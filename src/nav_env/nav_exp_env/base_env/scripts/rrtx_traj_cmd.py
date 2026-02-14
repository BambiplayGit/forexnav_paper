#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RRTX Trajectory B-Spline Commander (ROS1)

RRTX 专用：订阅 /planning/raw_path，B-spline 拟合后发布 PositionCommand。
与 frtree_traj_cmd 逻辑类似，但轨迹最远端（已知范围内的终点）强制速度为 0。

Subscribes:
  ~path_topic (default /planning/raw_path)  nav_msgs/Path
  /odom                                    nav_msgs/Odometry

Publishes:
  /planning/pos_cmd   PositionCommand (PVA)
  /planning/bspline_path  Path (visualization)
"""

import math
import threading
import numpy as np
import rospy

from scipy.interpolate import CubicSpline

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from quadrotor_msgs.msg import PositionCommand
from std_msgs.msg import Header


def quat_to_yaw(q):
    """Extract yaw from quaternion (geometry_msgs orientation)."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class RRTxTrajCmd(object):

    def __init__(self):
        rospy.init_node("rrtx_traj_cmd", anonymous=False)

        # --- Parameters ---
        self.max_vel = rospy.get_param("~max_vel", 2.0)
        self.max_acc = rospy.get_param("~max_acc", 2.0)
        self.publish_rate = rospy.get_param("~publish_rate", 50.0)
        self.fixed_z = rospy.get_param("~fixed_z", 1.0)
        self.path_topic = rospy.get_param("~path_topic", "/planning/raw_path")
        self.min_wp_dist = rospy.get_param("~min_wp_dist", 0.01)
        self.lookahead_dist = rospy.get_param("~lookahead_dist", 0.3)
        self.n_search = rospy.get_param("~n_search", 200)
        # 轨迹终点邻域弧长(m)：进入此范围后末端速度置 0（仅 RRTX 使用）
        self.end_zone_arc = rospy.get_param("~end_zone_arc", 0.05)

        # --- B-spline state (protected by lock) ---
        self.lock = threading.Lock()
        self.cs_x = None
        self.cs_y = None
        self.total_arc = 0.0
        self.traj_valid = False

        # --- Robot state ---
        self.robot_pos = None
        self.robot_yaw = 0.0
        self.last_yaw = 0.0

        # --- ROS I/O ---
        self.pub_cmd = rospy.Publisher(
            "/planning/pos_cmd", PositionCommand, queue_size=1)
        self.pub_bspline_path = rospy.Publisher(
            "/planning/bspline_path", Path, queue_size=1, latch=True)
        rospy.Subscriber(self.path_topic, Path, self._cb_path, queue_size=1)
        rospy.Subscriber("/odom", Odometry, self._cb_odom, queue_size=1)
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_rate), self._timer_cb)

        rospy.loginfo("[RRTxTrajCmd] B-spline, endpoint vel=0. v_max=%.1f path_topic=%s end_zone=%.2fm",
                      self.max_vel, self.path_topic, self.end_zone_arc)

    def _cb_odom(self, msg):
        self.robot_pos = np.array([msg.pose.pose.position.x,
                                   msg.pose.pose.position.y])
        self.robot_yaw = quat_to_yaw(msg.pose.pose.orientation)

    def _cb_path(self, msg):
        """Handle nav_msgs/Path from RRTX (/planning/raw_path)."""
        if len(msg.poses) < 2:
            return
        wps = []
        for ps in msg.poses:
            pt = np.array([ps.pose.position.x, ps.pose.position.y])
            if len(wps) == 0 or np.linalg.norm(pt - wps[-1]) > self.min_wp_dist:
                wps.append(pt)
        self._fit_and_update_spline(wps)

    def _fit_and_update_spline(self, wps):
        if len(wps) < 2:
            return
        if self.robot_pos is not None:
            wps[0] = self.robot_pos.copy()

        pts = np.array(wps)
        diffs = np.diff(pts, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        arc = np.zeros(len(pts))
        arc[1:] = np.cumsum(seg_lens)
        total_arc = arc[-1]
        if total_arc < 0.02:
            return

        try:
            if len(pts) >= 4:
                cs_x = CubicSpline(arc, pts[:, 0])
                cs_y = CubicSpline(arc, pts[:, 1])
            elif len(pts) >= 2:
                cs_x = CubicSpline(arc, pts[:, 0], bc_type='natural')
                cs_y = CubicSpline(arc, pts[:, 1], bc_type='natural')
            else:
                return
        except Exception as e:
            rospy.logwarn("[RRTxTrajCmd] CubicSpline fit failed: %s", e)
            return

        with self.lock:
            self.cs_x = cs_x
            self.cs_y = cs_y
            self.total_arc = total_arc
            self.traj_valid = True

        rospy.loginfo("[RRTxTrajCmd] New B-spline: arc=%.2fm, wps=%d", total_arc, len(wps))
        self._publish_bspline_vis(cs_x, cs_y, total_arc)

    def _publish_bspline_vis(self, cs_x, cs_y, total_arc):
        path_msg = Path()
        path_msg.header = Header(stamp=rospy.Time.now(), frame_id="odom")
        n_vis = max(int(total_arc / 0.02), 20)
        s_vals = np.linspace(0.0, total_arc, n_vis)
        x_vals = cs_x(s_vals)
        y_vals = cs_y(s_vals)
        for i in range(n_vis):
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = float(x_vals[i])
            ps.pose.position.y = float(y_vals[i])
            ps.pose.position.z = self.fixed_z
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)
        self.pub_bspline_path.publish(path_msg)

    def _timer_cb(self, event):
        with self.lock:
            if not self.traj_valid or self.robot_pos is None:
                return
            if self.cs_x is None or self.cs_y is None:
                return
            cs_x = self.cs_x
            cs_y = self.cs_y
            total_arc = self.total_arc

        n = self.n_search
        s_samples = np.linspace(0.0, total_arc, n + 1)
        x_samples = cs_x(s_samples)
        y_samples = cs_y(s_samples)
        dx = x_samples - self.robot_pos[0]
        dy = y_samples - self.robot_pos[1]
        dist_sq = dx * dx + dy * dy
        idx_closest = int(np.argmin(dist_sq))
        s_closest = s_samples[idx_closest]

        s_target = min(s_closest + self.lookahead_dist, total_arc)

        px = float(cs_x(s_target))
        py = float(cs_y(s_target))

        dpx_ds = float(cs_x(s_target, 1))
        dpy_ds = float(cs_y(s_target, 1))
        tangent_norm = math.sqrt(dpx_ds ** 2 + dpy_ds ** 2)
        if tangent_norm > 1e-6:
            tx = dpx_ds / tangent_norm
            ty = dpy_ds / tangent_norm
        else:
            tx, ty = 1.0, 0.0

        dist_to_end = total_arc - s_closest
        dist_from_start = s_closest

        # RRTX：轨迹最远端（已知范围内终点）速度强制为 0
        if dist_to_end <= self.end_zone_arc or s_target >= total_arc - 1e-6:
            speed = 0.0
            a_tangential = 0.0
            px = float(cs_x(total_arc))
            py = float(cs_y(total_arc))
            if total_arc >= 0.01:
                dpx_ds = float(cs_x(total_arc, 1))
                dpy_ds = float(cs_y(total_arc, 1))
                n_t = math.sqrt(dpx_ds ** 2 + dpy_ds ** 2)
                if n_t > 1e-6:
                    tx, ty = dpx_ds / n_t, dpy_ds / n_t
        else:
            v_dec = math.sqrt(max(2.0 * self.max_acc * dist_to_end, 0.0))
            v_acc = math.sqrt(max(2.0 * self.max_acc * dist_from_start, 0.0))
            speed = min(self.max_vel, v_dec, v_acc)
            speed = max(speed, 0.0)
            if dist_to_end < speed * speed / (2.0 * self.max_acc + 1e-6):
                a_tangential = -self.max_acc
            elif dist_from_start < speed * speed / (2.0 * self.max_acc + 1e-6):
                a_tangential = self.max_acc
            else:
                a_tangential = 0.0

        vx = tx * speed
        vy = ty * speed
        ax = tx * a_tangential
        ay = ty * a_tangential

        if abs(vx) > 0.01 or abs(vy) > 0.01:
            yaw = math.atan2(vy, vx)
            self.last_yaw = yaw
        else:
            yaw = self.last_yaw

        yaw_dot = 0.0
        ds_yaw = max(0.01, min(0.02 * total_arc, 0.1))
        if s_target + ds_yaw <= total_arc:
            dpx_next = float(cs_x(s_target + ds_yaw, 1))
            dpy_next = float(cs_y(s_target + ds_yaw, 1))
            n_next = math.sqrt(dpx_next ** 2 + dpy_next ** 2)
            if n_next > 1e-6 and tangent_norm > 1e-6:
                yaw_next = math.atan2(dpy_next, dpx_next)
                dyaw = yaw_next - yaw
                while dyaw > math.pi:
                    dyaw -= 2.0 * math.pi
                while dyaw < -math.pi:
                    dyaw += 2.0 * math.pi
                yaw_dot = (dyaw / ds_yaw) * speed

        msg = PositionCommand()
        msg.header = Header(stamp=rospy.Time.now(), frame_id="world")
        msg.position.x = px
        msg.position.y = py
        msg.position.z = self.fixed_z
        msg.velocity.x = vx
        msg.velocity.y = vy
        msg.velocity.z = 0.0
        msg.acceleration.x = ax
        msg.acceleration.y = ay
        msg.acceleration.z = 0.0
        msg.yaw = yaw
        msg.yaw_dot = yaw_dot
        msg.kx = [0.0, 0.0, 0.0]
        msg.kv = [0.0, 0.0, 0.0]
        msg.trajectory_flag = msg.TRAJECTORY_STATUS_READY
        self.pub_cmd.publish(msg)

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = RRTxTrajCmd()
        node.spin()
    except rospy.ROSInterruptException:
        pass
