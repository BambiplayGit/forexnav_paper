#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FRTree Trajectory B-Spline Commander (ROS1)

Subscribes to FRTree's raw ALTRO trajectory output (geometry_msgs/PoseArray
on /traj_pose_vis), fits a cubic B-spline via scipy, and publishes
quadrotor_msgs/PositionCommand with position (p) and velocity (v) at 50Hz
using closest-point + lookahead tracking.

Also publishes the full B-spline curve as nav_msgs/Path for RViz visualization.

Tracking method:
    At each 50Hz tick, find the closest point on the B-spline to the robot's
    current position, then look ahead by a configurable distance to get the
    target.  This adapts to the robot's actual speed and avoids stutter.
"""

import math
import threading
import numpy as np
import rospy

from scipy.interpolate import CubicSpline

from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import Odometry, Path
from quadrotor_msgs.msg import PositionCommand
from std_msgs.msg import Header


def quat_to_yaw(q):
    """Extract yaw from quaternion (geometry_msgs orientation)."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class FRTreeTrajCmd(object):

    def __init__(self):
        rospy.init_node("frtree_traj_cmd", anonymous=False)

        # --- Parameters ---
        self.max_vel = rospy.get_param("~max_vel", 2.0)
        self.max_acc = rospy.get_param("~max_acc", 2.0)
        self.publish_rate = rospy.get_param("~publish_rate", 50.0)
        self.fixed_z = rospy.get_param("~fixed_z", 1.0)
        self.traj_topic = rospy.get_param("~traj_topic", "/traj_pose_vis")
        self.min_wp_dist = rospy.get_param("~min_wp_dist", 0.01)
        self.lookahead_dist = rospy.get_param("~lookahead_dist", 0.3)
        self.n_search = rospy.get_param("~n_search", 200)

        # --- B-spline state (protected by lock) ---
        self.lock = threading.Lock()
        self.cs_x = None          # CubicSpline for x(s)
        self.cs_y = None          # CubicSpline for y(s)
        self.total_arc = 0.0
        self.traj_valid = False

        # --- Robot state ---
        self.robot_pos = None     # (x, y) numpy array
        self.robot_yaw = 0.0
        self.last_yaw = 0.0

        # --- ROS I/O ---
        self.pub_cmd = rospy.Publisher(
            "/planning/pos_cmd", PositionCommand, queue_size=1)
        self.pub_bspline_path = rospy.Publisher(
            "/planning/bspline_path", Path, queue_size=1, latch=True)
        rospy.Subscriber(self.traj_topic, PoseArray, self._cb_traj, queue_size=1)
        rospy.Subscriber("/odom", Odometry, self._cb_odom, queue_size=1)
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_rate), self._timer_cb)

        rospy.loginfo("[FRTreeTrajCmd] B-spline mode. v_max=%.1f lookahead=%.2fm topic=%s",
                      self.max_vel, self.lookahead_dist, self.traj_topic)

    # ==================================================================
    # Callbacks
    # ==================================================================
    def _cb_odom(self, msg):
        self.robot_pos = np.array([msg.pose.pose.position.x,
                                   msg.pose.pose.position.y])
        self.robot_yaw = quat_to_yaw(msg.pose.pose.orientation)

    def _cb_traj(self, msg):
        """Handle PoseArray from ALTRO solver output -- fit B-spline."""
        if len(msg.poses) < 2:
            return

        # Extract waypoints, skip duplicates
        wps = []
        for pose in msg.poses:
            pt = np.array([pose.position.x, pose.position.y])
            if len(wps) == 0 or np.linalg.norm(pt - wps[-1]) > self.min_wp_dist:
                wps.append(pt)
        if len(wps) < 2:
            return

        # Replace first waypoint with current robot position for continuity
        if self.robot_pos is not None:
            wps[0] = self.robot_pos.copy()

        pts = np.array(wps)  # (N, 2)

        # --- Arc-length parameterization ---
        diffs = np.diff(pts, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        arc = np.zeros(len(pts))
        arc[1:] = np.cumsum(seg_lens)
        total_arc = arc[-1]
        if total_arc < 0.02:
            return

        # --- Fit cubic B-spline x(s), y(s) ---
        # 'not-a-knot' (default): no artificial straightening at endpoints,
        #   preserves the natural curvature of the waypoints.
        # 'natural' forces zero curvature at endpoints -> makes curve too straight.
        try:
            if len(pts) >= 4:
                # Enough points for 'not-a-knot' (needs >= 4 points)
                cs_x = CubicSpline(arc, pts[:, 0])  # default = 'not-a-knot'
                cs_y = CubicSpline(arc, pts[:, 1])
            elif len(pts) >= 2:
                # Fallback for very few points
                cs_x = CubicSpline(arc, pts[:, 0], bc_type='natural')
                cs_y = CubicSpline(arc, pts[:, 1], bc_type='natural')
            else:
                return
        except Exception as e:
            rospy.logwarn("[FRTreeTrajCmd] CubicSpline fit failed: %s", e)
            return

        with self.lock:
            self.cs_x = cs_x
            self.cs_y = cs_y
            self.total_arc = total_arc
            self.traj_valid = True

        rospy.loginfo("[FRTreeTrajCmd] New B-spline: arc=%.2fm, wps=%d",
                      total_arc, len(wps))

        # --- Publish B-spline curve as Path for RViz visualization ---
        self._publish_bspline_vis(cs_x, cs_y, total_arc)

    # ==================================================================
    # B-spline visualization: publish the full curve as nav_msgs/Path
    # ==================================================================
    def _publish_bspline_vis(self, cs_x, cs_y, total_arc):
        """Publish the B-spline curve at high resolution for RViz."""
        path_msg = Path()
        path_msg.header = Header(stamp=rospy.Time.now(), frame_id="odom")

        n_vis = max(int(total_arc / 0.02), 20)  # ~2cm resolution
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

    # ==================================================================
    # Timer callback: publish PositionCommand at ~50 Hz
    # ==================================================================
    def _timer_cb(self, event):
        with self.lock:
            if not self.traj_valid or self.robot_pos is None:
                return
            if self.cs_x is None or self.cs_y is None:
                return

            cs_x = self.cs_x
            cs_y = self.cs_y
            total_arc = self.total_arc

        # ---------------------------------------------------------
        # 1. Find closest point on B-spline to robot position
        # ---------------------------------------------------------
        n = self.n_search
        s_samples = np.linspace(0.0, total_arc, n + 1)
        x_samples = cs_x(s_samples)
        y_samples = cs_y(s_samples)
        dx = x_samples - self.robot_pos[0]
        dy = y_samples - self.robot_pos[1]
        dist_sq = dx * dx + dy * dy
        idx_closest = int(np.argmin(dist_sq))
        s_closest = s_samples[idx_closest]

        # ---------------------------------------------------------
        # 2. Lookahead: target = closest + lookahead_dist
        # ---------------------------------------------------------
        s_target = min(s_closest + self.lookahead_dist, total_arc)

        # ---------------------------------------------------------
        # 3. Evaluate position p(s_target)
        # ---------------------------------------------------------
        px = float(cs_x(s_target))
        py = float(cs_y(s_target))

        # ---------------------------------------------------------
        # 4. Evaluate tangent dp/ds at s_target
        # ---------------------------------------------------------
        dpx_ds = float(cs_x(s_target, 1))
        dpy_ds = float(cs_y(s_target, 1))
        tangent_norm = math.sqrt(dpx_ds ** 2 + dpy_ds ** 2)

        if tangent_norm > 1e-6:
            tx = dpx_ds / tangent_norm
            ty = dpy_ds / tangent_norm
        else:
            tx, ty = 1.0, 0.0

        # ---------------------------------------------------------
        # 5. Speed profile (trapezoidal along arc-length)
        # ---------------------------------------------------------
        dist_to_end = total_arc - s_closest
        dist_from_start = s_closest

        v_dec = math.sqrt(max(2.0 * self.max_acc * dist_to_end, 0.0))
        v_acc = math.sqrt(max(2.0 * self.max_acc * dist_from_start, 0.0))
        speed = min(self.max_vel, v_dec, v_acc)
        speed = max(speed, 0.0)

        # ---------------------------------------------------------
        # 6. Compute velocity v = tangent * speed
        # ---------------------------------------------------------
        vx = tx * speed
        vy = ty * speed

        # ---------------------------------------------------------
        # 7. Compute acceleration
        # ---------------------------------------------------------
        if dist_to_end < speed * speed / (2.0 * self.max_acc + 1e-6):
            a_tangential = -self.max_acc
        elif dist_from_start < speed * speed / (2.0 * self.max_acc + 1e-6):
            a_tangential = self.max_acc
        else:
            a_tangential = 0.0

        ax = tx * a_tangential
        ay = ty * a_tangential

        # ---------------------------------------------------------
        # 8. Yaw from velocity direction
        # ---------------------------------------------------------
        if abs(vx) > 0.01 or abs(vy) > 0.01:
            yaw = math.atan2(vy, vx)
            self.last_yaw = yaw
        else:
            yaw = self.last_yaw

        # ---------------------------------------------------------
        # 9. Yaw rate from curvature
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # 10. Publish PositionCommand
        # ---------------------------------------------------------
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

    # ==================================================================
    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = FRTreeTrajCmd()
        node.spin()
    except rospy.ROSInterruptException:
        pass
