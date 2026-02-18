#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FRTree Trajectory B-Spline Commander (ROS1)

Subscribes to FRTree's local plan output and fits a cubic B-spline via scipy,
then publishes quadrotor_msgs/PositionCommand with full PVA at 50Hz using
closest-point + lookahead tracking.

Supported input sources (both can be active simultaneously):
  - nav_msgs/Path   on ~path_topic  (default "/localPlan")  -- primary
  - PoseArray       on ~traj_topic  (default "/traj_pose_vis") -- fallback

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

from scipy.interpolate import CubicSpline, splprep, splev

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
        self.path_topic = rospy.get_param("~path_topic", "/localPlan")
        self.min_wp_dist = rospy.get_param("~min_wp_dist", 0.01)
        self.lookahead_dist = rospy.get_param("~lookahead_dist", 0.3)
        self.n_search = rospy.get_param("~n_search", 200)
        self.smooth_dist = rospy.get_param("~smooth_dist", 0.1)

        # --- B-spline state (protected by lock) ---
        self.lock = threading.Lock()
        self.cs_x = None          # CubicSpline for x(s)
        self.cs_y = None          # CubicSpline for y(s)
        self.total_arc = 0.0
        self.traj_valid = False
        self.vel_profile_s = None   # arc-length samples for velocity profile
        self.vel_profile_v = None   # pre-computed speed at each sample

        # --- Robot state ---
        self.robot_pos = None     # (x, y) numpy array
        self.robot_vel = np.zeros(2)
        self.robot_yaw = 0.0
        self.last_yaw = 0.0

        # --- ROS I/O ---
        self.pub_cmd = rospy.Publisher(
            "/planning/pos_cmd", PositionCommand, queue_size=1)
        self.pub_bspline_path = rospy.Publisher(
            "/planning/bspline_path", Path, queue_size=1, latch=True)
        rospy.Subscriber(self.traj_topic, PoseArray, self._cb_traj, queue_size=1)
        rospy.Subscriber(self.path_topic, Path, self._cb_path, queue_size=1)
        rospy.Subscriber("/odom", Odometry, self._cb_odom, queue_size=1)
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_rate), self._timer_cb)

        rospy.loginfo("[FRTreeTrajCmd] B-spline mode. v_max=%.1f lookahead=%.2fm "
                      "path_topic=%s traj_topic=%s",
                      self.max_vel, self.lookahead_dist,
                      self.path_topic, self.traj_topic)

    # ==================================================================
    # Callbacks
    # ==================================================================
    def _cb_odom(self, msg):
        self.robot_pos = np.array([msg.pose.pose.position.x,
                                   msg.pose.pose.position.y])
        self.robot_vel = np.array([msg.twist.twist.linear.x,
                                   msg.twist.twist.linear.y])
        self.robot_yaw = quat_to_yaw(msg.pose.pose.orientation)

    def _cb_traj(self, msg):
        """Handle PoseArray from ALTRO solver output -- fit B-spline (fallback)."""
        if len(msg.poses) < 2:
            return

        # Extract waypoints, skip duplicates
        wps = []
        for pose in msg.poses:
            pt = np.array([pose.position.x, pose.position.y])
            if len(wps) == 0 or np.linalg.norm(pt - wps[-1]) > self.min_wp_dist:
                wps.append(pt)

        self._fit_and_update_spline(wps)

    def _cb_path(self, msg):
        """Handle nav_msgs/Path from FRTree localPlan -- fit B-spline (primary)."""
        if len(msg.poses) < 2:
            return

        # Extract waypoints, skip duplicates
        wps = []
        for ps in msg.poses:
            pt = np.array([ps.pose.position.x, ps.pose.position.y])
            if len(wps) == 0 or np.linalg.norm(pt - wps[-1]) > self.min_wp_dist:
                wps.append(pt)

        self._fit_and_update_spline(wps)

    # ==================================================================
    # Shared B-spline fitting
    # ==================================================================
    def _fit_and_update_spline(self, wps):
        """Fit cubic B-spline to waypoints and update tracking state.

        Args:
            wps: list of np.array([x, y]) waypoints (duplicates already filtered).
        """
        if len(wps) < 2:
            return

        # Replace first waypoint with current robot position for continuity
        if self.robot_pos is not None:
            wps[0] = self.robot_pos.copy()

        pts = np.array(wps)  # (N, 2)

        # --- Smooth waypoints with B-spline approximation ---
        # splprep with s>0 creates a smoothed B-spline that does not pass
        # through every waypoint, naturally cutting corners at sharp turns.
        if len(pts) >= 4 and self.smooth_dist > 0:
            try:
                s_param = len(pts) * self.smooth_dist ** 2
                tck, _u = splprep([pts[:, 0], pts[:, 1]], s=s_param, k=3)
                n_resample = max(len(pts) * 10, 100)
                u_fine = np.linspace(0.0, 1.0, n_resample)
                x_s, y_s = splev(u_fine, tck)
                x_s[0], y_s[0] = pts[0, 0], pts[0, 1]
                x_s[-1], y_s[-1] = pts[-1, 0], pts[-1, 1]
                pts = np.column_stack([x_s, y_s])
            except Exception as e:
                rospy.logwarn("[FRTreeTrajCmd] splprep smoothing failed, using raw waypoints: %s", e)

        # --- Arc-length parameterization ---
        diffs = np.diff(pts, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        arc = np.zeros(len(pts))
        arc[1:] = np.cumsum(seg_lens)
        total_arc = arc[-1]
        if total_arc < 0.02:
            return

        # --- Fit cubic spline x(s), y(s) on (possibly smoothed) points ---
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
            rospy.logwarn("[FRTreeTrajCmd] CubicSpline fit failed: %s", e)
            return

        # --- Pre-compute curvature-limited velocity profile ---
        robot_speed = float(np.linalg.norm(self.robot_vel))
        prof_s, prof_v = self._build_velocity_profile(cs_x, cs_y, total_arc, robot_speed)

        with self.lock:
            self.cs_x = cs_x
            self.cs_y = cs_y
            self.total_arc = total_arc
            self.vel_profile_s = prof_s
            self.vel_profile_v = prof_v
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
    # Curvature-limited velocity profile
    # ==================================================================
    def _build_velocity_profile(self, cs_x, cs_y, total_arc, start_speed=0.0):
        """Pre-compute a velocity profile that respects curvature + accel limits.

        Algorithm:
          1. Sample curvature kappa(s) along the spline.
          2. At each sample, the curvature speed limit is v_curv = sqrt(a_max / |kappa|).
          3. Forward pass:  v[i] = min(v_curv[i], sqrt(v[i-1]^2 + 2*a_max*ds))
          4. Backward pass: v[i] = min(v[i],      sqrt(v[i+1]^2 + 2*a_max*ds))
          5. Clamp to [0, max_vel].

        This is equivalent to what ForexNav achieves via feasibility-constrained
        B-spline optimisation, but computed directly.
        """
        n_samples = max(int(total_arc / 0.02), 50)
        s_arr = np.linspace(0.0, total_arc, n_samples)
        ds = s_arr[1] - s_arr[0] if n_samples > 1 else total_arc

        dx = cs_x(s_arr, 1)
        dy = cs_y(s_arr, 1)
        ddx = cs_x(s_arr, 2)
        ddy = cs_y(s_arr, 2)

        speed_sq = dx * dx + dy * dy
        speed_32 = np.power(np.maximum(speed_sq, 1e-12), 1.5)
        kappa = np.abs(dx * ddy - dy * ddx) / speed_32

        a_lat = self.max_acc * 0.8
        v_curv = np.where(kappa > 1e-6,
                          np.sqrt(a_lat / np.maximum(kappa, 1e-6)),
                          self.max_vel)
        v_curv = np.minimum(v_curv, self.max_vel)

        v_fwd = np.copy(v_curv)
        v_fwd[0] = min(v_curv[0], max(start_speed, 0.0))
        for i in range(1, n_samples):
            v_max_here = math.sqrt(v_fwd[i-1] ** 2 + 2.0 * self.max_acc * ds)
            v_fwd[i] = min(v_fwd[i], v_max_here)

        v_bwd = np.copy(v_fwd)
        v_bwd[-1] = 0.0
        for i in range(n_samples - 2, -1, -1):
            v_max_here = math.sqrt(v_bwd[i+1] ** 2 + 2.0 * self.max_acc * ds)
            v_bwd[i] = min(v_bwd[i], v_max_here)

        v_bwd = np.clip(v_bwd, 0.0, self.max_vel)
        return s_arr, v_bwd

    def _lookup_speed(self, s, prof_s, prof_v):
        """Linearly interpolate the pre-computed velocity profile at arc-length s."""
        return float(np.interp(s, prof_s, prof_v))

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
            prof_s = self.vel_profile_s
            prof_v = self.vel_profile_v

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
        # 5. Curvature-aware speed from pre-computed velocity profile
        # ---------------------------------------------------------
        speed = self._lookup_speed(s_closest, prof_s, prof_v)

        # ---------------------------------------------------------
        # 6. Compute velocity v = tangent * speed
        # ---------------------------------------------------------
        vx = tx * speed
        vy = ty * speed

        # ---------------------------------------------------------
        # 7. Acceleration from finite-difference of velocity profile
        # ---------------------------------------------------------
        ds_fd = 0.05
        if s_closest + ds_fd <= total_arc:
            v_ahead = self._lookup_speed(s_closest + ds_fd, prof_s, prof_v)
            a_tangential = (v_ahead ** 2 - speed ** 2) / (2.0 * ds_fd + 1e-9)
            a_tangential = max(-self.max_acc, min(a_tangential, self.max_acc))
        else:
            a_tangential = -self.max_acc if speed > 0.01 else 0.0

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
