#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cubic-Spline Trajectory Smoother (ROS1)

Subscribes to a raw path (nav_msgs/Path), fits a natural cubic spline,
and publishes smooth position commands at constant speed (decel near goal).

Subscriptions:
    <path_topic>  (nav_msgs/Path)      -- raw waypoint path
    /odom         (nav_msgs/Odometry)   -- robot state

Publications:
    /planning/pos_cmd  (geometry_msgs/PoseStamped)  -- smooth position cmd
"""

import math
import threading
import numpy as np
from scipy.interpolate import CubicSpline
import rospy

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header


class SplineTrajSmoother(object):

    def __init__(self):
        rospy.init_node("poly_traj_smoother", anonymous=False)

        self.max_vel = rospy.get_param("~max_vel", 4.0)
        self.max_acc = rospy.get_param("~max_acc", 2.0)
        self.publish_rate = rospy.get_param("~publish_rate", 50.0)
        self.fixed_z = rospy.get_param("~fixed_z", 1.0)
        self.path_topic = rospy.get_param("~path_topic", "/planning/raw_path")
        self.min_wp_dist = rospy.get_param("~min_wp_dist", 0.05)

        # Trajectory state (protected by lock)
        self.lock = threading.Lock()
        self.spline_x = None
        self.spline_y = None
        self.total_arc = 0.0
        self.total_time = 0.0
        self.traj_start = None
        self.traj_valid = False

        # Robot state
        self.robot_pos = None
        self.robot_yaw = 0.0
        self.last_traj_yaw = 0.0  # last yaw from spline tangent

        # ROS I/O
        self.pub_cmd = rospy.Publisher("/planning/pos_cmd", PoseStamped, queue_size=1)
        rospy.Subscriber(self.path_topic, Path, self._cb_path, queue_size=1)
        rospy.Subscriber("/odom", Odometry, self._cb_odom, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self._timer_cb)

        rospy.loginfo("[SplineTraj] Ready. v_max=%.1f a_max=%.1f topic=%s",
                      self.max_vel, self.max_acc, self.path_topic)

    # ===================================================================
    # Callbacks
    # ===================================================================
    def _cb_odom(self, msg):
        self.robot_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny, cosy)

    def _cb_path(self, msg):
        if len(msg.poses) < 2:
            return

        # Extract & deduplicate waypoints
        wps = []
        for ps in msg.poses:
            pt = np.array([ps.pose.position.x, ps.pose.position.y])
            if len(wps) == 0 or np.linalg.norm(pt - wps[-1]) > self.min_wp_dist:
                wps.append(pt)
        if len(wps) < 2:
            return

        # Replace first waypoint with robot position
        if self.robot_pos is not None:
            wps[0] = np.array(self.robot_pos)

        self._build_trajectory(wps, self.robot_yaw)

    # ===================================================================
    # Trajectory generation
    # ===================================================================
    def _build_trajectory(self, wps, start_yaw=None):
        pts = np.array(wps)

        # Cumulative arc length
        diffs = np.diff(pts, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        arc = np.zeros(len(pts))
        arc[1:] = np.cumsum(seg_lens)
        total_arc = arc[-1]

        if total_arc < 1e-4:
            return

        # Boundary conditions:
        # Few waypoints (<=3): clamp start tangent to robot heading for curvature
        # Many waypoints: natural BC, the spline shape is already good
        if start_yaw is not None and len(pts) <= 3:
            bc_x = ((1, math.cos(start_yaw)), (2, 0.0))
            bc_y = ((1, math.sin(start_yaw)), (2, 0.0))
        else:
            bc_x = 'natural'
            bc_y = 'natural'

        spline_x = CubicSpline(arc, pts[:, 0], bc_type=bc_x)
        spline_y = CubicSpline(arc, pts[:, 1], bc_type=bc_y)

        # Time: trapezoid  0 -> v_max -> 0
        v = self.max_vel
        a = self.max_acc
        s_ramp = v * v / a  # accel + decel distance
        if total_arc >= s_ramp:
            t_acc = v / a
            t_dec = v / a
            t_cruise = (total_arc - s_ramp) / v
        else:
            vp = math.sqrt(total_arc * a)
            t_acc = vp / a
            t_dec = vp / a
            t_cruise = 0.0
            v = vp

        total_time = t_acc + t_cruise + t_dec

        with self.lock:
            self.spline_x = spline_x
            self.spline_y = spline_y
            self.total_arc = total_arc
            self.total_time = total_time
            self.t_acc = t_acc
            self.t_cruise = t_cruise
            self.t_dec = t_dec
            self.v_peak = v
            self.traj_start = rospy.Time.now()
            self.traj_valid = True

        rospy.loginfo("[SplineTraj] arc=%.2fm time=%.2fs wps=%d", total_arc, total_time, len(wps))

    def _time_to_arc(self, t):
        a = self.max_acc
        vp = self.v_peak
        t1 = self.t_acc
        t2 = t1 + self.t_cruise

        if t <= t1:
            return 0.5 * a * t * t
        elif t <= t2:
            s1 = 0.5 * a * t1 * t1
            return s1 + vp * (t - t1)
        else:
            s1 = 0.5 * a * t1 * t1
            s2 = s1 + vp * self.t_cruise
            dt = t - t2
            return s2 + vp * dt - 0.5 * a * dt * dt

    # ===================================================================
    # Timer
    # ===================================================================
    def _timer_cb(self, event):
        with self.lock:
            if not self.traj_valid or self.traj_start is None:
                return

            t = (rospy.Time.now() - self.traj_start).to_sec()

            if t >= self.total_time:
                x = float(self.spline_x(self.total_arc))
                y = float(self.spline_y(self.total_arc))
                self._publish(x, y, self.last_traj_yaw)
                self.traj_valid = False
                return

            s = min(max(self._time_to_arc(t), 0.0), self.total_arc)
            x = float(self.spline_x(s))
            y = float(self.spline_y(s))

            dx = float(self.spline_x(s, 1))
            dy = float(self.spline_y(s, 1))
            if dx * dx + dy * dy > 1e-6:
                self.last_traj_yaw = math.atan2(dy, dx)

            self._publish(x, y, self.last_traj_yaw)

    def _publish(self, x, y, yaw=None):
        if yaw is None:
            yaw = self.robot_yaw
        msg = PoseStamped()
        msg.header = Header(stamp=rospy.Time.now(), frame_id="world")
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = self.fixed_z
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        msg.pose.orientation.z = sy
        msg.pose.orientation.w = cy
        self.pub_cmd.publish(msg)

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = SplineTrajSmoother()
        node.spin()
    except rospy.ROSInterruptException:
        pass
