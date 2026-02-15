#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmark Recorder Node

Subscribes to goal, odom, and occupancy grid topics to record navigation
benchmark metrics. When the robot reaches within 0.2 m of the goal, the
following metrics are saved to a CSV file (all rounded to 2 decimal places):

  1. Travel time  -- from goal receipt to arrival
  2. Length scale  -- path length / Euclidean(start, goal)
  3. Average linear velocity (from odom twist)
  4. Known area ratio in the 2-D occupancy grid

CSV naming: {map}_{algorithm}_{start}_{goal}_{timestamp}.csv
"""

import os
import csv
import math
from datetime import datetime

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, OccupancyGrid


# ───────────────────────── helpers ──────────────────────────

# ANSI purple (magenta) for terminal output
_PURPLE = "\033[35m"
_BOLD_PURPLE = "\033[1;35m"
_RESET = "\033[0m"


def purple(text):
    """Wrap *text* in ANSI purple escape codes."""
    return "{}{}{}".format(_PURPLE, text, _RESET)


def bold_purple(text):
    """Wrap *text* in ANSI bold-purple escape codes."""
    return "{}{}{}".format(_BOLD_PURPLE, text, _RESET)


# ───────────────────────── node ─────────────────────────────

class BenchmarkRecorder:
    """Records navigation metrics and writes them to CSV on goal arrival."""

    # States
    IDLE = 0
    NAVIGATING = 1

    def __init__(self):
        rospy.init_node("benchmark_recorder")

        # Parameters
        self.algorithm = rospy.get_param("~algorithm", "unknown")
        self.map_name = rospy.get_param("~map_name", "unknown")
        self.goal_thresh = rospy.get_param("~goal_reach_threshold", 0.2)
        output_dir = rospy.get_param("~output_dir",
                                     os.path.expanduser("~/benchmark_results"))

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.csv_path = None  # set when goal reached (needs start/goal info)

        # State
        self.state = self.IDLE
        self.goal_xy = None          # (gx, gy)
        self.start_xy = None         # (sx, sy)  position when goal received
        self.start_time = None       # rospy.Time

        self.path_length = 0.0       # accumulated 2-D path length
        self.prev_xy = None          # last odom (x, y) for path accumulation
        self.vel_samples = []        # list of |v| samples

        self.known_area_ratio = 0.0  # latest value from occupancy grid

        # Subscribers
        rospy.Subscriber("/move_base_simple/goal", PoseStamped,
                         self._goal_cb, queue_size=5)
        rospy.Subscriber("/odom", Odometry,
                         self._odom_cb, queue_size=50)
        rospy.Subscriber("/local_sensing/occupancy_grid", OccupancyGrid,
                         self._map_cb, queue_size=2)

        print(purple("[BenchmarkRecorder] ready  |  algorithm={}  map={}  "
                     "output_dir={}".format(self.algorithm, self.map_name,
                                            os.path.abspath(self.output_dir))))

    # ── callbacks ────────────────────────────────────────────

    def _goal_cb(self, msg):
        """New navigation goal received -- (re)start recording."""
        gx = msg.pose.position.x
        gy = msg.pose.position.y

        # Reset accumulators
        self.goal_xy = (gx, gy)
        self.start_time = rospy.Time.now()
        self.path_length = 0.0
        self.prev_xy = None
        self.vel_samples = []
        self.start_xy = None        # will be set on first odom after goal
        self.state = self.NAVIGATING

        print(purple("[BenchmarkRecorder] goal received ({:.2f}, {:.2f}) -- "
                     "recording started".format(gx, gy)))

    def _odom_cb(self, msg):
        """Odometry update -- accumulate path / velocity and check arrival."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        if self.state != self.NAVIGATING:
            return

        # Record start position on first odom after goal
        if self.start_xy is None:
            self.start_xy = (x, y)
            self.prev_xy = (x, y)

        # Accumulate path length (2-D)
        dx = x - self.prev_xy[0]
        dy = y - self.prev_xy[1]
        seg = math.hypot(dx, dy)
        self.path_length += seg
        self.prev_xy = (x, y)

        # Collect speed sample
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        speed = math.hypot(vx, vy)
        self.vel_samples.append(speed)

        # Check if goal reached
        dist_to_goal = math.hypot(x - self.goal_xy[0], y - self.goal_xy[1])
        if dist_to_goal < self.goal_thresh:
            self._on_goal_reached()

    def _map_cb(self, msg):
        """Occupancy grid update -- compute known area ratio."""
        data = msg.data
        total = len(data)
        if total == 0:
            return
        known = sum(1 for v in data if v >= 0)
        self.known_area_ratio = known / total

    # ── goal reached ─────────────────────────────────────────

    def _on_goal_reached(self):
        """Compute final metrics, log them, and save to CSV."""
        travel_time = (rospy.Time.now() - self.start_time).to_sec()

        # Length scale
        euclid = math.hypot(self.goal_xy[0] - self.start_xy[0],
                            self.goal_xy[1] - self.start_xy[1])
        if euclid < 1e-3:
            length_scale_str = "N/A"
        else:
            length_scale_str = "{:.2f}".format(
                round(self.path_length / euclid, 2))

        # Average velocity
        if self.vel_samples:
            avg_vel = sum(self.vel_samples) / len(self.vel_samples)
        else:
            avg_vel = 0.0

        # Round to 2 decimal places
        travel_time_r = round(travel_time, 2)
        avg_vel_r = round(avg_vel, 2)
        known_ratio_r = round(self.known_area_ratio, 2)

        # ── Purple terminal output ──
        sx, sy = self.start_xy
        gx, gy = self.goal_xy
        print(bold_purple("=" * 40))
        print(bold_purple("       Benchmark Result"))
        print(bold_purple("=" * 40))
        print(purple("  Algorithm      : {}".format(self.algorithm)))
        print(purple("  Map            : {}".format(self.map_name)))
        print(purple("  Start          : ({:.2f}, {:.2f})".format(sx, sy)))
        print(purple("  Goal           : ({:.2f}, {:.2f})".format(gx, gy)))
        print(purple("  Travel time    : {:.2f} s".format(travel_time_r)))
        print(purple("  Length scale   : {}".format(length_scale_str)))
        print(purple("  Avg velocity   : {:.2f} m/s".format(avg_vel_r)))
        print(purple("  Known area     : {:.2f}".format(known_ratio_r)))
        print(bold_purple("=" * 40))

        # ── Build CSV file path ──
        # Format: {map}_{algorithm}_{sx}_{sy}_to_{gx}_{gy}_{timestamp}.csv
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_name = "{map}_{alg}_({sx:.1f},{sy:.1f})_to_({gx:.1f},{gy:.1f})_{t}.csv".format(
            map=self.map_name, alg=self.algorithm,
            sx=sx, sy=sy, gx=gx, gy=gy, t=now_str)
        self.csv_path = os.path.join(self.output_dir, csv_name)

        # Write CSV (one file per goal, with header)
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "algorithm", "map_name",
                "start_x", "start_y", "goal_x", "goal_y",
                "travel_time_s", "length_scale",
                "avg_velocity_m_s", "known_area_ratio",
            ])
            writer.writerow([
                self.algorithm, self.map_name,
                round(sx, 2), round(sy, 2),
                round(gx, 2), round(gy, 2),
                travel_time_r, length_scale_str,
                avg_vel_r, known_ratio_r,
            ])

        print(purple("[BenchmarkRecorder] saved -> {}".format(
            os.path.abspath(self.csv_path))))

        # Return to idle (ready for next goal)
        self.state = self.IDLE


# ─────────────────────────────────────────────────────────────

def main():
    try:
        node = BenchmarkRecorder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
