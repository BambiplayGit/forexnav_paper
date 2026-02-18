#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive Benchmark Point Selector

Usage:
  roslaunch nav_exp_env set_benchmark_points.launch map_name:=hkust1

  1. Use rviz "2D Pose Estimate" tool to set the START point (init_x, init_y, init_yaw)
  2. Use rviz "2D Nav Goal" tool to set the GOAL point (goal_x, goal_y)
  3. Both values are written back to benchmark.launch automatically.
  4. You can re-click to update. Ctrl+C to exit.
"""

import os
import re
import math

import rospy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped


# ANSI colours for clear terminal output
_CYAN = "\033[36m"
_BOLD_CYAN = "\033[1;36m"
_GREEN = "\033[32m"
_BOLD_GREEN = "\033[1;32m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"


def cyan(t):
    return "{}{}{}".format(_CYAN, t, _RESET)


def bold_cyan(t):
    return "{}{}{}".format(_BOLD_CYAN, t, _RESET)


def green(t):
    return "{}{}{}".format(_GREEN, t, _RESET)


def bold_green(t):
    return "{}{}{}".format(_BOLD_GREEN, t, _RESET)


def yellow(t):
    return "{}{}{}".format(_YELLOW, t, _RESET)


class SetBenchmarkPoints:
    """Subscribe to rviz tools, collect start/goal, write to benchmark.launch."""

    def __init__(self):
        rospy.init_node("set_benchmark_points")

        self.map_name = rospy.get_param("~map_name", "office1")
        self.benchmark_launch = rospy.get_param("~benchmark_launch", "")

        if not self.benchmark_launch:
            import rospkg
            rospack = rospkg.RosPack()
            pkg_path = rospack.get_path("nav_exp_env")
            self.benchmark_launch = os.path.join(
                pkg_path, "base_env", "launch", "benchmark.launch")

        self.start_set = False
        self.goal_set = False
        self.start_x = None
        self.start_y = None
        self.start_yaw = None
        self.goal_x = None
        self.goal_y = None

        rospy.Subscriber("/initialpose", PoseWithCovarianceStamped,
                         self._start_cb, queue_size=5)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped,
                         self._goal_cb, queue_size=5)

        print(bold_cyan("=" * 62))
        print(bold_cyan("  Benchmark Point Selector"))
        print(bold_cyan("=" * 62))
        print(cyan("  Map           : {}".format(self.map_name)))
        print(cyan("  Launch file   : {}".format(self.benchmark_launch)))
        print(cyan("  Step 1  -->  Use '2D Pose Estimate' to set START"))
        print(cyan("  Step 2  -->  Use '2D Nav Goal'      to set GOAL"))
        print(bold_cyan("=" * 62))

    # ── helpers ───────────────────────────────────────────────

    @staticmethod
    def _quat_to_yaw(q):
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    # ── callbacks ─────────────────────────────────────────────

    def _start_cb(self, msg):
        self.start_x = msg.pose.pose.position.x
        self.start_y = msg.pose.pose.position.y
        self.start_yaw = self._quat_to_yaw(msg.pose.pose.orientation)
        self.start_set = True

        print(green("[START] ({:.2f}, {:.2f})  yaw={:.2f}".format(
            self.start_x, self.start_y, self.start_yaw)))

        if self.goal_set:
            self._write_to_launch()
        else:
            print(yellow("  --> Now use '2D Nav Goal' to set GOAL"))

    def _goal_cb(self, msg):
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        self.goal_set = True

        print(green("[GOAL]  ({:.2f}, {:.2f})".format(
            self.goal_x, self.goal_y)))

        if self.start_set:
            self._write_to_launch()
        else:
            print(yellow("  --> Now use '2D Pose Estimate' to set START"))

    # ── file writer ───────────────────────────────────────────

    def _update_arg_value(self, content, arg_name, new_value):
        """Replace the value="..." of an <arg name="ARG_NAME" value="..."/> tag."""
        pattern = r'(<arg\s+name="{}"\s+value=")([^"]*)(")'.format(
            re.escape(arg_name))
        return re.sub(pattern, r'\g<1>{}\3'.format(new_value), content)

    def _write_to_launch(self):
        """Write start/goal to benchmark.launch for the current map."""
        launch_path = self.benchmark_launch
        if not os.path.isfile(launch_path):
            rospy.logerr("benchmark.launch not found: %s", launch_path)
            return

        with open(launch_path, "r") as f:
            content = f.read()

        mn = self.map_name

        # Update init_x, init_y, init_yaw
        content = self._update_arg_value(
            content, "{}_init_x".format(mn), "{:.2f}".format(self.start_x))
        content = self._update_arg_value(
            content, "{}_init_y".format(mn), "{:.2f}".format(self.start_y))
        content = self._update_arg_value(
            content, "{}_init_yaw".format(mn), "{:.2f}".format(self.start_yaw))

        # Update goal_x, goal_y (insert if missing)
        goal_x_tag = '{}_goal_x'.format(mn)
        goal_y_tag = '{}_goal_y'.format(mn)

        if re.search(r'name="{}"'.format(re.escape(goal_x_tag)), content):
            content = self._update_arg_value(
                content, goal_x_tag, "{:.2f}".format(self.goal_x))
            content = self._update_arg_value(
                content, goal_y_tag, "{:.2f}".format(self.goal_y))
        else:
            # Insert goal_x/goal_y right after the init_yaw line
            init_yaw_pat = r'(<arg\s+name="{}_init_yaw"\s+value="[^"]*"\s*/>)'.format(
                re.escape(mn))
            replacement = (
                r'\1\n'
                '    <arg name="{gx}"    value="{vx}"/>\n'
                '    <arg name="{gy}"    value="{vy}"/>'
            ).format(
                gx=goal_x_tag, vx="{:.2f}".format(self.goal_x),
                gy=goal_y_tag, vy="{:.2f}".format(self.goal_y),
            )
            content = re.sub(init_yaw_pat, replacement, content)

        with open(launch_path, "w") as f:
            f.write(content)

        print(bold_green("=" * 62))
        print(bold_green("  SAVED to benchmark.launch"))
        print(bold_green("=" * 62))
        print(green("  Map   : {}".format(mn)))
        print(green("  Start : ({:.2f}, {:.2f})  yaw={:.2f}".format(
            self.start_x, self.start_y, self.start_yaw)))
        print(green("  Goal  : ({:.2f}, {:.2f})".format(
            self.goal_x, self.goal_y)))
        print(bold_green("=" * 62))
        print(yellow("  You can click again to update, or Ctrl+C to exit."))


def main():
    try:
        SetBenchmarkPoints()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
