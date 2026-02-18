#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preset Goal Sender

Subscribes to /clicked_point  (rviz "Publish Point" tool).
On each click, publishes the PRESET goal to /move_base_simple/goal.

This does NOT interfere with the "2D Nav Goal" rviz tool, which also
publishes to /move_base_simple/goal via a completely separate rviz tool.

Params:
  ~goal_x            (double)  preset goal X
  ~goal_y            (double)  preset goal Y
  ~planning_height   (double)  Z height for goal pose
"""

import rospy
from geometry_msgs.msg import PointStamped, PoseStamped


_BOLD_YELLOW = "\033[1;33m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"


class PresetGoalSender:
    def __init__(self):
        rospy.init_node("preset_goal_sender")

        self.goal_x = rospy.get_param("~goal_x", 0.0)
        self.goal_y = rospy.get_param("~goal_y", 0.0)
        self.planning_height = rospy.get_param("~planning_height", 0.4)

        self.pub = rospy.Publisher(
            "/move_base_simple/goal", PoseStamped, queue_size=10)
        rospy.Subscriber(
            "/clicked_point", PointStamped, self._cb, queue_size=10)

        rospy.loginfo(
            "{}[PresetGoalSender]{} preset=({:.2f}, {:.2f})  "
            "Click 'Publish Point' in rviz to trigger".format(
                _BOLD_YELLOW, _RESET, self.goal_x, self.goal_y))

    def _cb(self, msg):
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "world"
        goal.pose.position.x = self.goal_x
        goal.pose.position.y = self.goal_y
        goal.pose.position.z = self.planning_height
        goal.pose.orientation.w = 1.0

        self.pub.publish(goal)
        rospy.loginfo(
            "{}[PresetGoalSender]{} sent goal ({:.2f}, {:.2f})".format(
                _YELLOW, _RESET, self.goal_x, self.goal_y))


def main():
    try:
        PresetGoalSender()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
