#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
clicked_point_to_pos_cmd (ROS1)

Listen to RViz "/clicked_point" (geometry_msgs/PointStamped).
On each click, publish a PoseStamped to "/planning/pos_cmd".

Then existing simple_odom_simulator.py remains the ONLY /odom publisher.

Params:
- fixed_z (double, default 1.0)
- fixed_yaw (double, default 0.0)
- frame_id (string, default "world")
"""

import math
import rospy
from geometry_msgs.msg import PointStamped, PoseStamped


def yaw_to_quat(yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (0.0, 0.0, sy, cy)


class ClickedPointToPosCmd:
    def __init__(self):
        rospy.init_node("clicked_point_to_pos_cmd")

        self.fixed_z = rospy.get_param("~fixed_z", 1.0)
        self.fixed_yaw = rospy.get_param("~fixed_yaw", 0.0)
        self.frame_id = rospy.get_param("~frame_id", "world")

        self.pub = rospy.Publisher("/planning/pos_cmd", PoseStamped, queue_size=10)
        self.sub = rospy.Subscriber("/clicked_point", PointStamped, self._cb, queue_size=10)

        rospy.loginfo("clicked_point_to_pos_cmd started")
        rospy.loginfo("  fixed_z={}, fixed_yaw={}, frame_id={}".format(self.fixed_z, self.fixed_yaw, self.frame_id))
        rospy.loginfo("  Click in RViz (Publish Point tool) to send /planning/pos_cmd")

    def _cb(self, msg):
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = self.frame_id

        pose.pose.position.x = msg.point.x
        pose.pose.position.y = msg.point.y
        pose.pose.position.z = self.fixed_z

        qx, qy, qz, qw = yaw_to_quat(self.fixed_yaw)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        self.pub.publish(pose)
        rospy.loginfo("click -> /planning/pos_cmd: x={:.2f} y={:.2f} z={:.2f} yaw={:.2f}".format(
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z, self.fixed_yaw))


def main():
    try:
        node = ClickedPointToPosCmd()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
