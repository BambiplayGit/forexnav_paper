#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Waypoint to Position Command Converter

Converts /way_point (PointStamped) to /planning/pos_cmd (PoseStamped)
for the simple_odom_simulator to follow.

Interpolates between current position and target waypoint for smooth motion.
"""

import rospy
import math
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Header


class WaypointToPosCmd:
    def __init__(self):
        rospy.init_node('waypoint_to_pos_cmd')
        
        # Parameters
        self.fixed_z = rospy.get_param('~fixed_z', 1.0)
        self.frame_id = rospy.get_param('~frame_id', 'map')
        self.face_waypoint = rospy.get_param('~face_waypoint', True)
        self.publish_rate = rospy.get_param('~publish_rate', 50.0)  # Hz
        self.move_speed = rospy.get_param('~move_speed', 4.0)  # m/s
        self.arrival_threshold = rospy.get_param('~arrival_threshold', 0.1)  # meter
        self.raw_path_topic = rospy.get_param('~raw_path_topic', '/planning/raw_path')
        
        # Current position
        self.current_x = 0.0
        self.current_y = 0.0
        
        # Target waypoint
        self.target_x = None
        self.target_y = None
        
        # Last yaw (to keep orientation when arrived)
        self.last_yaw = 0.0
        
        # Publishers
        self.pub = rospy.Publisher('/planning/pos_cmd', PoseStamped, queue_size=1)
        self.pub_raw_path = rospy.Publisher(self.raw_path_topic, Path, queue_size=1)
        
        # Subscribers
        rospy.Subscriber('/way_point', PointStamped, self.waypoint_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        
        # Timer for continuous publishing
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.timer_callback)
        
        rospy.loginfo("waypoint_to_pos_cmd ready")
        rospy.loginfo("  publish_rate: {} Hz".format(self.publish_rate))
        rospy.loginfo("  move_speed: {} m/s".format(self.move_speed))
        rospy.loginfo("  raw_path_topic: {}".format(self.raw_path_topic))
        
    def odom_callback(self, msg):
        """Update current position"""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
    def waypoint_callback(self, msg):
        """Update target waypoint and publish 2-point Path on raw_path topic."""
        self.target_x = msg.point.x
        self.target_y = msg.point.y

        # Publish a 2-point Path (current -> target) for the trajectory smoother
        path_msg = Path()
        path_msg.header = Header(stamp=rospy.Time.now(), frame_id=self.frame_id)
        for (px, py) in [(self.current_x, self.current_y),
                         (self.target_x, self.target_y)]:
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = px
            ps.pose.position.y = py
            ps.pose.position.z = self.fixed_z
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)
        self.pub_raw_path.publish(path_msg)
        
    def timer_callback(self, event):
        """Publish interpolated position command at fixed rate"""
        if self.target_x is None or self.target_y is None:
            return
        
        # Compute direction and distance to target
        dx = self.target_x - self.current_x
        dy = self.target_y - self.current_y
        dist = math.sqrt(dx * dx + dy * dy)
        
        # Compute next position along the line
        step_size = self.move_speed / self.publish_rate
        
        if dist > step_size:
            # Move one step toward target
            ratio = step_size / dist
            next_x = self.current_x + dx * ratio
            next_y = self.current_y + dy * ratio
        else:
            # Close enough, go directly to target
            next_x = self.target_x
            next_y = self.target_y
        
        # Build and publish pose
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = self.frame_id
        
        pose.pose.position.x = next_x
        pose.pose.position.y = next_y
        pose.pose.position.z = self.fixed_z
        
        # Orientation: face the target, or keep last yaw when arrived
        if self.face_waypoint and dist > self.arrival_threshold:
            self.last_yaw = math.atan2(dy, dx)
        
        cy = math.cos(self.last_yaw * 0.5)
        sy = math.sin(self.last_yaw * 0.5)
        pose.pose.orientation.z = sy
        pose.pose.orientation.w = cy
        
        self.pub.publish(pose)


def main():
    try:
        node = WaypointToPosCmd()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
