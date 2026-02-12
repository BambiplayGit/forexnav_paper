#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Odometry Path Publisher (ROS1)

Subscribes to /odom and publishes a marker for visualization in rviz.
Uses color gradient to represent speed (red=fast, light yellow=slow).
"""

import rospy
import math
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA


class OdomPathPublisher:
    """Publishes path marker from odometry with speed-based color gradient"""
    
    def __init__(self):
        rospy.init_node('odom_path_publisher')
        
        # Parameters
        self.max_path_length = rospy.get_param('~max_path_length', 2000)
        self.publish_rate = rospy.get_param('~publish_rate', 10.0)
        self.min_distance = rospy.get_param('~min_distance', 0.05)
        self.line_width = rospy.get_param('~line_width', 0.20)  # Marker thickness
        self.max_speed = rospy.get_param('~max_speed', 4.0)  # For color scaling
        self.min_speed = rospy.get_param('~min_speed', 0.0)
        
        # Color settings (RGB)
        # High speed: red (1, 0, 0)
        # Low speed: light yellow (1, 1, 0.6)
        self.color_high = (1.0, 0.0, 0.0)
        self.color_low = (1.0, 1.0, 0.6)
        
        # Path storage: list of (Point, speed)
        self.path_points = []
        self.path_speeds = []
        
        # Last position
        self.last_x = None
        self.last_y = None
        
        # ROS interface
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback, queue_size=10)
        self.marker_pub = rospy.Publisher('/odom_path', Marker, queue_size=10)
        
        # Timer
        timer_period = 1.0 / self.publish_rate
        self.timer = rospy.Timer(rospy.Duration(timer_period), self.publish_path)
        
        rospy.loginfo("Odometry path publisher initialized")
        rospy.loginfo("  Max path length: {}".format(self.max_path_length))
        rospy.loginfo("  Line width: {} m".format(self.line_width))
        rospy.loginfo("  Speed range: [{}, {}] m/s".format(self.min_speed, self.max_speed))
        
    def speed_to_color(self, speed):
        """Convert speed to color (interpolate between low and high color)"""
        # Normalize speed to [0, 1]
        t = (speed - self.min_speed) / (self.max_speed - self.min_speed + 1e-6)
        t = max(0.0, min(1.0, t))
        
        # Interpolate color
        r = self.color_low[0] + t * (self.color_high[0] - self.color_low[0])
        g = self.color_low[1] + t * (self.color_high[1] - self.color_low[1])
        b = self.color_low[2] + t * (self.color_high[2] - self.color_low[2])
        
        return ColorRGBA(r, g, b, 1.0)
        
    def odom_callback(self, msg):
        """Callback for odometry messages"""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        
        # Get speed from twist
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        speed = math.sqrt(vx*vx + vy*vy)
        
        # Filter out points too close to the last one
        if self.last_x is not None:
            dist = ((x - self.last_x)**2 + (y - self.last_y)**2)**0.5
            if dist < self.min_distance:
                return
        
        self.last_x = x
        self.last_y = y
        
        # Add point and speed
        point = Point(x, y, z)
        self.path_points.append(point)
        self.path_speeds.append(speed)
        
        # Limit path length
        if len(self.path_points) > self.max_path_length:
            self.path_points.pop(0)
            self.path_speeds.pop(0)
            
    def publish_path(self, event):
        """Publish the path as marker"""
        if len(self.path_points) < 2:
            return
            
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "odom_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # Line width
        marker.scale.x = self.line_width
        
        # Pose (identity)
        marker.pose.orientation.w = 1.0
        
        # Add points and colors
        marker.points = self.path_points[:]
        marker.colors = [self.speed_to_color(s) for s in self.path_speeds]
        
        self.marker_pub.publish(marker)


def main():
    try:
        publisher = OdomPathPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
