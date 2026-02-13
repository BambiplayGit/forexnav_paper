#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手柄遥操作四足机器人 (仿照 CHAMP teleop)

将 sensor_msgs/Joy 消息转换为 geometry_msgs/Twist，发布到 /cmd_vel。
/cmd_vel 为 body frame (base_link): x=前进, y=左平移, angular.z=左转

支持：
  - 左摇杆：前进/后退 + 左右平移
  - 右摇杆：左右旋转
  - 按键调速（加速/减速）
  - 死区过滤
  - 急停按键

典型手柄映射 (Xbox / Logitech F710):
  左摇杆 Y轴 (axis 1): 前进/后退
  左摇杆 X轴 (axis 0): 左右平移
  右摇杆 X轴 (axis 3): 旋转
  A 按键 (button 0):  急停 (松开恢复)
  LB 按键 (button 4): 减速档
  RB 按键 (button 5): 加速档

所有轴和按键映射均可通过 ROS 参数配置。
"""

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy


class JoyTeleop:
    """手柄遥操作四足机器人节点"""

    def __init__(self):
        rospy.init_node("joy_teleop")

        # ===== 速度限制 =====
        self.max_linear_vel = rospy.get_param("~max_linear_vel", 1.5)   # m/s
        self.max_lateral_vel = rospy.get_param("~max_lateral_vel", 0.5)  # m/s (四足平移较慢)
        self.max_angular_vel = rospy.get_param("~max_angular_vel", 1.5)  # rad/s

        # ===== 摇杆轴映射 =====
        self.axis_linear = rospy.get_param("~axis_linear", 1)     # 左摇杆 Y轴
        self.axis_lateral = rospy.get_param("~axis_lateral", 0)    # 左摇杆 X轴
        self.axis_angular = rospy.get_param("~axis_angular", 3)    # 右摇杆 X轴

        # ===== 按键映射 =====
        self.btn_estop = rospy.get_param("~btn_estop", 0)         # A: 急停
        self.btn_speed_down = rospy.get_param("~btn_speed_down", 4)  # LB: 减速
        self.btn_speed_up = rospy.get_param("~btn_speed_up", 5)      # RB: 加速

        # ===== 死区 =====
        self.deadzone = rospy.get_param("~deadzone", 0.1)

        # ===== 速度档位 =====
        self.speed_levels = rospy.get_param("~speed_levels", [0.3, 0.5, 0.7, 1.0])
        self.current_speed_idx = rospy.get_param("~default_speed_idx", 1)  # 默认中低速
        self.current_speed_idx = max(0, min(self.current_speed_idx, len(self.speed_levels) - 1))

        # ===== 发布频率 =====
        self.publish_rate = rospy.get_param("~publish_rate", 30.0)  # Hz
        self.cmd_vel_timeout = rospy.get_param("~cmd_vel_timeout", 0.5)  # 超时停止 s

        # 状态
        self.estop_active = False
        self.last_joy_time = None
        self.latest_twist = Twist()

        # 防抖：记录上次按键状态
        self._prev_btn_speed_down = 0
        self._prev_btn_speed_up = 0

        # ROS 接口
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.joy_sub = rospy.Subscriber("/joy", Joy, self._joy_callback, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self._publish_cmd_vel)

        rospy.loginfo("===== 手柄遥操作 (Joy Teleop) 已启动 =====")
        rospy.loginfo("  最大速度: linear=%.1f m/s, lateral=%.1f m/s, angular=%.1f rad/s",
                      self.max_linear_vel, self.max_lateral_vel, self.max_angular_vel)
        rospy.loginfo("  速度档位: %s (当前: %.0f%%)",
                      ["{:.0f}%".format(s * 100) for s in self.speed_levels],
                      self.speed_levels[self.current_speed_idx] * 100)
        rospy.loginfo("  摇杆映射: linear=axis%d, lateral=axis%d, angular=axis%d",
                      self.axis_linear, self.axis_lateral, self.axis_angular)
        rospy.loginfo("  按键映射: 急停=btn%d, 减速=btn%d, 加速=btn%d",
                      self.btn_estop, self.btn_speed_down, self.btn_speed_up)
        rospy.loginfo("  死区: %.2f", self.deadzone)

    def _apply_deadzone(self, value):
        """应用死区过滤"""
        if abs(value) < self.deadzone:
            return 0.0
        # 重映射: deadzone~1.0 → 0.0~1.0
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - self.deadzone) / (1.0 - self.deadzone)

    def _joy_callback(self, msg):
        """手柄消息回调"""
        self.last_joy_time = rospy.Time.now()

        # ---- 急停按键 ----
        if len(msg.buttons) > self.btn_estop:
            self.estop_active = bool(msg.buttons[self.btn_estop])
            if self.estop_active:
                self.latest_twist = Twist()  # 清零
                return

        # ---- 速度调档 (上升沿触发) ----
        if len(msg.buttons) > self.btn_speed_down:
            btn_down = msg.buttons[self.btn_speed_down]
            if btn_down and not self._prev_btn_speed_down:
                self.current_speed_idx = max(0, self.current_speed_idx - 1)
                rospy.loginfo("速度档位: %d/%d (%.0f%%)",
                              self.current_speed_idx + 1, len(self.speed_levels),
                              self.speed_levels[self.current_speed_idx] * 100)
            self._prev_btn_speed_down = btn_down

        if len(msg.buttons) > self.btn_speed_up:
            btn_up = msg.buttons[self.btn_speed_up]
            if btn_up and not self._prev_btn_speed_up:
                self.current_speed_idx = min(len(self.speed_levels) - 1, self.current_speed_idx + 1)
                rospy.loginfo("速度档位: %d/%d (%.0f%%)",
                              self.current_speed_idx + 1, len(self.speed_levels),
                              self.speed_levels[self.current_speed_idx] * 100)
            self._prev_btn_speed_up = btn_up

        # ---- 摇杆 → 速度 (body frame) ----
        speed_scale = self.speed_levels[self.current_speed_idx]

        linear_raw = 0.0
        lateral_raw = 0.0
        angular_raw = 0.0

        if len(msg.axes) > self.axis_linear:
            linear_raw = self._apply_deadzone(msg.axes[self.axis_linear])
        if len(msg.axes) > self.axis_lateral:
            lateral_raw = self._apply_deadzone(msg.axes[self.axis_lateral])
        if len(msg.axes) > self.axis_angular:
            angular_raw = self._apply_deadzone(msg.axes[self.axis_angular])

        twist = Twist()
        twist.linear.x = linear_raw * self.max_linear_vel * speed_scale   # 前进/后退
        twist.linear.y = lateral_raw * self.max_lateral_vel * speed_scale  # 左/右平移
        twist.angular.z = angular_raw * self.max_angular_vel * speed_scale # 左/右旋转

        self.latest_twist = twist

    def _publish_cmd_vel(self, event):
        """定时发布 /cmd_vel"""
        # 急停
        if self.estop_active:
            self.cmd_vel_pub.publish(Twist())
            return

        # 超时保护: 长时间没有 Joy 消息则停止
        if self.last_joy_time is not None:
            elapsed = (rospy.Time.now() - self.last_joy_time).to_sec()
            if elapsed > self.cmd_vel_timeout:
                self.cmd_vel_pub.publish(Twist())
                return

        self.cmd_vel_pub.publish(self.latest_twist)


def main():
    try:
        node = JoyTeleop()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
