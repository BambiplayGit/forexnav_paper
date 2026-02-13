#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Position Command to Velocity Command Converter with Closed-Loop Control
将Fast-Planner的位置命令转换为四足机器人的速度命令，带闭环位置控制
"""

import rospy
import numpy as np
import tf
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand
from std_srvs.srv import SetBool, SetBoolResponse

class PositionToVelocityController:
    def __init__(self):
        rospy.init_node('position_to_velocity_controller', anonymous=False)
        
        # PID控制器参数（X和Y分开，因为四足机器人Y方向能力弱）
        self.kp_x = rospy.get_param('~kp_x', 2.0)   # X位置比例增益（前进）
        self.kp_y = rospy.get_param('~kp_y', 3.0)   # Y位置比例增益（侧移，需要更大）
        self.ki_x = rospy.get_param('~ki_x', 0.0)   # X积分增益
        self.ki_y = rospy.get_param('~ki_y', 0.3)   # Y积分增益（侧移需要积分补偿）
        self.kd_x = rospy.get_param('~kd_x', 0.0)   # X微分增益
        self.kd_y = rospy.get_param('~kd_y', 0.6)   # Y微分增益（侧移需要更强阻尼）
        
        self.kp_z = rospy.get_param('~kp_z', 0.0)    # Z位置比例增益（四足一般不用）
        self.kp_yaw = rospy.get_param('~kp_yaw', 1.8)  # 航向比例增益
        
        # 四足机器人参数
        self.ignore_z = rospy.get_param('~ignore_z', True)  # 忽略Z轴控制
        
        # 速度限制（四足机器人Y方向能力弱）
        self.max_linear_vel_x = rospy.get_param('~max_linear_vel_x', 1.0)
        self.max_linear_vel_y = rospy.get_param('~max_linear_vel_y', 0.8)
        self.max_angular_vel_z = rospy.get_param('~max_angular_vel_z', 1.2)
        
        # 控制模式
        # 'feedforward': 直接使用规划器的速度
        # 'feedback': 纯反馈控制
        # 'hybrid': 前馈+反馈
        self.control_mode = rospy.get_param('~control_mode', 'hybrid')
        
        # 前馈权重（混合模式下，X和Y可以不同）
        self.feedforward_weight_x = rospy.get_param('~feedforward_weight_x', 0.8)  # X方向跟规划
        self.feedforward_weight_y = rospy.get_param('~feedforward_weight_y', 0.3)  # Y方向更依赖反馈
        self.feedback_weight_x = rospy.get_param('~feedback_weight_x', 0.2)
        self.feedback_weight_y = rospy.get_param('~feedback_weight_y', 0.7)  # Y方向主要用反馈
        
        # 位置容差
        self.position_tolerance = rospy.get_param('~position_tolerance', 0.1)  # 米
        
        # 状态变量
        self.current_odom = None
        self.target_position = None
        self.target_velocity = None
        self.target_yaw = None
        
        # 控制器启用/禁用标志
        self.controller_enabled = True  # 默认启用
        
        # PID状态
        self.error_integral_x = 0.0
        self.error_integral_y = 0.0
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.last_time = None
        
        # 订阅器
        self.odom_sub = rospy.Subscriber(
            rospy.get_param('~odom_topic', '/odom'),
            Odometry,
            self.odom_callback,
            queue_size=1
        )
        
        self.pos_cmd_sub = rospy.Subscriber(
            rospy.get_param('~position_cmd_topic', '/planning/pos_cmd'),
            PositionCommand,
            self.position_cmd_callback,
            queue_size=1
        )
        
        # 发布器
        self.cmd_vel_pub = rospy.Publisher(
            rospy.get_param('~cmd_vel_topic', '/cmd_vel'),
            Twist,
            queue_size=1
        )
        
        # 控制循环定时器
        control_rate = rospy.get_param('~control_rate', 50.0)  # Hz
        self.control_timer = rospy.Timer(
            rospy.Duration(1.0 / control_rate),
            self.control_loop
        )
        
        # 服务：启用/禁用控制器
        self.enable_service = rospy.Service(
            '~set_enabled',
            SetBool,
            self.set_enabled_callback
        )
        
        rospy.loginfo("Position to Velocity Controller initialized")
        rospy.loginfo("Control mode: %s", self.control_mode)
        rospy.loginfo("PID gains X - kp: %.2f, ki: %.2f, kd: %.2f", 
                     self.kp_x, self.ki_x, self.kd_x)
        rospy.loginfo("PID gains Y - kp: %.2f, ki: %.2f, kd: %.2f", 
                     self.kp_y, self.ki_y, self.kd_y)
        rospy.loginfo("Feedforward weight - X: %.2f, Y: %.2f", 
                     self.feedforward_weight_x, self.feedforward_weight_y)
        rospy.loginfo("Controller enabled: %s", self.controller_enabled)
    
    def set_enabled_callback(self, req):
        """服务回调：启用/禁用控制器"""
        self.controller_enabled = req.data
        if self.controller_enabled:
            rospy.loginfo("Position controller ENABLED")
            # 重置PID状态，避免积分累积
            self.error_integral_x = 0.0
            self.error_integral_y = 0.0
            self.prev_error_x = 0.0
            self.prev_error_y = 0.0
        else:
            rospy.loginfo("Position controller DISABLED")
            # 停止发布速度命令
            self.publish_zero_velocity()
        
        return SetBoolResponse(success=True, message="Controller %s" % 
                              ("enabled" if self.controller_enabled else "disabled"))
    
    def publish_zero_velocity(self):
        """发布零速度命令"""
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.linear.y = 0.0
        cmd_vel.linear.z = 0.0
        cmd_vel.angular.x = 0.0
        cmd_vel.angular.y = 0.0
        cmd_vel.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_vel)
    
    def odom_callback(self, msg):
        """接收里程计信息"""
        self.current_odom = msg
    
    def position_cmd_callback(self, msg):
        """接收位置命令"""
        self.target_position = np.array([msg.position.x, msg.position.y, msg.position.z])
        self.target_velocity = np.array([msg.velocity.x, msg.velocity.y, msg.velocity.z])
        self.target_yaw = msg.yaw
        # 根据 velocity 设置 yaw
        # vel_xy = self.target_velocity[:2]
        # if np.linalg.norm(vel_xy) > 1e-3:  # 避免除0
        #     self.target_yaw = np.arctan2(vel_xy[1], vel_xy[0])
    
    def get_current_pose(self):
        """从里程计获取当前位姿"""
        if self.current_odom is None:
            return None, None, None
        
        pos = self.current_odom.pose.pose.position
        current_pos = np.array([pos.x, pos.y, pos.z])
        
        # 获取当前速度（注意：odom的twist是在child_frame_id坐标系下，通常是base_link）
        vel = self.current_odom.twist.twist.linear
        current_vel = np.array([vel.x, vel.y, vel.z])
        
        # 获取当前偏航角
        quaternion = (
            self.current_odom.pose.pose.orientation.x,
            self.current_odom.pose.pose.orientation.y,
            self.current_odom.pose.pose.orientation.z,
            self.current_odom.pose.pose.orientation.w
        )
        euler = tf.transformations.euler_from_quaternion(quaternion)
        current_yaw = euler[2]
        
        return current_pos, current_vel, current_yaw
    
    def compute_feedback_velocity(self, dt):
        """计算基于位置误差的反馈速度（世界坐标系）"""
        current_pos, current_vel, current_yaw = self.get_current_pose()
        
        if current_pos is None or self.target_position is None:
            return np.array([0.0, 0.0, 0.0])
        
        # 位置误差（世界坐标系）
        error = self.target_position - current_pos
        
        # 四足机器人：忽略Z轴误差
        if self.ignore_z:
            error[2] = 0.0
        
        # 如果误差很小，停止积分
        if np.linalg.norm(error[:2]) < self.position_tolerance:
            self.error_integral_x = 0.0
            self.error_integral_y = 0.0
        else:
            # 积分项（带抗饱和）
            max_integral = 0.5  # 限制积分项的最大值
            self.error_integral_x += error[0] * dt
            self.error_integral_y += error[1] * dt
            self.error_integral_x = np.clip(self.error_integral_x, -max_integral, max_integral)
            self.error_integral_y = np.clip(self.error_integral_y, -max_integral, max_integral)
        
        # 微分项
        if dt > 0:
            error_derivative_x = (error[0] - self.prev_error_x) / dt
            error_derivative_y = (error[1] - self.prev_error_y) / dt
        else:
            error_derivative_x = 0.0
            error_derivative_y = 0.0
        
        # 保存当前误差
        self.prev_error_x = error[0]
        self.prev_error_y = error[1]
        
        # PID控制律（X和Y使用不同参数）
        feedback_vel_x = (self.kp_x * error[0] + 
                         self.ki_x * self.error_integral_x + 
                         self.kd_x * error_derivative_x)
        
        feedback_vel_y = (self.kp_y * error[1] + 
                         self.ki_y * self.error_integral_y + 
                         self.kd_y * error_derivative_y)
        
        feedback_vel = np.array([feedback_vel_x, feedback_vel_y, 0.0])
        
        return feedback_vel
    
    def compute_yaw_control(self, current_yaw):
        """计算偏航角控制（自适应版本）
        
        核心思想：当位置误差大时，降低yaw跟踪强度，避免"yaw转过去了但xy还没到"的问题。
        这样可以防止提前转向导致的Y方向偏差累积。
        """
        if self.target_yaw is None or current_yaw is None:
            return 0.0
        
        # 偏航角误差（处理角度环绕）
        yaw_error = self.target_yaw - current_yaw
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
        
        # 自适应yaw增益：根据位置误差动态调整
        adaptive_kp_yaw = self.kp_yaw
        
        if self.current_odom is not None and self.target_position is not None:
            current_pos, _, _ = self.get_current_pose()
            if current_pos is not None:
                # 计算位置误差
                pos_error = self.target_position[:2] - current_pos[:2]
                pos_error_norm = np.linalg.norm(pos_error)
                
                # 位置误差越大，yaw增益越小（让位置先追上来）
                # 使用平滑的衰减函数，而不是突变
                if pos_error_norm > 0.15:  # 误差>15cm时开始衰减
                    # 衰减因子：误差0.15m时为1.0，误差0.5m时约为0.5，误差1.0m时约为0.3
                    decay_factor = 0.15 / (pos_error_norm + 0.05)
                    decay_factor = np.clip(decay_factor, 0.3, 1.0)  # 限制在[0.3, 1.0]
                    adaptive_kp_yaw = self.kp_yaw * decay_factor
                    
                    # 调试输出
                    if rospy.get_param('~debug', False):
                        rospy.loginfo_throttle(2.0, 
                            "Adaptive yaw control: pos_error=%.3f m, kp_yaw=%.2f->%.2f (decay=%.2f)",
                            pos_error_norm, self.kp_yaw, adaptive_kp_yaw, decay_factor)
        
        # 使用自适应增益计算角速度
        angular_vel = adaptive_kp_yaw * yaw_error
        
        return angular_vel
    
    def world_to_body_frame(self, world_vel, yaw):
        """将世界坐标系速度转换到机体坐标系"""
        # 标准旋转矩阵转换
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        body_vel_x = cos_yaw * world_vel[0] + sin_yaw * world_vel[1]
        body_vel_y = -sin_yaw * world_vel[0] + cos_yaw * world_vel[1]
        
        return np.array([body_vel_x, body_vel_y, world_vel[2]])
    
    def control_loop(self, event):
        """主控制循环"""
        # 如果控制器被禁用，不执行控制
        if not self.controller_enabled:
            return
        
        if self.current_odom is None or self.target_position is None:
            return
        
        # 计算时间间隔
        current_time = rospy.Time.now()
        if self.last_time is None:
            self.last_time = current_time
            return
        
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time
        
        if dt <= 0 or dt > 1.0:  # 防止异常时间间隔
            return
        
        # 获取当前位姿
        current_pos, current_vel, current_yaw = self.get_current_pose()
        if current_pos is None:
            return
        
        # 根据控制模式计算速度命令
        if self.control_mode == 'feedforward':
            # 纯前馈：直接使用规划器的速度
            desired_vel_world = self.target_velocity
            
        elif self.control_mode == 'feedback':
            # 纯反馈：基于位置误差
            desired_vel_world = self.compute_feedback_velocity(dt)
            
        else:  # hybrid
            # 混合：前馈 + 反馈（X和Y使用不同权重）
            feedforward_vel = self.target_velocity
            feedback_vel = self.compute_feedback_velocity(dt)
            
            # X方向：更多前馈（跟随规划速度）
            # Y方向：更多反馈（补偿侧移能力弱的问题）
            desired_vel_world = np.array([
                self.feedforward_weight_x * feedforward_vel[0] + self.feedback_weight_x * feedback_vel[0],
                self.feedforward_weight_y * feedforward_vel[1] + self.feedback_weight_y * feedback_vel[1],
                0.0
            ])
        
        # 转换到机体坐标系
        desired_vel_body = self.world_to_body_frame(desired_vel_world, current_yaw)
        
        # 计算偏航角速度
        angular_vel_z = self.compute_yaw_control(current_yaw)
        
        # 速度限幅
        desired_vel_body[0] = np.clip(desired_vel_body[0], 
                                     -self.max_linear_vel_x, 
                                     self.max_linear_vel_x)
        desired_vel_body[1] = np.clip(desired_vel_body[1], 
                                     -self.max_linear_vel_y, 
                                     self.max_linear_vel_y)
        angular_vel_z = np.clip(angular_vel_z, 
                               -self.max_angular_vel_z, 
                               self.max_angular_vel_z)
        
        # 发布速度命令
        cmd_vel = Twist()
        cmd_vel.linear.x = desired_vel_body[0]
        cmd_vel.linear.y = desired_vel_body[1]
        cmd_vel.linear.z = 0.0  # 四足机器人通常不控制Z
        cmd_vel.angular.x = 0.0
        cmd_vel.angular.y = 0.0
        cmd_vel.angular.z = angular_vel_z
        
        self.cmd_vel_pub.publish(cmd_vel)
        
        # 调试信息
        if rospy.get_param('~debug', False):
            pos_error = np.linalg.norm(self.target_position[:2] - current_pos[:2])
            rospy.loginfo_throttle(1.0, 
                "Pos error: %.3f m, Vel cmd: (%.2f, %.2f) m/s, Yaw: %.2f rad/s",
                pos_error, cmd_vel.linear.x, cmd_vel.linear.y, cmd_vel.angular.z)
    
    def run(self):
        """运行节点"""
        rospy.spin()

if __name__ == '__main__':
    try:
        controller = PositionToVelocityController()
        controller.run()
    except rospy.ROSInterruptException:
        pass


