#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quadruped Robot Odometry Simulator (ROS1)

This node simulates a quadruped robot's motion characteristics:
- Gait-induced body oscillation (periodic sway during walking)
- Response delay from leg kinematics
- Velocity fluctuation during gait transitions
- Speed reduction during turning
- Realistic second-order dynamics with PD control

Features:
- Quadruped-like motion dynamics
- Configurable gait parameters
- Body sway simulation
- Turn-speed coupling
- Publishes odom and TF transforms
"""

import rospy
import math
import random
import tf2_ros
from geometry_msgs.msg import TransformStamped, PoseStamped, Twist
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand


class SimpleOdomSimulator:
    """Quadruped robot odometry simulator with realistic dynamics"""
    
    def __init__(self):
        rospy.init_node('simple_odom_simulator')
        
        # Get parameters
        self.init_x = rospy.get_param('~init_x', 0.0)
        self.init_y = rospy.get_param('~init_y', 0.0)
        self.init_z = rospy.get_param('~init_z', 1.0)
        self.init_yaw = rospy.get_param('~init_yaw', 0.0)
        self.publish_freq = rospy.get_param('~publish_frequency', 200.0)
        
        # Dynamics parameters (控制器余量 > 规划器约束，给 PD 校正留空间)
        self.max_vel = rospy.get_param('~max_vel', 2.0)           # 最大速度 m/s
        self.max_acc = rospy.get_param('~max_acc', 4.0)           # 最大加速度 m/s^2 (规划器用2.0，控制器留2倍余量)
        self.max_yaw_rate = rospy.get_param('~max_yaw_rate', 2.09) # 最大角速度 rad/s
        self.max_yaw_acc = rospy.get_param('~max_yaw_acc', 4.0)   # 最大角加速度 rad/s^2
        
        # PD controller gains (控制器增益，参考 SO3 控制器)
        self.kp_pos = rospy.get_param('~kp_pos', 5.0)    # 位置P增益
        self.kd_pos = rospy.get_param('~kd_pos', 4.0)    # 位置D增益 (阻尼，防冲过头)
        self.kp_yaw = rospy.get_param('~kp_yaw', 4.0)    # 航向P增益
        self.kd_yaw = rospy.get_param('~kd_yaw', 3.0)    # 航向D增益
        
        # Feedforward scaling (前馈缩放，留余量给 PD 校正)
        self.ff_acc_scale = rospy.get_param('~ff_acc_scale', 0.8)  # 加速度前馈缩放 (0~1)
        
        # ===== 四足机器人特有参数 =====
        # 步态参数
        self.gait_freq = rospy.get_param('~gait_freq', 2.0)        # 步态频率 Hz (与 gait_joint_publisher 一致)
        self.body_sway_amp = rospy.get_param('~body_sway_amp', 0.008)  # 身体左右摆动幅度 m
        self.body_bob_amp = rospy.get_param('~body_bob_amp', 0.005)    # 身体上下起伏幅度 m
        self.yaw_sway_amp = rospy.get_param('~yaw_sway_amp', 0.01)     # 航向微小摆动 rad
        self.yaw_step_amp = rospy.get_param('~yaw_step_amp', 0.03)     # 步态同步旋转阶梯幅度 rad
        
        # 四足特性
        self.turn_speed_factor = rospy.get_param('~turn_speed_factor', 0.95)  # 转弯时速度衰减系数
        self.response_delay = rospy.get_param('~response_delay', 0.01)        # 响应延迟 s
        self.vel_noise_std = rospy.get_param('~vel_noise_std', 0.0)          # 速度噪声(0=无侧滑)
        self.angular_smooth_alpha = rospy.get_param('~angular_smooth_alpha', 0.5)  # 角速度平滑系数(0~1,越大越快跟随)
        
        # Current state (actual robot state)
        self.pos_x = self.init_x
        self.pos_y = self.init_y
        self.pos_z = self.init_z
        self.yaw = self.init_yaw
        
        # Velocity state
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.omega_z = 0.0
        
        # Delayed velocity (模拟腿部运动学延迟)
        self.delayed_vel_x = 0.0
        self.delayed_vel_y = 0.0
        self.delayed_omega_z = 0.0
        self.prev_omega_z = 0.0  # 用于角速度平滑
        
        # Target state (from pos_cmd)
        self.target_x = self.init_x
        self.target_y = self.init_y
        self.target_yaw = self.init_yaw
        
        # 目标速度和加速度（直接从 PositionCommand 获取，用于前馈控制）
        self.target_vel_x = 0.0
        self.target_vel_y = 0.0
        self.target_acc_x = 0.0
        self.target_acc_y = 0.0
        self.target_yaw_dot = 0.0
        self.use_position_command = False  # True when receiving PositionCommand
        
        # Commanded velocity from cmd_vel (for velocity control mode)
        self.cmd_vel_x = 0.0
        self.cmd_vel_y = 0.0
        self.cmd_omega_z = 0.0
        
        # Timing
        self.last_cmd_vel_time = None
        self.last_pos_cmd_time = None
        self.pos_cmd_timeout = 0.3  # seconds
        self.sim_time = 0.0  # 仿真时间，用于步态相位
        
        # Setup ROS interface
        self.setup_ros_interface()
        
        rospy.loginfo("Quadruped Odom Simulator initialized")
        rospy.loginfo("  Initial position: ({:.2f}, {:.2f}, {:.2f})".format(
            self.init_x, self.init_y, self.init_z))
        rospy.loginfo("  Dynamics: max_vel={:.1f} m/s, max_acc={:.1f} m/s^2".format(
            self.max_vel, self.max_acc))
        rospy.loginfo("  Gait: freq={:.1f}Hz, sway={:.3f}m, bob={:.3f}m".format(
            self.gait_freq, self.body_sway_amp, self.body_bob_amp))
        rospy.loginfo("  Publish frequency: {} Hz".format(self.publish_freq))
        
    def setup_ros_interface(self):
        """Setup ROS publishers and subscribers"""
        # Subscribe to position command (PositionCommand with p/v/a/yaw/yaw_dot)
        self.pos_cmd_sub = rospy.Subscriber(
            '/planning/pos_cmd', 
            PositionCommand, 
            self.position_cmd_callback,
            queue_size=1
        )
        
        # Subscribe to velocity command
        self.cmd_vel_sub = rospy.Subscriber(
            '/cmd_vel',
            Twist,
            self.cmd_vel_callback,
            queue_size=1
        )
        
        # Publishers
        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=10)
        # cmd_vel for Champ leg animation - velocity in body frame (must match odom motion!)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.publish_cmd_vel = rospy.get_param('~publish_cmd_vel', False)
        self.publish_base_footprint = rospy.get_param('~publish_base_footprint', False)  # leg mode: robot needs base_footprint
        
        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # Timer for dynamics update and publishing
        timer_period = 1.0 / self.publish_freq
        self.timer = rospy.Timer(rospy.Duration(timer_period), self.update_and_publish)
        
        rospy.loginfo("ROS interface setup complete")
        
    def cmd_vel_callback(self, msg):
        """Callback for velocity commands"""
        self.cmd_vel_x = msg.linear.x
        self.cmd_vel_y = msg.linear.y
        self.cmd_omega_z = msg.angular.z
        self.last_cmd_vel_time = rospy.Time.now()
        
    def position_cmd_callback(self, msg):
        """Callback for PositionCommand - extract position, velocity, acceleration, yaw, yaw_dot"""
        self.target_x = msg.position.x
        self.target_y = msg.position.y
        
        # Directly use trajectory velocity (no more noisy differentiation)
        self.target_vel_x = msg.velocity.x
        self.target_vel_y = msg.velocity.y
        
        # Directly use trajectory acceleration (feedforward)
        self.target_acc_x = msg.acceleration.x
        self.target_acc_y = msg.acceleration.y
        
        # Yaw and yaw rate
        self.target_yaw = msg.yaw
        self.target_yaw_dot = msg.yaw_dot
        
        self.use_position_command = True
        self.last_pos_cmd_time = rospy.Time.now()
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
    
    def clamp(self, value, min_val, max_val):
        """Clamp value to [min_val, max_val]"""
        return max(min_val, min(max_val, value))
    
    def update_and_publish(self, event):
        """Update dynamics and publish odometry"""
        current_time = rospy.Time.now()
        dt = 1.0 / self.publish_freq
        self.sim_time += dt
        
        # Determine control mode
        pos_cmd_active = False
        if self.last_pos_cmd_time is not None:
            time_since_pos_cmd = (current_time - self.last_pos_cmd_time).to_sec()
            if time_since_pos_cmd < self.pos_cmd_timeout:
                pos_cmd_active = True
        
        if pos_cmd_active:
            # Position tracking mode with PD controller
            self.update_position_tracking(dt)
        elif self.last_cmd_vel_time is not None:
            # Velocity control mode
            time_since_cmd = (current_time - self.last_cmd_vel_time).to_sec()
            if time_since_cmd < 0.5:
                self.update_velocity_control(dt)
            else:
                # No command, decelerate to stop
                self.decelerate_to_stop(dt)
        else:
            # No command, decelerate to stop
            self.decelerate_to_stop(dt)
        
        # Apply quadruped-specific effects
        self.apply_quadruped_dynamics(dt)
        
        # Publish odometry and TF
        self.publish_odometry_and_tf(current_time)
    
    def update_position_tracking(self, dt):
        """PD + feedforward controller for position tracking (inspired by UAV SO3 controller)
        
        Control law: acc = kp*(p_des - p) + kd*(v_des - v) + a_des
        This mirrors the UAV's geometric controller but adapted for 2D ground robot.
        """
        # Position error
        err_x = self.target_x - self.pos_x
        err_y = self.target_y - self.pos_y
        
        # Yaw error with wrap-around
        err_yaw = self.normalize_angle(self.target_yaw - self.yaw)
        
        # 四足特性：转弯时降低最大速度
        turn_rate = abs(self.omega_z)
        turn_penalty = 1.0 - (1.0 - self.turn_speed_factor) * min(1.0, turn_rate / self.max_yaw_rate)
        effective_max_vel = self.max_vel * turn_penalty
        
        # PD + scaled acceleration feedforward (like UAV: kp*(p_des-p) + kv*(v_des-v) + ff*a_des)
        # ff_acc_scale < 1.0 reserves headroom for PD corrections within max_acc budget
        acc_x = (self.kp_pos * err_x
                 + self.kd_pos * (self.target_vel_x - self.vel_x)
                 + self.ff_acc_scale * self.target_acc_x)
        acc_y = (self.kp_pos * err_y
                 + self.kd_pos * (self.target_vel_y - self.vel_y)
                 + self.ff_acc_scale * self.target_acc_y)
        
        # Limit acceleration magnitude
        acc_mag = math.sqrt(acc_x * acc_x + acc_y * acc_y)
        if acc_mag > self.max_acc:
            scale = self.max_acc / acc_mag
            acc_x *= scale
            acc_y *= scale
        
        # Update velocity with acceleration
        self.vel_x += acc_x * dt
        self.vel_y += acc_y * dt
        
        # Limit velocity magnitude (考虑转弯惩罚)
        vel_mag = math.sqrt(self.vel_x * self.vel_x + self.vel_y * self.vel_y)
        if vel_mag > effective_max_vel:
            scale = effective_max_vel / vel_mag
            self.vel_x *= scale
            self.vel_y *= scale
        
        # Update position using delayed velocity (模拟响应延迟)
        self.pos_x += self.delayed_vel_x * dt
        self.pos_y += self.delayed_vel_y * dt
        
        # Yaw: PD + yaw_dot feedforward
        yaw_acc = (self.kp_yaw * err_yaw
                   + self.kd_yaw * (self.target_yaw_dot - self.omega_z))
        yaw_acc = self.clamp(yaw_acc, -self.max_yaw_acc, self.max_yaw_acc)
        
        # Update angular velocity
        self.omega_z += yaw_acc * dt
        self.omega_z = self.clamp(self.omega_z, -self.max_yaw_rate, self.max_yaw_rate)
        
        # Update yaw using delayed omega
        self.yaw += self.delayed_omega_z * dt
        self.yaw = self.normalize_angle(self.yaw)
    
    def update_velocity_control(self, dt):
        """Velocity control mode with quadruped dynamics
        
        /cmd_vel 是 body frame (base_link): x=前进, y=左平移, angular.z=左转
        内部 vel_x/vel_y 是 world frame，这里做 body → world 转换
        """
        # 四足特性：转弯时降低最大速度
        turn_rate = abs(self.omega_z)
        turn_penalty = 1.0 - (1.0 - self.turn_speed_factor) * min(1.0, turn_rate / self.max_yaw_rate)
        effective_max_vel = self.max_vel * turn_penalty
        
        # Clamp body-frame cmd_vel
        body_vx = self.clamp(self.cmd_vel_x, -effective_max_vel, effective_max_vel)
        body_vy = self.clamp(self.cmd_vel_y, -effective_max_vel, effective_max_vel)
        target_omega = self.clamp(self.cmd_omega_z, -self.max_yaw_rate, self.max_yaw_rate)
        
        # Body frame → World frame
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        target_vel_x = cos_yaw * body_vx - sin_yaw * body_vy
        target_vel_y = sin_yaw * body_vx + cos_yaw * body_vy
        
        # Accelerate towards target velocity with limits
        vel_err_x = target_vel_x - self.vel_x
        vel_err_y = target_vel_y - self.vel_y
        omega_err = target_omega - self.omega_z
        
        # Limit acceleration
        acc_x = self.clamp(vel_err_x / dt, -self.max_acc, self.max_acc)
        acc_y = self.clamp(vel_err_y / dt, -self.max_acc, self.max_acc)
        yaw_acc = self.clamp(omega_err / dt, -self.max_yaw_acc, self.max_yaw_acc)
        
        # Update velocities
        self.vel_x += acc_x * dt
        self.vel_y += acc_y * dt
        self.omega_z += yaw_acc * dt
        
        # Update pose using delayed velocity
        self.pos_x += self.delayed_vel_x * dt
        self.pos_y += self.delayed_vel_y * dt
        self.yaw += self.delayed_omega_z * dt
        self.yaw = self.normalize_angle(self.yaw)
    
    def decelerate_to_stop(self, dt):
        """Smoothly decelerate to stop"""
        # Decelerate linear velocity
        vel_mag = math.sqrt(self.vel_x * self.vel_x + self.vel_y * self.vel_y)
        if vel_mag > 0.001:
            decel = min(self.max_acc * dt, vel_mag)
            scale = (vel_mag - decel) / vel_mag
            self.vel_x *= scale
            self.vel_y *= scale
        else:
            self.vel_x = 0.0
            self.vel_y = 0.0
        
        # Decelerate angular velocity
        if abs(self.omega_z) > 0.001:
            decel = min(self.max_yaw_acc * dt, abs(self.omega_z))
            if self.omega_z > 0:
                self.omega_z -= decel
            else:
                self.omega_z += decel
        else:
            self.omega_z = 0.0
        
        # Update pose using delayed velocity
        self.pos_x += self.delayed_vel_x * dt
        self.pos_y += self.delayed_vel_y * dt
        self.yaw += self.delayed_omega_z * dt
        self.yaw = self.normalize_angle(self.yaw)
    
    def apply_quadruped_dynamics(self, dt):
        """Apply quadruped-specific motion characteristics"""
        # 1. 角速度平滑 (alpha 越大越快跟随，不再过度平滑)
        omega_smooth = self.prev_omega_z + self.angular_smooth_alpha * (self.omega_z - self.prev_omega_z)
        self.prev_omega_z = omega_smooth

        # 2. 速度延迟滤波 (模拟腿部运动学延迟)
        delay_alpha = dt / (self.response_delay + dt)  # 一阶滤波
        self.delayed_vel_x += delay_alpha * (self.vel_x - self.delayed_vel_x)
        self.delayed_vel_y += delay_alpha * (self.vel_y - self.delayed_vel_y)
        self.delayed_omega_z += delay_alpha * (omega_smooth - self.delayed_omega_z)
        
        # 3. 速度噪声 (默认0，避免侧滑)
        if self.vel_noise_std > 0:
            vel_mag = math.sqrt(self.delayed_vel_x**2 + self.delayed_vel_y**2)
            if vel_mag > 0.1:
                noise_scale = min(1.0, vel_mag / self.max_vel)
                self.delayed_vel_x += random.gauss(0, self.vel_noise_std * noise_scale)
                self.delayed_vel_y += random.gauss(0, self.vel_noise_std * noise_scale)
        
    def publish_odometry_and_tf(self, current_time):
        """Publish odometry (smooth, for planner) and TF (stepped, for visualization)"""
        # 计算步态相位
        gait_phase = self.sim_time * self.gait_freq * 2.0 * math.pi
        
        # 计算当前速度大小 (用于调节摆动幅度)
        vel_mag = math.sqrt(self.delayed_vel_x**2 + self.delayed_vel_y**2)
        motion_factor = min(1.0, vel_mag / (self.max_vel * 0.5))  # 速度越快摆动越明显
        
        # 步态引起的身体摆动 (轻微)
        sway_offset = self.body_sway_amp * math.sin(gait_phase) * motion_factor
        bob_offset = self.body_bob_amp * math.sin(2.0 * gait_phase) * motion_factor
        yaw_sway = self.yaw_sway_amp * math.sin(gait_phase) * motion_factor

        sway_x = -sway_offset * math.sin(self.yaw)
        sway_y = sway_offset * math.cos(self.yaw)
        
        pub_x = self.pos_x + sway_x
        pub_y = self.pos_y + sway_y
        pub_z = self.pos_z + bob_offset
        
        # ===== Odom: 平滑 yaw (给规划器用) =====
        odom_yaw = self.yaw + yaw_sway
        
        # ===== TF: 阶梯式旋转 (给 RViz 可视化) =====
        # 步态同步阶梯旋转: trot 每个 stride 有 2 次"推"(每对对角腿推一次)
        # 每半周期: 先保持，后跳变 → 视觉上一顿一顿
        turn_factor = min(1.0, abs(self.delayed_omega_z) / 1.0)
        if turn_factor > 0.02:
            # half_phase: 每半个步态周期 0->1
            half_phase = (gait_phase / math.pi) % 1.0
            # 前 60% 保持, 后 40% 快速过渡 (smoothstep)
            if half_phase < 0.6:
                step_ease = 0.0
            else:
                te = (half_phase - 0.6) / 0.4
                step_ease = te * te * (3.0 - 2.0 * te)  # smoothstep
            # 零均值偏移: (step_ease - 0.5) 范围 [-0.5, 0.5]
            yaw_step = self.yaw_step_amp * turn_factor * (step_ease - 0.5) * 2.0
            yaw_step *= math.copysign(1.0, self.delayed_omega_z)
        else:
            yaw_step = 0.0
        
        tf_yaw = self.yaw + yaw_sway + yaw_step
        
        # ===== Publish odometry (smooth) =====
        odom = Odometry()
        odom.header.stamp = current_time
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        
        odom.pose.pose.position.x = pub_x
        odom.pose.pose.position.y = pub_y
        odom.pose.pose.position.z = pub_z
        
        cy_odom = math.cos(odom_yaw * 0.5)
        sy_odom = math.sin(odom_yaw * 0.5)
        odom.pose.pose.orientation.x = 0.0
        odom.pose.pose.orientation.y = 0.0
        odom.pose.pose.orientation.z = sy_odom
        odom.pose.pose.orientation.w = cy_odom
        
        # 发布实际速度 (带延迟的)
        odom.twist.twist.linear.x = self.delayed_vel_x
        odom.twist.twist.linear.y = self.delayed_vel_y
        odom.twist.twist.linear.z = 0.0
        odom.twist.twist.angular.z = self.delayed_omega_z
        
        self.odom_pub.publish(odom)
        
        # ===== Publish TF (stepped yaw for visual) =====
        # When publish_base_footprint or publish_cmd_vel: odom->base_footprint
        # When not: odom->base_link for backward compatibility
        use_footprint = self.publish_base_footprint or self.publish_cmd_vel
        cy_tf = math.cos(tf_yaw * 0.5)
        sy_tf = math.sin(tf_yaw * 0.5)
        t = TransformStamped()
        t.header.stamp = current_time
        t.header.frame_id = "odom"
        t.child_frame_id = "base_footprint" if use_footprint else "base_link"
        t.transform.translation.x = pub_x
        t.transform.translation.y = pub_y
        t.transform.translation.z = pub_z
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = sy_tf
        t.transform.rotation.w = cy_tf
        self.tf_broadcaster.sendTransform(t)
        
        # Publish cmd_vel for Champ leg animation (body frame: x=forward, y=left, z=angular)
        if self.publish_cmd_vel:
            body_vx = math.cos(self.yaw) * self.delayed_vel_x + math.sin(self.yaw) * self.delayed_vel_y
            body_vy = -math.sin(self.yaw) * self.delayed_vel_x + math.cos(self.yaw) * self.delayed_vel_y
            cmd = Twist()
            cmd.linear.x = body_vx
            cmd.linear.y = body_vy
            cmd.linear.z = 0.0
            cmd.angular.z = self.delayed_omega_z
            self.cmd_vel_pub.publish(cmd)


def main():
    try:
        simulator = SimpleOdomSimulator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()