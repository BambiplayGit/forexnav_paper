#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立步态关节发布器：CHAMP 风格 stance/swing 相位

直接根据 odom 速度和步态相位生成 joint_states，与 simple_odom_simulator 严格同步。
采用 CHAMP 风格：
  - 步态周期分为 stance（支撑相）和 swing（摆动相）
  - Stance: 大腿线性后摆，脚在地面上"定住"，匹配地面位移
  - Swing: 大腿弧线前摆 + 小腿抬起，脚在空中前进
  - Raibert 启发式: step_length = stance_duration * velocity，确保步幅与速度匹配
  - Trot 步态: LF/RH 同相, RF/LH 同相且偏移半周期
"""

import rospy
import math
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry

# ========== 机器人配置表 ==========
ROBOT_CONFIGS = {
    "go2": {
        "joint_names": [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ],
        "stance": {"hip": 0.0, "thigh": 0.9, "calf": -1.78},
        "leg_length": 0.32,
        "limits": {
            "hip": (-1.05, 1.05),
            "thigh_front": (-1.57, 3.49),
            "thigh_rear": (-0.52, 4.54),
            "calf": (-2.72, -0.84),
        },
    },
    "zsl-1": {
        "joint_names": [
            "FL_ABAD_JOINT", "FL_HIP_JOINT", "FL_KNEE_JOINT",
            "FR_ABAD_JOINT", "FR_HIP_JOINT", "FR_KNEE_JOINT",
            "RL_ABAD_JOINT", "RL_HIP_JOINT", "RL_KNEE_JOINT",
            "RR_ABAD_JOINT", "RR_HIP_JOINT", "RR_KNEE_JOINT",
        ],
        "stance": {"hip": 0.0, "thigh": 0.8, "calf": -1.6},
        "leg_length": 0.31,
        "limits": {
            "hip": (-0.4887, 0.4887),
            "thigh_front": (-1.1519, 2.967),
            "thigh_rear": (-1.1519, 2.967),
            "calf": (-2.723, -0.602),
        },
    },
}

# Trot 相位偏移 (归一化 0~1): LF&RH 同相=0, RF&LH 偏移=0.5 (与 CHAMP phase_generator 一致)
PHASE_OFFSET_NORM = [0.0, 0.5, 0.5, 0.0]  # LF, RF, LH, RH

# 腿侧 (左=1, 右=-1, 用于 hip 转弯调制)
LEG_SIDE = [1.0, -1.0, 1.0, -1.0]  # LF, RF, LH, RH


class GaitJointPublisher:
    def __init__(self):
        rospy.init_node("gait_joint_publisher")

        # 机器人类型 → 加载对应关节配置
        robot_type = rospy.get_param("~robot_type", "zsl-1")
        if robot_type not in ROBOT_CONFIGS:
            rospy.logwarn("Unknown robot_type '%s', falling back to 'zsl-1'", robot_type)
            robot_type = "zsl-1"
        cfg = ROBOT_CONFIGS[robot_type]
        self.joint_names = cfg["joint_names"]
        self.stance = cfg["stance"]
        self.leg_length = cfg["leg_length"]
        self.limits = cfg["limits"]
        rospy.loginfo("GaitJointPublisher: robot_type = %s", robot_type)

        # 步态参数
        self.gait_freq = rospy.get_param("~gait_freq", 2.0)          # 步态频率 Hz (stride_period = 1/freq)
        self.stance_duration = rospy.get_param("~stance_duration", 0.25)  # 支撑相时长 s
        self.max_vel = rospy.get_param("~max_vel", 2.0)              # 最大速度 m/s
        self.swing_lift = rospy.get_param("~swing_lift", 0.25)       # 摆动相小腿抬起幅度 rad
        self.max_thigh_amp = rospy.get_param("~max_thigh_amp", 0.45) # 大腿最大摆动半幅 rad
        self.hip_turn_amp = rospy.get_param("~hip_turn_amp", 0.08)   # 转弯时 hip 外展幅度 rad
        self.vel_smooth_alpha = rospy.get_param("~vel_smooth_alpha", 0.15)  # 速度平滑

        # 速度状态
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.omega_z = 0.0
        self.smooth_vel_x = 0.0
        self.smooth_vel_y = 0.0
        self.smooth_omega_z = 0.0
        self.last_odom_time = None
        
        # 步态相位（增量累积，支持动态频率）
        self.gait_phase = 0.0
        self.last_time = None

        # ROS 接口
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self._odom_cb, queue_size=1)
        self.js_pub = rospy.Publisher("joint_states", JointState, queue_size=1)

        rate = rospy.get_param("~publish_rate", 100.0)
        self.timer = rospy.Timer(rospy.Duration(1.0 / rate), self._publish)

        # 计算 stride 参数
        self.stride_period = 1.0 / self.gait_freq
        self.stance_ratio = min(0.7, max(0.3, self.stance_duration / self.stride_period))

        rospy.loginfo("GaitJointPublisher: CHAMP-style stance/swing gait")
        rospy.loginfo("  stride_period=%.3fs, stance_ratio=%.2f, swing_lift=%.2f rad",
                      self.stride_period, self.stance_ratio, self.swing_lift)

    def _odom_cb(self, msg):
        self.vel_x = msg.twist.twist.linear.x
        self.vel_y = msg.twist.twist.linear.y
        self.omega_z = msg.twist.twist.angular.z
        self.last_odom_time = msg.header.stamp

    @staticmethod
    def _smoothstep(x):
        """Hermite smoothstep: 3x^2 - 2x^3, x clamped to [0,1]"""
        x = max(0.0, min(1.0, x))
        return x * x * (3.0 - 2.0 * x)

    def _publish(self, event):
        t = rospy.Time.now().to_sec()

        # ---- 速度平滑 ----
        a = self.vel_smooth_alpha
        self.smooth_vel_x += a * (self.vel_x - self.smooth_vel_x)
        self.smooth_vel_y += a * (self.vel_y - self.smooth_vel_y)
        self.smooth_omega_z += a * (self.omega_z - self.smooth_omega_z)

        vel_mag = math.sqrt(self.smooth_vel_x ** 2 + self.smooth_vel_y ** 2)
        omega_mag = abs(self.smooth_omega_z)

        # ---- 自适应步频：freq * step_length ≈ stance_ratio * velocity ----
        # 目标步幅（~70% max_thigh_amp，视觉最自然）
        target_step = 2.0 * self.leg_length * self.max_thigh_amp * 0.7
        if vel_mag > 0.05:
            desired_freq = vel_mag * self.stance_ratio / target_step
            effective_freq = max(self.gait_freq * 0.5, min(self.gait_freq * 2.5, desired_freq))
        else:
            effective_freq = self.gait_freq

        # ---- 增量相位累积（支持动态频率）----
        if self.last_time is None:
            self.last_time = t
        dt_gait = t - self.last_time
        self.last_time = t
        self.gait_phase += effective_freq * dt_gait
        self.gait_phase %= 1.0

        # ---- Raibert 启发式: 用自适应的 stride_period 计算步长 ----
        effective_stride_period = 1.0 / effective_freq
        effective_stance_duration = effective_stride_period * self.stance_ratio
        step_length = effective_stance_duration * vel_mag
        thigh_amp = step_length / (2.0 * self.leg_length)
        thigh_amp = min(thigh_amp, self.max_thigh_amp)

        # 纯旋转时的小幅摆动 (避免太空步，但需要有一点腿部动作)
        omega_amp = min(0.08, 0.04 * omega_mag)

        # 综合运动幅度
        amp = max(thigh_amp, omega_amp)

        # motion_factor: 判断是否有运动 (用于开关步态)
        motion_factor = min(1.0, vel_mag / 0.05 + omega_mag / 0.1)
        if motion_factor < 0.05:
            motion_factor = 0.0

        # 转弯 hip 调制
        turn_factor = min(1.0, omega_mag / 1.5)
        turn_sign = 1.0 if self.smooth_omega_z >= 0 else -1.0

        # ---- 逐腿计算关节角 ----
        positions = []
        for leg in range(4):
            if motion_factor < 0.01:
                # 静止：保持站立姿态
                positions.extend([self.stance["hip"], self.stance["thigh"], self.stance["calf"]])
                continue

            # 计算该腿在 stride 周期内的相位 [0, 1)
            leg_phase = (self.gait_phase + PHASE_OFFSET_NORM[leg]) % 1.0

            if leg_phase < self.stance_ratio:
                # ======= STANCE (支撑相): 脚在地面，大腿线性后摆 =======
                # progress: 0 -> 1 在 stance 期间
                progress = leg_phase / self.stance_ratio
                # 大腿从 +amp 线性移动到 -amp (脚从前方滑到后方)
                thigh_delta = amp * (1.0 - 2.0 * progress)
                # 小腿保持不变 (脚在地面)
                calf_delta = 0.0
            else:
                # ======= SWING (摆动相): 脚在空中，大腿前摆 + 小腿抬起 =======
                # progress: 0 -> 1 在 swing 期间
                progress = (leg_phase - self.stance_ratio) / (1.0 - self.stance_ratio)
                # 大腿从 -amp 摆回 +amp (脚从后方回到前方)
                # 使用 smoothstep 使起止更自然
                smooth_p = self._smoothstep(progress)
                thigh_delta = amp * (-1.0 + 2.0 * smooth_p)
                # 小腿在 swing 中间最大抬起 (正弦包络)
                calf_delta = -self.swing_lift * math.sin(math.pi * progress) * min(1.0, amp / 0.1)

            # 合成关节角
            hip = self.stance["hip"]
            thigh = self.stance["thigh"] + thigh_delta
            calf = self.stance["calf"] + calf_delta

            # 转弯时 hip 外展/内收
            if turn_factor > 0.02:
                hip += self.hip_turn_amp * turn_factor * turn_sign * LEG_SIDE[leg]

            # 关节限位 (根据 robot_type 自动选择)
            lim = self.limits
            hip = max(lim["hip"][0], min(lim["hip"][1], hip))
            calf = max(lim["calf"][0], min(lim["calf"][1], calf))
            if leg < 2:  # 前腿
                thigh = max(lim["thigh_front"][0], min(lim["thigh_front"][1], thigh))
            else:  # 后腿
                thigh = max(lim["thigh_rear"][0], min(lim["thigh_rear"][1], thigh))

            positions.extend([hip, thigh, calf])

        # 发布 joint_states
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = ""
        msg.name = self.joint_names
        msg.position = positions
        msg.velocity = [0.0] * 12
        msg.effort = []
        self.js_pub.publish(msg)


def main():
    try:
        n = GaitJointPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
