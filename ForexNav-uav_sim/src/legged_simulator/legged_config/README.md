# 四足机器人规划系统 - 快速参考

## 🚀 快速启动

```bash
# 推荐：完整系统（MINCO 优化）
roslaunch legged_config leg_planner_bringup.launch
```

在 RViz 中点击 **"2D Nav Goal"** 设置目标点，机器人自动规划和移动！

---

## 📦 Launch 文件对比

| Launch 文件 | 优化器 | 包含内容 | 适用场景 |
|------------|--------|---------|---------|
| **leg_planner_bringup.launch** | MINCO | 完整系统 | ⭐ 推荐日常使用 |
| **leg_minco_planner.launch** | MINCO | 完整系统 | 明确使用 MINCO |
| **leg_bspline_planner.launch** | B-spline | 完整系统 | 快速响应场景 |
| **leg_teleop.launch** | - | 仅底盘 | 手动控制 |
| **leg_bringup.launch** | - | 仅底盘 | 调试基础 |
| **leg_rviz.launch** | - | 仅可视化 | 配合其他 launch |

**完整系统包含**：机器人底盘 + 环境感知 + 路径规划 + 轨迹优化 + 运动控制 + RViz 可视化

---

## 🎯 MINCO vs B-spline 对比

| 特性 | MINCO | B-spline |
|------|-------|----------|
| 优化方法 | 闭式最优解 | 迭代优化 |
| 轨迹质量 | ⭐⭐⭐⭐⭐ 最优 | ⭐⭐⭐⭐ 很好 |
| 计算速度 | 稍慢 (~100-300ms) | 较快 (~50-150ms) |
| 轨迹平滑度 | 最平滑 | 平滑 |
| 连续性 | C∞ | C² |
| 推荐场景 | 追求最优轨迹 | 需要快速响应 |

**默认配置**：所有 launch 文件默认使用 **MINCO**（轨迹质量最高）

---

## 🔧 参数说明

### 关键参数
```xml
<!-- 优化器后端选择 -->
<param name="fast_planner_node/manager/use_minco_backend" value="true"/>
<!-- true = MINCO, false = B-spline -->

<!-- 规划高度（机器人身体中心高度）-->
<arg name="planning_height" default="0.35"/>

<!-- 最大速度 -->
<arg name="max_vel_x" default="0.6"/>  <!-- 前进速度 (m/s) -->

<!-- 障碍物数量 -->
<arg name="pillar_num" default="40"/>
```

### 运行时查看当前配置
```bash
# 查看优化器
rosparam get /fast_planner_node/manager/use_minco_backend

# 查看规划高度
rosparam get /fast_planner_node/fsm/fixed_height
```

---

## 📖 详细文档

- **使用指南**: [四足规划器使用指南.md](四足规划器使用指南.md)
- **修复总结**: [../四足规划器修复总结.md](../四足规划器修复总结.md)

---

## ⚡ 常用命令

### 启动系统
```bash
cd /home/lsy/xue/quadruped_planner_ws
source devel/setup.bash

# 完整系统（MINCO）
roslaunch legged_config leg_planner_bringup.launch

# 或 B-spline 版本
roslaunch legged_config leg_bspline_planner.launch
```

### 手动遥控
```bash
# 启动底盘
roslaunch legged_config leg_teleop.launch

# 键盘控制
rosrun teleop_twist_keyboard teleop_twist_keyboard.py cmd_vel:=/cmd_vel
```

### 调试命令
```bash
# 查看所有话题
rostopic list

# 查看机器人位置
rostopic echo /odom/pose/pose/position -n 1

# 查看规划命令
rostopic echo /planning/pos_cmd

# 查看局部点云数量
rostopic echo /pcl_render_node/cloud/width -n 1
```

---

## 📞 技术支持

**配置文件位置**：
- Launch: `legged_config/launch/`
- 控制器参数: `legged_config/config/position_controller.yaml`
- RViz 配置: `legged_config/config/leg_visualization.rviz`
- 机器人模型: `legged_config/config/robots/go2/`

**已修复的问题**：
- ✅ 坐标系变换
- ✅ 相机 FOV 方向
- ✅ 障碍物可见性
- ✅ 规划起点设置
- ✅ UAV launch 猛冲问题

**系统状态**：完全可用！🎉

---

**最后更新**: 2025-10-24




