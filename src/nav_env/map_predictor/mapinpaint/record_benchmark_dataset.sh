#!/bin/bash

# =================配置区域=================
# 1. 设置保存的 Bag 文件名前缀
file_prefix="inpaint_experiment"

# 2. 生成带时间戳的文件名 (例如: inpaint_experiment_2023-10-27-14-30-00.bag)
timestamp=$(date +%Y-%m-%d-%H-%M-%S)
bag_name="${file_prefix}_${timestamp}.bag"

# 3. 定义要录制的 Topic 列表
# 包含: 核心输入图像, 地图信息, 导航目标触发器, 里程计/轨迹, 以及 TF 坐标变换
topics=(
    "/sdf_map/2d/image"       # [输入] 传感器图像/局部地图
    "/sdf_map/2d"             # [元数据] 地图分辨率和原点
    "/move_base_simple/goal"  # [触发器] 发送这个话题时，实验开始
    "/state_ukf/odom"         # [轨迹/终止] 机器人位置和结束判断
    "/tf"                     # [坐标系] 动态坐标变换 (必须)
    "/tf_static"              # [坐标系] 静态坐标变换 (必须)
)

# =================执行区域=================
echo "-------------------------------------------"
echo "开始录制 ROS Bag..."
echo "文件名: $bag_name"
echo "录制话题: ${topics[*]}"
echo "-------------------------------------------"
echo "注意: 请在实验结束后按 Ctrl+C 停止录制。"
echo "-------------------------------------------"

# 执行录制命令
# --split --size=2048: 如果文件超过2GB自动分割 (防止文件过大损坏)
# --buffsize=256: 增加缓冲区大小 (MB)，防止图像数据量大时丢帧
rosbag record -O "$bag_name" --buffsize=256 "${topics[@]}"