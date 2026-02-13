# Frtree 问题记录

## 问题1: Eigen 版本冲突导致 rviz 崩溃

### 报错信息
```
rviz: /home/lsy/.local/include/eigen3/Eigen/src/Eigenvalues/SelfAdjointEigenSolver.h:302: 
Assertion `m_isInitialized && "SelfAdjointEigenSolver is not initialized."' failed.
[rviz-2] process has died [pid 90163, exit code -6]
```

### 原因
本地安装了一个 Eigen (`~/.local/include/eigen3/`)，与 ROS 系统的 Eigen 版本冲突。

### 解决方法
删除本地 Eigen 安装：
```bash
rm -rf ~/.local/include/eigen3
rm -rf ~/.local/share/eigen3
```

---

## 问题2: altro CMake 配置引用已删除的 Eigen 路径

### 报错信息
```
CMake Error in CMakeLists.txt:
  Imported target "altro::altro" includes non-existent path
    "/home/lsy/.local/include/eigen3"
  in its INTERFACE_INCLUDE_DIRECTORIES.
```

### 原因
之前安装的 altro 库在 cmake 配置文件中硬编码了 Eigen 路径，删除 Eigen 后路径失效。

### 解决方法
清理旧的 altro 安装：
```bash
rm -rf ~/.local/lib/cmake/altro
rm -rf ~/.local/lib/cmake/AltroCpp
rm -rf ~/.local/lib/libaltro*
rm -rf ~/.local/include/altro
```

---

## 问题3: 项目需要的是 altro-cpp 而不是 altro

### 报错信息
```
fatal error: altro/eigentypes.hpp: No such file or directory
fatal error: altro/constraints/constraint.hpp: No such file or directory
```

### 原因
项目代码使用的 include 头文件（如 `eigentypes.hpp`, `constraints/constraint.hpp`）来自 **altro-cpp** 库，而不是 **altro** 库。这是两个不同的库：
- altro: https://github.com/bjack205/altro
- altro-cpp: https://github.com/bjack205/altro-cpp

### 解决方法

1. 下载并编译安装 altro-cpp：
```bash
cd ~/
git clone https://github.com/bjack205/altro-cpp.git
cd altro-cpp
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=~/.local \
         -DALTRO_BUILD_TESTS=OFF \
         -DALTRO_BUILD_EXAMPLES=OFF \
         -DALTRO_BUILD_BENCHMARKS=OFF \
         -DALTRO_BUILD_SHARED_LIBS=ON
make -j$(nproc)
make install
```

2. 修改项目 CMakeLists.txt（`planner_manage/CMakeLists.txt`）：
```cmake
# 原来的：
find_package(altro REQUIRED)
# 改为：
find_package(AltroCpp REQUIRED)

# 原来的：
altro::altro
# 改为：
altrocpp::altro
```

### 备注
altro-cpp 源码已保存在 `Frtree/altro-cpp/` 目录下，方便在新机器上重新编译安装。

---

## 问题4: 运行 test_path_decomp_2d.launch 时 rviz 崩溃

### 报错信息
```
4 on Mesa system.
rviz: /home/lsy/.local/include/eigen3/Eigen/src/Eigenvalues/SelfAdjointEigenSolver.h:302: 
Assertion `m_isInitialized && "SelfAdjointEigenSolver is not initialized."' failed.
[rviz-2] process has died [pid 294790, exit code -6]
```

### 原因
Eigen 版本冲突：
- 系统 Eigen (`/usr/include/eigen3`): **3.3.7**
- 本地 Eigen (`~/.local/include/eigen3`): **3.4.0**

ROS rviz 使用系统 Eigen 3.3.7 编译，但 `decomp_ros_utils` 插件可能链接了本地的 3.4.0 版本，导致 ABI 不兼容。rviz 配置中的 `EllipsoidArray` 和 `PolyhedronArray` 插件内部使用 Eigen 特征值分解，初始化时出错。

### 解决方法

1. 备份本地 Eigen：
```bash
mv ~/.local/include/eigen3 ~/.local/include/eigen3_backup_340
```

2. 重新编译 decomp_ros_utils：
```bash
cd ~/ForexNav
catkin build decomp_ros_utils --force-cmake
```

3. 如果还有问题，也重新编译 decomp_test_node：
```bash
catkin build decomp_test_node --force-cmake
```

---

## 问题5: local_map 运行崩溃 (exit code -11)

### 报错信息
```
[local_map-12] process has died [pid 326439, exit code -11]
```

### 原因
1. **点云坐标帧不匹配**：`local_sensing`发布的点云是world帧（世界坐标），但`local_map`期望的是传感器帧（相对于机器人的坐标）
2. **点云高度过滤**：`local_map`的`slicePointCloudInxyPlane`函数过滤z范围是-0.2~0.2，但world帧的点云z=1.0（机器人高度），导致所有点被滤掉
3. **点云数量为0**：过滤后点云为空，后续曲率计算访问越界导致段错误

### 解决方法

1. 修改`local_sensing_node.cpp`，添加参数支持相对坐标输出：
```cpp
pnh.param<bool>("use_relative_cloud", use_relative_cloud_, false);
pnh.param<std::string>("cloud_frame_id", cloud_frame_id_, "world");
```

2. 在launch文件中启用相对坐标：
```xml
<param name="use_relative_cloud" value="true" />
<param name="cloud_frame_id" value="unitree_scan" />
```

3. 放宽`local_map.cpp`中的z过滤范围：
```cpp
pass.setFilterLimits(-1.0, 1.0);  // 原来是-0.2, 0.2
```

### 涉及文件
- `nav_env/local_sensing/src/local_sensing_node.cpp`
- `nav_env/nav_exp_env/launch/nav_env.launch`
- `nav_env/nav_exp_env/launch/frtree_local_map_test.launch`
- `planner/Frtree/.../local_map/src/local_map.cpp`

---

## 问题6: 订阅 /planner/poly_test 导致 RViz 卡死

### 现象
在 RViz 中添加 `/planner/poly_test` 话题后，RViz 无响应卡死。

### 原因
`dynamic_plan_manager_pop.cpp` 中 `polys_test_` 向量只有 `push_back` 操作，但 `clear()` 被注释掉了（第669行和740行），导致多面体数据无限累积。每次规划都会添加新的多面体，但永远不清空，消息越来越大，RViz 渲染时卡死。

### 解决方法
不要订阅 `/planner/poly_test`，这是调试用的话题。改用以下话题：

| 话题 | 用途 | 推荐程度 |
|------|------|----------|
| `/planner/poly_for_current_node` | 当前节点的前2个多面体 | 推荐（数据量小） |
| `/planner/polyhedron_array` | 完整规划走廊 | 可用 |
| `/planner/poly_test` | 调试用，有bug | 不推荐 |

### 涉及文件
- `planner/Frtree/.../planner_manage/src/dynamic_plan_manager_pop.cpp` (第669、740行的clear被注释)

---

## 问题7: simple_odom_simulator 不支持速度命令

### 现象
FRTree planner 发布的是速度命令 `/cmd_vel` (Twist类型)，但 `simple_odom_simulator.py` 只订阅位置命令 `/planning/pos_cmd`，导致机器人不动。

### 原因
原始的 odom simulator 设计为直接跟踪位置命令，不支持速度控制模式。

### 解决方法
修改 `simple_odom_simulator.py`，添加 `/cmd_vel` 订阅：

1. 导入 Twist 消息类型
2. 添加 cmd_vel 订阅和回调函数
3. 在定时器中积分速度更新位置
4. 忽略 z 轴线速度（2D 导航）

### 涉及文件
- `nav_env/nav_exp_env/scripts/simple_odom_simulator.py`

---

## 问题8: ALTRO solver replan 时崩溃 (exit code -11)

### 报错信息
```
[ WARN] Start to solve the ALTRO problem
SDPConstraint created start
SDPConstraint created start
... (10次)
[planner-12] process has died [pid xxx, exit code -11]
```

### 原因
1. **F, g, c 未初始化**：`altro_problem.cpp` 中 F, g, c resize 后没有初始化值，直接传给 SDPConstraint
2. **多线程问题**：`nthreads = 12` 可能导致 HiGHS 求解器状态冲突

### 解决方法

1. 初始化 F, g, c（`altro_problem.cpp`）：
```cpp
F[0] = Eigen::MatrixXd::Zero(6, 3);
F[1] = Eigen::MatrixXd::Zero(6, 3);
g[0] = Eigen::VectorXd::Zero(6);
g[1] = Eigen::VectorXd::Zero(6);
c[0] = Eigen::VectorXd::Zero(6);
c[1] = Eigen::VectorXd::Zero(6);
```

2. 临时将线程数改为1（`dynamic_plan_fsm_pop.cpp`）：
```cpp
solver_al_ptr_->GetOptions().nthreads = 1;
```

### 涉及文件
- `planner/Frtree/.../planner_manage/src/Altro/altro_problem.cpp`
- `planner/Frtree/.../planner_manage/src/dynamic_plan_fsm_pop.cpp`

---

## 问题9: SDPsolver 求解失败导致段错误

### 报错信息
ALTRO 迭代几次后崩溃（exit code -11）

### 原因
1. **输入验证缺失**：空向量/矩阵传入时直接访问元素
2. **求解结果检查缺失**：HiGHS 求解失败后直接访问 solution.row_dual 和 solution.col_value

### 解决方法
在 `SDPsolver.cpp` 中添加安全检查：

1. 输入验证：
```cpp
if (b.size() == 0 || g.size() == 0 || q.size() < 6 || c.size() < 6 ||
    A.rows() == 0 || A.cols() < 3 || F.rows() == 0 || F.cols() < 3) {
    gradient.setZero();
    alpha_value = 1.0;
    return result;
}
```

2. 求解状态检查：
```cpp
HighsStatus status = highs.run();
if (status != HighsStatus::kOk) {
    // 返回安全默认值
}
```

3. Solution 有效性检查：
```cpp
if (solution.row_dual.empty() || solution.col_value.empty()) {
    // 返回安全默认值
}
```

### 涉及文件
- `planner/Frtree/.../planner_manage/src/SDPsolver/SDPsolver.cpp`

---

## 问题10: SDPsolver 是 3D 求解，可考虑简化为 2D

### 现状
当前 SDPsolver 使用 6 自由度位姿 (x, y, z, roll, pitch, yaw) 进行 3D 碰撞检测。

### 建议
对于地面机器人导航，可以简化为 2D：
- 只需要 x, y, yaw
- z, roll, pitch 固定
- 减少计算量，提高稳定性

### 机器人形状配置
在 `altro_problem.cpp` 中修改 `b[0]` 向量：
```cpp
// 原始（0.8m x 0.44m x 0.2m 长方体）
b[0] << 0.38, 0.42, 0.22, 0.22, 0.10, 0.10;

// 修改为 5cm 正方体
b[0] << 0.025, 0.025, 0.025, 0.025, 0.025, 0.025;
```

### 涉及文件
- `planner/Frtree/.../planner_manage/src/Altro/altro_problem.cpp`
- `planner/Frtree/.../planner_manage/src/SDPsolver/SDPsolver.cpp`

---

## 问题11: 轨迹跟踪卡顿、速度慢

### 现象
机器人运动时一卡一卡的，速度很慢，跟踪效果差。

### 原因分析

| 问题 | 原始值 | 说明 |
|------|--------|------|
| cmd发布频率 | 5Hz (0.2秒) | 太低，导致跳跃 |
| 速度上限 | 硬编码0.15 m/s | P控制器限速太严 |
| 轨迹点数 | 10个点 | ALTRO轨迹点间距大 |
| 轨迹形状 | 直线段 | 基于树节点中心的直线插值 |

### 轨迹生成原理
`traj_from_altro_` 是基于多边形中心点的直线插值：
1. 当前位置 → 交叉多边形中心 (`center_list_[1]`)
2. → 第二个多边形中心 (`center_list_[2]`)
3. 在这些点之间线性插值10个点

### 解决方法

#### 方案1: 使用pos_cmd直接位置跟踪（已实现）

1. **planner发布pos_cmd**（`dynamic_plan_fsm_pop.cpp`）：
```cpp
// 添加publisher
pos_cmd_pub_ = public_nh.advertise<geometry_msgs::PoseStamped>("/planning/pos_cmd", 10);

// 在publishCmdCallback中发布带插值的位置命令
```

2. **simulator优先跟踪pos_cmd**（`simple_odom_simulator.py`）：
```python
# pos_cmd优先级高于cmd_vel
if time_since_pos_cmd < pos_cmd_timeout:
    pos_cmd_active = True
# 只有pos_cmd不活跃时才使用cmd_vel积分
```

#### 方案2: 提高频率+插值（已实现）

1. **提高发布频率到50Hz**：
```cpp
cmd_timer_ = node_.createTimer(ros::Duration(0.02), &DynamicReplanFSMPOP::publishCmdCallback, this);
```

2. **添加平滑插值**：
```cpp
// 每次向目标移动一小步，而不是直接跳到目标点
double step_size = max_vel_ * cmd_dt;  // 基于max_vel计算步长
Eigen::Vector3d dir = target_pos - odom_pos_;
if (dist > step_size) {
    interp_pos = odom_pos_ + dir.normalized() * step_size;
} else {
    interp_pos = target_pos;
}
```

#### 方案3: 速度由yaml参数控制（已实现）

修改后，`pop_plan_fsm_param.yaml` 中的 `max_vel` 参数生效：
```yaml
max_vel: &max_vel 4.0    # 控制最大跟踪速度
max_acc: &max_acc 2.0    # 预留
```

读取参数：
```cpp
node_.param("plan_manager/max_vel", max_vel_, 0.6);
```

计算步长：
```cpp
double step_size = max_vel_ * 0.02;  // 50Hz * max_vel = 速度
```

### 涉及文件
- `planner/Frtree/.../planner_manage/src/dynamic_plan_fsm_pop.cpp`
- `planner/Frtree/.../planner_manage/include/.../dynamic_plan_fsm_pop.h`
- `nav_env/nav_exp_env/scripts/simple_odom_simulator.py`
- `planner/Frtree/.../planner_manage/launch/config/pop_plan_fsm_param.yaml`

### 参数调节建议

| 参数 | 位置 | 作用 | 建议值 |
|------|------|------|--------|
| `max_vel` | yaml | 最大速度 | 0.5-4.0 m/s |
| `step_size` | cpp | 每步移动距离 | max_vel * 0.02 |
| `yaw_step` | cpp | 每步转动角度 | 0.05 rad |
| `cmd_dt` | cpp | 发布周期 | 0.02秒 (50Hz) |

---

## 问题12: MPC控制器未使用，yaml中Q/R参数无效

### 现象
`pop_plan_fsm_param.yaml` 中的MPC参数（Q_vec, R_vec等）不起作用。

### 原因
`trackTrajCallback`（MPC跟踪器）被注释掉了：
```cpp
// traj_tracker_timer_ = node_.createTimer(ros::Duration(delta_time_), &DynamicReplanFSMPOP::trackTrajCallback, this);
```

当前使用的是简单P控制器 `publishCmdCallback`：
```cpp
twist_cmd.linear.x = (target(0) - odom_pos_(0)) * 1.5 + (- odom_vel_(0)) * 0.1;
```

### 现状
改用pos_cmd直接位置跟踪后，P控制器和MPC都被绕过，机器人直接跟踪规划的位置点。

### 涉及文件
- `planner/Frtree/.../planner_manage/src/dynamic_plan_fsm_pop.cpp` (第78行)
- `planner/Frtree/.../planner_manage/launch/config/pop_plan_fsm_param.yaml` (第104-110行)
