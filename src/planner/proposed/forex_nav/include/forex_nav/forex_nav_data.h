#ifndef FOREX_NAV_DATA_H_
#define FOREX_NAV_DATA_H_

#include <Eigen/Core>
#include <vector>
#include <memory>

namespace forex_nav {

using Vector3d = Eigen::Vector3d;

// Viewpoint data structure
struct Viewpoint {
  Vector3d pos_;
  double yaw_;
  int visible_cells_;
  double obstacle_distance_;
  
  Viewpoint() : pos_(Vector3d::Zero()), yaw_(0.0), visible_cells_(0), obstacle_distance_(0.0) {}
  Viewpoint(const Vector3d& pos, double yaw, int vis = 0, double dist = 0.0)
    : pos_(pos), yaw_(yaw), visible_cells_(vis), obstacle_distance_(dist) {}
};

// FSM data
struct FSMData {
  bool have_odom_;
  bool have_goal_;
  bool have_viewpoints_;
  
  Vector3d odom_pos_;
  Vector3d odom_vel_;
  double odom_yaw_;
  
  Vector3d goal_pos_;
  
  std::vector<Viewpoint> viewpoints_;
  
  Vector3d selected_viewpoint_pos_;
  double selected_viewpoint_yaw_;
  
  std::vector<Vector3d> planned_path_;
  
  FSMData() 
    : have_odom_(false), have_goal_(false), have_viewpoints_(false),
      odom_pos_(Vector3d::Zero()), odom_vel_(Vector3d::Zero()), odom_yaw_(0.0),
      goal_pos_(Vector3d::Zero()),
      selected_viewpoint_pos_(Vector3d::Zero()), selected_viewpoint_yaw_(0.0) {}
};

// FSM parameters
struct FSMParam {
  double replan_thresh1_;  // Time threshold for replanning when traj is almost finished (s)
  double replan_thresh2_;  // Distance threshold for replanning (m)
  double replan_thresh3_;  // Periodic replanning time threshold (s)
  double replan_time_;     // Minimum time between replans (s)
  double goal_reached_threshold_;  // Distance to goal to consider reached (m)
  double no_replan_dist_to_goal_;  // Disable replanning when close to goal (m)
  
  FSMParam()
    : replan_thresh1_(0.1), replan_thresh2_(0.3), replan_thresh3_(3.0), replan_time_(0.5),
      goal_reached_threshold_(0.5), no_replan_dist_to_goal_(1.5) {}
};

// Navigation parameters
struct NavParam {
  // Fuzzy A* weights
  double w_dist_;    // Weight for distance cost
  double w_pred_;    // Weight for predicted cost (fuzzy A* / path cost)
  double w_curve_;   // Weight for curve speed penalty
  double w_homo_;    // Weight for homotopy class consistency
  
  // Path planning
  double max_vel_;   // Maximum velocity (m/s)
  double max_acc_;   // Maximum acceleration (m/s^2)
  double max_yawdot_; // Maximum yaw rate (rad/s)
  
  // Trajectory generation
  double traj_dt_;   // Time step for trajectory (s)
  
  // Visualization: number of candidate paths to draw (-1 = all)
  int vis_candidate_path_count_;
  
  // Height
  double planning_height_;  // 规划高度 (m)
  
  // MINCO trajectory optimization parameters
  double minco_weight_time_;         // 时间代价权重 (越大越压缩时间, 轨迹越激进)
  double minco_weight_energy_;       // 能量/平滑权重 (越大越光滑)
  double minco_weight_pos_;          // 位置约束(走廊)惩罚
  double minco_weight_vel_;          // 速度约束惩罚
  double minco_weight_acc_;          // 加速度约束惩罚
  double minco_weight_jerk_;         // jerk约束惩罚
  double minco_max_jerk_;            // 最大jerk约束 (m/s^3)
  double minco_alloc_speed_ratio_;   // 初始时间分配速度 = max_vel * ratio
  double minco_length_per_piece_;    // 每段多项式目标长度 (m)
  
  NavParam()
    : w_dist_(2.0), w_pred_(1.5), w_curve_(0.3), w_homo_(0.3),
      max_vel_(2.0), max_acc_(1.0), max_yawdot_(2.09),
      traj_dt_(0.02), vis_candidate_path_count_(3),
      planning_height_(1.0),
      minco_weight_time_(30.0), minco_weight_energy_(0.1),
      minco_weight_pos_(2000.0), minco_weight_vel_(100.0),
      minco_weight_acc_(80.0), minco_weight_jerk_(30.0),
      minco_max_jerk_(15.0), minco_alloc_speed_ratio_(0.7),
      minco_length_per_piece_(2.0) {}
};

}  // namespace forex_nav

#endif  // FOREX_NAV_DATA_H_
