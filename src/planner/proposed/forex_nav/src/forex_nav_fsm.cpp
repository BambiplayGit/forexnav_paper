#include "forex_nav/forex_nav_fsm.h"
#include <tf2/LinearMath/Quaternion.h>
#include <quadrotor_msgs/PositionCommand.h>
#include <cmath>
#include <Eigen/Core>

namespace forex_nav {

ForexNavFSM::ForexNavFSM(ros::NodeHandle& nh, ros::NodeHandle& pnh) 
  : nh_(nh), pnh_(pnh), state_(INIT), traj_index_(0),
    minco_traj_duration_(0.0), minco_start_yaw_(0.0), minco_end_yaw_(0.0), has_minco_traj_(false) {
  
  fd_ = std::make_shared<FSMData>();
  fp_ = std::make_shared<FSMParam>();
  manager_ = std::make_shared<ForexNavManager>();
  
  // Initialize last_replan_time_ to zero (will be set when first planning succeeds)
  last_replan_time_ = ros::Time(0);
  
  state_str_ = {"INIT", "WAIT_TRIGGER", "PLAN_TRAJ", "EXEC_TRAJ", "FINISH"};
}

ForexNavFSM::~ForexNavFSM() {
}

void ForexNavFSM::init() {
  // Get parameters
  pnh_.param("fsm/replan_thresh1", fp_->replan_thresh1_, fp_->replan_thresh1_);
  pnh_.param("fsm/replan_thresh2", fp_->replan_thresh2_, fp_->replan_thresh2_);
  pnh_.param("fsm/replan_thresh3", fp_->replan_thresh3_, fp_->replan_thresh3_);
  pnh_.param("fsm/replan_time", fp_->replan_time_, fp_->replan_time_);
  pnh_.param("fsm/goal_reached_threshold", fp_->goal_reached_threshold_, fp_->goal_reached_threshold_);
  pnh_.param("fsm/no_replan_dist_to_goal", fp_->no_replan_dist_to_goal_, fp_->no_replan_dist_to_goal_);
  
  auto& nav_param = manager_->getNavParam();
  pnh_.param("nav/w_dist", nav_param.w_dist_, nav_param.w_dist_);
  pnh_.param("nav/w_pred", nav_param.w_pred_, nav_param.w_pred_);
  pnh_.param("nav/w_curve", nav_param.w_curve_, nav_param.w_curve_);
  pnh_.param("nav/w_homo", nav_param.w_homo_, nav_param.w_homo_);
  pnh_.param("nav/max_vel", nav_param.max_vel_, nav_param.max_vel_);
  pnh_.param("nav/max_acc", nav_param.max_acc_, nav_param.max_acc_);
  pnh_.param("nav/max_yawdot", nav_param.max_yawdot_, nav_param.max_yawdot_);
  pnh_.param("nav/traj_dt", nav_param.traj_dt_, nav_param.traj_dt_);
  pnh_.param("nav/vis_candidate_path_count", nav_param.vis_candidate_path_count_, nav_param.vis_candidate_path_count_);
  pnh_.param("nav/planning_height", nav_param.planning_height_, nav_param.planning_height_);
  
  // MINCO trajectory optimization parameters
  pnh_.param("minco/weight_time", nav_param.minco_weight_time_, nav_param.minco_weight_time_);
  pnh_.param("minco/weight_energy", nav_param.minco_weight_energy_, nav_param.minco_weight_energy_);
  pnh_.param("minco/weight_pos", nav_param.minco_weight_pos_, nav_param.minco_weight_pos_);
  pnh_.param("minco/weight_vel", nav_param.minco_weight_vel_, nav_param.minco_weight_vel_);
  pnh_.param("minco/weight_acc", nav_param.minco_weight_acc_, nav_param.minco_weight_acc_);
  pnh_.param("minco/weight_jerk", nav_param.minco_weight_jerk_, nav_param.minco_weight_jerk_);
  pnh_.param("minco/max_jerk", nav_param.minco_max_jerk_, nav_param.minco_max_jerk_);
  pnh_.param("minco/alloc_speed_ratio", nav_param.minco_alloc_speed_ratio_, nav_param.minco_alloc_speed_ratio_);
  pnh_.param("minco/length_per_piece", nav_param.minco_length_per_piece_, nav_param.minco_length_per_piece_);
  
  manager_->initialize();
  
  // Setup ROS interfaces
  odom_sub_ = nh_.subscribe("/odom", 10, &ForexNavFSM::odometryCallback, this);
  goal_sub_ = nh_.subscribe("/move_base_simple/goal", 10, &ForexNavFSM::goalCallback, this);
  viewpoints_sub_ = nh_.subscribe("/planner/viewpoints", 10, &ForexNavFSM::viewpointsCallback, this);
  
  // Subscribe to inflated occupancy grid for A* planning (安全膨胀版)
  map_sub_ = nh_.subscribe("/local_sensing/occupancy_grid_inflate", 10, &ForexNavFSM::mapCallback, this);
  
  // Subscribe to inflated 3D occupancy point cloud for 3D corridor generation (安全膨胀版)
  occ_cloud_3d_sub_ = nh_.subscribe("/local_sensing/occupancy_3d_inflate", 1, &ForexNavFSM::occCloud3DCallback, this);
  
  traj_pub_ = nh_.advertise<quadrotor_msgs::PositionCommand>("/planning/pos_cmd", 10);
  vis_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/planning_vis/trajectory", 10);
  vis_3d_corridor_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/planning_vis/corridor_3d", 10);
  vis_fuzzy_astar_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/planning/fuzzyAstar", 10);
  
  // Setup timers
  fsm_timer_ = nh_.createTimer(ros::Duration(0.1), &ForexNavFSM::FSMCallback, this);
  exec_timer_ = nh_.createTimer(ros::Duration(0.01), &ForexNavFSM::execTrajCallback, this);  // 20Hz
  
  ROS_INFO("ForexNav FSM initialized");
}

void ForexNavFSM::odometryCallback(const nav_msgs::Odometry::ConstPtr& msg) {
  fd_->odom_pos_ = Eigen::Vector3d(
    msg->pose.pose.position.x,
    msg->pose.pose.position.y,
    msg->pose.pose.position.z);
  
  fd_->odom_vel_ = Eigen::Vector3d(
    msg->twist.twist.linear.x,
    msg->twist.twist.linear.y,
    msg->twist.twist.linear.z);
  
  // Extract yaw from quaternion
  double qx = msg->pose.pose.orientation.x;
  double qy = msg->pose.pose.orientation.y;
  double qz = msg->pose.pose.orientation.z;
  double qw = msg->pose.pose.orientation.w;
  
  // Convert quaternion to yaw (simplified for 2D)
  double siny_cosp = 2.0 * (qw * qz + qx * qy);
  double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
  fd_->odom_yaw_ = std::atan2(siny_cosp, cosy_cosp);
  
  fd_->have_odom_ = true;
}

void ForexNavFSM::goalCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
  fd_->goal_pos_ = Eigen::Vector3d(
    msg->pose.position.x,
    msg->pose.position.y,
    fd_->odom_pos_.z());  // 使用机器人高度，忽略 RViz 的 z=0
  fd_->have_goal_ = true;
  
  ROS_INFO("Goal received: (%.2f, %.2f, %.2f)",
            fd_->goal_pos_.x(), fd_->goal_pos_.y(), fd_->goal_pos_.z());
  
  // Only trigger planning from WAIT_TRIGGER state (not FINISH)
  if (state_ == WAIT_TRIGGER) {
    transitState(PLAN_TRAJ, "goalCallback");
  }
}

void ForexNavFSM::viewpointsCallback(const geometry_msgs::PoseArray::ConstPtr& msg) {
  fd_->viewpoints_.clear();
  
  for (const auto& pose : msg->poses) {
    Viewpoint vp;
    vp.pos_ = Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
    
    // Extract yaw from quaternion
    double qx = pose.orientation.x;
    double qy = pose.orientation.y;
    double qz = pose.orientation.z;
    double qw = pose.orientation.w;
    
    double siny_cosp = 2.0 * (qw * qz + qx * qy);
    double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
    vp.yaw_ = std::atan2(siny_cosp, cosy_cosp);
    
    fd_->viewpoints_.push_back(vp);
  }
  
  fd_->have_viewpoints_ = !fd_->viewpoints_.empty();
  
  ROS_INFO_THROTTLE(2.0, "Received %zu viewpoints", fd_->viewpoints_.size());
}

void ForexNavFSM::mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg) {
  // Update map for A* planning
  manager_->setMap(msg);
}

void ForexNavFSM::occCloud3DCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromROSMsg(*msg, *cloud);
  // 从点云推断体素分辨率不太靠谱, 这里用launch参数或默认0.2
  double voxel_res = 0.2;
  pnh_.param("sfc/voxel_res_3d", voxel_res, 0.1);
  manager_->setOccCloud3D(cloud, voxel_res);
}

void ForexNavFSM::FSMCallback(const ros::TimerEvent&) {
  ROS_INFO_THROTTLE(1.0, "[FSM] State: %s", state_str_[state_].c_str());
  
  switch (state_) {
    case INIT: {
      if (!fd_->have_odom_) {
        ROS_WARN_THROTTLE(1.0, "Waiting for odometry...");
        return;
      }
      transitState(WAIT_TRIGGER, "FSM");
      break;
    }
    
    case WAIT_TRIGGER: {
      // Wait for goal
      break;
    }
    
    case PLAN_TRAJ: {
      int res = callPlanner();
      if (res == 0) {
        transitState(EXEC_TRAJ, "FSM");
      } else {
        ROS_WARN("Planning failed, staying in PLAN_TRAJ");
      }
      break;
    }
    
    case EXEC_TRAJ: {
      // Execution handled by execTrajCallback
      // Just log state periodically
      if (!traj_positions_.empty()) {
        double elapsed = (ros::Time::now() - traj_start_time_).toSec();
        ROS_INFO_THROTTLE(2.0, "[EXEC_TRAJ] Executing trajectory: elapsed=%.2f/%.2fs, %zu points",
                          elapsed, traj_times_.back(), traj_positions_.size());
      }
      break;
    }
    
    case FINISH: {
      // Reset goal and wait for next goal
      fd_->have_goal_ = false;
      fd_->goal_pos_ = Vector3d::Zero();  // Clear goal position
      // Clear trajectory
      traj_positions_.clear();
      traj_yaws_.clear();
      traj_times_.clear();
      has_minco_traj_ = false;
      // Clear visualization data
      astar_path_.clear();
      minco_traj_.clear();
      corridors_.clear();
      corridors_3d_.clear();
      // Publish empty markers to clear RViz
      visualize();
      ROS_INFO("\033[32m[FSM] Task finished, waiting for new goal\033[0m");
      transitState(WAIT_TRIGGER, "FSM");
      break;
    }
  }
}

void ForexNavFSM::execTrajCallback(const ros::TimerEvent&) {
  // PLAN_TRAJ 期间也继续发布旧轨迹，避免速度骤降
  if (state_ != EXEC_TRAJ && state_ != PLAN_TRAJ) return;
  
  if (traj_positions_.empty() || traj_times_.empty()) {
    if (state_ == EXEC_TRAJ) {
      // No trajectory, replan
      transitState(PLAN_TRAJ, "execTrajCallback");
    }
    return;
  }
  
  // Calculate elapsed time since trajectory start
  double elapsed = (ros::Time::now() - traj_start_time_).toSec();
  double total_time = traj_times_.back();
  
  // 仅在 EXEC_TRAJ 状态下做重规划判断（PLAN_TRAJ 时跳过，因为已经在重规划了）
  if (state_ == EXEC_TRAJ) {
    // Check if reached the final goal - unified distance-based check
    double dist_to_goal = (fd_->odom_pos_ - fd_->goal_pos_).norm();
    const double goal_reached_threshold = fp_->goal_reached_threshold_;
    
    // Unified goal check: use distance only
    if (dist_to_goal < goal_reached_threshold) {
      ROS_INFO("\033[32m[FSM] Goal reached! Distance: %.3f m. Stopping.\033[0m", dist_to_goal);
      // Clear trajectory to stop robot
      traj_positions_.clear();
      traj_yaws_.clear();
      traj_times_.clear();
      // Publish final position (goal position) to stop robot
      publishFinalPosition();
      transitState(FINISH, "execTrajCallback");
      return;
    }
    
    // Replan if traj is almost fully executed
    double time_to_end = total_time - elapsed;
    if (time_to_end < fp_->replan_thresh1_) {
      // When close to goal, do NOT replan to avoid oscillation
      if (dist_to_goal < fp_->no_replan_dist_to_goal_) {
        ROS_INFO("[FSM] Near goal (dist=%.3f m < no_replan_dist_to_goal=%.3f m), skip replan and keep executing", 
                  dist_to_goal, fp_->no_replan_dist_to_goal_);
      } else {
        // Before replanning, check if we're already at goal
        if (dist_to_goal < goal_reached_threshold) {
          ROS_INFO("\033[32m[FSM] Traj finished and goal reached (%.3f m). Stopping.\033[0m", dist_to_goal);
          transitState(FINISH, "execTrajCallback");
          return;
        }
        ROS_WARN("[REPLAN] Traj almost finished (time_to_end=%.3f < %.3f s), replanning", 
                  time_to_end, fp_->replan_thresh1_);
        transitState(PLAN_TRAJ, "execTrajCallback");
        // 不 return，继续发布旧轨迹
      }
    }
    
    // Periodic replanning after some time
    if (elapsed > fp_->replan_thresh3_) {
      // When close to goal, do NOT replan to avoid oscillation
      if (dist_to_goal < fp_->no_replan_dist_to_goal_) {
        ROS_INFO("[FSM] Near goal (dist=%.3f m < no_replan_dist_to_goal=%.3f m), skip periodic replan", 
                  dist_to_goal, fp_->no_replan_dist_to_goal_);
      } else {
        // Before replanning, check if we're already at goal
        if (dist_to_goal < goal_reached_threshold) {
          ROS_INFO("\033[32m[FSM] Periodic replan check: goal reached (%.3f m). Stopping.\033[0m", dist_to_goal);
          transitState(FINISH, "execTrajCallback");
          return;
        }
        
        // Check minimum time between replans
        double time_since_last_replan = (ros::Time::now() - last_replan_time_).toSec();
        if (time_since_last_replan > fp_->replan_time_) {
          // Optional: Check position deviation
          double traj_dt = manager_->getNavParam().traj_dt_;
          size_t expected_idx = std::min(static_cast<size_t>(elapsed / traj_dt), traj_positions_.size() - 1);
          Vector3d expected_pos = (expected_idx < traj_positions_.size()) ? traj_positions_[expected_idx] : traj_positions_.back();
          double pos_deviation = (fd_->odom_pos_ - expected_pos).norm();
          
          if (pos_deviation > fp_->replan_thresh2_) {
            ROS_WARN("[REPLAN] Position deviation too large: %.3f > %.3f m, replanning",
                      pos_deviation, fp_->replan_thresh2_);
          } else {
            ROS_INFO("[REPLAN] Periodic replanning: elapsed=%.3f > %.3f s",
                      elapsed, fp_->replan_thresh3_);
          }
          last_replan_time_ = ros::Time::now();
          transitState(PLAN_TRAJ, "execTrajCallback");
          // 不 return，继续发布旧轨迹
        }
      }
    }
  }  // end if (state_ == EXEC_TRAJ)
  
  // 始终发布轨迹（即使在 PLAN_TRAJ 期间，继续跟踪旧轨迹）
  publishTrajectory();
  
  // Update index for visualization
  double traj_dt = manager_->getNavParam().traj_dt_;
  traj_index_ = std::min(static_cast<size_t>(elapsed / traj_dt), traj_positions_.size() - 1);
}

int ForexNavFSM::callPlanner() {
  if (!fd_->have_odom_ || !fd_->have_goal_) {
    ROS_WARN("Missing odom or goal for planning");
    return -1;
  }
  
  // Check if goal is valid (not zero)
  if (fd_->goal_pos_.norm() < 0.01) {
    ROS_WARN("Invalid goal position (zero)");
    return -1;
  }
  
  // Check if already at goal
  double dist_to_goal = (fd_->odom_pos_ - fd_->goal_pos_).norm();
  if (dist_to_goal < fp_->goal_reached_threshold_) {
    ROS_INFO("Already at goal (dist=%.3f m), no planning needed", dist_to_goal);
    return -1;
  }
  
  std::vector<Eigen::Vector3d> path;
  std::vector<double> yaws;
  std::vector<double> times;
  std::vector<Eigen::Vector3d> astar_path;
  
  // 用旧轨迹预测速度作为 start_vel，避免因 PD 跟踪延迟导致 odom_vel 偏低
  Eigen::Vector3d start_vel = fd_->odom_vel_;  // 兜底：用 odom 速度
  if (has_minco_traj_) {
    // Use continuous trajectory for accurate velocity
    double elapsed = (ros::Time::now() - traj_start_time_).toSec();
    double t = std::max(0.0, std::min(elapsed, minco_traj_duration_));
    start_vel = minco_traj_obj_.getVel(t);
  } else if (!traj_positions_.empty() && traj_times_.size() >= 2) {
    double elapsed = (ros::Time::now() - traj_start_time_).toSec();
    double traj_dt = manager_->getNavParam().traj_dt_;
    size_t idx = std::min(static_cast<size_t>(elapsed / traj_dt), traj_positions_.size() - 1);
    if (idx > 0 && idx < traj_positions_.size()) {
      double dt = traj_times_[idx] - traj_times_[idx - 1];
      if (dt > 1e-6) {
        start_vel = (traj_positions_[idx] - traj_positions_[idx - 1]) / dt;
      }
    }
  }
  
  Trajectory<5> minco_traj;
  int res = manager_->planNavMotion(
    fd_->odom_pos_, start_vel, fd_->odom_yaw_,
    fd_->goal_pos_,
    fd_->viewpoints_,
    path, yaws, times, astar_path, &minco_traj);
  
  if (res < 0) {
    ROS_ERROR("Planning failed");
    return -1;
  }
  
  // Store trajectory (discrete points for visualization/debug)
  traj_positions_ = path;
  traj_yaws_ = yaws;
  traj_times_ = times;
  traj_index_ = 0;
  traj_start_time_ = ros::Time::now();
  last_replan_time_ = ros::Time::now();  // Initialize last replan time
  
  // Store continuous MINCO trajectory for real-time p/v/a evaluation
  if (minco_traj.getPieceNum() > 0) {
    minco_traj_obj_ = minco_traj;
    minco_traj_duration_ = minco_traj.getTotalDuration();
    minco_start_yaw_ = fd_->odom_yaw_;
    minco_end_yaw_ = yaws.empty() ? fd_->odom_yaw_ : yaws.back();
    has_minco_traj_ = true;
    ROS_INFO("[FSM] Stored continuous MINCO trajectory: duration=%.2f s, %d pieces",
              minco_traj_duration_, minco_traj.getPieceNum());
  } else {
    has_minco_traj_ = false;
    ROS_WARN("[FSM] No continuous MINCO trajectory available, using fallback discrete publishing");
  }
  
  // Store path data for visualization
  astar_path_ = astar_path;  // A* path (yellow)
  minco_traj_ = path;         // MINCO trajectory (red)
  corridors_ = manager_->getLastCorridors();  // SFC 2D corridors (green boxes)
  
  // 生成3D走廊并存储
  manager_->generateCorridors3D(astar_path);
  corridors_3d_ = manager_->getLastCorridors3D();
  
  ROS_INFO("Planning succeeded: %zu waypoints, %zu trajectory points", 
            path.size(), path.size());
  if (!path.empty()) {
    ROS_INFO("Start: (%.2f, %.2f, %.2f), End: (%.2f, %.2f, %.2f)",
              path[0].x(), path[0].y(), path[0].z(),
              path.back().x(), path.back().y(), path.back().z());
    
    // Debug: Check if trajectory points are different
    if (path.size() > 1) {
      double path_length = 0.0;
      for (size_t i = 1; i < path.size(); ++i) {
        path_length += (path[i] - path[i-1]).norm();
      }
      ROS_INFO("Trajectory path length: %.2f m, first-last distance: %.2f m",
                path_length, (path.back() - path[0]).norm());
      
      // Check if all points are the same
      bool all_same = true;
      for (size_t i = 1; i < path.size(); ++i) {
        if ((path[i] - path[0]).norm() > 0.01) {
          all_same = false;
          break;
        }
      }
      if (all_same) {
        ROS_WARN("WARNING: All trajectory points are the same!");
      }
    }
  }
  
  visualize();
  
  return 0;
}

double ForexNavFSM::normalizeAngle(double a) {
  while (a > M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}

void ForexNavFSM::publishTrajectory() {
  if (!has_minco_traj_) {
    // Fallback: use discrete points (backward compatibility)
    publishTrajectoryFallback();
    return;
  }
  
  double t = (ros::Time::now() - traj_start_time_).toSec();
  Vector3d pos, vel, acc;
  double yaw, yawdot;
  
  if (t >= 0.0 && t < minco_traj_duration_) {
    // Real-time analytical evaluation from continuous trajectory (like UAV traj_server)
    pos = minco_traj_obj_.getPos(t);
    vel = minco_traj_obj_.getVel(t);
    acc = minco_traj_obj_.getAcc(t);
    
    // Yaw from velocity direction (consistent with MINCO sampling logic)
    if (vel.head<2>().norm() > 0.1) {
      yaw = std::atan2(vel.y(), vel.x());
    } else {
      double alpha = t / minco_traj_duration_;
      double yaw_diff = normalizeAngle(minco_end_yaw_ - minco_start_yaw_);
      yaw = minco_start_yaw_ + alpha * yaw_diff;
    }
    yaw = normalizeAngle(yaw);
    
    // Yaw rate: compute from velocity direction change using jerk
    // For simplicity, use finite difference on yaw from velocity direction
    double dt_small = 0.005;
    double t_next = std::min(t + dt_small, minco_traj_duration_);
    Vector3d vel_next = minco_traj_obj_.getVel(t_next);
    if (vel.head<2>().norm() > 0.1 && vel_next.head<2>().norm() > 0.1) {
      double yaw_next = std::atan2(vel_next.y(), vel_next.x());
      yawdot = normalizeAngle(yaw_next - yaw) / dt_small;
    } else {
      yawdot = normalizeAngle(minco_end_yaw_ - minco_start_yaw_) / minco_traj_duration_;
    }
  } else {
    // Trajectory finished: hold final position, zero velocity
    pos = minco_traj_obj_.getPos(minco_traj_duration_);
    vel.setZero();
    acc.setZero();
    yaw = normalizeAngle(minco_end_yaw_);
    yawdot = 0.0;
  }
  
  // Publish PositionCommand with full trajectory information (p, v, a, yaw, yaw_dot)
  quadrotor_msgs::PositionCommand cmd;
  cmd.header.stamp = ros::Time::now();
  cmd.header.frame_id = "odom";
  cmd.position.x = pos.x();
  cmd.position.y = pos.y();
  cmd.position.z = pos.z();
  cmd.velocity.x = vel.x();
  cmd.velocity.y = vel.y();
  cmd.velocity.z = vel.z();
  cmd.acceleration.x = acc.x();
  cmd.acceleration.y = acc.y();
  cmd.acceleration.z = acc.z();
  cmd.yaw = yaw;
  cmd.yaw_dot = yawdot;
  // Default gains (controller can override via parameters)
  cmd.kx[0] = cmd.kx[1] = cmd.kx[2] = 0.0;
  cmd.kv[0] = cmd.kv[1] = cmd.kv[2] = 0.0;
  cmd.trajectory_flag = quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_READY;
  traj_pub_.publish(cmd);
  
  // Update index for visualization
  double traj_dt = manager_->getNavParam().traj_dt_;
  traj_index_ = std::min(static_cast<size_t>(t / traj_dt), 
                          traj_positions_.empty() ? 0 : traj_positions_.size() - 1);
}

void ForexNavFSM::publishTrajectoryFallback() {
  // Fallback: discrete-point interpolation + PositionCommand (no continuous trajectory)
  if (traj_positions_.empty() || traj_times_.empty()) {
    ROS_WARN_THROTTLE(1.0, "[PUBLISH] No trajectory to publish");
    return;
  }
  
  double elapsed = (ros::Time::now() - traj_start_time_).toSec();
  
  size_t idx = 0;
  for (size_t i = 0; i < traj_times_.size(); ++i) {
    if (traj_times_[i] <= elapsed) idx = i;
    else break;
  }
  if (idx >= traj_positions_.size()) idx = traj_positions_.size() - 1;
  
  Vector3d pos, vel;
  double yaw;
  vel.setZero();
  
  if (idx < traj_positions_.size() - 1 && elapsed < traj_times_.back()) {
    double t0 = traj_times_[idx];
    double t1 = traj_times_[idx + 1];
    double alpha = (t1 > t0) ? (elapsed - t0) / (t1 - t0) : 0.0;
    alpha = std::max(0.0, std::min(1.0, alpha));
    
    pos = traj_positions_[idx] + alpha * (traj_positions_[idx + 1] - traj_positions_[idx]);
    
    // Estimate velocity from discrete points
    double dt = t1 - t0;
    if (dt > 1e-6) {
      vel = (traj_positions_[idx + 1] - traj_positions_[idx]) / dt;
    }
    
    double yaw0 = normalizeAngle((idx < traj_yaws_.size()) ? traj_yaws_[idx] : 0.0);
    double yaw1 = normalizeAngle((idx + 1 < traj_yaws_.size()) ? traj_yaws_[idx + 1] : yaw0);
    double yaw_diff = normalizeAngle(yaw1 - yaw0);
    yaw = normalizeAngle(yaw0 + alpha * yaw_diff);
  } else {
    pos = traj_positions_[idx];
    yaw = normalizeAngle((idx < traj_yaws_.size()) ? traj_yaws_[idx] : 0.0);
  }
  
  quadrotor_msgs::PositionCommand cmd;
  cmd.header.stamp = ros::Time::now();
  cmd.header.frame_id = "odom";
  cmd.position.x = pos.x();
  cmd.position.y = pos.y();
  cmd.position.z = pos.z();
  cmd.velocity.x = vel.x();
  cmd.velocity.y = vel.y();
  cmd.velocity.z = vel.z();
  cmd.acceleration.x = 0.0;
  cmd.acceleration.y = 0.0;
  cmd.acceleration.z = 0.0;
  cmd.yaw = yaw;
  cmd.yaw_dot = 0.0;
  cmd.kx[0] = cmd.kx[1] = cmd.kx[2] = 0.0;
  cmd.kv[0] = cmd.kv[1] = cmd.kv[2] = 0.0;
  cmd.trajectory_flag = quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_READY;
  traj_pub_.publish(cmd);
  
  traj_index_ = idx;
}

void ForexNavFSM::publishFinalPosition() {
  // Publish goal position with zero velocity to make robot stop at goal
  quadrotor_msgs::PositionCommand cmd;
  cmd.header.stamp = ros::Time::now();
  cmd.header.frame_id = "odom";
  
  cmd.position.x = fd_->goal_pos_.x();
  cmd.position.y = fd_->goal_pos_.y();
  cmd.position.z = fd_->goal_pos_.z();
  cmd.velocity.x = 0.0;
  cmd.velocity.y = 0.0;
  cmd.velocity.z = 0.0;
  cmd.acceleration.x = 0.0;
  cmd.acceleration.y = 0.0;
  cmd.acceleration.z = 0.0;
  cmd.yaw = fd_->odom_yaw_;
  cmd.yaw_dot = 0.0;
  cmd.kx[0] = cmd.kx[1] = cmd.kx[2] = 0.0;
  cmd.kv[0] = cmd.kv[1] = cmd.kv[2] = 0.0;
  cmd.trajectory_flag = quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_COMPLETED;
  
  // Publish multiple times to ensure robot stops
  for (int i = 0; i < 5; ++i) {
    traj_pub_.publish(cmd);
  }
  
  ROS_INFO("[FSM] Published final position to stop robot at goal");
}

void ForexNavFSM::visualize() {
  visualization_msgs::MarkerArray markers;
  
  // Clear previous markers
  visualization_msgs::Marker delete_marker;
  delete_marker.action = visualization_msgs::Marker::DELETEALL;
  markers.markers.push_back(delete_marker);
  
  // Visualize A* path (yellow line)
  if (!astar_path_.empty()) {
    visualization_msgs::Marker astar_marker;
    astar_marker.header.frame_id = "world";
    astar_marker.header.stamp = ros::Time::now();
    astar_marker.ns = "astar_path";
    astar_marker.id = 0;
    astar_marker.type = visualization_msgs::Marker::LINE_STRIP;
    astar_marker.action = visualization_msgs::Marker::ADD;
    astar_marker.pose.orientation.w = 1.0;
    astar_marker.scale.x = 0.1;  // Line width
    astar_marker.color.r = 1.0;
    astar_marker.color.g = 1.0;
    astar_marker.color.b = 0.0;
    astar_marker.color.a = 1.0;
    
    for (const auto& pt : astar_path_) {
      geometry_msgs::Point p;
      p.x = pt.x();
      p.y = pt.y();
      p.z = pt.z();
      astar_marker.points.push_back(p);
    }
    markers.markers.push_back(astar_marker);
  }
  
  // Visualize MINCO trajectory (red line)
  if (!minco_traj_.empty()) {
    visualization_msgs::Marker minco_marker;
    minco_marker.header.frame_id = "world";
    minco_marker.header.stamp = ros::Time::now();
    minco_marker.ns = "minco_trajectory";
    minco_marker.id = 0;
    minco_marker.type = visualization_msgs::Marker::LINE_STRIP;
    minco_marker.action = visualization_msgs::Marker::ADD;
    minco_marker.pose.orientation.w = 1.0;
    minco_marker.scale.x = 0.15;  // Line width (thicker than A*)
    minco_marker.color.r = 1.0;
    minco_marker.color.g = 0.0;
    minco_marker.color.b = 0.0;
    minco_marker.color.a = 1.0;
    
    for (const auto& pt : minco_traj_) {
      geometry_msgs::Point p;
      p.x = pt.x();
      p.y = pt.y();
      p.z = pt.z();
      minco_marker.points.push_back(p);
    }
    markers.markers.push_back(minco_marker);
  }
  
  // Visualize SFC 2D corridors (green wireframe boxes, 12 edges)
  for (size_t i = 0; i < corridors_.size(); ++i) {
    const auto& corr = corridors_[i];
    double x_min = corr[0], x_max = corr[1];
    double y_min = corr[2], y_max = corr[3];
    double z_min = corr[4], z_max = corr[5];
    
    visualization_msgs::Marker corridor_marker;
    corridor_marker.header.frame_id = "world";
    corridor_marker.header.stamp = ros::Time::now();
    corridor_marker.ns = "sfc_corridors";
    corridor_marker.id = static_cast<int>(i);
    corridor_marker.type = visualization_msgs::Marker::LINE_LIST;
    corridor_marker.action = visualization_msgs::Marker::ADD;
    corridor_marker.pose.orientation.w = 1.0;
    corridor_marker.scale.x = 0.05;
    corridor_marker.color.r = 0.0;
    corridor_marker.color.g = 1.0;
    corridor_marker.color.b = 0.0;
    corridor_marker.color.a = 0.8;
    
    // 8个顶点
    geometry_msgs::Point p[8];
    p[0].x = x_min; p[0].y = y_min; p[0].z = z_min;
    p[1].x = x_max; p[1].y = y_min; p[1].z = z_min;
    p[2].x = x_max; p[2].y = y_max; p[2].z = z_min;
    p[3].x = x_min; p[3].y = y_max; p[3].z = z_min;
    p[4].x = x_min; p[4].y = y_min; p[4].z = z_max;
    p[5].x = x_max; p[5].y = y_min; p[5].z = z_max;
    p[6].x = x_max; p[6].y = y_max; p[6].z = z_max;
    p[7].x = x_min; p[7].y = y_max; p[7].z = z_max;
    
    // 底面4条边
    corridor_marker.points.push_back(p[0]); corridor_marker.points.push_back(p[1]);
    corridor_marker.points.push_back(p[1]); corridor_marker.points.push_back(p[2]);
    corridor_marker.points.push_back(p[2]); corridor_marker.points.push_back(p[3]);
    corridor_marker.points.push_back(p[3]); corridor_marker.points.push_back(p[0]);
    // 顶面4条边
    corridor_marker.points.push_back(p[4]); corridor_marker.points.push_back(p[5]);
    corridor_marker.points.push_back(p[5]); corridor_marker.points.push_back(p[6]);
    corridor_marker.points.push_back(p[6]); corridor_marker.points.push_back(p[7]);
    corridor_marker.points.push_back(p[7]); corridor_marker.points.push_back(p[4]);
    // 竖直4条边
    corridor_marker.points.push_back(p[0]); corridor_marker.points.push_back(p[4]);
    corridor_marker.points.push_back(p[1]); corridor_marker.points.push_back(p[5]);
    corridor_marker.points.push_back(p[2]); corridor_marker.points.push_back(p[6]);
    corridor_marker.points.push_back(p[3]); corridor_marker.points.push_back(p[7]);
    
    markers.markers.push_back(corridor_marker);
  }
  
  // Publish 3D SFC corridors on separate topic (cyan wireframe boxes, 12 edges)
  {
    visualization_msgs::MarkerArray markers_3d;
    
    visualization_msgs::Marker del;
    del.action = visualization_msgs::Marker::DELETEALL;
    markers_3d.markers.push_back(del);
    
    for (size_t i = 0; i < corridors_3d_.size(); ++i) {
      const auto& corr = corridors_3d_[i];
      double x_min = corr[0], x_max = corr[1];
      double y_min = corr[2], y_max = corr[3];
      double z_min = corr[4], z_max = corr[5];
      
      visualization_msgs::Marker m;
      m.header.frame_id = "world";
      m.header.stamp = ros::Time::now();
      m.ns = "sfc_corridors_3d";
      m.id = static_cast<int>(i);
      m.type = visualization_msgs::Marker::LINE_LIST;
      m.action = visualization_msgs::Marker::ADD;
      m.pose.orientation.w = 1.0;
      m.scale.x = 0.06;
      m.color.r = 0.0;
      m.color.g = 0.8;
      m.color.b = 1.0;
      m.color.a = 0.7;
      
      geometry_msgs::Point p[8];
      p[0].x = x_min; p[0].y = y_min; p[0].z = z_min;
      p[1].x = x_max; p[1].y = y_min; p[1].z = z_min;
      p[2].x = x_max; p[2].y = y_max; p[2].z = z_min;
      p[3].x = x_min; p[3].y = y_max; p[3].z = z_min;
      p[4].x = x_min; p[4].y = y_min; p[4].z = z_max;
      p[5].x = x_max; p[5].y = y_min; p[5].z = z_max;
      p[6].x = x_max; p[6].y = y_max; p[6].z = z_max;
      p[7].x = x_min; p[7].y = y_max; p[7].z = z_max;
      
      // 底面
      m.points.push_back(p[0]); m.points.push_back(p[1]);
      m.points.push_back(p[1]); m.points.push_back(p[2]);
      m.points.push_back(p[2]); m.points.push_back(p[3]);
      m.points.push_back(p[3]); m.points.push_back(p[0]);
      // 顶面
      m.points.push_back(p[4]); m.points.push_back(p[5]);
      m.points.push_back(p[5]); m.points.push_back(p[6]);
      m.points.push_back(p[6]); m.points.push_back(p[7]);
      m.points.push_back(p[7]); m.points.push_back(p[4]);
      // 竖直
      m.points.push_back(p[0]); m.points.push_back(p[4]);
      m.points.push_back(p[1]); m.points.push_back(p[5]);
      m.points.push_back(p[2]); m.points.push_back(p[6]);
      m.points.push_back(p[3]); m.points.push_back(p[7]);
      
      markers_3d.markers.push_back(m);
    }
    
    vis_3d_corridor_pub_.publish(markers_3d);
  }
  
  // Publish candidate paths separately on /planning/fuzzyAstar (thick green curves)
  {
    visualization_msgs::MarkerArray fuzzy_markers;
    visualization_msgs::Marker del_fuzzy;
    del_fuzzy.action = visualization_msgs::Marker::DELETEALL;
    fuzzy_markers.markers.push_back(del_fuzzy);

    const auto& candidate_paths = manager_->getLastCandidatePaths();
    for (size_t i = 0; i < candidate_paths.size(); ++i) {
      const auto& path = candidate_paths[i];
      if (path.empty()) continue;
      visualization_msgs::Marker line_marker;
      line_marker.header.frame_id = "world";
      line_marker.header.stamp = ros::Time::now();
      line_marker.ns = "fuzzy_astar";
      line_marker.id = static_cast<int>(i);
      line_marker.type = visualization_msgs::Marker::LINE_STRIP;
      line_marker.action = visualization_msgs::Marker::ADD;
      line_marker.pose.orientation.w = 1.0;
      line_marker.scale.x = 0.25;  // thick line
      line_marker.color.r = 0.0;
      line_marker.color.g = 1.0;
      line_marker.color.b = 0.0;
      line_marker.color.a = 1.0;
      for (const auto& pt : path) {
        geometry_msgs::Point p;
        p.x = pt.x();
        p.y = pt.y();
        p.z = pt.z();
        line_marker.points.push_back(p);
      }
      fuzzy_markers.markers.push_back(line_marker);
    }
    vis_fuzzy_astar_pub_.publish(fuzzy_markers);
  }
  
  // Draw trajectory
  visualization_msgs::Marker traj_marker;
  traj_marker.header.frame_id = "world";
  traj_marker.header.stamp = ros::Time::now();
  traj_marker.ns = "trajectory";
  traj_marker.id = 0;
  traj_marker.type = visualization_msgs::Marker::LINE_STRIP;
  traj_marker.action = visualization_msgs::Marker::ADD;
  traj_marker.pose.orientation.w = 1.0;
  traj_marker.scale.x = 0.1;
  traj_marker.color.r = 1.0;
  traj_marker.color.g = 0.0;
  traj_marker.color.b = 0.0;
  traj_marker.color.a = 1.0;
  
  for (const auto& pos : traj_positions_) {
    geometry_msgs::Point pt;
    pt.x = pos.x();
    pt.y = pos.y();
    pt.z = pos.z();
    traj_marker.points.push_back(pt);
  }
  
  markers.markers.push_back(traj_marker);
  vis_pub_.publish(markers);
}

void ForexNavFSM::transitState(EXPL_STATE new_state, const std::string& pos_call) {
  int pre_s = static_cast<int>(state_);
  state_ = new_state;
  ROS_INFO("[%s]: from %s to %s",
            pos_call.c_str(), state_str_[pre_s].c_str(), state_str_[static_cast<int>(new_state)].c_str());
}

}  // namespace forex_nav
