// #include <fstream>
#include <plan_manage/planner_manager.h>
#include <plan_env/sdf_map.h>
#include <plan_env/raycast.h>

#include <thread>
#include <cmath>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>

namespace fast_planner {
// SECTION interfaces for setup and query

FastPlannerManager::FastPlannerManager() {
}

FastPlannerManager::~FastPlannerManager() {
  std::cout << "des manager" << std::endl;
}

void FastPlannerManager::initPlanModules(ros::NodeHandle& nh) {
  /* read algorithm parameters */

  nh.param("manager/max_vel", pp_.max_vel_, -1.0);
  nh.param("manager/max_acc", pp_.max_acc_, -1.0);
  nh.param("manager/max_jerk", pp_.max_jerk_, -1.0);
  nh.param("manager/accept_vel", pp_.accept_vel_, pp_.max_vel_ + 0.5);
  nh.param("manager/accept_acc", pp_.accept_acc_, pp_.max_acc_ + 0.5);
  nh.param("manager/max_yawdot", pp_.max_yawdot_, -1.0);
  nh.param("manager/dynamic_environment", pp_.dynamic_, -1);
  nh.param("manager/clearance_threshold", pp_.clearance_, -1.0);
  nh.param("manager/local_segment_length", pp_.local_traj_len_, -1.0);
  nh.param("manager/control_points_distance", pp_.ctrl_pt_dist, -1.0);
  nh.param("manager/bspline_degree", pp_.bspline_degree_, 3);
  nh.param("manager/min_time", pp_.min_time_, false);
  
  // Frontend planner type selection
  // 0: Standard Kinodynamic A* (default, for drones/holonomic robots)
  // 1: Ackermann Kinodynamic A* (for car-like vehicles)
  nh.param("manager/frontend_planner_type", pp_.frontend_planner_type_, 0);
  ROS_WARN("========================================");
  ROS_WARN("[PlanManager::initPlanModules] Reading frontend_planner_type from param server");
  ROS_WARN("Read value: frontend_planner_type = %d", pp_.frontend_planner_type_);
  ROS_WARN("Expected: 1 for Ackermann, 0 for Standard");
  ROS_WARN("========================================");
  
  // MINCO backend configuration
  nh.param("manager/use_minco_backend", use_minco_backend_, true);
  nh.param("manager/debug_compare_mode", debug_compare_mode_, false);
  
  // Initialize debug publishers if in debug mode
  if (debug_compare_mode_) {
    debug_nh_ = nh;
    debug_astar_path_pub_ = nh.advertise<visualization_msgs::Marker>("/debug/astar_path", 10);
    debug_bspline_traj_pub_ = nh.advertise<visualization_msgs::Marker>("/debug/bspline_traj", 10);
    debug_minco_traj_pub_ = nh.advertise<visualization_msgs::Marker>("/debug/minco_traj", 10);
    ROS_WARN("[PlanManager] Debug compare mode ENABLED - publishing A*/B-spline/MINCO for comparison");
  }
  
  // Rectangle collision detection parameters (for legged robots)
  nh.param("manager/collision_rect_length", collision_rect_length_, 0.5);
  nh.param("manager/collision_rect_width", collision_rect_width_, 0.3);
  nh.param("manager/collision_check_interval", collision_check_interval_, 0.02);
  collision_rect_viz_pub_ = nh.advertise<visualization_msgs::Marker>("/planning_vis/collision_rects", 10);
  ROS_INFO("[PlanManager] Rectangle collision detection: length=%.2f, width=%.2f, interval=%.3f", 
           collision_rect_length_, collision_rect_width_, collision_check_interval_);

  bool use_geometric_path, use_kinodynamic_path, use_topo_path, use_optimization,
      use_active_perception;
  nh.param("manager/use_geometric_path", use_geometric_path, false);
  nh.param("manager/use_kinodynamic_path", use_kinodynamic_path, false);
  nh.param("manager/use_topo_path", use_topo_path, false);
  nh.param("manager/use_optimization", use_optimization, false);
  nh.param("manager/use_active_perception", use_active_perception, false);

  local_data_.traj_id_ = 0;
  sdf_map_.reset(new SDFMap);
  sdf_map_->initMap(nh);
  edt_environment_.reset(new EDTEnvironment);
  edt_environment_->setMap(sdf_map_);

  if (use_geometric_path) {
    path_finder_.reset(new Astar);
    // path_finder_->setParam(nh);
    // path_finder_->setEnvironment(edt_environment_);
    // path_finder_->init();
    path_finder_->init(nh, edt_environment_);
  }

  if (use_kinodynamic_path) {
    if (pp_.frontend_planner_type_ == 0) {
      // Standard Kinodynamic A* for drones/holonomic robots
      kino_path_finder_.reset(new KinodynamicAstar);
      kino_path_finder_->setParam(nh);
      kino_path_finder_->setEnvironment(edt_environment_);
      kino_path_finder_->init();
      ROS_WARN("========================================");
      ROS_WARN("[PlanManager] Using Standard Kinodynamic A* (for drones/holonomic robots)");
      ROS_WARN("frontend_planner_type = 0");
      ROS_WARN("========================================");
    } else if (pp_.frontend_planner_type_ == 1) {
      // Ackermann Kinodynamic A* for car-like vehicles
      ackermann_path_finder_.reset(new AckermannKinoAstar);
      ackermann_path_finder_->setParam(nh);  // 设置参数（内部会保存 nh_）
      ackermann_path_finder_->setEnvironment(edt_environment_);
      ackermann_path_finder_->init();
      ROS_WARN("========================================");
      ROS_WARN("[PlanManager] Using Ackermann Kinodynamic A* (for car-like vehicles)");
      ROS_WARN("frontend_planner_type = 1");
      ROS_WARN("State space: 2D (x, y, θ), with non-holonomic constraints");
      ROS_WARN("========================================");
    } else {
      ROS_ERROR("[PlanManager] Invalid frontend_planner_type: %d (must be 0 or 1)", pp_.frontend_planner_type_);
      kino_path_finder_.reset(new KinodynamicAstar);
      kino_path_finder_->setParam(nh);
      kino_path_finder_->setEnvironment(edt_environment_);
      kino_path_finder_->init();
    }
  }

  if (use_optimization) {
    bspline_optimizers_.resize(10);
    for (int i = 0; i < 10; ++i) {
      bspline_optimizers_[i].reset(new BsplineOptimizer);
      bspline_optimizers_[i]->setParam(nh);
      bspline_optimizers_[i]->setEnvironment(edt_environment_);
    }
  }

  if (use_topo_path) {
    topo_prm_.reset(new TopologyPRM);
    topo_prm_->setEnvironment(edt_environment_);
    topo_prm_->init(nh);
  }

  if (use_active_perception) {
    frontier_finder_.reset(new FrontierFinder(edt_environment_, nh));
    heading_planner_.reset(new HeadingPlanner(nh));
    heading_planner_->setMap(sdf_map_);
    visib_util_.reset(new VisibilityUtil(nh));
    visib_util_->setEDTEnvironment(edt_environment_);
    plan_data_.view_cons_.idx_ = -1;
  }
  
  // Initialize MINCO backend if enabled
  if (use_minco_backend_) {
    gcopter_config_.reset(new GcopterConfig);
    gcopter_config_->init(nh);
    gcopter_viz_.reset(new Visualizer);
    gcopter_viz_->init(nh);
    std::cout << "[PlanManager] Using MINCO backend for trajectory optimization" << std::endl;
  } else {
    std::cout << "[PlanManager] Using B-spline backend for trajectory optimization" << std::endl;
  }
}

void FastPlannerManager::setGlobalWaypoints(vector<Eigen::Vector3d>& waypoints) {
  plan_data_.global_waypoints_ = waypoints;
}

bool FastPlannerManager::checkTrajCollision(double& distance) {
  double t_now = (ros::Time::now() - local_data_.start_time_).toSec();

  Eigen::Vector3d cur_pt = local_data_.getPosition(t_now);  // Use unified interface
  double radius = 0.0;
  double fut_t = collision_check_interval_;
  int rect_id = 0;

  // Clear previous visualization markers
  visualization_msgs::Marker delete_marker;
  delete_marker.action = visualization_msgs::Marker::DELETEALL;
  collision_rect_viz_pub_.publish(delete_marker);

  while (radius < 6.0 && t_now + fut_t < local_data_.duration_) {
    Eigen::Vector3d fut_pt = local_data_.getPosition(t_now + fut_t);
    
    // Calculate yaw angle from velocity direction
    Eigen::Vector3d vel = local_data_.getVelocity(t_now + fut_t);
    double yaw = 0.0;
    if (vel.head(2).norm() > 0.01) {
      yaw = atan2(vel(1), vel(0));
    } else {
      // If velocity is too small, use direction to next point
      if (t_now + fut_t + collision_check_interval_ < local_data_.duration_) {
        Eigen::Vector3d next_pt = local_data_.getPosition(t_now + fut_t + collision_check_interval_);
        Eigen::Vector2d dir = (next_pt - fut_pt).head(2);
        if (dir.norm() > 0.01) {
          yaw = atan2(dir(1), dir(0));
        }
      }
    }
    
    // Check collision for rectangle at this position
    if (checkRectCollision(fut_pt, yaw, collision_rect_length_, collision_rect_width_)) {
      distance = radius;
      std::cout << "collision detected at: " << fut_pt.transpose() 
                << ", yaw: " << yaw * 180.0 / M_PI << " deg" << std::endl;
      return false;
    }
    
    // Visualize collision rectangle
    visualizeCollisionRect(fut_pt, yaw, collision_rect_length_, collision_rect_width_, rect_id++);
    
    radius = (fut_pt - cur_pt).norm();
    fut_t += collision_check_interval_;
  }

  return true;
}

// !SECTION

// SECTION kinodynamic replanning

bool FastPlannerManager::kinodynamicReplan(const Eigen::Vector3d& start_pt,
    const Eigen::Vector3d& start_vel, const Eigen::Vector3d& start_acc,
    const Eigen::Vector3d& end_pt, const Eigen::Vector3d& end_vel, const double& time_lb) {
  // ========== 强制输出，确保能看到 ==========
  ROS_ERROR("========================================");
  ROS_ERROR("[kinodynamicReplan] FUNCTION CALLED!");
  ROS_ERROR("========================================");
  
  std::cout << "[Kino replan]: start: " << start_pt.transpose() << ", " << start_vel.transpose()
            << ", " << start_acc.transpose() << ", goal:" << end_pt.transpose() << ", "
            << end_vel.transpose() << endl;

  if ((start_pt - end_pt).norm() < 1e-2) {
    cout << "Close goal" << endl;
    return false;
  }

  Eigen::Vector3d init_pos = start_pt;
  Eigen::Vector3d init_vel = start_vel;
  Eigen::Vector3d init_acc = start_acc;

  // Kinodynamic path searching
  auto t1 = ros::Time::now();
  int status;

  // 调试信息：检查规划器类型和初始化状态（使用ERROR级别确保能看到）
  ROS_ERROR("========================================");
  ROS_ERROR("[kinodynamicReplan] Checking planner configuration...");
  ROS_ERROR("frontend_planner_type_ = %d (1=Ackermann, 0=Standard)", pp_.frontend_planner_type_);
  ROS_ERROR("ackermann_path_finder_ is %s", ackermann_path_finder_ ? "INITIALIZED ✓" : "NULL ✗");
  ROS_ERROR("kino_path_finder_ is %s", kino_path_finder_ ? "INITIALIZED ✓" : "NULL ✗");
  ROS_ERROR("========================================");

  if (pp_.frontend_planner_type_ == 1 && ackermann_path_finder_) {
    // Use Ackermann Kinodynamic A* for car-like vehicles
    ROS_ERROR("========================================");
    ROS_ERROR("[ACKERMANN PLANNER] ====== ACKERMANN PATH SEARCH START ======");
    ROS_ERROR("Condition check: frontend_planner_type_==1 ✓ AND ackermann_path_finder_!=NULL ✓");
    ROS_ERROR("========================================");
    
    // Calculate yaw from velocity direction (for 2D ground vehicles)
    double start_yaw = atan2(start_vel(1), start_vel(0));
    double start_vel_mag = start_vel.head(2).norm();
    double end_yaw = atan2(end_vel(1), end_vel(0));
    double end_vel_mag = end_vel.head(2).norm();
    
    // If velocity is too small, use direction to goal
    if (start_vel_mag < 0.1) {
      Eigen::Vector2d dir_to_goal = (end_pt - start_pt).head(2);
      if (dir_to_goal.norm() > 0.1) {
        start_yaw = atan2(dir_to_goal(1), dir_to_goal(0));
      }
    }
    if (end_vel_mag < 0.1) {
      Eigen::Vector2d dir_from_start = (end_pt - start_pt).head(2);
      if (dir_from_start.norm() > 0.1) {
        end_yaw = atan2(dir_from_start(1), dir_from_start(0));
      }
    }
    
    Eigen::Vector4d ackermann_start(start_pt(0), start_pt(1), start_yaw, start_vel_mag);  // x, y, yaw, v
    Eigen::Vector4d ackermann_end(end_pt(0), end_pt(1), end_yaw, end_vel_mag);
    Eigen::Vector2d init_ctrl(0.0, 0.0);  // steering, arc
    
    ROS_ERROR("[Ackermann] Start: (%.2f, %.2f, %.1f°, %.2f m/s)", 
             ackermann_start(0), ackermann_start(1), 
             ackermann_start(2) * 180.0 / M_PI, ackermann_start(3));
    ROS_ERROR("[Ackermann] End: (%.2f, %.2f, %.1f°, %.2f m/s)", 
             ackermann_end(0), ackermann_end(1), 
             ackermann_end(2) * 180.0 / M_PI, ackermann_end(3));
    
    ackermann_path_finder_->reset();
    status = ackermann_path_finder_->search(ackermann_start, init_ctrl, ackermann_end, false);
    
    ROS_ERROR("[Ackermann Planner] Search status: %d (REACH_END=2, REACH_HORIZON=1, NO_PATH=3)", status);
    
    if (status == AckermannKinoAstar::NO_PATH) {
      ROS_WARN("[Ackermann Kino replan]: search 1 fail, retrying...");
      ackermann_path_finder_->reset();
      status = ackermann_path_finder_->search(ackermann_start, init_ctrl, ackermann_end, false);
      if (status == AckermannKinoAstar::NO_PATH) {
        ROS_ERROR("[Ackermann Kino replan]: Can't find path.");
        return false;
      }
    }
    plan_data_.kino_path_ = ackermann_path_finder_->getKinoTraj(0.01);
    
  } else {
    // Use Standard Kinodynamic A* for drones/holonomic robots
    ROS_ERROR("========================================");
    ROS_ERROR("[⚠️  WARNING] USING STANDARD KINODYNAMIC A* (NOT ACKERMANN)");
    ROS_ERROR("Reason: frontend_planner_type_=%d (expected 1) OR ackermann_path_finder_ is NULL", 
             pp_.frontend_planner_type_);
    ROS_ERROR("This is the DRONE/HOLONOMIC robot planner!");
    ROS_ERROR("If you want Ackermann, check:");
    ROS_ERROR("  1. Launch file has: <param name=\"manager/frontend_planner_type\" value=\"1\"/>");
    ROS_ERROR("  2. Ackermann planner was initialized in initPlanModules");
    ROS_ERROR("========================================");
    
    kino_path_finder_->reset();
    status = kino_path_finder_->search(start_pt, start_vel, start_acc, end_pt, end_vel, true);
    if (status == KinodynamicAstar::NO_PATH) {
      ROS_ERROR("search 1 fail");
      // Retry
      kino_path_finder_->reset();
      status = kino_path_finder_->search(start_pt, start_vel, start_acc, end_pt, end_vel, false);
      if (status == KinodynamicAstar::NO_PATH) {
        cout << "[Kino replan]: Can't find path." << endl;
        return false;
      }
    }
    plan_data_.kino_path_ = kino_path_finder_->getKinoTraj(0.01);
  }

  double t_search = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();

  // Parameterize path to B-spline
  double ts = pp_.ctrl_pt_dist / pp_.max_vel_;
  vector<Eigen::Vector3d> point_set, start_end_derivatives;
  
  // Get fixed height for 2D planning (if using Ackermann planner)
  double fixed_z = 0.5;  // Default height
  if (pp_.frontend_planner_type_ == 1 && ackermann_path_finder_) {
    // Use Ackermann planner's getSamples
    ackermann_path_finder_->getSamples(ts, point_set, start_end_derivatives);
    // Get fixed height from planner
    if (point_set.size() > 0) {
      fixed_z = point_set[0](2);
    }
    ROS_INFO("[Ackermann] Using 2D planning at fixed height: %.2f m", fixed_z);
  } else {
    // Use standard planner's getSamples
    kino_path_finder_->getSamples(ts, point_set, start_end_derivatives);
  }

  // std::cout << "point set:" << std::endl;
  // for (auto pt : point_set) std::cout << pt.transpose() << std::endl;
  // std::cout << "derivative:" << std::endl;
  // for (auto dr : start_end_derivatives) std::cout << dr.transpose() << std::endl;

  Eigen::MatrixXd ctrl_pts;
  NonUniformBspline::parameterizeToBspline(
      ts, point_set, start_end_derivatives, pp_.bspline_degree_, ctrl_pts);
  NonUniformBspline init(ctrl_pts, pp_.bspline_degree_, ts);

  // B-spline-based optimization
  int cost_function = BsplineOptimizer::NORMAL_PHASE;
  if (pp_.min_time_) cost_function |= BsplineOptimizer::MINTIME;
  vector<Eigen::Vector3d> start, end;
  init.getBoundaryStates(2, 0, start, end);
  bspline_optimizers_[0]->setBoundaryStates(start, end);
  if (time_lb > 0) bspline_optimizers_[0]->setTimeLowerBound(time_lb);

  bspline_optimizers_[0]->optimize(ctrl_pts, ts, cost_function, 1, 1);
  
  // For Ackermann vehicles, force Z coordinate to fixed height after optimization
  if (pp_.frontend_planner_type_ == 1) {
    for (int i = 0; i < ctrl_pts.rows(); ++i) {
      ctrl_pts(i, 2) = fixed_z;
    }
    ROS_INFO("[Ackermann] Fixed all control points Z to %.2f m after optimization", fixed_z);
  }
  
  local_data_.position_traj_.setUniformBspline(ctrl_pts, pp_.bspline_degree_, ts);

  vector<Eigen::Vector3d> start2, end2;
  local_data_.position_traj_.getBoundaryStates(2, 0, start2, end2);
  std::cout << "State error: (" << (start2[0] - start[0]).norm() << ", "
            << (start2[1] - start[1]).norm() << ", " << (start2[2] - start[2]).norm() << ")"
            << std::endl;

  double t_opt = (ros::Time::now() - t1).toSec();
  ROS_WARN("Kino t: %lf, opt: %lf", t_search, t_opt);

  // t1 = ros::Time::now();

  // // Adjust time and refine

  // double dt;
  // for (int i = 0; i < 2; ++i)
  // {
  //   NonUniformBspline pos = NonUniformBspline(ctrl_pts, pp_.bspline_degree_, ts);
  //   pos.setPhysicalLimits(pp_.max_vel_, pp_.max_acc_);
  //   pos.lengthenTime(min(1.01, pos.checkRatio()));
  //   double duration = pos.getTimeSum();
  //   dt = duration / double(pos.getControlPoint().rows() - pp_.bspline_degree_);

  //   point_set.clear();
  //   for (double time = 0.0; time <= duration + 1e-4; time += dt)
  //     point_set.push_back(pos.evaluateDeBoorT(time));
  //   NonUniformBspline::parameterizeToBspline(dt, point_set, start_end_derivatives,
  //   pp_.bspline_degree_, ctrl_pts);
  //   bspline_optimizers_[0]->optimize(ctrl_pts, dt, cost_function, 1, 1);
  // }
  // local_data_.position_traj_.setUniformBspline(ctrl_pts, pp_.bspline_degree_, dt);

  // iterative time adjustment

  // double to = pos.getTimeSum();
  // pos.setPhysicalLimits(pp_.max_vel_, pp_.max_acc_);
  // bool feasible = pos.checkFeasibility(false);

  // int iter_num = 0;
  // while (!feasible && ros::ok()) {

  //   feasible = pos.reallocateTime();

  //   if (++iter_num >= 3) break;
  // }

  // // pos.checkFeasibility(true);
  // // cout << "[Main]: iter num: " << iter_num << endl;

  // double tn = pos.getTimeSum();

  // cout << "[kino replan]: Reallocate ratio: " << tn / to << endl;
  // if (tn / to > 3.0) ROS_ERROR("reallocate error.");

  // t_adjust = (ros::Time::now() - t1).toSec();

  // // save planned results

  // local_data_.position_traj_ = pos;

  // double t_total = t_search + t_opt + t_adjust;
  // cout << "[kino replan]: time: " << t_total << ", search: " << t_search << ",
  // optimize: " << t_opt
  //      << ", adjust time:" << t_adjust << endl;

  // pp_.time_search_   = t_search;
  // pp_.time_optimize_ = t_opt;
  // pp_.time_adjust_   = t_adjust;

  // int rd = rand() % 2;
  // if (rd == 0) {
  //   updateTrajInfo();
  //   return true;
  // } else
  //   return false;

  updateTrajInfo();
  return true;
}

void FastPlannerManager::planExploreTraj(const vector<Eigen::Vector3d>& tour,
    const Eigen::Vector3d& cur_vel, const Eigen::Vector3d& cur_acc, const double& time_lb) {
  if (tour.empty()) {
    ROS_ERROR("Empty path to traj planner");
    return;
  }

  // Publish A* path for comparison
  if (debug_compare_mode_) {
    ROS_WARN("[Debug] ========== DEBUG COMPARISON MODE ==========");
    ROS_INFO("[Debug] A* path has %lu waypoints", tour.size());
    publishAStarPath(tour);
  }

  // Debug mode: run BOTH backends and publish for comparison
  if (debug_compare_mode_) {
    ROS_INFO("[Debug] Running BOTH backends for comparison...");
    
    // Save current trajectory
    auto saved_traj = local_data_.position_traj_;
    
    // Run B-spline
    ROS_INFO("[Debug] Running B-spline...");
    planExploreTrajBspline(tour, cur_vel, cur_acc, time_lb);
    NonUniformBspline bspline_result = local_data_.position_traj_;
    publishBsplineTraj(bspline_result, "B-spline");
    
    // Run MINCO
    ROS_INFO("[Debug] Running MINCO...");
    planExploreTrajMINCO(tour, cur_vel, cur_acc, time_lb);
    // MINCO result is already in local_data_.position_traj_
    
    // Use MINCO as the actual trajectory (or you can choose B-spline)
    // local_data_.position_traj_ already has MINCO result
    
    ROS_INFO("[Debug] Both trajectories published to /debug/* topics for visualization");
    
  } else {
    // Normal mode: choose backend based on configuration
    if (use_minco_backend_) {
      planExploreTrajMINCO(tour, cur_vel, cur_acc, time_lb);
    } else {
      planExploreTrajBspline(tour, cur_vel, cur_acc, time_lb);
    }
  }
}

// Original B-spline implementation (renamed, kept as fallback)
void FastPlannerManager::planExploreTrajBspline(const vector<Eigen::Vector3d>& tour,
    const Eigen::Vector3d& cur_vel, const Eigen::Vector3d& cur_acc, const double& time_lb) {
  if (tour.empty()) {
    ROS_ERROR("Empty path to traj planner");
    return;
  }
  
  // Ensure at least 2 points for trajectory planning
  if (tour.size() < 2) {
    ROS_ERROR("Path has only %zu point(s), need at least 2 for trajectory planning", tour.size());
    return;
  }

  // Generate traj through waypoints-based method
  const int pt_num = tour.size();
  Eigen::MatrixXd pos(pt_num, 3);
  for (int i = 0; i < pt_num; ++i) pos.row(i) = tour[i];

  Eigen::Vector3d zero(0, 0, 0);
  Eigen::VectorXd times(pt_num - 1);
  for (int i = 0; i < pt_num - 1; ++i)
    times(i) = (pos.row(i + 1) - pos.row(i)).norm() / (pp_.max_vel_ * 0.5);

  PolynomialTraj init_traj;
  PolynomialTraj::waypointsTraj(pos, cur_vel, zero, cur_acc, zero, times, init_traj);

  // B-spline-based optimization
  vector<Vector3d> points, boundary_deri;
  double duration = init_traj.getTotalTime();
  int seg_num = init_traj.getLength() / pp_.ctrl_pt_dist;
  seg_num = max(8, seg_num);
  double dt = duration / double(seg_num);

  std::cout << "duration: " << duration << ", seg_num: " << seg_num << ", dt: " << dt << std::endl;

  for (double ts = 0.0; ts <= duration + 1e-4; ts += dt)
    points.push_back(init_traj.evaluate(ts, 0));
  boundary_deri.push_back(init_traj.evaluate(0.0, 1));
  boundary_deri.push_back(init_traj.evaluate(duration, 1));
  boundary_deri.push_back(init_traj.evaluate(0.0, 2));
  boundary_deri.push_back(init_traj.evaluate(duration, 2));

  Eigen::MatrixXd ctrl_pts;
  NonUniformBspline::parameterizeToBspline(
      dt, points, boundary_deri, pp_.bspline_degree_, ctrl_pts);
  NonUniformBspline tmp_traj(ctrl_pts, pp_.bspline_degree_, dt);

  int cost_func = BsplineOptimizer::NORMAL_PHASE;
  if (pp_.min_time_) cost_func |= BsplineOptimizer::MINTIME;

  vector<Vector3d> start, end;
  tmp_traj.getBoundaryStates(2, 0, start, end);
  bspline_optimizers_[0]->setBoundaryStates(start, end);
  if (time_lb > 0) bspline_optimizers_[0]->setTimeLowerBound(time_lb);

  bspline_optimizers_[0]->optimize(ctrl_pts, dt, cost_func, 1, 1);
  local_data_.position_traj_.setUniformBspline(ctrl_pts, pp_.bspline_degree_, dt);

  updateTrajInfo();
}

// !SECTION

// SECTION topological replanning

bool FastPlannerManager::planGlobalTraj(const Eigen::Vector3d& start_pos) {
  plan_data_.clearTopoPaths();

  // Generate global reference trajectory
  vector<Eigen::Vector3d> points = plan_data_.global_waypoints_;
  if (points.size() == 0) std::cout << "no global waypoints!" << std::endl;

  points.insert(points.begin(), start_pos);

  // Insert intermediate points if two waypoints are too far
  vector<Eigen::Vector3d> inter_points;
  const double dist_thresh = 4.0;

  for (int i = 0; i < points.size() - 1; ++i) {
    inter_points.push_back(points.at(i));
    double dist = (points.at(i + 1) - points.at(i)).norm();
    if (dist > dist_thresh) {
      int id_num = floor(dist / dist_thresh) + 1;
      for (int j = 1; j < id_num; ++j) {
        Eigen::Vector3d inter_pt =
            points.at(i) * (1.0 - double(j) / id_num) + points.at(i + 1) * double(j) / id_num;
        inter_points.push_back(inter_pt);
      }
    }
  }
  inter_points.push_back(points.back());

  // At least 3 waypoints are required to solve the problem
  if (inter_points.size() == 2) {
    Eigen::Vector3d mid = (inter_points[0] + inter_points[1]) * 0.5;
    inter_points.insert(inter_points.begin() + 1, mid);
  }

  int pt_num = inter_points.size();
  Eigen::MatrixXd pos(pt_num, 3);
  for (int i = 0; i < pt_num; ++i) pos.row(i) = inter_points[i];

  Eigen::Vector3d zero(0, 0, 0);
  Eigen::VectorXd time(pt_num - 1);
  for (int i = 0; i < pt_num - 1; ++i)
    time(i) = (pos.row(i + 1) - pos.row(i)).norm() / (pp_.max_vel_ * 0.5);

  time(0) += pp_.max_vel_ / (2 * pp_.max_acc_);
  time(time.rows() - 1) += pp_.max_vel_ / (2 * pp_.max_acc_);

  PolynomialTraj gl_traj;
  PolynomialTraj::waypointsTraj(pos, zero, zero, zero, zero, time, gl_traj);

  auto time_now = ros::Time::now();
  global_data_.setGlobalTraj(gl_traj, time_now);

  // truncate a local trajectory

  double dt, duration;
  Eigen::MatrixXd ctrl_pts = paramLocalTraj(0.0, dt, duration);
  NonUniformBspline bspline(ctrl_pts, pp_.bspline_degree_, dt);

  std::cout << "ctrl pt: " << ctrl_pts.rows() << std::endl;

  global_data_.setLocalTraj(bspline, 0.0, duration, 0.0);
  local_data_.position_traj_ = bspline;
  local_data_.start_time_ = time_now;
  ROS_INFO("global trajectory generated.");

  updateTrajInfo();

  return true;
}

bool FastPlannerManager::topoReplan(bool collide) {
  ros::Time t1, t2;

  /* truncate a new local segment for replanning */
  ros::Time time_now = ros::Time::now();
  double t_now = (time_now - global_data_.global_start_time_).toSec();
  double local_traj_dt, local_traj_duration;

  Eigen::MatrixXd ctrl_pts = paramLocalTraj(t_now, local_traj_dt, local_traj_duration);
  NonUniformBspline init_traj(ctrl_pts, pp_.bspline_degree_, local_traj_dt);
  local_data_.start_time_ = time_now;

  std::cout << "dt: " << local_traj_dt << ", dur: " << local_traj_duration << std::endl;

  if (!collide) {
    // No collision detected, but we can further refine the trajectory
    refineTraj(init_traj);
    double time_change = init_traj.getTimeSum() - local_traj_duration;
    local_data_.position_traj_ = init_traj;
    global_data_.setLocalTraj(
        local_data_.position_traj_, t_now, local_traj_duration + time_change + t_now, time_change);
    // local_data_.position_traj_ = init_traj;
    // global_data_.setLocalTraj(init_traj, t_now, local_traj_duration + t_now, 0.0);
  } else {
    // Find topologically distinctive path and guide optimization in parallel
    plan_data_.initial_local_segment_ = init_traj;
    vector<Eigen::Vector3d> colli_start, colli_end, start_pts, end_pts;
    findCollisionRange(colli_start, colli_end, start_pts, end_pts);

    if (colli_start.size() == 1 && colli_end.size() == 0) {
      ROS_WARN("Init traj ends in obstacle, no replanning.");
      local_data_.position_traj_ = init_traj;
      global_data_.setLocalTraj(init_traj, t_now, local_traj_duration + t_now, 0.0);
    } else {
      // Call topological replanning when local segment is in collision
      /* Search topological distinctive paths */
      ROS_INFO("[Topo]: ---------");
      plan_data_.clearTopoPaths();
      list<GraphNode::Ptr> graph;
      vector<vector<Eigen::Vector3d>> raw_paths, filtered_paths, select_paths;
      topo_prm_->findTopoPaths(colli_start.front(), colli_end.back(), start_pts, end_pts, graph,
          raw_paths, filtered_paths, select_paths);

      if (select_paths.size() == 0) {
        ROS_WARN("No path.");
        return false;
      }
      plan_data_.addTopoPaths(graph, raw_paths, filtered_paths, select_paths);

      /* Optimize trajectory using different topo guiding paths */
      ROS_INFO("[Optimize]: ---------");
      t1 = ros::Time::now();

      plan_data_.topo_traj_pos1_.resize(select_paths.size());
      plan_data_.topo_traj_pos2_.resize(select_paths.size());
      vector<thread> optimize_threads;
      for (int i = 0; i < select_paths.size(); ++i) {
        optimize_threads.emplace_back(&FastPlannerManager::optimizeTopoBspline, this, t_now,
            local_traj_duration, select_paths[i], i);
        // optimizeTopoBspline(t_now, local_traj_duration,
        // select_paths[i], origin_len, i);
      }
      for (int i = 0; i < select_paths.size(); ++i) optimize_threads[i].join();

      double t_opt = (ros::Time::now() - t1).toSec();
      cout << "[planner]: optimization time: " << t_opt << endl;

      NonUniformBspline best_traj;
      selectBestTraj(best_traj);
      refineTraj(best_traj);
      double time_change = best_traj.getTimeSum() - local_traj_duration;

      local_data_.position_traj_ = best_traj;
      global_data_.setLocalTraj(local_data_.position_traj_, t_now,
          local_traj_duration + time_change + t_now, time_change);
    }
  }
  updateTrajInfo();

  double tr = (ros::Time::now() - time_now).toSec();
  ROS_WARN("Replan time: %lf", tr);

  return true;
}

void FastPlannerManager::selectBestTraj(NonUniformBspline& traj) {
  // sort by jerk
  vector<NonUniformBspline>& trajs = plan_data_.topo_traj_pos2_;
  sort(trajs.begin(), trajs.end(),
      [](NonUniformBspline& tj1, NonUniformBspline& tj2) { return tj1.getJerk() < tj2.getJerk(); });
  traj = trajs[0];
}

void FastPlannerManager::refineTraj(NonUniformBspline& best_traj) {
  ros::Time t1 = ros::Time::now();
  plan_data_.no_visib_traj_ = best_traj;

  int cost_function = BsplineOptimizer::NORMAL_PHASE;
  if (pp_.min_time_) cost_function |= BsplineOptimizer::MINTIME;

  // ViewConstraint view_cons;
  // visib_util_->calcViewConstraint(best_traj, view_cons);
  // plan_data_.view_cons_ = view_cons;
  // if (view_cons.idx_ >= 0)
  // {
  //   cost_function |= BsplineOptimizer::VIEWCONS;
  //   bspline_optimizers_[0]->setViewConstraint(view_cons);
  // }

  // Refine selected best traj
  Eigen::MatrixXd ctrl_pts = best_traj.getControlPoint();
  double dt = best_traj.getKnotSpan();
  vector<Eigen::Vector3d> start1, end1;
  best_traj.getBoundaryStates(2, 0, start1, end1);

  bspline_optimizers_[0]->setBoundaryStates(start1, end1);
  bspline_optimizers_[0]->optimize(ctrl_pts, dt, cost_function, 2, 2);
  best_traj.setUniformBspline(ctrl_pts, pp_.bspline_degree_, dt);

  vector<Eigen::Vector3d> start2, end2;
  best_traj.getBoundaryStates(2, 2, start2, end2);
  for (int i = 0; i < 3; ++i)
    std::cout << "error start: " << (start1[i] - start2[i]).norm() << std::endl;
  for (int i = 0; i < 1; ++i)
    std::cout << "error end  : " << (end1[i] - end2[i]).norm() << std::endl;
}

void FastPlannerManager::updateTrajInfo() {
  local_data_.velocity_traj_ = local_data_.position_traj_.getDerivative();
  local_data_.acceleration_traj_ = local_data_.velocity_traj_.getDerivative();

  local_data_.start_pos_ = local_data_.getPosition(0.0);
  local_data_.duration_ = local_data_.position_traj_.getTimeSum();

  local_data_.traj_id_ += 1;
  
  // IMPORTANT: Reset MINCO flag when using B-spline
  // This ensures getPosition() uses the new B-spline trajectory, not old MINCO
  local_data_.use_minco_traj_ = false;
}

void FastPlannerManager::reparamBspline(NonUniformBspline& bspline, double ratio,
    Eigen::MatrixXd& ctrl_pts, double& dt, double& time_inc) {
  int prev_num = bspline.getControlPoint().rows();
  double time_origin = bspline.getTimeSum();

  int seg_num = bspline.getControlPoint().rows() - pp_.bspline_degree_;
  ratio = min(1.01, ratio);

  bspline.lengthenTime(ratio);
  double duration = bspline.getTimeSum();
  dt = duration / double(seg_num);
  time_inc = duration - time_origin;

  vector<Eigen::Vector3d> point_set;
  for (double time = 0.0; time <= duration + 1e-4; time += dt)
    point_set.push_back(bspline.evaluateDeBoorT(time));
  NonUniformBspline::parameterizeToBspline(
      dt, point_set, plan_data_.local_start_end_derivative_, pp_.bspline_degree_, ctrl_pts);
  // ROS_WARN("prev: %d, new: %d", prev_num, ctrl_pts.rows());
}

void FastPlannerManager::optimizeTopoBspline(
    double start_t, double duration, vector<Eigen::Vector3d> guide_path, int traj_id) {
  auto t1 = ros::Time::now();

  // Re-parameterize B-spline according to the length of guide path
  int seg_num = topo_prm_->pathLength(guide_path) / pp_.ctrl_pt_dist;
  seg_num = max(6, seg_num);  // Min number required for optimizing
  double dt = duration / double(seg_num);
  Eigen::MatrixXd ctrl_pts = reparamLocalTraj(start_t, duration, dt);

  NonUniformBspline tmp_traj(ctrl_pts, pp_.bspline_degree_, dt);
  vector<Eigen::Vector3d> start, end;
  tmp_traj.getBoundaryStates(2, 0, start, end);

  // std::cout << "ctrl pt num: " << ctrl_pts.rows() << std::endl;

  // Discretize the guide path and align it with B-spline control points
  vector<Eigen::Vector3d> tmp_pts, guide_pts;
  if (pp_.bspline_degree_ == 3 || pp_.bspline_degree_ == 5) {
    topo_prm_->pathToGuidePts(guide_path, int(ctrl_pts.rows()) - 2, tmp_pts);
    guide_pts.insert(guide_pts.end(), tmp_pts.begin() + 2, tmp_pts.end() - 2);
    if (guide_pts.size() != int(ctrl_pts.rows()) - 6) ROS_WARN("Incorrect guide for 3 degree");
  } else if (pp_.bspline_degree_ == 4) {
    topo_prm_->pathToGuidePts(guide_path, int(2 * ctrl_pts.rows()) - 7, tmp_pts);
    for (int i = 0; i < tmp_pts.size(); ++i) {
      if (i % 2 == 1 && i >= 5 && i <= tmp_pts.size() - 6) guide_pts.push_back(tmp_pts[i]);
    }
    if (guide_pts.size() != int(ctrl_pts.rows()) - 8) ROS_WARN("Incorrect guide for 4 degree");
  }

  // std::cout << "guide pt num: " << guide_pt.size() << std::endl;

  double tm1 = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();

  // First phase, path-guided optimization
  bspline_optimizers_[traj_id]->setBoundaryStates(start, end);
  bspline_optimizers_[traj_id]->setGuidePath(guide_pts);
  bspline_optimizers_[traj_id]->optimize(ctrl_pts, dt, BsplineOptimizer::GUIDE_PHASE, 0, 1);
  plan_data_.topo_traj_pos1_[traj_id] = NonUniformBspline(ctrl_pts, pp_.bspline_degree_, dt);

  double tm2 = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();

  // Second phase, smooth+safety+feasibility
  int cost_func = BsplineOptimizer::NORMAL_PHASE;
  // if (pp_.min_time_)
  //   cost_func |= BsplineOptimizer::MINTIME;
  bspline_optimizers_[traj_id]->setBoundaryStates(start, end);
  bspline_optimizers_[traj_id]->optimize(ctrl_pts, dt, cost_func, 1, 1);
  plan_data_.topo_traj_pos2_[traj_id] = NonUniformBspline(ctrl_pts, pp_.bspline_degree_, dt);

  double tm3 = (ros::Time::now() - t1).toSec();
  // ROS_INFO("optimization %d cost %lf, %lf, %lf seconds.", traj_id, tm1, tm2, tm3);
}

Eigen::MatrixXd FastPlannerManager::paramLocalTraj(double start_t, double& dt, double& duration) {
  vector<Eigen::Vector3d> point_set;
  vector<Eigen::Vector3d> start_end_derivative;
  global_data_.getTrajInfoInSphere(start_t, pp_.local_traj_len_, pp_.ctrl_pt_dist, point_set,
      start_end_derivative, dt, duration);

  Eigen::MatrixXd ctrl_pts;
  NonUniformBspline::parameterizeToBspline(
      dt, point_set, start_end_derivative, pp_.bspline_degree_, ctrl_pts);
  plan_data_.local_start_end_derivative_ = start_end_derivative;

  return ctrl_pts;
}

Eigen::MatrixXd FastPlannerManager::reparamLocalTraj(
    const double& start_t, const double& duration, const double& dt) {
  vector<Eigen::Vector3d> point_set;
  vector<Eigen::Vector3d> start_end_derivative;

  global_data_.getTrajInfoInDuration(start_t, duration, dt, point_set, start_end_derivative);
  plan_data_.local_start_end_derivative_ = start_end_derivative;

  /* parameterization of B-spline */
  Eigen::MatrixXd ctrl_pts;
  NonUniformBspline::parameterizeToBspline(
      dt, point_set, start_end_derivative, pp_.bspline_degree_, ctrl_pts);
  // cout << "ctrl pts:" << ctrl_pts.rows() << endl;

  return ctrl_pts;
}

void FastPlannerManager::findCollisionRange(vector<Eigen::Vector3d>& colli_start,
    vector<Eigen::Vector3d>& colli_end, vector<Eigen::Vector3d>& start_pts,
    vector<Eigen::Vector3d>& end_pts) {
  bool last_safe = true, safe;
  double t_m, t_mp;
  NonUniformBspline* initial_traj = &plan_data_.initial_local_segment_;
  initial_traj->getTimeSpan(t_m, t_mp);

  /* find range of collision */
  double t_s = -1.0, t_e;
  for (double tc = t_m; tc <= t_mp + 1e-4; tc += 0.05) {
    Eigen::Vector3d ptc = initial_traj->evaluateDeBoor(tc);
    safe = edt_environment_->evaluateCoarseEDT(ptc, -1.0) < topo_prm_->clearance_ ? false : true;

    if (last_safe && !safe) {
      colli_start.push_back(initial_traj->evaluateDeBoor(tc - 0.05));
      if (t_s < 0.0) t_s = tc - 0.05;
    } else if (!last_safe && safe) {
      colli_end.push_back(ptc);
      t_e = tc;
    }

    last_safe = safe;
  }

  if (colli_start.size() == 0) return;

  if (colli_start.size() == 1 && colli_end.size() == 0) return;

  /* find start and end safe segment */
  double dt = initial_traj->getKnotSpan();
  int sn = ceil((t_s - t_m) / dt);
  dt = (t_s - t_m) / sn;

  for (double tc = t_m; tc <= t_s + 1e-4; tc += dt) {
    start_pts.push_back(initial_traj->evaluateDeBoor(tc));
  }

  dt = initial_traj->getKnotSpan();
  sn = ceil((t_mp - t_e) / dt);
  dt = (t_mp - t_e) / sn;
  // std::cout << "dt: " << dt << std::endl;
  // std::cout << "sn: " << sn << std::endl;
  // std::cout << "t_m: " << t_m << std::endl;
  // std::cout << "t_mp: " << t_mp << std::endl;
  // std::cout << "t_s: " << t_s << std::endl;
  // std::cout << "t_e: " << t_e << std::endl;

  if (dt > 1e-4) {
    for (double tc = t_e; tc <= t_mp + 1e-4; tc += dt) {
      end_pts.push_back(initial_traj->evaluateDeBoor(tc));
    }
  } else {
    end_pts.push_back(initial_traj->evaluateDeBoor(t_mp));
  }
}

// !SECTION

void FastPlannerManager::planYaw(const Eigen::Vector3d& start_yaw) {
  auto t1 = ros::Time::now();
  // calculate waypoints of heading

  auto& pos = local_data_.position_traj_;
  double duration = pos.getTimeSum();

  double dt_yaw = 0.3;
  int seg_num = ceil(duration / dt_yaw);
  dt_yaw = duration / seg_num;

  const double forward_t = 2.0;
  double last_yaw = start_yaw(0);
  vector<Eigen::Vector3d> waypts;
  vector<int> waypt_idx;

  // seg_num -> seg_num - 1 points for constraint excluding the boundary states

  for (int i = 0; i < seg_num; ++i) {
    double tc = i * dt_yaw;
    Eigen::Vector3d pc = pos.evaluateDeBoorT(tc);
    double tf = min(duration, tc + forward_t);
    Eigen::Vector3d pf = pos.evaluateDeBoorT(tf);
    Eigen::Vector3d pd = pf - pc;

    Eigen::Vector3d waypt;
    if (pd.norm() > 1e-6) {
      waypt(0) = atan2(pd(1), pd(0));
      waypt(1) = waypt(2) = 0.0;
      calcNextYaw(last_yaw, waypt(0));
    } else {
      waypt = waypts.back();
    }
    last_yaw = waypt(0);
    waypts.push_back(waypt);
    waypt_idx.push_back(i);
  }

  // calculate initial control points with boundary state constraints

  Eigen::MatrixXd yaw(seg_num + 3, 1);
  yaw.setZero();

  Eigen::Matrix3d states2pts;
  states2pts << 1.0, -dt_yaw, (1 / 3.0) * dt_yaw * dt_yaw, 1.0, 0.0, -(1 / 6.0) * dt_yaw * dt_yaw,
      1.0, dt_yaw, (1 / 3.0) * dt_yaw * dt_yaw;
  yaw.block(0, 0, 3, 1) = states2pts * start_yaw;

  Eigen::Vector3d end_v = local_data_.velocity_traj_.evaluateDeBoorT(duration - 0.1);
  Eigen::Vector3d end_yaw(atan2(end_v(1), end_v(0)), 0, 0);
  calcNextYaw(last_yaw, end_yaw(0));
  yaw.block(seg_num, 0, 3, 1) = states2pts * end_yaw;

  // solve
  bspline_optimizers_[1]->setWaypoints(waypts, waypt_idx);
  int cost_func = BsplineOptimizer::SMOOTHNESS | BsplineOptimizer::WAYPOINTS |
                  BsplineOptimizer::START | BsplineOptimizer::END;

  vector<Eigen::Vector3d> start = { Eigen::Vector3d(start_yaw[0], 0, 0),
    Eigen::Vector3d(start_yaw[1], 0, 0), Eigen::Vector3d(start_yaw[2], 0, 0) };
  vector<Eigen::Vector3d> end = { Eigen::Vector3d(end_yaw[0], 0, 0),
    Eigen::Vector3d(end_yaw[1], 0, 0), Eigen::Vector3d(end_yaw[2], 0, 0) };
  bspline_optimizers_[1]->setBoundaryStates(start, end);
  bspline_optimizers_[1]->optimize(yaw, dt_yaw, cost_func, 1, 1);

  // update traj info
  local_data_.yaw_traj_.setUniformBspline(yaw, pp_.bspline_degree_, dt_yaw);
  local_data_.yawdot_traj_ = local_data_.yaw_traj_.getDerivative();
  local_data_.yawdotdot_traj_ = local_data_.yawdot_traj_.getDerivative();

  vector<double> path_yaw;
  for (int i = 0; i < waypts.size(); ++i) path_yaw.push_back(waypts[i][0]);
  plan_data_.path_yaw_ = path_yaw;
  plan_data_.dt_yaw_ = dt_yaw;
  plan_data_.dt_yaw_path_ = dt_yaw;

  std::cout << "yaw time: " << (ros::Time::now() - t1).toSec() << std::endl;
}

void FastPlannerManager::planYawExplore(const Eigen::Vector3d& start_yaw, const double& end_yaw,
    bool lookfwd, const double& relax_time) {
  const int seg_num = 12;
  double dt_yaw = local_data_.duration_ / seg_num;  // time of B-spline segment
  Eigen::Vector3d start_yaw3d = start_yaw;
  std::cout << "dt_yaw: " << dt_yaw << ", start yaw: " << start_yaw3d.transpose()
            << ", end: " << end_yaw << std::endl;

  while (start_yaw3d[0] < -M_PI) start_yaw3d[0] += 2 * M_PI;
  while (start_yaw3d[0] > M_PI) start_yaw3d[0] -= 2 * M_PI;
  double last_yaw = start_yaw3d[0];

  // Yaw traj control points
  Eigen::MatrixXd yaw(seg_num + 3, 1);
  yaw.setZero();

  // Initial state
  Eigen::Matrix3d states2pts;
  states2pts << 1.0, -dt_yaw, (1 / 3.0) * dt_yaw * dt_yaw, 1.0, 0.0, -(1 / 6.0) * dt_yaw * dt_yaw,
      1.0, dt_yaw, (1 / 3.0) * dt_yaw * dt_yaw;
  yaw.block<3, 1>(0, 0) = states2pts * start_yaw3d;

  // Add waypoint constraints if look forward is enabled
  vector<Eigen::Vector3d> waypts;
  vector<int> waypt_idx;
  if (lookfwd) {
    const double forward_t = 2.0;
    const int relax_num = relax_time / dt_yaw;
    for (int i = 1; i < seg_num - relax_num; ++i) {
      double tc = i * dt_yaw;
      Eigen::Vector3d pc = local_data_.getPosition(tc);
      double tf = min(local_data_.duration_, tc + forward_t);
      Eigen::Vector3d pf = local_data_.getPosition(tf);
      Eigen::Vector3d pd = pf - pc;
      Eigen::Vector3d waypt;
      if (pd.norm() > 1e-6) {
        waypt(0) = atan2(pd(1), pd(0));
        waypt(1) = waypt(2) = 0.0;
        calcNextYaw(last_yaw, waypt(0));
      } else
        waypt = waypts.back();

      last_yaw = waypt(0);
      waypts.push_back(waypt);
      waypt_idx.push_back(i);
    }
  }
  // Final state
  Eigen::Vector3d end_yaw3d(end_yaw, 0, 0);
  calcNextYaw(last_yaw, end_yaw3d(0));
  yaw.block<3, 1>(seg_num, 0) = states2pts * end_yaw3d;

  // Check for rapid yaw change (> 180 degrees)
  double yaw_change = fabs(start_yaw3d[0] - end_yaw3d[0]);
  if (yaw_change >= M_PI) {
    ROS_WARN("[Yaw] Large yaw change detected: %.1f deg (%.1f → %.1f deg)", 
             yaw_change * 180.0 / M_PI,
             start_yaw3d[0] * 180.0 / M_PI,
             end_yaw3d[0] * 180.0 / M_PI);
    ROS_INFO("[Yaw] This is normal for exploration, optimizer will smooth it");
  }

  // // Interpolate start and end value for smoothness
  // for (int i = 1; i < seg_num; ++i)
  // {
  //   double tc = i * dt_yaw;
  //   Eigen::Vector3d waypt = (1 - double(i) / seg_num) * start_yaw3d + double(i) / seg_num *
  //   end_yaw3d;
  //   std::cout << "i: " << i << ", wp: " << waypt[0] << ", ";
  //   calcNextYaw(last_yaw, waypt(0));
  // }
  // std::cout << "" << std::endl;

  auto t1 = ros::Time::now();

  // Call B-spline optimization solver
  int cost_func = BsplineOptimizer::SMOOTHNESS | BsplineOptimizer::START | BsplineOptimizer::END |
                  BsplineOptimizer::WAYPOINTS;
  vector<Eigen::Vector3d> start = { Eigen::Vector3d(start_yaw3d[0], 0, 0),
    Eigen::Vector3d(start_yaw3d[1], 0, 0), Eigen::Vector3d(start_yaw3d[2], 0, 0) };
  vector<Eigen::Vector3d> end = { Eigen::Vector3d(end_yaw3d[0], 0, 0), Eigen::Vector3d(0, 0, 0) };
  bspline_optimizers_[1]->setBoundaryStates(start, end);
  bspline_optimizers_[1]->setWaypoints(waypts, waypt_idx);
  bspline_optimizers_[1]->optimize(yaw, dt_yaw, cost_func, 1, 1);

  // std::cout << "2: " << (ros::Time::now() - t1).toSec() << std::endl;

  // Update traj info
  local_data_.yaw_traj_.setUniformBspline(yaw, 3, dt_yaw);
  local_data_.yawdot_traj_ = local_data_.yaw_traj_.getDerivative();
  local_data_.yawdotdot_traj_ = local_data_.yawdot_traj_.getDerivative();
  plan_data_.dt_yaw_ = dt_yaw;
  
  // Check yaw velocity (optional warning, not critical)
  double max_yaw_rate = 0.0;
  for (double t = 0; t <= local_data_.duration_; t += dt_yaw) {
    double yaw_rate = fabs(local_data_.yawdot_traj_.evaluateDeBoorT(t)[0]);
    max_yaw_rate = std::max(max_yaw_rate, yaw_rate);
  }
  double max_yaw_rate_deg = max_yaw_rate * 180.0 / M_PI;
  if (max_yaw_rate_deg > 90.0) {  // 90 deg/s threshold
    ROS_WARN("[Yaw] High yaw rate: %.1f deg/s (normal during exploration turns)", max_yaw_rate_deg);
  }

  // plan_data_.path_yaw_ = path;
  // plan_data_.dt_yaw_path_ = dt_yaw * subsp;
}

void FastPlannerManager::calcNextYaw(const double& last_yaw, double& yaw) {
  // round yaw to [-PI, PI]
  double round_last = last_yaw;
  while (round_last < -M_PI) {
    round_last += 2 * M_PI;
  }
  while (round_last > M_PI) {
    round_last -= 2 * M_PI;
  }

  double diff = yaw - round_last;
  if (fabs(diff) <= M_PI) {
    yaw = last_yaw + diff;
  } else if (diff > M_PI) {
    yaw = last_yaw + diff - 2 * M_PI;
  } else if (diff < -M_PI) {
    yaw = last_yaw + diff + 2 * M_PI;
  }
}

// MINCO backend implementation
void FastPlannerManager::planExploreTrajMINCO(
    const vector<Eigen::Vector3d>& tour,
    const Eigen::Vector3d& cur_vel, 
    const Eigen::Vector3d& cur_acc, 
    const double& time_lb) {
  
  ros::Time start_time = ros::Time::now();
  
  // Path truncation (like EPIC) to avoid planning too long trajectories
  vector<Eigen::Vector3d> tour_truncated;
  double max_traj_len = pp_.local_traj_len_;  // From FUEL's local_segment_length parameter
  bool was_truncated = false;
  
  if (max_traj_len > 0 && tour.size() > 1) {
    double len = 0.0;
    tour_truncated.push_back(tour[0]);
    
    for (size_t i = 1; i < tour.size(); i++) {
      double seg_len = (tour[i] - tour[i-1]).norm();
      if (len + seg_len > max_traj_len) {
        // Interpolate to exactly max_traj_len
        double ratio = (max_traj_len - len) / seg_len;
        if (ratio > 0.1) {  // At least 10% of the segment
          Eigen::Vector3d truncated_pt = tour[i-1] + ratio * (tour[i] - tour[i-1]);
          tour_truncated.push_back(truncated_pt);
        }
        was_truncated = true;
        ROS_WARN("[MINCO] Path truncated: %lu -> %lu waypoints (%.2fm > %.2fm limit)",
                 tour.size(), tour_truncated.size(), len + seg_len, max_traj_len);
        break;
      }
      len += seg_len;
      tour_truncated.push_back(tour[i]);
    }
    
    if (tour_truncated.size() < 2 && tour.size() >= 2) {
      tour_truncated.push_back(tour[1]);  // Ensure at least 2 points
    }
    
    // If path was truncated, check if we should preserve the final goal
    // This is important for point-to-point navigation to ensure reaching the target
    if (was_truncated && tour.size() > 1) {
      const Eigen::Vector3d& original_final = tour.back();
      const Eigen::Vector3d& truncated_final = tour_truncated.back();
      double dist_to_original_final = (truncated_final - original_final).norm();
      
      // If the original final point is close enough (within 2m), preserve it
      // This ensures we can reach the goal even if path was truncated
      if (dist_to_original_final > 0.5 && dist_to_original_final < 2.0) {
        tour_truncated.push_back(original_final);
        ROS_INFO("[MINCO] Preserved original final goal (%.2f m away) despite truncation", dist_to_original_final);
      } else if (dist_to_original_final >= 2.0) {
        ROS_WARN("[MINCO] Original final goal is %.2f m away from truncated end - may need replanning", dist_to_original_final);
      }
    }
  } else {
    tour_truncated = tour;
  }
  
  // Use truncated path
  const vector<Eigen::Vector3d>& path = tour_truncated;
  ROS_INFO("[MINCO] Planning with %lu waypoints (original: %lu)", path.size(), tour.size());
  
  // 1. Calculate bounding box
  Eigen::Vector3d min_bd = path[0];
  Eigen::Vector3d max_bd = path[0];
  for (const auto& waypoint : path) {
    for (int i = 0; i < 3; i++) {
      min_bd[i] = std::min(min_bd[i], waypoint[i]);
      max_bd[i] = std::max(max_bd[i], waypoint[i]);
    }
  }
  // Expand boundaries with Z-axis limits
  for (int i = 0; i < 2; i++) {
    min_bd[i] -= 3.0;
    max_bd[i] += 3.0;
  }
  // Limit Z-axis expansion to stay within sensor range
  min_bd[2] = std::max(min_bd[2] - 1.0, 0.3);   // Don't go below 0.3m
  max_bd[2] = std::min(max_bd[2] + 1.0, 3.5);   // Don't go above 3.5m (exploration height)
  
  // 2. Extract obstacle point cloud from ESDF map
  std::vector<Eigen::Vector3d> surf_points;
  getPointCloudFromESDF(min_bd, max_bd, surf_points);
  
  if (surf_points.empty()) {
    ROS_ERROR("[MINCO] No obstacle points found, using B-spline fallback");
    ROS_ERROR("[Debug] MINCO FAILED - No obstacle points! Check ESDF map initialization.");
    planExploreTrajBspline(tour, cur_vel, cur_acc, time_lb);
    return;
  } else {
    ROS_INFO("[MINCO] Successfully extracted %lu obstacle points", surf_points.size());
  }
  
  // 3. Generate Safe Flight Corridor (SFC)
  std::vector<Eigen::MatrixX4d> hPolys;
  try {
    sfc_gen::convexCover(
        gcopter_viz_,  // pass unique_ptr directly
        path,  // Use truncated path
        surf_points,
        min_bd, max_bd, 
        7.0,  // initialRadius
        gcopter_config_->corridor_size,
        hPolys, 
        1e-6,  // epsilon
        gcopter_config_->dilateRadiusSoft
    );
  } catch (const std::exception& e) {
    ROS_ERROR("[MINCO] SFC generation failed: %s, using B-spline fallback", e.what());
    planExploreTrajBspline(tour, cur_vel, cur_acc, time_lb);
    return;
  }
  
  if (hPolys.size() < 2) {
    ROS_WARN("[MINCO] Corridor too short (%lu segments, need >= 2), using B-spline fallback", hPolys.size());
    ROS_INFO("[MINCO→B-spline] Switching to B-spline for short path (normal behavior)");
    planExploreTrajBspline(tour, cur_vel, cur_acc, time_lb);
    return;
  } else {
    ROS_INFO("[MINCO] Successfully generated corridor with %lu segments", hPolys.size());
  }
  
  // 4. Setup initial and final states
  Eigen::Matrix<double, 3, 4> iniState, finState;
  
  // Limit initial velocity and acceleration to avoid aggressive behavior
  Eigen::Vector3d limited_vel = cur_vel;
  double vel_norm = limited_vel.norm();
  if (vel_norm > gcopter_config_->maxVelMag * 0.8) {  // More conservative: 80% of max
    limited_vel = limited_vel / vel_norm * (gcopter_config_->maxVelMag * 0.8);
    ROS_WARN("[MINCO] Initial velocity limited: %.2f -> %.2f m/s", vel_norm, gcopter_config_->maxVelMag * 0.8);
  }
  
  // Limit initial acceleration (crucial for smooth start!)
  Eigen::Vector3d limited_acc = cur_acc;
  double acc_norm = limited_acc.norm();
  double max_acc = 2.0;  // Conservative acceleration limit
  if (acc_norm > max_acc) {
    limited_acc = limited_acc / acc_norm * max_acc;
    ROS_WARN("[MINCO] Initial acceleration limited: %.2f -> %.2f m/s^2", acc_norm, max_acc);
  }
  
  iniState << path.front(), limited_vel, limited_acc, Eigen::Vector3d::Zero();
  finState << path.back(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
  
  // 5. Setup GCOPTER parameters
  gcopter::GCOPTER_PolytopeSFC gcopter;
  Eigen::VectorXd magnitudeBounds(5);
  Eigen::VectorXd penaltyWeights(5);
  Eigen::VectorXd physicalParams(6);
  
  magnitudeBounds(0) = gcopter_config_->maxVelMag;
  magnitudeBounds(1) = gcopter_config_->maxBdrMag;
  magnitudeBounds(2) = gcopter_config_->maxTiltAngle;
  magnitudeBounds(3) = gcopter_config_->minThrust;
  magnitudeBounds(4) = gcopter_config_->maxThrust;
  
  for (int i = 0; i < 5; i++) {
    penaltyWeights(i) = gcopter_config_->chiVec[i];
  }
  
  physicalParams(0) = gcopter_config_->vehicleMass;
  physicalParams(1) = gcopter_config_->gravAcc;
  physicalParams(2) = gcopter_config_->horizDrag;
  physicalParams(3) = gcopter_config_->vertDrag;
  physicalParams(4) = gcopter_config_->parasDrag;
  physicalParams(5) = gcopter_config_->speedEps;
  
  // 6. GCOPTER optimization
  ROS_INFO("[MINCO] Setting up GCOPTER: weightT=%.1f, dilateRadius=%.3f, corridors=%lu",
           gcopter_config_->weightT, gcopter_config_->dilateRadiusSoft, hPolys.size());
  
  if (!gcopter.setup(
          gcopter_config_->weightT, 
          gcopter_config_->dilateRadiusSoft,
          iniState, finState, 
          hPolys, 
          INFINITY,  // timeWeight
          gcopter_config_->smoothingEps,
          gcopter_config_->integralIntervs,
          magnitudeBounds, 
          penaltyWeights, 
          physicalParams)) {
    ROS_ERROR("[MINCO] GCOPTER setup failed (weightT=%.1f, dilate=%.3f), using B-spline fallback",
              gcopter_config_->weightT, gcopter_config_->dilateRadiusSoft);
    ROS_ERROR("[Debug] Try increasing DilateRadiusSoft or MaxCorridorSize in algorithm.xml");
    planExploreTrajBspline(tour, cur_vel, cur_acc, time_lb);
    return;
  }
  
  ROS_INFO("[MINCO] GCOPTER setup SUCCESS, starting optimization...");
  
  double minco_time_lb = time_lb > 0 ? time_lb : 0.0;
  Trajectory<7> minco_traj;
  
  double opt_result = gcopter.optimize(minco_traj, gcopter_config_->relCostTol, minco_time_lb);
  
  // Check if optimization result is valid
  if (opt_result < 0 || std::isinf(opt_result) || std::isnan(opt_result)) {
    ROS_ERROR("[MINCO] GCOPTER optimization failed (result: %f), using B-spline fallback", opt_result);
    ROS_ERROR("[Debug] MINCO FAILED - Invalid optimization result!");
    planExploreTrajBspline(tour, cur_vel, cur_acc, time_lb);
    return;
  }
  
  // Check if trajectory is valid
  double traj_duration = minco_traj.getTotalDuration();
  if (std::isinf(traj_duration) || std::isnan(traj_duration) || traj_duration <= 0 || traj_duration > 30.0) {
    ROS_ERROR("[MINCO] Invalid trajectory duration: %f, using B-spline fallback", traj_duration);
    ROS_ERROR("[Debug] MINCO FAILED - Trajectory duration invalid!");
    planExploreTrajBspline(tour, cur_vel, cur_acc, time_lb);
    return;
  }
  
  ROS_INFO("[MINCO] Optimization SUCCESS! Cost: %f, Duration: %f s", opt_result, traj_duration);
  
  // Check trajectory quality: ensure it's not too aggressive
  double max_vel_found = 0.0, max_acc_found = 0.0;
  double max_vel_y_found = 0.0;  // Maximum Y-direction velocity
  int quality_check_samples = std::max(20, (int)(traj_duration / 0.1));
  for (int i = 0; i <= quality_check_samples; ++i) {
    double t = i * traj_duration / quality_check_samples;
    Eigen::Vector3d vel = minco_traj.getVel(t);
    double vel_norm = vel.norm();
    double acc = minco_traj.getAcc(t).norm();
    double vel_y_abs = fabs(vel(1));  // Y-direction velocity (lateral)
    
    max_vel_found = std::max(max_vel_found, vel_norm);
    max_acc_found = std::max(max_acc_found, acc);
    max_vel_y_found = std::max(max_vel_y_found, vel_y_abs);
  }
  
  // Safety check: reject if trajectory is too aggressive
  double safe_vel_limit = gcopter_config_->maxVelMag * 1.2;  // Allow 20% overshoot
  double safe_acc_limit = 5.0;  // Conservative acceleration limit
  bool check_y_vel = (gcopter_config_->maxVelY > 0.0);  // Check Y velocity if limit is set
  double safe_vel_y_limit = gcopter_config_->maxVelY * 1.2;  // Allow 20% overshoot for Y
  
  bool vel_exceeded = max_vel_found > safe_vel_limit;
  bool acc_exceeded = max_acc_found > safe_acc_limit;
  bool vel_y_exceeded = check_y_vel && (max_vel_y_found > safe_vel_y_limit);
  
  if (vel_exceeded || acc_exceeded || vel_y_exceeded) {
    if (vel_exceeded) {
      ROS_WARN("[MINCO→B-spline] Velocity exceeded: %.2f m/s > %.2f m/s limit", 
               max_vel_found, safe_vel_limit);
    }
    if (acc_exceeded) {
      ROS_WARN("[MINCO→B-spline] Acceleration exceeded: %.2f m/s² > %.2f m/s² limit", 
               max_acc_found, safe_acc_limit);
    }
    if (vel_y_exceeded) {
      ROS_WARN("[MINCO→B-spline] Y-direction velocity exceeded: %.2f m/s > %.2f m/s limit", 
               max_vel_y_found, safe_vel_y_limit);
    }
    ROS_INFO("[MINCO→B-spline] Switching to B-spline for safety");
    planExploreTrajBspline(tour, cur_vel, cur_acc, time_lb);
    return;
  }
  
  // Log quality check results
  if (check_y_vel) {
    ROS_INFO("[MINCO] Quality check ✓: vel=%.2f/%.2f m/s, vel_y=%.2f/%.2f m/s, acc=%.2f/%.2f m/s²", 
             max_vel_found, safe_vel_limit, max_vel_y_found, safe_vel_y_limit, max_acc_found, safe_acc_limit);
  } else {
    ROS_INFO("[MINCO] Quality check ✓: vel=%.2f/%.2f m/s, acc=%.2f/%.2f m/s²", 
             max_vel_found, safe_vel_limit, max_acc_found, safe_acc_limit);
  }
  
  // 7. Store MINCO trajectory directly (no conversion needed!)
  // NEW APPROACH: Let MINCO trajectory execute directly without degrading to B-spline
  // This preserves MINCO's closed-form optimality and avoids conversion overhead
  ROS_INFO("[MINCO] Storing MINCO trajectory for direct execution");
  
  local_data_.minco_traj_ = minco_traj;
  local_data_.use_minco_traj_ = true;
  
  // Update trajectory info from MINCO (not B-spline)
  updateTrajInfoMINCO(minco_traj);
  
  // Publish MINCO trajectory for comparison if in debug mode
  if (debug_compare_mode_) {
    publishMINCOTraj(minco_traj, "MINCO");
  }
  
  double time_cost = (ros::Time::now() - start_time).toSec() * 1000.0;
  std::cout << "[MINCO] Trajectory optimization time: " << time_cost << " ms, " 
            << "duration: " << minco_traj.getTotalDuration() << " s, "
            << "corridor segments: " << hPolys.size() << std::endl;
}

void FastPlannerManager::getPointCloudFromESDF(
    const Eigen::Vector3d& min_bd, 
    const Eigen::Vector3d& max_bd,
    std::vector<Eigen::Vector3d>& surf_points) {
    
  surf_points.clear();
  
  // Get map resolution
  double resolution = edt_environment_->sdf_map_->getResolution();
  
  // Convert boundaries to voxel indices
  Eigen::Vector3i min_id, max_id;
  edt_environment_->sdf_map_->posToIndex(min_bd, min_id);
  edt_environment_->sdf_map_->posToIndex(max_bd, max_id);
  edt_environment_->sdf_map_->boundIndex(min_id);
  edt_environment_->sdf_map_->boundIndex(max_id);
  
  ROS_INFO("[MINCO] Searching occupancy grid in range [%d,%d,%d] to [%d,%d,%d]", 
           min_id.x(), min_id.y(), min_id.z(), max_id.x(), max_id.y(), max_id.z());
  
  // Directly iterate through occupancy grid (like EPIC's boxSearch)
  for (int x = min_id.x(); x <= max_id.x(); ++x) {
    for (int y = min_id.y(); y <= max_id.y(); ++y) {
      for (int z = min_id.z(); z <= max_id.z(); ++z) {
        Eigen::Vector3i idx(x, y, z);
        
        // Check if this voxel is occupied
        int occ = edt_environment_->sdf_map_->getOccupancy(idx);
        
        if (occ == SDFMap::OCCUPIED) {
          // Convert index to position and add to point cloud
          Eigen::Vector3d pos;
          edt_environment_->sdf_map_->indexToPos(idx, pos);
          surf_points.push_back(pos);
        }
      }
    }
  }
  
  ROS_INFO("[MINCO] Found %lu occupied voxels in occupancy grid", surf_points.size());
  
  // Downsample if too many points (similar to EPIC's VoxelGrid with 0.2 leaf size)
  if (surf_points.size() > 10000) {
    // Random downsampling
    std::random_shuffle(surf_points.begin(), surf_points.end());
    surf_points.resize(10000);
    ROS_INFO("[MINCO] Downsampled to %lu points", surf_points.size());
  } else if (surf_points.size() > 5000) {
    // Moderate downsampling
    std::random_shuffle(surf_points.begin(), surf_points.end());
    surf_points.resize(5000);
    ROS_INFO("[MINCO] Downsampled to %lu points", surf_points.size());
  }
  
  ROS_INFO("[MINCO] Final obstacle point cloud: %lu points", surf_points.size());
}

void FastPlannerManager::convertMINCOToBspline(
    const Trajectory<7>& minco_traj,
    NonUniformBspline& bspline_traj) {
    
  // Sample MINCO trajectory
  double duration = minco_traj.getTotalDuration();
  
  // Safety check
  if (std::isinf(duration) || std::isnan(duration) || duration <= 0) {
    ROS_ERROR("[MINCO] Cannot convert: invalid duration %f", duration);
    return;
  }
  
  double dt = 0.05;  // Sampling interval
  int num_samples = std::ceil(duration / dt) + 1;
  
  if (num_samples > 10000) {
    ROS_ERROR("[MINCO] Too many samples (%d), duration too long (%f)", num_samples, duration);
    return;
  }
  
  std::vector<Eigen::Vector3d> samples;
  for (int i = 0; i < num_samples; ++i) {
    double t = std::min(i * dt, duration);
    Eigen::Vector3d pos = minco_traj.getPos(t);
    samples.push_back(pos);
  }
  
  // Boundary conditions
  std::vector<Eigen::Vector3d> boundary_deri;
  boundary_deri.push_back(minco_traj.getVel(0.0));
  boundary_deri.push_back(minco_traj.getVel(duration));
  boundary_deri.push_back(minco_traj.getAcc(0.0));
  boundary_deri.push_back(minco_traj.getAcc(duration));
  
  // Convert to B-spline
  Eigen::MatrixXd ctrl_pts;
  NonUniformBspline::parameterizeToBspline(
      dt, samples, boundary_deri, pp_.bspline_degree_, ctrl_pts);
  
  bspline_traj.setUniformBspline(ctrl_pts, pp_.bspline_degree_, dt);
  
  ROS_INFO("[MINCO] Converted MINCO traj (%.2fs) to B-spline with %d control points", 
           duration, (int)ctrl_pts.rows());
}

void FastPlannerManager::updateTrajInfoMINCO(const Trajectory<7>& minco_traj) {
  // Update trajectory info using MINCO trajectory directly
  // For compatibility, we still need to create B-spline wrappers for visualization
  // But the actual control uses MINCO trajectory via getPosition/getVelocity/getAcceleration
  
  local_data_.duration_ = minco_traj.getTotalDuration();
  local_data_.start_time_ = ros::Time::now();
  local_data_.start_pos_ = minco_traj.getPos(0.0);
  local_data_.traj_id_ += 1;
  
  // Sample MINCO trajectory to create B-spline wrapper
  // This is ONLY for visualization and compatibility, not for control
  double dt = 0.05;  // 50ms sampling
  int num_samples = std::max(10, (int)(local_data_.duration_ / dt) + 1);
  
  std::vector<Eigen::Vector3d> pos_samples, vel_samples, acc_samples;
  for (int i = 0; i < num_samples; ++i) {
    double t = std::min(i * dt, local_data_.duration_);
    pos_samples.push_back(minco_traj.getPos(t));
    vel_samples.push_back(minco_traj.getVel(t));
    acc_samples.push_back(minco_traj.getAcc(t));
  }
  
  // Convert to Eigen matrices (control points)
  Eigen::MatrixXd pos_ctrl_pts, vel_ctrl_pts, acc_ctrl_pts;
  
  // Use parameterizeToBspline to get proper B-spline control points
  std::vector<Eigen::Vector3d> boundary_deri;
  boundary_deri.push_back(minco_traj.getVel(0.0));
  boundary_deri.push_back(minco_traj.getVel(local_data_.duration_));
  boundary_deri.push_back(minco_traj.getAcc(0.0));
  boundary_deri.push_back(minco_traj.getAcc(local_data_.duration_));
  
  NonUniformBspline::parameterizeToBspline(
      dt, pos_samples, boundary_deri, pp_.bspline_degree_, pos_ctrl_pts);
  
  // Set B-spline wrappers
  local_data_.position_traj_.setUniformBspline(pos_ctrl_pts, pp_.bspline_degree_, dt);
  local_data_.velocity_traj_ = local_data_.position_traj_.getDerivative();
  local_data_.acceleration_traj_ = local_data_.velocity_traj_.getDerivative();
  
  // Mark that we're using MINCO trajectory for control
  local_data_.use_minco_traj_ = true;
  
  ROS_INFO("[MINCO] Updated trajectory info: duration=%.3f s, start_pos=(%.3f, %.3f, %.3f), ctrl_pts=%d",
           local_data_.duration_, 
           local_data_.start_pos_.x(), local_data_.start_pos_.y(), local_data_.start_pos_.z(),
           (int)pos_ctrl_pts.cols());
}

// Debug comparison: Publish A* path as LINE_STRIP Marker
void FastPlannerManager::publishAStarPath(const vector<Eigen::Vector3d>& path) {
  visualization_msgs::Marker marker;
  marker.header.frame_id = "world";
  marker.header.stamp = ros::Time::now();
  marker.ns = "astar_path";
  marker.id = 0;
  marker.type = visualization_msgs::Marker::LINE_STRIP;
  marker.action = visualization_msgs::Marker::ADD;
  
  // Line properties
  marker.scale.x = 0.05;  // Line width
  marker.color.r = 0.0;   // Blue
  marker.color.g = 0.0;
  marker.color.b = 1.0;
  marker.color.a = 0.9;
  marker.pose.orientation.w = 1.0;
  
  // Add waypoints
  for (const auto& pt : path) {
    geometry_msgs::Point p;
    p.x = pt.x();
    p.y = pt.y();
    p.z = pt.z();
    marker.points.push_back(p);
  }
  
  debug_astar_path_pub_.publish(marker);
  ROS_INFO("[Debug] Published A* path with %lu waypoints to /debug/astar_path", path.size());
}

// Debug comparison: Publish B-spline trajectory as LINE_STRIP Marker
void FastPlannerManager::publishBsplineTraj(NonUniformBspline& traj, const std::string& label) {
  visualization_msgs::Marker marker;
  marker.header.frame_id = "world";
  marker.header.stamp = ros::Time::now();
  marker.ns = "bspline_traj";
  marker.id = 1;
  marker.type = visualization_msgs::Marker::LINE_STRIP;
  marker.action = visualization_msgs::Marker::ADD;
  
  // Line properties
  marker.scale.x = 0.08;  // Line width
  marker.color.r = 1.0;   // Yellow
  marker.color.g = 1.0;
  marker.color.b = 0.0;
  marker.color.a = 0.9;
  marker.pose.orientation.w = 1.0;
  
  // Sample trajectory points
  double tm, tmp;
  traj.getTimeSpan(tm, tmp);
  
  for (double t = tm; t <= tmp; t += 0.01) {  // Dense sampling for smooth curve
    Eigen::Vector3d pt = traj.evaluateDeBoorT(t);
    geometry_msgs::Point p;
    p.x = pt.x();
    p.y = pt.y();
    p.z = pt.z();
    marker.points.push_back(p);
  }
  
  debug_bspline_traj_pub_.publish(marker);
  ROS_INFO("[Debug] Published %s trajectory with %lu points to /debug/bspline_traj", 
           label.c_str(), marker.points.size());
}

// Debug comparison: Publish MINCO trajectory as LINE_STRIP Marker
void FastPlannerManager::publishMINCOTraj(Trajectory<7>& traj, const std::string& label) {
  visualization_msgs::Marker marker;
  marker.header.frame_id = "world";
  marker.header.stamp = ros::Time::now();
  marker.ns = "minco_traj";
  marker.id = 2;
  marker.type = visualization_msgs::Marker::LINE_STRIP;
  marker.action = visualization_msgs::Marker::ADD;
  
  // Line properties
  marker.scale.x = 0.10;  // Line width (slightly thicker to emphasize)
  marker.color.r = 1.0;   // Red
  marker.color.g = 0.0;
  marker.color.b = 0.0;
  marker.color.a = 1.0;
  marker.pose.orientation.w = 1.0;
  
  // Sample trajectory points
  double duration = traj.getTotalDuration();
  
  for (double t = 0; t <= duration; t += 0.01) {  // Dense sampling for smooth curve
    Eigen::Vector3d pt = traj.getPos(t);
    geometry_msgs::Point p;
    p.x = pt.x();
    p.y = pt.y();
    p.z = pt.z();
    marker.points.push_back(p);
  }
  
  debug_minco_traj_pub_.publish(marker);
  ROS_INFO("[Debug] Published %s trajectory with %lu points to /debug/minco_traj", 
           label.c_str(), marker.points.size());
}

// Rectangle collision detection helper function
bool FastPlannerManager::checkRectCollision(const Eigen::Vector3d& center, double yaw, double length, double width) {
  // Sample points within the rectangle to check for collision
  // Create a grid of sample points inside the rectangle
  const double sample_resolution = 0.1;  // Sample every 10cm
  int samples_x = std::max(1, (int)(length / sample_resolution));
  int samples_y = std::max(1, (int)(width / sample_resolution));
  
  // Rotation matrix for yaw
  double cos_yaw = cos(yaw);
  double sin_yaw = sin(yaw);
  
  // Check each sample point
  for (int i = 0; i <= samples_x; ++i) {
    for (int j = 0; j <= samples_y; ++j) {
      // Local coordinates in rectangle frame (centered at origin, aligned with x-axis)
      double local_x = (i * 2.0 / samples_x - 1.0) * length * 0.5;
      double local_y = (j * 2.0 / samples_y - 1.0) * width * 0.5;
      
      // Transform to world coordinates
      Eigen::Vector3d world_pt;
      world_pt(0) = center(0) + local_x * cos_yaw - local_y * sin_yaw;
      world_pt(1) = center(1) + local_x * sin_yaw + local_y * cos_yaw;
      world_pt(2) = center(2);  // Keep same height
      
      // Check if this point is in collision
      if (sdf_map_->getInflateOccupancy(world_pt) == 1) {
        return true;  // Collision detected
      }
    }
  }
  
  return false;  // No collision
}

// Visualize collision detection rectangles
void FastPlannerManager::visualizeCollisionRect(const Eigen::Vector3d& center, double yaw, 
                                                 double length, double width, int id) {
  visualization_msgs::Marker marker;
  marker.header.frame_id = "world";
  marker.header.stamp = ros::Time::now();
  marker.ns = "collision_rects";
  marker.id = id;
  marker.type = visualization_msgs::Marker::LINE_STRIP;
  marker.action = visualization_msgs::Marker::ADD;
  
  // Rectangle properties
  marker.scale.x = 0.02;  // Line width
  marker.color.r = 1.0;   // Red color
  marker.color.g = 0.0;
  marker.color.b = 0.0;
  marker.color.a = 0.8;
  marker.pose.orientation.w = 1.0;
  
  // Calculate rectangle corners in local frame
  double half_length = length * 0.5;
  double half_width = width * 0.5;
  
  // Four corners in local frame (centered at origin, aligned with x-axis)
  std::vector<Eigen::Vector2d> local_corners(4);
  local_corners[0] = Eigen::Vector2d(-half_length, -half_width);  // Back-left
  local_corners[1] = Eigen::Vector2d(half_length, -half_width);   // Front-left
  local_corners[2] = Eigen::Vector2d(half_length, half_width);     // Front-right
  local_corners[3] = Eigen::Vector2d(-half_length, half_width);    // Back-right
  
  // Rotation matrix for yaw
  double cos_yaw = cos(yaw);
  double sin_yaw = sin(yaw);
  
  // Transform corners to world coordinates and add to marker
  for (const auto& local_corner : local_corners) {
    geometry_msgs::Point p;
    p.x = center(0) + local_corner(0) * cos_yaw - local_corner(1) * sin_yaw;
    p.y = center(1) + local_corner(0) * sin_yaw + local_corner(1) * cos_yaw;
    p.z = center(2);
    marker.points.push_back(p);
  }
  
  // Close the rectangle by adding first point again
  geometry_msgs::Point p_first;
  p_first.x = center(0) + local_corners[0](0) * cos_yaw - local_corners[0](1) * sin_yaw;
  p_first.y = center(1) + local_corners[0](0) * sin_yaw + local_corners[0](1) * cos_yaw;
  p_first.z = center(2);
  marker.points.push_back(p_first);
  
  collision_rect_viz_pub_.publish(marker);
}

}  // namespace fast_planner
