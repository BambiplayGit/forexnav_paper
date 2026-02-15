#include "forex_nav/forex_nav_manager.h"
#include "forex_nav/astar_2d.h"
#include "forex_nav/minco/minco_wrapper.h"
#include <algorithm>
#include <utility>
#include <cmath>
#include <limits>
#include <iostream>
#include <Eigen/Core>
#include <nav_msgs/OccupancyGrid.h>

namespace forex_nav {

ForexNavManager::ForexNavManager() : has_last_goal_(false), last_best_goal_(Vector3d::Zero()) {
  nav_param_ = NavParam();
  astar_2d_ = std::make_shared<Astar2D>();
  astar_2d_->setResolution(0.1);
  minco_wrapper_ = std::make_shared<MincoWrapper>();
}

ForexNavManager::~ForexNavManager() {
}

void ForexNavManager::initialize() {
  // 用统一参数设置规划高度
  double h = nav_param_.planning_height_;
  astar_2d_->setFixedHeight(h);
  minco_wrapper_->setFixedHeight(h);
  std::cout << "[ForexNav] Planning height set to " << h << " m" << std::endl;
  
  // 转发 MINCO 优化参数到 wrapper
  minco_wrapper_->setMincoOptConfig(
      nav_param_.minco_weight_time_,
      nav_param_.minco_weight_energy_,
      nav_param_.minco_weight_pos_,
      nav_param_.minco_weight_vel_,
      nav_param_.minco_weight_acc_,
      nav_param_.minco_weight_jerk_,
      nav_param_.minco_max_jerk_,
      nav_param_.minco_alloc_speed_ratio_,
      nav_param_.minco_length_per_piece_);
}

void ForexNavManager::setMap(const nav_msgs::OccupancyGrid::ConstPtr& map) {
  if (astar_2d_) {
    astar_2d_->setMap(map);
  }
  if (minco_wrapper_) {
    minco_wrapper_->setMap(map);
  }
}

void ForexNavManager::setOccCloud3D(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, double voxel_res) {
  if (minco_wrapper_) {
    minco_wrapper_->setOccCloud3D(cloud, voxel_res);
  }
}

bool ForexNavManager::generateCorridors3D(const std::vector<Vector3d>& path) {
  if (minco_wrapper_) {
    return minco_wrapper_->generateCorridors3D(path);
  }
  return false;
}

std::vector<std::array<double, 6>> ForexNavManager::getLastCorridors3D() const {
  std::vector<std::array<double, 6>> result;
  if (minco_wrapper_) {
    const auto& corridors = minco_wrapper_->getLastCorridors3D();
    for (const auto& corr : corridors) {
      result.push_back({corr.x_min, corr.x_max, corr.y_min, corr.y_max, corr.z_min, corr.z_max});
    }
  }
  return result;
}

int ForexNavManager::planNavMotion(
    const Vector3d& pos, const Vector3d& vel, double yaw,
    const Vector3d& goal_pos,
    const std::vector<Viewpoint>& viewpoints,
    std::vector<Vector3d>& path,
    std::vector<double>& yaws,
    std::vector<double>& times,
    std::vector<Vector3d>& astar_path,
    Trajectory<5>* out_traj) {
  
  // Step 1: Select best viewpoint
  Vector3d selected_pos;
  double selected_yaw;
  double selected_v_limit;
  int result = selectBestViewpoint(pos, vel, yaw, goal_pos, viewpoints, selected_pos, selected_yaw, selected_v_limit);
  
  if (result < 0) {
    return -1;  // FAIL
  }
  
  // Log selected viewpoint
  double dist_to_goal = (selected_pos - goal_pos).norm();
  double dist_from_current = (selected_pos - pos).norm();
  std::cout << "[ForexNav] Selected viewpoint: pos=(" << selected_pos.x() << ", " 
            << selected_pos.y() << ", " << selected_pos.z() << "), "
            << "dist_to_goal=" << dist_to_goal << "m, "
            << "dist_from_current=" << dist_from_current << "m" << std::endl;
  
  // Step 2: Plan path to selected viewpoint (frontend: A* path planning)
  std::vector<Vector3d> temp_path;
  if (!planPath(pos, selected_pos, temp_path)) {
    // A* failed, return error
    return -1;  // FAIL
  }
  
  // Store A* path for visualization (before shortening)
  astar_path = temp_path;
  
  // Step 2.5: Shorten path to remove redundant waypoints
  shortenPath(temp_path);
  
  // Ensure at least 3 points for MINCO (needs start, intermediate, end)
  if (temp_path.size() < 2) {
    temp_path.push_back(selected_pos);
  }
  if (temp_path.size() == 2) {
    // Insert intermediate point
    Vector3d mid = (temp_path[0] + temp_path[1]) * 0.5;
    temp_path.insert(temp_path.begin() + 1, mid);
  }
  
  // Step 3: Generate MINCO trajectory (backend: trajectory optimization)
  // Calculate yaws based on path direction
  std::vector<double> temp_yaws;
  for (size_t i = 0; i < temp_path.size(); ++i) {
    if (i < temp_path.size() - 1) {
      // Yaw points in the direction of next waypoint
      double path_yaw = std::atan2(temp_path[i + 1].y() - temp_path[i].y(),
                                   temp_path[i + 1].x() - temp_path[i].x());
      temp_yaws.push_back(path_yaw);
    } else {
      // Last point uses selected_yaw
      temp_yaws.push_back(selected_yaw);
    }
  }
  
  // Compute end_vel: non-zero for intermediate viewpoints, zero for final goal
  Vector3d end_vel = Vector3d::Zero();
  double dist_selected_to_goal = (selected_pos - goal_pos).norm();
  if (selected_v_limit > 0.01 && dist_selected_to_goal > 0.5) {
    // Intermediate viewpoint: maintain velocity toward goal
    Vector3d dir_to_goal = (goal_pos - selected_pos).normalized();
    end_vel = selected_v_limit * dir_to_goal;
  }
  Vector3d start_acc = Vector3d::Zero();
  generateMINCOTrajectory(temp_path, temp_yaws, vel, start_acc, yaw, end_vel, selected_yaw, path, yaws, times, out_traj);
  
  return 0;  // SUCCESS
}

int ForexNavManager::selectBestViewpoint(
    const Vector3d& pos, const Vector3d& vel, double yaw,
    const Vector3d& goal_pos,
    const std::vector<Viewpoint>& viewpoints,
    Vector3d& selected_pos,
    double& selected_yaw,
    double& selected_v_limit) {
  
  if (viewpoints.empty()) {
    return -1;
  }
  
  // Check if goal is reachable and within field of view
  const double VIEW_RANGE = 10.0;  // Maximum sensing/view range (meters)
  
  // Use proper reachability check with occupancy grid
  bool goal_reachable = isGoalReachable(pos, goal_pos);
  
  if (goal_reachable) {
    // Check if goal is within view range of any viewpoint
    bool goal_in_view = false;
    double min_dist_to_viewpoint = std::numeric_limits<double>::max();
    
    for (const auto& vp : viewpoints) {
      double dist = (goal_pos - vp.pos_).norm();
      min_dist_to_viewpoint = std::min(min_dist_to_viewpoint, dist);
      if (dist < VIEW_RANGE) {
        goal_in_view = true;
        break;
      }
    }
    
    // Also check if goal is close to current position (already in view)
    double dist_to_goal = (goal_pos - pos).norm();
    if (dist_to_goal < VIEW_RANGE) {
      goal_in_view = true;
    }
    
    if (goal_in_view) {
      // Goal is reachable and in view, use it directly
      candidate_paths_.clear();
      selected_pos = goal_pos;
      selected_yaw = std::atan2(goal_pos.y() - pos.y(), goal_pos.x() - pos.x());
      selected_v_limit = 0.0;  // Stop at goal
      std::cout << "[ForexNav] Goal is reachable and in view, selecting goal directly" << std::endl;
      std::cout << "  Goal pos: (" << goal_pos.x() << ", " << goal_pos.y() << ", " << goal_pos.z() << ")" << std::endl;
      std::cout << "  Distance to goal: " << dist_to_goal << " m" << std::endl;
      std::cout << "  Min distance to viewpoint: " << min_dist_to_viewpoint << " m" << std::endl;
      return 0;
    } else {
      std::cout << "[ForexNav] Goal is reachable but not in view (min_dist_to_viewpoint=" 
                << min_dist_to_viewpoint << "m > " << VIEW_RANGE << "m), selecting viewpoint" << std::endl;
    }
  } else {
    std::cout << "[ForexNav] Goal is not reachable, selecting viewpoint" << std::endl;
  }
  
  // Goal is not in view or not reachable, select best viewpoint
  // Compute 4 costs for each viewpoint (matching ref implementation)
  std::vector<double> all_pred_costs;   // Cost 1: predicted path cost (A* path length as proxy for fuzzy A*)
  std::vector<double> all_dists;        // Cost 2: distance to goal
  std::vector<double> all_curve_costs;  // Cost 3: curve dynamics cost
  std::vector<double> all_homo_costs;   // Cost 4: homotopy consistency cost
  std::vector<double> all_v_limits;     // Velocity limits from curve cost

  for (size_t i = 0; i < viewpoints.size(); ++i) {
    // [Cost 2] Distance cost
    double dist_cost = calculateDistanceCost(viewpoints[i].pos_, goal_pos);
    all_dists.push_back(dist_cost);

    // [Cost 3] Curve cost
    CurveResult res = calculateCurveCost(pos, yaw, vel, viewpoints[i].pos_, viewpoints[i].yaw_,
                                          nav_param_.max_vel_, nav_param_.max_acc_, nav_param_.max_yawdot_);
    all_curve_costs.push_back(res.cost);
    all_v_limits.push_back(res.v_limit);

    // [Cost 4] Homotopy consistency cost
    double homo_cost = 0.0;
    if (has_last_goal_) {
      Vector3d dir_old = (last_best_goal_ - pos).normalized();
      Vector3d dir_new = (viewpoints[i].pos_ - pos).normalized();
      double dot = dir_old.dot(dir_new);
      homo_cost = (1.0 - dot) * 0.5;
      if (dot < 0.0) {
        homo_cost *= 10.0;  // Heavy penalty for going backward
      }
    }
    all_homo_costs.push_back(homo_cost);

    // [Cost 1] Predicted path cost (use A* path length as proxy for fuzzy A*)
    std::vector<Vector3d> pred_path;
    double pred_cost = std::numeric_limits<double>::max();
    bool found = planPath(viewpoints[i].pos_, goal_pos, pred_path);
    if (found && pred_path.size() >= 2) {
      pred_cost = 0.0;
      for (size_t j = 1; j < pred_path.size(); ++j) {
        pred_cost += (pred_path[j] - pred_path[j - 1]).norm();
      }
      all_pred_costs.push_back(pred_cost);
    } else {
      all_pred_costs.push_back(-1.0);  // Mark as invalid
    }
  }

  // Compute min/max for normalization
  double min_pred = 1e9, max_pred = -1e9;
  double min_dist = 1e9, max_dist = -1e9;
  double min_curve = 1e9, max_curve = -1e9;
  double min_homo = 1e9, max_homo = -1e9;
  bool has_valid_pred = false;

  for (size_t i = 0; i < viewpoints.size(); ++i) {
    min_dist  = std::min(min_dist,  all_dists[i]);
    max_dist  = std::max(max_dist,  all_dists[i]);
    min_curve = std::min(min_curve, all_curve_costs[i]);
    max_curve = std::max(max_curve, all_curve_costs[i]);
    min_homo  = std::min(min_homo,  all_homo_costs[i]);
    max_homo  = std::max(max_homo,  all_homo_costs[i]);
    if (all_pred_costs[i] >= 0) {
      has_valid_pred = true;
      min_pred = std::min(min_pred, all_pred_costs[i]);
      max_pred = std::max(max_pred, all_pred_costs[i]);
    }
  }

  double pred_range  = has_valid_pred ? std::max(1e-6, max_pred - min_pred) : 1.0;
  double dist_range  = std::max(1e-6, max_dist - min_dist);
  double curve_range = std::max(1e-6, max_curve - min_curve);
  double homo_range  = std::max(1e-6, max_homo - min_homo);

  // Compute mixed cost
  const double PENALTY_FACTOR = 2.0;
  std::vector<std::pair<size_t, double>> idx_cost;
  int best_idx = -1;
  double min_mixed_cost = std::numeric_limits<double>::max();

  for (size_t i = 0; i < viewpoints.size(); ++i) {
    double norm_pred;
    if (all_pred_costs[i] < 0) {
      norm_pred = has_valid_pred ? 1.0 * PENALTY_FACTOR : 1.0;
    } else {
      norm_pred = (all_pred_costs[i] - min_pred) / pred_range;
    }

    double norm_dist  = (all_dists[i] - min_dist) / dist_range;
    double norm_curve = (all_curve_costs[i] - min_curve) / curve_range;
    double norm_homo  = (all_homo_costs[i] - min_homo) / homo_range;

    double mixed_cost = nav_param_.w_dist_  * norm_dist +
                        nav_param_.w_pred_  * norm_pred +
                        nav_param_.w_curve_ * norm_curve +
                        nav_param_.w_homo_  * norm_homo;

    idx_cost.push_back({i, mixed_cost});
    if (mixed_cost < min_mixed_cost) {
      min_mixed_cost = mixed_cost;
      best_idx = static_cast<int>(i);
    }
  }

  std::sort(idx_cost.begin(), idx_cost.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

  // Plan candidate paths for visualization (viewpoint -> goal)
  candidate_paths_.clear();
  int n_vis = nav_param_.vis_candidate_path_count_;
  int n_take = (n_vis < 0) ? static_cast<int>(idx_cost.size())
                           : std::min(n_vis, static_cast<int>(idx_cost.size()));
  for (int k = 0; k < n_take; ++k) {
    std::vector<Vector3d> path;
    if (planPath(viewpoints[idx_cost[k].first].pos_, goal_pos, path)) {
      candidate_paths_.push_back(path);
    }
  }

  if (best_idx >= 0) {
    selected_pos = viewpoints[best_idx].pos_;
    selected_yaw = viewpoints[best_idx].yaw_;
    selected_v_limit = all_v_limits[best_idx];

    // Update homotopy state
    has_last_goal_ = true;
    last_best_goal_ = selected_pos;

    // Log selected viewpoint info
    double dist_to_goal = (selected_pos - goal_pos).norm();
    double dist_from_current = (selected_pos - pos).norm();
    std::cout << "[Viewpoint Selection] Selected viewpoint #" << best_idx 
              << " out of " << viewpoints.size() << " candidates" << std::endl;
    std::cout << "  Position: (" << selected_pos.x() << ", " << selected_pos.y() << ", " << selected_pos.z() << ")" << std::endl;
    std::cout << "  Distance to goal: " << dist_to_goal << " m" << std::endl;
    std::cout << "  Distance from current: " << dist_from_current << " m" << std::endl;
    std::cout << "  Mixed cost: " << min_mixed_cost 
              << " (w_dist=" << nav_param_.w_dist_ << ", w_pred=" << nav_param_.w_pred_
              << ", w_curve=" << nav_param_.w_curve_ << ", w_homo=" << nav_param_.w_homo_ << ")" << std::endl;

    return 0;
  }

  selected_v_limit = 0.0;
  return -1;
}

bool ForexNavManager::planPath(
    const Vector3d& start, const Vector3d& end, std::vector<Vector3d>& path) {
  if (!astar_2d_) {
    // Fallback to straight line if A* not available
    path.clear();
    path.push_back(start);
    path.push_back(end);
    return true;
  }
  
  bool success = astar_2d_->search(start, end, path);
  
  if (!success || path.empty()) {
    // Fallback to straight line if A* fails
    path.clear();
    path.push_back(start);
    path.push_back(end);
    return true;
  }
  
  return true;
}

bool ForexNavManager::isPointInFreeSpace(const Vector3d& pos) {
  if (!astar_2d_) {
    return false;
  }
  
  // Use A*'s internal map check - check if occupancy value is 0 (free space)
  Eigen::Vector2i map_coord = astar_2d_->worldToMap(pos);
  return astar_2d_->isFree(map_coord.x(), map_coord.y());
}

bool ForexNavManager::isGoalReachable(const Vector3d& /* start */, const Vector3d& goal) {
  // Simply check if goal point is in free space (occupancy value is 0)
  return isPointInFreeSpace(goal);
}

void ForexNavManager::shortenPath(std::vector<Vector3d>& path) {
  if (path.size() <= 2) return;
  
  // Reduce threshold to keep more waypoints for smoother MINCO trajectory
  const double dist_thresh = 0.3;  // Minimum distance between waypoints (reduced from 0.5)
  std::vector<Vector3d> short_path;
  short_path.push_back(path[0]);
  
  for (size_t i = 1; i < path.size() - 1; ++i) {
    double dist = (path[i] - short_path.back()).norm();
    if (dist > dist_thresh) {
      short_path.push_back(path[i]);
    }
  }
  
  // Always include the last point
  if ((path.back() - short_path.back()).norm() > 1e-3) {
    short_path.push_back(path.back());
  }
  
  // Ensure we have at least 3 points for MINCO (needs intermediate points)
  if (short_path.size() < 3 && path.size() >= 3) {
    // Keep more points if shortened too much
    short_path = path;
    // Only remove very close points
    std::vector<Vector3d> filtered_path;
    filtered_path.push_back(short_path[0]);
    for (size_t i = 1; i < short_path.size() - 1; ++i) {
      if ((short_path[i] - filtered_path.back()).norm() > 0.2) {
        filtered_path.push_back(short_path[i]);
      }
    }
    filtered_path.push_back(short_path.back());
    path = filtered_path;
  } else {
    path = short_path;
  }
}

void ForexNavManager::generateMINCOTrajectory(
    const std::vector<Vector3d>& path,
    const std::vector<double>& yaws,
    const Vector3d& start_vel,
    const Vector3d& start_acc,
    double start_yaw,
    const Vector3d& end_vel,
    double end_yaw,
    std::vector<Vector3d>& traj_pos,
    std::vector<double>& traj_yaw,
    std::vector<double>& traj_time,
    Trajectory<5>* out_traj) {
  
  traj_pos.clear();
  traj_yaw.clear();
  traj_time.clear();
  
  if (path.empty()) {
    std::cout << "[ForexNav] Error: Empty path for MINCO generation" << std::endl;
    return;
  }
  
  std::cout << "[ForexNav] Generating MINCO trajectory from " << path.size() << " waypoints" << std::endl;
  
  // Use real MINCO trajectory generation
  if (minco_wrapper_) {
    bool success = minco_wrapper_->generateTrajectory(
        path,
        start_vel,
        start_acc,
        start_yaw,
        end_vel,
        end_yaw,
        nav_param_.max_vel_,
        nav_param_.max_acc_,
        nav_param_.max_yawdot_,
        nav_param_.traj_dt_,
        traj_pos,
        traj_yaw,
        traj_time,
        out_traj);
    
    if (success && !traj_pos.empty()) {
      std::cout << "[ForexNav] MINCO trajectory generated successfully: " 
                << traj_pos.size() << " points" << std::endl;
      
      // Verify trajectory quality
      if (traj_pos.size() < path.size()) {
        std::cout << "[ForexNav] Warning: MINCO trajectory has fewer points than waypoints" << std::endl;
      }
      
      // Check if trajectory is smooth (not just waypoints)
      bool is_smooth = false;
      if (traj_pos.size() > path.size() * 2) {
        is_smooth = true;
      } else {
        // Check if trajectory points differ from waypoints
        double max_deviation = 0.0;
        for (const auto& wp : path) {
          double min_dist = std::numeric_limits<double>::max();
          for (const auto& tp : traj_pos) {
            min_dist = std::min(min_dist, (tp - wp).norm());
          }
          max_deviation = std::max(max_deviation, min_dist);
        }
        if (max_deviation > 0.1) {
          is_smooth = true;
        }
      }
      
      if (is_smooth) {
        std::cout << "[ForexNav] MINCO trajectory is smooth and optimized" << std::endl;
      } else {
        std::cout << "[ForexNav] Warning: MINCO trajectory may not be optimized (too similar to waypoints)" << std::endl;
      }
      
      return;
    } else {
      std::cout << "[ForexNav] MINCO generation failed or returned empty trajectory" << std::endl;
    }
  } else {
    std::cout << "[ForexNav] Error: MINCO wrapper is null" << std::endl;
  }
  
  // Fallback to simple interpolation if MINCO fails
  std::cout << "[ForexNav] Using fallback interpolation (MINCO not available or failed)" << std::endl;
  
  double current_time = 0.0;
  for (size_t i = 0; i < path.size(); ++i) {
    traj_pos.push_back(path[i]);
    double yaw = (i < yaws.size()) ? yaws[i] : (i > 0 ? yaws.back() : start_yaw);
    traj_yaw.push_back(yaw);
    traj_time.push_back(current_time);
    
    if (i < path.size() - 1) {
      double dist = (path[i + 1] - path[i]).norm();
      double dt = dist / nav_param_.max_vel_;
      dt = std::max(dt, nav_param_.traj_dt_);
      current_time += dt;
    }
  }
  
  std::cout << "[ForexNav] Fallback interpolation completed: " << traj_pos.size() << " points" << std::endl;
}

ForexNavManager::CurveResult ForexNavManager::calculateCurveCost(
    const Vector3d& curr_p, double curr_y,
    const Vector3d& curr_v,
    const Vector3d& target_p, double target_y,
    double v_max, double a_max, double yv_max) {
  
  const double lambda_a = 2.0;
  const double yaw_gain = 1.0;
  
  Vector3d diff_p = target_p - curr_p;
  double dist = diff_p.norm();
  if (dist < 1e-3) return {0.0, 0.0};
  
  Vector3d dp = diff_p.normalized();
  double v_norm = curr_v.norm();
  double v_final_limit = v_max;
  double travel_distance = dist;
  
  if (v_norm > 0.1) {
    double cos_theta = curr_v.normalized().dot(dp);
    cos_theta = std::max(-1.0, std::min(1.0, cos_theta));
    double theta = std::acos(cos_theta);
    
    if (theta > 1e-3) {
      double r = dist / (2.0 * std::sin(theta * 0.5));
      double v_curve_limit = std::sqrt(lambda_a * a_max * r);
      v_final_limit = std::min(v_curve_limit, v_max);
      travel_distance = r * theta;
    }
  }
  
  double t_trans = travel_distance / v_final_limit;
  double yaw_diff = std::abs(target_y - curr_y);
  while (yaw_diff > M_PI) yaw_diff -= 2.0 * M_PI;
  while (yaw_diff < -M_PI) yaw_diff += 2.0 * M_PI;
  double t_yaw = std::abs(yaw_diff) / yv_max;
  
  return {std::max(t_trans, t_yaw * yaw_gain), v_final_limit};
}

double ForexNavManager::calculateDistanceCost(const Vector3d& pos, const Vector3d& goal) {
  return (pos - goal).norm();
}

const std::vector<std::vector<Vector3d>>& ForexNavManager::getLastCandidatePaths() const {
  return candidate_paths_;
}

std::vector<std::array<double, 6>> ForexNavManager::getLastCorridors() const {
  std::vector<std::array<double, 6>> result;
  if (minco_wrapper_) {
    const auto& corridors = minco_wrapper_->getLastCorridors();
    for (const auto& corr : corridors) {
      result.push_back({corr.x_min, corr.x_max, corr.y_min, corr.y_max, corr.z_min, corr.z_max});
    }
  }
  return result;
}

}  // namespace forex_nav
