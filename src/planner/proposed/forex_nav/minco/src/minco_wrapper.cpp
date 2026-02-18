#include "forex_nav/minco/minco_wrapper.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace forex_nav {

MincoWrapper::MincoWrapper() {
    sfc_generator_.setInflateRadius(0.4);
    sfc_generator_.setFixedHeight(1.0);  // 默认值，会被 setFixedHeight() 覆盖
}

void MincoWrapper::setFixedHeight(double height) {
    sfc_generator_.setFixedHeight(height);
}

void MincoWrapper::setMincoOptConfig(double weight_time, double weight_energy,
                                     double weight_pos, double weight_vel,
                                     double weight_acc, double weight_jerk,
                                     double max_jerk, double alloc_speed_ratio,
                                     double length_per_piece,
                                     double weight_guide) {
    stored_config_.weight_time = weight_time;
    stored_config_.weight_energy = weight_energy;
    stored_config_.weight_pos = weight_pos;
    stored_config_.weight_vel = weight_vel;
    stored_config_.weight_acc = weight_acc;
    stored_config_.weight_jerk = weight_jerk;
    stored_config_.weight_guide = weight_guide;
    stored_config_.max_jerk = max_jerk;
    stored_config_.alloc_speed = alloc_speed_ratio;  // 暂存 ratio, generateTrajectory 中乘以 max_vel
    stored_config_.length_per_piece = length_per_piece;
    has_user_config_ = true;
    std::cout << "[MincoWrapper] OptConfig set: wt=" << weight_time
              << " we=" << weight_energy << " wp=" << weight_pos
              << " wv=" << weight_vel << " wa=" << weight_acc
              << " wj=" << weight_jerk << " wg=" << weight_guide
              << " mj=" << max_jerk
              << " asr=" << alloc_speed_ratio << " lpp=" << length_per_piece << std::endl;
}

MincoWrapper::~MincoWrapper() {
}

void MincoWrapper::setMap(const nav_msgs::OccupancyGrid::ConstPtr& map) {
    if (map) {
        sfc_generator_.setMap(map);
        has_map_ = true;
    }
}

void MincoWrapper::setOccCloud3D(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, double voxel_res) {
    sfc_generator_.setOccCloud3D(cloud, voxel_res);
    has_cloud_3d_ = true;
}

bool MincoWrapper::generateCorridors3D(const std::vector<Eigen::Vector3d>& waypoints) {
    if (!has_cloud_3d_ || waypoints.size() < 2) {
        last_corridors_3d_.clear();
        return false;
    }
    std::vector<AABB2D> corridors;
    bool ok = sfc_generator_.generateCorridors3D(waypoints, corridors);
    last_corridors_3d_ = corridors;
    return ok;
}

bool MincoWrapper::generateTrajectory(
    const std::vector<Vector3d>& waypoints,
    const Vector3d& start_vel,
    const Vector3d& start_acc,
    double start_yaw,
    const Vector3d& end_vel,
    double end_yaw,
    double max_vel,
    double max_acc,
    double /* max_yawdot */,
    double traj_dt,
    std::vector<Vector3d>& traj_pos,
    std::vector<double>& traj_yaw,
    std::vector<double>& traj_time,
    Trajectory<5>* out_traj,
    const std::vector<Vector3d>& ref_path) {
    
    if (waypoints.size() < 2) {
        std::cout << "[MincoWrapper] Error: Need at least 2 waypoints, got " << waypoints.size() << std::endl;
        return false;
    }
    
    traj_pos.clear();
    traj_yaw.clear();
    traj_time.clear();
    
    std::cout << "[MincoWrapper] Generating trajectory from " << waypoints.size() << " waypoints" << std::endl;
    std::cout << "[MincoWrapper] Start: (" << waypoints[0].transpose() << "), End: (" << waypoints.back().transpose() << ")" << std::endl;
    std::cout << "[MincoWrapper] Constraints: max_vel=" << max_vel << ", max_acc=" << max_acc << std::endl;
    
    std::vector<Vector3d> actual_waypoints = waypoints;
    if (waypoints.size() == 2) {
        Vector3d mid = (waypoints[0] + waypoints[1]) * 0.5;
        actual_waypoints.insert(actual_waypoints.begin() + 1, mid);
        std::cout << "[MincoWrapper] Added midpoint for 2-waypoint case" << std::endl;
    }
    
    if (!has_map_) {
        std::cout << "[MincoWrapper] No map available, using fallback (no SFC optimization)" << std::endl;
        return generateFallbackTrajectory(
            actual_waypoints, start_vel, start_acc, start_yaw,
            end_vel, end_yaw, max_vel, max_acc, traj_dt,
            traj_pos, traj_yaw, traj_time, out_traj);
    }
    
    std::vector<AABB2D> corridors;
    if (!sfc_generator_.generateCorridors(actual_waypoints, corridors)) {
        std::cout << "[MincoWrapper] SFC generation failed, using fallback" << std::endl;
        last_corridors_.clear();
        return generateFallbackTrajectory(
            actual_waypoints, start_vel, start_acc, start_yaw,
            end_vel, end_yaw, max_vel, max_acc, traj_dt,
            traj_pos, traj_yaw, traj_time, out_traj);
    }
    
    last_corridors_ = corridors;
    
    std::cout << "[MincoWrapper] Generated " << corridors.size() << " SFC corridors" << std::endl;
    for (size_t i = 0; i < corridors.size(); ++i) {
        std::cout << "  Corridor " << i << ": x=[" << corridors[i].x_min << "," << corridors[i].x_max 
                  << "], y=[" << corridors[i].y_min << "," << corridors[i].y_max << "]" << std::endl;
    }
    
    Eigen::Matrix3d headPVA, tailPVA;
    headPVA.col(0) = actual_waypoints[0];
    headPVA.col(1) = start_vel;
    headPVA.col(2) = start_acc;
    
    tailPVA.col(0) = actual_waypoints.back();
    tailPVA.col(1) = end_vel;
    tailPVA.col(2) = Vector3d::Zero();
    
    GCopter2D::OptConfig config;
    config.max_vel = max_vel;
    config.max_acc = max_acc;
    if (has_user_config_) {
        config.max_jerk = stored_config_.max_jerk;
        config.weight_time = stored_config_.weight_time;
        config.weight_energy = stored_config_.weight_energy;
        config.weight_pos = stored_config_.weight_pos;
        config.weight_vel = stored_config_.weight_vel;
        config.weight_acc = stored_config_.weight_acc;
        config.weight_jerk = stored_config_.weight_jerk;
        config.weight_guide = stored_config_.weight_guide;
        config.alloc_speed = max_vel * stored_config_.alloc_speed;  // stored_config_.alloc_speed 存的是 ratio
        config.length_per_piece = stored_config_.length_per_piece;
    } else {
        config.max_jerk = 15.0;
        config.weight_time = 30.0;
        config.weight_pos = 2000.0;
        config.weight_vel = 100.0;
        config.weight_acc = 80.0;
        config.weight_jerk = 30.0;
        config.weight_guide = 100.0;
        config.alloc_speed = max_vel * 0.7;
    }
    config.smooth_eps = 0.01;
    config.integral_resolution = 16;
    config.rel_cost_tol = 1e-4;
    
    gcopter_.setConfig(config);
    
    if (!ref_path.empty()) {
        gcopter_.setReferencePath(ref_path);
    }
    
    Trajectory<5> traj;
    bool success = gcopter_.optimize(actual_waypoints, corridors, headPVA, tailPVA, traj);
    
    if (!success || traj.getPieceNum() == 0) {
        std::cout << "[MincoWrapper] GCOPTER optimization failed, using fallback" << std::endl;
        return generateFallbackTrajectory(
            actual_waypoints, start_vel, start_acc, start_yaw,
            end_vel, end_yaw, max_vel, max_acc, traj_dt,
            traj_pos, traj_yaw, traj_time, out_traj);
    }
    
    // Output continuous trajectory object for real-time evaluation (p/v/a)
    if (out_traj) *out_traj = traj;
    
    sampleTrajectory(traj, traj_dt, start_yaw, end_yaw, traj_pos, traj_yaw, traj_time);
    
    if (traj_pos.empty()) {
        std::cout << "[MincoWrapper] Error: Sampled trajectory is empty" << std::endl;
        return false;
    }
    
    double traj_length = 0.0;
    for (size_t i = 1; i < traj_pos.size(); ++i) {
        traj_length += (traj_pos[i] - traj_pos[i-1]).norm();
    }
    
    double max_vel_actual = 0.0;
    for (size_t i = 1; i < traj_pos.size(); ++i) {
        double dt = traj_time[i] - traj_time[i-1];
        if (dt > 1e-6) {
            double vel = (traj_pos[i] - traj_pos[i-1]).norm() / dt;
            max_vel_actual = std::max(max_vel_actual, vel);
        }
    }
    
    std::cout << "[MincoWrapper] SUCCESS: " << traj_pos.size() << " points, "
              << "duration=" << traj_time.back() << "s, "
              << "length=" << traj_length << "m, "
              << "max_vel=" << max_vel_actual << "m/s" << std::endl;
    
    return true;
}

void MincoWrapper::sampleTrajectory(
    const Trajectory<5>& traj,
    double dt,
    double start_yaw,
    double end_yaw,
    std::vector<Vector3d>& traj_pos,
    std::vector<double>& traj_yaw,
    std::vector<double>& traj_time) {
    
    traj_pos.clear();
    traj_yaw.clear();
    traj_time.clear();
    
    double total_duration = traj.getTotalDuration();
    if (total_duration <= 0) return;
    
    double t = 0.0;
    while (t < total_duration) {
        Vector3d pos = traj.getPos(t);
        Vector3d vel = traj.getVel(t);
        
        double yaw;
        if (vel.head<2>().norm() > 0.1) {
            yaw = std::atan2(vel.y(), vel.x());
        } else {
            double alpha = t / total_duration;
            double yaw_diff = end_yaw - start_yaw;
            while (yaw_diff > M_PI) yaw_diff -= 2.0 * M_PI;
            while (yaw_diff < -M_PI) yaw_diff += 2.0 * M_PI;
            yaw = start_yaw + alpha * yaw_diff;
        }
        
        while (yaw > M_PI) yaw -= 2.0 * M_PI;
        while (yaw < -M_PI) yaw += 2.0 * M_PI;
        
        traj_pos.push_back(pos);
        traj_yaw.push_back(yaw);
        traj_time.push_back(t);
        
        t += dt;
    }
    
    if (traj_time.empty() || std::abs(traj_time.back() - total_duration) > 1e-6) {
        Vector3d pos = traj.getPos(total_duration);
        traj_pos.push_back(pos);
        traj_yaw.push_back(end_yaw);
        traj_time.push_back(total_duration);
    }
}

bool MincoWrapper::generateFallbackTrajectory(
    const std::vector<Vector3d>& waypoints,
    const Vector3d& start_vel,
    const Vector3d& start_acc,
    double start_yaw,
    const Vector3d& end_vel,
    double end_yaw,
    double max_vel,
    double /* max_acc */,
    double traj_dt,
    std::vector<Vector3d>& traj_pos,
    std::vector<double>& traj_yaw,
    std::vector<double>& traj_time,
    Trajectory<5>* out_traj) {
    
    std::cout << "[MincoWrapper] Using fallback trajectory generation (no optimization)" << std::endl;
    
    int N = waypoints.size() - 1;
    if (N < 2) {
        std::vector<Vector3d> actual_waypoints = waypoints;
        if (N == 1) {
            Vector3d mid = (waypoints[0] + waypoints[1]) * 0.5;
            actual_waypoints.insert(actual_waypoints.begin() + 1, mid);
            N = 2;
        }
        return generateFallbackTrajectory(actual_waypoints, start_vel, start_acc, start_yaw,
                                          end_vel, end_yaw, max_vel, 2.0, traj_dt,
                                          traj_pos, traj_yaw, traj_time, out_traj);
    }
    
    Eigen::Matrix3Xd inPs(3, N - 1);
    for (int i = 0; i < N - 1; ++i) {
        inPs.col(i) = waypoints[i + 1];
    }
    
    Eigen::VectorXd ts(N);
    double estimated_vel = max_vel * 0.6;
    for (int i = 0; i < N; ++i) {
        Vector3d start_pt = (i == 0) ? waypoints[0] : waypoints[i];
        Vector3d end_pt = (i == N - 1) ? waypoints.back() : waypoints[i + 1];
        double dist = (end_pt - start_pt).norm();
        ts(i) = std::max(dist / estimated_vel, 0.1);
    }
    
    Eigen::Matrix3d headPVA, tailPVA;
    headPVA.col(0) = waypoints[0];
    headPVA.col(1) = start_vel;
    headPVA.col(2) = start_acc;
    
    tailPVA.col(0) = waypoints.back();
    tailPVA.col(1) = end_vel;
    tailPVA.col(2) = Vector3d::Zero();
    
    minco_solver_.setConditions(headPVA, tailPVA, N);
    minco_solver_.setParameters(inPs, ts);
    
    Trajectory<5> traj;
    minco_solver_.getTrajectory(traj);
    
    if (traj.getPieceNum() == 0) {
        std::cout << "[MincoWrapper] Fallback MINCO failed" << std::endl;
        return false;
    }
    
    // Also output continuous trajectory from fallback path
    if (out_traj) *out_traj = traj;
    
    sampleTrajectory(traj, traj_dt, start_yaw, end_yaw, traj_pos, traj_yaw, traj_time);
    
    std::cout << "[MincoWrapper] Fallback success: " << traj_pos.size() << " points, "
              << "duration=" << traj_time.back() << "s" << std::endl;
    
    return !traj_pos.empty();
}

}  // namespace forex_nav
