#ifndef FOREX_NAV_MINCO_WRAPPER_H_
#define FOREX_NAV_MINCO_WRAPPER_H_

#include "forex_nav/minco/minco.hpp"
#include "forex_nav/minco/trajectory.hpp"
#include "forex_nav/minco/sfc_generator.hpp"
#include "forex_nav/minco/gcopter_2d.hpp"
#include <Eigen/Core>
#include <vector>
#include <nav_msgs/OccupancyGrid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace forex_nav {

using Vector3d = Eigen::Vector3d;

// Wrapper class for MINCO trajectory generation with GCOPTER optimization
class MincoWrapper {
public:
  MincoWrapper();
  ~MincoWrapper();

  // Set occupancy grid map for SFC generation
  void setMap(const nav_msgs::OccupancyGrid::ConstPtr& map);

  // 设置3D占据点云用于3D走廊生成
  void setOccCloud3D(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, double voxel_res);

  // Generate MINCO trajectory from waypoints with full GCOPTER optimization
  bool generateTrajectory(
      const std::vector<Vector3d>& waypoints,
      const Vector3d& start_vel,
      const Vector3d& start_acc,
      double start_yaw,
      const Vector3d& end_vel,
      double end_yaw,
      double max_vel,
      double max_acc,
      double max_yawdot,
      double traj_dt,
      std::vector<Vector3d>& traj_pos,
      std::vector<double>& traj_yaw,
      std::vector<double>& traj_time);

  // Get the last generated corridors for visualization
  const std::vector<AABB2D>& getLastCorridors() const { return last_corridors_; }

  // 获取最近生成的3D走廊
  const std::vector<AABB2D>& getLastCorridors3D() const { return last_corridors_3d_; }
  
  // 生成3D走廊(仅用于可视化, 不影响轨迹优化)
  bool generateCorridors3D(const std::vector<Eigen::Vector3d>& waypoints);

  // 设置规划高度
  void setFixedHeight(double height);

private:
  SFCGenerator sfc_generator_;
  GCopter2D gcopter_;
  minco::MINCO_S3NU minco_solver_;  // Fallback solver
  bool has_map_ = false;
  bool has_cloud_3d_ = false;
  std::vector<AABB2D> last_corridors_;     // 2D corridors for visualization
  std::vector<AABB2D> last_corridors_3d_;  // 3D corridors for visualization
  
  // Sample trajectory at fixed dt
  void sampleTrajectory(
      const Trajectory<5>& traj,
      double dt,
      double start_yaw,
      double end_yaw,
      std::vector<Vector3d>& traj_pos,
      std::vector<double>& traj_yaw,
      std::vector<double>& traj_time);
  
  // Fallback: generate trajectory without optimization (when no map available)
  bool generateFallbackTrajectory(
      const std::vector<Vector3d>& waypoints,
      const Vector3d& start_vel,
      const Vector3d& start_acc,
      double start_yaw,
      const Vector3d& end_vel,
      double end_yaw,
      double max_vel,
      double max_acc,
      double traj_dt,
      std::vector<Vector3d>& traj_pos,
      std::vector<double>& traj_yaw,
      std::vector<double>& traj_time);
};

}  // namespace forex_nav

#endif  // FOREX_NAV_MINCO_WRAPPER_H_
