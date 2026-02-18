#ifndef FOREX_NAV_MANAGER_H_
#define FOREX_NAV_MANAGER_H_

#include <memory>
#include <vector>
#include <array>
#include <Eigen/Core>
#include <nav_msgs/OccupancyGrid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "forex_nav/forex_nav_data.h"
#include "forex_nav/fuzzy_astar_2d.h"
#include "forex_nav/minco/trajectory.hpp"

namespace forex_nav {

class Astar2D;
class MincoWrapper;

class ForexNavManager {
public:
  ForexNavManager();
  ~ForexNavManager();

  void initialize();
  
  // Plan navigation motion: select viewpoint and plan trajectory
  // out_traj: if non-null, receives the continuous Trajectory<5> object for real-time p/v/a evaluation
  int planNavMotion(
    const Vector3d& pos, const Vector3d& vel, double yaw,
    const Vector3d& goal_pos,
    const std::vector<Viewpoint>& viewpoints,
    std::vector<Vector3d>& path,
    std::vector<double>& yaws,
    std::vector<double>& times,
    std::vector<Vector3d>& astar_path,
    Trajectory<5>* out_traj = nullptr);

  // Select best viewpoint using fuzzy A* (simplified version)
  int selectBestViewpoint(
    const Vector3d& pos, const Vector3d& vel, double yaw,
    const Vector3d& goal_pos,
    const std::vector<Viewpoint>& viewpoints,
    Vector3d& selected_pos,
    double& selected_yaw,
    double& selected_v_limit);

  // 2D A* path planning
  bool planPath(const Vector3d& start, const Vector3d& end, std::vector<Vector3d>& path);
  
  // Check if a point is in free space (not in obstacle) - uses occupancy grid
  bool isPointInFreeSpace(const Vector3d& pos);
  
  // Check if a point collides with obstacles (inverse of isPointInFreeSpace) - uses inflated occupancy grid
  bool checkPointCollision(const Vector3d& pos);
  
  // Check if goal is reachable using occupancy grid (more reliable than A* with fallback)
  bool isGoalReachable(const Vector3d& start, const Vector3d& goal);
  
  // Shorten path by removing redundant waypoints
  void shortenPath(std::vector<Vector3d>& path);
  
  // Set map for A* planning
  void setMap(const nav_msgs::OccupancyGrid::ConstPtr& map);

  // Set whether inpainted map is available for predicted cost (fuzzy A*); when false, pred cost = 0
  void setUseInpaintForPred(bool use) { use_inpaint_for_pred_ = use; }

  // Set inpainted map for FuzzyAstar2D predicted cost computation
  void setInpaintedMap(const nav_msgs::OccupancyGrid::ConstPtr& map);

  // 设置3D占据点云
  void setOccCloud3D(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, double voxel_res);

  // Generate MINCO trajectory from path using real MINCO
  void generateMINCOTrajectory(
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
    Trajectory<5>* out_traj = nullptr);

  // Set parameters
  void setNavParam(const NavParam& param) { nav_param_ = param; }
  NavParam& getNavParam() { return nav_param_; }
  
  // Get last generated SFC corridors for visualization
  // Returns vector of [x_min, x_max, y_min, y_max, z_min, z_max] for each corridor
  std::vector<std::array<double, 6>> getLastCorridors() const;

  // 获取3D走廊
  std::vector<std::array<double, 6>> getLastCorridors3D() const;

  // 生成3D走廊(基于已有路径)
  bool generateCorridors3D(const std::vector<Vector3d>& path);

  // Get last candidate paths for visualization (paths to top N viewpoints by cost)
  const std::vector<std::vector<Vector3d>>& getLastCandidatePaths() const;

  // Get ranked viewpoint positions (sorted by mixed cost, best first) for viewpoint graph visualization
  const std::vector<Vector3d>& getRankedViewpoints() const;

private:
  NavParam nav_param_;
  std::shared_ptr<Astar2D> astar_2d_;
  std::shared_ptr<MincoWrapper> minco_wrapper_;
  std::vector<std::vector<Vector3d>> candidate_paths_;  // Paths to candidate viewpoints for vis
  std::vector<Vector3d> ranked_viewpoints_;  // Viewpoint positions sorted by mixed cost (best first)
  
  bool use_inpaint_for_pred_;  // When false, skip A* for pred cost and set to 0 (no inpainted map)
  
  // FuzzyAstar2D for predicted cost on inpainted map
  FuzzyAstar2D fuzzy_astar_;
  nav_msgs::OccupancyGrid::ConstPtr inpaint_map_;
  
  // Homotopy consistency state
  bool has_last_goal_;
  Vector3d last_best_goal_;
  
  // Helper structures
  struct CurveResult {
    double cost;
    double v_limit;
  };
  
  // Helper functions
  CurveResult calculateCurveCost(
    const Vector3d& curr_p, double curr_y, const Vector3d& curr_v,
    const Vector3d& target_p, double target_y,
    double v_max, double a_max, double yv_max);
  
  double calculateDistanceCost(const Vector3d& pos, const Vector3d& goal);
};

}  // namespace forex_nav

#endif  // FOREX_NAV_MANAGER_H_
