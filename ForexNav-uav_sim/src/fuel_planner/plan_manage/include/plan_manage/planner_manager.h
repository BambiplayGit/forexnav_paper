#ifndef _PLANNER_MANAGER_H_
#define _PLANNER_MANAGER_H_

#include <bspline_opt/bspline_optimizer.h>
#include <bspline/non_uniform_bspline.h>

#include <path_searching/astar2.h>
#include <path_searching/kinodynamic_astar.h>
#include <path_searching/ackermann_kino_astar.h>
#include <path_searching/topo_prm.h>

#include <plan_env/edt_environment.h>

#include <active_perception/frontier_finder.h>
#include <active_perception/heading_planner.h>

#include <plan_manage/plan_container.hpp>

#include <ros/ros.h>

// MINCO backend includes
#include "gcopter/firi.hpp"
#include "gcopter/flatness.hpp"
#include "gcopter/gcopter.hpp"
#include "gcopter/sfc_gen.hpp"
#include "gcopter/trajectory.hpp"
#include "gcopter/geo_utils.hpp"
#include "misc/visualizer.hpp"

namespace fast_planner {
// Fast Planner Manager
// Key algorithms of mapping and planning are called

// GCOPTER Configuration for MINCO backend
struct GcopterConfig {
  double dilateRadiusSoft, dilateRadiusHard;
  double maxVelMag;
  double maxVelY;  // Maximum velocity in Y direction (lateral). -1 means no limit
  double maxBdrMag;
  double maxTiltAngle;
  double minThrust;
  double maxThrust;
  double vehicleMass;
  double gravAcc;
  double horizDrag;
  double vertDrag;
  double parasDrag;
  double speedEps;
  double weightT;
  double WeightSafeT;
  std::vector<double> chiVec;
  double smoothingEps;
  int integralIntervs;
  double relCostTol;
  double corridor_size;

  void init(const ros::NodeHandle &nh_priv) {
    nh_priv.param("gcopter/DilateRadiusSoft", dilateRadiusSoft, 0.1);
    nh_priv.param("gcopter/DilateRadiusHard", dilateRadiusHard, 0.05);
    nh_priv.param("gcopter/MaxVelMag", maxVelMag, 2.0);
    nh_priv.param("gcopter/MaxVelY", maxVelY, -1.0);  // Default: no limit on Y velocity
    nh_priv.param("gcopter/maxBdrMag", maxBdrMag, 8.0);
    nh_priv.param("gcopter/MaxTiltAngle", maxTiltAngle, 0.785);
    nh_priv.param("gcopter/MinThrust", minThrust, 3.0);
    nh_priv.param("gcopter/MaxThrust", maxThrust, 15.0);
    nh_priv.param("gcopter/VehicleMass", vehicleMass, 1.0);
    nh_priv.param("gcopter/GravAcc", gravAcc, 9.8);
    nh_priv.param("gcopter/HorizDrag", horizDrag, 0.0);
    nh_priv.param("gcopter/VertDrag", vertDrag, 0.0);
    nh_priv.param("gcopter/ParasDrag", parasDrag, 0.0);
    nh_priv.param("gcopter/SpeedEps", speedEps, 1.0);
    nh_priv.param("gcopter/WeightT", weightT, 100.0);
    nh_priv.param("gcopter/WeightSafeT", WeightSafeT, 10.0);
    nh_priv.param("gcopter/ChiVec", chiVec, std::vector<double>{1.0, 1.0, 1.0, 1.0, 1.0});
    nh_priv.param("gcopter/SmoothingEps", smoothingEps, 1e-6);
    nh_priv.param("gcopter/IntegralIntervs", integralIntervs, 10);
    nh_priv.param("gcopter/RelCostTol", relCostTol, 1e-4);
    nh_priv.param("gcopter/MaxCorridorSize", corridor_size, 3.0);
  }
};

class FastPlannerManager {
  // SECTION stable
public:
  FastPlannerManager();
  ~FastPlannerManager();

  /* main planning interface */
  bool kinodynamicReplan(const Eigen::Vector3d& start_pt, const Eigen::Vector3d& start_vel,
                         const Eigen::Vector3d& start_acc, const Eigen::Vector3d& end_pt,
                         const Eigen::Vector3d& end_vel, const double& time_lb = -1);
  void planExploreTraj(const vector<Eigen::Vector3d>& tour, const Eigen::Vector3d& cur_vel,
                       const Eigen::Vector3d& cur_acc, const double& time_lb = -1);
  bool planGlobalTraj(const Eigen::Vector3d& start_pos);
  bool topoReplan(bool collide);

  void planYaw(const Eigen::Vector3d& start_yaw);
  void planYawExplore(const Eigen::Vector3d& start_yaw, const double& end_yaw, bool lookfwd,
                      const double& relax_time);

  void initPlanModules(ros::NodeHandle& nh);
  void setGlobalWaypoints(vector<Eigen::Vector3d>& waypoints);

  bool checkTrajCollision(double& distance);
  void calcNextYaw(const double& last_yaw, double& yaw);

  PlanParameters pp_;
  LocalTrajData local_data_;
  GlobalTrajData global_data_;
  MidPlanData plan_data_;
  EDTEnvironment::Ptr edt_environment_;
  unique_ptr<Astar> path_finder_;
  unique_ptr<TopologyPRM> topo_prm_;

private:
  /* main planning algorithms & modules */
  shared_ptr<SDFMap> sdf_map_;

  unique_ptr<KinodynamicAstar> kino_path_finder_;
  unique_ptr<AckermannKinoAstar> ackermann_path_finder_;
  vector<BsplineOptimizer::Ptr> bspline_optimizers_;

  void updateTrajInfo();

  // topology guided optimization

  void findCollisionRange(vector<Eigen::Vector3d>& colli_start, vector<Eigen::Vector3d>& colli_end,
                          vector<Eigen::Vector3d>& start_pts, vector<Eigen::Vector3d>& end_pts);

  void optimizeTopoBspline(double start_t, double duration, vector<Eigen::Vector3d> guide_path,
                           int traj_id);
  Eigen::MatrixXd paramLocalTraj(double start_t, double& dt, double& duration);
  Eigen::MatrixXd reparamLocalTraj(const double& start_t, const double& duration, const double& dt);

  void selectBestTraj(NonUniformBspline& traj);
  void refineTraj(NonUniformBspline& best_traj);
  void reparamBspline(NonUniformBspline& bspline, double ratio, Eigen::MatrixXd& ctrl_pts, double& dt,
                      double& time_inc);

  // Heading planning

  // !SECTION stable

  // SECTION developing

public:
  typedef shared_ptr<FastPlannerManager> Ptr;

  void planYawActMap(const Eigen::Vector3d& start_yaw);
  void test();
  void searchFrontier(const Eigen::Vector3d& p);

private:
  unique_ptr<FrontierFinder> frontier_finder_;
  unique_ptr<HeadingPlanner> heading_planner_;
  unique_ptr<VisibilityUtil> visib_util_;

  // MINCO backend
  unique_ptr<GcopterConfig> gcopter_config_;
  unique_ptr<Visualizer> gcopter_viz_;
  bool use_minco_backend_;
  bool debug_compare_mode_;  // Debug mode: run both backends and publish for comparison
  
  // Debug mode publishers (using Marker for better visualization)
  ros::Publisher debug_astar_path_pub_;
  ros::Publisher debug_bspline_traj_pub_;
  ros::Publisher debug_minco_traj_pub_;
  ros::NodeHandle debug_nh_;
  
  // Rectangle collision detection parameters (for legged robots)
  double collision_rect_length_;  // Length of collision detection rectangle
  double collision_rect_width_;   // Width of collision detection rectangle
  double collision_check_interval_; // Time interval for collision checking along trajectory
  ros::Publisher collision_rect_viz_pub_; // Publisher for visualizing collision rectangles
  
  // MINCO helper functions
  void planExploreTrajBspline(const vector<Eigen::Vector3d>& tour, const Eigen::Vector3d& cur_vel,
                              const Eigen::Vector3d& cur_acc, const double& time_lb);
  void planExploreTrajMINCO(const vector<Eigen::Vector3d>& tour, const Eigen::Vector3d& cur_vel,
                            const Eigen::Vector3d& cur_acc, const double& time_lb);
  void getPointCloudFromESDF(const Eigen::Vector3d& min_bd, const Eigen::Vector3d& max_bd,
                             std::vector<Eigen::Vector3d>& surf_points);
  void convertMINCOToBspline(const Trajectory<7>& minco_traj, NonUniformBspline& bspline_traj);
  void updateTrajInfoMINCO(const Trajectory<7>& minco_traj);
  
  // Debug comparison functions
  void publishAStarPath(const vector<Eigen::Vector3d>& path);
  void publishBsplineTraj(NonUniformBspline& traj, const std::string& label);
  void publishMINCOTraj(Trajectory<7>& traj, const std::string& label);
  
  // Rectangle collision detection helper functions
  bool checkRectCollision(const Eigen::Vector3d& center, double yaw, double length, double width);
  void visualizeCollisionRect(const Eigen::Vector3d& center, double yaw, double length, double width, int id);

  // Benchmark method, local exploration
public:
  bool localExplore(Eigen::Vector3d start_pt, Eigen::Vector3d start_vel, Eigen::Vector3d start_acc,
                    Eigen::Vector3d end_pt);

  // !SECTION
};
}  // namespace fast_planner

#endif