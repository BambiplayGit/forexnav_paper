#ifndef FOREX_NAV_FSM_H_
#define FOREX_NAV_FSM_H_

#include <memory>
#include <array>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <quadrotor_msgs/PositionCommand.h>
#include <geometry_msgs/PoseArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include "forex_nav/forex_nav_data.h"
#include "forex_nav/forex_nav_manager.h"
#include "forex_nav/minco/trajectory.hpp"

namespace forex_nav {

enum EXPL_STATE {
  INIT,
  WAIT_TRIGGER,
  PLAN_TRAJ,
  EXEC_TRAJ,
  FINISH
};

class ForexNavFSM {
public:
  ForexNavFSM(ros::NodeHandle& nh, ros::NodeHandle& pnh);
  ~ForexNavFSM();

  void init();

private:
  // ROS callbacks
  void odometryCallback(const nav_msgs::Odometry::ConstPtr& msg);
  void goalCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);
  void viewpointsCallback(const geometry_msgs::PoseArray::ConstPtr& msg);
  void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg);
  void inpaintMapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg);
  void occCloud3DCallback(const sensor_msgs::PointCloud2::ConstPtr& msg);
  
  // FSM callbacks
  void FSMCallback(const ros::TimerEvent&);
  void execTrajCallback(const ros::TimerEvent&);
  
  // Helper functions
  void transitState(EXPL_STATE new_state, const std::string& pos_call);
  int callPlanner();
  void publishTrajectory();
  void publishTrajectoryFallback();  // Fallback: discrete-point PoseStamped publishing
  void publishFinalPosition();  // Publish goal position to stop robot
  static double normalizeAngle(double a);
  void visualize();
  
  // ROS node handles
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  
  // Manager
  std::shared_ptr<ForexNavManager> manager_;
  
  // Data and parameters
  std::shared_ptr<FSMData> fd_;
  std::shared_ptr<FSMParam> fp_;
  EXPL_STATE state_;
  
  // Trajectory data (discrete points for visualization/debug)
  std::vector<Eigen::Vector3d> traj_positions_;
  std::vector<double> traj_yaws_;
  std::vector<double> traj_times_;
  size_t traj_index_;
  ros::Time traj_start_time_;
  ros::Time last_replan_time_;
  ros::Time last_map_update_time_;  // Throttle map updates for planning performance
  
  // Continuous MINCO trajectory for real-time p/v/a evaluation
  Trajectory<5> minco_traj_obj_;
  double minco_traj_duration_;
  double minco_start_yaw_;
  double minco_end_yaw_;
  bool has_minco_traj_;
  
  // Look-ahead collision check counter (runs every N exec callbacks to limit CPU)
  int collision_check_counter_;
  
  // Path data for visualization
  std::vector<Eigen::Vector3d> astar_path_;  // A* path (yellow)
  std::vector<Eigen::Vector3d> minco_traj_;  // MINCO trajectory (red)
  std::vector<std::array<double, 6>> corridors_;     // SFC 2D corridors
  std::vector<std::array<double, 6>> corridors_3d_;  // SFC 3D corridors
  
  // ROS interfaces
  ros::Subscriber odom_sub_;
  ros::Subscriber goal_sub_;
  ros::Subscriber viewpoints_sub_;
  ros::Subscriber map_sub_;
  ros::Subscriber inpaint_map_sub_;
  ros::Subscriber occ_cloud_3d_sub_;
  
  bool has_inpaint_map_;  // True when at least one inpainted map has been received
  nav_msgs::OccupancyGrid::ConstPtr inpaint_map_;  // 存储最新的 inpainted map
  
  ros::Publisher traj_pub_;
  ros::Publisher vis_pub_;
  ros::Publisher vis_3d_corridor_pub_;
  ros::Publisher vis_fuzzy_astar_pub_;
  ros::Publisher vis_viewpoint_graph_pub_;
  
  ros::Timer fsm_timer_;
  ros::Timer exec_timer_;
  
  // State strings for logging
  std::vector<std::string> state_str_;
};

}  // namespace forex_nav

#endif  // FOREX_NAV_FSM_H_
