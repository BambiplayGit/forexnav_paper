#ifndef _FAST_EXPLORATION_FSM_H_
#define _FAST_EXPLORATION_FSM_H_

#include <Eigen/Eigen>

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <std_msgs/Empty.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Twist.h>
#include <std_srvs/SetBool.h>
#include <exploration_manager/expl_data.h>

#include <algorithm>
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <thread>

using Eigen::Vector3d;
using std::vector;
using std::shared_ptr;
using std::unique_ptr;
using std::string;

namespace fast_planner {
class FastPlannerManager;
class FastExplorationManager;
class PlanningVisualization;
struct FSMParam;
struct FSMData;

enum EXPL_STATE { INIT, WAIT_TRIGGER, ALIGN_YAW, CHECK_GOAL, PLAN_TRAJ, PUB_TRAJ, EXEC_TRAJ, FINISH };

class FastExplorationFSM {
private:
  /* planning utils */
  shared_ptr<FastPlannerManager> planner_manager_;
  shared_ptr<FastExplorationManager> expl_manager_;
  shared_ptr<PlanningVisualization> visualization_;

  shared_ptr<FSMParam> fp_;
  shared_ptr<FSMData> fd_;
  EXPL_STATE state_;

  bool classic_;

  /* ROS utils */
  ros::NodeHandle node_;
  ros::Timer exec_timer_, safety_timer_, vis_timer_, frontier_timer_;
  ros::Subscriber trigger_sub_, odom_sub_, goal_sub_;
  ros::Publisher replan_pub_, new_pub_, bspline_pub_, historical_viewpoints_pub_, filtered_viewpoints_pub_, ellipse_pub_, cmd_vel_pub_, goal_check_circle_pub_;
  ros::ServiceClient controller_enable_client_;

  /* helper functions */
  int callExplorationPlanner();
  void transitState(EXPL_STATE new_state, string pos_call);

  /* ROS functions */
  void FSMCallback(const ros::TimerEvent& e);
  void safetyCallback(const ros::TimerEvent& e);
  void frontierCallback(const ros::TimerEvent& e);
  void triggerCallback(const nav_msgs::PathConstPtr& msg);
  void odometryCallback(const nav_msgs::OdometryConstPtr& msg);
  void visualize();
  void clearVisMarker();
  void goalCallback(const geometry_msgs::PoseStampedConstPtr& msg);
  void publishHistoricalViewpoints(const shared_ptr<ExplorationData>& ed);
  void publishFilteredViewpoints(const shared_ptr<ExplorationData>& ed);
  void publishEllipse(const shared_ptr<ExplorationData>& ed);
  
  /* Yaw control functions */
  void alignYawState();
  double computeYawToGoal();
  bool enableController(bool enable);
  void publishCmdVel(double linear_x, double linear_y, double angular_z);
  
  /* Goal check functions */
  void checkGoalState();
  void publishGoalCheckCircle();

public:
  FastExplorationFSM(/* args */) {
  }
  ~FastExplorationFSM() {
  }

  void init(ros::NodeHandle& nh);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace fast_planner

#endif