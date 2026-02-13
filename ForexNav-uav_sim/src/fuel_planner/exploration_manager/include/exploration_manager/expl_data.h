#ifndef _EXPL_DATA_H_
#define _EXPL_DATA_H_

#include <Eigen/Eigen>
#include <vector>
#include <string>
#include <utility>
#include <bspline/Bspline.h>

using std::vector;
using std::string;
using std::pair;
using Eigen::Vector3d;

namespace fast_planner {
struct FSMData {
  // FSM data
  bool trigger_, have_odom_, static_state_, last_planned_;
  vector<string> state_str_;

  Eigen::Vector3d odom_pos_, odom_vel_;  // odometry state
    Eigen::Vector3d final_pos_;  // goal state
  Eigen::Quaterniond odom_orient_;
  double odom_yaw_;

  Eigen::Vector3d start_pt_, start_vel_, start_acc_, start_yaw_;  // start state
  vector<Eigen::Vector3d> start_poss;
  bspline::Bspline newest_traj_;
  
  // Yaw control state
  double target_yaw_;   // Target yaw to align to
  
  // Goal check state
  bool goal_check_entered_;  // Whether CHECK_GOAL state has been entered for current goal (reset to false if moved away from goal, reset to false in FINISH)
  bool goal_check_aligned_;  // Whether yaw has been aligned in CHECK_GOAL state
  double last_dist_to_goal_;  // Last distance to goal for detecting approach/retreat (-1 if uninitialized)
};

struct FSMParam {
  double replan_thresh1_;
  double replan_thresh2_;
  double replan_thresh3_;
  double replan_time_;  // second
  
  // Yaw control parameters
  double align_yaw_angular_vel_;  // Fixed angular velocity for yaw alignment (rad/s)
  double align_yaw_tolerance_;  // Yaw alignment tolerance (rad)
  bool enable_align_yaw_;  // Enable align yaw state
  
  // Goal check parameters
  double check_goal_distance_;  // Distance threshold to enter CHECK_GOAL state (m)
};

struct ExplorationData {
  vector<vector<Vector3d>> frontiers_;
  vector<vector<Vector3d>> dead_frontiers_;
  vector<pair<Vector3d, Vector3d>> frontier_boxes_;
  vector<Vector3d> points_;
  vector<Vector3d> averages_;
  vector<Vector3d> views_;
  vector<double> yaws_;
  vector<Vector3d> global_tour_;

  vector<int> refined_ids_;
  vector<vector<Vector3d>> n_points_;
  vector<Vector3d> unrefined_points_;
  vector<Vector3d> refined_points_;
  vector<Vector3d> refined_views_;  // points + dir(yaw)
  vector<Vector3d> refined_views1_, refined_views2_;
  vector<Vector3d> refined_tour_;

  Vector3d next_goal_;
  vector<Vector3d> path_next_goal_;

  // viewpoint planning
  // vector<Vector4d> views_;
  vector<Vector3d> views_vis1_, views_vis2_;
  vector<Vector3d> centers_, scales_;
  
  // Historical viewpoints table (accumulated over time)
  // All discovered viewpoints are stored here for stable planning
  vector<Vector3d> historical_viewpoints_;
  vector<double> historical_yaws_;
  vector<Vector3d> historical_averages_;  // For compatibility with averages
  
  // Filtered viewpoints for planning (after clustering and ellipse filtering)
  vector<Vector3d> filtered_viewpoints_;
  vector<double> filtered_yaws_;
  vector<Vector3d> filtered_averages_;
  
  // Ellipse parameters for filtering (for visualization)
  Vector3d ellipse_center_;  // Center of ellipse (midpoint between current pos and goal)
  Vector3d ellipse_major_axis_;  // Major axis direction (from current to goal)
  double ellipse_major_radius_;  // Half-length of major axis
  double ellipse_minor_radius_;  // Half-length of minor axis
  bool ellipse_valid_;  // Whether ellipse filtering is active
};

struct ExplorationParam {
  // params
  bool refine_local_;
  int refined_num_;
  double refined_radius_;
  int top_view_num_;
  double max_decay_;
  string tsp_dir_;  // resource dir of tsp solver
  double relax_time_;
  bool use_fuzzy_astar_; // use predicted map and fuzzy astar as heuristic for path cost evaluation
  double w_dist_; // weight for distance cost in mixed cost
  double w_pred_; // weight for predicted cost in mixed cost
  
  // Viewpoint filtering parameters
  bool enable_ellipse_filter_;  // Enable ellipse filtering
  double ellipse_major_factor_;  // Factor for major axis (e.g., 1.2 means 20% longer than direct distance)
  double ellipse_minor_factor_;  // Factor for minor axis (e.g., 0.6 means 60% of major axis)
  int max_filtered_viewpoints_;  // Maximum number of viewpoints after filtering (0 = no limit)
  
  // Viewpoint selection mode
  bool use_historical_viewpoints_;  // If true, use historical_viewpoints_ for planning; if false, use current viewpoints (ed_->points_)
};

}  // namespace fast_planner

#endif