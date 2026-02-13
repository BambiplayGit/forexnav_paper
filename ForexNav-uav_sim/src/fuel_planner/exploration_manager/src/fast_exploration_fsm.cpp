
#include <plan_manage/planner_manager.h>
#include <exploration_manager/fast_exploration_manager.h>
#include <traj_utils/planning_visualization.h>

#include <exploration_manager/fast_exploration_fsm.h>
#include <exploration_manager/expl_data.h>
#include <plan_env/edt_environment.h>
#include <plan_env/sdf_map.h>
#include <geometry_msgs/Twist.h>
#include <std_srvs/SetBool.h>
#include <cmath>

using Eigen::Vector4d;

namespace fast_planner {
void FastExplorationFSM::init(ros::NodeHandle& nh) {
  fp_.reset(new FSMParam);
  fd_.reset(new FSMData);

  /*  Fsm param  */
  nh.param("fsm/thresh_replan1", fp_->replan_thresh1_, -1.0);
  nh.param("fsm/thresh_replan2", fp_->replan_thresh2_, -1.0);
  nh.param("fsm/thresh_replan3", fp_->replan_thresh3_, -1.0);
  nh.param("fsm/replan_time", fp_->replan_time_, -1.0);
  
  /* Yaw control params */
  nh.param("fsm/align_yaw_angular_vel", fp_->align_yaw_angular_vel_, 0.5);  // Fixed angular velocity for alignment (rad/s)
  nh.param("fsm/align_yaw_tolerance", fp_->align_yaw_tolerance_, 0.1);  // Yaw alignment tolerance (rad)
  nh.param("fsm/enable_align_yaw", fp_->enable_align_yaw_, true);
  
  /* Goal check params */
  nh.param("fsm/check_goal_distance", fp_->check_goal_distance_, 2.0);  // Distance threshold to enter CHECK_GOAL state (m)

  /* Initialize main modules */
  expl_manager_.reset(new FastExplorationManager);
  expl_manager_->initialize(nh);
  visualization_.reset(new PlanningVisualization(nh));

  planner_manager_ = expl_manager_->planner_manager_;
  state_ = EXPL_STATE::INIT;
  fd_->have_odom_ = false;
  fd_->state_str_ = { "INIT", "WAIT_TRIGGER", "ALIGN_YAW", "CHECK_GOAL", "PLAN_TRAJ", "PUB_TRAJ", "EXEC_TRAJ", "FINISH" };
  fd_->static_state_ = true;
  fd_->trigger_ = false;
  
  // Initialize yaw control state
  fd_->target_yaw_ = 0.0;
  
  // Initialize goal check state
  fd_->goal_check_entered_ = false;
  fd_->goal_check_aligned_ = false;
  fd_->last_dist_to_goal_ = -1.0;

  /* Ros sub, pub and timer */
  exec_timer_ = nh.createTimer(ros::Duration(0.01), &FastExplorationFSM::FSMCallback, this);
  safety_timer_ = nh.createTimer(ros::Duration(0.05), &FastExplorationFSM::safetyCallback, this);
  frontier_timer_ = nh.createTimer(ros::Duration(0.1), &FastExplorationFSM::frontierCallback, this);

  goal_sub_ = nh.subscribe("/move_base_simple/goal", 1, &FastExplorationFSM::goalCallback, this);

  odom_sub_ = nh.subscribe("/odom_world", 1, &FastExplorationFSM::odometryCallback, this);

  replan_pub_ = nh.advertise<std_msgs::Empty>("/planning/replan", 10);
  new_pub_ = nh.advertise<std_msgs::Empty>("/planning/new", 10);
  bspline_pub_ = nh.advertise<bspline::Bspline>("/planning/bspline", 10);
  historical_viewpoints_pub_ = nh.advertise<visualization_msgs::Marker>("/planning_vis/historical_viewpoints", 100);
  filtered_viewpoints_pub_ = nh.advertise<visualization_msgs::Marker>("/planning_vis/filtered_viewpoints", 100);
  ellipse_pub_ = nh.advertise<visualization_msgs::Marker>("/planning_vis/ellipse", 100);
  cmd_vel_pub_ = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
  goal_check_circle_pub_ = nh.advertise<visualization_msgs::Marker>("/planning_vis/goal_check_circle", 100);
  
  // Service client to enable/disable position_to_velocity_controller
  controller_enable_client_ = nh.serviceClient<std_srvs::SetBool>("/position_to_velocity_controller/set_enabled");
}

void FastExplorationFSM::FSMCallback(const ros::TimerEvent& e) {
  ROS_INFO_STREAM_THROTTLE(1.0, "[FSM]: state: " << fd_->state_str_[int(state_)]);

  // Publish goal check circle visualization (if goal is set)
  if (fd_->trigger_ || state_ != WAIT_TRIGGER) {
    publishGoalCheckCircle();
  }

  switch (state_) {
    case INIT: {
      // Wait for odometry ready
      if (!fd_->have_odom_) {
        ROS_WARN_THROTTLE(1.0, "no odom.");
        return;
      }
      // Go to wait trigger when odom is ok
      transitState(WAIT_TRIGGER, "FSM");
      break;
    }

    case WAIT_TRIGGER: {
      // Do nothing but wait for trigger
      ROS_WARN_THROTTLE(1.0, "wait for trigger.");
      break;
    }

    case ALIGN_YAW: {
      alignYawState();
      break;
    }

    case CHECK_GOAL: {
      checkGoalState();
      break;
    }

    case FINISH: {
      // For point-to-point planning: print planning summary and return to WAIT_TRIGGER
      LocalTrajData* info = &planner_manager_->local_data_;
      double dist_to_goal = (fd_->odom_pos_ - fd_->final_pos_).head<2>().norm();
      double straight_dist = (fd_->final_pos_ - fd_->start_pt_).head<2>().norm();
      double t_cur = (ros::Time::now() - info->start_time_).toSec();
      
      ROS_INFO("\033[32m========================================\033[0m");
      ROS_INFO("\033[32m[FSM] Point-to-Point Planning Summary\033[0m");
      ROS_INFO("\033[32m========================================\033[0m");
      ROS_INFO("  Start Position:     (%.3f, %.3f, %.3f) m", 
               fd_->start_pt_[0], fd_->start_pt_[1], fd_->start_pt_[2]);
      ROS_INFO("  Goal Position:      (%.3f, %.3f, %.3f) m", 
               fd_->final_pos_[0], fd_->final_pos_[1], fd_->final_pos_[2]);
      ROS_INFO("  Current Position:   (%.3f, %.3f, %.3f) m", 
               fd_->odom_pos_[0], fd_->odom_pos_[1], fd_->odom_pos_[2]);
      ROS_INFO("  Straight Distance:  %.3f m", straight_dist);
      ROS_INFO("  Final Distance:     %.3f m", dist_to_goal);
      ROS_INFO("  Trajectory Duration: %.3f s", info->duration_);
      ROS_INFO("  Actual Time:        %.3f s", t_cur);
      ROS_INFO("\033[32m========================================\033[0m");
      
      // Reset flags and return to WAIT_TRIGGER state for next goal
      fd_->static_state_ = true;
      fd_->trigger_ = false;
      fd_->last_planned_ = false;
      fd_->goal_check_entered_ = false;  // Reset goal check flag
      fd_->goal_check_aligned_ = false;  // Reset alignment flag
      fd_->last_dist_to_goal_ = -1.0;  // Reset distance tracking
      clearVisMarker();
      
      ROS_INFO("[FSM] Planning completed. Returning to WAIT_TRIGGER state for next goal.");
      transitState(WAIT_TRIGGER, "FSM");
      break;
    }

    case PLAN_TRAJ: {
           if (fd_->static_state_) {
            // Plan from static state (hover)
            fd_->start_pt_ = fd_->odom_pos_;
            fd_->start_vel_ = fd_->odom_vel_;
            fd_->start_acc_.setZero();

            fd_->start_yaw_(0) = fd_->odom_yaw_;
            fd_->start_yaw_(1) = fd_->start_yaw_(2) = 0.0;
          } else {
            // Replan from non-static state, starting from 'replan_time' seconds later
            LocalTrajData* info = &planner_manager_->local_data_;
            double t_r = (ros::Time::now() - info->start_time_).toSec() + fp_->replan_time_;
            if (t_r > info->duration_) 
            {fd_->static_state_ = true; return;}
            fd_->start_pt_ = info->position_traj_.evaluateDeBoorT(t_r);
            fd_->start_vel_ = info->velocity_traj_.evaluateDeBoorT(t_r);
            fd_->start_acc_ = info->acceleration_traj_.evaluateDeBoorT(t_r);
            fd_->start_yaw_(0) = fd_->odom_yaw_;
            fd_->start_yaw_(1) = fd_->start_yaw_(2) = 0.0;
          }

          // Check if already at goal before planning
          double dist_to_goal = (fd_->start_pt_ - fd_->final_pos_).head<2>().norm();
          const double goal_reached_threshold = 0.5;  
          
          if (dist_to_goal < goal_reached_threshold) {
            ROS_INFO("\033[32m[FSM] Already at goal before planning! Distance: %.4f m < %.4f m. Skipping to FINISH.\033[0m", 
                     dist_to_goal, goal_reached_threshold);
            transitState(FINISH, "FSM");
            fd_->static_state_ = true;
            fd_->trigger_ = false;
            clearVisMarker();
            break;
          }
          
          // Inform traj_server the replanning
          replan_pub_.publish(std_msgs::Empty());
          ROS_INFO("\033[35m[FSM] Before Calling Explore! start_pt=(%.2f,%.2f,%.2f), final_pos=(%.2f,%.2f,%.2f), dist=%.4f\033[0m",
                   fd_->start_pt_[0], fd_->start_pt_[1], fd_->start_pt_[2],
                   fd_->final_pos_[0], fd_->final_pos_[1], fd_->final_pos_[2], dist_to_goal);
          int res = callExplorationPlanner();
          if (res == SUCCEED) {
            ROS_INFO("[FSM] Planning succeeded, transitioning to PUB_TRAJ");
            transitState(PUB_TRAJ, "FSM");
            clearVisMarker();
          } else if (res == FAIL) {
            // Still in PLAN_TRAJ state, keep replanning
            ROS_WARN("[FSM] Planning failed, will retry. Result code: %d", res);
            fd_->static_state_ = true;
          } else {
            ROS_ERROR("[FSM] Unknown planning result: %d", res);
            fd_->static_state_ = true;
          }
          break;
        }

    case PUB_TRAJ: {
      double dt = (ros::Time::now() - fd_->newest_traj_.start_time).toSec();
      if (dt > 0) {
        bspline_pub_.publish(fd_->newest_traj_);
        fd_->static_state_ = false;
        transitState(EXEC_TRAJ, "FSM");

        thread vis_thread(&FastExplorationFSM::visualize, this);
        vis_thread.detach();
      }
      break;
    }

    case EXEC_TRAJ: {
      LocalTrajData* info = &planner_manager_->local_data_;
      double t_cur = (ros::Time::now() - info->start_time_).toSec();

      // Check if reached the final goal
      double dist_to_goal = (fd_->odom_pos_ - fd_->final_pos_).head<2>().norm();
      const double goal_reached_threshold = 0.5;
      
      // Check if approaching goal for the first time (enter CHECK_GOAL state)
      if (!fd_->goal_check_entered_ && dist_to_goal < fp_->check_goal_distance_) {
        ROS_INFO("\033[33m[FSM] Approaching goal for first time (%.3f m < %.3f m). Entering CHECK_GOAL state.\033[0m",
                 dist_to_goal, fp_->check_goal_distance_);
        fd_->goal_check_entered_ = true;
        fd_->last_dist_to_goal_ = dist_to_goal;
        transitState(CHECK_GOAL, "EXEC_TRAJ");
        return;
      }
      
      // Check if moving away from goal (reset flags so we can enter CHECK_GOAL again)
      if (fd_->last_dist_to_goal_ > 0 && dist_to_goal > fd_->last_dist_to_goal_ + 0.2) {  // Hysteresis: 20cm buffer
        if (fd_->goal_check_entered_) {
          ROS_INFO("\033[33m[FSM] Moving away from goal (%.3f m -> %.3f m). Resetting goal check flags.\033[0m",
                   fd_->last_dist_to_goal_, dist_to_goal);
          fd_->goal_check_entered_ = false;
          fd_->goal_check_aligned_ = false;  // Also reset alignment flag
        }
      }
      
      // Update last distance (only when getting closer or if uninitialized)
      if (fd_->last_dist_to_goal_ < 0 || dist_to_goal < fd_->last_dist_to_goal_) {
        fd_->last_dist_to_goal_ = dist_to_goal;
      }  
      
      // If this is the last planned trajectory, use more aggressive stopping
      if (fd_->last_planned_) {
        // For last planned traj, stop if close to goal (more lenient threshold)
        const double last_planned_threshold = 0.5;  // 50cm for last planned traj
        if (dist_to_goal < last_planned_threshold) {
          ROS_INFO("\033[32m[FSM] Last planned traj, goal reached (%.3f m). Stopping.\033[0m", dist_to_goal);
          transitState(FINISH, "FSM");
          fd_->static_state_ = true;
          fd_->trigger_ = false;
          clearVisMarker();
          return;
        }
        // For last planned traj, skip all frontier-related replanning
        // Only check if trajectory is finished
        double time_to_end = info->duration_ - t_cur;
        if (time_to_end < fp_->replan_thresh1_) {
          // Trajectory finished, check if we're at goal
          if (dist_to_goal < last_planned_threshold) {
            ROS_INFO("\033[32m[FSM] Last planned traj finished, goal reached (%.3f m). Stopping.\033[0m", dist_to_goal);
            transitState(FINISH, "FSM");
            fd_->static_state_ = true;
            fd_->trigger_ = false;
            clearVisMarker();
            return;
          }
          // If not at goal yet but traj finished, might need one more small adjustment
          // But don't replan aggressively
          ROS_WARN("[FSM] Last planned traj finished but not at goal (%.3f m). Will stop anyway.", dist_to_goal);
          transitState(FINISH, "FSM");
          fd_->static_state_ = true;
          fd_->trigger_ = false;
          clearVisMarker();
          return;
        }
        // For last planned traj, don't check frontier coverage or periodic replan
        break;
      }
      
      // Normal trajectory execution (not last planned)
      if (dist_to_goal < goal_reached_threshold) {
        ROS_INFO("\033[32m[FSM] Goal reached! Distance: %.3f m. Stopping.\033[0m", dist_to_goal);
        transitState(FINISH, "FSM");
        fd_->static_state_ = true;
        fd_->trigger_ = false;
        clearVisMarker();
        return;
      }

      // Replan if traj is almost fully executed
      double time_to_end = info->duration_ - t_cur;
      if (time_to_end < fp_->replan_thresh1_) {
        // Before replanning, check if we're already at goal
        if (dist_to_goal < goal_reached_threshold) {
          ROS_INFO("\033[32m[FSM] Traj finished and goal reached (%.3f m). Stopping.\033[0m", dist_to_goal);
          transitState(FINISH, "FSM");
          fd_->static_state_ = true;
          fd_->trigger_ = false;
          clearVisMarker();
          return;
        }
        transitState(PLAN_TRAJ, "FSM");
        ROS_WARN("Replan: traj fully executed=================================");
        return;
      }
      // Frontier coverage check removed - not needed for point-to-point navigation
      // if (t_cur > fp_->replan_thresh2_ && expl_manager_->frontier_finder_->isFrontierCovered()) {
      //   ...
      // }
      // Replan after some time
      if (t_cur > fp_->replan_thresh3_ && !classic_) {
        // Before replanning, check if we're already at goal
        if (dist_to_goal < goal_reached_threshold) {
          ROS_INFO("\033[32m[FSM] Periodic replan check: goal reached (%.3f m). Stopping.\033[0m", dist_to_goal);
          transitState(FINISH, "FSM");
          fd_->static_state_ = true;
          fd_->trigger_ = false;
          clearVisMarker();
          return;
        }
        transitState(PLAN_TRAJ, "FSM");
        ROS_WARN("Replan: periodic call=======================================");
      }
      break;
    }
  }
}

int FastExplorationFSM::callExplorationPlanner() {
  ros::Time time_r = ros::Time::now() + ros::Duration(fp_->replan_time_);
  int res = expl_manager_->planForexMotion(fd_->start_pt_, fd_->start_vel_, fd_->start_acc_,
                                             fd_->start_yaw_, fd_->final_pos_);
  classic_ = false;

  // int res = expl_manager_->classicFrontier(fd_->start_pt_, fd_->start_yaw_[0]);
  // classic_ = true;

  // int res = expl_manager_->rapidFrontier(fd_->start_pt_, fd_->start_vel_, fd_->start_yaw_[0],
  // classic_);

  if (res == SUCCEED) {
    auto info = &planner_manager_->local_data_;
    info->start_time_ = (ros::Time::now() - time_r).toSec() > 0 ? ros::Time::now() : time_r;

    bspline::Bspline bspline;
    bspline.order = planner_manager_->pp_.bspline_degree_;
    bspline.start_time = info->start_time_;
    bspline.traj_id = info->traj_id_;
    Eigen::MatrixXd pos_pts = info->position_traj_.getControlPoint();
    for (int i = 0; i < pos_pts.rows(); ++i) {
      geometry_msgs::Point pt;
      pt.x = pos_pts(i, 0);
      pt.y = pos_pts(i, 1);
      pt.z = pos_pts(i, 2);
      bspline.pos_pts.push_back(pt);
    }
    Eigen::VectorXd knots = info->position_traj_.getKnot();
    for (int i = 0; i < knots.rows(); ++i) {
      bspline.knots.push_back(knots(i));
    }
    Eigen::MatrixXd yaw_pts = info->yaw_traj_.getControlPoint();
    for (int i = 0; i < yaw_pts.rows(); ++i) {
      double yaw = yaw_pts(i, 0);
      bspline.yaw_pts.push_back(yaw);
    }
    bspline.yaw_dt = info->yaw_traj_.getKnotSpan();
    fd_->newest_traj_ = bspline;
  }
  return res;
}

void FastExplorationFSM::visualize() {
  auto info = &planner_manager_->local_data_;
  auto plan_data = &planner_manager_->plan_data_;
  auto ed_ptr = expl_manager_->ed_;

  // Draw updated box
  // Vector3d bmin, bmax;
  // planner_manager_->edt_environment_->sdf_map_->getUpdatedBox(bmin, bmax);
  // visualization_->drawBox((bmin + bmax) / 2.0, bmax - bmin, Vector4d(0, 1, 0, 0.3), "updated_box", 0,
  // 4);

  // Frontier visualization moved to frontierCallback() for continuous update
  // Only clean up old markers from previous visualize() calls to avoid duplicates
  // Note: frontierCallback() uses "frontier" namespace, while visualize() used "frontier_2d" namespace
  // So we need to clean up both namespaces on first call
  static bool first_call = true;
  static int last_ftr_num_viz = 0;  // Track markers drawn by visualize() (different namespace)
  
  if (first_call) {
    // On first call, clean up any existing frontier markers from previous sessions
    // Clean up both "frontier_2d" (from old visualize() calls) and "frontier" (from frontierCallback)
    if (expl_manager_->frontier_finder_->is2DMode()) {
      // Clean up old "frontier_2d" namespace markers
      for (int i = 0; i < 100; ++i) {
        visualization_->drawSpheres({}, 0.1, Eigen::Vector4d(0, 0, 0, 1), "frontier_2d", i * 3, 4);
        visualization_->drawLines({}, 0.03, Eigen::Vector4d(0, 0, 0, 1), "frontier_ring", i * 3 + 1, 4);
        visualization_->drawLines({}, {}, 0.04, Eigen::Vector4d(0, 0, 0, 1), "frontier_pillar", i * 3 + 2, 4);
      }
      // Also clean up "frontier" namespace (used by frontierCallback in 2D mode, but not actively used)
      for (int i = 0; i < 100; ++i) {
        visualization_->drawCubes({}, 0.1, Vector4d(0, 0, 0, 1), "frontier", i, 4);
      }
    } else {
      // Clean up old "frontier" namespace markers
      for (int i = 0; i < 100; ++i) {
        visualization_->drawCubes({}, 0.1, Vector4d(0, 0, 0, 1), "frontier", i, 4);
      }
    }
    first_call = false;
    last_ftr_num_viz = 0;
  }
  
  // Clean up old markers that were drawn in previous visualize() calls (if any)
  // This ensures old markers from visualize() are cleared when visualize() is called again
  if (last_ftr_num_viz > 0) {
    if (expl_manager_->frontier_finder_->is2DMode()) {
      for (int i = 0; i < last_ftr_num_viz; ++i) {
        visualization_->drawSpheres({}, 0.1, Eigen::Vector4d(0, 0, 0, 1), "frontier_2d", i * 3, 4);
        visualization_->drawLines({}, 0.03, Eigen::Vector4d(0, 0, 0, 1), "frontier_ring", i * 3 + 1, 4);
        visualization_->drawLines({}, {}, 0.04, Eigen::Vector4d(0, 0, 0, 1), "frontier_pillar", i * 3 + 2, 4);
      }
    } else {
      for (int i = 0; i < last_ftr_num_viz; ++i) {
        visualization_->drawCubes({}, 0.1, Vector4d(0, 0, 0, 1), "frontier", i, 4);
      }
    }
  }
  // Don't update last_ftr_num_viz since we're not drawing new markers here
  // It will be reset to 0 after cleanup
  last_ftr_num_viz = 0;
  // for (int i = 0; i < ed_ptr->dead_frontiers_.size(); ++i)
  //   visualization_->drawCubes(ed_ptr->dead_frontiers_[i], 0.1, Vector4d(0, 0, 0, 0.5), "dead_frontier",
  //                             i, 4);
  // for (int i = ed_ptr->dead_frontiers_.size(); i < 5; ++i)
  //   visualization_->drawCubes({}, 0.1, Vector4d(0, 0, 0, 0.5), "dead_frontier", i, 4);

  // Draw global top viewpoints info
  // visualization_->drawSpheres(ed_ptr->points_, 0.2, Vector4d(0, 0.5, 0, 1), "points", 0, 6);
  // visualization_->drawLines(ed_ptr->global_tour_, 0.07, Vector4d(0, 0.5, 0, 1), "global_tour", 0, 6);
  // visualization_->drawLines(ed_ptr->points_, ed_ptr->views_, 0.05, Vector4d(0, 1, 0.5, 1), "view", 0, 6);
  // visualization_->drawLines(ed_ptr->points_, ed_ptr->averages_, 0.03, Vector4d(1, 0, 0, 1),
  // "point-average", 0, 6);

  // Draw local refined viewpoints info
  // visualization_->drawSpheres(ed_ptr->refined_points_, 0.2, Vector4d(0, 0, 1, 1), "refined_pts", 0, 6);
  // visualization_->drawLines(ed_ptr->refined_points_, ed_ptr->refined_views_, 0.05,
  //                           Vector4d(0.5, 0, 1, 1), "refined_view", 0, 6);
  // visualization_->drawLines(ed_ptr->refined_tour_, 0.07, Vector4d(0, 0, 1, 1), "refined_tour", 0, 6);
  // visualization_->drawLines(ed_ptr->refined_views1_, ed_ptr->refined_views2_, 0.04, Vector4d(0, 0, 0,
  // 1),
  //                           "refined_view", 0, 6);
  // visualization_->drawLines(ed_ptr->refined_points_, ed_ptr->unrefined_points_, 0.05, Vector4d(1, 1,
  // 0, 1),
  //                           "refine_pair", 0, 6);
  // for (int i = 0; i < ed_ptr->n_points_.size(); ++i)
  //   visualization_->drawSpheres(ed_ptr->n_points_[i], 0.1,
  //                               visualization_->getColor(double(ed_ptr->refined_ids_[i]) /
  //                               ed_ptr->frontiers_.size()),
  //                               "n_points", i, 6);
  // for (int i = ed_ptr->n_points_.size(); i < 15; ++i)
  //   visualization_->drawSpheres({}, 0.1, Vector4d(0, 0, 0, 1), "n_points", i, 6);

  // Draw trajectory
  // visualization_->drawSpheres({ ed_ptr->next_goal_ }, 0.3, Vector4d(0, 1, 1, 1), "next_goal", 0, 6);
  visualization_->drawBspline(info->position_traj_, 0.1, Vector4d(1.0, 0.0, 0.0, 1), false, 0.15,
                              Vector4d(1, 1, 0, 1));
    visualization_->drawFrontierGraph(fd_->start_pt_, fd_->final_pos_, expl_manager_->frontier_finder_->frontier_nodes_,
                                    "frontier_graph", 4);
visualization_->drawFuzzyAstarPaths(
    expl_manager_->fuzzyastar_path_world_,
    "fuzzyastar_path",
    0.08,
    Eigen::Vector4d(0.0, 1.0, 0.0, 1.0),  // 绿色 RGBA
    9527,
    4  // 用 frontier_pub_ ("/planning_vis/frontier") 也可以
);
  // visualization_->drawSpheres(plan_data->kino_path_, 0.1, Vector4d(1, 0, 1, 1), "kino_path", 0, 0);
  // visualization_->drawLines(ed_ptr->path_next_goal_, 0.05, Vector4d(0, 1, 1, 1), "next_goal", 1, 6);
}

void FastExplorationFSM::clearVisMarker() {
  // visualization_->drawSpheres({}, 0.2, Vector4d(0, 0.5, 0, 1), "points", 0, 6);
  // visualization_->drawLines({}, 0.07, Vector4d(0, 0.5, 0, 1), "global_tour", 0, 6);
  // visualization_->drawSpheres({}, 0.2, Vector4d(0, 0, 1, 1), "refined_pts", 0, 6);
  // visualization_->drawLines({}, {}, 0.05, Vector4d(0.5, 0, 1, 1), "refined_view", 0, 6);
  // visualization_->drawLines({}, 0.07, Vector4d(0, 0, 1, 1), "refined_tour", 0, 6);
  // visualization_->drawSpheres({}, 0.1, Vector4d(0, 0, 1, 1), "B-Spline", 0, 0);

  // visualization_->drawLines({}, {}, 0.03, Vector4d(1, 0, 0, 1), "current_pose", 0, 6);
}

void FastExplorationFSM::frontierCallback(const ros::TimerEvent& e) {
  // Removed delay limit to allow continuous update during motion
  // Update both frontier and viewpoints in all states
  
  auto ft = expl_manager_->frontier_finder_;
  auto ed = expl_manager_->ed_;
  
  // Update frontiers
  ft->searchFrontiers();
  ft->computeFrontiersToVisit();
  ft->updateFrontierCostMatrix();

  ft->getFrontiers(ed->frontiers_);
  ft->getFrontierCenters();
  ft->getFrontierBoxes(ed->frontier_boxes_);

  // Update viewpoints (best viewing points) based on current position
  static size_t last_viewpoint_count = 0;
  static bool has_new_viewpoint = false;
  
  if (!ed->frontiers_.empty() && fd_->have_odom_) {
    // Get current position for viewpoint calculation
    Vector3d current_pos = fd_->odom_pos_;
    double current_yaw = fd_->odom_yaw_;
    
    // Update viewpoints for all frontiers (current viewpoints for visualization)
    ft->getTopViewpointsInfo(current_pos, ed->points_, ed->yaws_, ed->averages_);
    
    // Add new viewpoints to historical table (with deduplication) - only if use_historical_viewpoints is enabled
    if (expl_manager_->ep_->use_historical_viewpoints_) {
      const double dedup_threshold = 0.5;  // 50cm threshold for deduplication
      size_t new_viewpoints_added = 0;
      
      for (size_t i = 0; i < ed->points_.size(); ++i) {
        bool is_duplicate = false;
        // Check if this viewpoint already exists in historical table
        for (size_t j = 0; j < ed->historical_viewpoints_.size(); ++j) {
          if ((ed->points_[i] - ed->historical_viewpoints_[j]).norm() < dedup_threshold) {
            is_duplicate = true;
            break;
          }
        }
        
        // Add to historical table if not duplicate
        if (!is_duplicate) {
          ed->historical_viewpoints_.push_back(ed->points_[i]);
          ed->historical_yaws_.push_back(ed->yaws_[i]);
          ed->historical_averages_.push_back(ed->averages_[i]);
          new_viewpoints_added++;
        }
      }
      
      // Check if new viewpoints are discovered
      if (new_viewpoints_added > 0) {
        has_new_viewpoint = true;
        ROS_INFO("[FSM] Added %zu new viewpoints to historical table (total: %zu)", 
                 new_viewpoints_added, ed->historical_viewpoints_.size());
      }
    } else {
      // In current viewpoints mode, check if new viewpoints are discovered (for replanning trigger)
      if (ed->points_.size() > last_viewpoint_count) {
        has_new_viewpoint = true;
      }
    }
    
    if (ed->points_.size() > last_viewpoint_count) {
      ROS_INFO("[FSM] Current viewpoints count: %zu -> %zu", last_viewpoint_count, ed->points_.size());
    }
    last_viewpoint_count = ed->points_.size();
    
    // Update views (viewpoints + direction vectors) for current viewpoints
    ed->views_.clear();
    for (int i = 0; i < ed->points_.size(); ++i) {
      ed->views_.push_back(
          ed->points_[i] + 2.0 * Vector3d(cos(ed->yaws_[i]), sin(ed->yaws_[i]), 0));
    }
    
    // Publish historical viewpoints
    publishHistoricalViewpoints(ed);
    
    // Publish filtered viewpoints and ellipse (if available)
    publishFilteredViewpoints(ed);
    publishEllipse(ed);
  }

  // Check if final goal is now reachable (when in EXEC_TRAJ state)
  // Priority: If goal is reachable, go directly to goal; otherwise check for new viewpoints
  static bool goal_was_reachable = false;
  bool goal_now_reachable = false;
  
  if (state_ == EXPL_STATE::EXEC_TRAJ && fd_->have_odom_ && !fd_->last_planned_) {
    // Check if final goal is reachable (using A* path finder)
    planner_manager_->path_finder_->reset();
    bool reachable = planner_manager_->path_finder_->search(fd_->odom_pos_, fd_->final_pos_) == Astar::REACH_END;
    double dist_to_goal = (fd_->odom_pos_ - fd_->final_pos_).head<2>().norm();
    
    if (reachable && dist_to_goal > 0.5) {  // Goal is reachable and not too close (avoid frequent replanning)
      goal_now_reachable = true;
      if (!goal_was_reachable) {
        ROS_INFO("\033[33m[FSM] Final goal is now reachable! Distance: %.3f m. Triggering replan to go directly to goal.\033[0m", 
                 dist_to_goal);
        // Reset viewpoint flag to avoid triggering replan for viewpoints when goal is reachable
        has_new_viewpoint = false;
        // Trigger replanning - planForexMotion will automatically use final_pos_ since it's reachable
        transitState(PLAN_TRAJ, "frontierCallback");
        goal_was_reachable = true;
        return;  // Exit early to trigger replanning
      }
    } else {
      if (goal_was_reachable && !reachable) {
        ROS_INFO("[FSM] Final goal is no longer directly reachable. Will use viewpoints.");
      }
      goal_was_reachable = false;
    }
  }

  // Check if new viewpoint discovered and trigger replanning (when in EXEC_TRAJ state)
  // Only trigger if goal is not reachable (checked above)
  if (has_new_viewpoint && state_ == EXPL_STATE::EXEC_TRAJ && !fd_->last_planned_ && !goal_was_reachable) {
    ROS_INFO("\033[33m[FSM] New viewpoint discovered during execution (count: %zu). Triggering replanning.\033[0m", 
             ed->points_.size());
    has_new_viewpoint = false;  // Reset flag
    transitState(PLAN_TRAJ, "frontierCallback");
    return;  // Exit early to trigger replanning
  }

  // Draw frontier and bounding box (simple version - original callback style)
  for (int i = 0; i < ed->frontiers_.size(); ++i) {
    visualization_->drawCubes(ed->frontiers_[i], 0.1,
                              visualization_->getColor(double(i) / ed->frontiers_.size(), 0.4),
                              "frontier", i, 4);
  }
  for (int i = ed->frontiers_.size(); i < 50; ++i) {
    visualization_->drawCubes({}, 0.1, Vector4d(0, 0, 0, 1), "frontier", i, 4);
  }

  // Draw viewpoints (best viewing points) - pub_id = 6 -> /planning_vis/viewpoints
  if (!ed->points_.empty()) {
    // Draw viewpoint positions as spheres (green)
    visualization_->drawSpheres(ed->points_, 0.2, Vector4d(0, 0.5, 0, 1), "viewpoints", 0, 6);
    
    // Draw viewpoint directions (viewpoints -> views) as lines
    if (ed->points_.size() == ed->views_.size()) {
      visualization_->drawLines(ed->points_, ed->views_, 0.05, Vector4d(0, 1, 0.5, 1), "viewpoint_dirs", 1, 6);
    }
  } else {
    // Clear old markers if no viewpoints
    visualization_->drawSpheres({}, 0.2, Vector4d(0, 0, 0, 1), "viewpoints", 0, 6);
    visualization_->drawLines({}, {}, 0.05, Vector4d(0, 0, 0, 1), "viewpoint_dirs", 1, 6);
  }
}

void FastExplorationFSM::goalCallback(const geometry_msgs::PoseStampedConstPtr& msg) {
  if (msg->pose.position.z < -0.1) return;
  fd_->trigger_ = true;
  cout << "New Goal Triggered!" << endl;
  fd_->final_pos_ = Vector3d(msg->pose.position.x, msg->pose.position.y, 1.0);
  fd_->last_planned_ = false;
  fd_->goal_check_entered_ = false;  // Reset goal check flag when new goal is set
  fd_->goal_check_aligned_ = false;  // Reset alignment flag when new goal is set
  fd_->last_dist_to_goal_ = -1.0;  // Reset distance tracking
  
  // Compute target yaw (direction to goal)
  fd_->target_yaw_ = computeYawToGoal();
  
  // If align yaw is enabled, go to ALIGN_YAW state first
  if (fp_->enable_align_yaw_) {
    // Disable position_to_velocity_controller to avoid conflict
    enableController(false);
    transitState(ALIGN_YAW, "goalCallback");
  } else {
    transitState(PLAN_TRAJ, "goalCallback");
  }
}

void FastExplorationFSM::safetyCallback(const ros::TimerEvent& e) {
  if (state_ == EXPL_STATE::EXEC_TRAJ) {
    // Check safety and trigger replan if necessary
    double dist;
    bool safe = planner_manager_->checkTrajCollision(dist);
    if (!safe) {
      ROS_WARN("Replan: collision detected==================================");
      transitState(PLAN_TRAJ, "safetyCallback");
    }
  }
}

void FastExplorationFSM::odometryCallback(const nav_msgs::OdometryConstPtr& msg) {
  fd_->odom_pos_(0) = msg->pose.pose.position.x;
  fd_->odom_pos_(1) = msg->pose.pose.position.y;
  fd_->odom_pos_(2) = msg->pose.pose.position.z;

  fd_->odom_vel_(0) = msg->twist.twist.linear.x;
  fd_->odom_vel_(1) = msg->twist.twist.linear.y;
  fd_->odom_vel_(2) = msg->twist.twist.linear.z;

  fd_->odom_orient_.w() = msg->pose.pose.orientation.w;
  fd_->odom_orient_.x() = msg->pose.pose.orientation.x;
  fd_->odom_orient_.y() = msg->pose.pose.orientation.y;
  fd_->odom_orient_.z() = msg->pose.pose.orientation.z;

  Eigen::Vector3d rot_x = fd_->odom_orient_.toRotationMatrix().block<3, 1>(0, 0);
  fd_->odom_yaw_ = atan2(rot_x(1), rot_x(0));

  fd_->have_odom_ = true;
}

void FastExplorationFSM::transitState(EXPL_STATE new_state, string pos_call) {
  int pre_s = int(state_);
  state_ = new_state;
  cout << "[" + pos_call + "]: from " + fd_->state_str_[pre_s] + " to " + fd_->state_str_[int(new_state)]
       << endl;
}

void FastExplorationFSM::publishHistoricalViewpoints(const shared_ptr<ExplorationData>& ed) {
  if (ed->historical_viewpoints_.empty()) return;
  
  visualization_msgs::Marker marker;
  marker.header.frame_id = "world";
  marker.header.stamp = ros::Time::now();
  marker.ns = "historical_viewpoints";
  marker.id = 0;
  marker.type = visualization_msgs::Marker::SPHERE_LIST;
  marker.action = visualization_msgs::Marker::ADD;
  
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;
  
  marker.scale.x = 0.25;  // Larger than current viewpoints to distinguish
  marker.scale.y = 0.25;
  marker.scale.z = 0.25;
  
  marker.color.r = 1.0;  // Red color for historical viewpoints
  marker.color.g = 0.5;
  marker.color.b = 0.0;
  marker.color.a = 0.8;
  
  // Add all historical viewpoints
  for (size_t i = 0; i < ed->historical_viewpoints_.size(); ++i) {
    geometry_msgs::Point pt;
    pt.x = ed->historical_viewpoints_[i].x();
    pt.y = ed->historical_viewpoints_[i].y();
    pt.z = ed->historical_viewpoints_[i].z();
    marker.points.push_back(pt);
  }
  
  historical_viewpoints_pub_.publish(marker);
}

void FastExplorationFSM::publishFilteredViewpoints(const shared_ptr<ExplorationData>& ed) {
  if (ed->filtered_viewpoints_.empty()) {
    // Clear old markers if no filtered viewpoints
    visualization_msgs::Marker marker;
    marker.header.frame_id = "world";
    marker.header.stamp = ros::Time::now();
    marker.ns = "filtered_viewpoints";
    marker.id = 0;
    marker.action = visualization_msgs::Marker::DELETE;
    filtered_viewpoints_pub_.publish(marker);
    return;
  }
  
  visualization_msgs::Marker marker;
  marker.header.frame_id = "world";
  marker.header.stamp = ros::Time::now();
  marker.ns = "filtered_viewpoints";
  marker.id = 0;
  marker.type = visualization_msgs::Marker::SPHERE_LIST;
  marker.action = visualization_msgs::Marker::ADD;
  
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;
  
  marker.scale.x = 0.3;  // Larger than historical to distinguish
  marker.scale.y = 0.3;
  marker.scale.z = 0.3;
  
  marker.color.r = 0.0;  // Green color for filtered viewpoints
  marker.color.g = 1.0;
  marker.color.b = 0.0;
  marker.color.a = 1.0;
  
  // Add all filtered viewpoints
  for (size_t i = 0; i < ed->filtered_viewpoints_.size(); ++i) {
    geometry_msgs::Point pt;
    pt.x = ed->filtered_viewpoints_[i].x();
    pt.y = ed->filtered_viewpoints_[i].y();
    pt.z = ed->filtered_viewpoints_[i].z();
    marker.points.push_back(pt);
  }
  
  filtered_viewpoints_pub_.publish(marker);
}

void FastExplorationFSM::publishEllipse(const shared_ptr<ExplorationData>& ed) {
  if (!ed->ellipse_valid_) {
    // Clear old markers if ellipse is not valid
    visualization_msgs::Marker marker;
    marker.header.frame_id = "world";
    marker.header.stamp = ros::Time::now();
    marker.ns = "ellipse";
    marker.id = 0;
    marker.action = visualization_msgs::Marker::DELETE;
    ellipse_pub_.publish(marker);
    return;
  }
  
  visualization_msgs::Marker marker;
  marker.header.frame_id = "world";
  marker.header.stamp = ros::Time::now();
  marker.ns = "ellipse";
  marker.id = 0;
  marker.type = visualization_msgs::Marker::LINE_STRIP;
  marker.action = visualization_msgs::Marker::ADD;
  
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;
  
  marker.scale.x = 0.1;  // Line width
  
  marker.color.r = 0.0;  // Cyan color for ellipse
  marker.color.g = 1.0;
  marker.color.b = 1.0;
  marker.color.a = 0.8;
  
  // Generate ellipse points (in XY plane, at ellipse center z)
  const int num_points = 64;
  Vector3d minor_axis(-ed->ellipse_major_axis_.y(), ed->ellipse_major_axis_.x(), 0);
  
  for (int i = 0; i <= num_points; ++i) {
    double angle = 2.0 * M_PI * i / num_points;
    Vector3d pt = ed->ellipse_center_ + 
                  ed->ellipse_major_radius_ * cos(angle) * ed->ellipse_major_axis_ +
                  ed->ellipse_minor_radius_ * sin(angle) * minor_axis;
    
    geometry_msgs::Point p;
    p.x = pt.x();
    p.y = pt.y();
    p.z = pt.z();
    marker.points.push_back(p);
  }
  
  ellipse_pub_.publish(marker);
}

double FastExplorationFSM::computeYawToGoal() {
  // Compute yaw angle pointing from current position to goal
  Vector3d dir = fd_->final_pos_ - fd_->odom_pos_;
  return atan2(dir(1), dir(0));
}

bool FastExplorationFSM::enableController(bool enable) {
  std_srvs::SetBool srv;
  srv.request.data = enable;
  
  if (controller_enable_client_.call(srv)) {
    if (srv.response.success) {
      ROS_INFO("[FSM] Position controller %s", enable ? "enabled" : "disabled");
      return true;
    } else {
      ROS_WARN("[FSM] Failed to %s position controller: %s", 
               enable ? "enable" : "disable", srv.response.message.c_str());
      return false;
    }
  } else {
    ROS_WARN_THROTTLE(5.0, "[FSM] Service /position_to_velocity_controller/set_enabled not available");
    return false;
  }
}

void FastExplorationFSM::publishCmdVel(double linear_x, double linear_y, double angular_z) {
  geometry_msgs::Twist cmd_vel;
  cmd_vel.linear.x = linear_x;
  cmd_vel.linear.y = linear_y;
  cmd_vel.linear.z = 0.0;
  cmd_vel.angular.x = 0.0;
  cmd_vel.angular.y = 0.0;
  cmd_vel.angular.z = angular_z;
  cmd_vel_pub_.publish(cmd_vel);
}

void FastExplorationFSM::alignYawState() {
  if (!fd_->have_odom_) {
    ROS_WARN_THROTTLE(1.0, "[FSM] No odom in ALIGN_YAW state");
    return;
  }
  
  // Compute yaw error (normalize to [-pi, pi])
  double yaw_error = fd_->target_yaw_ - fd_->odom_yaw_;
  yaw_error = atan2(sin(yaw_error), cos(yaw_error));
  
  // Check if aligned
  if (fabs(yaw_error) < fp_->align_yaw_tolerance_) {
    ROS_INFO("[FSM] Yaw aligned! Error: %.3f rad (threshold: %.3f)", 
             fabs(yaw_error), fp_->align_yaw_tolerance_);
    
    // Stop rotation
    publishCmdVel(0.0, 0.0, 0.0);
    
    // Check if we came from CHECK_GOAL state (by checking if goal_check_entered_ is true but goal_check_aligned_ is false)
    if (fd_->goal_check_entered_ && !fd_->goal_check_aligned_) {
      // We came from CHECK_GOAL, mark as aligned and return to CHECK_GOAL
      fd_->goal_check_aligned_ = true;
      enableController(true);
      ROS_INFO("[FSM] Aligned to goal in CHECK_GOAL context. Returning to CHECK_GOAL state.");
      transitState(CHECK_GOAL, "alignYawState");
      return;
    }
    
    // Normal flow: re-enable controller and go to PLAN_TRAJ
    enableController(true);
    transitState(PLAN_TRAJ, "alignYawState");
    return;
  }
  
  // Use fixed angular velocity for alignment
  // Determine rotation direction based on yaw error
  double angular_vel = (yaw_error > 0) ? fp_->align_yaw_angular_vel_ : -fp_->align_yaw_angular_vel_;
  
  // Publish velocity command (only rotation, no translation)
  publishCmdVel(0.0, 0.0, angular_vel);
  
  ROS_INFO_THROTTLE(0.5, "[FSM] Aligning yaw: current=%.3f, target=%.3f, error=%.3f, vel=%.3f",
                    fd_->odom_yaw_, fd_->target_yaw_, yaw_error, angular_vel);
}

void FastExplorationFSM::checkGoalState() {
  if (!fd_->have_odom_) {
    ROS_WARN_THROTTLE(1.0, "[FSM] No odom in CHECK_GOAL state");
    return;
  }
  
  // Compute distance to goal
  double dist_to_goal = (fd_->odom_pos_ - fd_->final_pos_).head<2>().norm();
  
  // Update distance tracking
  if (fd_->last_dist_to_goal_ < 0 || dist_to_goal < fd_->last_dist_to_goal_) {
    fd_->last_dist_to_goal_ = dist_to_goal;
  }
  
  // Check if moved away from goal check zone (hysteresis: 30cm buffer)
  if (dist_to_goal > fp_->check_goal_distance_ + 0.3) {
    ROS_INFO("[FSM] Moved away from goal check zone (%.3f m > %.3f m). Resetting flags and returning to EXEC_TRAJ.",
             dist_to_goal, fp_->check_goal_distance_ + 0.3);
    fd_->goal_check_entered_ = false;  // Reset flag so we can enter again when approaching
    fd_->goal_check_aligned_ = false;  // Reset alignment flag
    fd_->last_dist_to_goal_ = dist_to_goal;
    transitState(EXEC_TRAJ, "checkGoalState");
    return;
  }
  
  // Check if goal reached (within final threshold)
  const double goal_reached_threshold = 0.5;
  if (dist_to_goal < goal_reached_threshold) {
    ROS_INFO("\033[32m[FSM] Goal reached in CHECK_GOAL state (%.3f m). Stopping.\033[0m", dist_to_goal);
    transitState(FINISH, "checkGoalState");
    fd_->static_state_ = true;
    fd_->trigger_ = false;
    fd_->goal_check_entered_ = false;
    fd_->goal_check_aligned_ = false;
    fd_->last_dist_to_goal_ = -1.0;
    clearVisMarker();
    return;
  }
  
  // First, align yaw towards goal if not yet aligned
  if (!fd_->goal_check_aligned_) {
    // Compute target yaw (direction to goal)
    fd_->target_yaw_ = computeYawToGoal();
    
    // Check current yaw error
    double yaw_error = fd_->target_yaw_ - fd_->odom_yaw_;
    yaw_error = atan2(sin(yaw_error), cos(yaw_error));
    
    // If already aligned, skip ALIGN_YAW
    if (fabs(yaw_error) < fp_->align_yaw_tolerance_) {
      ROS_INFO("[FSM] Already aligned to goal in CHECK_GOAL state. Error: %.3f rad", fabs(yaw_error));
      fd_->goal_check_aligned_ = true;
    } else {
      // Need to align yaw first - switch to ALIGN_YAW state
      ROS_INFO("[FSM] Aligning yaw to goal in CHECK_GOAL state. Current yaw: %.3f, target: %.3f, error: %.3f",
               fd_->odom_yaw_, fd_->target_yaw_, yaw_error);
      // Disable controller to avoid conflict during yaw alignment
      enableController(false);
      transitState(ALIGN_YAW, "checkGoalState");
      return;
    }
  }
  
  // Now that we're aligned, continue with normal CHECK_GOAL behavior
  // Continue in CHECK_GOAL state while trajectory executes
  // Check if trajectory needs replanning (similar to EXEC_TRAJ logic)
  LocalTrajData* info = &planner_manager_->local_data_;
  double t_cur = (ros::Time::now() - info->start_time_).toSec();
  double time_to_end = info->duration_ - t_cur;
  
  // Replan if trajectory is almost finished
  if (time_to_end < fp_->replan_thresh1_) {
    if (dist_to_goal < goal_reached_threshold) {
      ROS_INFO("\033[32m[FSM] Traj finished in CHECK_GOAL state, goal reached (%.3f m). Stopping.\033[0m", dist_to_goal);
      transitState(FINISH, "checkGoalState");
      fd_->static_state_ = true;
      fd_->trigger_ = false;
      fd_->goal_check_entered_ = false;
      fd_->goal_check_aligned_ = false;
      fd_->last_dist_to_goal_ = -1.0;
      clearVisMarker();
      return;
    }
    // Replan to continue towards goal
    transitState(PLAN_TRAJ, "checkGoalState");
    ROS_WARN("[FSM] Traj finished in CHECK_GOAL state, replanning...");
    return;
  }
  
  ROS_INFO_THROTTLE(1.0, "[FSM] In CHECK_GOAL state (aligned=%s): distance=%.3f m (threshold=%.3f m), remaining time=%.2f s",
                    fd_->goal_check_aligned_ ? "yes" : "no", dist_to_goal, fp_->check_goal_distance_, time_to_end);
}

void FastExplorationFSM::publishGoalCheckCircle() {
  if (!fd_->have_odom_) {
    return;
  }
  
  visualization_msgs::Marker marker;
  marker.header.frame_id = "world";
  marker.header.stamp = ros::Time::now();
  marker.ns = "goal_check_circle";
  marker.id = 0;
  marker.type = visualization_msgs::Marker::LINE_STRIP;
  marker.action = visualization_msgs::Marker::ADD;
  
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;
  
  marker.scale.x = 0.1;  // Line width
  
  // Color based on goal_check_entered flag
  if (fd_->goal_check_entered_) {
    marker.color.r = 0.0;  // Green when already entered
    marker.color.g = 1.0;
    marker.color.b = 0.0;
  } else {
    marker.color.r = 1.0;  // Red when not yet entered
    marker.color.g = 0.0;
    marker.color.b = 0.0;
  }
  marker.color.a = 0.8;
  
  // Generate circle points (in XY plane, at goal z height)
  const int num_points = 64;
  double radius = fp_->check_goal_distance_;
  double center_z = fd_->final_pos_[2];
  
  for (int i = 0; i <= num_points; ++i) {
    double angle = 2.0 * M_PI * i / num_points;
    geometry_msgs::Point p;
    p.x = fd_->final_pos_[0] + radius * cos(angle);
    p.y = fd_->final_pos_[1] + radius * sin(angle);
    p.z = center_z;
    marker.points.push_back(p);
  }
  
  goal_check_circle_pub_.publish(marker);
}

}  // namespace fast_planner
