#ifndef _ACKERMANN_KINODYNAMIC_ASTAR_H
#define _ACKERMANN_KINODYNAMIC_ASTAR_H

#include <path_searching/kinodynamic_astar.h>  // 复用基础结构
#include <plan_env/edt_environment.h>
#include <plan_env/sdf_map.h>
#include <plan_env/raycast.h>
#include <ompl/base/spaces/ReedsSheppStateSpace.h>
#include <ompl/base/ScopedState.h>
#include <ros/ros.h>

namespace fast_planner {

// 复用 KinodynamicAstar 的 PathNode，但适配 Ackermann 模型
// 状态空间：6D (x, y, z, vx, vy, vz)
// 对于 Ackermann：state.head(3) = (x, y, fixed_z), state.tail(3) = (vx, vy, 0)
// 其中 vx, vy 由 yaw 和速度决定

class AckermannKinoAstar {
private:
  /* ---------- main data structure (复用 kinodynamic_astar) ---------- */
  vector<PathNodePtr> path_node_pool_;
  int use_node_num_, iter_num_;
  NodeHashTable expanded_nodes_;
  std::priority_queue<PathNodePtr, std::vector<PathNodePtr>, NodeComparator> open_set_;
  std::vector<PathNodePtr> path_nodes_;

  /* ---------- record data ---------- */
  Eigen::Vector3d start_vel_, end_vel_, start_acc_;
  Eigen::Matrix<double, 6, 6> phi_;  // state transit matrix (复用)
  EDTEnvironment::Ptr edt_environment_;
  bool is_shot_succ_ = false;
  Eigen::MatrixXd coef_shot_;
  double t_shot_;
  bool has_path_ = false;

  /* ---------- parameter ---------- */
  double max_tau_, init_max_tau_;
  double max_vel_, max_acc_;
  double max_steer_;  // Ackermann 特有
  double max_cur_;    // 最大曲率
  double w_time_, horizon_, lambda_heu_;
  int allocate_num_, check_num_;
  double tie_breaker_;
  bool optimistic_;

  /* map */
  double resolution_, inv_resolution_, time_resolution_, inv_time_resolution_;
  Eigen::Vector3d origin_, map_size_3d_;
  double time_origin_;
  double fixed_height_2d_;  // 固定高度（2D规划）

  /* vehicle parameters */
  double car_width_, car_length_, car_wheelbase_;
  std::vector<Eigen::Vector2d> car_vertex_;  // 车辆顶点（用于碰撞检测）

  /* shot trajectory (Reeds-Shepp) */
  ompl::base::StateSpacePtr shotptr;
  double checkl_;

  /* helper */
  Eigen::Vector3i posToIndex(Eigen::Vector3d pt);
  int timeToIndex(double time);
  void retrievePath(PathNodePtr end_node);
  
  /* Ackermann specific */
  void stateTransit(Eigen::Matrix<double, 6, 1>& state0, Eigen::Matrix<double, 6, 1>& state1,
                    Eigen::Vector2d ctrl_input, double tau);
  void checkCollisionUsingPosAndYaw(const Eigen::Vector3d& state, bool& res);
  bool isInMap2d(const Eigen::Vector2d& pos);
  int getVoxelState2d(const Eigen::Vector2d& pos);
  
  /* shot trajectory */
  vector<double> cubic(double a, double b, double c, double d);
  vector<double> quartic(double a, double b, double c, double d, double e);
  bool computeShotTraj(Eigen::VectorXd state1, Eigen::VectorXd state2, double time_to_goal);
  double estimateHeuristic(Eigen::VectorXd x1, Eigen::VectorXd x2, double& optimal_time);

public:
  AckermannKinoAstar(){};
  ~AckermannKinoAstar();

  enum { REACH_HORIZON = 1, REACH_END = 2, NO_PATH = 3, NEAR_END = 4 };

  /* main API (兼容原 AckermannKinoAstar 接口) */
  ros::NodeHandle nh_;  // 用于参数读取
  void setParam(ros::NodeHandle& nh);
  void init();
  void reset();
  
  // 兼容原接口：search(Vector4d start_state, Vector2d init_ctrl, Vector4d end_state, bool use3d)
  // Vector4d: (x, y, yaw, v)
  int search(Eigen::Vector4d start_state, Eigen::Vector2d init_ctrl,
             Eigen::Vector4d end_state, bool use3d = false);
  
  // 也支持标准接口（用于兼容 KinodynamicAstar）
  int search(Eigen::Vector3d start_pt, Eigen::Vector3d start_vel, Eigen::Vector3d start_acc,
             Eigen::Vector3d end_pt, Eigen::Vector3d end_vel, bool init, bool dynamic = false,
             double time_start = -1.0);

  void setEnvironment(const EDTEnvironment::Ptr& env);

  std::vector<Eigen::Vector3d> getKinoTraj(double delta_t);

  void getSamples(double& ts, vector<Eigen::Vector3d>& point_set,
                  vector<Eigen::Vector3d>& start_end_derivatives);

  std::vector<PathNodePtr> getVisitedNodes();

  typedef shared_ptr<AckermannKinoAstar> Ptr;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace fast_planner

#endif

