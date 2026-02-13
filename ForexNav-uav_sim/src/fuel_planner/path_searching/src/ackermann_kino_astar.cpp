#include <path_searching/ackermann_kino_astar.h>
#include <sstream>
#include <plan_env/sdf_map.h>
#include <plan_env/raycast.h>

using namespace std;
using namespace Eigen;

namespace fast_planner {

AckermannKinoAstar::~AckermannKinoAstar() {
  for (int i = 0; i < allocate_num_; i++) {
    delete path_node_pool_[i];
  }
}

// ============================================================================
// 参数设置（复用 kinodynamic_astar 的参数，添加 Ackermann 特有参数）
// ============================================================================
void AckermannKinoAstar::setParam(ros::NodeHandle& nh) {
  nh_ = nh;  // 保存 node handle
  
  // 复用标准参数
  nh.param("search/max_tau", max_tau_, -1.0);
  nh.param("search/init_max_tau", init_max_tau_, -1.0);
  nh.param("search/max_vel", max_vel_, -1.0);
  nh.param("search/max_acc", max_acc_, -1.0);
  nh.param("search/w_time", w_time_, -1.0);
  nh.param("search/horizon", horizon_, -1.0);
  nh.param("search/resolution_astar", resolution_, -1.0);
  nh.param("search/time_resolution", time_resolution_, -1.0);
  nh.param("search/lambda_heu", lambda_heu_, -1.0);
  nh.param("search/allocate_num", allocate_num_, -1);
  nh.param("search/check_num", check_num_, -1);
  nh.param("search/optimistic", optimistic_, true);
  tie_breaker_ = 1.0 + 1.0 / 10000;

  // Ackermann 特有参数
  nh.param("search/max_cur", max_cur_, 0.3);
  nh.param("vehicle/car_max_steering_angle", max_steer_, 50.0);
  max_steer_ = max_steer_ * M_PI / 180.0;
  
  nh.param("vehicle/car_width", car_width_, 1.9);
  nh.param("vehicle/car_length", car_length_, 4.88);
  nh.param("vehicle/car_wheelbase", car_wheelbase_, 2.85);
  nh.param("search/fixed_height_2d", fixed_height_2d_, 0.5);
  nh.param("search/checkl", checkl_, 0.2);

  double vel_margin;
  nh.param("search/vel_margin", vel_margin, 0.0);
  max_vel_ += vel_margin;
}

void AckermannKinoAstar::init() {
  /* ---------- map params ---------- */
  this->inv_resolution_ = 1.0 / resolution_;
  inv_time_resolution_ = 1.0 / time_resolution_;
  edt_environment_->sdf_map_->getRegion(origin_, map_size_3d_);

  cout << "origin_: " << origin_.transpose() << endl;
  cout << "map size: " << map_size_3d_.transpose() << endl;

  /* ---------- pre-allocated node ---------- */
  path_node_pool_.resize(allocate_num_);
  for (int i = 0; i < allocate_num_; i++) {
    path_node_pool_[i] = new PathNode;
  }

  phi_ = Eigen::MatrixXd::Identity(6, 6);
  use_node_num_ = 0;
  iter_num_ = 0;

  /* ---------- vehicle vertices for collision checking ---------- */
  car_vertex_.clear();
  Eigen::Vector2d vertex;
  vertex << car_length_ / 2.0, car_width_ / 2.0;
  car_vertex_.push_back(vertex);
  vertex << car_length_ / 2.0, -car_width_ / 2.0;
  car_vertex_.push_back(vertex);
  vertex << -car_length_ / 2.0, -car_width_ / 2.0;
  car_vertex_.push_back(vertex);
  vertex << -car_length_ / 2.0, car_width_ / 2.0;
  car_vertex_.push_back(vertex);
  vertex << car_length_ / 2.0, car_width_ / 2.0;  // 闭合
  car_vertex_.push_back(vertex);

  /* ---------- Reeds-Shepp for shot trajectory ---------- */
  shotptr = std::make_shared<ompl::base::ReedsSheppStateSpace>(1.0 / max_cur_);
}

void AckermannKinoAstar::setEnvironment(const EDTEnvironment::Ptr& env) {
  this->edt_environment_ = env;
}

void AckermannKinoAstar::reset() {
  expanded_nodes_.clear();
  path_nodes_.clear();

  std::priority_queue<PathNodePtr, std::vector<PathNodePtr>, NodeComparator> empty_queue;
  open_set_.swap(empty_queue);

  for (int i = 0; i < use_node_num_; i++) {
    PathNodePtr node = path_node_pool_[i];
    node->parent = NULL;
    node->node_state = NOT_EXPAND;
  }

  use_node_num_ = 0;
  iter_num_ = 0;
  is_shot_succ_ = false;
  has_path_ = false;
}

// ============================================================================
// 兼容原接口：search(Vector4d start_state, Vector2d init_ctrl, Vector4d end_state, bool use3d)
// Vector4d: (x, y, yaw, v)
// ============================================================================
int AckermannKinoAstar::search(Eigen::Vector4d start_state, Eigen::Vector2d init_ctrl,
                                      Eigen::Vector4d end_state, bool use3d) {
  ROS_WARN("========================================");
  ROS_WARN("[AckermannKinoAstar::search] ACKERMANN PLANNER CALLED!");
  ROS_WARN("Start: (%.2f, %.2f, %.1f°, %.2f m/s)", 
           start_state(0), start_state(1), start_state(2)*180/M_PI, start_state(3));
  ROS_WARN("End: (%.2f, %.2f, %.1f°, %.2f m/s)", 
           end_state(0), end_state(1), end_state(2)*180/M_PI, end_state(3));
  ROS_WARN("Using Ackermann kinematic model (steering + arc length)");
  ROS_WARN("========================================");
  
  // 转换为标准接口
  // Vector4d: (x, y, yaw, v) -> Vector3d: (x, y, z)
  Eigen::Vector3d start_pt;
  start_pt << start_state(0), start_state(1), fixed_height_2d_;
  Eigen::Vector3d start_vel;
  start_vel << start_state(3) * cos(start_state(2)), 
               start_state(3) * sin(start_state(2)), 0.0;
  Eigen::Vector3d start_acc(0.0, 0.0, 0.0);  // 简化
  
  Eigen::Vector3d end_pt;
  end_pt << end_state(0), end_state(1), fixed_height_2d_;
  Eigen::Vector3d end_vel;
  end_vel << end_state(3) * cos(end_state(2)), 
             end_state(3) * sin(end_state(2)), 0.0;
  
  // 调用标准 search 函数
  return search(start_pt, start_vel, start_acc, end_pt, end_vel, false, false, -1.0);
}

// ============================================================================
// 主搜索函数（基于 kinodynamic_astar，只修改节点扩展和约束检查）
// ============================================================================
int AckermannKinoAstar::search(Eigen::Vector3d start_pt, Eigen::Vector3d start_v, 
                                     Eigen::Vector3d start_a, Eigen::Vector3d end_pt, 
                                     Eigen::Vector3d end_v, bool init, bool dynamic,
                                     double time_start) {
  ROS_INFO("[AckermannKinoAstar] Internal search function called");
  ROS_INFO("[AckermannKinoAstar] Max steering: %.1f°, Max curvature: %.3f, Wheelbase: %.2f", 
           max_steer_*180/M_PI, max_cur_, car_wheelbase_);
  
  // 对于 Ackermann：start_v 和 end_v 是 2D 速度，需要转换为 3D
  // start_pt 和 end_pt 的 z 坐标设为 fixed_height_2d_
  Eigen::Vector3d start_pt_3d(start_pt(0), start_pt(1), fixed_height_2d_);
  Eigen::Vector3d end_pt_3d(end_pt(0), end_pt(1), fixed_height_2d_);
  
  // 从 2D 速度计算 yaw（用于初始化）
  double start_yaw = atan2(start_v(1), start_v(0));
  double end_yaw = atan2(end_v(1), end_v(0));
  double start_vel_mag = start_v.head(2).norm();
  double end_vel_mag = end_v.head(2).norm();
  
  // 检查终点碰撞
  Eigen::Vector3d end_state_3d(end_pt_3d(0), end_pt_3d(1), end_yaw);
  bool isocc = false;
  checkCollisionUsingPosAndYaw(end_state_3d, isocc);
  if (isocc) {
    ROS_WARN("AckermannKinoAstar: end is not free!");
    return NO_PATH;
  }

  start_vel_ = Eigen::Vector3d(start_vel_mag * cos(start_yaw), start_vel_mag * sin(start_yaw), 0.0);
  end_vel_ = Eigen::Vector3d(end_vel_mag * cos(end_yaw), end_vel_mag * sin(end_yaw), 0.0);
  start_acc_ = start_a;

  PathNodePtr cur_node = path_node_pool_[0];
  cur_node->parent = NULL;
  cur_node->state.head(3) = start_pt_3d;
  cur_node->state.tail(3) = start_vel_;
  cur_node->index = posToIndex(start_pt_3d);
  cur_node->g_score = 0.0;

  Eigen::VectorXd end_state(6);
  Eigen::Vector3i end_index;
  double time_to_goal;

  end_state.head(3) = end_pt_3d;
  end_state.tail(3) = end_vel_;
  end_index = posToIndex(end_pt_3d);
  cur_node->f_score = lambda_heu_ * estimateHeuristic(cur_node->state, end_state, time_to_goal);
  cur_node->node_state = IN_OPEN_SET;
  open_set_.push(cur_node);
  use_node_num_ += 1;

  if (dynamic) {
    time_origin_ = time_start;
    cur_node->time = time_start;
    cur_node->time_idx = timeToIndex(time_start);
    expanded_nodes_.insert(cur_node->index, cur_node->time_idx, cur_node);
  } else
    expanded_nodes_.insert(cur_node->index, cur_node);

  PathNodePtr neighbor = NULL;
  PathNodePtr terminate_node = NULL;
  bool init_search = init;
  const int tolerance = ceil(1 / resolution_);

  while (!open_set_.empty()) {
    cur_node = open_set_.top();

    // Terminate?
    bool reach_horizon = (cur_node->state.head(3) - start_pt_3d).norm() >= horizon_;
    bool near_end = abs(cur_node->index(0) - end_index(0)) <= tolerance &&
                    abs(cur_node->index(1) - end_index(1)) <= tolerance &&
                    abs(cur_node->index(2) - end_index(2)) <= tolerance;

    if (reach_horizon || near_end) {
      terminate_node = cur_node;
      retrievePath(terminate_node);
      if (near_end) {
        // Check whether shot traj exist
        estimateHeuristic(cur_node->state, end_state, time_to_goal);
        computeShotTraj(cur_node->state, end_state, time_to_goal);
        if (init_search) ROS_ERROR("Shot in first search loop!");
      }
    }
    if (reach_horizon) {
      if (is_shot_succ_) {
        std::cout << "reach end" << std::endl;
        return REACH_END;
      } else {
        std::cout << "reach horizon" << std::endl;
        return REACH_HORIZON;
      }
    }

    if (near_end) {
      if (is_shot_succ_) {
        std::cout << "reach end" << std::endl;
        return REACH_END;
      } else if (cur_node->parent != NULL) {
        std::cout << "near end" << std::endl;
        return NEAR_END;
      } else {
        std::cout << "no path" << std::endl;
        return NO_PATH;
      }
    }
    open_set_.pop();
    cur_node->node_state = IN_CLOSE_SET;
    iter_num_ += 1;

    // ========================================================================
    // 节点扩展部分（修改：从加速度改为转向角+弧长）
    // ========================================================================
    double res = 0.5;
    double step_arc = max_tau_ * max_vel_;  // 最大弧长
    Eigen::Matrix<double, 6, 1> cur_state = cur_node->state;
    Eigen::Matrix<double, 6, 1> pro_state;
    vector<PathNodePtr> tmp_expand_nodes;
    
    // 控制输入：2D (steering_angle, arc_length)
    vector<Eigen::Vector2d> inputs;
    vector<double> durations;
    
    if (init_search) {
      // 初始化：使用小步长
      double init_arc = resolution_;
      for (double arc = init_arc; arc <= 2 * init_arc + 1e-3; arc += init_arc) {
        for (double steer = -max_steer_; steer <= max_steer_ + 1e-3; steer += res * max_steer_) {
          Eigen::Vector2d ctrl_input;
          ctrl_input << steer, arc;
          inputs.push_back(ctrl_input);
        }
      }
      // 持续时间由弧长和速度决定
      for (double arc = init_arc; arc <= 2 * init_arc + 1e-3; arc += init_arc) {
        durations.push_back(arc / max_vel_);
      }
      init_search = false;
    } else {
      // 正常扩展：枚举转向角和弧长
      for (double arc = -step_arc; arc <= step_arc + 1e-3; arc += 0.5 * step_arc) {
        if (fabs(arc) < 1.0e-2) continue;
        for (double steer = -max_steer_; steer <= max_steer_ + 1e-3; steer += res * max_steer_) {
          Eigen::Vector2d ctrl_input;
          ctrl_input << steer, arc;
          inputs.push_back(ctrl_input);
        }
      }
      // 持续时间由弧长决定
      for (double arc = step_arc * 0.5; arc <= step_arc + 1e-3; arc += 0.5 * step_arc) {
        durations.push_back(arc / max_vel_);
      }
    }

    // ========================================================================
    // 状态传播循环（修改：使用阿克曼运动学模型）
    // ========================================================================
    for (int i = 0; i < inputs.size(); ++i) {
      Eigen::Vector2d ctrl_input = inputs[i];
      double tau = durations[i % durations.size()];  // 持续时间由弧长决定
      
      // 使用阿克曼运动学模型进行状态传播
      stateTransit(cur_state, pro_state, ctrl_input, tau);
      double pro_t = cur_node->time + tau;

      // ======================================================================
      // 约束检查部分（修改：2D地图 + 车辆形状碰撞）
      // ======================================================================
      
      // Check inside map range (2D)
      Eigen::Vector2d pro_pos_2d = pro_state.head(2);
      if (!isInMap2d(pro_pos_2d)) {
        if (init_search) std::cout << "out of map" << std::endl;
        continue;
      }

      // Check if in close set
      Eigen::Vector3d pro_pos_3d(pro_pos_2d(0), pro_pos_2d(1), fixed_height_2d_);
      Eigen::Vector3i pro_id = posToIndex(pro_pos_3d);
      int pro_t_id = timeToIndex(pro_t);
      PathNodePtr pro_node =
          dynamic ? expanded_nodes_.find(pro_id, pro_t_id) : expanded_nodes_.find(pro_id);
      if (pro_node != NULL && pro_node->node_state == IN_CLOSE_SET) {
        if (init_search) std::cout << "in close set" << std::endl;
        continue;
      }

      // Check velocity (从 yaw 和速度计算)
      // 对于 Ackermann，速度由弧长和时间决定，这里检查速度大小
      double pro_vel_mag = pro_state.tail(3).head(2).norm();
      if (pro_vel_mag > max_vel_) {
        if (init_search) std::cout << "vel" << std::endl;
        continue;
      }

      // Check not in the same voxel
      Eigen::Vector3i diff = pro_id - cur_node->index;
      int diff_time = pro_t_id - cur_node->time_idx;
      if (diff.norm() == 0 && ((!dynamic) || diff_time == 0)) {
        if (init_search) std::cout << "same voxel" << std::endl;
        continue;
      }

      // Check collision along trajectory (车辆形状碰撞)
      Eigen::Matrix<double, 6, 1> xt;
      bool is_occ = false;
      for (int k = 1; k <= check_num_; ++k) {
        double tmparc = ctrl_input(1) * double(k) / double(check_num_);
        Eigen::Vector2d tmpctrl;
        tmpctrl << ctrl_input(0), tmparc;
        stateTransit(cur_state, xt, tmpctrl, tmparc / max_vel_);
        
        Eigen::Vector3d xt_pos_3d(xt(0), xt(1), fixed_height_2d_);
        checkCollisionUsingPosAndYaw(xt_pos_3d, is_occ);
        if (is_occ) {
          if (init_search) std::cout << "collision" << std::endl;
          break;
        }
      }
      if (is_occ) {
        continue;
      }

      // ======================================================================
      // 代价计算（修改：使用弧长和转向角）
      // ======================================================================
      double time_to_goal, tmp_g_score, tmp_f_score;
      // 代价 = 弧长 + 转向惩罚 + 时间惩罚
      tmp_g_score = std::fabs(ctrl_input(1)) + 
                   0.1 * std::fabs(ctrl_input(0)) * std::fabs(ctrl_input(1)) +
                   w_time_ * tau + 
                   cur_node->g_score;
      tmp_f_score = tmp_g_score + lambda_heu_ * estimateHeuristic(pro_state, end_state, time_to_goal);

      // Compare nodes expanded from the same parent
      bool prune = false;
      for (int j = 0; j < tmp_expand_nodes.size(); ++j) {
        PathNodePtr expand_node = tmp_expand_nodes[j];
        if ((pro_id - expand_node->index).norm() == 0 &&
            ((!dynamic) || pro_t_id == expand_node->time_idx)) {
          prune = true;
          if (tmp_f_score < expand_node->f_score) {
            expand_node->f_score = tmp_f_score;
            expand_node->g_score = tmp_g_score;
            expand_node->state = pro_state;
            // 存储控制输入（转换为3D以兼容接口）
            expand_node->input = Eigen::Vector3d(ctrl_input(0), ctrl_input(1), 0.0);
            expand_node->duration = tau;
            if (dynamic) expand_node->time = cur_node->time + tau;
          }
          break;
        }
      }

      // This node ends up in a voxel different from others
      if (!prune) {
        if (pro_node == NULL) {
          pro_node = path_node_pool_[use_node_num_];
          pro_node->index = pro_id;
          pro_node->state = pro_state;
          pro_node->f_score = tmp_f_score;
          pro_node->g_score = tmp_g_score;
          pro_node->input = Eigen::Vector3d(ctrl_input(0), ctrl_input(1), 0.0);
          pro_node->duration = tau;
          pro_node->parent = cur_node;
          pro_node->node_state = IN_OPEN_SET;
          if (dynamic) {
            pro_node->time = cur_node->time + tau;
            pro_node->time_idx = timeToIndex(pro_node->time);
          }
          open_set_.push(pro_node);

          if (dynamic)
            expanded_nodes_.insert(pro_id, pro_node->time, pro_node);
          else
            expanded_nodes_.insert(pro_id, pro_node);

          tmp_expand_nodes.push_back(pro_node);

          use_node_num_ += 1;
          if (use_node_num_ == allocate_num_) {
            cout << "run out of memory." << endl;
            return NO_PATH;
          }
        } else if (pro_node->node_state == IN_OPEN_SET) {
          if (tmp_g_score < pro_node->g_score) {
            pro_node->state = pro_state;
            pro_node->f_score = tmp_f_score;
            pro_node->g_score = tmp_g_score;
            pro_node->input = Eigen::Vector3d(ctrl_input(0), ctrl_input(1), 0.0);
            pro_node->duration = tau;
            pro_node->parent = cur_node;
            if (dynamic) pro_node->time = cur_node->time + tau;
          }
        } else {
          cout << "error type in searching: " << pro_node->node_state << endl;
        }
      }
    }
  }

  cout << "open set empty, no path!" << endl;
  cout << "use node num: " << use_node_num_ << endl;
  cout << "iter num: " << iter_num_ << endl;
  return NO_PATH;
}

// ============================================================================
// 阿克曼运动学模型（状态传播）
// ============================================================================
void AckermannKinoAstar::stateTransit(Eigen::Matrix<double, 6, 1>& state0,
                                            Eigen::Matrix<double, 6, 1>& state1,
                                            Eigen::Vector2d ctrl_input, double tau) {
  // state0: (x, y, z, vx, vy, vz)
  // 对于 Ackermann：z = fixed_height_2d_, vz = 0
  // ctrl_input: (steering_angle, arc_length)
  
  double psi = ctrl_input(0);  // 转向角
  double s = ctrl_input(1);     // 弧长
  
  Eigen::Vector3d state0_2d(state0(0), state0(1), atan2(state0(4), state0(3)));  // (x, y, yaw)
  Eigen::Vector3d state1_2d;
  
  if (fabs(psi) > 1e-6) {
    // 有转向：圆弧运动
    double k = car_wheelbase_ / tan(psi);  // 转弯半径
    state1_2d(0) = state0_2d(0) + k * (sin(state0_2d(2) + s / k) - sin(state0_2d(2)));
    state1_2d(1) = state0_2d(1) - k * (cos(state0_2d(2) + s / k) - cos(state0_2d(2)));
    state1_2d(2) = state0_2d(2) + s / k;
  } else {
    // 直线运动
    state1_2d(0) = state0_2d(0) + s * cos(state0_2d(2));
    state1_2d(1) = state0_2d(1) + s * sin(state0_2d(2));
    state1_2d(2) = state0_2d(2);
  }
  
  // 更新速度（从 yaw 和速度大小计算）
  double vel_mag = s / tau;  // 速度大小
  state1(0) = state1_2d(0);
  state1(1) = state1_2d(1);
  state1(2) = fixed_height_2d_;
  state1(3) = vel_mag * cos(state1_2d(2));
  state1(4) = vel_mag * sin(state1_2d(2));
  state1(5) = 0.0;
}

// ============================================================================
// 碰撞检测（车辆形状）
// ============================================================================
void AckermannKinoAstar::checkCollisionUsingPosAndYaw(const Eigen::Vector3d& state, bool& res) {
  res = false;
  Eigen::Vector2d pos = state.head(2);
  double yaw = state(2);
  Eigen::Matrix2d Rotation_matrix;
  Rotation_matrix << cos(yaw), -sin(yaw),
                     sin(yaw),  cos(yaw);
  
  for (int i = 0; i < 4; i++) {
    Eigen::Vector2d start_point_2d = pos + Rotation_matrix * car_vertex_[i];
    Eigen::Vector2d end_point_2d = pos + Rotation_matrix * car_vertex_[i + 1];
    
    // Convert to 3D for RayCaster
    Eigen::Vector3d start_point(start_point_2d(0), start_point_2d(1), fixed_height_2d_);
    Eigen::Vector3d end_point(end_point_2d(0), end_point_2d(1), fixed_height_2d_);
    
    RayCaster raycaster;
    if (!raycaster.input(start_point, end_point)) {
      continue;
    }
    
    Eigen::Vector3d ray_pt;
    while (raycaster.step(ray_pt)) {
      Eigen::Vector2d tmp(ray_pt(0), ray_pt(1));
      if (getVoxelState2d(tmp) == 1) {
        res = true;
        return;
      }
    }
  }
}

bool AckermannKinoAstar::isInMap2d(const Eigen::Vector2d& pos) {
  Eigen::Vector3d pos3d(pos(0), pos(1), fixed_height_2d_);
  return edt_environment_->sdf_map_->isInMap(pos3d);
}

int AckermannKinoAstar::getVoxelState2d(const Eigen::Vector2d& pos) {
  Eigen::Vector3d pos3d(pos(0), pos(1), fixed_height_2d_);
  int occ = edt_environment_->sdf_map_->getInflateOccupancy(pos3d);
  return (occ == SDFMap::OCCUPIED) ? 1 : 0;
}

// ============================================================================
// 辅助函数（复用 kinodynamic_astar）
// ============================================================================
Eigen::Vector3i AckermannKinoAstar::posToIndex(Eigen::Vector3d pt) {
  Vector3i idx = ((pt - origin_) * inv_resolution_).array().floor().cast<int>();
  return idx;
}

int AckermannKinoAstar::timeToIndex(double time) {
  int idx = floor((time - time_origin_) * inv_time_resolution_);
  return idx;
}

void AckermannKinoAstar::retrievePath(PathNodePtr end_node) {
  PathNodePtr cur_node = end_node;
  path_nodes_.push_back(cur_node);

  while (cur_node->parent != NULL) {
    cur_node = cur_node->parent;
    path_nodes_.push_back(cur_node);
  }

  reverse(path_nodes_.begin(), path_nodes_.end());
}

// ============================================================================
// 启发式函数（使用 Reeds-Shepp 距离）
// ============================================================================
double AckermannKinoAstar::estimateHeuristic(Eigen::VectorXd x1, Eigen::VectorXd x2,
                                                    double& optimal_time) {
  // 对于 Ackermann：x1 和 x2 是 6D，但实际只用前3维 (x, y, z)
  // 从速度计算 yaw
  double yaw1 = atan2(x1(4), x1(3));
  double yaw2 = atan2(x2(4), x2(3));
  
  namespace ob = ompl::base;
  ob::ScopedState<> from(shotptr), to(shotptr);
  
  from[0] = x1(0); from[1] = x1(1); from[2] = yaw1;
  to[0] = x2(0); to[1] = x2(1); to[2] = yaw2;
  
  double rs_distance = shotptr->distance(from(), to());
  optimal_time = rs_distance / max_vel_;
  double cost = rs_distance * (1.0 + tie_breaker_);
  
  return cost;
}

// ============================================================================
// One-shot 轨迹（使用 Reeds-Shepp）
// ============================================================================
bool AckermannKinoAstar::computeShotTraj(Eigen::VectorXd state1, Eigen::VectorXd state2,
                                                double time_to_goal) {
  double yaw1 = atan2(state1(4), state1(3));
  double yaw2 = atan2(state2(4), state2(3));
  
  namespace ob = ompl::base;
  ob::ScopedState<> from(shotptr), to(shotptr), s(shotptr);
  from[0] = state1(0); from[1] = state1(1); from[2] = yaw1;
  to[0] = state2(0); to[1] = state2(1); to[2] = yaw2;
  
  double rs_distance = shotptr->distance(from(), to());
  double t_d = time_to_goal;
  
  // 检查轨迹是否无碰撞
  double t_delta = t_d / 10;
  for (double time = t_delta; time <= t_d; time += t_delta) {
    double ratio = time / t_d;
    shotptr->interpolate(from(), to(), ratio, s());
    std::vector<double> reals = s.reals();
    
    Eigen::Vector3d coord(reals[0], reals[1], fixed_height_2d_);
    Eigen::Vector3d state_check(reals[0], reals[1], reals[2]);
    
    if (!edt_environment_->sdf_map_->isInMap(coord)) {
      return false;
    }
    
    bool is_occ = false;
    checkCollisionUsingPosAndYaw(state_check, is_occ);
    if (is_occ) {
      return false;
    }
  }
  
  // 存储轨迹系数（简化：使用 Reeds-Shepp 插值）
  t_shot_ = t_d;
  is_shot_succ_ = true;
  return true;
}

// ============================================================================
// 方程求解（复用 kinodynamic_astar）
// ============================================================================
vector<double> AckermannKinoAstar::cubic(double a, double b, double c, double d) {
  vector<double> dts;

  double a2 = b / a;
  double a1 = c / a;
  double a0 = d / a;

  double Q = (3 * a1 - a2 * a2) / 9;
  double R = (9 * a1 * a2 - 27 * a0 - 2 * a2 * a2 * a2) / 54;
  double D = Q * Q * Q + R * R;
  if (D > 0) {
    double S = std::cbrt(R + sqrt(D));
    double T = std::cbrt(R - sqrt(D));
    dts.push_back(-a2 / 3 + (S + T));
    return dts;
  } else if (D == 0) {
    double S = std::cbrt(R);
    dts.push_back(-a2 / 3 + S + S);
    dts.push_back(-a2 / 3 - S);
    return dts;
  } else {
    double theta = acos(R / sqrt(-Q * Q * Q));
    dts.push_back(2 * sqrt(-Q) * cos(theta / 3) - a2 / 3);
    dts.push_back(2 * sqrt(-Q) * cos((theta + 2 * M_PI) / 3) - a2 / 3);
    dts.push_back(2 * sqrt(-Q) * cos((theta + 4 * M_PI) / 3) - a2 / 3);
    return dts;
  }
}

vector<double> AckermannKinoAstar::quartic(double a, double b, double c, double d, double e) {
  vector<double> dts;

  double a3 = b / a;
  double a2 = c / a;
  double a1 = d / a;
  double a0 = e / a;

  vector<double> ys = cubic(1, -a2, a1 * a3 - 4 * a0, 4 * a2 * a0 - a1 * a1 - a3 * a3 * a0);
  double y1 = ys.front();
  double r = a3 * a3 / 4 - a2 + y1;
  if (r < 0) return dts;

  double R = sqrt(r);
  double D, E;
  if (R != 0) {
    D = sqrt(0.75 * a3 * a3 - R * R - 2 * a2 + 0.25 * (4 * a3 * a2 - 8 * a1 - a3 * a3 * a3) / R);
    E = sqrt(0.75 * a3 * a3 - R * R - 2 * a2 - 0.25 * (4 * a3 * a2 - 8 * a1 - a3 * a3 * a3) / R);
  } else {
    D = sqrt(0.75 * a3 * a3 - 2 * a2 + 2 * sqrt(y1 * y1 - 4 * a0));
    E = sqrt(0.75 * a3 * a3 - 2 * a2 - 2 * sqrt(y1 * y1 - 4 * a0));
  }

  if (!std::isnan(D)) {
    dts.push_back(-a3 / 4 + R / 2 + D / 2);
    dts.push_back(-a3 / 4 + R / 2 - D / 2);
  }
  if (!std::isnan(E)) {
    dts.push_back(-a3 / 4 - R / 2 + E / 2);
    dts.push_back(-a3 / 4 - R / 2 - E / 2);
  }

  return dts;
}

// ============================================================================
// 轨迹生成（复用 kinodynamic_astar，但适配 Ackermann）
// ============================================================================
std::vector<Eigen::Vector3d> AckermannKinoAstar::getKinoTraj(double delta_t) {
  vector<Vector3d> state_list;

  /* ---------- get traj of searching ---------- */
  PathNodePtr node = path_nodes_.back();
  Matrix<double, 6, 1> x0, xt;

  while (node->parent != NULL) {
    // 从 3D input 提取 2D 控制输入
    Eigen::Vector2d ut(node->input(0), node->input(1));
    double duration = node->duration;
    x0 = node->parent->state;

    for (double t = duration; t >= -1e-5; t -= delta_t) {
      double ratio = t / duration;
      double arc = ut(1) * ratio;
      Eigen::Vector2d tmpctrl(ut(0), arc);
      stateTransit(x0, xt, tmpctrl, t);
      state_list.push_back(xt.head(3));
    }
    node = node->parent;
  }
  reverse(state_list.begin(), state_list.end());

  /* ---------- get traj of one shot (Reeds-Shepp) ---------- */
  if (is_shot_succ_) {
    PathNodePtr last_node = path_nodes_.back();
    double yaw1 = atan2(last_node->state(4), last_node->state(3));
    Eigen::Vector3d end_pos = end_vel_.head(3);
    double yaw2 = atan2(end_vel_(1), end_vel_(0));
    
    namespace ob = ompl::base;
    ob::ScopedState<> from(shotptr), to(shotptr), s(shotptr);
    from[0] = last_node->state(0); from[1] = last_node->state(1); from[2] = yaw1;
    to[0] = end_pos(0); to[1] = end_pos(1); to[2] = yaw2;
    
    double rs_distance = shotptr->distance(from(), to());
    for (double l = checkl_; l <= rs_distance; l += checkl_) {
      shotptr->interpolate(from(), to(), l / rs_distance, s());
      std::vector<double> reals = s.reals();
      state_list.push_back(Eigen::Vector3d(reals[0], reals[1], fixed_height_2d_));
    }
  }

  return state_list;
}

void AckermannKinoAstar::getSamples(double& ts, vector<Eigen::Vector3d>& point_set,
                                          vector<Eigen::Vector3d>& start_end_derivatives) {
  /* ---------- path duration ---------- */
  double T_sum = 0.0;
  if (is_shot_succ_) T_sum += t_shot_;
  PathNodePtr node = path_nodes_.back();
  while (node->parent != NULL) {
    T_sum += node->duration;
    node = node->parent;
  }

  // Calculate boundary vel and acc
  Eigen::Vector3d end_vel, end_acc;
  double t;
  if (is_shot_succ_) {
    t = t_shot_;
    end_vel = end_vel_;
    end_acc = Eigen::Vector3d(0.0, 0.0, 0.0);  // 简化
  } else {
    t = path_nodes_.back()->duration;
    end_vel = node->state.tail(3);
    end_acc = Eigen::Vector3d(0.0, 0.0, 0.0);  // 简化
  }

  // Get point samples
  int seg_num = floor(T_sum / ts);
  seg_num = max(8, seg_num);
  ts = T_sum / double(seg_num);
  bool sample_shot_traj = is_shot_succ_;
  node = path_nodes_.back();

  for (double ti = T_sum; ti > -1e-5; ti -= ts) {
    if (sample_shot_traj) {
      // samples on shot traj (Reeds-Shepp)
      PathNodePtr last_node = path_nodes_.back();
      double yaw1 = atan2(last_node->state(4), last_node->state(3));
      Eigen::Vector3d end_pos = end_vel_.head(3);
      double yaw2 = atan2(end_vel_(1), end_vel_(0));
      
      namespace ob = ompl::base;
      ob::ScopedState<> from(shotptr), to(shotptr), s(shotptr);
      from[0] = last_node->state(0); from[1] = last_node->state(1); from[2] = yaw1;
      to[0] = end_pos(0); to[1] = end_pos(1); to[2] = yaw2;
      
      double rs_distance = shotptr->distance(from(), to());
      double ratio = (T_sum - ti) / t_shot_;
      ratio = std::max(0.0, std::min(1.0, ratio));
      
      shotptr->interpolate(from(), to(), ratio, s());
      std::vector<double> reals = s.reals();
      point_set.push_back(Eigen::Vector3d(reals[0], reals[1], fixed_height_2d_));
      t -= ts;

      if (t < -1e-5) {
        sample_shot_traj = false;
        if (node->parent != NULL) t += node->duration;
      }
    } else {
      // samples on searched traj
      Eigen::Matrix<double, 6, 1> x0 = node->parent->state;
      Eigen::Matrix<double, 6, 1> xt;
      Eigen::Vector2d ut(node->input(0), node->input(1));
      
      double ratio = t / node->duration;
      double arc = ut(1) * ratio;
      Eigen::Vector2d tmpctrl(ut(0), arc);
      stateTransit(x0, xt, tmpctrl, t);

      point_set.push_back(xt.head(3));
      t -= ts;

      if (t < -1e-5 && node->parent->parent != NULL) {
        node = node->parent;
        t += node->duration;
      }
    }
  }
  reverse(point_set.begin(), point_set.end());

  // calculate start acc
  Eigen::Vector3d start_acc;
  if (path_nodes_.back()->parent == NULL) {
    start_acc = Eigen::Vector3d(0.0, 0.0, 0.0);
  } else {
    start_acc = Eigen::Vector3d(0.0, 0.0, 0.0);  // 简化
  }

  start_end_derivatives.push_back(start_vel_);
  start_end_derivatives.push_back(end_vel);
  start_end_derivatives.push_back(start_acc);
  start_end_derivatives.push_back(end_acc);
}

std::vector<PathNodePtr> AckermannKinoAstar::getVisitedNodes() {
  vector<PathNodePtr> visited;
  visited.assign(path_node_pool_.begin(), path_node_pool_.begin() + use_node_num_ - 1);
  return visited;
}

}  // namespace fast_planner

