#include "forex_nav/astar_2d.h"
#include <algorithm>
#include <iostream>

namespace forex_nav {

Astar2D::Astar2D()
    : resolution_(0.1), fixed_height_(1.0), max_expansions_(80000),
      dist_field_dirty_(true), dist_field_w_(0), dist_field_h_(0) {
}

Astar2D::~Astar2D() {
}

void Astar2D::setMap(const nav_msgs::OccupancyGrid::ConstPtr& map) {
  map_ = map;
  dist_field_dirty_ = true;  // Mark distance field for lazy recomputation
}

void Astar2D::setResolution(double resolution) {
  resolution_ = resolution;
}

void Astar2D::setFixedHeight(double height) {
  fixed_height_ = height;
}

void Astar2D::setMaxExpansions(int max_exp) {
  max_expansions_ = max_exp;
}

void Astar2D::reset() {
  dist_field_.clear();
  dist_field_dirty_ = true;
}

// ----------------------------------------------------------------
//  Precomputed distance field: BFS from all obstacle cells
//  Chebyshev distance (8-connected), capped at check_radius.
//  Complexity: O(W*H) — runs once per map update, lazily.
// ----------------------------------------------------------------
void Astar2D::computeDistanceField() {
  if (!map_) return;

  const int w = static_cast<int>(map_->info.width);
  const int h = static_cast<int>(map_->info.height);
  const int total = w * h;
  const int check_radius = 5;  // Same as original getObstacleProximityCost

  dist_field_w_ = w;
  dist_field_h_ = h;
  dist_field_.assign(total, check_radius + 1);

  // Seed BFS with all obstacle cells
  std::queue<int> q;
  for (int i = 0; i < total; ++i) {
    if (map_->data[i] >= 50) {
      dist_field_[i] = 0;
      q.push(i);
    }
  }

  // 8-connected BFS: propagate Chebyshev distance
  const int dx8[8] = {1, -1, 0, 0, 1, -1, 1, -1};
  const int dy8[8] = {0, 0, 1, -1, 1, 1, -1, -1};

  while (!q.empty()) {
    int idx = q.front();
    q.pop();
    int cx = idx % w;
    int cy = idx / w;
    int cd = dist_field_[idx];
    if (cd >= check_radius) continue;  // Don't propagate beyond radius

    for (int i = 0; i < 8; ++i) {
      int nx = cx + dx8[i];
      int ny = cy + dy8[i];
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
      int nidx = ny * w + nx;
      int nd = cd + 1;
      if (nd < dist_field_[nidx]) {
        dist_field_[nidx] = nd;
        q.push(nidx);
      }
    }
  }
}

void Astar2D::ensureDistanceField() {
  if (dist_field_dirty_ && map_) {
    computeDistanceField();
    dist_field_dirty_ = false;
  }
}

bool Astar2D::search(const Eigen::Vector3d& start, const Eigen::Vector3d& goal,
                     std::vector<Eigen::Vector3d>& path) {
  path.clear();
  
  if (!map_) {
    return false;
  }
  
  // Ensure distance field is up-to-date (lazy computation)
  ensureDistanceField();
  
  Eigen::Vector2i start_map = worldToMap(start);
  Eigen::Vector2i goal_map = worldToMap(goal);
  
  if (!isInMap(start_map.x(), start_map.y()) || !isInMap(goal_map.x(), goal_map.y())) {
    return false;
  }
  
  if (isObstacle(start_map.x(), start_map.y()) || isObstacle(goal_map.x(), goal_map.y())) {
    return false;
  }
  
  // A* search
  std::priority_queue<Node, std::vector<Node>, std::greater<Node>> open_set;
  std::unordered_map<int, Node> all_nodes;
  std::unordered_map<int, bool> closed_set;
  
  Node start_node;
  start_node.x = start_map.x();
  start_node.y = start_map.y();
  start_node.g_score = 0.0;
  start_node.f_score = heuristic(start_map.x(), start_map.y(), goal_map.x(), goal_map.y());
  start_node.parent_x = -1;
  start_node.parent_y = -1;
  
  open_set.push(start_node);
  all_nodes[hash(start_node.x, start_node.y)] = start_node;
  
  // 8-connected neighbors
  const int dx[8] = {1, -1, 0, 0, 1, -1, 1, -1};
  const int dy[8] = {0, 0, 1, -1, 1, 1, -1, -1};
  const double cost[8] = {1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414};
  
  int expansions = 0;
  
  while (!open_set.empty()) {
    Node current = open_set.top();
    open_set.pop();
    
    int current_hash = hash(current.x, current.y);
    if (closed_set.find(current_hash) != closed_set.end()) {
      continue;
    }
    closed_set[current_hash] = true;
    
    ++expansions;
    if (expansions > max_expansions_) {
      // Expansion limit reached — return best-effort path to closest node
      std::cerr << "[Astar2D] Max expansions (" << max_expansions_ 
                << ") reached, search aborted" << std::endl;
      return false;
    }
    
    // Check if reached goal
    if (std::abs(current.x - goal_map.x()) <= 1 && std::abs(current.y - goal_map.y()) <= 1) {
      reconstructPath(all_nodes, current.x, current.y, path);
      return true;
    }
    
    // Explore neighbors
    for (int i = 0; i < 8; ++i) {
      int nx = current.x + dx[i];
      int ny = current.y + dy[i];
      
      if (!isInMap(nx, ny) || isObstacle(nx, ny)) {
        continue;
      }
      
      int n_hash = hash(nx, ny);
      if (closed_set.find(n_hash) != closed_set.end()) {
        continue;
      }
      
      // O(1) obstacle proximity cost from precomputed distance field
      double obstacle_cost = getObstacleProximityCost(nx, ny);
      double tentative_g = current.g_score + cost[i] + obstacle_cost;
      
      auto it = all_nodes.find(n_hash);
      if (it == all_nodes.end() || tentative_g < it->second.g_score) {
        Node neighbor;
        neighbor.x = nx;
        neighbor.y = ny;
        neighbor.g_score = tentative_g;
        neighbor.f_score = tentative_g + heuristic(nx, ny, goal_map.x(), goal_map.y());
        neighbor.parent_x = current.x;
        neighbor.parent_y = current.y;
        
        all_nodes[n_hash] = neighbor;
        open_set.push(neighbor);
      }
    }
  }
  
  return false;
}

bool Astar2D::isInMap(int x, int y) const {
  if (!map_) return false;
  return x >= 0 && y >= 0 && x < static_cast<int>(map_->info.width) && 
         y < static_cast<int>(map_->info.height);
}

bool Astar2D::isFree(int x, int y) const {
  if (!isInMap(x, y)) return false;
  int idx = y * map_->info.width + x;
  if (idx < 0 || idx >= static_cast<int>(map_->data.size())) return false;
  return map_->data[idx] == 0;
}

bool Astar2D::isObstacle(int x, int y) const {
  if (!isInMap(x, y)) return true;
  int idx = y * map_->info.width + x;
  if (idx < 0 || idx >= static_cast<int>(map_->data.size())) return true;
  int8_t val = map_->data[idx];
  return val >= 50;  // Consider >= 50 as obstacle
}

Eigen::Vector2i Astar2D::worldToMap(const Eigen::Vector3d& world) const {
  if (!map_) return Eigen::Vector2i(0, 0);
  double res = map_->info.resolution;
  double ox = map_->info.origin.position.x;
  double oy = map_->info.origin.position.y;
  int x = static_cast<int>((world.x() - ox) / res);
  int y = static_cast<int>((world.y() - oy) / res);
  return Eigen::Vector2i(x, y);
}

Eigen::Vector3d Astar2D::mapToWorld(int x, int y) const {
  if (!map_) return Eigen::Vector3d::Zero();
  double res = map_->info.resolution;
  double ox = map_->info.origin.position.x;
  double oy = map_->info.origin.position.y;
  double wx = ox + (static_cast<double>(x) + 0.5) * res;
  double wy = oy + (static_cast<double>(y) + 0.5) * res;
  return Eigen::Vector3d(wx, wy, fixed_height_);
}

double Astar2D::heuristic(int x1, int y1, int x2, int y2) const {
  double dx = static_cast<double>(x1 - x2);
  double dy = static_cast<double>(y1 - y2);
  return std::sqrt(dx * dx + dy * dy);
}

void Astar2D::reconstructPath(const std::unordered_map<int, Node>& all_nodes,
                               int goal_x, int goal_y,
                               std::vector<Eigen::Vector3d>& path) const {
  path.clear();
  
  int current_x = goal_x;
  int current_y = goal_y;
  
  std::vector<Eigen::Vector2i> path_map;
  
  while (current_x >= 0 && current_y >= 0) {
    path_map.push_back(Eigen::Vector2i(current_x, current_y));
    
    int current_hash = hash(current_x, current_y);
    auto it = all_nodes.find(current_hash);
    if (it == all_nodes.end()) break;
    
    current_x = it->second.parent_x;
    current_y = it->second.parent_y;
  }
  
  std::reverse(path_map.begin(), path_map.end());
  
  for (const auto& pt : path_map) {
    path.push_back(mapToWorld(pt.x(), pt.y()));
  }
}

int Astar2D::hash(int x, int y) const {
  return y * 100000 + x;  // Simple hash function
}

// ----------------------------------------------------------------
//  O(1) obstacle proximity cost using precomputed distance field.
//  Original: O(121) per call (checked 11x11 window every time).
// ----------------------------------------------------------------
double Astar2D::getObstacleProximityCost(int x, int y) const {
  const int check_radius = 5;
  const double weight = 2.0;

  // Use precomputed distance field (Chebyshev distance in cells)
  if (!dist_field_.empty() && x >= 0 && x < dist_field_w_ &&
      y >= 0 && y < dist_field_h_) {
    int d = dist_field_[y * dist_field_w_ + x];
    if (d < check_radius) {
      double normalized_dist = static_cast<double>(d) / check_radius;
      return weight * (1.0 - normalized_dist) * (1.0 - normalized_dist);
    }
    return 0.0;
  }

  // Fallback: no distance field available (should not happen in normal operation)
  return 0.0;
}

}  // namespace forex_nav
