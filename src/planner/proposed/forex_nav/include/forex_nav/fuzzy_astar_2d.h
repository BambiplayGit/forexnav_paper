#pragma once
#include <queue>
#include <vector>
#include <cmath>
#include <limits>
#include <Eigen/Core>
#include <nav_msgs/OccupancyGrid.h>
#include <ros/ros.h>

class FuzzyAstar2D {
public:
  explicit FuzzyAstar2D(const nav_msgs::OccupancyGrid* map_ptr = nullptr,
                        double tie_breaker = 1.001)
      : map_ptr_(map_ptr), tie_breaker_(tie_breaker) {}

  void setMap(const nav_msgs::OccupancyGrid* map_ptr) {
    map_ptr_ = map_ptr;
  }

  void setTieBreaker(double t) { tie_breaker_ = t; }

  bool search(const Eigen::Vector3d& start_w, const Eigen::Vector3d& goal_w,
              std::vector<Eigen::Vector2i>& path_pix,
              std::vector<Eigen::Vector3d>& path_world,
              double& total_cost) {
    if (!map_ptr_) {
      ROS_ERROR_STREAM("\033[31m[FuzzyAstar2D] map_ptr_ is null! Did you forget to setMap()?\033[0m");
      return false;
    }

    const auto& map = *map_ptr_;
    int width = map.info.width;
    int height = map.info.height;
    double resolution = map.info.resolution;
    double origin_x = map.info.origin.position.x;
    double origin_y = map.info.origin.position.y;

    auto worldToMap = [&](const Eigen::Vector3d& pos) {
      int x = static_cast<int>((pos.x() - origin_x) / resolution);
      int y = static_cast<int>((pos.y() - origin_y) / resolution);
      return Eigen::Vector2i(x, y);
    };

    Eigen::Vector2i start = worldToMap(start_w);
    Eigen::Vector2i goal = worldToMap(goal_w);

    if (!isInMap(start, width, height) || !isInMap(goal, width, height)) {
      ROS_ERROR_STREAM("\033[31m[FuzzyAstar2D] Start or goal out of bounds! Map size: "
                       << width << "x" << height << ", Start: " << start.transpose()
                       << ", Goal: " << goal.transpose() << "\033[0m");
      return false;
    }

    const int map_size = width * height;
    std::vector<double> g_score(map_size, std::numeric_limits<double>::infinity());
    std::vector<int> came_from(map_size, -1);

    auto idx = [&](const Eigen::Vector2i& p) { return p.y() * width + p.x(); };
    auto heuristic = [&](const Eigen::Vector2i& a, const Eigen::Vector2i& b) {
      return (a - b).cast<double>().norm();
    };

    using Node = std::pair<double, int>;
    auto cmp = [](const Node& a, const Node& b) { return a.first > b.first; };
    std::priority_queue<Node, std::vector<Node>, decltype(cmp)> open(cmp);

    int start_i = idx(start);
    int goal_i = idx(goal);
    g_score[start_i] = 0.0;
    open.emplace(0.0, start_i);

    const double sqrt2 = 1.41421356237;
    const Eigen::Vector2i dirs[8] = {
        {1, 0}, {-1, 0}, {0, 1}, {0, -1},
        {1, 1}, {-1, 1}, {1, -1}, {-1, -1}};

    while (!open.empty()) {
      int current_i = open.top().second;
      open.pop();

      if (current_i == goal_i) {
        reconstructPath(came_from, goal_i, width, path_pix);
        convertToWorldPath(path_pix, path_world, map);
        total_cost = g_score[goal_i];
        return true;
      }

      Eigen::Vector2i current(current_i % width, current_i / width);
      double g_curr = g_score[current_i];

      for (int k = 0; k < 8; ++k) {
        Eigen::Vector2i nb = current + dirs[k];
        if (!isInMap(nb, width, height)) continue;

        int ni = idx(nb);
        int cost = mapCost(nb, map);
        if (cost < 0) continue;  // obstacle

        double step = (k < 4) ? 1.0 : sqrt2;
        double tentative_g = g_curr + cost * step;
        if (tentative_g < g_score[ni]) {
          g_score[ni] = tentative_g;
          came_from[ni] = current_i;

          // Apply tie breaker here
          double h = heuristic(nb, goal);
          double f = tentative_g + tie_breaker_ * h;
          open.emplace(f, ni);
        }
      }
    }

    ROS_ERROR_STREAM("\033[31m[FuzzyAstar2D] Failed to find path from "
                     << start.transpose() << " to " << goal.transpose() << "\033[0m");
    return false;
  }

private:
  const nav_msgs::OccupancyGrid* map_ptr_ = nullptr;
  double tie_breaker_ = 1.001;

  inline bool isInMap(const Eigen::Vector2i& p, int w, int h) const {
    return p.x() >= 0 && p.y() >= 0 && p.x() < w && p.y() < h;
  }

  inline int mapCost(const Eigen::Vector2i& p, const nav_msgs::OccupancyGrid& map) const {
    int i = p.y() * map.info.width + p.x();
    if (i < 0 || i >= (int)map.data.size()) return -1;

    int8_t val = map.data[i];
    if (val >= 80) return 10;       // predicted/known obstacle: very expensive but traversable
    else if (val <= 40) return 1;  // free
    else return 2;                 // fuzzy/uncertain area
  }

  void reconstructPath(const std::vector<int>& came_from, int goal_i, int width,
                       std::vector<Eigen::Vector2i>& path_pix) const {
    path_pix.clear();
    int current_i = goal_i;
    while (current_i != -1) {
      int x = current_i % width;
      int y = current_i / width;
      path_pix.emplace_back(x, y);
      current_i = came_from[current_i];
    }
    std::reverse(path_pix.begin(), path_pix.end());
  }

  void convertToWorldPath(const std::vector<Eigen::Vector2i>& path_pix,
                          std::vector<Eigen::Vector3d>& path_world,
                          const nav_msgs::OccupancyGrid& map) const {
    path_world.clear();
    double resolution = map.info.resolution;
    double origin_x = map.info.origin.position.x;
    double origin_y = map.info.origin.position.y;

    path_world.reserve(path_pix.size());
    for (const auto& pix : path_pix) {
      double x = origin_x + pix.x() * resolution;
      double y = origin_y + pix.y() * resolution;
      path_world.emplace_back(x, y, 0.0);
    }
  }
};
