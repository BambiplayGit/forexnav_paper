#ifndef ASTAR_2D_H_
#define ASTAR_2D_H_

#include <nav_msgs/OccupancyGrid.h>
#include <Eigen/Core>
#include <vector>
#include <queue>
#include <unordered_map>
#include <limits>
#include <cmath>

namespace forex_nav {

class Astar2D {
public:
  Astar2D();
  ~Astar2D();

  void setMap(const nav_msgs::OccupancyGrid::ConstPtr& map);
  void setResolution(double resolution);
  void setFixedHeight(double height);
  
  bool search(const Eigen::Vector3d& start, const Eigen::Vector3d& goal, 
              std::vector<Eigen::Vector3d>& path);
  
  void reset();
  
  // Public access to map checking functions
  bool isFree(int x, int y) const;
  bool isObstacle(int x, int y) const;
  Eigen::Vector2i worldToMap(const Eigen::Vector3d& world) const;

private:
  struct Node {
    int x, y;
    double g_score;
    double f_score;
    int parent_x, parent_y;
    
    bool operator>(const Node& other) const {
      return f_score > other.f_score;
    }
  };

  nav_msgs::OccupancyGrid::ConstPtr map_;
  double resolution_;
  double fixed_height_;
  
  // Helper functions
  bool isInMap(int x, int y) const;
  Eigen::Vector3d mapToWorld(int x, int y) const;
  double heuristic(int x1, int y1, int x2, int y2) const;
  void reconstructPath(const std::unordered_map<int, Node>& all_nodes,
                       int goal_x, int goal_y,
                       std::vector<Eigen::Vector3d>& path) const;
  int hash(int x, int y) const;
  
  // Get cost based on proximity to obstacles (keeps path away from walls)
  double getObstacleProximityCost(int x, int y) const;
};

}  // namespace forex_nav

#endif  // ASTAR_2D_H_
