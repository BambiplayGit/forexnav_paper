#include <ros/ros.h>

#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>
#include <nav_msgs/OccupancyGrid.h>
#include <visualization_msgs/MarkerArray.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <list>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Cell {
  int x{};
  int y{};
};

// Frontier cluster with world coordinates
struct FrontierCluster {
  std::vector<Cell> cells;
  std::vector<Eigen::Vector2d> cells_world;  // World coordinates
  Eigen::Vector2d average{0.0, 0.0};
  Eigen::Vector2d box_min, box_max;
  Eigen::Vector2d principal_direction{1.0, 0.0};  // First principal component direction
};

inline bool inBounds(int x, int y, int w, int h) { return x >= 0 && y >= 0 && x < w && y < h; }

inline int idx(int x, int y, int w) { return y * w + x; }

inline double wrapYaw(double a) {
  while (a > M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}

// Bresenham-like stepping on grid cells, returns true if line is free
bool raycastVisible(
    const nav_msgs::OccupancyGrid& grid, int x0, int y0, int x1, int y1, bool unknown_blocks) {
  const int w = static_cast<int>(grid.info.width);
  const int h = static_cast<int>(grid.info.height);
  if (!inBounds(x0, y0, w, h) || !inBounds(x1, y1, w, h)) return false;

  int dx = std::abs(x1 - x0);
  int dy = std::abs(y1 - y0);
  int sx = (x0 < x1) ? 1 : -1;
  int sy = (y0 < y1) ? 1 : -1;
  int err = dx - dy;

  int x = x0;
  int y = y0;
  while (true) {
    const int8_t v = grid.data[idx(x, y, w)];
    if (v == 100) return false;
    if (unknown_blocks && v < 0) return false;

    if (x == x1 && y == y1) break;
    const int e2 = 2 * err;
    if (e2 > -dy) {
      err -= dy;
      x += sx;
    }
    if (e2 < dx) {
      err += dx;
      y += sy;
    }
    if (!inBounds(x, y, w, h)) return false;
  }
  return true;
}

}  // namespace

class ViewpointPlannerNode {
 public:
  ViewpointPlannerNode(ros::NodeHandle& nh, ros::NodeHandle& pnh) {
    // Topics
    std::string grid_topic;
    pnh.param<std::string>("grid_topic", grid_topic, "/local_sensing/occupancy_grid");

    // Frontier extraction
    pnh.param<bool>("use_8_connected_frontier", use_8_connected_frontier_, true);
    pnh.param<int>("cluster_min", cluster_min_, 20);
    pnh.param<double>("cluster_size_xy", cluster_size_xy_, 2.0);
    pnh.param<double>("cluster_size_z", cluster_size_z_, 10.0);
    pnh.param<int>("down_sample", down_sample_, 2);

    // Viewpoint sampling
    pnh.param<double>("candidate_rmin", candidate_rmin_, 0.4);
    pnh.param<double>("candidate_rmax", candidate_rmax_, 1.5);
    pnh.param<int>("candidate_rnum", candidate_rnum_, 3);
    pnh.param<double>("candidate_dphi", candidate_dphi_, 15.0 * M_PI / 180.0);
    pnh.param<double>("min_candidate_dist", min_candidate_dist_, 0.4);
    pnh.param<double>("min_candidate_clearance", min_candidate_clearance_, 0.21);

    // Viewpoint safety and scoring
    pnh.param<double>("min_viewpoint_distance", min_viewpoint_distance_, 0.5);
    pnh.param<double>("viewpoint_distance_weight", viewpoint_distance_weight_, 10.0);
    pnh.param<int>("min_visib_num", min_visib_num_, 10);
    pnh.param<double>("min_view_finish_fraction", min_view_finish_fraction_, 0.2);

    // Visibility/FOV
    pnh.param<double>("top_angle", top_angle_, 0.785398);
    pnh.param<double>("left_angle", left_angle_, 1.047198);
    pnh.param<double>("right_angle", right_angle_, 1.047198);
    pnh.param<double>("max_dist", max_dist_, 9.5);
    pnh.param<double>("vis_dist", vis_dist_, 1.0);
    pnh.param<bool>("unknown_blocks_visibility", unknown_blocks_visibility_, true);

    // 2D mode
    pnh.param<bool>("use_2d_mode", use_2d_mode_, true);
    pnh.param<double>("fixed_height_2d", fixed_height_2d_, 1.0);
    pnh.param<bool>("use_strict_2d_mode", use_strict_2d_mode_, false);

    // Incremental update
    pnh.param<double>("update_region_inflate", update_region_inflate_, 1.0);

    // Output
    pnh.param<int>("max_viewpoints_total", max_viewpoints_total_, 50);
    pnh.param<double>("publish_rate", publish_rate_, 2.0);

    grid_sub_ = nh.subscribe(grid_topic, 1, &ViewpointPlannerNode::onGrid, this);

    vps_pub_ = nh.advertise<geometry_msgs::PoseArray>("/planner/viewpoints", 10);
    markers_pub_ = nh.advertise<visualization_msgs::MarkerArray>("/planning_vis/viewpoints", 10);
    frontier_pub_ = nh.advertise<visualization_msgs::MarkerArray>("/planning_vis/frontiers", 10);

    const int period_ms = std::max(1, static_cast<int>(1000.0 / std::max(0.1, publish_rate_)));
    timer_ = nh.createTimer(ros::Duration(period_ms / 1000.0), &ViewpointPlannerNode::tick, this);

    ROS_INFO("Subscribing grid: %s", grid_topic.c_str());
  }

 private:
  // Parameters
  bool use_8_connected_frontier_;
  int cluster_min_;
  double cluster_size_xy_;
  double cluster_size_z_;
  int down_sample_;
  double candidate_rmin_;
  double candidate_rmax_;
  int candidate_rnum_;
  double candidate_dphi_;
  double min_candidate_dist_;
  double min_candidate_clearance_;
  double min_viewpoint_distance_;
  double viewpoint_distance_weight_;
  int min_visib_num_;
  double min_view_finish_fraction_;
  double top_angle_;
  double left_angle_;
  double right_angle_;
  double max_dist_;
  double vis_dist_;
  bool unknown_blocks_visibility_;
  bool use_2d_mode_;
  double fixed_height_2d_;
  bool use_strict_2d_mode_;
  double update_region_inflate_;
  int max_viewpoints_total_;
  double publish_rate_;

  void onGrid(const nav_msgs::OccupancyGrid::ConstPtr& msg) {
    last_grid_ = *msg;
    has_grid_ = true;
    grid_updated_ = true;
  }

  void tick(const ros::TimerEvent&) {
    if (!has_grid_) return;
    if (last_grid_.info.width == 0 || last_grid_.info.height == 0) return;
    if (last_grid_.data.empty()) return;

    // Compute updated region
    int update_min_x, update_min_y, update_max_x, update_max_y;
    computeUpdatedRegion(update_min_x, update_min_y, update_max_x, update_max_y);

    // Search frontiers incrementally
    searchFrontiersIncremental(last_grid_, update_min_x, update_min_y, update_max_x, update_max_y);

    // Split large frontiers
    splitLargeFrontiers();

    // Filter by min cluster size
    std::vector<FrontierCluster> valid_frontiers;
    for (auto& ftr : frontiers_) {
      if (static_cast<int>(ftr.cells.size()) >= cluster_min_) {
        valid_frontiers.push_back(std::move(ftr));
      }
    }
    frontiers_ = std::move(valid_frontiers);

    // Sort by size
    std::sort(frontiers_.begin(), frontiers_.end(),
              [](const auto& a, const auto& b) { return a.cells.size() > b.cells.size(); });

    publishFrontiers(frontiers_);

    const auto viewpoints = computeViewpoints(last_grid_, frontiers_);
    publishViewpoints(viewpoints);

    // Save current grid for next comparison
    prev_grid_ = last_grid_;
    has_prev_grid_ = true;
    grid_updated_ = false;
  }

  void computeUpdatedRegion(int& min_x, int& min_y, int& max_x, int& max_y) {
    const int w = static_cast<int>(last_grid_.info.width);
    const int h = static_cast<int>(last_grid_.info.height);
    const double res = last_grid_.info.resolution;
    const int inflate_cells = static_cast<int>(std::ceil(update_region_inflate_ / res));

    if (!has_prev_grid_ || prev_grid_.data.size() != last_grid_.data.size() ||
        prev_grid_.info.width != last_grid_.info.width ||
        prev_grid_.info.height != last_grid_.info.height) {
      min_x = 0;
      min_y = 0;
      max_x = w - 1;
      max_y = h - 1;
      return;
    }

    min_x = w;
    min_y = h;
    max_x = -1;
    max_y = -1;

    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const int id = idx(x, y, w);
        if (last_grid_.data[id] != prev_grid_.data[id]) {
          min_x = std::min(min_x, x);
          min_y = std::min(min_y, y);
          max_x = std::max(max_x, x);
          max_y = std::max(max_y, y);
        }
      }
    }

    if (max_x < 0) {
      min_x = 0;
      min_y = 0;
      max_x = w - 1;
      max_y = h - 1;
      return;
    }

    min_x = std::max(0, min_x - inflate_cells);
    min_y = std::max(0, min_y - inflate_cells);
    max_x = std::min(w - 1, max_x + inflate_cells);
    max_y = std::min(h - 1, max_y + inflate_cells);
  }

  void searchFrontiersIncremental(const nav_msgs::OccupancyGrid& grid,
                                   int search_min_x, int search_min_y,
                                   int search_max_x, int search_max_y) {
    const int w = static_cast<int>(grid.info.width);
    const int h = static_cast<int>(grid.info.height);
    const double res = grid.info.resolution;
    const double ox = grid.info.origin.position.x;
    const double oy = grid.info.origin.position.y;

    auto isFree = [&](int x, int y) {
      if (!inBounds(x, y, w, h)) return false;
      return grid.data[idx(x, y, w)] == 0;
    };
    auto isUnknown = [&](int x, int y) {
      if (!inBounds(x, y, w, h)) return false;
      return grid.data[idx(x, y, w)] < 0;
    };
    auto gridToWorld = [&](int gx, int gy) -> Eigen::Vector2d {
      return Eigen::Vector2d(ox + (static_cast<double>(gx) + 0.5) * res,
                             oy + (static_cast<double>(gy) + 0.5) * res);
    };

    const int nx8[8] = {1, 1, 1, 0, 0, -1, -1, -1};
    const int ny8[8] = {1, 0, -1, 1, -1, 1, 0, -1};
    const int nx4[4] = {1, -1, 0, 0};
    const int ny4[4] = {0, 0, 1, -1};

    auto isFrontierCell = [&](int x, int y) -> bool {
      if (!isFree(x, y)) return false;
      for (int k = 0; k < 8; ++k) {
        const int xx = x + nx8[k];
        const int yy = y + ny8[k];
        if (inBounds(xx, yy, w, h) && isUnknown(xx, yy)) {
          return true;
        }
      }
      return false;
    };

    auto haveOverlap = [](const Eigen::Vector2d& box_min, const Eigen::Vector2d& box_max,
                          double update_min_x, double update_min_y,
                          double update_max_x, double update_max_y) -> bool {
      return !(box_max.x() < update_min_x || box_min.x() > update_max_x ||
               box_max.y() < update_min_y || box_min.y() > update_max_y);
    };

    const double search_min_wx = ox + search_min_x * res;
    const double search_min_wy = oy + search_min_y * res;
    const double search_max_wx = ox + (search_max_x + 1) * res;
    const double search_max_wy = oy + (search_max_y + 1) * res;

    auto isFrontierChanged = [&](const FrontierCluster& ftr) -> bool {
      for (const auto& cell : ftr.cells) {
        if (!isFrontierCell(cell.x, cell.y)) return true;
      }
      return false;
    };

    std::vector<FrontierCluster> kept_frontiers;
    for (auto& ftr : frontiers_) {
      if (haveOverlap(ftr.box_min, ftr.box_max, search_min_wx, search_min_wy, search_max_wx, search_max_wy) &&
          isFrontierChanged(ftr)) {
        for (const auto& cell : ftr.cells) {
          if (inBounds(cell.x, cell.y, w, h)) {
            frontier_flag_[idx(cell.x, cell.y, w)] = 0;
          }
        }
      } else {
        kept_frontiers.push_back(std::move(ftr));
      }
    }
    frontiers_ = std::move(kept_frontiers);

    if (frontier_flag_.size() != static_cast<size_t>(w * h)) {
      frontier_flag_.assign(static_cast<size_t>(w * h), 0);
    }

    for (int y = search_min_y; y <= search_max_y; ++y) {
      for (int x = search_min_x; x <= search_max_x; ++x) {
        if (!inBounds(x, y, w, h)) continue;
        const int id = idx(x, y, w);
        if (frontier_flag_[id] != 0) continue;
        if (!isFrontierCell(x, y)) continue;

        std::deque<Cell> queue;
        FrontierCluster cluster;
        queue.push_back({x, y});
        frontier_flag_[id] = 1;

        while (!queue.empty()) {
          const Cell c = queue.front();
          queue.pop_front();
          cluster.cells.push_back(c);
          cluster.cells_world.push_back(gridToWorld(c.x, c.y));

          const int ncnt = use_8_connected_frontier_ ? 8 : 4;
          for (int k = 0; k < ncnt; ++k) {
            const int xx = c.x + (use_8_connected_frontier_ ? nx8[k] : nx4[k]);
            const int yy = c.y + (use_8_connected_frontier_ ? ny8[k] : ny4[k]);
            if (!inBounds(xx, yy, w, h)) continue;
            const int nid = idx(xx, yy, w);
            if (frontier_flag_[nid] != 0) continue;
            if (!isFrontierCell(xx, yy)) continue;
            frontier_flag_[nid] = 1;
            queue.push_back({xx, yy});
          }
        }

        if (!cluster.cells.empty()) {
          computeFrontierInfo(cluster);
          frontiers_.push_back(std::move(cluster));
        }
      }
    }
  }

  void computeFrontierInfo(FrontierCluster& ftr) {
    if (ftr.cells_world.empty()) return;

    ftr.average.setZero();
    ftr.box_min = ftr.cells_world.front();
    ftr.box_max = ftr.cells_world.front();

    for (const auto& cell : ftr.cells_world) {
      ftr.average += cell;
      ftr.box_min.x() = std::min(ftr.box_min.x(), cell.x());
      ftr.box_min.y() = std::min(ftr.box_min.y(), cell.y());
      ftr.box_max.x() = std::max(ftr.box_max.x(), cell.x());
      ftr.box_max.y() = std::max(ftr.box_max.y(), cell.y());
    }
    ftr.average /= static_cast<double>(ftr.cells_world.size());

    // PCA: compute covariance matrix and extract first principal component
    if (ftr.cells_world.size() >= 2) {
      Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
      for (const auto& cell : ftr.cells_world) {
        Eigen::Vector2d diff = cell - ftr.average;
        cov += diff * diff.transpose();
      }
      cov /= static_cast<double>(ftr.cells_world.size());

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(cov);
      int max_idx = (solver.eigenvalues()(0) > solver.eigenvalues()(1)) ? 0 : 1;
      ftr.principal_direction = solver.eigenvectors().col(max_idx);
    }
  }

  void splitLargeFrontiers() {
    std::vector<FrontierCluster> result;
    for (auto& ftr : frontiers_) {
      std::list<FrontierCluster> splits;
      if (splitHorizontally(ftr, cluster_size_xy_, splits)) {
        for (auto& s : splits) {
          result.push_back(std::move(s));
        }
      } else {
        result.push_back(std::move(ftr));
      }
    }
    frontiers_ = std::move(result);
  }

  bool splitHorizontally(const FrontierCluster& frontier, double max_size,
                         std::list<FrontierCluster>& splits) {
    if (frontier.cells_world.empty()) return false;

    const Eigen::Vector2d mean = frontier.average;
    bool need_split = false;
    for (const auto& cell : frontier.cells_world) {
      if ((cell - mean).norm() > max_size) {
        need_split = true;
        break;
      }
    }
    if (!need_split) return false;

    Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
    for (const auto& cell : frontier.cells_world) {
      Eigen::Vector2d diff = cell - mean;
      cov += diff * diff.transpose();
    }
    cov /= static_cast<double>(frontier.cells_world.size());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(cov);
    Eigen::Vector2d eigenvalues = solver.eigenvalues();
    Eigen::Matrix2d eigenvectors = solver.eigenvectors();

    int max_idx = (eigenvalues(0) > eigenvalues(1)) ? 0 : 1;
    Eigen::Vector2d first_pc = eigenvectors.col(max_idx);

    FrontierCluster ftr1, ftr2;
    for (size_t i = 0; i < frontier.cells.size(); ++i) {
      const Eigen::Vector2d& cell_world = frontier.cells_world[i];
      if ((cell_world - mean).dot(first_pc) >= 0) {
        ftr1.cells.push_back(frontier.cells[i]);
        ftr1.cells_world.push_back(cell_world);
      } else {
        ftr2.cells.push_back(frontier.cells[i]);
        ftr2.cells_world.push_back(cell_world);
      }
    }

    if (!ftr1.cells.empty()) computeFrontierInfo(ftr1);
    if (!ftr2.cells.empty()) computeFrontierInfo(ftr2);

    std::list<FrontierCluster> splits1, splits2;
    if (!ftr1.cells.empty()) {
      if (splitHorizontally(ftr1, max_size, splits1)) {
        splits.insert(splits.end(), splits1.begin(), splits1.end());
      } else {
        splits.push_back(std::move(ftr1));
      }
    }
    if (!ftr2.cells.empty()) {
      if (splitHorizontally(ftr2, max_size, splits2)) {
        splits.insert(splits.end(), splits2.begin(), splits2.end());
      } else {
        splits.push_back(std::move(ftr2));
      }
    }

    return true;
  }

  struct Viewpoint2D {
    double x{};
    double y{};
    double yaw{};
    int visible_cells{};
    double obstacle_distance{};
  };

  std::vector<Viewpoint2D> computeViewpoints(
      const nav_msgs::OccupancyGrid& grid, const std::vector<FrontierCluster>& frontiers) const {
    const int w = static_cast<int>(grid.info.width);
    const int h = static_cast<int>(grid.info.height);
    const double res = grid.info.resolution;
    const double ox = grid.info.origin.position.x;
    const double oy = grid.info.origin.position.y;

    auto worldToGrid = [&](double wx, double wy) -> std::pair<int, int> {
      const int gx = static_cast<int>(std::floor((wx - ox) / res));
      const int gy = static_cast<int>(std::floor((wy - oy) / res));
      return {gx, gy};
    };
    auto gridToWorld = [&](int gx, int gy) -> std::pair<double, double> {
      const double wx = ox + (static_cast<double>(gx) + 0.5) * res;
      const double wy = oy + (static_cast<double>(gy) + 0.5) * res;
      return {wx, wy};
    };
    auto isFree = [&](int gx, int gy) {
      if (!inBounds(gx, gy, w, h)) return false;
      return grid.data[idx(gx, gy, w)] == 0;
    };
    auto isUnknown = [&](int gx, int gy) {
      if (!inBounds(gx, gy, w, h)) return false;
      return grid.data[idx(gx, gy, w)] < 0;
    };
    auto isOccupied = [&](int gx, int gy) {
      if (!inBounds(gx, gy, w, h)) return false;
      return grid.data[idx(gx, gy, w)] == 100;
    };

    auto computeObstacleDistance = [&](int gx, int gy) -> double {
      const int max_search = static_cast<int>(std::ceil(max_dist_ / res));
      double min_dist = max_dist_;
      for (int dx = -max_search; dx <= max_search; ++dx) {
        for (int dy = -max_search; dy <= max_search; ++dy) {
          const int nx = gx + dx;
          const int ny = gy + dy;
          if (isOccupied(nx, ny)) {
            const double dist = std::hypot(static_cast<double>(dx) * res, static_cast<double>(dy) * res);
            if (dist < min_dist) min_dist = dist;
          }
        }
      }
      return min_dist;
    };

    auto isNearUnknown = [&](int gx, int gy) -> bool {
      const int vox_num = static_cast<int>(std::floor(min_candidate_clearance_ / res));
      for (int dx = -vox_num; dx <= vox_num; ++dx) {
        for (int dy = -vox_num; dy <= vox_num; ++dy) {
          const int nx = gx + dx;
          const int ny = gy + dy;
          if (isUnknown(nx, ny)) return true;
        }
      }
      return false;
    };

    std::vector<Viewpoint2D> out;
    out.reserve(std::min(128, std::max(1, max_viewpoints_total_)));

    const double dr = (candidate_rnum_ <= 1) ? 0.0 : (candidate_rmax_ - candidate_rmin_) / static_cast<double>(candidate_rnum_ - 1);

    for (const auto& cluster : frontiers) {
      if (static_cast<int>(out.size()) >= max_viewpoints_total_) break;
      if (cluster.cells.empty()) continue;

      const double cx = cluster.average.x();
      const double cy = cluster.average.y();

      Viewpoint2D best{};
      best.visible_cells = -1;

      for (int ri = 0; ri < candidate_rnum_; ++ri) {
        const double r = candidate_rmin_ + dr * static_cast<double>(ri);
        for (double phi = -M_PI; phi < M_PI; phi += candidate_dphi_) {
          const double vx = cx + r * std::cos(phi);
          const double vy = cy + r * std::sin(phi);
          auto [vgx, vgy] = worldToGrid(vx, vy);
          if (!isFree(vgx, vgy)) continue;

          if (isNearUnknown(vgx, vgy)) continue;

          const double obs_dist = computeObstacleDistance(vgx, vgy);
          if (obs_dist < min_viewpoint_distance_) continue;

          bool too_close = false;
          for (const auto& prev : out) {
            const double dx = prev.x - vx;
            const double dy = prev.y - vy;
            if (std::hypot(dx, dy) < min_candidate_dist_) {
              too_close = true;
              break;
            }
          }
          if (too_close) continue;

          // Viewpoint faces perpendicular to first PC, toward unknown region
          Eigen::Vector2d normal(-cluster.principal_direction.y(),
                                  cluster.principal_direction.x());
          // Disambiguate: pick the direction closest to viewpoint-to-center
          // (viewpoint is in free space, center is at frontier boundary toward unknown)
          Eigen::Vector2d vp_to_center(cx - vx, cy - vy);
          if (normal.dot(vp_to_center) < 0) normal = -normal;
          const double yaw = wrapYaw(std::atan2(normal.y(), normal.x()));

          int vis = 0;
          for (int i = 0; i < static_cast<int>(cluster.cells.size()); i += down_sample_) {
            const auto& fc = cluster.cells[i];
            auto [fwx, fwy] = gridToWorld(fc.x, fc.y);
            const double dist_to_frontier = std::hypot(fwx - vx, fwy - vy);
            if (dist_to_frontier > max_dist_) continue;
            if (raycastVisible(grid, vgx, vgy, fc.x, fc.y, unknown_blocks_visibility_)) vis++;
          }

          if (vis >= min_visib_num_) {
            const double score = static_cast<double>(vis) + viewpoint_distance_weight_ *
                                 std::max(0.0, obs_dist - min_viewpoint_distance_);
            const double best_score = static_cast<double>(best.visible_cells) + viewpoint_distance_weight_ *
                                     std::max(0.0, best.obstacle_distance - min_viewpoint_distance_);
            if (score > best_score || best.visible_cells < 0) {
              best = {vx, vy, yaw, vis, obs_dist};
            }
          }
        }
      }

      if (best.visible_cells >= min_visib_num_) out.push_back(best);
    }

    std::sort(out.begin(), out.end(), [this](const auto& a, const auto& b) {
      const double score_a = static_cast<double>(a.visible_cells) + viewpoint_distance_weight_ *
                            std::max(0.0, a.obstacle_distance - min_viewpoint_distance_);
      const double score_b = static_cast<double>(b.visible_cells) + viewpoint_distance_weight_ *
                            std::max(0.0, b.obstacle_distance - min_viewpoint_distance_);
      return score_a > score_b;
    });
    if (static_cast<int>(out.size()) > max_viewpoints_total_) out.resize(max_viewpoints_total_);
    return out;
  }

  // Convert HSV (h in [0,360), s,v in [0,1]) to RGB
  static void hsvToRgb(double h, double s, double v, float& r, float& g, float& b) {
    const int hi = static_cast<int>(std::floor(h / 60.0)) % 6;
    const double f = h / 60.0 - std::floor(h / 60.0);
    const double p = v * (1.0 - s);
    const double q = v * (1.0 - f * s);
    const double t = v * (1.0 - (1.0 - f) * s);
    switch (hi) {
      case 0: r = static_cast<float>(v); g = static_cast<float>(t); b = static_cast<float>(p); break;
      case 1: r = static_cast<float>(q); g = static_cast<float>(v); b = static_cast<float>(p); break;
      case 2: r = static_cast<float>(p); g = static_cast<float>(v); b = static_cast<float>(t); break;
      case 3: r = static_cast<float>(p); g = static_cast<float>(q); b = static_cast<float>(v); break;
      case 4: r = static_cast<float>(t); g = static_cast<float>(p); b = static_cast<float>(v); break;
      default: r = static_cast<float>(v); g = static_cast<float>(p); b = static_cast<float>(q); break;
    }
  }

  void publishFrontiers(const std::vector<FrontierCluster>& frontiers) {
    if (!has_grid_) return;
    visualization_msgs::MarkerArray arr;

    // Delete all previous frontier markers
    visualization_msgs::Marker del;
    del.header = last_grid_.header;
    del.ns = "frontiers";
    del.action = visualization_msgs::Marker::DELETEALL;
    arr.markers.push_back(del);

    const double res = last_grid_.info.resolution;
    const double ox = last_grid_.info.origin.position.x;
    const double oy = last_grid_.info.origin.position.y;
    const float pt_size = std::max(0.02f, static_cast<float>(res));
    const int n_clusters = static_cast<int>(frontiers.size());

    // One marker per cluster with a distinct color from HSV palette
    for (int ci = 0; ci < n_clusters; ++ci) {
      const auto& cl = frontiers[ci];

      visualization_msgs::Marker pts;
      pts.header = last_grid_.header;
      pts.ns = "frontiers";
      pts.id = ci;
      pts.type = visualization_msgs::Marker::POINTS;
      pts.action = visualization_msgs::Marker::ADD;
      pts.pose.orientation.w = 1.0;
      pts.scale.x = pt_size;
      pts.scale.y = pt_size;

      // Evenly spaced hue with high saturation and value
      const double hue = (n_clusters > 1)
          ? 360.0 * static_cast<double>(ci) / static_cast<double>(n_clusters)
          : 30.0;  // orange if only one cluster
      hsvToRgb(hue, 0.9, 0.95, pts.color.r, pts.color.g, pts.color.b);
      pts.color.a = 0.9f;

      const int stride = std::max(1, static_cast<int>(cl.cells.size()) / 400);
      for (int i = 0; i < static_cast<int>(cl.cells.size()); i += stride) {
        geometry_msgs::Point p;
        p.x = ox + (static_cast<double>(cl.cells[i].x) + 0.5) * res;
        p.y = oy + (static_cast<double>(cl.cells[i].y) + 0.5) * res;
        p.z = 0.0;
        pts.points.push_back(p);
      }

      arr.markers.push_back(pts);
    }

    frontier_pub_.publish(arr);
  }

  void publishViewpoints(const std::vector<Viewpoint2D>& vps) {
    if (!has_grid_) return;

    geometry_msgs::PoseArray pa;
    pa.header = last_grid_.header;

    visualization_msgs::MarkerArray marr;

    visualization_msgs::Marker del;
    del.header = last_grid_.header;
    del.ns = "vp_points";
    del.action = visualization_msgs::Marker::DELETEALL;
    marr.markers.push_back(del);

    del.ns = "vp_arrows";
    marr.markers.push_back(del);

    del.ns = "vp_frustum";
    marr.markers.push_back(del);

    visualization_msgs::Marker pts;
    pts.header = last_grid_.header;
    pts.ns = "vp_points";
    pts.id = 0;
    pts.type = visualization_msgs::Marker::SPHERE_LIST;
    pts.action = visualization_msgs::Marker::ADD;
    pts.pose.orientation.w = 1.0;
    pts.scale.x = 0.15;
    pts.scale.y = 0.15;
    pts.scale.z = 0.15;
    pts.color.r = 0.8f;
    pts.color.g = 0.2f;
    pts.color.b = 1.0f;
    pts.color.a = 1.0f;

    const double vp_z = fixed_height_2d_;  // Viewpoint marker height aligned to planning_height

    for (size_t i = 0; i < vps.size(); ++i) {
      geometry_msgs::Pose pose;
      pose.position.x = vps[i].x;
      pose.position.y = vps[i].y;
      pose.position.z = vp_z;
      pose.orientation.w = std::cos(vps[i].yaw * 0.5);
      pose.orientation.z = std::sin(vps[i].yaw * 0.5);
      pa.poses.push_back(pose);

      geometry_msgs::Point p;
      p.x = vps[i].x;
      p.y = vps[i].y;
      p.z = vp_z;
      pts.points.push_back(p);

      const double z = vp_z;
      const double py_d = 0.8;
      const double cx = vps[i].x + py_d * std::cos(vps[i].yaw);
      const double cy = vps[i].y + py_d * std::sin(vps[i].yaw);
      const double rx = -std::sin(vps[i].yaw);
      const double ry = std::cos(vps[i].yaw);
      geometry_msgs::Point apex;
      apex.x = vps[i].x;
      apex.y = vps[i].y;
      apex.z = z;
      geometry_msgs::Point c1, c2, c3, c4;
      c1.x = cx + py_d * rx; c1.y = cy + py_d * ry; c1.z = z + py_d;
      c2.x = cx - py_d * rx; c2.y = cy - py_d * ry; c2.z = z + py_d;
      c3.x = cx - py_d * rx; c3.y = cy - py_d * ry; c3.z = z - py_d;
      c4.x = cx + py_d * rx; c4.y = cy + py_d * ry; c4.z = z - py_d;

      visualization_msgs::Marker frustum;
      frustum.header = last_grid_.header;
      frustum.ns = "vp_frustum";
      frustum.id = static_cast<int>(i);
      frustum.type = visualization_msgs::Marker::LINE_LIST;
      frustum.action = visualization_msgs::Marker::ADD;
      frustum.pose.orientation.w = 1.0;
      frustum.scale.x = 0.03;
      frustum.color.r = 1.0f;
      frustum.color.g = 0.3f;
      frustum.color.b = 0.3f;
      frustum.color.a = 0.9f;
      frustum.points = {apex, c1, apex, c2, apex, c3, apex, c4,
                        c1, c2, c2, c3, c3, c4, c4, c1};
      marr.markers.push_back(frustum);
    }

    marr.markers.push_back(pts);

    markers_pub_.publish(marr);
    vps_pub_.publish(pa);
  }

  ros::Subscriber grid_sub_;
  ros::Publisher vps_pub_;
  ros::Publisher markers_pub_;
  ros::Publisher frontier_pub_;
  ros::Timer timer_;

  nav_msgs::OccupancyGrid last_grid_;
  nav_msgs::OccupancyGrid prev_grid_;
  bool has_grid_{false};
  bool has_prev_grid_{false};
  bool grid_updated_{false};

  std::vector<FrontierCluster> frontiers_;
  std::vector<uint8_t> frontier_flag_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "gen_viewpoint_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  
  ViewpointPlannerNode node(nh, pnh);
  
  ros::spin();
  return 0;
}
