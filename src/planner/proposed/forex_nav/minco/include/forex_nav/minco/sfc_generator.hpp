/*
 * Safe Flight Corridor (SFC) Generator for 2D ground navigation
 * Generates axis-aligned bounding box (AABB) corridors from occupancy grid
 * Features: corridor expansion, merging adjacent corridors
 */

#ifndef SFC_GENERATOR_HPP
#define SFC_GENERATOR_HPP

#include <Eigen/Core>
#include <vector>
#include <unordered_set>
#include <nav_msgs/OccupancyGrid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <iostream>

namespace forex_nav {

// Axis-Aligned Bounding Box for 2D corridor
struct AABB2D {
    double x_min, x_max;
    double y_min, y_max;
    double z_min, z_max;  // Fixed height for ground robot
    
    AABB2D() : x_min(0), x_max(0), y_min(0), y_max(0), z_min(0), z_max(2.0) {}
    
    AABB2D(double xmin, double xmax, double ymin, double ymax, double zmin = 0.0, double zmax = 2.0)
        : x_min(xmin), x_max(xmax), y_min(ymin), y_max(ymax), z_min(zmin), z_max(zmax) {}
    
    // Convert to half-plane representation (H-rep): ax + by + cz + d <= 0
    Eigen::MatrixX4d toHRep() const {
        Eigen::MatrixX4d H(6, 4);
        H.row(0) << -1.0, 0.0, 0.0, x_min;
        H.row(1) << 1.0, 0.0, 0.0, -x_max;
        H.row(2) << 0.0, -1.0, 0.0, y_min;
        H.row(3) << 0.0, 1.0, 0.0, -y_max;
        H.row(4) << 0.0, 0.0, -1.0, z_min;
        H.row(5) << 0.0, 0.0, 1.0, -z_max;
        return H;
    }
    
    Eigen::Matrix3Xd toVRep() const {
        Eigen::Matrix3Xd V(3, 8);
        V.col(0) << x_max, y_max, z_max;
        V.col(1) << x_min, y_max, z_max;
        V.col(2) << x_max, y_min, z_max;
        V.col(3) << x_min, y_min, z_max;
        V.col(4) << x_max, y_max, z_min;
        V.col(5) << x_min, y_max, z_min;
        V.col(6) << x_max, y_min, z_min;
        V.col(7) << x_min, y_min, z_min;
        return V;
    }
    
    Eigen::Vector3d center() const {
        return Eigen::Vector3d(
            (x_min + x_max) * 0.5,
            (y_min + y_max) * 0.5,
            (z_min + z_max) * 0.5
        );
    }
    
    bool contains(const Eigen::Vector3d& pt, double margin = 0.0) const {
        return pt.x() >= x_min - margin && pt.x() <= x_max + margin &&
               pt.y() >= y_min - margin && pt.y() <= y_max + margin &&
               pt.z() >= z_min - margin && pt.z() <= z_max + margin;
    }
    
    bool overlaps(const AABB2D& other, double margin = 0.0) const {
        return !(x_max + margin < other.x_min || other.x_max + margin < x_min ||
                 y_max + margin < other.y_min || other.y_max + margin < y_min);
    }
    
    AABB2D intersection(const AABB2D& other) const {
        return AABB2D(
            std::max(x_min, other.x_min),
            std::min(x_max, other.x_max),
            std::max(y_min, other.y_min),
            std::min(y_max, other.y_max),
            std::max(z_min, other.z_min),
            std::min(z_max, other.z_max)
        );
    }
    
    bool isValid() const {
        return x_max > x_min && y_max > y_min && z_max > z_min;
    }
    
    double area() const {
        return (x_max - x_min) * (y_max - y_min);
    }
};

class SFCGenerator {
public:
    SFCGenerator() : resolution_(0.1), inflate_radius_(0.35), fixed_height_(1.0),
                     has_cloud_3d_(false), voxel_res_3d_(0.2) {}
    
    void setMap(const nav_msgs::OccupancyGrid::ConstPtr& map) {
        map_ = map;
        if (map_) {
            resolution_ = map_->info.resolution;
            origin_x_ = map_->info.origin.position.x;
            origin_y_ = map_->info.origin.position.y;
            width_ = map_->info.width;
            height_ = map_->info.height;
        }
    }
    
    // 设置3D占据点云用于3D走廊生成
    void setOccCloud3D(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, double voxel_res) {
        voxel_res_3d_ = voxel_res;
        occ_voxels_3d_.clear();
        if (!cloud || cloud->empty()) {
            has_cloud_3d_ = false;
            return;
        }
        for (const auto& pt : cloud->points) {
            // 只取占据体素 (intensity对应z高度, 但所有发布的点都是占据的)
            int gx = static_cast<int>(std::floor(pt.x / voxel_res_3d_));
            int gy = static_cast<int>(std::floor(pt.y / voxel_res_3d_));
            int gz = static_cast<int>(std::floor(pt.z / voxel_res_3d_));
            occ_voxels_3d_.insert(voxelKey(gx, gy, gz));
        }
        has_cloud_3d_ = true;
        std::cout << "[SFC] Set 3D occ cloud: " << occ_voxels_3d_.size() 
                  << " voxels, res=" << voxel_res_3d_ << std::endl;
    }
    
    void setInflateRadius(double radius) { inflate_radius_ = radius; }
    void setFixedHeight(double height) { fixed_height_ = height; }
    
    bool generateCorridors(
        const std::vector<Eigen::Vector3d>& path,
        std::vector<AABB2D>& corridors) {
        
        corridors.clear();
        
        if (!map_ || path.size() < 2) {
            std::cout << "[SFC] Error: Invalid map or path" << std::endl;
            return false;
        }
        
        std::vector<AABB2D> initial_corridors;
        for (size_t i = 0; i < path.size() - 1; ++i) {
            AABB2D corridor;
            if (generateSegmentCorridor(path[i], path[i + 1], corridor)) {
                initial_corridors.push_back(corridor);
            } else {
                corridor = createMinimalCorridor(path[i], path[i + 1]);
                initial_corridors.push_back(corridor);
            }
        }
        
        std::cout << "[SFC] Initial corridors: " << initial_corridors.size() << std::endl;
        
        corridors = mergeCorridors(initial_corridors, path);
        
        std::cout << "[SFC] After merging: " << corridors.size() << " corridors for " 
                  << path.size() << " waypoints" << std::endl;
        
        return !corridors.empty();
    }
    
    // 生成3D安全走廊: 在x/y/z六个方向扩展
    bool generateCorridors3D(
        const std::vector<Eigen::Vector3d>& path,
        std::vector<AABB2D>& corridors) {
        
        corridors.clear();
        
        if (!has_cloud_3d_ || path.size() < 2) {
            std::cout << "[SFC3D] Error: No 3D cloud or invalid path" << std::endl;
            return false;
        }
        
        std::vector<AABB2D> initial_corridors;
        for (size_t i = 0; i < path.size() - 1; ++i) {
            AABB2D corridor;
            if (generateSegmentCorridor3D(path[i], path[i + 1], corridor)) {
                initial_corridors.push_back(corridor);
            } else {
                corridor = createMinimalCorridor3D(path[i], path[i + 1]);
                initial_corridors.push_back(corridor);
            }
        }
        
        std::cout << "[SFC3D] Initial 3D corridors: " << initial_corridors.size() << std::endl;
        
        corridors = mergeCorridors3D(initial_corridors, path);
        
        std::cout << "[SFC3D] After merging: " << corridors.size() << " 3D corridors" << std::endl;
        
        return !corridors.empty();
    }

private:
    nav_msgs::OccupancyGrid::ConstPtr map_;
    double resolution_;
    double origin_x_, origin_y_;
    int width_, height_;
    double inflate_radius_;
    double fixed_height_;
    
    // 3D体素相关
    bool has_cloud_3d_;
    double voxel_res_3d_;
    std::unordered_set<int64_t> occ_voxels_3d_;
    
    static int64_t voxelKey(int i, int j, int k) {
        return ((int64_t)(i & 0x1FFFFF) << 42) | ((int64_t)(j & 0x1FFFFF) << 21) | (int64_t)(k & 0x1FFFFF);
    }
    
    bool isOccupied3D(double wx, double wy, double wz) const {
        int gx = static_cast<int>(std::floor(wx / voxel_res_3d_));
        int gy = static_cast<int>(std::floor(wy / voxel_res_3d_));
        int gz = static_cast<int>(std::floor(wz / voxel_res_3d_));
        return occ_voxels_3d_.count(voxelKey(gx, gy, gz)) > 0;
    }
    
    bool isFreeSpace3D(double wx, double wy, double wz) const {
        return !isOccupied3D(wx, wy, wz);
    }
    
    bool worldToMap(double wx, double wy, int& mx, int& my) const {
        mx = static_cast<int>((wx - origin_x_) / resolution_);
        my = static_cast<int>((wy - origin_y_) / resolution_);
        return mx >= 0 && mx < width_ && my >= 0 && my < height_;
    }
    
    bool isOccupied(int mx, int my) const {
        if (mx < 0 || mx >= width_ || my < 0 || my >= height_) {
            return true;
        }
        int idx = my * width_ + mx;
        return map_->data[idx] > 50 || map_->data[idx] < 0;
    }
    
    bool isFreeSpace(double wx, double wy) const {
        int mx, my;
        if (!worldToMap(wx, wy, mx, my)) {
            return false;
        }
        return !isOccupied(mx, my);
    }
    
    bool isCorridorFree(const AABB2D& corridor) const {
        const int samples = 5;
        double dx = (corridor.x_max - corridor.x_min) / samples;
        double dy = (corridor.y_max - corridor.y_min) / samples;
        
        for (int i = 0; i <= samples; ++i) {
            for (int j = 0; j <= samples; ++j) {
                double x = corridor.x_min + i * dx;
                double y = corridor.y_min + j * dy;
                if (!isFreeSpace(x, y)) {
                    return false;
                }
            }
        }
        return true;
    }
    
    bool generateSegmentCorridor(
        const Eigen::Vector3d& start,
        const Eigen::Vector3d& end,
        AABB2D& corridor) {
        
        double min_x = std::min(start.x(), end.x()) - inflate_radius_;
        double max_x = std::max(start.x(), end.x()) + inflate_radius_;
        double min_y = std::min(start.y(), end.y()) - inflate_radius_;
        double max_y = std::max(start.y(), end.y()) + inflate_radius_;
        
        const double expand_step = resolution_;
        const double max_expand = 3.0;
        const int check_samples = 30;
        
        for (double dx = 0; dx < max_expand; dx += expand_step) {
            bool can_expand = true;
            for (int i = 0; i <= check_samples; ++i) {
                double y = min_y + (max_y - min_y) * i / check_samples;
                if (!isFreeSpace(min_x - dx - expand_step, y)) {
                    can_expand = false;
                    break;
                }
            }
            if (!can_expand) break;
            min_x -= expand_step;
        }
        
        for (double dx = 0; dx < max_expand; dx += expand_step) {
            bool can_expand = true;
            for (int i = 0; i <= check_samples; ++i) {
                double y = min_y + (max_y - min_y) * i / check_samples;
                if (!isFreeSpace(max_x + dx + expand_step, y)) {
                    can_expand = false;
                    break;
                }
            }
            if (!can_expand) break;
            max_x += expand_step;
        }
        
        for (double dy = 0; dy < max_expand; dy += expand_step) {
            bool can_expand = true;
            for (int i = 0; i <= check_samples; ++i) {
                double x = min_x + (max_x - min_x) * i / check_samples;
                if (!isFreeSpace(x, min_y - dy - expand_step)) {
                    can_expand = false;
                    break;
                }
            }
            if (!can_expand) break;
            min_y -= expand_step;
        }
        
        for (double dy = 0; dy < max_expand; dy += expand_step) {
            bool can_expand = true;
            for (int i = 0; i <= check_samples; ++i) {
                double x = min_x + (max_x - min_x) * i / check_samples;
                if (!isFreeSpace(x, max_y + dy + expand_step)) {
                    can_expand = false;
                    break;
                }
            }
            if (!can_expand) break;
            max_y += expand_step;
        }
        
        const double safety_margin = inflate_radius_ * 0.5;
        min_x += safety_margin;
        max_x -= safety_margin;
        min_y += safety_margin;
        max_y -= safety_margin;
        
        if (max_x <= min_x || max_y <= min_y) {
            return false;
        }
        
        corridor = AABB2D(min_x, max_x, min_y, max_y, 0.0, fixed_height_ * 2);
        return true;
    }
    
    AABB2D createMinimalCorridor(
        const Eigen::Vector3d& start,
        const Eigen::Vector3d& end) {
        
        double min_x = std::min(start.x(), end.x()) - inflate_radius_ * 0.5;
        double max_x = std::max(start.x(), end.x()) + inflate_radius_ * 0.5;
        double min_y = std::min(start.y(), end.y()) - inflate_radius_ * 0.5;
        double max_y = std::max(start.y(), end.y()) + inflate_radius_ * 0.5;
        
        const double min_size = inflate_radius_;
        if (max_x - min_x < min_size) {
            double cx = (max_x + min_x) * 0.5;
            min_x = cx - min_size * 0.5;
            max_x = cx + min_size * 0.5;
        }
        if (max_y - min_y < min_size) {
            double cy = (max_y + min_y) * 0.5;
            min_y = cy - min_size * 0.5;
            max_y = cy + min_size * 0.5;
        }
        
        return AABB2D(min_x, max_x, min_y, max_y, 0.0, fixed_height_ * 2);
    }
    
    std::vector<AABB2D> mergeCorridors(
        const std::vector<AABB2D>& initial_corridors,
        const std::vector<Eigen::Vector3d>& path) {
        
        if (initial_corridors.empty()) return {};
        if (initial_corridors.size() == 1) return initial_corridors;
        
        std::vector<AABB2D> merged;
        AABB2D current = initial_corridors[0];
        size_t path_idx = 1;
        
        for (size_t i = 1; i < initial_corridors.size(); ++i) {
            const AABB2D& next = initial_corridors[i];
            
            const double overlap_margin = 0.1;
            if (current.overlaps(next, overlap_margin)) {
                AABB2D intersection = current.intersection(next);
                
                double min_intersection_size = inflate_radius_ * 0.5;
                if (intersection.isValid() && 
                    (intersection.x_max - intersection.x_min) > min_intersection_size &&
                    (intersection.y_max - intersection.y_min) > min_intersection_size) {
                    
                    AABB2D merged_candidate = intersection;
                    
                    bool all_points_inside = true;
                    for (size_t j = path_idx; j <= i + 1 && j < path.size(); ++j) {
                        if (!merged_candidate.contains(path[j], -0.1)) {
                            all_points_inside = false;
                            break;
                        }
                    }
                    
                    if (all_points_inside && isCorridorFree(merged_candidate)) {
                        current = merged_candidate;
                        std::cout << "[SFC] Merged corridors " << (i-1) << " and " << i << std::endl;
                        continue;
                    }
                }
            }
            
            merged.push_back(current);
            current = next;
            path_idx = i + 1;
        }
        
        merged.push_back(current);
        
        return merged;
    }
    
    // ==================== 3D走廊生成方法 ====================
    
    bool generateSegmentCorridor3D(
        const Eigen::Vector3d& start,
        const Eigen::Vector3d& end,
        AABB2D& corridor) {
        
        double min_x = std::min(start.x(), end.x()) - inflate_radius_;
        double max_x = std::max(start.x(), end.x()) + inflate_radius_;
        double min_y = std::min(start.y(), end.y()) - inflate_radius_;
        double max_y = std::max(start.y(), end.y()) + inflate_radius_;
        double min_z = std::min(start.z(), end.z()) - inflate_radius_;
        double max_z = std::max(start.z(), end.z()) + inflate_radius_;
        
        const double expand_step = voxel_res_3d_;
        const double max_expand = 3.0;
        const double max_expand_z = 2.0;
        const int check_samples = 8;
        
        // 向x负方向扩展
        for (double d = 0; d < max_expand; d += expand_step) {
            bool can = true;
            double test_x = min_x - expand_step;
            for (int iy = 0; iy <= check_samples && can; ++iy) {
                double y = min_y + (max_y - min_y) * iy / check_samples;
                for (int iz = 0; iz <= check_samples && can; ++iz) {
                    double z = min_z + (max_z - min_z) * iz / check_samples;
                    if (!isFreeSpace3D(test_x, y, z)) can = false;
                }
            }
            if (!can) break;
            min_x -= expand_step;
        }
        
        // 向x正方向扩展
        for (double d = 0; d < max_expand; d += expand_step) {
            bool can = true;
            double test_x = max_x + expand_step;
            for (int iy = 0; iy <= check_samples && can; ++iy) {
                double y = min_y + (max_y - min_y) * iy / check_samples;
                for (int iz = 0; iz <= check_samples && can; ++iz) {
                    double z = min_z + (max_z - min_z) * iz / check_samples;
                    if (!isFreeSpace3D(test_x, y, z)) can = false;
                }
            }
            if (!can) break;
            max_x += expand_step;
        }
        
        // 向y负方向扩展
        for (double d = 0; d < max_expand; d += expand_step) {
            bool can = true;
            double test_y = min_y - expand_step;
            for (int ix = 0; ix <= check_samples && can; ++ix) {
                double x = min_x + (max_x - min_x) * ix / check_samples;
                for (int iz = 0; iz <= check_samples && can; ++iz) {
                    double z = min_z + (max_z - min_z) * iz / check_samples;
                    if (!isFreeSpace3D(x, test_y, z)) can = false;
                }
            }
            if (!can) break;
            min_y -= expand_step;
        }
        
        // 向y正方向扩展
        for (double d = 0; d < max_expand; d += expand_step) {
            bool can = true;
            double test_y = max_y + expand_step;
            for (int ix = 0; ix <= check_samples && can; ++ix) {
                double x = min_x + (max_x - min_x) * ix / check_samples;
                for (int iz = 0; iz <= check_samples && can; ++iz) {
                    double z = min_z + (max_z - min_z) * iz / check_samples;
                    if (!isFreeSpace3D(x, test_y, z)) can = false;
                }
            }
            if (!can) break;
            max_y += expand_step;
        }
        
        // 向z负方向扩展
        for (double d = 0; d < max_expand_z; d += expand_step) {
            bool can = true;
            double test_z = min_z - expand_step;
            for (int ix = 0; ix <= check_samples && can; ++ix) {
                double x = min_x + (max_x - min_x) * ix / check_samples;
                for (int iy = 0; iy <= check_samples && can; ++iy) {
                    double y = min_y + (max_y - min_y) * iy / check_samples;
                    if (!isFreeSpace3D(x, y, test_z)) can = false;
                }
            }
            if (!can) break;
            min_z -= expand_step;
        }
        
        // 向z正方向扩展
        for (double d = 0; d < max_expand_z; d += expand_step) {
            bool can = true;
            double test_z = max_z + expand_step;
            for (int ix = 0; ix <= check_samples && can; ++ix) {
                double x = min_x + (max_x - min_x) * ix / check_samples;
                for (int iy = 0; iy <= check_samples && can; ++iy) {
                    double y = min_y + (max_y - min_y) * iy / check_samples;
                    if (!isFreeSpace3D(x, y, test_z)) can = false;
                }
            }
            if (!can) break;
            max_z += expand_step;
        }
        
        // 安全边距
        const double safety_margin = inflate_radius_ * 0.5;
        min_x += safety_margin;
        max_x -= safety_margin;
        min_y += safety_margin;
        max_y -= safety_margin;
        min_z += safety_margin;
        max_z -= safety_margin;
        
        if (max_x <= min_x || max_y <= min_y || max_z <= min_z) {
            return false;
        }
        
        corridor = AABB2D(min_x, max_x, min_y, max_y, min_z, max_z);
        return true;
    }
    
    AABB2D createMinimalCorridor3D(
        const Eigen::Vector3d& start,
        const Eigen::Vector3d& end) {
        
        double min_x = std::min(start.x(), end.x()) - inflate_radius_ * 0.5;
        double max_x = std::max(start.x(), end.x()) + inflate_radius_ * 0.5;
        double min_y = std::min(start.y(), end.y()) - inflate_radius_ * 0.5;
        double max_y = std::max(start.y(), end.y()) + inflate_radius_ * 0.5;
        double min_z = std::min(start.z(), end.z()) - inflate_radius_ * 0.5;
        double max_z = std::max(start.z(), end.z()) + inflate_radius_ * 0.5;
        
        const double min_size = inflate_radius_;
        if (max_x - min_x < min_size) {
            double cx = (max_x + min_x) * 0.5;
            min_x = cx - min_size * 0.5;
            max_x = cx + min_size * 0.5;
        }
        if (max_y - min_y < min_size) {
            double cy = (max_y + min_y) * 0.5;
            min_y = cy - min_size * 0.5;
            max_y = cy + min_size * 0.5;
        }
        if (max_z - min_z < min_size) {
            double cz = (max_z + min_z) * 0.5;
            min_z = cz - min_size * 0.5;
            max_z = cz + min_size * 0.5;
        }
        
        return AABB2D(min_x, max_x, min_y, max_y, min_z, max_z);
    }
    
    bool isCorridorFree3D(const AABB2D& corridor) const {
        const int samples = 4;
        double dx = (corridor.x_max - corridor.x_min) / samples;
        double dy = (corridor.y_max - corridor.y_min) / samples;
        double dz = (corridor.z_max - corridor.z_min) / samples;
        
        for (int i = 0; i <= samples; ++i) {
            for (int j = 0; j <= samples; ++j) {
                for (int k = 0; k <= samples; ++k) {
                    double x = corridor.x_min + i * dx;
                    double y = corridor.y_min + j * dy;
                    double z = corridor.z_min + k * dz;
                    if (!isFreeSpace3D(x, y, z)) return false;
                }
            }
        }
        return true;
    }
    
    std::vector<AABB2D> mergeCorridors3D(
        const std::vector<AABB2D>& initial_corridors,
        const std::vector<Eigen::Vector3d>& path) {
        
        if (initial_corridors.empty()) return {};
        if (initial_corridors.size() == 1) return initial_corridors;
        
        std::vector<AABB2D> merged;
        AABB2D current = initial_corridors[0];
        size_t path_idx = 1;
        
        for (size_t i = 1; i < initial_corridors.size(); ++i) {
            const AABB2D& next = initial_corridors[i];
            
            const double overlap_margin = 0.1;
            if (current.overlaps(next, overlap_margin)) {
                AABB2D inter = current.intersection(next);
                
                double min_size = inflate_radius_ * 0.5;
                if (inter.isValid() &&
                    (inter.x_max - inter.x_min) > min_size &&
                    (inter.y_max - inter.y_min) > min_size &&
                    (inter.z_max - inter.z_min) > min_size) {
                    
                    bool all_inside = true;
                    for (size_t j = path_idx; j <= i + 1 && j < path.size(); ++j) {
                        if (!inter.contains(path[j], -0.1)) {
                            all_inside = false;
                            break;
                        }
                    }
                    
                    if (all_inside && isCorridorFree3D(inter)) {
                        current = inter;
                        continue;
                    }
                }
            }
            
            merged.push_back(current);
            current = next;
            path_idx = i + 1;
        }
        
        merged.push_back(current);
        return merged;
    }
};

}  // namespace forex_nav

#endif  // SFC_GENERATOR_HPP
