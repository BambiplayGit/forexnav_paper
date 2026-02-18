#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/OccupancyGrid.h>
#include <visualization_msgs/Marker.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <unordered_set>
#include <unordered_map>
#include <memory>

namespace {

int ij_to_key(int i, int j) { return i * 100000 + j; }

// 3D体素key: 每个轴21位, 支持索引到~2M
int64_t ijk_to_key_3d(int i, int j, int k) {
  return ((int64_t)i << 42) | ((int64_t)j << 21) | (int64_t)k;
}

void key_to_ijk_3d(int64_t key, int& i, int& j, int& k) {
  k = (int)(key & 0x1FFFFF);
  j = (int)((key >> 21) & 0x1FFFFF);
  i = (int)((key >> 42) & 0x1FFFFF);
}

}  // namespace

class LocalSensingNode {
 public:
  LocalSensingNode(ros::NodeHandle& nh, ros::NodeHandle& pnh) {
    // ---- 通用地图参数 ----
    double x_size, y_size;
    pnh.param<double>("map/x_size", x_size, 132.0);
    pnh.param<double>("map/y_size", y_size, 132.0);
    pnh.param<double>("map/resolution", occ_res_, 0.1);
    pnh.param<double>("occupancy_resolution", occ_res_, 0.1);

    pnh.param<bool>("publish_cloud", publish_cloud_, true);
    pnh.param<bool>("use_relative_cloud", use_relative_cloud_, false);
    pnh.param<std::string>("cloud_frame_id", cloud_frame_id_, "world");

    // 固定参数
    occ_mark_free_ = true;
    occ_accumulate_ = true;

    gl_xl_ = -x_size / 2.0;
    gl_yl_ = -y_size / 2.0;
    occ_w_ = (int)(x_size / occ_res_);
    occ_h_ = (int)(y_size / occ_res_);
    occ_size_ = occ_w_ * occ_h_;
    occ_accum_.resize(occ_size_, -1);

    // ---- 2D平面扫描参数 (仅 enable_3d_occ=false 时使用) ----
    pnh.param<double>("fov_angle", fov_angle_, 1.5708);
    pnh.param<double>("max_range", max_range_, 10.0);
    double sensing_rate;
    pnh.param<double>("sensing_rate", sensing_rate, 60.0);

    // ---- 3D深度感知开关与参数 ----
    pnh.param<bool>("enable_3d_occ", enable_3d_occ_, false);
    pnh.param<double>("cam_h_fov", cam_h_fov_, 1.22);
    pnh.param<double>("cam_v_fov", cam_v_fov_, 1.01);
    pnh.param<double>("cam_pitch", cam_pitch_, 0.0);
    pnh.param<double>("cam_max_range", cam_max_range_, 10.0);
    pnh.param<double>("depth_h_fov", depth_h_fov_, cam_h_fov_);
    pnh.param<double>("fov_vis_h_fov", fov_vis_h_fov_, cam_h_fov_);
    pnh.param<double>("voxel_resolution", voxel_res_, 0.1);
    double z_size;
    pnh.param<double>("map/z_size", z_size, 5.0);
    double cam_sensing_rate;
    pnh.param<double>("cam_sensing_rate", cam_sensing_rate, 10.0);
    pnh.param<bool>("occ3d_accumulate", occ3d_accumulate_, true);
    pnh.param<double>("occ3d_radius_limit", occ3d_radius_limit_, 20.0);
    pnh.param<bool>("publish_free_voxels", publish_free_voxels_, false);

    // ---- 深度渲染参数 (仅 enable_3d_occ=true 时使用) ----
    pnh.param<int>("cam_width", cam_width_, 320);
    pnh.param<int>("cam_height", cam_height_, 240);
    pnh.param<double>("depth_point_radius", depth_point_radius_, 0.0873);
    pnh.param<double>("slice_half_height", slice_half_height_, 0.5);

    // ---- 膨胀参数 (用于规划器的安全边距) ----
    pnh.param<double>("inflate_radius", inflate_radius_, 0.3);
    pnh.param<double>("inflate_publish_interval", inflate_publish_interval_, 0.1);

    // ---- FOV可视化参数 ----
    pnh.param<bool>("publish_fov_marker", publish_fov_marker_, true);
    pnh.param<double>("fov_marker_rate", fov_marker_rate_, 3.0);
    pnh.param<double>("fov_marker_scale", fov_marker_scale_, 0.6);
    pnh.param<double>("fov_marker_line_width", fov_marker_line_width_, 0.02);

    // ---- 3D地图尺寸与虚拟相机内参 (始终初始化, 深度管线总是运行) ----
    gl_zl_ = -z_size / 2.0;
    vox_w_ = (int)(x_size / voxel_res_);
    vox_h_ = (int)(y_size / voxel_res_);
    vox_d_ = (int)(z_size / voxel_res_);

    // 从水平/垂直FOV推算针孔相机内参
    cam_fx_ = cam_width_ / (2.0 * std::tan(cam_h_fov_ / 2.0));
    cam_fy_ = cam_height_ / (2.0 * std::tan(cam_v_fov_ / 2.0));
    cam_cx_ = cam_width_ / 2.0;
    cam_cy_ = cam_height_ / 2.0;

    // 构建相机→机体变换矩阵 (含俯仰角)
    buildCam02Body();

    // 预计算膨胀半径对应的体素格数
    inflate_r_voxels_ = (inflate_radius_ > 1e-6) ? (int)(inflate_radius_ / voxel_res_ + 0.5) : 0;

    // 预计算膨胀半径对应的2D栅格格数 (两种模式通用)
    inflate_r_2d_ = (inflate_radius_ > 1e-6) ? (int)(inflate_radius_ / occ_res_ + 0.5) : 0;

    last_inflate_time_ = ros::Time(0);

    // ---- 订阅器 ----
    map_sub_ = nh.subscribe("global_map", 1, &LocalSensingNode::mapCallback, this);
    odom_sub_ = nh.subscribe("odometry", 50, &LocalSensingNode::odomCallback, this);

    // ---- 发布器 ----
    occ_pub_ = nh.advertise<nav_msgs::OccupancyGrid>("occupancy_grid", 1);
    occ_image_pub_ = nh.advertise<sensor_msgs::Image>("occupancy_grid_image", 1);
    cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("cloud", 1);
    // 3D occupancy (两种模式均发布)
    occ3d_pub_ = nh.advertise<sensor_msgs::PointCloud2>("occupancy_3d", 1);
    // 膨胀版 (供规划器使用)
    if (inflate_radius_ > 1e-6) {
      occ_inflate_pub_ = nh.advertise<nav_msgs::OccupancyGrid>("occupancy_grid_inflate", 1);
      occ3d_inflate_pub_ = nh.advertise<sensor_msgs::PointCloud2>("occupancy_3d_inflate", 1);
    }

    // ---- 定时器: 深度管线始终运行 (发布 occupancy_3d) ----
    {
      double timer_period = 1.0 / cam_sensing_rate;
      timer_depth_ = nh.createTimer(ros::Duration(timer_period),
                                     &LocalSensingNode::depthSensingCallback, this);
      int depth_n_passes = std::max(1, (int)std::ceil(depth_h_fov_ / cam_h_fov_));
      ROS_INFO("[LocalSensing] Depth pipeline: %dx%d fx=%.1f fy=%.1f "
               "max_range=%.1f voxel=%.2f rate=%.1fHz "
               "depth_h_fov=%.2f(%d passes) vis_fov=%.2f",
               cam_width_, cam_height_, cam_fx_, cam_fy_,
               cam_max_range_, voxel_res_, cam_sensing_rate,
               depth_h_fov_, depth_n_passes, fov_vis_h_fov_);
      ROS_INFO("[LocalSensing] pitch=%.2f rad slice_half=%.2f point_r=%.4f",
               cam_pitch_, slice_half_height_, depth_point_radius_);
      if (inflate_radius_ > 1e-6)
        ROS_INFO("[LocalSensing] inflate_radius=%.2f (%d voxels, %d 2d cells)",
                 inflate_radius_, inflate_r_voxels_, inflate_r_2d_);
    }

    // ---- 2D模式: 同时运行2D平扫 (发布 occupancy_grid) ----
    if (!enable_3d_occ_) {
      double timer_period = 1.0 / sensing_rate;
      timer_ = nh.createTimer(ros::Duration(timer_period),
                               &LocalSensingNode::sensingTimerCallback, this);
      ROS_INFO("[LocalSensing] 2D scan also running: fov=%.2f rad max_range=%.1f", fov_angle_, max_range_);
    }

    // FOV可视化 (两种模式均可用)
    if (publish_fov_marker_) {
      fov_marker_pub_ = nh.advertise<visualization_msgs::Marker>("fov_marker", 1);
      double fov_timer_period = 1.0 / fov_marker_rate_;
      fov_timer_ = nh.createTimer(ros::Duration(fov_timer_period),
                                  &LocalSensingNode::fovMarkerTimerCallback, this);
      ROS_INFO("[LocalSensing] FOV marker at %.1f Hz", fov_marker_rate_);
    }

    ROS_INFO("[LocalSensing] use_relative_cloud=%s cloud_frame_id=%s",
             use_relative_cloud_ ? "true" : "false", cloud_frame_id_.c_str());
  }

 private:
  // ================================================================
  //  相机→机体变换矩阵 (含俯仰角)
  // ================================================================
  void buildCam02Body() {
    // ROS机体坐标系: x-前, y-左, z-上
    // OpenCV相机坐标系: x-右, y-下, z-前
    //
    // 无俯仰时:
    //   cam_x(右) → body(-y)  cam_y(下) → body(-z)  cam_z(前) → body(x)
    //
    // 俯仰角θ: 正值=仰视, 负值=俯视
    // 等价于绕body y轴旋转 Ry(-θ) 后再应用基础变换
    double ct = std::cos(cam_pitch_);
    double st = std::sin(cam_pitch_);

    cam02body_ = Eigen::Matrix4d::Identity();
    // 列0: camera x (右) 在机体坐标系中的方向 [0, -1, 0]
    cam02body_(0, 0) =  0.0;
    cam02body_(1, 0) = -1.0;
    cam02body_(2, 0) =  0.0;
    // 列1: camera y (下) 在机体坐标系中的方向 (受pitch影响)
    cam02body_(0, 1) =  st;
    cam02body_(1, 1) =  0.0;
    cam02body_(2, 1) = -ct;
    // 列2: camera z (前) 在机体坐标系中的方向 (受pitch影响)
    cam02body_(0, 2) =  ct;
    cam02body_(1, 2) =  0.0;
    cam02body_(2, 2) =  st;
    // 平移为零 (相机安装在机体原点)
  }

  // ================================================================
  //  Map & Odom 回调
  // ================================================================
  void mapCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
    if (has_map_) return;

    pcl::fromROSMsg(*msg, full_map_cloud_);
    has_map_ = true;
    ROS_INFO("[LocalSensing] Loaded map with %zu points", full_map_cloud_.points.size());

    // 仅2D平扫模式需要预计算障碍物集
    if (!enable_3d_occ_) {
      updateObstacleMap();
    }
  }

  void odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    bool z_changed = false;
    if (has_odom_) {
      double old_z = robot_z_;
      robot_z_ = msg->pose.pose.position.z;
      if (std::abs(old_z - robot_z_) > 0.2) z_changed = true;
    } else {
      has_odom_ = true;
      robot_z_ = msg->pose.pose.position.z;
      z_changed = true;
    }

    last_odom_stamp_ = msg->header.stamp;
    robot_x_ = msg->pose.pose.position.x;
    robot_y_ = msg->pose.pose.position.y;

    // 保存完整四元数 (深度管线需要)
    robot_quat_ = Eigen::Quaterniond(
        msg->pose.pose.orientation.w,
        msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z);
    Eigen::Matrix3d R = robot_quat_.toRotationMatrix();
    robot_yaw_ = std::atan2(R(1, 0), R(0, 0));

    // 仅2D平扫模式需要更新障碍物集
    if (!enable_3d_occ_ && z_changed && has_map_) {
      updateObstacleMap();
    }
  }

  // ================================================================
  //  2D 障碍物预计算 (仅 enable_3d_occ=false 时使用)
  // ================================================================
  void updateObstacleMap() {
    obstacle_2d_.clear();
    double z_tolerance = 0.2;
    double target_z = has_odom_ ? robot_z_ : 1.0;

    for (const auto& pt : full_map_cloud_.points) {
      if (std::abs(pt.z - target_z) > z_tolerance) continue;
      int gx = (int)((pt.x - gl_xl_) / occ_res_);
      int gy = (int)((pt.y - gl_yl_) / occ_res_);
      if (gx >= 0 && gx < occ_w_ && gy >= 0 && gy < occ_h_)
        obstacle_2d_.insert(ij_to_key(gx, gy));
    }
  }

  // ================================================================
  //  原有 2D 平面扫描 (enable_3d_occ=false 时使用, 保持不变)
  // ================================================================
  void sensingTimerCallback(const ros::TimerEvent&) {
    if (!has_map_ || !has_odom_) return;
    if (last_odom_stamp_.toSec() == 0.0) return;

    nav_msgs::OccupancyGrid grid;
    grid.header.stamp = last_odom_stamp_;
    grid.header.frame_id = "world";
    grid.info.resolution = occ_res_;
    grid.info.width = occ_w_;
    grid.info.height = occ_h_;
    grid.info.origin.position.x = gl_xl_;
    grid.info.origin.position.y = gl_yl_;
    grid.info.origin.orientation.w = 1.0;
    grid.data.resize(occ_size_);

    if (occ_initialized_) {
      for (int i = 0; i < occ_size_; i++) grid.data[i] = occ_accum_[i];
    } else {
      for (int i = 0; i < occ_size_; i++) grid.data[i] = -1;
      occ_accum_.resize(occ_size_, -1);
      occ_initialized_ = true;
    }

    double half_fov = fov_angle_ / 2.0;
    double max_angle_step = 0.5 * occ_res_ / max_range_;
    double angle_step = std::min(0.002, max_angle_step);
    int num_rays = (int)(fov_angle_ / angle_step) + 1;

    pcl::PointCloud<pcl::PointXYZ> hit_cloud;
    hit_cloud.reserve(num_rays * 100);

    for (int r = 0; r < num_rays; r++) {
      double angle = (num_rays > 1) ?
          robot_yaw_ - half_fov + (double)r / (double)(num_rays - 1) * fov_angle_ :
          robot_yaw_;
      double cos_a = std::cos(angle);
      double sin_a = std::sin(angle);

      double step_size = occ_res_ * 0.3;
      int max_steps = (int)(max_range_ / step_size) + 1;

      bool hit_obstacle = false;
      int last_gx = -1, last_gy = -1;
      for (int i = 0; i < max_steps && !hit_obstacle; i++) {
        double dist = i * step_size;
        if (dist > max_range_) break;

        double wx = robot_x_ + cos_a * dist;
        double wy = robot_y_ + sin_a * dist;

        int gx = (int)((wx - gl_xl_) / occ_res_);
        int gy = (int)((wy - gl_yl_) / occ_res_);

        if (gx < 0 || gx >= occ_w_ || gy < 0 || gy >= occ_h_) break;
        if (gx == last_gx && gy == last_gy) continue;
        last_gx = gx;
        last_gy = gy;

        int idx = gy * occ_w_ + gx;
        int key = ij_to_key(gx, gy);

        if (obstacle_2d_.count(key)) {
          for (int di = -1; di <= 1; di++) {
            for (int dj = -1; dj <= 1; dj++) {
              int ni = gx + di;
              int nj = gy + dj;
              if (ni >= 0 && ni < occ_w_ && nj >= 0 && nj < occ_h_) {
                int nidx = nj * occ_w_ + ni;
                grid.data[nidx] = 100;
              }
            }
          }
          if (publish_cloud_) {
            pcl::PointXYZ pt;
            if (use_relative_cloud_) {
              double dx = wx - robot_x_;
              double dy = wy - robot_y_;
              double cos_yaw = std::cos(-robot_yaw_);
              double sin_yaw = std::sin(-robot_yaw_);
              pt.x = dx * cos_yaw - dy * sin_yaw;
              pt.y = dx * sin_yaw + dy * cos_yaw;
              pt.z = 0;
            } else {
              pt.x = wx;
              pt.y = wy;
              pt.z = robot_z_;
            }
            hit_cloud.points.push_back(pt);
          }
          hit_obstacle = true;
        } else {
          if (grid.data[idx] != 100) grid.data[idx] = 0;
        }
      }
    }

    for (int i = 0; i < occ_size_; i++) occ_accum_[i] = grid.data[i];
    occ_pub_.publish(grid);
    publishOccupancyGridImage(grid);

    // 发布膨胀版2D栅格 (节流: 大地图膨胀开销大, 按10Hz发布)
    if (inflate_radius_ > 1e-6) {
      ros::Time now = ros::Time::now();
      if ((now - last_inflate_time_).toSec() >= inflate_publish_interval_) {
        last_inflate_time_ = now;
        inflateAndPublish2DGrid(grid);
      }
    }

    if (publish_cloud_ && !hit_cloud.points.empty()) {
      hit_cloud.width = hit_cloud.points.size();
      hit_cloud.height = 1;
      hit_cloud.is_dense = true;
      sensor_msgs::PointCloud2 cloud_msg;
      pcl::toROSMsg(hit_cloud, cloud_msg);
      cloud_msg.header.stamp = last_odom_stamp_;
      cloud_msg.header.frame_id = cloud_frame_id_;
      cloud_pub_.publish(cloud_msg);
    }

  }

  // ================================================================
  //  深度渲染管线: 深度图 → 三维体素 → occupancy_3d
  //  (enable_3d_occ=true 时替代2D平扫和旧3D光线追踪)
  // ================================================================
  void depthSensingCallback(const ros::TimerEvent&) {
    if (!has_map_ || !has_odom_) return;
    if (last_odom_stamp_.toSec() == 0.0) return;

    // 非累积模式每帧清空 (在多pass循环前统一清空)
    if (!occ3d_accumulate_) {
      occ_3d_accum_.clear();
      occ_3d_inflate_.clear();
    } else if (occ3d_radius_limit_ > 1e-6) {
      // 累积模式: 裁剪超出半径的体素以控制内存 (性能优化)
      pruneOcc3DByRadius();
    }

    // ---- 多pass渲染: depth_h_fov > cam_h_fov 时, 多次旋转相机覆盖全FOV ----
    int n_passes = std::max(1, (int)std::ceil(depth_h_fov_ / cam_h_fov_));
    double pass_step = depth_h_fov_ / n_passes;
    double start_offset = -(depth_h_fov_ - pass_step) / 2.0;

    Eigen::Matrix3d R_body = robot_quat_.toRotationMatrix();
    pcl::PointCloud<pcl::PointXYZ> all_hit_cloud;

    for (int pass = 0; pass < n_passes; pass++) {
      double delta_yaw = start_offset + pass * pass_step;

      // ---- 1. 计算相机外参 (含yaw偏移) ----
      Eigen::Matrix4d body_pose = Eigen::Matrix4d::Identity();
      if (n_passes > 1) {
        Eigen::Matrix3d R_yaw = Eigen::AngleAxisd(delta_yaw,
                                    Eigen::Vector3d::UnitZ()).toRotationMatrix();
        body_pose.block<3, 3>(0, 0) = R_yaw * R_body;
      } else {
        body_pose.block<3, 3>(0, 0) = R_body;
      }
      body_pose(0, 3) = robot_x_;
      body_pose(1, 3) = robot_y_;
      body_pose(2, 3) = robot_z_;

      // cam2world: 相机坐标 → 世界坐标
      Eigen::Matrix4d cam2world = body_pose * cam02body_;
      // Tcw: 世界坐标 → 相机坐标
      Eigen::Matrix4d Tcw = cam2world.inverse();
      Eigen::Matrix3d Rcw = Tcw.block<3, 3>(0, 0);
      Eigen::Vector3d tcw = Tcw.block<3, 1>(0, 3);
      Eigen::Vector3d cam_pos = cam2world.block<3, 1>(0, 3);
      Eigen::Matrix3d Rwc = cam2world.block<3, 3>(0, 0);

      // ---- 2. 渲染深度图 ----
      std::vector<float> depth_buf(cam_width_ * cam_height_, 0.0f);
      renderDepthBuffer(Rcw, tcw, cam_pos, depth_buf);

      // ---- 3. 从深度图构建三维体素地图 ----
      pcl::PointCloud<pcl::PointXYZ> hit_cloud;
      buildVoxelsFromDepth(depth_buf, cam_pos, Rwc, hit_cloud);

      all_hit_cloud += hit_cloud;
    }

    // ---- 4. 发布三维体素地图 (始终发布) ----
    publish3DOccupancy();

    // ---- 5. 膨胀版3D体素 (始终发布) ----
    if (inflate_radius_ > 1e-6) {
      publish3DOccupancyInflated();
    }

    // ---- 6. 仅 enable_3d_occ 模式: 三维切片→二维栅格 + 点云发布 ----
    //     (2D模式下, 2D栅格由 sensingTimerCallback 独立生成)
    if (enable_3d_occ_) {
      publishSlicedOccupancyGrid();

      if (inflate_radius_ > 1e-6) {
        ros::Time now = ros::Time::now();
        if ((now - last_inflate_time_).toSec() >= inflate_publish_interval_) {
          last_inflate_time_ = now;
          inflateAndPublish2DGrid(last_slice_grid_);
        }
      }

      if (publish_cloud_ && !all_hit_cloud.points.empty()) {
        all_hit_cloud.width = all_hit_cloud.points.size();
        all_hit_cloud.height = 1;
        all_hit_cloud.is_dense = true;
        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(all_hit_cloud, cloud_msg);
        cloud_msg.header.stamp = last_odom_stamp_;
        cloud_msg.header.frame_id = cloud_frame_id_;
        cloud_pub_.publish(cloud_msg);
      }
    }
  }

  // ----------------------------------------------------------------
  //  深度缓冲渲染: 将全局点云投影到虚拟深度相机
  // ----------------------------------------------------------------
  void renderDepthBuffer(const Eigen::Matrix3d& Rcw,
                          const Eigen::Vector3d& tcw,
                          const Eigen::Vector3d& cam_pos,
                          std::vector<float>& depth_buf) {
    double range_sq = cam_max_range_ * cam_max_range_;

    for (const auto& pt : full_map_cloud_.points) {
      Eigen::Vector3d pw(pt.x, pt.y, pt.z);
      // 距离过远的点跳过
      if ((cam_pos - pw).squaredNorm() > range_sq) continue;

      // 变换到相机坐标系
      Eigen::Vector3d pc = Rcw * pw + tcw;
      // 在相机后方的点不可见
      if (pc[2] <= 0.0) continue;

      // 针孔投影 → 像素坐标
      float u = (float)(pc[0] / pc[2] * cam_fx_ + cam_cx_);
      float v = (float)(pc[1] / pc[2] * cam_fy_ + cam_cy_);
      if (u < 0 || u >= cam_width_ || v < 0 || v >= cam_height_) continue;

      float depth = (float)pc[2];

      // 根据距离膨胀投影点 (模拟真实深度传感器的点面积)
      int r = (int)(depth_point_radius_ * cam_fx_ / depth + 0.5);
      r = std::max(r, 0);
      int min_u = std::max((int)(u - r), 0);
      int max_u = std::min((int)(u + r), cam_width_ - 1);
      int min_v = std::max((int)(v - r), 0);
      int max_v = std::min((int)(v + r), cam_height_ - 1);

      // Z-buffer: 保留最近深度
      for (int tu = min_u; tu <= max_u; tu++) {
        for (int tv = min_v; tv <= max_v; tv++) {
          int idx = tv * cam_width_ + tu;
          float val = depth_buf[idx];
          if (val < 1e-3f) {
            depth_buf[idx] = depth;
          } else {
            depth_buf[idx] = std::min(val, depth);
          }
        }
      }
    }
  }

  // ----------------------------------------------------------------
  //  从深度图构建三维体素: 占据 + 空闲标记
  // ----------------------------------------------------------------
  void buildVoxelsFromDepth(const std::vector<float>& depth_buf,
                             const Eigen::Vector3d& cam_pos,
                             const Eigen::Matrix3d& Rwc,
                             pcl::PointCloud<pcl::PointXYZ>& hit_cloud) {
    // 注: 非累积模式的清空已移至 depthSensingCallback (支持多pass渲染)

    // 像素采样步长 (不需要逐像素处理, 平衡精度与效率)
    int step = std::max(1, std::min(cam_width_, cam_height_) / 120);
    double voxel_step = voxel_res_ * 0.5;

    for (int v = 0; v < cam_height_; v += step) {
      for (int u = 0; u < cam_width_; u += step) {
        int idx = v * cam_width_ + u;
        float depth = depth_buf[idx];

        // 计算该像素的射线方向 (相机坐标系 → 世界坐标系)
        Eigen::Vector3d pc_dir;
        pc_dir[0] = (u - cam_cx_) / cam_fx_;
        pc_dir[1] = (v - cam_cy_) / cam_fy_;
        pc_dir[2] = 1.0;
        pc_dir.normalize();
        Eigen::Vector3d dir_world = Rwc * pc_dir;

        if (depth > 1e-3f && depth <= cam_max_range_) {
          // ---- 有深度: 碰到障碍物 ----

          // 反投影到世界坐标
          Eigen::Vector3d pc;
          pc[0] = (u - cam_cx_) / cam_fx_ * depth;
          pc[1] = (v - cam_cy_) / cam_fy_ * depth;
          pc[2] = depth;
          Eigen::Vector3d pw = Rwc * pc + cam_pos;

          // 标记占据体素
          markVoxelOccupied(pw);

          // 收集命中点用于发布点云
          if (publish_cloud_) {
            pcl::PointXYZ pt;
            if (use_relative_cloud_) {
              double dx = pw.x() - robot_x_;
              double dy = pw.y() - robot_y_;
              double dz = pw.z() - robot_z_;
              double cos_yaw = std::cos(-robot_yaw_);
              double sin_yaw = std::sin(-robot_yaw_);
              pt.x = dx * cos_yaw - dy * sin_yaw;
              pt.y = dx * sin_yaw + dy * cos_yaw;
              pt.z = dz;
            } else {
              pt.x = pw.x();
              pt.y = pw.y();
              pt.z = pw.z();
            }
            hit_cloud.points.push_back(pt);
          }

          // 射线路径上标记空闲体素 (从相机到障碍物前方)
          double total_dist = (pw - cam_pos).norm();
          rayTraceFree(cam_pos, dir_world, total_dist - voxel_res_, voxel_step);
        } else {
          // ---- 无深度: 射线范围内无障碍, 全部标记空闲 ----
          rayTraceFree(cam_pos, dir_world, cam_max_range_, voxel_step);
        }
      }
    }
  }

  // 标记单个体素为占据, 同时膨胀到邻域
  void markVoxelOccupied(const Eigen::Vector3d& pw) {
    int gx = (int)((pw.x() - gl_xl_) / voxel_res_);
    int gy = (int)((pw.y() - gl_yl_) / voxel_res_);
    int gz = (int)((pw.z() - gl_zl_) / voxel_res_);
    if (gx < 0 || gx >= vox_w_ || gy < 0 || gy >= vox_h_ ||
        gz < 0 || gz >= vox_d_)
      return;

    // 原始占据
    occ_3d_accum_[ijk_to_key_3d(gx, gy, gz)] = 100;

    // 膨胀: 在球形邻域内标记
    if (inflate_r_voxels_ > 0) {
      int r = inflate_r_voxels_;
      int r2 = r * r;
      for (int dx = -r; dx <= r; dx++) {
        for (int dy = -r; dy <= r; dy++) {
          for (int dz = -r; dz <= r; dz++) {
            if (dx * dx + dy * dy + dz * dz > r2) continue;
            int nx = gx + dx, ny = gy + dy, nz = gz + dz;
            if (nx >= 0 && nx < vox_w_ && ny >= 0 && ny < vox_h_ &&
                nz >= 0 && nz < vox_d_) {
              occ_3d_inflate_.insert(ijk_to_key_3d(nx, ny, nz));
            }
          }
        }
      }
    }
  }

  // 沿射线标记空闲体素 (跳过重复体素, 不覆盖已知占据)
  void rayTraceFree(const Eigen::Vector3d& origin,
                     const Eigen::Vector3d& dir,
                     double max_dist,
                     double step) {
    if (max_dist <= 0) return;
    int last_gx = -1, last_gy = -1, last_gz = -1;
    int max_steps = (int)(max_dist / step) + 1;

    for (int s = 0; s < max_steps; s++) {
      double d = s * step;
      if (d > max_dist) break;

      Eigen::Vector3d p = origin + dir * d;
      int gx = (int)((p.x() - gl_xl_) / voxel_res_);
      int gy = (int)((p.y() - gl_yl_) / voxel_res_);
      int gz = (int)((p.z() - gl_zl_) / voxel_res_);

      if (gx < 0 || gx >= vox_w_ || gy < 0 || gy >= vox_h_ ||
          gz < 0 || gz >= vox_d_)
        break;

      // 跳过重复体素
      if (gx == last_gx && gy == last_gy && gz == last_gz) continue;
      last_gx = gx; last_gy = gy; last_gz = gz;

      int64_t key = ijk_to_key_3d(gx, gy, gz);
      auto it = occ_3d_accum_.find(key);
      if (it == occ_3d_accum_.end() || it->second != 100) {
        occ_3d_accum_[key] = 0;  // 空闲
      }
    }
  }

  // ----------------------------------------------------------------
  //  裁剪超出机器人半径的3D体素 (控制内存, 避免随探索无界增长)
  // ----------------------------------------------------------------
  void pruneOcc3DByRadius() {
    if (occ3d_radius_limit_ <= 1e-6) return;
    const double r2 = occ3d_radius_limit_ * occ3d_radius_limit_;

    auto it = occ_3d_accum_.begin();
    while (it != occ_3d_accum_.end()) {
      int gi, gj, gk;
      key_to_ijk_3d(it->first, gi, gj, gk);
      double vx = gl_xl_ + (gi + 0.5) * voxel_res_;
      double vy = gl_yl_ + (gj + 0.5) * voxel_res_;
      double dx = vx - robot_x_;
      double dy = vy - robot_y_;
      if (dx * dx + dy * dy > r2) {
        it = occ_3d_accum_.erase(it);
      } else {
        ++it;
      }
    }

    auto it2 = occ_3d_inflate_.begin();
    while (it2 != occ_3d_inflate_.end()) {
      int gi, gj, gk;
      key_to_ijk_3d(*it2, gi, gj, gk);
      double vx = gl_xl_ + (gi + 0.5) * voxel_res_;
      double vy = gl_yl_ + (gj + 0.5) * voxel_res_;
      double dx = vx - robot_x_;
      double dy = vy - robot_y_;
      if (dx * dx + dy * dy > r2) {
        it2 = occ_3d_inflate_.erase(it2);
      } else {
        ++it2;
      }
    }
  }

  // ----------------------------------------------------------------
  //  发布三维体素占据地图
  // ----------------------------------------------------------------
  void publish3DOccupancy() {
    pcl::PointCloud<pcl::PointXYZI> occ_cloud;
    occ_cloud.reserve(occ_3d_accum_.size());

    for (const auto& kv : occ_3d_accum_) {
      if (!publish_free_voxels_ && kv.second == 0) continue;

      int gi, gj, gk;
      key_to_ijk_3d(kv.first, gi, gj, gk);

      pcl::PointXYZI pt;
      pt.x = gl_xl_ + (gi + 0.5) * voxel_res_;
      pt.y = gl_yl_ + (gj + 0.5) * voxel_res_;
      pt.z = gl_zl_ + (gk + 0.5) * voxel_res_;
      // intensity = z高度, RViz中用Intensity着色
      pt.intensity = pt.z;
      occ_cloud.points.push_back(pt);
    }

    if (occ_cloud.points.empty()) return;

    occ_cloud.width = occ_cloud.points.size();
    occ_cloud.height = 1;
    occ_cloud.is_dense = true;

    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(occ_cloud, msg);
    msg.header.stamp = last_odom_stamp_;
    msg.header.frame_id = "world";
    occ3d_pub_.publish(msg);
  }

  // ----------------------------------------------------------------
  //  三维体素切片 → 二维占据栅格
  // ----------------------------------------------------------------
  void publishSlicedOccupancyGrid() {
    nav_msgs::OccupancyGrid grid;
    grid.header.stamp = last_odom_stamp_;
    grid.header.frame_id = "world";
    grid.info.resolution = occ_res_;
    grid.info.width = occ_w_;
    grid.info.height = occ_h_;
    grid.info.origin.position.x = gl_xl_;
    grid.info.origin.position.y = gl_yl_;
    grid.info.origin.orientation.w = 1.0;
    grid.data.resize(occ_size_);

    // 从累积状态开始
    if (occ_initialized_) {
      for (int i = 0; i < occ_size_; i++) grid.data[i] = occ_accum_[i];
    } else {
      for (int i = 0; i < occ_size_; i++) grid.data[i] = -1;
      occ_initialized_ = true;
    }

    // Z切片范围
    double z_min = robot_z_ - slice_half_height_;
    double z_max = robot_z_ + slice_half_height_;

    // 遍历所有已知三维体素, 将z范围内的投影到二维
    for (const auto& kv : occ_3d_accum_) {
      int gi, gj, gk;
      key_to_ijk_3d(kv.first, gi, gj, gk);

      double vz = gl_zl_ + (gk + 0.5) * voxel_res_;
      if (vz < z_min || vz > z_max) continue;

      // 一个三维体素可能覆盖多个二维栅格 (当 voxel_res > occ_res 时)
      double vx_min = gl_xl_ + gi * voxel_res_;
      double vx_max = vx_min + voxel_res_;
      double vy_min = gl_yl_ + gj * voxel_res_;
      double vy_max = vy_min + voxel_res_;

      int gx_min_2d = std::max(0, (int)((vx_min - gl_xl_) / occ_res_));
      int gx_max_2d = std::min(occ_w_ - 1, (int)((vx_max - gl_xl_) / occ_res_));
      int gy_min_2d = std::max(0, (int)((vy_min - gl_yl_) / occ_res_));
      int gy_max_2d = std::min(occ_h_ - 1, (int)((vy_max - gl_yl_) / occ_res_));

      for (int gx2d = gx_min_2d; gx2d <= gx_max_2d; gx2d++) {
        for (int gy2d = gy_min_2d; gy2d <= gy_max_2d; gy2d++) {
          int idx2d = gy2d * occ_w_ + gx2d;
          if (kv.second == 100) {
            grid.data[idx2d] = 100;  // 占据优先
          } else if (kv.second == 0 && grid.data[idx2d] != 100) {
            grid.data[idx2d] = 0;    // 空闲 (不覆盖已知占据)
          }
        }
      }
    }

    // 更新二维累积
    for (int i = 0; i < occ_size_; i++) occ_accum_[i] = grid.data[i];

    occ_pub_.publish(grid);
    publishOccupancyGridImage(grid);

    // 缓存原始栅格供膨胀使用
    last_slice_grid_ = grid;
  }

  // ----------------------------------------------------------------
  //  将 OccupancyGrid 转为 Image 发布 (free=255, occupied=0, unknown=127)
  //  做 顺时针90° + 上下翻转，以匹配 OpenCV/ROS 坐标系约定
  // ----------------------------------------------------------------
  void publishOccupancyGridImage(const nav_msgs::OccupancyGrid& grid) {
    const int W = grid.info.width;
    const int H = grid.info.height;
    // 顺时针90°后尺寸 (W,H); 再flipud 尺寸不变
    sensor_msgs::Image img;
    img.header = grid.header;
    img.height = W;
    img.width = H;
    img.encoding = "mono8";
    img.is_bigendian = 0;
    img.step = img.width;
    img.data.resize(img.width * img.height);

    for (int r = 0; r < H; ++r) {
      for (int c = 0; c < W; ++c) {
        int8_t v = grid.data[r * W + c];
        uint8_t pix = (v == 0) ? 255 : (v == 100) ? 0 : 127;
        // 顺时针90°: (r,c)->(c, H-1-r); flipud: (r',c')->(W-1-r', c')
        // 综合: 原(r,c) -> rot -> (c, H-1-r) -> flipud -> (W-1-c, H-1-r)
        int r_out = W - 1 - c;
        int c_out = H - 1 - r;
        img.data[r_out * img.width + c_out] = pix;
      }
    }
    occ_image_pub_.publish(img);
  }

  // ----------------------------------------------------------------
  //  发布膨胀版三维体素 (供规划器使用)
  // ----------------------------------------------------------------
  void publish3DOccupancyInflated() {
    if (occ_3d_inflate_.empty()) return;

    pcl::PointCloud<pcl::PointXYZI> occ_cloud;
    occ_cloud.reserve(occ_3d_inflate_.size());

    for (const auto& key : occ_3d_inflate_) {
      int gi, gj, gk;
      key_to_ijk_3d(key, gi, gj, gk);

      pcl::PointXYZI pt;
      pt.x = gl_xl_ + (gi + 0.5) * voxel_res_;
      pt.y = gl_yl_ + (gj + 0.5) * voxel_res_;
      pt.z = gl_zl_ + (gk + 0.5) * voxel_res_;
      pt.intensity = pt.z;
      occ_cloud.points.push_back(pt);
    }

    occ_cloud.width = occ_cloud.points.size();
    occ_cloud.height = 1;
    occ_cloud.is_dense = true;

    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(occ_cloud, msg);
    msg.header.stamp = last_odom_stamp_;
    msg.header.frame_id = "world";
    occ3d_inflate_pub_.publish(msg);
  }

  // ----------------------------------------------------------------
  //  对二维栅格做圆形膨胀并发布
  // ----------------------------------------------------------------
  void inflateAndPublish2DGrid(const nav_msgs::OccupancyGrid& original) {
    nav_msgs::OccupancyGrid inflated = original;
    int r = inflate_r_2d_;
    int r2 = r * r;

    // 收集所有原始占据格的坐标 (避免遍历全图)
    std::vector<std::pair<int, int>> occ_cells;
    occ_cells.reserve(occ_size_ / 10);
    for (int y = 0; y < occ_h_; y++) {
      for (int x = 0; x < occ_w_; x++) {
        if (original.data[y * occ_w_ + x] == 100) {
          occ_cells.emplace_back(x, y);
        }
      }
    }

    // 对每个占据格, 在圆形邻域内膨胀
    for (const auto& cell : occ_cells) {
      int cx = cell.first, cy = cell.second;
      for (int dy = -r; dy <= r; dy++) {
        for (int dx = -r; dx <= r; dx++) {
          if (dx * dx + dy * dy > r2) continue;
          int nx = cx + dx, ny = cy + dy;
          if (nx >= 0 && nx < occ_w_ && ny >= 0 && ny < occ_h_) {
            inflated.data[ny * occ_w_ + nx] = 100;
          }
        }
      }
    }

    occ_inflate_pub_.publish(inflated);
  }

  // ================================================================
  //  FOV实时可视化 (与odom对齐, 两种模式均可用)
  // ================================================================
  void fovMarkerTimerCallback(const ros::TimerEvent&) {
    if (!has_odom_) return;

    visualization_msgs::Marker marker;
    marker.header.stamp = last_odom_stamp_;
    marker.header.frame_id = "world";
    marker.ns = "fov_frustum";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::LINE_LIST;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = fov_marker_line_width_;
    marker.color.r = 1.0f;
    marker.color.g = 0.0f;
    marker.color.b = 0.0f;
    marker.color.a = 0.8f;
    marker.lifetime = ros::Duration(0.2);

    double h_fov, v_fov, pitch, range;
    if (enable_3d_occ_) {
      h_fov = fov_vis_h_fov_;   // 仅可视化FOV, 不影响实际深度感知范围
      v_fov = cam_v_fov_;
      pitch = cam_pitch_;
      range = cam_max_range_ * fov_marker_scale_;
    } else {
      h_fov = fov_angle_;
      v_fov = 0.1;
      pitch = 0.0;
      range = max_range_ * fov_marker_scale_;
    }

    double half_h = h_fov / 2.0;
    double half_v = v_fov / 2.0;

    std::vector<geometry_msgs::Point> corners(4);
    double yaw = robot_yaw_;

    // 左上角 (h=-half_h, v=+half_v)
    double h_ang = -half_h;
    double v_ang = half_v + pitch;
    corners[0].x = robot_x_ + range * std::cos(v_ang) * std::cos(yaw + h_ang);
    corners[0].y = robot_y_ + range * std::cos(v_ang) * std::sin(yaw + h_ang);
    corners[0].z = robot_z_ + range * std::sin(v_ang);

    // 右上角 (h=+half_h, v=+half_v)
    h_ang = half_h;
    v_ang = half_v + pitch;
    corners[1].x = robot_x_ + range * std::cos(v_ang) * std::cos(yaw + h_ang);
    corners[1].y = robot_y_ + range * std::cos(v_ang) * std::sin(yaw + h_ang);
    corners[1].z = robot_z_ + range * std::sin(v_ang);

    // 右下角 (h=+half_h, v=-half_v)
    h_ang = half_h;
    v_ang = -half_v + pitch;
    corners[2].x = robot_x_ + range * std::cos(v_ang) * std::cos(yaw + h_ang);
    corners[2].y = robot_y_ + range * std::cos(v_ang) * std::sin(yaw + h_ang);
    corners[2].z = robot_z_ + range * std::sin(v_ang);

    // 左下角 (h=-half_h, v=-half_v)
    h_ang = -half_h;
    v_ang = -half_v + pitch;
    corners[3].x = robot_x_ + range * std::cos(v_ang) * std::cos(yaw + h_ang);
    corners[3].y = robot_y_ + range * std::cos(v_ang) * std::sin(yaw + h_ang);
    corners[3].z = robot_z_ + range * std::sin(v_ang);

    geometry_msgs::Point apex;
    apex.x = robot_x_;
    apex.y = robot_y_;
    apex.z = robot_z_;

    for (int i = 0; i < 4; i++) {
      marker.points.push_back(apex);
      marker.points.push_back(corners[i]);
    }
    for (int i = 0; i < 4; i++) {
      marker.points.push_back(corners[i]);
      marker.points.push_back(corners[(i + 1) % 4]);
    }

    fov_marker_pub_.publish(marker);
  }

  // ================================================================
  //  成员变量
  // ================================================================

  // ---- 通用参数 ----
  double fov_angle_;
  double max_range_;
  double occ_res_;
  bool publish_cloud_;
  bool use_relative_cloud_;
  std::string cloud_frame_id_;
  bool occ_mark_free_;
  bool occ_accumulate_;

  // ---- 3D深度感知参数 ----
  bool enable_3d_occ_ = false;
  double cam_h_fov_;
  double cam_v_fov_;
  double cam_pitch_;
  double cam_max_range_;
  double depth_h_fov_;        // 实际深度感知水平FOV (可360度, 多pass渲染)
  double fov_vis_h_fov_;      // FOV可视化marker的水平FOV
  double voxel_res_;
  bool occ3d_accumulate_;
  double occ3d_radius_limit_ = 20.0;  // 体素空间裁剪半径(m), <=0 禁用
  bool publish_free_voxels_;

  // ---- 深度渲染参数 ----
  int cam_width_ = 320;
  int cam_height_ = 240;
  double cam_fx_, cam_fy_, cam_cx_, cam_cy_;    // 虚拟相机内参 (从FOV计算)
  double depth_point_radius_ = 0.1;          // 投影点膨胀系数
  double slice_half_height_ = 0.5;              // 二维切片半高
  Eigen::Matrix4d cam02body_;                    // 相机→机体变换 (含俯仰)
  Eigen::Quaterniond robot_quat_ = Eigen::Quaterniond::Identity();  // 机体姿态四元数

  // ---- 膨胀参数 ----
  double inflate_radius_ = 0.3;                  // 膨胀半径 (meters)
  int inflate_r_2d_ = 0;                         // 2D膨胀格数 (occ_res单位)
  int inflate_r_voxels_ = 0;                     // 3D膨胀体素数 (voxel_res单位)
  double inflate_publish_interval_ = 0.1;        // 膨胀地图发布间隔 (秒, 10Hz)
  ros::Time last_inflate_time_;                  // 上次膨胀发布时间

  // ---- FOV可视化参数 ----
  bool publish_fov_marker_ = true;
  double fov_marker_rate_ = 10.0;
  double fov_marker_scale_ = 1.0;
  double fov_marker_line_width_ = 0.02;

  // ---- 地图参数 (2D) ----
  double gl_xl_, gl_yl_;
  int occ_w_, occ_h_, occ_size_;

  // ---- 地图参数 (3D) ----
  double gl_zl_ = 0.0;
  int vox_w_ = 0, vox_h_ = 0, vox_d_ = 0;

  // ---- 运行状态 ----
  pcl::PointCloud<pcl::PointXYZ> full_map_cloud_;
  std::unordered_set<int> obstacle_2d_;          // 仅2D模式使用
  std::vector<int8_t> occ_accum_;
  bool occ_initialized_ = false;
  bool has_map_ = false;
  bool has_odom_ = false;
  ros::Time last_odom_stamp_;
  double robot_x_ = 0, robot_y_ = 0, robot_z_ = 1.0, robot_yaw_ = 0;

  // ---- 3D体素累积 ----
  std::unordered_map<int64_t, int8_t> occ_3d_accum_;
  std::unordered_set<int64_t> occ_3d_inflate_;   // 膨胀版占据体素 (增量累积)
  nav_msgs::OccupancyGrid last_slice_grid_;       // 缓存最近一次切片栅格

  // ---- ROS接口 ----
  ros::Subscriber map_sub_;
  ros::Subscriber odom_sub_;
  ros::Publisher occ_pub_;
  ros::Publisher occ_image_pub_;       // 2D栅格图像 (供 map_predictor 等订阅)
  ros::Publisher occ_inflate_pub_;     // 膨胀版2D栅格
  ros::Publisher cloud_pub_;
  ros::Publisher occ3d_pub_;
  ros::Publisher occ3d_inflate_pub_;    // 膨胀版3D体素
  ros::Timer timer_;          // 2D平扫定时器 (仅2D模式)
  ros::Timer timer_depth_;    // 深度管线定时器 (仅3D模式)

  // ---- FOV可视化 ----
  ros::Publisher fov_marker_pub_;
  ros::Timer fov_timer_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "local_sensing_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  LocalSensingNode node(nh, pnh);

  ros::spin();
  return 0;
}
