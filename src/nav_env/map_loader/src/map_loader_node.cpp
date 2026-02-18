#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>

class MapLoaderNode {
 public:
  MapLoaderNode(ros::NodeHandle& nh, ros::NodeHandle& pnh) {
    // Get parameters
    std::string pcd_path;
    pnh.param<std::string>("pcd_path", pcd_path, "");
    if (pcd_path.empty()) {
      ROS_ERROR("Parameter 'pcd_path' is required");
      return;
    }

    double rotate_z, trans_x, trans_y, trans_z;
    std::string frame_id;
    pnh.param<double>("rotate_z", rotate_z, 0.0);
    pnh.param<double>("trans_x", trans_x, 0.0);
    pnh.param<double>("trans_y", trans_y, 0.0);
    pnh.param<double>("trans_z", trans_z, 0.0);
    pnh.param<std::string>("frame_id", frame_id, "world");

    // Display Z truncation: same height as local_sensing occupancy_3d (3D online mapping)
    // occupancy_3d uses gl_zl_=-z_size/2, range [-z_size/2, z_size/2]
    pnh.param<double>("display_z_size", display_z_size_, 5.0);

    ROS_INFO("Loading PCD: %s", pcd_path.c_str());
    ROS_INFO("Transform: RotZ=%.2f deg, Trans=[%.2f, %.2f, %.2f]",
              rotate_z, trans_x, trans_y, trans_z);

    // Publisher with latched topic
    cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("global_cloud", 10, true);

    // Load point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_path, *cloud) == -1) {
      ROS_ERROR("Couldn't read file: %s", pcd_path.c_str());
      return;
    }

    // Apply transform
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << trans_x, trans_y, trans_z;
    float theta_rad = rotate_z * M_PI / 180.0f;
    transform.rotate(Eigen::AngleAxisf(theta_rad, Eigen::Vector3f::UnitZ()));
    pcl::transformPointCloud(*cloud, *cloud, transform);

    // Generate ground
    Eigen::Vector2d mmin(1e9, 1e9), mmax(-1e9, -1e9);
    for (const auto& pt : cloud->points) {
      if (pt.x < mmin[0]) mmin[0] = pt.x;
      if (pt.y < mmin[1]) mmin[1] = pt.y;
      if (pt.x > mmax[0]) mmax[0] = pt.x;
      if (pt.y > mmax[1]) mmax[1] = pt.y;
    }

    for (double x = mmin[0]; x <= mmax[0]; x += 0.1) {
      for (double y = mmin[1]; y <= mmax[1]; y += 0.1) {
        cloud->push_back(pcl::PointXYZ(x, y, 0.0));
      }
    }

    // Truncate display to same Z range as occupancy_3d (3D online mapping): [-z_size/2, z_size/2]
    if (display_z_size_ > 0) {
      double z_min = -display_z_size_ / 2.0;
      double z_max = display_z_size_ / 2.0;
      size_t orig_count = cloud->points.size();
      pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
      for (const auto& pt : cloud->points) {
        if (pt.z >= z_min && pt.z <= z_max) {
          filtered->push_back(pt);
        }
      }
      cloud = filtered;
      ROS_INFO("Display Z truncated to [%.2f, %.2f]m (occupancy_3d height): %zu points (was %zu)",
               z_min, z_max, cloud->points.size(), orig_count);
    }

    // Convert to ROS message
    pcl::toROSMsg(*cloud, cloud_msg_);
    cloud_msg_.header.frame_id = frame_id;

    ROS_INFO("Publishing map with %zu points...", cloud->points.size());

    // Publish at 1Hz
    timer_ = nh.createTimer(ros::Duration(1.0), &MapLoaderNode::publishCallback, this);
  }

 private:
  void publishCallback(const ros::TimerEvent&) {
    cloud_msg_.header.stamp = ros::Time::now();
    cloud_pub_.publish(cloud_msg_);
  }

  ros::Publisher cloud_pub_;
  ros::Timer timer_;
  sensor_msgs::PointCloud2 cloud_msg_;
  double display_z_size_{5.0};
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "map_loader_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  
  MapLoaderNode node(nh, pnh);
  
  ros::spin();
  return 0;
}
