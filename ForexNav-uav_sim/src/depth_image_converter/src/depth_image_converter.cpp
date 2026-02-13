#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class DepthImageConverter {
public:
  DepthImageConverter() : nh_("~") {
    sub_ = nh_.subscribe("/image_raw/compressed/depth", 1, &DepthImageConverter::depthCallback, this);
    pub_ = nh_.advertise<sensor_msgs::Image>("/depth_image/converted", 1);
    nh_.param("depth_scaling_factor", k_depth_scaling_factor_, 1000.0);  // 默认从m转mm
    ROS_INFO("DepthImageConverter initialized.");
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber sub_;
  ros::Publisher pub_;
  double k_depth_scaling_factor_;

void depthCallback(const sensor_msgs::ImageConstPtr& msg) {
  // 正确构造 32FC1 的深度图：height 是行数（480），width 是列数（640）
  cv::Mat depth_image(480, 640, CV_32FC1, const_cast<uchar*>(msg->data.data()));

  // 转成 16UC1，单位从米转成毫米
  cv::Mat depth_16u;
  depth_image.convertTo(depth_16u, CV_16UC1, 1000.0);

  // 构造输出消息
  cv_bridge::CvImage out_msg;
  out_msg.header = msg->header;
  out_msg.encoding = sensor_msgs::image_encodings::TYPE_16UC1;
  out_msg.image = depth_16u;

  // 发布
  pub_.publish(out_msg.toImageMsg());
}


};

int main(int argc, char** argv) {
  ros::init(argc, argv, "depth_image_converter");
  DepthImageConverter converter;
  ros::spin();
  return 0;
}
