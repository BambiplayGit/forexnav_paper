#include <ros/ros.h>
#include "forex_nav/forex_nav_fsm.h"

int main(int argc, char** argv) {
  ros::init(argc, argv, "forex_nav_node");
  
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  
  forex_nav::ForexNavFSM fsm(nh, pnh);
  fsm.init();
  
  ros::spin();
  
  return 0;
}
