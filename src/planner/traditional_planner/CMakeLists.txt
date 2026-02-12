cmake_minimum_required(VERSION 3.0.2)
project(traditional_planner)

find_package(catkin REQUIRED COMPONENTS
  rospy
  nav_msgs
  geometry_msgs
  visualization_msgs
  std_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/traditional_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
