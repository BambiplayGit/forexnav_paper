cmake_minimum_required(VERSION 3.0.2)
project(local_sensing)

add_compile_options(-std=c++17)
set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  pcl_ros
  pcl_conversions
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp sensor_msgs nav_msgs geometry_msgs pcl_ros pcl_conversions
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_executable(local_sensing_node src/local_sensing_node.cpp)
target_link_libraries(local_sensing_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

install(TARGETS local_sensing_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY params/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/params
)
