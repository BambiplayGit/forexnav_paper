cmake_minimum_required(VERSION 3.0.2)
project(gen_viewpoint)

add_compile_options(-std=c++17)
set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  nav_msgs
  geometry_msgs
  visualization_msgs
)

find_package(Eigen3 REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp nav_msgs geometry_msgs visualization_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(gen_viewpoint_node src/viewpoint_planner_node.cpp)
target_link_libraries(gen_viewpoint_node
  ${catkin_LIBRARIES}
)

install(TARGETS gen_viewpoint_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
