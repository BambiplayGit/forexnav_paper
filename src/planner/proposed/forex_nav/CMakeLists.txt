cmake_minimum_required(VERSION 3.0.2)
project(forex_nav)

add_compile_options(-std=c++17)
set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  nav_msgs
  geometry_msgs
  visualization_msgs
  std_msgs
  sensor_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  pcl_conversions
  quadrotor_msgs
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common)

catkin_package(
  INCLUDE_DIRS include minco/include
  LIBRARIES minco_wrapper astar_2d forex_nav_manager forex_nav_fsm
  CATKIN_DEPENDS roscpp nav_msgs geometry_msgs visualization_msgs std_msgs sensor_msgs tf2 tf2_ros tf2_geometry_msgs pcl_conversions quadrotor_msgs
)

include_directories(
  include
  minco/include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

# MINCO library with GCOPTER optimization
add_library(minco_wrapper
  minco/src/minco_wrapper.cpp
)
target_link_libraries(minco_wrapper
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

# Astar2D library
add_library(astar_2d
  src/astar_2d.cpp
)
target_link_libraries(astar_2d
  ${catkin_LIBRARIES}
)

# ForexNavManager library
add_library(forex_nav_manager
  src/forex_nav_manager.cpp
)
target_link_libraries(forex_nav_manager
  astar_2d
  minco_wrapper
  ${catkin_LIBRARIES}
)

# ForexNavFSM library
add_library(forex_nav_fsm
  src/forex_nav_fsm.cpp
)
add_dependencies(forex_nav_fsm
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(forex_nav_fsm
  forex_nav_manager
  ${catkin_LIBRARIES}
)

# Main node
add_executable(forex_nav_node
  src/forex_nav_node.cpp
)
target_link_libraries(forex_nav_node
  forex_nav_manager
  forex_nav_fsm
  ${catkin_LIBRARIES}
)

install(TARGETS minco_wrapper astar_2d forex_nav_manager forex_nav_fsm forex_nav_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(DIRECTORY minco/include/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
