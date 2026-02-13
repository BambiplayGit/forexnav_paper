cmake_minimum_required(VERSION 2.8.3)
project(legged_fake_simulator)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  nav_msgs
  tf
  quadrotor_msgs
)

catkin_package(
  CATKIN_DEPENDS roscpp geometry_msgs nav_msgs tf quadrotor_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(fake_simulator_node src/fake_simulator_node.cpp)
target_link_libraries(fake_simulator_node ${catkin_LIBRARIES})

install(TARGETS fake_simulator_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

