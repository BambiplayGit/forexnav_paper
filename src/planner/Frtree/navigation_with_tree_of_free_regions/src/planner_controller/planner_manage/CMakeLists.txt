cmake_minimum_required(VERSION 3.16)
project(pop_planner)

set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS} -O3 -Wno-pedantic")
set(CMAKE_BUILD_TYPE "Release")

find_package(catkin REQUIRED COMPONENTS
    laser_geometry
    geometry_msgs
    message_filters
    roscpp
    rospy
    sensor_msgs
    std_msgs
    tf
    visualization_msgs
    frtree_local_map
    decomp_ros_msgs
    decomp_ros_utils
)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.7 REQUIRED)

# find_package(jps3d REQUIRED)
set(Eigen3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})

# altro-cpp - use in-tree source (relative: planner_manage -> ... -> Frtree/altro-cpp)
set(ALTRO_CPP_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../altro-cpp)
if(EXISTS "${ALTRO_CPP_SOURCE_DIR}/CMakeLists.txt")
  set(ALTRO_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(ALTRO_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(ALTRO_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
  set(ALTRO_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  add_subdirectory(${ALTRO_CPP_SOURCE_DIR} ${CMAKE_BINARY_DIR}/altro_cpp)
else()
  find_package(AltroCpp REQUIRED)
endif()

# HiGHS - use in-tree source if present (same Frtree level as altro-cpp)
set(HIGHS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../HiGHS)
if(EXISTS "${HIGHS_SOURCE_DIR}/CMakeLists.txt")
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(BUILD_CXX_EXE OFF CACHE BOOL "" FORCE)
  add_subdirectory(${HIGHS_SOURCE_DIR} ${CMAKE_BINARY_DIR}/highs_build)
  if(NOT TARGET highs::highs)
    add_library(highs::highs ALIAS highs)
  endif()
else()
  find_package(highs REQUIRED)
endif()

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES pop_planner 
 CATKIN_DEPENDS geometry_msgs roscpp roscpp rospy sensor_msgs std_msgs visualization_msgs frtree_local_map
)


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS} 
  ${PCL_INCLUDE_DIRS}
  # ${JPS3D_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/../DecompROS/DecompUtil/include
  ${PROJECT_SOURCE_DIR}/../third_party/geo_utils
)

add_executable(${PROJECT_NAME}_node
    src/dynamic_plan_manager_pop.cpp 
    src/dynamic_plan_fsm_pop.cpp 
    src/global_graph.cpp
    src/Altro/altro_constraints.cpp
    src/Altro/altro_cost.cpp
    src/Altro/altro_model.cpp
    src/Altro/altro_problem.cpp
    src/SDPsolver/SDPsolver.cpp
  )

target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  # ${JPS3D_LIBRARIES}
  altrocpp::altro 
  fmt::fmt
  highs::highs
)
