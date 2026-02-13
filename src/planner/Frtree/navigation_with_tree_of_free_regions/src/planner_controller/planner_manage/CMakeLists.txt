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

###############################################################################
# Auto-build AltroCpp and HiGHS from source if not found
###############################################################################
get_filename_component(FRTREE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
set(LOCAL_DEPS_PREFIX "${FRTREE_DIR}/local_install")
list(APPEND CMAKE_PREFIX_PATH "${LOCAL_DEPS_PREFIX}")
include(ProcessorCount)
ProcessorCount(NPROC)
if(NPROC EQUAL 0)
  set(NPROC 1)
endif()

# --- AltroCpp ---
find_package(AltroCpp QUIET)
if(NOT AltroCpp_FOUND)
  message(STATUS "===============================================")
  message(STATUS "AltroCpp not found – building from source ...")
  message(STATUS "===============================================")
  execute_process(
    COMMAND ${CMAKE_COMMAND}
      -S "${FRTREE_DIR}/altro-cpp"
      -B "${FRTREE_DIR}/altro-cpp/build"
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_INSTALL_PREFIX=${LOCAL_DEPS_PREFIX}
      -DALTRO_BUILD_TESTS=OFF
      -DALTRO_BUILD_EXAMPLES=OFF
      -DALTRO_BUILD_BENCHMARKS=OFF
    RESULT_VARIABLE _altro_cfg_result
  )
  if(NOT _altro_cfg_result EQUAL 0)
    message(FATAL_ERROR "Failed to configure AltroCpp")
  endif()
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build "${FRTREE_DIR}/altro-cpp/build" --parallel ${NPROC}
    RESULT_VARIABLE _altro_build_result
  )
  if(NOT _altro_build_result EQUAL 0)
    message(FATAL_ERROR "Failed to build AltroCpp")
  endif()
  execute_process(
    COMMAND ${CMAKE_COMMAND} --install "${FRTREE_DIR}/altro-cpp/build"
    RESULT_VARIABLE _altro_install_result
  )
  if(NOT _altro_install_result EQUAL 0)
    message(FATAL_ERROR "Failed to install AltroCpp")
  endif()
  find_package(AltroCpp REQUIRED)
endif()

# --- HiGHS ---
find_package(highs QUIET)
if(NOT highs_FOUND)
  message(STATUS "===============================================")
  message(STATUS "HiGHS not found – building from source ...")
  message(STATUS "===============================================")
  execute_process(
    COMMAND ${CMAKE_COMMAND}
      -S "${FRTREE_DIR}/HiGHS"
      -B "${FRTREE_DIR}/HiGHS/build"
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_INSTALL_PREFIX=${LOCAL_DEPS_PREFIX}
      -DBUILD_TESTING=OFF
      -DBUILD_CXX_EXE=OFF
    RESULT_VARIABLE _highs_cfg_result
  )
  if(NOT _highs_cfg_result EQUAL 0)
    message(FATAL_ERROR "Failed to configure HiGHS")
  endif()
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build "${FRTREE_DIR}/HiGHS/build" --parallel ${NPROC}
    RESULT_VARIABLE _highs_build_result
  )
  if(NOT _highs_build_result EQUAL 0)
    message(FATAL_ERROR "Failed to build HiGHS")
  endif()
  execute_process(
    COMMAND ${CMAKE_COMMAND} --install "${FRTREE_DIR}/HiGHS/build"
    RESULT_VARIABLE _highs_install_result
  )
  if(NOT _highs_install_result EQUAL 0)
    message(FATAL_ERROR "Failed to install HiGHS")
  endif()
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
