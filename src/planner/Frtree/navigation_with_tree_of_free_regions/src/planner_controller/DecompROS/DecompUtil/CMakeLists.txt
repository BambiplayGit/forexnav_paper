cmake_minimum_required(VERSION 2.8.3)
project(decomp_util)

# set default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wno-deprecated-declarations")

set(BUILD_SHARED_LIBS ON)

find_package(catkin REQUIRED)
find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  DEPENDS EIGEN3
)

include_directories(${EIGEN3_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS} include)

install(DIRECTORY include/decomp_basis include/decomp_geometry include/decomp_util
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
