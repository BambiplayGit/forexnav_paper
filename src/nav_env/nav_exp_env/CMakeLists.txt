cmake_minimum_required(VERSION 3.0.2)
project(nav_exp_env)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  tf2_ros
  tf2_geometry_msgs
  visualization_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs geometry_msgs nav_msgs tf2_ros tf2_geometry_msgs visualization_msgs sensor_msgs
)

# Install Python scripts
catkin_install_python(PROGRAMS
  base_env/scripts/simple_odom_simulator.py
  base_env/scripts/odom_path_publisher.py
  base_env/scripts/clicked_point_to_pos_cmd.py
  base_env/scripts/poly_traj_smoother.py
  base_env/scripts/gait_joint_publisher.py
  base_env/scripts/frtree_traj_cmd.py
  base_env/scripts/rrtx_traj_cmd.py
  base_env/scripts/benchmark_recorder.py
  base_env/scripts/set_benchmark_points.py
  base_env/scripts/preset_goal_sender.py
  cogniplan/scripts/rl_planner_nav.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch files
install(DIRECTORY base_env/launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch/base_env
)
install(DIRECTORY cogniplan/launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch/cogniplan
)
install(DIRECTORY frtree/launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch/frtree
)
install(DIRECTORY rrtx/launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch/rrtx
)
install(DIRECTORY proposed/launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch/proposed
)

# Install config files
install(DIRECTORY base_env/config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config/base_env
)
install(DIRECTORY cogniplan/config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config/cogniplan
)
install(DIRECTORY frtree/config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config/frtree
)
install(DIRECTORY rrtx/config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config/rrtx
)
install(DIRECTORY proposed/config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config/proposed
)
