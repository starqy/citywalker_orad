cmake_minimum_required(VERSION 3.0.2)
project(citywalker_ros)

find_package(catkin REQUIRED COMPONENTS
  gazebo_msgs
  geometry_msgs
  message_generation
  nav_msgs
  rospy
  sensor_msgs
  std_msgs
  tf
  visualization_msgs
)

add_message_files(
  FILES
  HistoryBundle.msg
)

catkin_python_setup()

generate_messages(
  DEPENDENCIES
  geometry_msgs
  sensor_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
  gazebo_msgs
  geometry_msgs
  message_runtime
  nav_msgs
  sensor_msgs
  std_msgs
  visualization_msgs
)

catkin_install_python(
  PROGRAMS
  scripts/global_path_node.py
  scripts/history_buffer_node.py
  scripts/inference_node_conda.py
  scripts/inference_node.py
  scripts/path_recorder_node.py
  scripts/citywalker_visualization_node.py
  scripts/waypoint_follower_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
