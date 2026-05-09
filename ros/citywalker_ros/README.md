CityWalker ROS Deployment
=========================

This package targets ROS Noetic and provides four nodes:

1. `global_path_node.py`: loads a reference path, finds the nearest path point from SLAM pose, and publishes a local lookahead goal.
2. `history_buffer_node.py`: caches recent images and poses, samples 5 timestamps over 2 seconds, and converts historical positions into the current `base_link` frame.
3. `inference_node.py`: runs CityWalker with a TensorRT DINOv2 backbone and PyTorch decoder/head, then publishes local waypoints.
4. `waypoint_follower_node.py`: tracks predicted waypoints with a pure-pursuit controller and publishes `geometry_msgs/Twist`.

Build
-----

Place or symlink this package into a catkin workspace `src` directory:

```bash
conda deactivate
source /opt/ros/noetic/setup.bash
cd ~/catkin_ws/src
ln -s /media/isee324/2a90eb70-2d62-4af4-b04a-0fcdde4122a5/qy/AD/citywalker_orad/ros/citywalker_ros .
cd ~/catkin_ws
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setup.bash
```

Build the ROS workspace with the system Python, not the `dino` conda Python.
The inference node is launched through the conda wrapper at runtime, so the
workspace itself does not need to be built inside conda.

Run
---

Use your real path CSV/YAML and hardware topics:

```bash
roslaunch citywalker_ros citywalker_deploy.launch \
  path_file:=/path/to/reference_path.csv \
  image_topic:=/camera/color/image_raw \
  pose_topic:=/slam_pose \
  pose_msg_type:=Odometry \
  cmd_topic:=/cmd_vel \
  cmd_msg_type:=TwistStamped \
  step_scale:=1.0
```

By default the inference node is launched through `scripts/inference_node_conda.py`,
which activates the `dino` conda environment while keeping ROS Python paths:

```bash
roslaunch citywalker_ros citywalker_deploy.launch \
  conda_setup:=/media/isee324/2a90eb70-2d62-4af4-b04a-0fcdde4122a5/anaconda3/etc/profile.d/conda.sh \
  conda_env:=dino
```

If the active Python environment already contains `torch`, `tensorrt`, `rospy`,
and `cv_bridge`, disable the wrapper:

```bash
roslaunch citywalker_ros citywalker_deploy.launch use_conda_inference:=false
```

The controller starts disabled. Enable it only after checking topics and RViz:

```bash
rostopic pub /citywalker/enable std_msgs/Bool "data: true" -1
```

Check why the controller is publishing zero velocity:

```bash
rostopic echo /waypoint_follower_node/status
```

Important Parameters
--------------------

- `step_scale`: runtime coordinate normalization scale. The offroad dataset divides input positions and waypoint targets by `step_scale`; deployment must use a matching value.
- `pose_msg_type`: either `PoseStamped` or `Odometry`; this deployment defaults to `Odometry`.
- `image_encoding`: defaults to `bgr8`. Change to match the camera if needed.
- `path_file`: CSV columns are `x,y[,yaw]`; YAML may be either a list or `{points: [...]}`.

Main Topics
-----------

- `/global_path_node/reference_path`: `nav_msgs/Path`
- `/global_path_node/goal_sequence`: `nav_msgs/Path`
- `/global_path_node/local_goal_base`: `geometry_msgs/PoseStamped`
- `/history_buffer_node/history`: `citywalker_ros/HistoryBundle`
- `/inference_node/waypoints`: `nav_msgs/Path`
- `/inference_node/arrived_prob`: `std_msgs/Float32`
- `/cmd_vel`: `geometry_msgs/Twist`

For simulators that subscribe to `geometry_msgs/TwistStamped`, launch with:

```bash
roslaunch citywalker_ros citywalker_deploy.launch \
  cmd_topic:=/cmd_vel \
  cmd_msg_type:=TwistStamped \
  cmd_frame_id:=base_link
```
- `/citywalker_visualization_node/markers`: `visualization_msgs/MarkerArray`

Gazebo And RViz Visualization
-----------------------------

`citywalker_deploy.launch` starts `citywalker_visualization_node.py` by default.
It visualizes:

- global reference path
- current local lookahead goal
- sampled 5 historical positions
- predicted local waypoints
- current `cmd_vel` direction

The node always publishes RViz markers:

```bash
rviz
```

Add a `MarkerArray` display and select:

```text
/citywalker_visualization_node/markers
```

When Gazebo is running, it also spawns colored sphere models through
`/gazebo/spawn_sdf_model` and updates them with `/gazebo/set_model_state`.
If you only want RViz markers and no Gazebo models:

```bash
roslaunch citywalker_ros citywalker_deploy.launch use_gazebo_visuals:=false
```

To run only visualization against already-running nodes:

```bash
roslaunch citywalker_ros citywalker_visualization.launch \
  pose_topic:=/slam_pose \
  pose_msg_type:=Odometry \
  use_gazebo_visuals:=true
```

Record A Global Path
--------------------

Run SLAM, manually drive the vehicle, and record a sparse path from the SLAM
odometry:

```bash
roslaunch citywalker_ros record_path.launch \
  pose_topic:=/slam_pose \
  pose_msg_type:=Odometry \
  output_file:=$HOME/citywalker_recorded_path.csv \
  min_distance:=0.3 \
  final_sparsify_distance:=0.5
```

The recorder publishes `/path_recorder_node/recorded_path` for RViz. Recording
starts automatically by default. To pause and save:

```bash
rostopic pub /path_recorder_node/recording std_msgs/Bool "data: false" -1
```

To resume:

```bash
rostopic pub /path_recorder_node/recording std_msgs/Bool "data: true" -1
```

Use the generated CSV as `path_file` in `citywalker_deploy.launch`.

Global Goal Publishing
----------------------

The global path node supports two modes:

- `waypoint_sequence`: generate fixed global goals every `waypoint_spacing`
  meters along the path. The current goal stays fixed until the vehicle is
  within `switch_distance`, then the next fixed goal is published.
- `lookahead`: old behavior; continuously publish a point at a fixed
  `lookahead_distance` ahead of the nearest path point.

Recommended fixed-goal launch:

```bash
roslaunch citywalker_ros citywalker_deploy.launch \
  path_file:=/path/to/recorded_path.csv \
  goal_mode:=waypoint_sequence \
  waypoint_spacing:=5.0 \
  switch_distance:=1.0
```

The fixed goal sequence is published on:

```text
/global_path_node/goal_sequence
```
