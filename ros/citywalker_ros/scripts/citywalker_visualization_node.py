#!/usr/bin/env python3
import math

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, SpawnModel
from geometry_msgs.msg import Point, Pose, PoseStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from citywalker_ros.geometry import base_point_to_map, pose_msg_to_xy_yaw
from citywalker_ros.msg import HistoryBundle


def color(r, g, b, a=1.0):
    return ColorRGBA(float(r), float(g), float(b), float(a))


def point_msg(x, y, z=0.0):
    msg = Point()
    msg.x = float(x)
    msg.y = float(y)
    msg.z = float(z)
    return msg


def sphere_sdf(radius, rgba):
    r, g, b, a = rgba
    return """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="citywalker_marker">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry><sphere><radius>{radius}</radius></sphere></geometry>
        <material>
          <ambient>{r} {g} {b} {a}</ambient>
          <diffuse>{r} {g} {b} {a}</diffuse>
        </material>
      </visual>
      <collision name="collision">
        <geometry><sphere><radius>{radius}</radius></sphere></geometry>
      </collision>
    </link>
  </model>
</sdf>""".format(radius=radius, r=r, g=g, b=b, a=a)


class CityWalkerVisualizationNode:
    def __init__(self):
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.pose_topic = rospy.get_param("~pose_topic", "/slam_pose")
        self.pose_msg_type = rospy.get_param("~pose_msg_type", "Odometry")
        self.reference_path_topic = rospy.get_param("~reference_path_topic", "/global_path_node/reference_path")
        self.goal_sequence_topic = rospy.get_param("~goal_sequence_topic", "/global_path_node/goal_sequence")
        self.goal_topic = rospy.get_param("~goal_topic", "/global_path_node/local_goal_base")
        self.history_topic = rospy.get_param("~history_topic", "/history_buffer_node/history")
        self.waypoints_topic = rospy.get_param("~waypoints_topic", "/inference_node/waypoints")
        self.cmd_topic = rospy.get_param("~cmd_topic", "/cmd_vel")
        self.cmd_msg_type = rospy.get_param("~cmd_msg_type", "Twist")
        self.publish_rate = float(rospy.get_param("~publish_rate", 5.0))
        self.use_gazebo_models = bool(rospy.get_param("~use_gazebo_models", True))
        self.gazebo_reference_frame = rospy.get_param("~gazebo_reference_frame", "world")
        self.gazebo_prefix = rospy.get_param("~gazebo_prefix", "citywalker_vis")
        self.max_path_models = int(rospy.get_param("~max_path_models", 80))
        self.marker_z = float(rospy.get_param("~marker_z", 0.08))

        self.current_pose = None
        self.reference_path = None
        self.goal_sequence = None
        self.goal = None
        self.history = None
        self.waypoints = None
        self.cmd = None
        self.spawned = set()
        self.gazebo_ready = False

        self.marker_pub = rospy.Publisher("~markers", MarkerArray, queue_size=1)

        msg_type = Odometry if self.pose_msg_type == "Odometry" else PoseStamped
        self.pose_sub = rospy.Subscriber(self.pose_topic, msg_type, self.pose_callback, queue_size=50)
        self.path_sub = rospy.Subscriber(self.reference_path_topic, Path, self.path_callback, queue_size=1)
        self.goal_sequence_sub = rospy.Subscriber(self.goal_sequence_topic, Path, self.goal_sequence_callback, queue_size=1)
        self.goal_sub = rospy.Subscriber(self.goal_topic, PoseStamped, self.goal_callback, queue_size=1)
        self.history_sub = rospy.Subscriber(self.history_topic, HistoryBundle, self.history_callback, queue_size=1)
        self.waypoint_sub = rospy.Subscriber(self.waypoints_topic, Path, self.waypoints_callback, queue_size=1)
        cmd_type = TwistStamped if self.cmd_msg_type == "TwistStamped" else Twist
        self.cmd_sub = rospy.Subscriber(self.cmd_topic, cmd_type, self.cmd_callback, queue_size=5)

        if self.use_gazebo_models:
            self.setup_gazebo()

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.publish)

    def setup_gazebo(self):
        try:
            rospy.wait_for_service("/gazebo/spawn_sdf_model", timeout=3.0)
            rospy.wait_for_service("/gazebo/set_model_state", timeout=3.0)
            self.spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
            self.set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
            self.gazebo_ready = True
            rospy.loginfo("Gazebo visualization enabled")
        except rospy.ROSException:
            self.gazebo_ready = False
            rospy.logwarn("Gazebo services unavailable; RViz markers will still be published")

    def pose_callback(self, msg):
        self.current_pose = pose_msg_to_xy_yaw(msg)

    def path_callback(self, msg):
        self.reference_path = msg

    def goal_sequence_callback(self, msg):
        self.goal_sequence = msg

    def goal_callback(self, msg):
        self.goal = msg

    def history_callback(self, msg):
        self.history = msg

    def waypoints_callback(self, msg):
        self.waypoints = msg

    def cmd_callback(self, msg):
        self.cmd = msg.twist if isinstance(msg, TwistStamped) else msg

    def map_xy_from_point(self, point, frame_id):
        if frame_id == self.base_frame:
            if self.current_pose is None:
                return None
            return base_point_to_map((point.x, point.y), self.current_pose)
        return point.x, point.y

    def marker_header(self, frame_id, stamp):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(0.5)
        return marker

    def line_marker(self, ns, marker_id, frame_id, points, rgba, width, stamp):
        marker = self.marker_header(frame_id, stamp)
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.scale.x = width
        marker.color = rgba
        marker.points = [point_msg(x, y, self.marker_z) for x, y in points]
        return marker

    def sphere_list_marker(self, ns, marker_id, frame_id, points, rgba, scale, stamp):
        marker = self.marker_header(frame_id, stamp)
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.SPHERE_LIST
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color = rgba
        marker.points = [point_msg(x, y, self.marker_z) for x, y in points]
        return marker

    def arrow_marker(self, ns, marker_id, frame_id, start_xy, end_xy, rgba, stamp):
        marker = self.marker_header(frame_id, stamp)
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.scale.x = 0.05
        marker.scale.y = 0.12
        marker.scale.z = 0.12
        marker.color = rgba
        marker.points = [point_msg(start_xy[0], start_xy[1], self.marker_z), point_msg(end_xy[0], end_xy[1], self.marker_z)]
        return marker

    def publish_markers(self, stamp):
        markers = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        if self.reference_path and self.reference_path.poses:
            pts = [(p.pose.position.x, p.pose.position.y) for p in self.reference_path.poses]
            markers.markers.append(self.line_marker("reference_path", 0, self.reference_path.header.frame_id or self.map_frame, pts, color(0.0, 0.7, 1.0, 1.0), 0.06, stamp))

        if self.goal_sequence and self.goal_sequence.poses:
            pts = [(p.pose.position.x, p.pose.position.y) for p in self.goal_sequence.poses]
            markers.markers.append(self.sphere_list_marker("fixed_goal_sequence", 0, self.goal_sequence.header.frame_id or self.map_frame, pts, color(1.0, 0.4, 0.0, 1.0), 0.22, stamp))

        if self.goal:
            p = self.goal.pose.position
            markers.markers.append(self.sphere_list_marker("local_goal", 0, self.goal.header.frame_id or self.base_frame, [(p.x, p.y)], color(1.0, 0.8, 0.0, 1.0), 0.35, stamp))

        if self.history and self.history.relative_positions:
            pts = [(p.x, p.y) for p in self.history.relative_positions]
            markers.markers.append(self.sphere_list_marker("history_positions", 0, self.history.header.frame_id or self.base_frame, pts, color(0.1, 0.6, 1.0, 1.0), 0.22, stamp))
            markers.markers.append(self.line_marker("history_positions", 1, self.history.header.frame_id or self.base_frame, pts, color(0.1, 0.6, 1.0, 0.8), 0.035, stamp))

        if self.waypoints and self.waypoints.poses:
            pts = [(p.pose.position.x, p.pose.position.y) for p in self.waypoints.poses]
            markers.markers.append(self.sphere_list_marker("predicted_waypoints", 0, self.waypoints.header.frame_id or self.base_frame, pts, color(0.0, 1.0, 0.2, 1.0), 0.24, stamp))
            markers.markers.append(self.line_marker("predicted_waypoints", 1, self.waypoints.header.frame_id or self.base_frame, pts, color(0.0, 1.0, 0.2, 0.9), 0.05, stamp))

        if self.cmd:
            length = max(0.2, min(1.5, abs(self.cmd.linear.x) * 2.0))
            lateral = max(-0.8, min(0.8, self.cmd.angular.z * 0.4))
            markers.markers.append(self.arrow_marker("cmd_vel", 0, self.base_frame, (0.0, 0.0), (length, lateral), color(1.0, 0.2, 0.1, 1.0), stamp))

        self.marker_pub.publish(markers)

    def spawn_or_move_sphere(self, name, xy, radius, rgba):
        if not self.gazebo_ready:
            return
        pose = Pose()
        pose.position.x = float(xy[0])
        pose.position.y = float(xy[1])
        pose.position.z = self.marker_z
        pose.orientation.w = 1.0
        try:
            if name not in self.spawned:
                self.spawn_model(name, sphere_sdf(radius, rgba), "", pose, self.gazebo_reference_frame)
                self.spawned.add(name)
            else:
                state = ModelState()
                state.model_name = name
                state.pose = pose
                state.reference_frame = self.gazebo_reference_frame
                self.set_model_state(state)
        except rospy.ServiceException as exc:
            rospy.logwarn_throttle(2.0, "Gazebo marker update failed: %s", exc)

    def sampled_path_points(self):
        if not self.reference_path or not self.reference_path.poses:
            return []
        poses = self.reference_path.poses
        if len(poses) <= self.max_path_models:
            selected = poses
        else:
            step = int(math.ceil(float(len(poses)) / self.max_path_models))
            selected = poses[::step]
        return [(p.pose.position.x, p.pose.position.y) for p in selected[: self.max_path_models]]

    def publish_gazebo_models(self):
        if not self.gazebo_ready:
            return

        for idx, xy in enumerate(self.sampled_path_points()):
            self.spawn_or_move_sphere("%s_path_%03d" % (self.gazebo_prefix, idx), xy, 0.10, (0.0, 0.5, 1.0, 1.0))

        if self.goal_sequence:
            for idx, pose in enumerate(self.goal_sequence.poses[: self.max_path_models]):
                xy = (pose.pose.position.x, pose.pose.position.y)
                self.spawn_or_move_sphere("%s_fixed_goal_%03d" % (self.gazebo_prefix, idx), xy, 0.13, (1.0, 0.4, 0.0, 1.0))

        if self.goal:
            xy = self.map_xy_from_point(self.goal.pose.position, self.goal.header.frame_id or self.base_frame)
            if xy:
                self.spawn_or_move_sphere("%s_goal" % self.gazebo_prefix, xy, 0.22, (1.0, 0.8, 0.0, 1.0))

        if self.history:
            for idx, p in enumerate(self.history.relative_positions):
                xy = self.map_xy_from_point(p, self.history.header.frame_id or self.base_frame)
                if xy:
                    self.spawn_or_move_sphere("%s_history_%02d" % (self.gazebo_prefix, idx), xy, 0.13, (0.1, 0.6, 1.0, 1.0))

        if self.waypoints:
            for idx, pose in enumerate(self.waypoints.poses):
                xy = self.map_xy_from_point(pose.pose.position, self.waypoints.header.frame_id or self.base_frame)
                if xy:
                    self.spawn_or_move_sphere("%s_waypoint_%02d" % (self.gazebo_prefix, idx), xy, 0.15, (0.0, 1.0, 0.2, 1.0))

    def publish(self, _event):
        stamp = rospy.Time.now()
        self.publish_markers(stamp)
        self.publish_gazebo_models()


if __name__ == "__main__":
    rospy.init_node("citywalker_visualization_node")
    CityWalkerVisualizationNode()
    rospy.spin()
