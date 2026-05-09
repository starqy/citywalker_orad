#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path

from citywalker_ros.geometry import (
    load_path_file,
    lookahead_index,
    make_path_msg,
    make_pose_stamped,
    map_point_to_base,
    nearest_path_index,
    pose_msg_to_xy_yaw,
)


def build_spaced_waypoints(path, spacing):
    if len(path) == 0:
        return path
    if spacing <= 0.0:
        return path

    waypoints = [path[0]]
    accum = 0.0
    last = path[0]
    for idx in range(1, len(path)):
        current = path[idx]
        seg_len = ((current[0] - last[0]) ** 2 + (current[1] - last[1]) ** 2) ** 0.5
        accum += seg_len
        if accum >= spacing:
            waypoints.append(current)
            accum = 0.0
        last = current
    if not (waypoints[-1][0] == path[-1][0] and waypoints[-1][1] == path[-1][1]):
        waypoints.append(path[-1])
    return np.asarray(waypoints, dtype=path.dtype)


class GlobalPathNode:
    def __init__(self):
        self.path_file = rospy.get_param("~path_file", "")
        self.path_frame = rospy.get_param("~path_frame", "map")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.pose_topic = rospy.get_param("~pose_topic", "/slam_pose")
        self.pose_msg_type = rospy.get_param("~pose_msg_type", "PoseStamped")
        self.goal_mode = rospy.get_param("~goal_mode", "waypoint_sequence")
        self.lookahead_distance = float(rospy.get_param("~lookahead_distance", 5.0))
        self.waypoint_spacing = float(rospy.get_param("~waypoint_spacing", 5.0))
        self.switch_distance = float(rospy.get_param("~switch_distance", 1.0))
        self.search_window = int(rospy.get_param("~nearest_search_window", 300))
        self.publish_rate = float(rospy.get_param("~publish_rate", 10.0))

        if not self.path_file:
            raise rospy.ROSInitException("~path_file is required")
        self.path = load_path_file(self.path_file)
        if len(self.path) < 2:
            raise rospy.ROSInitException("Path must contain at least two points")
        self.path_xy = self.path[:, :2]
        self.goal_waypoints = build_spaced_waypoints(self.path, self.waypoint_spacing)
        self.goal_waypoints_xy = self.goal_waypoints[:, :2]
        self.goal_idx = 0
        self.nearest_idx = 0
        self.current_pose = None
        self.last_pose_stamp = rospy.Time(0)

        self.path_pub = rospy.Publisher("~reference_path", Path, queue_size=1, latch=True)
        self.goal_sequence_pub = rospy.Publisher("~goal_sequence", Path, queue_size=1, latch=True)
        self.goal_map_pub = rospy.Publisher("~local_goal_map", PoseStamped, queue_size=1)
        self.goal_base_pub = rospy.Publisher("~local_goal_base", PoseStamped, queue_size=1)

        msg_type = Odometry if self.pose_msg_type == "Odometry" else PoseStamped
        self.pose_sub = rospy.Subscriber(self.pose_topic, msg_type, self.pose_callback, queue_size=20)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.publish)
        rospy.loginfo("global_path_node loaded %d path points from %s", len(self.path), self.path_file)
        rospy.loginfo(
            "global_path_node goal_mode=%s waypoint_spacing=%.2f switch_distance=%.2f generated_goals=%d",
            self.goal_mode,
            self.waypoint_spacing,
            self.switch_distance,
            len(self.goal_waypoints),
        )

    def pose_callback(self, msg):
        self.current_pose = pose_msg_to_xy_yaw(msg)
        self.last_pose_stamp = msg.header.stamp if msg.header.stamp else rospy.Time.now()

    def advance_sequence_goal(self):
        if len(self.goal_waypoints) == 0:
            return None
        while self.goal_idx < len(self.goal_waypoints) - 1:
            goal = self.goal_waypoints[self.goal_idx]
            dx = goal[0] - self.current_pose[0]
            dy = goal[1] - self.current_pose[1]
            if (dx * dx + dy * dy) ** 0.5 > self.switch_distance:
                break
            self.goal_idx += 1
            rospy.loginfo("global_path_node switched to fixed goal %d/%d", self.goal_idx + 1, len(self.goal_waypoints))
        return self.goal_waypoints[self.goal_idx]

    def current_goal(self):
        if self.goal_mode == "lookahead":
            self.nearest_idx = nearest_path_index(
                self.path_xy,
                self.current_pose[:2],
                start_index=self.nearest_idx,
                search_window=self.search_window,
            )
            goal_idx = lookahead_index(self.path_xy, self.nearest_idx, self.lookahead_distance)
            return self.path[goal_idx]
        if self.goal_mode == "waypoint_sequence":
            return self.advance_sequence_goal()
        raise rospy.ROSException("Unsupported ~goal_mode: {}".format(self.goal_mode))

    def publish(self, _event):
        stamp = rospy.Time.now()
        self.path_pub.publish(make_path_msg(self.path, self.path_frame, stamp))
        self.goal_sequence_pub.publish(make_path_msg(self.goal_waypoints, self.path_frame, stamp))
        if self.current_pose is None:
            return

        goal = self.current_goal()
        if goal is None:
            return

        goal_map = make_pose_stamped(goal[0], goal[1], goal[2], self.path_frame, stamp)
        gx_base, gy_base = map_point_to_base(goal[:2], self.current_pose)
        goal_base = make_pose_stamped(gx_base, gy_base, 0.0, self.base_frame, stamp)
        self.goal_map_pub.publish(goal_map)
        self.goal_base_pub.publish(goal_base)


if __name__ == "__main__":
    rospy.init_node("global_path_node")
    try:
        GlobalPathNode()
        rospy.spin()
    except Exception as exc:
        rospy.logerr("global_path_node failed: %s", exc)
        raise
