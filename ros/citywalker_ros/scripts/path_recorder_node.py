#!/usr/bin/env python3
import csv
import math
import os

import rospy
import yaml
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

from citywalker_ros.geometry import make_path_msg, pose_msg_to_xy_yaw


class PathRecorderNode:
    def __init__(self):
        self.pose_topic = rospy.get_param("~pose_topic", "/slam_pose")
        self.pose_msg_type = rospy.get_param("~pose_msg_type", "Odometry")
        self.output_file = rospy.get_param("~output_file", "recorded_path.csv")
        self.frame_id = rospy.get_param("~frame_id", "map")
        self.min_distance = float(rospy.get_param("~min_distance", 0.3))
        self.min_yaw_delta = float(rospy.get_param("~min_yaw_delta", 0.25))
        self.final_sparsify_distance = float(rospy.get_param("~final_sparsify_distance", self.min_distance))
        self.save_format = rospy.get_param("~save_format", "csv").lower()
        self.autostart = bool(rospy.get_param("~autostart", True))
        self.publish_rate = float(rospy.get_param("~publish_rate", 2.0))

        self.recording = self.autostart
        self.points = []
        self.last_saved = None

        self.path_pub = rospy.Publisher("~recorded_path", Path, queue_size=1, latch=True)
        self.enable_sub = rospy.Subscriber("~recording", Bool, self.recording_callback, queue_size=1)
        msg_type = Odometry if self.pose_msg_type == "Odometry" else PoseStamped
        self.pose_sub = rospy.Subscriber(self.pose_topic, msg_type, self.pose_callback, queue_size=100)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.publish_path)
        rospy.on_shutdown(self.save)
        rospy.loginfo("path_recorder_node writing to %s, recording=%s", self.output_file, self.recording)

    def recording_callback(self, msg):
        self.recording = bool(msg.data)
        rospy.loginfo("Path recording set to %s", self.recording)
        if not self.recording:
            self.save()

    def angle_diff(self, a, b):
        return math.atan2(math.sin(a - b), math.cos(a - b))

    def should_keep(self, point):
        if self.last_saved is None:
            return True
        dx = point[0] - self.last_saved[0]
        dy = point[1] - self.last_saved[1]
        dist = math.hypot(dx, dy)
        yaw_delta = abs(self.angle_diff(point[2], self.last_saved[2]))
        return dist >= self.min_distance or yaw_delta >= self.min_yaw_delta

    def pose_callback(self, msg):
        if not self.recording:
            return
        x, y, yaw = pose_msg_to_xy_yaw(msg)
        point = (x, y, yaw)
        if self.should_keep(point):
            self.points.append(point)
            self.last_saved = point

    def sparsify(self, points):
        if not points:
            return []
        sparse = [points[0]]
        for point in points[1:]:
            dx = point[0] - sparse[-1][0]
            dy = point[1] - sparse[-1][1]
            yaw_delta = abs(self.angle_diff(point[2], sparse[-1][2]))
            if math.hypot(dx, dy) >= self.final_sparsify_distance or yaw_delta >= self.min_yaw_delta:
                sparse.append(point)
        if sparse[-1] != points[-1]:
            sparse.append(points[-1])
        return sparse

    def publish_path(self, _event):
        if not self.points:
            return
        self.path_pub.publish(make_path_msg(self.points, self.frame_id, rospy.Time.now()))

    def save(self):
        sparse = self.sparsify(self.points)
        if not sparse:
            return
        output_dir = os.path.dirname(os.path.abspath(self.output_file))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if self.save_format == "yaml" or self.output_file.endswith((".yaml", ".yml")):
            data = {"frame_id": self.frame_id, "points": [{"x": x, "y": y, "yaw": yaw} for x, y, yaw in sparse]}
            with open(self.output_file, "w") as f:
                yaml.safe_dump(data, f, default_flow_style=False)
        else:
            with open(self.output_file, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y", "yaw"])
                for x, y, yaw in sparse:
                    writer.writerow([f"{x:.6f}", f"{y:.6f}", f"{yaw:.6f}"])
        rospy.loginfo("Saved %d sparse path points to %s", len(sparse), self.output_file)


if __name__ == "__main__":
    rospy.init_node("path_recorder_node")
    PathRecorderNode()
    rospy.spin()
