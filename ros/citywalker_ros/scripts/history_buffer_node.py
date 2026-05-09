#!/usr/bin/env python3
from collections import deque

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

from citywalker_ros.geometry import make_point, map_point_to_base, pose_msg_to_xy_yaw
from citywalker_ros.msg import HistoryBundle


class HistoryBufferNode:
    def __init__(self):
        self.image_topic = rospy.get_param("~image_topic", "/camera/image_raw")
        self.pose_topic = rospy.get_param("~pose_topic", "/slam_pose")
        self.pose_msg_type = rospy.get_param("~pose_msg_type", "PoseStamped")
        self.goal_topic = rospy.get_param("~goal_topic", "/global_path_node/local_goal_base")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.context_size = int(rospy.get_param("~context_size", 5))
        self.history_duration = float(rospy.get_param("~history_duration", 2.0))
        self.sample_interval = float(
            rospy.get_param("~sample_interval", self.history_duration / max(1, self.context_size - 1))
        )
        self.publish_rate = float(rospy.get_param("~publish_rate", 10.0))
        self.max_lookup_dt = float(rospy.get_param("~max_lookup_dt", 1.0))
        self.cache_extra = float(rospy.get_param("~cache_extra", 1.0))

        self.images = deque()
        self.poses = deque()
        self.latest_goal = None
        self.latest_goal_stamp = rospy.Time(0)

        self.pub = rospy.Publisher("~history", HistoryBundle, queue_size=1)
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=30)
        msg_type = Odometry if self.pose_msg_type == "Odometry" else PoseStamped
        self.pose_sub = rospy.Subscriber(self.pose_topic, msg_type, self.pose_callback, queue_size=100)
        self.goal_sub = rospy.Subscriber(self.goal_topic, PoseStamped, self.goal_callback, queue_size=10)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.publish)

    def stamp_or_now(self, header):
        return header.stamp if header.stamp and header.stamp.to_sec() > 0.0 else rospy.Time.now()

    def image_callback(self, msg):
        stamp = self.stamp_or_now(msg.header)
        self.images.append((stamp, msg))
        self.trim_cache(stamp)

    def pose_callback(self, msg):
        stamp = self.stamp_or_now(msg.header)
        self.poses.append((stamp, pose_msg_to_xy_yaw(msg)))
        self.trim_cache(stamp)

    def goal_callback(self, msg):
        self.latest_goal = msg
        self.latest_goal_stamp = self.stamp_or_now(msg.header)

    def trim_cache(self, now):
        min_time = now.to_sec() - self.history_duration - self.cache_extra
        while self.images and self.images[0][0].to_sec() < min_time:
            self.images.popleft()
        while self.poses and self.poses[0][0].to_sec() < min_time:
            self.poses.popleft()

    def nearest_item(self, items, target_time):
        if not items:
            return None
        best = min(items, key=lambda item: abs(item[0].to_sec() - target_time))
        if abs(best[0].to_sec() - target_time) > self.max_lookup_dt:
            return None
        return best[1]

    def sample_end_time(self):
        latest_image_time = self.images[-1][0].to_sec()
        latest_pose_time = self.poses[-1][0].to_sec()
        return min(latest_image_time, latest_pose_time)

    def target_point_base(self, current_pose):
        if self.latest_goal is None:
            return None
        point = self.latest_goal.pose.position
        frame_id = self.latest_goal.header.frame_id
        if frame_id == self.base_frame or frame_id == "":
            return make_point(point.x, point.y, point.z)
        x_base, y_base = map_point_to_base((point.x, point.y), current_pose)
        return make_point(x_base, y_base, 0.0)

    def publish(self, _event):
        if len(self.images) < self.context_size or len(self.poses) < self.context_size:
            rospy.logwarn_throttle(
                2.0,
                "Waiting for history: images=%d/%d poses=%d/%d",
                len(self.images),
                self.context_size,
                len(self.poses),
                self.context_size,
            )
            return

        now = rospy.Time.now()
        end_time = self.sample_end_time()
        oldest_needed = end_time - self.sample_interval * (self.context_size - 1)
        if self.images[0][0].to_sec() > oldest_needed or self.poses[0][0].to_sec() > oldest_needed:
            rospy.logwarn_throttle(
                2.0,
                "Waiting for enough history span: need %.2fs, image_span=%.2fs, pose_span=%.2fs",
                self.sample_interval * (self.context_size - 1),
                self.images[-1][0].to_sec() - self.images[0][0].to_sec(),
                self.poses[-1][0].to_sec() - self.poses[0][0].to_sec(),
            )
            return
        sample_times = [
            end_time - self.sample_interval * (self.context_size - 1 - i)
            for i in range(self.context_size)
        ]
        sampled_images = []
        sampled_poses = []
        for sample_time in sample_times:
            image = self.nearest_item(self.images, sample_time)
            pose = self.nearest_item(self.poses, sample_time)
            if image is None or pose is None:
                rospy.logwarn_throttle(
                    2.0,
                    "History sampling failed at t=%.3f. Try increasing ~max_lookup_dt or lowering sample_interval.",
                    sample_time,
                )
                rospy.logwarn_throttle(
                    2.0,
                    "History cache ranges: images %.3f..%.3f poses %.3f..%.3f max_lookup_dt=%.2f",
                    self.images[0][0].to_sec(),
                    self.images[-1][0].to_sec(),
                    self.poses[0][0].to_sec(),
                    self.poses[-1][0].to_sec(),
                    self.max_lookup_dt,
                )
                return
            sampled_images.append(image)
            sampled_poses.append(pose)

        current_pose = sampled_poses[-1]
        relative_positions = []
        for pose in sampled_poses:
            x_base, y_base = map_point_to_base((pose[0], pose[1]), current_pose)
            relative_positions.append(make_point(x_base, y_base, 0.0))

        target = self.target_point_base(current_pose)
        if target is None:
            rospy.logwarn_throttle(2.0, "History bundle has no valid local target yet")
        msg = HistoryBundle()
        msg.header.stamp = now
        msg.header.frame_id = self.base_frame
        msg.images = sampled_images
        msg.relative_positions = relative_positions
        msg.target_valid = target is not None
        msg.target_point = target if target is not None else make_point(0.0, 0.0, 0.0)
        self.pub.publish(msg)


if __name__ == "__main__":
    rospy.init_node("history_buffer_node")
    HistoryBufferNode()
    rospy.spin()
