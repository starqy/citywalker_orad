#!/usr/bin/env python3
import math

import rospy
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Path
from std_msgs.msg import Bool, Float32, String


class WaypointFollowerNode:
    def __init__(self):
        self.waypoints_topic = rospy.get_param("~waypoints_topic", "/inference_node/waypoints")
        self.arrived_topic = rospy.get_param("~arrived_topic", "/inference_node/arrived_prob")
        self.cmd_topic = rospy.get_param("~cmd_topic", "/cmd_vel")
        self.cmd_msg_type = rospy.get_param("~cmd_msg_type", "Twist")
        self.cmd_frame_id = rospy.get_param("~cmd_frame_id", "base_link")
        self.enable_topic = rospy.get_param("~enable_topic", "/citywalker/enable")
        self.control_rate = float(rospy.get_param("~control_rate", 20.0))
        self.nominal_speed = float(rospy.get_param("~nominal_speed", 0.4))
        self.min_speed = float(rospy.get_param("~min_speed", 0.0))
        self.max_speed = float(rospy.get_param("~max_speed", 0.6))
        self.max_angular = float(rospy.get_param("~max_angular", 1.2))
        self.lookahead_distance = float(rospy.get_param("~lookahead_distance", 1.0))
        self.goal_stop_distance = float(rospy.get_param("~goal_stop_distance", 0.3))
        self.path_timeout = float(rospy.get_param("~path_timeout", 0.5))
        self.arrived_threshold = float(rospy.get_param("~arrived_threshold", 0.5))
        self.stop_on_arrived = bool(rospy.get_param("~stop_on_arrived", True))
        self.enabled = bool(rospy.get_param("~enabled", False))

        self.latest_path = None
        self.latest_path_stamp = rospy.Time(0)
        self.arrived_prob = 0.0

        if self.cmd_msg_type == "TwistStamped":
            self.cmd_pub = rospy.Publisher(self.cmd_topic, TwistStamped, queue_size=1)
        elif self.cmd_msg_type == "Twist":
            self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        else:
            raise rospy.ROSInitException("~cmd_msg_type must be Twist or TwistStamped")
        self.status_pub = rospy.Publisher("~status", String, queue_size=1, latch=True)
        self.path_sub = rospy.Subscriber(self.waypoints_topic, Path, self.path_callback, queue_size=1)
        self.arrived_sub = rospy.Subscriber(self.arrived_topic, Float32, self.arrived_callback, queue_size=5)
        self.enable_sub = rospy.Subscriber(self.enable_topic, Bool, self.enable_callback, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.control_rate), self.control)
        self.set_status("initialized: enabled={}".format(self.enabled))

    def set_status(self, text):
        self.status_pub.publish(String(text))

    def path_callback(self, msg):
        self.latest_path = msg
        self.latest_path_stamp = msg.header.stamp if msg.header.stamp else rospy.Time.now()

    def arrived_callback(self, msg):
        self.arrived_prob = float(msg.data)

    def enable_callback(self, msg):
        self.enabled = bool(msg.data)
        self.set_status("enabled={}".format(self.enabled))
        if not self.enabled:
            self.publish_stop()

    def publish_stop(self):
        self.publish_cmd(0.0, 0.0)

    def make_cmd_msg(self, linear_x, angular_z):
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        if self.cmd_msg_type == "Twist":
            return twist
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.cmd_frame_id
        msg.twist = twist
        return msg

    def publish_cmd(self, linear_x, angular_z):
        self.cmd_pub.publish(self.make_cmd_msg(linear_x, angular_z))

    def select_target(self):
        if self.latest_path is None or not self.latest_path.poses:
            return None
        points = [(pose.pose.position.x, pose.pose.position.y) for pose in self.latest_path.poses]
        for x, y in points:
            if math.hypot(x, y) >= self.lookahead_distance:
                return x, y
        return points[-1]

    def control(self, _event):
        if not self.enabled:
            self.set_status("stop: controller disabled")
            self.publish_stop()
            return
        if self.latest_path is None or (rospy.Time.now() - self.latest_path_stamp).to_sec() > self.path_timeout:
            self.set_status("stop: no fresh waypoint path")
            rospy.logwarn_throttle(2.0, "No fresh waypoint path; stopping")
            self.publish_stop()
            return
        if self.stop_on_arrived and self.arrived_prob >= self.arrived_threshold:
            self.set_status("stop: arrived_prob {:.3f} >= {:.3f}".format(self.arrived_prob, self.arrived_threshold))
            self.publish_stop()
            return

        target = self.select_target()
        if target is None:
            self.set_status("stop: empty waypoint path")
            self.publish_stop()
            return
        x, y = target
        distance = math.hypot(x, y)
        if distance < self.goal_stop_distance:
            self.set_status("stop: target distance {:.3f} < {:.3f}".format(distance, self.goal_stop_distance))
            self.publish_stop()
            return

        speed = max(self.min_speed, min(self.nominal_speed, self.max_speed))
        curvature = 2.0 * y / max(distance * distance, 1e-4)
        angular = max(-self.max_angular, min(self.max_angular, speed * curvature))

        self.publish_cmd(speed, angular)
        self.set_status("track: v={:.3f} w={:.3f} target=({:.3f},{:.3f}) arrived_prob={:.3f}".format(speed, angular, x, y, self.arrived_prob))


if __name__ == "__main__":
    rospy.init_node("waypoint_follower_node")
    WaypointFollowerNode()
    rospy.spin()
