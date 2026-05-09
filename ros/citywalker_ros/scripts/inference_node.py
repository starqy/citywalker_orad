#!/usr/bin/env python3
import os
import sys

import numpy as np
import rospy
import torch
from nav_msgs.msg import Path
from std_msgs.msg import Float32, String

from citywalker_ros.geometry import make_pose_stamped
from citywalker_ros.msg import HistoryBundle
from citywalker_ros.trt_model import CityWalkerTRTModel, load_config


class InferenceNode:
    def __init__(self):
        self.repo_root = rospy.get_param(
            "~repo_root",
            "/media/isee324/2a90eb70-2d62-4af4-b04a-0fcdde4122a5/qy/AD/citywalker_orad",
        )
        if self.repo_root not in sys.path:
            sys.path.insert(0, self.repo_root)

        self.config_path = rospy.get_param("~config", os.path.join(self.repo_root, "config/offroad_finetune.yaml"))
        self.checkpoint_path = rospy.get_param("~checkpoint", os.path.join(self.repo_root, "ckpts/epoch=14-step=134983.ckpt"))
        self.engine_path = rospy.get_param(
            "~engine",
            os.path.join(self.repo_root, "exports/dinov2_vitb14_offroad_392x392_fp16.engine"),
        )
        self.history_topic = rospy.get_param("~history_topic", "/history_buffer_node/history")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.dtype = rospy.get_param("~dtype", "fp16")
        self.step_scale = float(rospy.get_param("~step_scale", 1.0))
        self.image_encoding = rospy.get_param("~image_encoding", "rgb8")
        self.max_bundle_age = float(rospy.get_param("~max_bundle_age", 0.5))
        self.publish_arrived_threshold = float(rospy.get_param("~arrived_threshold", 0.5))

        if self.step_scale <= 0.0:
            raise rospy.ROSInitException("~step_scale must be positive")

        cfg = load_config(self.config_path)
        self.context_size = cfg.model.obs_encoder.context_size
        self.model = CityWalkerTRTModel(
            cfg,
            engine_path=self.engine_path,
            checkpoint_path=self.checkpoint_path,
            repo_root=self.repo_root,
            dtype=self.dtype,
        ).cuda().eval()

        self.path_pub = rospy.Publisher("~waypoints", Path, queue_size=1)
        self.arrived_pub = rospy.Publisher("~arrived_prob", Float32, queue_size=1)
        self.status_pub = rospy.Publisher("~status", String, queue_size=1, latch=True)
        self.sub = rospy.Subscriber(self.history_topic, HistoryBundle, self.callback, queue_size=1)
        rospy.loginfo("inference_node loaded TensorRT backbone engine: %s", self.engine_path)

    def set_status(self, text):
        self.status_pub.publish(String(text))

    def image_msg_to_tensor(self, msg):
        encoding = (msg.encoding or self.image_encoding).lower()
        if encoding in {"bgr8", "rgb8"}:
            channels = 3
        elif encoding in {"bgra8", "rgba8"}:
            channels = 4
        elif encoding in {"mono8", "8uc1"}:
            channels = 1
        else:
            raise ValueError("Unsupported image encoding: {}".format(msg.encoding))

        row = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        image = row[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
        if channels == 4:
            image = image[:, :, :3]
            channels = 3
        if channels == 1:
            image = np.repeat(image, 3, axis=2)
        elif encoding == "bgr8" or encoding == "bgra8":
            image = image[:, :, ::-1]

        tensor = torch.from_numpy(np.ascontiguousarray(image)).cuda(non_blocking=True)
        tensor = tensor.permute(2, 0, 1).float() / 255.0
        return tensor

    def callback(self, msg):
        if not msg.target_valid:
            self.set_status("skip: target invalid")
            rospy.logwarn_throttle(2.0, "Skipping inference because target is not valid")
            return
        if len(msg.images) != self.context_size or len(msg.relative_positions) != self.context_size:
            self.set_status("skip: invalid history bundle size")
            rospy.logwarn_throttle(2.0, "Invalid history bundle size")
            return
        if msg.header.stamp and (rospy.Time.now() - msg.header.stamp).to_sec() > self.max_bundle_age:
            self.set_status("skip: stale history bundle")
            rospy.logwarn_throttle(2.0, "Skipping stale history bundle")
            return

        try:
            frames = torch.stack([self.image_msg_to_tensor(image) for image in msg.images], dim=0).unsqueeze(0)
            coords = [[point.x, point.y] for point in msg.relative_positions]
            coords.append([msg.target_point.x, msg.target_point.y])
            cord = torch.tensor(coords, dtype=torch.float32, device="cuda").unsqueeze(0) / self.step_scale

            with torch.inference_mode():
                wp_pred, arrive_logits = self.model(frames, cord)
            waypoints = (wp_pred[0].detach().float().cpu().numpy() * self.step_scale)
            arrive_prob = torch.sigmoid(arrive_logits[0, 0]).detach().float().cpu().item()
        except Exception as exc:
            self.set_status("error: {}".format(exc))
            rospy.logerr_throttle(2.0, "Inference failed: %s", exc)
            return

        stamp = rospy.Time.now()
        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = self.base_frame
        for x, y in waypoints:
            path.poses.append(make_pose_stamped(float(x), float(y), 0.0, self.base_frame, stamp))
        self.path_pub.publish(path)
        self.arrived_pub.publish(Float32(arrive_prob))
        self.set_status("ok: published {} waypoints, arrived_prob={:.3f}".format(len(path.poses), arrive_prob))


if __name__ == "__main__":
    rospy.init_node("inference_node")
    InferenceNode()
    rospy.spin()
