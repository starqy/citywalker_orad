import json
import os
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset


class OffroadDataset(Dataset):
    def __init__(self, cfg, mode):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.root_dir = cfg.data.root_dir
        self.context_size = cfg.model.obs_encoder.context_size
        self.wp_length = cfg.model.decoder.len_traj_pred
        self.search_window = cfg.data.search_window
        self.arrived_threshold = cfg.data.arrived_threshold
        self.arrived_prob = cfg.data.arrived_prob
        self.default_category = getattr(cfg.data, "default_category", "other")
        self.frame_size = tuple(getattr(cfg.data, "frame_size", cfg.model.obs_encoder.crop))
        self.subdataset_cfg = getattr(cfg.data, "subdatasets", None)
        self.category_names = [
            "crowd",
            "person_close_by",
            "turn",
            "action_target_mismatch",
            "crossing",
            "other",
        ]
        self.default_category_vector = self._build_default_category_vector(self.default_category)

        sequence_dirs_by_dataset = self._discover_sequences(self.root_dir)
        if not sequence_dirs_by_dataset:
            raise FileNotFoundError(f"No offroad sequences found under {self.root_dir}")
        selected_dirs = self._select_sequence_dirs(sequence_dirs_by_dataset, mode)

        self.sequence_infos = []
        self.count = []
        for sequence_dir in selected_dirs:
            sequence_info = self._load_sequence(sequence_dir)
            if sequence_info is None:
                continue
            usable = sequence_info["poses"].shape[0] - self.context_size - max(
                self.arrived_threshold * 2, self.wp_length
            )
            if usable <= 0:
                continue
            self.sequence_infos.append(sequence_info)
            self.count.append(usable)

        if not self.sequence_infos:
            raise ValueError(
                f"No usable offroad sequences found for mode={mode} under {self.root_dir}"
            )

        self.step_scale = []
        for sequence_info in self.sequence_infos:
            positions = sequence_info["poses"][:, :2]
            if positions.shape[0] > 1:
                step_scale = np.linalg.norm(np.diff(positions, axis=0), axis=1).mean()
            else:
                step_scale = 1.0
            self.step_scale.append(max(step_scale, 1e-2))

        self.lut = []
        self.sequence_ranges = []
        idx_counter = 0
        for sequence_idx, usable in enumerate(self.count):
            start_idx = idx_counter
            interval = self.context_size if self.mode == "train" else 1
            for pose_start in range(0, usable, interval):
                self.lut.append((sequence_idx, pose_start))
                idx_counter += 1
            self.sequence_ranges.append((start_idx, idx_counter))

        if not self.lut:
            raise ValueError(f"No valid samples found for mode={mode} under {self.root_dir}")

    def __len__(self):
        return len(self.lut)

    def __getitem__(self, index):
        sequence_idx, pose_start = self.lut[index]
        sequence_info = self.sequence_infos[sequence_idx]
        poses = sequence_info["poses"]
        image_paths = sequence_info["image_paths"]

        future_poses = poses[
            pose_start + self.context_size : pose_start + self.context_size + self.search_window
        ]
        if future_poses.shape[0] == 0:
            raise IndexError(f"No future poses available for index {pose_start}")

        target_idx, arrived = self.select_target_index(future_poses)

        input_poses = poses[pose_start : pose_start + self.context_size]
        waypoint_start = pose_start + self.context_size
        waypoint_end = waypoint_start + self.wp_length
        gt_waypoint_poses = poses[waypoint_start:waypoint_end]

        current_pose = input_poses[-1]
        history_positions = self.transform_poses(input_poses, current_pose)
        gt_waypoints = self.transform_poses(gt_waypoint_poses, current_pose)

        target_pose = poses[pose_start + self.context_size + target_idx]
        target_transformed = self.transform_pose(target_pose, current_pose)

        history_positions_xy = history_positions[:, :2]
        target_position_xy = target_transformed[np.newaxis, :2]

        if self.cfg.model.cord_embedding.type == "polar":
            input_positions = self.input2target(history_positions_xy, target_transformed[:2])
            if self.mode == "train":
                rand_angle = np.random.uniform(-np.pi, np.pi)
                rot_matrix = np.array(
                    [
                        [np.cos(rand_angle), -np.sin(rand_angle)],
                        [np.sin(rand_angle), np.cos(rand_angle)],
                    ]
                )
                input_positions = input_positions @ rot_matrix.T
        elif self.cfg.model.cord_embedding.type == "input_target":
            input_positions = np.concatenate([history_positions_xy, target_position_xy], axis=0)
        else:
            raise NotImplementedError(
                f"Coordinate embedding type {self.cfg.model.cord_embedding.type} not implemented"
            )

        input_positions = torch.tensor(input_positions, dtype=torch.float32)
        arrived = torch.tensor(arrived, dtype=torch.float32)

        frame_paths = image_paths[pose_start : pose_start + self.context_size]
        frames = self.load_frames(frame_paths)

        waypoints_transformed = torch.tensor(gt_waypoints[:, :2], dtype=torch.float32)
        step_scale = torch.tensor(self.step_scale[sequence_idx], dtype=torch.float32)
        waypoints_scaled = waypoints_transformed / step_scale
        input_positions_scaled = input_positions / step_scale

        sample = {
            "video_frames": frames,
            "input_positions": input_positions_scaled,
            "waypoints": waypoints_scaled,
            "arrived": arrived,
            "step_scale": step_scale,
            "vis_history_positions": torch.tensor(history_positions[:, :2], dtype=torch.float32),
            "vis_gt_waypoints": waypoints_transformed,
            "vis_target_transformed": torch.tensor(target_transformed[:2], dtype=torch.float32),
            "vis_current_pose_raw": torch.tensor(current_pose, dtype=torch.float32),
            "vis_history_world_positions": torch.tensor(input_poses[:, :3], dtype=torch.float32),
            "vis_gt_waypoints_world": torch.tensor(gt_waypoint_poses[:, :3], dtype=torch.float32),
            "vis_target_world_position": torch.tensor(target_pose[:3], dtype=torch.float32),
        }

        if self.mode in ["val", "test"]:
            if self.cfg.model.cord_embedding.type == "polar":
                sample["original_input_positions"] = sample["vis_history_positions"]
                sample["noisy_input_positions"] = sample["vis_history_positions"]
                sample["gt_waypoints"] = waypoints_transformed
                sample["target_transformed"] = sample["vis_target_transformed"]
            elif self.cfg.model.cord_embedding.type == "input_target":
                sample["original_input_positions"] = sample["vis_history_positions"]
                sample["noisy_input_positions"] = input_positions[:-1, :]
                sample["gt_waypoints"] = waypoints_transformed
                sample["target_transformed"] = input_positions[-1, :]

        if self.mode == "test":
            sample["categories"] = torch.tensor(
                self.default_category_vector, dtype=torch.float32
            )

        return sample

    def _build_default_category_vector(self, category_name):
        category_vector = np.zeros(len(self.category_names), dtype=np.float32)
        try:
            category_vector[self.category_names.index(category_name)] = 1.0
        except ValueError:
            category_vector[-1] = 1.0
        return category_vector

    def _discover_sequences(self, root_dir):
        sequence_dirs_by_dataset = {}
        for dirpath, dirnames, _ in os.walk(root_dir):
            if {"image", "poses"}.issubset(set(dirnames)):
                rel_path = os.path.relpath(dirpath, root_dir)
                dataset_name = rel_path.split(os.sep)[0]
                sequence_dirs_by_dataset.setdefault(dataset_name, []).append(dirpath)
        for dataset_name in sequence_dirs_by_dataset:
            sequence_dirs_by_dataset[dataset_name].sort()
        return sequence_dirs_by_dataset

    def _select_sequence_dirs(self, sequence_dirs_by_dataset, mode):
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Invalid mode {mode}")

        if self.subdataset_cfg is None:
            flat_sequence_dirs = []
            for dataset_name in sorted(sequence_dirs_by_dataset):
                flat_sequence_dirs.extend(sequence_dirs_by_dataset[dataset_name])
            return self._select_global_slices(flat_sequence_dirs, mode)

        selected_dirs = []
        subdataset_cfg = vars(self.subdataset_cfg)
        for dataset_name, split_cfg in subdataset_cfg.items():
            if dataset_name not in sequence_dirs_by_dataset:
                continue
            if split_cfg is None:
                continue
            train_count = int(getattr(split_cfg, "train", 0) or 0)
            val_count = int(getattr(split_cfg, "val", 0) or 0)
            test_count = int(getattr(split_cfg, "test", 0) or 0)
            sequence_dirs = sequence_dirs_by_dataset[dataset_name]
            if mode == "train":
                start = 0
                end = train_count
            elif mode == "val":
                start = train_count
                end = train_count + val_count
            else:
                start = train_count + val_count
                end = train_count + val_count + test_count
            selected_dirs.extend(sequence_dirs[start:end])
        return selected_dirs

    def _select_global_slices(self, sequence_dirs, mode):
        if mode == "train":
            return sequence_dirs[: self.cfg.data.num_train]
        if mode == "val":
            start = self.cfg.data.num_train
            end = start + self.cfg.data.num_val
            return sequence_dirs[start:end]
        start = self.cfg.data.num_train + self.cfg.data.num_val
        end = start + self.cfg.data.num_test
        return sequence_dirs[start:end]

    def _load_sequence(self, sequence_dir):
        image_dir = os.path.join(sequence_dir, "image")
        pose_dir = os.path.join(sequence_dir, "poses")

        image_map = {
            os.path.splitext(file_name)[0]: os.path.join(image_dir, file_name)
            for file_name in os.listdir(image_dir)
            if file_name.lower().endswith((".png", ".jpg", ".jpeg"))
        }
        pose_map = {
            os.path.splitext(file_name)[0]: os.path.join(pose_dir, file_name)
            for file_name in os.listdir(pose_dir)
            if file_name.endswith(".json")
        }

        common_keys = sorted(set(image_map) & set(pose_map), key=self._sort_key)
        if not common_keys:
            return None

        poses = []
        image_paths = []
        for key in common_keys:
            pose = self._load_pose_json(pose_map[key])
            if pose is None:
                continue
            poses.append(pose)
            image_paths.append(image_map[key])

        if not poses:
            return None

        return {
            "sequence_dir": sequence_dir,
            "poses": np.array(poses, dtype=np.float64),
            "image_paths": image_paths,
        }

    def _load_pose_json(self, pose_path):
        with open(pose_path, "r") as file:
            pose_json = json.load(file)

        position = pose_json.get("position")
        orientation = pose_json.get("orientation")
        if position is None or orientation is None:
            return None

        return [
            float(position["x"]),
            float(position["y"]),
            float(position["z"]),
            float(orientation["x"]),
            float(orientation["y"]),
            float(orientation["z"]),
            float(orientation["w"]),
        ]

    def _sort_key(self, key):
        try:
            return (0, int(key))
        except ValueError:
            return (1, key)

    def input2target(self, input_positions, target_position):
        return input_positions - target_position

    def transform_input(self, input_positions):
        current_position = input_positions[-1]
        translated_input = input_positions - current_position
        second_last = translated_input[-2]
        angle = -np.pi / 2 - np.arctan2(second_last[1], second_last[0])
        rotation_matrix = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        return np.dot(translated_input[:, :2], rotation_matrix.T)

    def select_target_index(self, future_positions):
        arrived = np.random.rand() < self.arrived_prob
        max_idx = future_positions.shape[0] - 1
        if arrived:
            target_idx = random.randint(
                self.wp_length, min(self.wp_length + self.arrived_threshold, max_idx)
            )
        else:
            target_idx = random.randint(self.wp_length + self.arrived_threshold, max_idx)
        return target_idx, arrived

    def transform_poses(self, poses, current_pose_array):
        current_pose_matrix = self.pose_to_matrix(current_pose_array)
        current_pose_inv = np.linalg.inv(current_pose_matrix)
        pose_matrices = self.poses_to_matrices(poses)
        transformed_matrices = np.matmul(current_pose_inv[np.newaxis, :, :], pose_matrices)
        return transformed_matrices[:, :3, 3]

    def transform_pose(self, pose, current_pose_array):
        current_pose_matrix = self.pose_to_matrix(current_pose_array)
        current_pose_inv = np.linalg.inv(current_pose_matrix)
        pose_matrix = self.pose_to_matrix(pose)
        transformed_matrix = np.matmul(current_pose_inv, pose_matrix)
        return transformed_matrix[:3, 3]

    def pose_to_matrix(self, pose):
        position = pose[:3]
        rotation = R.from_quat(pose[3:])
        matrix = np.eye(4)
        matrix[:3, :3] = rotation.as_matrix()
        matrix[:3, 3] = position
        return matrix

    def poses_to_matrices(self, poses):
        positions = poses[:, :3]
        quats = poses[:, 3:]
        rotations = R.from_quat(quats)
        matrices = np.tile(np.eye(4), (poses.shape[0], 1, 1))
        matrices[:, :3, :3] = rotations.as_matrix()
        matrices[:, :3, 3] = positions
        return matrices

    def load_frames(self, image_paths):
        frames = []
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            image = TF.resize(image, self.frame_size, antialias=True)
            frames.append(TF.to_tensor(image))
        return torch.stack(frames)
