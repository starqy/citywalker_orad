import argparse
import os
import random
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.offroad_dataset import OffroadDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class DictNamespace(argparse.Namespace):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, DictNamespace(**value))
            else:
                setattr(self, key, value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize and sanity-check samples produced by OffroadDataset."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/offroad_finetune.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/offroad_dataset_vis",
        help="Directory for rendered PNG files.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print sequence-level summary before rendering.",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Sequence basename or unique path fragment. If omitted, global indices are used.",
    )
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Comma-separated global sample indices, or local indices when --sequence is set.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of samples to render when --indices is not set.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride for automatic sequence-local sample selection.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Randomly choose samples instead of taking evenly spaced samples.",
    )
    parser.add_argument(
        "--per-subdataset",
        action="store_true",
        help="Randomly render samples from every selected subdataset in the split.",
    )
    parser.add_argument(
        "--samples-per-subdataset",
        type=int,
        default=3,
        help="Number of random samples rendered for each subdataset when --per-subdataset is set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for deterministic target selection and optional random sampling.",
    )
    parser.add_argument(
        "--max-summary",
        type=int,
        default=30,
        help="Maximum number of sequences printed by --summary. Use -1 for all.",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as file:
        cfg_dict = yaml.safe_load(file)
    return DictNamespace(**cfg_dict)


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def parse_indices(indices_arg):
    if indices_arg is None or indices_arg.strip() == "":
        return None
    return [int(item.strip()) for item in indices_arg.split(",") if item.strip()]


def find_sequence_index(dataset, query):
    matches = []
    for idx, info in enumerate(dataset.sequence_infos):
        sequence_dir = info["sequence_dir"]
        basename = os.path.basename(sequence_dir)
        if query == basename or query in sequence_dir:
            matches.append((idx, sequence_dir))
    if not matches:
        raise FileNotFoundError(f"No sequence matching '{query}' found in split '{dataset.mode}'")
    if len(matches) > 1:
        matched_paths = "\n".join(path for _, path in matches[:20])
        raise ValueError(f"Sequence query '{query}' is ambiguous. Matches:\n{matched_paths}")
    return matches[0]


def choose_global_indices(dataset, args):
    if args.per_subdataset:
        return choose_indices_per_subdataset(dataset, args), "per-subdataset random indices"

    explicit_indices = parse_indices(args.indices)
    if args.sequence is not None:
        sequence_idx, sequence_dir = find_sequence_index(dataset, args.sequence)
        start, end = dataset.sequence_ranges[sequence_idx]
        local_count = end - start
        if explicit_indices is not None:
            indices = [start + local_idx for local_idx in explicit_indices]
        elif args.random:
            rng = random.Random(args.seed)
            local_indices = rng.sample(range(local_count), k=min(args.num_samples, local_count))
            indices = [start + local_idx for local_idx in sorted(local_indices)]
        else:
            indices = list(range(start, end, max(args.stride, 1)))[: args.num_samples]
        return indices, f"sequence={sequence_dir}"

    if explicit_indices is not None:
        return explicit_indices, "global indices"

    if args.random:
        rng = random.Random(args.seed)
        indices = rng.sample(range(len(dataset)), k=min(args.num_samples, len(dataset)))
        return sorted(indices), "random global indices"

    if args.num_samples <= 1:
        return [0], "first global index"
    indices = np.linspace(0, len(dataset) - 1, num=min(args.num_samples, len(dataset)), dtype=int)
    return sorted(set(indices.tolist())), "evenly spaced global indices"


def get_subdataset_name(dataset, sequence_idx):
    sequence_dir = dataset.sequence_infos[sequence_idx]["sequence_dir"]
    rel_path = os.path.relpath(sequence_dir, dataset.root_dir)
    return rel_path.split(os.sep)[0]


def choose_indices_per_subdataset(dataset, args):
    rng = random.Random(args.seed)
    indices_by_subdataset = {}
    for global_idx, (sequence_idx, _) in enumerate(dataset.lut):
        subdataset_name = get_subdataset_name(dataset, sequence_idx)
        indices_by_subdataset.setdefault(subdataset_name, []).append(global_idx)

    selected_indices = []
    for subdataset_name in sorted(indices_by_subdataset):
        candidates = indices_by_subdataset[subdataset_name]
        sample_count = min(args.samples_per_subdataset, len(candidates))
        sampled = sorted(rng.sample(candidates, k=sample_count))
        print(
            f"Selected {len(sampled)} sample(s) from {subdataset_name} "
            f"out of {len(candidates)} available samples"
        )
        selected_indices.extend(sampled)
    return selected_indices


def print_summary(dataset, max_summary):
    print(f"Split: {dataset.mode}")
    print(f"Dataset samples: {len(dataset)}")
    print(f"Usable sequences: {len(dataset.sequence_infos)}")
    print(f"Context size: {dataset.context_size}")
    print(f"Waypoint length: {dataset.wp_length}")
    print(f"Search window: {dataset.search_window}")
    print("Sequences:")
    limit = len(dataset.sequence_infos) if max_summary < 0 else min(max_summary, len(dataset.sequence_infos))
    for idx in range(limit):
        info = dataset.sequence_infos[idx]
        rel_dir = os.path.relpath(info["sequence_dir"], dataset.root_dir)
        start, end = dataset.sequence_ranges[idx]
        print(
            f"  [{idx:04d}] samples={end - start:5d} "
            f"frames={len(info['image_paths']):5d} "
            f"usable={dataset.count[idx]:5d} "
            f"step_scale={dataset.step_scale[idx]:.4f} "
            f"dir={rel_dir}"
        )
    if limit < len(dataset.sequence_infos):
        print(f"  ... {len(dataset.sequence_infos) - limit} more sequences not shown")


def axis_limits_from_points(point_groups, padding=0.5):
    valid_groups = []
    for points in point_groups:
        points = np.asarray(points)
        if points.size == 0:
            continue
        valid = points[np.isfinite(points).all(axis=1)]
        if valid.size > 0:
            valid_groups.append(valid[:, :2])
    if not valid_groups:
        return (-1, 1), (-1, 1)
    stacked = np.concatenate(valid_groups, axis=0)
    min_xy = stacked.min(axis=0)
    max_xy = stacked.max(axis=0)
    center = (min_xy + max_xy) / 2
    half_range = max((max_xy - min_xy).max() / 2 + padding, 1.0)
    return (center[0] - half_range, center[0] + half_range), (center[1] - half_range, center[1] + half_range)


def plot_points(ax, points, style, label, color, marker_size=6):
    points = np.asarray(points)
    if points.size == 0:
        return
    ax.plot(points[:, 0], points[:, 1], style, label=label, color=color, markersize=marker_size)


def sample_diagnostics(dataset, sample, global_idx):
    sequence_idx, pose_start = dataset.lut[global_idx]
    step_scale = float(to_numpy(sample["step_scale"]))
    input_positions = to_numpy(sample["input_positions"]) * step_scale
    history_ego = to_numpy(sample["vis_history_positions"])
    target_ego = to_numpy(sample["vis_target_transformed"])
    gt_waypoints = to_numpy(sample["vis_gt_waypoints"])

    diagnostics = []
    diagnostics.append(f"global_idx={global_idx}")
    diagnostics.append(f"sequence_idx={sequence_idx}")
    diagnostics.append(f"pose_start={pose_start}")
    diagnostics.append(f"step_scale={step_scale:.4f}")
    diagnostics.append(f"frames={tuple(sample['video_frames'].shape)}")
    diagnostics.append(f"input={tuple(sample['input_positions'].shape)}")
    diagnostics.append(f"waypoints={tuple(sample['waypoints'].shape)}")

    if getattr(dataset.cfg.model.cord_embedding, "type", None) == "input_target":
        encoded_history = input_positions[:-1]
        encoded_target = input_positions[-1]
        history_error = np.linalg.norm(encoded_history - history_ego, axis=1).max()
        target_error = np.linalg.norm(encoded_target - target_ego)
        diagnostics.append(f"encoded_history_vs_ego_max={history_error:.4f}m")
        diagnostics.append(f"encoded_target_vs_ego={target_error:.4f}m")

    finite_ok = all(
        np.isfinite(array).all()
        for array in [input_positions, history_ego, target_ego, gt_waypoints]
    )
    diagnostics.append(f"finite={finite_ok}")
    return diagnostics


def render_sample(dataset, sample, global_idx, output_dir):
    sequence_idx, pose_start = dataset.lut[global_idx]
    sequence_info = dataset.sequence_infos[sequence_idx]
    sequence_name = os.path.basename(sequence_info["sequence_dir"])
    subdataset_name = get_subdataset_name(dataset, sequence_idx)
    step_scale = float(to_numpy(sample["step_scale"]))

    frames = to_numpy(sample["video_frames"])
    input_positions = to_numpy(sample["input_positions"]) * step_scale
    history_ego = to_numpy(sample["vis_history_positions"])
    gt_waypoints = to_numpy(sample["vis_gt_waypoints"])
    target_ego = to_numpy(sample["vis_target_transformed"])
    history_world = to_numpy(sample["vis_history_world_positions"])
    gt_world = to_numpy(sample["vis_gt_waypoints_world"])
    target_world = to_numpy(sample["vis_target_world_position"])

    if getattr(dataset.cfg.model.cord_embedding, "type", None) == "input_target":
        encoded_history = input_positions[:-1]
        encoded_target = input_positions[-1]
    else:
        encoded_history = input_positions
        encoded_target = None

    context_size = frames.shape[0]
    ncols = max(context_size, 2)
    fig = plt.figure(figsize=(4 * ncols, 9))
    grid = fig.add_gridspec(2, ncols, height_ratios=[1, 1.5])

    for frame_idx in range(context_size):
        ax = fig.add_subplot(grid[0, frame_idx])
        frame = np.transpose(frames[frame_idx], (1, 2, 0))
        ax.imshow(np.clip(frame, 0, 1))
        ax.axis("off")
        title = "current" if frame_idx == context_size - 1 else f"t-{context_size - 1 - frame_idx}"
        ax.set_title(title)

    ego_ax = fig.add_subplot(grid[1, : ncols // 2])
    world_ax = fig.add_subplot(grid[1, ncols // 2 :])

    plot_points(ego_ax, history_ego, "o-", "history ego", "#5771DB")
    plot_points(ego_ax, encoded_history, "x--", "encoded history", "#DBC257")
    plot_points(ego_ax, gt_waypoints, "s-", "gt waypoints", "#42A948")
    ego_ax.plot(target_ego[0], target_ego[1], "*", label="target ego", color="#A157DB", markersize=14)
    if encoded_target is not None:
        ego_ax.plot(
            encoded_target[0],
            encoded_target[1],
            "P",
            label="encoded target",
            color="#E08D2D",
            markersize=10,
        )
    ego_xlim, ego_ylim = axis_limits_from_points(
        [history_ego, encoded_history, gt_waypoints, target_ego[None, :]], padding=0.5
    )
    ego_ax.set_xlim(*ego_xlim)
    ego_ax.set_ylim(*ego_ylim)
    ego_ax.set_aspect("equal", adjustable="box")
    ego_ax.grid(True)
    ego_ax.legend(fontsize=8)
    ego_ax.set_title("Current-pose / model-input frame")
    ego_ax.set_xlabel("x")
    ego_ax.set_ylabel("y")

    plot_points(world_ax, history_world[:, :2], "o-", "history world", "#5771DB")
    plot_points(world_ax, gt_world[:, :2], "s-", "gt waypoints world", "#42A948")
    world_ax.plot(
        target_world[0],
        target_world[1],
        "*",
        label="target world",
        color="#A157DB",
        markersize=14,
    )
    world_xlim, world_ylim = axis_limits_from_points(
        [history_world[:, :2], gt_world[:, :2], target_world[None, :2]], padding=0.5
    )
    world_ax.set_xlim(*world_xlim)
    world_ax.set_ylim(*world_ylim)
    world_ax.set_aspect("equal", adjustable="box")
    world_ax.grid(True)
    world_ax.legend(fontsize=8)
    world_ax.set_title("Raw world XY")
    world_ax.set_xlabel("x")
    world_ax.set_ylabel("y")

    diagnostics = sample_diagnostics(dataset, sample, global_idx)
    fig.suptitle(f"{sequence_name} | " + " | ".join(diagnostics), fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset.mode}_{subdataset_name}_{global_idx:06d}_seq{sequence_idx:04d}_pose{pose_start:06d}.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path, diagnostics


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    if cfg.data.type != "offroad":
        raise ValueError("This script expects cfg.data.type == 'offroad'.")

    dataset = OffroadDataset(cfg, mode=args.split)
    if args.summary:
        print_summary(dataset, args.max_summary)

    indices, source = choose_global_indices(dataset, args)
    print(f"Rendering {len(indices)} samples from {source}")

    output_paths = []
    for render_idx, global_idx in enumerate(indices):
        if global_idx < 0 or global_idx >= len(dataset):
            raise IndexError(f"Sample index {global_idx} out of range [0, {len(dataset) - 1}]")
        random.seed(args.seed + render_idx)
        np.random.seed(args.seed + render_idx)
        torch.manual_seed(args.seed + render_idx)
        sample = dataset[global_idx]
        output_path, diagnostics = render_sample(dataset, sample, global_idx, args.output_dir)
        output_paths.append(output_path)
        print(f"Saved {output_path}")
        print("  " + " | ".join(diagnostics))

    if output_paths:
        print(f"Done. Wrote {len(output_paths)} file(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
