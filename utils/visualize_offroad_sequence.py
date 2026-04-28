import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.offroad_dataset import OffroadDataset
from pl_modules.citywalker_feat_module import CityWalkerFeatModule
from pl_modules.citywalker_module import CityWalkerModule

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
        description="Render model predictions for one offroad sequence as a video."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/offroad_finetune.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to draw the sequence from.",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        required=True,
        help="Sequence basename or unique path fragment.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output mp4 path. Defaults to results/<run_name>/<sequence>.mp4",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="Output video fps.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Render every Nth sample from the chosen sequence.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=-1,
        help="Maximum number of rendered timesteps. -1 means no limit.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device. Defaults to cuda if available else cpu.",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as file:
        cfg_dict = yaml.safe_load(file)
    return DictNamespace(**cfg_dict)


def load_model(cfg, checkpoint_path, device):
    if cfg.model.type == "citywalker":
        model = CityWalkerModule.load_from_checkpoint(checkpoint_path, cfg=cfg)
    elif cfg.model.type == "citywalker_feat":
        model = CityWalkerFeatModule.load_from_checkpoint(checkpoint_path, cfg=cfg)
    else:
        raise ValueError(f"Unsupported model type: {cfg.model.type}")
    model.eval()
    model.to(device)
    return model


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
        raise ValueError(
            f"Sequence query '{query}' is ambiguous. Matches:\n{matched_paths}"
        )
    return matches[0]


def locate_sequence_in_cfg(cfg, query):
    root_dir = cfg.data.root_dir
    grouped = {}
    for dirpath, dirnames, _ in os.walk(root_dir):
        if {"image", "poses"}.issubset(set(dirnames)):
            rel_path = os.path.relpath(dirpath, root_dir)
            dataset_name = rel_path.split(os.sep)[0]
            grouped.setdefault(dataset_name, []).append(dirpath)
    for dataset_name in grouped:
        grouped[dataset_name].sort()

    if hasattr(cfg.data, "subdatasets"):
        subdatasets = vars(cfg.data.subdatasets)
        for dataset_name, sequence_dirs in grouped.items():
            for idx, sequence_dir in enumerate(sequence_dirs):
                basename = os.path.basename(sequence_dir)
                if query != basename and query not in sequence_dir:
                    continue
                split_cfg = subdatasets.get(dataset_name)
                if split_cfg is None:
                    return {
                        "dataset_name": dataset_name,
                        "sequence_dir": sequence_dir,
                        "index": idx,
                        "split": "unselected",
                    }
                train_count = int(getattr(split_cfg, "train", 0) or 0)
                val_count = int(getattr(split_cfg, "val", 0) or 0)
                test_count = int(getattr(split_cfg, "test", 0) or 0)
                if idx < train_count:
                    split = "train"
                elif idx < train_count + val_count:
                    split = "val"
                elif idx < train_count + val_count + test_count:
                    split = "test"
                else:
                    split = "unselected"
                return {
                    "dataset_name": dataset_name,
                    "sequence_dir": sequence_dir,
                    "index": idx,
                    "split": split,
                }
    else:
        flat_sequence_dirs = []
        for dataset_name in sorted(grouped):
            flat_sequence_dirs.extend(grouped[dataset_name])
        for idx, sequence_dir in enumerate(flat_sequence_dirs):
            basename = os.path.basename(sequence_dir)
            if query != basename and query not in sequence_dir:
                continue
            train_count = int(getattr(cfg.data, "num_train", 0) or 0)
            val_count = int(getattr(cfg.data, "num_val", 0) or 0)
            test_count = int(getattr(cfg.data, "num_test", 0) or 0)
            if idx < train_count:
                split = "train"
            elif idx < train_count + val_count:
                split = "val"
            elif idx < train_count + val_count + test_count:
                split = "test"
            else:
                split = "unselected"
            return {
                "dataset_name": os.path.relpath(sequence_dir, root_dir).split(os.sep)[0],
                "sequence_dir": sequence_dir,
                "index": idx,
                "split": split,
            }
    return None


def render_frame(sample, prediction, sequence_name, sample_idx):
    obs = sample["video_frames"]
    frame = obs[-1].permute(1, 2, 0).cpu().numpy()
    frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)

    history_positions = prediction["history_positions"]
    gt_waypoints = prediction["gt_waypoints"]
    target_transformed = prediction["target_position"]
    pred_waypoints = prediction["pred_waypoints"]

    arrive_gt = bool(sample["arrived"].item())
    arrive_prob = prediction["arrive_prob"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(wspace=0.3)

    ax1.imshow(frame)
    ax1.axis("off")
    ax1.set_title(
        f"{sequence_name}\nstep={sample_idx} arrived_gt={arrive_gt} pred={arrive_prob:.2f}",
        fontsize=14,
    )

    ax2.axis("equal")
    ax2.plot(
        history_positions[:, 0],
        history_positions[:, 1],
        "o-",
        color="#5771DB",
        label="history",
    )
    ax2.plot(
        gt_waypoints[:, 0],
        gt_waypoints[:, 1],
        "X-",
        color="#92DB58",
        label="gt",
    )
    ax2.plot(
        pred_waypoints[:, 0],
        pred_waypoints[:, 1],
        "s-",
        color="#DB6057",
        label="pred",
    )
    ax2.plot(
        target_transformed[0],
        target_transformed[1],
        marker="*",
        markersize=15,
        color="#A157DB",
        label="target",
    )
    ax2.set_title("Trajectory in current vehicle frame")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.grid(True)
    ax2.legend(loc="best")

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    image = image[:, :, :3].copy()
    plt.close(fig)
    return image


def save_video(frames, output_path, fps):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    with tempfile.TemporaryDirectory(prefix="citywalker_vis_") as tmp_dir:
        for idx, frame in enumerate(frames):
            frame_path = Path(tmp_dir) / f"frame_{idx:06d}.png"
            plt.imsave(frame_path, frame)

        if ffmpeg is None:
            fallback_dir = output_path.with_suffix("")
            fallback_dir = fallback_dir.parent / f"{fallback_dir.name}_frames"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            for idx, frame in enumerate(frames):
                plt.imsave(fallback_dir / f"frame_{idx:06d}.png", frame)
            print(f"ffmpeg was not found. Saved PNG frames to {fallback_dir}")
            return fallback_dir

        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(Path(tmp_dir) / "frame_%06d.png"),
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)
        return output_path


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if cfg.data.type != "offroad":
        raise ValueError("This script currently only supports cfg.data.type == 'offroad'.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = OffroadDataset(cfg, mode=args.split)
    try:
        sequence_idx, sequence_dir = find_sequence_index(dataset, args.sequence)
    except FileNotFoundError as exc:
        located = locate_sequence_in_cfg(cfg, args.sequence)
        if located is not None:
            raise FileNotFoundError(
                f"{exc}\n"
                f"Sequence '{args.sequence}' exists at '{located['sequence_dir']}' "
                f"(dataset={located['dataset_name']}, index={located['index']}), "
                f"but under the current config it belongs to split '{located['split']}'."
            ) from None
        raise
    sequence_name = os.path.basename(sequence_dir)
    sequence_start, sequence_end = dataset.sequence_ranges[sequence_idx]

    model = load_model(cfg, args.checkpoint, device)

    rendered_frames = []
    with torch.no_grad():
        for local_idx, global_idx in enumerate(range(sequence_start, sequence_end, args.stride)):
            if args.max_frames > 0 and len(rendered_frames) >= args.max_frames:
                break

            sample = dataset[global_idx]
            obs = sample["video_frames"].unsqueeze(0).to(device)
            cord = sample["input_positions"].unsqueeze(0).to(device)

            if cfg.model.type == "citywalker_feat":
                wp_pred, arrive_pred, _, _ = model(obs, cord, None)
            else:
                wp_pred, arrive_pred = model(obs, cord)

            pred_waypoints_local = wp_pred[0].detach().cpu().numpy()
            pred_waypoints_local = pred_waypoints_local * sample["step_scale"].item()
            arrive_prob = torch.sigmoid(arrive_pred.flatten())[0].item()

            prediction = {
                "history_positions": sample["vis_history_positions"].cpu().numpy(),
                "gt_waypoints": sample["vis_gt_waypoints"].cpu().numpy(),
                "target_position": sample["vis_target_transformed"].cpu().numpy(),
                "pred_waypoints": pred_waypoints_local,
                "arrive_prob": arrive_prob,
            }
            rendered_frames.append(render_frame(sample, prediction, sequence_name, local_idx))

    if not rendered_frames:
        raise RuntimeError(f"No frames rendered for sequence '{sequence_name}'.")

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(
            cfg.project.result_dir,
            cfg.project.run_name,
            "sequence_videos",
            f"{sequence_name}.mp4",
        )

    saved_path = save_video(rendered_frames, output_path, args.fps)
    print(f"Saved visualization to {saved_path}")


if __name__ == "__main__":
    main()
