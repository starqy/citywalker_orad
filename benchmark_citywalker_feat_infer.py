#!/usr/bin/env python3
"""Benchmark CityWalkerFeat inference latency with synthetic inputs."""

from __future__ import annotations

import argparse
import os
import statistics
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
import torchvision.transforms.functional as TF
import yaml


class DictNamespace(argparse.Namespace):
    """Compatibility shim for checkpoints saved by train.py."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, DictNamespace(**value))
            else:
                setattr(self, key, value)


def dict_to_namespace(value):
    if isinstance(value, dict):
        return DictNamespace(**{k: dict_to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [dict_to_namespace(v) for v in value]
    return value


def percentile(values, pct):
    ordered = sorted(values)
    idx = round((pct / 100.0) * (len(ordered) - 1))
    return ordered[max(0, min(len(ordered) - 1, idx))]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark CityWalkerFeat forward-pass latency.")
    parser.add_argument("--config", default="config/offroad_finetune.yaml", help="Path to config YAML.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint to load before benchmarking.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device to benchmark on.")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"], help="Autocast dtype.")
    parser.add_argument("--batch-size", type=int, default=1, help="Synthetic batch size.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=50, help="Timed iterations.")
    parser.add_argument("--height", type=int, default=None, help="Override synthetic input height.")
    parser.add_argument("--width", type=int, default=None, help="Override synthetic input width.")
    parser.add_argument("--context-size", type=int, default=None, help="Override observation context size.")
    parser.add_argument("--future-obs", action="store_true", help="Also pass future_obs, doubling backbone calls.")
    parser.add_argument("--quick", action="store_true", help="Use a small CPU-friendly smoke benchmark.")
    parser.add_argument("--profile-stages", action="store_true", help="Also time backbone and decoder stages separately.")
    parser.add_argument(
        "--enable-xformers",
        action="store_true",
        help="Use xFormers attention. Disabled by default because some Jetson builds lack matching kernels.",
    )
    parser.add_argument(
        "--disable-xformers",
        action="store_true",
        help="Deprecated alias; xFormers is already disabled unless --enable-xformers is set.",
    )
    return parser.parse_args()


def choose_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(requested)


def autocast_context(device, dtype_name):
    if dtype_name == "fp32":
        return nullcontext()
    dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    if state_dict and all(k.startswith("model.") for k in state_dict):
        state_dict = {k[len("model.") :]: v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_stage(device, timings, name, fn):
    synchronize(device)
    start = time.perf_counter()
    result = fn()
    synchronize(device)
    timings[name].append((time.perf_counter() - start) * 1000.0)
    return result


def profile_forward_once(model, obs, cord, future_obs, device, timings):
    B, N, _, H, W = obs.shape

    def preprocess():
        obs_flat = obs.view(B * N, 3, H, W)
        future_flat = None
        if future_obs is not None:
            future_flat = future_obs.view(B * N, 3, H, W)

        if model.do_rgb_normalize:
            obs_flat = (obs_flat - model.mean) / model.std
            if future_flat is not None:
                future_flat = (future_flat - model.mean) / model.std

        if model.do_resize:
            obs_flat = TF.center_crop(obs_flat, model.crop)
            obs_flat = TF.resize(obs_flat, model.resize)
            if future_flat is not None:
                future_flat = TF.center_crop(future_flat, model.crop)
                future_flat = TF.resize(future_flat, model.resize)

        return obs_flat, future_flat

    obs_flat, future_flat = timed_stage(device, timings, "preprocess", preprocess)

    obs_enc = timed_stage(
        device,
        timings,
        "obs_backbone",
        lambda: model.obs_encoder(obs_flat).view(B, N, -1),
    )

    if future_flat is not None:
        timed_stage(
            device,
            timings,
            "future_backbone",
            lambda: model.obs_encoder(future_flat).view(B, N, -1),
        )

    cord_enc = timed_stage(
        device,
        timings,
        "cord_embed",
        lambda: model.compress_goal_enc(model.cord_embedding(cord).view(B, -1)).view(B, 1, -1),
    )

    def decode():
        tokens = torch.cat([obs_enc, cord_enc], dim=1)
        feature_pred = model.predictor(tokens)
        dec_out = model.predictor_mlp(feature_pred.view(B, -1))
        return feature_pred, dec_out

    feature_pred, dec_out = timed_stage(device, timings, "decoder", decode)

    def heads():
        wp_pred = model.wp_predictor(dec_out).view(B, model.len_traj_pred, 2)
        arrive_pred = model.arrive_predictor(dec_out).view(B, 1)
        if model.output_coordinate_repr == "euclidean":
            wp_pred = torch.cumsum(wp_pred, dim=1)
            return wp_pred, arrive_pred, feature_pred[:, :-1]
        raise NotImplementedError(f"Output coordinate representation {model.output_coordinate_repr} not implemented")

    return timed_stage(device, timings, "heads", heads)


def print_stage_profile(stage_times, total_mean_ms, images_per_forward):
    print("\n=== Stage Profile ===")
    for name, values in stage_times.items():
        mean_ms = statistics.mean(values)
        pct = (mean_ms / total_mean_ms) * 100.0
        print(
            f"{name}: mean_ms={mean_ms:.3f}, median_ms={statistics.median(values):.3f}, "
            f"p95_ms={percentile(values, 95):.3f}, pct_of_forward={pct:.1f}%"
        )

    if "obs_backbone" in stage_times:
        backbone_mean_ms = statistics.mean(stage_times["obs_backbone"])
        print(f"obs_backbone_ms_per_image: {backbone_mean_ms / images_per_forward:.3f}")
        print(f"obs_backbone_images_per_sec: {images_per_forward * 1000.0 / backbone_mean_ms:.2f}")


def main():
    args = parse_args()
    device = choose_device(args.device)

    if not args.enable_xformers:
        os.environ["XFORMERS_DISABLED"] = "1"

    from model.citywalker_feat import CityWalkerFeat

    config_path = Path(args.config)
    with config_path.open("r") as f:
        cfg = dict_to_namespace(yaml.safe_load(f))

    if args.quick:
        cfg.model.obs_encoder.type = "dinov2_vits14"
        cfg.model.obs_encoder.context_size = 1
        cfg.model.obs_encoder.crop = [98, 98]
        cfg.model.obs_encoder.resize = [98, 98]
        cfg.model.decoder.num_layers = 1
        cfg.model.decoder.ff_dim_factor = 1

    if args.context_size is not None:
        cfg.model.obs_encoder.context_size = args.context_size

    input_h = args.height or cfg.model.obs_encoder.crop[0]
    input_w = args.width or cfg.model.obs_encoder.crop[1]
    context_size = cfg.model.obs_encoder.context_size

    model = CityWalkerFeat(cfg).to(device).eval()
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, device)

    obs = torch.rand(args.batch_size, context_size, 3, input_h, input_w, device=device)
    cord = torch.rand(args.batch_size, context_size + 1, 2, device=device)
    future_obs = None
    if args.future_obs:
        future_obs = torch.rand(args.batch_size, context_size, 3, input_h, input_w, device=device)

    amp = autocast_context(device, args.dtype)

    with torch.inference_mode():
        for _ in range(args.warmup):
            with amp:
                model(obs, cord, future_obs)
        synchronize(device)

        times_ms = []
        for _ in range(args.iters):
            start = time.perf_counter()
            with amp:
                model(obs, cord, future_obs)
            synchronize(device)
            times_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.mean(times_ms)
    images_per_forward = args.batch_size * context_size * (2 if args.future_obs else 1)

    print("\n=== CityWalkerFeat Inference Benchmark ===")
    print(f"config: {config_path}")
    print(f"device: {device}")
    print(f"dtype: {args.dtype}")
    print(f"xformers_disabled: {os.environ.get('XFORMERS_DISABLED') == '1'}")
    print(f"encoder: {cfg.model.obs_encoder.type}")
    print(f"decoder_layers: {cfg.model.decoder.num_layers}")
    print(f"batch_size: {args.batch_size}")
    print(f"context_size: {context_size}")
    print(f"input_shape: {(args.batch_size, context_size, 3, input_h, input_w)}")
    print(f"future_obs: {args.future_obs}")
    print(f"warmup_iters: {args.warmup}")
    print(f"timed_iters: {args.iters}")
    print(f"mean_ms: {mean_ms:.3f}")
    print(f"median_ms: {statistics.median(times_ms):.3f}")
    print(f"p90_ms: {percentile(times_ms, 90):.3f}")
    print(f"p95_ms: {percentile(times_ms, 95):.3f}")
    print(f"min_ms: {min(times_ms):.3f}")
    print(f"max_ms: {max(times_ms):.3f}")
    print(f"forwards_per_sec: {1000.0 / mean_ms:.2f}")
    print(f"backbone_images_per_sec: {images_per_forward * 1000.0 / mean_ms:.2f}")

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        print(f"cuda_peak_memory_mb: {peak_mb:.1f}")

    if args.profile_stages:
        stage_names = ["preprocess", "obs_backbone", "future_backbone", "cord_embed", "decoder", "heads"]
        stage_times = {name: [] for name in stage_names}
        if future_obs is None:
            stage_times.pop("future_backbone")

        with torch.inference_mode():
            for _ in range(args.warmup):
                with amp:
                    profile_forward_once(model, obs, cord, future_obs, device, {name: [] for name in stage_times})

            for _ in range(args.iters):
                with amp:
                    profile_forward_once(model, obs, cord, future_obs, device, stage_times)

        print_stage_profile(stage_times, mean_ms, images_per_forward)


if __name__ == "__main__":
    main()
