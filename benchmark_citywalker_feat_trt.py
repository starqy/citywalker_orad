#!/usr/bin/env python3
"""Benchmark CityWalkerFeat with a TensorRT DINOv2 backbone and PyTorch heads."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import yaml

from model.model_utils import FeatPredictor, PolarEmbedding


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


def import_tensorrt():
    try:
        import tensorrt as trt

        return trt
    except ImportError:
        system_dist = f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages"
        if system_dist not in sys.path:
            sys.path.append(system_dist)
        try:
            import tensorrt as trt

            return trt
        except ImportError as exc:
            raise ImportError(
                "Could not import TensorRT Python bindings. On JetPack, try:\n"
                f"  export PYTHONPATH={system_dist}:$PYTHONPATH\n"
                "or run with the system Python that has python3-libnvinfer installed."
            ) from exc


def percentile(values, pct):
    ordered = sorted(values)
    idx = round((pct / 100.0) * (len(ordered) - 1))
    return ordered[max(0, min(len(ordered) - 1, idx))]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark CityWalkerFeat with TensorRT backbone.")
    parser.add_argument("--config", default="config/offroad_finetune.yaml", help="Path to config YAML.")
    parser.add_argument("--checkpoint", required=True, help="CityWalkerFeat checkpoint.")
    parser.add_argument("--engine", required=True, help="TensorRT engine for DINOv2 backbone.")
    parser.add_argument("--batch-size", type=int, default=1, help="Synthetic batch size.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=50, help="Timed iterations.")
    parser.add_argument("--height", type=int, default=None, help="Synthetic raw input height before model resize.")
    parser.add_argument("--width", type=int, default=None, help="Synthetic raw input width before model resize.")
    parser.add_argument("--context-size", type=int, default=None, help="Override observation context size.")
    parser.add_argument("--dtype", default="fp16", choices=["fp32", "fp16"], help="PyTorch decoder autocast dtype.")
    parser.add_argument("--profile-stages", action="store_true", help="Time preprocess/TRT backbone/decoder separately.")
    return parser.parse_args()


def autocast_context(dtype_name):
    if dtype_name == "fp32":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def synchronize():
    torch.cuda.synchronize()


class TensorRTBackbone:
    def __init__(self, engine_path):
        trt = import_tensorrt()
        self.trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.input_idx = None
        self.output_idx = None
        for idx in range(self.engine.num_bindings):
            if self.engine.binding_is_input(idx):
                self.input_idx = idx
            else:
                self.output_idx = idx
        if self.input_idx is None or self.output_idx is None:
            raise RuntimeError("Expected one TensorRT input binding and one output binding.")

        self.input_name = self.engine.get_binding_name(self.input_idx)
        self.output_name = self.engine.get_binding_name(self.output_idx)

    def torch_dtype(self, binding_idx):
        dtype = self.engine.get_binding_dtype(binding_idx)
        if dtype == self.trt.float16:
            return torch.float16
        if dtype == self.trt.float32:
            return torch.float32
        if dtype == self.trt.int32:
            return torch.int32
        if dtype == self.trt.int8:
            return torch.int8
        raise TypeError(f"Unsupported TensorRT binding dtype: {dtype}")

    def binding_shape(self, idx):
        shape = tuple(self.context.get_binding_shape(idx))
        if any(dim < 0 for dim in shape):
            shape = tuple(self.engine.get_binding_shape(idx))
        return shape

    def __call__(self, images):
        if not images.is_cuda:
            raise ValueError("TensorRTBackbone expects a CUDA tensor.")
        if not images.is_contiguous():
            images = images.contiguous()

        input_shape = tuple(images.shape)
        engine_input_shape = tuple(self.engine.get_binding_shape(self.input_idx))
        if any(dim < 0 for dim in engine_input_shape):
            self.context.set_binding_shape(self.input_idx, input_shape)
        else:
            expected = engine_input_shape
            if input_shape != expected:
                raise ValueError(f"Engine input shape is static {expected}, but got {input_shape}.")

        output_shape = tuple(self.context.get_binding_shape(self.output_idx))
        output = torch.empty(output_shape, device=images.device, dtype=self.torch_dtype(self.output_idx))
        bindings = [0] * self.engine.num_bindings
        bindings[self.input_idx] = int(images.data_ptr())
        bindings[self.output_idx] = int(output.data_ptr())
        ok = self.context.execute_async_v2(bindings=bindings, stream_handle=torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v2 failed.")
        return output


class CityWalkerFeatTorchHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.context_size = cfg.model.obs_encoder.context_size
        self.num_obs_features = cfg.model.encoder_feat_dim
        self.len_traj_pred = cfg.model.decoder.len_traj_pred
        self.output_coordinate_repr = cfg.model.output_coordinate_repr

        self.cord_embedding = PolarEmbedding(cfg)
        self.dim_cord_embedding = self.cord_embedding.out_dim * (self.context_size + 1)
        self.compress_goal_enc = nn.Linear(self.dim_cord_embedding, self.num_obs_features)
        self.predictor = FeatPredictor(
            embed_dim=self.num_obs_features,
            seq_len=self.context_size + 1,
            nhead=cfg.model.decoder.num_heads,
            num_layers=cfg.model.decoder.num_layers,
            ff_dim_factor=cfg.model.decoder.ff_dim_factor,
        )
        self.predictor_mlp = nn.Sequential(
            nn.Linear((self.context_size + 1) * self.num_obs_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        self.wp_predictor = nn.Linear(32, self.len_traj_pred * 2)
        self.arrive_predictor = nn.Linear(32, 1)

    def forward(self, obs_enc, cord):
        batch_size = obs_enc.shape[0]
        cord_enc = self.cord_embedding(cord).view(batch_size, -1)
        cord_enc = self.compress_goal_enc(cord_enc).view(batch_size, 1, -1)
        tokens = torch.cat([obs_enc, cord_enc], dim=1)
        feature_pred = self.predictor(tokens)
        dec_out = self.predictor_mlp(feature_pred.view(batch_size, -1))
        wp_pred = self.wp_predictor(dec_out).view(batch_size, self.len_traj_pred, 2)
        arrive_pred = self.arrive_predictor(dec_out).view(batch_size, 1)
        if self.output_coordinate_repr == "euclidean":
            wp_pred = torch.cumsum(wp_pred, dim=1)
            return wp_pred, arrive_pred, feature_pred[:, :-1]
        raise NotImplementedError(f"Output coordinate representation {self.output_coordinate_repr} not implemented")


class CityWalkerFeatTRT(nn.Module):
    def __init__(self, cfg, engine_path):
        super().__init__()
        self.context_size = cfg.model.obs_encoder.context_size
        self.do_rgb_normalize = cfg.model.do_rgb_normalize
        self.do_resize = cfg.model.do_resize
        self.crop = cfg.model.obs_encoder.crop
        self.resize = cfg.model.obs_encoder.resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.backbone = TensorRTBackbone(engine_path)
        self.head = CityWalkerFeatTorchHead(cfg)

    def preprocess(self, obs):
        batch_size, context_size, _, height, width = obs.shape
        images = obs.view(batch_size * context_size, 3, height, width)
        if self.do_rgb_normalize:
            images = (images - self.mean) / self.std
        if self.do_resize:
            images = TF.center_crop(images, self.crop)
            images = TF.resize(images, self.resize)
        return images.contiguous()

    def forward(self, obs, cord):
        batch_size, context_size = obs.shape[:2]
        images = self.preprocess(obs)
        obs_enc = self.backbone(images).view(batch_size, context_size, -1)
        return self.head(obs_enc, cord)

    def forward_profiled(self, obs, cord):
        timings = {}

        def timed(name, fn):
            synchronize()
            start = time.perf_counter()
            result = fn()
            synchronize()
            timings[name] = (time.perf_counter() - start) * 1000.0
            return result

        batch_size, context_size = obs.shape[:2]
        images = timed("preprocess", lambda: self.preprocess(obs))
        obs_enc = timed("trt_backbone", lambda: self.backbone(images).view(batch_size, context_size, -1))
        output = timed("torch_decoder_heads", lambda: self.head(obs_enc, cord))
        return output, timings


def load_head_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if state_dict and all(k.startswith("model.") for k in state_dict):
        state_dict = {k[len("model.") :]: v for k, v in state_dict.items()}

    head_state = {}
    for key, value in state_dict.items():
        if key.startswith("obs_encoder.") or key in {"mean", "std"}:
            continue
        head_state[f"head.{key}"] = value

    missing, unexpected = model.load_state_dict(head_state, strict=False)
    missing = [key for key in missing if not key.startswith("mean") and not key.startswith("std")]
    print(f"Loaded decoder/head weights from: {checkpoint_path}")
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")


def print_stats(title, times_ms):
    mean_ms = statistics.mean(times_ms)
    print(f"\n=== {title} ===")
    print(f"timed_iters: {len(times_ms)}")
    print(f"mean_ms: {mean_ms:.3f}")
    print(f"median_ms: {statistics.median(times_ms):.3f}")
    print(f"p90_ms: {percentile(times_ms, 90):.3f}")
    print(f"p95_ms: {percentile(times_ms, 95):.3f}")
    print(f"min_ms: {min(times_ms):.3f}")
    print(f"max_ms: {max(times_ms):.3f}")
    print(f"forwards_per_sec: {1000.0 / mean_ms:.2f}")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TensorRT benchmark.")

    with open(args.config, "r") as f:
        cfg = dict_to_namespace(yaml.safe_load(f))
    if args.context_size is not None:
        cfg.model.obs_encoder.context_size = args.context_size

    default_h = cfg.data.frame_size[0] if hasattr(cfg.data, "frame_size") else cfg.model.obs_encoder.crop[0]
    default_w = cfg.data.frame_size[1] if hasattr(cfg.data, "frame_size") else cfg.model.obs_encoder.crop[1]
    raw_h = args.height or default_h
    raw_w = args.width or default_w
    context_size = cfg.model.obs_encoder.context_size

    model = CityWalkerFeatTRT(cfg, args.engine).cuda().eval()
    load_head_checkpoint(model, args.checkpoint)

    obs = torch.rand(args.batch_size, context_size, 3, raw_h, raw_w, device="cuda")
    cord = torch.rand(args.batch_size, context_size + 1, 2, device="cuda")
    amp = autocast_context(args.dtype)

    with torch.inference_mode():
        for _ in range(args.warmup):
            with amp:
                model(obs, cord)
        synchronize()

        times_ms = []
        for _ in range(args.iters):
            start = time.perf_counter()
            with amp:
                model(obs, cord)
            synchronize()
            times_ms.append((time.perf_counter() - start) * 1000.0)

    print("\n=== CityWalkerFeat TensorRT Backbone Benchmark ===")
    print(f"config: {args.config}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"engine: {args.engine}")
    print(f"dtype: {args.dtype}")
    print(f"batch_size: {args.batch_size}")
    print(f"context_size: {context_size}")
    print(f"raw_input_shape: {(args.batch_size, context_size, 3, raw_h, raw_w)}")
    print(f"trt_input_shape: {(args.batch_size * context_size, 3, cfg.model.obs_encoder.resize[0], cfg.model.obs_encoder.resize[1])}")
    print_stats("End-to-End", times_ms)

    images_per_forward = args.batch_size * context_size
    mean_ms = statistics.mean(times_ms)
    print(f"images_per_forward: {images_per_forward}")
    print(f"end_to_end_ms_per_image: {mean_ms / images_per_forward:.3f}")
    print(f"cuda_peak_memory_mb: {torch.cuda.max_memory_allocated() / (1024.0 * 1024.0):.1f}")

    if args.profile_stages:
        stage_times = {"preprocess": [], "trt_backbone": [], "torch_decoder_heads": []}
        with torch.inference_mode():
            for _ in range(args.warmup):
                with amp:
                    model.forward_profiled(obs, cord)
            for _ in range(args.iters):
                with amp:
                    _, timings = model.forward_profiled(obs, cord)
                for key, value in timings.items():
                    stage_times[key].append(value)

        print("\n=== Stage Profile ===")
        for key, values in stage_times.items():
            stage_mean = statistics.mean(values)
            print(
                f"{key}: mean_ms={stage_mean:.3f}, median_ms={statistics.median(values):.3f}, "
                f"p95_ms={percentile(values, 95):.3f}, pct_of_forward={(stage_mean / mean_ms) * 100.0:.1f}%"
            )
        backbone_mean = statistics.mean(stage_times["trt_backbone"])
        print(f"trt_backbone_ms_per_image: {backbone_mean / images_per_forward:.3f}")
        print(f"trt_backbone_images_per_sec: {images_per_forward * 1000.0 / backbone_mean:.2f}")


if __name__ == "__main__":
    main()
