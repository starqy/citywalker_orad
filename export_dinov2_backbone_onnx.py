#!/usr/bin/env python3
"""Export CityWalkerFeat's DINOv2 observation encoder to ONNX.

The exported model is the backbone only. It expects a preprocessed tensor shaped
[B, 3, H, W], matching the tensor passed into CityWalkerFeat.obs_encoder after
normalization/crop/resize.
"""

from __future__ import annotations

import argparse
import os
import types
from pathlib import Path

import torch
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


def parse_args():
    parser = argparse.ArgumentParser(description="Export CityWalkerFeat DINOv2 backbone to ONNX.")
    parser.add_argument("--config", default="config/offroad_finetune.yaml", help="Path to config YAML.")
    parser.add_argument("--checkpoint", default=None, help="Optional CityWalkerFeat checkpoint.")
    parser.add_argument("--output", default="exports/dinov2_backbone.onnx", help="Output ONNX path.")
    parser.add_argument("--batch-images", type=int, default=5, help="Number of images per backbone call.")
    parser.add_argument("--height", type=int, default=None, help="Backbone input height.")
    parser.add_argument("--width", type=int, default=None, help="Backbone input width.")
    parser.add_argument(
        "--opset",
        type=int,
        default=16,
        help="ONNX opset version. TensorRT 8.5 does not parse opset-17 LayerNormalization without a plugin.",
    )
    parser.add_argument("--dynamic-batch", action="store_true", help="Export dynamic image batch dimension.")
    parser.add_argument(
        "--keep-sdpa",
        action="store_true",
        help="Keep PyTorch scaled_dot_product_attention. Default rewrites attention to MatMul/Softmax.",
    )
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path):
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


def patch_attention_for_onnx(module):
    """Replace DINOv2 attention forward with ONNX-friendly MatMul/Softmax ops."""

    def forward(self, x, attn_bias=None):
        if attn_bias is not None:
            raise RuntimeError("attn_bias/nested tensors are not supported by this export path.")
        bsz, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(bsz, num_tokens, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(bsz, num_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    patched = 0
    for child in module.modules():
        if child.__class__.__name__ in {"Attention", "MemEffAttention"} and hasattr(child, "qkv"):
            child.forward = types.MethodType(forward, child)
            patched += 1
    print(f"Patched attention modules for ONNX export: {patched}")


class BackboneOnly(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, images):
        return self.backbone(images)


def main():
    args = parse_args()

    # Must be set before importing/loading local DINOv2 modules.
    os.environ["XFORMERS_DISABLED"] = "1"

    from model.citywalker_feat import CityWalkerFeat

    with open(args.config, "r") as f:
        cfg = dict_to_namespace(yaml.safe_load(f))

    input_h = args.height or cfg.model.obs_encoder.resize[0]
    input_w = args.width or cfg.model.obs_encoder.resize[1]

    model = CityWalkerFeat(cfg).eval()
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint)

    backbone = BackboneOnly(model.obs_encoder).eval()
    if not args.keep_sdpa:
        patch_attention_for_onnx(backbone)

    dummy = torch.randn(args.batch_images, 3, input_h, input_w)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {"images": {0: "batch_images"}, "features": {0: "batch_images"}}

    with torch.inference_mode():
        features = backbone(dummy)
        print(f"Backbone input shape: {tuple(dummy.shape)}")
        print(f"Backbone output shape: {tuple(features.shape)}")
        torch.onnx.export(
            backbone,
            dummy,
            str(output_path),
            input_names=["images"],
            output_names=["features"],
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
        )

    print(f"Exported ONNX: {output_path}")


if __name__ == "__main__":
    main()
