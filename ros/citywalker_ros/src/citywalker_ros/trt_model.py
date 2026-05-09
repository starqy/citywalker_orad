import argparse
import __main__
import sys
from contextlib import nullcontext

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import yaml


class DictNamespace(argparse.Namespace):
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
        import tensorrt as trt

        return trt


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

    def __call__(self, images):
        if not images.is_cuda:
            raise ValueError("TensorRTBackbone expects a CUDA tensor.")
        images = images.contiguous()
        input_shape = tuple(images.shape)
        engine_input_shape = tuple(self.engine.get_binding_shape(self.input_idx))
        if any(dim < 0 for dim in engine_input_shape):
            self.context.set_binding_shape(self.input_idx, input_shape)
        elif input_shape != engine_input_shape:
            raise ValueError(f"Engine input shape is static {engine_input_shape}, but got {input_shape}.")

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
    def __init__(self, cfg, repo_root):
        super().__init__()
        if repo_root and repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from model.model_utils import FeatPredictor, PolarEmbedding

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
            return torch.cumsum(wp_pred, dim=1), arrive_pred
        raise NotImplementedError(f"Output coordinate representation {self.output_coordinate_repr} not implemented")


class CityWalkerTRTModel(nn.Module):
    def __init__(self, cfg, engine_path, checkpoint_path, repo_root, dtype="fp16"):
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("CityWalkerTRTModel requires CUDA.")
        self.cfg = cfg
        self.context_size = cfg.model.obs_encoder.context_size
        self.crop = cfg.model.obs_encoder.crop
        self.resize = cfg.model.obs_encoder.resize
        self.do_rgb_normalize = cfg.model.do_rgb_normalize
        self.do_resize = cfg.model.do_resize
        self.autocast = nullcontext if dtype == "fp32" else lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.backbone = TensorRTBackbone(engine_path)
        self.head = CityWalkerFeatTorchHead(cfg, repo_root)
        self.load_head_checkpoint(checkpoint_path)

    def load_head_checkpoint(self, checkpoint_path):
        setattr(__main__, "DictNamespace", DictNamespace)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        if state_dict and all(k.startswith("model.") for k in state_dict):
            state_dict = {k[len("model.") :]: v for k, v in state_dict.items()}
        head_state = {}
        for key, value in state_dict.items():
            if key.startswith("obs_encoder.") or key in {"mean", "std"}:
                continue
            head_state[f"head.{key}"] = value
        self.load_state_dict(head_state, strict=False)

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
        with self.autocast():
            obs_enc = self.backbone(images).view(batch_size, context_size, -1)
            return self.head(obs_enc, cord)


def load_config(config_path):
    with open(config_path, "r") as f:
        return dict_to_namespace(yaml.safe_load(f))
