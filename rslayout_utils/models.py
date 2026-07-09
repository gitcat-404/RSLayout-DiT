import json
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import FluxControlNetModel, FluxControlNetPipeline, FluxPipeline
from safetensors.torch import load_file, save_file

from .layout_condition import channel_count


def zero_module(module: nn.Module) -> nn.Module:
    for param in module.parameters():
        nn.init.zeros_(param)
    return module


class LayoutConditioningEmbedding(nn.Module):
    """Small image-space encoder that maps layout conditions to FLUX packed latent channels."""

    def __init__(
        self,
        conditioning_channels: int,
        conditioning_embedding_channels: int,
        block_out_channels=(32, 64, 128, 256, 320),
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)
        self.blocks = nn.ModuleList()
        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))
        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        embedding = F.silu(self.conv_in(conditioning))
        for block in self.blocks:
            embedding = F.silu(block(embedding))
        return self.conv_out(embedding)


def _controlnet_metadata(
    method: str,
    input_channels: int,
    num_layers: int,
    num_single_layers: int,
    resolution: int,
) -> Dict[str, Any]:
    return {
        "method": method,
        "input_channels": input_channels,
        "num_layers": num_layers,
        "num_single_layers": num_single_layers,
        "resolution": resolution,
        "format": "rslayout_utils_controlnet_v1",
    }


def create_flux_controlnet(
    transformer,
    method: str = "layout_control",
    input_channels: Optional[int] = None,
    num_layers: int = 4,
    num_single_layers: int = 8,
) -> FluxControlNetModel:
    if input_channels is None:
        input_channels = channel_count(method)
    controlnet = FluxControlNetModel.from_transformer(
        transformer,
        num_layers=num_layers,
        num_single_layers=num_single_layers,
        attention_head_dim=transformer.config.attention_head_dim,
        num_attention_heads=transformer.config.num_attention_heads,
        load_weights_from_transformer=True,
    )
    controlnet.input_hint_block = LayoutConditioningEmbedding(
        conditioning_channels=input_channels,
        conditioning_embedding_channels=controlnet.config.in_channels,
    )
    controlnet.controlnet_x_embedder = nn.Linear(controlnet.config.in_channels, controlnet.inner_dim)
    return controlnet


def save_controlnet(controlnet: FluxControlNetModel, output_dir: str, metadata: Dict[str, Any]):
    os.makedirs(output_dir, exist_ok=True)
    controlnet.save_pretrained(output_dir, safe_serialization=True)
    with open(os.path.join(output_dir, "rslayout_utils_config.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def save_controlnet_trainable(
    controlnet: FluxControlNetModel,
    output_dir: str,
    metadata: Dict[str, Any],
    trainable_prefixes=("input_hint_block", "controlnet_x_embedder", "controlnet_blocks", "controlnet_single_blocks"),
):
    os.makedirs(output_dir, exist_ok=True)
    controlnet.save_config(output_dir)
    state_dict = {
        key: value.detach().cpu()
        for key, value in controlnet.state_dict().items()
        if key.startswith(trainable_prefixes)
    }
    save_file(state_dict, os.path.join(output_dir, "diffusion_pytorch_model.safetensors"))
    metadata = dict(metadata)
    metadata["trainable_only"] = True
    metadata["trainable_prefixes"] = list(trainable_prefixes)
    with open(os.path.join(output_dir, "rslayout_utils_config.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_controlnet_metadata(controlnet_dir: str) -> Dict[str, Any]:
    path = os.path.join(controlnet_dir, "rslayout_utils_config.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing layout metadata: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_weight_file(controlnet_dir: str) -> str:
    candidates = [
        os.path.join(controlnet_dir, "diffusion_pytorch_model.safetensors"),
        os.path.join(controlnet_dir, "pytorch_model.bin"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No controlnet weight file found in {controlnet_dir}")


def load_layout_controlnet(controlnet_dir: str, transformer, map_location: str = "cpu") -> FluxControlNetModel:
    metadata = load_controlnet_metadata(controlnet_dir)
    controlnet = create_flux_controlnet(
        transformer,
        method=metadata["method"],
        input_channels=int(metadata["input_channels"]),
        num_layers=int(metadata["num_layers"]),
        num_single_layers=int(metadata["num_single_layers"]),
    )
    weight_file = _find_weight_file(controlnet_dir)
    if weight_file.endswith(".safetensors"):
        state_dict = load_file(weight_file, device=map_location)
    else:
        state_dict = torch.load(weight_file, map_location=map_location)
    missing, unexpected = controlnet.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys while loading controlnet: {unexpected[:8]}")
    if missing and not metadata.get("trainable_only", False):
        raise RuntimeError(f"Missing keys while loading controlnet: {missing[:8]}")
    return controlnet


def build_control_pipeline(
    pretrained_model_name_or_path: str,
    controlnet_dir: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: Optional[torch.device] = None,
) -> FluxControlNetPipeline:
    base = FluxPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch_dtype)
    controlnet = load_layout_controlnet(controlnet_dir, base.transformer)
    controlnet.to(dtype=torch_dtype)
    pipe = FluxControlNetPipeline(**base.components, controlnet=controlnet)
    if device is not None:
        pipe = pipe.to(device)
    return pipe


def dtype_from_string(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def metadata_for_training(args, input_channels: int) -> Dict[str, Any]:
    return _controlnet_metadata(
        method=args.method,
        input_channels=input_channels,
        num_layers=args.controlnet_num_layers,
        num_single_layers=args.controlnet_num_single_layers,
        resolution=args.resolution,
    )
