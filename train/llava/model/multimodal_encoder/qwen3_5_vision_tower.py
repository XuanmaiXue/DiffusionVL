# coding=utf-8
# Copyright 2025 The HustVL Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on Qwen3.5 (hybrid Gated DeltaNet + full attention).
# It has been modified to create DiffusionVL.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3.5 vision tower for DiffusionVL.

Qwen3.5 has NO DeepStack (deepstack_visual_indexes=[]), so this is simpler
than the Qwen3-VL tower: just patch_embed + pos_embed + blocks, returning
patch-granularity features (pre-merger) to the projector.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoConfig, AutoImageProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5VisionModel,
    Qwen3_5VisionRotaryEmbedding,
)

from llava.utils import rank0_print

DEFAULT_MIN_PIXELS = 384 * 384
DEFAULT_MAX_PIXELS = 512 * 512


class LlavaQwen3_5VisionTower(nn.Module):
    """Vision tower wrapping Qwen3_5VisionModel, stopping before the main merger.

    Qwen3.5 has no DeepStack, so forward returns a single hidden_states tensor
    (no deepstack_feature_lists).
    """

    def __init__(self, vision_tower_path, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower_path
        self.select_layer = getattr(args, "mm_vision_select_layer", -1)
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        # Image processor
        processor_kwargs = {"use_fast": False}
        min_pixels = getattr(args, "min_pixels", None)
        max_pixels = getattr(args, "max_pixels", None)
        processor_kwargs["min_pixels"] = min_pixels if min_pixels is not None else DEFAULT_MIN_PIXELS
        processor_kwargs["max_pixels"] = max_pixels if max_pixels is not None else DEFAULT_MAX_PIXELS
        if min_pixels is not None:
            rank0_print(f"Using min_pixels: {min_pixels}")
        else:
            rank0_print(f"Using default min_pixels: {processor_kwargs['min_pixels']}")
        if max_pixels is not None:
            rank0_print(f"Using max_pixels: {max_pixels}")
        else:
            rank0_print(f"Using default max_pixels: {processor_kwargs['max_pixels']}")

        self.image_processor = AutoImageProcessor.from_pretrained(vision_tower_path, **processor_kwargs)
        self.vision_tower = None

        if hasattr(args, "vision_config"):
            self._config = args.vision_config
        else:
            self._config = AutoConfig.from_pretrained(vision_tower_path).vision_config

        head_dim = self._config.hidden_size // self._config.num_heads
        self.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(head_dim // 2)

    def load_model(self, model_path=None, device_map=None):
        if self.is_loaded:
            return

        path_to_load = model_path if model_path is not None else self.vision_tower_name
        rank0_print(f"Loading Qwen3.5 vision tower weights from path: {path_to_load}")
        self.vision_tower = Qwen3_5VisionModel(self._config)

        try:
            full_state_dict = {}
            vision_tower_path = os.path.join(path_to_load, "vision_tower.safetensors")
            safetensors_index_path = os.path.join(path_to_load, "model.safetensors.index.json")
            safetensors_path = os.path.join(path_to_load, "model.safetensors")

            if os.path.exists(vision_tower_path):
                rank0_print(f"Found dedicated vision tower file: {vision_tower_path}")
                full_state_dict = load_file(vision_tower_path, device="cpu")
            elif os.path.exists(safetensors_path):
                rank0_print(f"Found single safetensors file: {safetensors_path}")
                full_state_dict = load_file(safetensors_path, device="cpu")
            elif os.path.exists(safetensors_index_path):
                rank0_print(f"Found sharded safetensors index: {safetensors_index_path}")
                import json
                with open(safetensors_index_path, "r") as f:
                    index = json.load(f)
                shard_files = set(index["weight_map"].values())
                for shard_file in shard_files:
                    shard_path = os.path.join(path_to_load, shard_file)
                    rank0_print(f"Loading shard: {shard_path}")
                    full_state_dict.update(load_file(shard_path, device="cpu"))
            else:
                raise FileNotFoundError(
                    f"Could not find model weights file in {path_to_load}")

            incompatible_keys = self.vision_tower.load_state_dict(full_state_dict, strict=False)
            real_missing = [k for k in incompatible_keys.missing_keys if not k.startswith("merger.")]
            real_unexpected = [
                k for k in incompatible_keys.unexpected_keys
                if not k.startswith("model.") and not k.startswith("lm_head.")
            ]
            if real_missing or real_unexpected:
                rank0_print(f"Vision tower loaded with incompatibilities: missing={real_missing}, unexpected={real_unexpected}")
            else:
                rank0_print("Vision tower backbone weights loaded successfully.")
        except Exception as e:
            rank0_print(f"ERROR: Failed to load vision tower weights: {e}")

        self.vision_tower.rotary_pos_emb = self.rotary_pos_emb

        if hasattr(self.vision_tower, "merger"):
            delattr(self.vision_tower, "merger")
            rank0_print("Removed top-level merger module from vision tower (owned by projector).")

        self.is_loaded = True

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
        """Qwen3.5 vision forward (no DeepStack, no window shuffle).

        Returns hidden_states at patch granularity (pre-merger).
        """
        hidden_states = self.vision_tower.patch_embed(hidden_states)

        if hasattr(self.vision_tower, "fast_pos_embed_interpolate"):
            pos_embeds = self.vision_tower.fast_pos_embed_interpolate(grid_thw)
        elif hasattr(self.vision_tower, "get_pos_embed"):
            pos_embeds = self.vision_tower.get_pos_embed(grid_thw)
        else:
            pos_embeds = None
        if pos_embeds is not None:
            pos_embeds = pos_embeds.to(dtype=hidden_states.dtype, device=hidden_states.device)
            hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.vision_tower.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos().to(hidden_states.dtype), emb.sin().to(hidden_states.dtype))

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.vision_tower.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        return hidden_states

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        return self._config

    @property
    def hidden_size(self):
        return self._config.hidden_size
