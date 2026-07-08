# coding=utf-8
# Copyright 2025 The HustVL Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on Qwen3-VL (https://github.com/QwenLM/Qwen3-VL). It has been modified to create DiffusionVL.
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

"""Qwen3-VL vision tower for DiffusionVL.

This mirrors `LlavaQwenVisionTower` but targets Qwen3-VL, which differs from
Qwen2.5-VL in three ways that matter here:

  1. A learned positional embedding (`pos_embed`) added after `patch_embed`,
     interpolated to the actual grid via `fast_pos_embed_interpolate`.
  2. No window shuffle / hybrid window-full attention — Qwen3-VL uses plain
     `cu_seqlens` full attention over every image's patches.
  3. DeepStack: intermediate block outputs at `deepstack_visual_indexes`
     (e.g. [5, 11, 17]) are merged by `deepstack_merger_list` and later injected
     into the early decoder layers of the language model.

The forward stops *before* the main `merger` (owned by `LlavaQwen3Projector`)
but *after* the DeepStack mergers, returning
`(hidden_states, deepstack_features)` where `hidden_states` is at patch
granularity and `deepstack_features` is a list of already-merged tensors.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoConfig, AutoImageProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionModel,
    Qwen3VLVisionRotaryEmbedding,
)

from llava.utils import rank0_print

DEFAULT_MIN_PIXELS = 384 * 384  # 147456
DEFAULT_MAX_PIXELS = 512 * 512  # 262144


class LlavaQwen3VisionTower(nn.Module):
    """Vision tower wrapping `Qwen3VLVisionModel`, stopping before the main merger."""

    def __init__(self, vision_tower_path, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower_path
        self.select_layer = getattr(args, "mm_vision_select_layer", -1)
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        # Image processor
        processor_kwargs = {"use_fast": False}
        min_pixels = getattr(args, "min_pixels", None)
        if min_pixels is None:
            min_pixels = DEFAULT_MIN_PIXELS
            rank0_print(f"Using default min_pixels: {min_pixels}")
        else:
            rank0_print(f"Using min_pixels: {min_pixels}")
        processor_kwargs["min_pixels"] = min_pixels

        max_pixels = getattr(args, "max_pixels", None)
        if max_pixels is None:
            max_pixels = DEFAULT_MAX_PIXELS
            rank0_print(f"Using default max_pixels: {max_pixels}")
        else:
            rank0_print(f"Using max_pixels: {max_pixels}")
        processor_kwargs["max_pixels"] = max_pixels

        self.image_processor = AutoImageProcessor.from_pretrained(vision_tower_path, **processor_kwargs)
        self.vision_tower = None

        if hasattr(args, "vision_config"):
            self._config = args.vision_config
        else:
            self._config = AutoConfig.from_pretrained(vision_tower_path).vision_config

        head_dim = self._config.hidden_size // self._config.num_heads
        # Qwen3-VL uses the same vision rotary embedding class as Qwen2.5-VL
        # (operates on half the head dim).
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

    def load_model(self, model_path=None, device_map=None):
        if self.is_loaded:
            return

        path_to_load = model_path if model_path is not None else self.vision_tower_name
        rank0_print(f"Loading Qwen3-VL vision tower weights from path: {path_to_load}")
        # Build the full Qwen3VLVisionModel (includes patch_embed, pos_embed,
        # blocks, merger, and deepstack_merger_list). We delete the top-level
        # `merger` afterwards because the main merger is owned by the projector,
        # but we keep `deepstack_merger_list` since it is integral to the tower.
        self.vision_tower = Qwen3VLVisionModel(self._config)

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
                    f"Could not find model weights file (safetensors or bin) in {path_to_load}"
                )

            # The converted vision_tower.safetensors already has keys without any
            # `model.visual.` / `visual.` prefix (stripped by the converter), so
            # we can load directly. Keys include blocks.*, patch_embed.*, pos_embed.*.
            # The top-level `merger.*` keys are NOT in vision_tower.safetensors
            # (they live in projector.safetensors), and deepstack_merger_list.*
            # are loaded separately below.
            incompatible_keys = self.vision_tower.load_state_dict(full_state_dict, strict=False)
            # merger.* are expected to be missing (owned by the projector).
            real_missing = [k for k in incompatible_keys.missing_keys if not k.startswith("merger.")]
            real_unexpected = [
                k for k in incompatible_keys.unexpected_keys
                if not k.startswith("model.") and not k.startswith("lm_head.")
            ]
            if real_missing or real_unexpected:
                rank0_print(
                    f"Vision tower weights loaded with incompatibilities: "
                    f"missing={real_missing}, unexpected={real_unexpected}"
                )
            else:
                rank0_print("Vision tower backbone weights loaded successfully.")

            # DeepStack mergers are loaded by `LlavaMetaModel.initialize_vision_modules`
            # via `config.deepstack_state_dict` (same pattern as the vision tower
            # and projector), so they are NOT loaded here to avoid double-loading.

        except Exception as e:
            rank0_print(
                f"ERROR: Failed to load Qwen3-VL vision tower weights from {path_to_load}: {e}. "
                f"The vision tower will have random weights."
            )

        self.vision_tower.rotary_pos_emb = self.rotary_pos_emb

        # Remove the top-level merger: it is owned by LlavaQwen3Projector.
        # Keep deepstack_merger_list — it is part of the tower forward.
        if hasattr(self.vision_tower, "merger"):
            delattr(self.vision_tower, "merger")
            rank0_print("Removed top-level merger module from vision tower (owned by projector).")

        self.is_loaded = True

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
        """Replicate `Qwen3VLVisionModel.forward` but stop before the main merger.

        Returns:
            (hidden_states, deepstack_features):
              - hidden_states: (seq_len, hidden_size) patch-granularity features
                in the order the merger expects (merged_h, merge_size, merged_w,
                merge_size). The caller's projector runs the main merger on this.
              - deepstack_features: list of (visual_seq_len, out_hidden_size)
                tensors, already merged, one per deepstack visual index.
        """
        hidden_states = self.vision_tower.patch_embed(hidden_states)

        # Learned absolute positional embedding, interpolated to the actual grid.
        pos_embeds = self.vision_tower.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.vision_tower.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # Full attention over each image's patches (no windowing in Qwen3-VL).
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.vision_tower.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            # DeepStack: at configured layers, merge the intermediate features
            # and collect them for injection into the LM decoder later.
            if layer_num in self.vision_tower.deepstack_visual_indexes:
                deepstack_feature = self.vision_tower.deepstack_merger_list[
                    self.vision_tower.deepstack_visual_indexes.index(layer_num)
                ](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        return hidden_states, deepstack_feature_lists

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
