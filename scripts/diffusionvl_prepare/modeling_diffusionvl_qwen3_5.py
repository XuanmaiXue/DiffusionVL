# coding=utf-8
# Copyright 2025 The HustVL Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on Qwen3.5 (hybrid Gated DeltaNet + full attention),
# LLaDA-V, and Block Diffusion. It has been modified to create DiffusionVL.
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
"""DiffusionVL-Qwen3.5 model — self-contained inference implementation.

Qwen3.5 uses a HYBRID architecture: ~75% Gated DeltaNet (linear attention)
layers + ~25% full softmax attention layers. The key challenge for BD3-LM
block-diffusion is that linear attention layers have NO "read-only" cache mode
— passing a non-None cache always writes. We solve this with a snapshot-restore
strategy: before each denoising step, we snapshot the linear layers' conv_states
and recurrent_states, let the forward write, then restore before the next step.
"""

import os
from typing import List, Optional, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, AutoImageProcessor, PretrainedConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.utils import logging
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5TextModel as Qwen3_5TextModelOriginal,
    Qwen3_5PreTrainedModel,
    Qwen3_5VisionModel,
    Qwen3_5VisionPatchMerger,
)

try:
    from .configuration_diffusionvl_qwen3_5 import DiffusionVLQwen3_5Config
except ImportError:
    from configuration_diffusionvl_qwen3_5 import DiffusionVLQwen3_5Config

logger = logging.get_logger(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"


def rank0_print(*args):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(*args)


# ============================================================
# Linear attention cache snapshot/restore utilities
# ============================================================

def snapshot_linear_cache(cache):
    """Snapshot conv_states + recurrent_states of all linear-attention layers.

    Returns a dict: {layer_idx: (conv_clone, recurrent_clone)} for layers that
    are LinearAttentionCacheLayerMixin instances.
    """
    from transformers.cache_utils import LinearAttentionCacheLayerMixin
    snapshot = {}
    layers = getattr(cache, "layers", None)
    if layers is None:
        return snapshot
    for idx, layer in enumerate(layers):
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            conv = layer.conv_states.clone() if layer.conv_states is not None else None
            recur = layer.recurrent_states.clone() if layer.recurrent_states is not None else None
            snapshot[idx] = (conv, recur, layer.is_conv_states_initialized, layer.is_recurrent_states_initialized)
    return snapshot


def restore_linear_cache(cache, snapshot):
    """Restore conv_states + recurrent_states from a snapshot (in-place copy_)."""
    layers = getattr(cache, "layers", None)
    if layers is None:
        return
    for idx, (conv, recur, conv_init, recur_init) in snapshot.items():
        layer = layers[idx]
        if conv is not None and layer.conv_states is not None:
            layer.conv_states.copy_(conv)
        if recur is not None and layer.recurrent_states is not None:
            layer.recurrent_states.copy_(recur)
        layer.is_conv_states_initialized = conv_init
        layer.is_recurrent_states_initialized = recur_init


def snapshot_full_attn_cache_seq_len(cache):
    """Record the KV-cache sequence length of all full-attention layers.

    Used to crop back after a trial forward that appended to the KV cache.
    Returns a dict: {layer_idx: seq_length}.
    """
    lengths = {}
    layers = getattr(cache, "layers", None)
    if layers is None:
        return lengths
    for idx, layer in enumerate(layers):
        if not hasattr(layer, "keys"):  # not a full-attention DynamicLayer
            continue
        lengths[idx] = layer.get_seq_length() if hasattr(layer, "get_seq_length") else 0
    return lengths


def crop_full_attn_cache(cache, lengths):
    """Crop full-attention KV caches back to recorded lengths (undo append)."""
    layers = getattr(cache, "layers", None)
    if layers is None:
        return
    for idx, target_len in lengths.items():
        layer = layers[idx]
        cur_len = layer.get_seq_length() if hasattr(layer, "get_seq_length") else 0
        if cur_len > target_len and hasattr(layer, "crop"):
            layer.crop(target_len)
        elif cur_len > target_len and hasattr(layer, "keys") and layer.keys is not None:
            # Manual crop for DynamicLayer
            layer.keys = layer.keys[:, :, :target_len, :]
            layer.values = layer.values[:, :, :target_len, :]


# ============================================================
# Vision Tower (wraps Qwen3_5VisionModel, NO DeepStack)
# ============================================================

class DiffusionVLQwen3_5VisionTower(nn.Module):
    """Wraps Qwen3_5VisionModel, stopping before the main merger.

    Qwen3.5 has NO DeepStack (deepstack_visual_indexes=[]), so the forward
    is simpler than Qwen3-VL: just patch_embed + pos_embed + blocks.
    Returns hidden_states at patch granularity (pre-merger).
    """

    def __init__(self, config, vision_tower_path=None):
        super().__init__()
        self.vision_tower_name = vision_tower_path or ""
        self._config = config

        self.vision_tower = Qwen3_5VisionModel(config)
        if hasattr(self.vision_tower, "merger"):
            delattr(self.vision_tower, "merger")

        self.image_processor = None
        if vision_tower_path and os.path.isdir(vision_tower_path):
            try:
                self.image_processor = AutoImageProcessor.from_pretrained(vision_tower_path, use_fast=False)
            except Exception as e:
                rank0_print(f"WARNING: could not load image processor: {e}")

    def forward(self, hidden_states, grid_thw, **kwargs):
        """Qwen3.5 vision forward (no DeepStack, no window shuffle)."""
        hidden_states = self.vision_tower.patch_embed(hidden_states)

        # pos_embed
        if hasattr(self.vision_tower, "fast_pos_embed_interpolate"):
            pos_embeds = self.vision_tower.fast_pos_embed_interpolate(grid_thw)
        elif hasattr(self.vision_tower, "get_pos_embed"):
            pos_embeds = self.vision_tower.get_pos_embed(grid_thw)
        else:
            pos_embeds = 0
        if isinstance(pos_embeds, torch.Tensor):
            pos_embeds = pos_embeds.to(dtype=hidden_states.dtype, device=hidden_states.device)
            hidden_states = hidden_states + pos_embeds

        # rotary
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
    def hidden_size(self):
        return self._config.hidden_size

    @property
    def spatial_merge_size(self):
        return self._config.spatial_merge_size


# ============================================================
# Projector
# ============================================================

class DiffusionVLQwen3_5Projector(nn.Module):
    """Main patch merger (Qwen3.5 has no DeepStack)."""

    def __init__(self, vision_config):
        super().__init__()
        if isinstance(vision_config, dict):
            vision_config = PretrainedConfig.from_dict(vision_config)
        self.merger = Qwen3_5VisionPatchMerger(config=vision_config, use_postshuffle_norm=False)

    def forward(self, hidden_states):
        return self.merger(hidden_states)


# ============================================================
# Full Attention (non-causal for BD3-LM, with store_kv hook)
# ============================================================

class DiffusionVLQwen3_5Attention(Qwen3_5Attention):
    """Non-causal full attention with KV-cache store_kv hook."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.is_causal = False

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        store_kv = kwargs.pop("store_kv", True)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Qwen3.5's q_proj outputs 2× the query dim (query + gate).
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = self._apply_rotary(query_states, key_states, cos, sin)

        if past_key_values is not None:
            if store_kv:
                cache_kwargs = {"sin": sin, "cos": cos}
                if "cache_position" in kwargs:
                    cache_kwargs["cache_position"] = kwargs["cache_position"]
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
            else:
                _layers = getattr(past_key_values, "layers", None)
                if _layers is not None and self.layer_idx < len(_layers):
                    layer_cache = _layers[self.layer_idx]
                    past_k = getattr(layer_cache, "keys", None)
                    past_v = getattr(layer_cache, "values", None)
                    if past_k is not None and past_k.numel() > 0:
                        key_states = torch.cat([past_k, key_states], dim=2)
                        value_states = torch.cat([past_v, value_states], dim=2)

        attention_interface = self._get_attn_interface()
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling, **kwargs,
        )

        # Apply output gate (Qwen3.5 attn_output_gate)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def _apply_rotary(self, q, k, cos, sin):
        """Apply rotary embedding with partial_rotary_factor support."""
        # Check if partial rotary is configured
        partial = getattr(self.config, "partial_rotary_factor", None)
        if partial is not None and partial < 1.0:
            # Split into rotary and non-rotary parts
            rot_dim = int(self.head_dim * partial)
            q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
            k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]
            cos = cos[..., :rot_dim]
            sin = sin[..., :rot_dim]
            q_rot, k_rot = self._rotate_half_apply(q_rot, k_rot, cos, sin)
            return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)
        return self._rotate_half_apply(q, k, cos, sin)

    @staticmethod
    def _rotate_half_apply(q, k, cos, sin):
        """Standard RoPE apply."""
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)
        cos = cos.unsqueeze(1)  # (B, 1, S, D) from (B, S, D)
        sin = sin.unsqueeze(1)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def _get_attn_interface(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.qwen3_5.modeling_qwen3_5 import eager_attention_forward
        if self.config._attn_implementation != "eager":
            return ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        return eager_attention_forward


# ============================================================
# Text Model (hybrid: replaces full-attn layers, keeps linear layers)
# ============================================================

class DiffusionVLQwen3_5TextModel(Qwen3_5TextModelOriginal):
    """Hybrid text model: non-causal full attention + stock linear attention.

    For full-attention layers, we replace self_attn with DiffusionVLQwen3_5Attention
    (is_causal=False + store_kv hook). Linear-attention layers (Gated DeltaNet)
    are kept as-is — their recurrence is inherently causal, and we handle the
    cache write problem via snapshot-restore in generate_with_bd3lm.
    """

    def __init__(self, config):
        super().__init__(config)

        # Replace only full-attention layers' self_attn
        layer_types = getattr(config, "layer_types", [])
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx < len(layer_types) and layer_types[layer_idx] == "full_attention":
                if hasattr(layer, "self_attn"):
                    original_layer_idx = layer.self_attn.layer_idx
                    layer.self_attn = DiffusionVLQwen3_5Attention(config, layer_idx=original_layer_idx)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        store_kv=False,
        **kwargs,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        kwargs["store_kv"] = store_kv

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        # position_ids: Qwen3.5 uses (4, B, S) — text + 3D MRoPE
        if position_ids is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)
        elif position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0).unsqueeze(0).expand(4, inputs_embeds.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids_3d = position_ids[1:]
        else:
            text_position_ids = position_ids[0] if position_ids.ndim == 3 else position_ids
            position_ids_3d = position_ids

        # Build mask dict if not already a dict
        if not isinstance(attention_mask, dict):
            from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }
        else:
            causal_mask_mapping = attention_mask

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids_3d)

        layer_types = getattr(self.config, "layer_types", [])

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_mask_key = layer_types[i] if i < len(layer_types) else "full_attention"
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask_mapping[layer_mask_key],
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


# ============================================================
# Inner Model
# ============================================================

class DiffusionVLQwen3_5Model(DiffusionVLQwen3_5TextModel):
    """Inner model: vision_tower + mm_projector + text model."""

    def __init__(self, config):
        super().__init__(config)
        vision_cfg = config.vision_config
        if isinstance(vision_cfg, dict):
            vision_cfg = PretrainedConfig.from_dict(vision_cfg)
        self.vision_tower = DiffusionVLQwen3_5VisionTower(vision_cfg)
        self.mm_projector = DiffusionVLQwen3_5Projector(vision_cfg)

    def get_vision_tower(self):
        return self.vision_tower

    def encode_images(self, images, image_grid_thw=None):
        vt = self.get_vision_tower()
        if image_grid_thw is None:
            raise ValueError("`image_grid_thw` is required.")
        return vt(images, grid_thw=image_grid_thw)


# ============================================================
# Multimodal input preparation (LLaVA-style, NO DeepStack)
# ============================================================

def prepare_inputs_labels_for_multimodal(
    model_self, input_ids, position_ids, attention_mask, past_key_values,
    labels, images, modalities=None, image_grid_thws=None,
):
    """Splice image features into text embeddings. Returns 6-tuple (no DeepStack)."""
    vision_tower = model_self.get_vision_tower()

    if vision_tower is None or images is None or input_ids.shape[1] == 1:
        return input_ids, position_ids, attention_mask, past_key_values, None, labels

    images_list = []
    for image in images:
        if image.dim() == 2:
            images_list.append(image)
        elif image.ndim == 4:
            images_list.append(image)
        else:
            images_list.append(image.unsqueeze(0))
    concat_images = torch.cat(images_list, dim=0)

    if image_grid_thws is None:
        raise ValueError("`image_grid_thws` must be provided.")
    image_grid_thw = torch.tensor(image_grid_thws, device=model_self.device)

    hidden_states = model_self.encode_images(concat_images, image_grid_thw=image_grid_thw)
    ordered_image_features = model_self.mm_projector(hidden_states)

    spatial_merge_size = vision_tower.spatial_merge_size
    split_sizes = (image_grid_thw.prod(dim=1) // (spatial_merge_size ** 2)).tolist()
    ordered_image_features_list = torch.split(ordered_image_features, split_sizes)

    image_features = list(ordered_image_features_list)

    _attention_mask = attention_mask
    _labels = labels
    _position_ids = position_ids

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if position_ids is None:
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    input_ids = [cur[cur_am] for cur, cur_am in zip(input_ids, attention_mask)]
    labels = [cur[cur_am] for cur, cur_am in zip(labels, attention_mask)]

    new_input_embeds = []
    new_labels = []
    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        if num_images == 0:
            cur_image_features = image_features[cur_image_idx]
            cur_embeds = model_self.embed_tokens(cur_input_ids)
            cur_embeds = torch.cat([cur_embeds, cur_image_features[0:0]], dim=0)
            new_input_embeds.append(cur_embeds)
            new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            continue

        image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1: image_token_indices[i + 1]])
            cur_labels_noim.append(cur_labels[image_token_indices[i] + 1: image_token_indices[i + 1]])
        split_sizes_text = [x.shape[0] for x in cur_labels_noim]
        cur_input_embeds = model_self.embed_tokens(torch.cat(cur_input_ids_noim))
        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes_text, dim=0)
        cur_new_input_embeds = []
        cur_new_labels = []

        for i in range(num_images + 1):
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            cur_new_labels.append(cur_labels_noim[i])
            if i < num_images:
                try:
                    cur_image_features = image_features[cur_image_idx]
                except IndexError:
                    cur_image_features = image_features[cur_image_idx - 1]
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_image_features)
                n_vis = cur_image_features.shape[0]
                cur_new_labels.append(torch.full((n_vis,), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

        cur_new_input_embeds = [x.to(model_self.device) for x in cur_new_input_embeds]
        cur_new_input_embeds = torch.cat(cur_new_input_embeds)
        cur_new_labels = torch.cat(cur_new_labels)
        new_input_embeds.append(cur_new_input_embeds)
        new_labels.append(cur_new_labels)

    tokenizer_model_max_length = getattr(model_self.config, "tokenizer_model_max_length", None) or getattr(model_self.config, "model_max_length", None)
    new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
    new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

    max_len = max(x.shape[0] for x in new_input_embeds)
    batch_size = len(new_input_embeds)

    new_input_embeds_padded = []
    new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

    for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
        cur_len = cur_new_embed.shape[0]
        new_input_embeds_padded.append(torch.cat(
            (cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
        if cur_len > 0:
            new_labels_padded[i, :cur_len] = cur_new_labels
            attention_mask[i, :cur_len] = True
            position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

    new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

    if _labels is None:
        new_labels = None
    else:
        new_labels = new_labels_padded
    if _attention_mask is None:
        attention_mask = None
    if _position_ids is None:
        position_ids = None

    return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


# ============================================================
# ForConditionalGeneration
# ============================================================

class DiffusionVLQwen3_5PreTrainedModel(Qwen3_5PreTrainedModel):
    config_class = DiffusionVLQwen3_5Config
    _no_split_modules = ["Qwen3_5DecoderLayer", "Qwen3_5VisionBlock"]


class DiffusionVLQwen3_5ForConditionalGeneration(DiffusionVLQwen3_5PreTrainedModel):
    """DiffusionVL-Qwen3.5 inference model with BD3-LM block diffusion.

    Uses snapshot-restore for linear attention cache during block denoising.
    """

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = DiffusionVLQwen3_5Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mask_token_id = getattr(config, "mask_token_id", 248319)
        self.block_size = getattr(config, "bd3lm_block_size", 8)
        self.post_init()

    def tie_weights(self, *args, **kwargs):
        if getattr(self.config, "tie_word_embeddings", False):
            super().tie_weights(*args, **kwargs)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def get_model(self):
        return self.model

    def get_vision_tower(self):
        return self.model.get_vision_tower()

    def encode_images(self, images, image_grid_thw=None):
        return self.model.encode_images(images, image_grid_thw=image_grid_thw)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                images=None, image_grid_thws=None, modalities=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            (_, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = \
                prepare_inputs_labels_for_multimodal(
                    self.model, input_ids, position_ids, attention_mask, past_key_values,
                    labels, images, image_grid_thws=image_grid_thws)

        outputs = self.model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            use_cache=use_cache, return_dict=return_dict)
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1), ignore_index=IGNORE_INDEX)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values)

    @torch.no_grad()
    def generate(self, inputs=None, images=None, image_sizes=None, image_grid_thws=None,
                 modalities=None, gen_length=256, steps=8, temperature=0.0, **kwargs):
        if modalities is None:
            modalities = ["image"]
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)

        if images is not None:
            (_, _, _, _, inputs_embeds, _) = prepare_inputs_labels_for_multimodal(
                self.model, inputs, position_ids, attention_mask, None, None,
                images, image_grid_thws=image_grid_thws)
        else:
            inputs_embeds = self.get_input_embeddings()(inputs)

        kwargs.pop("input_ids", None)
        return self.generate_with_bd3lm(
            inputs_embeds=inputs_embeds, gen_length=gen_length, steps=steps,
            temperature=temperature, **kwargs)

    @torch.no_grad()
    def generate_with_bd3lm(self, inputs_embeds, gen_length=256, steps=8, temperature=0.0, **kwargs):
        """BD3-LM block diffusion with snapshot-restore for linear attention cache."""
        device = inputs_embeds.device
        batch_size = inputs_embeds.shape[0]
        prompt_len = inputs_embeds.shape[1]
        block_size = self.block_size
        mask_id = self.mask_token_id

        is_full_diffusion = block_size >= (prompt_len + gen_length)
        if is_full_diffusion:
            total_length = prompt_len + gen_length
            num_blocks = 1
        else:
            num_blocks = (prompt_len + gen_length + block_size - 1) // block_size
            total_length = num_blocks * block_size

        x_ids = torch.full((batch_size, total_length), mask_id, dtype=torch.long, device=device)
        mask_embed = self.get_input_embeddings()(torch.tensor([mask_id], device=device))
        x_embeds = mask_embed.repeat(batch_size, total_length, 1)
        x_embeds[:, :prompt_len] = inputs_embeds.clone()

        prompt_logits = self.lm_head(inputs_embeds)
        x_ids[:, :prompt_len] = torch.argmax(prompt_logits, dim=-1)

        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device)).to(inputs_embeds.dtype)
        block_diffusion_mask_bool = block_mask.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1).unsqueeze(0)
        block_diffusion_mask = block_diffusion_mask_bool.unsqueeze(1)
        block_diffusion_mask = torch.where(block_diffusion_mask == 0., torch.full_like(block_diffusion_mask, float('-inf')), 0.)
        if is_full_diffusion:
            block_diffusion_mask = block_diffusion_mask[:, :, :total_length, :total_length]

        position_ids = torch.arange(total_length, device=device).unsqueeze(0).expand(batch_size, -1)

        prefill_blocks = prompt_len // block_size
        prefill_length = prefill_blocks * block_size

        past_key_values = DynamicCache(config=self.model.config)

        # Prefill: process prompt blocks, store KV + linear states
        if prefill_length > 0:
            prefill_embeds = x_embeds[:, :prefill_length]
            prefill_mask = block_diffusion_mask[:, :, :prefill_length, :prefill_length]
            model_mask = {"full_attention": prefill_mask, "linear_attention": None}
            prefill_pos_ids = position_ids[:, :prefill_length]

            self.model(
                inputs_embeds=prefill_embeds, attention_mask=model_mask,
                position_ids=prefill_pos_ids, past_key_values=past_key_values,
                use_cache=True, store_kv=True)

        num_transfer_tokens = self._get_num_transfer_tokens(block_size, steps)

        for block_idx in range(prefill_blocks, num_blocks):
            block_start = block_idx * block_size
            block_end = block_start + block_size

            cur_block_embeds = x_embeds[:, block_start:block_end].clone()
            cur_block_ids = x_ids[:, block_start:block_end]

            cur_mask = block_diffusion_mask[:, :, block_start:block_end, :block_end]
            cur_pos_ids = position_ids[:, block_start:block_end]
            model_mask = {"full_attention": cur_mask, "linear_attention": None}

            # Snapshot linear cache + full-attn cache length BEFORE denoising
            linear_snapshot = snapshot_linear_cache(past_key_values)
            full_attn_lengths = snapshot_full_attn_cache_seq_len(past_key_values)

            for step in range(steps + 1):
                # Restore linear cache + crop full-attn cache to pre-block state
                restore_linear_cache(past_key_values, linear_snapshot)
                crop_full_attn_cache(past_key_values, full_attn_lengths)

                is_mask = torch.all(torch.abs(cur_block_embeds - mask_embed) < 1e-5, dim=-1)
                if not is_mask.any():
                    # All tokens revealed: commit this block to cache
                    self.model(
                        inputs_embeds=cur_block_embeds, attention_mask=model_mask,
                        position_ids=cur_pos_ids, past_key_values=past_key_values,
                        use_cache=True, store_kv=True)
                    break

                # Trial forward (writes to both linear + full-attn cache)
                outputs = self.model(
                    inputs_embeds=cur_block_embeds, attention_mask=model_mask,
                    position_ids=cur_pos_ids, past_key_values=past_key_values,
                    use_cache=True, store_kv=False)
                logits = self.lm_head(outputs[0]).float()

                top_k = kwargs.get("top_k", 0)
                top_p = kwargs.get("top_p", 1.0)
                x0, x0_p = self._sample_tokens(logits, temperature=temperature, top_k=top_k, top_p=top_p)
                remasking_strategy = kwargs.get("remasking_strategy", "low_confidence_static")
                num_to_transfer = num_transfer_tokens[step].item()

                transfer_mask = torch.zeros_like(x0, dtype=torch.bool, device=device)
                if remasking_strategy == "low_confidence_static":
                    confidence = torch.where(is_mask, x0_p, -torch.inf)
                    for j in range(confidence.shape[0]):
                        num_masks = is_mask[j].sum().item()
                        k = min(num_to_transfer, num_masks)
                        if k > 0 and not torch.all(torch.isinf(confidence[j])):
                            _, idx = torch.topk(confidence[j], k)
                            transfer_mask[j, idx] = True
                elif remasking_strategy == "low_confidence_dynamic":
                    confidence_threshold = kwargs.get("confidence_threshold", 0.85)
                    confidence = torch.where(is_mask, x0_p, -torch.inf)
                    for j in range(confidence.shape[0]):
                        high_conf = confidence[j] > confidence_threshold
                        if high_conf.sum().item() >= num_to_transfer:
                            transfer_mask[j] = high_conf
                        else:
                            num_masks = is_mask[j].sum().item()
                            k = min(num_to_transfer, num_masks)
                            if k > 0:
                                _, idx = torch.topk(confidence[j], k)
                                transfer_mask[j, idx] = True
                else:
                    raise ValueError(f"Unknown remasking strategy: {remasking_strategy}")

                cur_block_ids = torch.where(transfer_mask, x0, cur_block_ids)
                x0_embeds = self.get_input_embeddings()(x0)
                cur_block_embeds = torch.where(transfer_mask.unsqueeze(-1), x0_embeds, cur_block_embeds)

            x_embeds[:, block_start:block_end] = cur_block_embeds
            x_ids[:, block_start:block_end] = cur_block_ids

            if block_end > prompt_len:
                gen_start = max(prompt_len, block_start)
                gen_ids_check = x_ids[:, gen_start:block_end]
                eos_token_id = kwargs.get("eos_token_id", 248044)
                if eos_token_id in gen_ids_check:
                    break

        return x_ids[:, prompt_len:prompt_len + gen_length]

    @staticmethod
    def _top_k_logits(logits, k):
        if k <= 0:
            return logits
        values, _ = torch.topk(logits, k)
        return torch.where(logits < values[..., -1, None], torch.full_like(logits, float('-inf')), logits)

    @staticmethod
    def _top_p_logits(logits, p):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        mask_indices = torch.scatter(torch.full_like(logits, False, dtype=torch.bool), -1, sorted_indices, sorted_mask)
        return logits.masked_fill(mask_indices, float('-inf'))

    def _sample_tokens(self, logits, temperature=0.0, top_k=0, top_p=1.0):
        orig_shape = logits.shape[:-1]
        vocab_size = logits.shape[-1]
        logits_2d = logits.reshape(-1, vocab_size)
        if temperature == 0:
            token = torch.argmax(logits_2d, dim=-1, keepdim=True)
            probs = F.softmax(logits_2d, dim=-1)
            token_prob = torch.gather(probs, -1, token)
        else:
            logits_mod = logits_2d.clone()
            if temperature != 1.0:
                logits_mod = logits_mod / temperature
            if top_k > 0:
                logits_mod = self._top_k_logits(logits_mod, top_k)
            if top_p < 1.0:
                logits_mod = self._top_p_logits(logits_mod, top_p)
            probs = F.softmax(logits_mod, dim=-1)
            token = torch.multinomial(probs, num_samples=1)
            token_prob = torch.gather(probs, -1, token)
        return token.view(*orig_shape), token_prob.view(*orig_shape)

    @staticmethod
    def _get_num_transfer_tokens(block_length, steps):
        if steps == 0:
            return torch.zeros(1, dtype=torch.int64)
        base = block_length // steps
        remainder = block_length % steps
        num_transfer_tokens = torch.zeros(steps + 1, dtype=torch.int64) + base
        num_transfer_tokens[:remainder] += 1
        return num_transfer_tokens


AutoConfig.register("diffusionvl_qwen3_5", DiffusionVLQwen3_5Config)
AutoModelForCausalLM.register(DiffusionVLQwen3_5Config, DiffusionVLQwen3_5ForConditionalGeneration)

__all__ = [
    "DiffusionVLQwen3_5Config", "DiffusionVLQwen3_5PreTrainedModel",
    "DiffusionVLQwen3_5Model", "DiffusionVLQwen3_5ForConditionalGeneration",
]
