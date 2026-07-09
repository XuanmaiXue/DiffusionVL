# coding=utf-8
# Copyright 2025 The HustVL Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on Qwen3-VL (https://github.com/QwenLM/Qwen3-VL),
# LLaDA-V (https://github.com/ML-GSAI/LLaDA-V), and
# Block Diffusion (https://github.com/kuleshov-group/bd3lm). It has been
# modified to create DiffusionVL.
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
"""DiffusionVL-Qwen3VL model — self-contained inference implementation.

This file merges the DiffusionVL-Qwen3VL training code into a single
self-contained module with no dependency on the `llava` package, so a checkpoint
can be loaded via `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`.

It contains ONLY the inference path (generate / generate_with_bd3lm /
prepare_inputs_labels_for_multimodal / forward for eval loss). Training-only
logic (LogLinearNoise, block_diff_mask, _apply_bd3lm_noise_embedding,
_compute_bd3lm_loss_embedding) is NOT included.

Qwen3-VL backbone classes are imported from `transformers.models.qwen3_vl`
(requires transformers >= 4.57.0).
"""

import os
from typing import List, Optional, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoImageProcessor, AutoModelForCausalLM, PretrainedConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.utils import logging
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextAttention,
    Qwen3VLTextModel as Qwen3VLTextModelOriginal,
    Qwen3VLPreTrainedModel,
    Qwen3VLVisionModel,
    Qwen3VLVisionRotaryEmbedding,
    Qwen3VLVisionPatchMerger,
    apply_rotary_pos_emb,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
)

try:
    from .configuration_diffusionvl_qwen3_vl import DiffusionVLQwen3VLConfig
except ImportError:
    # When loaded as standalone scripts (not as a package), use direct import.
    from configuration_diffusionvl_qwen3_vl import DiffusionVLQwen3VLConfig

logger = logging.get_logger(__name__)

# --- Constants (inlined from llava.constants) ---
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"


def rank0_print(*args):
    """Print only on rank 0 (inlined from llava.utils)."""
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(*args)


# ============================================================
# Vision Tower (simplified from LlavaQwen3VisionTower — no load_model,
# built as a regular submodule; weights loaded by HF from_pretrained)
# ============================================================

class DiffusionVLQwen3VLVisionTower(nn.Module):
    """Wraps Qwen3VLVisionModel, stopping before the main merger (owned by the
    projector) but keeping the DeepStack mergers.

    The forward returns (hidden_states, deepstack_features) where hidden_states
    is at patch granularity (pre-merger) and deepstack_features is a list of
    already-merged tensors (one per deepstack visual index).
    """

    def __init__(self, config, vision_tower_path=None):
        super().__init__()
        self.vision_tower_name = vision_tower_path or ""
        self._config = config

        # Build the full Qwen3VLVisionModel (patch_embed, pos_embed, blocks,
        # merger, deepstack_merger_list). The top-level merger is deleted because
        # the main merger is owned by the projector; deepstack_merger_list stays.
        self.vision_tower = Qwen3VLVisionModel(config)
        if hasattr(self.vision_tower, "merger"):
            delattr(self.vision_tower, "merger")

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)
        self.vision_tower.rotary_pos_emb = self.rotary_pos_emb

        # Load image processor if a path is available.
        self.image_processor = None
        if vision_tower_path and os.path.isdir(vision_tower_path):
            try:
                self.image_processor = AutoImageProcessor.from_pretrained(vision_tower_path, use_fast=False)
            except Exception as e:
                rank0_print(f"WARNING: could not load image processor from {vision_tower_path}: {e}")

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
        """Replicate Qwen3VLVisionModel.forward but stop before the main merger."""
        hidden_states = self.vision_tower.patch_embed(hidden_states)

        pos_embeds = self.vision_tower.fast_pos_embed_interpolate(grid_thw)
        # Align dtype: pos_embeds (from nn.Embedding) may be float32 while
        # hidden_states / model weights are bfloat16. NPU's aclnnLayerNorm
        # requires input and weight to share a dtype.
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

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.vision_tower.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
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
    def hidden_size(self):
        return self._config.hidden_size

    @property
    def spatial_merge_size(self):
        return self._config.spatial_merge_size


# ============================================================
# Projector (from LlavaQwen3Projector)
# ============================================================

class DiffusionVLQwen3VLProjector(nn.Module):
    """Main patch merger. Qwen3-VL has no window shuffle, so no un-shuffle."""

    def __init__(self, vision_config):
        super().__init__()
        if isinstance(vision_config, dict):
            vision_config = PretrainedConfig.from_dict(vision_config)
        self.merger = Qwen3VLVisionPatchMerger(
            config=vision_config,
            use_postshuffle_norm=False,
        )

    def forward(self, features_tuple):
        hidden_states, deepstack_features = features_tuple
        final_features = self.merger(hidden_states)
        return final_features, deepstack_features


# ============================================================
# Attention (non-causal for BD3-LM, with KV-cache store_kv hook)
# ============================================================

class DiffusionVLQwen3VLAttention(Qwen3VLTextAttention):
    """Non-causal attention with KV-cache store_kv hook for block-diffusion."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        store_kv = kwargs.pop("store_kv", True)
        flash_attn_kwargs = kwargs

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            if store_kv:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
            else:
                # Read cached KV without updating (for block-diffusion re-evaluation).
                # transformers >= 5.x DynamicCache uses .layers[idx].keys/.values.
                _layers = getattr(past_key_values, "layers", None)
                if _layers is not None and self.layer_idx < len(_layers):
                    layer_cache = _layers[self.layer_idx]
                    past_key_states = getattr(layer_cache, "keys", getattr(layer_cache, "key_cache", None))
                    past_value_states = getattr(layer_cache, "values", getattr(layer_cache, "value_cache", None))
                    if past_key_states is not None and past_key_states.numel() > 0:
                        key_states = torch.cat([past_key_states, key_states], dim=2)
                        value_states = torch.cat([past_value_states, value_states], dim=2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **flash_attn_kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# ============================================================
# Text Model (DeepStack injection + dict attention mask)
# ============================================================

class DiffusionVLQwen3VLTextModel(Qwen3VLTextModelOriginal):
    """Text model with non-causal attention and DeepStack visual injection."""

    def __init__(self, config):
        super().__init__(config)
        for layer in self.layers:
            original_layer_idx = layer.self_attn.layer_idx
            layer.self_attn = DiffusionVLQwen3VLAttention(config, layer_idx=original_layer_idx)
        # Qwen3-VL has no sliding-window layers.
        self.has_sliding_layers = False

    def _deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        store_kv=False,
        visual_pos_masks=None,
        deepstack_visual_embeds=None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs["store_kv"] = store_kv

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                use_cache = False

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        if position_ids is not None and position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0).expand(inputs_embeds.shape[0], -1)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        if not isinstance(attention_mask, dict):
            causal_mask_mapping = {
                "full_attention": attention_mask,
                "sliding_attention": attention_mask if self.has_sliding_layers else None,
            }
        else:
            causal_mask_mapping = attention_mask

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping["full_attention"],
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states, visual_pos_masks, deepstack_visual_embeds[layer_idx]
                )

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values, all_hidden_states] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )


# ============================================================
# Inner Model (vision_tower + mm_projector + text model)
# ============================================================

class DiffusionVLQwen3VLModel(DiffusionVLQwen3VLTextModel):
    """Inner model combining vision tower, projector, and text model."""

    def __init__(self, config):
        super().__init__(config)
        # Build vision tower and projector as regular submodules. Weights are
        # loaded by HF from_pretrained via key matching (model.vision_tower.*,
        # model.mm_projector.*). No initialize_vision_modules / load_model needed.
        vision_cfg = config.vision_config
        if isinstance(vision_cfg, dict):
            vision_cfg = PretrainedConfig.from_dict(vision_cfg)
        self.vision_tower = DiffusionVLQwen3VLVisionTower(vision_cfg)
        self.mm_projector = DiffusionVLQwen3VLProjector(vision_cfg)

    def get_vision_tower(self):
        return self.vision_tower

    def encode_images(self, images, image_grid_thw=None):
        vision_tower = self.get_vision_tower()
        if image_grid_thw is None:
            raise ValueError("`image_grid_thw` is required for Qwen vision tower.")
        return vision_tower(images, grid_thw=image_grid_thw)


# ============================================================
# Multimodal input preparation (from llava_arch, Qwen3 branch)
# ============================================================

def prepare_inputs_labels_for_multimodal(
    model_self,
    input_ids,
    position_ids,
    attention_mask,
    past_key_values,
    labels,
    images,
    modalities=None,
    image_sizes=None,
    image_grid_thws=None,
):
    """Merge image features into text embeddings (LLaVA-style splice).

    Returns 8-tuple: (None, position_ids, attention_mask, past_key_values,
    inputs_embeds, labels, visual_pos_masks, deepstack_visual_embeds).
    """
    import random

    vision_tower = model_self.get_vision_tower()

    if vision_tower is None or images is None or input_ids.shape[1] == 1:
        return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None

    if isinstance(modalities, str):
        modalities = [modalities]

    images_list = []
    for image in images:
        if image.dim() == 2:
            images_list.append(image)
        else:
            if image.ndim == 4:
                images_list.append(image)
            else:
                images_list.append(image.unsqueeze(0))
    concat_images = torch.cat(images_list, dim=0)

    if image_grid_thws is None:
        raise ValueError("`image_grid_thws` must be provided by the dataloader for Qwen models.")
    image_grid_thw = torch.tensor(image_grid_thws, device=model_self.device)

    # Encode images: (hidden_states, deepstack_features)
    image_features_tuple = model_self.encode_images(concat_images, image_grid_thw=image_grid_thw)
    projector_out = model_self.mm_projector(image_features_tuple)

    # Qwen3 projector returns (features, deepstack_features).
    if isinstance(projector_out, tuple):
        ordered_image_features, deepstack_image_embeds = projector_out
    else:
        ordered_image_features, deepstack_image_embeds = projector_out, None

    spatial_merge_size = vision_tower.spatial_merge_size
    split_sizes = (image_grid_thw.prod(dim=1) // (spatial_merge_size ** 2)).tolist()
    ordered_image_features_list = torch.split(ordered_image_features, split_sizes)

    # Split DeepStack features per image.
    deepstack_per_image = None
    if deepstack_image_embeds is not None:
        deepstack_per_image = [torch.split(ds, split_sizes) for ds in deepstack_image_embeds]

    image_features = []
    for idx, image_feat in enumerate(ordered_image_features_list):
        image_features.append(image_feat)

    _attention_mask = attention_mask
    _labels = labels  # save original for later None-check
    _position_ids = position_ids  # save original for later None-check
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if position_ids is None:
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
    labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

    new_input_embeds = []
    new_labels = []
    per_batch_visual_spans = []
    per_batch_deepstack = []
    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        visual_spans = []
        batch_deepstack = []
        if num_images == 0:
            cur_image_features = image_features[cur_image_idx]
            cur_input_embeds_1 = model_self.embed_tokens(cur_input_ids)
            cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
            new_input_embeds.append(cur_input_embeds)
            new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            per_batch_visual_spans.append(visual_spans)
            per_batch_deepstack.append(batch_deepstack)
            continue

        image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
        split_sizes_text = [x.shape[0] for x in cur_labels_noim]
        cur_input_embeds = model_self.embed_tokens(torch.cat(cur_input_ids_noim))
        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes_text, dim=0)
        cur_new_input_embeds = []
        cur_new_labels = []

        running_len = 0
        for i in range(num_images + 1):
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            cur_new_labels.append(cur_labels_noim[i])
            running_len += cur_input_embeds_no_im[i].shape[0]
            if i < num_images:
                try:
                    cur_image_features = image_features[cur_image_idx]
                except IndexError:
                    cur_image_features = image_features[cur_image_idx - 1]
                if deepstack_per_image is not None:
                    img_ds = [deepstack_per_image[layer][cur_image_idx] for layer in range(len(deepstack_per_image))]
                    batch_deepstack.append(img_ds)
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_image_features)
                n_vis = cur_image_features.shape[0]
                cur_new_labels.append(torch.full((n_vis,), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                visual_spans.append((running_len, running_len + n_vis))
                running_len += n_vis

        cur_new_input_embeds = [x.to(model_self.device) for x in cur_new_input_embeds]
        cur_new_input_embeds = torch.cat(cur_new_input_embeds)
        cur_new_labels = torch.cat(cur_new_labels)
        new_input_embeds.append(cur_new_input_embeds)
        new_labels.append(cur_new_labels)
        per_batch_visual_spans.append(visual_spans)
        per_batch_deepstack.append(batch_deepstack)

    tokenizer_model_max_length = getattr(model_self.config, "tokenizer_model_max_length", None) or getattr(model_self.config, "model_max_length", None)
    new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
    new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

    max_len = max(x.shape[0] for x in new_input_embeds)
    batch_size = len(new_input_embeds)

    new_input_embeds_padded = []
    new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
    visual_pos_masks = torch.zeros((batch_size, max_len), dtype=torch.bool, device=attention_mask.device)
    deepstack_layer_features = None
    if deepstack_per_image is not None:
        deepstack_layer_features = [None] * len(deepstack_per_image)

    for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
        cur_len = cur_new_embed.shape[0]
        pad_offset = max_len - cur_len if getattr(model_self.config, "tokenizer_padding_side", "right") == "left" else 0
        if pad_offset:
            new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
            if cur_len > 0:
                new_labels_padded[i, -cur_len:] = cur_new_labels
                attention_mask[i, -cur_len:] = True
                position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
        else:
            new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        for img_i, (vstart, vend) in enumerate(per_batch_visual_spans[i]):
            if vstart >= cur_len:
                break
            vend_clipped = min(vend, cur_len)
            pstart = pad_offset + vstart
            pend = pad_offset + vend_clipped
            visual_pos_masks[i, pstart:pend] = True
            if deepstack_layer_features is not None and img_i < len(per_batch_deepstack[i]):
                n_kept = vend_clipped - vstart
                for layer in range(len(deepstack_layer_features)):
                    ds_tensor = per_batch_deepstack[i][img_i][layer][:n_kept]
                    deepstack_layer_features[layer] = (
                        ds_tensor if deepstack_layer_features[layer] is None
                        else torch.cat([deepstack_layer_features[layer], ds_tensor], dim=0)
                    )

    new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

    if _labels is None:
        new_labels = None
    else:
        new_labels = new_labels_padded
    if _attention_mask is None:
        attention_mask = None
    if _position_ids is None:
        position_ids = None

    deepstack_visual_embeds = None
    if deepstack_layer_features is not None and any(f is not None for f in deepstack_layer_features):
        deepstack_visual_embeds = [
            f.to(new_input_embeds.device, new_input_embeds.dtype) for f in deepstack_layer_features if f is not None
        ]
    elif deepstack_per_image is not None:
        visual_pos_masks = None

    return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, visual_pos_masks, deepstack_visual_embeds


# ============================================================
# ForConditionalGeneration (main model)
# ============================================================

class DiffusionVLQwen3VLPreTrainedModel(Qwen3VLPreTrainedModel):
    config_class = DiffusionVLQwen3VLConfig
    _no_split_modules = ["DiffusionVLQwen3VLTextModel"]


class DiffusionVLQwen3VLForConditionalGeneration(DiffusionVLQwen3VLPreTrainedModel):
    """DiffusionVL-Qwen3VL for inference (BD3-LM block-diffusion generation)."""

    # transformers >= 5.x expects _tied_weights_keys as a dict (target -> source).
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = DiffusionVLQwen3VLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mask_token_id = getattr(config, "mask_token_id", 151671)
        self.block_size = getattr(config, "bd3lm_block_size", 8)
        self.post_init()

    def tie_weights(self, *args, **kwargs):
        """Tie lm_head with embed_tokens if config.tie_word_embeddings is True."""
        if getattr(self.config, "tie_word_embeddings", False):
            super().tie_weights(*args, **kwargs)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_model(self):
        return self.model

    def get_vision_tower(self):
        return self.model.get_vision_tower()

    def encode_images(self, images, image_grid_thw=None):
        return self.model.encode_images(images, image_grid_thw=image_grid_thw)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        images=None,
        image_grid_thws=None,
        modalities=None,
        return_dict=None,
    ):
        """Forward for eval loss (standard causal LM loss, no BD3-LM noise)."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            (
                _, position_ids, attention_mask, past_key_values, inputs_embeds, labels,
                _, _,
            ) = prepare_inputs_labels_for_multimodal(
                self.model, input_ids, position_ids, attention_mask, past_key_values,
                labels, images, modalities=modalities, image_grid_thws=image_grid_thws,
            )

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs=None,
        images=None,
        image_sizes=None,
        image_grid_thws=None,
        modalities=None,
        gen_length=256,
        steps=8,
        temperature=0.0,
        **kwargs,
    ):
        """BD3-LM diffusion-based generation."""
        if modalities is None:
            modalities = ["image"]

        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if images is not None:
            (_, _, _, _, inputs_embeds, _, visual_pos_masks, deepstack_visual_embeds) = prepare_inputs_labels_for_multimodal(
                self.model, inputs, position_ids, attention_mask, None, None,
                images, modalities=modalities, image_grid_thws=image_grid_thws,
            )
        else:
            inputs_embeds = self.get_input_embeddings()(inputs)

        kwargs.pop("input_ids", None)
        return self.generate_with_bd3lm(
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            gen_length=gen_length,
            steps=steps,
            temperature=temperature,
            **kwargs,
        )

    @torch.no_grad()
    def generate_with_bd3lm(
        self,
        inputs_embeds,
        gen_length=256,
        steps=8,
        temperature=0.0,
        visual_pos_masks=None,
        deepstack_visual_embeds=None,
        **kwargs,
    ):
        """BD3-LM block-diffusion generation with KV-cache and DeepStack injection."""
        device = inputs_embeds.device
        batch_size = inputs_embeds.shape[0]
        prompt_len = inputs_embeds.shape[1]
        block_size = self.block_size
        mask_id = self.mask_token_id

        is_full_diffusion_ablation = block_size >= (prompt_len + gen_length)
        if is_full_diffusion_ablation:
            rank0_print("Full-Diffusion ablation mode enabled.")
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
        prompt_ids_reconstructed = torch.argmax(prompt_logits, dim=-1)
        x_ids[:, :prompt_len] = prompt_ids_reconstructed

        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device)).to(inputs_embeds.dtype)
        block_diffusion_mask_bool = block_mask.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1).unsqueeze(0)
        block_diffusion_mask = block_diffusion_mask_bool.unsqueeze(1)
        block_diffusion_mask = torch.where(block_diffusion_mask == 0., torch.full_like(block_diffusion_mask, float('-inf')), 0.)
        if is_full_diffusion_ablation:
            block_diffusion_mask = block_diffusion_mask[:, :, :total_length, :total_length]

        position_ids = torch.arange(total_length, device=device).unsqueeze(0).expand(batch_size, -1)

        prefill_blocks = prompt_len // block_size
        prefill_length = prefill_blocks * block_size

        # DeepStack for prefill.
        prefill_visual_pos_masks = None
        prefill_deepstack_embeds = None
        prefill_vis_count = 0
        if visual_pos_masks is not None and deepstack_visual_embeds is not None and prefill_length > 0:
            prefill_visual_pos_masks = visual_pos_masks[:, :prefill_length]
            prefill_vis_count = int(prefill_visual_pos_masks.sum().item())
            if prefill_vis_count > 0:
                prefill_deepstack_embeds = [e[:prefill_vis_count] for e in deepstack_visual_embeds]
            else:
                prefill_visual_pos_masks = None

        past_key_values = DynamicCache()
        if prefill_length > 0:
            prefill_embeds = x_embeds[:, :prefill_length]
            prefill_mask = block_diffusion_mask[:, :, :prefill_length, :prefill_length]
            prefill_pos_ids = position_ids[:, :prefill_length]
            model_mask = {"full_attention": prefill_mask, "sliding_attention": prefill_mask}

            prefill_outputs = self.model(
                inputs_embeds=prefill_embeds,
                attention_mask=model_mask,
                position_ids=prefill_pos_ids,
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=True,
                visual_pos_masks=prefill_visual_pos_masks,
                deepstack_visual_embeds=prefill_deepstack_embeds,
            )
            past_key_values = prefill_outputs.past_key_values

        # Per-block DeepStack slices.
        block_deepstack = {}
        if visual_pos_masks is not None and deepstack_visual_embeds is not None:
            vis_cursor = prefill_vis_count
            for bidx in range(prefill_blocks, num_blocks):
                bs = bidx * block_size
                be = bs + block_size
                block_mask_slice = visual_pos_masks[:, bs:min(be, visual_pos_masks.shape[1])]
                if block_mask_slice.shape[1] < block_size:
                    pad = torch.zeros(
                        block_mask_slice.shape[0], block_size - block_mask_slice.shape[1],
                        dtype=block_mask_slice.dtype, device=block_mask_slice.device,
                    )
                    block_mask_slice = torch.cat([block_mask_slice, pad], dim=1)
                n_vis = int(block_mask_slice.sum().item())
                if n_vis > 0:
                    block_embeds = [e[vis_cursor:vis_cursor + n_vis] for e in deepstack_visual_embeds]
                    block_deepstack[bidx] = (block_mask_slice, block_embeds)
                    vis_cursor += n_vis

        num_transfer_tokens = self._get_num_transfer_tokens(block_size, steps)

        for block_idx in range(prefill_blocks, num_blocks):
            block_start = block_idx * block_size
            block_end = block_start + block_size

            cur_block_embeds = x_embeds[:, block_start:block_end].clone()
            cur_block_ids = x_ids[:, block_start:block_end]

            cur_mask = block_diffusion_mask[:, :, block_start:block_end, :block_end]
            cur_pos_ids = position_ids[:, block_start:block_end]
            model_mask = {"full_attention": cur_mask, "sliding_attention": cur_mask}

            blk_vis_mask, blk_ds_embeds = block_deepstack.get(block_idx, (None, None))

            for step in range(steps + 1):
                is_mask = torch.all(torch.abs(cur_block_embeds - mask_embed) < 1e-5, dim=-1)
                if not is_mask.any():
                    _ = self.model(
                        inputs_embeds=cur_block_embeds,
                        attention_mask=model_mask,
                        position_ids=cur_pos_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        store_kv=True,
                        visual_pos_masks=blk_vis_mask,
                        deepstack_visual_embeds=blk_ds_embeds,
                    )
                    break

                outputs = self.model(
                    inputs_embeds=cur_block_embeds,
                    attention_mask=model_mask,
                    position_ids=cur_pos_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=False,
                    visual_pos_masks=blk_vis_mask,
                    deepstack_visual_embeds=blk_ds_embeds,
                )
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
                        high_conf_mask = confidence[j] > confidence_threshold
                        num_high_confidence = high_conf_mask.sum().item()
                        if num_high_confidence >= num_to_transfer:
                            transfer_mask[j] = high_conf_mask
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
                gen_start_in_block = max(prompt_len, block_start)
                gen_ids_check = x_ids[:, gen_start_in_block:block_end]
                eos_token_id = kwargs.get("eos_token_id", 151645)
                if eos_token_id in gen_ids_check:
                    break

        return x_ids[:, prompt_len:prompt_len + gen_length]

    @staticmethod
    def _top_k_logits(logits, k):
        if k <= 0:
            return logits
        values, _ = torch.topk(logits, k)
        min_values = values[..., -1, None]
        return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)

    @staticmethod
    def _top_p_logits(logits, p):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        mask_indices = torch.scatter(torch.full_like(logits, False, dtype=torch.bool), -1, sorted_indices, sorted_mask)
        logits = logits.masked_fill(mask_indices, float('-inf'))
        return logits

    def _sample_tokens(self, logits, temperature=0.0, top_k=0, top_p=1.0):
        orig_shape = logits.shape[:-1]
        vocab_size = logits.shape[-1]
        logits_2d = logits.reshape(-1, vocab_size)

        if temperature == 0:
            token = torch.argmax(logits_2d, dim=-1, keepdim=True)
            probs_original = F.softmax(logits_2d, dim=-1)
            token_prob = torch.gather(probs_original, -1, token)
        else:
            logits_modified = logits_2d.clone()
            if temperature != 1.0:
                logits_modified = logits_modified / temperature
            if top_k > 0:
                logits_modified = self._top_k_logits(logits_modified, top_k)
            if top_p < 1.0:
                logits_modified = self._top_p_logits(logits_modified, top_p)
            probs_modified = F.softmax(logits_modified, dim=-1)
            token = torch.multinomial(probs_modified, num_samples=1)
            token_prob = torch.gather(probs_modified, -1, token)

        return token.view(*orig_shape), token_prob.view(*orig_shape)

    @staticmethod
    def _get_num_transfer_tokens(block_length, steps):
        if steps == 0:
            return torch.zeros(0, dtype=torch.int64)
        base = block_length // steps
        remainder = block_length % steps
        num_transfer_tokens = torch.zeros(steps + 1, dtype=torch.int64) + base
        num_transfer_tokens[:remainder] += 1
        return num_transfer_tokens


# Register the model
AutoConfig.register("diffusionvl_qwen3vl", DiffusionVLQwen3VLConfig)
AutoModelForCausalLM.register(DiffusionVLQwen3VLConfig, DiffusionVLQwen3VLForConditionalGeneration)

__all__ = [
    "DiffusionVLQwen3VLConfig",
    "DiffusionVLQwen3VLPreTrainedModel",
    "DiffusionVLQwen3VLModel",
    "DiffusionVLQwen3VLForConditionalGeneration",
]
