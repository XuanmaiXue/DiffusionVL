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

"""DiffusionVL-Qwen3.5 model implementation (training + inference).

Qwen3.5 uses a hybrid architecture: ~75% Gated DeltaNet (linear attention)
layers + ~25% full softmax attention layers. Key adaptations for BD3-LM:

1. Full-attention layers: is_causal=False (BD3-LM needs non-causal), with a
   store_kv hook for block-diffusion sampling.
2. Linear-attention layers (Gated DeltaNet): kept AS-IS. Their recurrence is
   inherently causal, so no is_causal change. During training (doubling mode),
   the linear layers process the doubled sequence normally — the block_diff_mask
   is applied only to full-attention layers; linear layers use a recurrent mask.
3. No DeepStack (deepstack_visual_indexes=[]).

The BD3-LM diffusion logic (noise/loss/generate) is carried over from the
Qwen3-VL implementation, adapted for the hybrid mask structure.
"""

import os
import json
from typing import List, Optional, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig
from transformers.configuration_utils import PretrainedConfig as HFPretrainedConfig
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig, Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5TextModel as Qwen3_5TextModelOriginal,
    Qwen3_5PreTrainedModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
)

from llava.constants import IGNORE_INDEX
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.utils import rank0_print

logger = logging.get_logger(__name__)


class DiffusionVLQwen3_5Config(HFPretrainedConfig):
    """Configuration class for DiffusionVL-Qwen3.5 model."""

    model_type = "diffusionvl_qwen3_5"
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_id=248056,
        video_token_id=248057,
        enable_bd3lm=False,
        bd3lm_block_size=4,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {}
        if text_config is None:
            text_config = {}

        self.vision_config = Qwen3_5VisionConfig(**vision_config)
        self.text_config = Qwen3_5TextConfig(**text_config)
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

        self.enable_bd3lm = enable_bd3lm
        self.bd3lm_block_size = bd3lm_block_size

        if self.enable_bd3lm:
            self.bd3lm_antithetic_sampling = True
            self.bd3lm_sampling_eps_min = 1e-3
            self.bd3lm_sampling_eps_max = 1.0

        for key, value in self.text_config.to_dict().items():
            setattr(self, key, value)

    def to_dict(self):
        output = super().to_dict()
        output["vision_config"] = self.vision_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        return output


class DiffusionVLQwen3_5Attention(Qwen3_5Attention):
    """Non-causal full attention with KV-cache store_kv hook for BD3-LM.

    Only applied to full_attention layers. Linear attention layers (Gated
    DeltaNet) are left untouched — their recurrence is inherently causal.
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        store_kv = kwargs.pop("store_kv", True)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Qwen3.5's q_proj outputs 2× the query dim (query + gate).
        # Chunk on the head_dim*2 axis, then reshape.
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

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

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
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


class DiffusionVLQwen3_5TextModel(Qwen3_5TextModelOriginal):
    """Hybrid text model with non-causal full attention for BD3-LM.

    Replaces only full_attention layers' self_attn with DiffusionVLQwen3_5Attention.
    Linear attention layers (Gated DeltaNet) are kept as-is.
    """

    def __init__(self, config):
        super().__init__(config)

        # Replace only full-attention layers
        layer_types = getattr(config, "layer_types", [])
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx < len(layer_types) and layer_types[layer_idx] == "full_attention":
                if hasattr(layer, "self_attn"):
                    original_layer_idx = layer.self_attn.layer_idx
                    layer.self_attn = DiffusionVLQwen3_5Attention(config, layer_idx=original_layer_idx)

        if getattr(config, "enable_bd3lm", False):
            self._init_bd3lm_components(config)

    def _init_bd3lm_components(self, config):
        from .bd3lm_utils import LogLinearNoise
        self.noise_scheduler = LogLinearNoise()
        self.mask_token_id = getattr(config, "mask_token_id", 248319)
        self.bd3lm_block_size = config.bd3lm_block_size
        self.antithetic_sampling = getattr(config, "bd3lm_antithetic_sampling", True)
        self.sampling_eps_min = getattr(config, "bd3lm_sampling_eps_min", 1e-3)
        self.sampling_eps_max = getattr(config, "bd3lm_sampling_eps_max", 1.0)

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
        store_kv=False,
        **kwargs,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs["store_kv"] = store_kv

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once("use_cache incompatible with gradient checkpointing")
                use_cache = False

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Qwen3.5 position_ids: (4, B, S) — text + 3D MRoPE
        if position_ids is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        if position_ids is not None and position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0).expand(inputs_embeds.shape[0], -1)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids_3d = position_ids[1:]
        else:
            text_position_ids = position_ids[0] if position_ids.ndim == 3 else position_ids
            position_ids_3d = position_ids

        # Build mask dict for hybrid layers
        if not isinstance(attention_mask, dict):
            from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
            mask_kwargs = {
                "config": self.config, "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask, "past_key_values": past_key_values,
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


class DiffusionVLQwen3_5ForCausalLM_Base(Qwen3_5PreTrainedModel):
    """Base CausalLM with BD3-LM diffusion logic."""

    # transformers >= 5.x expects _tied_weights_keys as a dict
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = DiffusionVLQwen3_5TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def tie_weights(self, *args, **kwargs):
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

    def _apply_bd3lm_noise_embedding(self, inputs_embeds, labels):
        batch_size, seq_len, hidden_size = inputs_embeds.shape
        device = inputs_embeds.device
        block_size = self.model.bd3lm_block_size
        num_blocks = (seq_len + block_size - 1) // block_size
        _eps_b = torch.rand((batch_size, num_blocks), device=device)

        if self.model.antithetic_sampling:
            num_samples = _eps_b.numel()
            offset = torch.arange(num_samples, device=device) / num_samples
            offset = offset.view(_eps_b.shape)
            _eps_b = (_eps_b / num_samples + offset) % 1

        t = _eps_b.repeat_interleave(block_size, dim=-1)
        t = t[:, :seq_len]
        t = t * (self.model.sampling_eps_max - self.model.sampling_eps_min) + self.model.sampling_eps_min

        loss_scale, p = self.model.noise_scheduler(t)

        move_probabilities = torch.rand(batch_size, seq_len, device=device)
        move_chance = p
        text_token_mask = (labels != IGNORE_INDEX)
        move_indices = (move_probabilities <= move_chance) & text_token_mask

        mask_embed = self.get_input_embeddings()(torch.tensor([self.model.mask_token_id], device=device))
        xt_embeds = torch.where(move_indices.unsqueeze(-1), mask_embed, inputs_embeds)

        avg_noise_level = torch.mean(move_chance).item()
        bd3lm_inputs = torch.cat([xt_embeds, inputs_embeds], dim=1)
        return bd3lm_inputs, move_indices, loss_scale, inputs_embeds, avg_noise_level

    def _compute_bd3lm_loss_embedding(self, logits, labels, move_indices, loss_scale):
        masked_positions = move_indices & (labels != IGNORE_INDEX)

        if not masked_positions.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        logits_flat = logits[masked_positions]
        labels_flat = labels[masked_positions]
        token_loss_unweighted = F.cross_entropy(logits_flat, labels_flat, reduction="none")

        loss_scale_flat = loss_scale[masked_positions]
        weighted_loss = token_loss_unweighted * loss_scale_flat.abs()

        prompt_index = (labels == IGNORE_INDEX).to(torch.int64)
        noisy_data_length = torch.sum((1 - prompt_index), dim=-1, keepdim=True)
        noisy_data_length = torch.max(noisy_data_length, torch.ones_like(noisy_data_length))
        noisy_data_length_flat = noisy_data_length.expand_as(labels)[masked_positions]

        loss = torch.sum(weighted_loss / noisy_data_length_flat) / labels.shape[0]
        return loss

    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None,
        past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
        output_attentions=None, output_hidden_states=None, return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        bd3lm_inputs, move_indices, loss_scale, x0_embeds, avg_noise_level = \
            self._apply_bd3lm_noise_embedding(inputs_embeds, labels)

        from .bd3lm_utils import block_diff_mask

        seq_len = inputs_embeds.shape[1]
        device = inputs_embeds.device

        q_idx = torch.arange(seq_len * 2, device=device)[:, None]
        kv_idx = torch.arange(seq_len * 2, device=device)[None, :]

        mask = block_diff_mask(b=None, h=None, q_idx=q_idx, kv_idx=kv_idx,
                               block_size=self.model.bd3lm_block_size, n=seq_len)

        if attention_mask is not None and attention_mask.dim() == 2:
            extended = torch.cat([attention_mask, attention_mask], dim=1).bool()
            query_validity_mask = extended.unsqueeze(-1)
            key_validity_mask = extended.unsqueeze(-2)
            combined_padding_mask_2d = query_validity_mask & key_validity_mask
            mask = mask & combined_padding_mask_2d

        attention_mask_4d = torch.zeros(mask.shape, dtype=inputs_embeds.dtype, device=device)
        attention_mask_4d.masked_fill_(~mask, torch.finfo(inputs_embeds.dtype).min)
        attention_mask_4d = attention_mask_4d.unsqueeze(1)

        if position_ids is None:
            pos_ids_part = torch.arange(seq_len, device=device)
            position_ids = torch.cat([pos_ids_part, pos_ids_part], dim=0)

        # For the doubled sequence, build hybrid mask dict.
        # full_attention layers get the 4D block_diff_mask.
        # linear_attention layers get a simple padding mask (they are causal by recurrence).
        linear_mask = None
        if attention_mask is not None and attention_mask.dim() == 2:
            linear_mask = torch.cat([attention_mask, attention_mask], dim=1).bool()
        model_mask = {"full_attention": attention_mask_4d, "linear_attention": linear_mask}

        outputs = self.model(
            inputs_embeds=bd3lm_inputs,
            attention_mask=model_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = hidden_states[:, :inputs_embeds.shape[1]]

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            loss = self._compute_bd3lm_loss_embedding(logits, labels, move_indices, loss_scale)

        if self.training:
            if not hasattr(self, "_current_custom_metrics"):
                self._current_custom_metrics = {}
            self._current_custom_metrics["anneal/noise_level"] = avg_noise_level

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss, logits=logits,
            past_key_values=outputs.past_key_values,
        )

    @torch.no_grad()
    def generate_with_bd3lm(self, inputs_embeds, steps=4, gen_length=128, temperature=0.0, **kwargs):
        """BD3-LM inference with snapshot-restore for hybrid cache."""
        # Reuse the self-contained generate_with_bd3lm logic
        from .bd3lm_cache_utils import snapshot_linear_cache, restore_linear_cache, \
            snapshot_full_attn_cache_seq_len, crop_full_attn_cache

        device = inputs_embeds.device
        batch_size = inputs_embeds.shape[0]
        prompt_len = inputs_embeds.shape[1]
        block_size = self.model.bd3lm_block_size
        mask_id = self.model.mask_token_id

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

        if prefill_length > 0:
            prefill_embeds = x_embeds[:, :prefill_length]
            prefill_mask = block_diffusion_mask[:, :, :prefill_length, :prefill_length]
            model_mask = {"full_attention": prefill_mask, "linear_attention": None}
            prefill_pos_ids = position_ids[:, :prefill_length]

            self.model(inputs_embeds=prefill_embeds, attention_mask=model_mask,
                       position_ids=prefill_pos_ids, past_key_values=past_key_values,
                       use_cache=True, store_kv=True)

        num_transfer_tokens = self.get_bd3lm_num_transfer_tokens(block_size, steps)

        for block_idx in range(prefill_blocks, num_blocks):
            block_start = block_idx * block_size
            block_end = block_start + block_size

            cur_block_embeds = x_embeds[:, block_start:block_end].clone()
            cur_block_ids = x_ids[:, block_start:block_end]

            cur_mask = block_diffusion_mask[:, :, block_start:block_end, :block_end]
            cur_pos_ids = position_ids[:, block_start:block_end]
            model_mask = {"full_attention": cur_mask, "linear_attention": None}

            linear_snapshot = snapshot_linear_cache(past_key_values)
            full_attn_lengths = snapshot_full_attn_cache_seq_len(past_key_values)

            for step in range(steps + 1):
                restore_linear_cache(past_key_values, linear_snapshot)
                crop_full_attn_cache(past_key_values, full_attn_lengths)

                is_mask = torch.all(torch.abs(cur_block_embeds - mask_embed) < 1e-5, dim=-1)
                if not is_mask.any():
                    self.model(inputs_embeds=cur_block_embeds, attention_mask=model_mask,
                               position_ids=cur_pos_ids, past_key_values=past_key_values,
                               use_cache=True, store_kv=True)
                    break

                outputs = self.model(inputs_embeds=cur_block_embeds, attention_mask=model_mask,
                                     position_ids=cur_pos_ids, past_key_values=past_key_values,
                                     use_cache=True, store_kv=False)
                logits = self.lm_head(outputs[0]).float()

                top_k = kwargs.get("top_k", 0)
                top_p = kwargs.get("top_p", 1.0)
                x0, x0_p = self._sample_with_temperature_topk_topp(logits, temperature=temperature, top_k=top_k, top_p=top_p)
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
                    raise ValueError(f"Unknown remasking: {remasking_strategy}")

                cur_block_ids = torch.where(transfer_mask, x0, cur_block_ids)
                x0_embeds = self.get_input_embeddings()(x0)
                cur_block_embeds = torch.where(transfer_mask.unsqueeze(-1), x0_embeds, cur_block_embeds)

            x_embeds[:, block_start:block_end] = cur_block_embeds
            x_ids[:, block_start:block_end] = cur_block_ids

            if block_end > prompt_len:
                gen_start = max(prompt_len, block_start)
                gen_ids_check = x_ids[:, gen_start:block_end]
                eos_token_id = 248044
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

    def _sample_with_temperature_topk_topp(self, logits, temperature=1.0, top_k=0, top_p=1.0):
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

    def get_bd3lm_num_transfer_tokens(self, block_length, steps):
        if steps == 0:
            return torch.zeros(1, dtype=torch.int64)
        base = block_length // steps
        remainder = block_length % steps
        ntt = torch.zeros(steps + 1, dtype=torch.int64) + base
        ntt[:remainder] += 1
        return ntt


class DiffusionVLQwen3_5MultiModalModel(LlavaMetaModel, DiffusionVLQwen3_5TextModel):
    config_class = DiffusionVLQwen3_5Config

    def __init__(self, config):
        super(DiffusionVLQwen3_5MultiModalModel, self).__init__(config)


class DiffusionVLQwen3_5ForCausalLM(DiffusionVLQwen3_5ForCausalLM_Base, LlavaMetaForCausalLM):
    config_class = DiffusionVLQwen3_5Config

    def __init__(self, config):
        super(DiffusionVLQwen3_5ForCausalLM, self).__init__(config)
        self.model = DiffusionVLQwen3_5MultiModalModel(config)

    def get_model(self):
        return self.model

    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None,
        past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
        output_attentions=None, output_hidden_states=None,
        images=None, image_sizes=None, image_grid_thws=None, modalities=None,
        return_dict=None,
    ):
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values,
             inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels,
                images, modalities=modalities, image_sizes=image_sizes,
                image_grid_thws=image_grid_thws)

        return super(DiffusionVLQwen3_5ForCausalLM, self).forward(
            inputs_embeds=inputs_embeds, labels=labels,
            attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, return_dict=return_dict)

    @torch.no_grad()
    def generate(self, inputs=None, images=None, image_sizes=None, image_grid_thws=None,
                 modalities=None, **kwargs):
        if modalities is None:
            modalities = ["image"]
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)

        if images is not None:
            (_, _, _, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(
                inputs, position_ids, attention_mask, None, None,
                images, modalities=modalities, image_sizes=image_sizes,
                image_grid_thws=image_grid_thws)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        kwargs.pop("input_ids", None)
        return self.generate_with_bd3lm(inputs_embeds=inputs_embeds, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        print(f">>> Loading DiffusionVL-Qwen3.5 model from: {pretrained_model_name_or_path}")
        model = super(DiffusionVLQwen3_5ForCausalLM, cls).from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs)

        vision_config_path = os.path.join(pretrained_model_name_or_path, "vision_config.json")
        if os.path.exists(vision_config_path):
            print(">>> Loading visual components from .safetensors files...")
            with open(vision_config_path, "r") as f:
                vision_config_dict = json.load(f)
            model.config.vision_config = PretrainedConfig.from_dict(vision_config_dict)
            model.config.vision_tower_state_dict = load_file(
                os.path.join(pretrained_model_name_or_path, "vision_tower.safetensors"), device="cpu")
            model.config.projector_state_dict = load_file(
                os.path.join(pretrained_model_name_or_path, "projector.safetensors"), device="cpu") \
                if os.path.exists(os.path.join(pretrained_model_name_or_path, "projector.safetensors")) else None

        # Auto-build vision modules
        import types
        _torch_dtype = kwargs.get("torch_dtype", kwargs.get("dtype", None))
        _compute_dtype = torch.bfloat16 if _torch_dtype in (torch.bfloat16, "bfloat16") else torch.float16
        model_args = types.SimpleNamespace(
            vision_tower=pretrained_model_name_or_path,
            mm_vision_select_layer=getattr(model.config, "mm_vision_select_layer", -2),
            mm_vision_select_feature=getattr(model.config, "mm_vision_select_feature", "patch"),
            pretrain_mm_mlp_adapter=None,
            mm_patch_merge_type=getattr(model.config, "mm_patch_merge_type", "flat"),
            mm_projector_type=getattr(model.config, "mm_projector_type", "qwen3_5_merger"),
            add_faster_video=getattr(model.config, "add_faster_video", False),
            vision_tower_pretrained=getattr(model.config, "vision_tower_pretrained", ""),
        )
        model.get_model().initialize_vision_modules(model_args=model_args)
        _vt = model.get_vision_tower()
        if _vt is not None:
            _vt.to(dtype=_compute_dtype, device=next(model.parameters()).device)
        if model.get_model().mm_projector is not None:
            model.get_model().mm_projector.to(dtype=_compute_dtype, device=next(model.parameters()).device)

        print(">>> DiffusionVL-Qwen3.5 model loaded successfully.")
        return model


AutoConfig.register("diffusionvl_qwen3_5", DiffusionVLQwen3_5Config)
AutoModelForCausalLM.register(DiffusionVLQwen3_5Config, DiffusionVLQwen3_5ForCausalLM)
