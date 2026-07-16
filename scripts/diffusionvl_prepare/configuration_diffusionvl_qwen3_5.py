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
"""DiffusionVL-Qwen3.5 configuration."""

from transformers.configuration_utils import PretrainedConfig


class DiffusionVLQwen3_5Config(PretrainedConfig):
    """Configuration for DiffusionVL-Qwen3.5.

    Qwen3.5 uses a hybrid architecture: ~75% Gated DeltaNet (linear attention)
    layers + ~25% full softmax attention layers. This config stores the
    layer_types array that controls per-layer dispatch.
    """

    model_type = "diffusionvl_qwen3_5"
    is_composition = True
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
        enable_bd3lm=True,
        bd3lm_block_size=8,
        mask_token_id=None,
        tie_word_embeddings=True,
        **kwargs,
    ):
        # Strip nested text_config so it doesn't leak into GenerationConfig.
        kwargs.pop("text_config", None)

        # Qwen3.5 checkpoints don't define pad_token_id; use eos as pad.
        if "pad_token_id" not in kwargs:
            kwargs["pad_token_id"] = kwargs.get("eos_token_id", 248044)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        if vision_config is None:
            vision_config = {}
        if text_config is None:
            text_config = {}

        # Store as PretrainedConfig objects for GenerationConfig compatibility.
        if isinstance(vision_config, dict):
            vision_config = PretrainedConfig.from_dict(vision_config)
        if isinstance(text_config, dict):
            text_config = PretrainedConfig.from_dict(text_config)
        self.vision_config = vision_config
        self.text_config = text_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

        # BD3-LM inference parameters.
        self.enable_bd3lm = enable_bd3lm
        self.bd3lm_block_size = bd3lm_block_size
        # Qwen3.5 vocab is 248320; use the last token as mask token if not set.
        self.mask_token_id = mask_token_id if mask_token_id is not None else 248319

        # Flatten text_config fields onto self so modeling code can access
        # hidden_size / num_hidden_layers / layer_types etc. directly.
        text_config_dict = self.text_config.to_dict() if hasattr(self.text_config, "to_dict") else self.text_config
        for key, value in text_config_dict.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self):
        output = super().to_dict()
        output["vision_config"] = self.vision_config.to_dict() if hasattr(self.vision_config, "to_dict") else self.vision_config
        output["text_config"] = self.text_config.to_dict() if hasattr(self.text_config, "to_dict") else self.text_config
        return output


__all__ = ["DiffusionVLQwen3_5Config"]
