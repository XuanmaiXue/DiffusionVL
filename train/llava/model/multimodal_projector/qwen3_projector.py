# coding=utf-8
# Copyright 2025 The HustVL Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on Qwen3-VL (https://github.com/QwenLM/Qwen3-VL),
# LLaDA-V (https://github.com/ML-GSAI/LLaDA-V), and
# Block Diffusion (https://github.com/kuleshov-group/bd3lm). It has been modified to create DiffusionVL.
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

"""Qwen3-VL projector for DiffusionVL.

Unlike the Qwen2.5-VL projector, the Qwen3-VL main patch merger is constructed
from a `Qwen3VLVisionConfig` directly, and Qwen3-VL has no window shuffle to
reverse. The forward pass additionally threads out the DeepStack visual features
(one tensor per deepstack visual index) so they can be injected into the early
decoder layers of the language model by the caller (see
`llava_arch.prepare_inputs_labels_for_multimodal`).
"""

import torch
from torch import nn
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchMerger


class LlavaQwen3Projector(nn.Module):
    """Projector wrapping the Qwen3-VL main patch merger.

    The vision tower returns `(hidden_states, deepstack_features)`.
    This projector applies the main merger to `hidden_states` and passes the
    DeepStack features through unchanged (they are already produced by
    `Qwen3VLVisionPatchMerger(use_postshuffle_norm=True)` inside the tower, so
    they are already at the LM hidden size).
    """

    def __init__(self, vision_config):
        super().__init__()
        # The main merger uses postshuffle_norm=False, matching Qwen3VLVisionModel.
        self.merger = Qwen3VLVisionPatchMerger(
            config=vision_config,
            use_postshuffle_norm=False,
        )

    def forward(self, features_tuple):
        """Project vision-tower output into LM embedding space.

        Args:
            features_tuple: `(hidden_states, deepstack_features)`
                - hidden_states: (seq_len, hidden_size) pre-merger features at
                  patch granularity, already in the (merged_h, merge_size,
                  merged_w, merge_size) order required by the merger.
                - deepstack_features: list of (visual_seq_len, out_hidden_size)
                  tensors, one per deepstack visual index, already merged.
        Returns:
            (final_features, deepstack_features) where final_features is
            (visual_seq_len, out_hidden_size), the merged visual embeddings.
        """
        hidden_states, deepstack_features = features_tuple

        # Merge patches (spatial_merge_size**2 -> 1) into LM hidden size.
        # Qwen3-VL has no window shuffle, so no un-shuffle is needed.
        final_features = self.merger(hidden_states)

        return final_features, deepstack_features
