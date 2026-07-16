# coding=utf-8
# Copyright 2025 The HustVL Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on Qwen3.5 (hybrid Gated DeltaNet + full attention).
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

"""Qwen3.5 projector for DiffusionVL.

Qwen3.5 has NO DeepStack, so the projector simply wraps the main patch merger.
"""

import torch
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionPatchMerger


class LlavaQwen3_5Projector(nn.Module):
    """Projector wrapping the Qwen3.5 main patch merger."""

    def __init__(self, vision_config):
        super().__init__()
        self.merger = Qwen3_5VisionPatchMerger(
            config=vision_config,
            use_postshuffle_norm=False,
        )

    def forward(self, hidden_states):
        """Merge patches into LM embedding space."""
        return self.merger(hidden_states)
