"""Inspect actual state_dict keys of the training-side model class."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "train"))

import torch
# Use a tiny config to avoid loading real weights; we just need key structure.
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig, Qwen2_5_VLVisionConfig
from llava.model.language_model.llava_diffusionvl_qwenvl import DiffusionVLQwenVLConfig, DiffusionVLQwenVLForCausalLM

# Minimal config to instantiate a tiny model (just to read key names)
vc = Qwen2_5_VLVisionConfig(depth=2, hidden_size=16, num_heads=2, out_hidden_size=32, spatial_merge_size=2, in_chans=3)
tc = Qwen2_5_VLTextConfig(vocab_size=32, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                          num_key_value_heads=2, intermediate_size=64, max_position_embeddings=64)
cfg = DiffusionVLQwenVLConfig(vision_config=vc.to_dict(), text_config=tc.to_dict(),
                              image_token_id=10, video_token_id=11, enable_bd3lm=True, bd3lm_block_size=4)

model = DiffusionVLQwenVLForCausalLM(cfg)

# Trigger vision module init so vision_tower / mm_projector appear in state_dict
import types as _types
model_args = _types.SimpleNamespace(
    vision_tower="dummy-qwen",  # name triggers LlavaQwenVisionTower
    mm_vision_select_layer=-2,
    mm_vision_select_feature="patch",
    pretrain_mm_mlp_adapter=None,
    mm_patch_merge_type="flat",
    mm_projector_type="qwen_merger",
    add_faster_video=False,
    vision_tower_pretrained="",
)
try:
    model.get_model().initialize_vision_modules(model_args=model_args)
except Exception as e:
    print("INIT NOTE (expected in tiny model):", type(e).__name__, str(e)[:120])

sd = model.state_dict()
keys = list(sd.keys())
print("total keys:", len(keys))

from collections import Counter
prefixes = Counter()
for k in keys:
    prefixes[".".join(k.split(".")[:3])] += 1
print("=== top-3-level prefix counts ===")
for p, c in prefixes.most_common():
    print(f"  {p:50s} {c}")

print()
print("=== vision_tower keys (sample) ===")
for k in keys:
    if "vision_tower" in k:
        print(" ", k)
print()
print("=== mm_projector keys ===")
for k in keys:
    if "mm_projector" in k:
        print(" ", k)
print()
print("=== lm_head / embed / norm / layers.0 ===")
for k in keys:
    if k.startswith("lm_head") or k.startswith("model.embed") or k.startswith("model.norm") or k.startswith("model.layers.0."):
        print(" ", k)
