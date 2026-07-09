# coding=utf-8
# Licensed under the Apache License, Version 2.0.
"""End-to-end self-test for convert_diffusionvl_to_hf_release.py.

Builds a *tiny* fake training checkpoint that mimics the real
DiffusionVLQwenVLForCausalLM state-dict layout (LM layers + vision tower +
projector + tied lm_head), runs the packaging script on it, and asserts that
the produced release directory is internally consistent:

  - every source tensor (minus a tied lm_head) is present with the SAME key;
  - shard files exist and their union of keys == index.weight_map keys;
  - config.json has architectures / auto_map / tie_word_embeddings set and
    local-path fields scrubbed;
  - tokenizer/preprocessor files are copied through;
  - re-loading the shards reproduces every tensor byte-for-byte.

This does NOT exercise trust_remote_code loading (that needs the real HF
remote-code files, which aren't in this repo). It validates the packaging
contract that those files rely on.
"""

import json
import os
import shutil
import sys
import tempfile

import torch
from safetensors.torch import save_file, load_file

# Make the converter importable.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from convert_diffusionvl_to_hf_release import (
    convert,
    _REMOTE_CODE_FILES,
    _NON_WEIGHT_FILES_TO_COPY,
)


def _build_fake_training_checkpoint(dst, tie_lm_head_duplicate=False):
    """Write a minimal checkpoint in the training-side layout."""
    os.makedirs(dst, exist_ok=True)

    state_dict = {}
    # 2 LLM layers, tiny dims.
    for i in range(2):
        state_dict[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.randn(8, 8)
        state_dict[f"model.layers.{i}.self_attn.k_proj.weight"] = torch.randn(8, 8)
        state_dict[f"model.layers.{i}.self_attn.v_proj.weight"] = torch.randn(8, 8)
        state_dict[f"model.layers.{i}.self_attn.o_proj.weight"] = torch.randn(8, 8)
        state_dict[f"model.layers.{i}.mlp.gate_proj.weight"] = torch.randn(16, 8)
        state_dict[f"model.layers.{i}.mlp.up_proj.weight"] = torch.randn(16, 8)
        state_dict[f"model.layers.{i}.mlp.down_proj.weight"] = torch.randn(8, 16)
        state_dict[f"model.layers.{i}.input_layernorm.weight"] = torch.randn(8)
        state_dict[f"model.layers.{i}.post_attention_layernorm.weight"] = torch.randn(8)
    state_dict["model.embed_tokens.weight"] = torch.randn(32, 8)
    state_dict["model.norm.weight"] = torch.randn(8)

    # Vision tower (2 ViT blocks + patch_embed).
    for i in range(2):
        state_dict[f"model.vision_tower.vision_tower.blocks.{i}.attn.qkv.weight"] = torch.randn(16, 4)
        state_dict[f"model.vision_tower.vision_tower.blocks.{i}.attn.proj.weight"] = torch.randn(4, 4)
        state_dict[f"model.vision_tower.vision_tower.blocks.{i}.norm1.weight"] = torch.randn(4)
        state_dict[f"model.vision_tower.vision_tower.blocks.{i}.norm2.weight"] = torch.randn(4)
    state_dict["model.vision_tower.vision_tower.patch_embed.proj.weight"] = torch.randn(4, 3, 2, 2)

    # Projector (Qwen2.5-VL PatchMerger: ln_q + 2-layer MLP).
    state_dict["model.mm_projector.merger.ln_q.weight"] = torch.randn(4)
    state_dict["model.mm_projector.merger.mlp.0.weight"] = torch.randn(16, 16)
    state_dict["model.mm_projector.merger.mlp.0.bias"] = torch.randn(16)
    state_dict["model.mm_projector.merger.mlp.2.weight"] = torch.randn(8, 16)
    state_dict["model.mm_projector.merger.mlp.2.bias"] = torch.randn(8)

    # Tied lm_head: embed_tokens.weight is the canonical copy. A well-behaved
    # trainer omits lm_head.weight, but we optionally inject a duplicate to
    # exercise the de-dup path.
    if tie_lm_head_duplicate:
        # Same storage -> safetensors would refuse; use a clone to simulate
        # an accidental separate-but-equal copy.
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"].clone()

    save_file(state_dict, os.path.join(dst, "model.safetensors"))

    # Training-style config, including the fields that must be scrubbed.
    config = {
        "model_type": "diffusionvl_qwenvl",
        "architectures": ["DiffusionVLQwenVLForCausalLM"],  # training name, must be rewritten
        "tie_word_embeddings": True,
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "vocab_size": 32,
        "mm_vision_tower": "/absolute/local/path/Qwen2.5-VL-3B",   # must be scrubbed
        "vision_tower_pretrained": "/another/local/path",           # must be scrubbed
        "_name_or_path": "Qwen/Qwen2.5-VL-3B-Instruct",            # must be scrubbed
        "vision_config": {"depth": 2, "hidden_size": 4},
        "text_config": {"hidden_size": 8, "num_hidden_layers": 2},
        "bd3lm_block_size": 4,
        "enable_bd3lm": True,
    }
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Minimal tokenizer / processor files so the copy path is exercised.
    for fname in ["tokenizer_config.json", "preprocessor_config.json"]:
        with open(os.path.join(dst, fname), "w") as f:
            json.dump({"_test": fname}, f)
    return state_dict


def _load_release_weights(dest):
    """Re-read every shard in the release dir into one dict."""
    index_path = os.path.join(dest, "model.safetensors.index.json")
    with open(index_path) as f:
        idx = json.load(f)
    shards = sorted(set(idx["weight_map"].values()))
    out = {}
    for s in shards:
        out.update(load_file(os.path.join(dest, s)))
    return out, idx


def _make_fake_remote_code(dst):
    """Write dummy remote-code files so the copy path succeeds."""
    os.makedirs(dst, exist_ok=True)
    for fname in _REMOTE_CODE_FILES:
        with open(os.path.join(dst, fname), "w") as f:
            f.write(f"# stub {fname}\n")


def run():
    tmp = tempfile.mkdtemp(prefix="diffusionvl_release_test_")
    try:
        src = os.path.join(tmp, "train_ckpt")
        rcs = os.path.join(tmp, "remote_code")
        dst = os.path.join(tmp, "release")

        # --- Case 1: clean checkpoint (no duplicate lm_head) ---
        print("=== Case 1: clean training checkpoint ===")
        src_sd = _build_fake_training_checkpoint(src, tie_lm_head_duplicate=False)
        _make_fake_remote_code(rcs)

        convert(
            src_checkpoint_dir=src,
            dest_dir=dst,
            remote_code_source=rcs,
            max_shard_size_bytes=1024,  # tiny -> forces multiple shards
        )

        # Assert: weights preserved, key-for-key.
        release_sd, idx = _load_release_weights(dst)
        assert len(release_sd) == len(src_sd), (
            f"tensor count mismatch: src={len(src_sd)} release={len(release_sd)}"
        )
        for k, v in src_sd.items():
            assert k in release_sd, f"missing key in release: {k}"
            assert torch.equal(release_sd[k], v), f"tensor value changed: {k}"
        print("  [OK] all source tensors present and byte-identical in release")

        # Assert: index weight_map matches actual shard keys.
        shard_keys = set()
        for s in sorted(set(idx["weight_map"].values())):
            assert os.path.exists(os.path.join(dst, s)), f"shard file missing: {s}"
            shard_keys.update(load_file(os.path.join(dst, s)).keys())
        assert shard_keys == set(idx["weight_map"].keys()), "index weight_map != shard keys"
        print("  [OK] index.weight_map matches shard contents")

        # Assert: config rewritten.
        with open(os.path.join(dst, "config.json")) as f:
            cfg = json.load(f)
        assert cfg["architectures"] == ["DiffusionVL_Qwen2_5_VL_ForConditionalGeneration"]
        assert "AutoConfig" in cfg["auto_map"]
        assert cfg["tie_word_embeddings"] is True
        assert cfg["model_type"] == "diffusionvl_qwenvl"
        assert "mm_vision_tower" not in cfg, "local path leaked into release config"
        assert "vision_tower_pretrained" not in cfg
        assert "_name_or_path" not in cfg
        assert "vision_config" in cfg and "bd3lm_block_size" in cfg, "legit config fields dropped"
        print("  [OK] config.json rewritten (architectures/auto_map set, paths scrubbed)")

        # Assert: tokenizer/preprocessor copied.
        for fname in ["tokenizer_config.json", "preprocessor_config.json"]:
            assert os.path.exists(os.path.join(dst, fname)), f"{fname} not copied"
        print("  [OK] tokenizer/preprocessor files copied")

        # Assert: remote-code files copied.
        for fname in _REMOTE_CODE_FILES:
            assert os.path.exists(os.path.join(dst, fname)), f"{fname} not copied"
        print("  [OK] remote-code files copied")

        # Assert: more than one shard (tiny max_shard_size forces splitting).
        assert len(set(idx["weight_map"].values())) > 1, "expected multiple shards"
        print(f"  [OK] sharded into {len(set(idx['weight_map'].values()))} files")

        # --- Case 2: checkpoint with a duplicate tied lm_head ---
        print("\n=== Case 2: training checkpoint with duplicate lm_head.weight ===")
        shutil.rmtree(dst)
        src_sd2 = _build_fake_training_checkpoint(src, tie_lm_head_duplicate=True)
        convert(src_checkpoint_dir=src, dest_dir=dst, remote_code_source=rcs,
                max_shard_size_bytes=1024)
        release_sd2, _ = _load_release_weights(dst)
        assert "lm_head.weight" not in release_sd2, "tied lm_head should be dropped"
        assert torch.equal(
            release_sd2["model.embed_tokens.weight"], src_sd2["model.embed_tokens.weight"]
        ), "embed_tokens corrupted after dropping tied lm_head"
        print("  [OK] duplicate tied lm_head.weight dropped, embed_tokens intact")

        # --- Case 3: stray key should fail loudly ---
        print("\n=== Case 3: stray unrecognized key raises ===")
        shutil.rmtree(dst)
        bad_sd = dict(src_sd2)
        bad_sd["something.unexpected.weight"] = torch.randn(4)
        save_file(bad_sd, os.path.join(src, "model.safetensors"))
        try:
            convert(src_checkpoint_dir=src, dest_dir=dst, remote_code_source=rcs)
            print("  [FAIL] expected ValueError for stray key, got none")
            return 1
        except ValueError as e:
            assert "Unexpected weight keys" in str(e)
            print("  [OK] stray key correctly rejected")

        print("\nAll checks passed.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(run())
