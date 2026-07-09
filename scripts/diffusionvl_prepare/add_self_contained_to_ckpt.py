#!/usr/bin/env python3
"""Prepare a trained DiffusionVL-Qwen3VL checkpoint for HF trust_remote_code inference.

Copies the three self-contained .py files into the checkpoint directory and
adds the required `auto_map` (and bd3lm inference fields if missing) to
config.json. After running this, the checkpoint can be loaded with:

    from transformers import AutoModelForCausalLM, AutoProcessor
    model = AutoModelForCausalLM.from_pretrained("ckpt", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("ckpt", trust_remote_code=True)

Usage:
    python add_self_contained_to_ckpt.py --ckpt_path /path/to/checkpoint
"""

import argparse
import json
import os
import shutil


SELF_CONTAINED_FILES = [
    "configuration_diffusionvl_qwen3_vl.py",
    "modeling_diffusionvl_qwen3_vl.py",
    "processing_diffusionvl_qwen3_vl.py",
]

AUTO_MAP = {
    "AutoConfig": "configuration_diffusionvl_qwen3_vl.DiffusionVLQwen3VLConfig",
    "AutoModelForCausalLM": "modeling_diffusionvl_qwen3_vl.DiffusionVLQwen3VLForConditionalGeneration",
    "AutoProcessor": "processing_diffusionvl_qwen3_vl.DiffusionVLQwen3VLProcessor",
}


def main():
    parser = argparse.ArgumentParser(description="Add self-contained inference files to a DiffusionVL-Qwen3VL checkpoint.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the trained checkpoint directory.")
    parser.add_argument("--src_dir", type=str, default=None,
                        help="Directory containing the self-contained .py files (default: same dir as this script).")
    args = parser.parse_args()

    ckpt_path = args.ckpt_path
    if not os.path.isdir(ckpt_path):
        raise NotADirectoryError(f"Checkpoint directory not found: {ckpt_path}")

    src_dir = args.src_dir or os.path.dirname(os.path.abspath(__file__))

    # --- Step 1: Copy the three self-contained .py files ---
    print("Copying self-contained files...")
    for fname in SELF_CONTAINED_FILES:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(ckpt_path, fname)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Source file not found: {src}")
        shutil.copy2(src, dst)
        print(f"  - {fname}")

    # --- Step 2: Update config.json with auto_map + bd3lm fields ---
    config_path = os.path.join(ckpt_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {ckpt_path}")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    cfg["auto_map"] = AUTO_MAP

    # Ensure bd3lm inference fields are present (training checkpoints usually
    # have them, but add defaults just in case).
    cfg.setdefault("enable_bd3lm", True)
    cfg.setdefault("bd3lm_block_size", 8)
    cfg.setdefault("mask_token_id", 151671)

    # Ensure model_type matches (training checkpoints should already have it).
    cfg.setdefault("model_type", "diffusionvl_qwen3vl")

    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Updated config.json: auto_map + bd3lm fields")

    # --- Step 3: Check for preprocessor_config.json ---
    preprocessor_path = os.path.join(ckpt_path, "preprocessor_config.json")
    if not os.path.exists(preprocessor_path):
        print()
        print("WARNING: preprocessor_config.json not found in checkpoint!")
        print("  The image processor is needed for inference. Copy it from the")
        print("  original Qwen3-VL checkpoint:")
        print(f"    cp /path/to/Qwen3-VL-4B-Instruct/preprocessor_config.json {ckpt_path}/")

    print()
    print(f"Done! The checkpoint at {ckpt_path} is now loadable via trust_remote_code.")
    print()
    print("Example inference:")
    print("```python")
    print("from transformers import AutoModelForCausalLM, AutoProcessor")
    print(f'model = AutoModelForCausalLM.from_pretrained("{ckpt_path}", trust_remote_code=True, torch_dtype="bfloat16")')
    print(f'processor = AutoProcessor.from_pretrained("{ckpt_path}", trust_remote_code=True)')
    print("```")


if __name__ == "__main__":
    main()
