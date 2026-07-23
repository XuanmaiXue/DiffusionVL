#!/usr/bin/env python3
"""Prepare a trained DiffusionVL checkpoint for HF trust_remote_code inference.

Copies the three self-contained .py files into the checkpoint directory and
adds the required `auto_map` (and bd3lm inference fields if missing) to
config.json. Supports both Qwen3-VL and Qwen3.5 checkpoints.

Usage:
    # Qwen3-VL checkpoint:
    python add_self_contained_to_ckpt.py --ckpt_path /path/to/checkpoint --variant qwen3vl

    # Qwen3.5 checkpoint:
    python add_self_contained_to_ckpt.py --ckpt_path /path/to/checkpoint --variant qwen3_5
"""

import argparse
import json
import os
import shutil


VARIANTS = {
    "qwen3vl": {
        "files": [
            "configuration_diffusionvl_qwen3_vl.py",
            "modeling_diffusionvl_qwen3_vl.py",
            "processing_diffusionvl_qwen3_vl.py",
        ],
        "auto_map": {
            "AutoConfig": "configuration_diffusionvl_qwen3_vl.DiffusionVLQwen3VLConfig",
            "AutoModelForCausalLM": "modeling_diffusionvl_qwen3_vl.DiffusionVLQwen3VLForConditionalGeneration",
            "AutoProcessor": "processing_diffusionvl_qwen3_vl.DiffusionVLQwen3VLProcessor",
        },
        "model_type": "diffusionvl_qwen3vl",
        "mask_token_id": 151671,
    },
    "qwen3_5": {
        "files": [
            "configuration_diffusionvl_qwen3_5.py",
            "modeling_diffusionvl_qwen3_5.py",
            "processing_diffusionvl_qwen3_5.py",
        ],
        "auto_map": {
            "AutoConfig": "configuration_diffusionvl_qwen3_5.DiffusionVLQwen3_5Config",
            "AutoModelForCausalLM": "modeling_diffusionvl_qwen3_5.DiffusionVLQwen3_5ForConditionalGeneration",
            "AutoProcessor": "processing_diffusionvl_qwen3_5.DiffusionVLQwen3_5Processor",
        },
        "model_type": "diffusionvl_qwen3_5",
        "mask_token_id": 248319,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Add self-contained inference files to a DiffusionVL checkpoint.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the trained checkpoint directory.")
    parser.add_argument("--variant", type=str, default=None, choices=["qwen3vl", "qwen3_5"],
                        help="Model variant. If not specified, auto-detect from config.json model_type.")
    parser.add_argument("--src_dir", type=str, default=None,
                        help="Directory containing the self-contained .py files (default: same dir as this script).")
    args = parser.parse_args()

    ckpt_path = args.ckpt_path
    if not os.path.isdir(ckpt_path):
        raise NotADirectoryError(f"Checkpoint directory not found: {ckpt_path}")

    src_dir = args.src_dir or os.path.dirname(os.path.abspath(__file__))

    # Auto-detect variant from config.json if not specified
    variant_key = args.variant
    if variant_key is None:
        config_path = os.path.join(ckpt_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            mt = cfg.get("model_type", "")
            if "qwen3_5" in mt:
                variant_key = "qwen3_5"
            elif "qwen3vl" in mt or "qwen3_vl" in mt:
                variant_key = "qwen3vl"
            elif "qwenvl" in mt:
                variant_key = "qwen3vl"
        if variant_key is None:
            variant_key = "qwen3vl"
        print(f"Auto-detected variant: {variant_key}")

    variant = VARIANTS[variant_key]

    # --- Step 1: Copy the three self-contained .py files ---
    print("Copying self-contained files...")
    for fname in variant["files"]:
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

    cfg["auto_map"] = variant["auto_map"]
    cfg.setdefault("enable_bd3lm", True)
    cfg.setdefault("bd3lm_block_size", 8)
    cfg.setdefault("mask_token_id", variant["mask_token_id"])
    cfg.setdefault("model_type", variant["model_type"])

    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Updated config.json: auto_map + bd3lm fields ({variant_key})")

    # --- Step 2b: Fix preprocessor_config.json ---
    preprocessor_cfg_path = os.path.join(ckpt_path, "preprocessor_config.json")
    if os.path.exists(preprocessor_cfg_path):
        with open(preprocessor_cfg_path, "r") as f:
            pp_cfg = json.load(f)
        if "processor_class" in pp_cfg:
            del pp_cfg["processor_class"]
            with open(preprocessor_cfg_path, "w") as f:
                json.dump(pp_cfg, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print("Removed processor_class from preprocessor_config.json")

    video_pp_path = os.path.join(ckpt_path, "video_preprocessor_config.json")
    if os.path.exists(video_pp_path):
        os.remove(video_pp_path)
        print("Removed video_preprocessor_config.json")

    # --- Step 3: Check for preprocessor_config.json ---
    if not os.path.exists(preprocessor_cfg_path):
        print()
        print("WARNING: preprocessor_config.json not found in checkpoint!")

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
