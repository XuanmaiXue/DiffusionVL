import argparse
import json
import os
import shutil

from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import snapshot_download

# NOTE on scope:
# This script only performs a *weight-format* conversion from the upstream
# Qwen3-VL layout to the DiffusionVL checkpoint layout (separate
# model / vision_tower / projector / deepstack_merger safetensors files).
# Qwen3-VL introduces several visual-encoder changes over Qwen2.5-VL:
#   - DeepStack multi-layer feature fusion (deepstack_merger_list)
#   - a learned positional embedding (pos_embed)
#   - interleaved MRope (mrope_interleaved)
#   - QK-norm in the language-model attention (self_attn.q/k_norm)
# These weights are preserved faithfully, but actually loading and running
# them inside DiffusionVL additionally requires:
#   1. upgrading `transformers` to a version that ships
#      `Qwen3VLForConditionalGeneration`, and
#   2. adapting the DiffusionVL visual code (llava_diffusionvl_qwenvl.py,
#      qwen_vision_tower.py, qwen_projector.py) to the Qwen3-VL visual classes.
# This script makes no code-side assumptions, so the produced checkpoint can
# serve as the faithful source of truth for that follow-up work.


def convert_qwen3vl_to_diffusionvl(source_path, dest_path):
    """
    Converts a Hugging Face Qwen3-VL checkpoint to the DiffusionVL checkpoint
    layout by remapping tensor keys via pure prefix substitution.

    Unlike the Qwen2.5-VL converter, we avoid instantiating
    `Qwen3VLForConditionalGeneration` (not available in older `transformers`
    releases and memory-heavy). Instead we read the safetensors directly and
    route each tensor into one of four output files based on its key prefix,
    which is sufficient because the upstream key naming is already regular.
    """
    # --- Resolve the source path (local dir or HF Hub repo id) ---
    if os.path.isdir(source_path):
        print(f"Source path '{source_path}' is a local directory. Using it directly.")
        source_local_path = source_path
    else:
        print(f"Source path '{source_path}' not found locally. Assuming it's a Hugging Face Hub repo id and attempting to download.")
        try:
            source_local_path = snapshot_download(source_path)
            print(f"Model successfully downloaded to: {source_local_path}")
        except Exception as e:
            print(f"Error downloading from Hugging Face Hub: {e}")
            print("Please ensure the source_path is either a valid local directory or a correct Hub repo id.")
            return

    os.makedirs(dest_path, exist_ok=True)

    # --- Copy all non-weight configuration / tokenizer files first ---
    print("Copying all configuration and tokenizer files...")
    files_to_ignore = [".git", ".gitattributes"]
    weight_extensions = [".safetensors", ".bin", ".pth"]
    for filename in os.listdir(source_local_path):
        if filename in files_to_ignore:
            continue
        # Skip weight files and the sharded-safetensors index; we create our own.
        if any(filename.endswith(ext) for ext in weight_extensions):
            continue
        if filename == "model.safetensors.index.json" or filename.endswith(".index.json"):
            continue

        src_file = os.path.join(source_local_path, filename)
        if os.path.isfile(src_file):
            if filename == "config.json":
                # Rewrite model_type / architectures so the DiffusionVL config
                # class and model class are selected at load time. All other
                # fields (vision_config, text_config, token ids, etc.) are kept.
                with open(src_file, "r") as fp:
                    cfg = json.load(fp)
                cfg["model_type"] = "diffusionvl_qwen3vl"
                cfg["architectures"] = ["DiffusionVLQwen3VLForCausalLM"]
                with open(os.path.join(dest_path, "config.json"), "w") as fp:
                    json.dump(cfg, fp, indent=2, ensure_ascii=False)
                    fp.write("\n")
                print("  - Writing config.json (model_type -> diffusionvl_qwen3vl)")
            else:
                print(f"  - Copying {filename}")
                shutil.copy2(src_file, dest_path)

    # --- Route tensors into four buckets by key prefix ---
    # Upstream Qwen3-VL tensor naming (all under a top-level `model.`):
    #   model.language_model.*                       -> language backbone
    #   model.visual.merger.*                        -> main patch merger (projector)
    #   model.visual.deepstack_merger_list.{0,1,2}.* -> DeepStack mergers
    #   model.visual.* (the rest)                    -> vision tower backbone
    print("\nRouting tensors from source safetensors into DiffusionVL buckets...")
    safetensors_files = sorted(
        f for f in os.listdir(source_local_path) if f.endswith(".safetensors")
    )
    if not safetensors_files:
        print("ERROR: No .safetensors weight files found in the source directory.")
        return

    main_state_dict = {}          # -> model.safetensors
    vision_tower_state_dict = {}  # -> vision_tower.safetensors
    projector_state_dict = {}     # -> projector.safetensors
    deepstack_state_dict = {}     # -> deepstack_merger.safetensors

    LM_PREFIX = "model.language_model."
    MERGER_PREFIX = "model.visual.merger."
    DEEPSTACK_PREFIX = "model.visual.deepstack_merger_list."
    VISUAL_PREFIX = "model.visual."

    for st_file in safetensors_files:
        st_path = os.path.join(source_local_path, st_file)
        print(f"  - Reading {st_file}")
        with safe_open(st_path, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if key.startswith(LM_PREFIX):
                    # Strip "model.language_model." -> "model."
                    new_key = "model." + key[len(LM_PREFIX):]
                    main_state_dict[new_key] = tensor
                elif key.startswith(MERGER_PREFIX):
                    # Strip "model.visual.merger."
                    new_key = key[len(MERGER_PREFIX):]
                    projector_state_dict[new_key] = tensor
                elif key.startswith(DEEPSTACK_PREFIX):
                    # Keep the deepstack_merger_list.N. structure, strip only "model.visual."
                    new_key = key[len(VISUAL_PREFIX):]
                    deepstack_state_dict[new_key] = tensor
                elif key.startswith(VISUAL_PREFIX):
                    # Remaining visual backbone (blocks.*, patch_embed.*, pos_embed.*)
                    new_key = key[len(VISUAL_PREFIX):]
                    vision_tower_state_dict[new_key] = tensor
                else:
                    # Unrecognized key; surface it rather than silently dropping.
                    print(f"  WARNING: unclassified tensor (skipped): {key}")

    # --- Sanity report ---
    total = (
        len(main_state_dict)
        + len(vision_tower_state_dict)
        + len(projector_state_dict)
        + len(deepstack_state_dict)
    )
    print("\nRouting summary:")
    print(f"  model.safetensors             (language_model): {len(main_state_dict)} tensors")
    print(f"  vision_tower.safetensors      (visual backbone): {len(vision_tower_state_dict)} tensors")
    print(f"  projector.safetensors         (main merger)    : {len(projector_state_dict)} tensors")
    print(f"  deepstack_merger.safetensors  (DeepStack)      : {len(deepstack_state_dict)} tensors")
    print(f"  total routed tensors                            : {total}")

    # --- Save the four weight files ---
    print("\nSaving converted weights...")
    save_file(main_state_dict, os.path.join(dest_path, "model.safetensors"))
    save_file(vision_tower_state_dict, os.path.join(dest_path, "vision_tower.safetensors"))
    save_file(projector_state_dict, os.path.join(dest_path, "projector.safetensors"))
    if deepstack_state_dict:
        save_file(deepstack_state_dict, os.path.join(dest_path, "deepstack_merger.safetensors"))

    # --- Export the vision config separately (matches the Qwen2.5-VL converter) ---
    config_path = os.path.join(source_local_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as fp:
            full_config = json.load(fp)
        vision_config = full_config.get("vision_config")
        if vision_config is not None:
            with open(os.path.join(dest_path, "vision_config.json"), "w") as fp:
                json.dump(vision_config, fp, indent=2)
                fp.write("\n")
            print("Saved vision_config.json")

    # Free memory before returning
    del main_state_dict, vision_tower_state_dict, projector_state_dict, deepstack_state_dict

    print(f"\nConversion successful! DiffusionVL-Qwen3VL checkpoint saved at: {dest_path}")
    print("NOTE: Only the weight layout was converted. To actually load/run the")
    print("      Qwen3-VL visual encoder (DeepStack, pos_embed, mrope_interleaved)")
    print("      inside DiffusionVL, you still need to (1) upgrade transformers to a")
    print("      release with Qwen3VLForConditionalGeneration and (2) adapt the")
    print("      DiffusionVL visual code (qwen_vision_tower.py / qwen_projector.py /")
    print("      llava_diffusionvl_qwenvl.py) to the Qwen3-VL visual classes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a Hugging Face Qwen3-VL model to the DiffusionVL checkpoint format."
    )
    parser.add_argument(
        "--source_path",
        type=str,
        required=True,
        help="Path or name of the original Qwen3-VL model (e.g., 'Qwen/Qwen3-VL-2B-Instruct').",
    )
    parser.add_argument(
        "--dest_path",
        type=str,
        required=True,
        help="Path to save the converted DiffusionVL checkpoint.",
    )
    args = parser.parse_args()
    convert_qwen3vl_to_diffusionvl(args.source_path, args.dest_path)
