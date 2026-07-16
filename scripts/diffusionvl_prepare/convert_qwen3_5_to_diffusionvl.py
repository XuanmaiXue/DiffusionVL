import argparse
import json
import os
import shutil

from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import snapshot_download

# NOTE on scope:
# This script converts a Qwen3.5 checkpoint (hybrid Gated DeltaNet + full
# attention + dense MoE-free) to the DiffusionVL checkpoint layout for inference.
#
# Qwen3.5-4B specifics:
#   - model.language_model.*  (426 keys) → language backbone (32 hybrid layers)
#   - model.visual.*          (297 keys) → vision tower (24 ViT blocks, NO DeepStack)
#   - mtp.*                   ( 15 keys) → Multi-Token Prediction head (IGNORED for inference)
#
# The MTP head is not used during inference and is dropped.
# deepstack_visual_indexes is empty ([]), so no deepstack_merger.safetensors.


def convert_qwen3_5_to_diffusionvl(source_path, dest_path):
    """Convert a Qwen3.5 checkpoint to the DiffusionVL checkpoint layout."""
    if os.path.isdir(source_path):
        print(f"Source path '{source_path}' is a local directory. Using it directly.")
        source_local_path = source_path
    else:
        print(f"Source path '{source_path}' not found locally. Attempting HF Hub download.")
        source_local_path = snapshot_download(source_path)

    os.makedirs(dest_path, exist_ok=True)

    # --- Copy non-weight configuration / tokenizer files ---
    print("Copying configuration and tokenizer files...")
    files_to_ignore = [".git", ".gitattributes"]
    weight_extensions = [".safetensors", ".bin", ".pth"]
    for filename in os.listdir(source_local_path):
        if filename in files_to_ignore:
            continue
        if any(filename.endswith(ext) for ext in weight_extensions):
            continue
        if filename == "model.safetensors.index.json" or filename.endswith(".index.json"):
            continue

        src_file = os.path.join(source_local_path, filename)
        if os.path.isfile(src_file):
            if filename == "config.json":
                with open(src_file, "r") as fp:
                    cfg = json.load(fp)
                cfg["model_type"] = "diffusionvl_qwen3_5"
                cfg["architectures"] = ["DiffusionVLQwen3_5ForConditionalGeneration"]
                with open(os.path.join(dest_path, "config.json"), "w") as fp:
                    json.dump(cfg, fp, indent=2, ensure_ascii=False)
                    fp.write("\n")
                print("  - Writing config.json (model_type -> diffusionvl_qwen3_5)")
            else:
                print(f"  - Copying {filename}")
                shutil.copy2(src_file, dest_path)

    # --- Route tensors into buckets by key prefix ---
    print("\nRouting tensors into DiffusionVL buckets...")
    safetensors_files = sorted(
        f for f in os.listdir(source_local_path) if f.endswith(".safetensors")
    )
    if not safetensors_files:
        print("ERROR: No .safetensors weight files found.")
        return

    main_state_dict = {}          # -> model.safetensors (language model)
    vision_tower_state_dict = {}  # -> vision_tower.safetensors
    projector_state_dict = {}     # -> projector.safetensors (main merger)
    skipped = {}                  # -> ignored (mtp)

    LM_PREFIX = "model.language_model."
    VISUAL_PREFIX = "model.visual."

    for st_file in safetensors_files:
        st_path = os.path.join(source_local_path, st_file)
        print(f"  - Reading {st_file}")
        with safe_open(st_path, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if key.startswith(LM_PREFIX):
                    new_key = "model." + key[len(LM_PREFIX):]
                    main_state_dict[new_key] = tensor
                elif key.startswith(VISUAL_PREFIX + "merger."):
                    # Visual merger → projector.safetensors (strip "model.visual.merger.")
                    new_key = key[len(VISUAL_PREFIX + "merger."):]
                    projector_state_dict[new_key] = tensor
                elif key.startswith(VISUAL_PREFIX):
                    new_key = key[len(VISUAL_PREFIX):]
                    vision_tower_state_dict[new_key] = tensor
                else:
                    skipped[key] = tensor

    # --- Report ---
    total = len(main_state_dict) + len(vision_tower_state_dict) + len(projector_state_dict) + len(skipped)
    print("\nRouting summary:")
    print(f"  model.safetensors        (language_model): {len(main_state_dict)} tensors")
    print(f"  vision_tower.safetensors (visual backbone): {len(vision_tower_state_dict)} tensors")
    print(f"  projector.safetensors    (main merger)   : {len(projector_state_dict)} tensors")
    print(f"  ignored (mtp)                           : {len(skipped)} tensors")
    print(f"  total routed                             : {total}")

    if skipped:
        # Verify all skipped are mtp
        non_mtp = [k for k in skipped if not k.startswith("mtp.")]
        if non_mtp:
            print(f"  WARNING: unexpected non-mtp skipped keys: {non_mtp[:5]}")

    # --- Save ---
    print("\nSaving converted weights...")
    save_file(main_state_dict, os.path.join(dest_path, "model.safetensors"))
    save_file(vision_tower_state_dict, os.path.join(dest_path, "vision_tower.safetensors"))
    save_file(projector_state_dict, os.path.join(dest_path, "projector.safetensors"))

    # --- Export vision config ---
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

    del main_state_dict, vision_tower_state_dict, projector_state_dict, skipped

    print(f"\nConversion successful! DiffusionVL-Qwen3.5 checkpoint saved at: {dest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a Qwen3.5 model to the DiffusionVL checkpoint format."
    )
    parser.add_argument("--source_path", type=str, required=True,
                        help="Path or name of the original Qwen3.5 model.")
    parser.add_argument("--dest_path", type=str, required=True,
                        help="Path to save the converted DiffusionVL checkpoint.")
    args = parser.parse_args()
    convert_qwen3_5_to_diffusionvl(args.source_path, args.dest_path)
