# coding=utf-8
# Copyright 2025 The HustVL Team. All rights reserved.
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
"""Package a training-side DiffusionVL checkpoint as a self-contained HF release.

What this does (and does NOT do)
--------------------------------
A training run (``train/llava/``) saves a checkpoint that is *already* in the
exact weight layout the HF release expects — verified against
``huggingface.co/hustvl/DiffusionVL-Qwen2.5VL-3B``:

    model.embed_tokens.weight
    model.layers.{N}.*                                  # Qwen2.5 decoder layers
    model.norm.weight
    model.vision_tower.vision_tower.blocks.{N}.*        # ViT backbone
    model.vision_tower.vision_tower.patch_embed.*
    model.mm_projector.merger.{ln_q,mlp.*}.*            # Qwen2.5-VL PatchMerger
    (lm_head.weight is absent when tie_word_embeddings=True)

So this script performs **no key remapping**. Its job is the *release packaging*
that sits between ``trainer.save_model()`` and ``huggingface_hub.upload``:

  1. re-shard the weights to a clean ``model-0000N-of-0000M.safetensors``
     layout + ``model.safetensors.index.json`` (training saves a single file
     or an ad-hoc shard layout);
  2. drop a duplicated ``lm_head.weight`` if it slipped in despite tied weights
     (safetensors refuses to save a tensor stored twice);
  3. rewrite ``config.json`` for release — set ``architectures``,
     ``auto_map`` (pointing at the trust_remote_code files), scrub
     machine-specific local paths, and ensure ``tie_word_embeddings`` is set;
  4. copy tokenizer / preprocessor / generation files verbatim;
  5. copy the three ``*_diffusionvl_qwen2_5_vl.py`` remote-code files from a
     source directory (without them, ``trust_remote_code=True`` loading fails).

The companion script ``convert_qwen2.5vl_to_diffusionvl.py`` goes the *other*
direction (base Qwen2.5-VL -> training layout). This one closes the loop from
training output to publishable release.
"""

import argparse
import json
import os
import shutil

from safetensors import safe_open
from safetensors.torch import save_file


# ---------------------------------------------------------------------------
# Weight handling
# ---------------------------------------------------------------------------

# When tie_word_embeddings is True, lm_head.weight shares storage with
# embed_tokens.weight. HF's save_pretrained normally skips it, but a checkpoint
# produced by a manual state_dict dump may include both copies. safetensors
# raises on shared storage, so we drop lm_head.weight in that case (the tie is
# re-established at load time from config.tie_word_embeddings).
def _should_drop(key, tie_word_embeddings):
    if tie_word_embeddings and key == "lm_head.weight":
        return True
    # The training vision tower deletes its own `merger` submodule at load time
    # (qwen_vision_tower.py: delattr), so this should never appear, but guard
    # against a stale copy that would collide with mm_projector.merger.
    if key.startswith("model.vision_tower.vision_tower.merger."):
        return True
    return False


_DEFAULT_MAX_SHARD_SIZE_BYTES = 5 * 1024 ** 3  # 5 GiB, matches Qwen2.5-VL


def _load_all_tensors(src_dir, src_shards, tie_word_embeddings):
    """Read every tensor from the source shards into one dict, dropping dups."""
    state_dict = {}
    dropped = []
    for shard_name in src_shards:
        shard_path = os.path.join(src_dir, shard_name)
        with safe_open(shard_path, framework="pt") as f:
            for key in f.keys():
                if _should_drop(key, tie_word_embeddings):
                    dropped.append(key)
                    continue
                state_dict[key] = f.get_tensor(key)
    return state_dict, dropped


def _shard_state_dict(state_dict, max_bytes):
    """Greedily pack tensors into shards under ``max_bytes`` (raw byte size)."""
    shards, current, current_bytes = [], {}, 0
    # Sorted keys -> reproducible shard layout across runs / machines.
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        nbytes = tensor.numel() * tensor.element_size()
        if current and current_bytes + nbytes > max_bytes:
            shards.append(current)
            current, current_bytes = {}, 0
        current[key] = tensor
        current_bytes += nbytes
    if current:
        shards.append(current)
    return shards


# ---------------------------------------------------------------------------
# Config rewriting
# ---------------------------------------------------------------------------

# Machine-specific / in-memory fields written by the training code that must
# NOT ship in a release config.
_CONFIG_DROP_FIELDS = {
    "mm_vision_tower",          # absolute local path to the base model
    "vision_tower_pretrained",  # absolute local path
    "vision_tower_state_dict",  # in-memory artifact (never JSON-serialized, but guard)
    "projector_state_dict",     # in-memory artifact
    "deepstack_state_dict",     # in-memory artifact (Qwen3-VL path)
    "_name_or_path",            # upstream HF id; misleading for a release
}

_RELEASE_ARCHITECTURE = "DiffusionVL_Qwen2_5_VL_ForConditionalGeneration"
_AUTO_MAP = {
    "AutoConfig": "configuration_diffusionvl_qwen2_5_vl.DiffusionVL_Qwen2_5_VL_Config",
    "AutoModel": "modeling_diffusionvl_qwen2_5_vl.DiffusionVL_Qwen2_5_VL_ForConditionalGeneration",
    "AutoModelForImageTextToText": "modeling_diffusionvl_qwen2_5_vl.DiffusionVL_Qwen2_5_VL_ForConditionalGeneration",
}


def rewrite_config(src_config, tie_word_embeddings):
    """Produce a release-ready config dict from the training checkpoint's config."""
    cfg = {k: v for k, v in src_config.items() if k not in _CONFIG_DROP_FIELDS}
    cfg["architectures"] = [_RELEASE_ARCHITECTURE]
    cfg["auto_map"] = dict(_AUTO_MAP)
    cfg["model_type"] = "diffusionvl_qwenvl"
    cfg["tie_word_embeddings"] = tie_word_embeddings
    return cfg


# ---------------------------------------------------------------------------
# File copying
# ---------------------------------------------------------------------------

# Files copied verbatim from the source checkpoint. Weights + config are
# handled separately.
_NON_WEIGHT_FILES_TO_COPY = [
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "added_tokens.json",
    "preprocessor_config.json", "chat_template.jinja",
    "generation_config.json",
]

# The trust_remote_code bundle. Copied verbatim from --remote_code_source.
_REMOTE_CODE_FILES = [
    "configuration_diffusionvl_qwen2_5_vl.py",
    "modeling_diffusionvl_qwen2_5_vl.py",
    "processing_diffusionvl_qwen2_5_vl.py",
]


def _copy_files(filenames, src_dir, dst_dir, label):
    copied, missing = [], []
    for fname in filenames:
        fpath = os.path.join(src_dir, fname)
        if os.path.isfile(fpath):
            shutil.copy2(fpath, os.path.join(dst_dir, fname))
            copied.append(fname)
        else:
            missing.append(fname)
    if copied:
        print(f"  copied: {', '.join(copied)}")
    if missing:
        print(f"  MISSING ({label}): {', '.join(missing)}")
    return copied, missing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert(
    src_checkpoint_dir,
    dest_dir,
    remote_code_source=None,
    max_shard_size_bytes=_DEFAULT_MAX_SHARD_SIZE_BYTES,
    skip_config_rewrite=False,
):
    src = os.path.abspath(src_checkpoint_dir)
    dst = os.path.abspath(dest_dir)
    if src == dst:
        raise ValueError("source and destination must differ")
    os.makedirs(dst, exist_ok=True)

    # ---- 1. Source config drives the tie-weights decision ------------------
    config_path = os.path.join(src, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config.json in {src}")
    with open(config_path, "r") as fp:
        src_config = json.load(fp)
    tie_word_embeddings = bool(src_config.get("tie_word_embeddings", True))

    # ---- 2. Discover source weight files ----------------------------------
    index_path = os.path.join(src, "model.safetensors.index.json")
    single_path = os.path.join(src, "model.safetensors")
    if os.path.exists(index_path):
        with open(index_path) as fp:
            src_index = json.load(fp)
        src_shards = sorted(set(src_index["weight_map"].values()))
        print(f"Source: sharded ({len(src_shards)} file(s))")
    elif os.path.exists(single_path):
        src_shards = ["model.safetensors"]
        print("Source: single model.safetensors")
    else:
        raise FileNotFoundError(
            f"No model.safetensors or model.safetensors.index.json in {src}"
        )

    # ---- 3. Load + de-dup tensors (no key remapping needed) ----------------
    print("\nReading tensors...")
    state_dict, dropped = _load_all_tensors(src, src_shards, tie_word_embeddings)
    if dropped:
        print(f"  dropped {len(dropped)} duplicate/stale key(s):")
        for k in dropped[:10]:
            print(f"    - {k}")
        if len(dropped) > 10:
            print(f"    ... ({len(dropped)} total)")
    print(f"  tensors to ship: {len(state_dict)}")

    # Sanity: every key must be in the expected namespaces. If a stray key
    # appears it likely means the training layout changed and this script
    # needs updating — fail loudly rather than ship a broken release.
    _EXPECTED_PREFIXES = (
        "model.embed_tokens", "model.layers", "model.norm",
        "model.vision_tower", "model.mm_projector",
        "lm_head",  # present only when not tied
    )
    stray = [k for k in state_dict if not k.startswith(_EXPECTED_PREFIXES)]
    if stray:
        raise ValueError(
            "Unexpected weight keys not in the known DiffusionVL-QwenVL layout:\n  "
            + "\n  ".join(stray[:20])
            + ("\n  ..." if len(stray) > 20 else "")
            + "\nThis script assumes training-side and release-side key naming "
            "match. If the training layout changed, update _EXPECTED_PREFIXES."
        )

    # ---- 4. Re-shard + write index ----------------------------------------
    print("\nRe-sharding...")
    shards = _shard_state_dict(state_dict, max_shard_size_bytes)
    n_shards = len(shards)
    weight_map = {}
    for i, shard in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        save_file(shard, os.path.join(dst, fname), metadata={"format": "pt"})
        for k in shard:
            weight_map[k] = fname
        print(f"  wrote {fname} ({len(shard)} tensors)")

    total_params = sum(t.numel() for t in state_dict.values())
    index_out = {
        "metadata": {
            "total_parameters": total_params,
            "total_size": sum(t.numel() * t.element_size() for t in state_dict.values()),
        },
        "weight_map": weight_map,
    }
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as fp:
        json.dump(index_out, fp, indent=2)
        fp.write("\n")

    # ---- 5. Config ---------------------------------------------------------
    if not skip_config_rewrite:
        release_cfg = rewrite_config(src_config, tie_word_embeddings)
        with open(os.path.join(dst, "config.json"), "w") as fp:
            json.dump(release_cfg, fp, indent=2, ensure_ascii=False)
            fp.write("\n")
        print("\nWrote release config.json (architectures + auto_map, scrubbed paths).")
    else:
        shutil.copy2(config_path, os.path.join(dst, "config.json"))
        print("\nCopied config.json unchanged (--skip_config_rewrite).")

    # ---- 6. Tokenizer / processor / generation files ----------------------
    print("\nTokenizer / processor files:")
    _copy_files(_NON_WEIGHT_FILES_TO_COPY, src, dst, "non-weight files")

    # ---- 7. Remote-code bundle --------------------------------------------
    print("\nRemote-code files:")
    if remote_code_source:
        rcs = os.path.abspath(remote_code_source)
        if not os.path.isdir(rcs):
            raise FileNotFoundError(f"--remote_code_source not a directory: {rcs}")
        copied, missing = _copy_files(_REMOTE_CODE_FILES, rcs, dst, "remote-code")
        if missing:
            print(
                f"WARNING: {len(missing)} remote-code file(s) missing. The release "
                f"will NOT load with trust_remote_code until all three are present."
            )
    else:
        print("  --remote_code_source not given; remote-code files NOT copied.")
        print("  Without them, trust_remote_code loading will fail. Pass")
        print("  --remote_code_source <dir> pointing at a folder containing:")
        for f in _REMOTE_CODE_FILES:
            print(f"    {f}")

    # ---- 8. Summary --------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Release checkpoint written to: {dst}")
    print(f"  shards        : {n_shards}")
    print(f"  tensors       : {len(state_dict)}")
    print(f"  total params  : {total_params:,}")
    print(f"  tie_weights   : {tie_word_embeddings}")
    print("=" * 60)
    print("Next: upload with `huggingface-cli upload <repo> <dst> --repo-type model`")


def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Package a training-side DiffusionVL-Qwen2.5VL checkpoint as a "
            "self-contained HF release (trust_remote_code layout)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python convert_diffusionvl_to_hf_release.py \\\n"
            "      --src_checkpoint_dir ./outputs/diffusionvl_qwenvl_final \\\n"
            "      --dest_dir ./DiffusionVL-Qwen2.5VL-3B-release \\\n"
            "      --remote_code_source ./hf_remote_code"
        ),
    )
    p.add_argument("--src_checkpoint_dir", required=True,
                   help="Training output dir (model*.safetensors + config.json).")
    p.add_argument("--dest_dir", required=True,
                   help="Where to write the self-contained HF release.")
    p.add_argument("--remote_code_source", default=None,
                   help="Dir containing the three *_diffusionvl_qwen2_5_vl.py files. "
                        "If omitted, remote-code files are NOT copied.")
    p.add_argument("--max_shard_size_gb", type=float, default=5.0,
                   help="Max size per safetensors shard in GiB (default 5).")
    p.add_argument("--skip_config_rewrite", action="store_true",
                   help="Copy config.json verbatim instead of rewriting for release.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    convert(
        src_checkpoint_dir=args.src_checkpoint_dir,
        dest_dir=args.dest_dir,
        remote_code_source=args.remote_code_source,
        max_shard_size_bytes=int(args.max_shard_size_gb * 1024 ** 3),
        skip_config_rewrite=args.skip_config_rewrite,
    )
