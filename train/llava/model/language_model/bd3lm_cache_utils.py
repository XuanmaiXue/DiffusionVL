# coding=utf-8
# Copyright 2025 The HustVL Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0.
"""Cache snapshot/restore utilities for hybrid (linear + full) attention models.

During BD3-LM block-diffusion denoising, each denoising step evaluates the
same block without committing its cache state. Full-attention layers support
store_kv=False (read cache without writing), but linear-attention layers
(Gated DeltaNet) write unconditionally when cache_params is not None.

These utilities snapshot/restore the linear layers' conv_states + recurrent_states
and crop the full-attention layers' KV caches back to their pre-block lengths.
"""

import torch


def snapshot_linear_cache(cache):
    """Snapshot conv_states + recurrent_states of all linear-attention layers.

    conv_states and recurrent_states are dict[int, Tensor] (keyed by state_idx),
    so we deep-copy each tensor inside the dict.
    """
    from transformers.cache_utils import LinearAttentionCacheLayerMixin
    snapshot = {}
    layers = getattr(cache, "layers", None)
    if layers is None:
        return snapshot
    for idx, layer in enumerate(layers):
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            conv_snap = {}
            if isinstance(layer.conv_states, dict):
                for s_idx, t in layer.conv_states.items():
                    conv_snap[s_idx] = t.clone() if t is not None else None
            elif isinstance(layer.conv_states, torch.Tensor):
                conv_snap[0] = layer.conv_states.clone()
            else:
                conv_snap[0] = None

            recur_snap = {}
            if isinstance(layer.recurrent_states, dict):
                for s_idx, t in layer.recurrent_states.items():
                    recur_snap[s_idx] = t.clone() if t is not None else None
            elif isinstance(layer.recurrent_states, torch.Tensor):
                recur_snap[0] = layer.recurrent_states.clone()
            else:
                recur_snap[0] = None

            conv_init = dict(layer.is_conv_states_initialized) if isinstance(layer.is_conv_states_initialized, dict) else layer.is_conv_states_initialized
            recur_init = dict(layer.is_recurrent_states_initialized) if isinstance(layer.is_recurrent_states_initialized, dict) else layer.is_recurrent_states_initialized

            snapshot[idx] = (conv_snap, recur_snap, conv_init, recur_init)
    return snapshot


def restore_linear_cache(cache, snapshot):
    """Restore conv_states + recurrent_states from a snapshot (in-place copy_)."""
    layers = getattr(cache, "layers", None)
    if layers is None:
        return
    for idx, (conv_snap, recur_snap, conv_init, recur_init) in snapshot.items():
        layer = layers[idx]
        if isinstance(layer.conv_states, dict):
            for s_idx, t in conv_snap.items():
                if t is not None and s_idx in layer.conv_states and layer.conv_states[s_idx] is not None:
                    layer.conv_states[s_idx].copy_(t)
        elif isinstance(layer.conv_states, torch.Tensor) and conv_snap.get(0) is not None:
            layer.conv_states.copy_(conv_snap[0])
        if isinstance(layer.recurrent_states, dict):
            for s_idx, t in recur_snap.items():
                if t is not None and s_idx in layer.recurrent_states and layer.recurrent_states[s_idx] is not None:
                    layer.recurrent_states[s_idx].copy_(t)
        elif isinstance(layer.recurrent_states, torch.Tensor) and recur_snap.get(0) is not None:
            layer.recurrent_states.copy_(recur_snap[0])
        layer.is_conv_states_initialized = conv_init
        layer.is_recurrent_states_initialized = recur_init


def snapshot_full_attn_cache_seq_len(cache):
    """Record the KV-cache sequence length of all full-attention layers."""
    lengths = {}
    layers = getattr(cache, "layers", None)
    if layers is None:
        return lengths
    for idx, layer in enumerate(layers):
        if not hasattr(layer, "keys"):
            continue
        lengths[idx] = layer.get_seq_length() if hasattr(layer, "get_seq_length") else 0
    return lengths


def crop_full_attn_cache(cache, lengths):
    """Crop full-attention KV caches back to recorded lengths."""
    layers = getattr(cache, "layers", None)
    if layers is None:
        return
    for idx, target_len in lengths.items():
        layer = layers[idx]
        cur_len = layer.get_seq_length() if hasattr(layer, "get_seq_length") else 0
        if cur_len <= target_len:
            continue
        if hasattr(layer, "crop"):
            layer.crop(target_len)
        elif hasattr(layer, "keys") and layer.keys is not None:
            layer.keys = layer.keys[:, :, :target_len, :]
            layer.values = layer.values[:, :, :target_len, :]
