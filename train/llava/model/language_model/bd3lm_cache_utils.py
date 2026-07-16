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
    """Snapshot conv_states + recurrent_states of all linear-attention layers."""
    from transformers.cache_utils import LinearAttentionCacheLayerMixin
    snapshot = {}
    layers = getattr(cache, "layers", None)
    if layers is None:
        return snapshot
    for idx, layer in enumerate(layers):
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            conv = layer.conv_states.clone() if layer.conv_states is not None else None
            recur = layer.recurrent_states.clone() if layer.recurrent_states is not None else None
            snapshot[idx] = (conv, recur, layer.is_conv_states_initialized, layer.is_recurrent_states_initialized)
    return snapshot


def restore_linear_cache(cache, snapshot):
    """Restore conv_states + recurrent_states from a snapshot."""
    layers = getattr(cache, "layers", None)
    if layers is None:
        return
    for idx, (conv, recur, conv_init, recur_init) in snapshot.items():
        layer = layers[idx]
        if conv is not None and layer.conv_states is not None:
            layer.conv_states.copy_(conv)
        if recur is not None and layer.recurrent_states is not None:
            layer.recurrent_states.copy_(recur)
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
