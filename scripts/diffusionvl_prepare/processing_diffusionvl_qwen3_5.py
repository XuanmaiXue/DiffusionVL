# coding=utf-8
# Copyright 2025 The HustVL Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on Qwen3.5, LLaDA-V, and Block Diffusion.
# Licensed under the Apache License, Version 2.0.
"""DiffusionVL-Qwen3.5 Processor — combines image processor and tokenizer."""

from typing import List, Optional, Union

import torch

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.video_utils import VideoInput


IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"


class DiffusionVLQwen3_5ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {"text_kwargs": {"padding": False}}


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    """Tokenize text with <image> placeholders → IMAGE_TOKEN_INDEX."""
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])
    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors == "pt":
        return torch.tensor(input_ids, dtype=torch.long)
    return input_ids


class DiffusionVLQwen3_5Processor(ProcessorMixin):
    """Combines a Qwen2VL image processor and a Qwen2 tokenizer.

    Text should contain <image> placeholders → replaced with -200 in input_ids.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "Qwen2VLImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        self.image_token = DEFAULT_IMAGE_TOKEN
        self.image_token_index = IMAGE_TOKEN_INDEX
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(self, images=None, text=None, videos=None, **kwargs):
        output_kwargs = self._merge_kwargs(
            DiffusionVLQwen3_5ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs)

        image_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs.get("images_kwargs", {}))

        if text is None:
            return BatchFeature(data=image_inputs)

        if not isinstance(text, list):
            text = [text]

        return_tensors = output_kwargs.get("text_kwargs", {}).pop("return_tensors", None)
        all_input_ids = []
        for t in text:
            input_ids = tokenizer_image_token(t, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors=None)
            all_input_ids.append(input_ids)

        max_len = max(len(ids) for ids in all_input_ids)
        padded_input_ids, attention_masks = [], []
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        for ids in all_input_ids:
            padding_length = max_len - len(ids)
            padded_input_ids.append(ids + [pad_token_id] * padding_length)
            attention_masks.append([1] * len(ids) + [0] * padding_length)

        text_inputs = {"input_ids": padded_input_ids, "attention_mask": attention_masks}
        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return list(dict.fromkeys(self.tokenizer.model_input_names + self.image_processor.model_input_names))


__all__ = ["DiffusionVLQwen3_5Processor", "tokenizer_image_token"]
