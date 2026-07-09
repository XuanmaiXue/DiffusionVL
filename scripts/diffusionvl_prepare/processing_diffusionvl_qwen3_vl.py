# coding=utf-8
# Copyright 2025 The HustVL Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on Qwen3-VL, LLaDA-V, and Block Diffusion. It has been
# modified to create DiffusionVL.
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
"""DiffusionVL-Qwen3VL Processor — combines image processor and tokenizer.

This is the self-contained processor for the DiffusionVL-Qwen3VL inference
checkpoint. It wraps a Qwen2VL image processor and a Qwen2 tokenizer, and
handles LLaVA-style <image> placeholder tokenization (replacing <image> with
IMAGE_TOKEN_INDEX = -200 in input_ids).
"""

from typing import List, Optional, Union

import torch

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.video_utils import VideoInput


IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"


class DiffusionVLQwen3VLProcessorKwargs(ProcessingKwargs, total=False):
    """Keyword arguments for DiffusionVLQwen3VLProcessor."""

    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
    }


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    """Tokenize text with <image> placeholders, replacing them with IMAGE_TOKEN_INDEX.

    Matches the training code (llava/mm_utils.py::tokenizer_image_token).
    """
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

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


class DiffusionVLQwen3VLProcessor(ProcessorMixin):
    """Combines a Qwen2VL image processor and a Qwen2 tokenizer.

    The text should contain <image> placeholders where images should be inserted.
    These are replaced with IMAGE_TOKEN_INDEX (-200) in the output input_ids.
    The model's `prepare_inputs_labels_for_multimodal` replaces -200 with the
    actual image features.

    Example:

    ```python
    >>> from transformers import AutoProcessor
    >>> from PIL import Image

    >>> processor = AutoProcessor.from_pretrained("path/to/model", trust_remote_code=True)
    >>> image = Image.open("image.jpg")
    >>> messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]}]
    >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    >>> inputs = processor(text=[text], images=[image], return_tensors="pt")
    ```
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "Qwen2VLImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        self.image_token = DEFAULT_IMAGE_TOKEN
        self.image_token_index = IMAGE_TOKEN_INDEX
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        videos: Optional[VideoInput] = None,
        **kwargs: Unpack[DiffusionVLQwen3VLProcessorKwargs],
    ) -> BatchFeature:
        """Process text and images into model inputs.

        Returns a BatchFeature with:
        - input_ids: token IDs with -200 at <image> positions (right-padded)
        - attention_mask
        - pixel_values (when images provided)
        - image_grid_thw (when images provided)
        """
        output_kwargs = self._merge_kwargs(
            DiffusionVLQwen3VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # Process images
        image_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs.get("images_kwargs", {}))

        # Handle text input
        if text is None:
            return BatchFeature(data=image_inputs)

        if not isinstance(text, list):
            text = [text]

        return_tensors = output_kwargs.get("text_kwargs", {}).pop("return_tensors", None)

        all_input_ids = []
        for t in text:
            input_ids = tokenizer_image_token(t, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors=None)
            all_input_ids.append(input_ids)

        # Right-pad sequences
        max_len = max(len(ids) for ids in all_input_ids)
        padded_input_ids = []
        attention_masks = []

        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        for ids in all_input_ids:
            padding_length = max_len - len(ids)
            padded_ids = ids + [pad_token_id] * padding_length
            mask = [1] * len(ids) + [0] * padding_length
            padded_input_ids.append(padded_ids)
            attention_masks.append(mask)

        text_inputs = {
            "input_ids": padded_input_ids,
            "attention_mask": attention_masks,
        }

        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def build_conversation_input_ids(self, messages, images=None, add_generation_prompt=True):
        """Build a prompt string with <image> placeholders from chat messages.

        Produces ChatML format:
        <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n<image>\nPrompt<|im_end|>\n<|im_start|>assistant\n
        """
        text_parts = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            text_parts.append(f"<|im_start|>{role}\n")

            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "image":
                            text_parts.append(DEFAULT_IMAGE_TOKEN)
                        elif item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                    else:
                        text_parts.append(str(item))

            text_parts.append("<|im_end|>\n")

        if add_generation_prompt:
            text_parts.append("<|im_start|>assistant\n")

        text = "".join(text_parts)
        return {"text": text}

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_names = self.tokenizer.model_input_names
        image_processor_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_names + image_processor_names))


__all__ = ["DiffusionVLQwen3VLProcessor", "tokenizer_image_token"]
