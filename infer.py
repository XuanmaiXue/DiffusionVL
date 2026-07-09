"""
Minimal inference script for DiffusionVL (Qwen2.5-VL / Qwen3-VL).

Pure local checkpoint loading: a single from_pretrained() call builds and
loads everything (language model + vision tower + projector + DeepStack),
because from_pretrained() now auto-calls initialize_vision_modules internally.

DiffusionVL does NOT use the standard `processor(text=, images=)` API. The
image is processed with the Qwen vision `image_processor` and the text with
the `tokenizer` separately, then model.generate(input_ids, images=,
image_grid_thws=...) runs BD3-LM block-diffusion sampling.
"""

import os
import sys

# Make the `llava` package importable so the model class registers itself
# with AutoModelForCausalLM (model_type: diffusionvl_qwenvl / diffusionvl_qwen3vl).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "train"))
import llava.model  # noqa: F401  (triggers register())

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from llava.mm_utils import tokenizer_image_token
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from transformers import AutoImageProcessor

# ============================================
# TODO: configure these paths
# ============================================
local_model_path = '/home/ma-user/work/xuemaixuan/data/dVLM/outputs/diffusionvl_qwenvl_finetune_npu/debug'
# Path containing preprocessor_config.json. Training checkpoints only save
# weights + config.json, NOT the image processor. Point this to the ORIGINAL
# (pre-conversion) checkpoint of the SAME generation as local_model_path:
#   - Qwen2.5-VL trained checkpoint  -> original Qwen2.5-VL (patch_size=14)
#   - Qwen3-VL   trained checkpoint  -> original Qwen3-VL   (patch_size=16)
# Cross-generation is NOT compatible (different patch_size / processor_class).
# Alternatively, copy preprocessor_config.json from there into local_model_path
# and set preprocessor_path = local_model_path.
#
# For Qwen2.5-VL:
preprocessor_path = '/home/ma-user/work/xuemaixuan/data/dVLM/ckpt/Qwen2.5-VL-7B-Instruct'
# For Qwen3-VL, uncomment and use instead:
# preprocessor_path = '/home/ma-user/work/xuemaixuan/data/dVLM/ckpt/Qwen3-VL-4B-Instruct'
image_path = "/home/ma-user/work/xuemaixuan/data/dVLM/dataset/LLaVA-Pretrain/images_subset/00000/000007653.jpg"
question = "Describe this image."
# ============================================

# --- 1. Load everything with a single from_pretrained ---
# trust_remote_code is not needed: the model classes are registered by the
# `import llava.model` above. from_pretrained() auto-builds vision modules.
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto" if (torch.cuda.is_available() or (hasattr(torch, "npu") and torch.npu.is_available())) else None,
    low_cpu_mem_usage=True,
)
model.eval()
device = next(model.parameters()).device

tokenizer = AutoTokenizer.from_pretrained(local_model_path, fix_mistral_regex=True)
# Image processor is loaded from the original checkpoint (has preprocessor_config.json).
_pp = preprocessor_path if os.path.exists(os.path.join(preprocessor_path, "preprocessor_config.json")) else local_model_path
image_processor = AutoImageProcessor.from_pretrained(_pp, use_fast=False)

# --- 2. Build the conversation prompt (ChatML, single <image> placeholder) ---
image = Image.open(image_path).convert("RGB")
prompt_question = DEFAULT_IMAGE_TOKEN + "\n" + question

conv = conv_templates["qwen_2_5"].copy()  # works for both Qwen2.5-VL and Qwen3-VL
conv.append_message(conv.roles[0], prompt_question)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

# --- 3. Tokenize text + process image SEPARATELY (not via processor(...)) ---
input_ids = tokenizer_image_token(
    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
).unsqueeze(0).to(device)

processor_output = image_processor(images=image, return_tensors="pt")
image_tensor = processor_output["pixel_values"].to(dtype=torch.bfloat16, device=device)
image_grid_thws = processor_output.get("image_grid_thw")
if image_grid_thws is not None:
    image_grid_thws = [tuple(t.tolist()) for t in image_grid_thws]

attention_mask = torch.ones_like(input_ids)

# --- 4. BD3-LM block-diffusion generation ---
print(f"Question: {question}\n")
with torch.inference_mode():
    gen_kwargs = dict(
        gen_length=128,
        steps=8,
        temperature=0.0,
        remasking_strategy="low_confidence_static",
    )
    if image_grid_thws is not None:
        gen_kwargs["image_grid_thws"] = image_grid_thws

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        images=image_tensor,
        modalities=["image"],
        **gen_kwargs,
    )

# output_ids contains only the newly generated tokens.
response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
print(f"Response: {response}")
