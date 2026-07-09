"""
Inference script for DiffusionVL-Qwen3VL using self-contained checkpoint files.

This uses the STANDARD HF API (AutoModelForCausalLM + AutoProcessor with
trust_remote_code=True), which works once the checkpoint directory contains:
  - configuration_diffusionvl_qwen3_vl.py
  - modeling_diffusionvl_qwen3_vl.py
  - processing_diffusionvl_qwen3_vl.py
  - config.json (with auto_map)
  - preprocessor_config.json
  - tokenizer files

Prepare the checkpoint first with:
    python scripts/diffusionvl_prepare/add_self_contained_to_ckpt.py --ckpt_path /path/to/checkpoint
"""

import argparse

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


def main():
    parser = argparse.ArgumentParser(description="DiffusionVL-Qwen3VL inference (self-contained)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the checkpoint with self-contained .py files.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--question", type=str, default="Describe this image.")
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    # --- Load model + processor (standard HF API, trust_remote_code) ---
    print(f"Loading model from {args.model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, fix_mistral_regex=True)
    device = next(model.parameters()).device

    # --- Build chat input ---
    image = Image.open(args.image_path).convert("RGB")
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": args.question}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # --- Process inputs (standard processor call) ---
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    # --- BD3-LM block-diffusion generation ---
    print(f"Question: {args.question}\n")
    with torch.inference_mode():
        output_ids = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            images=inputs.get("pixel_values"),
            image_grid_thws=inputs.get("image_grid_thw"),
            modalities=["image"],
            gen_length=args.gen_length,
            steps=args.steps,
            temperature=args.temperature,
            remasking_strategy="low_confidence_static",
        )

    response = processor.decode(output_ids[0], skip_special_tokens=True).strip()
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
