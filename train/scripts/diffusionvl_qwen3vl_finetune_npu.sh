#!/bin/bash
# Finetune script for DiffusionVL-Qwen3VL model on NPU (Ascend)
# Model type: diffusionvl_qwen3vl (Qwen3-VL built-in vision tower + DeepStack + BD3-LM)
#
# Prerequisites:
#   - transformers >= 4.57.0 (ships Qwen3VLForConditionalGeneration)
#   - torch_npu installed and CANN toolkit configured
#   - A Qwen3-VL checkpoint converted to DiffusionVL format via
#     scripts/diffusionvl_prepare/convert_qwen3vl_to_diffusionvl.py
#
# Differences from the GPU (CUDA/NCCL) version:
#   - HCCL replaces NCCL for distributed communication
#   - torch_npu provides the Ascend backend
#   - TF32 is NVIDIA-only and is disabled; NPU trains in bf16 natively
#   - Attention defaults to "eager" (CANN's SDPA/Flash support varies by version)
#   - NCCL_* env vars are removed; HCCL_* env vars are set instead
#
# Dependency note:
#   DeepSpeed >= 0.16 is REQUIRED on NPU. Earlier versions abort ZeRO-3 init
#   with `assert len(set(t.dtype for t in tensors)) == 1` because the ZeRO-3
#   parameter partitioning did not handle the mixed bf16/fp32 layout produced
#   by `initialize_vision_modules` correctly on Ascend.

export OMP_NUM_THREADS=8

# ============================================
# NPU / CANN / HCCL configuration
# ============================================
# HCCL is the Ascend equivalent of NCCL. These env vars control the
# collective-communication backend used by torch.distributed.
export HCCL_TIMEOUT=7200              # seconds; large enough for long prefills
export HCCL_CONNECT_TIMEOUT=7200
# Let HCCL auto-select the collective algorithm. If you must set it manually,
# the required format is "level0:NA;level1:<algo>" (e.g. ring / mesh), NOT a
# bare "level0" — a malformed value aborts init with ERR02200.
export HCCL_WHITELIST_DISABLE=1       # Allow all algorithms (auto-selection)
export HCCL_IF_BASE_PORT=16000        # Base port for HCCL P2P
# Set the network interface for HCCL (like NCCL_SOCKET_IFNAME).
# TODO: replace with your actual NPU-facing network interface (e.g. eth0 / bond0)
export HCCL_SOCKET_IFNAME=eth0

# Make torch_npu available before training starts. Most CANN installs ship
# torch_npu as a regular site-package; if yours is elsewhere, adjust PYTHONPATH.
# export PYTHONPATH=/usr/local/lib/python3.*/site-packages:$PYTHONPATH

# Optimize memory reuse on Ascend (avoid fragmentation during checkpointing)
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# ============================================
# TODO: Configure these paths before running
# ============================================

# Wandb configuration (optional, set report_to="none" to disable)
export WANDB_DIR="./wandb"
export WANDB_PROJECT="diffusionvl"

# Model checkpoint path - Qwen3-VL model (converted to DiffusionVL format)
# TODO: Download Qwen3-VL (e.g. Qwen/Qwen3-VL-4B-Instruct) from HuggingFace and
#       convert to DiffusionVL format, then set the path here.
PRETRAINED_CHECKPOINT="/path/to/Qwen3-VL-4B-Instruct-Reformat"

# Training data paths
# TODO: Set your training data paths
DATA_PATH="/path/to/your/training_data.json"
IMAGE_FOLDER="/path/to/your/images"

# Output directory
# TODO: Set your output directory
OUTPUT_DIR="./outputs/diffusionvl_qwen3vl_finetune_npu"

# ============================================
# Training configuration
# ============================================
# we use 4 node and 8 npu per node and global batch size is 256
num_node=$1
gpu_num=$2                          # number of NPUs per node
custom_run_name=${3:-"diffusionvl_qwen3vl_finetune_npu"}
BD3LM_BLOCK_SIZE=${4:-8}

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29199"}
RANK=${RANK:-"0"}

echo "=========================================="
echo "DiffusionVL-Qwen3VL Finetune (Qwen3-VL + DeepStack + BD3-LM) on NPU"
echo "=========================================="
echo "master_addr ${MASTER_ADDR}"
echo "master_port ${MASTER_PORT}"
echo "node_rank ${RANK}"
echo "npu_num ${gpu_num}"
echo "num_node ${num_node}"
echo "BD3LM Block Size: ${BD3LM_BLOCK_SIZE}"

LLM_VERSION=${PRETRAINED_CHECKPOINT}
VISION_MODEL_VERSION=${PRETRAINED_CHECKPOINT}

echo "Checkpoint: ${PRETRAINED_CHECKPOINT}"

# Qwen3-VL uses the same ChatML format as Qwen2.5-VL, so the conversation
# template is reused. The system prompt differs slightly ("Qwen3" vs "Qwen")
# but the trained weights learn the actual template from the data.
PROMPT_VERSION=qwen_2_5
BASE_RUN_NAME=${custom_run_name}

echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

torchrun --nproc_per_node=${gpu_num} --nnodes=${num_node} --master_addr=${MASTER_ADDR} --master_port ${MASTER_PORT} --node_rank=${RANK} \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type qwen3_merger \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio pad \
    --bf16 True \
    --run_name $BASE_RUN_NAME \
    --output_dir "${OUTPUT_DIR}/$BASE_RUN_NAME" \
    --num_train_epochs 1 \
    --max_steps -1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 800 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --force_model_type "diffusionvl_qwen3vl" \
    --bd3lm_block_aligned_eos True \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --dataloader_drop_last True \
    --attn_implementation eager \
    --use_conversation_mask False \
    --enable_bd3lm True \
    --bd3lm_block_size ${BD3LM_BLOCK_SIZE}
