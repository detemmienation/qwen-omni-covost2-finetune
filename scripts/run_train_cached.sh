#!/bin/bash
# Fine-tuning with pre-cached audio encoder features.
# Prerequisites:
#   1. conda activate qwen3-omni
#   2. python scripts/extract_audio_features.py  (builds data/audio_cache/)
# Then run this script.
set -e

DATA_DIR="/home/ubuntu/leili-cmu-lab/CMU-project/Y4_data/data"
OUTPUT_DIR="/home/ubuntu/leili-cmu-lab/CMU-project/Y4_data/output/cached_e2e"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
USE_HF=1 \
python "$SCRIPT_DIR/run_train_cached.py" \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --use_hf true \
  --dataset "$DATA_DIR/train_5000.jsonl" \
  --val_dataset "$DATA_DIR/val_500.jsonl" \
  --train_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --max_length 256 \
  --logging_steps 10 \
  --save_steps 200 \
  --eval_steps 200 \
  --lora_rank 4 \
  --lora_alpha 8 \
  --lora_dropout 0.05 \
  --target_modules q_proj v_proj \
  --attn_impl flash_attn \
  --dataloader_num_workers 0 \
  --output_dir "$OUTPUT_DIR"
