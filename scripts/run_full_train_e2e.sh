#!/bin/bash
# Full-parameter fine-tuning: English speech → Chinese text (e2e)
# Model: Qwen/Qwen3-Omni-30B-A3B-Instruct
# Hardware: single NVIDIA GH200 480GB
set -e

DATA_DIR="/home/ubuntu/leili-cmu-lab/CMU-project/Y4_data/data"
OUTPUT_DIR="/home/ubuntu/leili-cmu-lab/CMU-project/Y4_data/output/full_e2e"

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
USE_HF=1 \
swift sft \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --use_hf true \
  --dataset "$DATA_DIR/train_10000_20000_cot.jsonl" \
  --val_dataset "$DATA_DIR/val_500_cot.jsonl" \
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
  --target_modules q_proj v_proj o_proj \
  --attn_impl flash_attn \
  --dataloader_num_workers 0 \
  --adapters "$OUTPUT_DIR/v16-20260418-060538/checkpoint-1250" \
  --output_dir "$OUTPUT_DIR"
