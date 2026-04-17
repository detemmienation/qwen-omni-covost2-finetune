#!/bin/bash
# Full-parameter fine-tuning: text translation (English → Chinese)
# Model: Qwen/Qwen3-Omni-30B-A3B-Instruct
# Hardware: single NVIDIA GH200 480GB
# Prerequisites: run prepare_text_translation_from_whisper.py first
set -e

DATA_DIR="/home/ubuntu/leili-cmu-lab/CMU-project/Y4_data/data"
OUTPUT_DIR="/home/ubuntu/leili-cmu-lab/CMU-project/Y4_data/output/full_text"

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
USE_HF=1 \
swift sft \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --use_hf true \
  --dataset "$DATA_DIR/text_train_1000.jsonl" \
  --val_dataset "$DATA_DIR/text_val_200.jsonl" \
  --train_type full \
  --torch_dtype bfloat16 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --max_length 512 \
  --logging_steps 10 \
  --save_steps 200 \
  --eval_steps 100 \
  --output_dir "$OUTPUT_DIR"
