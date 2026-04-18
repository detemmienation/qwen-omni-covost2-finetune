#!/bin/bash
# E2E fine-tuning with focal + entity-upweighted loss on ~500 training samples.
#
# Prerequisites:
#   1. source swift-clean/bin/activate  (or conda activate qwen3-omni)
#   2. python scripts/extract_audio_features.py   # builds audio cache
#
# Loss settings are in run_train_custom_loss.py:
#   FOCAL_GAMMA   = 2.0
#   ENTITY_WEIGHT = 5.0
set -e

DATA_DIR="/home/ubuntu/leili-cmu-lab/CMU-project/Y4_data/data"
OUTPUT_DIR="/home/ubuntu/leili-cmu-lab/CMU-project/Y4_data/output/custom_loss_500"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
USE_HF=1 \
python "$SCRIPT_DIR/run_train_custom_loss.py" \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --use_hf true \
  --dataset       "$DATA_DIR/train_500.jsonl" \
  --val_dataset   "$DATA_DIR/val_100.jsonl" \
  --train_type lora \
  --torch_dtype float16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 \
  --max_length 320 \
  --logging_steps 10 \
  --save_steps 200 \
  --eval_steps 200 \
  --lora_rank 4 \
  --lora_alpha 8 \
  --lora_dropout 0.05 \
  --dataloader_num_workers 0 \
  --output_dir "$OUTPUT_DIR"
