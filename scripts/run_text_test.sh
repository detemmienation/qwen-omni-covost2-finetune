#!/bin/bash
set -e

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

BASE_DIR="/home/ubuntu/leili-cmu-lab/CMU-project/Y4_data"
TEST_DIR="$BASE_DIR/data/whisper/test"
OUT_DIR="$BASE_DIR/text_test_outputs"
LORA_PATH="$BASE_DIR/output/text_translation_omni_1000/checkpoint-750"
mkdir -p "$OUT_DIR"

for f in "$TEST_DIR"/*.jsonl; do
  name=$(basename "$f" .jsonl)
  out_file="$OUT_DIR/${name}_predictions.jsonl"

  echo "=================================================="
  echo "Running: $f"
  echo "Output : $out_file"
  echo "=================================================="

  python3 "/home/ubuntu/Y4/scripts/infer_text.py" \
    --lora_path "$LORA_PATH" \
    --input_jsonl "$f" \
    --output_jsonl "$out_file" \
    --max_new_tokens 64
done

echo "All done."