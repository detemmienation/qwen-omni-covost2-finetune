# Qwen2.5-Omni Fine-tuning on CoVoST2 (English Speech → Chinese Text)

This repository contains a minimal pipeline for fine-tuning **Qwen2.5-Omni-7B** with **SWIFT** on a small subset of **CoVoST2** for speech translation from English audio to Chinese text.

The main goal of this repo is to provide a **reproducible debug pipeline**:
- download a small subset of CoVoST2
- convert it into SWIFT multimodal JSONL format
- run LoRA fine-tuning on Qwen2.5-Omni/Qwem3.0-Omni

## Environment

Recommended environment:
- AWS GPU instance: `g5.xlarge` or similar
- GPU: NVIDIA A10G 24GB
- Python: 3.10
- Ubuntu: 22.04

## Setup

Create a virtual environment:

```
python3 -m venv swift-clean
source swift-clean/bin/activate
pip install -U pip setuptools wheel
```

Install dependencies:
``` 
pip install "ms-swift>=3.9,<3.10"
pip install "transformers==4.57.*"
pip install "huggingface-hub==0.36.0"
pip install "qwen_omni_utils==0.0.8"
pip install datasets soundfile librosa decord torchvision
pip install torch torchaudio
```

## Data Preparation

We use the Hugging Face version of CoVoST2:
	•	dataset: fixie-ai/covost2
	•	config: en_zh-CN

Run:

`python scripts/prepare_covost2.py`

This script:
	1.	downloads a small subset of CoVoST2
	2.	saves audio files locally as .wav
	3.	converts the dataset into SWIFT JSONL format

Expected output structure:
```
data/
├── audio/
│   ├── train/
│   └── val/
├── train.jsonl
└── val.jsonl
```

Each JSONL record looks like this:
```

{
  "messages": [
    {"role": "system", "content": "You are a speech translation assistant."},
    {"role": "user", "content": "<audio>Please translate the spoken English into simplified Chinese text only."},
    {"role": "assistant", "content": "今天天气很好。"}
  ],
  "audios": ["/absolute/path/to/audio.wav"]
}
```

## Create Tiny Debug Split

To reduce memory usage, create a debug split:
```
head -n 20 data/debug_train.jsonl > data/tiny_train.jsonl
head -n 5 data/debug_val.jsonl > data/tiny_val.jsonl
```

to train a size of 500 and 100, create split like 
```
head -n 500 data/debug_train.jsonl > data/train_500.jsonl
head -n 100 data/debug_val.jsonl > data/val_100.jsonl
```

## Training

example training command(use size of 500 and 100):

```
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
USE_HF=1 \
swift sft \
  --model Qwen/Qwen2.5-Omni-7B \
  --use_hf true \
  --dataset /home/ubuntu/project/data/train_500.jsonl \
  --val_dataset /home/ubuntu/project/data/val_100.jsonl \
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
  --output_dir /home/ubuntu/project/output/run_500
```
-> try to modify learning_rate/lora_rank/num_train_epochs...

