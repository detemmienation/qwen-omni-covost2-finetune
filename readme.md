# Qwen2.5-Omni Fine-tuning on CoVoST2 (English Speech → Chinese Text)

This repository contains a minimal pipeline for fine-tuning **Qwen2.5-Omni-7B** with **SWIFT** on a small subset of **CoVoST2** for speech translation from English audio to Chinese text.

The main goal of this repo is to provide a **reproducible debug pipeline**:
- download a small subset of CoVoST2
- convert it into SWIFT multimodal JSONL format
- run LoRA fine-tuning on Qwen2.5-Omni

## Project Status

This pipeline has been tested on AWS and successfully runs a small debug fine-tuning job.

Because Qwen2.5-Omni-7B is large, full-scale training is difficult on a single 24GB GPU. The current setup is mainly intended for:
- pipeline validation
- small-scale debugging
- team reproduction

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

To reduce memory usage, create a very small debug split:
```
head -n 20 data/debug_train.jsonl > data/tiny_train.jsonl
head -n 5 data/debug_val.jsonl > data/tiny_val.jsonl
```

If debug_train.jsonl and debug_val.jsonl do not exist yet, create them first from train.jsonl and val.jsonl.

## Training

A minimal debug training command:
```

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
USE_HF=1 \
swift sft \
  --model Qwen/Qwen2.5-Omni-7B \
  --use_hf true \
  --dataset /home/ubuntu/project/data/tiny_train.jsonl \
  --val_dataset /home/ubuntu/project/data/tiny_val.jsonl \
  --train_type lora \
  --torch_dtype float16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --max_length 256 \
  --logging_steps 1 \
  --save_steps 200 \
  --eval_steps 1000 \
  --lora_rank 2 \
  --lora_alpha 4 \
  --lora_dropout 0.05 \
  --output_dir /home/ubuntu/project/output/debug_run_tiny_fp16
```

## Notes
	•	Do not commit downloaded model weights or dataset audio files.
	•	Use Hugging Face download instead of ModelScope for faster downloads on AWS US instances.
	•	Full training may exceed memory limits on a single 24GB GPU.
	•	This repository is intended for reproducibility of the pipeline, not for storing training artifacts.
