"""
Modal training pipeline for Qwen3-Omni CoVoST2 e2e fine-tuning.

Three-stage pipeline:
  1. prepare  — download CoVoST2 subset from HuggingFace, save audio + JSONL
  2. extract  — pre-extract audio encoder features (skips encoder during training)
  3. train    — LoRA fine-tuning with focal + entity-upweighted loss

Usage:
    modal run modal_train.py                         # full pipeline
    modal run modal_train.py --stage prepare
    modal run modal_train.py --stage extract
    modal run modal_train.py --stage train

Volumes (created automatically on first run):
    qwen-omni-data     — audio files, JSONL, audio cache, checkpoints
    qwen-omni-hfcache  — HuggingFace model cache (reused across runs)

Secrets required (create once via `modal secret create`):
    huggingface-secret  with key HF_TOKEN

GPU requirements:
    extract  → A100-80GB  (loads full 30B model, then frees all but audio tower)
    train    → A100-40GB  (audio encoder bypassed; only LLM LoRA in float16)
"""

import os
import sys
from pathlib import Path

import modal

# ── App + image ────────────────────────────────────────────────────────────────

APP_NAME = "qwen-omni-covost2"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git", "ffmpeg", "libsndfile1"])
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "ms-swift>=3.9,<3.10",
        "transformers==4.51.*",
        "huggingface-hub>=0.36.0",
        "qwen_omni_utils==0.0.8",
        "datasets",
        "soundfile",
        "librosa",
        "decord",
        "torchvision",
        "sacrebleu",
        "tqdm",
    )
    # Mount the scripts directory so all helpers are importable
    .add_local_dir("scripts", "/app/scripts")
)

app = modal.App(APP_NAME, image=image)

# Persistent volumes
data_vol   = modal.Volume.from_name("qwen-omni-data",    create_if_missing=True)
hfcache_vol = modal.Volume.from_name("qwen-omni-hfcache", create_if_missing=True)

VOL_DATA    = "/vol/data"       # audio files, JSONL, audio_cache/, checkpoints
VOL_HFCACHE = "/vol/hfcache"   # HuggingFace model weights cache

# Training hyper-parameters — edit here to experiment
TRAIN_N          = 500
VAL_N            = 100
MODEL_ID         = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
LORA_RANK        = 4
LORA_ALPHA       = 8
LORA_DROPOUT     = 0.05
LEARNING_RATE    = "5e-5"
NUM_EPOCHS       = 1
MAX_LENGTH       = 320
FOCAL_GAMMA      = 2.0    # passed via env to run_train_custom_loss.py
ENTITY_WEIGHT    = 5.0

# Prompt matching prepare_covost2.py (e2e format, no CoT)
SYSTEM_PROMPT = "You are a speech translation assistant."
USER_PROMPT   = "<audio>Please translate the spoken English into simplified Chinese text only."


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Data preparation
# ══════════════════════════════════════════════════════════════════════════════

@app.function(
    volumes={VOL_DATA: data_vol, VOL_HFCACHE: hfcache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    cpu=4,
    memory=16384,
    timeout=3600,
)
def prepare_data(train_n: int = TRAIN_N, val_n: int = VAL_N):
    """Download CoVoST2 from HuggingFace and write JSONL + .wav files."""
    import json
    import soundfile as sf
    from datasets import load_dataset

    os.environ["HF_HOME"] = VOL_HFCACHE

    audio_train_dir = Path(VOL_DATA) / "audio" / "train"
    audio_val_dir   = Path(VOL_DATA) / "audio" / "val"
    audio_train_dir.mkdir(parents=True, exist_ok=True)
    audio_val_dir.mkdir(parents=True, exist_ok=True)

    def normalize(s):
        return " ".join(s.strip().replace("\n", " ").split())

    def save_wav(audio_obj, path):
        sf.write(str(path), audio_obj["array"], audio_obj["sampling_rate"])

    def make_record(audio_path, translation):
        return {
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": USER_PROMPT},
                {"role": "assistant", "content": translation},
            ],
            "audios": [str(audio_path)],
        }

    def process(ds, audio_dir, jsonl_path, label):
        skipped = 0
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for i in range(len(ds)):
                wav_path = audio_dir / f"{i}.wav"
                try:
                    item = ds[i]
                    save_wav(item["audio"], wav_path)
                    tgt = normalize(item["translation"])
                    if not tgt:
                        continue
                    f.write(json.dumps(make_record(wav_path, tgt), ensure_ascii=False) + "\n")
                except Exception as e:
                    skipped += 1
                    print(f"[{label}] skip {i}: {e}")
                if i % 100 == 0:
                    print(f"[{label}] {i}/{len(ds)} ...", flush=True)
        print(f"[{label}] done — skipped {skipped}", flush=True)

    print(f"Downloading CoVoST2 train[:{train_n}] ...", flush=True)
    train_ds = load_dataset("fixie-ai/covost2", "en_zh-CN", split=f"train[:{train_n}]")
    process(train_ds, audio_train_dir, Path(VOL_DATA) / "train_500.jsonl", "train")

    print(f"Downloading CoVoST2 validation[:{val_n}] ...", flush=True)
    val_ds = load_dataset("fixie-ai/covost2", "en_zh-CN", split=f"validation[:{val_n}]")
    process(val_ds, audio_val_dir, Path(VOL_DATA) / "val_100.jsonl", "val")

    data_vol.commit()
    print("prepare_data done.", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Audio feature extraction
# ══════════════════════════════════════════════════════════════════════════════

@app.function(
    volumes={VOL_DATA: data_vol, VOL_HFCACHE: hfcache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu=modal.gpu.A100(size="80GB"),
    timeout=7200,
)
def extract_features():
    """
    Load Qwen3-Omni, extract audio encoder features for all train/val samples,
    save as .pt files.  Frees the LLM weights after loading so only the audio
    tower stays in GPU memory (~8 GB after cleanup).
    """
    import gc
    import json
    import torch
    import librosa
    from tqdm import tqdm
    from transformers import AutoProcessor, Qwen3OmniMoeForConditionalGeneration

    os.environ["HF_HOME"] = VOL_HFCACHE

    CACHE_DIR = Path(VOL_DATA) / "audio_cache"
    TARGET_SR = 16000

    SPLITS = [
        (Path(VOL_DATA) / "train_500.jsonl", CACHE_DIR / "train"),
        (Path(VOL_DATA) / "val_100.jsonl",   CACHE_DIR / "val"),
    ]

    print("Loading processor ...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    print("Loading model (will trim to audio tower) ...", flush=True)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )

    # Keep only the audio tower; free the rest to save GPU memory
    audio_tower = model.thinker.audio_tower.eval()
    del model.thinker.model, model.thinker.lm_head
    del model.thinker.visual, model.talker, model.code2wav
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU after trim: {torch.cuda.memory_allocated(0)/1e9:.1f} GB", flush=True)

    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        _get_feat_extract_output_lengths,
    )

    for jsonl_path, split_cache_dir in SPLITS:
        if not jsonl_path.exists():
            print(f"Skip (not found): {jsonl_path}", flush=True)
            continue
        split_cache_dir.mkdir(parents=True, exist_ok=True)

        with open(jsonl_path) as f:
            lines = f.readlines()

        path_index: dict[str, int] = {}
        skipped = 0

        for i, line in enumerate(tqdm(lines, desc=jsonl_path.stem)):
            item       = json.loads(line.strip())
            audio_path = item["audios"][0]
            cache_file = split_cache_dir / f"{i}.pt"
            path_index[audio_path] = i

            if cache_file.exists():
                continue
            try:
                audio = librosa.load(audio_path, sr=TARGET_SR)[0]
                feat_out = processor.feature_extractor(
                    audio, sampling_rate=TARGET_SR,
                    return_tensors="pt", return_attention_mask=True,
                )
                attn_mask      = feat_out["attention_mask"].bool()
                feature_lens   = attn_mask.sum(-1).to("cuda")
                input_features = (
                    feat_out["input_features"]
                    .to("cuda", torch.bfloat16)
                    .permute(0, 2, 1)[attn_mask]
                    .permute(1, 0)
                )
                with torch.no_grad():
                    out = audio_tower(input_features, feature_lens=feature_lens)
                feats = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                torch.save(feats.squeeze(0).cpu().half(), cache_file)
                torch.cuda.empty_cache()
            except Exception as e:
                skipped += 1
                print(f"\n[{i}] skip {audio_path}: {e}", flush=True)

        idx_file = split_cache_dir / "path_index.json"
        with open(idx_file, "w") as f:
            json.dump(path_index, f, ensure_ascii=False)
        print(f"{jsonl_path.stem}: {len(lines)-skipped}/{len(lines)} cached → {idx_file}", flush=True)

    data_vol.commit()
    print("extract_features done.", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Training with custom loss
# ══════════════════════════════════════════════════════════════════════════════

@app.function(
    volumes={VOL_DATA: data_vol, VOL_HFCACHE: hfcache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu=modal.gpu.A100(size="40GB"),
    timeout=72000,   # 20 h — enough for 1 epoch on 500 samples
)
def train():
    """Run LoRA fine-tuning with focal + entity-upweighted loss."""
    import subprocess

    os.environ["HF_HOME"]          = VOL_HFCACHE
    os.environ["USE_HF"]           = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # Env vars read by run_train_custom_loss.py
    os.environ["SWIFT_DATA_DIR"]   = VOL_DATA
    os.environ["SWIFT_CACHE_DIR"]  = str(Path(VOL_DATA) / "audio_cache")

    output_dir = str(Path(VOL_DATA) / "output" / "custom_loss_500")
    sys.path.insert(0, "/app/scripts")

    cmd = [
        sys.executable, "/app/scripts/run_train_custom_loss.py",
        "--model",                      MODEL_ID,
        "--use_hf",                     "true",
        "--dataset",                    str(Path(VOL_DATA) / "train_500.jsonl"),
        "--val_dataset",                str(Path(VOL_DATA) / "val_100.jsonl"),
        "--train_type",                 "lora",
        "--torch_dtype",                "float16",
        "--num_train_epochs",           str(NUM_EPOCHS),
        "--per_device_train_batch_size","1",
        "--per_device_eval_batch_size", "1",
        "--gradient_accumulation_steps","1",
        "--learning_rate",              LEARNING_RATE,
        "--max_length",                 str(MAX_LENGTH),
        "--logging_steps",              "10",
        "--save_steps",                 "200",
        "--eval_steps",                 "200",
        "--lora_rank",                  str(LORA_RANK),
        "--lora_alpha",                 str(LORA_ALPHA),
        "--lora_dropout",               str(LORA_DROPOUT),
        "--dataloader_num_workers",     "0",
        "--output_dir",                 output_dir,
    ]

    print("Launching training ...", flush=True)
    print(" ".join(cmd), flush=True)

    result = subprocess.run(cmd, check=False)
    data_vol.commit()

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")
    print("train done.", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Local entrypoint
# ══════════════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main(stage: str = "all"):
    """
    --stage all      run prepare → extract → train  (default)
    --stage prepare  data download only
    --stage extract  audio feature extraction only
    --stage train    training only
    """
    stages = stage.lower().split(",")
    run_all = "all" in stages

    if run_all or "prepare" in stages:
        print("=== Stage 1: prepare_data ===")
        prepare_data.remote()

    if run_all or "extract" in stages:
        print("=== Stage 2: extract_features ===")
        extract_features.remote()

    if run_all or "train" in stages:
        print("=== Stage 3: train ===")
        train.remote()

    print("Pipeline complete.")
