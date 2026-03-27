import os
import json
import soundfile as sf
from datasets import load_dataset

# ===== 路径配置 =====
BASE_DIR = "/home/ubuntu/project"
DATA_DIR = os.path.join(BASE_DIR, "data")
AUDIO_TRAIN_DIR = os.path.join(DATA_DIR, "audio/train")
AUDIO_VAL_DIR = os.path.join(DATA_DIR, "audio/val")

TRAIN_JSONL = os.path.join(DATA_DIR, "train.jsonl")
VAL_JSONL = os.path.join(DATA_DIR, "val.jsonl")

os.makedirs(AUDIO_TRAIN_DIR, exist_ok=True)
os.makedirs(AUDIO_VAL_DIR, exist_ok=True)

# ===== prompt =====
SYSTEM_PROMPT = "You are a speech translation assistant."
USER_PROMPT = "<audio>Please translate the spoken English into simplified Chinese text only."

def normalize_text(s):
    s = s.strip()
    s = s.replace("\n", " ")
    s = " ".join(s.split())
    return s

def save_audio(audio_obj, path):
    array = audio_obj["array"]
    sr = audio_obj["sampling_rate"]
    sf.write(path, array, sr)

def build_record(audio_path, tgt_text):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
            {"role": "assistant", "content": tgt_text},
        ],
        "audios": [audio_path]
    }

# ===== 主逻辑 =====
print("Loading CoVoST2...")

train_ds = load_dataset("fixie-ai/covost2", "en_zh-CN", split="train")
val_ds = load_dataset("fixie-ai/covost2", "en_zh-CN", split="validation")

# 👉 只取小数据（先跑通）
train_ds = train_ds.select(range(200))
val_ds = val_ds.select(range(50))

print("Processing train...")

with open(TRAIN_JSONL, "w", encoding="utf-8") as f:
    for i, item in enumerate(train_ds):
        audio_path = os.path.join(AUDIO_TRAIN_DIR, f"{i}.wav")

        try:
            save_audio(item["audio"], audio_path)
        except Exception as e:
            print("skip audio:", e)
            continue

        tgt = normalize_text(item["translation"])
        if not tgt:
            continue

        record = build_record(audio_path, tgt)
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("Processing val...")

with open(VAL_JSONL, "w", encoding="utf-8") as f:
    for i, item in enumerate(val_ds):
        audio_path = os.path.join(AUDIO_VAL_DIR, f"{i}.wav")

        try:
            save_audio(item["audio"], audio_path)
        except Exception as e:
            print("skip audio:", e)
            continue

        tgt = normalize_text(item["translation"])
        if not tgt:
            continue

        record = build_record(audio_path, tgt)
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("Done!")
print("Train JSON:", TRAIN_JSONL)
print("Val JSON:", VAL_JSONL)