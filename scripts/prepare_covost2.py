import os
import json
import soundfile as sf
from datasets import load_dataset

# ===== 路径配置 =====
BASE_DIR = "/home/ubuntu/leili-cmu-lab/CMU-project/Y4_data"
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

# ===== 数据配置 =====
# 全量数据，如需子集可取消注释以下几行
# TRAIN_SIZE = 2000
# VAL_SIZE = 200
# train_ds = train_ds.select(range(TRAIN_SIZE))
# val_ds = val_ds.select(range(VAL_SIZE))

def process_split(ds, audio_dir, jsonl_path, split_name):
    skipped = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i in range(len(ds)):
            audio_path = os.path.join(audio_dir, f"{i}.wav")
            try:
                item = ds[i]
                save_audio(item["audio"], audio_path)
                tgt = normalize_text(item["translation"])
                if not tgt:
                    continue
                record = build_record(audio_path, tgt)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                skipped += 1
                print(f"[{split_name}] skip {i}: {e}")
                continue
            if i % 5000 == 0:
                print(f"[{split_name}] {i}/{len(ds)} ...")
    print(f"[{split_name}] done, skipped {skipped}")

print("Processing train...")
process_split(train_ds, AUDIO_TRAIN_DIR, TRAIN_JSONL, "train")

print("Processing val...")
process_split(val_ds, AUDIO_VAL_DIR, VAL_JSONL, "val")

print("Done!")
print("Train JSON:", TRAIN_JSONL)
print("Val JSON:", VAL_JSONL)