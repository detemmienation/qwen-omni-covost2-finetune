import os
import json
import soundfile as sf
from datasets import load_dataset
BASE_DIR = "/home/ubuntu/leili-cmu-lab/CMU-project/Y4_data"
DATA_DIR = os.path.join(BASE_DIR, "data")
# ===== prompt =====
SYSTEM_PROMPT = "You are a speech translation assistant."
USER_PROMPT = "<audio>Please translate the spoken English into simplified Chinese text only."

# 1. 路径改成 test 专用
AUDIO_TEST_DIR = os.path.join(DATA_DIR, "audio/test")
TEST_JSONL = os.path.join(DATA_DIR, "test.jsonl")
os.makedirs(AUDIO_TEST_DIR, exist_ok=True)

# 2. load test split
test_ds = load_dataset("fixie-ai/covost2", "en_zh-CN", split="test")

# 3. 数量对齐 seamless（跑完整 test，或者先取500条验证）
TEST_SIZE = 1000
test_ds = test_ds.select(range(TEST_SIZE))

def build_record(audio_path, tgt_text):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
            {"role": "assistant", "content": tgt_text},
        ],
        "audios": [audio_path]
    }

def normalize_text(s):
    s = s.strip()
    s = s.replace("\n", " ")
    s = " ".join(s.split())
    return s
    
def save_audio(audio_obj, path):
    array = audio_obj["array"]
    sr = audio_obj["sampling_rate"]
    sf.write(path, array, sr)

# 4. 处理逻辑和之前 val 一样，路径换成 test
print("Processing test...")
skipped = 0
with open(TEST_JSONL, "w", encoding="utf-8") as f:
    for i in range(len(test_ds)):
        audio_path = os.path.join(AUDIO_TEST_DIR, f"{i}.wav")
        try:
            item = test_ds[i]
            save_audio(item["audio"], audio_path)
            tgt = normalize_text(item["translation"])
            if not tgt:
                continue
            record = build_record(audio_path, tgt)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            skipped += 1
            print(f"skip {i}: {e}")
            continue
print(f"Done. skipped {skipped}/{len(test_ds)}")