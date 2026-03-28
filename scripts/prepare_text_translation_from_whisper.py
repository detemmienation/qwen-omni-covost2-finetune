import json
from pathlib import Path

TRAIN_DIR = Path("/home/ubuntu/project/data/whisper/training")
VAL_DIR = Path("/home/ubuntu/project/data/whisper/validation")

TRAIN_OUT = Path("/home/ubuntu/project/data/text_train_1000.jsonl")
VAL_OUT = Path("/home/ubuntu/project/data/text_val_200.jsonl")

TRAIN_LIMIT = 1000
VAL_LIMIT = 200

SYSTEM_PROMPT = "You are a translation assistant. Translate the input English text into simplified Chinese only."
def normalize_text(s: str) -> str:
    s = s.strip()
    s = s.replace("\n", " ")
    s = " ".join(s.split())
    return s

def iter_jsonl_files(folder: Path):
    for path in sorted(folder.glob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

def convert_record(obj):
    src = normalize_text(obj["pred_zh"])   # Whisper 输出的英文
    tgt = normalize_text(obj["ref_zh"])    # 中文参考

    if not src or not tgt:
        return None

    rec = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Translate into Chinese: {src}"},
            {"role": "assistant", "content": tgt},
        ]
    }
    return rec

def collect_records(folder: Path, limit: int):
    records = []
    for obj in iter_jsonl_files(folder):
        rec = convert_record(obj)
        if rec is not None:
            records.append(rec)
        if len(records) >= limit:
            break
    return records

def write_jsonl(records, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    train_records = collect_records(TRAIN_DIR, TRAIN_LIMIT)
    val_records = collect_records(VAL_DIR, VAL_LIMIT)

    write_jsonl(train_records, TRAIN_OUT)
    write_jsonl(val_records, VAL_OUT)

    print(f"train records: {len(train_records)} -> {TRAIN_OUT}")
    print(f"val records:   {len(val_records)} -> {VAL_OUT}")

if __name__ == "__main__":
    main()