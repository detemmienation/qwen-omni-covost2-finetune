"""
Build CoT-format training data by merging:
  - audio paths from existing train.jsonl / val_500.jsonl
  - English sentences + Chinese translations from CoVoST2 arrow cache

Output format (assistant):
    Transcription: <English text>
    Translation: <Chinese text>

Usage:
    python make_cot_data.py                  # default: 10000 train, 500 val
    python make_cot_data.py --train_n 5000 --val_n 200
"""

import argparse
import json
import glob
from datasets import load_dataset

CACHE_DIR = "/home/ubuntu/.cache/huggingface/datasets/fixie-ai___covost2/en_zh-CN/0.0.0/17c8c81e331e7a6929118121771a58c7ef7331d8"
DATA_DIR  = "/home/ubuntu/leili-cmu-lab/CMU-project/Y4_data/data"

SYSTEM_PROMPT = "You are a speech translation assistant."
USER_PROMPT = (
    "Step 1 – Transcription: Listen carefully and write down the exact "
    "English words spoken in the audio.\n"
    "Step 2 – Translation: Translate the transcription into natural Chinese.\n\n"
    "Use this exact output format:\n"
    "Transcription: <English text>\n"
    "Translation: <Chinese text>"
)

def load_covost2_split(split_prefix):
    files = sorted(glob.glob(f"{CACHE_DIR}/covost2-{split_prefix}-*.arrow"))
    return load_dataset("arrow", data_files=files, split="train")

def convert(src_jsonl, covost2_split_prefix, out_jsonl, n, offset=0):
    with open(src_jsonl) as f:
        src_items = [json.loads(l) for l in f]
    src_items = src_items[offset:offset + n]

    ds = load_covost2_split(covost2_split_prefix)
    assert len(ds) >= offset + len(src_items), \
        f"CoVoST2 has {len(ds)} entries but requested offset={offset} n={n}"

    first_zh = src_items[0]["messages"][-1]["content"]
    if ds[offset]["translation"] != first_zh:
        print(f"WARNING: translation mismatch at index {offset}")
        print(f"  src:     {first_zh[:60]}")
        print(f"  covost2: {ds[offset]['translation'][:60]}")

    out = []
    for i, item in enumerate(src_items):
        en = ds[offset + i]["sentence"]
        zh = ds[offset + i]["translation"]
        out.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": f"<audio>{USER_PROMPT}"},
                {"role": "assistant", "content": f"Transcription: {en}\nTranslation: {zh}"},
            ],
            "audios": item["audios"],
        })

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out)} records → {out_jsonl}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_n",      type=int, default=10000)
    parser.add_argument("--train_offset", type=int, default=0)
    parser.add_argument("--val_n",        type=int, default=500)
    args = parser.parse_args()

    suffix = f"{args.train_offset}_{args.train_offset + args.train_n}" if args.train_offset else str(args.train_n)
    convert(
        f"{DATA_DIR}/train.jsonl", "train",
        f"{DATA_DIR}/train_{suffix}_cot.jsonl",
        n=args.train_n, offset=args.train_offset,
    )
    convert(
        f"{DATA_DIR}/val_500.jsonl", "validation",
        f"{DATA_DIR}/val_{args.val_n}_cot.jsonl",
        n=args.val_n,
    )
