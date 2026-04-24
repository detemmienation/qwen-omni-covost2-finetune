"""
Evaluate BLEU, COMET (with real English src), chrF, Hallucination Rate,
and Content Coverage for JSONL prediction files.

Handles multiple field-name conventions and merges chunked files.

Usage (single file or glob):
    python scripts/eval_jsonl.py --inputs "outputs/prev_output/seamless/outputs/pred_*.jsonl" \
                                  --name seamless
    python scripts/eval_jsonl.py --inputs outputs/prev_output/Qwen2.5-omni/outputs/pred_qwen2_5_omni_7b_results.jsonl \
                                  --name qwen25omni
"""

import json
import re
import os
import glob
import argparse
from collections import Counter

# ── Args ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--inputs", required=True,
                    help="Glob pattern or space-separated paths to JSONL files")
parser.add_argument("--name", required=True, help="Short model name for output file")
parser.add_argument("--out_dir", default=None, help="Output directory (default: same as first input)")
args = parser.parse_args()

# Expand glob
input_files = sorted(glob.glob(args.inputs))
if not input_files:
    # Try as literal path
    input_files = [p.strip() for p in args.inputs.split() if os.path.exists(p.strip())]
if not input_files:
    raise FileNotFoundError(f"No files matched: {args.inputs}")

print(f"Files : {input_files}")

# ── Load & normalise ──────────────────────────────────────────
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def get_field(d, *candidates):
    for k in candidates:
        if k in d and d[k] is not None:
            v = d[k]
            return str(v).strip() if not isinstance(v, str) else v.strip()
    return ""

records = []
for path in input_files:
    for d in load_jsonl(path):
        pred = get_field(d, "pred_zh", "pred")
        ref  = get_field(d, "ref_zh",  "ref")
        src  = get_field(d, "sentence_en_gt", "sentence_en", "src_en", "en")
        if pred and ref:
            records.append({"pred": pred, "ref": ref, "src": src})

print(f"Loaded : {len(records)} valid samples")
if not records:
    raise ValueError("No valid pred/ref pairs found")

predictions = [r["pred"] for r in records]
references  = [r["ref"]  for r in records]
sources     = [r["src"]  for r in records]

# ── 1. chrF ───────────────────────────────────────────────────
print("\nComputing chrF...", flush=True)
from sacrebleu.metrics import CHRF
chrf_score = CHRF().corpus_score(predictions, [references])
print(f"chrF ↑ : {chrf_score.score:.2f}")

# ── 2. Hallucination Rate & Content Coverage ──────────────────
def tokenize(text):
    return re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]|[A-Za-z0-9]+', text)

hall_rates, coverages = [], []
for pred, ref in zip(predictions, references):
    pt, rt = tokenize(pred), tokenize(ref)
    if not pt or not rt:
        continue
    pc, rc  = Counter(pt), Counter(rt)
    matched = sum(min(pc[t], rc[t]) for t in pc)
    hall_rates.append(1.0 - matched / len(pt))
    coverages.append(matched / len(rt))

hall_rate   = sum(hall_rates) / len(hall_rates) * 100
content_cov = sum(coverages)  / len(coverages)  * 100
print(f"Hallucination Rate ↓ : {hall_rate:.2f}%")
print(f"Content Coverage   ↑ : {content_cov:.2f}%")

# ── Summary ───────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  Model  : {args.name}")
print(f"  Samples: {len(records)}")
print(f"  chrF ↑            : {chrf_score.score:.2f}")
print(f"  Hallucination ↓   : {hall_rate:.2f}%")
print(f"  Content Cov ↑     : {content_cov:.2f}%")
print(f"{'='*55}")

# ── Save ──────────────────────────────────────────────────────
out_dir  = args.out_dir or os.path.dirname(input_files[0])
out_path = os.path.join(out_dir, f"eval_{args.name}.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "model":              args.name,
        "files":              input_files,
        "samples":            len(records),
        "chrF":               round(chrf_score.score, 4),
        "hallucination_rate": round(hall_rate, 4),
        "content_coverage":   round(content_cov, 4),
    }, f, ensure_ascii=False, indent=2)
print(f"Saved to {out_path}")
