"""
Compute Hallucination Rate, chrF, and Content Coverage for speech translation predictions.

Definitions:
  - chrF         : character n-gram F-score (sacrebleu, β=1, char_order=6)
  - Hallucination: fraction of predicted tokens NOT supported by the reference
                   = 1 - token-level precision  (↓ is better)
  - Content Cov  : fraction of reference tokens covered by the prediction
                   = token-level recall          (↑ is better)

Tokenization: CJK characters individually; ASCII alphanumeric words as units;
              punctuation ignored — so Chinese and mixed en/zh text both work.

Usage:
    python scripts/eval_intrinsic.py --input outputs/cot/predictions_v16_cot.json
    python scripts/eval_intrinsic.py --input outputs/predictions_seamless.json
"""

import json
import re
import os
import argparse
from collections import Counter

# ── Args ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Path to predictions JSON file")
args = parser.parse_args()

# ── Load ──────────────────────────────────────────────────────
with open(args.input, "r", encoding="utf-8") as f:
    results = json.load(f)

ok = [r for r in results if r["status"] == "ok" and r["prediction"].strip()]
predictions = [r["prediction"] for r in ok]
references  = [r["reference"]  for r in ok]

print(f"Total : {len(results)}")
print(f"Valid : {len(ok)}")

# ── Tokenizer ─────────────────────────────────────────────────
def tokenize(text: str):
    """CJK chars individually; ASCII alphanumeric words as units; skip punctuation."""
    return re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]|[A-Za-z0-9]+', text)

# ── 1. chrF ───────────────────────────────────────────────────
print("\nComputing chrF...", flush=True)
from sacrebleu.metrics import CHRF

chrf_metric = CHRF()
chrf_score  = chrf_metric.corpus_score(predictions, [references])
print(f"chrF ↑ : {chrf_score.score:.2f}")

# ── 2. Hallucination Rate & Content Coverage ──────────────────
print("\nComputing Hallucination Rate & Content Coverage...", flush=True)

hall_rates = []
coverages  = []

for pred, ref in zip(predictions, references):
    pred_tok = tokenize(pred)
    ref_tok  = tokenize(ref)

    if not pred_tok or not ref_tok:
        continue

    pred_cnt = Counter(pred_tok)
    ref_cnt  = Counter(ref_tok)

    # clipped matches (same as BLEU precision clipping)
    matched = sum(min(pred_cnt[t], ref_cnt[t]) for t in pred_cnt)

    precision = matched / len(pred_tok)   # how much of pred is grounded in ref
    recall    = matched / len(ref_tok)    # how much of ref is covered by pred

    hall_rates.append(1.0 - precision)
    coverages.append(recall)

hall_rate   = sum(hall_rates) / len(hall_rates) * 100
content_cov = sum(coverages)  / len(coverages)  * 100

print(f"Hallucination Rate ↓ : {hall_rate:.2f}%")
print(f"Content Coverage   ↑ : {content_cov:.2f}%")

# ── Summary ───────────────────────────────────────────────────
print(f"\n{'='*50}")
print("SUMMARY")
print(f"{'='*50}")
print(f"  chrF               ↑ : {chrf_score.score:.2f}")
print(f"  Hallucination Rate ↓ : {hall_rate:.2f}%")
print(f"  Content Coverage   ↑ : {content_cov:.2f}%")
print(f"{'='*50}")

# ── Save ──────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(args.input), "eval_intrinsic.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "input":             args.input,
        "valid":             len(ok),
        "chrF":              round(chrf_score.score, 4),
        "hallucination_rate": round(hall_rate, 4),
        "content_coverage":  round(content_cov, 4),
    }, f, ensure_ascii=False, indent=2)
print(f"Saved to {out_path}")
