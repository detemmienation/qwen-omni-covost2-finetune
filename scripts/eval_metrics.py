import json
import re
import os

# ===== 路径配置 =====
PREDICTIONS_FILE = "/home/ubuntu/leili-cmu-lab/CMU-project/Y4_data/outputs/predictions_cot_smoothing.json"

# ===== 1. 读取结果 =====
with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
    results = json.load(f)

ok_results  = [r for r in results if r["status"] == "ok" and r["prediction"].strip()]
predictions = [r["prediction"] for r in ok_results]
references  = [r["reference"]  for r in ok_results]
total       = len(results)
n           = len(ok_results)

print(f"Total samples : {total}")
print(f"Valid for eval: {n}")

# ===== 2. 辅助函数 =====

def extract_capitalized_words(text: str):
    """从文本中提取大写开头的词（专有名词近似），排除句首单词"""
    # 匹配连续大写开头的单词（如 Mozilla, Mavis, Google）
    words = re.findall(r'\b[A-Z][a-zA-Z]{1,}\b', text)
    return set(words)

def extract_numbers(text: str):
    """提取数字，数字是关键信息"""
    return set(re.findall(r'\d+\.?\d*', text))

def extract_entities(text: str):
    """提取大写词 + 数字作为关键实体"""
    return extract_capitalized_words(text) | extract_numbers(text)

# ===== 3. ETA：source 实体是否保留在 prediction =====
# 因为没有英文 source，用 reference 里能找到的英文词和数字作为实体
# ETA = 有多少 reference 里的实体出现在了 prediction 里

def compute_eta(refs, preds):
    """
    ETA: Entity Translation Accuracy
    reference 中的关键实体（英文专名、数字）是否出现在 prediction 中
    衡量信息是否被保留
    """
    total_entities = 0
    retained       = 0

    for ref, pred in zip(refs, preds):
        entities = extract_entities(ref)
        if not entities:
            continue
        for e in entities:
            total_entities += 1
            if e in pred:
                retained += 1

    return retained / total_entities * 100 if total_entities > 0 else 0.0, total_entities, retained

# ===== 4. ERR：reference 实体是否被正确翻译到 prediction =====
# ERR = prediction 里有多少 reference 实体被正确保留（精确匹配）
# 衡量语义对齐的准确性

def compute_err(refs, preds):
    """
    ERR: Entity Retention Rate
    计算每条样本：prediction 中保留了多少 reference 实体 / reference 总实体数
    取所有样本的平均
    """
    scores = []
    for ref, pred in zip(refs, preds):
        entities = extract_entities(ref)
        if not entities:
            continue
        matched = sum(1 for e in entities if e in pred)
        scores.append(matched / len(entities))

    return sum(scores) / len(scores) * 100 if scores else 0.0

# ===== 5. 计算 ETA 和 ERR =====
eta_score, total_entities, retained_entities = compute_eta(references, predictions)
err_score = compute_err(references, predictions)

print(f"\n{'='*50}")
print(f"ETA ↑ (Entity Translation Accuracy) : {eta_score:.2f}%")
print(f"      ({retained_entities}/{total_entities} entities retained in predictions)")
print(f"ERR ↑ (Entity Retention Rate)        : {err_score:.2f}%")

# ===== 6. BLEU =====
print(f"\n{'='*50}")
print("Computing BLEU...", flush=True)
from sacrebleu.metrics import BLEU

bleu = BLEU(tokenize="char")
score_bleu = bleu.corpus_score(predictions, [references])
print(f"BLEU ↑ (char-level) : {score_bleu}")

# ===== 7. COMET =====
print(f"\n{'='*50}")
print("Computing COMET...", flush=True)
comet_score = None
try:
    from comet import download_model, load_from_checkpoint

    model_path  = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(model_path)

    comet_data = [
        {"src": r, "mt": p, "ref": r}
        for p, r in zip(predictions, references)
    ]
    comet_output = comet_model.predict(comet_data, batch_size=32, gpus=1)
    comet_score  = comet_output.system_score
    print(f"COMET ↑ : {comet_score:.4f}")
except Exception as e:
    print(f"COMET failed: {e}")

# ===== 8. 汇总 =====
print(f"\n{'='*50}")
print("SUMMARY")
print(f"{'='*50}")
print(f"  Total samples : {total}")
print(f"  Valid for eval: {n}")
print(f"  ETA  ↑        : {eta_score:.2f}%")
print(f"  ERR  ↑        : {err_score:.2f}%")
print(f"  BLEU ↑        : {score_bleu}")
if comet_score is not None:
    print(f"  COMET ↑       : {comet_score:.4f}")
print(f"{'='*50}")

# ===== 9. 保存 =====
out_path = os.path.join(os.path.dirname(PREDICTIONS_FILE), "eval_metrics.json")
metrics = {
    "total":        total,
    "valid":        n,
    "ETA":          round(eta_score, 4),
    "ERR":          round(err_score, 4),
    "BLEU":         str(score_bleu),
    "COMET":        round(comet_score, 4) if comet_score is not None else None,
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print(f"\nMetrics saved to {out_path}")