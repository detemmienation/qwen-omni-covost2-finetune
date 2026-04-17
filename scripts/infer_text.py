import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"

import json
import time
import argparse
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3OmniMoeForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel


def normalize_text(s: str) -> str:
    s = s.strip()
    s = s.replace("\n", " ")
    s = " ".join(s.split())
    return s


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",     type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--lora_path",      type=str, default="/home/ubuntu/leili-cmu-lab/CMU-project/Y4_data/output/text_translation_omni_1000/checkpoint-500")
    parser.add_argument("--input_jsonl",    type=str, required=True)
    parser.add_argument("--output_jsonl",   type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    # ===== 1. Processor =====
    print("Loading processor...", flush=True)
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)

    # ===== 2. 4bit 量化加载（16GB 显存稳定跑，不需要 offload） =====
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print("Loading base model (4bit)...", flush=True)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # ===== 3. LoRA（不传 device_map） =====
    print("Loading LoRA...", flush=True)
    model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval()
    print(f"Model ready. GPU: {torch.cuda.memory_allocated(0)/1e9:.2f} GB", flush=True)

    device = torch.device("cuda:0")

    # ===== 4. 推理循环 =====
    results = []
    total   = 0
    ok      = 0

    for obj in iter_jsonl(args.input_jsonl):
        total += 1

        # pred_zh 字段存的是 ASR 识别出的英文，作为翻译输入
        src_en = normalize_text(obj.get("pred_zh", ""))
        ref_zh = normalize_text(obj.get("ref_zh", ""))

        if not src_en or not ref_zh:
            results.append({
                "idx":      obj.get("idx", total - 1),
                "src_en":   src_en,
                "ref_zh":   ref_zh,
                "pred_zh":  "",
                "error":    "empty src/ref",
            })
            print(f"[{total:>3}] skip: empty", flush=True)
            continue

        try:
            # prompt 格式和训练数据完全对齐
            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a translation assistant. Translate the input English text into simplified Chinese only."}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"Translate into Chinese: {src_en}"}],
                },
            ]
            prompt = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )

            inputs = processor(text=prompt, return_tensors="pt")
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            input_len = inputs["input_ids"].shape[1]

            t0 = time.time()
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )
            dt = time.time() - t0

            # outputs 可能是 tuple（Omni 模型），取第 0 个
            output_ids = outputs[0] if isinstance(outputs, tuple) else outputs
            gen_ids    = output_ids[:, input_len:]
            pred       = processor.batch_decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            results.append({
                "idx":         obj.get("idx", total - 1),
                "src_en":      src_en,
                "ref_zh":      ref_zh,
                "pred_zh":     pred,
                "latency_sec": round(dt, 3),
            })
            ok += 1
            print(f"[{total:>3}] {dt:.1f}s | ref: {ref_zh[:35]} | pred: {pred[:50]}", flush=True)

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            results.append({"idx": obj.get("idx", total-1), "src_en": src_en, "ref_zh": ref_zh, "pred_zh": "", "error": "OOM"})
            print(f"[{total:>3}] OOM", flush=True)

        except Exception as e:
            import traceback
            traceback.print_exc()   # 打印完整报错栈，方便定位
            results.append({"idx": obj.get("idx", total-1), "src_en": src_en, "ref_zh": ref_zh, "pred_zh": "", "error": str(e)})
            print(f"[{total:>3}] ERR: {e}", flush=True)

    # ===== 5. 保存 =====
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone. {ok}/{total} successful")
    print(f"Saved to {args.output_jsonl}")


if __name__ == "__main__":
    main()