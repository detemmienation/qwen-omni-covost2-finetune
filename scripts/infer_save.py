import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"

import json
import time
import torch
import soundfile as sf
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

# ===== 路径配置 =====
BASE_MODEL  = "Qwen/Qwen2.5-Omni-7B"
LORA_PATH   = "/home/ubuntu/project/output/run_500/v0-20260327-185755/checkpoint-500"
# DATA_FILE   = "/home/ubuntu/project/data/val_100.jsonl"
# OUTPUT_FILE = "/home/ubuntu/project/predictions.json"

# 用text集来推理
DATA_FILE   = "/home/ubuntu/project/data/test.jsonl"     
# 输出路径
OUTPUT_FILE = "/home/ubuntu/project/predictions_test.json" 
OFFLOAD_DIR = "/home/ubuntu/project/offload"
os.makedirs(OFFLOAD_DIR, exist_ok=True)

# ===== 1. Processor =====
print("Loading processor...", flush=True)
processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

# ===== 2. 4bit 量化加载 =====
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print("Loading base model (4bit)...", flush=True)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
print(f"GPU after base model: {torch.cuda.memory_allocated(0)/1e9:.2f} GB", flush=True)

# ===== 3. LoRA =====
print("Loading LoRA...", flush=True)
model = PeftModel.from_pretrained(model, LORA_PATH)
print(f"GPU after LoRA: {torch.cuda.memory_allocated(0)/1e9:.2f} GB", flush=True)

model.eval()
print("Model ready.", flush=True)

# ===== 4. 推理函数 =====
def infer(item):
    messages   = item["messages"]
    audio_path = item["audios"][0]

    user_text = [m["content"] for m in messages if m["role"] == "user"][-1]
    user_text = user_text.replace("<audio>", "").strip()
    reference = [m["content"] for m in messages if m["role"] == "assistant"][-1]

    audio, sr = sf.read(audio_path)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text",  "text": user_text},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = processor(
        text=text_prompt,
        audio=[audio],
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
    )

    device = torch.device("cuda:0")
    inputs = {
        k: (v.to(device=device, dtype=torch.float16)
            if isinstance(v, torch.Tensor) and v.dtype in (torch.float32, torch.float64)
            else v.to(device=device) if isinstance(v, torch.Tensor)
            else v)
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            use_audio_in_video=False,
        )

    # outputs 是 tuple，第 0 个是 text token ids
    output_ids = outputs[0]
    input_len  = inputs["input_ids"].shape[1]
    prediction = processor.batch_decode(
        output_ids[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return prediction, reference


# ===== 5. 跑全部 =====
print("Loading dataset...", flush=True)
dataset = load_dataset("json", data_files=DATA_FILE)["train"]
print(f"Dataset size: {len(dataset)}, starting inference...\n", flush=True)

results  = []
t_total  = time.time()

for i, item in enumerate(dataset):
    t0 = time.time()
    try:
        pred, ref = infer(item)
        status = "ok"
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        pred   = ""
        ref    = [m["content"] for m in item["messages"] if m["role"] == "assistant"][-1]
        status = "OOM"
    except Exception as e:
        pred   = ""
        ref    = [m["content"] for m in item["messages"] if m["role"] == "assistant"][-1]
        status = f"ERR:{e}"

    results.append({
        "id":         i,
        "audio":      item["audios"][0],
        "reference":  ref,
        "prediction": pred,
        "status":     status,
    })
    print(
        f"[{i+1:3d}/{len(dataset)}] {time.time()-t0:.1f}s | {status}"
        f" | ref: {ref[:35]} | pred: {pred[:35]}",
        flush=True
    )

# ===== 6. 保存 =====
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

ok_count = sum(1 for r in results if r["status"] == "ok")
print(f"\nDone. {ok_count}/{len(dataset)} successful in {(time.time()-t_total)/60:.1f} min", flush=True)
print(f"Saved to {OUTPUT_FILE}", flush=True)