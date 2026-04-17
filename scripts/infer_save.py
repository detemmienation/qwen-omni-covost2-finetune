import os
os.environ["TRANSFORMERS_OFFLINE"]  = "1"
os.environ["HF_DATASETS_OFFLINE"]   = "1"
os.environ["CUDA_LAUNCH_BLOCKING"]  = "1"

import json
import time
import torch
import soundfile as sf
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3OmniMoeForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

# ===== 路径配置 =====
BASE_MODEL  = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
LORA_PATH   = "/home/ubuntu/leili-cmu-lab/CMU-project/Y4_data/output/full_e2e/v10-20260417-014453/checkpoint-625"
DATA_FILE   = "/home/ubuntu/leili-cmu-lab/CMU-project/Y4_data/data/test.jsonl"
OUTPUT_FILE = "/home/ubuntu/leili-cmu-lab/CMU-project/Y4_data/outputs/predictions_cot_smoothing.json"
OFFLOAD_DIR = "/home/ubuntu/leili-cmu-lab/CMU-project/Y4_data/offload"
os.makedirs(OFFLOAD_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ===== 1. Processor =====
print("Loading processor...", flush=True)
processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

# ★ 用 len(tokenizer) 而不是 vocab_size，Qwen 里两者不一样
VOCAB_SIZE = len(processor.tokenizer)
print(f"Vocab size (len): {VOCAB_SIZE}", flush=True)

print("Loading base model (bfloat16)...", flush=True)
model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    enable_audio_output=False,   # ★ 关键：禁用音频输出，不加载 token2wav
)
print(f"GPU after base model: {torch.cuda.memory_allocated(0)/1e9:.2f} GB", flush=True)

# ===== 3. LoRA =====
print("Loading LoRA...", flush=True)
model = PeftModel.from_pretrained(model, LORA_PATH)
print(f"GPU after LoRA: {torch.cuda.memory_allocated(0)/1e9:.2f} GB", flush=True)

model.eval()
print("Model ready.", flush=True)

# ===== 4. 后处理：去掉末尾残留的特殊 token =====
TAIL_TOKENS = ["⽗", "父", "<|im_end|>", "<|endoftext|>"]

SYSTEM_PROMPT = "You are a speech translation assistant."
USER_PROMPT = (
    "Step 1 – Transcription: Listen carefully and write down the exact "
    "English words spoken in the audio.\n"
    "Step 2 – Translation: Translate the transcription into natural Chinese.\n\n"
    "Use this exact output format:\n"
    "Transcription: <English text>\n"
    "Translation: <Chinese text>"
)

def clean_prediction(text: str) -> str:
    text = text.strip()
    changed = True
    while changed:
        changed = False
        for tail in TAIL_TOKENS:
            if text.endswith(tail):
                text = text[:-len(tail)].strip()
                changed = True
    return text

def parse_cot(text: str) -> str:
    """Extract only the Translation line from CoT output."""
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("translation:"):
            return line[len("translation:"):].strip()
    # Fallback: return full text if format not found
    return text.strip()

# ===== 5. 推理函数 =====
def infer(item):
    audio_path = item["audios"][0]
    reference  = [m["content"] for m in item["messages"] if m["role"] == "assistant"][-1]

    audio, sr = sf.read(audio_path)

    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text",  "text": USER_PROMPT},
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
        k: (v.to(device=device, dtype=torch.bfloat16)
            if isinstance(v, torch.Tensor) and v.dtype in (torch.float32, torch.float64)
            else v.to(device=device) if isinstance(v, torch.Tensor)
            else v)
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            use_audio_in_video=False,
            return_audio=False,       # ★ 加这行，明确不返回音频
        )

    # outputs 是 tuple，取第 0 个
    output_ids = outputs[0] if isinstance(outputs, tuple) else outputs
    input_len  = inputs["input_ids"].shape[1]
    gen_ids    = output_ids[:, input_len:]

    # ★ clamp 用正确的 vocab size 防止越界
    gen_ids = gen_ids.clamp(0, VOCAB_SIZE - 1)

    prediction = processor.batch_decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    # ★ 去掉末尾残留的特殊 token 解码字符
    prediction = parse_cot(clean_prediction(prediction))

    torch.cuda.empty_cache()

    return prediction, reference


# ===== 6. 跑全部 =====
print("Loading dataset...", flush=True)
dataset = load_dataset("json", data_files=DATA_FILE)["train"]
print(f"Dataset size: {len(dataset)}, starting inference...\n", flush=True)

results = []
t_total = time.time()

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
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
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

# ===== 7. 保存 =====
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

ok_count = sum(1 for r in results if r["status"] == "ok")
print(f"\nDone. {ok_count}/{len(dataset)} successful in {(time.time()-t_total)/60:.1f} min", flush=True)
print(f"Saved to {OUTPUT_FILE}", flush=True)