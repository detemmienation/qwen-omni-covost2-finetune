import os
os.environ["TRANSFORMERS_OFFLINE"]  = "1"
os.environ["HF_DATASETS_OFFLINE"]   = "1"
os.environ["CUDA_LAUNCH_BLOCKING"]  = "1"

import json
import time
import torch
import soundfile as sf
from datasets import load_dataset
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

BASE_MODEL  = "Qwen/Qwen2.5-Omni-7B"
DATA_FILE   = "/home/ubuntu/leili-cmu-lab/CMU-project/Y4_data/data/test.jsonl"
OUTPUT_FILE = "/home/ubuntu/leili-cmu-lab/CMU-project/Y4_data/outputs/predictions_baseline_qwen25omni.json"
MAX_SAMPLES = 200

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

print("Loading processor...", flush=True)
processor = Qwen2_5OmniProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
VOCAB_SIZE = len(processor.tokenizer)

print("Loading model (bfloat16)...", flush=True)
model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print("Model ready.", flush=True)

TAIL_TOKENS = ["⽗", "父", "<|im_end|>", "<|endoftext|>"]

SYSTEM_PROMPT = "You are a speech translation assistant."
USER_PROMPT = "Please translate the spoken English into simplified Chinese text only."

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
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("translation:"):
            return line[len("translation:"):].strip()
    return text.strip()

def infer(item):
    audio_path = item["audios"][0]
    reference  = [m["content"] for m in item["messages"] if m["role"] == "assistant"][-1]
    audio, sr  = sf.read(audio_path)

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text",  "text": USER_PROMPT},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False,
    )
    inputs = processor(
        text=text_prompt, audio=[audio], sampling_rate=sr,
        return_tensors="pt", padding=True,
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
            repetition_penalty=1.3,
            use_audio_in_video=False,
        )

    output_ids = outputs[0] if isinstance(outputs, tuple) else outputs
    input_len  = inputs["input_ids"].shape[1]
    gen_ids    = output_ids[:, input_len:].clamp(0, VOCAB_SIZE - 1)

    prediction = processor.batch_decode(
        gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]
    prediction = clean_prediction(prediction)

    torch.cuda.empty_cache()
    return prediction, reference


print("Loading dataset...", flush=True)
dataset = load_dataset("json", data_files=DATA_FILE)["train"]
dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))
print(f"Running inference on {len(dataset)} samples...\n", flush=True)

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

    latency = round(time.time() - t0, 3)
    results.append({
        "id":         i,
        "audio":      item["audios"][0],
        "reference":  ref,
        "prediction": pred,
        "status":     status,
        "latency_s":  latency,
    })
    print(
        f"[{i+1:4d}/{len(dataset)}] {latency:.1f}s | {status}"
        f" | ref: {ref[:35]} | pred: {pred[:35]}",
        flush=True,
    )

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

ok_count  = sum(1 for r in results if r["status"] == "ok")
latencies = [r["latency_s"] for r in results if r["status"] == "ok"]
print(f"\nDone. {ok_count}/{len(dataset)} successful in {(time.time()-t_total)/60:.1f} min", flush=True)
if latencies:
    print(f"Latency: mean={sum(latencies)/len(latencies):.2f}s  min={min(latencies):.2f}s  max={max(latencies):.2f}s", flush=True)
print(f"Saved to {OUTPUT_FILE}", flush=True)
