import os
import json
import time
import torch
import soundfile as sf
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration
from peft import PeftModel

# ===== 1. base model + LoRA adapter =====
base_model = "Qwen/Qwen2.5-Omni-7B"
lora_path = "/home/ubuntu/project/output/run_500/v0-20260327-185755/checkpoint-500"
offload_dir = "/home/ubuntu/project/offload"

os.makedirs(offload_dir, exist_ok=True)

print("Loading processor...", flush=True)
processor = AutoProcessor.from_pretrained(
    base_model,
    trust_remote_code=True,
)

print("Loading base model...", flush=True)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    base_model,
    dtype=torch.float16,          # ← 用 dtype 不用 torch_dtype（你代码里有 deprecation warning）
    device_map="auto",
    offload_folder=offload_dir,
    offload_buffers=True,
    trust_remote_code=True,
)

print("Loading LoRA adapter...", flush=True)
model = PeftModel.from_pretrained(
    model,
    lora_path,
    dtype=torch.float16,
    device_map="auto",
    offload_folder=offload_dir,
)

model.eval()
print("Model loaded successfully.", flush=True)

# ===== 2. load eval dataset =====
print("Loading dataset...", flush=True)
dataset = load_dataset(
    "json",
    data_files="/home/ubuntu/project/data/val_100.jsonl"
)["train"]

print(f"Dataset size: {len(dataset)}", flush=True)

# ===== 3. 推理函数 =====
def run_inference(item):
    messages = item["messages"]
    audio_path = item["audios"][0]

    # 取 user 文本（去掉 <audio> 标签，processor 会处理）
    user_text = [m["content"] for m in messages if m["role"] == "user"][-1]
    reference = [m["content"] for m in messages if m["role"] == "assistant"][-1]

    audio, sr = sf.read(audio_path)

    # ★ 关键修复：用 chat template 格式构造输入
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},   # 传路径或 ndarray 都行
                {"type": "text", "text": user_text.replace("<audio>", "").strip()},
            ],
        }
    ]

    # apply_chat_template 生成文本，processor 处理音频
    text_prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )

    t0 = time.time()
    inputs = processor(
        text=text_prompt,
        audio=[audio],                # ← 必须是列表
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
    )
    print(f"Processor done in {time.time() - t0:.2f}s", flush=True)

    # ★ 关键修复：统一转 float16，并移动到 cuda:0
    #    不要依赖 get_input_device，直接指定
    device = torch.device("cuda:0")
    inputs = {
        k: (v.to(device=device, dtype=torch.float16)
            if v.dtype in (torch.float32, torch.float64)   # 浮点 tensor → fp16
            else v.to(device=device))                        # 整型 tensor 只移设备
        for k, v in inputs.items()
        if isinstance(v, torch.Tensor)
    }

    print("Starting generation...", flush=True)
    t0 = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            use_audio_in_video=False,   # ★ Omni 模型必须加这个
        )
    print(f"Generation done in {time.time() - t0:.2f}s", flush=True)

    # 只解码新生成的 token（去掉 input prompt 部分）
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    pred = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return pred, reference, user_text, audio_path


# ===== 4. 跑第一条验证 =====
item = dataset[0]
pred, reference, user_text, audio_path = run_inference(item)

print("Prediction:", pred, flush=True)
print("Reference: ", reference, flush=True)

with open("/home/ubuntu/project/prediction_single.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "input": user_text,
            "audio": audio_path,
            "prediction": pred,
            "reference": reference,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print("Saved to /home/ubuntu/project/prediction_single.json", flush=True)