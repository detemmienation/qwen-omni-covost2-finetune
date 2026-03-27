from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import json

model_path = "output/debug_run_tiny_fp16/checkpoint-xxx"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model.eval()

# 你的 val dataset
dataset = load_dataset("json", data_files="val.jsonl")["train"]

results = []

for item in dataset:
    input_text = item["messages"][0]["content"]
    reference = item["messages"][1]["content"]

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100
        )

    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

    results.append({
        "input": input_text,
        "prediction": pred,
        "reference": reference
    })

# 保存
with open("predictions.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")