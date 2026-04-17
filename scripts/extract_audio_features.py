"""
Pre-extract audio encoder features from Qwen3-Omni.
Run ONCE before training to cache audio encoder outputs to disk.

Usage:
    python scripts/extract_audio_features.py

Output:
    data/audio_cache/train/<i>.pt      -- feature tensor per train sample
    data/audio_cache/val/<i>.pt        -- feature tensor per val sample
    data/audio_cache/train/path_index.json  -- {audio_path: index}
    data/audio_cache/val/path_index.json

Estimated time: ~10-20 min for 5500 samples on GH200.
Expected GPU memory: ~65 GB (audio tower loaded on top of LLM).
"""

import os, gc, json, torch
import librosa
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3OmniMoeForConditionalGeneration

TARGET_SR = 16000   # must match processor.feature_extractor.sampling_rate

MODEL_ID  = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
DATA_DIR  = "/home/ubuntu/leili-cmu-lab/CMU-project/Y4_data/data"
CACHE_DIR = f"{DATA_DIR}/audio_cache"

SPLITS = [
    (f"{DATA_DIR}/train_5000.jsonl", f"{CACHE_DIR}/train"),
    (f"{DATA_DIR}/val_500.jsonl",    f"{CACHE_DIR}/val"),
]


def load_audio(path: str):
    # Use librosa at TARGET_SR — identical to what SWIFT does during training
    # (swift/llm/template/vision_utils.py: librosa.load(audio_io, sr=sampling_rate))
    audio = librosa.load(path, sr=TARGET_SR)[0]
    return audio, TARGET_SR


def extract_split(jsonl_path, split_cache_dir, processor, audio_tower):
    Path(split_cache_dir).mkdir(parents=True, exist_ok=True)

    with open(jsonl_path) as f:
        lines = f.readlines()

    path_index = {}   # audio_path → integer index
    skipped    = 0

    for i, line in enumerate(tqdm(lines, desc=Path(jsonl_path).stem)):
        item       = json.loads(line.strip())
        audio_path = item["audios"][0]
        cache_file = Path(split_cache_dir) / f"{i}.pt"

        path_index[audio_path] = i   # always register, even if already cached

        if cache_file.exists():
            continue

        try:
            audio, sr = load_audio(audio_path)

            feat_out = processor.feature_extractor(
                audio,
                sampling_rate=sr,
                return_tensors="pt",
                return_attention_mask=True,
            )
            # Strip padding: audio_tower expects (n_mel, valid_frames), not the
            # padded (1, n_mel, 3000) that the feature extractor produces.
            # Mirror what get_audio_features() does internally:
            #   input_features.permute(0,2,1)[mask].permute(1,0)
            attn_mask      = feat_out["attention_mask"].bool()          # (1, 3000)
            feature_lens   = attn_mask.sum(-1).to("cuda")               # (1,)
            input_features = (
                feat_out["input_features"]
                .to("cuda", torch.bfloat16)
                .permute(0, 2, 1)[attn_mask]   # (valid_frames, n_mel)
                .permute(1, 0)                 # (n_mel, valid_frames)
            )

            with torch.no_grad():
                out = audio_tower(input_features, feature_lens=feature_lens)

            # Handle tuple/dataclass output
            if isinstance(out, (tuple, list)):
                features = out[0]
            elif hasattr(out, "last_hidden_state"):
                features = out.last_hidden_state
            else:
                features = out

            # Save as (num_audio_tokens, hidden_dim) in float16
            torch.save(features.squeeze(0).cpu().half(), cache_file)
            torch.cuda.empty_cache()

        except Exception as e:
            skipped += 1
            print(f"\n[{i}] skip {audio_path}: {e}")
            continue

    # Save path → index lookup
    idx_file = Path(split_cache_dir) / "path_index.json"
    with open(idx_file, "w") as f:
        json.dump(path_index, f, ensure_ascii=False)

    n_ok = len(lines) - skipped
    print(f"\n{Path(jsonl_path).stem}: {n_ok}/{len(lines)} extracted. "
          f"Index saved to {idx_file}")


def main():
    print("Loading processor ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    print("Loading model to GPU (will free non-audio parts after) ...")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )

    # Keep only audio tower, free everything else
    audio_tower = model.thinker.audio_tower.eval()
    del model.thinker.model
    del model.thinker.lm_head
    del model.thinker.visual
    del model.talker
    del model.code2wav
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU after cleanup: {torch.cuda.memory_allocated(0)/1e9:.1f} GB")

    for jsonl_path, split_cache_dir in SPLITS:
        if not os.path.exists(jsonl_path):
            print(f"Skip (not found): {jsonl_path}")
            continue
        print(f"\n=== Processing {jsonl_path} ===")
        extract_split(jsonl_path, split_cache_dir, processor, audio_tower)

    print("\nAll done.")


if __name__ == "__main__":
    main()
