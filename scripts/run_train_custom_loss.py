#!/usr/bin/env python3
"""
Training wrapper: pre-cached audio features + focal + entity-upweighted loss.

Combines two earlier scripts:
  - run_train_cached.py  : skips audio encoder by serving pre-cached features
  - custom_loss.py       : focal CE + entity-token upweighting

Prerequisites
-------------
1. Run extract_audio_features.py first to build data/audio_cache/.
2. Pass the same CLI args you would pass to "swift sft".

How the loss patch works
------------------------
SWIFT's Seq2SeqTrainer.compute_loss is replaced at the class level BEFORE
sft_main() is called.  Because Python resolves method calls on the class object
at runtime, any trainer instance created by SWIFT will use the patched method —
no subclassing or trainer-registry hacking required.  This is the same pattern
used in run_train_cached.py to patch Qwen3OmniMoeAudioEncoder.forward.

Example
-------
    CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    USE_HF=1 \
    python scripts/run_train_custom_loss.py \
      --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
      --use_hf true \
      --dataset  /path/to/train_500.jsonl \
      --val_dataset /path/to/val_100.jsonl \
      --train_type lora \
      --torch_dtype float16 \
      --num_train_epochs 1 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 1 \
      --learning_rate 5e-5 \
      --max_length 320 \
      --lora_rank 4 \
      --lora_alpha 8 \
      --lora_dropout 0.05 \
      --output_dir /path/to/output/custom_loss_500
"""

import sys
import json
import collections
from pathlib import Path
from types import MethodType

import torch
from transformers.modeling_outputs import BaseModelOutput

# ── Loss hyper-parameters (edit here or extend CLI if needed) ──────────────────
FOCAL_GAMMA    = 2.0   # focal exponent; 0 = standard CE, 2 = standard focal
ENTITY_WEIGHT  = 5.0   # loss multiplier for digit / Latin-letter tokens

# ══════════════════════════════════════════════════════════════════════════════
# Part 1 — Audio encoder cache  (copied verbatim from run_train_cached.py)
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR  = "/home/ubuntu/leili-cmu-lab/CMU-project/Y4_data/data"
CACHE_DIR = Path(DATA_DIR) / "audio_cache"

_CACHE: dict[str, Path] = {}
for _split in ("train", "val"):
    _idx_file = CACHE_DIR / _split / "path_index.json"
    if _idx_file.exists():
        with open(_idx_file) as _f:
            _idx = json.load(_f)
        for _audio_path, _i in _idx.items():
            _CACHE[_audio_path] = CACHE_DIR / _split / f"{_i}.pt"
        print(f"[cache] {_split}: {len(_idx)} entries loaded", flush=True)

if not _CACHE:
    print(
        "[cache] WARNING: no cached features found — "
        "run scripts/extract_audio_features.py first!",
        flush=True,
    )

_PATH_QUEUE: collections.deque[str] = collections.deque()

from swift.llm.template import vision_utils as _vu
_orig_load_audio = _vu.load_audio


def _hooked_load_audio(audio, sampling_rate, return_sr=False):
    _PATH_QUEUE.append(str(audio))
    return _orig_load_audio(audio, sampling_rate, return_sr)


_vu.load_audio = _hooked_load_audio
import swift.llm.template.template.qwen as _qt  # noqa: E402
_qt.load_audio = _hooked_load_audio
print("[cache] Hooked swift load_audio", flush=True)

from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (  # noqa: E402
    Qwen3OmniMoeAudioEncoder,
    _get_feat_extract_output_lengths,
)

_cache_misses: list[str] = []
_adjusted: list[tuple]   = []


def _cached_audio_forward(self, input_features, feature_lens=None, aftercnn_lens=None):
    batch_size = feature_lens.shape[0] if feature_lens is not None else 1
    feats = []
    for i in range(batch_size):
        if not _PATH_QUEUE:
            raise RuntimeError(
                "[cache] Path queue is empty when audio encoder was called."
            )
        path    = _PATH_QUEUE.popleft()
        pt_file = _CACHE.get(path)

        if pt_file is not None and pt_file.exists():
            feat = torch.load(pt_file, map_location="cpu", weights_only=True)
            feat = feat.to(device=input_features.device, dtype=input_features.dtype)
            if feature_lens is not None:
                expected_T = int(_get_feat_extract_output_lengths(feature_lens[i]).item())
                actual_T   = feat.shape[0]
                if actual_T != expected_T:
                    _adjusted.append((path, actual_T, expected_T))
                    if actual_T < expected_T:
                        pad  = feat[-1:].expand(expected_T - actual_T, -1)
                        feat = torch.cat([feat, pad], dim=0)
                    else:
                        feat = feat[:expected_T]
            feats.append(feat)
        else:
            _cache_misses.append(path)
            raise RuntimeError(
                f"[cache] No cached .pt file for: {path!r}\n"
                "Re-run extract_audio_features.py to rebuild the cache."
            )

    hidden_states = torch.cat(feats, dim=0)
    return BaseModelOutput(last_hidden_state=hidden_states)


Qwen3OmniMoeAudioEncoder.forward = _cached_audio_forward
print("[cache] Patched Qwen3OmniMoeAudioEncoder.forward", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Part 2 — Custom loss patch
# ══════════════════════════════════════════════════════════════════════════════

from custom_loss import compute_custom_loss  # noqa: E402  (local script)
from swift.trainers import Seq2SeqTrainer    # noqa: E402

_original_compute_loss = Seq2SeqTrainer.compute_loss


def _patched_compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    """
    Drop-in replacement for Seq2SeqTrainer.compute_loss.

    Falls back to the original loss if:
      - labels are absent (shouldn't happen in SFT but be safe)
      - tokenizer is not attached to the trainer yet
    """
    outputs = model(**{k: v for k, v in inputs.items() if k != "labels"
                        if False} or inputs)

    # SWIFT passes labels inside inputs; retrieve them for loss computation.
    labels = inputs.get("labels")

    if labels is None or not hasattr(self, "tokenizer") or self.tokenizer is None:
        # Fallback to original behaviour
        return _original_compute_loss(self, model, inputs,
                                      return_outputs=return_outputs, **kwargs)

    logits = outputs.logits  # (B, T, V)

    loss = compute_custom_loss(
        logits=logits,
        labels=labels,
        tokenizer=self.tokenizer,
        gamma=FOCAL_GAMMA,
        entity_weight=ENTITY_WEIGHT,
    )

    # Attach loss to outputs so SWIFT can log it correctly
    outputs.loss = loss
    return (loss, outputs) if return_outputs else loss


Seq2SeqTrainer.compute_loss = _patched_compute_loss
print(
    f"[custom_loss] Patched Seq2SeqTrainer.compute_loss "
    f"(focal γ={FOCAL_GAMMA}, entity_weight={ENTITY_WEIGHT})",
    flush=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# Part 3 — Hand off to SWIFT sft_main (sys.argv forwarded transparently)
# ══════════════════════════════════════════════════════════════════════════════

# Ensure scripts/ dir is on the path so custom_loss.py is importable
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from swift.cli.sft import sft_main  # noqa: E402

sft_main()
