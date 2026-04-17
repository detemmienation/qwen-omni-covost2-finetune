#!/usr/bin/env python3
"""
Training wrapper that skips the audio encoder by using pre-cached features.

Run extract_audio_features.py first to populate the cache.

Usage (same args as "swift sft"):
    conda run -n qwen3-omni python scripts/run_train_cached.py \\
        --model Qwen/Qwen3-Omni-30B-A3B-Instruct \\
        --dataset /path/to/train.jsonl \\
        ...

How it works:
  1. SWIFT loads each audio file via librosa (inside load_audio).
     We hook load_audio to push the audio path into a queue — the real
     audio is still loaded so WhisperFeatureExtractor can compute the
     correct mel length and thus the correct number of <audio_pad> tokens.
  2. When Qwen3OmniMoeAudioEncoder.forward is called during the model
     forward pass, we pop the matching path from the queue and return the
     pre-cached (T, output_dim) feature tensor instead of running the
     encoder.  This saves the GPU encoder compute (~0.1-0.2 s/sample).
  3. Everything else (LLM forward+backward, LoRA updates) runs normally.

Requirement:
    data/audio_cache/{train,val}/path_index.json  and  *.pt  files
    produced by scripts/extract_audio_features.py.
"""

import sys
import json
import collections
from pathlib import Path
from types import MethodType

import torch
from transformers.modeling_outputs import BaseModelOutput

# ── 1. Build cache lookup: audio_path → .pt file ──────────────────────────────
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

# ── 2. Queue to track which audio files the data loader opened ─────────────────
# With num_workers=0 everything is single-threaded, so push order matches
# the order in which audio_tower.forward is eventually called.
_PATH_QUEUE: collections.deque[str] = collections.deque()

# ── 3. Hook load_audio in SWIFT so we capture each audio path ─────────────────
from swift.llm.template import vision_utils as _vu
_orig_load_audio = _vu.load_audio


def _hooked_load_audio(audio, sampling_rate, return_sr=False):
    """Push the path to the queue, then load the file normally."""
    _PATH_QUEUE.append(str(audio))
    return _orig_load_audio(audio, sampling_rate, return_sr)


# Patch both the module-level name and the name imported into the qwen template.
_vu.load_audio = _hooked_load_audio
import swift.llm.template.template.qwen as _qt  # noqa: E402
_qt.load_audio = _hooked_load_audio
print("[cache] Hooked swift load_audio", flush=True)

# ── 4. Replace audio encoder forward with cache lookup ─────────────────────────
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (  # noqa: E402
    Qwen3OmniMoeAudioEncoder,
    _get_feat_extract_output_lengths,
)

_cache_misses: list[str] = []
_adjusted: list[tuple] = []   # log (path, cached_T, expected_T) mismatches


def _cached_audio_forward(self, input_features, feature_lens=None, aftercnn_lens=None):
    """Return pre-cached features; skip actual encoder computation."""
    batch_size = feature_lens.shape[0] if feature_lens is not None else 1
    feats = []

    for i in range(batch_size):
        if not _PATH_QUEUE:
            raise RuntimeError(
                "[cache] Path queue is empty when audio encoder was called. "
                "This usually means load_audio was not patched in time or "
                "num_workers > 0 was used."
            )
        path = _PATH_QUEUE.popleft()
        pt_file = _CACHE.get(path)

        if pt_file is not None and pt_file.exists():
            feat = torch.load(pt_file, map_location="cpu", weights_only=True)
            feat = feat.to(device=input_features.device, dtype=input_features.dtype)

            # The number of audio tokens the model expects for this sample is
            # determined by feature_lens[i] via _get_feat_extract_output_lengths.
            # Small rounding differences between extraction and training can make
            # cached T differ by ±1-2 tokens — adjust to match exactly.
            if feature_lens is not None:
                expected_T = int(_get_feat_extract_output_lengths(feature_lens[i]).item())
                actual_T = feat.shape[0]
                if actual_T != expected_T:
                    _adjusted.append((path, actual_T, expected_T))
                    if actual_T < expected_T:
                        # Pad by repeating the last token
                        pad = feat[-1:].expand(expected_T - actual_T, -1)
                        feat = torch.cat([feat, pad], dim=0)
                    else:
                        feat = feat[:expected_T]

            feats.append(feat)
        else:
            _cache_misses.append(path)
            raise RuntimeError(
                f"[cache] No cached .pt file for audio path: {path!r}\n"
                "Re-run extract_audio_features.py to rebuild the cache."
            )

    # Concatenate along the token axis: (sum_T, output_dim)
    hidden_states = torch.cat(feats, dim=0)
    return BaseModelOutput(last_hidden_state=hidden_states)


Qwen3OmniMoeAudioEncoder.forward = _cached_audio_forward
print("[cache] Patched Qwen3OmniMoeAudioEncoder.forward", flush=True)

# ── 5. Launch swift sft (sys.argv are forwarded transparently) ─────────────────
from swift.cli.sft import sft_main  # noqa: E402

sft_main()
