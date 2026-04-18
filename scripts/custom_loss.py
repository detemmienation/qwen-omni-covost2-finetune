"""
Custom loss functions for Qwen-Omni e2e speech translation fine-tuning.

Two losses are combined:
  1. Focal Loss      — down-weights easy tokens so hard ones (entities, rare chars)
                       get more gradient. Replaces label smoothing without causing
                       format drift.
  2. Entity Upweight — multiplies the loss on digit / Latin-letter tokens by
                       `entity_weight`. Directly targets ETA and ERR metrics,
                       which measure whether numbers and English proper nouns are
                       preserved in the Chinese output.

Usage: imported by run_train_custom_loss.py which patches swift.trainers before
calling sft_main().
"""

import re
import torch
from torch.nn import CrossEntropyLoss


# --------------------------------------------------------------------------- #
# Token-ID lookup for "entity" tokens                                          #
# --------------------------------------------------------------------------- #

_ENTITY_IDS: set[int] | None = None


def get_entity_token_ids(tokenizer) -> set[int]:
    """
    Build (once) the set of token IDs whose surface form contains a digit or
    ASCII letter.  These correspond to numbers and English proper nouns that
    the ETA / ERR metrics expect to see preserved in the Chinese translation.
    """
    global _ENTITY_IDS
    if _ENTITY_IDS is not None:
        return _ENTITY_IDS

    entity_ids: set[int] = set()
    # Strip common BPE / SentencePiece prefixes before testing
    _prefix_re = re.compile(r'^[▁Ġ#\s]+')

    for token, tid in tokenizer.get_vocab().items():
        clean = _prefix_re.sub("", token)
        if re.search(r"[0-9A-Za-z]", clean):
            entity_ids.add(tid)

    _ENTITY_IDS = entity_ids
    print(
        f"[custom_loss] Entity token IDs built: {len(entity_ids)} tokens "
        f"(digits + Latin letters)",
        flush=True,
    )
    return entity_ids


# --------------------------------------------------------------------------- #
# Core loss computation                                                         #
# --------------------------------------------------------------------------- #

def compute_custom_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tokenizer,
    gamma: float = 2.0,
    entity_weight: float = 5.0,
) -> torch.Tensor:
    """
    Focal + entity-upweighted cross-entropy.

    Args:
        logits:        (B, T, V)  — raw model output logits
        labels:        (B, T)     — target token IDs; -100 = ignored (padding)
        tokenizer:     used to look up entity token IDs (cached after first call)
        gamma:         focal exponent; 2.0 is a standard default
        entity_weight: loss multiplier for entity tokens; 5.0 means 5× gradient
                       on digits / English proper nouns

    Returns:
        Scalar loss tensor.
    """
    flat_logits = logits.view(-1, logits.size(-1))   # (B*T, V)
    flat_labels = labels.view(-1)                     # (B*T,)

    # ── per-token CE (no reduction) ──────────────────────────────────────────
    ce_fct = CrossEntropyLoss(reduction="none", ignore_index=-100)
    per_token_ce = ce_fct(flat_logits, flat_labels)   # (B*T,)

    # ── focal weight: (1 - p_correct)^gamma ─────────────────────────────────
    # p_correct = exp(-ce).  Detach so focal weight doesn't backprop through
    # itself; we only want it to re-weight the actual CE gradient.
    pt = torch.exp(-per_token_ce.detach())
    focal_w = (1.0 - pt) ** gamma                     # (B*T,)

    # ── entity weight ────────────────────────────────────────────────────────
    entity_ids = get_entity_token_ids(tokenizer)
    entity_tensor = torch.tensor(
        list(entity_ids), device=labels.device, dtype=flat_labels.dtype
    )
    is_entity = torch.isin(flat_labels, entity_tensor)
    entity_w = torch.where(is_entity, entity_weight, 1.0)  # (B*T,)

    # ── combine; ignore padding positions ───────────────────────────────────
    valid = (flat_labels != -100).float()
    weighted = focal_w * entity_w * per_token_ce * valid
    loss = weighted.sum() / valid.sum().clamp(min=1.0)
    return loss
