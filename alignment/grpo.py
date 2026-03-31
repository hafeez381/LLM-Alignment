"""
alignment/grpo.py — Group Relative Policy Optimisation (Task C5)
================================================================

PSEUDOCODE (for Phase C5):
───────────────────────────

┌─────────────────────────────────────────────────────────────────────────┐
│  GROUP ROLLOUT (K completions per prompt)                                │
│  ──────────────────────────────────────────────────────────────────────  │
│  For each prompt x_b in batch (b = 1..B):                               │
│    Sample K completions: {y_{b,k}}_{k=1}^K via temperature sampling    │
│    Score each: r_{b,k} = rm(x_b ⊕ y_{b,k})   [frozen, no_grad]        │
│    Compute group mean: μ_b = (1/K) Σ_k r_{b,k}                         │
│    Group-relative advantage: A_{b,k} = r_{b,k} - μ_b                   │
│      → No value model needed! μ_b IS the baseline.                      │
│      → A_{b,k} is BROADCAST to ALL tokens of completion y_{b,k}        │
│         (uniform credit — weaker than PPO token-level advantages)       │
│    Cache log_probs_old_{b,k,t} = log π_old(y_{b,k,t}|s_t) per token   │
│    Cache log_probs_ref_{b,k,t} = log π_ref(y_{b,k,t}|s_t) per token   │
│                                  [disable_adapter, no_grad]             │
│                                                                          │
│  DEGENERATE BATCH CHECK:                                                 │
│    if std({r_{b,k}}) ≈ 0 for all k → all A_{b,k} = 0 → gradient = 0   │
│    Increment degenerate_batch_counter; log if frequent                  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  GRPO OBJECTIVE                                                          │
│  ──────────────────────────────────────────────────────────────────────  │
│  L_GRPO = -(1/K) Σ_k (1/T_k) Σ_t                                      │
│              min(ρ_{k,t}·A_{k,t}, clip(ρ_{k,t},1-ε,1+ε)·A_{k,t})     │
│           + β · KL(π_θ || π_ref)                                        │
│                                                                          │
│  where KL = E_t[log π_θ(y_t|s_t) - log π_ref(y_t|s_t)]               │
│         ← per-token KL, averaged over all response tokens               │
│                                                                          │
│  NOTE: Length normalisation (1/T_k) prevents short correct answers     │
│  from dominating. But it creates bias: short correct answers get a      │
│  stronger per-token push than equally correct long answers.             │
└─────────────────────────────────────────────────────────────────────────┘

CRITICAL BUGS:
    ① Recomputing log_probs_old inside the update → same bug as PPO
    ② Not normalising by T_k → long sequences dominate the gradient
    ③ Forgetting the KL term → reward hacking without policy anchoring
    ④ Degenerate batches (all r_{b,k} equal) silently produce zero grad
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

from config import cfg


# TODO (Phase C5): Implement all functions below.

def group_rollout(
    policy,
    rm,
    prompts: torch.Tensor,    # (B, prompt_len)
    prompt_mask: torch.Tensor,# (B, prompt_len)
    K: int = 4,
    max_new_tokens: int = 128,
    device=None,
) -> List[Dict]:
    """
    For each of B prompts, generate K completions and compute:
        - r_{b,k}        : RM score for each completion
        - A_{b,k}        : group-relative advantage
        - log_probs_old  : cached per-token log-probs under π_old
        - log_probs_ref  : cached per-token log-probs under π_ref

    Returns a list of B·K trajectory dicts.
    """
    raise NotImplementedError("TODO: Implement in Phase C5")


def grpo_loss(
    log_probs_new: torch.Tensor,  # (B*K, response_len)
    log_probs_old: torch.Tensor,  # (B*K, response_len) — CACHED
    log_probs_ref: torch.Tensor,  # (B*K, response_len) — CACHED
    advantages: torch.Tensor,     # (B*K, response_len) — broadcast from A_{b,k}
    response_mask: torch.Tensor,  # (B*K, response_len) — 1=real response token
    epsilon: float = 0.2,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    GRPO loss = clipped surrogate + KL penalty.

    Returns (loss scalar, metrics dict)
    """
    raise NotImplementedError("TODO: Implement in Phase C5")