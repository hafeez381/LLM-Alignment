"""
alignment/ppo.py — Proximal Policy Optimisation for LLMs (Task C3)
===================================================================

PSEUDOCODE (full implementation plan for Phase C3):
────────────────────────────────────────────────────

┌─────────────────────────────────────────────────────────────────────────┐
│  ROLLOUT PHASE (collect trajectories with π_old)                        │
│  ──────────────────────────────────────────────────────────────────────  │
│  For each prompt x_i in batch:                                           │
│    1. Generate y_i = model.generate(x_i, do_sample=True, max_new=128)  │
│    2. Compute old log-probs:                                             │
│         log_probs_old[i,t] = log π_old(y_{i,t} | x_i, y_{i,<t})       │
│         ← MUST be computed HERE at rollout time, NOT inside update loop │
│         ← If you recompute inside the loop, ρ_t = 1.0 always → no clip │
│    3. Get ref log-probs (disable_adapter ctx):                           │
│         log_probs_ref[i,t] = log π_ref(y_{i,t} | x_i, y_{i,<t})       │
│    4. Query RM (frozen, no_grad):                                        │
│         r_task[i] = rm(x_i ⊕ y_i)                                      │
│    5. Query value model (frozen for old values):                         │
│         V_old[i,t] = value_head(hidden_state at s_{i,t})                │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  REWARD SHAPING (per-token reward with KL penalty)                      │
│  ──────────────────────────────────────────────────────────────────────  │
│  r_{i,t} = r_task_i · 1[t = T_i]                                       │
│           - β · (log_probs_old[i,t] - log_probs_ref[i,t])              │
│                  ↑ KL shaping per token: penalises divergence at each t │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  GAE ADVANTAGE ESTIMATION                                                │
│  ──────────────────────────────────────────────────────────────────────  │
│  δ_t = r_t + γ · V_old(s_{t+1}) - V_old(s_t)    # TD residual         │
│  A_t^GAE = Σ_{k≥0} (γλ)^k · δ_{t+k}             # GAE accumulation   │
│            computed by a RIGHT-TO-LEFT scan over the sequence           │
│  V^GAE(s_t) = V_old(s_t) + A_t^GAE               # value target        │
│  DETACH V_GAE before using as critic target! Otherwise gradients from  │
│  the critic's own output flow back through the GAE computation.         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  PPO UPDATE LOOP (K epochs over the rollout buffer)                     │
│  ──────────────────────────────────────────────────────────────────────  │
│  For epoch in range(ppo_epochs):                                        │
│    For mini_batch in rollout_buffer:                                    │
│      1. Recompute current log-probs under π_θ (trainable):             │
│           log_probs_new[t] = log π_θ(y_t | x, y_{<t})                 │
│           ← Only over RESPONSE tokens; mask out prompt + pad            │
│      2. Importance ratio:                                               │
│           ρ_t = exp(log_probs_new[t] - log_probs_old[t])               │
│                 shape: (batch, response_len)                            │
│      3. Clipped surrogate objective:                                    │
│           L_clip = E_t[min(ρ_t·Ã_t, clip(ρ_t, 1-ε, 1+ε)·Ã_t)]        │
│      4. Critic loss:                                                    │
│           L_V = E_{i,t}[(V_θ(s_{i,t}) - V^GAE(s_{i,t}))²]            │
│      5. Entropy bonus:                                                  │
│           H = -E_t[Σ_a π_θ(a|s_t) log π_θ(a|s_t)]                    │
│      6. Total loss:                                                     │
│           L = -L_clip + c_V·L_V - c_ent·H                             │
│      7. Backward + clip grad norm + optimizer step                      │
└─────────────────────────────────────────────────────────────────────────┘

CRITICAL BUGS TO AVOID:
    ① Recomputing π_old inside update loop → ρ_t ≡ 1.0 → clipping never activates
    ② Not masking prompt tokens in log-prob sum → wrong importance ratios
    ③ Not detaching V^GAE targets → gradients through critic self-predict
    ④ Not calling rm() and value_head() under torch.no_grad() at rollout time
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from config import cfg


# TODO (Phase C3): Implement all functions below.

def collect_rollouts(
    policy,
    rm,
    value_head,
    prompt_loader,
    policy_tokenizer,
    rm_tokenizer,
    device,
    ppo_cfg=None,
) -> List[Dict]:
    """
    Collect a batch of trajectories under the current π_θ.

    Returns a list of dicts, each containing:
        input_ids         : (seq_len,) — prompt + response token ids
        attention_mask    : (seq_len,)
        log_probs_old     : (response_len,) — log π_old(y_t|s_t) CACHED HERE
        log_probs_ref     : (response_len,) — log π_ref(y_t|s_t)
        values_old        : (response_len,) — V_old(s_t), DETACHED
        rewards           : (response_len,) — shaped rewards r_{i,t}
        advantages        : (response_len,) — A_t^GAE, DETACHED
        value_targets     : (response_len,) — V^GAE(s_t), DETACHED
        response_mask     : (seq_len,) — 1 for response tokens, 0 for prompt+pad
    """
    raise NotImplementedError("TODO: Implement in Phase C3")


def compute_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,       # shape: (batch, seq_len)
    attention_mask: torch.Tensor,  # shape: (batch, seq_len)
    response_mask: torch.Tensor,   # shape: (batch, seq_len), 1=response token
) -> torch.Tensor:
    """
    Compute per-token log-probabilities for response tokens only.

    Math:
        log π(y_t | x, y_{<t}) = log_softmax(logits[t])[y_t]

    The response_mask ensures we only SUM log-probs over response tokens.
    Prompt and padding tokens are EXCLUDED from the sum.

    Returns: shape (batch, seq_len) — 0.0 for non-response positions
    """
    raise NotImplementedError("TODO: Implement in Phase C3")


def compute_gae(
    rewards: torch.Tensor,    # shape: (batch, response_len)
    values: torch.Tensor,     # shape: (batch, response_len), V_old — DETACHED
    gamma: float = 1.0,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generalised Advantage Estimation (GAE-λ).

    Math:
        δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
        A_t^GAE = Σ_{k≥0} (γλ)^k · δ_{t+k}

    Computed by a right-to-left scan:
        A_T = δ_T
        A_t = δ_t + (γλ) · A_{t+1}

    Returns:
        advantages   : shape (batch, response_len) — A_t^GAE, DETACHED
        value_targets: shape (batch, response_len) — V(s_t)+A_t, DETACHED
    """
    raise NotImplementedError("TODO: Implement in Phase C3")


def ppo_clip_loss(
    log_probs_new: torch.Tensor,   # shape: (batch, response_len)
    log_probs_old: torch.Tensor,   # shape: (batch, response_len) — CACHED
    advantages: torch.Tensor,      # shape: (batch, response_len) — DETACHED
    epsilon: float = 0.2,
) -> torch.Tensor:
    """
    PPO clipped surrogate objective.

    Math:
        ρ_t(θ) = π_θ(y_t|s_t) / π_old(y_t|s_t)
               = exp(log π_θ(y_t) - log π_old(y_t))   [log-space ratio]

        L_clip = E_t[min(ρ_t·Ã_t,  clip(ρ_t, 1-ε, 1+ε)·Ã_t)]

    Returns: scalar loss (to be negated for gradient ascent)
    """
    raise NotImplementedError("TODO: Implement in Phase C3")


def ppo_update(
    policy,
    value_head,
    optimizer,
    rollout_buffer: List[Dict],
    ppo_cfg=None,
) -> Dict[str, float]:
    """
    Full PPO update: K epochs of mini-batch gradient steps over the buffer.

    Returns a dict of logged metrics (loss, ratio, etc.)
    """
    raise NotImplementedError("TODO: Implement in Phase C3")
