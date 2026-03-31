"""
alignment/dpo.py — Direct Preference Optimisation (Task C4)
============================================================

PSEUDOCODE (for Phase C4):
────────────────────────────

The DPO loss is derived from the RLHF objective via the reward reparameterisation:
    r_ψ(x, y) = β · log[π*(y|x) / π_ref(y|x)] + β · log Z(x)

Substituting into the Bradley-Terry preference model and cancelling log Z(x):
    L_DPO(θ) = -E_{(x,y⁺,y⁻)} [
        log σ(
            β · log[π_θ(y⁺|x) / π_ref(y⁺|x)]
          - β · log[π_θ(y⁻|x) / π_ref(y⁻|x)]
        )
    ]

where:
    log π(y|x) = Σ_{t=1}^{T} log π(y_t | x, y_{<t})   ← SUM over response tokens only
    log Z(x) cancels because it appears identically in both terms.

IMPLEMENTATION STEPS:
─────────────────────
1. Forward pass through π_θ (trainable, LoRA adapters ON):
     logits_theta_pos = policy(chosen_input_ids)           shape: (B,L,V)
     logits_theta_neg = policy(rejected_input_ids)         shape: (B,L,V)

2. Forward pass through π_ref (NO_GRAD, LoRA adapters OFF via disable_adapter):
     logits_ref_pos   = ref_policy(chosen_input_ids)       shape: (B,L,V)
     logits_ref_neg   = ref_policy(rejected_input_ids)     shape: (B,L,V)

3. Sum log-probs over RESPONSE TOKENS ONLY:
     log_π_θ(y⁺|x) = sum_response_logprobs(logits_theta_pos, chosen_labels)
     log_π_θ(y⁻|x) = sum_response_logprobs(logits_theta_neg, rejected_labels)
     log_π_ref(y⁺|x) = sum_response_logprobs(logits_ref_pos, chosen_labels)
     log_π_ref(y⁻|x) = sum_response_logprobs(logits_ref_neg, rejected_labels)
     ← chosen_labels / rejected_labels have -100 for prompt+pad tokens
     ← we build a mask: mask = (labels != -100), then sum only at mask positions

4. Compute DPO margin:
     Δ_θ   = log_π_θ(y⁺|x) - log_π_θ(y⁻|x)    shape: (B,)
     Δ_ref  = log_π_ref(y⁺|x) - log_π_ref(y⁻|x) shape: (B,)
     z      = β · (Δ_θ - Δ_ref)                  shape: (B,)
     L_DPO  = -logsigmoid(z).mean()               scalar

5. Monitoring:
     Implicit chosen reward:   β · (log_π_θ(y⁺|x) - log_π_ref(y⁺|x))  shape: (B,)
     Implicit rejected reward: β · (log_π_θ(y⁻|x) - log_π_ref(y⁻|x))  shape: (B,)
     Preference accuracy:      P(implicit_chosen_r > implicit_rejected_r)

CRITICAL BUGS:
    ① NOT masking prompt/pad tokens in log-prob sum → overestimates long-response
      log-probs, creates spurious gradient signal (verbosity bias amplified)
    ② Letting π_ref gradients flow → if ref is same model (disable_adapter),
      must be inside no_grad context; if separate model, freeze ALL params
    ③ Calling π_ref inside the outer computation graph → can leak into policy grad
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from config import cfg


# TODO (Phase C4): Implement all functions below.

def sum_response_log_probs(
    logits: torch.Tensor,   # shape: (batch, seq_len, vocab_size)
    labels: torch.Tensor,   # shape: (batch, seq_len), -100 for prompt+pad
) -> torch.Tensor:
    """
    Compute log π(y|x) = Σ_{t: response token} log π(y_t | x, y_{<t}).

    Uses the causal shift: logits[:, t, :] predicts labels[:, t+1].

    The response_mask is derived from labels: positions where labels != -100
    are response tokens. We only SUM these log-probs.

    Returns: shape (batch,) — one scalar per sequence
    """
    raise NotImplementedError("TODO: Implement in Phase C4")


def dpo_loss(
    policy,           # π_θ with LoRA adapters ON
    chosen_ids: torch.Tensor,      # (B, L)
    chosen_mask: torch.Tensor,     # (B, L)
    chosen_labels: torch.Tensor,   # (B, L), -100 for prompt+pad
    rejected_ids: torch.Tensor,    # (B, L)
    rejected_mask: torch.Tensor,   # (B, L)
    rejected_labels: torch.Tensor, # (B, L), -100 for prompt+pad
    beta: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Full DPO loss for one batch.

    Math:
        L_DPO = -E[log σ(β·(Δ_θ - Δ_ref))]

    Returns (loss scalar, metrics_dict)
    """
    raise NotImplementedError("TODO: Implement in Phase C4")