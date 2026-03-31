"""
model/reward_head.py — Reward Model with Scalar Head
=====================================================

PSEUDOCODE:
────────────
RewardModel.__init__(backbone)
  → Stores AutoModelForSequenceClassification backbone
  → backbone already has Linear(hidden_size, 1) head attached by HuggingFace
  → Optionally apply LoRA to backbone layers (or just train the head)

RewardModel.forward(input_ids, attention_mask)
  → backbone(input_ids, attention_mask)
  → outputs.logits  : shape (batch, 1)
  → squeeze(-1)     : shape (batch,)   ← scalar reward r(x,y) ∈ ℝ
  → return reward

compute_rm_loss(r_pos, r_neg, lambda_reg)
  → Margin ranking loss (Bradley-Terry):
        L_RM = -E[log σ(r⁺ - r⁻)]  +  λ · E[(r⁺)² + (r⁻)²]
  → First term: pushes r⁺ > r⁻ (preference learning)
  → Second term: L2 regularisation on reward magnitudes
    WHY reg? Without it, r can grow to ±∞ while still satisfying r⁺>r⁻.
    Unbounded rewards cause NaN in later RL training and make rewards
    non-comparable across batches (Goodhart's Law amplification).

How AutoModelForSequenceClassification finds the last non-pad token:
  → It computes sequence_lengths = (input_ids == pad_token_id).int().argmax(-1) - 1
  → This finds the INDEX of the first pad token, then subtracts 1 → last real token
  → This ONLY works with RIGHT-PADDING (pad tokens at the end)
  → If we accidentally used left-padding for the RM, we'd read from the wrong position
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from config import cfg


# ─────────────────────────────────────────────────────────────────────────────
# RewardModel wrapper
# ─────────────────────────────────────────────────────────────────────────────

class RewardModel(nn.Module):
    """
    Wraps Llama-3.2-1B-Instruct as a scalar reward function r_ψ(x, y) ∈ ℝ.

    Architecture:
        Llama backbone (frozen or LoRA-adapted)
        └── AutoModelForSequenceClassification head: Linear(hidden_size, 1)

    The scalar is read at the LAST NON-PAD TOKEN position of the sequence.
    For a right-padded sequence [tok1...tokN, PAD...PAD], this is tokN,
    which carries the accumulated attention over the full (x, y) pair.

    Usage (after training):
        reward = rm(input_ids, attention_mask)   # shape: (batch,)
    """

    def __init__(self, backbone: nn.Module):
        """
        Parameters
        ----------
        backbone : AutoModelForSequenceClassification loaded via model/loader.py
                   (already has the Linear head; this class is a thin wrapper)
        """
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        input_ids: torch.Tensor,       # shape: (batch, seq_len)
        attention_mask: torch.Tensor,  # shape: (batch, seq_len)
    ) -> torch.Tensor:
        """
        Returns
        -------
        reward : shape (batch,)
            Raw scalar reward scores. NOT normalised — can be any real number.
            During training: (r⁺ - r⁻) should be > 0 for preferred responses.
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # outputs.logits : shape (batch, 1)  ← from Linear(hidden_size, 1)
        reward = outputs.logits.squeeze(-1)  # shape: (batch,)
        return reward

    @torch.no_grad()
    def score(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Inference-only scoring (no grad). Convenient alias for evaluation."""
        return self.forward(input_ids, attention_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Loss function
# ─────────────────────────────────────────────────────────────────────────────

def compute_rm_loss(
    r_pos: torch.Tensor,           # shape: (batch,) — scores for chosen responses
    r_neg: torch.Tensor,           # shape: (batch,) — scores for rejected responses
    lambda_reg: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Bradley-Terry margin ranking loss with L2 regularisation.

    Mathematical formulation:
    ─────────────────────────
    Preference loss (from Bradley-Terry model):
        L_pref = -E[log σ(r⁺ - r⁻)]

    This is the negative log-likelihood of the observed preference under
    the assumption that P(y⁺ ≻ y⁻ | x) = σ(r⁺ - r⁻).

    Regularisation (prevents reward magnitude explosion):
        L_reg = λ · E[(r⁺)² + (r⁻)²]

    Total:
        L_RM = L_pref + L_reg

    Parameters
    ----------
    r_pos      : reward scores for chosen (preferred) responses
    r_neg      : reward scores for rejected responses
    lambda_reg : regularisation coefficient (default: cfg.rm.reg_lambda = 1e-3)

    Returns
    -------
    (total_loss, pref_loss, reg_loss) — all scalars (for logging)
    """
    if lambda_reg is None:
        lambda_reg = cfg.rm.reg_lambda

    # ── Preference loss ──────────────────────────────────────────────────
    # logsigmoid(x) = log(1 / (1 + e^{-x})) is numerically stable.
    # We want: -E[log σ(r⁺ - r⁻)] = -E[logsigmoid(r⁺ - r⁻)]
    pref_loss = -F.logsigmoid(r_pos - r_neg).mean()  # scalar

    # ── Regularisation loss ──────────────────────────────────────────────
    # Penalise both r⁺ and r⁻ for having large magnitudes.
    # Without this, the model can set r⁺ = 1000, r⁻ = -1000 — technically
    # correct preference ranking but useless for RL training where rewards
    # are used as advantage estimates.
    reg_loss = lambda_reg * (r_pos.pow(2) + r_neg.pow(2)).mean()  # scalar

    total_loss = pref_loss + reg_loss

    return total_loss, pref_loss, reg_loss


def compute_preference_accuracy(
    r_pos: torch.Tensor,  # shape: (batch,)
    r_neg: torch.Tensor,  # shape: (batch,)
) -> float:
    """
    Fraction of pairs where the chosen response scores higher.
        P(r⁺ > r⁻) on this batch.
    Target: ≥ 60% before using RM in downstream RL.
    """
    correct = (r_pos > r_neg).float()  # shape: (batch,), 1.0 if correct
    return correct.mean().item()        # scalar in [0, 1]
