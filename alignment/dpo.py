"""
alignment/dpo.py — Direct Preference Optimisation (Task C4)
============================================================

PSEUDOCODE:
────────────
dpo_loss(policy, batch):
  # π_θ forward passes (gradients flow here)
  logits_θ_pos  = policy(chosen_input_ids)           shape: (B, L, V)
  logits_θ_neg  = policy(rejected_input_ids)         shape: (B, L, V)
  log_θ_pos  = sum_response_log_probs(logits_θ_pos, chosen_labels)   shape: (B,)
  log_θ_neg  = sum_response_log_probs(logits_θ_neg, rejected_labels) shape: (B,)

  # π_ref forward passes (NO GRAD — reference model is frozen)
  with reference_model_ctx(policy):
      logits_ref_pos = policy(chosen_input_ids)
      logits_ref_neg = policy(rejected_input_ids)
  log_ref_pos = sum_response_log_probs(logits_ref_pos, chosen_labels).detach()
  log_ref_neg = sum_response_log_probs(logits_ref_neg, rejected_labels).detach()

  Δ_θ   = log_θ_pos  - log_θ_neg           shape: (B,)
  Δ_ref  = log_ref_pos - log_ref_neg        shape: (B,)
  z      = β·(Δ_θ - Δ_ref)                 shape: (B,)
  L_DPO  = -logsigmoid(z).mean()            scalar

CRITICAL BUGS:
  ① Forgetting to mask prompt + pad tokens in log-prob sum.
     If you sum ALL token log-probs, longer responses get larger gradient
     signals purely by length (verbosity bias), not quality.
  ② Not using no_grad for π_ref. If gradients flow through the reference
     model, the loss gradient will include ∂Δ_ref/∂θ terms that corrupt
     the intended update direction.
  ③ At initialisation: π_θ ≈ π_ref, so Δ_θ ≈ Δ_ref, z ≈ 0, σ(z) ≈ 0.5
     → preference accuracy ≈ 50%. Verify this before training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from config import cfg
from model.lora_setup import reference_model_ctx


# ─────────────────────────────────────────────────────────────────────────────
# Core: sum log-probs over response tokens only
# ─────────────────────────────────────────────────────────────────────────────

def sum_response_log_probs(
    logits: torch.Tensor,   # shape: (batch, seq_len, vocab_size)
    labels: torch.Tensor,   # shape: (batch, seq_len)  —  -100 for prompt+pad
) -> torch.Tensor:
    """
    Compute log π(y|x) = Σ_{t ∈ response} log π(y_t | x, y_{<t}).

    This is the SUM (not mean) of log-probs over response tokens.
    We use SUM because the DPO gradient uses the TOTAL log-probability
    of the full response, not a per-token average. Using mean would
    introduce a hidden normalisation by response length, changing the
    effective β and creating different gradient magnitudes for
    chosen vs. rejected sequences of different lengths.

    Implementation
    ──────────────
    Causal shift: logits[:, t, :] predicts labels[:, t+1].
    So we shift logits left by 1 and labels right by 1:
        shift_logits = logits[:, :-1, :]   (B, L-1, V)
        shift_labels = labels[:, 1:]       (B, L-1)
    Response positions: where shift_labels != -100.
    At -100 positions we substitute index 0 (dummy) and mask the log-prob out.

    Upcast to float32 before log_softmax — same reason as in ppo.py
    (bfloat16 mantissa too coarse for accurate tail log-probs).

    Parameters
    ----------
    logits : raw (un-normalised) model outputs
    labels : token ids at response positions, -100 elsewhere

    Returns
    -------
    log_prob_sum : shape (batch,)  — Σ_t log π(y_t|s_t) for response tokens
    """
    B, L, V = logits.shape

    # Causal shift
    shift_logits = logits[:, :-1, :].float()    # (B, L-1, V)
    shift_labels = labels[:, 1:]                # (B, L-1)

    # Response mask: 1 where labels != -100 (real response token), 0 elsewhere
    response_mask = (shift_labels != -100).float()  # (B, L-1)

    # Replace -100 with dummy index 0 before gather (avoid index-out-of-range).
    # These positions will be zeroed out by response_mask anyway.
    gather_labels = shift_labels.clone()
    gather_labels[shift_labels == -100] = 0    # (B, L-1)

    # Log-prob of each token
    log_probs = F.log_softmax(shift_logits, dim=-1)   # (B, L-1, V)
    token_log_probs = log_probs.gather(
        dim=-1,
        index=gather_labels.unsqueeze(-1),             # (B, L-1, 1)
    ).squeeze(-1)                                       # (B, L-1)

    # Sum ONLY over response tokens (mask zeros out prompt and pad)
    log_prob_sum = (token_log_probs * response_mask).sum(dim=-1)   # (B,)

    return log_prob_sum


# ─────────────────────────────────────────────────────────────────────────────
# DPO loss
# ─────────────────────────────────────────────────────────────────────────────

def dpo_loss(
    policy,
    chosen_ids: torch.Tensor,       # (B, L)
    chosen_mask: torch.Tensor,      # (B, L)
    chosen_labels: torch.Tensor,    # (B, L)  -100 for prompt+pad
    rejected_ids: torch.Tensor,     # (B, L)
    rejected_mask: torch.Tensor,    # (B, L)
    rejected_labels: torch.Tensor,  # (B, L)  -100 for prompt+pad
    beta: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    DPO loss for one batch of preference pairs.

    Math:
        L_DPO = -E[(x,y⁺,y⁻)] [ log σ(β·(Δ_θ - Δ_ref)) ]

    where Δ_θ = log π_θ(y⁺|x) - log π_θ(y⁻|x)
          Δ_ref = log π_ref(y⁺|x) - log π_ref(y⁻|x)

    Four forward passes total:
        (a) π_θ on chosen          ← gradient flows (trainable LoRA)
        (b) π_θ on rejected        ← gradient flows (trainable LoRA)
        (c) π_ref on chosen        ← NO gradient (reference_model_ctx)
        (d) π_ref on rejected      ← NO gradient (reference_model_ctx)

    Returns
    -------
    loss    : scalar DPO loss
    metrics : dict with monitoring quantities
    """

    # ── π_θ forward passes (gradients flow through here) ─────────────────
    # We do chosen and rejected separately because they may have different
    # sequence lengths. (In practice with our padding they're the same, but
    # keeping them separate is explicit and correct.)
    logits_theta_pos = policy(
        input_ids=chosen_ids, attention_mask=chosen_mask
    ).logits                                              # (B, L, V)

    logits_theta_neg = policy(
        input_ids=rejected_ids, attention_mask=rejected_mask
    ).logits                                              # (B, L, V)

    # Σ_t log π_θ(y_t|s_t) over response tokens
    log_theta_pos = sum_response_log_probs(logits_theta_pos, chosen_labels)    # (B,)
    log_theta_neg = sum_response_log_probs(logits_theta_neg, rejected_labels)  # (B,)

    # ── π_ref forward passes (NO gradients) ──────────────────────────────
    # reference_model_ctx disables LoRA adapters on the policy, making it
    # behave as π_ref (the merged SFT checkpoint). The nested no_grad
    # ensures ZERO gradient flows through these computations.
    with reference_model_ctx(policy) as ref:
        logits_ref_pos = ref(
            input_ids=chosen_ids, attention_mask=chosen_mask
        ).logits                                          # (B, L, V)

        logits_ref_neg = ref(
            input_ids=rejected_ids, attention_mask=rejected_mask
        ).logits                                          # (B, L, V)

    # log_ref values are computed under no_grad → they have no gradient.
    # We still explicitly call .detach() as defensive programming.
    log_ref_pos = sum_response_log_probs(logits_ref_pos, chosen_labels).detach()    # (B,)
    log_ref_neg = sum_response_log_probs(logits_ref_neg, rejected_labels).detach()  # (B,)

    # ── DPO margin ────────────────────────────────────────────────────────
    # Δ_θ = log π_θ(y⁺|x) - log π_θ(y⁻|x)    how much more the policy
    #        prefers chosen over rejected (in log-space)
    # Δ_ref = same quantity for π_ref
    # z = β·(Δ_θ - Δ_ref): how much π_θ has shifted its preference
    #     RELATIVE to π_ref. Positive z → π_θ more strongly prefers y⁺.
    delta_theta = log_theta_pos - log_theta_neg   # (B,)
    delta_ref   = log_ref_pos   - log_ref_neg     # (B,)
    z           = beta * (delta_theta - delta_ref) # (B,)

    # ── Loss ─────────────────────────────────────────────────────────────
    # -log σ(z) = log(1 + e^{-z})
    # logsigmoid(z) = -log(1 + e^{-z}) is numerically stable via PyTorch.
    loss = -F.logsigmoid(z).mean()   # scalar

    # ── Monitoring metrics ────────────────────────────────────────────────
    with torch.no_grad():
        # Implicit rewards: r_impl(x,y) = β·(log π_θ(y|x) - log π_ref(y|x))
        # This is the reward DPO IMPLICITLY learns (from PA2 Problem 3.1(c))
        impl_r_pos  = beta * (log_theta_pos - log_ref_pos)   # (B,)
        impl_r_neg  = beta * (log_theta_neg - log_ref_neg)   # (B,)

        pref_acc    = (impl_r_pos > impl_r_neg).float().mean().item()
        margin      = z.mean().item()
        sigma_z     = torch.sigmoid(z).mean().item()

        # Log-prob lengths for monitoring verbosity bias
        chosen_len  = (chosen_labels != -100).float().sum(dim=1).mean().item()
        rejected_len = (rejected_labels != -100).float().sum(dim=1).mean().item()

    metrics = {
        "dpo_loss":            loss.item(),
        "z_mean":              margin,
        "sigma_z_mean":        sigma_z,
        "pref_acc":            pref_acc,
        "impl_r_pos_mean":     impl_r_pos.mean().item(),
        "impl_r_neg_mean":     impl_r_neg.mean().item(),
        "impl_r_margin":       (impl_r_pos - impl_r_neg).mean().item(),
        "log_theta_pos_mean":  log_theta_pos.mean().item(),
        "log_theta_neg_mean":  log_theta_neg.mean().item(),
        "chosen_resp_len":     chosen_len,
        "rejected_resp_len":   rejected_len,
    }

    return loss, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_dpo(
    policy,
    dataloader,
    device: torch.device,
    beta: float = 0.1,
    max_batches: int = 50,
) -> Dict[str, float]:
    """
    Evaluate DPO metrics on held-out preference pairs.

    Reports:
        - DPO loss
        - Preference accuracy P(implicit r⁺ > implicit r⁻)
        - Mean reward margin (implicit r⁺ - implicit r⁻)

    Sanity check at initialisation (step 0):
        pref_acc ≈ 50%  (π_θ ≈ π_ref → z ≈ 0 → random preference)
    After 100 steps:
        pref_acc should be rising above 55–60%.
        If stuck at 50%, check π_ref is frozen and response masking is correct.
    """
    policy.eval()
    accum = {
        "dpo_loss": 0.0, "pref_acc": 0.0,
        "impl_r_margin": 0.0, "z_mean": 0.0
    }
    n = 0

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break

        chosen_ids      = batch["chosen_input_ids"].to(device)
        chosen_mask     = batch["chosen_attention_mask"].to(device)
        chosen_labels   = batch["chosen_labels"].to(device)
        rejected_ids    = batch["rejected_input_ids"].to(device)
        rejected_mask   = batch["rejected_attention_mask"].to(device)
        rejected_labels = batch["rejected_labels"].to(device)

        _, metrics = dpo_loss(
            policy,
            chosen_ids, chosen_mask, chosen_labels,
            rejected_ids, rejected_mask, rejected_labels,
            beta=beta,
        )

        for k in accum:
            accum[k] += metrics[k]
        n += 1

    result = {k: v / max(n, 1) for k, v in accum.items()}
    policy.train()
    return result
