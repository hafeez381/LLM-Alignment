"""
alignment/grpo.py — Group Relative Policy Optimisation (Task C5)
=================================================================

Mathematical Reference
──────────────────────
GRPO eliminates the value network by using the GROUP MEAN as the baseline.

For prompt x_b, K completions {y_{b,k}}_{k=1}^K:
    r_{b,k}   = r_ψ(x_b ⊕ y_{b,k})          ← RM score for each completion
    μ_b        = (1/K) · Σ_k r_{b,k}          ← group mean baseline
    A_{b,k}   = r_{b,k} - μ_b                  ← group-relative advantage
    Ã_{b,k,t} = (A_{b,k} - μ_A) / (σ_A + ε)  ← batch-wide standardisation

GRPO loss (eq. 12):
    L^GRPO = -(1/K) · Σ_k (1/T_k) · Σ_t min(ρ_{k,t}·Ã_{k,t}, clip(ρ_{k,t},1-ε,1+ε)·Ã_{k,t})
             + β · KL(π_θ || π_ref)

where ρ_{k,t} = π_θ(y_{k,t}|s_t) / π_old(y_{k,t}|s_t)

KL term (two options, see below):
    Full:   KL_t = Σ_v π_θ(v|s_t) · [log π_θ(v|s_t) - log π_ref(v|s_t)]
    MC approx: KL_t ≈ log π_θ(y_t|s_t) - log π_ref(y_t|s_t)

Key difference from PPO:
    PPO:  A_t varies at each TOKEN (from GAE with V_φ)
    GRPO: A_{b,k} is the SAME for all tokens in completion k (no V_φ)
    → GRPO gives uniform credit to every token in a completion.
       A correct completion's every token (even filler words) gets +A.
       This is the "credit assignment weakness" from PA2 Problem 4.1(b).

Length normalisation (1/T_k):
    Prevents long completions from dominating the gradient.
    BUT creates a bias: identical A_{b,k} gives a stronger PER-TOKEN gradient
    to short completions than long ones (PA2 Problem 4.2).

PSEUDOCODE:
────────────
group_rollout(policy, rm, prompts, prompt_mask, K, max_new_tokens):
  policy.eval()
  For prompt b in batch (B prompts):
    1. Repeat prompt_ids K times → (K, P) tiled prompt batch
    2. Generate K completions in ONE batched generate() call:
       full_ids = policy.generate(tiled_prompts, do_sample=True, temp=0.7, top_p=0.9)
                  shape: (K, P+R_k)  — R_k may differ per completion due to EOS
    3. Pad all K completions to same R = max R_k with EOS tokens
    4. Build response_mask for each completion
    5. Cache log_probs_old: forward full_ids through π_θ (LoRA ON)
    6. Cache log_probs_ref: forward full_ids through π_ref (LoRA OFF, no_grad)
    7. Score all K completions with frozen RM → {r_{b,k}}
    8. Compute μ_b, A_{b,k} = r_{b,k} - μ_b
    9. Broadcast A_{b,k} → shape (R,) by repeating across all token positions
    10. Check if batch is degenerate: std(r_{b,:}) < 1e-6

grpo_update(policy, optimizer, rollout_buffer):
  policy.train()
  For mini_batch from rollout_buffer:
    a. Recompute log_probs_new under current π_θ  (grad flows)
    b. ρ_{k,t} = exp(log_new - log_old_CACHED)
    c. L_GRPO = -(1/K)·Σ_k (1/T_k)·Σ_t min(ρ·Ã, clip(ρ,1-ε,1+ε)·Ã)
    d. KL term (full or MC approx)
    e. loss = L_GRPO + β·KL
    f. backward → clip_grad_norm → optimizer.step
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from config import cfg
from model.lora_setup import reference_model_ctx


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (shared logic with ppo.py, duplicated here for module independence)
# ─────────────────────────────────────────────────────────────────────────────

def _token_log_probs(
    model: nn.Module,
    full_ids: torch.Tensor,     # (B, P+R)
    full_attn: torch.Tensor,    # (B, P+R)
    prompt_len: int,
) -> torch.Tensor:
    """
    Per-token log π(y_t | x, y_{<t}) at every response position.

    Causal shift: logits[:, P-1+k, :] predicts full_ids[:, P+k].
    Cast to float32 before log_softmax (bfloat16 mantissa insufficient
    for accurate rare-token log-probs → NaN in ρ_t).

    Returns shape (B, R).
    """
    outputs = model(input_ids=full_ids, attention_mask=full_attn)
    logits  = outputs.logits                                        # (B, P+R, V)
    R = full_ids.shape[1] - prompt_len
    resp_logits = logits[:, prompt_len - 1 : prompt_len - 1 + R, :]  # (B, R, V)
    resp_ids    = full_ids[:, prompt_len:]                            # (B, R)
    lp = F.log_softmax(resp_logits.float(), dim=-1)                   # (B, R, V)
    return lp.gather(-1, resp_ids.unsqueeze(-1)).squeeze(-1)          # (B, R)


def _token_log_probs_full(
    model: nn.Module,
    full_ids: torch.Tensor,     # (B, P+R)
    full_attn: torch.Tensor,    # (B, P+R)
    prompt_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns BOTH the per-token sampled log-prob AND the full log-prob
    distribution over the vocabulary at every response position.

    Used for the exact full-vocabulary KL term.

    Returns
    -------
    token_lp   : shape (B, R)   — log π(y_t|s_t) for the sampled token
    full_log_p : shape (B, R, V) — log π(·|s_t) for all V tokens
    """
    outputs = model(input_ids=full_ids, attention_mask=full_attn)
    logits  = outputs.logits
    R = full_ids.shape[1] - prompt_len
    resp_logits = logits[:, prompt_len - 1 : prompt_len - 1 + R, :]  # (B, R, V)
    resp_ids    = full_ids[:, prompt_len:]                            # (B, R)
    log_p       = F.log_softmax(resp_logits.float(), dim=-1)          # (B, R, V)
    token_lp    = log_p.gather(-1, resp_ids.unsqueeze(-1)).squeeze(-1)  # (B, R)
    return token_lp, log_p


def _build_response_mask(
    resp_ids: torch.Tensor,   # (B, R)
    eos_token_id: int,
) -> torch.Tensor:
    """
    1 for real tokens up to and including first EOS, 0 for padding after EOS.
    Returns shape (B, R), dtype=long.
    """
    B, R   = resp_ids.shape
    device = resp_ids.device
    is_eos    = (resp_ids == eos_token_id)
    first_eos = is_eos.long().argmax(dim=1)
    has_eos   = is_eos.any(dim=1)
    cutoff    = torch.where(has_eos, first_eos,
                            torch.tensor(R - 1, device=device))
    positions     = torch.arange(R, device=device).unsqueeze(0)
    response_mask = (positions <= cutoff.unsqueeze(1)).long()
    return response_mask


@torch.no_grad()
def _rm_score_batch(
    rm,
    rm_tok,
    raw_prompts: List[str],
    resp_ids: torch.Tensor,    # (B, R)
    policy_tok,
    device: torch.device,
) -> torch.Tensor:
    """Decode generated tokens → re-tokenise with RM tokeniser → score. Returns (B,)."""
    texts = [
        raw_prompts[i] + " " + policy_tok.decode(resp_ids[i], skip_special_tokens=True)
        for i in range(len(raw_prompts))
    ]
    enc    = rm_tok(texts, max_length=cfg.tokenizer.max_seq_len,
                   padding=True, truncation=True, return_tensors="pt").to(device)
    scores = rm(enc["input_ids"], enc["attention_mask"])
    return scores.detach()


# ─────────────────────────────────────────────────────────────────────────────
# Group rollout
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def group_rollout(
    policy,
    rm,
    rm_tok,
    prompt_batch: Dict,       # one batch from PromptDataset DataLoader
    policy_tok,
    device: torch.device,
    grpo_cfg=None,
    reward_fn=None,           # optional: callable(raw_prompts, resp_ids, policy_tok) → (B*K,)
                              #  used to swap in verifiable reward for RLVR
) -> Dict:
    """
    For each of B prompts, generate K completions and collect all data
    needed for the GRPO update.

    Returns a SINGLE buffer dict (not a list) containing:
        full_ids       : (B*K, P+R)   — prompt + response token ids
        full_attn      : (B*K, P+R)
        prompt_len     : int
        response_mask  : (B*K, R)
        log_probs_old  : (B*K, R)     ← CACHED under π_old, NOT recomputed later
        log_probs_ref  : (B*K, R)     ← π_ref (LoRA OFF)
        advantages     : (B*K, R)     ← A_{b,k} broadcast to all token positions
        rm_scores      : (B*K,)       ← raw (pre-normalisation) RM scores
        group_means    : (B,)         ← μ_b for each prompt
        response_lens  : (B*K,)       ← T_k = number of real response tokens
        is_degenerate  : (B,)         ← bool, True if std(r_{b,:}) < 1e-6

    WHY single dict instead of list-of-dicts (unlike PPO)?
        GRPO processes all K completions for a prompt TOGETHER to compute
        μ_b. Keeping them in a single tensor block makes the group mean
        calculation trivial (reshape to (B, K, ...) and mean over K).
    """
    if grpo_cfg is None:
        grpo_cfg = cfg.grpo

    policy.eval()
    K = grpo_cfg.K

    prompt_ids   = prompt_batch["input_ids"].to(device)        # (B, P)
    prompt_mask  = prompt_batch["attention_mask"].to(device)   # (B, P)
    raw_prompts  = prompt_batch["raw_prompt"]                  # list[str], len=B

    B, P = prompt_ids.shape

    # ── Step 1: Tile each prompt K times for one batched generate() call ──
    # Tiling: prompt_ids[b] → [prompt_ids[b]] × K in the batch dimension.
    # This lets us sample K independent completions per prompt in one pass.
    tiled_ids  = prompt_ids.repeat_interleave(K, dim=0)    # (B*K, P)
    tiled_mask = prompt_mask.repeat_interleave(K, dim=0)   # (B*K, P)
    tiled_raw  = [p for p in raw_prompts for _ in range(K)]  # len B*K

    # ── Step 2: Generate K completions per prompt (batched) ───────────────
    # temperature=0.7, top_p=0.9 for diversity. Without randomness, all K
    # completions would be identical → μ_b = r_{b,1} → all A_{b,k} = 0.
    full_ids_gen = policy.generate(
        input_ids=tiled_ids,
        attention_mask=tiled_mask,
        max_new_tokens=grpo_cfg.max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=policy_tok.pad_token_id,
        eos_token_id=policy_tok.eos_token_id,
        use_cache=True,
    )  # (B*K, P+R_gen) — R_gen ≤ max_new_tokens

    R = full_ids_gen.shape[1] - P

    resp_ids = full_ids_gen[:, P:]  # (B*K, R)

    # ── Step 3: Response masks and full attention mask ─────────────────────
    response_mask = _build_response_mask(resp_ids, policy_tok.eos_token_id)  # (B*K, R)
    full_attn     = torch.cat([tiled_mask, response_mask], dim=1)             # (B*K, P+R)

    # ── Step 4: Cache π_old log-probs (LoRA ON) ───────────────────────────
    # These are the denominator of ρ_{k,t} throughout ALL update steps.
    # MUST be cached here — not recomputed in the update loop.
    log_probs_old = _token_log_probs(policy, full_ids_gen, full_attn, P)  # (B*K, R)

    # ── Step 5: Cache π_ref log-probs (LoRA OFF, no_grad) ─────────────────
    with reference_model_ctx(policy) as ref:
        log_probs_ref = _token_log_probs(ref, full_ids_gen, full_attn, P)  # (B*K, R)

    # ── Step 6: Score all completions ─────────────────────────────────────
    if reward_fn is not None:
        # RLVR path: verifiable reward r_v ∈ {0, 1}
        rm_scores = reward_fn(tiled_raw, resp_ids, policy_tok)  # (B*K,)
    else:
        # GRPO path: learned RM
        rm_scores = _rm_score_batch(rm, rm_tok, tiled_raw, resp_ids,
                                    policy_tok, device)           # (B*K,)

    # ── Step 7: Group-relative advantages ────────────────────────────────
    # Reshape to (B, K) to compute group mean μ_b per prompt
    scores_bk    = rm_scores.view(B, K)                          # (B, K)
    group_means  = scores_bk.mean(dim=1)                         # (B,)
    group_stds   = scores_bk.std(dim=1)                          # (B,)

    # A_{b,k} = r_{b,k} - μ_b
    adv_bk = scores_bk - group_means.unsqueeze(1)               # (B, K)
    adv_flat = adv_bk.view(B * K)                                # (B*K,)

    # Degenerate batch: all K completions for a prompt have the same reward
    # → std ≈ 0 → all A_{b,k} ≈ 0 → zero gradient contribution.
    # This is expected early in training for hard tasks (all wrong) or
    # trivially easy prompts (all perfect). Track fraction over training.
    is_degenerate = (group_stds < 1e-6)                          # (B,) bool

    # ── Step 8: Broadcast advantage to all token positions ────────────────
    # A_{b,k} is a sequence-level scalar: the same value applies to every
    # token in completion k. Broadcast: (B*K,) → (B*K, R).
    #
    # WHY broadcast vs. token-level? GRPO has no value function, so there's
    # no way to know WHICH tokens contributed to the final reward. All tokens
    # are credited equally. This is the "uniform credit assignment" weakness
    # from PA2 Problem 4.1(b) — correct tokens and filler tokens get the
    # same advantage signal.
    advantages = adv_flat.unsqueeze(1).expand(-1, R)             # (B*K, R)
    advantages = advantages * response_mask                       # zero padding

    # ── Step 9: Batch-wide advantage standardisation ──────────────────────
    # Standardise over ALL non-padding response positions in the batch.
    # This stabilises the gradient magnitude regardless of RM scale.
    real_advs = advantages[response_mask.bool()]
    if real_advs.numel() > 1:
        mu_A  = real_advs.mean()
        sig_A = real_advs.std() + 1e-8
        advantages = torch.where(
            response_mask.bool(),
            (advantages - mu_A) / sig_A,
            torch.zeros_like(advantages),
        )

    # Response lengths T_k: number of real response tokens per completion
    response_lens = response_mask.sum(dim=1).float()             # (B*K,)

    # ── Step 10: Store on CPU to free GPU VRAM for the update phase ────────
    return {
        "full_ids":      full_ids_gen.cpu(),   # (B*K, P+R)
        "full_attn":     full_attn.cpu(),       # (B*K, P+R)
        "prompt_len":    P,
        "response_mask": response_mask.cpu(),  # (B*K, R)
        "log_probs_old": log_probs_old.cpu(),  # (B*K, R)  ← CACHED π_old
        "log_probs_ref": log_probs_ref.cpu(),  # (B*K, R)
        "advantages":    advantages.cpu(),      # (B*K, R)  ← standardised
        "rm_scores":     rm_scores.cpu(),       # (B*K,)
        "group_means":   group_means.cpu(),     # (B,)
        "response_lens": response_lens.cpu(),   # (B*K,)
        "is_degenerate": is_degenerate.cpu(),   # (B,)
        # Monitoring
        "mean_rm":           rm_scores.mean().item(),
        "frac_degenerate":   is_degenerate.float().mean().item(),
        "mean_resp_len":     response_lens.mean().item(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GRPO loss
# ─────────────────────────────────────────────────────────────────────────────

def grpo_loss(
    log_probs_new: torch.Tensor,    # (B*K, R) — current π_θ, GRAD FLOWS
    log_probs_old: torch.Tensor,    # (B*K, R) — CACHED at rollout, NO GRAD
    log_probs_ref: torch.Tensor,    # (B*K, R) — cached π_ref, NO GRAD
    advantages: torch.Tensor,       # (B*K, R) — batch-standardised, NO GRAD
    response_mask: torch.Tensor,    # (B*K, R) — 1=real token
    response_lens: torch.Tensor,    # (B*K,)   — T_k for each completion
    epsilon: float = 0.2,
    beta: float = 0.1,
    full_log_p_new: Optional[torch.Tensor] = None,  # (B*K, R, V) for full KL
    full_log_p_ref: Optional[torch.Tensor] = None,  # (B*K, R, V) for full KL
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    GRPO loss = clipped surrogate objective + KL penalty (eq. 12).

    Math:
        L^GRPO = -(1/K)·Σ_k (1/T_k)·Σ_t min(ρ·Ã, clip(ρ,1-ε,1+ε)·Ã)
                 + β·KL(π_θ||π_ref)

    The (1/T_k) normalisation divides each completion's contribution by its
    length. This prevents long completions from having larger raw gradient
    magnitudes just because they have more tokens — a necessary fairness
    correction when mixing completions of different lengths in the same batch.

    KL term options:
        Full (exact):
            KL_t = Σ_v π_θ(v|s_t)·[log π_θ(v|s_t) - log π_ref(v|s_t)]
            Requires full_log_p_new and full_log_p_ref (expensive: V=32k vocab)

        MC approximation (default if full distributions not provided):
            KL_t ≈ log π_θ(y_t|s_t) - log π_ref(y_t|s_t)
            Unbiased but high variance for low-probability tokens.
            Preferred for T4 with limited VRAM.

    Parameters
    ----------
    log_probs_new  : per-token log π_θ(y_t|s_t) for sampled tokens (grad flows)
    log_probs_old  : CACHED per-token log π_old(y_t|s_t) (no grad)
    log_probs_ref  : CACHED per-token log π_ref(y_t|s_t) (no grad)
    advantages     : Ã_{b,k,t} — standardised, broadcast per completion
    response_mask  : binary mask (1=real response token)
    response_lens  : T_k for each of the B*K completions
    epsilon        : PPO clip threshold
    beta           : KL penalty coefficient
    full_log_p_new : optional (B*K, R, V) for exact KL
    full_log_p_ref : optional (B*K, R, V) for exact KL

    Returns
    -------
    loss    : scalar — total GRPO loss (to minimise)
    metrics : dict with named components for logging
    """
    BK, R  = log_probs_new.shape
    device = log_probs_new.device

    # ── Importance ratio ──────────────────────────────────────────────────
    # ρ_{k,t} = π_θ(y_{k,t}|s_t) / π_old(y_{k,t}|s_t)
    #         = exp(log π_θ - log π_old_CACHED)
    log_ratio = log_probs_new - log_probs_old                    # (B*K, R)
    rho       = torch.exp(log_ratio)                              # (B*K, R)

    # ── Clipped surrogate with length normalisation ───────────────────────
    surr_unclipped = rho         * advantages                     # (B*K, R)
    rho_clipped    = rho.clamp(1.0 - epsilon, 1.0 + epsilon)
    surr_clipped   = rho_clipped * advantages                     # (B*K, R)
    surr_min       = torch.min(surr_unclipped, surr_clipped)      # (B*K, R)

    # Apply response_mask and length normalisation (1/T_k):
    # For each completion k, sum masked surrogate values then divide by T_k.
    # T_k is the NUMBER OF REAL response tokens (from response_lens).
    T_k          = response_lens.clamp(min=1.0).to(device)        # (B*K,)
    # Sum surrogate over response tokens per completion, then normalise by T_k
    surr_per_seq = (surr_min * response_mask).sum(dim=1)           # (B*K,)
    # Divide each completion's contribution by its length
    surr_normed  = surr_per_seq / T_k                              # (B*K,)
    # Mean over all B*K completions
    clip_loss    = surr_normed.mean()                              # scalar

    # ── KL penalty ────────────────────────────────────────────────────────
    if full_log_p_new is not None and full_log_p_ref is not None:
        # Exact full-vocabulary KL:
        # KL_t = Σ_v π_θ(v|s_t)·[log π_θ(v|s_t) - log π_ref(v|s_t)]
        p_theta  = full_log_p_new.exp()                            # (B*K, R, V)
        kl_per_t = (p_theta * (full_log_p_new - full_log_p_ref)
                    ).sum(dim=-1)                                   # (B*K, R)
    else:
        # MC approximation: KL_t ≈ log π_θ(y_t) - log π_ref(y_t)
        # Unbiased, low-cost. Sufficient for T4 memory budget.
        kl_per_t = log_probs_new - log_probs_ref                   # (B*K, R)

    # Average KL over real response positions
    n_real   = response_mask.sum().clamp(min=1.0)
    kl_mean  = (kl_per_t * response_mask).sum() / n_real           # scalar

    # ── Total loss ────────────────────────────────────────────────────────
    # L^GRPO = -L_clip  +  β·KL
    # We MAXIMISE L_clip (hence negate it), and add KL penalty to
    # prevent the policy from drifting too far from π_ref.
    loss = -clip_loss + beta * kl_mean                             # scalar

    # ── Metrics ───────────────────────────────────────────────────────────
    with torch.no_grad():
        real_rho = rho[response_mask.bool()]
    metrics = {
        "grpo_loss":   loss.item(),
        "clip_loss":   clip_loss.item(),
        "kl_mean":     kl_mean.item(),
        "rho_mean":    real_rho.mean().item(),
        "rho_max":     real_rho.max().item(),
        "rho_min":     real_rho.min().item(),
    }
    return loss, metrics


# ─────────────────────────────────────────────────────────────────────────────
# GRPO update loop
# ─────────────────────────────────────────────────────────────────────────────

def grpo_update(
    policy,
    optimizer,
    rollout_buffer: Dict,      # single dict from group_rollout()
    device: torch.device,
    grpo_cfg=None,
    use_full_kl: bool = False, # set True for exact KL (more VRAM)
) -> Dict[str, float]:
    """
    Single gradient update step on the collected group rollout.

    Unlike PPO which runs K epochs over a buffer, GRPO typically does
    a SINGLE update pass per rollout (the original paper's formulation).
    Multiple update epochs are possible but increase the staleness of
    ρ_{k,t} (since log_probs_old is cached from the rollout).

    Returns
    -------
    metrics : dict of loss components
    """
    if grpo_cfg is None:
        grpo_cfg = cfg.grpo

    policy.train()

    # Move buffer tensors to device
    full_ids      = rollout_buffer["full_ids"].to(device)        # (B*K, P+R)
    full_attn     = rollout_buffer["full_attn"].to(device)       # (B*K, P+R)
    prompt_len    = rollout_buffer["prompt_len"]
    response_mask = rollout_buffer["response_mask"].to(device)   # (B*K, R)
    log_probs_old = rollout_buffer["log_probs_old"].to(device)   # (B*K, R)
    log_probs_ref = rollout_buffer["log_probs_ref"].to(device)   # (B*K, R)
    advantages    = rollout_buffer["advantages"].to(device)      # (B*K, R)
    response_lens = rollout_buffer["response_lens"].to(device)   # (B*K,)

    R  = full_ids.shape[1] - prompt_len
    BK = full_ids.shape[0]

    # ── Recompute log-probs under CURRENT π_θ ────────────────────────────
    # This is the NUMERATOR of ρ. Gradients flow through here.
    # log_probs_old is the cached DENOMINATOR — NOT recomputed.
    if use_full_kl:
        log_probs_new, full_log_p_new = _token_log_probs_full(
            policy, full_ids, full_attn, prompt_len
        )  # (B*K, R),  (B*K, R, V)

        with reference_model_ctx(policy) as ref:
            with torch.no_grad():
                _, full_log_p_ref = _token_log_probs_full(
                    ref, full_ids, full_attn, prompt_len
                )  # (B*K, R, V) — no grad
    else:
        outputs     = policy(input_ids=full_ids, attention_mask=full_attn)
        logits      = outputs.logits
        resp_logits = logits[:, prompt_len - 1 : prompt_len - 1 + R, :]  # (B*K, R, V)
        resp_ids    = full_ids[:, prompt_len:]                            # (B*K, R)
        log_p_all   = F.log_softmax(resp_logits.float(), dim=-1)          # (B*K, R, V)
        log_probs_new = log_p_all.gather(
            -1, resp_ids.unsqueeze(-1)
        ).squeeze(-1)                                                       # (B*K, R)
        full_log_p_new = None
        full_log_p_ref = None

    # ── GRPO loss ─────────────────────────────────────────────────────────
    loss, metrics = grpo_loss(
        log_probs_new=log_probs_new,
        log_probs_old=log_probs_old,
        log_probs_ref=log_probs_ref,
        advantages=advantages,
        response_mask=response_mask,
        response_lens=response_lens,
        epsilon=grpo_cfg.epsilon,
        beta=grpo_cfg.beta,
        full_log_p_new=full_log_p_new,
        full_log_p_ref=full_log_p_ref,
    )

    # ── Backward ──────────────────────────────────────────────────────────
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in policy.parameters() if p.requires_grad], max_norm=1.0
    )
    optimizer.step()

    metrics["grad_norm"] = grad_norm.item()
    return metrics
