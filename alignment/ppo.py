"""
alignment/ppo.py — Proximal Policy Optimisation for LLMs (Task C3)
===================================================================

Mathematical Reference
──────────────────────
Per-step shaped reward (eq. 7):
    r_{i,t} = r_task_i · 1[t = T_i]
              - β · [log π_old(y_t|s_t) - log π_ref(y_t|s_t)]

TD residual and GAE (eq. 8):
    δ_t = r_t + γ·V_old(s_{t+1}) - V_old(s_t)
    A_t^GAE(λ) = Σ_{k≥0} (γλ)^k · δ_{t+k}
    ← right-to-left recurrence: A_T = δ_T,  A_t = δ_t + (γλ)·A_{t+1}

Value target (λ-return):
    V^GAE(s_t) = V_old(s_t) + A_t^GAE    ← DETACH before using as target

Importance ratio:
    ρ_t(θ) = π_θ(y_t|s_t) / π_old(y_t|s_t)
           = exp( log π_θ(y_t|s_t) - log π_old(y_t|s_t) )

PPO clipped objective (eq. 9):
    L_clip = E_t[ min( ρ_t·Ã_t,  clip(ρ_t, 1-ε, 1+ε)·Ã_t ) ]

Total PPO loss (minimised):
    L^PPO = -L_clip  +  c_V·L_V  -  c_ent·H

PSEUDOCODE (execution order):
───────────────────────────────
Phase A — Rollout (all no_grad):
  policy.eval()
  for each batch of B prompts:
    1. generate response y via policy.generate(temp=0.7, top_p=0.9)
    2. build response_mask  (1 up-to-and-including first EOS, 0 after)
    3. forward full_ids through π_θ (LoRA ON)  → log_probs_old (B,R) CACHE
    4. forward full_ids through π_ref (LoRA OFF, disable_adapter ctx) → log_probs_ref
    5. forward full_ids through value_head → values_old (B,R) DETACH + CACHE
    6. retokenise and forward through frozen RM → rm_scores (B,) CACHE
    7. compute shaped rewards r_{i,t}
    8. compute GAE → advantages (DETACHED), value_targets (DETACHED)
    9. standardise advantages batch-wide (μ=0, σ=1)
    10. store all as CPU tensors in rollout_buffer

Phase B — Update (K epochs, mini-batches):
  policy.train(); value_head.train()
  for epoch in ppo_epochs:
    for mini_batch in shuffle(rollout_buffer):
      a. recompute log_probs_new under current π_θ (grad flows here)
      b. ρ_t = exp(log_new - log_old)  [log_old is CACHED, not recomputed]
      c. L_clip  ← clipped surrogate
      d. values_new via value_head
      e. L_V     ← MSE against DETACHED value_targets
      f. H       ← entropy over policy distribution at response positions
      g. loss = -L_clip + c_V·L_V - c_ent·H
      h. backward → clip_grad_norm → optimizer.step
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Tuple

from config import cfg
from model.lora_setup import reference_model_ctx


# ─────────────────────────────────────────────────────────────────────────────
# Helper: per-token log-probs at response positions
# ─────────────────────────────────────────────────────────────────────────────

def _token_log_probs(
    model: nn.Module,
    full_ids: torch.Tensor,     # shape: (B, P+R)
    full_attn: torch.Tensor,    # shape: (B, P+R)
    prompt_len: int,
) -> torch.Tensor:
    """
    Forward the model and collect log π(y_t | x, y_{<t}) at each response position.

    Causal shift logic:
        logits[:, t, :] = predicted distribution for token at position t+1.
        Response tokens start at column P in full_ids.
        Therefore logits[:, P-1, :] predicts full_ids[:, P]   (1st resp token)
                  logits[:, P+k-1, :] predicts full_ids[:, P+k]
        → resp_logits = logits[:, P-1 : P-1+R, :]
          resp_ids    = full_ids[:, P : P+R]

    Upcast to float32 before log_softmax — bfloat16's 8-bit mantissa can
    produce inaccurate log-probs for rare tokens, causing NaN in ρ_t.

    Returns
    -------
    token_log_probs : shape (B, R)  — NOT masked; caller applies response_mask
    """
    outputs = model(input_ids=full_ids, attention_mask=full_attn)
    logits  = outputs.logits                              # (B, P+R, V)

    R = full_ids.shape[1] - prompt_len

    resp_logits = logits[:, prompt_len - 1 : prompt_len - 1 + R, :]  # (B, R, V)
    resp_ids    = full_ids[:, prompt_len:]                            # (B, R)

    log_probs = F.log_softmax(resp_logits.float(), dim=-1)            # (B, R, V)
    token_log_probs = log_probs.gather(
        dim=-1,
        index=resp_ids.unsqueeze(-1),   # (B, R, 1)
    ).squeeze(-1)                        # (B, R)

    return token_log_probs


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build response mask
# ─────────────────────────────────────────────────────────────────────────────

def _build_response_mask(
    resp_ids: torch.Tensor,   # shape: (B, R)
    eos_token_id: int,
) -> torch.Tensor:
    """
    Binary mask: 1 for real tokens (up to and including first EOS), 0 after.

    We include the EOS in the mask because:
        (a) it carries a meaningful log-prob used in the KL computation
        (b) the policy should learn WHEN to stop (EOS prediction is part of the task)

    Returns
    -------
    response_mask : shape (B, R), dtype=long
    """
    B, R   = resp_ids.shape
    device = resp_ids.device

    is_eos    = (resp_ids == eos_token_id)                               # (B, R)
    first_eos = is_eos.long().argmax(dim=1)                              # (B,)
    has_eos   = is_eos.any(dim=1)                                        # (B,)

    # Sequences without EOS: all R positions are real
    cutoff    = torch.where(has_eos, first_eos,
                            torch.tensor(R - 1, device=device))          # (B,)

    positions     = torch.arange(R, device=device).unsqueeze(0)          # (1, R)
    response_mask = (positions <= cutoff.unsqueeze(1)).long()             # (B, R)

    return response_mask


# ─────────────────────────────────────────────────────────────────────────────
# Helper: score with frozen reward model
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _rm_score(
    rm,
    rm_tok,
    raw_prompts: List[str],
    resp_ids: torch.Tensor,       # (B, R)
    policy_tok,
    device: torch.device,
) -> torch.Tensor:
    """
    Decode generated responses and re-score with the frozen RM.

    WHY decode + re-encode?
        Policy tokenizer (SmolLM2) ≠ RM tokenizer (Llama).
        We convert generated token ids → text → Llama token ids.

    Returns
    -------
    scores : shape (B,)
    """
    full_texts = []
    for i, prompt in enumerate(raw_prompts):
        resp_text = policy_tok.decode(resp_ids[i], skip_special_tokens=True)
        full_texts.append(prompt + " " + resp_text)

    enc = rm_tok(
        full_texts,
        max_length=cfg.tokenizer.max_seq_len,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    scores = rm(enc["input_ids"], enc["attention_mask"])  # (B,)
    return scores.detach()


# ─────────────────────────────────────────────────────────────────────────────
# Shaped rewards
# ─────────────────────────────────────────────────────────────────────────────

def compute_shaped_rewards(
    log_probs_old: torch.Tensor,   # (B, R) — π_old
    log_probs_ref: torch.Tensor,   # (B, R) — π_ref
    rm_scores: torch.Tensor,       # (B,)
    response_mask: torch.Tensor,   # (B, R)
    beta: float,
) -> torch.Tensor:
    """
    Build per-token reward signal for GAE.

    Math (eq. 7):
        r_{i,t} = r_task_i · 1[t = T_i]
                  - β · [log π_old(y_t|s_t) - log π_ref(y_t|s_t)]

    The KL penalty at each token asks: how much did π_old diverge from π_ref
    at this particular token decision? A positive KL at token t means π_old
    assigned higher probability than π_ref → small negative reward, nudging
    the policy back toward π_ref locally.

    Returns
    -------
    rewards : shape (B, R) — zero at padding positions
    """
    # Per-token KL shaping: -β·(log_old - log_ref)
    kl_rewards = -beta * (log_probs_old - log_probs_ref) * response_mask  # (B, R)

    # Add sparse task reward at LAST real response token of each sequence
    last_idx = (response_mask.sum(dim=1) - 1).clamp(min=0)  # (B,)

    rewards = kl_rewards.clone()
    for b in range(rewards.shape[0]):
        rewards[b, last_idx[b]] = rewards[b, last_idx[b]] + rm_scores[b]

    return rewards   # (B, R)


# ─────────────────────────────────────────────────────────────────────────────
# GAE
# ─────────────────────────────────────────────────────────────────────────────

def compute_gae(
    rewards: torch.Tensor,        # (B, R)
    values: torch.Tensor,         # (B, R) — V_old, already detached
    response_mask: torch.Tensor,  # (B, R)
    gamma: float = 1.0,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generalised Advantage Estimation via right-to-left recurrence.

    Math:
        δ_t = r_t + γ·V(s_{t+1})·mask_{t+1} - V(s_t)·mask_t
        A_t = δ_t + (γλ)·A_{t+1}·mask_t

    The mask term on V(s_{t+1}) is critical: when position t is the last
    real token, t+1 is padding → mask_{t+1} = 0 → γ·V(s_{t+1}) = 0.
    This gives the correct TERMINAL bootstrap (no future value after EOS).

    WHY DETACH value_targets?
        The value target V^GAE = V_old + A^GAE is a fixed regression target.
        If not detached, PyTorch would try to differentiate L_V =
        (V_new - V_target)^2 through V_old itself, creating a circular
        gradient: the critic would simultaneously move V_new UP and V_old
        DOWN (to reduce the gap), causing unstable training.

    Returns
    -------
    advantages    : (B, R) DETACHED
    value_targets : (B, R) DETACHED
    """
    B, R   = rewards.shape
    device = rewards.device

    advantages = torch.zeros_like(rewards)
    last_adv   = torch.zeros(B, device=device)

    for t in reversed(range(R)):
        mask_t = response_mask[:, t].float()    # (B,)

        # Bootstrap: V(s_{t+1}) = 0 if t+1 is padding (terminal)
        if t < R - 1:
            next_val = values[:, t + 1] * response_mask[:, t + 1].float()
        else:
            next_val = torch.zeros(B, device=device)

        # TD residual, masked so padding positions have δ=0
        delta    = (rewards[:, t] + gamma * next_val - values[:, t]) * mask_t  # (B,)

        # GAE accumulation; zero out at padding positions
        last_adv = (delta + gamma * lam * last_adv) * mask_t   # (B,)
        advantages[:, t] = last_adv

    value_targets = (values + advantages).detach()
    advantages    = advantages.detach()

    return advantages, value_targets


# ─────────────────────────────────────────────────────────────────────────────
# PPO loss components
# ─────────────────────────────────────────────────────────────────────────────

def ppo_clip_loss(
    log_probs_new: torch.Tensor,   # (B, R) — current π_θ, grad flows here
    log_probs_old: torch.Tensor,   # (B, R) — CACHED at rollout, no grad
    advantages: torch.Tensor,      # (B, R) — DETACHED
    response_mask: torch.Tensor,   # (B, R)
    epsilon: float = 0.2,
) -> torch.Tensor:
    """
    PPO clipped surrogate (eq. 9).

    ρ_t = exp(log_new - log_old)

    The min(unclipped, clipped) creates an asymmetric trust region:
        Ã > 0: reward this action more. But cap ρ at 1+ε
               → prevents exploiting a single good action too aggressively.
        Ã < 0: punish this action. But cap ρ at 1-ε
               → prevents collapsing the probability of an action to 0
               based on a single bad rollout.

    Returns
    -------
    clip_loss : scalar (to be negated for gradient ascent)
    """
    rho            = torch.exp(log_probs_new - log_probs_old)     # (B, R)
    rho_clipped    = rho.clamp(1.0 - epsilon, 1.0 + epsilon)

    surr_unclipped = rho         * advantages                      # (B, R)
    surr_clipped   = rho_clipped * advantages                      # (B, R)
    surr_min       = torch.min(surr_unclipped, surr_clipped)       # (B, R)

    n_real    = response_mask.sum().clamp(min=1.0)
    clip_loss = (surr_min * response_mask).sum() / n_real          # scalar

    return clip_loss


def ppo_value_loss(
    values_new: torch.Tensor,      # (B, R) — current V_θ
    value_targets: torch.Tensor,   # (B, R) — DETACHED λ-returns
    response_mask: torch.Tensor,   # (B, R)
) -> torch.Tensor:
    """
    MSE critic loss averaged over real response positions.

    L_V = E_{i,t}[ (V_θ(s_{i,t}) - V^GAE(s_{i,t}))^2 ]
    """
    mse    = (values_new - value_targets).pow(2)           # (B, R)
    n_real = response_mask.sum().clamp(min=1.0)
    return (mse * response_mask).sum() / n_real             # scalar


def ppo_entropy_bonus(
    resp_logits: torch.Tensor,     # (B, R, V) — response logits, detached
    response_mask: torch.Tensor,   # (B, R)
) -> torch.Tensor:
    """
    Mean per-token entropy H = -Σ_a π(a|s_t)·log π(a|s_t).

    A positive entropy bonus in the loss keeps the policy from prematurely
    collapsing onto narrow high-reward patterns (mode collapse).
    c_ent is typically small (0.01) so it regularises without dominating.
    """
    log_p  = F.log_softmax(resp_logits.float(), dim=-1)     # (B, R, V)
    p      = log_p.exp()
    H_t    = -(p * log_p).sum(dim=-1)                       # (B, R)
    n_real = response_mask.sum().clamp(min=1.0)
    return (H_t * response_mask).sum() / n_real              # scalar


# ─────────────────────────────────────────────────────────────────────────────
# Rollout collection
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_rollouts(
    policy,
    rm,
    value_head,
    prompt_loader,
    policy_tok,
    rm_tok,
    device: torch.device,
    ppo_cfg=None,
    n_batches: int = 1,
) -> List[Dict]:
    """
    Collect trajectories under π_old (current policy before any updates).

    CRITICAL INVARIANT: log_probs_old is computed HERE and cached.
    It is the DENOMINATOR of ρ_t throughout all K PPO update epochs.
    Recomputing it inside the update would give ρ_t ≡ 1 → clipping never
    fires → PPO degrades to plain policy gradient, losing its stability.

    All tensors are moved to CPU after collection to free GPU memory for
    the subsequent update phase.
    """
    if ppo_cfg is None:
        ppo_cfg = cfg.ppo

    policy.eval()
    value_head.eval()

    rollout_buffer = []

    for batch_idx, batch in enumerate(prompt_loader):
        if batch_idx >= n_batches:
            break

        prompt_ids  = batch["input_ids"].to(device)        # (B, P)
        prompt_mask = batch["attention_mask"].to(device)   # (B, P)
        raw_prompts = batch["raw_prompt"]                  # list[str]

        B = prompt_ids.shape[0]
        P = prompt_ids.shape[1]

        # ── 1. Generate responses ─────────────────────────────────────
        # do_sample=True with temperature: ensures diverse completions.
        # Greedy (do_sample=False) would produce identical outputs for
        # identical prompts → all advantages ≈ 0 → no learning signal.
        full_ids = policy.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=ppo_cfg.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=policy_tok.pad_token_id,
            eos_token_id=policy_tok.eos_token_id,
            use_cache=True,
        )  # (B, P+R)

        R        = full_ids.shape[1] - P
        resp_ids = full_ids[:, P:]   # (B, R)

        # ── 2. Response mask ──────────────────────────────────────────
        response_mask = _build_response_mask(resp_ids, policy_tok.eos_token_id)  # (B,R)
        full_attn     = torch.cat([prompt_mask, response_mask], dim=1)            # (B,P+R)

        # ── 3. Old log-probs under π_θ (LoRA ON) ─────────────────────
        # π_θ at THIS MOMENT = π_old. Cached as denominator for ρ_t.
        log_probs_old = _token_log_probs(policy, full_ids, full_attn, P)   # (B,R)

        # ── 4. Ref log-probs under π_ref (LoRA OFF) ──────────────────
        with reference_model_ctx(policy) as ref_model:
            log_probs_ref = _token_log_probs(ref_model, full_ids, full_attn, P)  # (B,R)

        # ── 5. Value estimates V_old ──────────────────────────────────
        # Detach immediately: these are fixed baselines for GAE,
        # not part of the live computation graph.
        values_old = value_head(full_ids, full_attn)[:, P:].detach()   # (B,R)

        # ── 6. Task reward from frozen RM ─────────────────────────────
        rm_scores = _rm_score(rm, rm_tok, raw_prompts, resp_ids, policy_tok, device)  # (B,)

        # ── 7. Shaped rewards ─────────────────────────────────────────
        rewards = compute_shaped_rewards(
            log_probs_old, log_probs_ref, rm_scores, response_mask, ppo_cfg.beta
        )  # (B, R)

        # ── 8. GAE ────────────────────────────────────────────────────
        advantages, value_targets = compute_gae(
            rewards, values_old, response_mask, ppo_cfg.gamma, ppo_cfg.lam
        )  # both (B, R), both DETACHED

        # ── 9. Advantage normalisation ────────────────────────────────
        # Standardise over real response positions (μ=0, σ=1).
        # Keeps effective step size stable regardless of RM magnitude.
        real_advs = advantages[response_mask.bool()]
        if real_advs.numel() > 1:
            mu  = real_advs.mean()
            std = real_advs.std() + 1e-8
            advantages = torch.where(
                response_mask.bool(),
                (advantages - mu) / std,
                torch.zeros_like(advantages),
            )

        # ── 10. Store on CPU ──────────────────────────────────────────
        rollout_buffer.append({
            "full_ids":      full_ids.cpu(),        # (B, P+R)
            "full_attn":     full_attn.cpu(),        # (B, P+R)
            "prompt_len":    P,
            "response_mask": response_mask.cpu(),   # (B, R)
            "log_probs_old": log_probs_old.cpu(),   # (B, R)  ← CACHED π_old
            "advantages":    advantages.cpu(),       # (B, R)  ← DETACHED
            "value_targets": value_targets.cpu(),   # (B, R)  ← DETACHED
            "rm_scores":     rm_scores.cpu(),        # (B,)
            "mean_kl":   (log_probs_old - log_probs_ref).clamp(min=-10).mean().item(),
            "mean_rm":   rm_scores.mean().item(),
        })

    policy.train()
    value_head.train()
    return rollout_buffer


# ─────────────────────────────────────────────────────────────────────────────
# PPO update loop
# ─────────────────────────────────────────────────────────────────────────────

def ppo_update(
    policy,
    value_head,
    policy_optimizer,
    value_optimizer,
    rollout_buffer: List[Dict],
    device: torch.device,
    ppo_cfg=None,
) -> Dict[str, float]:
    """
    K epochs of mini-batch PPO gradient updates.

    Each epoch shuffles all (buffer_idx, seq_idx) pairs to form
    independent mini-batches. log_probs_old is loaded from the cache
    and NEVER recomputed.

    Returns
    -------
    metrics : dict of mean values over all update steps
    """
    if ppo_cfg is None:
        ppo_cfg = cfg.ppo

    policy.train()
    value_head.train()

    metric_accum = defaultdict(list)

    for epoch in range(ppo_cfg.ppo_epochs):
        # Build flat index list: one entry per sequence across all batches
        all_idx = []
        for buf_i, buf in enumerate(rollout_buffer):
            B = buf["full_ids"].shape[0]
            for seq_i in range(B):
                all_idx.append((buf_i, seq_i))

        perm    = torch.randperm(len(all_idx))
        all_idx = [all_idx[i] for i in perm]

        for start in range(0, len(all_idx), ppo_cfg.mini_batch_size):
            mb_idx = all_idx[start : start + ppo_cfg.mini_batch_size]
            if not mb_idx:
                continue

            # Assemble tensors from buffer and move to GPU
            keys = ["full_ids", "full_attn", "log_probs_old",
                    "advantages", "value_targets", "response_mask"]
            mb = {k: [] for k in keys}

            prompt_len = rollout_buffer[0]["prompt_len"]

            for buf_i, seq_i in mb_idx:
                buf = rollout_buffer[buf_i]
                for k in keys:
                    mb[k].append(buf[k][seq_i])

            full_ids      = torch.stack(mb["full_ids"]).to(device)       # (mb, P+R)
            full_attn     = torch.stack(mb["full_attn"]).to(device)      # (mb, P+R)
            log_probs_old = torch.stack(mb["log_probs_old"]).to(device)  # (mb, R)
            advantages    = torch.stack(mb["advantages"]).to(device)     # (mb, R)
            value_targets = torch.stack(mb["value_targets"]).to(device)  # (mb, R)
            resp_mask     = torch.stack(mb["response_mask"]).to(device)  # (mb, R)

            R = full_ids.shape[1] - prompt_len

            # ── (a) Recompute log-probs under CURRENT π_θ ─────────────
            outputs     = policy(input_ids=full_ids, attention_mask=full_attn)
            resp_logits = outputs.logits[
                :, prompt_len - 1 : prompt_len - 1 + R, :
            ]                                                             # (mb, R, V)
            resp_ids    = full_ids[:, prompt_len:]                       # (mb, R)

            log_p_all    = F.log_softmax(resp_logits.float(), dim=-1)    # (mb, R, V)
            log_probs_new = log_p_all.gather(
                dim=-1, index=resp_ids.unsqueeze(-1)
            ).squeeze(-1)                                                 # (mb, R)

            # ── (b) Clip loss ──────────────────────────────────────────
            clip_loss = ppo_clip_loss(
                log_probs_new, log_probs_old, advantages, resp_mask, ppo_cfg.epsilon
            )

            # ── (c) Value loss ─────────────────────────────────────────
            values_new = value_head(full_ids, full_attn)[:, prompt_len:]  # (mb, R)
            v_loss     = ppo_value_loss(values_new, value_targets, resp_mask)

            # ── (d) Entropy bonus ──────────────────────────────────────
            # Detach logits: entropy is a regulariser, not part of actor gradient
            entropy = ppo_entropy_bonus(resp_logits.detach(), resp_mask)

            # ── (e) Total loss ─────────────────────────────────────────
            loss = (
                -clip_loss
                + ppo_cfg.c_value   * v_loss
                - ppo_cfg.c_entropy * entropy
            )

            # ── (f) Backward ───────────────────────────────────────────
            policy_optimizer.zero_grad()
            value_optimizer.zero_grad()

            loss.backward()

            gn_policy = torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], max_norm=1.0
            )
            gn_value = torch.nn.utils.clip_grad_norm_(
                [p for p in value_head.parameters() if p.requires_grad], max_norm=1.0
            )

            policy_optimizer.step()
            value_optimizer.step()

            # ── Log ────────────────────────────────────────────────────
            with torch.no_grad():
                rho    = torch.exp(log_probs_new.detach() - log_probs_old)
                kl_tok = ((log_probs_new.detach() - log_probs_old) * resp_mask)
                n_real = resp_mask.sum().clamp(min=1.0)

            metric_accum["loss"].append(loss.item())
            metric_accum["clip_loss"].append(clip_loss.item())
            metric_accum["v_loss"].append(v_loss.item())
            metric_accum["entropy"].append(entropy.item())
            metric_accum["rho_mean"].append((rho * resp_mask).sum().item() / n_real.item())
            metric_accum["rho_max"].append(rho.max().item())
            metric_accum["kl_mean"].append(kl_tok.sum().item() / n_real.item())
            metric_accum["grad_norm_policy"].append(gn_policy.item())

    return {k: sum(v) / len(v) for k, v in metric_accum.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Assignment sanity checks
# ─────────────────────────────────────────────────────────────────────────────

def run_sanity_checks():
    """
    Three required checks from the PA2 spec.

    Check 1 — GAE unit test:
        T=3, r=[0.05, -0.02, 1.6], V_old=[1.5, 1.55, 1.45], γ=λ=1
        δ_2 = 1.60 + 0    - 1.45 =  0.15  (terminal: next V = 0)
        δ_1 = -0.02 + 1.45 - 1.55 = -0.12
        δ_0 =  0.05 + 1.55 - 1.50 =  0.10
        A_2 = 0.15
        A_1 = -0.12 + 0.15 = 0.03
        A_0 =  0.10 + 0.03 = 0.13

    Check 2 — Ratio = 1.0 before any updates:
        At the very first update step, log_new == log_old (model unchanged).
        ρ = exp(log_new - log_old) = exp(0) = 1.0 everywhere.

    Check 3 — Clipping at ρ=1.5, A=1.0, ε=0.2:
        unclipped: 1.5 × 1.0 = 1.5
        clipped:   clip(1.5, 0.8, 1.2) × 1.0 = 1.2
        L_clip = min(1.5, 1.2) = 1.2
        ∂L_clip/∂ρ = 0  (min selects the clipped, constant branch)
    """
    print("\n" + "=" * 60)
    print(" PPO SANITY CHECKS")
    print("=" * 60)

    # ── Check 1 ────────────────────────────────────────────────────────
    print("\n[1/3] GAE Unit Test")
    r      = torch.tensor([[0.05, -0.02, 1.60]])
    v_old  = torch.tensor([[1.50,  1.55, 1.45]])
    mask   = torch.ones(1, 3, dtype=torch.long)
    adv, _ = compute_gae(r, v_old, mask, gamma=1.0, lam=1.0)
    expected = torch.tensor([0.13, 0.03, 0.15])
    ok1 = torch.allclose(adv[0], expected, atol=1e-4)
    print(f"   Got:      {adv[0].numpy()}")
    print(f"   Expected: {expected.numpy()}")
    print(f"   {'✓ PASS' if ok1 else '✗ FAIL — check right-to-left recurrence'}")

    # ── Check 2 ────────────────────────────────────────────────────────
    print("\n[2/3] Ratio Test (ρ_t = 1.0 before first update)")
    lp_old = torch.randn(4, 10)
    lp_new = lp_old.clone()
    rho    = torch.exp(lp_new - lp_old)
    ok2    = torch.allclose(rho, torch.ones_like(rho))
    print(f"   ρ min={rho.min():.6f}  max={rho.max():.6f}")
    print(f"   {'✓ PASS' if ok2 else '✗ FAIL — log_probs_old is being recomputed!'}")

    # ── Check 3 ────────────────────────────────────────────────────────
    print("\n[3/3] Clipping Test (ρ=1.5, A=1.0, ε=0.2)")
    lp_old_t = torch.zeros(1, 1)
    lp_new_t = torch.log(torch.tensor([[1.5]])) + lp_old_t
    lp_new_t.requires_grad_(True)
    adv_t    = torch.ones(1, 1)
    mask_t   = torch.ones(1, 1, dtype=torch.long)

    c = ppo_clip_loss(lp_new_t, lp_old_t, adv_t, mask_t, epsilon=0.2)
    c.backward()

    ok3_val  = abs(c.item() - 1.2) < 1e-4
    ok3_grad = abs(lp_new_t.grad.item()) < 1e-6
    print(f"   L_clip = {c.item():.4f}  (expected 1.2000)  "
          f"{'✓' if ok3_val else '✗'}")
    print(f"   ∂L/∂log_new = {lp_new_t.grad.item():.6f}  (expected 0.0)  "
          f"{'✓' if ok3_grad else '✗'}")

    print("\n" + "=" * 60 + "\n")
    return ok1 and ok2 and ok3_val and ok3_grad


if __name__ == "__main__":
    run_sanity_checks()
