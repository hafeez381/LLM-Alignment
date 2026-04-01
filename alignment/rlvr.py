"""
alignment/rlvr.py — RLVR: RL with Verifiable Rewards (Task C6)
===============================================================

RLVR = GRPO + verifiable reward r_v ∈ {0, 1} (no learned RM).

The ONLY difference from standard GRPO:
    GRPO: r_{b,k} = r_ψ(x_b ⊕ y_{b,k})         (learned RM)
    RLVR: r_{b,k} = 1[extract_answer(y_{b,k}) == g(x_b)]  (rule check)

Why is r_v ∈ {0, 1} special?
    Group advantages: A_{b,k} = r_{b,k} - μ_b
    With r_v ∈ {0, 1}, μ_b = fraction of K completions that are correct.
    A_{b,k} ∈ {-μ_b, 1-μ_b}:
        All wrong (μ_b=0): A_{b,k}=0 for all k → DEGENERATE, zero gradient
        All right (μ_b=1): A_{b,k}=0 for all k → DEGENERATE, zero gradient
        Mixed: A_{b,k}∈{-μ_b, 1-μ_b} → non-zero gradient, learning signal!

    This is the PAD Problem (Problem 4.1(c) in PA2): easy or hard prompts
    give zero signal. The training set difficulty distribution matters.

PSEUDOCODE — RLVR training loop:
──────────────────────────────────
1. Load policy from SFT checkpoint (HH-RLHF SFT — same π_ref as other methods)
   WHY: Do NOT use PPO/GRPO/DPO checkpoint. Those are aligned for HH preferences.
   We want to start from a general-purpose SFT model and teach math.

2. Load GSM8K train/test splits

3. For step in range(rlvr_cfg.total_steps):
   a. Sample B prompts from GSM8K train
   b. group_rollout(policy, rm=None, reward_fn=verifiable_reward_fn, ...)
      → K completions per prompt, scored by r_v ∈ {0,1}
      → advantages computed from group mean
      → log all degenerate batches
   c. grpo_update(policy, optimizer, rollout_buffer)
      → same GRPO loss as C5, just different reward source
   d. Log: mean_reward, frac_degenerate, mean_resp_len, kl_from_ref

4. Every eval_every steps:
   a. pass@1 on held-out GSM8K test (greedy decode)
   b. format_compliance: fraction of responses containing any number
   c. credit_assignment_analysis: fraction of tokens with nonzero gradient signal

ANALYSIS METRICS TO LOG:
    pass@1             ← primary quality metric
    frac_degenerate    ← tells us about difficulty match (>30% = problem)
    mean_resp_len      ← detects verbosity drift (length increasing over time?)
    format_compliance  ← fraction responses with parseable answer
    kl_from_ref        ← how far policy has moved from SFT init
    pct_nonzero_grad   ← r_v is sparse: only terminal reward → mostly zero grad
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

from config import cfg
from alignment.grpo import group_rollout, grpo_update
from data.gsm8k import (
    extract_answer, verifiable_reward, make_verifiable_reward_fn
)
from model.lora_setup import reference_model_ctx


# ─────────────────────────────────────────────────────────────────────────────
# Verifiable reward batch function (wraps scalar verifiable_reward for GRPO)
# ─────────────────────────────────────────────────────────────────────────────

def make_rlvr_reward_fn(gold_answer_map: Dict[str, float], policy_tok):
    """
    Returns a reward_fn(tiled_raw_prompts, resp_ids, policy_tok) → (B*K,) tensor.

    Plugs directly into group_rollout(reward_fn=...) replacing the RM call.

    Parameters
    ----------
    gold_answer_map : dict mapping formatted_prompt_string → numeric gold answer
                      Built from the GSM8K batch before rollout.
    policy_tok      : policy tokeniser for decoding resp_ids → text
    """
    def reward_fn(
        tiled_prompts: List[str],
        resp_ids: torch.Tensor,    # (B*K, R)
        tok,
    ) -> torch.Tensor:
        rewards = []
        for i, prompt in enumerate(tiled_prompts):
            generated = tok.decode(resp_ids[i], skip_special_tokens=True)
            gold = gold_answer_map.get(prompt)
            if gold is None:
                rewards.append(0.0)
                continue
            rv = verifiable_reward(generated, str(gold))
            rewards.append(rv)
        return torch.tensor(rewards, dtype=torch.float32,
                            device=resp_ids.device)

    return reward_fn


# ─────────────────────────────────────────────────────────────────────────────
# RLVR evaluation: pass@1 and format compliance
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_rlvr(
    policy,
    test_examples: List[Dict],
    policy_tok,
    device: torch.device,
    max_new_tokens: int = 256,
    n_eval: int = 200,
) -> Dict[str, float]:
    """
    Evaluate RLVR policy on GSM8K test set.

    Metrics:
        pass@1          : fraction of greedy responses with correct answer
        format_compliance: fraction of responses that contain ANY number
                           (proxy for whether the model learned to produce answers)
        mean_resp_len   : average response length (monitor verbosity drift)
        frac_extractable: fraction of responses where extract_answer succeeds

    Greedy decoding (do_sample=False) gives deterministic, comparable results.
    Use the SAME examples every eval to make pass@1 curves comparable.
    """
    policy.eval()
    examples = test_examples[:n_eval]

    n_correct     = 0
    n_extractable = 0
    n_has_number  = 0
    total_len     = 0.0
    n_total       = 0

    import re
    has_number_pattern = re.compile(r"-?\d+(?:\.\d+)?")

    for ex in examples:
        enc = policy_tok(
            ex["prompt"],
            max_length=256,           # leave room for generation
            truncation=True,
            return_tensors="pt",
        ).to(device)

        out = policy.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy — reproducible eval
            pad_token_id=policy_tok.pad_token_id,
            eos_token_id=policy_tok.eos_token_id,
        )

        prompt_len    = enc["input_ids"].shape[1]
        resp_ids      = out[0][prompt_len:]
        generated     = policy_tok.decode(resp_ids, skip_special_tokens=True)

        pred   = extract_answer(generated)
        gold   = ex["gold_answer"]

        # pass@1
        if pred is not None:
            n_extractable += 1
            try:
                if abs(float(pred) - float(gold)) < 1e-6:
                    n_correct += 1
            except (TypeError, ValueError):
                pass

        # format compliance: does the response contain any number at all?
        if has_number_pattern.search(generated):
            n_has_number += 1

        total_len += len(resp_ids)
        n_total   += 1

    policy.train()

    return {
        "pass_at_1":          n_correct     / max(n_total, 1),
        "frac_extractable":   n_extractable / max(n_total, 1),
        "format_compliance":  n_has_number  / max(n_total, 1),
        "mean_resp_len":      total_len     / max(n_total, 1),
        "n_eval":             n_total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Credit assignment analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_credit_assignment_fraction(
    response_mask: torch.Tensor,    # (B*K, R)
    response_lens: torch.Tensor,    # (B*K,) — T_k
    is_degenerate: torch.Tensor,    # (B,) — bool per prompt
    K: int,
) -> float:
    """
    Fraction of response tokens that carry a nonzero gradient signal.

    With r_v ∈ {0,1} and group-relative advantages:
        Degenerate batches (all-0 or all-1 rewards) → ALL tokens have zero grad.
        Non-degenerate batches → ALL tokens in each completion get the same A_{b,k}.

    This is a fundamental difference from PPO:
        PPO: token-level KL penalty gives a nonzero gradient at EVERY response token.
        RLVR: only non-degenerate batches contribute; within those, all tokens count.

    Returns fraction ∈ [0, 1] of response token positions with nonzero gradient.
    Relates to PA2 Problem 4.3(b): "most tokens get gradient zero before
    the final answer is verified."
    """
    BK, R = response_mask.shape
    B     = BK // K

    # Expand is_degenerate from (B,) to (B*K,)
    is_deg_expanded = is_degenerate.repeat_interleave(K)  # (B*K,)

    # Tokens in NON-degenerate sequences AND real (not padding)
    non_deg_mask    = (~is_deg_expanded).unsqueeze(1).expand(-1, R)  # (B*K, R)
    active_tokens   = (response_mask * non_deg_mask.long()).sum().item()
    total_tokens    = response_mask.sum().item()

    if total_tokens == 0:
        return 0.0
    return active_tokens / total_tokens


# ─────────────────────────────────────────────────────────────────────────────
# Sample response table (for analysis requirement C6.4 item 15)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_sample_table(
    policy,
    test_examples: List[Dict],
    policy_tok,
    device: torch.device,
    n_samples: int = 5,
    max_new_tokens: int = 256,
) -> List[Dict]:
    """
    Generate greedy responses for n_samples test problems.

    Returns a list of dicts with:
        question     : str
        generated    : str — model's solution
        extracted    : int/float/None — extracted answer
        gold         : int — correct answer
        correct      : bool
    """
    policy.eval()
    results = []

    for ex in test_examples[:n_samples]:
        enc = policy_tok(
            ex["prompt"], max_length=256, truncation=True, return_tensors="pt"
        ).to(device)

        out = policy.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=policy_tok.pad_token_id,
            eos_token_id=policy_tok.eos_token_id,
        )

        prompt_len = enc["input_ids"].shape[1]
        generated  = policy_tok.decode(out[0][prompt_len:], skip_special_tokens=True)
        pred       = extract_answer(generated)
        gold       = ex["gold_answer"]

        correct = False
        if pred is not None:
            try:
                correct = abs(float(pred) - float(gold)) < 1e-6
            except (TypeError, ValueError):
                pass

        results.append({
            "question":  ex["question"],
            "generated": generated,
            "extracted": pred,
            "gold":      gold,
            "correct":   correct,
        })

    policy.train()
    return results


def print_sample_table(samples: List[Dict]):
    """Pretty-print the sample response table to stdout."""
    print(f"\n{'━'*70}")
    print(f" RLVR SAMPLE RESPONSE TABLE")
    print(f"{'━'*70}")
    for i, s in enumerate(samples):
        status = "✓ CORRECT" if s["correct"] else "✗ WRONG"
        print(f"\n[{i+1}] {status}")
        print(f"Question: {s['question'][:120]}...")
        print(f"Generated (last 300 chars): ...{s['generated'][-300:]}")
        print(f"Extracted: {s['extracted']}  |  Gold: {s['gold']}")
    print(f"\n{'━'*70}\n")
