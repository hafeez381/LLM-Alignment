"""
eval.py — Evaluation Suite (Task C8)
======================================

Metrics implemented:
    1. compute_win_rate()          → RM win-rate vs. SFT baseline
    2. compute_kl()                → Monte-Carlo KL(π_θ || π_ref)
    3. build_sample_table()        → 5-prompt side-by-side response table
    4. ResourceTracker             → VRAM + timing per method
    5. run_full_eval()             → one-shot driver that calls all of the above
                                     for every trained checkpoint and prints a
                                     markdown-formatted report to stdout + file

PSEUDOCODE — win-rate:
    For each of N=200 held-out prompts:
        y_aligned = greedy_decode(aligned_model, prompt)
        y_sft     = greedy_decode(sft_model,     prompt)
        r_aligned = rm(prompt ⊕ y_aligned)
        r_sft     = rm(prompt ⊕ y_sft)
    win_rate = mean(r_aligned > r_sft)

PSEUDOCODE — KL (Monte-Carlo approximation):
    E_x[ KL(π_θ(·|x) || π_ref(·|x)) ]
    ≈ (1/N) Σ_x  (1/T) Σ_t log π_θ(y_t|s_t) - log π_ref(y_t|s_t)
    where y ~ π_θ(·|x) via greedy decode (T = length of greedy response).
    WHY MC approximation?  The true KL sums over the full vocabulary V at each
    step, costing O(N·T·V) — prohibitive.  The single-sample estimate is
    unbiased with variance O(1/N) when averaged over N prompts.

IMPORTANT: all models use greedy decoding (temperature=0, do_sample=False)
during evaluation to ensure reproducibility.  Sampling would introduce noise
that makes run-to-run comparisons unreliable.
"""

import os
import time
import json
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from model.lora_setup import reference_model_ctx


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _greedy_decode(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    device=None,
) -> str:
    """Greedy (temperature=0) response for a single prompt string."""
    if device is None:
        device = next(model.parameters()).device
    enc = tokenizer(
        prompt,
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,                     # greedy — reproducible
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    resp_ids = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(resp_ids, skip_special_tokens=True).strip()


@torch.no_grad()
def _rm_score_text(
    rm,
    rm_tok,
    prompt: str,
    response: str,
    device=None,
) -> float:
    """Score a single (prompt, response) pair with the frozen RM."""
    if device is None:
        device = next(rm.parameters()).device
    full_text = prompt + " " + response
    enc = rm_tok(
        full_text,
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    score = rm(enc["input_ids"], enc["attention_mask"])
    return score.item()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Win-rate
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_win_rate(
    aligned_model,
    sft_model,
    rm,
    rm_tok,
    policy_tok,
    test_examples: List[Dict],
    n_eval: int = 200,
    max_new_tokens: int = 128,
    device=None,
) -> Dict:
    """
    Fraction of held-out prompts where the aligned model's greedy response
    scores higher under the frozen RM than the SFT model's response.

        win_rate = P(r_aligned > r_sft)

    This is the PRIMARY quality metric for C8.

    We also track mean scores and mean response lengths so we can detect
    verbosity bias (a model that wins simply by generating longer responses).

    Parameters
    ----------
    aligned_model : PPO/DPO/GRPO/RLVR trained policy (with LoRA)
    sft_model     : SFT baseline policy (loaded from sft_merged, frozen)
    rm            : frozen reward model
    """
    if device is None:
        device = next(aligned_model.parameters()).device

    aligned_model.eval()
    sft_model.eval()
    examples = test_examples[:n_eval]
    batch_size = 16   # process 16 prompts at once — saturates A100 well

    def batch_decode(model, exs, label="decode"):
        """Greedy-decode a list of examples in mini-batches."""
        all_responses = []
        for i in tqdm(range(0, len(exs), batch_size),
                      desc=f"  {label}", unit="batch", leave=False, dynamic_ncols=True):
            batch   = exs[i : i + batch_size]
            prompts = [e["prompt"] for e in batch]
            enc = policy_tok(
                prompts, max_length=512, truncation=True,
                padding=True, return_tensors="pt"
            ).to(device)
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=policy_tok.pad_token_id,
                eos_token_id=policy_tok.eos_token_id,
            )
            P = enc["input_ids"].shape[1]
            for j in range(len(batch)):
                resp = policy_tok.decode(out[j][P:], skip_special_tokens=True).strip()
                all_responses.append(resp)
        return all_responses

    def batch_score(prompts, responses):
        """Score (prompt, response) pairs in mini-batches with the RM."""
        scores = []
        for i in range(0, len(prompts), batch_size):
            texts = [
                prompts[i + j] + " " + responses[i + j]
                for j in range(min(batch_size, len(prompts) - i))
            ]
            enc = rm_tok(
                texts, max_length=512, padding=True,
                truncation=True, return_tensors="pt"
            ).to(device)
            s = rm(enc["input_ids"], enc["attention_mask"])
            scores.extend(s.cpu().float().tolist())
        return scores

    prompts      = [e["prompt"] for e in examples]
    resp_aligned = batch_decode(aligned_model, examples, label="aligned decode")
    resp_sft     = batch_decode(sft_model,     examples, label="sft decode")
    r_aligned    = batch_score(prompts, resp_aligned)
    r_sft        = batch_score(prompts, resp_sft)

    wins = sum(1 for a, s in zip(r_aligned, r_sft) if a > s)
    n    = len(examples)
    return {
        "win_rate":         wins / n,
        "mean_r_aligned":   sum(r_aligned) / n,
        "mean_r_sft":       sum(r_sft) / n,
        "mean_len_aligned": sum(len(r.split()) for r in resp_aligned) / n,
        "mean_len_sft":     sum(len(r.split()) for r in resp_sft) / n,
        "n_eval":           n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. KL divergence (Monte-Carlo)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_kl(
    policy,         # π_θ (peft model with LoRA)
    policy_tok,
    test_examples: List[Dict],
    n_eval: int = 200,
    max_new_tokens: int = 64,  # short generations to keep cost low
    device=None,
) -> float:
    """
    Monte-Carlo estimate of E_x[ KL(π_θ(·|x) || π_ref(·|x)) ].

    Algorithm:
        For each prompt x:
            1. Greedy-decode y ~ π_θ(·|x)  (length T)
            2. Teacher-force y through π_θ  → log π_θ(y_t|s_t) per token
            3. Teacher-force y through π_ref (disable_adapter) → log π_ref(y_t|s_t)
            4. KL_x ≈ (1/T) Σ_t [log π_θ(y_t|s_t) - log π_ref(y_t|s_t)]
        Return mean KL over all prompts.

    WHY greedy y (not sampled)?
        Using the policy's OWN greedy output means we're measuring KL where
        the policy actually generates — the most relevant region of the
        output space. Sampling would average over low-probability responses.

    WHY (1/T) normalisation?
        Per-sequence KL grows linearly with T; normalising gives a
        per-token KL that is comparable across responses of different lengths.
    """
    if device is None:
        device = next(policy.parameters()).device

    policy.eval()
    kl_values = []

    for ex in tqdm(test_examples[:n_eval], desc="KL", unit="prompt", dynamic_ncols=True):
        prompt = ex["prompt"]
        enc = policy_tok(
            prompt, max_length=512, truncation=True, return_tensors="pt"
        ).to(device)

        # Step 1: greedy decode
        out = policy.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=policy_tok.pad_token_id,
            eos_token_id=policy_tok.eos_token_id,
        )
        P    = enc["input_ids"].shape[1]
        full = out  # (1, P+R)
        R    = full.shape[1] - P
        if R == 0:
            continue

        resp_ids = full[:, P:]           # (1, R)
        resp_mask = torch.ones(1, R, device=device, dtype=torch.long)
        full_attn = torch.cat([enc["attention_mask"], resp_mask], dim=1)

        # Step 2: log π_θ(y_t|s_t) — LoRA ON
        outputs_theta = policy(input_ids=full, attention_mask=full_attn)
        logits_theta  = outputs_theta.logits[:, P-1 : P-1+R, :].float()   # (1,R,V)
        lp_theta = F.log_softmax(logits_theta, dim=-1)
        lp_theta_tok = lp_theta.gather(-1, resp_ids.unsqueeze(-1)).squeeze(-1)  # (1,R)

        # Step 3: log π_ref(y_t|s_t) — LoRA OFF
        with reference_model_ctx(policy) as ref:
            outputs_ref = ref(input_ids=full, attention_mask=full_attn)
        logits_ref = outputs_ref.logits[:, P-1 : P-1+R, :].float()
        lp_ref = F.log_softmax(logits_ref, dim=-1)
        lp_ref_tok = lp_ref.gather(-1, resp_ids.unsqueeze(-1)).squeeze(-1)

        # Step 4: per-token KL, averaged over T
        kl_per_token = (lp_theta_tok - lp_ref_tok).mean().item()
        kl_values.append(kl_per_token)

    return sum(kl_values) / max(len(kl_values), 1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Sample response table
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def build_sample_table(
    method_models: Dict[str, object],  # {"SFT": model, "PPO": model, ...}
    rm,
    rm_tok,
    policy_tok,
    test_examples: List[Dict],
    n_prompts: int = 5,
    max_new_tokens: int = 128,
    device=None,
) -> List[Dict]:
    """
    For each of n_prompts held-out prompts, generate greedy responses from
    every method and score them with the RM.

    Returns a list of dicts:
        {
          "prompt": str,
          "responses": {"SFT": str, "PPO": str, ...},
          "scores":    {"SFT": float, "PPO": float, ...},
          "best_method": str,   (method with highest RM score)
          "disagreement": bool, (True if methods disagree on best response)
        }

    Disagreement is defined as the top-scoring method NOT being SFT
    (since SFT is the baseline), and different RL methods ranking differently.
    """
    if device is None:
        for m in method_models.values():
            if hasattr(m, 'parameters'):
                device = next(m.parameters()).device
                break

    results = []
    for ex in test_examples[:n_prompts]:
        prompt = ex["prompt"]
        responses = {}
        scores    = {}

        for name, model in method_models.items():
            if model is None:
                continue
            model.eval()
            resp  = _greedy_decode(model, policy_tok, prompt, max_new_tokens, device)
            score = _rm_score_text(rm, rm_tok, prompt, resp, device)
            responses[name] = resp
            scores[name]    = score

        best_method  = max(scores, key=scores.get)
        # Mark disagreement: different methods have a spread > 0.5 in RM scores
        score_vals   = list(scores.values())
        disagreement = (max(score_vals) - min(score_vals)) > 0.5

        results.append({
            "prompt":       prompt,
            "responses":    responses,
            "scores":       scores,
            "best_method":  best_method,
            "disagreement": disagreement,
        })

    return results


def print_sample_table(
    table: List[Dict],
    output_file: Optional[str] = None,
) -> None:
    """Pretty-print the sample response table to stdout and optionally to a file."""
    lines = []
    lines.append("\n" + "="*80)
    lines.append(" C8: SAMPLE RESPONSE TABLE")
    lines.append("="*80)

    for i, row in enumerate(table):
        tag = " [DISAGREEMENT]" if row["disagreement"] else ""
        lines.append(f"\n── Prompt {i+1}{tag} ──")
        lines.append(f"PROMPT (last 200 chars): ...{row['prompt'][-200:]}")
        lines.append("")

        # Sort methods by RM score descending
        ranked = sorted(row["scores"].items(), key=lambda x: x[1], reverse=True)
        for method, score in ranked:
            resp = row["responses"].get(method, "N/A")
            lines.append(f"  [{method}]  RM={score:.3f}")
            lines.append(f"  Response: {resp[:250]}{'...' if len(resp)>250 else ''}")
            lines.append("")

    lines.append("="*80 + "\n")
    output = "\n".join(lines)
    print(output)
    if output_file:
        with open(output_file, "a") as f:
            f.write(output)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Resource tracker
# ─────────────────────────────────────────────────────────────────────────────

class ResourceTracker:
    """
    Records peak VRAM and timing for each training method.

    Usage:
        tracker = ResourceTracker()
        tracker.start("PPO")
        ... training ...
        tracker.stop("PPO", n_steps=200)
        tracker.report()
    """

    def __init__(self):
        self.records: Dict[str, Dict] = {}
        self._start_time: Dict[str, float] = {}
        self._start_vram: Dict[str, float] = {}

    def start(self, method: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self._start_vram[method] = torch.cuda.memory_allocated() / 1e9
        self._start_time[method] = time.time()

    def stop(self, method: str, n_steps: int) -> None:
        elapsed = time.time() - self._start_time.get(method, time.time())
        peak_vram = 0.0
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / 1e9
        self.records[method] = {
            "total_time_s":    elapsed,
            "n_steps":         n_steps,
            "time_per_step_s": elapsed / max(n_steps, 1),
            "peak_vram_gb":    peak_vram,
        }

    def report(self, output_file: Optional[str] = None) -> None:
        """Print a markdown resource table."""
        header = (
            f"\n{'Method':<10} | {'Total time':>12} | "
            f"{'Time/step':>10} | {'Peak VRAM':>10}"
        )
        sep    = "-" * len(header)
        lines  = ["\n" + "="*60, " C8: RESOURCE TABLE", "="*60, header, sep]
        for method, r in self.records.items():
            row = (
                f"{method:<10} | "
                f"{r['total_time_s']:>10.0f}s | "
                f"{r['time_per_step_s']:>9.1f}s | "
                f"{r['peak_vram_gb']:>8.2f} GB"
            )
            lines.append(row)
        lines.append("="*60 + "\n")
        output = "\n".join(lines)
        print(output)
        if output_file:
            with open(output_file, "a") as f:
                f.write(output)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Full evaluation driver
# ─────────────────────────────────────────────────────────────────────────────

def run_full_eval(
    method_models: Dict[str, object],   # {"SFT": model, "PPO": model, ...}
    sft_model,
    rm,
    rm_tok,
    policy_tok,
    test_examples: List[Dict],
    resource_tracker: Optional[ResourceTracker] = None,
    n_eval: int = 200,
    n_sample_prompts: int = 5,
    output_dir: str = "plots",
    device=None,
) -> Dict:
    """
    Run all C8 metrics for every method and produce a full report.

    Returns a results dict with all computed metrics for further analysis
    (e.g., to save as JSON for a working document).
    """
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, "c8_evaluation_report.txt")
    # Clear previous report
    open(report_file, "w").close()

    all_results = {}

    # ── Win-rate and KL for each method ──────────────────────────────────────
    print("\n" + "="*60)
    print(" C8: WIN-RATE AND KL SUMMARY")
    print("="*60)

    summary_lines = []
    header = f"{'Method':<10} | {'Win-Rate':>9} | {'Mean r(aligned)':>15} | {'Mean r(SFT)':>11} | {'KL(θ||ref)':>10} | {'Len(aligned)':>13}"
    print(header)
    print("-" * len(header))

    for method, model in tqdm(method_models.items(), desc="Methods", unit="method", dynamic_ncols=True):
        if model is None or method == "SFT":
            continue

        print(f"\n[eval] Computing win-rate for {method}...")
        wr_metrics = compute_win_rate(
            aligned_model=model,
            sft_model=sft_model,
            rm=rm, rm_tok=rm_tok, policy_tok=policy_tok,
            test_examples=test_examples,
            n_eval=n_eval,
            device=device,
        )

        print(f"[eval] Computing KL for {method}...")
        kl = compute_kl(
            policy=model,
            policy_tok=policy_tok,
            test_examples=test_examples,
            n_eval=n_eval,
            device=device,
        )

        all_results[method] = {**wr_metrics, "kl": kl}

        row = (
            f"{method:<10} | "
            f"{wr_metrics['win_rate']:>8.3f}  | "
            f"{wr_metrics['mean_r_aligned']:>14.3f}  | "
            f"{wr_metrics['mean_r_sft']:>10.3f}  | "
            f"{kl:>9.4f}  | "
            f"{wr_metrics['mean_len_aligned']:>12.1f}"
        )
        print(row)
        summary_lines.append(row)

    with open(report_file, "a") as f:
        f.write("C8: WIN-RATE AND KL\n")
        f.write(header + "\n" + "-"*len(header) + "\n")
        f.write("\n".join(summary_lines) + "\n\n")

    # ── Sample response table ─────────────────────────────────────────────────
    print("\n[eval] Building sample response table...")
    all_models = {"SFT": sft_model, **{k: v for k, v in method_models.items() if v}}
    sample_table = build_sample_table(
        method_models=all_models,
        rm=rm, rm_tok=rm_tok, policy_tok=policy_tok,
        test_examples=test_examples,
        n_prompts=n_sample_prompts,
        device=device,
    )
    print_sample_table(sample_table, output_file=report_file)

    # ── Resource table ────────────────────────────────────────────────────────
    if resource_tracker is not None:
        resource_tracker.report(output_file=report_file)

    # ── Save results JSON ──────────────────────────────────────────────────────
    results_path = os.path.join(output_dir, "c8_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[eval] Results saved to {results_path}")
    print(f"[eval] Full report saved to {report_file}")

    return all_results