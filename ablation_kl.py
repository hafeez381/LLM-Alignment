"""
ablation_kl.py — Task C7: KL Coefficient Sweep (β ablation on PPO)
===================================================================

Hypothesis:
    Increasing β reduces reward hacking by penalising divergence from π_ref,
    but also limits the policy's ability to improve beyond the SFT baseline.
    Specifically, we hypothesise that:
        β = 0    → highest RM score but significant reward hacking
                   (policy exploits RM quirks with no constraint)
        β = 0.05 → mild improvement with moderate KL
        β = 0.1  → baseline config; balanced reward/KL trade-off
        β = 0.5  → constrained policy; RM score close to SFT baseline

    We expect reward gaming to first appear at β = 0 or β = 0.05,
    visible as (a) high RM score but low generation quality, and
    (b) large KL from π_ref.

METHODOLOGY:
    - All hyperparameters fixed EXCEPT β (epsilon=0.2, steps=200, batch=16)
    - Each β value trains from the SAME SFT-merged checkpoint (same init)
    - Evaluation after training: RM score on 200 held-out prompts (greedy)
    - KL from π_ref on same 200 prompts (MC approximation)
    - Results saved to plots/c7_kl_sweep.png and plots/c7_kl_sweep.json

Usage:
    python ablation_kl.py

Runtime estimate on A100:
    Each PPO run: ~200 steps × ~15s/step ≈ 50 minutes
    4 β values × 50 min ≈ 3.5 hours total
"""

import os
import json
import time
import copy
import argparse
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import cfg
from model.loader import (
    load_policy_tokenizer, load_rm_tokenizer,
    load_policy_base_model, load_reward_backbone,
)
from model.lora_setup import apply_lora, merge_and_save
from model.reward_head import RewardModel
from model.value_head import load_value_model
from data.hh_rlhf import load_hh_rlhf, parse_dataset, PromptDataset
from alignment.ppo import collect_rollouts, ppo_update, run_sanity_checks
from eval import compute_kl, compute_win_rate


# β values to sweep (from assignment spec)
BETA_VALUES = [0.0, 0.05, 0.1, 0.5]


def _load_rm(device):
    """Load the frozen, already-trained RM."""
    rm_path = cfg.rm.save_dir
    backbone = load_reward_backbone(model_id=rm_path)
    rm = RewardModel(backbone)
    rm.to(device).eval()
    for p in rm.parameters():
        p.requires_grad_(False)
    return rm


def _infinite_iter(loader):
    while True:
        for batch in loader:
            yield batch


def run_ppo_with_beta(
    beta: float,
    policy_tok,
    rm_tok,
    rm,
    train_examples,
    test_examples,
    device: torch.device,
    total_steps: int = 200,
    prompts_per_step: int = 16,
) -> dict:
    """
    Train PPO for `total_steps` steps with a fixed β and return eval metrics.

    Returns
    -------
    dict with:
        beta          : the β value used
        mean_rm_score : mean RM score on 200 greedy outputs (proxy for reward)
        kl            : KL(π_θ || π_ref) on 200 prompts (proxy for drift)
        win_rate      : fraction of prompts beating SFT baseline
        step_rewards  : list of per-step mean RM scores (for plotting)
    """
    print(f"\n{'─'*60}")
    print(f" KL SWEEP: β = {beta}")
    print(f"{'─'*60}")

    # ── Fresh policy from SFT merged ──────────────────────────────────────
    sft_merged = cfg.sft.merged_save_dir
    base  = load_policy_base_model(model_id=sft_merged)
    policy = apply_lora(base)
    policy.to(device).train()

    # ── Value head ─────────────────────────────────────────────────────────
    value_head = load_value_model(freeze_backbone=True)
    value_head.to(device)
    # Fix dtype mismatch: v_head must match backbone's bfloat16
    value_head.v_head = value_head.v_head.to(torch.bfloat16)
    value_head.train()

    # ── SFT model (baseline for win-rate) — frozen, no LoRA ───────────────
    from model.loader import load_policy_base_model as _lpbm
    from model.lora_setup import freeze_model
    sft_for_eval = _lpbm(model_id=sft_merged)
    sft_for_eval = freeze_model(sft_for_eval)
    sft_for_eval.to(device).eval()

    # ── Dataloader ─────────────────────────────────────────────────────────
    prompt_ds = PromptDataset(train_examples, policy_tok, max_prompt_len=512)
    prompt_loader = DataLoader(
        prompt_ds, batch_size=prompts_per_step,
        shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True,
    )
    train_iter = _infinite_iter(prompt_loader)

    # ── Optimisers ─────────────────────────────────────────────────────────
    policy_opt = AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=2e-4
    )
    value_opt = AdamW(
        [p for p in value_head.parameters() if p.requires_grad], lr=1e-3
    )

    # ── Build a custom PPO config with this beta ───────────────────────────
    # We override cfg.ppo.beta without mutating the global cfg singleton.
    class _PpoCfg:
        total_steps       = total_steps
        prompts_per_step  = prompts_per_step
        max_new_tokens    = 128
        beta              = beta          # ← the sweep variable
        epsilon           = 0.2
        gamma             = 1.0
        lam               = 0.95
        ppo_epochs        = 4
        mini_batch_size   = 8
        c_value           = 0.5
        c_entropy         = 0.01
        eval_every        = 999           # disable mid-training eval

    ppo_cfg = _PpoCfg()
    step_rewards = []
    t0 = time.time()

    for step in range(1, total_steps + 1):
        batch  = next(train_iter)
        buffer = collect_rollouts(
            policy=policy, rm=rm, value_head=value_head,
            prompt_loader=[batch], policy_tok=policy_tok, rm_tok=rm_tok,
            device=device, ppo_cfg=ppo_cfg, n_batches=1,
        )
        metrics = ppo_update(
            policy=policy, value_head=value_head,
            policy_optimizer=policy_opt, value_optimizer=value_opt,
            rollout_buffer=buffer, device=device, ppo_cfg=ppo_cfg,
        )
        mean_rm = sum(b["mean_rm"] for b in buffer) / len(buffer)
        step_rewards.append(mean_rm)

        if step % 25 == 0:
            print(
                f"  [β={beta}] step={step:3d} | "
                f"rm={mean_rm:.3f} | "
                f"kl={sum(b['mean_kl'] for b in buffer)/len(buffer):.4f} | "
                f"clip={metrics.get('clip_loss',0):.4f} | "
                f"t={time.time()-t0:.0f}s"
            )

    # ── Post-training evaluation ───────────────────────────────────────────
    print(f"\n  [β={beta}] Running post-training evaluation...")
    kl = compute_kl(
        policy=policy, policy_tok=policy_tok,
        test_examples=test_examples, n_eval=200, device=device,
    )
    wr = compute_win_rate(
        aligned_model=policy, sft_model=sft_for_eval,
        rm=rm, rm_tok=rm_tok, policy_tok=policy_tok,
        test_examples=test_examples, n_eval=200, device=device,
    )
    mean_rm_eval = wr["mean_r_aligned"]

    print(
        f"  [β={beta}] FINAL — "
        f"RM score={mean_rm_eval:.3f} | "
        f"KL={kl:.4f} | "
        f"win-rate={wr['win_rate']:.3f}"
    )

    # Free GPU memory before next run
    del policy, value_head, sft_for_eval
    torch.cuda.empty_cache()

    return {
        "beta":           beta,
        "mean_rm_score":  mean_rm_eval,
        "kl":             kl,
        "win_rate":       wr["win_rate"],
        "mean_r_sft":     wr["mean_r_sft"],
        "step_rewards":   step_rewards,
    }


def plot_kl_sweep(results: list, output_dir: str = "plots") -> None:
    """
    Save two plots:
        (a) RM score vs β  (reward hacking proxy)
        (b) KL vs β        (policy drift proxy)
    And a bonus subplot showing per-step RM score curves for all β values.
    """
    os.makedirs(output_dir, exist_ok=True)

    betas       = [r["beta"]          for r in results]
    rm_scores   = [r["mean_rm_score"] for r in results]
    kl_values   = [r["kl"]            for r in results]
    win_rates   = [r["win_rate"]       for r in results]

    # ── Figure 1: Summary bar chart ────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    x = range(len(betas))
    b_labels = [str(b) for b in betas]

    axes[0].bar(x, rm_scores, color="steelblue")
    axes[0].set_xticks(x); axes[0].set_xticklabels(b_labels)
    axes[0].set_xlabel("β (KL coefficient)"); axes[0].set_ylabel("Mean RM score")
    axes[0].set_title("RM Score vs β\n(proxy for reward; higher=better)")
    for i, v in enumerate(rm_scores):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(x, kl_values, color="tomato")
    axes[1].set_xticks(x); axes[1].set_xticklabels(b_labels)
    axes[1].set_xlabel("β (KL coefficient)"); axes[1].set_ylabel("KL(π_θ || π_ref)")
    axes[1].set_title("KL Divergence vs β\n(higher=more drift from SFT)")
    for i, v in enumerate(kl_values):
        axes[1].text(i, v + 0.001, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    axes[2].bar(x, win_rates, color="seagreen")
    axes[2].axhline(0.5, color="gray", linestyle="--", label="50% baseline")
    axes[2].set_xticks(x); axes[2].set_xticklabels(b_labels)
    axes[2].set_xlabel("β (KL coefficient)"); axes[2].set_ylabel("Win-rate vs SFT")
    axes[2].set_title("Win-Rate vs SFT vs β")
    axes[2].legend()
    for i, v in enumerate(win_rates):
        axes[2].text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("C7 Ablation: KL Coefficient (β) Sweep — PPO", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "c7_kl_sweep_summary.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: Per-step RM score curves ────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    colors = ["#e41a1c", "#ff7f00", "#377eb8", "#4daf4a"]
    for r, color in zip(results, colors):
        steps = list(range(1, len(r["step_rewards"]) + 1))
        ax2.plot(steps, r["step_rewards"], label=f"β={r['beta']}", color=color,
                 linewidth=1.5, alpha=0.85)

    ax2.set_xlabel("Training step")
    ax2.set_ylabel("Mean RM score (rollout batch)")
    ax2.set_title("C7: RM Score over Training for Different β Values")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "c7_kl_sweep_curves.png"),
                 dpi=150)
    plt.close(fig2)
    print(f"[c7] Plots saved to {output_dir}/")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[c7] Device: {device}")

    # ── Sanity checks ──────────────────────────────────────────────────────
    if not run_sanity_checks():
        raise RuntimeError("PPO sanity checks FAILED")

    # ── Shared resources ───────────────────────────────────────────────────
    policy_tok = load_policy_tokenizer()
    rm_tok     = load_rm_tokenizer()
    rm         = _load_rm(device)

    raw_train, raw_test = load_hh_rlhf()
    train_ex = parse_dataset(raw_train)
    test_ex  = parse_dataset(raw_test)

    # ── Run sweep ──────────────────────────────────────────────────────────
    all_results = []
    for beta in BETA_VALUES:
        result = run_ppo_with_beta(
            beta=beta,
            policy_tok=policy_tok, rm_tok=rm_tok, rm=rm,
            train_examples=train_ex, test_examples=test_ex,
            device=device,
            total_steps=cfg.ppo.total_steps,
            prompts_per_step=cfg.ppo.prompts_per_step,
        )
        all_results.append(result)

    # ── Save JSON ──────────────────────────────────────────────────────────
    os.makedirs("plots", exist_ok=True)
    with open("plots/c7_kl_sweep.json", "w") as f:
        # step_rewards are long lists — save separately to keep JSON readable
        summary = [{k: v for k, v in r.items() if k != "step_rewards"}
                   for r in all_results]
        json.dump(summary, f, indent=2)

    # ── Print summary table ────────────────────────────────────────────────
    print("\n" + "="*65)
    print(" C7 KL SWEEP RESULTS")
    print("="*65)
    print(f"  Hypothesis: β=0 causes reward hacking (high RM, high KL).")
    print(f"  β values tested: {BETA_VALUES}\n")
    print(f"  {'β':>6} | {'RM score':>9} | {'KL':>9} | {'Win-rate':>9}")
    print(f"  {'-'*42}")
    for r in all_results:
        print(
            f"  {r['beta']:>6.2f} | "
            f"{r['mean_rm_score']:>8.3f}  | "
            f"{r['kl']:>8.4f}  | "
            f"{r['win_rate']:>8.3f}"
        )

    # Identify first β where reward gaming appears (RM score + KL both high)
    # Heuristic: RM score > mean + 0.1 AND KL > 0.1
    mean_rm = sum(r["mean_rm_score"] for r in all_results) / len(all_results)
    gaming  = [r["beta"] for r in all_results
               if r["mean_rm_score"] > mean_rm + 0.1 and r["kl"] > 0.1]
    if gaming:
        print(f"\n  First reward gaming detected at β = {min(gaming)}")
    else:
        print(f"\n  No clear reward gaming detected across tested β values.")

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_kl_sweep(all_results, output_dir="plots")
    print("\n[c7] Done.")


if __name__ == "__main__":
    import random, numpy as np
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    main()
