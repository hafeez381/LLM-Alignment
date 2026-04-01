"""
train_rl.py — RL Training Loop Dispatcher (Tasks C3–C6)
========================================================

Usage:
    python train_rl.py --method ppo
    python train_rl.py --method dpo
    python train_rl.py --method grpo
    python train_rl.py --method rlvr

All RL methods (ppo, grpo, rlvr) start from the SFT-MERGED checkpoint.
DPO also starts from the SFT-MERGED checkpoint.
RLVR MUST start from the plain SFT checkpoint — NOT from a PPO/GRPO/DPO
checkpoint, even though it uses the same GSM8K data. The HH-RLHF SFT
policy is the shared π_ref for ALL methods (including RLVR).

DO NOT start from the raw pretrained model — you'd be penalising KL
divergence from an unaligned model, making KL a meaningless signal.

PSEUDOCODE — PPO:
──────────────────
1.  Load policy (sft_merged + fresh LoRA)    ← π_θ, trainable
2.  Load value model (Llama-1B + v_head)     ← V_φ, trainable
3.  Load frozen RM (from checkpoints/rm/)    ← r_ψ, frozen
4.  Create prompt pool dataloader
5.  For step in range(total_steps):
    a.  collect_rollouts(...)    → rollout_buffer
    b.  ppo_update(...)          → metrics
    c.  Log: mean RM score, KL, clip_loss, v_loss, grad_norm
    d.  Every eval_every: eval RM win-rate on 200 held-out prompts
6.  Save policy adapter + merge

PSEUDOCODE — DPO:
──────────────────
1.  Load policy (sft_merged + fresh LoRA)    ← π_θ, trainable
2.  Build DPO dataloader (preference pairs)
3.  For each batch (with grad accum):
    a.  dpo_loss(policy, chosen, rejected)   → loss, metrics
    b.  loss / grad_accum_steps → .backward()
    c.  Every grad_accum_steps: clip_grad_norm, optimizer.step
    d.  Log: loss, pref_acc, z_mean
    e.  Every 25 steps: evaluate on held-out DPO pairs
4.  Save adapter + merge

PSEUDOCODE — GRPO:
──────────────────
1. Load policy (sft_merged + fresh LoRA) ← π_θ, trainable
2. Load frozen RM                         ← r_ψ, frozen
3. Create prompt pool dataloader
4. For step in range(total_steps):
   a. batch = next(prompt_iter)
   b. rollout = group_rollout(policy, rm, batch, K=4, ...)
   c. log degenerate fraction
   d. metrics = grpo_update(policy, optimizer, rollout)
   e. Log + eval every eval_every steps
5. Save + merge

PSEUDOCODE — RLVR:
──────────────────
Same as GRPO but:
   b. reward_fn = make_rlvr_reward_fn(gold_map, policy_tok)
      rollout = group_rollout(policy, rm=None, reward_fn=reward_fn, batch, K=4, ...)
   Extra logging: pass@1, format_compliance, credit_assignment_frac
"""

import os
import sys
import argparse
import time
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.stdout.reconfigure(line_buffering=True)

from config import cfg
from model.loader import (
    load_policy_tokenizer,
    load_rm_tokenizer,
    load_policy_base_model,
    load_frozen_reference_model,
    load_reward_backbone,
    print_model_stats,
)
from model.lora_setup import apply_lora, merge_and_save
from model.reward_head import RewardModel
from model.value_head import load_value_model
from data.hh_rlhf import (
    load_hh_rlhf, parse_dataset,
    DPODataset, PromptDataset,
)
from data.gsm8k import load_gsm8k, GSM8KDataset
from alignment.ppo import (
    collect_rollouts, ppo_update, run_sanity_checks as ppo_sanity,
)
from alignment.dpo import dpo_loss, evaluate_dpo
from alignment.grpo import group_rollout, grpo_update
from alignment.rlvr import (
    make_rlvr_reward_fn,
    evaluate_rlvr,
    compute_credit_assignment_fraction,
    generate_sample_table,
    print_sample_table,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_policy_from_sft(device, model_id=None):
    """Load SFT-merged checkpoint + fresh LoRA → π_θ."""
    path = model_id or cfg.sft.merged_save_dir
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"SFT merged checkpoint not found: {path}. Run train_sft.py first."
        )
    print(f"[train_rl] Loading policy from: {path}")
    base  = load_policy_base_model(model_id=path)
    model = apply_lora(base)
    model.to(device).train()
    return model


def _load_frozen_rm(device):
    """Load frozen RM from checkpoints/rm/."""
    rm_path = cfg.rm.save_dir
    if not os.path.exists(rm_path):
        raise FileNotFoundError(
            f"RM checkpoint not found: {rm_path}. Run train_rm.py first."
        )
    print(f"[train_rl] Loading frozen RM from: {rm_path}")
    backbone = load_reward_backbone(model_id=rm_path)
    rm       = RewardModel(backbone)
    rm.to(device).eval()
    for p in rm.parameters():
        p.requires_grad_(False)
    return rm


def _infinite_iter(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def _eval_rm_score(policy, rm, rm_tok, policy_tok, prompt_loader, device,
                   step_label, n_batches=5):
    """Quick held-out eval: mean RM score on greedy responses."""
    policy.eval()
    scores = []
    for i, batch in enumerate(prompt_loader):
        if i >= n_batches:
            break
        prompt_ids  = batch["input_ids"].to(device)
        prompt_mask = batch["attention_mask"].to(device)
        raw_prompts = batch["raw_prompt"]

        full_ids = policy.generate(
            input_ids=prompt_ids, attention_mask=prompt_mask,
            max_new_tokens=128, do_sample=False,
            pad_token_id=policy_tok.pad_token_id,
        )
        P        = prompt_ids.shape[1]
        resp_ids = full_ids[:, P:]

        texts = [
            raw_prompts[j] + " " + policy_tok.decode(resp_ids[j], skip_special_tokens=True)
            for j in range(resp_ids.shape[0])
        ]
        enc = rm_tok(texts, max_length=512, padding=True, truncation=True,
                     return_tensors="pt").to(device)
        s   = rm(enc["input_ids"], enc["attention_mask"])
        scores.extend(s.cpu().tolist())

    mean_s = sum(scores) / max(len(scores), 1)
    print(f"[eval] {step_label} | mean RM score = {mean_s:.4f} "
          f"over {len(scores)} responses")
    policy.train()


# ─────────────────────────────────────────────────────────────────────────────
# PPO training
# ─────────────────────────────────────────────────────────────────────────────

def train_ppo():
    ppo_cfg = cfg.ppo
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ppo] Device: {device}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── Run sanity checks before touching any real data ──────────────────
    all_pass = ppo_sanity()
    if not all_pass:
        raise RuntimeError("PPO sanity checks FAILED — fix implementation before training.")

    # ── Tokenizers ────────────────────────────────────────────────────────
    policy_tok = load_policy_tokenizer()
    rm_tok     = load_rm_tokenizer()

    # ── Policy: load SFT-merged checkpoint + fresh LoRA ──────────────────
    # We start from sft_merged (not pretrained) so that:
    #   (a) disable_adapter() gives π_ref = SFT policy (correct KL anchor)
    #   (b) the policy already generates coherent text from step 0
    policy     = _load_policy_from_sft(device)

    # ── Value model ───────────────────────────────────────────────────────
    # Freeze backbone, train only the scalar head. More stable for short runs.
    # TODO: try unfreeze + LoRA backbone after establishing a working baseline.
    value_head = load_value_model(freeze_backbone=True)
    value_head.to(device).train()

    # ── Frozen reward model ───────────────────────────────────────────────
    rm = _load_frozen_rm(device)

    print_model_stats(policy,     "Policy + LoRA (PPO)")
    print_model_stats(value_head, "Value Head (frozen backbone)")

    # ── Dataset ───────────────────────────────────────────────────────────
    raw_train, raw_test = load_hh_rlhf()
    train_examples      = parse_dataset(raw_train)
    test_examples       = parse_dataset(raw_test)

    train_loader = DataLoader(
        PromptDataset(train_examples, policy_tok, max_prompt_len=512),
        batch_size=ppo_cfg.prompts_per_step, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    test_loader = DataLoader(
        PromptDataset(test_examples, policy_tok, max_prompt_len=512),
        batch_size=ppo_cfg.prompts_per_step, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    # Infinite iterator over training prompts
    train_iter = _infinite_iter(train_loader)

    # ── Optimisers ────────────────────────────────────────────────────────
    # Separate optimisers for policy (LoRA) and value head (v_head only when
    # backbone is frozen). Using one shared optimiser can cause the value
    # head LR to conflict with the policy LoRA LR.
    policy_optimizer = AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.01
    )
    value_optimizer = AdamW(
        [p for p in value_head.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.01
    )
    # Higher LR for value head: it's a simple linear layer starting from near-zero

    os.makedirs(ppo_cfg.save_dir, exist_ok=True)
    print(f"\n[ppo] Starting PPO training for {ppo_cfg.total_steps} steps...")
    t0 = time.time()

    # ── Training loop ─────────────────────────────────────────────────────
    for step in tqdm(range(1, ppo_cfg.total_steps + 1),
                     desc="PPO", unit="step", dynamic_ncols=True):

        # Phase A: Rollout
        # We wrap prompt_train_loader into a single-batch loader for collect_rollouts
        # by passing a list with one batch dict
        batch = next(train_iter)

        # Temporarily wrap the single batch as a 1-element iterable
        rollout_buffer = collect_rollouts(
            policy=policy,
            rm=rm,
            value_head=value_head,
            prompt_loader=[batch],
            policy_tok=policy_tok,
            rm_tok=rm_tok,
            device=device,
            ppo_cfg=ppo_cfg,
            n_batches=1,
        )

        # Phase B: Update
        metrics = ppo_update(
            policy=policy,
            value_head=value_head,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
            rollout_buffer=rollout_buffer,
            device=device,
            ppo_cfg=ppo_cfg,
        )

        # ── Logging ───────────────────────────────────────────────────────
        mean_rm  = sum(b["mean_rm"]  for b in rollout_buffer) / len(rollout_buffer)
        mean_kl  = sum(b["mean_kl"]  for b in rollout_buffer) / len(rollout_buffer)

        print(
            f"[ppo] step={step:4d} | "
            f"rm={mean_rm:.3f} | "
            f"kl={mean_kl:.4f} | "
            f"clip={metrics.get('clip_loss',0):.4f} | "
            f"v_loss={metrics.get('v_loss',0):.4f} | "
            f"ρ_mean={metrics.get('rho_mean',1):.3f} | "
            f"gn={metrics.get('grad_norm_policy',0):.3f} | "
            f"t={time.time()-t0:.0f}s"
        )

        # ── Evaluation ────────────────────────────────────────────────────
        if step % ppo_cfg.eval_every == 0:
            print(f"\n── Eval @ step {step} ──")
            _eval_rm_score(policy, rm, rm_tok, policy_tok, test_loader, device,
                           f"ppo step={step}")
            policy.train()

    # ── Save ──────────────────────────────────────────────────────────────
    print(f"\n[ppo] Training done in {time.time()-t0:.0f}s. Saving...")
    policy.save_pretrained(ppo_cfg.save_dir)
    policy_tok.save_pretrained(ppo_cfg.save_dir)
    merge_and_save(policy, os.path.join(ppo_cfg.save_dir, "merged"), policy_tok)
    print(f"[ppo] Saved to {ppo_cfg.save_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# DPO training
# ─────────────────────────────────────────────────────────────────────────────

def train_dpo():
    dpo_cfg = cfg.dpo
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[dpo] Device: {device}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── Tokenizer + policy ────────────────────────────────────────────────
    policy_tok = load_policy_tokenizer()
    policy     = _load_policy_from_sft(device)
    print_model_stats(policy, "Policy + LoRA (DPO)")

    # ── Dataset ───────────────────────────────────────────────────────────
    raw_train, raw_test = load_hh_rlhf()
    train_examples      = parse_dataset(raw_train)
    test_examples       = parse_dataset(raw_test)

    train_ds = DPODataset(train_examples, policy_tok, cfg.tokenizer.max_seq_len)
    test_ds  = DPODataset(test_examples,  policy_tok, cfg.tokenizer.max_seq_len)

    train_loader = DataLoader(
        train_ds, batch_size=dpo_cfg.batch_size,
        shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=dpo_cfg.batch_size,
        shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True,
    )
    print(f"[dpo] Train batches: {len(train_loader):,} | Test: {len(test_loader):,}")

    # ── Optimiser ─────────────────────────────────────────────────────────
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=dpo_cfg.learning_rate, weight_decay=0.01)

    total_opt_steps = len(train_loader) * dpo_cfg.num_epochs // dpo_cfg.grad_accum_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=30, num_training_steps=total_opt_steps
    )

    os.makedirs(dpo_cfg.save_dir, exist_ok=True)
    print(f"\n[dpo] Starting DPO training ({dpo_cfg.num_epochs} epoch)...")

    # ── Sanity check at initialisation ───────────────────────────────────
    # At step 0, π_θ ≈ π_ref → z ≈ 0 → pref_acc ≈ 0.50.
    # If pref_acc is not ≈ 0.50, something is wrong with the reference model.
    print("\n[dpo] Pre-training sanity check (should see pref_acc ≈ 0.50)...")
    init_metrics = evaluate_dpo(policy, test_loader, device,
                                beta=dpo_cfg.beta, max_batches=5)
    print(f"[dpo] Init: pref_acc={init_metrics['pref_acc']:.3f}  "
          f"z_mean={init_metrics['z_mean']:.4f}  "
          f"loss={init_metrics['dpo_loss']:.4f}")
    if init_metrics["pref_acc"] > 0.60:
        print("[dpo] ⚠ pref_acc >> 0.50 at init. Check that π_ref == π_θ baseline.")

    t0     = time.time()
    gstep  = 0   # optimizer steps
    rstep  = 0   # raw batch steps

    for epoch in range(dpo_cfg.num_epochs):
        for batch in tqdm(train_loader, desc=f"DPO epoch {epoch+1}/{dpo_cfg.num_epochs}",
                          unit="batch", dynamic_ncols=True):
            rstep += 1

            chosen_ids      = batch["chosen_input_ids"].to(device)
            chosen_mask     = batch["chosen_attention_mask"].to(device)
            chosen_labels   = batch["chosen_labels"].to(device)
            rejected_ids    = batch["rejected_input_ids"].to(device)
            rejected_mask   = batch["rejected_attention_mask"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)

            # ── Forward + loss ────────────────────────────────────────────
            loss, metrics = dpo_loss(
                policy,
                chosen_ids, chosen_mask, chosen_labels,
                rejected_ids, rejected_mask, rejected_labels,
                beta=dpo_cfg.beta,
            )

            # Scale for gradient accumulation
            (loss / dpo_cfg.grad_accum_steps).backward()

            # ── Optimiser step ────────────────────────────────────────────
            if rstep % dpo_cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, dpo_cfg.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                gstep += 1

                # ── Logging ───────────────────────────────────────────────
                if gstep % dpo_cfg.log_every == 0:
                    print(
                        f"[dpo] step={gstep:4d} | "
                        f"loss={metrics['dpo_loss']:.4f} | "
                        f"pref_acc={metrics['pref_acc']:.3f} | "
                        f"z={metrics['z_mean']:.3f} | "
                        f"σ(z)={metrics['sigma_z_mean']:.3f} | "
                        f"r⁺={metrics['impl_r_pos_mean']:.3f} | "
                        f"r⁻={metrics['impl_r_neg_mean']:.3f} | "
                        f"t={time.time()-t0:.0f}s"
                    )

                # ── Evaluation ─────────────────────────────────────────────
                if gstep % 25 == 0:
                    print(f"\n[dpo] ── Eval @ step {gstep} ──")
                    eval_m = evaluate_dpo(policy, test_loader, device,
                                          beta=dpo_cfg.beta, max_batches=50)
                    print(
                        f"[dpo-eval] loss={eval_m['dpo_loss']:.4f} | "
                        f"pref_acc={eval_m['pref_acc']:.3f} | "
                        f"z={eval_m['z_mean']:.3f} | "
                        f"margin={eval_m['impl_r_margin']:.3f}\n"
                    )
                    if eval_m["pref_acc"] < 0.52 and gstep > 100:
                        print("[dpo] ⚠ pref_acc stuck at ~50% after 100 steps. "
                              "Check: (1) response masking, (2) π_ref is frozen.")
                    policy.train()

    # ── Save ──────────────────────────────────────────────────────────────
    print(f"\n[dpo] Training done in {time.time()-t0:.0f}s. Saving...")
    policy.save_pretrained(dpo_cfg.save_dir)
    policy_tok.save_pretrained(dpo_cfg.save_dir)
    merge_and_save(policy, os.path.join(dpo_cfg.save_dir, "merged"), policy_tok)
    print(f"[dpo] Saved to {dpo_cfg.save_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# GRPO training
# ─────────────────────────────────────────────────────────────────────────────

def train_grpo():
    grpo_cfg = cfg.grpo
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[grpo] Device: {device}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    policy_tok = load_policy_tokenizer()
    rm_tok     = load_rm_tokenizer()
    policy     = _load_policy_from_sft(device)
    rm         = _load_frozen_rm(device)

    print_model_stats(policy, "Policy + LoRA (GRPO)")

    raw_train, raw_test = load_hh_rlhf()
    train_ex            = parse_dataset(raw_train)
    test_ex             = parse_dataset(raw_test)

    train_loader = DataLoader(
        PromptDataset(train_ex, policy_tok, max_prompt_len=512),
        batch_size=grpo_cfg.prompts_per_step, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    test_loader = DataLoader(
        PromptDataset(test_ex, policy_tok, max_prompt_len=512),
        batch_size=grpo_cfg.prompts_per_step, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    train_iter = _infinite_iter(train_loader)

    optimizer = AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.01
    )
    os.makedirs(grpo_cfg.save_dir, exist_ok=True)
    deg_frac_history = []   # (step, frac_degenerate) pairs

    # Running stats for degenerate batch monitoring
    total_batches    = 0
    degenerate_count = 0.0

    print(f"\n[grpo] Training for {grpo_cfg.total_steps} steps, K={grpo_cfg.K}...")
    t0 = time.time()

    for step in tqdm(range(1, grpo_cfg.total_steps + 1),
                     desc="GRPO", unit="step", dynamic_ncols=True):
        batch = next(train_iter)

        # ── Rollout ────────────────────────────────────────────────────────
        # group_rollout generates K completions per prompt using π_old,
        # caches log_probs_old and log_probs_ref, and computes group advantages.
        rollout = group_rollout(
            policy=policy,
            rm=rm,
            rm_tok=rm_tok,
            prompt_batch=batch,
            policy_tok=policy_tok,
            device=device,
            grpo_cfg=grpo_cfg,
            reward_fn=None,    # use learned RM (not verifiable reward)
        )

        # ── Degenerate batch tracking ─────────────────────────────────────
        # If std(r_{b,k}) ≈ 0 for a prompt, all A_{b,k}=0 → zero gradient.
        # Assignment spec: if >30% are degenerate, RM is not discriminating.
        frac_deg = rollout["frac_degenerate"]
        deg_frac_history.append((step, frac_deg))
        degenerate_count += frac_deg
        total_batches    += 1
        running_frac_deg  = degenerate_count / total_batches

        # ── Update ────────────────────────────────────────────────────────
        metrics = grpo_update(
            policy=policy,
            optimizer=optimizer,
            rollout_buffer=rollout,
            device=device,
            grpo_cfg=grpo_cfg,
            use_full_kl=False,    # MC-approx KL (less VRAM)
        )

        print(
            f"[grpo] step={step:4d} | "
            f"rm={rollout['mean_rm']:.3f} | "
            f"kl={metrics.get('kl_mean',0):.4f} | "
            f"clip={metrics.get('clip_loss',0):.4f} | "
            f"ρ={metrics.get('rho_mean',1):.3f} | "
            f"deg={frac_deg:.2f}(run={running_frac_deg:.2f}) | "
            f"len={rollout['mean_resp_len']:.0f} | "
            f"gn={metrics.get('grad_norm',0):.3f} | "
            f"t={time.time()-t0:.0f}s"
        )

        if running_frac_deg > 0.30 and step > 10:
            print(f"[grpo] ⚠ Degenerate fraction {running_frac_deg:.2f} > 0.30. "
                  f"RM may not discriminate well — check RM accuracy.")

        if step % grpo_cfg.eval_every == 0:
            print(f"\n── Eval @ step {step} ──")
            _eval_rm_score(policy, rm, rm_tok, policy_tok, test_loader, device,
                           f"grpo step={step}")
            policy.train()

    # ── Save degenerate batch fraction plot ───────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("plots", exist_ok=True)
    method_label = "grpo"
    steps_list = [s for s, _ in deg_frac_history]
    deg_list   = [d for _, d in deg_frac_history]
    window = 10
    running_mean = [
        sum(deg_list[max(0, i - window + 1):i + 1]) / (i - max(0, i - window + 1) + 1)
        for i in range(len(deg_list))
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps_list, deg_list,     alpha=0.3, color="steelblue", linewidth=1,  label="Per-step")
    ax.plot(steps_list, running_mean,             color="steelblue", linewidth=2,  label=f"Running mean (window={window})")
    ax.axhline(0.30, color="tomato", linestyle="--", linewidth=1.5, label="30% warning threshold")
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Fraction of degenerate batches", fontsize=12)
    ax.set_title(f"{method_label.upper()}: Degenerate Batch Fraction Over Training", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    save_path = f"plots/{method_label}_degenerate_batches.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[{method_label}] Degenerate batch plot saved → {save_path}")

    # ── Save ──────────────────────────────────────────────────────────────
    policy.save_pretrained(grpo_cfg.save_dir)
    policy_tok.save_pretrained(grpo_cfg.save_dir)
    merge_and_save(policy, os.path.join(grpo_cfg.save_dir, "merged"), policy_tok)
    print(f"[grpo] Done. Saved to {grpo_cfg.save_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# RLVR training
# ─────────────────────────────────────────────────────────────────────────────

def train_rlvr():
    """
    RLVR = GRPO on GSM8K with r_v ∈ {0,1} instead of a learned RM.

    Key differences from GRPO:
        - Dataset: GSM8K (math word problems), not HH-RLHF
        - Reward: verifiable_reward() (exact answer match), not r_ψ(x,y)
        - Hyperparameters: β=0.05 (less KL penalty; r_v is binary, not continuous)
        - max_new_tokens=256 (reasoning chains are longer than dialogue)
        - CRITICAL: initialise from SFT checkpoint, NOT from GRPO/PPO checkpoint
    """
    rlvr_cfg = cfg.rlvr
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[rlvr] Device: {device}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    policy_tok  = load_policy_tokenizer()
    # RLVR starts from the SAME SFT checkpoint as all other methods.
    # We do NOT use the GRPO/PPO checkpoint because those were aligned for
    # HH-RLHF preferences — starting from them would conflate HH alignment
    # signal with math reasoning signal.
    policy = _load_policy_from_sft(device)
    print_model_stats(policy, "Policy + LoRA (RLVR)")

    # ── Dataset ───────────────────────────────────────────────────────────
    train_examples, test_examples = load_gsm8k()

    train_ds = GSM8KDataset(train_examples, policy_tok, max_prompt_len=256)
    train_loader = DataLoader(
        train_ds,
        batch_size=rlvr_cfg.prompts_per_step,
        shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    train_iter = _infinite_iter(train_loader)

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer = AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.01
    )
    os.makedirs(rlvr_cfg.save_dir, exist_ok=True)
    deg_frac_history = []   # (step, frac_degenerate) pairs

    # ── Degenerate batch tracking ─────────────────────────────────────────
    total_batches    = 0
    total_deg_frac   = 0.0
    total_pass       = 0
    total_scored     = 0

    print(f"\n[rlvr] Training for {rlvr_cfg.total_steps} steps, K={rlvr_cfg.K}...")
    t0 = time.time()

    for step in tqdm(range(1, rlvr_cfg.total_steps + 1),
                     desc="RLVR", unit="step", dynamic_ncols=True):
        batch = next(train_iter)

        # Build gold answer map for this batch's prompts
        # (group_rollout needs it to compute r_v per completion)
        gold_map = {
            batch["raw_prompt"][i]: batch["gold_answer"][i].item()
            if hasattr(batch["gold_answer"][i], "item")
            else batch["gold_answer"][i]
            for i in range(len(batch["raw_prompt"]))
        }
        reward_fn = make_rlvr_reward_fn(gold_map, policy_tok)

        # ── Group rollout with verifiable reward ──────────────────────────
        # Overrides the GRPO config's beta with RLVR's lower beta (0.05).
        # We pass a modified grpo_cfg proxy by temporarily overriding fields.
        # Cleaner approach: create a dataclass copy. We do it inline here.
        class _RLVRGRPOCfg:
            K              = rlvr_cfg.K
            max_new_tokens = rlvr_cfg.max_new_tokens
            beta           = rlvr_cfg.beta
            epsilon        = rlvr_cfg.epsilon
            prompts_per_step = rlvr_cfg.prompts_per_step

        rollout = group_rollout(
            policy=policy,
            rm=None,              # no learned RM in RLVR
            rm_tok=None,
            prompt_batch=batch,
            policy_tok=policy_tok,
            device=device,
            grpo_cfg=_RLVRGRPOCfg(),
            reward_fn=reward_fn,  # verifiable reward
        )

        # ── Credit assignment analysis ─────────────────────────────────────
        # Fraction of response tokens carrying nonzero gradient.
        # For RLVR: degenerate batches give ZERO gradient to ALL tokens.
        # Even non-degenerate batches: ALL tokens in a completion get the
        # same gradient (unlike PPO which has per-token KL signal).
        # This illustrates PA2 Problem 4.3(b): sparse reward → sparse gradient.
        credit_frac = compute_credit_assignment_fraction(
            rollout["response_mask"],
            rollout["response_lens"],
            rollout["is_degenerate"],
            K=rlvr_cfg.K,
        )

        # ── Update ────────────────────────────────────────────────────────
        metrics = grpo_update(
            policy=policy,
            optimizer=optimizer,
            rollout_buffer=rollout,
            device=device,
            grpo_cfg=_RLVRGRPOCfg(),
            use_full_kl=False,
        )

        # ── Stats accumulation ────────────────────────────────────────────
        frac_deg       = rollout["frac_degenerate"]
        deg_frac_history.append((step, frac_deg))
        total_deg_frac += frac_deg
        total_batches  += 1
        running_deg     = total_deg_frac / total_batches

        # Track pass rate in training batch (mean reward = mean pass rate)
        batch_rm = rollout["mean_rm"]
        total_pass    += batch_rm * rlvr_cfg.K * rlvr_cfg.prompts_per_step
        total_scored  += rlvr_cfg.K * rlvr_cfg.prompts_per_step
        running_pass   = total_pass / max(total_scored, 1)

        print(
            f"[rlvr] step={step:4d} | "
            f"train_pass={batch_rm:.3f} | "
            f"run_pass={running_pass:.3f} | "
            f"kl={metrics.get('kl_mean',0):.4f} | "
            f"clip={metrics.get('clip_loss',0):.4f} | "
            f"deg={frac_deg:.2f}(run={running_deg:.2f}) | "
            f"credit={credit_frac:.2f} | "
            f"len={rollout['mean_resp_len']:.0f} | "
            f"gn={metrics.get('grad_norm',0):.3f} | "
            f"t={time.time()-t0:.0f}s"
        )

        if running_deg > 0.30 and step > 50:
            print(f"[rlvr] ⚠ Degenerate fraction {running_deg:.2f} > 0.30. "
                  f"Possible causes: (1) prompts too hard (all wrong) early in training, "
                  f"(2) answer extractor too strict. Check extract_answer on recent outputs.")

        # ── Evaluation ────────────────────────────────────────────────────
        if step % rlvr_cfg.eval_every == 0:
            print(f"\n── RLVR Eval @ step {step} ──")
            eval_m = evaluate_rlvr(
                policy=policy,
                test_examples=test_examples,
                policy_tok=policy_tok,
                device=device,
                max_new_tokens=rlvr_cfg.max_new_tokens,
                n_eval=200,
            )
            print(
                f"[rlvr-eval] pass@1={eval_m['pass_at_1']:.3f} | "
                f"format_compliance={eval_m['format_compliance']:.3f} | "
                f"extractable={eval_m['frac_extractable']:.3f} | "
                f"mean_len={eval_m['mean_resp_len']:.0f}\n"
            )
            if eval_m["pass_at_1"] == 0.0 and step > 100:
                print(
                    "[rlvr] ⚠ pass@1 still 0.0 after 100 steps. Check:\n"
                    "  (1) Is extract_answer finding '#### N' in gold solutions?\n"
                    "  (2) Does the prompt format match SFT training format?\n"
                    "  (3) Is max_new_tokens=256 (reasoning needs more space than 128)?"
                )
            policy.train()

    # ── Final sample table ─────────────────────────────────────────────────
    print("\n[rlvr] Generating final sample table...")
    samples = generate_sample_table(policy, test_examples, policy_tok, device, n_samples=5)
    print_sample_table(samples)

    # ── Save degenerate batch fraction plot ───────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("plots", exist_ok=True)
    method_label = "rlvr"
    steps_list = [s for s, _ in deg_frac_history]
    deg_list   = [d for _, d in deg_frac_history]
    window = 10
    running_mean = [
        sum(deg_list[max(0, i - window + 1):i + 1]) / (i - max(0, i - window + 1) + 1)
        for i in range(len(deg_list))
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps_list, deg_list,     alpha=0.3, color="steelblue", linewidth=1,  label="Per-step")
    ax.plot(steps_list, running_mean,             color="steelblue", linewidth=2,  label=f"Running mean (window={window})")
    ax.axhline(0.30, color="tomato", linestyle="--", linewidth=1.5, label="30% warning threshold")
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Fraction of degenerate batches", fontsize=12)
    ax.set_title(f"{method_label.upper()}: Degenerate Batch Fraction Over Training", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    save_path = f"plots/{method_label}_degenerate_batches.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[{method_label}] Degenerate batch plot saved → {save_path}")

    # ── Save ──────────────────────────────────────────────────────────────
    policy.save_pretrained(rlvr_cfg.save_dir)
    policy_tok.save_pretrained(rlvr_cfg.save_dir)
    merge_and_save(policy, os.path.join(rlvr_cfg.save_dir, "merged"), policy_tok)
    print(f"[rlvr] Done. Saved to {rlvr_cfg.save_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random, numpy as np

    parser = argparse.ArgumentParser(description="RL training dispatcher")
    parser.add_argument(
        "--method", type=str, required=True,
        choices=["ppo", "dpo", "grpo", "rlvr"],
        help="Which alignment method to train",
    )
    args = parser.parse_args()

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        # Faster matmul on A100/T4; minor accuracy trade-off acceptable for LLM training
        torch.backends.cuda.matmul.allow_tf32 = True

    dispatch = {"ppo": train_ppo, "dpo": train_dpo,
                "grpo": train_grpo, "rlvr": train_rlvr}
    dispatch[args.method]()
