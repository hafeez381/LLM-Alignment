"""
train_rl.py — RL Training Loop Dispatcher (Tasks C3 & C4)
==========================================================

Usage:
    python train_rl.py --method ppo
    python train_rl.py --method dpo

Both methods start from the SFT-merged checkpoint (π_ref).
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
"""

import os
import argparse
import time
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

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
from alignment.ppo import (
    collect_rollouts, ppo_update, run_sanity_checks as ppo_sanity,
)
from alignment.dpo import dpo_loss, evaluate_dpo


# ─────────────────────────────────────────────────────────────────────────────
# PPO training
# ─────────────────────────────────────────────────────────────────────────────

def train_ppo():
    ppo_cfg = cfg.ppo
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ppo] Device: {device}")

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
    sft_merged_path = cfg.sft.merged_save_dir
    if not os.path.exists(sft_merged_path):
        raise FileNotFoundError(
            f"SFT merged checkpoint not found at {sft_merged_path}. "
            "Run train_sft.py first."
        )

    print(f"[ppo] Loading policy from SFT merged: {sft_merged_path}")
    base_model = load_policy_base_model(model_id=sft_merged_path)
    policy     = apply_lora(base_model)   # π_θ with fresh LoRA
    policy.to(device)
    policy.train()

    # ── Value model ───────────────────────────────────────────────────────
    # Freeze backbone, train only the scalar head. More stable for short runs.
    # TODO: try unfreeze + LoRA backbone after establishing a working baseline.
    value_head = load_value_model(freeze_backbone=True)
    value_head.to(device)
    value_head.train()

    print_model_stats(policy,     "Policy + LoRA (PPO)")
    print_model_stats(value_head, "Value Head (frozen backbone)")

    # ── Frozen reward model ───────────────────────────────────────────────
    rm_backbone = load_reward_backbone()
    rm          = RewardModel(rm_backbone)
    rm.to(device)
    rm.eval()
    for p in rm.parameters():
        p.requires_grad_(False)
    print("[ppo] RM loaded and frozen")

    # ── Dataset ───────────────────────────────────────────────────────────
    raw_train, raw_test = load_hh_rlhf()
    train_examples      = parse_dataset(raw_train)
    test_examples       = parse_dataset(raw_test)

    prompt_train_ds = PromptDataset(train_examples, policy_tok, max_prompt_len=512)
    prompt_test_ds  = PromptDataset(test_examples,  policy_tok, max_prompt_len=512)

    prompt_train_loader = DataLoader(
        prompt_train_ds,
        batch_size=ppo_cfg.prompts_per_step,
        shuffle=True,
        num_workers=0,
    )
    prompt_test_loader = DataLoader(
        prompt_test_ds,
        batch_size=ppo_cfg.prompts_per_step,
        shuffle=False,
        num_workers=0,
    )
    # Infinite iterator over training prompts
    def cycle(loader):
        while True:
            for batch in loader:
                yield batch
    train_iter = cycle(prompt_train_loader)

    # ── Optimisers ────────────────────────────────────────────────────────
    # Separate optimisers for policy (LoRA) and value head (v_head only when
    # backbone is frozen). Using one shared optimiser can cause the value
    # head LR to conflict with the policy LoRA LR.
    policy_params = [p for p in policy.parameters() if p.requires_grad]
    value_params  = [p for p in value_head.parameters() if p.requires_grad]

    policy_optimizer = AdamW(policy_params, lr=2e-4, weight_decay=0.01)
    value_optimizer  = AdamW(value_params,  lr=1e-3, weight_decay=0.01)
    # Higher LR for value head: it's a simple linear layer starting from near-zero

    os.makedirs(ppo_cfg.save_dir, exist_ok=True)
    print(f"\n[ppo] Starting PPO training for {ppo_cfg.total_steps} steps...")
    t0 = time.time()

    # ── Training loop ─────────────────────────────────────────────────────
    for step in range(1, ppo_cfg.total_steps + 1):

        # Phase A: Rollout
        # We wrap prompt_train_loader into a single-batch loader for collect_rollouts
        # by passing a list with one batch dict
        batch = next(train_iter)

        # Temporarily wrap the single batch as a 1-element iterable
        single_batch_iter = [batch]

        rollout_buffer = collect_rollouts(
            policy=policy,
            rm=rm,
            value_head=value_head,
            prompt_loader=single_batch_iter,
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
            print(f"\n[ppo] ── Evaluation @ step {step} ──")
            _eval_ppo(policy, rm, rm_tok, policy_tok, prompt_test_loader, device, step)
            policy.train()

    # ── Save ──────────────────────────────────────────────────────────────
    print(f"\n[ppo] Training done in {time.time()-t0:.0f}s. Saving...")
    policy.save_pretrained(ppo_cfg.save_dir)
    policy_tok.save_pretrained(ppo_cfg.save_dir)
    merged_path = os.path.join(ppo_cfg.save_dir, "merged")
    merge_and_save(policy, merged_path, policy_tok)
    print(f"[ppo] Saved to {ppo_cfg.save_dir}")


@torch.no_grad()
def _eval_ppo(policy, rm, rm_tok, policy_tok, prompt_loader, device, step,
              n_batches=5):
    """
    Quick evaluation: mean RM score on greedy responses from n_batches prompts.
    A rising RM score over PPO steps is the primary quality signal.
    """
    policy.eval()
    rm.eval()
    all_scores = []

    for i, batch in enumerate(prompt_loader):
        if i >= n_batches:
            break
        prompt_ids  = batch["input_ids"].to(device)
        prompt_mask = batch["attention_mask"].to(device)
        raw_prompts = batch["raw_prompt"]

        full_ids = policy.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=128,
            do_sample=False,     # greedy for eval (reproducible)
            pad_token_id=policy_tok.pad_token_id,
        )
        P        = prompt_ids.shape[1]
        resp_ids = full_ids[:, P:]

        full_texts = [
            raw_prompts[j] + " " + policy_tok.decode(resp_ids[j], skip_special_tokens=True)
            for j in range(resp_ids.shape[0])
        ]
        enc = rm_tok(full_texts, max_length=512, padding=True, truncation=True,
                     return_tensors="pt").to(device)
        scores = rm(enc["input_ids"], enc["attention_mask"])
        all_scores.extend(scores.cpu().tolist())

    mean_score = sum(all_scores) / max(len(all_scores), 1)
    print(f"[ppo-eval] step={step} | mean RM score = {mean_score:.4f} "
          f"over {len(all_scores)} responses")


# ─────────────────────────────────────────────────────────────────────────────
# DPO training
# ─────────────────────────────────────────────────────────────────────────────

def train_dpo():
    dpo_cfg = cfg.dpo
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[dpo] Device: {device}")

    # ── Tokenizer + policy ────────────────────────────────────────────────
    policy_tok = load_policy_tokenizer()

    sft_merged_path = cfg.sft.merged_save_dir
    if not os.path.exists(sft_merged_path):
        raise FileNotFoundError(
            f"SFT merged checkpoint not found at {sft_merged_path}. "
            "Run train_sft.py first."
        )

    print(f"[dpo] Loading policy from SFT merged: {sft_merged_path}")
    base_model = load_policy_base_model(model_id=sft_merged_path)
    policy     = apply_lora(base_model)
    policy.to(device)
    policy.train()
    print_model_stats(policy, "Policy + LoRA (DPO)")

    # ── Dataset ───────────────────────────────────────────────────────────
    raw_train, raw_test = load_hh_rlhf()
    train_examples      = parse_dataset(raw_train)
    test_examples       = parse_dataset(raw_test)

    train_ds = DPODataset(train_examples, policy_tok, cfg.tokenizer.max_seq_len)
    test_ds  = DPODataset(test_examples,  policy_tok, cfg.tokenizer.max_seq_len)

    train_loader = DataLoader(
        train_ds, batch_size=dpo_cfg.batch_size,
        shuffle=True, num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=dpo_cfg.batch_size,
        shuffle=False, num_workers=0,
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
        for batch in train_loader:
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
    merged_path = os.path.join(dpo_cfg.save_dir, "merged")
    merge_and_save(policy, merged_path, policy_tok)
    print(f"[dpo] Saved to {dpo_cfg.save_dir}")


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

    if args.method == "ppo":
        train_ppo()
    elif args.method == "dpo":
        train_dpo()
    elif args.method == "grpo":
        raise NotImplementedError("GRPO: implement in Phase C5 (alignment/grpo.py)")
    elif args.method == "rlvr":
        raise NotImplementedError("RLVR: implement in Phase C6 (alignment/rlvr.py)")
