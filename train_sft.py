"""
train_sft.py — Supervised Fine-Tuning Warm-Up (Task C2)
========================================================

PSEUDOCODE:
────────────
1. Load policy tokenizer + base model
2. Apply LoRA → π_θ
3. Build SFT dataloaders (chosen responses only)
4. For each (input_ids, attention_mask, labels) in train_loader:
   a. Forward pass: logits = model(input_ids, attention_mask).logits
                   shape: (batch, seq_len, vocab_size)
   b. Compute CE loss: CrossEntropyLoss(ignore_index=-100)
      Only response tokens contribute (prompt + pad tokens masked to -100)
   c. loss / grad_accum_steps   ← scale for gradient accumulation
   d. loss.backward()
   e. Every grad_accum_steps: clip_grad_norm, optimizer.step(), scheduler.step()
5. Every eval_every steps: compute perplexity on held-out set
   PPL = exp(mean CE loss on test set)
   If PPL < 5: label masking bug — you're training on prompt tokens too!
6. After training: merge LoRA → base weights (merge_and_save)
   → saves 'checkpoints/sft_merged/' which becomes π_ref

Key masking invariant:
    labels[prompt_tokens] = -100
    labels[pad_tokens]    = -100
    labels[response_tokens] = actual token ids
    PyTorch CrossEntropyLoss(ignore_index=-100) skips -100 positions.
    → gradient flows ONLY through response token predictions.
"""

import os
import math
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from config import cfg
from data.hh_rlhf import load_hh_rlhf, parse_dataset, SFTDataset
from model.loader import (
    load_policy_tokenizer,
    load_policy_base_model,
    print_model_stats,
)
from model.lora_setup import apply_lora, merge_and_save

from torch.utils.data import DataLoader


def compute_sft_loss(
    logits: torch.Tensor,   # shape: (batch, seq_len, vocab_size)
    labels: torch.Tensor,   # shape: (batch, seq_len), -100 for prompt+pad
) -> torch.Tensor:
    """
    Cross-entropy loss over response tokens only.

    PyTorch CrossEntropyLoss with ignore_index=-100 automatically skips
    positions where labels == -100. We reshape for the loss function:
        logits: (batch × seq_len, vocab_size)
        labels: (batch × seq_len,)

    Returns a scalar loss.
    """
    vocab_size = logits.size(-1)

    # Shift for causal LM: predict token t+1 from token t
    # logits[:, :-1, :]  → predictions at positions 0 to T-2
    # labels[:, 1:]      → targets at positions 1 to T-1
    # This ensures the model is trained to predict the NEXT token, not current.
    shift_logits = logits[:, :-1, :].contiguous()  # shape: (batch, seq_len-1, vocab)
    shift_labels = labels[:, 1:].contiguous()       # shape: (batch, seq_len-1)

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(
        shift_logits.view(-1, vocab_size),  # shape: (batch*(seq_len-1), vocab)
        shift_labels.view(-1),              # shape: (batch*(seq_len-1),)
    )
    return loss  # scalar


def evaluate_perplexity(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> float:
    """
    Compute perplexity on a held-out set (greedy, no sampling).

    PPL = exp(mean CE loss over all response tokens in the held-out set)

    Interpretation:
        PPL ≈ 50–200  : normal range for a language model on this task
        PPL < 5       : likely leaking prompt tokens into the loss (bug!)
        PPL > 500     : model is not learning / wrong labels
    """
    model.eval()
    total_loss   = 0.0
    total_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break

            input_ids      = batch["input_ids"].to(device)      # (B, L)
            attention_mask = batch["attention_mask"].to(device)  # (B, L)
            labels         = batch["labels"].to(device)          # (B, L)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = compute_sft_loss(outputs.logits, labels)

            total_loss    += loss.item()
            total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    ppl      = math.exp(avg_loss)
    model.train()
    return ppl


def generate_samples(
    model: nn.Module,
    tokenizer,
    prompts: list,
    max_new_tokens: int = 100,
    device: torch.device = None,
) -> list:
    """
    Generate greedy responses for a list of raw prompt strings.
    Used for manual quality inspection after SFT.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    responses = []
    with torch.no_grad():
        for prompt in prompts:
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=cfg.tokenizer.max_seq_len - max_new_tokens,
            ).to(device)

            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,        # greedy decoding for reproducibility
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            # Decode only the newly generated tokens (not the prompt)
            new_tokens = out[0][enc["input_ids"].shape[1]:]
            response   = tokenizer.decode(new_tokens, skip_special_tokens=True)
            responses.append(response.strip())

    model.train()
    return responses


def train_sft(cfg_override=None):
    """
    Main SFT training loop.

    Steps:
        1. Setup
        2. For each batch: forward → loss (response tokens only) → backward
        3. Gradient accumulation every `grad_accum_steps`
        4. Eval perplexity every `eval_every` steps
        5. Post-training: generate 5 sample responses
        6. Merge LoRA → save π_ref checkpoint
    """
    sft_cfg = cfg.sft if cfg_override is None else cfg_override
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sft] Device: {device}")

    # ── 1. Load tokenizer ────────────────────────────────────────────────
    tokenizer = load_policy_tokenizer()
    print("[sft] Tokenizer loaded")

    # ── 2. Load dataset ──────────────────────────────────────────────────
    raw_train, raw_test = load_hh_rlhf()
    train_examples      = parse_dataset(raw_train)
    test_examples       = parse_dataset(raw_test)

    train_ds = SFTDataset(train_examples, tokenizer, cfg.tokenizer.max_seq_len)
    test_ds  = SFTDataset(test_examples,  tokenizer, cfg.tokenizer.max_seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=sft_cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=sft_cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )
    print(f"[sft] Train batches: {len(train_loader):,} | Test batches: {len(test_loader):,}")

    # ── 3. Load model + apply LoRA ───────────────────────────────────────
    base_model = load_policy_base_model()
    model      = apply_lora(base_model)     # wraps with LoRA → π_θ
    model.to(device)
    model.train()
    print_model_stats(model, "Policy + LoRA (SFT)")

    # ── 4. Optimiser + scheduler ─────────────────────────────────────────
    # Only optimise LoRA parameters (base model weights are frozen by PEFT)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=sft_cfg.learning_rate, weight_decay=0.01)

    # Total optimiser steps (NOT gradient accumulation steps)
    total_optimizer_steps = (
        len(train_loader) * sft_cfg.num_epochs // sft_cfg.grad_accum_steps
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=sft_cfg.warmup_steps,
        num_training_steps=total_optimizer_steps,
    )
    print(f"[sft] Total optimizer steps: {total_optimizer_steps:,}")

    # ── 5. Training loop ─────────────────────────────────────────────────
    global_step      = 0   # counts optimizer steps (after accumulation)
    raw_step         = 0   # counts dataloader steps (before accumulation)
    running_loss     = 0.0
    running_n_tokens = 0

    os.makedirs(sft_cfg.save_dir, exist_ok=True)

    print("\n[sft] Starting training...")
    t0 = time.time()

    for epoch in range(sft_cfg.num_epochs):
        for batch in train_loader:
            raw_step += 1

            input_ids      = batch["input_ids"].to(device)       # (B, L)
            attention_mask = batch["attention_mask"].to(device)   # (B, L)
            labels         = batch["labels"].to(device)           # (B, L)

            # ── Forward pass ────────────────────────────────────────────
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # logits: shape (batch, seq_len, vocab_size)

            loss = compute_sft_loss(outputs.logits, labels)

            # ── Scale for gradient accumulation ─────────────────────────
            # Dividing by grad_accum_steps makes the gradient equal to the
            # mean over the effective (larger) batch, not just the micro-batch.
            scaled_loss = loss / sft_cfg.grad_accum_steps
            scaled_loss.backward()

            running_loss += loss.item()

            # ── Optimiser step every grad_accum_steps ───────────────────
            if raw_step % sft_cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, sft_cfg.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # ── Logging ─────────────────────────────────────────────
                if global_step % sft_cfg.log_every == 0:
                    avg_loss = running_loss / (sft_cfg.log_every * sft_cfg.grad_accum_steps)
                    ppl      = math.exp(avg_loss)
                    elapsed  = time.time() - t0
                    print(
                        f"[sft] step={global_step:4d} | "
                        f"loss={avg_loss:.4f} | ppl={ppl:.1f} | "
                        f"lr={scheduler.get_last_lr()[0]:.2e} | "
                        f"elapsed={elapsed:.0f}s"
                    )
                    running_loss = 0.0

                # ── Evaluation ──────────────────────────────────────────
                if global_step % sft_cfg.eval_every == 0:
                    test_ppl = evaluate_perplexity(model, test_loader, device)
                    print(f"[sft] ── EVAL @ step {global_step}: test PPL = {test_ppl:.2f}")
                    if test_ppl < 5.0:
                        print("[sft] ⚠ PPL < 5 — check label masking! "
                              "You may be computing loss over prompt tokens.")

    print(f"\n[sft] Training complete in {time.time()-t0:.0f}s")

    # ── 6. Generate 5 sample responses (sanity check) ───────────────────
    print("\n[sft] Generating 5 sample responses (greedy)...")
    sample_prompts = [ex["prompt"] for ex in test_examples[:5]]
    responses = generate_samples(model, tokenizer, sample_prompts)
    for i, (p, r) in enumerate(zip(sample_prompts, responses)):
        print(f"\n── Sample {i+1} ──")
        print(f"PROMPT (last 100 chars): ...{p[-100:]}")
        print(f"RESPONSE: {r[:200]}")

    # ── 7. Save LoRA adapter checkpoint ─────────────────────────────────
    print(f"\n[sft] Saving LoRA adapter to {sft_cfg.save_dir}...")
    model.save_pretrained(sft_cfg.save_dir)
    tokenizer.save_pretrained(sft_cfg.save_dir)
    print("[sft] LoRA adapter saved.")

    # ── 8. Merge LoRA → base weights → save as π_ref ────────────────────
    # This merged checkpoint is used as:
    #   (a) π_ref (frozen KL anchor) in all downstream RL training
    #   (b) Starting point for applying fresh LoRA in PPO/GRPO
    print(f"\n[sft] Merging LoRA into base weights → {sft_cfg.merged_save_dir}")
    merged = merge_and_save(model, sft_cfg.merged_save_dir, tokenizer)
    print("[sft] Done. π_ref checkpoint ready at:", sft_cfg.merged_save_dir)

    return merged, tokenizer


if __name__ == "__main__":
    import random
    import numpy as np

    # Reproducibility
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    train_sft()
