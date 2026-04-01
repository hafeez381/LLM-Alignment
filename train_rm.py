"""
train_rm.py — Reward Model Fine-Tuning (Task C1)
=================================================

PSEUDOCODE:
────────────
1. Load Llama-3.2-1B-Instruct backbone via AutoModelForSequenceClassification
   (num_labels=1 → scalar head; right-padded Llama tokenizer)
2. Wrap in RewardModel class (thin PyTorch Module wrapper)
3. Optionally apply LoRA to backbone — or just train the linear head
   (spec says "alternatively, just train the linear projection layer")
   → We train the head only for simplicity; LoRA variant in TODO comment
4. Build RM dataloaders: each batch = (chosen_ids, chosen_mask, rej_ids, rej_mask)
5. For each batch:
   a. Forward chosen  → r⁺ = rm(chosen_ids, chosen_mask)   shape: (batch,)
   b. Forward rejected → r⁻ = rm(rej_ids,   rej_mask)      shape: (batch,)
   c. L_RM = -E[log σ(r⁺ - r⁻)] + λ · E[(r⁺)² + (r⁻)²]
   d. Compute preference accuracy = P(r⁺ > r⁻)
   e. Backward + optimiser step
6. Evaluate every eval_every steps on held-out test set
7. Save frozen RM checkpoint → used in PPO/GRPO evaluation

CRITICAL NOTE on r⁺ and r⁻ computation:
    The RM backbone is passed through TWICE per batch (once for chosen,
    once for rejected). There is NO weight sharing between r⁺ and r⁻
    computations — both use the SAME frozen (or LoRA-adapted) backbone.
    This is correct: r_ψ(x, y⁺) and r_ψ(x, y⁻) are two evaluations of
    the same model on different inputs.
"""

import os
import sys
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.stdout.reconfigure(line_buffering=True)

from config import cfg
from data.hh_rlhf import load_hh_rlhf, parse_dataset, RMDataset
from model.loader import load_rm_tokenizer, load_reward_backbone, print_model_stats
from model.lora_setup import freeze_model
from model.reward_head import RewardModel, compute_rm_loss, compute_preference_accuracy


def evaluate_rm(
    rm: RewardModel,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 100,
):
    """
    Evaluate RM on held-out test set.

    Reports:
        - Average preference loss
        - Preference accuracy P(r⁺ > r⁻)
        - Histograms of r⁺ and r⁻ distributions (as min/mean/max stats)
    """
    rm.eval()
    total_loss     = 0.0
    total_acc      = 0.0
    total_batches  = 0
    all_r_pos      = []
    all_r_neg      = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break

            chosen_ids   = batch["chosen_input_ids"].to(device)        # (B, L)
            chosen_mask  = batch["chosen_attention_mask"].to(device)   # (B, L)
            rej_ids      = batch["rejected_input_ids"].to(device)      # (B, L)
            rej_mask     = batch["rejected_attention_mask"].to(device)  # (B, L)

            r_pos = rm(chosen_ids, chosen_mask)   # shape: (batch,)
            r_neg = rm(rej_ids,    rej_mask)       # shape: (batch,)

            loss, _, _ = compute_rm_loss(r_pos, r_neg)
            acc        = compute_preference_accuracy(r_pos, r_neg)

            total_loss    += loss.item()
            total_acc     += acc
            total_batches += 1
            all_r_pos.extend(r_pos.cpu().float().tolist())
            all_r_neg.extend(r_neg.cpu().float().tolist())

    avg_loss = total_loss / max(total_batches, 1)
    avg_acc  = total_acc  / max(total_batches, 1)

    r_pos_t = torch.tensor(all_r_pos)
    r_neg_t = torch.tensor(all_r_neg)

    print(f"[rm-eval] loss={avg_loss:.4f} | acc={avg_acc:.3f}")
    print(f"[rm-eval] r⁺ distribution: min={r_pos_t.min():.2f} "
          f"mean={r_pos_t.mean():.2f} max={r_pos_t.max():.2f}")
    print(f"[rm-eval] r⁻ distribution: min={r_neg_t.min():.2f} "
          f"mean={r_neg_t.mean():.2f} max={r_neg_t.max():.2f}")
    print(f"[rm-eval] r⁺ - r⁻ mean = {(r_pos_t - r_neg_t).mean():.2f}  "
          f"(positive = model prefers chosen ✓)")

    rm.train()
    return avg_loss, avg_acc


def train_rm(cfg_override=None):
    """
    Main reward model training loop.

    Architecture decision: we fine-tune ONLY the linear classification head
    (not the backbone layers) for speed and to prevent overfitting on the
    ~42k HH-RLHF training pairs.

    To train backbone layers too (adds ~2M LoRA params):
        # Uncomment the lines marked TODO below and set train_head_only=False
    """
    import os
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    rm_cfg = cfg.rm if cfg_override is None else cfg_override
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[rm] Device: {device}")

    # ── 1. Load RM tokenizer and backbone ───────────────────────────────
    rm_tokenizer = load_rm_tokenizer()
    backbone     = load_reward_backbone()  # Llama-1B with scalar head
    rm           = RewardModel(backbone)
    rm.to(device)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── Decide which parameters to train ────────────────────────────────
    # Option A (default): train only the scalar classification head.
    #   Very fast, low memory, sufficient for ≥60% accuracy target.
    train_head_only = False

    if train_head_only:
        # Freeze ALL backbone layers; only score head is trainable
        for name, param in rm.named_parameters():
            if "score" in name:  # 'score' is HuggingFace's name for the head
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        print("[rm] Training HEAD ONLY (backbone frozen)")
    else:
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            target_modules=cfg.lora.target_modules,
            lora_dropout=cfg.lora.lora_dropout,
            bias=cfg.lora.bias,
        )
        rm.backbone = get_peft_model(rm.backbone, lora_config)
        rm.backbone.print_trainable_parameters()

        # Ensure score head stays trainable (PEFT can reset it)
        for name, param in rm.backbone.named_parameters():
            if "score" in name:
                param.requires_grad_(True)

    rm.train()
    print_model_stats(rm, "RewardModel")

    # ── 2. Load dataset ──────────────────────────────────────────────────
    raw_train, raw_test = load_hh_rlhf()
    train_examples      = parse_dataset(raw_train)
    test_examples       = parse_dataset(raw_test)

    train_ds = RMDataset(train_examples, rm_tokenizer, cfg.tokenizer.max_seq_len)
    test_ds  = RMDataset(test_examples,  rm_tokenizer, cfg.tokenizer.max_seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=rm_cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=rm_cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )
    print(f"[rm] Train batches: {len(train_loader):,} | Test: {len(test_loader):,}")

    # ── 3. Optimiser ─────────────────────────────────────────────────────
    trainable_params = [p for p in rm.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=rm_cfg.learning_rate, weight_decay=0.01)

    total_steps = len(train_loader) * rm_cfg.num_epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=rm_cfg.warmup_steps,
        num_training_steps=total_steps,
    )

    # ── 4. Training loop ─────────────────────────────────────────────────
    global_step  = 0
    running_loss = 0.0
    running_acc  = 0.0

    os.makedirs(rm_cfg.save_dir, exist_ok=True)
    print("\n[rm] Starting RM training...")
    t0 = time.time()

    for epoch in range(rm_cfg.num_epochs):
        for batch in tqdm(train_loader, desc=f"RM epoch {epoch+1}/{rm_cfg.num_epochs}",
                          unit="batch", dynamic_ncols=True):
            global_step += 1

            # Move to device
            chosen_ids   = batch["chosen_input_ids"].to(device)         # (B, L)
            chosen_mask  = batch["chosen_attention_mask"].to(device)    # (B, L)
            rej_ids      = batch["rejected_input_ids"].to(device)       # (B, L)
            rej_mask     = batch["rejected_attention_mask"].to(device)  # (B, L)

            # ── Forward chosen ───────────────────────────────────────────
            # WHY two separate forward passes?
            #   The RM scores x⊕y⁺ and x⊕y⁻ as separate sequences.
            #   They share weights (same RM) but differ in input.
            #   We could stack them in one batch for efficiency, but keeping
            #   them separate is clearer and avoids a shape bug from stacking.
            r_pos = rm(chosen_ids,  chosen_mask)   # shape: (batch,)

            # ── Forward rejected ─────────────────────────────────────────
            r_neg = rm(rej_ids, rej_mask)           # shape: (batch,)

            # ── Loss ─────────────────────────────────────────────────────
            # L_RM = -E[log σ(r⁺ - r⁻)] + λ·E[(r⁺)² + (r⁻)²]
            loss, pref_loss, reg_loss = compute_rm_loss(
                r_pos, r_neg, rm_cfg.reg_lambda
            )                                      # all scalars

            # ── Backward ─────────────────────────────────────────────────
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, rm_cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()

            # ── Logging ──────────────────────────────────────────────────
            acc = compute_preference_accuracy(r_pos.detach(), r_neg.detach())
            running_loss += loss.item()
            running_acc  += acc

            if global_step % rm_cfg.log_every == 0:
                avg_loss = running_loss / rm_cfg.log_every
                avg_acc  = running_acc  / rm_cfg.log_every
                elapsed  = time.time() - t0
                print(
                    f"[rm] step={global_step:4d} | "
                    f"loss={avg_loss:.4f} | "
                    f"pref_loss={pref_loss.item():.4f} | "
                    f"reg_loss={reg_loss.item():.5f} | "
                    f"acc={avg_acc:.3f} | "
                    f"elapsed={elapsed:.0f}s"
                )
                running_loss = 0.0
                running_acc  = 0.0

            # ── Evaluation ───────────────────────────────────────────────
            if global_step % rm_cfg.eval_every == 0:
                print(f"\n[rm] ── EVAL @ step {global_step} ──")
                test_loss, test_acc = evaluate_rm(rm, test_loader, device)
                if test_acc < rm_cfg.min_accuracy_threshold:
                    print(
                        f"[rm] ⚠ Accuracy {test_acc:.3f} < "
                        f"{rm_cfg.min_accuracy_threshold:.2f} threshold. "
                        f"Check: (1) pad_token_id in RM config, "
                        f"(2) right-padding in RM tokenizer, "
                        f"(3) prompt+response concatenated before tokenising."
                    )
                print()

    print(f"\n[rm] Training complete in {time.time()-t0:.0f}s")

    # ── 5. Final evaluation ──────────────────────────────────────────────
    print("\n[rm] Final evaluation on full test set...")
    final_loss, final_acc = evaluate_rm(rm, test_loader, device, max_batches=999)

    if final_acc >= rm_cfg.min_accuracy_threshold:
        print(f"[rm] ✓ Preference accuracy {final_acc:.3f} ≥ "
              f"{rm_cfg.min_accuracy_threshold:.2f} — RM ready for RL training!")
    else:
        print(f"[rm] ✗ Accuracy {final_acc:.3f} below threshold. "
              f"Do NOT proceed to RL training until RM is fixed.")

    # ── 6. Save frozen RM checkpoint ─────────────────────────────────────
    # Freeze all parameters BEFORE saving — the RM must NOT be updated
    # during any subsequent RL training. Saving in frozen state makes
    # this explicit and prevents accidental gradient flow.
    rm = freeze_model(rm)  # freeze_model sets requires_grad=False + eval()
    rm.backbone.save_pretrained(rm_cfg.save_dir)
    rm_tokenizer.save_pretrained(rm_cfg.save_dir)
    # Save the full wrapper state dict too (for easy reloading)
    torch.save(rm.state_dict(), os.path.join(rm_cfg.save_dir, "rm_state_dict.pt"))
    print(f"[rm] Frozen RM saved to: {rm_cfg.save_dir}")
    print(f"[rm] Final accuracy: {final_acc:.3f}")

    # ── Save reward distribution histograms ──────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")          # headless backend — no display needed on Colab
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs("plots", exist_ok=True)

    # Collect r⁺ and r⁻ on the full test set
    rm.eval()
    all_r_pos, all_r_neg = [], []
    with torch.no_grad():
        for batch in test_loader:
            r_pos = rm(batch["chosen_input_ids"].to(device),
                    batch["chosen_attention_mask"].to(device)).cpu().float().tolist()
            r_neg = rm(batch["rejected_input_ids"].to(device),
                    batch["rejected_attention_mask"].to(device)).cpu().float().tolist()
            all_r_pos.extend(r_pos)
            all_r_neg.extend(r_neg)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(
        min(min(all_r_pos), min(all_r_neg)),
        max(max(all_r_pos), max(all_r_neg)),
        40
    )
    ax.hist(all_r_pos, bins=bins, alpha=0.6, label=r"$r^+$ (chosen)",   color="steelblue")
    ax.hist(all_r_neg, bins=bins, alpha=0.6, label=r"$r^-$ (rejected)", color="tomato")
    ax.axvline(np.mean(all_r_pos), color="steelblue", linestyle="--", linewidth=1.5,
            label=f"mean $r^+$ = {np.mean(all_r_pos):.2f}")
    ax.axvline(np.mean(all_r_neg), color="tomato",    linestyle="--", linewidth=1.5,
            label=f"mean $r^-$ = {np.mean(all_r_neg):.2f}")
    ax.set_xlabel("Reward score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Reward Model: Score Distribution on Test Set", fontsize=13)
    ax.legend()
    fig.tight_layout()
    fig.savefig("plots/rm_reward_distribution.png", dpi=150)
    plt.close(fig)
    print("[rm] Reward distribution plot saved → plots/rm_reward_distribution.png")

    return rm, rm_tokenizer


if __name__ == "__main__":
    import random
    import numpy as np

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    train_rm()
