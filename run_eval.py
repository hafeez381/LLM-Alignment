"""
run_eval.py — Task C8: Full Evaluation Driver
=============================================

Run this AFTER all alignment methods have been trained:
    python train_sft.py
    python train_rm.py
    python train_rl.py --method ppo
    python train_rl.py --method dpo
    python train_rl.py --method grpo
    python train_rl.py --method rlvr  (optional)

Usage:
    python run_eval.py

This script:
    1. Loads all trained checkpoints
    2. Computes win-rate vs SFT for each method
    3. Computes KL(π_θ || π_ref) for each method
    4. Builds the 5-prompt sample response table
    5. Reads timing data from a JSON log (written by train_rl.py)
    6. Saves everything to plots/ and prints the full report

CHECKPOINT LAYOUT expected:
    checkpoints/sft_merged/         ← SFT policy = π_ref baseline
    checkpoints/rm/                 ← frozen RM
    checkpoints/ppo/merged/         ← PPO-trained policy (merged)
    checkpoints/dpo/merged/         ← DPO-trained policy (merged)
    checkpoints/grpo/merged/        ← GRPO-trained policy (merged)
    checkpoints/rlvr/merged/        ← RLVR-trained policy (merged, optional)

Methods with missing checkpoints are gracefully skipped.
"""

import os
import sys
import json
import torch

from config import cfg
from model.loader import (
    load_policy_tokenizer,
    load_rm_tokenizer,
    load_policy_base_model,
    load_reward_backbone,
    print_model_stats,
)
from model.lora_setup import freeze_model, apply_lora
from model.reward_head import RewardModel
from data.hh_rlhf import load_hh_rlhf, parse_dataset
from eval import (
    ResourceTracker,
    run_full_eval,
    compute_win_rate,
    compute_kl,
    build_sample_table,
    print_sample_table,
)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint paths for each method
# ─────────────────────────────────────────────────────────────────────────────
# We load the MERGED checkpoints for evaluation.
# A merged checkpoint is a plain (non-PEFT) model that can be loaded with
# AutoModelForCausalLM.from_pretrained() directly.
CHECKPOINT_PATHS = {
    "SFT":  cfg.sft.merged_save_dir,                    # baseline
    "PPO":  os.path.join(cfg.ppo.save_dir,  "merged"),
    "DPO":  os.path.join(cfg.dpo.save_dir,  "merged"),
    "GRPO": os.path.join(cfg.grpo.save_dir, "merged"),
    "RLVR": os.path.join(cfg.rlvr.save_dir, "merged"),
}

# Optional: if train_rl.py wrote resource logs, read them here
RESOURCE_LOG_PATH = "plots/resource_log.json"


def _load_merged_policy(ckpt_path: str, device: torch.device):
    """
    Load a merged (non-PEFT) policy checkpoint.

    For EVALUATION we load WITHOUT applying LoRA — the merged checkpoint
    already has the adapter weights baked in.  We freeze all parameters
    since we only need inference.
    """
    if not os.path.exists(ckpt_path):
        return None
    print(f"[run_eval] Loading {ckpt_path}...")
    model = load_policy_base_model(model_id=ckpt_path)
    model = freeze_model(model)
    model.to(device).eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run_eval] Device: {device}")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    # ── Tokenizers ────────────────────────────────────────────────────────
    policy_tok = load_policy_tokenizer()
    rm_tok     = load_rm_tokenizer()

    # ── Frozen RM ─────────────────────────────────────────────────────────
    rm_path  = cfg.rm.save_dir
    backbone = load_reward_backbone(model_id=rm_path)
    rm       = RewardModel(backbone)
    rm.to(device).eval()
    for p in rm.parameters():
        p.requires_grad_(False)
    print("[run_eval] RM loaded and frozen")

    # ── Dataset ───────────────────────────────────────────────────────────
    raw_train, raw_test = load_hh_rlhf()
    test_examples = parse_dataset(raw_test)
    print(f"[run_eval] Test set: {len(test_examples):,} examples")

    # ── Load all method checkpoints ───────────────────────────────────────
    method_models = {}
    sft_model     = None

    for method, ckpt_path in CHECKPOINT_PATHS.items():
        model = _load_merged_policy(ckpt_path, device)
        if model is not None:
            method_models[method] = model
            if method == "SFT":
                sft_model = model
            print(f"  ✓ {method} loaded from {ckpt_path}")
        else:
            print(f"  ✗ {method} checkpoint not found at {ckpt_path} — skipping")

    if sft_model is None:
        raise FileNotFoundError(
            "SFT checkpoint not found. Run train_sft.py first."
        )

    # ── Resource tracker: read from log if it exists ──────────────────────
    tracker = ResourceTracker()
    if os.path.exists(RESOURCE_LOG_PATH):
        with open(RESOURCE_LOG_PATH) as f:
            resource_data = json.load(f)
        for method, rec in resource_data.items():
            tracker.records[method] = rec
        print(f"[run_eval] Loaded resource log from {RESOURCE_LOG_PATH}")
    else:
        print(f"[run_eval] No resource log found at {RESOURCE_LOG_PATH}. "
              f"Resource table will be empty.")
        print(f"           To populate it, add ResourceTracker calls to train_rl.py.")

    # ── Run full evaluation ───────────────────────────────────────────────
    print("\n[run_eval] Starting C8 evaluation...")
    results = run_full_eval(
        method_models={k: v for k, v in method_models.items() if k != "SFT"},
        sft_model=sft_model,
        rm=rm, rm_tok=rm_tok, policy_tok=policy_tok,
        test_examples=test_examples,
        resource_tracker=tracker,
        n_eval=200,
        n_sample_prompts=5,
        output_dir="plots",
        device=device,
    )

    # ── Analysis: print the three required discussion points ──────────────
    print("\n" + "="*70)
    print(" C8 ANALYSIS DISCUSSION POINTS (fill in with your observations)")
    print("="*70)

    print("""
1. REWARD VS. QUALITY GAP
   - Check the sample table for responses where RM score is high but
     the text is repetitive, sycophantic, or factually wrong.
   - High-RM but low-quality responses reveal RM 'gaming' — the policy
     learned surface patterns (e.g. polite openings) that the RM rewards
     regardless of content.

2. ONLINE vs. OFFLINE COMPARISON
   - PPO/GRPO make `prompts_per_step × K` RM queries per step.
     PPO:  200 steps × 16 prompts × 1 sample  = 3,200 RM calls
     GRPO: 200 steps × 16 prompts × 4 samples = 12,800 RM calls
   - DPO uses the fixed preference dataset: 0 RM calls during training.
     (RM used only at eval time.)
   - Compare win-rates per RM call budget to assess sample efficiency.

3. KL AS ALIGNMENT MEASURE
   - A model with KL ≈ 0 hasn't learned anything new vs. SFT.
   - A model with very large KL may have hacked the RM.
   - Look for the 'sweet spot': moderate KL + high win-rate.
   - Cross-reference with the sample table to see if large-KL responses
     are genuinely better or just noisier.
""")
    print("="*70)
    print(f"\n[run_eval] All results saved to plots/")


if __name__ == "__main__":
    main()
