"""
model/lora_setup.py — LoRA Application and Reference Model Utilities
=====================================================================

PSEUDOCODE:
────────────
apply_lora(base_model, lora_cfg)
  → peft.LoraConfig(r, lora_alpha, target_modules, ...)
  → peft_model = get_peft_model(base_model, lora_config)
  → Print trainable parameter table
  → Return peft_model  (π_θ, trainable)

freeze_model(model)
  → model.eval()
  → for param in model.parameters(): param.requires_grad_(False)
  → Return model

reference_model_ctx(peft_model)
  → Context manager: temporarily disables LoRA adapters
  → Inside context: model behaves as π_ref (the merged SFT base)
  → Outside context: model behaves as π_θ (with LoRA active)
  → WHY: avoids loading a SECOND copy of the model just for KL computation
         — huge VRAM saving on T4

merge_and_save(peft_model, save_path)
  → peft_model.merge_and_unload()  (bakes LoRA delta into weights)
  → Save merged model to save_path
  → This checkpoint becomes π_ref for all downstream RL methods
"""

from contextlib import contextmanager
from typing import Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model

from config import cfg


# ─────────────────────────────────────────────────────────────────────────────
# Apply LoRA adapters
# ─────────────────────────────────────────────────────────────────────────────

def apply_lora(
    base_model: torch.nn.Module,
    lora_cfg=None,
) -> torch.nn.Module:
    """
    Wrap a base model with PEFT LoRA adapters to create π_θ.

    The LoRA reparameterisation replaces each targeted weight matrix W₀:
        W = W₀ + (α/r) · B · A
    where W₀ is frozen, and only B ∈ ℝ^{d×r} and A ∈ ℝ^{r×d} are trainable.

    With r=8 and targeting only q_proj and v_proj:
        SmolLM2-360M normally: ~360M params
        After LoRA:  trainable ≈ 2 × 2 × d_model × r × n_layers
                                ≈ 2 × 2 × 960 × 8 × 32 ≈ 983,040  (~0.27%)

    Parameters
    ----------
    base_model : the pretrained (or SFT-merged) model to adapt
    lora_cfg   : LoRAConfig from config.py; uses default if None
    """
    if lora_cfg is None:
        lora_cfg = cfg.lora

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        target_modules=lora_cfg.target_modules,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        # init_lora_weights='gaussian' — use default (kaiming) for stability
    )

    peft_model = get_peft_model(base_model, peft_config)

    print(f"[lora] LoRA applied: r={lora_cfg.r}, α={lora_cfg.lora_alpha}, "
          f"modules={lora_cfg.target_modules}")
    peft_model.print_trainable_parameters()

    return peft_model


# ─────────────────────────────────────────────────────────────────────────────
# Freeze all parameters
# ─────────────────────────────────────────────────────────────────────────────

def freeze_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Set requires_grad=False on ALL parameters and switch to eval mode.

    Used to freeze:
        - The reward model r_ψ  (must not update during RL)
        - π_ref when loaded as a separate copy
        - The value head's base backbone if we want frozen features
    """
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Reference model context manager (no separate copy needed!)
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def reference_model_ctx(peft_model: torch.nn.Module):
    """
    Deploying two discrete copies of a language model to compute the 
    KL divergence penalty drastically increases the required VRAM footprint.

    Context manager temporarily disables LoRA adapters on π_θ.

    While inside this context, the model behaves as π_ref (the SFT-merged
    base weights, without any of the RL-trained LoRA delta).

    Usage:
        with reference_model_ctx(policy) as ref:
            ref_logits = ref(input_ids=ids, attention_mask=mask).logits

    WHY this approach vs. loading a separate π_ref model?
        A separate π_ref copy would cost ~720 MB extra VRAM on T4.
        Disabling adapters is zero-cost: same weights, adapter matrices set
        aside. The context manager restores the adapters atomically on exit,
        so no risk of forgetting to re-enable them.

    The underlying base model must be the SFT-MERGED checkpoint
    (i.e., trained via merge_and_save() after SFT). If it's the raw pretrained
    checkpoint, disable_adapter() gives you pretrained, not SFT — wrong π_ref!
    """
    # Store training state
    training_was = peft_model.training
    try:
        peft_model.eval()
        # Temporarily remove the LoRA delta (B.A)
        with peft_model.disable_adapter():
            # Inside here: model == π_ref (SFT base, LoRA delta disabled)
            with torch.no_grad():
                yield peft_model
    finally:
        # Restore training mode that was active before entering context
        if training_was:
            peft_model.train()


# ─────────────────────────────────────────────────────────────────────────────
# Merge LoRA into base weights and save
# ─────────────────────────────────────────────────────────────────────────────

def merge_and_save(
    peft_model: torch.nn.Module,
    save_path: str,
    tokenizer=None,
) -> torch.nn.Module:
    """
    Bake the LoRA delta (B·A) into the base weight W₀:
        W_merged = W₀ + (α/r) · B · A

    After merging, the model is a standard (non-PEFT) model.
    This merged checkpoint is saved as π_ref — the frozen anchor for all
    downstream PPO/GRPO/DPO/RLVR training.

    WHY merge?
        For π_ref we want a single, loadable checkpoint that we can:
        (a) load frozen as a standalone model
        (b) OR load and apply fresh LoRA on top (for π_θ in PPO/GRPO)
        Keeping unmerged LoRA adapters would require the original base model
        to always be present, complicating checkpoint management.

    Returns the merged (non-PEFT) model.
    """
    import os
    os.makedirs(save_path, exist_ok=True)

    print(f"[lora] Merging LoRA adapters into base weights...")
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(save_path)
    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)
    print(f"[lora] Merged model saved to: {save_path}")
    print(f"       This checkpoint is π_ref for all downstream RL training.")
    return merged_model
