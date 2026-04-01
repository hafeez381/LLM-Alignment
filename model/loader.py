"""
model/loader.py — Model and Tokenizer Loading Utilities
========================================================

PSEUDOCODE:
────────────
load_policy_tokenizer(model_id)
  → Load SmolLM2 tokenizer
  → Set pad_token = eos_token (model has no dedicated pad token)
  → Set padding_side = 'left'  (for autoregressive generation)
  → Return tokenizer

load_rm_tokenizer(model_id)
  → Load Llama-3.2 tokenizer
  → Set pad_token = eos_token
  → Set padding_side = 'right'  (so SequenceClassifier finds last real token)
  → Return tokenizer

load_policy_base_model(model_id, dtype)
  → AutoModelForCausalLM.from_pretrained(..., torch_dtype=dtype)
  → Enable gradient checkpointing (saves activations, trades compute for memory)
  → Enable input require_grads (required for PEFT + grad checkpointing combo)
  → Return model (base, no LoRA yet — LoRA applied in lora_setup.py)

load_reward_backbone(model_id, load_in_8bit, dtype)
  → AutoModelForSequenceClassification.from_pretrained(
        ..., num_labels=1,          ← scalar head r(x,y) ∈ ℝ
        load_in_8bit=load_in_8bit,  ← halves VRAM for frozen RM
    )
  → Set model.config.pad_token_id correctly
  → Return model

print_model_stats(model, name)
  → Count trainable and total parameters
  → Report torch.cuda.memory_allocated() in GB
"""

import os
from typing import Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)

from config import cfg


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizers
# ─────────────────────────────────────────────────────────────────────────────

def load_policy_tokenizer(
    model_id: Optional[str] = None,
) -> AutoTokenizer:
    """
    Load the SmolLM2 tokenizer configured for the POLICY model.

    Key settings:
    ┌──────────────────┬────────────────────────────────────────────────────┐
    │ pad_token        │ eos_token (SmolLM2 has no dedicated pad token)     │
    │ padding_side     │ 'left' — see WHY below                             │
    └──────────────────┴────────────────────────────────────────────────────┘

    WHY left-padding?
        During batch generation, sequences in a batch have different prompt
        lengths. We need all sequences to END at the same position so that
        the model's first generated token (position T+1) is consistent.
        Left-padding achieves this: [PAD PAD PAD tok1 tok2 ... tokN].
        Right-padding would leave the "start of generation" at different
        positions for each sequence in the batch → corrupted generation.
    """
    if model_id is None:
        model_id = cfg.model.policy_model_id

    print(f"[loader] Loading policy tokenizer: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)

    # SmolLM2 uses its EOS token as the pad token (common for decoder-only)
    tok.pad_token    = tok.eos_token
    tok.padding_side = cfg.tokenizer.padding_side  # 'left'
    return tok


def load_rm_tokenizer(
    model_id: Optional[str] = None,
) -> AutoTokenizer:
    """
    Load the Llama-3.2 tokenizer configured for the REWARD MODEL.

    Uses RIGHT-padding because AutoModelForSequenceClassification reads the
    hidden state at the last non-pad token. HuggingFace's Llama implementation
    finds this position by locating the first pad token (argmax of pad mask),
    which is only correct for right-padded sequences.
    """
    if model_id is None:
        model_id = cfg.model.reward_model_id

    print(f"[loader] Loading RM tokenizer: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token    = tok.eos_token
    tok.padding_side = cfg.tokenizer.rm_padding_side  # 'right'
    return tok


# ─────────────────────────────────────────────────────────────────────────────
# Policy model
# ─────────────────────────────────────────────────────────────────────────────

def load_policy_base_model(
    model_id: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> AutoModelForCausalLM:
    """
    Load the SmolLM2-360M base model (NO LoRA yet).

    LoRA is applied separately in model/lora_setup.py via get_peft_model().
    We separate loading from LoRA application so the base model can be:
        (a) wrapped with LoRA for π_θ  (trainable)
        (b) OR loaded frozen as π_ref  (if not using disable_adapter trick)

    Gradient checkpointing:
        Instead of storing ALL intermediate activations for backprop, only
        store activations at "checkpoints" and recompute the rest on the
        backward pass. This trades ~30% extra compute for ~60% less VRAM
        — essential when fine-tuning on a T4 with max_seq_len=1024.

    enable_input_require_grads():
        PEFT bug-fix: when gradient checkpointing is on, the first layer's
        inputs may not have requires_grad=True (since they come from the
        embedding lookup, not a differentiable op). This call forces it,
        ensuring LoRA gradients can flow through the checkpointed layers.
    """
    if model_id is None:
        model_id = cfg.model.policy_model_id

    print(f"[loader] Loading policy base model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",  # place on GPU if available, CPU otherwise
    )

    # Enable gradient checkpointing BEFORE applying PEFT
    model.gradient_checkpointing_enable()

    # Required for PEFT + gradient checkpointing (see docstring)
    model.enable_input_require_grads()

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Reward model backbone
# ─────────────────────────────────────────────────────────────────────────────

def load_reward_backbone(
    model_id: Optional[str] = None,
    load_in_8bit: Optional[bool] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> AutoModelForSequenceClassification:
    """
    Load Llama-3.2-1B-Instruct as a scalar reward model backbone.

    AutoModelForSequenceClassification does two things for us:
        1. Strips the LM head (unembedding + softmax)
        2. Attaches a new linear head: Linear(hidden_size, num_labels=1)
           This is r_ψ(x,y) ∈ ℝ — a raw scalar, not a probability.

    The scalar is read from the hidden state at the LAST NON-PAD TOKEN,
    which captures the full sequence representation in a decoder-only model.

    8-bit loading:
        The RM is FROZEN — we never backprop through it.
        Loading in 8-bit halves VRAM (~2 GB → ~1 GB) at a minor speed cost.
        This is the bitsandbytes 'LLM.int8()' method.
    """
    if model_id is None:
        model_id = cfg.model.reward_model_id
    if load_in_8bit is None:
        load_in_8bit = cfg.model.load_frozen_in_8bit and torch.cuda.is_available()

    print(f"[loader] Loading reward backbone: {model_id}")
    print(f"         8-bit={load_in_8bit}, dtype={dtype}")

    kwargs = dict(
        num_labels=1,       # scalar reward head
        device_map="auto",
    )

    if load_in_8bit:
        # bitsandbytes 8-bit: weights stored as int8, computed in float16.
        # Must use device_map='auto' for bitsandbytes to work correctly.
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["quantization_config"] = bnb_config
    else:
        kwargs["torch_dtype"] = dtype

    model = AutoModelForSequenceClassification.from_pretrained(model_id, **kwargs)

    # Set pad_token_id so the model correctly identifies the last real token
    # (needed by the internal sequence_lengths computation in forward())
    rm_tok = load_rm_tokenizer(model_id)
    model.config.pad_token_id = rm_tok.pad_token_id

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Reference model (frozen copy of the SFT-merged checkpoint)
# ─────────────────────────────────────────────────────────────────────────────

def load_frozen_reference_model(
    checkpoint_path: str,
    dtype: torch.dtype = torch.bfloat16,
    load_in_8bit: Optional[bool] = None,
) -> AutoModelForCausalLM:
    """
    Load π_ref: the frozen SFT-merged model checkpoint.

    This is used during PPO/GRPO/DPO as the KL anchor.
    All parameters are frozen — no gradients, no optimizer state.

    WHY load from 'sft_merged'?
        After SFT, we call peft_model.merge_and_unload() which bakes the
        LoRA delta into the base weights. The merged checkpoint IS the SFT
        policy. When we apply a fresh LoRA on top for PPO/GRPO, disabling
        the adapter gives us back the merged SFT weights — i.e., π_ref.
        If instead we loaded π_ref from the pretrained checkpoint, we'd be
        penalising divergence from the raw base model, not the SFT model.
    """
    if load_in_8bit is None:
        load_in_8bit = cfg.model.load_frozen_in_8bit and torch.cuda.is_available()

    print(f"[loader] Loading frozen π_ref from: {checkpoint_path}")

    kwargs = dict(device_map="auto")
    if load_in_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **kwargs)
    model.eval()

    # Freeze ALL parameters — π_ref must NEVER receive gradient updates
    for param in model.parameters():
        param.requires_grad_(False)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def print_model_stats(model: torch.nn.Module, name: str = "Model") -> None:
    """Print trainable/total parameter counts and GPU memory footprint."""
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params    = total_params - trainable_params

    print(f"\n{'━'*55}")
    print(f" {name}")
    print(f"{'━'*55}")
    print(f"  Total params:     {total_params:>12,}")
    print(f"  Trainable params: {trainable_params:>12,}  ({100*trainable_params/total_params:.2f}%)")
    print(f"  Frozen params:    {frozen_params:>12,}  ({100*frozen_params/total_params:.2f}%)")

    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1e9
        reserved_gb  = torch.cuda.memory_reserved()  / 1e9
        print(f"  VRAM allocated:   {allocated_gb:.2f} GB")
        print(f"  VRAM reserved:    {reserved_gb:.2f} GB")
    print(f"{'━'*55}\n")
