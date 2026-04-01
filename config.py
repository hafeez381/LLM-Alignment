"""
config.py — Central Configuration
================================================================
All hyperparameters live here as frozen dataclasses.
Import `cfg` for configurations needed.
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Model IDs
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    # Policy: SmolLM2-360M chosen for its tiny VRAM footprint (~720 MB in bf16).
    # This lets us stack policy + reference + reward model on one T4 (15 GB).
    policy_model_id: str = "HuggingFaceTB/SmolLM2-360M-Instruct"

    # RM/value backbone: Llama-3.2-1B-Instruct.
    # A larger backbone → richer hidden states → better reward signal.
    # NOTE: requires accepting the model licence on HuggingFace and
    # authenticating via `huggingface-cli login` or HF_TOKEN env var.
    reward_model_id: str = "meta-llama/Llama-3.2-1B-Instruct"

    # Load frozen models (π_ref, RM) in 8-bit to halve their VRAM footprint.
    # bitsandbytes 8-bit only works on CUDA; we guard against CPU in loader.py.
    load_frozen_in_8bit: bool = False

    # bfloat16 preferred over float16: same range as float32, no overflow risk.
    # float16 can NaN with the large logit values LLMs produce.
    dtype: str = "bfloat16"


# ─────────────────────────────────────────────────────────────────────────────
# LoRA
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LoRAConfig:
    # r = 8: rank of the low-rank decomposition W ≈ W₀ + (B·A), where
    # B ∈ ℝ^{d×r}, A ∈ ℝ^{r×d}. Smaller r → fewer trainable params.
    r: int = 8

    # α = 16: scaling factor. Effective weight delta = (α/r) * B·A.
    # Keeping α/r = 2 is a common default that prevents adapter outputs from
    # being too small at init (where B·A ≈ 0) also preventoing vanishing gradients.
    lora_alpha: int = 16

    # Only adapt Q and V projection matrices.
    # Q and V control attention content; adapting them is sufficient for
    # alignment tasks and keeps the parameter budget tiny.
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_dropout: float = 0.05

    # Don't adapt bias terms — keeps adapter parameter count minimal
    bias: str = "none"

    # After SFT, we merge LoRA back into the base weights so that the merged
    # checkpoint IS π_ref. PPO/GRPO then apply a *fresh* LoRA on top.
    # This means disable_adapter() during PPO gives us π_ref (SFT merged), not
    # the pretrained base — which is exactly what we want for the KL penalty.
    task_type: str = "CAUSAL_LM"


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TokenizerConfig:
    max_seq_len: int = 512  # prompt + response ≤ 1024 tokens total

    # Left-padding for the POLICY tokenizer.
    # WHY: In autoregressive generation, we batch prompts of different lengths.
    # Left-padding ensures all sequences in a batch END at the same position,
    # so the model's first generated token is always at position -1 (the same
    # KV-cache slot). Right-padding would misalign the generation start point.
    padding_side: str = "left"

    # RM tokenizer uses RIGHT-padding.
    # WHY: AutoModelForSequenceClassification reads the hidden state at the
    # LAST NON-PAD TOKEN. HuggingFace finds this by locating the first pad
    # token (argmax of pad_mask), which only works when padding is on the right.
    rm_padding_side: str = "right"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    dataset_id: str = "Anthropic/hh-rlhf"
    subset: str = "default"

    # Set to an integer to use a smaller subset for fast iteration on Colab.
    # Set to None to use the full ~42 k train / ~2.3 k test split.
    train_subset_size: Optional[int] = 42000  # e.g. 5000 for quick runs
    test_subset_size: Optional[int] = 2300   # e.g. 500


# ─────────────────────────────────────────────────────────────────────────────
# SFT
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SFTConfig:
    num_epochs: int = 1
    batch_size: int = 16
    grad_accum_steps: int = 2    # effective batch = 16 × 2 = 32
    learning_rate: float = 2e-4  # LoRA adapts far fewer params → higher LR ok
    max_grad_norm: float = 1.0
    warmup_steps: int = 50
    log_every: int = 10          # print loss every N optimizer steps
    eval_every: int = 100        # eval perplexity on held-out set every N steps
    save_dir: str = "checkpoints/sft"
    merged_save_dir: str = "checkpoints/sft_merged"  # merged model = π_ref


# ─────────────────────────────────────────────────────────────────────────────
# Reward Model
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RMConfig:
    num_epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    warmup_steps: int = 30

    # L2 regularisation on reward magnitudes:
    #   λ_reg · E[(r⁺)² + (r⁻)²]
    # Without this, the model can make |r| → ∞ while still ranking correctly,
    # which causes overflow/NaN and makes rewards non-comparable across batches.
    reg_lambda: float = 1e-3

    log_every: int = 10
    eval_every: int = 50
    save_dir: str = "checkpoints/rm"

    # Minimum target before using RM in downstream RL training.
    # If accuracy < 60%, the RM is no better than random — stop and debug.
    min_accuracy_threshold: float = 0.60


# ─────────────────────────────────────────────────────────────────────────────
# PPO
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PPOConfig:
    total_steps: int = 200
    prompts_per_step: int = 16      # |batch| of prompts sampled per update
    max_new_tokens: int = 128
    beta: float = 0.1              # KL penalty coefficient β in J(θ)
    epsilon: float = 0.2           # clipping range: ρ_t ∈ [1-ε, 1+ε]
    gamma: float = 1.0             # discount γ (1.0 = no discounting for text)
    lam: float = 0.95              # GAE λ: bias-variance trade-off in A_t^GAE

    # Number of gradient update passes over EACH rollout buffer.
    # Each pass uses mini-batches drawn without replacement from the buffer.
    ppo_epochs: int = 4
    mini_batch_size: int = 8

    # PPO full loss: L = -L_clip + c_V·L_V - c_ent·H(π_θ) + c_KL·L_KL
    c_value: float = 0.5       # value loss coefficient
    c_entropy: float = 0.01    # entropy bonus (prevents premature collapse)

    eval_every: int = 25
    save_dir: str = "checkpoints/ppo"


# ─────────────────────────────────────────────────────────────────────────────
# GRPO
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class GRPOConfig:
    total_steps: int = 200
    prompts_per_step: int = 16
    K: int = 4                 # rollouts per prompt; group size for advantage
    max_new_tokens: int = 128
    beta: float = 0.1
    epsilon: float = 0.2
    eval_every: int = 25
    save_dir: str = "checkpoints/grpo"


# ─────────────────────────────────────────────────────────────────────────────
# DPO
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DPOConfig:
    num_epochs: int = 1
    batch_size: int = 16
    grad_accum_steps: int = 2
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    beta: float = 0.1          # temperature β in the DPO loss
    log_every: int = 10
    save_dir: str = "checkpoints/dpo"


# ─────────────────────────────────────────────────────────────────────────────
# RLVR  (GSM8K)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RLVRConfig:
    total_steps: int = 300
    prompts_per_step: int = 8
    K: int = 4
    max_new_tokens: int = 256   # 2× PPO/GRPO — reasoning chains are longer
    beta: float = 0.05          # lower β: reward signal is verifiable, less hack risk
    epsilon: float = 0.2
    eval_every: int = 25
    save_dir: str = "checkpoints/rlvr"
    gsm8k_dataset_id: str = "openai/gsm8k"


# ─────────────────────────────────────────────────────────────────────────────
# Root config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PA2Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    rm: RMConfig = field(default_factory=RMConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    rlvr: RLVRConfig = field(default_factory=RLVRConfig)
    seed: int = 42


# Singleton instance — import this everywhere:
#   from config import cfg
cfg = PA2Config()
