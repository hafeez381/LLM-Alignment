"""
model/value_head.py — PPO Value Model V_φ(s_t)
================================================

PSEUDOCODE:
────────────
ValueHead.__init__(backbone, hidden_size)
  → Store backbone (Llama-1B AutoModel — pure transformer, no LM head)
  → v_head = Linear(hidden_size, 1, bias=False)
  → init v_head with std=0.01  (near-zero: V starts near 0, not random)
  → WHY near-zero? If V starts large, GAE advantages are dominated by
    the initial critic error, not the actual rewards. Near-zero init
    lets advantages reflect true reward signal from the first step.

ValueHead.forward(input_ids, attention_mask)
  → backbone forward → last_hidden_state: (B, L, hidden_size)
  → v_head linear → values: (B, L, 1) → squeeze → (B, L)
  → Returns V_φ(s_t) for EVERY token position t in [0, L)
  → Caller selects only response positions: values[:, prompt_len:]

ValueHead.freeze_backbone()
  → Set requires_grad=False on all backbone params
  → Only v_head remains trainable
  → WHY: With a frozen backbone, the critic learns a linear projection
    of frozen Llama features. Cheap, stable, often sufficient for short
    fine-tuning runs. LoRA backbone is more expressive but costs VRAM.

load_value_model(model_id, load_in_8bit, dtype, freeze_backbone)
  → AutoModel.from_pretrained (no LM head → less VRAM)
  → Wrap in ValueHead
  → Optionally freeze backbone
  → Return model

ARCHITECTURE NOTE:
    We use AutoModel (LlamaModel), NOT AutoModelForCausalLM.
    AutoModel gives us just the transformer stack.
    Its output.last_hidden_state has shape (B, L, H) at every position,
    which is exactly what we need to compute V(s_t) at every time step t.
    AutoModelForCausalLM would also give hidden states (via output_hidden_states=True)
    but wastes memory by loading the unembedding matrix (vocab_size × H) we don't use.
"""

import torch
import torch.nn as nn
from typing import Optional

from transformers import AutoModel, BitsAndBytesConfig

from config import cfg


# ─────────────────────────────────────────────────────────────────────────────
# Value Head
# ─────────────────────────────────────────────────────────────────────────────

class ValueHead(nn.Module):
    """
    Scalar value function V_φ(s_t) for PPO's critic.

    Architecture:
        Llama-1B backbone (LlamaModel — transformer only, no LM head)
        └── Linear(hidden_size → 1)   ← v_head, near-zero init

    forward() returns V(s_t) at EVERY token position.
    The caller indexes [:, prompt_len:] to get response-position values.

    WHY Llama-1B and not SmolLM2 (same as the policy)?
        Using a DIFFERENT model for the critic keeps policy and critic
        gradients fully decoupled. If both shared the same backbone,
        a large critic update could corrupt policy representations and
        vice versa. Separate models avoid this interference.
    """

    def __init__(
        self,
        backbone: nn.Module,      # LlamaModel (AutoModel output)
        hidden_size: int,
    ):
        super().__init__()
        self.backbone = backbone

        # Scalar projection: one value per token position
        self.v_head = nn.Linear(hidden_size, 1, bias=False)

        # Near-zero initialisation — see module docstring
        nn.init.normal_(self.v_head.weight, std=0.01)
        self.v_head = self.v_head.to(torch.bfloat16)


    def forward(
        self,
        input_ids: torch.Tensor,       # shape: (batch, seq_len)
        attention_mask: torch.Tensor,  # shape: (batch, seq_len)
    ) -> torch.Tensor:
        """
        Returns
        -------
        values : shape (batch, seq_len)
            V_φ(s_t) for every token position t.
            At padding positions the value is defined but meaningless —
            callers should always apply a response_mask before using these.
        """
        # backbone forward: pure transformer, no language model head
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # last_hidden_state: shape (batch, seq_len, hidden_size)
        hidden = outputs.last_hidden_state

        # Project each token's hidden state to a scalar
        # v_head output: (batch, seq_len, 1) → squeeze → (batch, seq_len)
        values = self.v_head(hidden).squeeze(-1)   # shape: (batch, seq_len)
        return values

    def freeze_backbone(self) -> None:
        """
        Freeze the Llama backbone so only the linear value head is trained.

        This is the lightweight critic option:
            Trainable params ≈ hidden_size × 1 ≈ 2048 params (Llama-1B)
        vs. LoRA backbone:
            Trainable params ≈ 2 × r × d_model × n_layers ≈ ~500 k params

        Frozen backbone is more stable early in training; LoRA is more
        expressive once the policy is learning non-trivial behaviour.
        """
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        print("[value_head] Backbone frozen — only v_head is trainable")

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone (for LoRA setup via apply_lora in lora_setup.py)."""
        for param in self.backbone.parameters():
            param.requires_grad_(True)


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_value_model(
    model_id: Optional[str] = None,
    load_in_8bit: Optional[bool] = None,
    dtype: torch.dtype = torch.bfloat16,
    freeze_backbone: bool = True,
) -> ValueHead:
    """
    Load the Llama-1B backbone and wrap it in a ValueHead.

    Parameters
    ----------
    model_id        : HuggingFace model ID (default: Llama-3.2-1B-Instruct)
    load_in_8bit    : quantise backbone weights to 8-bit (saves ~1 GB VRAM)
                      ONLY valid when freeze_backbone=True (can't train int8)
    dtype           : compute dtype for non-quantised loading
    freeze_backbone : if True, only train the linear v_head (cheap + stable)
                      if False, apply LoRA to backbone after calling this fn

    Returns
    -------
    value_model : ValueHead ready for PPO training
    """
    if model_id is None:
        model_id = cfg.model.reward_model_id  # reuse Llama-1B ID
    if load_in_8bit is None:
        # 8-bit is only safe when backbone is frozen; trainable backbone needs bf16
        load_in_8bit = (
            cfg.model.load_frozen_in_8bit
            and freeze_backbone
            and torch.cuda.is_available()
        )

    print(f"[value_head] Loading backbone: {model_id}")
    print(f"             8-bit={load_in_8bit}, dtype={dtype}, "
          f"freeze_backbone={freeze_backbone}")

    kwargs = dict(device_map="auto")
    if load_in_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kwargs["torch_dtype"] = dtype

    # AutoModel = pure transformer (LlamaModel for Llama family)
    # No LM head → saves ~(vocab_size × hidden_size) ≈ 256 MB VRAM
    backbone = AutoModel.from_pretrained(model_id, **kwargs)

    hidden_size = backbone.config.hidden_size  # 2048 for Llama-1B
    value_model = ValueHead(backbone, hidden_size)

    if freeze_backbone:
        value_model.freeze_backbone()

    return value_model
