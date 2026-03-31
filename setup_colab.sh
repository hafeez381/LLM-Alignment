#!/usr/bin/env bash
# setup_colab.sh — Run this ONCE at the start of each Colab session.
#
# Usage (in a Colab cell):
#   !bash setup_colab.sh
#
# Or to authenticate with HuggingFace in the same step:
#   !HF_TOKEN=hf_xxx bash setup_colab.sh

set -e  # exit immediately on any error

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " PA2 Colab Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 1. Install packages ────────────────────────────────────────────────────
echo "[1/4] Installing Python packages..."
pip install -q \
    "torch>=2.1.0" \
    "transformers>=4.40.0" \
    "datasets>=2.18.0" \
    "peft>=0.10.0" \
    "accelerate>=0.28.0" \
    "bitsandbytes>=0.43.0" \
    "sentencepiece>=0.1.99" \
    "protobuf>=3.20.0" \
    "tqdm>=4.66.0" \
    "scipy>=1.11.0" \
    "matplotlib>=3.7.0"
echo "    ✓ packages installed"

# ── 2. Authenticate with HuggingFace ──────────────────────────────────────
# meta-llama/Llama-3.2-1B-Instruct requires accepting the licence on
# https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
# then logging in here. SmolLM2-360M does NOT require authentication.
echo "[2/4] Checking HuggingFace authentication..."
if [ -n "$HF_TOKEN" ]; then
    echo "    Found HF_TOKEN env var — logging in..."
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
    echo "    ✓ logged in"
else
    echo "    HF_TOKEN not set."
    echo "    To authenticate, run:"
    echo "      import os; os.environ['HF_TOKEN'] = 'hf_xxx'"
    echo "    or re-run this script with HF_TOKEN=hf_xxx bash setup_colab.sh"
fi

# ── 3. GPU sanity check ───────────────────────────────────────────────────
echo "[3/4] GPU check..."
python - <<'EOF'
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1e9
    print(f"    ✓ GPU: {props.name}, VRAM: {vram_gb:.1f} GB")
    if vram_gb < 12:
        print("    ⚠ VRAM < 12 GB — consider reducing batch_size or enabling 8-bit")
else:
    print("    ✗ No GPU found — switch to T4 runtime in Runtime → Change runtime type")
EOF

# ── 4. Create checkpoint dirs ─────────────────────────────────────────────
echo "[4/4] Creating checkpoint directories..."
mkdir -p checkpoints/{sft,sft_merged,rm,ppo,grpo,dpo,rlvr}
echo "    ✓ checkpoints/ ready"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Setup complete! You can now run:"
echo "   python train_sft.py    # C2"
echo "   python train_rm.py     # C1"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
