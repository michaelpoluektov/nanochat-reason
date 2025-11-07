#!/bin/bash

# Minimal pipeline to download an RL pretraining checkpoint from Hugging Face
# and continue reinforcement learning on GSM8K.

set -euo pipefail

# -----------------------------------------------------------------------------
# Edit the values in this block to fit your setup.

HF_PRERL_REPO_ID="accountblabla/nanochat-fp8"
WANDB_RUN="rl"
NPROC_PER_NODE=1

# -----------------------------------------------------------------------------

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p "$NANOCHAT_BASE_DIR"

if [ "$HF_PRERL_REPO_ID" = "your-username/rl-pretrain-model" ]; then
    echo "Please edit speedrun_rl.sh and set HF_PRERL_REPO_ID to your Hugging Face repo." >&2
    exit 1
fi

uv sync --extra gpu
source .venv/bin/activate

export WANDB_RUN

echo "📥 Downloading pretraining checkpoint from Hugging Face..."
python -m scripts.download_checkpoint \
    --repo-id "$HF_PRERL_REPO_ID" \
    --source-or-dir=prerl

python -m nanochat.report reset

if [ "$NPROC_PER_NODE" -gt 1 ]; then
    echo "🚀 Launching RL training with torchrun ($NPROC_PER_NODE processes)..."
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
        -m scripts.chat_rl -- --source=prerl --run="$WANDB_RUN"
else
    echo "🚀 Launching RL training with python..."
    python -m scripts.chat_rl --source=prerl --run="$WANDB_RUN"
fi

python -m scripts.chat_eval \
    -i rl \
    -a GSM8K \
    --num-samples 32 \
    --max-new-tokens 2048 \
    --temperature 1.0 \
    --top-k 50 \
    --max-problems 100

python -m nanochat.report generate

echo "🎉 RL training complete. Check $NANOCHAT_BASE_DIR/chatrl_checkpoints for outputs."
