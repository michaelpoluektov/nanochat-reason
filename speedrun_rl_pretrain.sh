#!/bin/bash

# Minimal pipeline to take the public nanochat d32 SFT checkpoint,
# prepare it for nanochat, run RL pretraining, and push the result to Hugging Face.

set -euo pipefail

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p "$NANOCHAT_BASE_DIR"

HF_MODEL_STEP="000650"
HF_LOCAL_DIR="$NANOCHAT_BASE_DIR/downloads/nanochat-d32"
mkdir -p "$HF_LOCAL_DIR"

HF_DATASET_REPO="LazyAGI/GSM8K_Deepseek_R1_Distill-Data-7148"
HF_DATASET_FILE="gsm8k_deepseek_R1_7148.json"
DATASET_DIR="$NANOCHAT_BASE_DIR/datasets/gsm8k_deepseek"
mkdir -p "$DATASET_DIR"

# Require Hugging Face upload target so the run fails fast if uploading is misconfigured.
if [ -z "${HF_UPLOAD_REPO_ID:-}" ]; then
    echo "HF_UPLOAD_REPO_ID is not set. Export it to the destination repository before running this script." >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# Python environment (reuse the same steps as speedrun.sh)

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync
source .venv/bin/activate

# -----------------------------------------------------------------------------
# Optional wandb logging (defaults to 'rl-pretrain' so you can override or disable with WANDB_RUN=dummy)

WANDB_RUN="rl-pretrain"
export WANDB_RUN

# -----------------------------------------------------------------------------
# Download released SFT checkpoint from Hugging Face and place files where nanochat expects them.

command -v huggingface-cli >/dev/null 2>&1 || { echo "huggingface-cli not found. Ensure huggingface-hub is installed."; exit 1; }

hf download "karpathy/nanochat-d32" \
    --repo-type=model \
    --local-dir "$HF_LOCAL_DIR"

if [ -n "$HF_DATASET_FILE" ]; then
    hf  download "$HF_DATASET_REPO" \
        --repo-type=dataset \
        --local-dir "$DATASET_DIR" \
        --include "$HF_DATASET_FILE"
else
    hf download "$HF_DATASET_REPO" \
        --repo-type=dataset \
        --local-dir "$DATASET_DIR"
fi

TARGET_DATASET_PATH="$DATASET_DIR/$HF_DATASET_FILE"
if [ -n "$HF_DATASET_FILE" ] && [ ! -f "$TARGET_DATASET_PATH" ]; then
    DATASET_PATH="$(python - "$DATASET_DIR" "$HF_DATASET_FILE" <<'PY'
import os, sys
dataset_dir = sys.argv[1]
dataset_file = sys.argv[2]
for root, _, files in os.walk(dataset_dir):
    if dataset_file in files:
        print(os.path.join(root, dataset_file))
        break
PY
)"
    if [ -z "$DATASET_PATH" ]; then
        echo "Failed to locate $HF_DATASET_FILE under $DATASET_DIR after download." >&2
        exit 1
    fi
    cp "$DATASET_PATH" "$TARGET_DATASET_PATH"
fi

if [ -n "$HF_DATASET_FILE" ]; then
    echo "✅ Downloaded GSM8K DeepSeek dataset to $TARGET_DATASET_PATH"
else
    echo "✅ Downloaded GSM8K DeepSeek dataset repository to $DATASET_DIR"
fi

TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"
CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/chatsft_checkpoints/d32"
mkdir -p "$TOKENIZER_DIR" "$CHECKPOINT_DIR"

TOKENIZER_DATA_DIR="$NANOCHAT_BASE_DIR/base_data"
TOKENIZER_DATASET_SHARDS="${TOKENIZER_DATASET_SHARDS:-8}"
if ! compgen -G "$TOKENIZER_DATA_DIR"/*.parquet >/dev/null 2>&1; then
    echo "Tokenizer training shards not found under $TOKENIZER_DATA_DIR." >&2
    echo "Downloading $TOKENIZER_DATASET_SHARDS shard(s) with python -m nanochat.dataset..." >&2
    python -m nanochat.dataset -n "$TOKENIZER_DATASET_SHARDS"
fi

if ! compgen -G "$TOKENIZER_DATA_DIR"/*.parquet >/dev/null 2>&1; then
    echo "Failed to locate tokenizer training shards under $TOKENIZER_DATA_DIR after download attempt." >&2
    echo "Ensure the machine has internet access or pre-download the shards manually with" >&2
    echo "  python -m nanochat.dataset -n <num_shards>" >&2
    exit 1
fi

cp "$HF_LOCAL_DIR/tokenizer.pkl" "$TOKENIZER_DIR/tokenizer.pkl"
cp "$HF_LOCAL_DIR/token_bytes.pt" "$TOKENIZER_DIR/token_bytes.pt"
cp "$HF_LOCAL_DIR/meta_${HF_MODEL_STEP}.json" "$CHECKPOINT_DIR/meta_${HF_MODEL_STEP}.json"
cp "$HF_LOCAL_DIR/model_${HF_MODEL_STEP}.pt" "$CHECKPOINT_DIR/model_${HF_MODEL_STEP}.pt"

python scripts/ensure_think_tokens.py \
    --tokenizer-dir "$TOKENIZER_DIR" \
    --data-dir "$TOKENIZER_DATA_DIR" \
    --max-chars 200000000

echo "✅ Downloaded SFT checkpoint (step ${HF_MODEL_STEP}) and placed it under $NANOCHAT_BASE_DIR"

python scripts/gsm8k_token_length_stats.py \
    --dataset "$TARGET_DATASET_PATH" \
    --tokenizer-dir "$TOKENIZER_DIR"

# -----------------------------------------------------------------------------
# Reset report so the RL pretrain run is tracked cleanly.

python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Run RL pretraining (defaults to 1 process, override via NPROC_PER_NODE=8 if you have more GPUs).

NPROC_PER_NODE=1
if [ "$NPROC_PER_NODE" -gt 1 ]; then
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_rl_pretrain -- --run="$WANDB_RUN"
else
    python -m scripts.chat_rl_pretrain --run="$WANDB_RUN"
fi

# -----------------------------------------------------------------------------
# Evaluate the pretraining checkpoint on GSM8K (test split, pass@k style settings from chat_rl.py).

python -m scripts.chat_eval \
    -i prerl \
    -a GSM8K \
    --num-samples 8 \
    --max-new-tokens 256 \
    --temperature 1.0 \
    --top-k 50 \
    --max-problems 400

# -----------------------------------------------------------------------------
# Generate final report markdown for convenience.

python -m nanochat.report generate

echo "🎉 RL pretraining complete. Check $NANOCHAT_BASE_DIR/chatrl_pretrain_checkpoints for outputs."
