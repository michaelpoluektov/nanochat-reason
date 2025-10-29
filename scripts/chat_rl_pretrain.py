"""
Offline fine-tuning on GSM8K reasoning traces prior to RL.

This script trains on the DeepSeek R1-style chain-of-thought data stored in
``data/gsm8k_deepseek_R1_7148.json``. Usage mirrors other nanoChat training
scripts, so you can launch it either directly or via ``torchrun``.

Example:
    python -m scripts.chat_rl_pretrain -- --run=rl-pretrain
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import random
from typing import Iterable

import torch
import torch.distributed as dist
import wandb

from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, get_base_dir
from nanochat.checkpoint_manager import (
    load_model,
    save_checkpoint,
)
from nanochat.device import get_autocast_kwargs
from nanochat.report import get_report
from tasks.gsm8k_deepseek import GSM8KDeepSeekR1

# -----------------------------------------------------------------------------
# Hyperparameters (override via configurator)
run = os.environ.get("WANDB_RUN")  # wandb run name; use "dummy" to disable logging
wandb.login()

# Model loading
source = "sft"  # base|mid|sft|rl -- which checkpoint family to start from
model_tag = None  # optional specific model tag
step = None  # optional step number

# Data
data_path = None  # defaults to ~/.cache/nanochat/datasets/gsm8k_deepseek/gsm8k_deepseek_R1_7148.json
val_fraction = 0.05
data_seed = 42
max_tokens = 2048  # truncate rendered conversations to this many tokens

# Optimization / training loop
dtype = "bfloat16"
device_batch_size = 2
target_examples_per_step = 16  # total across all ranks per optimizer step
num_epochs = 1
max_iterations = -1  # -1 means derive from dataset size and epochs
grad_clip = 1.0
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 0.05

# Evaluation / logging
eval_every = 50
eval_batches = 32
save_every = 0  # 0 => only save final checkpoint

# allow CLI overrides
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join("nanochat", "configurator.py")).read())  # noqa: E402, pylint: disable=exec-used
user_config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# Compute / precision init
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0
dtype = torch.float32 if dtype == "float32" else torch.bfloat16
autocast_kwargs = get_autocast_kwargs(device)
autocast_ctx = torch.amp.autocast(**{**autocast_kwargs, "dtype": dtype})

# wandb
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
    project="nanochat-rl-pretrain",
    name=run,
    config=user_config,
    save_code=True,
)

# Load model and tokenizer
model, tokenizer, meta = load_model(source, device, phase="train", model_tag=model_tag, step=step)
optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr,
    embedding_lr=embedding_lr,
    matrix_lr=matrix_lr,
    weight_decay=weight_decay,
)
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * init_lr_frac
        group["initial_lr"] = group["lr"]

# -----------------------------------------------------------------------------
# Dataset + data loaders

train_ds = GSM8KDeepSeekR1(
    split="train",
    path=data_path,
    val_fraction=val_fraction,
    seed=data_seed,
)
val_ds = GSM8KDeepSeekR1(
    split="val",
    path=data_path,
    val_fraction=val_fraction,
    seed=data_seed,
)

def build_data_iterator(dataset, batch_size, *, shuffle) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")
    rng = random.Random(data_seed + ddp_rank + (1337 if shuffle else 0))

    def collate(rows: list[tuple[list[int], list[int]]]) -> tuple[torch.Tensor, torch.Tensor]:
        nrows = len(rows)
        ncols = max(len(ids) for ids, _ in rows) - 1
        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long)
        for i, (ids, mask) in enumerate(rows):
            seq_len = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, :seq_len - 1] = ids_tensor[:-1]
            row_targets = ids_tensor[1:]
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1
            targets[i, :seq_len - 1] = row_targets
        return inputs.to(device), targets.to(device)

    while True:
        indices = list(range(ddp_rank, len(dataset), ddp_world_size))
        if len(indices) < batch_size:
            raise ValueError(
                f"Not enough examples ({len(indices)}) for rank {ddp_rank} "
                f"with batch_size={batch_size}. Reduce batch_size or adjust val_fraction."
            )
        if shuffle:
            rng.shuffle(indices)
        batch: list[tuple[list[int], list[int]]] = []
        samples_used = False
        for idx in indices:
            ids, mask = tokenizer.render_conversation(dataset[idx], max_tokens=max_tokens + 1)
            if len(ids) > max_tokens:
                continue  # skip conversations that exceed the model context window
            samples_used = True
            batch.append((ids, mask))
            if len(batch) == batch_size:
                yield collate(batch)
                batch = []
        if not samples_used:
            raise RuntimeError(
                f"All samples exceeded the max sequence length ({max_tokens}) "
                f"for rank {ddp_rank}. Consider increasing max_tokens or filtering the dataset."
            )

train_iter = iter(build_data_iterator(train_ds, device_batch_size, shuffle=True))
def make_val_iter():
    return iter(build_data_iterator(val_ds, device_batch_size, shuffle=False))

# -----------------------------------------------------------------------------
# Training loop prep

examples_per_step = device_batch_size * ddp_world_size
if target_examples_per_step <= 0:
    target_examples_per_step = examples_per_step
assert target_examples_per_step % examples_per_step == 0, "target_examples_per_step must be divisible by per-rank batch contributions"
grad_accum_steps = target_examples_per_step // examples_per_step
print0(f"Training examples per step (global): {target_examples_per_step}")
print0(f"Gradient accumulation steps: {grad_accum_steps}")

derived_iterations = (len(train_ds) // target_examples_per_step) * num_epochs
if derived_iterations == 0:
    derived_iterations = max(1, len(train_ds) // examples_per_step)
num_iterations = derived_iterations if max_iterations < 0 else min(derived_iterations, max_iterations)
if num_iterations <= 0:
    num_iterations = 1
print0(f"Training iterations: {num_iterations}")

def get_lr_multiplier(it):
    return max(0.0, 1.0 - it / max(num_iterations, 1))

# -----------------------------------------------------------------------------
# Training loop

latest_val_loss = None
for step_idx in range(num_iterations):
    last_step = step_idx == num_iterations - 1

    # Validation
    should_eval = eval_every > 0 and (step_idx % eval_every == 0 or last_step)
    if should_eval and len(val_ds) > 0:
        model.eval()
        val_iter = make_val_iter()
        losses = []
        with torch.no_grad(), autocast_ctx:
            for _ in range(eval_batches):
                val_inputs, val_targets = next(val_iter)
                loss = model(val_inputs, val_targets)
                losses.append(loss)
        if losses:
            val_loss = torch.stack(losses).mean()
            if ddp:
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            latest_val_loss = val_loss.item()
            print0(f"Step {step_idx:05d} | Validation loss: {latest_val_loss:.6f}")
            wandb_run.log({
                "step": step_idx,
                "val_loss": latest_val_loss,
            })
        model.train()

    # Train
    running_loss = torch.zeros(1, device=device)
    num_tokens = torch.tensor(0, device=device)
    for micro_step in range(grad_accum_steps):
        train_inputs, train_targets = next(train_iter)
        with autocast_ctx:
            loss = model(train_inputs, train_targets)
        running_loss += loss.detach()
        num_tokens += (train_targets >= 0).sum()
        (loss / grad_accum_steps).backward()
    if ddp:
        dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM)

    if grad_clip is not None and grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    lrm = get_lr_multiplier(step_idx)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

    train_loss = running_loss / grad_accum_steps
    if ddp:
        dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
    train_loss_item = train_loss.item()
    num_tokens_item = num_tokens.item()

    print0(f"Step {step_idx:05d}/{num_iterations:05d} | Train loss: {train_loss_item:.6f} | lrm: {lrm:.6f} | num_tokens: {num_tokens_item:,}")
    wandb_run.log({
        "step": step_idx,
        "train_loss": train_loss_item,
        "lrm": lrm,
        "num_tokens": num_tokens_item,
    })

    # Optional periodic checkpointing
    if master_process and save_every > 0 and ((step_idx + 1) % save_every == 0 or last_step):
        depth = model.config.n_layer
        model_tag_out = f"d{depth}"
        base_dir = get_base_dir()
        checkpoints_root = os.path.join(base_dir, "chatrl_pretrain_checkpoints")
        checkpoint_dir = os.path.join(checkpoints_root, model_tag_out)
        os.makedirs(checkpoint_dir, exist_ok=True)
        meta_payload = {
            "step": step_idx + 1,
            "train_loss": train_loss_item,
            "val_loss": latest_val_loss,
            "model_config": model.config.__dict__,
        }
        save_checkpoint(
            checkpoint_dir,
            step_idx + 1,
            model.state_dict(),
            None,
            meta_payload,
        )
        print0(f"✅ Saved model checkpoint to {checkpoint_dir}")

# Final save (if not already saved above)
if master_process and (save_every == 0 or (num_iterations % save_every) != 0):
    depth = model.config.n_layer
    model_tag_out = f"d{depth}"
    base_dir = get_base_dir()
    checkpoints_root = os.path.join(base_dir, "chatrl_pretrain_checkpoints")
    checkpoint_dir = os.path.join(checkpoints_root, model_tag_out)
    os.makedirs(checkpoint_dir, exist_ok=True)
    meta_payload = {
        "step": num_iterations,
        "train_loss": train_loss_item,
        "val_loss": latest_val_loss,
        "model_config": model.config.__dict__,
    }
    save_checkpoint(
        checkpoint_dir,
        num_iterations,
        model.state_dict(),
        None,
        meta_payload,
    )
    print0(f"✅ Saved model checkpoint to {checkpoint_dir}")

# Report summary
get_report().log(
    section="Chat RL Pretrain",
    data=[
        user_config,
        {
            "Training rows": len(train_ds),
            "Validation rows": len(val_ds),
            "Iterations": num_iterations,
            "Latest train loss": train_loss_item,
            "Latest val loss": latest_val_loss,
        },
    ],
)

wandb_run.finish()
compute_cleanup()
