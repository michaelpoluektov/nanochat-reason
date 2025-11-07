#!/usr/bin/env python
"""
Profile the chat RL training step (forward + backward) with torch.compile + torch.profiler.
"""

from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_cleanup, compute_init, print0
from nanochat.device import get_autocast_kwargs

# ---------------------------------------------------------------------------
# Configuration (edit these constants instead of passing CLI flags)
SOURCE = "sft"
BATCH_SIZE = 16
SEQ_LEN = 2048
PROMPT_FRACTION = 0.3
COMPILER_BACKEND = None  # e.g. "inductor"
COMPILER_MODE = None  # e.g. "reduce-overhead"
DISABLE_COMPILE = False
USE_AMP = True
SORT_BY = "self_cuda_time_total"
ROW_LIMIT = 25
TRACE_FILE = "chat_rl_train_trace.json"


# ---------------------------------------------------------------------------
def build_representative_batch(tokenizer, batch_size, seq_len, prompt_fraction, device):
    """
    Create a synthetic batch of RL rollouts:
    BOS token at position 0, random ids afterwards, and prompt tokens ignored via -1 targets.
    """
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2 to build inputs/targets.")
    vocab_size = tokenizer.get_vocab_size()
    ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    ids[:, 0] = tokenizer.get_bos_token_id()
    inputs = ids[:, :-1].contiguous()
    targets = ids[:, 1:].clone()
    prompt_tokens = max(1, int((seq_len - 1) * prompt_fraction))
    targets[:, :prompt_tokens] = -1
    return inputs, targets


def maybe_compile(model, backend, mode, disable_compile):
    if disable_compile:
        return model
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        raise RuntimeError("torch.compile is not available in this PyTorch build.")
    compile_kwargs: dict[str, object] = {}
    if backend is not None:
        compile_kwargs["backend"] = backend
    if mode is not None:
        compile_kwargs["mode"] = mode
    return compile_fn(model, **compile_kwargs)


def main():
    *_, ddp_world_size, device = compute_init()
    if ddp_world_size > 1:
        raise RuntimeError("Run the profiler in a single-process environment (no torchrun/DDP).")
    model, tokenizer, _ = load_model(SOURCE, device, phase="eval")
    seq_len = min(SEQ_LEN, model.config.sequence_len)
    inputs, targets = build_representative_batch(
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        seq_len=seq_len,
        prompt_fraction=PROMPT_FRACTION,
        device=device,
    )
    model_c = maybe_compile(model, COMPILER_BACKEND, COMPILER_MODE, DISABLE_COMPILE)
    autocast_kwargs = get_autocast_kwargs(device)
    autocast_kwargs["enabled"] = USE_AMP and autocast_kwargs["enabled"]
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    def train_step():
        model_c.zero_grad(set_to_none=True)
        with torch.amp.autocast(**autocast_kwargs):
            loss = model_c(inputs, targets, loss_reduction="mean")
        loss.backward()
        return loss

    torch.set_grad_enabled(True)
    _ = train_step()  # warmup + compilation
    if device.type == "cuda":
        torch.cuda.synchronize()
    model_c.zero_grad(set_to_none=True)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        loss = train_step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        model_c.zero_grad(set_to_none=True)
        print0(f"Profiled loss: {loss.item():.6f}")

    table = prof.key_averages().table(sort_by=SORT_BY, row_limit=ROW_LIMIT)
    print(table)
    trace_path = TRACE_FILE.strip()
    if trace_path:
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(trace_path)
        print0(f"Chrome trace exported to {trace_path}")

    compute_cleanup()


if __name__ == "__main__":
    main()
