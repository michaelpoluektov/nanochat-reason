#!/usr/bin/env python
"""
Profile the chat RL model with torch.compile + torch.profiler.
"""

import argparse
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_cleanup, compute_init, print0
from nanochat.device import get_autocast_kwargs


def build_representative_batch(tokenizer, batch_size, seq_len, prompt_fraction, device):
    """
    Create a synthetic batch of tokens that looks like an RL rollout:
    random ids with the BOS token up front and part of the sequence masked out.
    """
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2 to build inputs/targets.")
    vocab_size = tokenizer.get_vocab_size()
    ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    ids[:, 0] = tokenizer.get_bos_token_id()
    inputs = ids[:, :-1].contiguous()
    targets = ids[:, 1:].clone()
    prompt_tokens = max(1, int((seq_len - 1) * prompt_fraction))
    targets[:, :prompt_tokens] = -1  # ignore the prompt portion the RL loop would skip
    return inputs, targets


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="sft", help="Checkpoint source used in chat_rl.py.")
    parser.add_argument("--batch-size", type=int, default=16, help="Per-device batch size.")
    parser.add_argument("--seq-len", type=int, default=2048, help="Number of tokens incl. EOS placeholder.")
    parser.add_argument("--prompt-fraction", type=float, default=0.3, help="Fraction of tokens treated as prompt/ignored.")
    parser.add_argument("--compiler-backend", default=None, help="torch.compile backend (default: inductor).")
    parser.add_argument("--compiler-mode", default=None, help="torch.compile mode (e.g. default, reduce-overhead).")
    parser.add_argument("--no-amp", action="store_true", help="Disable autocast during profiling.")
    parser.add_argument("--sort-by", default="self_cuda_time_total", help="Column to sort the profiler table by.")
    parser.add_argument("--row-limit", type=int, default=25, help="Number of rows to display in the profiler table.")
    parser.add_argument(
        "--trace-file",
        default="chat_rl_trace.json",
        help="Chrome trace output path. Set to '' to skip exporting.",
    )
    args = parser.parse_args()
    if isinstance(args.compiler_backend, str) and args.compiler_backend.lower() == "none":
        args.compiler_backend = None
    if isinstance(args.compiler_mode, str) and args.compiler_mode.lower() == "none":
        args.compiler_mode = None
    return args


def main():
    args = parse_args()
    *_, ddp_world_size, device = compute_init()
    if ddp_world_size > 1:
        raise RuntimeError("Run the profiler in a single-process environment (no torchrun/DDP).")
    model, tokenizer, _ = load_model(args.source, device, phase="eval")
    seq_len = min(args.seq_len, model.config.sequence_len)
    inputs, targets = build_representative_batch(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        seq_len=seq_len,
        prompt_fraction=args.prompt_fraction,
        device=device,
    )
    compile_kwargs = {}
    if args.compiler_backend is not None:
        compile_kwargs["backend"] = args.compiler_backend
    if args.compiler_mode is not None:
        compile_kwargs["mode"] = args.compiler_mode
    model_c = torch.compile(model, **compile_kwargs)
    autocast_kwargs = get_autocast_kwargs(device)
    if args.no_amp:
        autocast_kwargs["enabled"] = False
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    def forward_pass():
        with torch.amp.autocast(**autocast_kwargs):
            return model_c(inputs, targets, loss_reduction="none")

    with torch.inference_mode():
        _ = forward_pass()  # warmup + compilation
    if device.type == "cuda":
        torch.cuda.synchronize()

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with torch.inference_mode():
            _ = forward_pass()

    table = prof.key_averages().table(sort_by=args.sort_by, row_limit=args.row_limit)
    print(table)
    trace_path = args.trace_file.strip()
    if trace_path:
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(trace_path)
        print0(f"Chrome trace exported to {trace_path}")

    compute_cleanup()


if __name__ == "__main__":
    main()
