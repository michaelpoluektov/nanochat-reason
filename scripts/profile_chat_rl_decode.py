#!/usr/bin/env python
"""
Profile the chat RL decoding stage (prefill + generation) with torch.profiler.
"""

from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_cleanup, compute_init, print0
from nanochat.device import get_autocast_kwargs
from nanochat.engine import Engine

SOURCE = "sft"
PROMPT_TEXT = "Solve 12 + 34. Show your work."
NUM_SAMPLES = 16
MAX_NEW_TOKENS = 256
TEMPERATURE = 1.0
TOP_K = 50  # set <=0 to disable top-k
SAMPLING_SEED = 0
COMPILER_BACKEND = None
COMPILER_MODE = None
DISABLE_COMPILE = False
USE_AMP = True
SORT_BY = "self_cuda_time_total"
ROW_LIMIT = 25
TRACE_FILE = "chat_rl_decode_trace.json"


# ---------------------------------------------------------------------------
def build_prompt_tokens(tokenizer, prompt_text):
    """
    Build a simple conversation and return tokens primed for Assistant completion.
    """
    conversation = {
        "messages": [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": "Let me think step by step.\n\\boxed{0}"},
        ]
    }
    return tokenizer.render_for_completion(conversation)


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
    model_c = maybe_compile(model, COMPILER_BACKEND, COMPILER_MODE, DISABLE_COMPILE)
    engine = Engine(model_c, tokenizer)
    prompt_tokens = build_prompt_tokens(tokenizer, PROMPT_TEXT)
    autocast_kwargs = get_autocast_kwargs(device)
    autocast_kwargs["enabled"] = USE_AMP and autocast_kwargs["enabled"]
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    def decode():
        with torch.amp.autocast(**autocast_kwargs):
            sequences, masks = engine.generate_batch(
                prompt_tokens,
                num_samples=NUM_SAMPLES,
                max_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_k=(TOP_K if TOP_K and TOP_K > 0 else None),
                seed=SAMPLING_SEED,
            )
        return sequences, masks

    with torch.inference_mode():
        decode()  # warmup + compilation
    if device.type == "cuda":
        torch.cuda.synchronize()

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with torch.inference_mode():
            sequences, _ = decode()
        if device.type == "cuda":
            torch.cuda.synchronize()
        print0(f"Generated {len(sequences)} samples (first length: {len(sequences[0]) if sequences else 0})")

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
