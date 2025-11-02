"""
Benchmark the NanoChat engine generation throughput for fixed settings.

Edit the global constants below to point the script at a checkpoint and control
the generation parameters. Run with:

    python -m tests.benchmark_engine
"""

from __future__ import annotations

import statistics
import time

import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_cleanup, compute_init
from nanochat.engine import Engine, KVCache

# -----------------------------------------------------------------------------
# Benchmark configuration (tune as needed)

MODEL_TAG = None  # e.g. "d12"; defaults to the largest model if None
STEP = None  # set to an integer to lock to a specific step; None = latest
PROMPT = ""  # empty string is fine; BOS token is always prepended
BATCH_SIZES = [1, 2, 4, 8, 12, 16, 24, 32]
MAX_NEW_TOKENS = 2048
NUM_ITERS = 3
# Note: the local BenchmarkEngine ignores EOS/BOS to always generate MAX_NEW_TOKENS.

# -----------------------------------------------------------------------------


class BenchmarkEngine(Engine):
    """Minimal engine variant that ignores EOS to force long rollouts."""

    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)

    @torch.inference_mode()
    def generate(
        self,
        tokens,
        num_samples=1,
        max_tokens=None,
    ):
        assert isinstance(tokens, list) and tokens, "tokens must be a non-empty list"
        device = self.model.get_device()

        # Prefill prompt once, then clone KV cache for batched decoding.
        model_cfg = self.model.config
        kv_kwargs = {
            "num_heads": model_cfg.n_kv_head,
            "head_dim": model_cfg.n_embd // model_cfg.n_head,
            "num_layers": model_cfg.n_layer,
        }
        kv_cache_prefill = KVCache(batch_size=1, seq_len=len(tokens), **kv_kwargs)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)[:, -1, :]
        first_token = torch.argmax(logits, dim=-1).item()

        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else model_cfg.sequence_len
        kv_cache_decode = KVCache(batch_size=num_samples, seq_len=kv_length_hint, **kv_kwargs)
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill

        num_generated = 0
        first_iteration = True
        while True:
            if max_tokens is not None and num_generated >= max_tokens:
                break

            if first_iteration:
                token_column = [first_token] * num_samples
                first_iteration = False
            else:
                logits = self.model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]
                token_column = torch.argmax(logits, dim=-1).tolist()

            yield token_column, [1] * num_samples
            num_generated += 1
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)

    @torch.inference_mode()
    def generate_batch(self, tokens, num_samples=1, **kwargs):
        sequences = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        for token_column, token_masks in self.generate(tokens, num_samples=num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                sequences[i].append(token)
                masks[i].append(mask)
        return sequences, masks


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_once(
    engine: BenchmarkEngine,
    prompt_tokens: list[int],
    *,
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[float, int]:
    kwargs = dict(max_tokens=max_new_tokens)
    prompt_len = len(prompt_tokens)
    start_ns = time.perf_counter_ns()
    sequences, _ = engine.generate_batch(prompt_tokens, num_samples=batch_size, **kwargs)
    _synchronize(device)
    end_ns = time.perf_counter_ns()
    total_new_tokens = sum(max(len(seq) - prompt_len, 0) for seq in sequences)
    latency_s = (end_ns - start_ns) / 1e9
    return latency_s, total_new_tokens


def main() -> None:
    ddp, rank, _, world_size, device = compute_init()
    if ddp and world_size > 1:
        raise RuntimeError("This benchmark is intended for single-process runs.")

    if rank == 0:
        print(f"Loading model from source 'sft' (model_tag={MODEL_TAG}, step={STEP})")
    model, tokenizer, _ = load_model(
        "sft",
        device=device,
        phase="eval",
        model_tag=MODEL_TAG,
        step=STEP,
    )
    engine = BenchmarkEngine(model, tokenizer)

    bos = tokenizer.get_bos_token_id()
    prompt_tokens = tokenizer.encode(PROMPT, prepend=bos) if PROMPT else [bos]

    results = []
    for batch_size in BATCH_SIZES:
        if batch_size <= 0:
            raise ValueError(f"Invalid batch size: {batch_size}")
        if rank == 0:
            print(f"\nBatch size {batch_size}: running warmup...")
        # Warmup (not timed)
        benchmark_once(
            engine,
            prompt_tokens,
            batch_size=batch_size,
            max_new_tokens=MAX_NEW_TOKENS,
            device=device,
        )
        latencies = []
        new_token_counts = []
        for iter_idx in range(NUM_ITERS):
            latency_s, total_new_tokens = benchmark_once(
                engine,
                prompt_tokens,
                batch_size=batch_size,
                max_new_tokens=MAX_NEW_TOKENS,
                device=device,
            )
            latencies.append(latency_s)
            new_token_counts.append(total_new_tokens)
            if rank == 0:
                print(
                    f"  Iter {iter_idx + 1}/{NUM_ITERS}: "
                    f"{latency_s:.3f}s, {total_new_tokens} new tokens"
                )
        mean_latency = statistics.fmean(latencies)
        mean_new_tokens = statistics.fmean(new_token_counts)
        tokens_per_sec = mean_new_tokens / mean_latency if mean_latency > 0 else float("inf")
        results.append(
            {
                "batch_size": batch_size,
                "latency_s": mean_latency,
                "tokens_per_sec": tokens_per_sec,
                "tokens_per_sample": mean_new_tokens / batch_size if batch_size else 0.0,
            }
        )

    if rank == 0:
        print("\n=== Benchmark Summary ===")
        header = f"{'Batch':>8} {'Latency (s)':>12} {'Tokens/sample':>15} {'Tokens/s':>12}"
        print(header)
        print("-" * len(header))
        for item in results:
            print(
                f"{item['batch_size']:>8d} "
                f"{item['latency_s']:>12.3f} "
                f"{item['tokens_per_sample']:>15.2f} "
                f"{item['tokens_per_sec']:>12.1f}"
            )

    compute_cleanup()


if __name__ == "__main__":
    main()
