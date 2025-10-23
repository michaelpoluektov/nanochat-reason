#!/usr/bin/env python3
"""
Compute tokenizer length coverage statistics for the GSM8K DeepSeek RL pretrain dataset.

The script loads the JSON dataset, renders each record into a chat conversation using the
project tokenizer, and reports what fraction of samples fit under several token limits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from nanochat.tokenizer import RustBPETokenizer
from tasks.gsm8k_deepseek import GSM8KDeepSeekR1


DEFAULT_THRESHOLDS = (1024, 2048, 4096, 8192)
DEFAULT_MAX_TOKENS = 65536


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report the percentage of dataset samples under specified token limits."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path.home() / ".cache" / "nanochat" / "datasets" / "gsm8k_deepseek" / "gsm8k_deepseek_R1_7148.json",
        help="Path to the GSM8K DeepSeek dataset JSON file.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path.home() / ".cache" / "nanochat" / "tokenizer",
        help="Directory containing tokenizer.pkl and token_bytes.pt.",
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
        help="Token count thresholds to evaluate.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Safety cap passed to tokenizer.render_conversation to avoid runaway lengths.",
    )
    return parser.parse_args()


def compute_percentages(lengths: Iterable[int], thresholds: list[int], total: int) -> list[tuple[int, float, int]]:
    counts = {limit: 0 for limit in thresholds}
    for length in lengths:
        for limit in thresholds:
            if length <= limit:
                counts[limit] += 1
    return [(limit, (counts[limit] / total) * 100.0, counts[limit]) for limit in thresholds]


def main() -> int:
    args = parse_args()
    thresholds = sorted(set(args.thresholds))
    dataset = GSM8KDeepSeekR1(
        split="train",
        path=args.dataset,
        val_fraction=0.0,
    )
    tokenizer = RustBPETokenizer.from_directory(str(args.tokenizer_dir))

    total = dataset.num_examples()
    lengths: list[int] = []
    truncated = 0

    for index in range(total):
        conversation = dataset.get_example(index)
        token_ids, _ = tokenizer.render_conversation(conversation, max_tokens=args.max_tokens)
        if len(token_ids) == args.max_tokens:
            truncated += 1
        lengths.append(len(token_ids))

    percentages = compute_percentages(lengths, thresholds, total)

    print(f"Dataset: {args.dataset}")
    print(f"Tokenizer directory: {args.tokenizer_dir}")
    print(f"Records processed: {total}")
    for limit, pct, count in percentages:
        print(f"<= {limit:5d} tokens: {pct:6.2f}% ({count}/{total})")
    if truncated:
        print(
            f"WARNING: {truncated} samples hit the max_tokens cap of {args.max_tokens}. "
            "Increase --max-tokens for precise counts.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
