#!/usr/bin/env python3
"""
Replace the globally least-used mergeable tokens with new special tokens without
changing the overall vocabulary size.

The script:
1. Iterates the dataset to count token usage.
2. Selects the lowest-frequency mergeable tokens (excluding raw byte tokens).
3. Verifies each candidate is composed of multiple bytes so it can fall back to
   existing smaller tokens, and prints the decomposition.
4. Removes those tokens from the mergeable ranks and reuses their ids for the
   requested special tokens.
5. Updates tokenizer.pkl and token_bytes.pt in-place.

Run inside the project virtualenv, e.g.:
    python tokenizer/replace_rare_tokens.py --tokens <think> </think>
"""

from __future__ import annotations

import argparse
import collections
import os
import pickle
import sys
from typing import Sequence

import tiktoken
import torch

from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched
from nanochat.tokenizer import RustBPETokenizer

DEFAULT_TOKENS: Sequence[str] = ("<think>", "</think>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Swap least-used mergeable tokens for new special tokens."
    )
    parser.add_argument(
        "--tokens",
        nargs="+",
        default=DEFAULT_TOKENS,
        help="Special tokens to add (defaults to <think> </think>)",
    )
    parser.add_argument(
        "--tokenizer-dir",
        default=os.path.expanduser("~/.cache/nanochat/tokenizer"),
        help="Directory containing tokenizer.pkl and token_bytes.pt",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional cap on number of documents to count (useful for dry runs).",
    )
    return parser.parse_args()


def load_encoding(path: str) -> tiktoken.core.Encoding:
    with open(path, "rb") as fh:
        enc = pickle.load(fh)
    if not isinstance(enc, tiktoken.core.Encoding):
        raise TypeError(f"Unexpected object type in {path}: {type(enc)!r}")
    return enc


def count_token_usage(tokenizer: RustBPETokenizer, max_docs: int | None) -> list[int]:
    counts = collections.Counter()
    num_docs = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            num_docs += 1
            ids = tokenizer.encode(doc)
            counts.update(ids)
            if max_docs is not None and num_docs >= max_docs:
                break
        if max_docs is not None and num_docs >= max_docs:
            break
    max_id = tokenizer.enc.max_token_value
    freq = [0] * (max_id + 1)
    for token_id, count in counts.items():
        freq[token_id] = count
    return freq


def choose_tokens_to_replace(
    enc: tiktoken.core.Encoding,
    frequencies: Sequence[int],
    num_required: int,
) -> list[tuple[bytes, int, int]]:
    mergeable_items = sorted(enc._mergeable_ranks.items(), key=lambda kv: kv[1])
    if num_required > len(mergeable_items):
        raise ValueError("Number of tokens to replace exceeds mergeable vocabulary size")

    candidates: list[tuple[bytes, int, int]] = []
    for token_bytes, rank in mergeable_items:
        if rank < 256:
            continue  # keep raw byte tokens intact
        count = frequencies[rank] if rank < len(frequencies) else 0
        if len(token_bytes) <= 1:
            continue  # single-byte tokens have no decomposition
        if not all(bytes([b]) in enc._mergeable_ranks for b in token_bytes):
            continue  # should not happen with byte-level BPE, but guard anyway
        candidates.append((token_bytes, rank, count))

    if len(candidates) < num_required:
        raise RuntimeError(
            f"Not enough eligible tokens to replace (need {num_required}, found {len(candidates)})."
        )

    candidates.sort(key=lambda item: (item[2], item[1]))
    selections = candidates[:num_required]

    print("Lowest-frequency replaceable tokens:")
    for token_bytes, rank, count in selections:
        display = token_bytes.decode("utf-8", errors="backslashreplace")
        print(f"  id={rank:6d} freq={count:8d} repr={display!r}")

    return selections


def print_token_info(enc: tiktoken.core.Encoding, selections: list[tuple[bytes, int, int]]) -> None:
    print("Replacing the following tokens:")
    for token_bytes, rank, count in selections:
        try:
            decoded = token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            decoded = repr(token_bytes)
        fallback_ids = [enc._mergeable_ranks[bytes([b])] for b in token_bytes]
        fallback_str = " ".join(f"{b:#04x}" for b in token_bytes)
        print(
            f"  id={rank:6d} freq={count:8d} token={decoded!r} "
            f"bytes=[{fallback_str}] fallback_ids={fallback_ids}"
        )


def rebuild_encoding(
    enc: tiktoken.core.Encoding,
    selections: list[tuple[bytes, int, int]],
    new_tokens: Sequence[str],
) -> tiktoken.core.Encoding:
    mergeable_ranks = dict(enc._mergeable_ranks)
    free_ids = sorted(rank for _, rank, _ in selections)

    for token_bytes, _, _ in selections:
        mergeable_ranks.pop(token_bytes, None)

    specials = dict(enc._special_tokens)
    for token_str, token_id in zip(new_tokens, free_ids):
        if token_str in specials:
            raise ValueError(f"Special token {token_str!r} already exists")
        specials[token_str] = token_id

    new_enc = tiktoken.Encoding(
        name=enc.name,
        pat_str=enc._pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens=specials,
    )
    return new_enc


def update_token_bytes(
    path: str,
    selections: list[tuple[bytes, int, int]],
) -> None:
    with open(path, "rb") as fh:
        token_bytes = torch.load(fh, map_location="cpu")

    for _, rank, _ in selections:
        token_bytes[rank] = 0

    with open(path, "wb") as fh:
        torch.save(token_bytes, fh)


def main() -> None:
    args = parse_args()
    tokenizer_dir = args.tokenizer_dir
    tokenizer_pkl = os.path.join(tokenizer_dir, "tokenizer.pkl")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")

    if not os.path.exists(tokenizer_pkl):
        sys.exit(f"tokenizer.pkl not found at {tokenizer_pkl}")
    if not os.path.exists(token_bytes_path):
        sys.exit(f"token_bytes.pt not found at {token_bytes_path}")

    enc = load_encoding(tokenizer_pkl)
    tokenizer = RustBPETokenizer(enc, "<|bos|>")
    frequencies = count_token_usage(tokenizer, args.max_docs)

    selections = choose_tokens_to_replace(enc, frequencies, len(args.tokens))
    print_token_info(enc, selections)

    new_enc = rebuild_encoding(enc, selections, args.tokens)
    with open(tokenizer_pkl, "wb") as fh:
        pickle.dump(new_enc, fh)

    update_token_bytes(token_bytes_path, selections)

    base_dir = get_base_dir()
    print(
        f"Replaced {len(selections)} tokens with {len(args.tokens)} new specials. "
        f"Tokenizer assets updated in {tokenizer_dir}. "
        f"Project base dir: {base_dir}"
    )


if __name__ == "__main__":
    main()
