#!/usr/bin/env python3
"""
Extend the cached nanoChat tokenizer by appending the <think> and </think>
special tokens without retraining the BPE merges.

This script rewrites both tokenizer.pkl (a pickled tiktoken Encoding) and
token_bytes.pt (per-token UTF-8 byte counts) in-place. Run it once after
activating the project's virtualenv.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import Iterable, Sequence

import tiktoken

try:
    import torch
except Exception as exc:  # pragma: no cover - defensive guard for broken torch builds
    torch = None
    _torch_error = exc
else:
    _torch_error = None


DEFAULT_TOKENS: Sequence[str] = ("<think>", "</think>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append new special tokens to cached nanoChat tokenizer assets."
    )
    parser.add_argument(
        "--tokenizer-dir",
        default=os.path.expanduser("~/.cache/nanochat/tokenizer"),
        help="Directory containing tokenizer.pkl and token_bytes.pt",
    )
    parser.add_argument(
        "--tokens",
        nargs="+",
        default=DEFAULT_TOKENS,
        help="Special tokens to append (defaults to <think> </think>)",
    )
    return parser.parse_args()


def load_encoding(path: str) -> tiktoken.core.Encoding:
    with open(path, "rb") as fh:
        enc = pickle.load(fh)
    if not isinstance(enc, tiktoken.core.Encoding):
        raise TypeError(f"Unexpected object type in {path}: {type(enc)!r}")
    return enc


def extend_encoding(enc: tiktoken.core.Encoding, tokens: Iterable[str]) -> tiktoken.core.Encoding:
    mergeable = dict(enc._mergeable_ranks)
    specials = dict(enc._special_tokens)
    next_id = len(mergeable) + len(specials)

    for i, token in enumerate(tokens):
        if token in specials:
            raise ValueError(f"Token {token!r} already exists with id {specials[token]}")
        specials[token] = next_id + i

    return tiktoken.Encoding(
        name=enc.name,
        pat_str=enc._pat_str,
        mergeable_ranks=mergeable,
        special_tokens=specials,
    )


def update_token_bytes(path: str, additions: int, original_vocab_size: int, target_vocab_size: int) -> None:
    if torch is None:
        raise RuntimeError(f"torch is required to update {path} ({_torch_error})")

    with open(path, "rb") as fh:
        token_bytes = torch.load(fh, map_location="cpu")

    if token_bytes.numel() != original_vocab_size:
        raise ValueError(
            f"Expected {original_vocab_size} entries in {path}, found {token_bytes.numel()}"
        )

    padding = torch.zeros(additions, dtype=token_bytes.dtype)
    token_bytes = torch.cat((token_bytes, padding), dim=0)

    if token_bytes.numel() != target_vocab_size:
        raise AssertionError("Token bytes tensor has unexpected length after extension")

    with open(path, "wb") as fh:
        torch.save(token_bytes, fh)


def main() -> None:
    args = parse_args()
    tokenizer_pkl = os.path.join(args.tokenizer_dir, "tokenizer.pkl")
    token_bytes_path = os.path.join(args.tokenizer_dir, "token_bytes.pt")

    if not os.path.exists(tokenizer_pkl):
        sys.exit(f"tokenizer.pkl not found at {tokenizer_pkl}")
    if not os.path.exists(token_bytes_path):
        sys.exit(f"token_bytes.pt not found at {token_bytes_path}")

    tokens = tuple(args.tokens)
    enc = load_encoding(tokenizer_pkl)
    print("SPECIAL TOKENS BEFORE")
    print(enc.special_tokens_set)

    original_vocab_size = enc.n_vocab
    new_enc = extend_encoding(enc, tokens)
    target_vocab_size = new_enc.n_vocab
    print("SPECIAL TOKENS BEFORE")
    print(new_enc.special_tokens_set)

    with open(tokenizer_pkl, "wb") as fh:
        pickle.dump(new_enc, fh)

    update_token_bytes(
        token_bytes_path,
        additions=len(tokens),
        original_vocab_size=original_vocab_size,
        target_vocab_size=target_vocab_size,
    )

    print(
        f"Added {len(tokens)} special tokens. "
        f"Vocab size {original_vocab_size} -> {target_vocab_size}."
    )


if __name__ == "__main__":
    main()
