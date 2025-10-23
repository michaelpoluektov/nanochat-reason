#!/usr/bin/env python3
"""
Remove previously appended <think> and </think> special tokens from the cached
nanoChat tokenizer assets.

This restores tokenizer.pkl and token_bytes.pt to their pre-extension state,
assuming the tokens were added by tokenizer/add_think_tokens.py (i.e. they live
at the tail end of the vocabulary).
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
        description="Remove special tokens that were appended to nanoChat tokenizer assets."
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
        help="Special tokens to remove (defaults to <think> </think>)",
    )
    return parser.parse_args()


def load_encoding(path: str) -> tiktoken.core.Encoding:
    with open(path, "rb") as fh:
        enc = pickle.load(fh)
    if not isinstance(enc, tiktoken.core.Encoding):
        raise TypeError(f"Unexpected object type in {path}: {type(enc)!r}")
    return enc


def strip_encoding(enc: tiktoken.core.Encoding, tokens: Iterable[str]) -> tiktoken.core.Encoding:
    mergeable = dict(enc._mergeable_ranks)
    specials = dict(enc._special_tokens)
    token_ids = []
    for token in tokens:
        if token not in specials:
            raise ValueError(f"Token {token!r} does not exist in special tokens")
        token_ids.append(specials[token])

    tail_ids = list(range(enc.n_vocab - len(token_ids), enc.n_vocab))
    if sorted(token_ids) != tail_ids:
        raise ValueError(
            "Tokens to remove are not the last entries in the vocabulary; "
            "refusing to mutate tokenizer."
        )

    for token in tokens:
        specials.pop(token, None)

    return tiktoken.Encoding(
        name=enc.name,
        pat_str=enc._pat_str,
        mergeable_ranks=mergeable,
        special_tokens=specials,
    )


def trim_token_bytes(path: str, removals: int, original_vocab_size: int, target_vocab_size: int) -> None:
    if torch is None:
        raise RuntimeError(f"torch is required to update {path} ({_torch_error})")

    with open(path, "rb") as fh:
        token_bytes = torch.load(fh, map_location="cpu")

    if token_bytes.numel() != original_vocab_size:
        raise ValueError(
            f"Expected {original_vocab_size} entries in {path}, found {token_bytes.numel()}"
        )

    tail = token_bytes[-removals:]
    if not torch.all(tail == 0):
        raise ValueError(
            "Tail entries in token_bytes.pt are not zero; refusing to trim "
            "because the vocabulary layout looks unexpected."
        )

    token_bytes = token_bytes[:-removals]

    if token_bytes.numel() != target_vocab_size:
        raise AssertionError("Token bytes tensor has unexpected length after trimming")

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

    original_vocab_size = enc.n_vocab
    new_enc = strip_encoding(enc, tokens)
    target_vocab_size = new_enc.n_vocab

    with open(tokenizer_pkl, "wb") as fh:
        pickle.dump(new_enc, fh)

    trim_token_bytes(
        token_bytes_path,
        removals=len(tokens),
        original_vocab_size=original_vocab_size,
        target_vocab_size=target_vocab_size,
    )

    print(
        f"Removed {len(tokens)} special tokens. "
        f"Vocab size {original_vocab_size} -> {target_vocab_size}."
    )


if __name__ == "__main__":
    main()
