#!/usr/bin/env python3
"""
Replace low-frequency mergeable tokens with the <think> and </think> markers.

The script replays the tokenizer training corpus (FineWeb EDU shards under
~/.cache/nanochat/base_data by default), counts token usage, and repurposes
the least-used mergeable ids as special tokens while keeping the vocab size
unchanged.
"""

from __future__ import annotations

import argparse
import pickle
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import torch
import tiktoken
import pyarrow.parquet as pq

from nanochat.tokenizer import RustBPETokenizer
from nanochat.dataset import list_parquet_files

DEFAULT_TOKENS: Sequence[str] = ("<think>", "</think>")
BYTE_LEVEL_VOCAB = 256  # First 256 mergeable tokens are 1-byte symbols; never repurpose them.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repurpose low-frequency tokens to serve as <think> special markers."
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path.home() / ".cache" / "nanochat" / "tokenizer",
        help="Directory containing tokenizer.pkl and token_bytes.pt",
    )
    parser.add_argument(
        "--tokens",
        nargs=2,
        default=list(DEFAULT_TOKENS),
        help="Special tokens to insert (defaults to <think> </think>).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2_000_000_000,
        help="Maximum number of characters to scan from the tokenizer training corpus.",
    )
    parser.add_argument(
        "--doc-cap",
        type=int,
        default=10_000,
        help="Crop each document to at most this many characters (mirrors tok_train.py).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / ".cache" / "nanochat" / "base_data",
        help="Directory containing the tokenizer training parquet shards.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="train",
        help="Dataset split to sample from when computing token frequencies.",
    )
    return parser.parse_args()


def iter_training_documents(data_dir: Path, split: str) -> Iterable[str]:
    parquet_paths = list_parquet_files(str(data_dir))
    if not parquet_paths:
        raise FileNotFoundError(
            f"No parquet shards found under {data_dir}. "
            "Download the tokenizer training dataset first."
        )
    if split == "train":
        parquet_paths = parquet_paths[:-1] or parquet_paths
    else:
        parquet_paths = parquet_paths[-1:]

    for path in parquet_paths:
        pf = pq.ParquetFile(path)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx, columns=["text"])
            for text in rg.column("text").to_pylist():
                yield text


def compute_token_frequencies(
    tokenizer: RustBPETokenizer,
    data_dir: Path,
    split: str,
    max_chars: int,
    doc_cap: int,
) -> Counter[int]:
    counts: Counter[int] = Counter()
    nchars = 0
    ndocs = 0

    for doc in iter_training_documents(data_dir, split):
        if doc_cap > 0 and len(doc) > doc_cap:
            doc = doc[:doc_cap]
        nchars += len(doc)
        token_ids = tokenizer.encode(doc)
        counts.update(token_ids)
        ndocs += 1
        if nchars >= max_chars:
            break
    return counts, nchars, ndocs


def find_replacement_ids(
    mergeable_ranks: dict[bytes, int],
    special_tokens: dict[str, int],
    frequencies: Counter[int],
    num_needed: int,
) -> list[int]:
    reserved_ids = set(special_tokens.values())
    candidates: list[int] = []
    for token_bytes, token_id in mergeable_ranks.items():
        if token_id < BYTE_LEVEL_VOCAB:
            continue  # keep byte-level primitives intact
        if token_id in reserved_ids:
            continue
        candidates.append(token_id)

    if len(candidates) < num_needed:
        raise RuntimeError(f"Only {len(candidates)} eligible tokens available, need {num_needed}.")

    candidates.sort(key=lambda tid: (frequencies.get(tid, 0), tid))
    return candidates[:num_needed]


def update_token_bytes(path: Path, replacement_ids: Sequence[int], tokens: Sequence[str]) -> None:
    with path.open("rb") as fh:
        token_bytes = torch.load(fh, map_location="cpu")

    for token_id, token in zip(replacement_ids, tokens):
        token_bytes[token_id] = len(token.encode("utf-8"))

    with path.open("wb") as fh:
        torch.save(token_bytes, fh)


def main() -> int:
    args = parse_args()
    tokenizer_pkl = args.tokenizer_dir / "tokenizer.pkl"
    token_bytes_path = args.tokenizer_dir / "token_bytes.pt"

    if not tokenizer_pkl.exists():
        raise FileNotFoundError(f"tokenizer.pkl not found at {tokenizer_pkl}")
    if not token_bytes_path.exists():
        raise FileNotFoundError(f"token_bytes.pt not found at {token_bytes_path}")

    tokenizer = RustBPETokenizer.from_directory(str(args.tokenizer_dir))
    enc = tokenizer.enc
    existing_specials = enc.special_tokens_set

    missing_tokens = [tok for tok in args.tokens if tok not in existing_specials]
    if not missing_tokens:
        print("Tokenizer already contains requested special tokens; nothing to do.")
        return 0
    if len(missing_tokens) != len(args.tokens):
        raise RuntimeError(
            f"Tokenizer already defines some of {args.tokens}; please remove them first."
        )

    data_dir = args.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Tokenizer training data directory not found: {data_dir}. "
            "Download the tokenizer corpus before running this script."
        )

    frequencies, scanned_chars, scanned_docs = compute_token_frequencies(
        tokenizer,
        data_dir,
        args.split,
        args.max_chars,
        args.doc_cap,
    )
    print(
        f"Collected token statistics over {scanned_docs:,} documents "
        f"({scanned_chars:,} characters)."
    )

    mergeable_ranks = dict(enc._mergeable_ranks)
    special_tokens = dict(enc._special_tokens)
    replacement_ids = find_replacement_ids(
        mergeable_ranks,
        special_tokens,
        frequencies,
        num_needed=len(args.tokens),
    )

    id_to_bytes = {rank: token_bytes for token_bytes, rank in mergeable_ranks.items()}
    replaced_tokens = [(token_id, id_to_bytes[token_id]) for token_id in replacement_ids]
    for token_id, token_bytes in replaced_tokens:
        mergeable_ranks.pop(token_bytes, None)

    # Ensure deterministic ordering when rebuilding the encoding
    ordered_mergeables = dict(sorted(mergeable_ranks.items(), key=lambda kv: kv[1]))

    for token, token_id in zip(args.tokens, replacement_ids):
        special_tokens[token] = token_id

    new_enc = tiktoken.Encoding(
        name=enc.name,
        pat_str=enc._pat_str,
        mergeable_ranks=ordered_mergeables,
        special_tokens=special_tokens,
    )

    with tokenizer_pkl.open("wb") as fh:
        pickle.dump(new_enc, fh)

    update_token_bytes(token_bytes_path, replacement_ids, args.tokens)

    for (token_id, raw_bytes), token in zip(replaced_tokens, args.tokens):
        count = frequencies.get(token_id, 0)
        printable = raw_bytes.decode("utf-8", errors="replace")
        print(
            f"Replaced token id {token_id} (freq={count}, bytes={printable!r}) "
            f"with special token {token!r}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
