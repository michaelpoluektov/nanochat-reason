"""
Prepare GSM8K difficulty splits and push them to the Hugging Face Hub.

This script downloads the lime-nlp/GSM8K_Difficulty dataset, produces a
deterministic 80/20 train/test split that remains balanced across the full
difficulty range (easiest to hardest), and then uploads the resulting
DatasetDict to a new repository.

Example:
    python -m scripts.prepare_gsm8k_split \
        --subset main \
        --train-ratio 0.8

You must be logged in (`huggingface-cli login`) or provide a token via
`--hf-token` or the HF_TOKEN/HUGGINGFACEHUB_API_TOKEN environment variables.
"""

import argparse
from typing import Sequence

from datasets import DatasetDict, load_dataset

SOURCE_DATASET = "lime-nlp/GSM8K_Difficulty"
TARGET_DATASET = "accountblabla/gsm8k-sorted"
SOURCE_CONFIG_MAP = {
    "main": "Difficulty Score",
    "socratic": "Response",
}
SOLVED_COLUMN = "solved_percentage"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dataset",
        default=SOURCE_DATASET,
        help=f"Dataset to download (default: {SOURCE_DATASET}).",
    )
    parser.add_argument(
        "--subset",
        default="main",
        help="Subset name to load (default: main).",
    )
    parser.add_argument(
        "--target-repo-id",
        default=TARGET_DATASET,
        help=f"Destination Hugging Face dataset repo id (default: {TARGET_DATASET}).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of examples to place in the train split (default: 0.8).",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face access token; defaults to environment token.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Push the generated dataset as a private repo.",
    )
    parser.add_argument(
        "--commit-message",
        default="Add deterministic GSM8K difficulty split",
        help="Commit message used when pushing to the hub.",
    )
    return parser.parse_args()


def parse_solved_percentage(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.endswith("%"):
            stripped = stripped[:-1]
        try:
            return float(stripped)
        except ValueError:
            return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def sort_by_difficulty(dataset) -> list[tuple[int, float]]:
    pairs: list[tuple[int, float]] = []
    solved_col = SOLVED_COLUMN if SOLVED_COLUMN in dataset.column_names else None
    for idx in range(len(dataset)):
        solved_value = dataset[idx][solved_col] if solved_col else None
        difficulty = parse_solved_percentage(solved_value)
        pairs.append((idx, difficulty))
    pairs.sort(key=lambda item: item[1], reverse=True)
    return pairs


def split_indices(
    sorted_pairs: Sequence[tuple[int, float]],
    train_ratio: float,
) -> tuple[list[int], list[int]]:
    total = len(sorted_pairs)
    if total == 0:
        return [], []
    train_ratio = max(0.0, min(1.0, train_ratio))
    desired_train = int(round(total * train_ratio))
    if total > 1:
        desired_train = max(1, min(total - 1, desired_train))
    desired_train = max(1, desired_train)
    desired_test = total - desired_train
    if total > 1 and desired_test == 0:
        desired_train -= 1
        desired_test += 1
    train_indices: list[int] = []
    test_indices: list[int] = []
    for index, _ in sorted_pairs:
        current_total = len(train_indices) + len(test_indices)
        if current_total == 0:
            assign_to_train = desired_train > 0
        else:
            if len(train_indices) >= desired_train:
                assign_to_train = False
            elif len(test_indices) >= desired_test:
                assign_to_train = True
            else:
                ratio_if_train = (len(train_indices) + 1) / (current_total + 1)
                ratio_if_test = len(train_indices) / (current_total + 1)
                diff_train = abs(ratio_if_train - train_ratio)
                diff_test = abs(ratio_if_test - train_ratio)
                assign_to_train = diff_train <= diff_test
        if assign_to_train:
            train_indices.append(index)
        else:
            test_indices.append(index)
    if not test_indices and train_indices:
        test_indices.append(train_indices.pop())
    if not train_indices and test_indices:
        train_indices.append(test_indices.pop())
    return train_indices, test_indices


def standardize_columns(dataset):
    rename_map = {}
    if "problem" in dataset.column_names and "question" not in dataset.column_names:
        rename_map["problem"] = "question"
    if "ground_truth" in dataset.column_names and "answer" not in dataset.column_names:
        rename_map["ground_truth"] = "answer"
    if rename_map:
        dataset = dataset.rename_columns(rename_map)
    return dataset


def main():
    args = parse_args()
    source_config = SOURCE_CONFIG_MAP.get(args.subset, args.subset if args.subset else None)
    load_kwargs = {}
    if source_config:
        load_kwargs["name"] = source_config
    dataset = load_dataset(args.source_dataset, split="train", **load_kwargs)
    dataset = standardize_columns(dataset)
    sorted_pairs = sort_by_difficulty(dataset)
    train_idx, test_idx = split_indices(sorted_pairs, args.train_ratio)
    train_dataset = dataset.select(train_idx)
    test_dataset = dataset.select(test_idx)
    ds_dict = DatasetDict(
        {
            "train": train_dataset,
            "test": test_dataset,
        }
    )
    print(f"Prepared dataset with {len(train_dataset)} train and {len(test_dataset)} test examples.")
    ds_dict.push_to_hub(
        repo_id=args.target_repo_id,
        token=args.hf_token,
        config_name=args.subset,
        private=args.private,
        commit_message=args.commit_message,
    )
    print(f"✅ Uploaded dataset to {args.target_repo_id}")


if __name__ == "__main__":
    main()
