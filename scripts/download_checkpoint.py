#!/usr/bin/env python3
"""
CLI helper to download nanochat checkpoints from the Hugging Face Hub.

Downloads a single model_tag/step pair into the local checkpoints directory
structure so that load_model(...) can pick it up afterwards.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

from huggingface_hub import HfApi, snapshot_download

from nanochat.checkpoint_manager import MODEL_SOURCE_DIRS
from nanochat.common import get_base_dir, print0


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face repository identifier (e.g. username/model-name).",
    )
    parser.add_argument(
        "--source-or-dir",
        default="prerl",
        help="Checkpoint destination. Either a source key (base|mid|sft|prerl|rl) "
        "or a filesystem path (default: prerl).",
    )
    parser.add_argument(
        "--model-tag",
        help="Specific model tag to download (defaults to the largest tag found).",
    )
    parser.add_argument(
        "--step",
        type=int,
        help="Specific checkpoint step to download (defaults to latest available).",
    )
    parser.add_argument("--branch", help="Target branch or revision on the repo.")
    parser.add_argument(
        "--token",
        help="Hugging Face access token (defaults to HF_TOKEN / cached credentials).",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        dest="allow_patterns",
        help="Additional allow pattern(s). May be provided multiple times.",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        dest="ignore_patterns",
        help="Additional ignore pattern(s). May be provided multiple times.",
    )
    return parser.parse_known_args()


def resolve_destination(source_or_dir: str) -> str:
    base_dir = get_base_dir()
    if source_or_dir in MODEL_SOURCE_DIRS:
        return os.path.join(base_dir, MODEL_SOURCE_DIRS[source_or_dir])
    if not os.path.isabs(source_or_dir):
        return os.path.join(base_dir, source_or_dir)
    return source_or_dir


def select_model_tag_and_step(
    api: HfApi,
    repo_id: str,
    revision: str | None,
    model_tag: str | None,
    step: int | None,
) -> tuple[str, int]:
    try:
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="model", revision=revision)
    except Exception as exc:
        raise RuntimeError(f"Failed to list files for repo {repo_id}: {exc}") from exc

    structure: dict[str, set[int]] = {}
    pattern = re.compile(r"^(?P<tag>[^/]+)/step_(?P<step>\d{6})/model_\d{6}\.pt$")
    for path in repo_files:
        match = pattern.match(path)
        if not match:
            continue
        tag = match.group("tag")
        step_value = int(match.group("step"))
        structure.setdefault(tag, set()).add(step_value)

    if not structure:
        raise FileNotFoundError(
            f"No checkpoint files found in repo {repo_id}. "
            "Expected paths like <model_tag>/step_000123/model_000123.pt."
        )

    selected_tag = model_tag
    if selected_tag is None:
        candidates = []
        for tag in structure:
            depth_match = re.match(r"d(\d+)", tag)
            if depth_match:
                candidates.append((int(depth_match.group(1)), tag))
        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            selected_tag = candidates[0][1]
        else:
            selected_tag = sorted(structure.keys())[-1]
        print0(f"Selected model tag '{selected_tag}' from repo {repo_id}.")
    if selected_tag not in structure:
        raise FileNotFoundError(
            f"Model tag '{selected_tag}' not found in repo {repo_id}. "
            f"Available tags: {sorted(structure.keys())}"
        )

    available_steps = sorted(structure[selected_tag])
    if not available_steps:
        raise FileNotFoundError(
            f"No steps found for model tag '{selected_tag}' in repo {repo_id}."
        )

    selected_step = step
    if selected_step is None:
        selected_step = available_steps[-1]
        print0(f"Selected step {selected_step:06d} for model tag '{selected_tag}'.")
    elif selected_step not in structure[selected_tag]:
        raise FileNotFoundError(
            f"Step {selected_step:06d} not available for model tag '{selected_tag}' in repo {repo_id}. "
            f"Available steps: {[f'{value:06d}' for value in available_steps]}"
        )

    return selected_tag, selected_step


def download_checkpoint(
    repo_id: str,
    source_or_dir: str,
    model_tag: str | None,
    step: int | None,
    token: str | None,
    branch: str | None,
    allow_patterns: list[str] | None,
    ignore_patterns: list[str] | None,
) -> tuple[str, int]:
    if not repo_id:
        raise ValueError("repo_id must be provided to download a checkpoint.")

    api = HfApi(token=token)
    revision = branch or None
    selected_tag, selected_step = select_model_tag_and_step(api, repo_id, revision, model_tag, step)
    step_str = f"{selected_step:06d}"

    destination_root = resolve_destination(source_or_dir)
    os.makedirs(destination_root, exist_ok=True)
    destination_dir = os.path.join(destination_root, selected_tag)
    os.makedirs(destination_dir, exist_ok=True)

    allow = [f"{selected_tag}/step_{step_str}/*"]
    if allow_patterns:
        allow = list(allow_patterns) + allow
    ignore = list(ignore_patterns) if ignore_patterns else None

    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        token=token,
        allow_patterns=allow,
        ignore_patterns=ignore,
    )

    source_dir = os.path.join(snapshot_path, selected_tag, f"step_{step_str}")
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(
            f"Downloaded snapshot is missing expected directory: {source_dir}"
        )

    for filename in os.listdir(source_dir):
        shutil.copy2(os.path.join(source_dir, filename), os.path.join(destination_dir, filename))

    print0(
        f"Downloaded checkpoint '{selected_tag}' step {step_str} "
        f"from {repo_id} into {destination_dir}."
    )
    return selected_tag, selected_step


def main() -> int:
    args, unknown = parse_args()
    if unknown:
        raise SystemExit(f"Unknown arguments: {' '.join(unknown)}")

    model_tag, step = download_checkpoint(
        repo_id=args.repo_id,
        source_or_dir=args.source_or_dir,
        model_tag=args.model_tag,
        step=args.step,
        token=args.token,
        branch=args.branch,
        allow_patterns=args.allow_patterns,
        ignore_patterns=args.ignore_patterns,
    )
    print0(
        f"Ready to load checkpoint '{model_tag}' step {step:06d} "
        f"from {resolve_destination(args.source_or_dir)}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
