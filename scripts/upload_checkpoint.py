#!/usr/bin/env python3
"""
CLI helper to upload nanochat checkpoints to the Hugging Face Hub.

By default the script mirrors the behaviour of maybe_upload_checkpoint by reading
environment variables (HF_UPLOAD_*) via get_hf_upload_config_from_env, but each
option can be overridden on the command line.
"""

import argparse
import sys

from nanochat.checkpoint_manager import (
    get_hf_upload_config_from_env,
    maybe_upload_checkpoint,
)
from nanochat.common import print0


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-or-dir",
        default="prerl",
        help="Checkpoint source key (base|mid|sft|prerl|rl) or path to directory (default: prerl).",
    )
    parser.add_argument("--model-tag", help="Specific model tag (defaults to largest available).")
    parser.add_argument("--step", type=int, help="Specific checkpoint step (defaults to latest).")
    parser.add_argument("--repo-id", help="Override Hugging Face repo id (defaults to env config).")
    parser.add_argument("--branch", help="Override target branch or revision.")
    parser.add_argument("--commit-message", help="Commit message to use for the upload.")
    parser.add_argument(
        "--token",
        help="Hugging Face access token to use (defaults to HF_UPLOAD_TOKEN / HF_TOKEN).",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        dest="allow_patterns",
        help="Additional allow pattern(s). May be supplied multiple times.",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        dest="ignore_patterns",
        help="Additional ignore pattern(s). May be supplied multiple times.",
    )
    parser.add_argument(
        "--private",
        dest="private",
        action="store_true",
        help="Upload to a private repository (overrides env config).",
    )
    parser.add_argument(
        "--public",
        dest="private",
        action="store_false",
        help="Upload to a public repository (overrides env config).",
    )
    parser.add_argument(
        "--create-pr",
        dest="create_pr",
        action="store_true",
        help="Create a PR instead of committing directly.",
    )
    parser.add_argument(
        "--no-create-pr",
        dest="create_pr",
        action="store_false",
        help="Commit directly instead of creating a PR.",
    )
    parser.set_defaults(private=None, create_pr=None)
    return parser.parse_known_args()


def main() -> int:
    args, unknown = parse_args()
    if unknown:
        raise SystemExit(f"Unknown arguments: {' '.join(unknown)}")

    cfg = get_hf_upload_config_from_env()
    if args.repo_id is not None:
        cfg.repo_id = args.repo_id
    if args.branch is not None:
        cfg.branch = args.branch or None
    if args.commit_message is not None:
        cfg.commit_message = args.commit_message or None
    if args.token is not None:
        cfg.token = args.token or None
    if args.private is not None:
        cfg.private = bool(args.private)
    if args.create_pr is not None:
        cfg.create_pr = bool(args.create_pr)

    if not cfg.effective_repo_id():
        raise SystemExit(
            "No Hugging Face repo id specified. Provide --repo-id or set HF_UPLOAD_REPO_ID."
        )

    success = maybe_upload_checkpoint(
        args.source_or_dir,
        model_tag=args.model_tag,
        step=args.step,
        config=cfg,
        allow_patterns=args.allow_patterns,
        ignore_patterns=args.ignore_patterns,
    )
    if success:
        print0(
            f"Uploaded checkpoint from '{args.source_or_dir}' "
            f"to repo '{cfg.effective_repo_id()}'."
        )
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
