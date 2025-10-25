"""
Utilities for saving and loading model/optim/state checkpoints.
"""
import os
import re
import glob
import json
import logging
import torch
from dataclasses import dataclass
from typing import Optional, Sequence

from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import setup_default_logging
from huggingface_hub import HfApi

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)

MODEL_SOURCE_DIRS = {
    "base": "base_checkpoints",
    "mid": "mid_checkpoints",
    "sft": "chatsft_checkpoints",
    "prerl": "chatrl_pretrain_checkpoints",
    "rl": "chatrl_checkpoints",
}

def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}

def _split_env_list(value: str) -> Optional[list[str]]:
    if not value:
        return None
    items = [part.strip() for part in value.split(",")]
    items = [item for item in items if item]
    return items or None

@dataclass
class HFUploadConfig:
    repo_id: str = ""
    branch: Optional[str] = None
    commit_message: Optional[str] = None
    private: bool = False
    create_pr: bool = False
    allow_patterns: Optional[Sequence[str]] = None
    ignore_patterns: Optional[Sequence[str]] = None
    token: Optional[str] = None
    upload_every_save: bool = False

    def effective_repo_id(self) -> str:
        return self.repo_id.strip()

def get_hf_upload_config_from_env() -> HFUploadConfig:
    """
    Snapshot the Hugging Face upload related environment variables.
    """
    repo_id = os.environ.get("HF_UPLOAD_REPO_ID", "").strip()
    branch = os.environ.get("HF_UPLOAD_BRANCH", "").strip() or None
    commit_message = os.environ.get("HF_UPLOAD_COMMIT_MESSAGE", "").strip() or None
    allow_patterns = _split_env_list(os.environ.get("HF_UPLOAD_ALLOW_PATTERNS", ""))
    ignore_patterns = _split_env_list(os.environ.get("HF_UPLOAD_IGNORE_PATTERNS", ""))
    token = os.environ.get("HF_UPLOAD_TOKEN") or os.environ.get("HF_TOKEN")
    return HFUploadConfig(
        repo_id=repo_id,
        branch=branch,
        commit_message=commit_message,
        private=_env_flag("HF_UPLOAD_PRIVATE", False),
        create_pr=_env_flag("HF_UPLOAD_CREATE_PR", False),
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        token=token if token else None,
        upload_every_save=_env_flag("HF_UPLOAD_EVERY_SAVE", False),
    )

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data):
    assert int(os.environ.get('RANK', 0)) == 0 # prevent footguns for now
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Save the model state (parameters)
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    torch.save(model_data, model_path)
    log0(f"Saved model file to: {model_path}")
    # Save the optimizer state (useful for SFT or any other fine-tuning)
    if optimizer_data is not None:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}.pt")
        torch.save(optimizer_data, optimizer_path)
        log0(f"Saved optimizer file to: {optimizer_path}")
    # Save the metadata dict as json
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "w") as f:
        json.dump(meta_data, f, indent=2)
    log0(f"Saved metadata file to: {meta_path}")


def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False):
    # Load the model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    # Load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.lstrip("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    with torch.device("meta"):
        model = GPT(model_config)
    # Load the model state
    model.to_empty(device=device)
    model.init_weights() # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    # Load the Tokenizer
    tokenizer = get_tokenizer()
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"]
    return model, tokenizer, meta_data


def find_largest_model(checkpoint_dir):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
    return last_step

# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta_data

def load_model(source, *args, **kwargs):
    model_dir = MODEL_SOURCE_DIRS[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)


def upload_model_to_hf(
    source_or_dir,
    repo_id,
    model_tag=None,
    step=None,
    *,
    token=None,
    private=False,
    branch=None,
    commit_message=None,
    allow_patterns=None,
    ignore_patterns=None,
    create_pr=False,
):
    """
    Upload a checkpoint to the Hugging Face Hub.

    Parameters
    ----------
    source_or_dir : str
        Either a nanochat source key (\"base\", \"mid\", \"sft\", \"rl\") or a path
        to the directory that contains model checkpoints.
    repo_id : str
        Repository identifier on the Hugging Face Hub (e.g. \"username/model-name\").
    model_tag : str, optional
        Specific model subdirectory inside ``source_or_dir``; defaults to the largest model.
    step : int, optional
        Training step to upload; defaults to the last available step.
    token : str, optional
        Hugging Face access token. If omitted, the hub library uses the cached token.
    private : bool, optional
        Whether to create the repository as private when it does not exist yet.
    branch : str, optional
        Target branch or revision on the Hub. Defaults to the repository's default branch.
    commit_message : str, optional
        Custom commit message for the upload.
    allow_patterns : Union[str, list[str]], optional
        Glob patterns selecting which files to upload. Defaults to the chosen checkpoint files.
    ignore_patterns : Union[str, list[str]], optional
        Glob patterns for files that should be ignored during the upload.
    create_pr : bool, optional
        Whether to open a pull request instead of committing directly.
    """
    assert int(os.environ.get("RANK", 0)) == 0, "Only rank 0 should upload checkpoints"
    if not repo_id:
        raise ValueError("repo_id must be provided (e.g. 'username/model-name').")

    base_dir = get_base_dir()
    if source_or_dir in MODEL_SOURCE_DIRS:
        checkpoints_dir = os.path.join(base_dir, MODEL_SOURCE_DIRS[source_or_dir])
    else:
        checkpoints_dir = source_or_dir
        if not os.path.isabs(checkpoints_dir):
            checkpoints_dir = os.path.join(base_dir, checkpoints_dir)
    if not os.path.isdir(checkpoints_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoints_dir}")

    if model_tag is None:
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, defaulting to largest model: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Model directory not found: {checkpoint_dir}")

    if step is None:
        step = find_last_step(checkpoint_dir)
        log0(f"No step provided, defaulting to last step: {step}")
    step_str = f"{step:06d}"

    checkpoint_files = sorted(
        glob.glob(os.path.join(checkpoint_dir, f"*_{step_str}.*"))
    )
    if not checkpoint_files:
        raise FileNotFoundError(
            f"No files for step {step} found in {checkpoint_dir}. "
            "Expected names like model_{step:06d}.pt."
        )

    default_allow = [os.path.basename(path) for path in checkpoint_files]
    effective_allow = default_allow
    if allow_patterns is not None:
        if isinstance(allow_patterns, str):
            allow_patterns = [allow_patterns]
        # Merge caller-provided patterns with defaults while preserving order.
        seen = set()
        effective_allow = [
            pattern
            for pattern in allow_patterns + default_allow
            if not (pattern in seen or seen.add(pattern))
        ]

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )

    path_in_repo = f"{model_tag}/step_{step_str}"
    commit_message = (
        commit_message
        if commit_message is not None
        else f"Upload checkpoint {model_tag} step {step_str}"
    )
    log0(
        f"Uploading checkpoint files {effective_allow} from {checkpoint_dir} "
        f"to {repo_id}:{path_in_repo} "
        f"(branch: {branch or 'default'})"
    )
    api.upload_folder(
        folder_path=checkpoint_dir,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=effective_allow,
        ignore_patterns=ignore_patterns,
        commit_message=commit_message,
        create_pr=create_pr,
        revision=branch,
    )
    log0(f"Upload complete for {repo_id}:{path_in_repo}")


def maybe_upload_checkpoint(
    source_or_dir,
    model_tag,
    step,
    *,
    config: Optional[HFUploadConfig] = None,
    default_commit_message: Optional[str] = None,
    allow_patterns: Optional[Sequence[str]] = None,
    ignore_patterns: Optional[Sequence[str]] = None,
) -> bool:
    """
    Convenience helper that reads Hugging Face upload settings and performs the upload.

    Returns True when an upload was attempted, False otherwise.
    """
    cfg = config or get_hf_upload_config_from_env()
    repo_id = cfg.effective_repo_id()
    if not repo_id:
        return False
    commit_message = cfg.commit_message if cfg.commit_message is not None else default_commit_message
    try:
        resolved_allow = allow_patterns or cfg.allow_patterns
        resolved_ignore = ignore_patterns or cfg.ignore_patterns
        if resolved_allow is not None:
            resolved_allow = list(resolved_allow)
        if resolved_ignore is not None:
            resolved_ignore = list(resolved_ignore)
        upload_model_to_hf(
            source_or_dir,
            repo_id=repo_id,
            model_tag=model_tag,
            step=step,
            token=cfg.token,
            private=cfg.private,
            branch=cfg.branch,
            commit_message=commit_message,
            allow_patterns=resolved_allow,
            ignore_patterns=resolved_ignore,
            create_pr=cfg.create_pr,
        )
        log0(f"Uploaded checkpoint {model_tag} step {step:06d} to Hugging Face repo {repo_id}.")
        return True
    except Exception as exc:
        log0(f"Failed to upload checkpoint to Hugging Face: {exc}")
        return False
