"""
Dataset wrapper for DeepSeek's GSM8K reasoning traces.

By default the JSON payload is expected at
``~/.cache/nanochat/datasets/gsm8k_deepseek/gsm8k_deepseek_R1_7148.json`` (or
``$NANOCHAT_BASE_DIR`` if set). It consists of records with ``instruction``
(the math word problem), optional ``input``, and ``output`` containing the full
chain-of-thought style response (complete with ``<think>...</think>`` markers).

We adapt this into the Task interface so that training scripts can reuse the
standard conversation rendering utilities.
"""

import json
import random
from pathlib import Path

from tasks.common import Task
from nanochat.common import get_base_dir


class GSM8KDeepSeekR1(Task):
    """
    Lightweight dataset backed by the DeepSeek GSM8K JSON dump.

    Parameters
    ----------
    split : str
        Either ``"train"`` or ``"val"``. A deterministic shuffle is applied
        first, and then we take a prefix for validation and the remainder for
        training.
    path : Union[str, Path], optional
        Optional override for the dataset location.
    val_fraction : float, optional
        Fraction of the dataset to reserve for validation. Defaults to 0.05.
    seed : int, optional
        RNG seed that controls the shuffle and split.
    """

    def __init__(
        self,
        split: str,
        path=None,
        *,
        val_fraction: float = 0.05,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert split in ["train", "val"], "split must be 'train' or 'val'"
        assert 0.0 <= val_fraction < 1.0, "val_fraction must be in [0.0, 1.0)"

        if path is None:
            base_dir = Path(get_base_dir())
            path = base_dir / "datasets" / "gsm8k_deepseek" / "gsm8k_deepseek_R1_7148.json"
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            raw_records = json.load(f)
        if not isinstance(raw_records, list):
            raise ValueError(f"Expected list of records in {path}, got {type(raw_records)}")
        if len(raw_records) == 0:
            raise ValueError(f"Dataset {path} is empty")

        rng = random.Random(seed)
        indices = list(range(len(raw_records)))
        rng.shuffle(indices)

        val_count = int(len(indices) * val_fraction)
        if val_fraction > 0.0 and val_count == 0:
            val_count = 1

        if split == "val":
            chosen_indices = indices[:val_count]
        else:
            chosen_indices = indices[val_count:]
        self.examples = [raw_records[i] for i in chosen_indices]

    def num_examples(self):
        return len(self.examples)

    def get_example(self, index):
        record = self.examples[index]
        instruction = (record.get("instruction") or "").strip()
        supplementary = (record.get("input") or "").strip()
        user_parts = [part for part in [instruction, supplementary] if part]
        if not user_parts:
            raise ValueError(f"Record {index} is missing both instruction and input")
        user_message = "\n\n".join(user_parts)

        assistant_message = (record.get("output") or "").strip()
        if not assistant_message:
            raise ValueError(f"Record {index} is missing an assistant output")

        conversation = {
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ]
        }
        return conversation
