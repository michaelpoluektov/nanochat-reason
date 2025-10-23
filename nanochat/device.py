"""
Helpers for selecting the default torch device across nanochat.
"""

from functools import lru_cache
import torch


def _mps_is_available() -> bool:
    return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()


@lru_cache(maxsize=None)
def get_default_device() -> torch.device:
    """
    Prefer CUDA, fall back to MPS, otherwise CPU.
    The result is cached so callers can reuse the same torch.device instance.
    """
    if torch.cuda.is_available():
        # Always return a CUDA device with an explicit index (default 0) so
        # downstream torch.cuda.set_device() calls receive a fully qualified device.
        return torch.device("cuda", torch.cuda.current_device() if torch.cuda.device_count() > 0 else 0)
    if _mps_is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_autocast_kwargs(device: torch.device) -> dict:
    """
    Return kwargs for torch.amp.autocast that match the selected device.
    """
    device_type = device.type if device.type in {"cuda", "mps"} else "cpu"
    enabled = device_type != "cpu"
    dtype = torch.bfloat16 if enabled else torch.float32
    return {"device_type": device_type, "dtype": dtype, "enabled": enabled}


def supports_non_blocking(device: torch.device) -> bool:
    """
    True if `.to(non_blocking=True)` transfers are supported for the device.
    """
    return device.type in {"cuda", "mps"}
