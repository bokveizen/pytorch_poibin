"""Conservative opt-in dispatch from measured, exact-workload profiles."""

import json
import math
import platform
from functools import lru_cache
from importlib.resources import files

import torch
from torch import Tensor


def hardware_key(device: torch.device) -> dict:
    """Include runtime and thread configuration; never extrapolate across hardware."""
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
    else:
        name = platform.processor() or platform.machine()
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as info:
                name = next(
                    line.split(":", 1)[1].strip()
                    for line in info
                    if line.startswith("model name")
                )
        except (OSError, StopIteration):
            pass
    return {
        "device": device.type,
        "hardware": name,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda if device.type == "cuda" else None,
        "threads": torch.get_num_threads(),
        "platform": platform.system(),
    }


@lru_cache(maxsize=1)
def load_profiles() -> list[dict]:
    data = json.loads(files("pytorch_poibin").joinpath("profiles.json").read_text())
    if data.get("schema_version") != 1:
        return []
    return data["profiles"]


def select_method(probs: Tensor) -> str:
    if probs.requires_grad or probs.device.type not in {"cpu", "cuda"}:
        return "dp"
    shape = (math.prod(probs.shape[:-1]), probs.shape[-1])
    key = hardware_key(probs.device)
    for profile in load_profiles():
        if (
            profile["runtime"] == key
            and profile["dtype"] == str(probs.dtype)
            and (profile["batch_size"], profile["n"]) == shape
        ):
            return "fft_tree"
    return "dp"
