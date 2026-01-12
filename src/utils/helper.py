import json
import os
import sys
from contextlib import contextmanager

import torch
import yaml


def load_config(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_device(verbose=True):
    """
    Detects and returns the best available device for PyTorch operations.
    Priority: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU

    Args:
        verbose (bool): If True, prints the detected device to console.

    Returns:
        str: 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        device = "cuda"
        info = f"NVIDIA GPU ({torch.cuda.get_device_name(0)})"

    elif torch.backends.mps.is_available():
        device = "mps"
        info = "Apple Silicon GPU (Metal Performance Shaders)"

    else:
        device = "cpu"
        info = "Standard CPU"

    if verbose:
        print(f"Device Detected: {device.upper()} [{info}]")

    return device


@contextmanager
def suppress_c_stderr():
    """
    Redirects C-level stderr to /dev/null to hide annoying
    backend logs (like ggml_metal_init) from C++ libraries.
    """
    with open(os.devnull, "w") as devnull:
        old_stderr = os.dup(sys.stderr.fileno())
        try:
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            yield
        finally:
            os.dup2(old_stderr, sys.stderr.fileno())
            os.close(old_stderr)
