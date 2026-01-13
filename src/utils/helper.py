import json
import os
import sys
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Union

import torch
import yaml


def load_config(file_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file.

    Args:
        file_path (str): The path to the .yaml or .yml file.

    Returns:
        Dict[str, Any]: The configuration parameters as a dictionary.
    """
    with open(file_path, "r") as file:
        config: Dict[str, Any] = yaml.safe_load(file)
    return config


def load_json(file_path: str) -> Union[Dict[str, Any], List[Any]]:
    """Loads data from a JSON file.

    Args:
        file_path (str): The path to the .json file.

    Returns:
        Union[Dict[str, Any], List[Any]]: The parsed JSON data (usually a dict or list).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data: Union[Dict[str, Any], List[Any]] = json.load(f)
    return data


def save_json(data: Union[Dict[str, Any], List[Any]], file_path: str) -> None:
    """Saves data to a JSON file, creating directories if they don't exist.

    Args:
        data (Union[Dict[str, Any], List[Any]]): The serializable data to save.
        file_path (str): The destination path for the JSON file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_device(verbose: bool = True) -> str:
    """Detects and returns the best available device for PyTorch operations.

    The selection priority follows the hierarchy:
    CUDA (NVIDIA) > MPS (Apple Silicon) > CPU.



    Args:
        verbose (bool): If True, prints the detected device and hardware info to console.

    Returns:
        str: The device string ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        device: str = "cuda"
        info: str = f"NVIDIA GPU ({torch.cuda.get_device_name(0)})"

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
def suppress_c_stderr() -> Generator[None, None, None]:
    """Redirects C-level stderr to /dev/null to hide backend logs.

    This is particularly useful for suppressing low-level C++ library logs (such
    as ggml_metal_init) that cannot be caught by standard Python logging captures.

    Yields:
        None: Continues execution within the context manager with suppressed stderr.
    """
    with open(os.devnull, "w") as devnull:
        old_stderr: int = os.dup(sys.stderr.fileno())
        try:
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            yield
        finally:
            os.dup2(old_stderr, sys.stderr.fileno())
            os.close(old_stderr)
