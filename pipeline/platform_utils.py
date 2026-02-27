# Module: platform_utils
# License: MIT (ARVTON project)
# Description: Platform detection and path resolution for Colab, AMD ROCm, and local environments.
# Platform: Both
# Dependencies: torch, yaml, os, pathlib

"""
Platform Utilities
==================
Detects the current execution environment (Colab, AMD ROCm, or local)
and resolves paths accordingly. All other modules import from here.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

logger = logging.getLogger("arvton.platform")


def detect_platform() -> str:
    """
    Detect the current execution platform.

    Returns:
        str: One of 'colab', 'amd', or 'local'.
    """
    # Check for Google Colab environment
    if "google.colab" in sys.modules or os.path.exists("/content"):
        logger.info("Platform detected: Google Colab")
        return "colab"

    # Check for AMD ROCm
    try:
        import torch
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            logger.info("Platform detected: AMD ROCm (HIP version: %s)", torch.version.hip)
            return "amd"
    except ImportError:
        pass

    logger.info("Platform detected: Local")
    return "local"


def detect_gpu_info() -> Dict[str, object]:
    """
    Detect GPU information including name, memory, and precision support.

    Returns:
        dict: GPU metadata with keys:
            - available (bool)
            - name (str)
            - total_memory_gb (float)
            - is_rocm (bool)
            - supports_bf16 (bool)
            - recommended_precision (str): 'bf16', 'fp16', or 'fp32'
    """
    info = {
        "available": False,
        "name": "CPU",
        "total_memory_gb": 0.0,
        "is_rocm": False,
        "supports_bf16": False,
        "recommended_precision": "fp32",
    }

    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning("No GPU detected. Running on CPU — this will be very slow.")
            return info

        info["available"] = True
        info["name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["total_memory_gb"] = round(props.total_mem / (1024 ** 3), 2)
        info["is_rocm"] = hasattr(torch.version, "hip") and torch.version.hip is not None

        # Precision detection
        try:
            info["supports_bf16"] = torch.cuda.is_bf16_supported()
        except Exception:
            info["supports_bf16"] = False

        # Recommend precision based on GPU
        if info["supports_bf16"] and info["total_memory_gb"] >= 20:
            info["recommended_precision"] = "bf16"
        elif info["total_memory_gb"] >= 10:
            info["recommended_precision"] = "fp16"
        else:
            info["recommended_precision"] = "fp32"

        logger.info(
            "GPU: %s | Memory: %.1f GB | ROCm: %s | Precision: %s",
            info["name"],
            info["total_memory_gb"],
            info["is_rocm"],
            info["recommended_precision"],
        )

    except ImportError:
        logger.warning("PyTorch not installed. GPU detection unavailable.")

    return info


def load_config(config_path: Optional[str] = None) -> dict:
    """
    Load the dataset configuration YAML file with environment variable expansion.

    Args:
        config_path: Path to the YAML config file. If None, auto-detects from project root.

    Returns:
        dict: Parsed configuration.
    """
    if config_path is None:
        # Search relative to this file's location
        here = Path(__file__).resolve().parent.parent
        config_path = here / "configs" / "dataset_config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Expand environment variables in the YAML
    for key in ["PEXELS_API_KEY", "PIXABAY_API_KEY"]:
        env_val = os.environ.get(key, "")
        raw = raw.replace(f"${{{key}}}", env_val)

    config = yaml.safe_load(raw)
    return config


def get_paths(platform: Optional[str] = None) -> Dict[str, Path]:
    """
    Get resolved paths for the current platform.

    Args:
        platform: Override platform detection. If None, auto-detects.

    Returns:
        dict: Mapping of path names to Path objects:
            - base, datasets, checkpoints, outputs
            - vivid_raw, vivid_processed
            - cc0_raw, cc0_processed
            - synthetic_raw, synthetic_processed
    """
    if platform is None:
        platform = detect_platform()

    config = load_config()
    platform_paths = config["paths"].get(platform, config["paths"]["local"])

    base = Path(platform_paths["base"])
    datasets = Path(platform_paths["datasets"])
    checkpoints = Path(platform_paths["checkpoints"])
    outputs = Path(platform_paths["outputs"])

    paths = {
        "base": base,
        "datasets": datasets,
        "checkpoints": checkpoints,
        "outputs": outputs,
        # ViViD
        "vivid_raw": datasets / "vivid" / "raw",
        "vivid_processed": datasets / "vivid" / "processed",
        # CC0
        "cc0_raw": datasets / "cc0" / "raw",
        "cc0_processed": datasets / "cc0" / "processed",
        # Synthetic
        "synthetic_raw": datasets / "synthetic" / "raw",
        "synthetic_processed": datasets / "synthetic" / "processed",
        # Manifests
        "train_manifest": datasets / "train_manifest.json",
        "val_manifest": datasets / "val_manifest.json",
        # SAM2 checkpoint
        "sam2_checkpoint": checkpoints / "sam2",
        # Samples and eval (training)
        "samples": outputs / "samples",
        "eval": outputs / "eval",
    }

    return paths


def ensure_dirs(paths: Dict[str, Path]) -> None:
    """
    Create all directories in the paths dict if they don't exist.

    Args:
        paths: Dictionary of path names to Path objects.
    """
    for name, p in paths.items():
        if "manifest" not in name and not p.suffix:
            p.mkdir(parents=True, exist_ok=True)
            logger.debug("Ensured directory: %s", p)


def get_torch_device() -> "torch.device":
    """
    Get the appropriate torch device for inference.

    Returns:
        torch.device: CUDA device if available, else CPU.
    """
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_precision_dtype() -> "torch.dtype":
    """
    Get the recommended precision dtype for the current GPU.

    Returns:
        torch.dtype: bf16, fp16, or fp32 based on GPU capabilities.
    """
    import torch
    info = detect_gpu_info()
    if info["recommended_precision"] == "bf16":
        return torch.bfloat16
    elif info["recommended_precision"] == "fp16":
        return torch.float16
    return torch.float32


def mount_google_drive() -> bool:
    """
    Mount Google Drive if running on Colab.

    Returns:
        bool: True if mounted successfully, False if not on Colab.
    """
    if detect_platform() != "colab":
        return False

    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        logger.info("Google Drive mounted at /content/drive")
        return True
    except Exception as e:
        logger.error("Failed to mount Google Drive: %s", e)
        return False


def setup_logging(level: str = "INFO") -> None:
    """
    Configure structured logging for the ARVTON pipeline.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.setLevel(log_level)


def print_system_info() -> None:
    """
    Print a complete system information summary for debugging.
    """
    import platform as plat

    print("=" * 60)
    print("ARVTON System Information")
    print("=" * 60)
    print(f"  Python:    {sys.version.split()[0]}")
    print(f"  OS:        {plat.system()} {plat.release()}")
    print(f"  Platform:  {detect_platform()}")

    try:
        import torch
        print(f"  PyTorch:   {torch.__version__}")
        if hasattr(torch.version, "hip") and torch.version.hip:
            print(f"  ROCm HIP:  {torch.version.hip}")
        elif hasattr(torch.version, "cuda") and torch.version.cuda:
            print(f"  CUDA:      {torch.version.cuda}")
    except ImportError:
        print("  PyTorch:   NOT INSTALLED")

    gpu = detect_gpu_info()
    print(f"  GPU:       {gpu['name']}")
    print(f"  VRAM:      {gpu['total_memory_gb']} GB")
    print(f"  Precision: {gpu['recommended_precision']}")
    print("=" * 60)
