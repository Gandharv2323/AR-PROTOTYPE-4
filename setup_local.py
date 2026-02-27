#!/usr/bin/env python3
"""
ARVTON -- Local Environment Setup
==================================
Run this ONCE to prepare your local machine for ARVTON development.

Usage:
    python setup_local.py
"""

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def create_directories():
    """Create the full directory tree for local development."""
    dirs = [
        # Pipeline data
        "data/arvton/datasets/vivid/raw",
        "data/arvton/datasets/vivid/processed",
        "data/arvton/datasets/cc0/raw",
        "data/arvton/datasets/cc0/processed",
        "data/arvton/datasets/synthetic/raw",
        "data/arvton/datasets/synthetic/processed",
        "data/arvton/checkpoints/sam2",
        "data/arvton/checkpoints/gan",
        "data/arvton/outputs/samples",
        "data/arvton/outputs/eval",
        # Backend
        "outputs",
        "uploads",
    ]

    created = 0
    for d in dirs:
        p = PROJECT_ROOT / d
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            created += 1

    print(f"  [OK] {created} directories created, {len(dirs) - created} already existed")


def create_sample_manifest():
    """Create a sample train manifest so fine-tuning can be tested."""
    manifest_path = PROJECT_ROOT / "data" / "arvton" / "datasets" / "train_manifest.json"
    val_manifest_path = PROJECT_ROOT / "data" / "arvton" / "datasets" / "val_manifest.json"

    if manifest_path.exists():
        print(f"  [OK] Train manifest already exists at {manifest_path}")
        return

    sample = {
        "__info__": "Sample manifest -- replace with real data after running dataset pipeline",
        "entries": [
            {
                "person": "data/arvton/datasets/vivid/processed/person_001.jpg",
                "garment": "data/arvton/datasets/vivid/processed/garment_001.jpg",
                "mask": "data/arvton/datasets/vivid/processed/mask_001.png",
                "densepose": "data/arvton/datasets/vivid/processed/densepose_001.png",
                "gt": "data/arvton/datasets/vivid/processed/gt_001.jpg",
            }
        ],
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2)

    with open(val_manifest_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2)

    print("  [OK] Sample manifests created (replace with real data for training)")


def check_python():
    """Check Python version."""
    ver = sys.version_info
    print(f"  Python: {ver.major}.{ver.minor}.{ver.micro}")
    if ver.major < 3 or (ver.major == 3 and ver.minor < 10):
        print("  [WARN] Python 3.10+ recommended")
    else:
        print("  [OK] Python version OK")


def check_gpu():
    """Check GPU and CUDA availability."""
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram = props.total_memory / (1024 ** 3)
            print(f"  [OK] GPU: {name}")
            print(f"  [OK] VRAM: {vram:.1f} GB")
            print(f"  [OK] CUDA: {torch.version.cuda}")

            if vram < 4:
                print("  [WARN] Very low VRAM -- consider using CPU or cloud")
            elif vram < 8:
                print("  [WARN] Low VRAM -- use batch_size=1, disable SyncHuman")
            else:
                print("  [OK] VRAM is sufficient for inference")

            return True
        else:
            print("  [WARN] No CUDA GPU -- pipeline will run on CPU (slow)")
            return False
    except ImportError:
        print("  [FAIL] PyTorch not installed")
        print("    Run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return False


def check_dependencies():
    """Check that all critical packages are installed."""
    packages = {
        "fastapi": "FastAPI (backend)",
        "uvicorn": "Uvicorn (ASGI server)",
        "PIL": "Pillow (image processing)",
        "numpy": "NumPy",
        "trimesh": "Trimesh (3D meshes)",
        "yaml": "PyYAML (config)",
        "tqdm": "tqdm (progress bars)",
        "scipy": "SciPy",
    }

    ok = 0
    missing = []
    for pkg, name in packages.items():
        try:
            __import__(pkg)
            ok += 1
        except ImportError:
            missing.append(name)

    print(f"  [OK] {ok}/{len(packages)} core packages installed")
    if missing:
        print(f"  [FAIL] Missing: {', '.join(missing)}")
        print(f"    Run: pip install -r requirements.txt")


def print_next_steps(has_gpu):
    """Print what to do next."""
    print()
    print("=" * 60)
    print("  Next Steps")
    print("=" * 60)
    print()
    print("  1. Install dependencies (if not done):")
    print("     pip install -r requirements.txt")
    print()
    print("  2. Download model checkpoints:")
    print("     - SAM2 (segmentation):")
    print("       pip install segment-anything-2")
    print("       (auto-downloads from HuggingFace on first run)")
    print()
    print("     - Leffa (virtual try-on):")
    print("       pip install leffa")
    print("       (auto-downloads from HuggingFace on first run)")
    print()
    print("     - TripoSR (3D reconstruction):")
    print("       pip install tsr")
    print("       (auto-downloads from HuggingFace on first run)")
    print()
    print("  3. Start the backend:")
    print("     python run_local.py")
    print()
    print("  4. Test the API:")
    print("     curl http://localhost:8000/health")
    print()
    print("  5. Run Flutter app:")
    print("     cd arvton_app")
    print("     flutter run -d chrome")
    print()
    if has_gpu:
        print("  6. Fine-tune model (after getting training data):")
        print("     python -m pipeline.train_local --epochs 10 --batch-size 2 --amp")
    else:
        print("  6. Fine-tune model (after getting training data):")
        print("     python -m pipeline.train_local --epochs 10 --batch-size 1 --device cpu")
    print()
    print("=" * 60)


def main():
    print()
    print("=" * 60)
    print("  ARVTON -- Local Environment Setup")
    print("=" * 60)
    print()

    print("[1/5] Creating directories...")
    create_directories()
    print()

    print("[2/5] Creating sample manifests...")
    create_sample_manifest()
    print()

    print("[3/5] Checking Python...")
    check_python()
    print()

    print("[4/5] Checking GPU...")
    has_gpu = check_gpu()
    print()

    print("[5/5] Checking dependencies...")
    check_dependencies()

    print_next_steps(has_gpu)


if __name__ == "__main__":
    main()
