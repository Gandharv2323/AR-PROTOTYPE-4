#!/usr/bin/env python3
"""
ARVTON -- Local Development Server
===================================
One-command launcher for the ARVTON backend on your laptop.

Usage:
    python run_local.py              # Start dev server (auto-reload)
    python run_local.py --no-reload  # Start without auto-reload
    python run_local.py --port 9000  # Custom port
"""

import argparse
import os
import sys
from pathlib import Path

# -- Ensure project root is on sys.path --
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def setup_directories():
    """Create required directories for local dev."""
    dirs = [
        PROJECT_ROOT / "outputs",
        PROJECT_ROOT / "uploads",
        PROJECT_ROOT / "data" / "arvton" / "checkpoints",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print(f"  [OK] Directories ready ({len(dirs)} checked)")


def load_env():
    """Load environment variables from app/.env if it exists."""
    env_file = PROJECT_ROOT / "app" / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())
        print(f"  [OK] Environment loaded from {env_file}")
    else:
        print(f"  [WARN] No .env file found at {env_file}")
        print(f"    Copy app/.env.example -> app/.env and customize")


def check_gpu():
    """Check GPU availability and report."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"  [OK] GPU: {name} ({vram:.1f} GB VRAM)")
            return True
        else:
            print("  [WARN] No CUDA GPU detected -- running on CPU (slow)")
            print("    Pipeline will use CPU fallback for inference")
            return False
    except ImportError:
        print("  [FAIL] PyTorch not installed -- run: pip install torch torchvision")
        return False


def check_dependencies():
    """Quick check that critical packages are installed."""
    missing = []
    for pkg in ["fastapi", "uvicorn", "PIL", "numpy", "trimesh"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"  [WARN] Missing packages: {', '.join(missing)}")
        print(f"    Run: pip install -r requirements.txt")
    else:
        print(f"  [OK] Core dependencies OK")


def main():
    parser = argparse.ArgumentParser(description="ARVTON Local Dev Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  ARVTON -- Local Development Server")
    print("=" * 60)
    print()

    # Step 1: Load environment
    load_env()

    # Step 2: Create directories
    setup_directories()

    # Step 3: Check GPU
    has_gpu = check_gpu()

    # Step 4: Check dependencies
    check_dependencies()

    # Set dev mode
    os.environ["ARVTON_DEV"] = "1"
    if not has_gpu:
        os.environ.setdefault("ARVTON_CPU_MODE", "1")

    print()
    print("-" * 60)
    print(f"  Server starting at: http://localhost:{args.port}")
    print(f"  API docs:           http://localhost:{args.port}/docs")
    print(f"  Health check:       http://localhost:{args.port}/health")
    print(f"  Auto-reload:        {'OFF' if args.no_reload else 'ON'}")
    print(f"  GPU:                {'YES' if has_gpu else 'CPU only'}")
    print("-" * 60)
    print()
    print("  Test commands:")
    print(f"    curl http://localhost:{args.port}/health")
    print()
    print(f'    curl -X POST http://localhost:{args.port}/tryon \\')
    print(f'      -F "person_image=@test_person.jpg" \\')
    print(f'      -F "garment_image=@test_garment.jpg"')
    print()
    print("  Press Ctrl+C to stop.")
    print("=" * 60)
    print()

    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        reload_dirs=[str(PROJECT_ROOT / "app"), str(PROJECT_ROOT / "pipeline")],
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
