# ARVTON — Local Development Guide

Run the full ARVTON pipeline on your laptop for prototyping and fine-tuning.

## Prerequisites

- **Python** 3.10+
- **GPU** (recommended): NVIDIA with 6+ GB VRAM and CUDA 12.1+
- **Flutter** 3.x (for the mobile app)
- **Node.js** (Flutter dependency)

## Quick Start (5 minutes)

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

> If you have an NVIDIA GPU, install PyTorch with CUDA:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 2. Run setup

```bash
python setup_local.py
```

This creates data directories, checks your GPU, and prints next steps.

### 3. Start the backend

```bash
python run_local.py
```

The server starts at `http://localhost:8000` with auto-reload enabled.

### 4. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Open interactive docs
# → http://localhost:8000/docs
```

### 5. Run the Flutter app

```bash
cd arvton_app
flutter pub get
flutter run -d chrome
```

For physical device testing, use ngrok:
```bash
ngrok http 8000
# Update arvton_app/.env with the ngrok URL
```

---

## Pipeline Stages

When you `POST /tryon`, the backend runs 4 stages:

| Stage | Module | What it does | GPU VRAM |
|-------|--------|--------------|----------|
| 1. Segment | `pipeline/segment.py` | SAM2 removes background | ~3 GB |
| 2. Try-On | `pipeline/tryon.py` | Leffa + CatVTON swap garment | ~4 GB |
| 3. Reconstruct | `pipeline/reconstruct.py` | TripoSR → 3D mesh | ~3 GB |
| 4. Export | `pipeline/export.py` | Compress → GLB (< 5 MB) | ~0 GB |

Models auto-download from HuggingFace on first run (~4 GB total).

---

## Fine-Tuning

### Prepare training data

1. Place person/garment image pairs in `data/arvton/datasets/`
2. Run segmentation: `python -m pipeline.segment --batch data/arvton/datasets/`
3. Update `data/arvton/datasets/train_manifest.json` with your entries

### Train the GAN refinement model

```bash
# With GPU (recommended)
python -m pipeline.train_local --epochs 50 --batch-size 4 --amp

# With limited VRAM (< 8 GB)
python -m pipeline.train_local --epochs 50 --batch-size 1 --amp

# CPU only (very slow, for testing only)
python -m pipeline.train_local --epochs 5 --batch-size 1 --device cpu

# Resume from checkpoint
python -m pipeline.train_local --resume checkpoints/gan/gan_epoch_010.pt --epochs 50

# With LoRA adapters (less VRAM)
python -m pipeline.train_local --epochs 50 --batch-size 2 --amp --lora --lora-rank 16
```

Checkpoints save to `data/arvton/checkpoints/gan/` every 10 epochs.

### Profile pipeline performance

```bash
python profile_pipeline.py --pairs 5
```

Reports average latency and peak VRAM per stage.

---

## Project Structure

```
AR-PROTOTYPE-4/
├── run_local.py          ← Start here: dev server
├── setup_local.py        ← One-time setup
├── pipeline/             ← ML pipeline modules
│   ├── segment.py        ← SAM2 segmentation
│   ├── tryon.py          ← Virtual try-on
│   ├── reconstruct.py    ← 3D reconstruction
│   ├── export.py         ← GLB export
│   ├── fine_tune.py      ← LoRA fine-tuning
│   ├── train_local.py    ← Training orchestrator
│   └── platform_utils.py ← GPU/platform detection
├── app/                  ← FastAPI backend
│   ├── main.py           ← API endpoints
│   ├── jobs/             ← Async job queue
│   └── storage/          ← File storage
├── arvton_app/           ← Flutter mobile app
├── tests/                ← pytest test suite
├── configs/              ← YAML configs
└── data/arvton/          ← Local data (gitignored)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: app` | Run from project root: `python run_local.py` |
| `CUDA out of memory` | Use `--batch-size 1` or add `--device cpu` |
| SAM2 won't download | `pip install segment-anything-2` then retry |
| Flutter red errors | `cd arvton_app && flutter pub get` |
| Port 8000 in use | `python run_local.py --port 9000` |
| Models loading slow | First run downloads ~4 GB; subsequent starts are fast |

---

## After Fine-Tuning → Cloud Deployment

Once you're happy with model quality locally:

1. Build Docker image: `docker build -t arvton -f app/Dockerfile .`
2. Deploy to AWS/GCP (see `DEPLOY.md`)
3. Update `arvton_app/.env` with production URL
4. Build release APK: `cd arvton_app && flutter build apk --release`
