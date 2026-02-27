# Module: main
# License: MIT (ARVTON project)
# Description: Production FastAPI application with all endpoints, rate limiting, CORS, structured logging.
# Platform: Cloud GPU (T4/A100/MI250)
# Dependencies: fastapi, uvicorn, slowapi, python-multipart, torch

"""
===================================
PRODUCTION FASTAPI — ARVTON API
File: app/main.py
===================================

Endpoints:
    POST   /tryon          — submit try-on job (returns 202 + job_id)
    GET    /result/{job_id} — poll job status
    GET    /health         — GPU stats, queue length, model status
    DELETE /job/{job_id}   — cancel/delete a job
"""

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Structured JSON Logger ──────────────────────────────────────────────


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        if hasattr(record, "job_id"):
            log_entry["job_id"] = record.job_id
        if hasattr(record, "stage"):
            log_entry["stage"] = record.stage
        if hasattr(record, "latency_ms"):
            log_entry["latency_ms"] = record.latency_ms
        if record.exc_info and record.exc_info[1]:
            log_entry["error"] = str(record.exc_info[1])
        return json.dumps(log_entry)


# Setup logging
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger("arvton.api")

# Add pipeline to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Model Tracking ─────────────────────────────────────────────────────

_models_loaded = {"sam2": False, "tryon": False, "hmr2": False, "triposr": False}


# ── Lifespan ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up models on startup, clean up on shutdown."""
    global _models_loaded

    logger.info("Starting ARVTON API — warming up models...")

    try:
        from pipeline.segment import _load_sam2
        _load_sam2(device="cpu")
        _models_loaded["sam2"] = True
        logger.info("SAM2 loaded")
    except Exception as e:
        logger.warning("SAM2 warm-up skipped: %s", str(e))

    try:
        from pipeline.tryon import load_leffa
        load_leffa()
        _models_loaded["tryon"] = True
        logger.info("Try-on model loaded ✓")
    except Exception as e:
        logger.warning("Try-on warm-up failed: %s", str(e))

    logger.info("All models loaded. Ready to accept requests.")

    yield

    # Cleanup
    logger.info("Shutting down ARVTON API...")
    try:
        from pipeline.segment import clear_model_cache
        from pipeline.tryon import clear_cache as clear_tryon
        from pipeline.reconstruct import clear_cache as clear_recon
        clear_model_cache()
        clear_tryon()
        clear_recon()
    except Exception:
        pass


# ── App ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ARVTON API",
    description="AR/VR Virtual Try-On — Production API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── CORS ────────────────────────────────────────────────────────────────

_raw_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000")
if _raw_origins.strip() == "*":
    ALLOWED_ORIGINS = ["*"]
else:
    ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Rate Limiting ───────────────────────────────────────────────────────

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    RATE_LIMIT = "10/minute"
    logger.info("Rate limiting enabled: %s", RATE_LIMIT)
except ImportError:
    limiter = None
    RATE_LIMIT = None
    logger.warning("slowapi not installed. Rate limiting disabled.")


# ── Storage & Jobs ──────────────────────────────────────────────────────

from app.jobs.store import JobStore
from app.storage.local import setup_local_storage

job_store = JobStore(max_jobs=1000)
setup_local_storage(app)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/jpg"}
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")


# ── Request Logging Middleware ──────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency_ms = int((time.time() - start) * 1000)

    extra = {"latency_ms": latency_ms}
    logger.info(
        "%s %s → %d (%dms)",
        request.method, request.url.path,
        response.status_code, latency_ms,
        extra=extra,
    )
    return response


# ═══════════════════════════════════════════════════════════════════════
# POST /tryon
# ═══════════════════════════════════════════════════════════════════════

@app.post("/tryon", status_code=202)
async def create_tryon(
    request: Request,
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    quality: str = Form("auto"),
):
    """
    Submit a virtual try-on job.
    Returns HTTP 202 with job_id for async polling.
    """
    # Rate limit check
    if limiter:
        await limiter.check(RATE_LIMIT, request)

    # Validate MIME types
    for upload, name in [(person_image, "person"), (garment_image, "garment")]:
        if upload.content_type and upload.content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                422,
                f"Invalid {name} image type: {upload.content_type}. "
                f"Allowed: JPEG, PNG.",
            )

    # Read and validate file sizes
    person_content = await person_image.read()
    garment_content = await garment_image.read()

    if len(person_content) > MAX_FILE_SIZE:
        raise HTTPException(413, "Person image exceeds 10 MB limit.")
    if len(garment_content) > MAX_FILE_SIZE:
        raise HTTPException(413, "Garment image exceeds 10 MB limit.")

    # Save uploads
    from app.storage.local import save_upload

    job_id = str(uuid.uuid4())
    person_path = save_upload(person_content, job_id, "person.jpg")
    garment_path = save_upload(garment_content, job_id, "garment.jpg")

    # Create job
    job = await job_store.create(person_path, garment_path, quality)
    # Override the auto-generated ID with our pre-created one
    job.job_id = job_id

    # Enqueue pipeline
    from app.jobs.worker import enqueue_job
    await enqueue_job(job, job_store, base_url=BASE_URL)

    return {"job_id": job_id}


# ═══════════════════════════════════════════════════════════════════════
# GET /result/{job_id}
# ═══════════════════════════════════════════════════════════════════════

@app.get("/result/{job_id}")
async def get_result(job_id: str):
    """Poll job status and get result when done."""
    job = await job_store.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job not found: {job_id}")

    return job.to_dict()


# ═══════════════════════════════════════════════════════════════════════
# GET /health
# ═══════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    """Health check with GPU stats, queue length, and model status."""
    gpu_name = "N/A"
    gpu_used = 0.0
    gpu_total = 0.0

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_used = torch.cuda.memory_allocated(0) / (1024 ** 3)
            gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        pass

    queue_len = await job_store.queue_length()
    last_jobs = await job_store.last_n_jobs(5)

    return {
        "status": "ok",
        "gpu_name": gpu_name,
        "gpu_memory_used_gb": round(gpu_used, 2),
        "gpu_memory_total_gb": round(gpu_total, 2),
        "queue_length": queue_len,
        "models_loaded": _models_loaded,
        "last_5_jobs": last_jobs,
    }


# ═══════════════════════════════════════════════════════════════════════
# DELETE /job/{job_id}
# ═══════════════════════════════════════════════════════════════════════

@app.delete("/job/{job_id}", status_code=204)
async def delete_job(job_id: str):
    """Delete a job and its associated files."""
    deleted = await job_store.delete(job_id)

    if not deleted:
        raise HTTPException(404, f"Job not found: {job_id}")

    # Clean up files
    from app.storage.local import cleanup_job
    cleanup_job(job_id)

    return None


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=False,
        workers=1,
        log_level="info",
    )
