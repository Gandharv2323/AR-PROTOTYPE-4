# Module: server
# License: MIT (ARVTON project)
# Description: Phase 7 — FastAPI backend with /segment, /tryon, /reconstruct, /export endpoints.
# Platform: Cloud GPU deployment (T4/A100/MI250)
# Dependencies: fastapi, uvicorn, python-multipart, torch, Pillow

"""
===================================
PHASE 7 — FASTAPI BACKEND
File: server.py
===================================

ARVTON API Server
==================
REST API endpoints:
    POST /segment         — garment/person segmentation → RGBA PNG
    POST /tryon           — virtual try-on → result image
    POST /reconstruct     — 3D mesh reconstruction → GLB URL
    POST /export          — mesh export with compression
    GET  /health          — health check with GPU status

WebSocket endpoint:
    WS   /ws/stream       — streaming real-time inference
"""

import asyncio
import io
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from PIL import Image

logger = logging.getLogger("arvton.server")

# ═══════════════════════════════════════════════════════════════════════
# APP CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="ARVTON API",
    description="AR/VR Virtual Try-On Pipeline — REST API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS for Flutter frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Output directory for generated files
OUTPUT_DIR = Path(os.environ.get("ARVTON_OUTPUT_DIR", "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# STARTUP / SHUTDOWN
# ═══════════════════════════════════════════════════════════════════════


@app.on_event("startup")
async def startup():
    """Pre-load models on server startup for fast inference."""
    logger.info("ARVTON API starting...")
    from pipeline.platform_utils import detect_gpu_info, print_system_info
    gpu = detect_gpu_info()
    logger.info("GPU: %s (VRAM: %s)", gpu.get("name", "N/A"), gpu.get("vram_gb", "N/A"))
    print_system_info()


@app.on_event("shutdown")
async def shutdown():
    """Clean up GPU memory on shutdown."""
    logger.info("ARVTON API shutting down...")
    try:
        from pipeline.segment import clear_model_cache as clear_seg
        from pipeline.tryon import clear_cache as clear_tryon
        from pipeline.reconstruct import clear_cache as clear_recon
        clear_seg()
        clear_tryon()
        clear_recon()
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════


@app.get("/health")
async def health():
    """Health check with GPU status."""
    from pipeline.platform_utils import detect_gpu_info, detect_platform

    gpu = detect_gpu_info()
    return {
        "status": "ok",
        "platform": detect_platform(),
        "gpu": gpu.get("name", "N/A"),
        "gpu_available": gpu.get("available", False),
        "vram_gb": gpu.get("vram_gb", 0),
        "version": "1.0.0",
    }


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT: /segment
# ═══════════════════════════════════════════════════════════════════════


@app.post("/segment")
async def segment_endpoint(
    image: UploadFile = File(...),
    mode: str = Form("garment"),
):
    """
    Segment an image to remove background.

    Args:
        image: Uploaded image file (JPEG/PNG).
        mode: 'garment' or 'person'.

    Returns:
        RGBA PNG with transparent background.
    """
    if mode not in ("garment", "person"):
        raise HTTPException(400, "mode must be 'garment' or 'person'")

    # Save uploaded file
    request_id = str(uuid.uuid4())[:8]
    upload_dir = OUTPUT_DIR / "uploads"
    upload_dir.mkdir(exist_ok=True)

    ext = Path(image.filename or "image.jpg").suffix or ".jpg"
    upload_path = upload_dir / f"{request_id}{ext}"

    content = await image.read()
    with open(upload_path, "wb") as f:
        f.write(content)

    try:
        from pipeline.segment import segment_garment, segment_person

        start = time.time()

        if mode == "garment":
            result = segment_garment(str(upload_path))
        else:
            result = segment_person(str(upload_path))

        elapsed = time.time() - start
        logger.info("Segmentation [%s] completed in %.2fs", mode, elapsed)

        # Return as PNG
        img_bytes = io.BytesIO()
        result.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        return StreamingResponse(
            img_bytes,
            media_type="image/png",
            headers={
                "X-Request-Id": request_id,
                "X-Inference-Time": f"{elapsed:.3f}s",
            },
        )

    except Exception as e:
        logger.error("Segmentation failed: %s", str(e))
        raise HTTPException(500, f"Segmentation failed: {str(e)}")

    finally:
        # Cleanup upload
        if upload_path.exists():
            upload_path.unlink()


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT: /tryon
# ═══════════════════════════════════════════════════════════════════════


@app.post("/tryon")
async def tryon_endpoint(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    category: str = Form("upper"),
    blend_alpha: float = Form(0.5),
):
    """
    Run virtual try-on on uploaded person and garment images.

    Args:
        person_image: Person photo (JPEG/PNG).
        garment_image: Garment photo (JPEG/PNG).
        category: 'upper', 'lower', or 'dress'.
        blend_alpha: Leffa weight (0 = CatVTON only, 1 = Leffa only).

    Returns:
        Try-on result as JPEG.
    """
    if category not in ("upper", "lower", "dress"):
        raise HTTPException(400, "category must be 'upper', 'lower', or 'dress'")

    request_id = str(uuid.uuid4())[:8]
    upload_dir = OUTPUT_DIR / "uploads"
    upload_dir.mkdir(exist_ok=True)

    # Save uploads
    person_path = upload_dir / f"{request_id}_person.jpg"
    garment_path = upload_dir / f"{request_id}_garment.jpg"

    person_content = await person_image.read()
    with open(person_path, "wb") as f:
        f.write(person_content)

    garment_content = await garment_image.read()
    with open(garment_path, "wb") as f:
        f.write(garment_content)

    try:
        from pipeline.segment import segment_person, segment_garment
        from pipeline.tryon import tryon

        start = time.time()

        # Auto-segment person
        person_mask_result = segment_person(str(person_path))
        mask_path = upload_dir / f"{request_id}_mask.png"
        person_mask_result.save(str(mask_path), "PNG")

        # Create placeholder densepose and body parse
        person_img = Image.open(person_path)
        w, h = person_img.size
        placeholder = np.zeros((h, w, 3), dtype=np.uint8)
        densepose_path = upload_dir / f"{request_id}_densepose.png"
        bodyparse_path = upload_dir / f"{request_id}_bodyparse.png"
        Image.fromarray(placeholder).save(str(densepose_path))
        Image.fromarray(placeholder).save(str(bodyparse_path))

        # Run try-on
        output_path = str(OUTPUT_DIR / f"{request_id}_tryon.jpg")
        result = tryon(
            str(person_path), str(garment_path),
            str(mask_path), str(bodyparse_path), str(densepose_path),
            category=category,
            blend_alpha=blend_alpha,
            output_path=output_path,
        )

        elapsed = time.time() - start
        logger.info("Try-on completed in %.2fs", elapsed)

        img_bytes = io.BytesIO()
        result.save(img_bytes, format="JPEG", quality=95)
        img_bytes.seek(0)

        return StreamingResponse(
            img_bytes,
            media_type="image/jpeg",
            headers={
                "X-Request-Id": request_id,
                "X-Inference-Time": f"{elapsed:.3f}s",
            },
        )

    except Exception as e:
        logger.error("Try-on failed: %s", str(e))
        raise HTTPException(500, f"Try-on failed: {str(e)}")

    finally:
        for p in [person_path, garment_path]:
            if p.exists():
                p.unlink()


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT: /reconstruct
# ═══════════════════════════════════════════════════════════════════════


@app.post("/reconstruct")
async def reconstruct_endpoint(
    image: UploadFile = File(...),
    include_usdz: bool = Form(False),
):
    """
    Generate 3D mesh from try-on result image.

    Args:
        image: Try-on result image (JPEG/PNG).
        include_usdz: Generate USDZ for Apple AR Quick Look.

    Returns:
        GLB file download.
    """
    request_id = str(uuid.uuid4())[:8]
    upload_dir = OUTPUT_DIR / "uploads"
    upload_dir.mkdir(exist_ok=True)

    upload_path = upload_dir / f"{request_id}.jpg"
    content = await image.read()
    with open(upload_path, "wb") as f:
        f.write(content)

    try:
        from pipeline.reconstruct import reconstruct

        start = time.time()
        result = reconstruct(
            str(upload_path),
            str(OUTPUT_DIR / "meshes" / request_id),
            use_synchuman=False,  # Skip for speed in API mode
        )
        elapsed = time.time() - start

        if not result["success"]:
            raise HTTPException(500, "3D reconstruction failed")

        glb_path = result["exported_files"].get("glb")
        if not glb_path or not Path(glb_path).exists():
            raise HTTPException(500, "GLB export failed")

        logger.info("Reconstruction completed in %.2fs", elapsed)

        return FileResponse(
            glb_path,
            media_type="model/gltf-binary",
            filename=f"{request_id}.glb",
            headers={
                "X-Request-Id": request_id,
                "X-Inference-Time": f"{elapsed:.3f}s",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Reconstruction failed: %s", str(e))
        raise HTTPException(500, f"Reconstruction failed: {str(e)}")

    finally:
        if upload_path.exists():
            upload_path.unlink()


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT: /export
# ═══════════════════════════════════════════════════════════════════════


@app.post("/export")
async def export_endpoint(
    glb_file: UploadFile = File(...),
    target_size_mb: float = Form(5.0),
    include_usdz: bool = Form(False),
):
    """
    Compress and optimize a GLB file for mobile delivery.

    Args:
        glb_file: Input GLB file.
        target_size_mb: Target compressed size.
        include_usdz: Generate USDZ.

    Returns:
        Compressed GLB file.
    """
    request_id = str(uuid.uuid4())[:8]
    upload_dir = OUTPUT_DIR / "uploads"
    upload_dir.mkdir(exist_ok=True)

    input_path = upload_dir / f"{request_id}_input.glb"
    content = await glb_file.read()
    with open(input_path, "wb") as f:
        f.write(content)

    try:
        from pipeline.export import compress_glb, convert_to_usdz

        output_path = str(OUTPUT_DIR / "exports" / f"{request_id}_compressed.glb")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        stats = compress_glb(str(input_path), output_path, target_size_mb=target_size_mb)

        if include_usdz:
            convert_to_usdz(output_path)

        return FileResponse(
            output_path,
            media_type="model/gltf-binary",
            filename=f"compressed_{request_id}.glb",
            headers={
                "X-Original-Size": f"{stats['original_size_mb']} MB",
                "X-Compressed-Size": f"{stats['compressed_size_mb']} MB",
                "X-Reduction": f"{stats['reduction_pct']}%",
            },
        )

    except Exception as e:
        logger.error("Export failed: %s", str(e))
        raise HTTPException(500, f"Export failed: {str(e)}")

    finally:
        if input_path.exists():
            input_path.unlink()


# ═══════════════════════════════════════════════════════════════════════
# WEBSOCKET: Real-time streaming
# ═══════════════════════════════════════════════════════════════════════


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time try-on streaming.
    Receives frames, processes, and streams back results.
    """
    await websocket.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            # Receive frame as bytes
            data = await websocket.receive_bytes()

            # Decode image
            img = Image.open(io.BytesIO(data)).convert("RGB")

            # Quick segmentation (lightweight mode)
            try:
                from pipeline.segment import segment_person
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    img.save(tmp.name, "JPEG")
                    result = segment_person(tmp.name)
                    os.unlink(tmp.name)

                # Send back as PNG bytes
                output_bytes = io.BytesIO()
                result.save(output_bytes, format="PNG")
                await websocket.send_bytes(output_bytes.getvalue())

            except Exception as e:
                await websocket.send_json({"error": str(e)})

    except Exception:
        logger.info("WebSocket client disconnected")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════


def main():
    """Start the ARVTON API server."""
    from pipeline.platform_utils import setup_logging
    setup_logging("INFO")

    host = os.environ.get("ARVTON_HOST", "0.0.0.0")
    port = int(os.environ.get("ARVTON_PORT", "8000"))

    logger.info("Starting ARVTON API on %s:%d", host, port)

    uvicorn.run(
        "pipeline.server:app",
        host=host,
        port=port,
        reload=False,
        workers=1,  # Single worker for GPU model sharing
        log_level="info",
    )


if __name__ == "__main__":
    main()
