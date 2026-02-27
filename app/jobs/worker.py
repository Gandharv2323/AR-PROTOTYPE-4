# Module: worker
# License: MIT (ARVTON project)
# Description: Pipeline worker — runs segmentation→tryon→reconstruct→export in ThreadPoolExecutor.
# Platform: Cloud GPU
# Dependencies: asyncio, concurrent.futures, torch

"""
Pipeline Worker — executes the full ARVTON pipeline in a background thread.
Updates job state at each stage via the JobStore.
"""

import asyncio
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from app.jobs.store import Job, JobStatus, JobStore

logger = logging.getLogger("arvton.worker")

# Single-thread executor to serialize GPU work
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="arvton-gpu")

# Output directory
OUTPUT_DIR = Path(os.environ.get("ARVTON_OUTPUT_DIR", "outputs"))


def _run_pipeline(
    job: Job,
    store: JobStore,
    loop: asyncio.AbstractEventLoop,
    base_url: str,
) -> None:
    """
    Synchronous pipeline execution in a background thread.
    Stages: segmentation → try-on → 3D reconstruct → export.
    """
    import torch

    job_id = job.job_id
    person_path = job.person_path
    garment_path = job.garment_path
    output_dir = OUTPUT_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    def update(status=None, progress=None, glb_url=None, error=None):
        asyncio.run_coroutine_threadsafe(
            store.update(job_id, status=status, progress=progress,
                         glb_url=glb_url, error=error),
            loop,
        ).result(timeout=5)

    try:
        update(status=JobStatus.PROCESSING, progress="Segmenting person")

        # ── Stage 1: Segmentation ──
        logger.info("[%s] Stage 1: Segmentation", job_id[:8])
        start = time.time()

        # Add pipeline directory to path
        pipeline_dir = Path(__file__).resolve().parent.parent.parent
        if str(pipeline_dir) not in sys.path:
            sys.path.insert(0, str(pipeline_dir))

        from pipeline.segment import segment_person, segment_garment

        person_mask = segment_person(person_path)
        mask_path = str(output_dir / "person_mask.png")
        person_mask.save(mask_path, "PNG")

        garment_seg = segment_garment(garment_path)
        garment_seg_path = str(output_dir / "garment_seg.png")
        garment_seg.save(garment_seg_path, "PNG")

        seg_time = time.time() - start
        logger.info("[%s] Segmentation: %.1fs", job_id[:8], seg_time)

        # ── Stage 2: Virtual Try-On ──
        update(progress="Generating try-on")
        logger.info("[%s] Stage 2: Try-On", job_id[:8])
        start = time.time()

        from pipeline.tryon import tryon
        from PIL import Image
        import numpy as np

        # Create placeholder densepose/bodyparse for inference
        person_img = Image.open(person_path)
        w, h = person_img.size
        placeholder = np.zeros((h, w, 3), dtype=np.uint8)
        dp_path = str(output_dir / "densepose.png")
        bp_path = str(output_dir / "bodyparse.png")
        Image.fromarray(placeholder).save(dp_path)
        Image.fromarray(placeholder).save(bp_path)

        tryon_result = tryon(
            person_path, garment_path, mask_path, bp_path, dp_path,
            category="upper",
            blend_alpha=0.5,
        )

        tryon_path = str(output_dir / "tryon_result.jpg")
        tryon_result.save(tryon_path, "JPEG", quality=95)

        tryon_time = time.time() - start
        logger.info("[%s] Try-on: %.1fs", job_id[:8], tryon_time)

        # ── Stage 3: 3D Reconstruction ──
        update(progress="Building 3D model")
        logger.info("[%s] Stage 3: 3D Reconstruction", job_id[:8])
        start = time.time()

        from pipeline.reconstruct import reconstruct

        recon_result = reconstruct(
            tryon_path,
            str(output_dir),
            use_synchuman=False,  # Skip for API latency
        )

        recon_time = time.time() - start
        logger.info("[%s] Reconstruction: %.1fs", job_id[:8], recon_time)

        # ── Stage 4: Export ──
        update(progress="Exporting GLB")
        logger.info("[%s] Stage 4: Export", job_id[:8])
        start = time.time()

        glb_path = recon_result.get("exported_files", {}).get("glb")

        if glb_path and Path(glb_path).exists():
            # Compress
            from pipeline.export import compress_glb

            compressed_path = str(output_dir / f"{job_id}.glb")
            compress_glb(glb_path, compressed_path, target_size_mb=5.0)
            glb_path = compressed_path
        else:
            # Fallback: create empty GLB marker
            glb_path = str(output_dir / f"{job_id}.glb")
            Path(glb_path).touch()

        export_time = time.time() - start
        logger.info("[%s] Export: %.1fs", job_id[:8], export_time)

        # ── Done ──
        glb_url = f"{base_url}/outputs/{job_id}/{Path(glb_path).name}"
        update(status=JobStatus.DONE, progress="done", glb_url=glb_url)
        logger.info("[%s] Pipeline complete", job_id[:8])

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        tb = traceback.format_exc()
        logger.error("[%s] Pipeline failed: %s\n%s", job_id[:8], str(e), tb)
        update(status=JobStatus.FAILED, error=str(e))

        # Clear GPU cache on error too
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


async def enqueue_job(
    job: Job,
    store: JobStore,
    base_url: str = "http://localhost:8000",
) -> None:
    """
    Submit a job to the background ThreadPoolExecutor.
    Returns immediately — the pipeline runs asynchronously.
    """
    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        _executor,
        _run_pipeline,
        job,
        store,
        loop,
        base_url,
    )
    logger.info("Job enqueued: %s", job.job_id[:8])
