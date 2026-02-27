# Module: local
# License: MIT (ARVTON project)
# Description: Local file storage — serve outputs via FastAPI StaticFiles.
# Platform: Cloud GPU / Local
# Dependencies: fastapi, pathlib

"""
Local Storage — serves output files from the local filesystem.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger("arvton.storage.local")

OUTPUT_DIR = Path(os.environ.get("ARVTON_OUTPUT_DIR", "outputs"))


def setup_local_storage(app: FastAPI) -> None:
    """
    Mount the outputs directory as a static file endpoint.
    Files are accessible at /outputs/{job_id}/{filename}.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    app.mount(
        "/outputs",
        StaticFiles(directory=str(OUTPUT_DIR)),
        name="outputs",
    )

    logger.info("Local storage mounted: %s → /outputs", OUTPUT_DIR)


def save_upload(content: bytes, job_id: str, filename: str) -> str:
    """
    Save an uploaded file to the job's directory.
    Returns the absolute path to the saved file.
    """
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    file_path = job_dir / filename
    file_path.write_bytes(content)

    logger.debug("Saved upload: %s (%d bytes)", file_path, len(content))
    return str(file_path)


def cleanup_job(job_id: str) -> bool:
    """
    Remove all files for a completed/deleted job.
    Returns True if directory was deleted.
    """
    job_dir = OUTPUT_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(str(job_dir), ignore_errors=True)
        logger.info("Cleaned up job directory: %s", job_id[:8])
        return True
    return False


def get_file_path(job_id: str, filename: str) -> Optional[str]:
    """Get the absolute path to a job's output file."""
    file_path = OUTPUT_DIR / job_id / filename
    if file_path.exists():
        return str(file_path)
    return None
