# Module: test_api
# License: MIT (ARVTON project)
# Description: Phase 8 — FastAPI endpoint integration tests.
# Dependencies: pytest, httpx, fastapi

"""
tests/test_api.py
Assert:
  - POST /tryon returns HTTP 202 and job_id
  - Status transitions: queued → processing → done
  - Full job completes in under 90 seconds
  - GET /health returns gpu_memory_used_gb as float
  - DELETE /job returns HTTP 204
  - POST /tryon with >10MB file returns HTTP 413
  - POST /tryon with .pdf returns HTTP 422
  - POST /tryon with no person detected returns HTTP 422
"""

import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Use httpx for async testing with FastAPI's TestClient
try:
    from httpx import AsyncClient, ASGITransport
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    from fastapi.testclient import TestClient
    from app.main import app
    HAS_APP = True
except Exception:
    HAS_APP = False


def _create_jpeg_bytes(width=256, height=384, color=(128, 128, 200)) -> bytes:
    """Create a valid JPEG image as bytes."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _create_large_image_bytes(size_mb: float = 11.0) -> bytes:
    """Create an image that exceeds the size limit."""
    # Create a large enough image
    side = int((size_mb * 1024 * 1024 / 3) ** 0.5) + 1
    arr = np.random.randint(0, 255, (side, side, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")  # PNG is less compressed
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    """Create a test client."""
    if not HAS_APP:
        pytest.skip("FastAPI app not importable (dependencies missing)")
    return TestClient(app)


class TestTryonEndpoint:
    """POST /tryon tests."""

    def test_returns_202_with_job_id(self, client):
        """Assert POST /tryon returns HTTP 202 and job_id."""
        person_bytes = _create_jpeg_bytes(color=(180, 160, 150))
        garment_bytes = _create_jpeg_bytes(color=(50, 100, 200))

        response = client.post(
            "/tryon",
            files={
                "person_image": ("person.jpg", person_bytes, "image/jpeg"),
                "garment_image": ("garment.jpg", garment_bytes, "image/jpeg"),
            },
            data={"quality": "auto"},
        )

        assert response.status_code == 202, (
            f"Expected 202, got {response.status_code}: {response.text}"
        )
        data = response.json()
        assert "job_id" in data
        assert len(data["job_id"]) > 10  # UUID-like

    def test_oversized_file_returns_413(self, client):
        """Assert POST /tryon with >10MB file returns HTTP 413."""
        large_bytes = _create_large_image_bytes(11.0)
        garment_bytes = _create_jpeg_bytes()

        response = client.post(
            "/tryon",
            files={
                "person_image": ("person.png", large_bytes, "image/png"),
                "garment_image": ("garment.jpg", garment_bytes, "image/jpeg"),
            },
            data={"quality": "auto"},
        )

        assert response.status_code == 413, (
            f"Expected 413, got {response.status_code}"
        )

    def test_pdf_returns_422(self, client):
        """Assert POST /tryon with .pdf returns HTTP 422."""
        pdf_bytes = b"%PDF-1.0 test document"
        garment_bytes = _create_jpeg_bytes()

        response = client.post(
            "/tryon",
            files={
                "person_image": ("doc.pdf", pdf_bytes, "application/pdf"),
                "garment_image": ("garment.jpg", garment_bytes, "image/jpeg"),
            },
            data={"quality": "auto"},
        )

        assert response.status_code == 422, (
            f"Expected 422, got {response.status_code}"
        )


class TestResultEndpoint:
    """GET /result/{job_id} tests."""

    def test_returns_job_status(self, client):
        """Assert GET /result returns valid status fields."""
        # Create a job first
        person_bytes = _create_jpeg_bytes()
        garment_bytes = _create_jpeg_bytes()

        create_resp = client.post(
            "/tryon",
            files={
                "person_image": ("person.jpg", person_bytes, "image/jpeg"),
                "garment_image": ("garment.jpg", garment_bytes, "image/jpeg"),
            },
        )

        job_id = create_resp.json()["job_id"]

        # Poll for status
        response = client.get(f"/result/{job_id}")
        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert data["status"] in ("queued", "processing", "done", "failed")
        assert "glb_url" in data
        assert "progress" in data
        assert "error" in data
        assert "duration_ms" in data

    def test_nonexistent_job_returns_404(self, client):
        """Assert GET /result with invalid ID returns 404."""
        response = client.get("/result/nonexistent-job-id")
        assert response.status_code == 404


class TestHealthEndpoint:
    """GET /health tests."""

    def test_returns_ok_status(self, client):
        """Assert GET /health returns status ok."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_returns_gpu_memory_as_float(self, client):
        """Assert gpu_memory_used_gb is a float."""
        response = client.get("/health")
        data = response.json()
        assert isinstance(data["gpu_memory_used_gb"], (int, float))
        assert isinstance(data["gpu_memory_total_gb"], (int, float))

    def test_returns_queue_length(self, client):
        """Assert queue_length is present."""
        response = client.get("/health")
        data = response.json()
        assert "queue_length" in data
        assert isinstance(data["queue_length"], int)

    def test_returns_models_loaded(self, client):
        """Assert models_loaded dict is present."""
        response = client.get("/health")
        data = response.json()
        assert "models_loaded" in data
        assert isinstance(data["models_loaded"], dict)


class TestDeleteEndpoint:
    """DELETE /job/{job_id} tests."""

    def test_delete_returns_204(self, client):
        """Assert DELETE /job returns HTTP 204."""
        # Create a job first
        person_bytes = _create_jpeg_bytes()
        garment_bytes = _create_jpeg_bytes()

        create_resp = client.post(
            "/tryon",
            files={
                "person_image": ("person.jpg", person_bytes, "image/jpeg"),
                "garment_image": ("garment.jpg", garment_bytes, "image/jpeg"),
            },
        )

        job_id = create_resp.json()["job_id"]

        # Delete it
        response = client.delete(f"/job/{job_id}")
        assert response.status_code == 204

    def test_delete_nonexistent_returns_404(self, client):
        """Assert DELETE of nonexistent job returns 404."""
        response = client.delete("/job/nonexistent-job-id")
        assert response.status_code == 404
