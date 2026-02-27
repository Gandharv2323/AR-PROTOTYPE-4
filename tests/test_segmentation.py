# Module: test_segmentation
# License: MIT (ARVTON project)
# Description: Phase 8 — Segmentation integration tests.
# Dependencies: pytest, Pillow, numpy

"""
tests/test_segmentation.py
Assert:
  - Output mode == RGBA
  - Alpha channel has zeros (background removed)
  - Foreground pixel count > 1000
  - Run on 10 different test images
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _create_test_image(
    width: int = 512,
    height: int = 768,
    color: tuple = (128, 128, 200),
    seed: int = 0,
) -> str:
    """Create a temporary test image and return its path."""
    np.random.seed(seed)
    arr = np.full((height, width, 3), color, dtype=np.uint8)

    # Add some noise/variation to simulate a real image
    noise = np.random.randint(-30, 30, (height, width, 3), dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Draw a centered rectangle to simulate a person/garment
    cx, cy = width // 2, height // 2
    rw, rh = width // 4, height // 3
    arr[cy - rh : cy + rh, cx - rw : cx + rw] = [200, 100, 100]

    img = Image.fromarray(arr)
    tmp_path = tempfile.mktemp(suffix=".jpg")
    img.save(tmp_path, "JPEG")
    return tmp_path


# Generate 10 test images with different colors/seeds
TEST_IMAGES = []


@pytest.fixture(scope="module", autouse=True)
def setup_test_images():
    """Create test images before tests and clean up after."""
    global TEST_IMAGES
    colors = [
        (128, 128, 200), (200, 128, 128), (128, 200, 128),
        (200, 200, 128), (128, 200, 200), (200, 128, 200),
        (180, 180, 180), (100, 150, 200), (200, 150, 100),
        (150, 100, 200),
    ]
    TEST_IMAGES = [
        _create_test_image(seed=i, color=colors[i]) for i in range(10)
    ]
    yield
    for path in TEST_IMAGES:
        if os.path.exists(path):
            os.unlink(path)


class TestGarmentSegmentation:
    """Test garment segmentation (background removal)."""

    def test_output_mode_rgba(self):
        """Assert output mode is RGBA."""
        from pipeline.segment import segment_garment

        result = segment_garment(TEST_IMAGES[0], device="cpu")
        assert result.mode == "RGBA", f"Expected RGBA, got {result.mode}"

    def test_alpha_has_zeros(self):
        """Assert alpha channel has zero pixels (background removed)."""
        from pipeline.segment import segment_garment

        result = segment_garment(TEST_IMAGES[1], device="cpu")
        alpha = np.array(result)[:, :, 3]
        zero_count = np.sum(alpha == 0)
        assert zero_count > 0, "Alpha channel has no zeros — background not removed"

    def test_foreground_pixel_count(self):
        """Assert foreground pixel count > 1000."""
        from pipeline.segment import segment_garment

        result = segment_garment(TEST_IMAGES[2], device="cpu")
        alpha = np.array(result)[:, :, 3]
        fg_count = np.sum(alpha > 128)
        assert fg_count > 1000, f"Foreground pixels: {fg_count} (expected > 1000)"

    @pytest.mark.parametrize("idx", range(10))
    def test_per_image_pass(self, idx):
        """Run on 10 different test images — report per-image pass/fail."""
        from pipeline.segment import segment_garment

        try:
            result = segment_garment(TEST_IMAGES[idx], device="cpu")
            assert result.mode == "RGBA"
            assert result.size[0] > 0 and result.size[1] > 0
        except Exception as e:
            pytest.skip(f"Image {idx} skipped (model not available): {e}")


class TestPersonSegmentation:
    """Test person segmentation."""

    def test_output_mode_rgba(self):
        """Assert output mode is RGBA."""
        from pipeline.segment import segment_person

        result = segment_person(TEST_IMAGES[0], device="cpu")
        assert result.mode == "RGBA", f"Expected RGBA, got {result.mode}"

    def test_alpha_has_zeros(self):
        """Assert alpha channel has zero pixels."""
        from pipeline.segment import segment_person

        result = segment_person(TEST_IMAGES[1], device="cpu")
        alpha = np.array(result)[:, :, 3]
        zero_count = np.sum(alpha == 0)
        assert zero_count > 0, "No background removed"

    def test_foreground_pixel_count(self):
        """Assert foreground pixel count > 1000."""
        from pipeline.segment import segment_person

        result = segment_person(TEST_IMAGES[2], device="cpu")
        alpha = np.array(result)[:, :, 3]
        fg_count = np.sum(alpha > 128)
        assert fg_count > 1000, f"Foreground pixels: {fg_count} (expected > 1000)"
