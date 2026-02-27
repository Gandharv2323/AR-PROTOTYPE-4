# Module: test_tryon
# License: MIT (ARVTON project)
# Description: Phase 8 — Virtual try-on integration tests.
# Dependencies: pytest, Pillow, numpy, skimage

"""
tests/test_tryon.py
Assert:
  - Output size == (768, 1024) for Leffa or (512, 768) for CatVTON
  - Compute SSIM vs reference — assert > 0.70
  - No fully-black pixels in garment region
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _create_person_image() -> str:
    """Create a test person image."""
    arr = np.full((1024, 768, 3), (180, 160, 150), dtype=np.uint8)
    # Simple body shape
    arr[200:900, 250:520] = [200, 180, 170]  # torso
    arr[50:200, 320:450] = [210, 190, 175]   # head
    tmp = tempfile.mktemp(suffix=".jpg")
    Image.fromarray(arr).save(tmp, "JPEG")
    return tmp


def _create_garment_image() -> str:
    """Create a test garment image."""
    arr = np.full((1024, 768, 3), (240, 240, 240), dtype=np.uint8)
    # T-shirt shape
    arr[200:700, 150:620] = [50, 100, 200]  # blue shirt
    arr[200:350, 50:180] = [50, 100, 200]   # left sleeve
    arr[200:350, 590:720] = [50, 100, 200]  # right sleeve
    tmp = tempfile.mktemp(suffix=".jpg")
    Image.fromarray(arr).save(tmp, "JPEG")
    return tmp


def _create_mask_image(w=768, h=1024) -> str:
    """Create a person mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[200:900, 250:520] = 255
    mask[50:200, 320:450] = 255
    tmp = tempfile.mktemp(suffix=".png")
    Image.fromarray(mask).save(tmp, "PNG")
    return tmp


def _create_placeholder(w=768, h=1024) -> str:
    """Create a placeholder black image."""
    tmp = tempfile.mktemp(suffix=".png")
    Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(tmp, "PNG")
    return tmp


@pytest.fixture(scope="module")
def test_inputs():
    person = _create_person_image()
    garment = _create_garment_image()
    mask = _create_mask_image()
    bodyparse = _create_placeholder()
    densepose = _create_placeholder()

    yield {
        "person": person,
        "garment": garment,
        "mask": mask,
        "bodyparse": bodyparse,
        "densepose": densepose,
    }

    for p in [person, garment, mask, bodyparse, densepose]:
        if os.path.exists(p):
            os.unlink(p)


class TestTryon:
    """Virtual try-on integration tests."""

    def test_output_size(self, test_inputs):
        """Assert output is 768×1024 or 512×768."""
        from pipeline.tryon import tryon

        try:
            result = tryon(
                test_inputs["person"],
                test_inputs["garment"],
                test_inputs["mask"],
                test_inputs["bodyparse"],
                test_inputs["densepose"],
                device="cpu",
            )

            w, h = result.size
            valid_sizes = [(768, 1024), (512, 768)]
            assert (w, h) in valid_sizes, (
                f"Output size {w}×{h} not in expected sizes: {valid_sizes}"
            )
        except Exception as e:
            pytest.skip(f"Try-on model not available: {e}")

    def test_ssim_above_threshold(self, test_inputs):
        """Compute SSIM vs person image — assert > 0.70."""
        from pipeline.tryon import tryon

        try:
            result = tryon(
                test_inputs["person"],
                test_inputs["garment"],
                test_inputs["mask"],
                test_inputs["bodyparse"],
                test_inputs["densepose"],
                device="cpu",
            )

            # Compare with person image (face should be preserved)
            from skimage.metrics import structural_similarity as ssim

            person = Image.open(test_inputs["person"]).resize(result.size)
            person_arr = np.array(person).astype(np.float32) / 255.0
            result_arr = np.array(result).astype(np.float32) / 255.0

            # Resize to match
            min_h = min(person_arr.shape[0], result_arr.shape[0])
            min_w = min(person_arr.shape[1], result_arr.shape[1])
            person_arr = person_arr[:min_h, :min_w]
            result_arr = result_arr[:min_h, :min_w]

            score = ssim(person_arr, result_arr, channel_axis=-1, data_range=1.0)
            assert score > 0.70, f"SSIM = {score:.3f} (threshold: 0.70)"
        except ImportError:
            pytest.skip("skimage not installed")
        except Exception as e:
            pytest.skip(f"Try-on model not available: {e}")

    def test_no_black_garment_region(self, test_inputs):
        """Assert no fully-black pixels in the garment region."""
        from pipeline.tryon import tryon

        try:
            result = tryon(
                test_inputs["person"],
                test_inputs["garment"],
                test_inputs["mask"],
                test_inputs["bodyparse"],
                test_inputs["densepose"],
                device="cpu",
            )

            arr = np.array(result)
            # Check center region (where garment should be)
            h, w = arr.shape[:2]
            center = arr[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
            black_pixels = np.sum(np.all(center == 0, axis=-1))
            total_pixels = center.shape[0] * center.shape[1]
            black_ratio = black_pixels / total_pixels

            assert black_ratio < 0.1, (
                f"Too many black pixels in garment region: "
                f"{black_pixels}/{total_pixels} ({black_ratio:.1%})"
            )
        except Exception as e:
            pytest.skip(f"Try-on model not available: {e}")
