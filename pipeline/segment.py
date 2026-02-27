# Module: segment
# License: MIT (ARVTON project)
# Description: SAM 2 segmentation module — garment background removal and person isolation.
# Platform: Both (Colab T4/A100 + AMD ROCm)
# Dependencies: torch, segment-anything-2, Pillow, numpy, opencv-python-headless

"""
===================================
PHASE 1 — SEGMENTATION
File: segment.py
===================================

SAM 2 Segmentation Module
==========================
Provides garment background removal and person body isolation using
Meta's SAM 2 (Segment Anything Model 2) with the Hiera-Large architecture.

TRAINING PATH: Skip SAM 2, load masks directly from dataset manifest.
INFERENCE PATH: Always run SAM 2 on new user-uploaded images.

VRAM usage target: under 10GB.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("arvton.segment")

# Module-level cached model — loaded once, reused across calls
_sam2_model = None
_sam2_device = None


# ═══════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════


def _load_sam2(
    checkpoint_dir: Optional[str] = None,
    device: str = "cuda",
):
    """
    Load and cache the SAM 2 Hiera-Large model.
    Subsequent calls return the cached model.

    Args:
        checkpoint_dir: Directory containing SAM 2 checkpoint.
            If None, auto-resolves from config.
        device: Torch device string.

    Returns:
        SAM 2 automatic mask generator.
    """
    global _sam2_model, _sam2_device

    if _sam2_model is not None and _sam2_device == device:
        logger.debug("Returning cached SAM 2 model")
        return _sam2_model

    import torch

    # ROCm compatible: YES — SAM 2 uses standard PyTorch ops
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError:
        logger.error(
            "segment-anything-2 not installed. "
            "Install via: pip install segment-anything-2"
        )
        raise

    # Resolve checkpoint
    if checkpoint_dir is None:
        from pipeline.platform_utils import get_paths
        paths = get_paths()
        checkpoint_dir = str(paths["sam2_checkpoint"])

    ckpt_path = Path(checkpoint_dir)
    ckpt_files = list(ckpt_path.glob("*.pt")) + list(ckpt_path.glob("*.pth"))

    if not ckpt_files:
        logger.info("SAM 2 checkpoint not found. Downloading from HuggingFace...")
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id="facebook/sam2-hiera-large",
                filename="sam2_hiera_large.pt",
                local_dir=str(ckpt_path),
            )
            sam2_ckpt = downloaded
        except Exception as e:
            raise RuntimeError(
                f"Failed to download SAM 2 checkpoint: {e}. "
                f"Manually download from facebook/sam2-hiera-large and place in {ckpt_path}"
            ) from e
    else:
        sam2_ckpt = str(ckpt_files[0])

    logger.info("Loading SAM 2 Hiera-Large from %s", sam2_ckpt)

    try:
        model_cfg = "sam2_hiera_l.yaml"
        sam2 = build_sam2(model_cfg, sam2_ckpt, device=device)
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=1000,
        )

        _sam2_model = mask_generator
        _sam2_device = device

        # Log VRAM usage
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated(0) / (1024 ** 3)
            logger.info("SAM 2 loaded. VRAM used: %.2f GB", vram_used)
        else:
            logger.info("SAM 2 loaded on CPU")

        return mask_generator

    except Exception as e:
        vram_info = ""
        if torch.cuda.is_available():
            vram_info = f" (VRAM: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB)"
        raise RuntimeError(
            f"SAM 2 model loading failed{vram_info}: {e}"
        ) from e


# ═══════════════════════════════════════════════════════════════════════
# GARMENT SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════


def segment_garment(
    image_path: str,
    checkpoint_dir: Optional[str] = None,
    device: str = "cuda",
) -> Image.Image:
    """
    Remove background from a garment flat-lay image.
    Returns an RGBA PNG with clean edges.

    Handles:
        - White backgrounds
        - Gradient backgrounds
        - Shadows
        - Non-square images

    Args:
        image_path: Path to the garment image (JPEG/PNG).
        checkpoint_dir: SAM 2 checkpoint directory (auto-resolved if None).
        device: Torch device string.

    Returns:
        PIL.Image in RGBA mode with transparent background.

    Raises:
        FileNotFoundError: If image_path does not exist.
        RuntimeError: If SAM 2 fails to generate any masks.
    """
    import torch

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Garment image not found: {image_path}")

    logger.info("Segmenting garment: %s", image_path.name)

    # Load image
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Load SAM 2
    try:
        mask_generator = _load_sam2(checkpoint_dir, device)
    except Exception as e:
        vram_used = 0.0
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / (1024 ** 3)
        raise RuntimeError(
            f"SAM 2 loading failed during garment segmentation "
            f"(VRAM used: {vram_used:.2f} GB): {e}"
        ) from e

    # Generate masks
    try:
        masks = mask_generator.generate(image_rgb)
    except Exception as e:
        vram_used = 0.0
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / (1024 ** 3)
        raise RuntimeError(
            f"SAM 2 mask generation failed for garment "
            f"(VRAM used: {vram_used:.2f} GB): {e}"
        ) from e

    if not masks:
        raise RuntimeError(f"SAM 2 generated no masks for garment: {image_path.name}")

    # For garments on white/gradient backgrounds:
    # The garment is typically NOT the largest mask (background is largest).
    # Strategy: Select the mask that is NOT the background.
    # Background heuristic: mask touching all 4 edges of the image.
    h, w = image_rgb.shape[:2]

    garment_mask = None
    best_score = -1.0

    for mask_data in masks:
        seg = mask_data["segmentation"]
        area = mask_data["area"]
        iou = mask_data["predicted_iou"]

        # Check if this mask touches all 4 edges (likely background)
        touches_top = np.any(seg[0, :])
        touches_bottom = np.any(seg[-1, :])
        touches_left = np.any(seg[:, 0])
        touches_right = np.any(seg[:, -1])
        edges_touched = sum([touches_top, touches_bottom, touches_left, touches_right])

        # Garment typically doesn't touch all 4 edges
        if edges_touched >= 4 and area > (h * w * 0.6):
            continue  # Skip background-like masks

        # Prefer larger garment masks with high IoU
        # Score = area-normalized * IoU
        area_ratio = area / (h * w)
        score = area_ratio * iou

        if score > best_score and area_ratio > 0.02:  # Minimum 2% of image
            best_score = score
            garment_mask = seg

    # Fallback: if no good mask found, invert the largest mask (background removal)
    if garment_mask is None:
        masks_sorted = sorted(masks, key=lambda m: m["area"], reverse=True)
        largest_mask = masks_sorted[0]["segmentation"]
        garment_mask = ~largest_mask  # Invert: garment = NOT background

    # Convert to alpha mask
    alpha = (garment_mask.astype(np.uint8)) * 255

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=2)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel, iterations=1)

    # Apply slight Gaussian blur to edges for clean anti-aliasing
    edge_mask = cv2.Canny(alpha, 100, 200)
    edge_dilated = cv2.dilate(edge_mask, kernel, iterations=1)
    alpha_float = alpha.astype(np.float32)
    alpha_blurred = cv2.GaussianBlur(alpha_float, (3, 3), 1.0)
    alpha_final = np.where(edge_dilated > 0, alpha_blurred, alpha_float).astype(np.uint8)

    # Compose RGBA
    rgba = np.dstack([image_rgb, alpha_final])
    result = Image.fromarray(rgba, mode="RGBA")

    logger.info(
        "Garment segmented: %s — foreground pixels: %d",
        image_path.name,
        int(np.sum(alpha_final > 127)),
    )
    return result


# ═══════════════════════════════════════════════════════════════════════
# PERSON SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════


def segment_person(
    image_path: str,
    checkpoint_dir: Optional[str] = None,
    device: str = "cuda",
) -> Image.Image:
    """
    Isolate the human body from the background.
    Returns an RGBA PNG with transparent background.

    Handles:
        - Complex backgrounds
        - Partial occlusion
        - Multiple people — keeps the largest or most centered person

    Args:
        image_path: Path to the person image (JPEG/PNG).
        checkpoint_dir: SAM 2 checkpoint directory (auto-resolved if None).
        device: Torch device string.

    Returns:
        PIL.Image in RGBA mode with the person isolated.

    Raises:
        FileNotFoundError: If image_path does not exist.
        RuntimeError: If SAM 2 fails or no person-like mask is found.
    """
    import torch

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Person image not found: {image_path}")

    logger.info("Segmenting person: %s", image_path.name)

    # Load image
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Load SAM 2
    try:
        mask_generator = _load_sam2(checkpoint_dir, device)
    except Exception as e:
        vram_used = 0.0
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / (1024 ** 3)
        raise RuntimeError(
            f"SAM 2 loading failed during person segmentation "
            f"(VRAM used: {vram_used:.2f} GB): {e}"
        ) from e

    # Generate masks
    try:
        masks = mask_generator.generate(image_rgb)
    except Exception as e:
        vram_used = 0.0
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / (1024 ** 3)
        raise RuntimeError(
            f"SAM 2 mask generation failed for person "
            f"(VRAM used: {vram_used:.2f} GB): {e}"
        ) from e

    if not masks:
        raise RuntimeError(f"SAM 2 generated no masks for person: {image_path.name}")

    h, w = image_rgb.shape[:2]
    center_x, center_y = w / 2, h / 2

    # For person images, the person is typically:
    # 1. One of the larger masks
    # 2. Vertically oriented (taller than wide)
    # 3. Centered in the image
    # We score based on these heuristics.

    best_mask = None
    best_score = -1.0

    for mask_data in masks:
        seg = mask_data["segmentation"]
        area = mask_data["area"]
        iou = mask_data["predicted_iou"]

        area_ratio = area / (h * w)
        if area_ratio < 0.05:  # Too small to be a person
            continue
        if area_ratio > 0.95:  # Entire image — likely background
            continue

        # Compute bounding box
        ys, xs = np.where(seg)
        if len(ys) == 0:
            continue
        bbox_h = ys.max() - ys.min()
        bbox_w = xs.max() - xs.min()
        bbox_center_x = (xs.min() + xs.max()) / 2
        bbox_center_y = (ys.min() + ys.max()) / 2

        # Person heuristics:
        # 1. Vertical aspect ratio (person is taller than wide)
        aspect_score = min(bbox_h / max(bbox_w, 1), 3.0) / 3.0  # Normalize to 0-1

        # 2. Centeredness
        dist_from_center = np.sqrt(
            ((bbox_center_x - center_x) / w) ** 2 +
            ((bbox_center_y - center_y) / h) ** 2
        )
        center_score = 1.0 - min(dist_from_center, 1.0)

        # 3. Size (larger is better, but not too large)
        size_score = min(area_ratio / 0.4, 1.0)

        # Combined score
        score = (aspect_score * 0.3 + center_score * 0.3 + size_score * 0.2 + iou * 0.2)

        if score > best_score:
            best_score = score
            best_mask = seg

    if best_mask is None:
        # Fallback: take the largest non-background mask
        masks_sorted = sorted(masks, key=lambda m: m["area"], reverse=True)
        for m in masks_sorted:
            area_ratio = m["area"] / (h * w)
            if 0.05 < area_ratio < 0.95:
                best_mask = m["segmentation"]
                break

    if best_mask is None:
        raise RuntimeError(f"No person-like mask found in: {image_path.name}")

    # Convert to alpha
    alpha = (best_mask.astype(np.uint8)) * 255

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=3)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel, iterations=1)

    # Fill small holes inside the person mask
    contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Keep only the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        filled_mask = np.zeros_like(alpha)
        cv2.drawContours(filled_mask, [largest_contour], -1, 255, -1)
        alpha = filled_mask

    # Edge anti-aliasing
    edge_mask = cv2.Canny(alpha, 100, 200)
    edge_dilated = cv2.dilate(edge_mask, kernel, iterations=1)
    alpha_float = alpha.astype(np.float32)
    alpha_blurred = cv2.GaussianBlur(alpha_float, (5, 5), 1.5)
    alpha_final = np.where(edge_dilated > 0, alpha_blurred, alpha_float).astype(np.uint8)

    # Compose RGBA
    rgba = np.dstack([image_rgb, alpha_final])
    result = Image.fromarray(rgba, mode="RGBA")

    logger.info(
        "Person segmented: %s — foreground pixels: %d",
        image_path.name,
        int(np.sum(alpha_final > 127)),
    )
    return result


# ═══════════════════════════════════════════════════════════════════════
# BATCH PROCESSING
# ═══════════════════════════════════════════════════════════════════════


def batch_segment(
    image_dir: str,
    output_dir: str,
    mode: str = "garment",
    checkpoint_dir: Optional[str] = None,
    device: str = "cuda",
    batch_log_interval: int = 8,
) -> int:
    """
    Process an entire directory of images, segmenting each one.
    Skips already-processed images by checking output_dir.

    Args:
        image_dir: Directory containing input images.
        output_dir: Directory to save RGBA PNG outputs.
        mode: 'garment' or 'person' — determines segmentation strategy.
        checkpoint_dir: SAM 2 checkpoint directory.
        device: Torch device string.
        batch_log_interval: Print progress every N images.

    Returns:
        int: Number of successfully processed images.
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode not in ("garment", "person"):
        raise ValueError(f"Invalid mode '{mode}'. Must be 'garment' or 'person'.")

    segment_fn = segment_garment if mode == "garment" else segment_person

    # Find all images
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = sorted([
        f for f in image_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    total = len(image_files)
    logger.info("Batch segment: %d images in '%s' (mode=%s)", total, image_dir.name, mode)

    processed = 0
    skipped = 0
    failed = 0

    for i, img_path in enumerate(image_files):
        output_path = output_dir / f"{img_path.stem}_seg.png"

        # Skip already processed
        if output_path.exists():
            skipped += 1
            continue

        try:
            result = segment_fn(
                str(img_path),
                checkpoint_dir=checkpoint_dir,
                device=device,
            )
            result.save(str(output_path), "PNG")
            processed += 1

        except Exception as e:
            logger.warning("Failed to segment %s: %s", img_path.name, str(e))
            failed += 1

        # Progress logging
        done = processed + skipped + failed
        if done % batch_log_interval == 0 or done == total:
            print(f"Processed {done}/{total} images "
                  f"(success={processed}, skipped={skipped}, failed={failed})")

    logger.info(
        "Batch segment complete: %d processed, %d skipped, %d failed (total: %d)",
        processed, skipped, failed, total,
    )
    return processed


# ═══════════════════════════════════════════════════════════════════════
# CLEANUP
# ═══════════════════════════════════════════════════════════════════════


def clear_model_cache() -> None:
    """
    Clear the cached SAM 2 model to free VRAM.
    Call this when transitioning between pipeline stages.
    """
    global _sam2_model, _sam2_device
    if _sam2_model is not None:
        del _sam2_model
        _sam2_model = None
        _sam2_device = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("SAM 2 model cache cleared. VRAM freed.")
        except ImportError:
            pass


# ═══════════════════════════════════════════════════════════════════════
# MAIN (CLI usage)
# ═══════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ARVTON — SAM 2 Segmentation Module")
    parser.add_argument("--image", type=str, help="Single image to segment")
    parser.add_argument("--dir", type=str, help="Directory of images to batch segment")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--mode", type=str, default="garment", choices=["garment", "person"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    from pipeline.platform_utils import setup_logging
    setup_logging("INFO")

    if args.image:
        fn = segment_garment if args.mode == "garment" else segment_person
        result = fn(args.image, device=args.device)
        out_path = Path(args.output) / f"{Path(args.image).stem}_seg.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(str(out_path), "PNG")
        print(f"Saved: {out_path}")
    elif args.dir:
        count = batch_segment(args.dir, args.output, mode=args.mode, device=args.device)
        print(f"Processed {count} images")
    else:
        parser.print_help()

    print("\nPhase 1 complete. Files written: [segment.py]")
    print("Reply 'continue' to proceed to Phase 2.")
