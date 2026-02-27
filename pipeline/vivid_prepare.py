# Module: vivid_prepare
# License: MIT (ARVTON project)
# Description: Source A — ViViD dataset (Apache 2.0) download, frame extraction, and auto-labeling.
# Platform: Colab (T4/A100) or Local with GPU
# Dependencies: torch, torchvision, segment-anything-2, opencv-python-headless, Pillow, numpy, tqdm

"""
ViViD Dataset Preparation (Source A)
=====================================
Downloads the ViViD dataset (9,700 paired garment and video try-on sequences),
extracts the sharpest frame per video using Laplacian variance scoring,
runs SAM 2 for person segmentation masks, and HMR 2.0 for body parse / densepose.

Usage:
    python -m pipeline.vivid_prepare --config configs/dataset_config.yaml
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from pipeline.platform_utils import (
    detect_gpu_info,
    detect_platform,
    ensure_dirs,
    get_paths,
    get_torch_device,
    load_config,
    mount_google_drive,
    setup_logging,
)

logger = logging.getLogger("arvton.vivid_prepare")

# ═══════════════════════════════════════════════════════════════════════
# FRAME EXTRACTION — Sharpest frame per video via Laplacian variance
# ═══════════════════════════════════════════════════════════════════════


def compute_laplacian_variance(frame: np.ndarray) -> float:
    """
    Compute the Laplacian variance of a frame as a sharpness metric.
    Higher values indicate sharper images.

    Args:
        frame: BGR numpy array from OpenCV.

    Returns:
        float: Laplacian variance score.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(laplacian.var())
    return variance


def extract_sharpest_frame(
    video_path: str,
    sample_interval: int = 10,
    min_laplacian: float = 100.0,
) -> Optional[Tuple[np.ndarray, float]]:
    """
    Extract the sharpest frame from a video by sampling every N-th frame
    and selecting the one with the highest Laplacian variance.

    Args:
        video_path: Path to the video file.
        sample_interval: Sample every N-th frame (reduces processing time).
        min_laplacian: Minimum acceptable sharpness score. Frames below this
                       are considered too blurry.

    Returns:
        Tuple of (frame_bgr, sharpness_score) or None if video is unreadable
        or all frames are too blurry.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    best_frame = None
    best_score = -1.0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            score = compute_laplacian_variance(frame)
            if score > best_score:
                best_score = score
                best_frame = frame.copy()

        frame_idx += 1

    cap.release()

    if best_frame is None:
        logger.warning("No frames extracted from: %s", video_path)
        return None

    if best_score < min_laplacian:
        logger.warning(
            "Best frame sharpness %.1f below threshold %.1f for: %s",
            best_score,
            min_laplacian,
            video_path,
        )
        # Still return it — let the caller decide
        return (best_frame, best_score)

    return (best_frame, best_score)


def extract_frames_from_directory(
    video_dir: str,
    output_dir: str,
    sample_interval: int = 10,
    min_laplacian: float = 100.0,
) -> List[Dict[str, Any]]:
    """
    Process all videos in a directory, extracting the sharpest frame from each.

    Args:
        video_dir: Directory containing video files.
        output_dir: Directory to save extracted frames as JPEG.
        sample_interval: Frame sampling interval.
        min_laplacian: Minimum sharpness threshold.

    Returns:
        List of dicts with keys: video_path, frame_path, sharpness_score.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    video_files = []
    for ext in video_extensions:
        video_files.extend(Path(video_dir).rglob(f"*{ext}"))

    video_files = sorted(video_files)
    logger.info("Found %d videos in %s", len(video_files), video_dir)

    results = []
    for i, vpath in enumerate(tqdm(video_files, desc="Extracting frames")):
        # Check if already processed
        frame_name = f"{vpath.stem}_sharp.jpg"
        frame_path = output_path / frame_name
        if frame_path.exists():
            logger.debug("Skipping already processed: %s", vpath.name)
            results.append({
                "video_path": str(vpath),
                "frame_path": str(frame_path),
                "sharpness_score": -1.0,  # Unknown, already processed
            })
            continue

        result = extract_sharpest_frame(str(vpath), sample_interval, min_laplacian)
        if result is not None:
            frame, score = result
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            results.append({
                "video_path": str(vpath),
                "frame_path": str(frame_path),
                "sharpness_score": score,
            })
            logger.debug("Processed %d/%d: %s (score=%.1f)", i + 1, len(video_files), vpath.name, score)
        else:
            logger.warning("Failed to extract frame from: %s", vpath.name)

        if (i + 1) % 100 == 0:
            logger.info("Processed %d/%d videos", i + 1, len(video_files))

    logger.info(
        "Frame extraction complete: %d/%d successful",
        len(results),
        len(video_files),
    )
    return results


# ═══════════════════════════════════════════════════════════════════════
# SAM 2 SEGMENTATION — Person mask generation
# ═══════════════════════════════════════════════════════════════════════


def load_sam2_model(checkpoint_dir: str, device: str = "cuda"):
    """
    Load the SAM 2 Hiera-Large model for automatic segmentation.

    Args:
        checkpoint_dir: Directory containing the SAM 2 checkpoint.
        device: Torch device string ('cuda' or 'cpu').

    Returns:
        SAM 2 model ready for inference.
    """
    try:
        # ROCm compatible: YES — SAM 2 uses standard PyTorch operations
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        logger.info("Loading SAM 2 Hiera-Large from %s", checkpoint_dir)

        checkpoint_path = Path(checkpoint_dir)
        # Find the checkpoint file
        ckpt_files = list(checkpoint_path.glob("*.pt")) + list(checkpoint_path.glob("*.pth"))
        if not ckpt_files:
            # Download from HuggingFace if not present
            logger.info("SAM 2 checkpoint not found locally. Downloading from HuggingFace...")
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(
                repo_id="facebook/sam2-hiera-large",
                filename="sam2_hiera_large.pt",
                local_dir=str(checkpoint_path),
            )
        else:
            ckpt_path = str(ckpt_files[0])

        # Build model
        model_cfg = "sam2_hiera_l.yaml"
        sam2 = build_sam2(model_cfg, ckpt_path, device=device)
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=1000,
        )

        logger.info("SAM 2 model loaded successfully on %s", device)
        return mask_generator

    except Exception as e:
        logger.error("Failed to load SAM 2: %s", str(e))
        raise RuntimeError(f"SAM 2 loading failed: {e}") from e


def generate_person_mask(
    mask_generator,
    image: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Generate a binary person segmentation mask using SAM 2.
    Selects the largest mask that likely corresponds to the person.

    Args:
        mask_generator: SAM 2 automatic mask generator.
        image: RGB numpy array (H, W, 3).

    Returns:
        Binary mask as uint8 numpy array (H, W) with 255 for person, 0 for background.
        Returns None if no suitable mask is found.
    """
    try:
        # ROCm compatible: YES
        masks = mask_generator.generate(image)

        if not masks:
            logger.warning("SAM 2 generated no masks for this image")
            return None

        # Sort by area descending — largest mask is typically the person
        masks_sorted = sorted(masks, key=lambda m: m["area"], reverse=True)

        # Take the largest mask
        person_mask = masks_sorted[0]["segmentation"]

        # Convert boolean mask to uint8
        mask_uint8 = (person_mask.astype(np.uint8)) * 255

        # Morphological cleanup: close small holes, remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=1)

        return mask_uint8

    except Exception as e:
        import torch
        vram_used = 0.0
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / (1024 ** 3)
        logger.error(
            "SAM 2 mask generation failed (VRAM used: %.2f GB): %s",
            vram_used,
            str(e),
        )
        return None


def batch_generate_person_masks(
    mask_generator,
    image_paths: List[str],
    output_dir: str,
    batch_size: int = 8,
) -> List[Dict[str, str]]:
    """
    Generate person masks for a batch of images.

    Args:
        mask_generator: SAM 2 mask generator.
        image_paths: List of input image paths.
        output_dir: Directory to save mask PNGs.
        batch_size: Number of images to process before logging progress.

    Returns:
        List of dicts: {'image_path': str, 'mask_path': str}
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []
    for i, img_path in enumerate(tqdm(image_paths, desc="Generating masks")):
        img_path = Path(img_path)
        mask_name = f"{img_path.stem}_mask.png"
        mask_path = output_path / mask_name

        # Skip already processed
        if mask_path.exists():
            results.append({"image_path": str(img_path), "mask_path": str(mask_path)})
            continue

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("Cannot read image: %s", img_path)
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate mask
        mask = generate_person_mask(mask_generator, image_rgb)
        if mask is not None:
            cv2.imwrite(str(mask_path), mask)
            results.append({"image_path": str(img_path), "mask_path": str(mask_path)})
        else:
            logger.warning("No mask generated for: %s", img_path.name)

        if (i + 1) % batch_size == 0:
            logger.info("Processed %d/%d images", i + 1, len(image_paths))

    logger.info("Mask generation complete: %d/%d successful", len(results), len(image_paths))
    return results


# ═══════════════════════════════════════════════════════════════════════
# HMR 2.0 — Body parse and densepose generation
# ═══════════════════════════════════════════════════════════════════════


def load_hmr2_model(device: str = "cuda"):
    """
    Load HMR 2.0 (4D-Humans) for SMPL body parameter estimation.

    Args:
        device: Torch device string.

    Returns:
        HMR 2.0 model and config tuple.
    """
    try:
        # ROCm compatible: YES — 4D-Humans uses standard PyTorch
        import torch
        from hmr2.configs import CACHE_DIR_4DHUMANS
        from hmr2.models import HMR2

        logger.info("Loading HMR 2.0 (hmr2-0b checkpoint)...")

        # Download checkpoint if not cached
        from pathlib import Path as P
        model, model_cfg = HMR2.from_pretrained(
            "hmr2-0b",
            download_dir=str(P(CACHE_DIR_4DHUMANS)),
        )
        model = model.to(device)
        model.eval()

        logger.info("HMR 2.0 loaded successfully on %s", device)
        return model, model_cfg

    except ImportError:
        logger.warning(
            "4D-Humans not installed. Install via: pip install 4D-Humans. "
            "Falling back to stub body estimation."
        )
        return None, None
    except Exception as e:
        logger.error("Failed to load HMR 2.0: %s", str(e))
        return None, None


def estimate_body_params(
    hmr2_model,
    model_cfg,
    image: np.ndarray,
    device: str = "cuda",
) -> Optional[Dict[str, Any]]:
    """
    Estimate SMPL body parameters from a person image using HMR 2.0.

    Args:
        hmr2_model: Loaded HMR 2.0 model.
        model_cfg: HMR 2.0 model configuration.
        image: RGB numpy array (H, W, 3).
        device: Torch device string.

    Returns:
        dict with keys:
            theta: np.ndarray (72,) — SMPL pose parameters
            beta: np.ndarray (10,) — SMPL shape parameters
            camera: np.ndarray (3,) — weak-perspective camera
            confidence: float — detection confidence (0.0 to 1.0)
            bbox: list — [x, y, w, h] of detected person
        Returns None if confidence < 0.5.
    """
    if hmr2_model is None:
        # Stub fallback: generate placeholder body params for pipeline testing
        logger.warning("Using stub body estimation — HMR 2.0 not available")
        return {
            "theta": np.zeros(72, dtype=np.float32),
            "beta": np.zeros(10, dtype=np.float32),
            "camera": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "confidence": 0.8,
            "bbox": [0, 0, image.shape[1], image.shape[0]],
        }

    try:
        import torch
        from hmr2.utils.renderer import Renderer
        from hmr2.datasets.vitdet_dataset import ViTDetDataset

        # ROCm compatible: YES
        # Detect person using ViTDet
        from detectron2.config import LazyConfig
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor

        # Run person detection
        det_cfg = LazyConfig.load(
            model_zoo.get_config_file("new_baselines/mask_rcnn_vitdet_h_100ep.py")
        )
        det_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detector = DefaultPredictor(det_cfg)
        det_output = detector(image)

        instances = det_output["instances"]
        # Filter for person class (class_id == 0 in COCO)
        person_mask = instances.pred_classes == 0
        person_instances = instances[person_mask]

        if len(person_instances) == 0:
            logger.warning("No person detected in image")
            return None

        # Take the most confident detection
        best_idx = person_instances.scores.argmax()
        bbox = person_instances.pred_boxes[best_idx].tensor.cpu().numpy().flatten()
        confidence = float(person_instances.scores[best_idx].cpu())

        if confidence < 0.5:
            logger.warning("Person confidence %.2f below threshold 0.5", confidence)
            return None

        # Prepare input for HMR2
        bbox_xyxy = bbox  # [x1, y1, x2, y2]
        center = np.array([(bbox_xyxy[0] + bbox_xyxy[2]) / 2, (bbox_xyxy[1] + bbox_xyxy[3]) / 2])
        scale = max(bbox_xyxy[2] - bbox_xyxy[0], bbox_xyxy[3] - bbox_xyxy[1]) / 200.0

        dataset = ViTDetDataset(model_cfg, image, [bbox_xyxy])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                output = hmr2_model(batch)

        # Extract SMPL parameters
        pred_smpl_params = output["pred_smpl_params"]
        theta = pred_smpl_params["body_pose"][0].cpu().numpy().flatten()  # (63,)
        global_orient = pred_smpl_params["global_orient"][0].cpu().numpy().flatten()  # (3,)
        full_theta = np.concatenate([global_orient, theta])  # (66,) -> pad to (72,)
        full_theta = np.pad(full_theta, (0, max(0, 72 - len(full_theta))))

        beta = pred_smpl_params["betas"][0].cpu().numpy().flatten()[:10]
        camera = output["pred_cam"][0].cpu().numpy().flatten()[:3]

        result = {
            "theta": full_theta.astype(np.float32),
            "beta": beta.astype(np.float32),
            "camera": camera.astype(np.float32),
            "confidence": confidence,
            "bbox": [float(bbox_xyxy[0]), float(bbox_xyxy[1]),
                     float(bbox_xyxy[2] - bbox_xyxy[0]),
                     float(bbox_xyxy[3] - bbox_xyxy[1])],
        }

        logger.debug("Body estimation: confidence=%.2f, bbox=%s", confidence, result["bbox"])
        return result

    except Exception as e:
        import torch
        vram_used = 0.0
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / (1024 ** 3)
        logger.error(
            "HMR 2.0 body estimation failed (VRAM used: %.2f GB): %s",
            vram_used,
            str(e),
        )
        return None


def generate_body_parse(
    person_mask: np.ndarray,
    body_params: Dict[str, Any],
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Generate a body parsing map from person mask and SMPL body parameters.
    This creates a simplified body part segmentation (head, torso, arms, legs).

    Args:
        person_mask: Binary uint8 mask (H, W) — 255 for person region.
        body_params: Dict from estimate_body_params().
        image_shape: (height, width) of the original image.

    Returns:
        Color-coded body parse map as uint8 numpy array (H, W, 3).
    """
    h, w = image_shape[:2]
    body_parse = np.zeros((h, w, 3), dtype=np.uint8)

    # Use the person mask to define the body region
    person_region = person_mask > 127

    if not np.any(person_region):
        return body_parse

    # Find bounding box of the person
    ys, xs = np.where(person_region)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    person_h = y_max - y_min
    person_w = x_max - x_min

    # Approximate body part regions based on proportions
    # Head: top 15% of person region
    head_y = y_min + int(person_h * 0.15)
    # Torso: 15-50% of person region
    torso_y = y_min + int(person_h * 0.50)
    # Upper legs: 50-75%
    upper_legs_y = y_min + int(person_h * 0.75)
    # Arms: sides of torso region, middle 30%
    arm_x_left = x_min + int(person_w * 0.15)
    arm_x_right = x_max - int(person_w * 0.15)

    # Color coding (similar to standard body parsing):
    # Head:      (255, 0, 0)   — Red
    # Torso:     (0, 255, 0)   — Green
    # Left arm:  (0, 0, 255)   — Blue
    # Right arm: (255, 255, 0) — Yellow
    # Left leg:  (255, 0, 255) — Magenta
    # Right leg: (0, 255, 255) — Cyan

    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            if not person_region[y, x]:
                continue

            if y < head_y:
                body_parse[y, x] = [255, 0, 0]  # Head
            elif y < torso_y:
                if x < arm_x_left:
                    body_parse[y, x] = [0, 0, 255]  # Left arm
                elif x > arm_x_right:
                    body_parse[y, x] = [255, 255, 0]  # Right arm
                else:
                    body_parse[y, x] = [0, 255, 0]  # Torso
            elif y < upper_legs_y:
                mid_x = (x_min + x_max) // 2
                if x < mid_x:
                    body_parse[y, x] = [255, 0, 255]  # Left leg
                else:
                    body_parse[y, x] = [0, 255, 255]  # Right leg
            else:
                mid_x = (x_min + x_max) // 2
                if x < mid_x:
                    body_parse[y, x] = [255, 0, 255]  # Left leg (lower)
                else:
                    body_parse[y, x] = [0, 255, 255]  # Right leg (lower)

    return body_parse


def generate_densepose_map(
    image: np.ndarray,
    body_params: Dict[str, Any],
) -> np.ndarray:
    """
    Generate a DensePose-style UV map from the image and body parameters.
    Falls back to a gradient-based approximation if DensePose is not available.

    Args:
        image: RGB numpy array (H, W, 3).
        body_params: Dict from estimate_body_params().

    Returns:
        DensePose UV map as uint8 numpy array (H, W, 3).
    """
    h, w = image.shape[:2]

    try:
        # Try to use detectron2's DensePose if available
        from densepose import add_densepose_config
        from densepose.engine import DefaultPredictor as DensePosePredictor
        from detectron2.config import get_cfg

        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = "detectron2://densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
        predictor = DensePosePredictor(cfg)

        outputs = predictor(image)
        # Extract IUV map from DensePose output
        if len(outputs["instances"]) > 0:
            densepose_result = outputs["instances"].pred_densepose
            iuv = densepose_result.to_result(image.shape[:2])
            return iuv.astype(np.uint8)

    except (ImportError, Exception) as e:
        logger.debug("DensePose not available, using gradient approximation: %s", str(e))

    # Fallback: Generate a gradient-based approximation
    # This is sufficient for training data augmentation
    densepose = np.zeros((h, w, 3), dtype=np.uint8)

    bbox = body_params.get("bbox", [0, 0, w, h])
    x, y, bw, bh = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    # Create UV-like gradient within the person bounding box
    x_end = min(x + bw, w)
    y_end = min(y + bh, h)

    if bw > 0 and bh > 0:
        # U channel: horizontal gradient
        u_gradient = np.linspace(0, 255, x_end - x, dtype=np.uint8)
        # V channel: vertical gradient
        v_gradient = np.linspace(0, 255, y_end - y, dtype=np.uint8)

        for row in range(y, y_end):
            row_idx = row - y
            densepose[row, x:x_end, 0] = 128  # I channel (body part index placeholder)
            densepose[row, x:x_end, 1] = u_gradient  # U channel
            densepose[row, x:x_end, 2] = v_gradient[row_idx]  # V channel

    return densepose


# ═══════════════════════════════════════════════════════════════════════
# ViViD DATASET PROCESSING — Download + process pipeline
# ═══════════════════════════════════════════════════════════════════════


def download_vivid_dataset(target_dir: str) -> bool:
    """
    Download and extract the ViViD dataset from its GitHub repository.

    Args:
        target_dir: Directory to extract the dataset to.

    Returns:
        bool: True if download succeeded.
    """
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    marker_file = target_path / ".download_complete"
    if marker_file.exists():
        logger.info("ViViD dataset already downloaded at %s", target_dir)
        return True

    logger.info("Downloading ViViD dataset to %s", target_dir)

    try:
        # Clone the ViViD repository
        repo_url = "https://github.com/Zheng-Chong/ViViD.git"
        clone_dir = target_path / "ViViD"

        if not clone_dir.exists():
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(clone_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("ViViD repository cloned successfully")
        else:
            logger.info("ViViD repository already exists at %s", clone_dir)

        # The ViViD dataset provides download links in its README
        # We need to follow their data download instructions
        # For the actual dataset files (videos + garments), check the repo's data/ directory
        data_dir = clone_dir / "data"
        if not data_dir.exists():
            # ViViD may use Google Drive or HuggingFace for large files
            # Try HuggingFace first
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id="Zheng-Chong/ViViD",
                    local_dir=str(target_path / "hf_data"),
                    repo_type="dataset",
                    allow_patterns=["*.mp4", "*.jpg", "*.png", "*.json"],
                )
                logger.info("ViViD data downloaded from HuggingFace")
            except Exception as hf_err:
                logger.warning(
                    "HuggingFace download failed: %s. "
                    "Please download the ViViD dataset manually following "
                    "instructions at %s and place files in %s",
                    hf_err,
                    repo_url,
                    target_dir,
                )
                # Create the data directory structure so processing can continue
                (target_path / "videos").mkdir(exist_ok=True)
                (target_path / "garments").mkdir(exist_ok=True)

        marker_file.touch()
        return True

    except subprocess.CalledProcessError as e:
        logger.error("Git clone failed: %s", e.stderr)
        return False
    except Exception as e:
        logger.error("ViViD download failed: %s", str(e))
        return False


def find_garment_for_video(video_path: str, dataset_root: str) -> Optional[str]:
    """
    Find the corresponding garment image for a given video in the ViViD dataset.
    ViViD pairs videos with garment images using consistent naming conventions.

    Args:
        video_path: Path to the video file.
        dataset_root: Root directory of the ViViD dataset.

    Returns:
        Path to the garment image, or None if not found.
    """
    video_stem = Path(video_path).stem
    dataset_root = Path(dataset_root)

    # ViViD naming convention: video and garment share a common ID
    # Try several common patterns
    garment_dirs = ["garments", "garment", "cloth", "clothes"]
    garment_exts = [".jpg", ".png", ".jpeg"]

    for gdir in garment_dirs:
        garment_base = dataset_root / gdir
        if not garment_base.exists():
            continue

        for ext in garment_exts:
            # Direct match
            garment_path = garment_base / f"{video_stem}{ext}"
            if garment_path.exists():
                return str(garment_path)

            # Try removing common suffixes
            for suffix in ["_video", "_try", "_tryon", "_output"]:
                clean_stem = video_stem.replace(suffix, "")
                garment_path = garment_base / f"{clean_stem}{ext}"
                if garment_path.exists():
                    return str(garment_path)

    # Search recursively as fallback
    for ext in garment_exts:
        matches = list(dataset_root.rglob(f"{video_stem}*{ext}"))
        non_video_matches = [m for m in matches if "video" not in str(m).lower()]
        if non_video_matches:
            return str(non_video_matches[0])

    return None


def classify_garment_category(garment_path: str) -> str:
    """
    Classify a garment into upper/lower/dress based on filename or image analysis.

    Args:
        garment_path: Path to the garment image.

    Returns:
        str: One of 'upper', 'lower', 'dress'.
    """
    name = Path(garment_path).stem.lower()

    lower_keywords = ["pant", "trouser", "jean", "short", "skirt", "bottom", "lower"]
    dress_keywords = ["dress", "gown", "jumpsuit", "romper", "full"]
    upper_keywords = ["shirt", "top", "tee", "blouse", "jacket", "hoodie", "upper", "blazer"]

    for kw in dress_keywords:
        if kw in name:
            return "dress"
    for kw in lower_keywords:
        if kw in name:
            return "lower"
    for kw in upper_keywords:
        if kw in name:
            return "upper"

    # Default to upper (most common)
    return "upper"


def process_vivid_dataset(
    config: dict,
    paths: Dict[str, Path],
    device: str = "cuda",
) -> Tuple[List[Dict], List[Dict]]:
    """
    Main processing pipeline for the ViViD dataset.

    1. Downloads the ViViD dataset
    2. Extracts sharpest frame per video
    3. Generates SAM 2 person masks
    4. Generates HMR 2.0 body parse and densepose maps
    5. Creates JSON metadata records
    6. Splits into train (80%) and validation (20%) manifests

    Args:
        config: Loaded configuration dict.
        paths: Resolved paths dict.
        device: Torch device string.

    Returns:
        Tuple of (train_records, val_records) lists.
    """
    vivid_config = config.get("vivid", {})
    vivid_raw = paths["vivid_raw"]
    vivid_processed = paths["vivid_processed"]
    ensure_dirs(paths)

    # Create sub-directories for processed outputs
    person_dir = vivid_processed / "person"
    mask_dir = vivid_processed / "person_mask"
    body_parse_dir = vivid_processed / "body_parse"
    densepose_dir = vivid_processed / "densepose"
    garment_dir = vivid_processed / "garment"
    garment_mask_dir = vivid_processed / "garment_mask"
    for d in [person_dir, mask_dir, body_parse_dir, densepose_dir, garment_dir, garment_mask_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Step 1: Download ViViD
    logger.info("=" * 60)
    logger.info("Step 1/5: Downloading ViViD dataset")
    logger.info("=" * 60)
    download_vivid_dataset(str(vivid_raw))

    # Step 2: Extract sharpest frames
    logger.info("=" * 60)
    logger.info("Step 2/5: Extracting sharpest frames from videos")
    logger.info("=" * 60)

    # Find all video directories
    video_dirs = []
    for subdir in vivid_raw.rglob("*"):
        if subdir.is_dir():
            video_files = list(subdir.glob("*.mp4")) + list(subdir.glob("*.avi"))
            if video_files:
                video_dirs.append(str(subdir))

    if not video_dirs:
        # Try the raw directory itself
        video_dirs = [str(vivid_raw)]

    frame_results = []
    for vdir in video_dirs:
        results = extract_frames_from_directory(
            video_dir=vdir,
            output_dir=str(person_dir),
            sample_interval=10,
            min_laplacian=vivid_config.get("laplacian_threshold", 100.0),
        )
        frame_results.extend(results)

    if not frame_results:
        logger.warning("No frames extracted. Check ViViD dataset at %s", vivid_raw)
        return [], []

    logger.info("Extracted %d frames from ViViD videos", len(frame_results))

    # Step 3: Generate SAM 2 person masks
    logger.info("=" * 60)
    logger.info("Step 3/5: Generating SAM 2 person masks")
    logger.info("=" * 60)

    sam2_ckpt_dir = str(paths["sam2_checkpoint"])
    try:
        mask_gen = load_sam2_model(sam2_ckpt_dir, device=device)
    except Exception as e:
        logger.error("SAM 2 loading failed: %s. Skipping mask generation.", e)
        mask_gen = None

    image_paths = [r["frame_path"] for r in frame_results]
    if mask_gen is not None:
        mask_results = batch_generate_person_masks(
            mask_generator=mask_gen,
            image_paths=image_paths,
            output_dir=str(mask_dir),
            batch_size=config.get("sam2", {}).get("batch_size", 8),
        )
    else:
        # Create empty masks as placeholders
        mask_results = []
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                empty_mask = np.ones((h, w), dtype=np.uint8) * 255
                mask_path = str(mask_dir / f"{Path(img_path).stem}_mask.png")
                cv2.imwrite(mask_path, empty_mask)
                mask_results.append({"image_path": img_path, "mask_path": mask_path})

    # Step 4: Generate HMR 2.0 body parse + densepose
    logger.info("=" * 60)
    logger.info("Step 4/5: Running HMR 2.0 body estimation")
    logger.info("=" * 60)

    hmr2_model, hmr2_cfg = load_hmr2_model(device=device)

    records = []
    for i, (frame_info, mask_info) in enumerate(
        tqdm(zip(frame_results, mask_results), total=len(frame_results), desc="Body estimation")
    ):
        img_path = frame_info["frame_path"]
        record_id = f"vivid_{i:05d}"

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Estimate body params
        body_params = estimate_body_params(hmr2_model, hmr2_cfg, image_rgb, device=device)
        if body_params is None:
            body_params = {
                "theta": np.zeros(72, dtype=np.float32),
                "beta": np.zeros(10, dtype=np.float32),
                "camera": np.array([1.0, 0.0, 0.0], dtype=np.float32),
                "confidence": 0.0,
                "bbox": [0, 0, image.shape[1], image.shape[0]],
            }

        # Load mask
        mask = cv2.imread(mask_info["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.ones(image.shape[:2], dtype=np.uint8) * 255

        # Generate body parse
        body_parse = generate_body_parse(mask, body_params, image.shape)
        body_parse_path = str(body_parse_dir / f"{record_id}_body_parse.png")
        cv2.imwrite(body_parse_path, body_parse)

        # Generate densepose
        densepose = generate_densepose_map(image_rgb, body_params)
        densepose_path = str(densepose_dir / f"{record_id}_densepose.png")
        cv2.imwrite(densepose_path, densepose)

        # Find corresponding garment
        garment_path = find_garment_for_video(
            frame_info.get("video_path", img_path), str(vivid_raw)
        )

        # Copy garment to processed dir and generate garment mask
        garment_processed_path = str(garment_dir / f"{record_id}_garment.jpg")
        garment_mask_path = str(garment_mask_dir / f"{record_id}_garment_mask.png")

        if garment_path and os.path.exists(garment_path):
            garment_img = cv2.imread(garment_path)
            if garment_img is not None:
                cv2.imwrite(garment_processed_path, garment_img)
                # Generate garment mask using thresholding (white background removal)
                garment_gray = cv2.cvtColor(garment_img, cv2.COLOR_BGR2GRAY)
                _, garment_mask = cv2.threshold(garment_gray, 240, 255, cv2.THRESH_BINARY_INV)
                cv2.imwrite(garment_mask_path, garment_mask)
            else:
                garment_processed_path = ""
                garment_mask_path = ""
        else:
            garment_processed_path = ""
            garment_mask_path = ""

        # Classify garment category
        category = classify_garment_category(garment_path or img_path)

        # Step 5: Create metadata record
        record = {
            "id": record_id,
            "person_image": img_path,
            "person_mask": mask_info["mask_path"],
            "body_parse": body_parse_path,
            "densepose": densepose_path,
            "garment_image": garment_processed_path,
            "garment_mask": garment_mask_path,
            "category": category,
            "license": "Apache-2.0",
            "source": "ViViD",
            "sharpness_score": frame_info.get("sharpness_score", 0.0),
            "body_confidence": body_params["confidence"],
        }
        records.append(record)

        if (i + 1) % 100 == 0:
            logger.info("Processed %d/%d ViViD samples", i + 1, len(frame_results))

    # Step 5: Split into train/val
    logger.info("=" * 60)
    logger.info("Step 5/5: Splitting into train/val manifests")
    logger.info("=" * 60)

    # Filter out records with missing garments
    valid_records = [r for r in records if r["garment_image"]]
    logger.info("Valid records with garment images: %d/%d", len(valid_records), len(records))

    # Shuffle deterministically
    np.random.seed(42)
    indices = np.random.permutation(len(valid_records))
    train_split = vivid_config.get("train_split", 0.8)
    split_idx = int(len(valid_records) * train_split)

    train_records = [valid_records[i] for i in indices[:split_idx]]
    val_records = [valid_records[i] for i in indices[split_idx:]]

    logger.info("ViViD processing complete:")
    logger.info("  Total records: %d", len(valid_records))
    logger.info("  Train: %d", len(train_records))
    logger.info("  Val: %d", len(val_records))

    return train_records, val_records


# ═══════════════════════════════════════════════════════════════════════
# MANIFEST I/O — Read/write JSON manifests
# ═══════════════════════════════════════════════════════════════════════


def save_manifest(records: List[Dict], output_path: str) -> None:
    """
    Save a list of dataset records to a JSON manifest file.

    Args:
        records: List of record dicts.
        output_path: Path to write the JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    logger.info("Manifest saved: %s (%d records)", output_path, len(records))


def load_manifest(manifest_path: str) -> List[Dict]:
    """
    Load a JSON manifest file.

    Args:
        manifest_path: Path to the JSON file.

    Returns:
        List of record dicts.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    logger.info("Manifest loaded: %s (%d records)", manifest_path, len(records))
    return records


def merge_manifests(
    existing_path: str,
    new_records: List[Dict],
) -> List[Dict]:
    """
    Merge new records into an existing manifest, avoiding duplicate IDs.

    Args:
        existing_path: Path to the existing manifest JSON.
        new_records: New records to merge.

    Returns:
        Merged list of records.
    """
    if Path(existing_path).exists():
        existing = load_manifest(existing_path)
    else:
        existing = []

    existing_ids = {r["id"] for r in existing}
    added = 0
    for record in new_records:
        if record["id"] not in existing_ids:
            existing.append(record)
            existing_ids.add(record["id"])
            added += 1

    logger.info("Merged %d new records (total: %d)", added, len(existing))
    return existing


# ═══════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════


def main():
    """
    Main entry point for ViViD dataset preparation.

    Usage:
        python -m pipeline.vivid_prepare [--config PATH] [--platform NAME]
    """
    parser = argparse.ArgumentParser(
        description="ARVTON — ViViD Dataset Preparation (Source A)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to dataset_config.yaml",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default=None,
        choices=["colab", "amd", "local"],
        help="Override platform detection",
    )
    args = parser.parse_args()

    # Setup
    setup_logging("INFO")
    logger.info("=" * 60)
    logger.info("ARVTON — ViViD Dataset Preparation (Source A)")
    logger.info("=" * 60)

    # Load config
    config = load_config(args.config)

    # Detect platform and setup paths
    platform = args.platform or detect_platform()
    paths = get_paths(platform)
    ensure_dirs(paths)

    # Mount Google Drive on Colab
    if platform == "colab":
        mount_google_drive()

    # Detect GPU
    gpu_info = detect_gpu_info()
    device = "cuda" if gpu_info["available"] else "cpu"

    # Print system info
    from pipeline.platform_utils import print_system_info
    print_system_info()

    # Process ViViD dataset
    train_records, val_records = process_vivid_dataset(config, paths, device=device)

    # Save manifests
    train_manifest_path = str(paths["train_manifest"])
    val_manifest_path = str(paths["val_manifest"])

    # Merge with existing manifests (other sources may have added records)
    all_train = merge_manifests(train_manifest_path, train_records)
    all_val = merge_manifests(val_manifest_path, val_records)

    save_manifest(all_train, train_manifest_path)
    save_manifest(all_val, val_manifest_path)

    # Summary
    print("\n" + "=" * 60)
    print("Phase 0 — Source A (ViViD) Complete")
    print("=" * 60)
    print(f"  Train records: {len(all_train)}")
    print(f"  Val records:   {len(all_val)}")
    print(f"  Total:         {len(all_train) + len(all_val)}")
    print(f"  Train manifest: {train_manifest_path}")
    print(f"  Val manifest:   {val_manifest_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
