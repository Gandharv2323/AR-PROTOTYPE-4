# Module: synthetic_gen
# License: MIT (ARVTON project)
# Description: Source C — Synthetic garment and person generation using SDXL + ControlNet-OpenPose.
# Platform: Colab (T4/A100) or AMD ROCm
# Dependencies: torch, diffusers, transformers, accelerate, Pillow, numpy, opencv-python-headless, tqdm

"""
Synthetic Data Generation (Source C)
======================================
Generates diverse synthetic garment flat-lay images using SDXL and
synthetic person images using SDXL + ControlNet-OpenPose. Applies the
same SAM 2 and HMR 2.0 auto-labeling pipeline and appends to manifests.

Usage:
    python -m pipeline.synthetic_gen --config configs/dataset_config.yaml
"""

import argparse
import itertools
import json
import logging
import os
import random
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
    get_precision_dtype,
    get_torch_device,
    load_config,
    mount_google_drive,
    setup_logging,
)

logger = logging.getLogger("arvton.synthetic_gen")


# ═══════════════════════════════════════════════════════════════════════
# GARMENT GENERATION — SDXL flat-lay product photos
# ═══════════════════════════════════════════════════════════════════════


def load_sdxl_pipeline(device: str = "cuda", dtype=None):
    """
    Load Stable Diffusion XL pipeline for image generation.

    Args:
        device: Torch device string.
        dtype: Torch dtype (auto-detected if None).

    Returns:
        Loaded SDXL pipeline.
    """
    import torch
    from diffusers import StableDiffusionXLPipeline

    if dtype is None:
        dtype = get_precision_dtype()

    logger.info("Loading SDXL pipeline (stabilityai/stable-diffusion-xl-base-1.0)...")

    try:
        # ROCm compatible: YES — Diffusers uses standard PyTorch
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if dtype == torch.float16 else None,
        )
        pipeline = pipeline.to(device)

        # Memory optimizations
        if device == "cuda":
            pipeline.enable_attention_slicing()
            pipeline.enable_vae_tiling()
            try:
                pipeline.enable_xformers_memory_efficient_attention()
                logger.info("xFormers memory-efficient attention enabled")
            except Exception:
                logger.debug("xFormers not available, using default attention")

        logger.info("SDXL pipeline loaded on %s (dtype: %s)", device, dtype)
        return pipeline

    except Exception as e:
        logger.error("Failed to load SDXL: %s", str(e))
        raise RuntimeError(f"SDXL loading failed: {e}") from e


def generate_garment_images(
    pipeline,
    config: dict,
    output_dir: str,
    target_count: int = 500,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generate synthetic garment flat-lay images using SDXL.

    Generates images using structured prompts combining colors, materials,
    and garment types. Each combination produces a unique flat-lay product photo.

    Args:
        pipeline: Loaded SDXL pipeline.
        config: Synthetic config dict with colors, materials, garment_types.
        output_dir: Directory to save generated images.
        target_count: Target number of garment images.
        seed: Random seed for reproducibility.

    Returns:
        List of metadata dicts for generated garment images.
    """
    import torch

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    colors = config.get("colors", ["black", "white", "navy", "red"])
    materials = config.get("materials", ["cotton", "linen", "denim"])
    garment_types = config.get("garment_types", ["t-shirt", "dress", "jeans"])
    steps = config.get("inference_steps", 30)
    guidance = config.get("guidance_scale", 7.5)

    # Generate all combinations
    all_combos = list(itertools.product(colors, materials, garment_types))
    random.seed(seed)
    random.shuffle(all_combos)

    # If we have more combos than target, truncate; otherwise, repeat with variation
    if len(all_combos) >= target_count:
        combos = all_combos[:target_count]
    else:
        combos = all_combos.copy()
        while len(combos) < target_count:
            extra = random.choice(all_combos)
            combos.append(extra)

    results = []
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    for i, (color, material, garment_type) in enumerate(
        tqdm(combos[:target_count], desc="Generating garments")
    ):
        garment_id = f"synth_garment_{i:05d}"
        output_file = output_path / f"{garment_id}.jpg"

        # Skip already generated
        if output_file.exists():
            results.append({
                "id": garment_id,
                "local_path": str(output_file),
                "color": color,
                "material": material,
                "garment_type": garment_type,
                "source": "synthetic",
            })
            continue

        prompt = (
            f"flat-lay photograph of a {color} {material} {garment_type}, "
            f"white background, product photography, studio lighting, "
            f"ultra sharp, no mannequin, centered, high quality, 8k"
        )

        negative_prompt = (
            "blurry, low quality, watermark, text, logo, human, mannequin, "
            "hanger, wrinkled, distorted, deformed"
        )

        try:
            with torch.no_grad():
                image = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    height=1024,
                    width=768,
                    generator=generator,
                ).images[0]

            image.save(str(output_file), "JPEG", quality=95)

            results.append({
                "id": garment_id,
                "local_path": str(output_file),
                "color": color,
                "material": material,
                "garment_type": garment_type,
                "source": "synthetic",
            })

        except Exception as e:
            vram_used = 0.0
            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated() / (1024 ** 3)
            logger.error(
                "Garment generation failed at %d (VRAM: %.2f GB): %s",
                i, vram_used, str(e),
            )
            continue

        if (i + 1) % 50 == 0:
            logger.info("Generated %d/%d garment images", i + 1, target_count)

    logger.info("Garment generation complete: %d/%d", len(results), target_count)
    return results


# ═══════════════════════════════════════════════════════════════════════
# PERSON GENERATION — SDXL + ControlNet-OpenPose
# ═══════════════════════════════════════════════════════════════════════


def load_controlnet_pipeline(device: str = "cuda", dtype=None):
    """
    Load SDXL + ControlNet-OpenPose pipeline for pose-guided person generation.

    Args:
        device: Torch device string.
        dtype: Torch dtype.

    Returns:
        ControlNet pipeline tuple (pipeline, controlnet).
    """
    import torch
    from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

    if dtype is None:
        dtype = get_precision_dtype()

    logger.info("Loading ControlNet-OpenPose pipeline...")

    try:
        # ROCm compatible: YES
        controlnet = ControlNetModel.from_pretrained(
            "thibaud/controlnet-openpose-sdxl-1.0",
            torch_dtype=dtype,
        )

        pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            torch_dtype=dtype,
            use_safetensors=True,
        )
        pipeline = pipeline.to(device)

        # Memory optimizations
        if device == "cuda":
            pipeline.enable_attention_slicing()
            pipeline.enable_vae_tiling()

        logger.info("ControlNet pipeline loaded on %s", device)
        return pipeline

    except Exception as e:
        logger.error("Failed to load ControlNet pipeline: %s", str(e))
        logger.info("Falling back to SDXL-only person generation (no pose control)")
        return None


def generate_pose_skeleton(
    body_type: str,
    image_size: Tuple[int, int] = (768, 1024),
) -> np.ndarray:
    """
    Generate a synthetic OpenPose skeleton for ControlNet conditioning.
    Creates realistic standing poses with body type variation.

    Args:
        body_type: One of 'slim', 'athletic', 'average', 'plus-size', 'tall'.
        image_size: (width, height) of the output skeleton image.

    Returns:
        RGB numpy array of the skeleton visualization.
    """
    w, h = image_size
    skeleton = np.zeros((h, w, 3), dtype=np.uint8)

    # Define body proportions based on body type
    proportions = {
        "slim": {"shoulder_w": 0.25, "hip_w": 0.20, "torso_h": 0.30},
        "athletic": {"shoulder_w": 0.30, "hip_w": 0.22, "torso_h": 0.28},
        "average": {"shoulder_w": 0.28, "hip_w": 0.24, "torso_h": 0.30},
        "plus-size": {"shoulder_w": 0.32, "hip_w": 0.30, "torso_h": 0.32},
        "tall": {"shoulder_w": 0.26, "hip_w": 0.22, "torso_h": 0.35},
    }
    prop = proportions.get(body_type, proportions["average"])

    center_x = w // 2
    head_y = int(h * 0.08)
    neck_y = int(h * 0.15)
    shoulder_y = int(h * 0.18)
    hip_y = shoulder_y + int(h * prop["torso_h"])
    knee_y = int(h * 0.70)
    ankle_y = int(h * 0.92)
    elbow_y = shoulder_y + int((hip_y - shoulder_y) * 0.55)
    wrist_y = hip_y + int((hip_y - shoulder_y) * 0.15)

    shoulder_offset = int(w * prop["shoulder_w"] / 2)
    hip_offset = int(w * prop["hip_w"] / 2)
    knee_offset = int(w * 0.08)

    # Keypoints: [x, y] format
    keypoints = {
        "nose": (center_x, head_y),
        "neck": (center_x, neck_y),
        "r_shoulder": (center_x - shoulder_offset, shoulder_y),
        "l_shoulder": (center_x + shoulder_offset, shoulder_y),
        "r_elbow": (center_x - shoulder_offset - 10, elbow_y),
        "l_elbow": (center_x + shoulder_offset + 10, elbow_y),
        "r_wrist": (center_x - shoulder_offset - 5, wrist_y),
        "l_wrist": (center_x + shoulder_offset + 5, wrist_y),
        "r_hip": (center_x - hip_offset, hip_y),
        "l_hip": (center_x + hip_offset, hip_y),
        "r_knee": (center_x - knee_offset, knee_y),
        "l_knee": (center_x + knee_offset, knee_y),
        "r_ankle": (center_x - knee_offset, ankle_y),
        "l_ankle": (center_x + knee_offset, ankle_y),
    }

    # Draw skeleton connections
    connections = [
        ("nose", "neck"), ("neck", "r_shoulder"), ("neck", "l_shoulder"),
        ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
        ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
        ("neck", "r_hip"), ("neck", "l_hip"),
        ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
        ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
    ]

    # Limb colors (OpenPose convention)
    colors = [
        (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
        (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
        (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
        (0, 0, 255),
    ]

    for idx, (p1_name, p2_name) in enumerate(connections):
        p1 = keypoints[p1_name]
        p2 = keypoints[p2_name]
        color = colors[idx % len(colors)]
        cv2.line(skeleton, p1, p2, color, thickness=3)

    # Draw keypoints
    for name, point in keypoints.items():
        cv2.circle(skeleton, point, 5, (255, 255, 255), -1)

    return skeleton


def generate_person_images(
    pipeline,
    controlnet_pipeline,
    config: dict,
    output_dir: str,
    target_count: int = 500,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generate synthetic person images using SDXL (optionally with ControlNet-OpenPose).

    Args:
        pipeline: Base SDXL pipeline (used if ControlNet unavailable).
        controlnet_pipeline: ControlNet pipeline (may be None).
        config: Synthetic config dict with body_types.
        output_dir: Directory to save generated images.
        target_count: Target number of person images.
        seed: Random seed.

    Returns:
        List of metadata dicts for generated person images.
    """
    import torch

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    body_types = config.get("body_types", ["slim", "athletic", "average", "plus-size", "tall"])
    steps = config.get("inference_steps", 30)
    guidance = config.get("guidance_scale", 7.5)

    # Ethnicity and appearance descriptors for diversity
    appearances = [
        "East Asian",
        "South Asian",
        "Black African",
        "White European",
        "Hispanic Latino",
        "Middle Eastern",
        "Southeast Asian",
        "Pacific Islander",
        "Mixed race",
        "Indigenous",
    ]

    # Generate combinations
    combos = list(itertools.product(body_types, appearances))
    random.seed(seed)
    random.shuffle(combos)

    # Expand to target count
    while len(combos) < target_count:
        combos.extend(combos[:target_count - len(combos)])
    combos = combos[:target_count]

    active_pipeline = controlnet_pipeline if controlnet_pipeline is not None else pipeline
    generator = torch.Generator(device=active_pipeline.device).manual_seed(seed)

    results = []
    for i, (body_type, ethnicity) in enumerate(
        tqdm(combos, desc="Generating persons")
    ):
        person_id = f"synth_person_{i:05d}"
        output_file = output_path / f"{person_id}.jpg"

        # Skip already generated
        if output_file.exists():
            results.append({
                "id": person_id,
                "local_path": str(output_file),
                "body_type": body_type,
                "ethnicity": ethnicity,
                "source": "synthetic",
            })
            continue

        prompt = (
            f"full body photo of a person standing, {body_type} body, "
            f"{ethnicity}, neutral background, fashion photography, "
            f"sharp focus, natural lighting, professional studio, 8k, "
            f"centered, looking at camera"
        )

        negative_prompt = (
            "blurry, low quality, watermark, text, logo, nsfw, nude, "
            "distorted, deformed, extra limbs, cropped, partial body"
        )

        try:
            if controlnet_pipeline is not None:
                # Generate pose skeleton
                pose_skeleton = generate_pose_skeleton(body_type)
                pose_image = Image.fromarray(pose_skeleton)

                with torch.no_grad():
                    image = controlnet_pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=pose_image,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        height=1024,
                        width=768,
                        controlnet_conditioning_scale=0.5,
                        generator=generator,
                    ).images[0]
            else:
                with torch.no_grad():
                    image = pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        height=1024,
                        width=768,
                        generator=generator,
                    ).images[0]

            image.save(str(output_file), "JPEG", quality=95)

            results.append({
                "id": person_id,
                "local_path": str(output_file),
                "body_type": body_type,
                "ethnicity": ethnicity,
                "source": "synthetic",
            })

        except Exception as e:
            vram_used = 0.0
            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated() / (1024 ** 3)
            logger.error(
                "Person generation failed at %d (VRAM: %.2f GB): %s",
                i, vram_used, str(e),
            )
            continue

        if (i + 1) % 50 == 0:
            logger.info("Generated %d/%d person images", i + 1, target_count)

    logger.info("Person generation complete: %d/%d", len(results), target_count)
    return results


# ═══════════════════════════════════════════════════════════════════════
# FULL PIPELINE — Orchestrate synthetic generation and labeling
# ═══════════════════════════════════════════════════════════════════════


def process_synthetic_pipeline(
    config: dict,
    paths: Dict[str, Path],
    device: str = "cuda",
) -> List[Dict]:
    """
    Full synthetic generation pipeline: generate → label → manifest.

    Args:
        config: Loaded configuration dict.
        paths: Resolved paths dict.
        device: Torch device string.

    Returns:
        List of manifest records for synthetic source.
    """
    synth_config = config.get("synthetic", {})
    synth_raw = paths["synthetic_raw"]
    synth_processed = paths["synthetic_processed"]

    garment_dir = synth_raw / "garments"
    person_dir = synth_raw / "persons"
    mask_dir = synth_processed / "person_mask"
    body_parse_dir = synth_processed / "body_parse"
    densepose_dir = synth_processed / "densepose"
    garment_mask_dir = synth_processed / "garment_mask"
    for d in [garment_dir, person_dir, mask_dir, body_parse_dir,
              densepose_dir, garment_mask_dir]:
        d.mkdir(parents=True, exist_ok=True)

    target_garments = synth_config.get("target_garments", 500)
    target_persons = synth_config.get("target_persons", 500)

    # Step 1: Load SDXL pipeline
    logger.info("=" * 60)
    logger.info("Step 1/4: Loading SDXL pipeline")
    logger.info("=" * 60)
    sdxl_pipeline = load_sdxl_pipeline(device=device)

    # Step 2: Generate garment images
    logger.info("=" * 60)
    logger.info("Step 2/4: Generating %d synthetic garment images", target_garments)
    logger.info("=" * 60)
    garment_results = generate_garment_images(
        sdxl_pipeline, synth_config, str(garment_dir),
        target_count=target_garments,
    )

    # Step 3: Generate person images (with ControlNet if available)
    logger.info("=" * 60)
    logger.info("Step 3/4: Generating %d synthetic person images", target_persons)
    logger.info("=" * 60)

    import torch
    # Free SDXL pipeline before loading ControlNet to save VRAM
    controlnet_pipeline = None
    try:
        del sdxl_pipeline
        torch.cuda.empty_cache()
        controlnet_pipeline = load_controlnet_pipeline(device=device)
    except Exception:
        logger.info("ControlNet not available. Using SDXL-only for person generation.")

    # Reload SDXL if ControlNet failed
    if controlnet_pipeline is None:
        sdxl_pipeline = load_sdxl_pipeline(device=device)
    else:
        sdxl_pipeline = None

    person_results = generate_person_images(
        sdxl_pipeline, controlnet_pipeline, synth_config, str(person_dir),
        target_count=target_persons,
    )

    # Cleanup pipelines
    del sdxl_pipeline, controlnet_pipeline
    torch.cuda.empty_cache()

    # Step 4: Label with SAM 2 + HMR 2.0 and create records
    logger.info("=" * 60)
    logger.info("Step 4/4: Running SAM 2 + HMR 2.0 auto-labeling")
    logger.info("=" * 60)

    from pipeline.vivid_prepare import (
        load_sam2_model,
        generate_person_mask,
        load_hmr2_model,
        estimate_body_params,
        generate_body_parse,
        generate_densepose_map,
    )

    sam2_ckpt_dir = str(paths["sam2_checkpoint"])
    try:
        mask_gen = load_sam2_model(sam2_ckpt_dir, device=device)
    except Exception:
        mask_gen = None

    hmr2_model, hmr2_cfg = load_hmr2_model(device=device)

    # Generate random person-garment pairs
    np.random.seed(42)
    num_pairs = min(len(person_results), len(garment_results))
    if num_pairs == 0:
        logger.warning("No person/garment images generated. Cannot create pairs.")
        return []

    garment_indices = np.random.choice(len(garment_results), size=num_pairs, replace=True)

    records = []
    for i in tqdm(range(num_pairs), desc="Labeling synthetic pairs"):
        person_meta = person_results[i]
        garment_meta = garment_results[garment_indices[i]]
        record_id = f"synth_{i:05d}"

        person_path = person_meta["local_path"]
        garment_path = garment_meta["local_path"]

        try:
            # Load person image
            person_img = cv2.imread(person_path)
            if person_img is None:
                continue
            person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)

            # Generate person mask
            mask_path = str(mask_dir / f"{record_id}_mask.png")
            if not Path(mask_path).exists():
                if mask_gen is not None:
                    mask = generate_person_mask(mask_gen, person_rgb)
                else:
                    mask = np.ones(person_img.shape[:2], dtype=np.uint8) * 255
                if mask is None:
                    mask = np.ones(person_img.shape[:2], dtype=np.uint8) * 255
                cv2.imwrite(mask_path, mask)

            # Estimate body params
            body_params = estimate_body_params(hmr2_model, hmr2_cfg, person_rgb, device=device)
            if body_params is None:
                body_params = {
                    "theta": np.zeros(72, dtype=np.float32),
                    "beta": np.zeros(10, dtype=np.float32),
                    "camera": np.array([1.0, 0.0, 0.0], dtype=np.float32),
                    "confidence": 0.0,
                    "bbox": [0, 0, person_img.shape[1], person_img.shape[0]],
                }

            # Generate body parse
            mask_loaded = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            body_parse = generate_body_parse(mask_loaded, body_params, person_img.shape)
            bp_path = str(body_parse_dir / f"{record_id}_body_parse.png")
            cv2.imwrite(bp_path, body_parse)

            # Generate densepose
            densepose = generate_densepose_map(person_rgb, body_params)
            dp_path = str(densepose_dir / f"{record_id}_densepose.png")
            cv2.imwrite(dp_path, densepose)

            # Generate garment mask (white background removal)
            garment_img = cv2.imread(garment_path)
            gm_path = str(garment_mask_dir / f"{record_id}_garment_mask.png")
            if garment_img is not None and not Path(gm_path).exists():
                garment_gray = cv2.cvtColor(garment_img, cv2.COLOR_BGR2GRAY)
                _, garment_mask = cv2.threshold(garment_gray, 240, 255, cv2.THRESH_BINARY_INV)
                cv2.imwrite(gm_path, garment_mask)

            # Determine category from garment type
            garment_type = garment_meta.get("garment_type", "t-shirt").lower()
            if garment_type in ["dress"]:
                category = "dress"
            elif garment_type in ["trousers", "jeans", "skirt"]:
                category = "lower"
            else:
                category = "upper"

            record = {
                "id": record_id,
                "person_image": person_path,
                "person_mask": mask_path,
                "body_parse": bp_path,
                "densepose": dp_path,
                "garment_image": garment_path,
                "garment_mask": gm_path,
                "category": category,
                "license": "MIT",
                "source": "synthetic",
                "body_type": person_meta.get("body_type", "average"),
                "garment_type": garment_type,
            }
            records.append(record)

        except Exception as e:
            logger.warning("Failed to label pair %s: %s", record_id, str(e))

        if (i + 1) % 100 == 0:
            logger.info("Labeled %d/%d synthetic pairs", i + 1, num_pairs)

    logger.info("Synthetic pipeline complete: %d records", len(records))
    return records


# ═══════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════


def main():
    """
    Main entry point for synthetic data generation.

    Usage:
        python -m pipeline.synthetic_gen [--config PATH] [--platform NAME]
    """
    parser = argparse.ArgumentParser(
        description="ARVTON — Synthetic Data Generation (Source C)",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to dataset_config.yaml")
    parser.add_argument("--platform", type=str, default=None, choices=["colab", "amd", "local"])
    parser.add_argument("--garments-only", action="store_true", help="Generate only garment images")
    parser.add_argument("--persons-only", action="store_true", help="Generate only person images")
    args = parser.parse_args()

    # Setup
    setup_logging("INFO")
    logger.info("=" * 60)
    logger.info("ARVTON — Synthetic Data Generation (Source C)")
    logger.info("=" * 60)

    config = load_config(args.config)
    platform = args.platform or detect_platform()
    paths = get_paths(platform)
    ensure_dirs(paths)

    if platform == "colab":
        mount_google_drive()

    gpu_info = detect_gpu_info()
    device = "cuda" if gpu_info["available"] else "cpu"

    from pipeline.platform_utils import print_system_info
    print_system_info()

    # Run pipeline
    records = process_synthetic_pipeline(config, paths, device=device)

    # Merge with existing manifests
    from pipeline.vivid_prepare import merge_manifests, save_manifest

    train_path = str(paths["train_manifest"])
    all_train = merge_manifests(train_path, records)
    save_manifest(all_train, train_path)

    # Summary
    print("\n" + "=" * 60)
    print("Phase 0 — Source C (Synthetic) Complete")
    print("=" * 60)
    print(f"  New synthetic records: {len(records)}")
    print(f"  Total train records: {len(all_train)}")
    print(f"  Train manifest: {train_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
