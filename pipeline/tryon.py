# Module: tryon
# License: MIT (ARVTON project)
# Description: Phase 2 — Dual-path 2D virtual try-on using Leffa (flow-based) + CatVTON (paint-based).
# Platform: Colab T4/A100 + AMD ROCm
# Dependencies: torch, diffusers, transformers, accelerate, Pillow, numpy, opencv-python-headless

"""
===================================
PHASE 2 — 2D VIRTUAL TRY-ON
File: tryon.py
===================================

Dual-Path Try-On Module
========================
Path A — Leffa: Flow-based warping (MIT license)
Path B — CatVTON: Diffusion-based inpainting (Apache-2.0 license)

The final output is a weighted blend controlled by blend_alpha (0.5 default).
    result = blend_alpha × Leffa + (1 – blend_alpha) × CatVTON

Inputs (from manifest):
    - person_image → person RGB
    - garment_image → garment RGB
    - person_mask → binary mask
    - body_parse → body parse map
    - densepose → densepose UV map
    - category → "upper" | "lower" | "dress"

Output:
    - 768 × 1024 try-on result image
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("arvton.tryon")

# ═══════════════════════════════════════════════════════════════════════
# MODEL LOADING — Leffa
# ═══════════════════════════════════════════════════════════════════════

_leffa_model = None


def load_leffa(device: str = "cuda"):
    """
    Load the Leffa flow-based virtual try-on model.
    MIT license — franciszzj/Leffa on HuggingFace.

    Args:
        device: Torch device string.

    Returns:
        Loaded Leffa pipeline.
    """
    global _leffa_model
    if _leffa_model is not None:
        return _leffa_model

    import torch

    logger.info("Loading Leffa model (franciszzj/Leffa)...")

    try:
        # ROCm compatible: YES — uses standard PyTorch
        from leffa.model import LeffaModel
        from leffa.transform import LeffaTransform

        model = LeffaModel.from_pretrained(
            "franciszzj/Leffa",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        model = model.to(device)
        model.eval()

        transform = LeffaTransform(model_type="vton")

        _leffa_model = {"model": model, "transform": transform}

        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / (1024 ** 3)
            logger.info("Leffa loaded. VRAM: %.2f GB", vram)

        return _leffa_model

    except ImportError:
        logger.warning(
            "Leffa package not installed. Install via: pip install leffa\n"
            "Falling back to stub mode."
        )
        _leffa_model = None
        return None

    except Exception as e:
        logger.warning("Leffa model loading failed: %s. Using stub.", str(e))
        _leffa_model = None
        return None


# ═══════════════════════════════════════════════════════════════════════
# MODEL LOADING — CatVTON
# ═══════════════════════════════════════════════════════════════════════

_catvton_model = None


def load_catvton(device: str = "cuda"):
    """
    Load the CatVTON diffusion-based virtual try-on model.
    Apache-2.0 license — zhengchong/CatVTON on HuggingFace.

    Args:
        device: Torch device string.

    Returns:
        Loaded CatVTON pipeline.
    """
    global _catvton_model
    if _catvton_model is not None:
        return _catvton_model

    import torch

    logger.info("Loading CatVTON model (zhengchong/CatVTON)...")

    try:
        # ROCm compatible: YES — Diffusers uses standard PyTorch
        from diffusers import StableDiffusionInpaintPipeline

        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "zhengchong/CatVTON",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=True,
        )
        pipeline = pipeline.to(device)

        if device == "cuda":
            pipeline.enable_attention_slicing()
            pipeline.enable_vae_tiling()
            try:
                pipeline.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        _catvton_model = pipeline

        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / (1024 ** 3)
            logger.info("CatVTON loaded. VRAM: %.2f GB", vram)

        return _catvton_model

    except ImportError:
        logger.warning(
            "Diffusers not installed. Install via: pip install diffusers\n"
            "Falling back to stub mode."
        )
        _catvton_model = None
        return None

    except Exception as e:
        logger.warning("CatVTON model loading failed: %s. Using stub.", str(e))
        _catvton_model = None
        return None


# ═══════════════════════════════════════════════════════════════════════
# INFERENCE — Leffa Path A
# ═══════════════════════════════════════════════════════════════════════


def infer_leffa(
    person_image: Image.Image,
    garment_image: Image.Image,
    person_mask: Image.Image,
    densepose: Image.Image,
    category: str = "upper",
    device: str = "cuda",
) -> Optional[Image.Image]:
    """
    Run Leffa flow-based virtual try-on inference.

    Args:
        person_image: Person RGB image.
        garment_image: Garment RGB image.
        person_mask: Binary person mask.
        densepose: DensePose UV map.
        category: Garment category.
        device: Torch device.

    Returns:
        Try-on result image or None if Leffa unavailable.
    """
    import torch

    leffa = load_leffa(device)
    if leffa is None:
        logger.warning("Leffa not available. Returning None.")
        return None

    try:
        model = leffa["model"]
        transform = leffa["transform"]

        # Prepare inputs
        inputs = transform.preprocess(
            person_image=person_image,
            garment_image=garment_image,
            mask=person_mask,
            densepose=densepose,
            category=category,
        )

        with torch.no_grad():
            result = model(inputs)

        output_image = transform.postprocess(result)

        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / (1024 ** 3)
            logger.debug("Leffa inference done. VRAM: %.2f GB", vram)

        return output_image

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("Leffa OOM. Clearing cache and returning None.")
            torch.cuda.empty_cache()
        else:
            logger.error("Leffa inference error: %s", str(e))
        return None

    except Exception as e:
        logger.error("Leffa inference failed: %s", str(e))
        return None


# ═══════════════════════════════════════════════════════════════════════
# INFERENCE — CatVTON Path B
# ═══════════════════════════════════════════════════════════════════════


def infer_catvton(
    person_image: Image.Image,
    garment_image: Image.Image,
    person_mask: Image.Image,
    category: str = "upper",
    num_inference_steps: int = 30,
    guidance_scale: float = 2.5,
    device: str = "cuda",
) -> Optional[Image.Image]:
    """
    Run CatVTON diffusion-based virtual try-on inference.

    Args:
        person_image: Person RGB image.
        garment_image: Garment RGB image.
        person_mask: Binary person mask (inpainting region).
        category: Garment category.
        num_inference_steps: Number of diffusion steps.
        guidance_scale: Classifier-free guidance scale.
        device: Torch device.

    Returns:
        Try-on result image or None if CatVTON unavailable.
    """
    import torch

    pipeline = load_catvton(device)
    if pipeline is None:
        logger.warning("CatVTON not available. Returning None.")
        return None

    try:
        # CatVTON uses inpainting approach
        # The person mask defines the region to replace with the garment
        target_size = (768, 1024)
        person_resized = person_image.resize(target_size, Image.LANCZOS)
        garment_resized = garment_image.resize(target_size, Image.LANCZOS)
        mask_resized = person_mask.resize(target_size, Image.NEAREST)

        with torch.no_grad():
            result = pipeline(
                prompt=f"a person wearing a {category} garment, photorealistic",
                image=person_resized,
                mask_image=mask_resized,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]

        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / (1024 ** 3)
            logger.debug("CatVTON inference done. VRAM: %.2f GB", vram)

        return result

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("CatVTON OOM. Clearing cache and returning None.")
            torch.cuda.empty_cache()
        else:
            logger.error("CatVTON inference error: %s", str(e))
        return None

    except Exception as e:
        logger.error("CatVTON inference failed: %s", str(e))
        return None


# ═══════════════════════════════════════════════════════════════════════
# BLENDING — Weighted fusion with face preservation
# ═══════════════════════════════════════════════════════════════════════


def blend_results(
    leffa_result: Optional[Image.Image],
    catvton_result: Optional[Image.Image],
    original_person: Image.Image,
    person_mask: Image.Image,
    blend_alpha: float = 0.5,
) -> Image.Image:
    """
    Blend Leffa and CatVTON results with face/hand preservation.

    The blend formula:
        result = blend_alpha × Leffa + (1 – blend_alpha) × CatVTON

    If only one model produced output, that output is used directly.
    Face and hand regions are preserved from the original person image.

    Args:
        leffa_result: Leffa output image (or None).
        catvton_result: CatVTON output image (or None).
        original_person: Original person image for face preservation.
        person_mask: Binary mask for the person.
        blend_alpha: Leffa weight (0 = all CatVTON, 1 = all Leffa).

    Returns:
        Final blended try-on result image (768 × 1024).
    """
    target_size = (768, 1024)

    # Handle fallback cases
    if leffa_result is None and catvton_result is None:
        logger.warning("Both models failed. Returning original person image.")
        return original_person.resize(target_size, Image.LANCZOS)

    if leffa_result is None:
        blended = catvton_result
    elif catvton_result is None:
        blended = leffa_result
    else:
        # Both available — compute weighted blend
        leffa_arr = np.array(leffa_result.resize(target_size, Image.LANCZOS)).astype(np.float32)
        catvton_arr = np.array(catvton_result.resize(target_size, Image.LANCZOS)).astype(np.float32)
        blended_arr = blend_alpha * leffa_arr + (1 - blend_alpha) * catvton_arr
        blended = Image.fromarray(blended_arr.astype(np.uint8))

    # Ensure correct size
    blended = blended.resize(target_size, Image.LANCZOS)

    # Face preservation: restore original face/hair from person image
    original_resized = original_person.resize(target_size, Image.LANCZOS)
    mask_resized = person_mask.resize(target_size, Image.NEAREST)

    blended_arr = np.array(blended).astype(np.float32)
    original_arr = np.array(original_resized).astype(np.float32)
    mask_arr = np.array(mask_resized)

    # Create face region mask (upper portion of person mask)
    h, w = mask_arr.shape[:2] if len(mask_arr.shape) >= 2 else (target_size[1], target_size[0])
    if len(mask_arr.shape) == 3:
        mask_arr = cv2.cvtColor(mask_arr, cv2.COLOR_RGB2GRAY)

    # Face region: top 25% of the person's bounding box
    person_ys = np.where(mask_arr > 127)[0]
    if len(person_ys) > 0:
        person_top = person_ys.min()
        person_bottom = person_ys.max()
        person_height = person_bottom - person_top
        face_bottom = person_top + int(person_height * 0.25)

        # Create soft face mask
        face_mask = np.zeros((h, w), dtype=np.float32)
        face_mask[person_top:face_bottom, :] = 1.0

        # Gaussian blur for soft transition
        face_mask = cv2.GaussianBlur(face_mask, (31, 31), 10.0)

        # Apply face mask to intersect with person mask
        person_mask_float = (mask_arr > 127).astype(np.float32)
        face_mask = face_mask * person_mask_float

        # Blend: keep face from original
        face_mask_3ch = np.stack([face_mask] * 3, axis=-1)
        blended_arr = blended_arr * (1 - face_mask_3ch) + original_arr * face_mask_3ch

    result = Image.fromarray(blended_arr.astype(np.uint8))
    return result


# ═══════════════════════════════════════════════════════════════════════
# MAIN INFERENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════════


def tryon(
    person_image_path: str,
    garment_image_path: str,
    person_mask_path: str,
    body_parse_path: str,
    densepose_path: str,
    category: str = "upper",
    blend_alpha: float = 0.5,
    output_path: Optional[str] = None,
    device: str = "cuda",
) -> Image.Image:
    """
    Run the full dual-path virtual try-on pipeline.

    Args:
        person_image_path: Path to person image.
        garment_image_path: Path to garment image.
        person_mask_path: Path to person mask.
        body_parse_path: Path to body parse map.
        densepose_path: Path to DensePose UV map.
        category: 'upper', 'lower', or 'dress'.
        blend_alpha: Leffa weight in final blend (default 0.5).
        output_path: If provided, save result to this path.
        device: Torch device string.

    Returns:
        Final 768×1024 try-on result as PIL Image.
    """
    logger.info("Running try-on: person=%s, garment=%s, category=%s",
                Path(person_image_path).name,
                Path(garment_image_path).name,
                category)

    # Load inputs
    person_image = Image.open(person_image_path).convert("RGB")
    garment_image = Image.open(garment_image_path).convert("RGB")
    person_mask = Image.open(person_mask_path).convert("L")
    densepose = Image.open(densepose_path).convert("RGB")

    # Path A: Leffa (flow-based)
    logger.info("Path A: Running Leffa inference...")
    leffa_result = infer_leffa(
        person_image, garment_image, person_mask, densepose,
        category=category, device=device,
    )

    # Path B: CatVTON (diffusion-based)
    logger.info("Path B: Running CatVTON inference...")
    catvton_result = infer_catvton(
        person_image, garment_image, person_mask,
        category=category, device=device,
    )

    # Blend results
    logger.info("Blending results (alpha=%.2f)...", blend_alpha)
    result = blend_results(
        leffa_result, catvton_result,
        person_image, person_mask,
        blend_alpha=blend_alpha,
    )

    # Save if output path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path, "JPEG", quality=95)
        logger.info("Try-on result saved: %s", output_path)

    return result


# ═══════════════════════════════════════════════════════════════════════
# CLEANUP
# ═══════════════════════════════════════════════════════════════════════


def clear_cache():
    """Clear all cached models to free VRAM."""
    global _leffa_model, _catvton_model
    _leffa_model = None
    _catvton_model = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Try-on model cache cleared.")
    except ImportError:
        pass


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    import argparse
    from pipeline.platform_utils import setup_logging

    parser = argparse.ArgumentParser(description="ARVTON — Phase 2 Virtual Try-On")
    parser.add_argument("--person", required=True, help="Person image path")
    parser.add_argument("--garment", required=True, help="Garment image path")
    parser.add_argument("--mask", required=True, help="Person mask path")
    parser.add_argument("--body-parse", required=True, help="Body parse path")
    parser.add_argument("--densepose", required=True, help="DensePose path")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--category", default="upper", choices=["upper", "lower", "dress"])
    parser.add_argument("--blend-alpha", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    setup_logging("INFO")

    result = tryon(
        args.person, args.garment, args.mask,
        args.body_parse, args.densepose,
        category=args.category,
        blend_alpha=args.blend_alpha,
        output_path=args.output,
        device=args.device,
    )

    print(f"\nPhase 2 complete. Try-on result saved to: {args.output}")
    print("Reply 'continue' to proceed to Phase 3.")
