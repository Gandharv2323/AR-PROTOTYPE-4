# Module: cc0_scraper
# License: MIT (ARVTON project)
# Description: Source B — CC0 image scraper from Pexels and Pixabay APIs with CLIP classification.
# Platform: Colab (T4/A100) or Local with GPU
# Dependencies: torch, open-clip-torch, requests, Pillow, numpy, opencv-python-headless, tqdm

"""
CC0 Web Scraper (Source B)
===========================
Scrapes fashion/garment images from Pexels (CC0) and Pixabay (CC0) APIs,
classifies them as person or garment using CLIP, generates masks and body
estimates, creates synthetic try-on ground truth, and appends to manifests.

Usage:
    python -m pipeline.cc0_scraper --config configs/dataset_config.yaml \\
           --pexels-key YOUR_KEY --pixabay-key YOUR_KEY
"""

import argparse
import hashlib
import json
import logging
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
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

logger = logging.getLogger("arvton.cc0_scraper")

# ═══════════════════════════════════════════════════════════════════════
# API CLIENTS — Pexels and Pixabay
# ═══════════════════════════════════════════════════════════════════════


class PexelsClient:
    """
    Client for the Pexels API (CC0 licensed images).
    API docs: https://www.pexels.com/api/documentation/
    """

    BASE_URL = "https://api.pexels.com/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"Authorization": api_key})

    def search(
        self,
        query: str,
        per_page: int = 80,
        page: int = 1,
        orientation: str = "portrait",
    ) -> List[Dict[str, Any]]:
        """
        Search for images on Pexels.

        Args:
            query: Search term.
            per_page: Results per page (max 80).
            page: Page number.
            orientation: 'landscape', 'portrait', or 'square'.

        Returns:
            List of image metadata dicts with download URLs.
        """
        params = {
            "query": query,
            "per_page": min(per_page, 80),
            "page": page,
            "orientation": orientation,
        }

        try:
            response = self.session.get(
                f"{self.BASE_URL}/search",
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for photo in data.get("photos", []):
                results.append({
                    "id": f"pexels_{photo['id']}",
                    "url": photo["src"]["large2x"],  # High-res version
                    "url_medium": photo["src"]["medium"],
                    "width": photo["width"],
                    "height": photo["height"],
                    "photographer": photo["photographer"],
                    "source": "pexels",
                    "license": "CC0",
                    "query": query,
                })

            logger.debug("Pexels: '%s' page %d returned %d results", query, page, len(results))
            return results

        except requests.RequestException as e:
            logger.error("Pexels API error for query '%s': %s", query, str(e))
            return []

    def search_all_pages(
        self,
        query: str,
        max_pages: int = 10,
        per_page: int = 80,
    ) -> List[Dict[str, Any]]:
        """
        Search across multiple pages.

        Args:
            query: Search term.
            max_pages: Maximum pages to fetch.
            per_page: Results per page.

        Returns:
            Aggregated list of image metadata dicts.
        """
        all_results = []
        for page in range(1, max_pages + 1):
            results = self.search(query, per_page=per_page, page=page)
            if not results:
                break
            all_results.extend(results)
            time.sleep(0.5)  # Rate limiting courtesy

        logger.info("Pexels total for '%s': %d images", query, len(all_results))
        return all_results


class PixabayClient:
    """
    Client for the Pixabay API (CC0 licensed images).
    API docs: https://pixabay.com/api/docs/
    """

    BASE_URL = "https://pixabay.com/api/"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()

    def search(
        self,
        query: str,
        per_page: int = 100,
        page: int = 1,
        image_type: str = "photo",
    ) -> List[Dict[str, Any]]:
        """
        Search for images on Pixabay.

        Args:
            query: Search term.
            per_page: Results per page (max 200).
            page: Page number.
            image_type: 'photo', 'illustration', or 'vector'.

        Returns:
            List of image metadata dicts with download URLs.
        """
        params = {
            "key": self.api_key,
            "q": query,
            "per_page": min(per_page, 200),
            "page": page,
            "image_type": image_type,
            "safesearch": "true",
            "min_width": 512,
            "min_height": 512,
        }

        try:
            response = self.session.get(
                self.BASE_URL,
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for hit in data.get("hits", []):
                results.append({
                    "id": f"pixabay_{hit['id']}",
                    "url": hit["largeImageURL"],
                    "url_medium": hit["webformatURL"],
                    "width": hit["imageWidth"],
                    "height": hit["imageHeight"],
                    "photographer": hit.get("user", "unknown"),
                    "source": "pixabay",
                    "license": "CC0",
                    "query": query,
                })

            logger.debug("Pixabay: '%s' page %d returned %d results", query, page, len(results))
            return results

        except requests.RequestException as e:
            logger.error("Pixabay API error for query '%s': %s", query, str(e))
            return []

    def search_all_pages(
        self,
        query: str,
        max_pages: int = 10,
        per_page: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Search across multiple pages.

        Args:
            query: Search term.
            max_pages: Maximum pages to fetch.
            per_page: Results per page.

        Returns:
            Aggregated list of image metadata dicts.
        """
        all_results = []
        for page in range(1, max_pages + 1):
            results = self.search(query, per_page=per_page, page=page)
            if not results:
                break
            all_results.extend(results)
            time.sleep(0.5)  # Rate limiting courtesy

        logger.info("Pixabay total for '%s': %d images", query, len(all_results))
        return all_results


# ═══════════════════════════════════════════════════════════════════════
# IMAGE DOWNLOAD — Download and deduplicate
# ═══════════════════════════════════════════════════════════════════════


def download_image(
    url: str,
    save_path: str,
    timeout: int = 30,
) -> bool:
    """
    Download an image from URL and save to disk.

    Args:
        url: Image URL.
        save_path: Local path to save.
        timeout: Request timeout in seconds.

    Returns:
        bool: True if download succeeded.
    """
    if Path(save_path).exists():
        return True

    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "image" not in content_type:
            logger.warning("Non-image content type '%s' for URL: %s", content_type, url)
            return False

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return True

    except requests.RequestException as e:
        logger.warning("Download failed for %s: %s", url, str(e))
        return False


def download_batch(
    image_metadata: List[Dict[str, Any]],
    output_dir: str,
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    """
    Download a batch of images and add local paths to metadata.

    Args:
        image_metadata: List of image metadata dicts (must contain 'id' and 'url').
        output_dir: Directory to save downloaded images.
        max_workers: Number of max concurrent downloads (sequential for courtesy).

    Returns:
        Updated metadata with 'local_path' field added.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []
    for i, meta in enumerate(tqdm(image_metadata, desc="Downloading images")):
        # Generate safe filename from ID
        ext = ".jpg"
        filename = f"{meta['id']}{ext}"
        local_path = str(output_path / filename)

        if download_image(meta["url"], local_path):
            meta["local_path"] = local_path
            results.append(meta)
        else:
            logger.warning("Skipping failed download: %s", meta["id"])

        # Rate limiting
        if (i + 1) % 50 == 0:
            time.sleep(1.0)

    logger.info("Downloaded %d/%d images", len(results), len(image_metadata))
    return results


# ═══════════════════════════════════════════════════════════════════════
# CLIP CLASSIFICATION — Person vs Garment
# ═══════════════════════════════════════════════════════════════════════


def load_clip_model(device: str = "cuda"):
    """
    Load the CLIP model for image classification.
    Uses open_clip (MIT license) with ViT-B/32 architecture.

    Args:
        device: Torch device string.

    Returns:
        Tuple of (model, preprocess, tokenizer).
    """
    try:
        # ROCm compatible: YES — OpenCLIP uses standard PyTorch
        import open_clip
        import torch

        logger.info("Loading CLIP model (ViT-B/32)...")

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai",
        )
        model = model.to(device)
        model.eval()

        tokenizer = open_clip.get_tokenizer("ViT-B-32")

        logger.info("CLIP model loaded on %s", device)
        return model, preprocess, tokenizer

    except Exception as e:
        logger.error("Failed to load CLIP model: %s", str(e))
        raise RuntimeError(f"CLIP loading failed: {e}") from e


def classify_image_clip(
    model,
    preprocess,
    tokenizer,
    image_path: str,
    device: str = "cuda",
    confidence_threshold: float = 0.75,
) -> Optional[Tuple[str, float]]:
    """
    Classify an image as 'person' or 'garment' using CLIP zero-shot classification.

    Args:
        model: CLIP model.
        preprocess: CLIP preprocessing transform.
        tokenizer: CLIP tokenizer.
        image_path: Path to the image.
        device: Torch device string.
        confidence_threshold: Minimum confidence to accept classification.

    Returns:
        Tuple of (class_label, confidence) or None if below threshold.
    """
    import torch

    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        # Define classification prompts
        text_prompts = [
            "a person wearing clothes, full body photo",
            "a flat-lay garment product photo, white background",
        ]
        text_tokens = tokenizer(text_prompts).to(device)

        # Compute similarities
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text_tokens)

            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Compute cosine similarity
            similarity = (image_features @ text_features.T).softmax(dim=-1)
            probs = similarity[0].cpu().numpy()

        person_prob = float(probs[0])
        garment_prob = float(probs[1])

        if person_prob > garment_prob:
            class_label = "person"
            confidence = person_prob
        else:
            class_label = "garment"
            confidence = garment_prob

        if confidence < confidence_threshold:
            logger.debug(
                "Image %s classified as '%s' (%.2f) but below threshold %.2f",
                Path(image_path).name,
                class_label,
                confidence,
                confidence_threshold,
            )
            return None

        return (class_label, confidence)

    except Exception as e:
        logger.warning("CLIP classification failed for %s: %s", image_path, str(e))
        return None


def classify_batch(
    model,
    preprocess,
    tokenizer,
    image_metadata: List[Dict[str, Any]],
    device: str = "cuda",
    confidence_threshold: float = 0.75,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Classify a batch of images into person and garment groups.

    Args:
        model: CLIP model.
        preprocess: CLIP preprocess function.
        tokenizer: CLIP tokenizer.
        image_metadata: List of image metadata dicts (must have 'local_path').
        device: Torch device string.
        confidence_threshold: Minimum confidence threshold.

    Returns:
        Tuple of (person_images, garment_images) lists.
    """
    persons = []
    garments = []
    discarded = 0

    for meta in tqdm(image_metadata, desc="Classifying images"):
        if "local_path" not in meta or not Path(meta["local_path"]).exists():
            continue

        result = classify_image_clip(
            model, preprocess, tokenizer,
            meta["local_path"],
            device=device,
            confidence_threshold=confidence_threshold,
        )

        if result is None:
            discarded += 1
            continue

        class_label, confidence = result
        meta["classification"] = class_label
        meta["clip_confidence"] = confidence

        if class_label == "person":
            persons.append(meta)
        else:
            garments.append(meta)

    logger.info(
        "Classification: %d persons, %d garments, %d discarded (below %.2f threshold)",
        len(persons),
        len(garments),
        discarded,
        confidence_threshold,
    )
    return persons, garments


# ═══════════════════════════════════════════════════════════════════════
# PAIR GENERATION — Random person-garment pairing (unpaired mode)
# ═══════════════════════════════════════════════════════════════════════


def generate_unpaired_pairs(
    person_images: List[Dict[str, Any]],
    garment_images: List[Dict[str, Any]],
    target_pairs: int = 2000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Randomly pair person images with garment images in unpaired mode.
    Each pair combines a person photo with a randomly selected garment.

    Args:
        person_images: List of person image metadata.
        garment_images: List of garment image metadata.
        target_pairs: Number of pairs to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of pair dicts with 'person' and 'garment' fields.
    """
    if not person_images or not garment_images:
        logger.warning("Cannot generate pairs: %d persons, %d garments",
                       len(person_images), len(garment_images))
        return []

    np.random.seed(seed)

    # Limit target pairs to available data
    max_pairs = min(target_pairs, len(person_images) * len(garment_images))
    actual_pairs = min(target_pairs, max_pairs)

    pairs = []
    person_indices = np.random.choice(len(person_images), size=actual_pairs, replace=True)
    garment_indices = np.random.choice(len(garment_images), size=actual_pairs, replace=True)

    for i, (pi, gi) in enumerate(zip(person_indices, garment_indices)):
        pair = {
            "pair_id": f"cc0_{i:05d}",
            "person": person_images[pi],
            "garment": garment_images[gi],
        }
        pairs.append(pair)

    logger.info("Generated %d unpaired person-garment pairs", len(pairs))
    return pairs


# ═══════════════════════════════════════════════════════════════════════
# SYNTHETIC GROUND TRUTH — CatVTON inference for training labels
# ═══════════════════════════════════════════════════════════════════════


def generate_synthetic_tryon(
    pairs: List[Dict[str, Any]],
    output_dir: str,
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    """
    Generate synthetic try-on ground truth by running CatVTON inference
    on each person-garment pair. The CatVTON output becomes the training label.

    Args:
        pairs: List of pair dicts from generate_unpaired_pairs().
        output_dir: Directory to save synthetic try-on results.
        device: Torch device string.

    Returns:
        Updated pairs with 'tryon_result' path added.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Try to load CatVTON for synthetic ground truth generation
    try:
        import torch
        from diffusers import StableDiffusionInpaintPipeline

        logger.info("Loading CatVTON for synthetic try-on ground truth generation...")

        # CatVTON uses a diffusion-based approach
        # Load the CatVTON model from HuggingFace
        from huggingface_hub import hf_hub_download

        # Note: CatVTON model loading depends on the specific implementation
        # We use the official pipeline from zhengchong/CatVTON
        try:
            from CatVTON.pipeline import CatVTONPipeline
            pipeline = CatVTONPipeline.from_pretrained(
                "zhengchong/CatVTON",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            pipeline = pipeline.to(device)
            if device == "cuda":
                pipeline.enable_attention_slicing()
                pipeline.enable_vae_tiling()
            catvton_available = True
        except (ImportError, Exception) as e:
            logger.warning("CatVTON not available: %s. Using composite fallback.", str(e))
            catvton_available = False

    except ImportError:
        logger.warning("Diffusers not available. Using composite fallback.")
        catvton_available = False

    results = []
    for i, pair in enumerate(tqdm(pairs, desc="Generating synthetic try-on")):
        pair_id = pair["pair_id"]
        person_path = pair["person"]["local_path"]
        garment_path = pair["garment"]["local_path"]
        output_file = str(output_path / f"{pair_id}_tryon.jpg")

        # Skip already processed
        if Path(output_file).exists():
            pair["tryon_result"] = output_file
            results.append(pair)
            continue

        try:
            person_img = Image.open(person_path).convert("RGB")
            garment_img = Image.open(garment_path).convert("RGB")

            if catvton_available:
                # Run CatVTON inference
                try:
                    result_img = pipeline(
                        image=person_img,
                        cloth=garment_img,
                        num_inference_steps=30,
                    ).images[0]
                except Exception as e:
                    import torch
                    vram_used = 0.0
                    if torch.cuda.is_available():
                        vram_used = torch.cuda.memory_allocated() / (1024 ** 3)
                    logger.error(
                        "CatVTON inference failed (VRAM: %.2f GB): %s",
                        vram_used,
                        str(e),
                    )
                    # Fallback to composite
                    result_img = create_composite_fallback(person_img, garment_img)
            else:
                # Composite fallback: overlay garment on person region
                result_img = create_composite_fallback(person_img, garment_img)

            # Resize to standard output size
            result_img = result_img.resize((768, 1024), Image.LANCZOS)
            result_img.save(output_file, "JPEG", quality=95)

            pair["tryon_result"] = output_file
            results.append(pair)

        except Exception as e:
            logger.warning("Failed to generate try-on for pair %s: %s", pair_id, str(e))

        if (i + 1) % 100 == 0:
            logger.info("Generated %d/%d synthetic try-on images", i + 1, len(pairs))

    logger.info("Synthetic try-on complete: %d/%d successful", len(results), len(pairs))
    return results


def create_composite_fallback(
    person_img: Image.Image,
    garment_img: Image.Image,
) -> Image.Image:
    """
    Create a simple composite by alpha-blending the garment onto the person's
    torso region. Used as fallback when CatVTON is not available.

    Args:
        person_img: Person PIL image (RGB).
        garment_img: Garment PIL image (RGB).

    Returns:
        Composite PIL image (RGB).
    """
    # Resize both to target
    target_w, target_h = 768, 1024
    person_resized = person_img.resize((target_w, target_h), Image.LANCZOS)
    garment_resized = garment_img.resize((target_w // 2, target_h // 2), Image.LANCZOS)

    # Place garment in the center-upper region of the person
    composite = person_resized.copy()
    paste_x = (target_w - garment_resized.width) // 2
    paste_y = int(target_h * 0.15)  # Upper torso area

    # Create a soft mask for blending
    mask = Image.new("L", garment_resized.size, 180)  # Semi-transparent
    composite.paste(garment_resized, (paste_x, paste_y), mask=mask)

    return composite


# ═══════════════════════════════════════════════════════════════════════
# FULL PIPELINE — Orchestrate scraping, classification, and processing
# ═══════════════════════════════════════════════════════════════════════


def scrape_all_sources(
    config: dict,
    pexels_key: str,
    pixabay_key: str,
) -> List[Dict[str, Any]]:
    """
    Scrape images from all CC0 sources (Pexels + Pixabay).

    Args:
        config: Loaded configuration dict.
        pexels_key: Pexels API key.
        pixabay_key: Pixabay API key.

    Returns:
        Combined list of image metadata.
    """
    cc0_config = config.get("cc0", {})
    queries_garment = cc0_config.get("search_queries", {}).get("garment", [])
    queries_person = cc0_config.get("search_queries", {}).get("person", [])
    all_queries = queries_garment + queries_person
    pages_per_query = cc0_config.get("pages_per_query", 10)

    all_results = []
    seen_urls = set()

    # Pexels
    if pexels_key:
        logger.info("Scraping Pexels API...")
        pexels = PexelsClient(pexels_key)
        for query in all_queries:
            results = pexels.search_all_pages(query, max_pages=pages_per_query)
            for r in results:
                if r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    all_results.append(r)
            time.sleep(1.0)  # API courtesy delay
    else:
        logger.warning("No Pexels API key provided. Skipping Pexels.")

    # Pixabay
    if pixabay_key:
        logger.info("Scraping Pixabay API...")
        pixabay = PixabayClient(pixabay_key)
        for query in all_queries:
            results = pixabay.search_all_pages(query, max_pages=pages_per_query)
            for r in results:
                if r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    all_results.append(r)
            time.sleep(1.0)
    else:
        logger.warning("No Pixabay API key provided. Skipping Pixabay.")

    logger.info("Total unique images scraped: %d", len(all_results))
    return all_results


def process_cc0_pipeline(
    config: dict,
    paths: Dict[str, Path],
    pexels_key: str,
    pixabay_key: str,
    device: str = "cuda",
) -> List[Dict]:
    """
    Full CC0 pipeline: scrape → download → classify → pair → label → manifest.

    Args:
        config: Loaded configuration dict.
        paths: Resolved paths dict.
        pexels_key: Pexels API key.
        pixabay_key: Pixabay API key.
        device: Torch device string.

    Returns:
        List of manifest records for CC0 source.
    """
    cc0_config = config.get("cc0", {})
    cc0_raw = paths["cc0_raw"]
    cc0_processed = paths["cc0_processed"]

    # Create sub-directories
    person_dir = cc0_processed / "person"
    garment_dir = cc0_processed / "garment"
    mask_dir = cc0_processed / "person_mask"
    body_parse_dir = cc0_processed / "body_parse"
    densepose_dir = cc0_processed / "densepose"
    garment_mask_dir = cc0_processed / "garment_mask"
    tryon_dir = cc0_processed / "tryon"
    for d in [person_dir, garment_dir, mask_dir, body_parse_dir,
              densepose_dir, garment_mask_dir, tryon_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Step 1: Scrape
    logger.info("=" * 60)
    logger.info("Step 1/6: Scraping CC0 sources (Pexels + Pixabay)")
    logger.info("=" * 60)
    raw_metadata = scrape_all_sources(config, pexels_key, pixabay_key)

    if not raw_metadata:
        logger.warning("No images scraped. Check API keys.")
        return []

    # Step 2: Download
    logger.info("=" * 60)
    logger.info("Step 2/6: Downloading images")
    logger.info("=" * 60)
    downloaded = download_batch(raw_metadata, str(cc0_raw))

    # Step 3: CLIP Classification
    logger.info("=" * 60)
    logger.info("Step 3/6: Classifying images with CLIP")
    logger.info("=" * 60)
    clip_model, clip_preprocess, clip_tokenizer = load_clip_model(device=device)
    threshold = cc0_config.get("clip_confidence_threshold", 0.75)
    persons, garments = classify_batch(
        clip_model, clip_preprocess, clip_tokenizer,
        downloaded, device=device, confidence_threshold=threshold,
    )

    # Move classified images to appropriate directories
    for meta in persons:
        src = Path(meta["local_path"])
        dst = person_dir / src.name
        if not dst.exists():
            import shutil
            shutil.copy2(str(src), str(dst))
        meta["processed_path"] = str(dst)

    for meta in garments:
        src = Path(meta["local_path"])
        dst = garment_dir / src.name
        if not dst.exists():
            import shutil
            shutil.copy2(str(src), str(dst))
        meta["processed_path"] = str(dst)

    # Step 4: Generate pairs
    logger.info("=" * 60)
    logger.info("Step 4/6: Generating unpaired person-garment pairs")
    logger.info("=" * 60)
    target_pairs = cc0_config.get("target_pairs", 2000)
    pairs = generate_unpaired_pairs(persons, garments, target_pairs=target_pairs)

    # Step 5: Generate synthetic try-on ground truth
    logger.info("=" * 60)
    logger.info("Step 5/6: Generating synthetic try-on ground truth (CatVTON)")
    logger.info("=" * 60)
    processed_pairs = generate_synthetic_tryon(pairs, str(tryon_dir), device=device)

    # Step 6: Generate SAM 2 masks and body estimates, create records
    logger.info("=" * 60)
    logger.info("Step 6/6: Running SAM 2 + HMR 2.0 labeling")
    logger.info("=" * 60)

    # Load SAM 2 and HMR 2.0 (reuse from vivid_prepare)
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

    records = []
    for i, pair in enumerate(tqdm(processed_pairs, desc="Labeling pairs")):
        pair_id = pair["pair_id"]
        person_path = pair["person"].get("processed_path", pair["person"]["local_path"])
        garment_path = pair["garment"].get("processed_path", pair["garment"]["local_path"])

        try:
            # Read person image
            person_img = cv2.imread(person_path)
            if person_img is None:
                continue
            person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)

            # Generate person mask
            mask_path = str(mask_dir / f"{pair_id}_mask.png")
            if not Path(mask_path).exists():
                if mask_gen is not None:
                    mask = generate_person_mask(mask_gen, person_rgb)
                else:
                    # Fallback: full-frame mask
                    mask = np.ones(person_img.shape[:2], dtype=np.uint8) * 255
                if mask is not None:
                    cv2.imwrite(mask_path, mask)
                else:
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
            body_parse_path = str(body_parse_dir / f"{pair_id}_body_parse.png")
            cv2.imwrite(body_parse_path, body_parse)

            # Generate densepose
            densepose = generate_densepose_map(person_rgb, body_params)
            densepose_path = str(densepose_dir / f"{pair_id}_densepose.png")
            cv2.imwrite(densepose_path, densepose)

            # Generate garment mask
            garment_img = cv2.imread(garment_path)
            garment_mask_path = str(garment_mask_dir / f"{pair_id}_garment_mask.png")
            if garment_img is not None and not Path(garment_mask_path).exists():
                garment_gray = cv2.cvtColor(garment_img, cv2.COLOR_BGR2GRAY)
                _, garment_mask = cv2.threshold(garment_gray, 240, 255, cv2.THRESH_BINARY_INV)
                cv2.imwrite(garment_mask_path, garment_mask)

            # Determine garment category from query
            query = pair["garment"].get("query", "")
            if any(kw in query.lower() for kw in ["dress", "gown"]):
                category = "dress"
            elif any(kw in query.lower() for kw in ["jeans", "trouser", "pant", "skirt"]):
                category = "lower"
            else:
                category = "upper"

            record = {
                "id": pair_id,
                "person_image": person_path,
                "person_mask": mask_path,
                "body_parse": body_parse_path,
                "densepose": densepose_path,
                "garment_image": garment_path,
                "garment_mask": garment_mask_path,
                "category": category,
                "license": "CC0",
                "source": "CC0",
                "clip_confidence": pair["garment"].get("clip_confidence", 0.0),
            }
            records.append(record)

        except Exception as e:
            logger.warning("Failed to label pair %s: %s", pair_id, str(e))

        if (i + 1) % 100 == 0:
            logger.info("Labeled %d/%d pairs", i + 1, len(processed_pairs))

    logger.info("CC0 pipeline complete: %d records generated", len(records))
    return records


# ═══════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════


def main():
    """
    Main entry point for CC0 web scraping pipeline.

    Usage:
        python -m pipeline.cc0_scraper --pexels-key KEY --pixabay-key KEY
    """
    parser = argparse.ArgumentParser(
        description="ARVTON — CC0 Web Scraper (Source B)",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to dataset_config.yaml")
    parser.add_argument("--platform", type=str, default=None, choices=["colab", "amd", "local"])
    parser.add_argument("--pexels-key", type=str, default=None, help="Pexels API key")
    parser.add_argument("--pixabay-key", type=str, default=None, help="Pixabay API key")
    args = parser.parse_args()

    # Setup
    setup_logging("INFO")
    logger.info("=" * 60)
    logger.info("ARVTON — CC0 Web Scraper (Source B)")
    logger.info("=" * 60)

    config = load_config(args.config)
    platform = args.platform or detect_platform()
    paths = get_paths(platform)
    ensure_dirs(paths)

    if platform == "colab":
        mount_google_drive()

    gpu_info = detect_gpu_info()
    device = "cuda" if gpu_info["available"] else "cpu"

    # Get API keys from args or environment
    pexels_key = args.pexels_key or os.environ.get("PEXELS_API_KEY", "")
    pixabay_key = args.pixabay_key or os.environ.get("PIXABAY_API_KEY", "")

    if not pexels_key and not pixabay_key:
        logger.error(
            "No API keys provided. Set PEXELS_API_KEY and/or PIXABAY_API_KEY "
            "environment variables, or use --pexels-key / --pixabay-key flags."
        )
        return

    # Run pipeline
    records = process_cc0_pipeline(config, paths, pexels_key, pixabay_key, device=device)

    # Merge with existing manifests
    from pipeline.vivid_prepare import merge_manifests, save_manifest

    train_path = str(paths["train_manifest"])
    all_train = merge_manifests(train_path, records)
    save_manifest(all_train, train_path)

    # Summary
    print("\n" + "=" * 60)
    print("Phase 0 — Source B (CC0) Complete")
    print("=" * 60)
    print(f"  New CC0 records: {len(records)}")
    print(f"  Total train records: {len(all_train)}")
    print(f"  Train manifest: {train_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
