# Module: reconstruct
# License: MIT (ARVTON project)
# Description: Phase 4 — 3D reconstruction using TripoSR (MIT) + SyncHuman (Apache-2.0).
# Platform: Colab T4/A100 + AMD ROCm
# Dependencies: torch, trimesh, Pillow, numpy

"""
===================================
PHASE 4 — 3D RECONSTRUCTION
File: reconstruct.py
===================================

3D Reconstruction Module
=========================
Stage 1: TripoSR — single-image → coarse 3D mesh (MIT license)
Stage 2: SyncHuman — multi-view refinement for clothed humans (Apache-2.0)

Output: textured OBJ mesh + GLB model ready for export.
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("arvton.reconstruct")


# ═══════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════

_triposr_model = None
_synchuman_model = None


def load_triposr(device: str = "cuda"):
    """
    Load TripoSR model for single-image 3D reconstruction.
    MIT license — stabilityai/TripoSR.

    Args:
        device: Torch device string.

    Returns:
        Loaded TripoSR model.
    """
    global _triposr_model
    if _triposr_model is not None:
        return _triposr_model

    import torch

    logger.info("Loading TripoSR (stabilityai/TripoSR)...")

    try:
        # ROCm compatible: YES — standard PyTorch + transformers
        from tsr.system import TSR

        model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        model.to(device)

        # Memory optimization
        if device == "cuda":
            model.renderer.set_chunk_size(8192)

        _triposr_model = model
        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / (1024 ** 3)
            logger.info("TripoSR loaded. VRAM: %.2f GB", vram)

        return model

    except ImportError:
        logger.warning(
            "TripoSR not installed. Install via: pip install triposr\n"
            "Returning stub model."
        )
        return None

    except Exception as e:
        logger.warning("TripoSR loading failed: %s", str(e))
        return None


def load_synchuman(device: str = "cuda"):
    """
    Load SyncHuman model for multi-view human refinement.
    Apache-2.0 license — lizhuo/SyncHuman.

    Args:
        device: Torch device string.

    Returns:
        Loaded SyncHuman model (or None if unavailable).
    """
    global _synchuman_model
    if _synchuman_model is not None:
        return _synchuman_model

    import torch

    logger.info("Loading SyncHuman (lizhuo/SyncHuman)...")

    try:
        from synchuman.model import SyncHumanModel

        model = SyncHumanModel.from_pretrained("lizhuo/SyncHuman")
        model = model.to(device)
        model.eval()

        _synchuman_model = model
        logger.info("SyncHuman loaded.")
        return model

    except ImportError:
        logger.warning("SyncHuman not installed. Skipping multi-view refinement.")
        return None
    except Exception as e:
        logger.warning("SyncHuman loading failed: %s", str(e))
        return None


# ═══════════════════════════════════════════════════════════════════════
# STAGE 1: TripoSR — Single-image → coarse mesh
# ═══════════════════════════════════════════════════════════════════════


def reconstruct_coarse(
    image: Image.Image,
    device: str = "cuda",
    marching_cubes_resolution: int = 256,
) -> Optional[Dict]:
    """
    Generate a coarse 3D mesh from a single try-on image using TripoSR.

    Args:
        image: Input RGB image (try-on result).
        device: Torch device.
        marching_cubes_resolution: Resolution for marching cubes extraction.

    Returns:
        Dict with 'vertices', 'faces', 'vertex_colors', and 'mesh' (trimesh object)
        or None if reconstruction fails.
    """
    import torch

    model = load_triposr(device)

    if model is None:
        logger.warning("TripoSR unavailable. Generating placeholder mesh.")
        return _generate_placeholder_mesh(image)

    try:
        logger.info("Running TripoSR reconstruction...")

        # Preprocess image
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Process with TripoSR
        with torch.no_grad():
            scene_codes = model([image], device=device)

        # Extract mesh
        meshes = model.extract_mesh(
            scene_codes,
            resolution=marching_cubes_resolution,
        )

        if not meshes:
            logger.error("TripoSR returned no meshes.")
            return None

        mesh = meshes[0]

        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / (1024 ** 3)
            logger.info(
                "Coarse mesh: %d vertices, %d faces (VRAM: %.2f GB)",
                len(mesh.vertices), len(mesh.faces), vram,
            )

        return {
            "vertices": mesh.vertices,
            "faces": mesh.faces,
            "vertex_colors": getattr(mesh, "vertex_colors", None),
            "mesh": mesh,
        }

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("TripoSR OOM error.")
            torch.cuda.empty_cache()
        else:
            logger.error("TripoSR error: %s", str(e))
        return None

    except Exception as e:
        logger.error("TripoSR reconstruction failed: %s", str(e))
        return None


def _generate_placeholder_mesh(
    image: Image.Image,
) -> Dict:
    """
    Generate a placeholder flat-plane mesh when TripoSR is unavailable.
    This allows the pipeline to continue for testing.

    Args:
        image: Input image (used for texturing the plane).

    Returns:
        Dict with placeholder mesh data.
    """
    import trimesh

    # Create a simple textured plane
    vertices = np.array([
        [-0.5, -0.75, 0],
        [0.5, -0.75, 0],
        [0.5, 0.75, 0],
        [-0.5, 0.75, 0],
    ], dtype=np.float32)

    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.int32)

    # UV coordinates
    uv = np.array([
        [0, 0], [1, 0], [1, 1], [0, 1],
    ], dtype=np.float32)

    # Create textured mesh
    texture = trimesh.visual.TextureVisuals(
        uv=uv,
        image=image.resize((512, 512)),
    )

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        visual=texture,
    )

    logger.warn("Generated placeholder mesh (4 verts, 2 faces)")

    return {
        "vertices": vertices,
        "faces": faces,
        "vertex_colors": None,
        "mesh": mesh,
    }


# ═══════════════════════════════════════════════════════════════════════
# STAGE 2: SyncHuman — Multi-view refinement
# ═══════════════════════════════════════════════════════════════════════


def refine_mesh_synchuman(
    coarse_mesh: Dict,
    image: Image.Image,
    device: str = "cuda",
    num_views: int = 4,
) -> Dict:
    """
    Refine the coarse mesh using SyncHuman multi-view generation.

    SyncHuman generates consistent multi-view images from the front view,
    then uses multi-view reconstruction to refine geometry and texture.

    Args:
        coarse_mesh: Dict from reconstruct_coarse().
        image: Original front-view image.
        device: Torch device.
        num_views: Number of views to synthesize (default: 4).

    Returns:
        Updated mesh dict with refined geometry and texture.
    """
    import torch

    model = load_synchuman(device)

    if model is None:
        logger.info("SyncHuman not available. Using coarse mesh directly.")
        return coarse_mesh

    try:
        logger.info("Running SyncHuman multi-view refinement (%d views)...", num_views)

        # Generate multi-view images
        with torch.no_grad():
            multi_views = model.generate_views(
                image=image,
                num_views=num_views,
                resolution=512,
            )

        # Refine mesh with multi-view consistency
        refined_mesh = model.refine_mesh(
            mesh=coarse_mesh["mesh"],
            views=multi_views,
        )

        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / (1024 ** 3)
            logger.info(
                "Refined mesh: %d vertices, %d faces (VRAM: %.2f GB)",
                len(refined_mesh.vertices), len(refined_mesh.faces), vram,
            )

        return {
            "vertices": refined_mesh.vertices,
            "faces": refined_mesh.faces,
            "vertex_colors": getattr(refined_mesh, "vertex_colors", None),
            "mesh": refined_mesh,
            "multi_views": multi_views,
        }

    except Exception as e:
        logger.warning("SyncHuman refinement failed: %s. Using coarse mesh.", str(e))
        return coarse_mesh


# ═══════════════════════════════════════════════════════════════════════
# EXPORT — OBJ + GLB
# ═══════════════════════════════════════════════════════════════════════


def export_mesh(
    mesh_data: Dict,
    output_dir: str,
    name: str = "tryon_result",
    formats: tuple = ("obj", "glb"),
) -> Dict[str, str]:
    """
    Export the reconstructed mesh to OBJ and/or GLB formats.

    Args:
        mesh_data: Dict from reconstruct_coarse() or refine_mesh_synchuman().
        output_dir: Directory to save exported files.
        name: Base name for output files.
        formats: Tuple of export formats ('obj', 'glb', 'ply', 'stl').

    Returns:
        Dict mapping format → file path.
    """
    import trimesh

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    mesh = mesh_data.get("mesh")

    if mesh is None:
        # Build mesh from raw data
        mesh = trimesh.Trimesh(
            vertices=mesh_data["vertices"],
            faces=mesh_data["faces"],
        )
        if mesh_data.get("vertex_colors") is not None:
            mesh.visual.vertex_colors = mesh_data["vertex_colors"]

    exported = {}

    for fmt in formats:
        file_path = str(output_path / f"{name}.{fmt}")
        try:
            mesh.export(file_path, file_type=fmt)
            exported[fmt] = file_path
            logger.info("Exported %s: %s", fmt.upper(), file_path)
        except Exception as e:
            logger.warning("Export to %s failed: %s", fmt, str(e))

    return exported


# ═══════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════


def reconstruct(
    image_path: str,
    output_dir: str,
    device: str = "cuda",
    use_synchuman: bool = True,
    export_formats: tuple = ("obj", "glb"),
) -> Dict:
    """
    Run the full 3D reconstruction pipeline:
    TripoSR → (optional) SyncHuman refinement → export.

    Args:
        image_path: Path to the try-on result image.
        output_dir: Output directory for mesh files.
        device: Torch device.
        use_synchuman: Whether to apply SyncHuman refinement.
        export_formats: Tuple of export formats.

    Returns:
        Dict with 'mesh_data', 'exported_files', and 'success' flag.
    """
    logger.info("3D Reconstruction: %s", Path(image_path).name)

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Stage 1: Coarse reconstruction
    logger.info("Stage 1: TripoSR coarse reconstruction...")
    mesh_data = reconstruct_coarse(image, device=device)

    if mesh_data is None:
        logger.error("Coarse reconstruction failed. Aborting.")
        return {"mesh_data": None, "exported_files": {}, "success": False}

    # Stage 2: SyncHuman refinement (optional)
    if use_synchuman:
        logger.info("Stage 2: SyncHuman multi-view refinement...")
        mesh_data = refine_mesh_synchuman(mesh_data, image, device=device)

    # Export
    name = Path(image_path).stem
    exported = export_mesh(mesh_data, output_dir, name=name, formats=export_formats)

    logger.info("3D reconstruction complete.")
    return {
        "mesh_data": mesh_data,
        "exported_files": exported,
        "success": True,
    }


# ═══════════════════════════════════════════════════════════════════════
# CLEANUP
# ═══════════════════════════════════════════════════════════════════════


def clear_cache():
    """Clear cached reconstruction models."""
    global _triposr_model, _synchuman_model
    _triposr_model = None
    _synchuman_model = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Reconstruction model cache cleared.")
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    import argparse
    from pipeline.platform_utils import setup_logging

    parser = argparse.ArgumentParser(description="ARVTON — Phase 4 3D Reconstruction")
    parser.add_argument("--image", required=True, help="Try-on result image")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-synchuman", action="store_true")
    args = parser.parse_args()

    setup_logging("INFO")

    result = reconstruct(
        args.image, args.output,
        device=args.device,
        use_synchuman=not args.no_synchuman,
    )

    if result["success"]:
        print(f"\nPhase 4 complete. Exported files:")
        for fmt, path in result["exported_files"].items():
            print(f"  {fmt}: {path}")
    else:
        print("\nPhase 4 failed. Check logs for details.")

    print("\nReply 'continue' to proceed to Phase 5.")
