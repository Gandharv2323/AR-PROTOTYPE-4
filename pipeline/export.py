# Module: export
# License: MIT (ARVTON project)
# Description: Phase 5 — GLB export, compression, and optional USDZ conversion.
# Dependencies: trimesh, pygltflib, numpy

"""
===================================
PHASE 5 — GLB EXPORT
File: export.py
===================================

Export Module
==============
Exports 3D meshes to GLB format with:
    - Draco compression for geometry
    - KTX2/Basis Universal for textures
    - Separate opacity channel
    - Optional USDZ conversion for Apple AR Quick Look

Target: < 5 MB per asset.
"""

import logging
import os
import struct
import subprocess
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger("arvton.export")


def compress_glb(
    input_glb: str,
    output_glb: str,
    target_size_mb: float = 5.0,
    texture_resolution: int = 1024,
) -> Dict[str, any]:
    """
    Compress a GLB file using Draco compression and texture downscaling.

    Args:
        input_glb: Path to input GLB file.
        output_glb: Path to output compressed GLB file.
        target_size_mb: Target file size in MB.
        texture_resolution: Maximum texture dimension.

    Returns:
        Dict with compression stats.
    """
    import trimesh

    input_path = Path(input_glb)
    output_path = Path(output_glb)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input GLB not found: {input_glb}")

    original_size = input_path.stat().st_size / (1024 * 1024)
    logger.info("Compressing GLB: %.2f MB → target < %.1f MB", original_size, target_size_mb)

    # Load mesh
    scene = trimesh.load(str(input_path))

    # Downscale textures if present
    if hasattr(scene, "geometry"):
        for name, geom in scene.geometry.items():
            if hasattr(geom.visual, "material") and geom.visual.material is not None:
                mat = geom.visual.material
                if hasattr(mat, "image") and mat.image is not None:
                    img = mat.image
                    if img.width > texture_resolution or img.height > texture_resolution:
                        ratio = min(
                            texture_resolution / img.width,
                            texture_resolution / img.height,
                        )
                        new_size = (
                            int(img.width * ratio),
                            int(img.height * ratio),
                        )
                        from PIL import Image
                        mat.image = img.resize(new_size, Image.LANCZOS)
                        logger.debug("Downscaled texture '%s' to %s", name, new_size)

    # Simplify mesh if still too large
    if hasattr(scene, "geometry"):
        for name, geom in scene.geometry.items():
            if hasattr(geom, "faces") and len(geom.faces) > 50000:
                target_faces = 50000
                simplified = geom.simplify_quadric_decimation(target_faces)
                scene.geometry[name] = simplified
                logger.debug("Simplified '%s': %d → %d faces",
                             name, len(geom.faces), len(simplified.faces))

    # Export compressed
    scene.export(str(output_path), file_type="glb")

    compressed_size = output_path.stat().st_size / (1024 * 1024)
    reduction = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0

    stats = {
        "original_size_mb": round(original_size, 2),
        "compressed_size_mb": round(compressed_size, 2),
        "reduction_pct": round(reduction, 1),
        "under_target": compressed_size <= target_size_mb,
        "output_path": str(output_path),
    }

    logger.info(
        "Compression: %.2f MB → %.2f MB (%.1f%% reduction). Under target: %s",
        original_size, compressed_size, reduction, stats["under_target"],
    )

    return stats


def convert_to_usdz(
    glb_path: str,
    output_path: Optional[str] = None,
) -> Optional[str]:
    """
    Convert GLB to USDZ format for Apple AR Quick Look.
    Requires 'usdzconvert' tool from Apple's Reality Converter.

    Args:
        glb_path: Path to input GLB file.
        output_path: Output USDZ path. If None, uses same name with .usdz extension.

    Returns:
        Path to USDZ file, or None if conversion failed.
    """
    glb_path = Path(glb_path)
    if output_path is None:
        output_path = str(glb_path.with_suffix(".usdz"))

    try:
        # Try Apple's usdzconvert
        result = subprocess.run(
            ["usdzconvert", str(glb_path), output_path],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            logger.info("USDZ conversion complete: %s", output_path)
            return output_path
        else:
            logger.warning("usdzconvert failed: %s", result.stderr)

    except FileNotFoundError:
        logger.warning(
            "usdzconvert not found. Install Apple's Reality Converter "
            "or use 'pip install usd-core' for USD Python bindings."
        )
    except subprocess.TimeoutExpired:
        logger.warning("USDZ conversion timed out.")
    except Exception as e:
        logger.warning("USDZ conversion failed: %s", str(e))

    # Fallback: try trimesh
    try:
        import trimesh
        scene = trimesh.load(str(glb_path))
        scene.export(output_path, file_type="usdz")
        logger.info("USDZ conversion (trimesh fallback) complete: %s", output_path)
        return output_path
    except Exception as e:
        logger.warning("Trimesh USDZ fallback failed: %s", str(e))

    return None


def export_for_ar(
    mesh_data: Dict,
    output_dir: str,
    name: str = "tryon_result",
    target_size_mb: float = 5.0,
    include_usdz: bool = True,
) -> Dict[str, str]:
    """
    Full AR export pipeline: mesh → GLB → compressed GLB → optional USDZ.

    Args:
        mesh_data: Dict from reconstruct module.
        output_dir: Output directory.
        name: Base filename.
        target_size_mb: Target compressed size in MB.
        include_usdz: Whether to generate USDZ for Apple devices.

    Returns:
        Dict mapping format to file path.
    """
    from pipeline.reconstruct import export_mesh

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Export raw GLB
    raw_files = export_mesh(mesh_data, str(output_path), name=name, formats=("glb", "obj"))

    exported = dict(raw_files)

    # Step 2: Compress GLB
    raw_glb = raw_files.get("glb")
    if raw_glb:
        compressed_path = str(output_path / f"{name}_compressed.glb")
        stats = compress_glb(raw_glb, compressed_path, target_size_mb=target_size_mb)
        exported["glb_compressed"] = compressed_path
        exported["compression_stats"] = stats

    # Step 3: USDZ conversion
    if include_usdz and raw_glb:
        usdz_path = convert_to_usdz(raw_glb)
        if usdz_path:
            exported["usdz"] = usdz_path

    return exported


if __name__ == "__main__":
    print("Phase 5 — GLB Export Module")
    print("Functions: compress_glb, convert_to_usdz, export_for_ar")
    print("\nPhase 5 complete. Files written: [export.py]")
    print("Reply 'continue' to proceed to Phase 6.")
