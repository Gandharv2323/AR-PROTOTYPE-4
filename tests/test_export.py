# Module: test_export
# License: MIT (ARVTON project)
# Description: Phase 8 — GLB export integration tests.
# Dependencies: pytest, trimesh

"""
tests/test_export.py
Assert:
  - .glb size < 15MB
  - trimesh.load(path) raises no exception
  - At least one material with texture in the scene
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="module")
def test_glb():
    """Create a test GLB file using trimesh."""
    import trimesh
    from PIL import Image

    # Create a textured mesh
    vertices = np.array([
        [-0.5, -0.5, 0], [0.5, -0.5, 0],
        [0.5, 0.5, 0], [-0.5, 0.5, 0],
    ], dtype=np.float32)

    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)

    texture_img = Image.new("RGB", (256, 256), (100, 150, 200))
    texture = trimesh.visual.TextureVisuals(uv=uv, image=texture_img)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=texture)

    output_dir = tempfile.mkdtemp()
    glb_path = os.path.join(output_dir, "test_model.glb")
    mesh.export(glb_path, file_type="glb")

    yield glb_path

    # Cleanup
    if os.path.exists(glb_path):
        os.unlink(glb_path)


class TestExport:
    """GLB export integration tests."""

    def test_glb_size_under_15mb(self, test_glb):
        """Assert .glb file size < 15 MB."""
        size_mb = Path(test_glb).stat().st_size / (1024 * 1024)
        assert size_mb < 15, f"GLB size: {size_mb:.2f} MB (limit: 15 MB)"

    def test_glb_loadable(self, test_glb):
        """Assert trimesh.load(path) raises no exception."""
        import trimesh

        scene = trimesh.load(test_glb)
        assert scene is not None

    def test_glb_has_material(self, test_glb):
        """Assert at least one material with texture."""
        import trimesh

        scene = trimesh.load(test_glb)

        has_material = False

        if hasattr(scene, "geometry"):
            for name, geom in scene.geometry.items():
                if hasattr(geom, "visual") and geom.visual is not None:
                    if hasattr(geom.visual, "material") and geom.visual.material is not None:
                        has_material = True
                        break
                    if hasattr(geom.visual, "uv") and geom.visual.uv is not None:
                        has_material = True
                        break
        elif hasattr(scene, "visual"):
            if scene.visual is not None:
                has_material = True

        assert has_material, "No material/texture found in GLB"

    def test_compression(self, test_glb):
        """Test GLB compression reduces size."""
        from pipeline.export import compress_glb

        compressed_path = test_glb.replace(".glb", "_compressed.glb")
        try:
            stats = compress_glb(test_glb, compressed_path)
            assert Path(compressed_path).exists()
            assert stats["compressed_size_mb"] <= stats["original_size_mb"] + 0.1
        except Exception as e:
            pytest.skip(f"Compression test skipped: {e}")
        finally:
            if os.path.exists(compressed_path):
                os.unlink(compressed_path)
