# Module: pipeline
# License: MIT (ARVTON project)
# Description: ARVTON pipeline package — dataset construction, segmentation, try-on, 3D reconstruction, export.
# Platform: Both (Colab + AMD ROCm)
# Dependencies: See requirements.txt

"""
ARVTON Pipeline Package
========================
Provides the complete AR/VR Virtual Try-On pipeline:
  - Dataset construction (vivid_prepare, cc0_scraper, synthetic_gen)
  - Segmentation (segment)
  - 2D virtual try-on (tryon)
  - 3D body reconstruction (reconstruct3d)
  - GLB export (export3d)
"""

__version__ = "0.1.0"
__license__ = "MIT"
