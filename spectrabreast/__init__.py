"""
SpectraBreast — HSI + 3D Mesh registration pipeline.

Public API (re-exported for convenience):
    from spectrabreast import run_full_pipeline, run_full_pipeline_roi, ...
"""

from .pipeline import (
    run_full_pipeline,
    extract_suspicious_centroids,
    load_mesh,
    save_render,
    save_turbo_render,
)

from .pipeline_roi import run_full_pipeline_roi
from .roi_align import (
    compute_roi_to_png_homography,
    load_liveview_png,
)

# render_gpu è opzionale: dipende da torch (e/o cupy)
try:
    from .render_gpu import render_orthographic_topview_gpu
except ImportError:
    render_orthographic_topview_gpu = None

__all__ = [
    "run_full_pipeline",
    "run_full_pipeline_roi",
    "compute_roi_to_png_homography",
    "load_liveview_png",
    "extract_suspicious_centroids",
    "load_mesh",
    "save_render",
    "save_turbo_render",
    "render_orthographic_topview_gpu",
]

__version__ = "0.3.0"
