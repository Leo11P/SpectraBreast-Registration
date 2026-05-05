"""
SpectraBreast — HSI + 3D Mesh registration pipeline.

Public API (re-exported for convenience):
    from spectrabreast import run_full_pipeline, load_mesh, ...
"""

from .pipeline import (
    run_full_pipeline,
    extract_suspicious_centroids,
    load_mesh,
    save_render,
    save_turbo_render,
)

# render_gpu è opzionale: dipende da torch (e/o cupy)
try:
    from .render_gpu import render_orthographic_topview_gpu
except ImportError:
    render_orthographic_topview_gpu = None

__all__ = [
    "run_full_pipeline",
    "extract_suspicious_centroids",
    "load_mesh",
    "save_render",
    "save_turbo_render",
    "render_orthographic_topview_gpu",
]

__version__ = "0.2.0"
