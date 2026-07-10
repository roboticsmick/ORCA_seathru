"""
@file __init__.py
@brief Pluggable range-map sources for Sea-thru.
@author Michael Venz
"""
from .base import DepthSource, ImageMeta
from .colmap_source import ColmapDepthSource
from .estimated import EstimatedDepthSource
from .file_source import FileDepthSource
from .plane_source import PlaneDepthSource

__all__ = ["DepthSource", "ImageMeta", "FileDepthSource", "PlaneDepthSource",
           "ColmapDepthSource", "EstimatedDepthSource", "MonocularDepthSource"]


def __getattr__(name):
    """@brief PEP 562 module-level lazy attribute lookup.

    Defers importing seathru.depth.monocular (and its PyTorch dependency)
    until MonocularDepthSource is actually accessed, so the rest of the
    library works with no torch installed.

    @param name Attribute being accessed on this module.
    @return The MonocularDepthSource class.
    @throws AttributeError for any other undefined attribute.
    """
    if name == "MonocularDepthSource":
        from .monocular import MonocularDepthSource
        return MonocularDepthSource
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
