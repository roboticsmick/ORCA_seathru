"""
@file plane_source.py
@brief Flat-plane range map from a single scalar altitude.
@author Michael Venz
"""
from __future__ import annotations

import numpy as np

from .base import DepthSource, ImageMeta


class PlaneDepthSource(DepthSource):
    """
    @brief Broadcast a single altitude to a constant range map.

    For a downward (nadir) camera over roughly flat reef, the range to every
    pixel is approximately the camera altitude, so a constant plane is a
    reasonable first approximation. Uses ``meta.depth_m`` when it is a valid
    (positive) reading, otherwise falls back to ``default_m``.

    Note: with a *constant* range map Sea-thru's range-dependent attenuation
    term becomes a global scale, so this mode mainly performs backscatter
    removal + white balance. Use monocular or SfM depth to get the full
    range-varying colour correction.
    """

    def __init__(self, default_m=5.0):
        """@brief @param default_m Altitude in metres used when ``meta.depth_m``
        is missing/invalid."""
        self.default_m = default_m

    def get_depth(self, img, meta: ImageMeta):
        """@brief @copydoc DepthSource.get_depth"""
        z = meta.depth_m if (meta and meta.depth_m and meta.depth_m > 0) else self.default_m
        return np.full(img.shape[:2], float(z), dtype=np.float32)
