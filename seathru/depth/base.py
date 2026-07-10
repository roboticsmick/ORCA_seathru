"""
@file base.py
@brief Depth-source interface.

Sea-thru needs a per-pixel *range* map (metres) the same size as the working
image. Different deployments obtain range differently:

  * ``MonocularDepthSource`` - neural monocular depth (Depth Anything V2 / MiDaS),
    scaled to metres using the ASV altitude when available. Works with no SfM.
  * ``FileDepthSource``      - per-image range maps exported from photogrammetry
    (COLMAP / Metashape / OpenDroneMap). Most accurate.
  * ``PlaneDepthSource``     - a single scalar altitude (e.g. CSV ``depth_m``)
    broadcast to a flat plane. A coarse fallback for a nadir camera over flat
    reef, or when nothing better exists.

All sources return a float32 ``(H, W)`` array in metres where ``<= 0`` marks
invalid / unknown pixels.

@author Michael Venz
"""
from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass
class ImageMeta:
    """
    @brief Per-image metadata from the survey CSV (only what depth sources may use).
    @param image_name File name (basename) the metadata applies to.
    @param depth_m ASV altitude over the scene, metres; ``None`` if unreported.
    @param heading_deg Vehicle heading in degrees, if known.
    @param latitude Latitude in decimal degrees, if known.
    @param longitude Longitude in decimal degrees, if known.
    """
    image_name: str
    depth_m: float | None = None
    heading_deg: float | None = None
    latitude: float | None = None
    longitude: float | None = None


class DepthSource(abc.ABC):
    """@brief Produce a per-pixel range map (metres) for a working-resolution image."""

    @abc.abstractmethod
    def get_depth(self, img, meta: ImageMeta):
        """
        @brief Compute or load the range map for one image.
        @param img (H, W, 3) float working-resolution image in [0, 1]; some
            sources (e.g. EstimatedDepthSource, MonocularDepthSource) derive
            depth from the image content itself.
        @param meta ImageMeta for this image (may carry a known altitude/GPS).
        @return float32 ``(H, W)`` range map in metres; ``<= 0`` marks invalid
            pixels.
        """
        raise NotImplementedError
