"""
@file file_source.py
@brief Range maps loaded from photogrammetry / SfM output files.
@author Michael Venz
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .base import DepthSource, ImageMeta


class FileDepthSource(DepthSource):
    """
    @brief Load a precomputed per-image range map from disk (SfM / stereo).

    For each image ``NAME.JPG`` the source looks for ``NAME.<ext>`` in
    ``depth_dir`` trying, in order, the given ``extensions``. Supported formats:

      * ``.npy``            - float array of range in metres (preferred).
      * ``.tif`` / ``.tiff``/ ``.exr`` - float image of range in metres.
      * ``.png`` / ``.jpg`` - 8/16-bit; scaled by ``scale`` into metres
                              (e.g. a normalised SfM depth exported as 16-bit).

    Maps are resized (nearest) to the working image size. Non-positive and
    non-finite values are treated as invalid.
    """

    def __init__(self, depth_dir, extensions=(".npy", ".tif", ".tiff", ".exr", ".png"),
                 scale=1.0, invalid_below=0.0):
        """
        @param depth_dir Directory containing one range-map file per image.
        @param extensions Extensions tried, in order, when locating a map.
        @param scale Multiplier applied to loaded 8/16-bit maps to convert to metres.
        @param invalid_below Values at or below this (after scaling) are marked invalid.
        """
        self.depth_dir = Path(depth_dir)
        self.extensions = extensions
        self.scale = scale
        self.invalid_below = invalid_below

    def _find(self, image_name):
        """@brief Locate the depth-map file for one image.
        @param image_name Source image file name.
        @return Path to the matching depth file, or None if not found."""
        stem = Path(image_name).stem
        for ext in self.extensions:
            cand = self.depth_dir / f"{stem}{ext}"
            if cand.exists():
                return cand
        return None

    def get_depth(self, img, meta: ImageMeta):
        """@brief @copydoc DepthSource.get_depth
        @throws FileNotFoundError if no matching depth file exists in ``depth_dir``."""
        path = self._find(meta.image_name)
        if path is None:
            raise FileNotFoundError(
                f"No depth map for {meta.image_name} in {self.depth_dir}")
        if path.suffix == ".npy":
            depth = np.load(path).astype(np.float32)
        else:
            depth = np.asarray(Image.open(path), dtype=np.float32)
            if depth.ndim == 3:
                depth = depth[..., 0]
        depth = depth * self.scale

        H, W = img.shape[:2]
        if depth.shape != (H, W):
            depth = np.asarray(
                Image.fromarray(depth).resize((W, H), Image.NEAREST),
                dtype=np.float32)

        depth[~np.isfinite(depth)] = 0.0
        depth[depth <= self.invalid_below] = 0.0
        return depth
