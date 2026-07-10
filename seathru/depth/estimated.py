"""
@file estimated.py
@brief Image-derived depth *prior* - a no-dependency, no-SfM range estimate.

Underwater, red light attenuates far faster than green/blue, so pixels where the
red channel is depleted relative to green/blue tend to be farther from the
camera. This module turns that cue into a smooth per-pixel range map. It is an
approximate monocular prior (a red coral is red, not necessarily far), not a
metric measurement - but it is spatially varying, which is what activates
Sea-thru's illuminant and attenuation stages (`f`, `l`, `p`, `epsilon`).

Use it to:
  * preview the FULL range-dependent Sea-thru today, with no torch / no COLMAP;
  * see the effect of every tuning parameter (unlike a flat plane).

Upgrade to `ColmapDepthSource` (metric SfM depth) for final quality.

@author Michael Venz
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .base import DepthSource, ImageMeta

## Numerical floor for percentile-normalisation ratios.
EPS = 1e-6


class EstimatedDepthSource(DepthSource):
    """@brief Red-attenuation-cue depth prior; see module docstring."""

    def __init__(self, near=1.0, far=10.0, blur_frac=0.02,
                 lo_pct=5.0, hi_pct=95.0):
        """
        @param near Metres mapped to the nearest cue value.
        @param far Metres mapped to the farthest cue value.
        @param blur_frac Gaussian smoothing sigma as a fraction of the image's
            long edge; larger smoothing gives clean range gradients for the
            neighbourhood machinery.
        @param lo_pct Lower robust-normalisation percentile for the cue.
        @param hi_pct Upper robust-normalisation percentile for the cue.
        """
        self.near = near
        self.far = far
        self.blur_frac = blur_frac
        self.lo_pct = lo_pct
        self.hi_pct = hi_pct

    def get_depth(self, img, meta: ImageMeta):
        """@brief @copydoc DepthSource.get_depth
        Derived from the input image's own colour (see module docstring); if
        ``meta.depth_m`` is valid the mean range is anchored to it."""
        img = np.asarray(img, dtype=np.float64)
        r, g, b = img[..., 0], img[..., 1], img[..., 2]

        # Red-attenuation cue: large where red is depleted vs green/blue -> far.
        cue = np.maximum(g, b) - r

        sigma = max(img.shape[:2]) * self.blur_frac
        cue = gaussian_filter(cue, sigma=sigma)

        lo, hi = np.percentile(cue, [self.lo_pct, self.hi_pct])
        norm = np.clip((cue - lo) / (hi - lo + EPS), 0.0, 1.0)
        depth = self.near + norm * (self.far - self.near)

        # If the CSV reports a valid altitude, anchor the mean range to it so the
        # absolute scale is roughly right (relative shape is unchanged).
        if meta and meta.depth_m and meta.depth_m > 0:
            depth *= meta.depth_m / max(depth.mean(), EPS)

        return depth.astype(np.float32)
