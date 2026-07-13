"""
@file colmap_source.py
@brief Range maps from a COLMAP dense (MVS) workspace.

After ``colmap patch_match_stereo`` runs, per-image depth maps live in
``<workspace>/stereo/depth_maps/<image>.geometric.bin`` (COLMAP's binary array
format). If the sparse model was georegistered to the ASV GPS with
``colmap model_aligner``, these depths are in **metres** along the camera axis,
which is what Sea-thru needs.

Values <= 0 are "no estimate"; extreme outliers past a percentile are clipped to
invalid so a few bad MVS pixels don't wreck the range statistics.

@author Michael Venz
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .base import DepthSource, ImageMeta


def fill_small_depth_holes(depth, max_hole_frac=0.02, fill_border=False):
    """
    @brief Fill invalid (<=0) holes in a range map with the nearest valid depth.

    MVS leaves holes where matching failed - speckle, moving objects (fish!),
    texture-poor patches. Pixels without depth get no Sea-thru correction, so
    they survive as raw hazy patches in an otherwise corrected image; a hole in
    the *middle* of a frame is especially visible and carries into every
    downstream product built from that frame.

    Two kinds of hole are treated differently:

    * **Interior holes** (fully enclosed by valid depth) are filled if smaller
      than ``max_hole_frac`` of the image. This is bounded *interpolation* -
      the surrounding real measurements cap the error - so the default is
      generous (2% of the image; a fish occluder is typically ~0.3%).
    * **Border-touching regions** (the undistortion frame and edge strips where
      the first/last survey frames lack a matching neighbour) are
      *extrapolation*: only filled when ``fill_border`` is set. In a survey
      with good overlap these edges are covered by neighbouring frames in any
      multi-view product, so the default leaves them invalid.

    @param depth (H, W) float range map; ``<= 0`` marks invalid.
    @param max_hole_frac Largest *interior* hole to fill, as a fraction of
        image area. 0 disables all filling.
    @param fill_border Also fill border-touching invalid regions (nearest-valid
        extrapolation), regardless of size.
    @return (H, W) float depth with holes filled (a copy).
    """
    if not max_hole_frac and not fill_border:
        return depth
    from scipy import ndimage

    invalid = depth <= 0
    if not invalid.any() or invalid.all():
        return depth
    labels, n = ndimage.label(invalid)
    border_ids = set(np.unique(np.concatenate(
        [labels[0], labels[-1], labels[:, 0], labels[:, -1]]))) - {0}
    sizes = ndimage.sum(np.ones_like(labels, dtype=np.float32), labels,
                        index=np.arange(1, n + 1))
    max_area = (max_hole_frac or 0) * depth.size
    fill_ids = [i for i in range(1, n + 1)
                if (i in border_ids and fill_border)
                or (i not in border_ids and sizes[i - 1] <= max_area)]
    if not fill_ids:
        return depth
    to_fill = np.isin(labels, fill_ids)
    # nearest valid pixel for every pixel (indices), then copy depth from it
    idx = ndimage.distance_transform_edt(invalid, return_distances=False,
                                         return_indices=True)
    filled = depth.copy()
    filled[to_fill] = depth[tuple(idx[:, to_fill])]
    return filled


def read_colmap_array(path):
    """
    @brief Read a COLMAP ``.bin`` dense array (depth/normal map).
    @param path Path to a ``.geometric.bin`` / ``.photometric.bin`` file.
    @return float32 array of shape ``(H, W)`` for depth maps (normal maps
        would be ``(H, W, 3)`` before the trailing ``squeeze()``).
    """
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        n = 0
        while True:
            b = fid.read(1)
            if b == b"&":
                n += 1
                if n >= 3:
                    break
        data = np.fromfile(fid, np.float32)
    arr = data.reshape((int(width), int(height), int(channels)), order="F")
    return np.transpose(arr, (1, 0, 2)).squeeze()


class ColmapDepthSource(DepthSource):
    """@brief Metric depth from a COLMAP dense-MVS workspace; see module docstring."""

    def __init__(self, workspace, kind="geometric", clip_percentile=99.5,
                 clip_low_percentile=2.0, fill_holes_max_frac=0.02,
                 fill_border=False):
        """
        @param workspace COLMAP dense workspace directory (containing ``stereo/``).
        @param kind ``"geometric"`` (recommended, multi-view consistent) or
            ``"photometric"`` depth-map variant.
        @param clip_percentile Depths above this percentile (of the valid
            pixels in this image) are marked invalid, to drop MVS outliers.
        @param clip_low_percentile Depths *below* this percentile are marked
            invalid. MVS produces spurious near-camera points (e.g. 0.2 m on a
            reef imaged from 3 m). These matter far more than they look: the
            coarse attenuation estimate is ``beta = -ln(illuminant) / z``, so a
            tiny ``z`` explodes beta, and ``_spread_samples`` gives each range
            window equal weight in the two-term fit — so a handful of junk
            near-range pixels can dominate it and force beta_D(z) to decay when
            it should rise. Set to 0 to disable.
        @param fill_holes_max_frac Fill *interior* invalid holes smaller than
            this fraction of image area with the nearest valid depth (see
            ``fill_small_depth_holes``), so MVS speckle and occluder holes
            (fish!) don't survive as untreated raw-colour patches mid-frame.
            Set to 0 to disable.
        @param fill_border Also fill border-touching invalid regions by
            nearest-valid extrapolation (see ``fill_small_depth_holes``).
        """
        self.depth_dir = Path(workspace) / "stereo" / "depth_maps"
        self.kind = kind
        self.clip_percentile = clip_percentile
        self.clip_low_percentile = clip_low_percentile
        self.fill_holes_max_frac = fill_holes_max_frac
        self.fill_border = fill_border

    def _find(self, image_name):
        """@brief Locate the COLMAP depth-map file for one image.
        @param image_name Source image file name.
        @return Path to the matching ``.bin`` file, or None if not found."""
        for name in (f"{image_name}.{self.kind}.bin",
                     f"{Path(image_name).stem}.{self.kind}.bin"):
            cand = self.depth_dir / name
            if cand.exists():
                return cand
        return None

    def get_depth(self, img, meta: ImageMeta):
        """@brief @copydoc DepthSource.get_depth
        @throws FileNotFoundError if no matching COLMAP depth map exists."""
        path = self._find(meta.image_name)
        if path is None:
            raise FileNotFoundError(
                f"No COLMAP depth map for {meta.image_name} in {self.depth_dir}")
        depth = read_colmap_array(path).astype(np.float32)

        H, W = img.shape[:2]
        if depth.shape != (H, W):
            # np.array (not asarray): PIL's buffer is read-only, and the
            # normalisation below writes in place, so force a writable copy.
            depth = np.array(
                Image.fromarray(depth).resize((W, H), Image.NEAREST),
                dtype=np.float32)
        else:
            depth = np.ascontiguousarray(depth, dtype=np.float32)

        depth[~np.isfinite(depth)] = 0.0
        depth[depth <= 0] = 0.0
        if self.clip_percentile and np.any(depth > 0):
            hi = np.percentile(depth[depth > 0], self.clip_percentile)
            depth[depth > hi] = 0.0
        if self.clip_low_percentile and np.any(depth > 0):
            lo = np.percentile(depth[depth > 0], self.clip_low_percentile)
            depth[(depth > 0) & (depth < lo)] = 0.0
        if self.fill_holes_max_frac or self.fill_border:
            depth = fill_small_depth_holes(depth, self.fill_holes_max_frac,
                                           fill_border=self.fill_border)
        return depth
