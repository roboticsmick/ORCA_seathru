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


def remove_valid_islands(depth, min_island_frac=0.01, erosion_px=5):
    """
    @brief Invalidate untrustworthy *valid* speckles/blobs (MVS noise) so hole
    fills can only propagate from large, coherent regions.

    Where MVS fails over a region (motion blur, texture-poor patch), the failed
    zone is rarely 100% invalid — it contains blobs of "valid" pure-noise depth
    that survive percentile clipping (that clip is global, not spatial). If
    holes are filled by nearest-valid propagation, those in-zone junk blobs win
    over the trustworthy rim and the zone inherits garbage depth (observed: a
    bright motion-blurred band assigned 3.8–7.2 m against a 2.5 m rim → strong
    false red band after correction).

    Plain connected-component size filtering is NOT sufficient: the junk blobs
    are usually attached to the main valid region by thin bridges, making them
    part of one huge component. So instead:

      1. **Erode** the valid mask by ``erosion_px`` — thin bridges and small
         blobs lose their core; large coherent regions keep one.
      2. Keep only eroded cores bigger than ``min_island_frac`` of the image.
      3. Trust exactly the valid pixels within ``erosion_px + 2`` of a kept
         core (restores the eroded boundary of genuine regions without
         re-growing through bridges to the junk).

    @param depth (H, W) float range map; ``<= 0`` marks invalid.
    @param min_island_frac Minimum eroded-core size, as a fraction of image
        area, for a region to be trusted. 0 disables the filter entirely.
    @param erosion_px Erosion radius in pixels (bridge-cutting scale).
    @return (H, W) float depth with untrusted valid pixels zeroed (a copy).
    """
    if not min_island_frac:
        return depth
    from scipy import ndimage

    valid = depth > 0
    if valid.all() or not valid.any():
        return depth
    core = ndimage.binary_erosion(valid, iterations=erosion_px)
    labels, n = ndimage.label(core)
    if n == 0:
        return depth                     # nothing survives erosion: keep as-is
    sizes = ndimage.sum(np.ones_like(labels, dtype=np.float32), labels,
                        index=np.arange(1, n + 1))
    big = np.nonzero(sizes >= min_island_frac * depth.size)[0] + 1
    if big.size == 0:
        return depth
    keep_core = np.isin(labels, big)
    dist = ndimage.distance_transform_edt(~keep_core)
    trusted = valid & (dist <= erosion_px + 2)
    if trusted.sum() == valid.sum():
        return depth
    out = depth.copy()
    out[valid & ~trusted] = 0.0
    return out


class ColmapDepthSource(DepthSource):
    """@brief Metric depth from a COLMAP dense-MVS workspace; see module docstring."""

    def __init__(self, workspace, kind="geometric", clip_percentile=99.5,
                 clip_low_percentile=2.0, fill_holes_max_frac=0.02,
                 fill_border=False, min_island_frac=0.01,
                 fill_mono=False, mono_backend="midas"):
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
        @param min_island_frac Invalidate isolated "valid" components smaller
            than this fraction of image area before filling (MVS noise
            speckles inside failed zones — see ``remove_valid_islands``).
        @param fill_mono Patch invalid regions with **monocular neural depth**
            aligned per-image to the valid COLMAP pixels (needs torch; see
            ``_mono_fill``). Much better than nearest-valid extrapolation for
            large holes, where the true surface is not a continuation of the
            rim. When enabled it replaces the nearest-valid fill entirely
            (which remains the fallback if the alignment fails).
        @param mono_backend Backend for ``fill_mono`` ("midas" or
            "depth_anything_v2"), see seathru.depth.monocular.
        """
        self.depth_dir = Path(workspace) / "stereo" / "depth_maps"
        self.kind = kind
        self.clip_percentile = clip_percentile
        self.clip_low_percentile = clip_low_percentile
        self.fill_holes_max_frac = fill_holes_max_frac
        self.fill_border = fill_border
        self.min_island_frac = min_island_frac
        self.fill_mono = fill_mono
        self.mono_backend = mono_backend
        self._mono = None          # lazy MonocularDepthSource
        self.last_notes = []       # per-frame processing notes (for run logs)

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

    def _mono_fill(self, img, depth):
        """
        @brief Patch invalid depth with monocular depth aligned to COLMAP.

        Monocular networks predict depth up to an unknown affine transform in
        *inverse* depth. With ~90%+ of the frame carrying metric COLMAP depth,
        that transform is over-determined: fit ``1/z_colmap ≈ s·d_mono + t``
        by least squares over the valid overlap (one trimming pass drops the
        worst 20% residuals — object boundaries and MVS noise), then evaluate
        the aligned mono depth in the holes only. No training required — the
        survey's own depth is the per-image supervision.

        @param img (H, W, 3) float image in [0, 1] (working resolution).
        @param depth (H, W) float COLMAP depth, ``<= 0`` invalid.
        @return (filled_depth, note_string) — depth unchanged (and a reason in
            the note) if torch/the model is unavailable or the fit degenerates.
        """
        hole = depth <= 0
        if not hole.any():
            return depth, None
        try:
            if self._mono is None:
                from .monocular import MonocularDepthSource
                self._mono = MonocularDepthSource(backend=self.mono_backend)
            rel = self._mono._infer_relative(img).astype(np.float32)
        except Exception as err:  # torch missing, hub download failed, OOM...
            return depth, f"mono-fill UNAVAILABLE ({type(err).__name__}); nearest-fill fallback"

        inv_mono = 1.0 / np.maximum(rel, 1e-6)
        valid = depth > 0
        x = inv_mono[valid].ravel()
        y = (1.0 / depth[valid]).ravel()
        if x.size > 20000:                      # subsample for speed
            idx = np.random.default_rng(0).choice(x.size, 20000, replace=False)
            x, y = x[idx], y[idx]
        for _ in range(2):                      # LSQ with one trimming pass
            A = np.c_[x, np.ones_like(x)]
            (s, t), *_ = np.linalg.lstsq(A, y, rcond=None)
            resid = np.abs(A @ np.array([s, t]) - y)
            keep = resid < np.percentile(resid, 80)
            if keep.sum() < 100:
                return depth, "mono-fill DEGENERATE fit; nearest-fill fallback"
            x, y = x[keep], y[keep]
        if s <= 0:
            return depth, "mono-fill NEGATIVE scale; nearest-fill fallback"
        aligned = 1.0 / np.clip(s * inv_mono + t, 1e-6, None)
        # residual sanity on the (trimmed) overlap, in metres
        err_m = float(np.median(np.abs(1.0 / np.clip(s * x + t, 1e-6, None) - 1.0 / y)))
        # clamp holes to a plausible band so a bad mono region cannot explode
        v = depth[valid]
        lo_b, hi_b = 0.5 * np.percentile(v, 2), 2.0 * np.percentile(v, 98)
        out = depth.copy()
        out[hole] = np.clip(aligned[hole], lo_b, hi_b)
        return out, f"mono-filled {100 * hole.mean():.1f}% (align err {err_m:.2f} m)"

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

        self.last_notes = []
        depth[~np.isfinite(depth)] = 0.0
        depth[depth <= 0] = 0.0
        if self.clip_percentile and np.any(depth > 0):
            hi = np.percentile(depth[depth > 0], self.clip_percentile)
            depth[depth > hi] = 0.0
        if self.clip_low_percentile and np.any(depth > 0):
            lo = np.percentile(depth[depth > 0], self.clip_low_percentile)
            depth[(depth > 0) & (depth < lo)] = 0.0
        if self.min_island_frac:
            before = (depth > 0).mean()
            depth = remove_valid_islands(depth, self.min_island_frac)
            removed = before - (depth > 0).mean()
            if removed > 0.001:
                self.last_notes.append(f"islands-removed {100 * removed:.1f}%")
        if self.fill_mono:
            depth, note = self._mono_fill(img, depth)
            if note:
                self.last_notes.append(note)
        if (depth <= 0).any() and (self.fill_holes_max_frac or self.fill_border):
            depth = fill_small_depth_holes(depth, self.fill_holes_max_frac,
                                           fill_border=self.fill_border)
        return depth
