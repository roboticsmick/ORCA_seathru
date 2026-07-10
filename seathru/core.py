"""
@file core.py
@brief Core Sea-thru algorithm (Akkaynak & Treibitz, CVPR 2019).

A clean, from-scratch reimplementation of the method described in
"Sea-thru: A Method for Removing Water From Underwater Images"
(Derya Akkaynak & Tali Treibitz, CVPR 2019).

The pipeline recovers the unattenuated scene J from a captured RGB image I and
a per-pixel range map z (metres), using the revised underwater image
formation model:

    I_c = J_c * exp(-beta_D_c(z) * z) + B_inf_c * (1 - exp(-beta_B_c * z))
          \\___________ direct signal ___________/   \\_____ backscatter _____/

Equation numbers in the docstrings refer to the paper. This module is pure
NumPy/SciPy/scikit-image and has no deep-learning dependency; the range map is
supplied by the caller (see seathru.depth).

This file also implements "survey-locked" processing: instead of fitting the
backscatter/attenuation/white-balance statistics independently on every
frame (the paper's per-image design), a SurveyStats bundle fits them once
from a handful of representative frames and freezes them across an entire
photo survey. This trades a small amount of per-image adaptivity for
frame-to-frame radiometric consistency, which matters for downstream
orthomosaic blending and Gaussian-splat/NeRF training. See seathru.survey
for the calibration entry point (seathru.survey.calibrate_survey_stats) and
the README's "Survey-locked mode" section for usage.

@author Michael Venz
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.optimize
import scipy.stats
from scipy import ndimage
from skimage.morphology import closing, disk
from skimage.restoration import denoise_bilateral

## Numerical floor used throughout to avoid divide-by-zero in ratios/logs.
EPS = 1e-8


# --------------------------------------------------------------------------- #
# 4.3  Backscatter estimation                                                 #
# --------------------------------------------------------------------------- #
def find_backscatter_points(img, depths, num_bins=10, fraction=0.01, max_vals=20):
    """
    @brief Section 4.3: collect candidate dark pixels for backscatter estimation.

    The range map is split into ``num_bins`` evenly spaced clusters. Inside each
    cluster the darkest ``fraction`` of RGB *triplets* (ranked by mean intensity,
    not per channel) are kept, capped at ``max_vals`` per cluster. These pixels
    are assumed to be scene points with negligible direct signal, so their
    residual intensity is (almost) pure backscatter.

    @param img (H, W, 3) float array in [0, 1].
    @param depths (H, W) float array of range in metres; <= 0 marks invalid pixels.
    @param num_bins Number of evenly spaced range bins spanning the valid depth range.
    @param fraction Fraction of pixels (by darkness rank) kept per bin.
    @param max_vals Hard cap on candidate pixels kept per bin.
    @return Tuple of three ``(N, 2)`` arrays ``(depth, value)`` for the R, G, B channels.
    """
    valid = depths > 0
    z_min, z_max = depths[valid].min(), depths[valid].max()
    z_ranges = np.linspace(z_min, z_max, num_bins + 1)
    img_norms = np.mean(img, axis=2)
    pts = {0: [], 1: [], 2: []}
    for i in range(num_bins):
        a, b = z_ranges[i], z_ranges[i + 1]
        locs = np.where(valid & (depths >= a) & (depths <= b))
        if locs[0].size == 0:
            continue
        norms, px, dz = img_norms[locs], img[locs], depths[locs]
        order = np.argsort(norms)
        n_take = min(int(np.ceil(fraction * order.size)), max_vals)
        sel = order[:n_take]
        for c in (0, 1, 2):
            pts[c].extend(zip(dz[sel], px[sel, c]))
    return (np.array(pts[0]), np.array(pts[1]), np.array(pts[2]))


def _backscatter_model(z, B_inf, beta_B, J_prime, beta_D_prime):
    """@brief Equation 10: saturating veiling term + small residual direct term.
    @param z Range in metres (scalar or array).
    @param B_inf Asymptotic backscatter (veiling-light) level.
    @param beta_B Backscatter attenuation coefficient.
    @param J_prime Residual direct-signal amplitude of the dark-pixel set.
    @param beta_D_prime Residual direct-signal attenuation coefficient.
    @return Model backscatter value(s), same shape as ``z``.
    """
    return (B_inf * (1.0 - np.exp(-beta_B * z))
            + J_prime * np.exp(-beta_D_prime * z))


def estimate_backscatter(points, depths, restarts=25, max_mean_loss_fraction=0.1):
    """
    @brief Fit Equation 10 to the dark-pixel set and evaluate over the full range map.

    Bounds follow the paper: B_inf in [0, 1], beta_B in [0, 5], J' in [0, 1],
    beta_D' in [0, 5]. Multiple random restarts are used because the fit is
    non-convex. Falls back to a linear model if no good fit is found.

    @param points (N, 2) array of (depth, value) dark-pixel samples for one channel,
        as returned by find_backscatter_points.
    @param depths (H, W) float range map in metres.
    @param restarts Number of random-initialisation curve-fit attempts.
    @param max_mean_loss_fraction Fraction of the depth span used as the maximum
        acceptable mean fit residual before falling back to a linear model.
    @return Tuple ``(B_channel, coefs)`` where ``B_channel`` is the per-pixel
        backscatter for this channel (same shape as ``depths``), and ``coefs`` is
        either the 4-parameter Equation 10 fit or a 2-parameter ``[slope, intercept]``
        linear fallback.
    """
    z_data, v_data = points[:, 0], points[:, 1]
    valid = depths > 0
    z_min, z_max = depths[valid].min(), depths[valid].max()
    max_mean_loss = max_mean_loss_fraction * (z_max - z_min)
    lower, upper = [0, 0, 0, 0], [1, 5, 1, 5]

    best_coefs, best_loss = None, np.inf
    for _ in range(restarts):
        try:
            coefs, _ = scipy.optimize.curve_fit(
                _backscatter_model, z_data, v_data,
                p0=np.random.random(4) * upper,
                bounds=(lower, upper), maxfev=2000,
            )
        except (RuntimeError, ValueError) as err:
            print(err, file=sys.stderr)
            continue
        loss = np.mean(np.abs(v_data - _backscatter_model(z_data, *coefs)))
        if loss < best_loss:
            best_loss, best_coefs = loss, coefs

    if best_coefs is None or best_loss > max_mean_loss:
        # Degenerate scene: fall back to a robust linear backscatter model.
        slope, intercept, *_ = scipy.stats.linregress(z_data, v_data)
        B = np.clip(slope * depths + intercept, 0, None)
        return B, np.array([slope, intercept])

    B = _backscatter_model(depths, *best_coefs)
    return B, best_coefs


# --------------------------------------------------------------------------- #
# 4.4.2  Neighbourhood map (Equation 15) and illuminant (Equations 13-14)     #
# --------------------------------------------------------------------------- #
def construct_neighborhood_map(depths, epsilon=0.05, min_size=50):
    """
    @brief Equation 15: partition the image into iso-range neighbourhoods.

    Pixels are grouped into connected components whose range falls in the same
    ``epsilon``-wide band. Components smaller than ``min_size`` are merged into
    the nearest surviving neighbourhood. Invalid pixels (range <= 0) get label 0.

    This is a fast, vectorised approximation of the flood-fill grouping used to
    define the support region for local-space-average-colour illuminant
    estimation.

    @param depths (H, W) float range map in metres; <= 0 marks invalid pixels.
    @param epsilon Range-band width as a fraction of the full depth span (Eq. 15).
    @param min_size Minimum pixel count for a neighbourhood to survive un-merged.
    @return Tuple ``(nmap, num_neighborhoods)``: an ``(H, W)`` int32 label map
        (0 = invalid/background) and the number of surviving labels.
    """
    valid = depths > 0
    if not np.any(valid):
        return np.zeros_like(depths, dtype=np.int32), 0
    z_min, z_max = depths[valid].min(), depths[valid].max()
    band = max((z_max - z_min) * epsilon, EPS)

    quant = np.zeros_like(depths, dtype=np.int64)
    quant[valid] = np.floor((depths[valid] - z_min) / band).astype(np.int64) + 1

    nmap = np.zeros_like(depths, dtype=np.int32)
    count = 0
    for level in np.unique(quant[valid]):
        lab, n = ndimage.label(quant == level)
        lab[lab > 0] += count
        nmap = np.where(quant == level, lab, nmap)
        count += n

    return _refine_neighborhood_map(nmap, min_size)


def _refine_neighborhood_map(nmap, min_size):
    """@brief Drop tiny neighbourhoods, reassign their pixels to the nearest
    survivor, and relabel contiguously starting at 1.
    @param nmap (H, W) int label map (0 = background).
    @param min_size Minimum pixel count for a label to survive.
    @return Tuple ``(nmap, num_labels)`` with small labels merged away.
    """
    labels, counts = np.unique(nmap, return_counts=True)
    keep = {int(l) for l, c in zip(labels, counts) if l != 0 and c >= min_size}
    if not keep:
        keep = {int(labels[np.argmax(counts * (labels != 0))])}

    survivor = np.isin(nmap, list(keep))
    # Nearest-survivor assignment for every foreground pixel.
    _, (iy, ix) = ndimage.distance_transform_edt(~survivor, return_indices=True)
    filled = nmap[iy, ix]
    filled[nmap == 0] = 0  # keep background as background

    remap = {old: new for new, old in enumerate(sorted(keep), start=1)}
    out = np.zeros_like(nmap)
    for old, new in remap.items():
        out[filled == old] = new
    return out, len(remap)


def estimate_illumination(channel, B, nmap, num_neighborhoods,
                          p=0.5, f=2.0, max_iters=100, tol=1e-4):
    """
    @brief Equations 13-14: local-space-average-colour illuminant for one channel.

    The direct signal ``D = channel - B`` is diffused within each iso-range
    neighbourhood. ``a'`` is iteratively updated as a convex combination of the
    per-pixel direct signal (weight ``p``) and the neighbourhood average of
    ``a'`` excluding the pixel itself (weight ``1 - p``). The illuminant is the
    converged average scaled by geometry factor ``f`` (paper uses 2 for a
    perpendicular camera-scene orientation), lightly denoised.

    @param channel (H, W) float array, one colour channel of the input image.
    @param B (H, W) float backscatter for this channel.
    @param nmap (H, W) int neighbourhood label map from construct_neighborhood_map.
    @param num_neighborhoods Number of labels in ``nmap`` (excluding background).
    @param p Illuminant locality: weight on the per-pixel direct signal vs. the
        neighbourhood average (Eq. 14 support weight). Tunable, see SeathruParams.p.
    @param f Illuminant geometry factor (Section 4.4.2). Tunable, see SeathruParams.f.
    @param max_iters Maximum diffusion iterations.
    @param tol Convergence tolerance on the maximum per-pixel update.
    @return (H, W) float illuminant map for this channel.
    """
    D = np.clip(channel - B, 0, None)
    flat_labels = nmap.ravel()
    flat_D = D.ravel()
    L = num_neighborhoods + 1
    counts = np.bincount(flat_labels, minlength=L).astype(np.float64)
    counts[0] = 0.0  # background contributes nothing

    avg = np.zeros_like(flat_D)
    fg = flat_labels > 0
    denom = np.maximum(counts[flat_labels] - 1.0, 1.0)
    for _ in range(max_iters):
        sums = np.bincount(flat_labels, weights=avg, minlength=L)
        neigh_mean = (sums[flat_labels] - avg) / denom
        new_avg = np.where(fg, p * flat_D + (1.0 - p) * neigh_mean, 0.0)
        if np.max(np.abs(new_avg - avg)) < tol:
            avg = new_avg
            break
        avg = new_avg

    illum = f * denoise_bilateral(np.clip(avg.reshape(D.shape), 0, None),
                                  sigma_spatial=3)
    return illum


# --------------------------------------------------------------------------- #
# 4.4.1 / 4.4.3  Wideband attenuation coefficient beta_D(z)                    #
# --------------------------------------------------------------------------- #
def _beta_D_two_term(z, a, b, c, d):
    """@brief Equation 11: direct attenuation coefficient as a two-term exponential.
    @param z Range in metres (scalar or array).
    @param a, b, c, d Fitted coefficients.
    @return beta_D(z), same shape as ``z``.
    """
    return a * np.exp(b * z) + c * np.exp(d * z)


def coarse_attenuation(depths, illum, max_val=10.0):
    """@brief Equation 12: coarse per-pixel beta_D from the illuminant map.
    @param depths (H, W) float range map in metres.
    @param illum (H, W) float illuminant map for one channel.
    @param max_val Clamp on the coarse coefficient (numerical stability).
    @return (H, W) float coarse beta_D estimate for this channel.
    """
    beta = np.minimum(max_val, -np.log(illum + EPS) / (np.maximum(0, depths) + EPS))
    mask = (depths > EPS) & (illum > EPS)
    return closing(np.clip(beta * mask, 0, None), disk(6))


def _spread_samples(x, y, radius_fraction=0.01):
    """@brief Thin dense samples so the two-term fit is not dominated by one range,
    keeping the median y within each range window (Hodges' ``filter_data``).
    @param x, y Sample arrays of equal length (range, coarse beta_D).
    @param radius_fraction Window width as a fraction of the x-span.
    @return Tuple of thinned ``(x, y)`` arrays.
    """
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    radius = radius_fraction * (xs.max() - xs.min() + EPS)
    out_x, out_y, buf_x, buf_y = [xs[0]], [ys[0]], [], []
    anchor = xs[0]
    for xi, yi in zip(xs[1:], ys[1:]):
        buf_x.append(xi); buf_y.append(yi)
        if xi - anchor >= radius:
            m = len(buf_y) // 2
            idx = np.argsort(buf_y)[m]
            out_x.append(buf_x[idx]); out_y.append(buf_y[idx])
            anchor = xi
    return np.array(out_x), np.array(out_y)


def refine_attenuation(depths, illum, coarse, restarts=10,
                       min_depth_fraction=0.1, l=1.0, spread_fraction=0.01):
    """
    @brief Equations 16-17: fit the two-term-exponential beta_D(z) so the implied
    range matches the known range map.

    @param depths (H, W) float range map in metres.
    @param illum (H, W) float illuminant map for this channel.
    @param coarse (H, W) float coarse beta_D from coarse_attenuation.
    @param restarts Number of random-initialisation curve-fit attempts.
    @param min_depth_fraction Fraction of the depth span excluded near the camera
        (avoids fitting the unstable near-range region).
    @param l Attenuation/brightness balance knob (tunable, see SeathruParams.l);
        scales the final coefficient, matching the reference implementation.
    @param spread_fraction Sample-thinning window, see _spread_samples.
    @return Tuple ``(beta_D_map, coefs)``. ``coefs`` is the fitted 4-parameter
        Equation 11 coefficients, or ``None`` if the fit is skipped/degenerate
        (fewer than 8 usable samples) or falls back to a 2-parameter linear fit.
    """
    valid = depths > 0
    z_min, z_max = depths[valid].min(), depths[valid].max()
    min_depth = z_min + min_depth_fraction * (z_max - z_min)
    locs = np.where((illum > 0) & (depths > min_depth) & (coarse > EPS))
    if locs[0].size < 8:
        return l * coarse, None

    def implied_range(z, il, a, b, c, d):
        return -np.log(il + 1e-5) / (_beta_D_two_term(z, a, b, c, d) + 1e-5)

    def loss(coefs):
        return np.mean(np.abs(depths[locs] - implied_range(depths[locs],
                                                           illum[locs], *coefs)))

    dx, dy = _spread_samples(depths[locs], coarse[locs], spread_fraction)
    lower, upper = [0, -100, 0, -100], [100, 0, 100, 0]
    best_coefs, best_loss = None, np.inf
    for _ in range(restarts):
        try:
            coefs, _ = scipy.optimize.curve_fit(
                _beta_D_two_term, dx, dy,
                p0=np.abs(np.random.random(4)) * np.array([1., -1., 1., -1.]),
                bounds=(lower, upper), maxfev=2000,
            )
        except (RuntimeError, ValueError):
            continue
        L = loss(coefs)
        if L < best_loss:
            best_loss, best_coefs = L, coefs

    if best_coefs is None:
        slope, intercept, *_ = scipy.stats.linregress(depths[locs], coarse[locs])
        return l * (slope * depths + intercept), None
    return l * _beta_D_two_term(depths, *best_coefs), best_coefs


# --------------------------------------------------------------------------- #
# 4.2  Scene reconstruction and white balance                                 #
# --------------------------------------------------------------------------- #
def _compute_wb_gains(img, protect_red=True):
    """
    @brief Equation 9: per-channel Gray-World white-balance gains.

    ``protect_red`` balances only green/blue and gently lifts red, which avoids
    the pink cast that pure gray-world produces on red-starved deep imagery.

    @param img (H, W, 3) float array the gains are computed from (typically the
        post-backscatter-removal "direct" image, background pixels zeroed).
    @param protect_red Use the red-protected variant (SeathruParams.protect_red).
    @return (3,) float array of multiplicative gains ``[gain_r, gain_g, gain_b]``.
    """
    if protect_red:
        dg, db = 1.0 / (img[..., 1].mean() + EPS), 1.0 / (img[..., 2].mean() + EPS)
        s = dg + db
        dg, db = dg / s * 2.0, db / s * 2.0
        return np.array([(dg + db) / 2.0, dg, db])
    d = 1.0 / (img.reshape(-1, 3).mean(0) + EPS)
    return d / d.sum() * 3.0


def _apply_wb_gains(img, gains):
    """@brief Apply precomputed per-channel gains (see _compute_wb_gains).
    @param img (H, W, 3) float array.
    @param gains (3,) float array of multiplicative gains.
    @return (H, W, 3) gain-scaled array.
    """
    return img * gains


def _scale01(img):
    """@brief Min-max normalise an array to [0, 1] for visualisation only."""
    lo, hi = np.min(img), np.max(img)
    return (img - lo) / (hi - lo + EPS)


def _robust_stretch(img, fg_mask, pct=(0.5, 99.5), bounds=None):
    """
    @brief Percentile-based contrast stretch over foreground pixels only.

    A raw min/max stretch lets a handful of saturated pixels (sun glints,
    caustic highlights - common in shallow reef frames) compress the whole
    tonal range. Clipping to the ``pct`` percentiles is robust to them.

    @param img (H, W, 3) float array to stretch.
    @param fg_mask Boolean foreground mask, or ``None`` to use every pixel.
    @param pct Tuple of (low, high) percentiles used when ``bounds`` is ``None``.
    @param bounds Optional explicit ``(lo, hi)`` values that override ``pct``;
        used by survey-locked mode (SurveyStats.stretch_bounds) so every frame
        is stretched to the same absolute range instead of its own percentiles.
    @return (H, W, 3) array stretched to [0, 1] (clipped).
    """
    if bounds is not None:
        lo, hi = bounds
    else:
        vals = img[fg_mask] if fg_mask is not None else img
        lo, hi = np.percentile(vals, pct)
    return np.clip((img - lo) / (hi - lo + EPS), 0.0, 1.0)


def _compute_direct(img, depths, B, beta_D, nmap):
    """@brief Undo backscatter and range attenuation (Eq. 8), zeroing background.
    @param img (H, W, 3) float input image in [0, 1].
    @param depths (H, W) float range map in metres.
    @param B (H, W, 3) float per-pixel backscatter.
    @param beta_D (H, W, 3) float per-pixel direct attenuation coefficient.
    @param nmap (H, W) int neighbourhood label map (0 = invalid/background).
    @return (H, W, 3) float "direct signal" image, clipped to [0, 1], with
        background pixels set to 0.
    """
    direct = (img - B) * np.exp(beta_D * depths[..., None])
    direct = np.clip(direct, 0.0, 1.0)
    direct[nmap == 0] = 0.0
    return direct


def recover_image(img, depths, B, beta_D, nmap, protect_red=True,
                  stretch_pct=(0.5, 99.5), wb_gains=None, stretch_bounds=None):
    """
    @brief Equation 8 + Equation 9: remove backscatter, undo range attenuation, then
    white balance. Invalid pixels keep their original colour.

    @param img (H, W, 3) float input image in [0, 1].
    @param depths (H, W) float range map in metres.
    @param B (H, W, 3) float per-pixel backscatter.
    @param beta_D (H, W, 3) float per-pixel direct attenuation coefficient.
    @param nmap (H, W) int neighbourhood label map (0 = invalid/background).
    @param protect_red Use the red-protected white-balance variant.
    @param stretch_pct Percentiles used for the output contrast stretch when
        ``stretch_bounds`` is not supplied.
    @param wb_gains Optional precomputed (3,) white-balance gains; when given,
        the per-image Gray-World fit is skipped (survey-locked mode).
    @param stretch_bounds Optional precomputed ``(lo, hi)`` stretch bounds; when
        given, the per-image percentile stretch is skipped (survey-locked mode).
    @return (H, W, 3) float recovered image in [0, 1].
    """
    direct = _compute_direct(img, depths, B, beta_D, nmap)
    bg = nmap == 0
    gains = wb_gains if wb_gains is not None else _compute_wb_gains(direct, protect_red)
    out = _robust_stretch(_apply_wb_gains(direct, gains), ~bg, stretch_pct, stretch_bounds)
    out[bg] = img[bg]
    return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# CONFIG - tunable parameters                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class SeathruParams:
    """
    @brief All tunable knobs for one Sea-thru run. This is the single config
    surface for the library - every parameter that affects the algorithm's
    output lives here (mirrored 1:1 by seathru.cli's command-line flags).

    Grouped by what they control:

    **Recovery strength / look** (only active with a spatially-varying range
    map - SfM, monocular, or the image-derived prior; a flat plane leaves
    these inert, see seathru.depth.PlaneDepthSource):
    @param p Illuminant locality (Eq. 14 support weight), in [0, 1]. Higher
        trusts the local pixel more over its neighbourhood average.
    @param f Illuminant geometry factor (Section 4.4.2); the paper uses 2 for a
        perpendicular camera-scene orientation. Raises overall brightness.
    @param l Attenuation/brightness balance; the main strength dial for the
        range-dependent correction. Lower if far/deep areas over-brighten.
    @param epsilon Iso-range neighbourhood band width (Eq. 15), as a fraction
        of the scene's depth span.

    **Output tone**:
    @param protect_red White-balance red gently instead of pure gray-world
        (avoids a pink cast on red-starved deep frames).
    @param stretch_pct Robust output-stretch percentiles ``(lo, hi)``; widen
        toward ``(0.1, 99.9)`` for a flatter, safer result, tighten for more
        contrast/punch.

    **Fit robustness / cost** (raise for a harder scene, lower for speed):
    @param min_neighborhood Minimum neighbourhood size before it is merged
        into its nearest survivor (Eq. 15).
    @param backscatter_restarts Random-restart count for the Eq. 10 backscatter
        fit. Dominates per-image runtime; see the README timing table.
    @param attenuation_restarts Random-restart count for the Eq. 11 attenuation
        fit. Also runtime-dominant.
    @param spread_fraction Sample-thinning window for the attenuation fit
        (see core._spread_samples).

    **Survey-locked mode** (see seathru.survey and the README):
    @param locked_stats Optional SurveyStats. When set, backscatter/attenuation
        coefficients and white-balance gains are taken from here instead of
        being fit per image; also skips the fit entirely (faster). Leave
        ``None`` for the paper's original per-image adaptive behaviour.

    **Debug**:
    @param return_debug When True, SeathruResult carries the intermediate maps
        (backscatter, illuminant, beta_D, neighbourhood map) for --debug montages.
    """
    p: float = 0.5
    f: float = 2.0
    l: float = 1.0
    epsilon: float = 0.05
    protect_red: bool = True
    stretch_pct: tuple = (0.5, 99.5)
    min_neighborhood: int = 50
    backscatter_restarts: int = 25
    attenuation_restarts: int = 10
    spread_fraction: float = 0.01
    locked_stats: "SurveyStats | None" = None
    return_debug: bool = False


@dataclass
class SurveyStats:
    """
    @brief Frozen per-survey radiometric statistics for "survey-locked" mode.

    Produced by seathru.survey.calibrate_survey_stats from a representative
    sample of a survey's images, then reused for every frame in the batch so
    the backscatter colour, water attenuation, and white balance stay
    consistent across the whole dataset (important for orthomosaic blending
    and NeRF/3DGS training, where per-frame colour drift shows up as visible
    seams or view-dependent colour artefacts).

    Any field may be ``None`` if calibration could not fit it reliably (e.g.
    a constant/plane depth map cannot support a spatial backscatter fit) - in
    that case seathru.core.run_seathru falls back to per-image adaptive
    fitting for just that piece.

    @param backscatter_coefs (3, 4) float array, one Equation 10 coefficient
        row ``[B_inf, beta_B, J', beta_D']`` per RGB channel, or ``None``.
    @param attenuation_coefs (3, 4) float array, one Equation 11 coefficient
        row ``[a, b, c, d]`` per RGB channel (pre- ``l`` scaling), or ``None``.
    @param wb_gains (3,) float array of white-balance gains ``[gain_r, gain_g,
        gain_b]`` (Equation 9), or ``None``.
    @param stretch_bounds Optional ``(lo, hi)`` output contrast-stretch bounds.
        Only set when calibration was run with exposure locking enabled
        (``--lock-exposure``); otherwise each frame keeps its own percentile
        stretch even in survey-locked mode, since scene brightness legitimately
        varies with altitude/sun angle.
    @param n_calibration_images Number of sample frames actually used.
    @param source_images File names of the sampled calibration frames (for
        provenance / debugging).
    """
    backscatter_coefs: "np.ndarray | None" = None
    attenuation_coefs: "np.ndarray | None" = None
    wb_gains: "np.ndarray | None" = None
    stretch_bounds: "tuple | None" = None
    n_calibration_images: int = 0
    source_images: list = field(default_factory=list)

    def to_dict(self):
        """@brief Convert to a JSON-serialisable plain dict.
        @return dict with numpy arrays converted to nested lists."""
        return {
            "backscatter_coefs": (self.backscatter_coefs.tolist()
                                  if self.backscatter_coefs is not None else None),
            "attenuation_coefs": (self.attenuation_coefs.tolist()
                                  if self.attenuation_coefs is not None else None),
            "wb_gains": self.wb_gains.tolist() if self.wb_gains is not None else None,
            "stretch_bounds": (list(self.stretch_bounds)
                               if self.stretch_bounds is not None else None),
            "n_calibration_images": self.n_calibration_images,
            "source_images": self.source_images,
        }

    @classmethod
    def from_dict(cls, d):
        """@brief Inverse of to_dict.
        @param d dict as produced by to_dict / json.load.
        @return SurveyStats instance."""
        return cls(
            backscatter_coefs=(np.array(d["backscatter_coefs"])
                               if d.get("backscatter_coefs") is not None else None),
            attenuation_coefs=(np.array(d["attenuation_coefs"])
                               if d.get("attenuation_coefs") is not None else None),
            wb_gains=(np.array(d["wb_gains"]) if d.get("wb_gains") is not None else None),
            stretch_bounds=(tuple(d["stretch_bounds"])
                            if d.get("stretch_bounds") is not None else None),
            n_calibration_images=d.get("n_calibration_images", 0),
            source_images=d.get("source_images", []),
        )

    def save(self, path):
        """@brief Write this SurveyStats to a JSON file.
        @param path Destination file path (str or Path)."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path):
        """@brief Read a SurveyStats previously written by save().
        @param path Source file path (str or Path).
        @return SurveyStats instance."""
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass
class SeathruResult:
    """
    @brief Output of one run_seathru call.
    @param recovered (H, W, 3) float recovered image in [0, 1]. Always present.
    @param backscatter (H, W, 3) float per-pixel backscatter B. Only populated
        when ``SeathruParams.return_debug`` is True.
    @param illuminant (H, W, 3) float illuminant map. Only populated in debug
        mode, and only computed at all when needed (see run_seathru).
    @param beta_D (H, W, 3) float direct attenuation coefficient map. Debug only.
    @param neighborhood_map (H, W) int iso-range neighbourhood labels. Debug only.
    """
    recovered: np.ndarray
    backscatter: np.ndarray = field(default=None, repr=False)
    illuminant: np.ndarray = field(default=None, repr=False)
    beta_D: np.ndarray = field(default=None, repr=False)
    neighborhood_map: np.ndarray = field(default=None, repr=False)


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def run_seathru(img, depths, params: SeathruParams | None = None) -> SeathruResult:
    """
    @brief Run the full Sea-thru pipeline on one linear-ish RGB image and range map.

    With ``params.locked_stats`` unset, this reproduces the paper's per-image
    adaptive fitting (independently estimates backscatter, illuminant, and
    attenuation for this image, and white-balances from this image's own
    statistics). With ``params.locked_stats`` set (see seathru.survey), the
    backscatter/attenuation coefficients and white-balance gains are taken
    from the frozen survey statistics instead of being re-fit, which is both
    faster (skips the nonlinear curve fits) and radiometrically consistent
    across a batch.

    @param img (H, W, 3) float array in [0, 1].
    @param depths (H, W) float array of range in metres; <= 0 marks invalid pixels.
    @param params SeathruParams, optional (defaults are used if omitted).
    @return SeathruResult with the recovered image (and intermediate maps if
        ``params.return_debug``).
    """
    params = params or SeathruParams()
    img = np.asarray(img, dtype=np.float64)
    depths = np.asarray(depths, dtype=np.float64)
    locked = params.locked_stats

    valid = depths > 0
    if not np.any(valid) or (depths[valid].max() - depths[valid].min()) < 1e-3:
        # Constant / missing range map (e.g. PlaneDepthSource): the range-fit is
        # degenerate, so fall back to backscatter removal + white balance only.
        return _run_constant_depth(img, params)

    if locked is not None and locked.backscatter_coefs is not None:
        B = np.stack([_backscatter_model(depths, *locked.backscatter_coefs[c])
                     for c in range(3)], axis=2)
    else:
        ptsR, ptsG, ptsB = find_backscatter_points(img, depths)
        Br, _ = estimate_backscatter(ptsR, depths, params.backscatter_restarts)
        Bg, _ = estimate_backscatter(ptsG, depths, params.backscatter_restarts)
        Bb, _ = estimate_backscatter(ptsB, depths, params.backscatter_restarts)
        B = np.stack([Br, Bg, Bb], axis=2)

    nmap, n = construct_neighborhood_map(depths, params.epsilon,
                                         params.min_neighborhood)

    need_attenuation_fit = locked is None or locked.attenuation_coefs is None
    need_illum = params.return_debug or need_attenuation_fit
    illum = None
    if need_illum:
        illum = np.stack([
            estimate_illumination(img[..., c], B[..., c], nmap, n,
                                  p=params.p, f=params.f)
            for c in range(3)], axis=2)

    if locked is not None and locked.attenuation_coefs is not None:
        beta_D = np.stack([params.l * _beta_D_two_term(depths, *locked.attenuation_coefs[c])
                          for c in range(3)], axis=2)
    else:
        beta_D = np.stack([
            refine_attenuation(depths, illum[..., c],
                               coarse_attenuation(depths, illum[..., c]),
                               restarts=params.attenuation_restarts,
                               l=params.l, spread_fraction=params.spread_fraction)[0]
            for c in range(3)], axis=2)

    wb_gains = locked.wb_gains if locked is not None else None
    stretch_bounds = (locked.stretch_bounds
                      if (locked is not None and locked.stretch_bounds is not None)
                      else None)
    recovered = recover_image(img, depths, B, beta_D, nmap, params.protect_red,
                              params.stretch_pct, wb_gains=wb_gains,
                              stretch_bounds=stretch_bounds)

    if not params.return_debug:
        return SeathruResult(recovered=recovered)
    return SeathruResult(recovered, B, illum, beta_D, nmap)


def upsample_and_recover(full_img, full_depths, result: "SeathruResult",
                         params: SeathruParams):
    """
    @brief Apply a low-resolution Sea-thru estimate to a full-resolution image.

    The backscatter, attenuation and neighbourhood maps are estimated cheaply at
    working resolution, then bilinearly upsampled and used to recover the scene
    at full resolution so the output matches the input dimensions exactly. This
    keeps runtime/RAM low while producing a directly comparable full-size image.
    Honours ``params.locked_stats`` the same way run_seathru does, for the
    white-balance gains and (optional) locked exposure bounds.

    @param full_img (H, W, 3) float native-resolution image in [0, 1].
    @param full_depths (H, W) float native-resolution range map in metres.
    @param result SeathruResult from a prior run_seathru(..., return_debug=True)
        call at working resolution.
    @param params SeathruParams used for that run (for protect_red/stretch/lock).
    @return (H, W, 3) float recovered image at native resolution.
    @pre ``result`` was produced with ``return_debug=True`` (its backscatter,
        beta_D, and neighborhood_map fields must be populated).
    """
    import cv2

    assert result.backscatter is not None and result.beta_D is not None, \
        "upsample_and_recover needs a SeathruResult from a return_debug=True run"

    H, W = full_img.shape[:2]
    B = cv2.resize(result.backscatter.astype(np.float32), (W, H),
                   interpolation=cv2.INTER_LINEAR)
    beta_D = cv2.resize(result.beta_D.astype(np.float32), (W, H),
                        interpolation=cv2.INTER_LINEAR)
    nmap = cv2.resize(result.neighborhood_map.astype(np.int32), (W, H),
                      interpolation=cv2.INTER_NEAREST)
    locked = params.locked_stats
    wb_gains = locked.wb_gains if locked is not None else None
    stretch_bounds = (locked.stretch_bounds
                      if (locked is not None and locked.stretch_bounds is not None)
                      else None)
    return recover_image(full_img, full_depths, B, beta_D, nmap,
                         params.protect_red, params.stretch_pct,
                         wb_gains=wb_gains, stretch_bounds=stretch_bounds)


def _run_constant_depth(img, params: SeathruParams):
    """
    @brief Simplified recovery when the range map carries no spatial information.

    Estimates a per-channel constant backscatter from the darkest 1% of pixels
    (Section 4.3 intuition without the range fit), subtracts it, and applies the
    white balance of Equation 9. No range-dependent attenuation term. Honours
    ``params.locked_stats.wb_gains`` / ``.stretch_bounds`` when present, since
    those are still meaningful without a spatial range map even though the
    backscatter/attenuation coefficients are not (see PlaneDepthSource).

    @param img (H, W, 3) float array in [0, 1].
    @param params SeathruParams.
    @return SeathruResult.
    """
    locked = params.locked_stats
    flat = img.reshape(-1, 3)
    dark = flat[np.argsort(flat.mean(1))[:max(1, flat.shape[0] // 100)]]
    B = dark.mean(0)
    direct = np.clip(img - B, 0.0, 1.0)
    gains = (locked.wb_gains if locked is not None and locked.wb_gains is not None
            else _compute_wb_gains(direct, params.protect_red))
    stretch_bounds = (locked.stretch_bounds
                      if (locked is not None and locked.stretch_bounds is not None)
                      else None)
    out = _robust_stretch(_apply_wb_gains(direct, gains), None,
                          params.stretch_pct, stretch_bounds)
    if not params.return_debug:
        return SeathruResult(recovered=out)
    B_map = np.broadcast_to(B, img.shape)
    nmap = np.ones(img.shape[:2], dtype=np.int32)
    zeros = np.zeros(img.shape[:2])
    return SeathruResult(out, B_map, np.zeros_like(img), np.stack([zeros]*3, 2), nmap)


# --------------------------------------------------------------------------- #
# Survey-locked calibration internals (used by seathru.survey)               #
# --------------------------------------------------------------------------- #
def _fit_calibration_sample(img, depths, params: SeathruParams):
    """
    @brief Fit one calibration frame's backscatter/attenuation coefficients and
    white-balance gain, for averaging across a survey sample.

    Mirrors run_seathru's per-image adaptive path, but returns the raw fitted
    *coefficients* (not just the evaluated per-pixel maps) so they can be
    re-evaluated against every other frame's own depth map at apply time. Not
    part of the public API; used by seathru.survey.calibrate_survey_stats.

    @param img (H, W, 3) float array in [0, 1] (spatially-varying depth assumed).
    @param depths (H, W) float range map in metres.
    @param params SeathruParams controlling fit restarts / knobs.
    @return dict with keys ``backscatter_coefs`` (3,4)|None, ``attenuation_coefs``
        (3,4)|None, ``wb_gains`` (3,), ``stretch_bounds`` (lo, hi).
    """
    ptsR, ptsG, ptsB = find_backscatter_points(img, depths)
    back_coefs, B_channels = [], []
    for pts in (ptsR, ptsG, ptsB):
        B_ch, coefs = estimate_backscatter(pts, depths, params.backscatter_restarts)
        B_channels.append(B_ch)
        back_coefs.append(coefs if coefs.shape == (4,) else None)
    B = np.stack(B_channels, axis=2)
    backscatter_coefs = None if any(c is None for c in back_coefs) else np.stack(back_coefs)

    nmap, n = construct_neighborhood_map(depths, params.epsilon, params.min_neighborhood)
    illum_channels, atten_coefs = [], []
    for c in range(3):
        illum_c = estimate_illumination(img[..., c], B[..., c], nmap, n,
                                        p=params.p, f=params.f)
        illum_channels.append(illum_c)
        _, coefs = refine_attenuation(depths, illum_c, coarse_attenuation(depths, illum_c),
                                      restarts=params.attenuation_restarts, l=1.0,
                                      spread_fraction=params.spread_fraction)
        atten_coefs.append(coefs if (coefs is not None and coefs.shape == (4,)) else None)
    attenuation_coefs = None if any(c is None for c in atten_coefs) else np.stack(atten_coefs)

    if attenuation_coefs is not None:
        beta_D = np.stack([params.l * _beta_D_two_term(depths, *attenuation_coefs[c])
                          for c in range(3)], axis=2)
    else:
        beta_D = np.stack([
            refine_attenuation(depths, illum_channels[c],
                               coarse_attenuation(depths, illum_channels[c]),
                               restarts=params.attenuation_restarts,
                               l=params.l, spread_fraction=params.spread_fraction)[0]
            for c in range(3)], axis=2)

    direct = _compute_direct(img, depths, B, beta_D, nmap)
    fg = nmap != 0
    wb_gains = _compute_wb_gains(direct, params.protect_red)
    balanced = _apply_wb_gains(direct, wb_gains)
    if np.any(fg):
        lo, hi = np.percentile(balanced[fg], params.stretch_pct)
    else:
        lo, hi = 0.0, 1.0

    return {
        "backscatter_coefs": backscatter_coefs,
        "attenuation_coefs": attenuation_coefs,
        "wb_gains": wb_gains,
        "stretch_bounds": (float(lo), float(hi)),
    }


def _run_constant_depth_stats(img, params: SeathruParams):
    """
    @brief Calibration-sample stats for a constant/degenerate-depth frame.

    Only a white-balance gain and stretch bounds can be calibrated without a
    spatial range map (mirrors _run_constant_depth's simplified recovery).

    @param img (H, W, 3) float array in [0, 1].
    @param params SeathruParams.
    @return dict, same shape as _fit_calibration_sample's, with
        ``backscatter_coefs`` and ``attenuation_coefs`` always ``None``.
    """
    flat = img.reshape(-1, 3)
    dark = flat[np.argsort(flat.mean(1))[:max(1, flat.shape[0] // 100)]]
    B = dark.mean(0)
    direct = np.clip(img - B, 0.0, 1.0)
    wb_gains = _compute_wb_gains(direct, params.protect_red)
    balanced = _apply_wb_gains(direct, wb_gains)
    lo, hi = np.percentile(balanced.reshape(-1, 3), params.stretch_pct)
    return {
        "backscatter_coefs": None,
        "attenuation_coefs": None,
        "wb_gains": wb_gains,
        "stretch_bounds": (float(lo), float(hi)),
    }
