"""
@file survey.py
@brief Multi-image calibration for "survey-locked" Sea-thru processing.

The paper's Sea-thru fits backscatter, illuminant and attenuation
independently on every image. That is the right choice for a handful of
photos, but across a multi-thousand-frame ASV survey it lets ambient
lighting, turbidity, and white-balance drift frame to frame - which shows up
as visible seams in an orthomosaic and view-dependent colour flicker in a
NeRF/3DGS reconstruction.

This module fits those statistics once, from a small representative sample
of the survey (spread evenly across the whole image sequence), and returns a
seathru.core.SurveyStats bundle that seathru.core.run_seathru can apply to
every frame instead of re-fitting per image. See the README's "Survey-locked
mode" section for the recommended CLI workflow
(``seathru --survey-locked ...``).

@author Michael Venz
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .core import (SeathruParams, SurveyStats, _fit_calibration_sample,
                   _run_constant_depth_stats)
from .depth.base import DepthSource, ImageMeta
from .io_images import load_image


def select_calibration_sample(image_paths, sample_size, seed=0):
    """
    @brief Choose a representative subset of survey images to calibrate from.

    Indices are spread evenly across the sorted file list (rather than drawn
    randomly) so the sample spans the whole survey - start/mid/end lighting,
    altitude, and turbidity - instead of clustering in one time window.

    @param image_paths Sorted list of image Path objects for the whole survey.
    @param sample_size Desired number of calibration frames.
    @param seed Reserved for future randomised-sampling strategies; unused by
        the current deterministic even-spacing selection.
    @return List of Path objects, length ``min(sample_size, len(image_paths))``.
    """
    del seed  # reserved; even spacing is deterministic today
    n = len(image_paths)
    k = min(sample_size, n)
    if k <= 0:
        return []
    idx = sorted(set(np.linspace(0, n - 1, k).round().astype(int).tolist()))
    return [image_paths[i] for i in idx]


def calibrate_survey_stats(image_paths, depth_source: DepthSource, meta_map,
                           params: SeathruParams, max_size=1024, sample_size=12,
                           seed=0, on_progress=print) -> SurveyStats:
    """
    @brief Fit frozen backscatter/attenuation/white-balance statistics from a
    representative sample of survey images.

    Runs the full per-image adaptive fit (seathru.core._fit_calibration_sample)
    on each sampled frame, then takes the per-coefficient *median* across the
    sample (robust to one or two outlier frames - e.g. an over/under-exposed
    or texture-poor scene). Fields that fail to fit on any sampled frame are
    left as ``None`` so run_seathru falls back to per-image adaptive fitting
    for just that piece rather than failing outright.

    Runtime is ``sample_size`` full per-image adaptive fits (the same cost as
    processing that many images without locking) - see the README timing
    table. This cost is paid once per survey, not per image.

    @param image_paths Sorted list of image Path objects for the whole survey
        (the same list seathru.pipeline.process_folder builds).
    @param depth_source A seathru.depth.DepthSource used to obtain each sample
        frame's range map.
    @param meta_map ``{image_name: ImageMeta}``, e.g. from
        seathru.metadata.load_metadata. Pass ``{}`` if there is no CSV.
    @param params SeathruParams controlling fit restarts / knobs. Its
        ``locked_stats`` field is ignored (calibration always fits fresh).
    @param max_size Working resolution (long edge, pixels) for calibration
        frames; should match the resolution used for the main batch run.
    @param sample_size Number of images to sample and fit.
    @param seed Reserved for future randomised sampling.
    @param on_progress Callback invoked with a one-line status string per
        sampled image (defaults to ``print``).
    @return A SurveyStats with median-aggregated coefficients.
    @pre ``image_paths`` is non-empty.
    @throws RuntimeError if every sampled frame fails to yield even a
        white-balance gain (i.e. calibration produced nothing usable).
    """
    assert len(image_paths) > 0, "calibrate_survey_stats needs at least one image"
    sample = select_calibration_sample(image_paths, sample_size, seed)

    backscatter_fits, attenuation_fits, gains, bounds, used = [], [], [], [], []
    wc_pool = [([], []), ([], []), ([], [])]   # per-channel pooled (z, lnE)
    for i, path in enumerate(sample, 1):
        on_progress(f"[calib {i}/{len(sample)}] {path.name} ...")
        meta = meta_map.get(path.name, ImageMeta(image_name=path.name))
        img, _ = load_image(path, max_size=max_size)
        depths = depth_source.get_depth(img, meta)

        valid = depths > 0
        is_constant = (not np.any(valid)
                      or (depths[valid].max() - depths[valid].min()) < 1e-3)
        stats = (_run_constant_depth_stats(img, params) if is_constant
                else _fit_calibration_sample(img, depths, params))

        if stats["backscatter_coefs"] is not None:
            backscatter_fits.append(stats["backscatter_coefs"])
        if stats["attenuation_coefs"] is not None:
            attenuation_fits.append(stats["attenuation_coefs"])
        if stats.get("wc_samples") is not None:
            for c in range(3):
                wc_pool[c][0].append(np.asarray(stats["wc_samples"][c][0]))
                wc_pool[c][1].append(np.asarray(stats["wc_samples"][c][1]))
        gains.append(stats["wb_gains"])
        bounds.append(stats["stretch_bounds"])
        used.append(path.name)

    if not gains:
        raise RuntimeError(
            "Survey calibration failed on every sampled image; try a larger "
            "--calib-sample-size, or check that the depth source is returning "
            "valid range maps."
        )

    # Pooled water-column fit: one E_c(z) = exp(a + b z) per channel across the
    # calibration frames, covering the survey's full depth range so even frames
    # too flat to fit their own exponential get the correct survey-wide water
    # model at apply time.
    #
    # The slope is the pooled WITHIN-frame estimator (fixed-effects): each
    # frame's samples are de-meaned before contributing. Fitting the raw pooled
    # samples instead lets between-frame differences (auto-exposure: deeper
    # frames are darker overall) masquerade as depth attenuation, inflating the
    # slope — observed as pink over-correction on the deepest frames.
    wc_coefs, wc_z_range = None, None
    n_wc_frames = len(wc_pool[0][0])
    if n_wc_frames >= 3:
        wc_coefs = np.zeros((3, 2))
        for c in range(3):
            num = den = 0.0
            offsets = []
            for zs, ys in zip(wc_pool[c][0], wc_pool[c][1]):
                zc, yc = zs - zs.mean(), ys - ys.mean()
                num += float(np.dot(zc, yc))
                den += float(np.dot(zc, zc))
            b = num / max(den, 1e-9)
            for zs, ys in zip(wc_pool[c][0], wc_pool[c][1]):
                offsets.append(float(ys.mean() - b * zs.mean()))
            wc_coefs[c] = (float(np.median(offsets)), b)
        z_all = np.concatenate(wc_pool[0][0])
        wc_z_range = (float(z_all.min()), float(z_all.max()))

    return SurveyStats(
        backscatter_coefs=(np.median(np.stack(backscatter_fits), axis=0)
                           if backscatter_fits else None),
        attenuation_coefs=(np.median(np.stack(attenuation_fits), axis=0)
                           if attenuation_fits else None),
        wb_gains=np.median(np.stack(gains), axis=0),
        stretch_bounds=tuple(np.median(np.array(bounds), axis=0).tolist()),
        wc_illum_coefs=wc_coefs,
        wc_z_range=wc_z_range,
        n_calibration_images=len(used),
        source_images=used,
    )
