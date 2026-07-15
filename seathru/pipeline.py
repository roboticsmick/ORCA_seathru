"""
@file pipeline.py
@brief Batch driver: run Sea-thru over a folder of images using a depth source.

Two processing modes, selected by ``survey_locked``:
  - **Adaptive** (default): every image independently fits its own
    backscatter/illuminant/attenuation/white-balance statistics, exactly as
    in the paper.
  - **Survey-locked**: seathru.survey.calibrate_survey_stats fits those
    statistics once from a sample of the batch, caches them to a JSON file,
    and every image reuses the frozen statistics - faster and radiometrically
    consistent across the survey. See the README's "Survey-locked mode"
    section.

@author Michael Venz
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import replace
from pathlib import Path

from .core import (SeathruParams, SeathruResult, SurveyStats, run_seathru,
                   upsample_and_recover)
from .depth.base import DepthSource, ImageMeta
from .io_images import load_image, save_debug_panel, save_image
from .metadata import load_metadata
from .survey import calibrate_survey_stats

## Image file extensions treated as inputs by process_folder.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def process_image(image_path, depth_source: DepthSource, meta: ImageMeta,
                  params: SeathruParams | None = None,
                  max_size=1024, full_res=False):
    """
    @brief Load one image, obtain its range map, and run Sea-thru.

    When ``full_res`` is set, the model is estimated at ``max_size`` but the
    recovered image is produced at the image's native resolution, so it can
    be compared with the input at 1:1.

    @param image_path Path to the source image.
    @param depth_source A seathru.depth.DepthSource providing the range map.
    @param meta ImageMeta for this image (from seathru.metadata.load_metadata,
        or a bare ``ImageMeta(image_name=...)`` if there is no CSV).
    @param params SeathruParams, optional. Pass ``locked_stats`` (via
        ``dataclasses.replace``) for survey-locked processing.
    @param max_size Working (estimation) resolution, long edge, in pixels.
    @param full_res Estimate at ``max_size`` but recover at native resolution.
    @return Tuple ``(result, working_image)``: the SeathruResult and the
        (possibly downscaled) working-resolution input image used for QA panels.
    """
    params = params or SeathruParams()
    img, _ = load_image(image_path, max_size=max_size)
    depths = depth_source.get_depth(img, meta)
    if not full_res:
        return run_seathru(img, depths, params), img

    est = run_seathru(img, depths, replace(params, return_debug=True))
    full_img, _ = load_image(image_path, max_size=None)
    full_depths = depth_source.get_depth(full_img, meta)
    recovered = upsample_and_recover(full_img, full_depths, est, params)
    result = SeathruResult(recovered, est.backscatter, est.illuminant,
                           est.beta_D, est.neighborhood_map, notes=est.notes)
    return result, img


def process_folder(input_dir, out_dir, depth_source: DepthSource,
                   csv_path=None, params: SeathruParams | None = None,
                   max_size=1024, debug=False, full_res=False, on_progress=print,
                   survey_locked=False, calib_sample_size=12, calib_seed=0,
                   lock_exposure=False, lock_backscatter=True, stats_path=None):
    """
    @brief Process every image in ``input_dir``; write recovered images to
    ``out_dir``. This is the function ``seathru.cli`` drives.

    In survey-locked mode (``survey_locked=True``), this first calls
    seathru.survey.calibrate_survey_stats (or loads a previously saved
    ``stats_path`` JSON, so re-runs / resumed batches don't recalibrate), then
    attaches the resulting SurveyStats to ``params.locked_stats`` for every
    image in the batch.

    @param input_dir Directory of source images.
    @param out_dir Output directory (created if missing); also holds the
        default survey-stats JSON when ``survey_locked`` is set.
    @param depth_source A seathru.depth.DepthSource providing range maps.
    @param csv_path Optional ASV processed-images CSV (depth_m / heading / GPS).
    @param params SeathruParams, optional.
    @param max_size Working (estimation) resolution, long edge, in pixels.
    @param debug Save an intermediate-map montage per image (see io_images).
    @param full_res Output at native resolution (see process_image).
    @param on_progress Callback invoked with a one-line status string per image.
    @param survey_locked Freeze backscatter/attenuation/white-balance
        statistics across the whole batch instead of adapting per image.
    @param calib_sample_size Number of images sampled to fit the locked stats.
    @param calib_seed Reserved for future randomised calibration sampling.
    @param lock_exposure Also freeze the output contrast-stretch bounds from
        calibration; otherwise each frame keeps its own exposure normalisation
        even in survey-locked mode.
    @param lock_backscatter Keep the frozen per-survey backscatter (default).
        Set False to re-estimate backscatter per image while still locking
        white-balance (and, if requested, exposure). Backscatter
        ``B = veiling * (1 - exp(-beta_bs * z))`` is intrinsically
        range-dependent, so on surveys with large depth relief (e.g. a reef
        dropoff) a single frozen backscatter under-subtracts veiling light at
        depth and leaves the deep water uncorrected; per-image backscatter with
        locked white-balance keeps cross-view colour consistency without that
        failure.
    @param stats_path Path to save/load the locked-stats JSON. Defaults to
        ``<out_dir>/survey_stats.json``. If the file already exists it is
        loaded (not recalibrated) - delete it to force a fresh calibration.
    @return List of ``(image_name, status)`` tuples, ``status`` is ``"ok"`` or
        an ``"error: ..."`` message.
    """
    input_dir, out_dir = Path(input_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_map = load_metadata(csv_path) if csv_path else {}
    params = params or SeathruParams()

    images = sorted(p for p in input_dir.iterdir()
                    if p.suffix.lower() in IMAGE_EXTS)

    if survey_locked:
        resolved_stats_path = Path(stats_path) if stats_path else out_dir / "survey_stats.json"
        if resolved_stats_path.exists():
            on_progress(f"Loading locked survey stats from {resolved_stats_path}")
            stats = SurveyStats.load(resolved_stats_path)
        else:
            on_progress(
                f"Calibrating survey-locked stats from "
                f"{min(calib_sample_size, len(images))} sample image(s) ..."
            )
            stats = calibrate_survey_stats(
                images, depth_source, meta_map, params, max_size=max_size,
                sample_size=calib_sample_size, seed=calib_seed,
                on_progress=on_progress,
            )
            resolved_stats_path.parent.mkdir(parents=True, exist_ok=True)
            stats.save(resolved_stats_path)
            on_progress(f"Saved locked stats -> {resolved_stats_path}")
        if not lock_exposure:
            stats = replace(stats, stretch_bounds=None)
        if not lock_backscatter:
            stats = replace(stats, backscatter_coefs=None)
        params = replace(params, locked_stats=stats)

    results = []
    path_counts = Counter()
    for i, path in enumerate(images, 1):
        meta = meta_map.get(path.name, ImageMeta(image_name=path.name))
        on_progress(f"[{i}/{len(images)}] {path.name} ...")
        try:
            result, img = process_image(path, depth_source, meta, params,
                                        max_size, full_res=full_res)
        except Exception as err:  # keep the batch going
            on_progress(f"    FAILED: {err}")
            results.append((path.name, f"error: {err}"))
            path_counts["FAILED"] += 1
            continue
        # Processing-path notes: which code path each component took (illum
        # mode/fallbacks from run_seathru, depth fill actions from the source).
        notes = list(result.notes)
        notes += getattr(depth_source, "last_notes", [])
        for note in notes:
            # Aggregate on the note's kind, not its per-frame numbers:
            # "mono-filled 13.0% (align err 0.10 m)" -> "mono-filled".
            path_counts[re.split(r"[\d(]", note)[0].strip()] += 1
        if notes:
            on_progress(f"    [{'; '.join(notes)}]")
        save_image(out_dir / f"{path.stem}_seathru.png", result.recovered)
        if debug and result.backscatter is not None:
            save_debug_panel(out_dir / f"{path.stem}_debug.png", result, img)
        results.append((path.name, "ok"))

    # End-of-run audit: every distinct processing path and how many frames
    # took it. An unexpected entry here (fallbacks, failures, heavy fills) is
    # the first place to look when some frames come out different.
    if path_counts:
        on_progress("\nProcessing-path summary:")
        for note, count in path_counts.most_common():
            on_progress(f"  {count:5d}x  {note}")
    return results
