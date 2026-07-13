"""
@file cli.py
@brief Command-line interface for Sea-thru.

Examples
--------
Plane depth (works with no torch / no SfM), over a test set::

    python -m seathru.cli \\
        --input-dir images --csv processed_images.csv \\
        --out-dir seathru_out --depth plane --plane-default 5 --debug

SfM depth maps exported per image::

    python -m seathru.cli --input-dir imgs --out-dir out \\
        --depth file --depth-dir sfm_depth --depth-scale 1.0

Monocular depth (needs torch)::

    python -m seathru.cli --input-dir imgs --out-dir out \\
        --depth mono --mono-backend midas

Survey-locked mode, for a large consistent dataset (see the README)::

    python -m seathru.cli --input-dir imgs --out-dir out --csv meta.csv \\
        --depth colmap --colmap-workspace path/to/dense \\
        --survey-locked --calib-sample-size 20

@author Michael Venz
"""
from __future__ import annotations

import argparse

from .core import SeathruParams
from .pipeline import process_folder


def build_depth_source(args):
    """
    @brief Instantiate the seathru.depth.DepthSource selected by ``--depth``.
    @param args Parsed argparse.Namespace from main().
    @return A DepthSource instance.
    @throws ValueError if ``args.depth`` is not a recognised choice.
    """
    if args.depth == "plane":
        from .depth import PlaneDepthSource
        return PlaneDepthSource(default_m=args.plane_default)
    if args.depth == "estimated":
        from .depth import EstimatedDepthSource
        return EstimatedDepthSource(near=args.est_near, far=args.est_far)
    if args.depth == "file":
        from .depth import FileDepthSource
        return FileDepthSource(args.depth_dir, scale=args.depth_scale)
    if args.depth == "colmap":
        from .depth import ColmapDepthSource
        return ColmapDepthSource(args.colmap_workspace, kind=args.colmap_depth_kind,
                                 clip_low_percentile=args.colmap_clip_low,
                                 fill_holes_max_frac=args.colmap_fill_holes,
                                 fill_border=args.colmap_fill_border)
    if args.depth == "mono":
        from .depth import MonocularDepthSource
        rng = (args.mono_near, args.mono_far) if args.mono_near else None
        return MonocularDepthSource(backend=args.mono_backend, fixed_range=rng)
    raise ValueError(args.depth)


def main(argv=None):
    """
    @brief Entry point for the ``seathru`` console script / ``python -m seathru.cli``.
    @param argv Optional argument list (defaults to ``sys.argv[1:]`` via argparse).
    @return None. Prints a summary line and exits normally; per-image failures
        are logged and do not abort the batch (see seathru.pipeline.process_folder).
    """
    p = argparse.ArgumentParser(prog="seathru", description="Sea-thru underwater color recovery")
    p.add_argument("--input-dir", required=True, help="Folder of input images")
    p.add_argument("--out-dir", required=True, help="Folder recovered images are written to")
    p.add_argument("--csv", default=None, help="ASV processed-images CSV (for depth_m / heading)")
    p.add_argument("--max-size", type=int, default=1024, help="Working (estimation) resolution, long edge")
    p.add_argument("--full-res", action="store_true", help="Output at native image resolution (estimate at --max-size, apply at full size)")
    p.add_argument("--debug", action="store_true", help="Save intermediate-map montages")

    depth_group = p.add_argument_group("depth source")
    depth_group.add_argument("--depth", choices=["plane", "estimated", "file", "colmap", "mono"], default="plane")
    depth_group.add_argument("--plane-default", type=float, default=5.0, help="Altitude (m) when CSV depth_m is missing")
    depth_group.add_argument("--est-near", type=float, default=1.0, help="Nearest range (m) for --depth estimated")
    depth_group.add_argument("--est-far", type=float, default=10.0, help="Farthest range (m) for --depth estimated")
    depth_group.add_argument("--depth-dir", default=None, help="Directory of SfM range maps (--depth file)")
    depth_group.add_argument("--depth-scale", type=float, default=1.0, help="Multiply loaded maps into metres")
    depth_group.add_argument("--colmap-workspace", default=None, help="COLMAP dense workspace with stereo/depth_maps (--depth colmap)")
    depth_group.add_argument("--colmap-depth-kind", choices=["geometric", "photometric"], default="geometric",
                             help="Which patch_match_stereo map to read (--depth colmap). 'photometric' is a "
                                  "single-pass map ~2x faster to compute; 'geometric' is multi-view consistent")
    depth_group.add_argument("--colmap-clip-low", type=float, default=2.0,
                             help="Drop depths below this percentile (--depth colmap). MVS emits spurious "
                                  "near-camera points; because beta = -ln(illuminant)/z, a tiny z explodes "
                                  "beta and corrupts the attenuation fit. 0 disables")
    depth_group.add_argument("--colmap-fill-holes", type=float, default=0.02,
                             help="Fill INTERIOR invalid depth holes smaller than this fraction of image area "
                                  "with the nearest valid depth (--depth colmap) - bounded interpolation, so "
                                  "MVS speckle and occluder holes (fish) don't survive as untreated raw-colour "
                                  "patches mid-frame. 0 disables")
    depth_group.add_argument("--colmap-fill-border", action="store_true",
                             help="Also fill border-touching invalid depth regions by nearest-valid "
                                  "extrapolation (--depth colmap). Off by default: edge gaps are covered by "
                                  "overlapping frames in multi-view products; enable for clean standalone frames")
    depth_group.add_argument("--mono-backend", choices=["midas", "depth_anything_v2"], default="midas")
    depth_group.add_argument("--mono-near", type=float, default=None)
    depth_group.add_argument("--mono-far", type=float, default=None)

    # Algorithm knobs (see SeathruParams in core.py for the full CONFIG reference).
    algo_group = p.add_argument_group("algorithm config (see SeathruParams docstring)")
    algo_group.add_argument("--p", type=float, default=0.5, help="Illuminant locality, Eq. 14 support weight, in [0,1]")
    algo_group.add_argument("--f", type=float, default=2.0, help="Illuminant geometry factor, Section 4.4.2")
    algo_group.add_argument("--l", type=float, default=1.0, help="Attenuation/brightness balance (main strength dial)")
    algo_group.add_argument("--epsilon", type=float, default=0.05, help="Iso-range neighbourhood band width, Eq. 15")
    algo_group.add_argument("--stretch-low", type=float, default=0.5,
                            help="Lower output-stretch percentile (robust to dark outliers)")
    algo_group.add_argument("--stretch-high", type=float, default=99.5,
                            help="Upper output-stretch percentile (robust to glints/caustic highlights)")
    algo_group.add_argument("--no-protect-red", action="store_true",
                            help="Disable red-protected white balance; use pure Gray-World")
    algo_group.add_argument("--backscatter-restarts", type=int, default=25,
                            help="Curve-fit random restarts for backscatter (Eq. 10); lower = faster, less robust")
    algo_group.add_argument("--attenuation-restarts", type=int, default=10,
                            help="Curve-fit random restarts for attenuation (Eq. 11); lower = faster, less robust")
    algo_group.add_argument("--attenuation-mode", choices=["two-term", "coarse"], default="two-term",
                            help="beta_D estimation. 'two-term' = the paper's Eq. 11 decaying fit (correct for "
                                 "HORIZONTAL imaging). 'coarse' = use the illuminant-derived beta_D (Eq. 12) "
                                 "directly; REQUIRED for DOWNWARD-looking surveys with depth relief (e.g. a reef "
                                 "dropoff), where range ~= depth so beta_D RISES with range and the decaying "
                                 "two-term form cannot represent it, leaving deep water uncorrected")

    survey_group = p.add_argument_group("survey-locked mode (see README)")
    survey_group.add_argument("--survey-locked", action="store_true",
                              help="Freeze backscatter/attenuation/white-balance stats across the whole "
                                   "batch instead of adapting per image (recommended for orthomosaic / "
                                   "splatting radiometric consistency)")
    survey_group.add_argument("--calib-sample-size", type=int, default=12,
                              help="Images sampled (evenly spread across the survey) to fit the locked stats")
    survey_group.add_argument("--calib-seed", type=int, default=0,
                              help="Reserved for future randomised calibration sampling")
    survey_group.add_argument("--lock-exposure", action="store_true",
                              help="Also freeze the output contrast-stretch bounds; by default exposure "
                                   "still adapts per image even in --survey-locked mode")
    survey_group.add_argument("--no-lock-backscatter", dest="lock_backscatter",
                              action="store_false",
                              help="In --survey-locked mode, re-estimate backscatter per image while still "
                                   "locking white-balance/exposure. Backscatter is range-dependent, so on "
                                   "surveys with large depth relief (reef dropoffs) a frozen backscatter "
                                   "under-corrects the deep frames. Recommended for downward surveys")
    survey_group.add_argument("--stats-file", default=None,
                              help="Path to save/load the locked-stats JSON "
                                   "(default: <out-dir>/survey_stats.json). If it already exists it is "
                                   "loaded, not recalibrated.")

    args = p.parse_args(argv)

    params = SeathruParams(
        p=args.p, f=args.f, l=args.l, epsilon=args.epsilon,
        protect_red=not args.no_protect_red,
        stretch_pct=(args.stretch_low, args.stretch_high),
        backscatter_restarts=args.backscatter_restarts,
        attenuation_restarts=args.attenuation_restarts,
        attenuation_mode=args.attenuation_mode,
        return_debug=args.debug,
    )
    depth_source = build_depth_source(args)
    results = process_folder(
        args.input_dir, args.out_dir, depth_source,
        csv_path=args.csv, params=params,
        max_size=args.max_size, debug=args.debug, full_res=args.full_res,
        survey_locked=args.survey_locked, calib_sample_size=args.calib_sample_size,
        calib_seed=args.calib_seed, lock_exposure=args.lock_exposure,
        stats_path=args.stats_file,
    )
    ok = sum(1 for _, s in results if s == "ok")
    print(f"\nDone: {ok}/{len(results)} images recovered -> {args.out_dir}")


if __name__ == "__main__":
    main()
