#!/usr/bin/env python3
"""
@file make_gt_depth.py
@brief Convert COLMAP dense depth maps to SeaSplat ground-truth-depth .npy files.

SeaSplat's dataset loader picks up per-image metric depth from
``<dataset>/images/depth/<image_name>.npy`` (used by ``--use_depth_l1_loss``).
This script reads every ``*.geometric.bin`` in a COLMAP dense workspace's
``stereo/depth_maps`` and writes the matching ``.npy`` (float32 metres,
invalid pixels set to 0) into ``<out-dir>``.

Usage
-----
    python make_gt_depth.py \
        --dense 2024_02_PALFREY_TEST_R2/colmap/dense \
        --out-dir 2024_02_PALFREY_TEST_R2/seathru_out/depth

@author Michael Venz
@date 2026
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Works installed (pip install -e .) or straight from this repo checkout.
try:
    from seathru.depth.colmap_source import read_colmap_array
except ImportError:  # not installed: add the repo root (parent of docs/)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from seathru.depth.colmap_source import read_colmap_array  # noqa: E402


def main(argv=None):
    """
    @brief Entry point: convert all geometric depth maps in a dense workspace.
    @param argv Optional argument list (defaults to ``sys.argv[1:]``).
    @return Process exit code (0 on success, 2 if no depth maps found).
    """
    ap = argparse.ArgumentParser(
        description="COLMAP *.geometric.bin depth maps -> SeaSplat gt-depth .npy files")
    ap.add_argument("--dense", type=Path, required=True,
                    help="COLMAP dense workspace (contains stereo/depth_maps)")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Destination for <image_name>.npy files "
                         "(use <splat images dir>/depth)")
    args = ap.parse_args(argv)

    depth_dir = args.dense / "stereo" / "depth_maps"
    files = sorted(depth_dir.glob("*.geometric.bin"))
    if not files:
        print(f"ERROR: no *.geometric.bin in {depth_dir}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(files, 1):
        d = read_colmap_array(f).astype(np.float32)
        d[~np.isfinite(d)] = 0.0
        d[d < 0] = 0.0
        name = f.name.replace(".geometric.bin", "")  # e.g. G0023380.JPG
        np.save(args.out_dir / f"{name}.npy", d)
        if i % 25 == 0 or i == len(files):
            print(f"  {i}/{len(files)} ...")

    print(f"Done: {len(files)} depth maps -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
