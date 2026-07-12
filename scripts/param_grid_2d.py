"""
@file param_grid_2d.py
@brief Sweep TWO Sea-thru parameters together as a 2D grid on one baseline image.

Columns vary the x-parameter, rows vary the y-parameter; every other parameter
stays at its default. This shows how two knobs interact (e.g. brightness `f`
against range-correction strength `l`) on a single fixed scene, so the only
thing changing across the grid is the two parameters.

Uses the image-derived depth prior by default so all parameters are active
(a flat plane would leave f/l/p/epsilon inert - see param_effect_grid.py).

Examples
--------
    # default: f (columns) x l (rows) on one image
    python scripts/param_grid_2d.py \\
        --input-dir ../2024_02_PALFREY/images \\
        --csv ../2024_02_PALFREY/processed_images.csv \\
        --out-dir ../2024_02_PALFREY/seathru_param_effects \\
        --image G0022406.JPG

    # p (columns) x epsilon (rows), custom values
    python scripts/param_grid_2d.py --input-dir ... --csv ... --out-dir ... \\
        --image G0022406.JPG \\
        --x-param p   --x-values 0.1,0.3,0.5,0.9 \\
        --y-param epsilon --y-values 0.02,0.05,0.10
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seathru.core import SeathruParams, run_seathru          # noqa: E402
from seathru.depth import (ColmapDepthSource, EstimatedDepthSource,  # noqa: E402
                           PlaneDepthSource)
from seathru.depth.base import ImageMeta                     # noqa: E402
from seathru.io_images import _linear_to_srgb, load_image    # noqa: E402
from seathru.metadata import load_metadata                   # noqa: E402

# Sweepable scalars. stretch_low/stretch_high map into the stretch_pct tuple;
# the rest are direct SeathruParams fields.
SCALAR_PARAMS = ("f", "l", "p", "epsilon", "stretch_low", "stretch_high")


def apply_params(base, updates):
    """Return a SeathruParams copy with `updates` applied, handling the
    stretch_low/stretch_high tuple members specially."""
    stretch = list(base.stretch_pct)
    direct = {}
    for name, val in updates.items():
        if name == "stretch_low":
            stretch[0] = val
        elif name == "stretch_high":
            stretch[1] = val
        else:
            direct[name] = val
    return replace(base, stretch_pct=tuple(stretch), **direct)
COL_HDR_H = 24
ROW_HDR_W = 74
TITLE_H = 58
PAD = 6


def _font(size=13):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def to_srgb_uint8(img_lin):
    return (_linear_to_srgb(img_lin) * 255 + 0.5).astype(np.uint8)


def thumb(arr_uint8, width):
    img = Image.fromarray(arr_uint8).convert("RGB")
    h = int(img.height * (width / img.width))
    return img.resize((width, h), Image.LANCZOS)


def parse_values(text):
    return [float(v) for v in text.split(",") if v.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--image", required=True, help="Baseline image filename")
    ap.add_argument("--x-param", choices=SCALAR_PARAMS, default="f")
    ap.add_argument("--y-param", choices=SCALAR_PARAMS, default="l")
    ap.add_argument("--x-values", default="1.5,2.0,2.5,3.0")
    ap.add_argument("--y-values", default="0.5,1.0,1.5")
    ap.add_argument("--depth", choices=["estimated", "plane", "colmap"], default="estimated")
    ap.add_argument("--colmap-workspace", default=None,
                    help="COLMAP dense workspace (metric depth). Use --input-dir "
                         "<workspace>/images so image names match the depth maps.")
    ap.add_argument("--colmap-depth-kind", choices=["geometric", "photometric"],
                    default="geometric")
    ap.add_argument("--max-size", type=int, default=800)
    ap.add_argument("--thumb-width", type=int, default=360)
    ap.add_argument("--plane-default", type=float, default=5.0)
    args = ap.parse_args(argv)

    if args.x_param == args.y_param:
        raise SystemExit("--x-param and --y-param must differ.")

    input_dir, out_dir = Path(args.input_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_map = load_metadata(args.csv) if args.csv else {}
    meta = meta_map.get(args.image, ImageMeta(image_name=args.image))

    img_lin, _ = load_image(input_dir / args.image, max_size=args.max_size)
    if args.depth == "colmap":
        if not args.colmap_workspace:
            raise SystemExit("--depth colmap needs --colmap-workspace")
        source = ColmapDepthSource(args.colmap_workspace, kind=args.colmap_depth_kind)
        depth_note = f"depth = COLMAP {args.colmap_depth_kind} (metric, all params active)"
    elif args.depth == "estimated":
        source = EstimatedDepthSource()
        depth_note = "depth = image-derived prior (all params active)"
    else:
        source = PlaneDepthSource(default_m=args.plane_default)
        depth_note = "depth = flat plane (f/l/p/epsilon inert!)"
    depths = source.get_depth(img_lin, meta)

    xs, ys = parse_values(args.x_values), parse_values(args.y_values)
    base = SeathruParams()
    w = args.thumb_width

    print(f"{args.x_param} x {args.y_param} = {len(xs)} x {len(ys)} grid on "
          f"{args.image} ...")
    t0 = time.time()
    tiles = {}  # (yi, xi) -> uint8 thumbnail
    for yi, yv in enumerate(ys):
        for xi, xv in enumerate(xs):
            params = apply_params(base, {args.x_param: xv, args.y_param: yv})
            rec = run_seathru(img_lin, depths, params).recovered
            tiles[(yi, xi)] = thumb(to_srgb_uint8(rec), w)
    cell_h = max(t.height for t in tiles.values())
    print(f"  {len(xs) * len(ys)} runs in {time.time() - t0:.1f}s")

    # ---- compose ---------------------------------------------------------- #
    grid_w = ROW_HDR_W + len(xs) * (w + PAD)
    grid_h = TITLE_H + COL_HDR_H + len(ys) * (cell_h + PAD)
    sheet = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(sheet)

    # Title + original reference thumbnail.
    draw.text((PAD, 5), f"Sea-thru 2D sweep  -  {args.image}   "
              f"(columns: {args.x_param},  rows: {args.y_param})",
              fill="black", font=_font(15))
    draw.text((PAD, 26), depth_note, fill=(120, 60, 0), font=_font(11))
    orig_small = thumb(to_srgb_uint8(img_lin), ROW_HDR_W - PAD)
    sheet.paste(orig_small, (PAD, TITLE_H + COL_HDR_H))
    draw.text((PAD, TITLE_H + COL_HDR_H + orig_small.height + 2), "original",
              fill="black", font=_font(10))

    # Column headers (x-param values).
    for xi, xv in enumerate(xs):
        x0 = ROW_HDR_W + xi * (w + PAD)
        draw.text((x0 + PAD, TITLE_H + 5), f"{args.x_param} = {xv:g}",
                  fill="black", font=_font(13))

    # Rows: y-param header + tiles.
    for yi, yv in enumerate(ys):
        y0 = TITLE_H + COL_HDR_H + yi * (cell_h + PAD)
        draw.text((PAD, y0 + cell_h // 2), f"{args.y_param} = {yv:g}",
                  fill="black", font=_font(13))
        for xi in range(len(xs)):
            x0 = ROW_HDR_W + xi * (w + PAD)
            sheet.paste(tiles[(yi, xi)], (x0, y0))

    out_path = out_dir / f"{Path(args.image).stem}_{args.x_param}_x_{args.y_param}.png"
    sheet.save(out_path)
    print(f"2D grid -> {out_path}")


if __name__ == "__main__":
    main()
