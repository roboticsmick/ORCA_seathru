"""
@file param_effect_grid.py
@brief Visualise the effect of each Sea-thru parameter, one row per parameter.

For each sampled image this builds a tall sheet: one banner + row of tiles per
tunable parameter. Within a row the parameter is swept low -> recommended ->
high while EVERYTHING ELSE stays at its default, so each row isolates exactly
what that one knob does. The original (input) image is the first tile in every
row for reference, giving four tiles per row (three for the on/off red switch).

Run it on a locked sample (shares sample_list.txt with sample_test.py):

    python scripts/param_effect_grid.py \\
        --input-dir ../2024_02_PALFREY/images \\
        --csv ../2024_02_PALFREY/processed_images.csv \\
        --out-dir ../2024_02_PALFREY/seathru_param_effects \\
        --n 10 --seed 42

Output: one `<image>_params.png` per sampled image in --out-dir.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seathru.core import SeathruParams, run_seathru          # noqa: E402
from seathru.depth import EstimatedDepthSource, PlaneDepthSource  # noqa: E402
from seathru.depth.base import ImageMeta                     # noqa: E402
from seathru.io_images import _linear_to_srgb, load_image    # noqa: E402
from seathru.metadata import load_metadata                   # noqa: E402
from seathru.pipeline import IMAGE_EXTS                       # noqa: E402

# Each entry: (title, help text, [(cell-label, {param overrides}), ...])
# The "(rec)" cell is the all-defaults result and is reused across every row.
SWEEPS = [
    ("f  -  overall brightness",
     "illuminant geometry factor; raise if the result looks dark",
     [("f = 1.5  (dimmer)", {"f": 1.5}),
      ("f = 2.0  (recommended)", {}),
      ("f = 3.0  (brighter)", {"f": 3.0})]),
    ("l  -  range-correction strength",
     "scales attenuation beta_D; lower if far/deep areas over-brighten",
     [("l = 0.5  (gentle)", {"l": 0.5}),
      ("l = 1.0  (recommended)", {}),
      ("l = 1.5  (aggressive)", {"l": 1.5})]),
    ("p  -  illuminant locality",
     "low = smoother illuminant (global), high = adapts locally",
     [("p = 0.1  (smooth)", {"p": 0.1}),
      ("p = 0.5  (recommended)", {}),
      ("p = 0.9  (local)", {"p": 0.9})]),
    ("epsilon  -  neighbourhood band width",
     "range band for iso-depth regions; smaller = finer (slower)",
     [("eps = 0.02  (fine)", {"epsilon": 0.02}),
      ("eps = 0.05  (recommended)", {}),
      ("eps = 0.10  (coarse)", {"epsilon": 0.10})]),
    ("stretch  -  output contrast",
     "percentile clip on output; wider = flatter/safer, tighter = punchier",
     [("2 / 98  (punchy)", {"stretch_pct": (2.0, 98.0)}),
      ("0.5 / 99.5  (recommended)", {}),
      ("0.1 / 99.9  (flat)", {"stretch_pct": (0.1, 99.9)})]),
    ("protect_red  -  white balance mode",
     "on = protect red from over-correction, off = pure gray-world",
     [("protect_red ON  (recommended)", {}),
      ("protect_red OFF  (gray-world)", {"protect_red": False})]),
]

LABEL_H = 22
BANNER_H = 30
PAD = 6


def get_or_create_sample(input_dir, out_dir, n, seed):
    list_path = out_dir / "sample_list.txt"
    if list_path.exists():
        sample = [ln.strip() for ln in list_path.read_text().splitlines() if ln.strip()]
        print(f"Reusing cached sample of {len(sample)} images.")
        return sample
    names = sorted(p.name for p in input_dir.iterdir()
                   if p.suffix.lower() in IMAGE_EXTS)
    random.seed(seed)
    sample = sorted(random.sample(names, min(n, len(names))))
    out_dir.mkdir(parents=True, exist_ok=True)
    list_path.write_text("\n".join(sample) + "\n")
    print(f"Locked sample of {len(sample)} of {len(names)} images (seed {seed}).")
    return sample


def _font(size=13):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def to_srgb_uint8(img_lin):
    return (_linear_to_srgb(img_lin) * 255 + 0.5).astype(np.uint8)


def labelled_tile(arr_uint8, text, width):
    img = Image.fromarray(arr_uint8).convert("RGB")
    h = int(img.height * (width / img.width))
    img = img.resize((width, h), Image.LANCZOS)
    tile = Image.new("RGB", (width, LABEL_H + h), "white")
    draw = ImageDraw.Draw(tile)
    draw.text((PAD, 4), text, fill="black", font=_font(12))
    tile.paste(img, (0, LABEL_H))
    return tile


def banner(text, subtext, width):
    b = Image.new("RGB", (width, BANNER_H), (32, 40, 52))
    draw = ImageDraw.Draw(b)
    draw.text((PAD, 3), text, fill="white", font=_font(14))
    draw.text((PAD, 16), subtext, fill=(170, 185, 205), font=_font(11))
    return b


def build_sheet(name, img_lin, runner, thumb_w, depth_note):
    """runner(overrides_dict) -> recovered linear image (memoised by caller)."""
    orig_uint8 = to_srgb_uint8(img_lin)
    rows = []
    for title, subtext, cells in SWEEPS:
        tiles = [labelled_tile(orig_uint8, "original (input)", thumb_w)]
        for label, override in cells:
            rec_lin = runner(override)
            tiles.append(labelled_tile(to_srgb_uint8(rec_lin), label, thumb_w))
        row_w = thumb_w * len(tiles) + PAD * (len(tiles) - 1)
        row_h = max(t.height for t in tiles)
        row = Image.new("RGB", (row_w, BANNER_H + row_h), "white")
        row.paste(banner(title, subtext, row_w), (0, 0))
        x = 0
        for t in tiles:
            row.paste(t, (x, BANNER_H))
            x += thumb_w + PAD
        rows.append(row)

    sheet_w = max(r.width for r in rows)
    head_h = 42
    total_h = head_h + sum(r.height for r in rows) + PAD * len(rows)
    sheet = Image.new("RGB", (sheet_w, total_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((PAD, 5), f"Sea-thru parameter effects  -  {name}",
              fill="black", font=_font(15))
    draw.text((PAD, 24), depth_note, fill=(120, 60, 0), font=_font(11))
    y = head_h
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + PAD
    return sheet


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image", default=None,
                    help="Process only this one image (filename), ignoring the "
                         "sample - best for comparing parameters on a fixed scene")
    ap.add_argument("--max-size", type=int, default=800,
                    help="Working resolution (smaller = faster sweep)")
    ap.add_argument("--thumb-width", type=int, default=360)
    ap.add_argument("--depth", choices=["estimated", "plane"], default="estimated",
                    help="Depth for the sweep. 'estimated' (default) is a "
                         "spatially-varying image prior so f/l/p/epsilon are "
                         "active; 'plane' is flat and only exercises "
                         "stretch/protect_red.")
    ap.add_argument("--plane-default", type=float, default=5.0)
    ap.add_argument("--est-near", type=float, default=1.0)
    ap.add_argument("--est-far", type=float, default=10.0)
    args = ap.parse_args(argv)

    input_dir, out_dir = Path(args.input_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_map = load_metadata(args.csv) if args.csv else {}
    if args.image:
        sample = [args.image]
        print(f"Single-image mode: {args.image}")
    else:
        sample = get_or_create_sample(input_dir, out_dir, args.n, args.seed)

    if args.depth == "estimated":
        source = EstimatedDepthSource(near=args.est_near, far=args.est_far)
        depth_note = ("depth = image-derived prior (red-attenuation cue); "
                      "spatially varying so every parameter is active. "
                      "Use metric COLMAP depth for final results.")
    else:
        source = PlaneDepthSource(default_m=args.plane_default)
        depth_note = ("depth = flat plane; only stretch & protect_red change "
                      "the result (f/l/p/epsilon need varying depth).")
    base = SeathruParams()

    for i, name in enumerate(sample, 1):
        t0 = time.time()
        img_lin, _ = load_image(input_dir / name, max_size=args.max_size)
        meta = meta_map.get(name, ImageMeta(image_name=name))
        depths = source.get_depth(img_lin, meta)

        cache = {}

        def runner(override, _img=img_lin, _z=depths, _cache=cache):
            key = json.dumps(override, sort_keys=True)
            if key not in _cache:
                params = replace(base, **override)
                _cache[key] = run_seathru(_img, _z, params).recovered
            return _cache[key]

        sheet = build_sheet(name, img_lin, runner, args.thumb_width, depth_note)
        sheet.save(out_dir / f"{Path(name).stem}_params.png")
        print(f"  [{i}/{len(sample)}] {name}  ({time.time() - t0:.1f}s, "
              f"{len(cache)} runs)")

    print(f"\nParameter-effect sheets -> {out_dir}")


if __name__ == "__main__":
    main()
