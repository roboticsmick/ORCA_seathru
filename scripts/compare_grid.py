"""
@file compare_grid.py
@brief Build a side-by-side comparison grid from sample_test.py runs.

Reads `sample_list.txt` and the `originals/` + `<tag>/` folders written by
`sample_test.py` and lays out one contact sheet per sampled image: original
first, then one recovered thumbnail per tag (in the order given, or every tag
found under --out-dir), each labelled with its tag name and the parameters
used (from `<tag>/params.json`). Also writes one big stacked sheet with every
sampled image, for a single-glance overview.

Usage:
    python scripts/compare_grid.py --out-dir ../2024_02_PALFREY/seathru_sweep
    python scripts/compare_grid.py --out-dir ../2024_02_PALFREY/seathru_sweep \\
        --tags baseline f2.5 l0.5_stretch_wide --thumb-width 500
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SAMPLE_LIST_NAME = "sample_list.txt"
ORIGINALS_DIR = "originals"
LABEL_H = 26
PAD = 6


def discover_tags(out_dir: Path):
    tags = [p.name for p in out_dir.iterdir()
            if p.is_dir() and p.name != ORIGINALS_DIR
            and (p / "params.json").exists()]
    return sorted(tags)


def load_thumb(path: Path, width: int):
    img = Image.open(path).convert("RGB")
    h = int(img.height * (width / img.width))
    return img.resize((width, h), Image.LANCZOS)


def label_text(tag, params_path):
    if not params_path.exists():
        return tag
    p = json.loads(params_path.read_text())
    return (f"{tag}  (f={p.get('f')} l={p.get('l')} p={p.get('p')} "
            f"eps={p.get('epsilon')} stretch={p.get('stretch_pct')})")


def tile_with_label(img, text, width):
    tile = Image.new("RGB", (width, LABEL_H + img.height), "white")
    draw = ImageDraw.Draw(tile)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((PAD, 6), text, fill="black", font=font)
    tile.paste(img, (0, LABEL_H))
    return tile


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True,
                    help="The sweep root passed as --out-dir to sample_test.py")
    ap.add_argument("--tags", nargs="*", default=None,
                    help="Tags to include, in display order (default: all found)")
    ap.add_argument("--thumb-width", type=int, default=420)
    ap.add_argument("--grid-out", default=None,
                    help="Output filename (default: <out-dir>/comparison_grid.png)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    sample_list = out_dir / SAMPLE_LIST_NAME
    if not sample_list.exists():
        raise SystemExit(f"No {SAMPLE_LIST_NAME} in {out_dir} - run sample_test.py first.")
    names = [ln.strip() for ln in sample_list.read_text().splitlines() if ln.strip()]

    tags = args.tags or discover_tags(out_dir)
    if not tags:
        raise SystemExit(f"No tagged runs found under {out_dir}.")
    print(f"Images: {len(names)}   Tags: {tags}")

    w = args.thumb_width
    rows = []
    for name in names:
        stem = Path(name).stem
        tiles = [tile_with_label(
            load_thumb(out_dir / ORIGINALS_DIR / f"{stem}.png", w),
            "original", w)]
        for tag in tags:
            img_path = out_dir / tag / f"{stem}.png"
            if not img_path.exists():
                print(f"  [skip] {tag}/{stem}.png missing")
                continue
            text = label_text(tag, out_dir / tag / "params.json")
            tiles.append(tile_with_label(load_thumb(img_path, w), text, w))

        row_h = max(t.height for t in tiles)
        row = Image.new("RGB", (w * len(tiles) + PAD * (len(tiles) - 1), row_h), "white")
        x = 0
        for t in tiles:
            row.paste(t, (x, 0))
            x += w + PAD
        rows.append((stem, row))

    grid_out = Path(args.grid_out) if args.grid_out else out_dir / "comparison_grid.png"
    total_h = sum(r.height for _, r in rows) + PAD * (len(rows) - 1)
    grid_w = max(r.width for _, r in rows)
    grid = Image.new("RGB", (grid_w, total_h), "white")
    y = 0
    for stem, row in rows:
        grid.paste(row, (0, y))
        y += row.height + PAD
    grid.save(grid_out)
    print(f"\nComparison grid ({len(names)} images x {len(tags) + 1} columns) -> {grid_out}")

    # Also save one grid per image for full-resolution review.
    per_image_dir = out_dir / "comparison_per_image"
    per_image_dir.mkdir(exist_ok=True)
    for stem, row in rows:
        row.save(per_image_dir / f"{stem}_compare.png")
    print(f"Per-image rows -> {per_image_dir}")


if __name__ == "__main__":
    main()
