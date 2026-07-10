#!/usr/bin/env python3
"""
@file make_test_subset.py
@brief Cut a small spatial test dataset out of a big ASV survey.

Given a seed image (or an explicit lat/lon) and a radius in metres, this
script finds every image in the survey CSV whose camera GPS position lies
within that radius, copies those images into ``<out-dir>/images/``, and
writes a matching ``processed_images.csv`` next to them containing only the
selected rows. The result is a self-contained mini-survey that drops straight
into the normal pipeline (COLMAP -> Sea-thru -> splat) for a quick end-to-end
test. Originals are never modified.

Distances are computed with a local equirectangular approximation, which is
accurate to well under a centimetre at the tens-of-metres scale this is
meant for.

Usage
-----
    python make_test_subset.py \
        --image G0022406.JPG --radius 2 \
        --out-dir 2024_02_PALFREY_TEST_R2

    # or seed by coordinates instead of an image:
    python make_test_subset.py --lat -14.699 --lon 145.447 --radius 3 --out-dir ...

Run with ``--dry-run`` first to see how many images the radius captures
without copying anything.

@author Michael Venz
@date 2026
"""
from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from pathlib import Path

## Default survey this tool was built around (override with --csv/--images-dir).
DEFAULT_CSV = Path(__file__).parent / "2024_02_PALFREY" / "processed_images.csv"
DEFAULT_IMAGES = Path(__file__).parent / "2024_02_PALFREY" / "images"

## Metres per degree of latitude (WGS-84 mean); longitude is scaled by cos(lat).
M_PER_DEG_LAT = 111_320.0


def local_offset_m(lat, lon, lat0, lon0):
    """
    @brief Convert a lat/lon pair to (east, north) metres from a reference point.
    @param lat, lon Point of interest, decimal degrees.
    @param lat0, lon0 Reference point, decimal degrees.
    @return Tuple ``(dx_east_m, dy_north_m)``.
    """
    dx = (lon - lon0) * M_PER_DEG_LAT * math.cos(math.radians(lat0))
    dy = (lat - lat0) * M_PER_DEG_LAT
    return dx, dy


def load_rows(csv_path):
    """
    @brief Read the survey CSV, keeping only rows with a parseable GPS fix.
    @param csv_path Path to the processed-images CSV
        (columns: image_name,image_path,timestamp_utc,latitude,longitude,heading_deg,depth_m).
    @return Tuple ``(fieldnames, rows)`` where each row is the raw dict plus
        float ``_lat``/``_lon`` keys.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        for row in reader:
            try:
                row["_lat"] = float(row["latitude"])
                row["_lon"] = float(row["longitude"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(row)
    return fieldnames, rows


def main(argv=None):
    """
    @brief Entry point: select, copy, and re-CSV a radius-bounded image subset.
    @param argv Optional argument list (defaults to ``sys.argv[1:]``).
    @return Process exit code (0 on success, 2 on bad arguments).
    """
    ap = argparse.ArgumentParser(
        description="Extract all survey images within X metres of a seed point "
                    "into a standalone mini-dataset (images/ + filtered CSV).")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                    help=f"Survey CSV (default: {DEFAULT_CSV})")
    ap.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES,
                    help=f"Folder holding the survey images (default: {DEFAULT_IMAGES})")
    ap.add_argument("--image", default=None,
                    help="Seed image name, e.g. G0022406.JPG (its GPS position becomes the centre)")
    ap.add_argument("--lat", type=float, default=None, help="Seed latitude (instead of --image)")
    ap.add_argument("--lon", type=float, default=None, help="Seed longitude (instead of --image)")
    ap.add_argument("--radius", type=float, default=2.0,
                    help="Selection radius in metres around the seed (default 2)")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output dataset folder (created; gets images/ + processed_images.csv)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be selected without copying anything")
    args = ap.parse_args(argv)

    fieldnames, rows = load_rows(args.csv)
    if not rows:
        print(f"ERROR: no rows with GPS in {args.csv}", file=sys.stderr)
        return 2

    # Resolve the seed point.
    if args.image:
        seed = next((r for r in rows if r["image_name"] == args.image), None)
        if seed is None:
            print(f"ERROR: seed image {args.image!r} not found in {args.csv}", file=sys.stderr)
            return 2
        lat0, lon0 = seed["_lat"], seed["_lon"]
    elif args.lat is not None and args.lon is not None:
        lat0, lon0 = args.lat, args.lon
    else:
        print("ERROR: give either --image NAME or both --lat and --lon", file=sys.stderr)
        return 2

    # Select rows within the radius.
    selected = []
    for row in rows:
        dx, dy = local_offset_m(row["_lat"], row["_lon"], lat0, lon0)
        if math.hypot(dx, dy) <= args.radius:
            selected.append(row)

    if not selected:
        print(f"No images within {args.radius} m of ({lat0:.6f}, {lon0:.6f}).")
        return 2

    n_depth = sum(1 for r in selected
                  if r.get("depth_m") not in (None, "") and float(r["depth_m"]) > 0)
    xs, ys = zip(*(local_offset_m(r["_lat"], r["_lon"], lat0, lon0) for r in selected))
    print(f"Seed:      ({lat0:.7f}, {lon0:.7f})"
          + (f"  [{args.image}]" if args.image else ""))
    print(f"Selected:  {len(selected)} images within {args.radius} m "
          f"(extent {max(xs)-min(xs):.1f} x {max(ys)-min(ys):.1f} m)")
    print(f"Depth:     {n_depth}/{len(selected)} rows have a valid sonar depth_m")

    missing = [r["image_name"] for r in selected
               if not (args.images_dir / r["image_name"]).exists()]
    if missing:
        print(f"WARNING: {len(missing)} selected images not found in {args.images_dir} "
              f"(first few: {missing[:3]})")

    if args.dry_run:
        print("Dry run - nothing copied.")
        return 0

    # Copy images and write the filtered CSV.
    out_images = args.out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    for i, row in enumerate(selected, 1):
        src = args.images_dir / row["image_name"]
        if not src.exists():
            continue
        dst = out_images / row["image_name"]
        if not dst.exists():
            shutil.copy2(src, dst)
        total_bytes += dst.stat().st_size
        if i % 25 == 0 or i == len(selected):
            print(f"  copied {i}/{len(selected)} ...")

    out_csv = args.out_dir / "processed_images.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            clean = {k: v for k, v in row.items() if not k.startswith("_")}
            clean["image_path"] = str((out_images / row["image_name"]).resolve())
            writer.writerow(clean)

    print(f"\nDone: {len(selected) - len(missing)} images "
          f"({total_bytes / 1e6:.0f} MB) -> {out_images}")
    print(f"CSV:  {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
