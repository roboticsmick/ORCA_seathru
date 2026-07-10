"""
@file colmap_geo_from_csv.py
@brief Turn the ASV processed-images CSV into COLMAP georegistration inputs.

Produces two files from lat/lon (and heading):

  1. ``geo_ref.txt``  - ``image_name X Y Z`` in local ENU metres, for
     ``colmap model_aligner --ref_images_path geo_ref.txt`` (sets metric scale +
     orientation of the sparse model to the GPS track).

  2. ``image_list.txt`` - one image name per line, in capture order. Handy for
     chunked processing.

Z is set to 0 for every frame: the GPS is at the surface and the CSV ``depth_m``
is unreliable (-1), so we anchor scale/orientation from the horizontal track
only. That is enough to make COLMAP metric, because the along-track GPS spacing
fixes the scale.

Usage:
    python scripts/colmap_geo_from_csv.py \
        --csv ../2024_02_PALFREY/processed_images.csv \
        --out-dir ../2024_02_PALFREY/colmap
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

WGS84_A = 6378137.0            # earth radius (m), spherical approx is plenty here


def latlon_to_local_enu(lat, lon, lat0, lon0):
    """Equirectangular local tangent-plane projection (metres) about (lat0, lon0).
    Accurate to well under a metre over a single reef survey extent."""
    lat_r, lat0_r = math.radians(lat), math.radians(lat0)
    x = WGS84_A * math.radians(lon - lon0) * math.cos(0.5 * (lat_r + lat0_r))  # east
    y = WGS84_A * math.radians(lat - lat0)                                     # north
    return x, y


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.csv, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append((r["image_name"], float(r["latitude"]),
                             float(r["longitude"])))
            except (KeyError, ValueError):
                continue
    if not rows:
        raise SystemExit("No usable lat/lon rows found in CSV.")

    lat0 = sum(r[1] for r in rows) / len(rows)
    lon0 = sum(r[2] for r in rows) / len(rows)

    geo = out / "geo_ref.txt"
    lst = out / "image_list.txt"
    with open(geo, "w") as gf, open(lst, "w") as lf:
        for name, lat, lon in rows:
            x, y = latlon_to_local_enu(lat, lon, lat0, lon0)
            gf.write(f"{name} {x:.4f} {y:.4f} 0.0000\n")
            lf.write(f"{name}\n")

    print(f"Wrote {geo} and {lst} ({len(rows)} images).")
    print(f"Local origin (lat0, lon0) = ({lat0:.8f}, {lon0:.8f})")


if __name__ == "__main__":
    main()
