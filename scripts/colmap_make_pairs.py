"""
@file colmap_make_pairs.py
@brief Build a COLMAP custom match-pair list from the ASV CSV (GPS + time).

This is the single biggest lever for running COLMAP on a laptop: instead of
matching all N*(N-1)/2 image pairs (10k images -> ~53 million pairs), we match
only pairs that could plausibly overlap:

  * temporal neighbours  - frames within +/- ``--seq`` positions in capture order
    (the along-track overlap of the survey line);
  * spatial neighbours    - frames whose GPS position is within ``--radius`` metres
    (the cross-track overlap between adjacent survey lines).

The result is a ``pairs.txt`` you feed to COLMAP's custom matcher:

    colmap matches_importer --database_path database.db \
        --match_list_path pairs.txt --match_type pairs

Note on heading: for a **downward-facing** camera the seabed footprint overlaps
regardless of boat heading (heading only rotates the image in-plane), so we do
NOT filter pairs by heading by default - that would wrongly drop good matches.
``--max-heading-diff`` is offered for forward/oblique cameras only.

Usage:
    python scripts/colmap_make_pairs.py \
        --csv ../2024_02_PALFREY/processed_images.csv \
        --out ../2024_02_PALFREY/colmap/pairs.txt \
        --seq 12 --radius 4.0
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

WGS84_A = 6378137.0


def latlon_to_local_enu(lat, lon, lat0, lon0):
    lat_r, lat0_r = math.radians(lat), math.radians(lat0)
    x = WGS84_A * math.radians(lon - lon0) * math.cos(0.5 * (lat_r + lat0_r))
    y = WGS84_A * math.radians(lat - lat0)
    return x, y


def load_rows(csv_path):
    rows = []
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append(dict(
                    name=r["image_name"],
                    lat=float(r["latitude"]),
                    lon=float(r["longitude"]),
                    heading=float(r.get("heading_deg", "nan") or "nan"),
                ))
            except (KeyError, ValueError):
                continue
    return rows


def heading_diff(a, b):
    if math.isnan(a) or math.isnan(b):
        return 0.0
    d = abs((a - b + 180.0) % 360.0 - 180.0)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=12,
                    help="Match each frame to +/- this many temporal neighbours")
    ap.add_argument("--radius", type=float, default=4.0,
                    help="Also match frames within this many metres (cross-track)")
    ap.add_argument("--max-neighbors", type=int, default=None,
                    help="Cap spatial neighbours per image to the nearest N "
                         "(bounds match count / RAM on dense surveys)")
    ap.add_argument("--max-heading-diff", type=float, default=None,
                    help="Optional: only pair frames whose heading differs by less "
                         "than this (deg). Leave unset for downward cameras.")
    args = ap.parse_args(argv)

    rows = load_rows(args.csv)
    if not rows:
        raise SystemExit("No usable rows in CSV.")
    lat0 = sum(r["lat"] for r in rows) / len(rows)
    lon0 = sum(r["lon"] for r in rows) / len(rows)
    xs, ys = [], []
    for r in rows:
        x, y = latlon_to_local_enu(r["lat"], r["lon"], lat0, lon0)
        xs.append(x); ys.append(y)

    n = len(rows)
    r2 = args.radius ** 2
    pairs = set()

    # Temporal neighbours (cheap, always overlap along the survey line).
    for i in range(n):
        for j in range(i + 1, min(i + args.seq + 1, n)):
            pairs.add((i, j))

    # Spatial neighbours via a coarse grid bucket (avoids O(N^2) distance loop).
    cell = max(args.radius, 1e-6)
    grid = {}
    for i in range(n):
        gx, gy = int(xs[i] // cell), int(ys[i] // cell)
        grid.setdefault((gx, gy), []).append(i)
    for i in range(n):
        gx, gy = int(xs[i] // cell), int(ys[i] // cell)
        cand = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((gx + dx, gy + dy), ()):
                    if j == i:
                        continue
                    d2 = (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2
                    if d2 > r2:
                        continue
                    if (args.max_heading_diff is not None and
                            heading_diff(rows[i]["heading"], rows[j]["heading"])
                            > args.max_heading_diff):
                        continue
                    cand.append((d2, j))
        if args.max_neighbors is not None:
            cand = sorted(cand)[:args.max_neighbors]
        for _, j in cand:
            pairs.add((min(i, j), max(i, j)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for i, j in sorted(pairs):
            fh.write(f"{rows[i]['name']} {rows[j]['name']}\n")

    avg = 2 * len(pairs) / n
    print(f"Wrote {len(pairs):,} pairs to {out} "
          f"({avg:.1f} matches/image avg, vs {n - 1:,} for exhaustive).")


if __name__ == "__main__":
    main()
