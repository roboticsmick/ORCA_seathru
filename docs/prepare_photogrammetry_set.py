#!/usr/bin/env python3
"""
Prepare a photogrammetry image set from an ASV mission.

Pipeline
--------
1. Parse the raw ASV CSV ($ASVSM sentences). Each row gives a UTC timestamp
   (GPS date DDMMYY + GPS time HHMMSS.s), longitude, latitude, compass heading,
   depth and the current mission waypoint.
2. Keep only the mission window: rows whose waypoint number is >= 0. Waypoint
   numbers are negative while the boat drives out to the first waypoint, so this
   crops the transit. The window is [first wp>=0 time, last wp>=0 time] (UTC).
3. Walk the GoPro image folders. Read each image's EXIF DateTimeOriginal and
   OffsetTimeOriginal (+ SubSec) to get a true UTC capture time. Any image whose
   capture time falls inside the mission window is "usable".
4. For each usable image, find the two GPS samples bracketing its timestamp and
   linearly interpolate longitude, latitude and depth. Heading is interpolated
   circularly (shortest-arc). A depth of -1 means the sonar was not confident;
   if either bracketing sample is -1 the interpolated depth is kept as -1.
5. Copy the usable images into a new output folder and write a new CSV
   describing them. Originals are never modified.

Usage
-----
    python prepare_photogrammetry_set.py \
        --csv    "C:/.../Palfrey_Raw_Data.csv" \
        --images "C:/.../Palfrey" \
        --out    "C:/.../Palfrey_processed"

Run with no arguments to use the LIRS_FEB_24_GP1 / Palfrey defaults below.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import math
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PIL import Image

# ---------------------------------------------------------------------------
# Defaults for the dataset this script was written against.
# ---------------------------------------------------------------------------
DEFAULT_CSV = r"C:\Users\mickv\OneDrive\Documents\Data\LIRS_2024\LIRS_FEB_24_GP1\Palfrey_Raw_Data.csv"
DEFAULT_IMAGES = r"C:\Users\mickv\OneDrive\Documents\Data\LIRS_2024\LIRS_FEB_24_GP1\Palfrey"
DEFAULT_OUT = r"C:\Users\mickv\OneDrive\Documents\Data\LIRS_2024\LIRS_FEB_24_GP1\Palfrey_processed"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# EXIF tag ids (from the EXIF spec / Exif sub-IFD).
TAG_DATETIME_ORIGINAL = 0x9003   # 36867 "DateTimeOriginal"
TAG_OFFSET_ORIGINAL = 0x9011     # 36881 "OffsetTimeOriginal"
TAG_SUBSEC_ORIGINAL = 0x9291     # 37521 "SubsecTimeOriginal"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class GpsSample:
    t: datetime          # UTC
    lon: float
    lat: float
    heading: float       # degrees, compass
    depth: float         # metres, or -1 == no confident reading
    waypoint: int
    mission: str


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------
def _parse_gps_datetime(date_field: str, time_field: str) -> datetime:
    """DDMMYY + HHMMSS.s (leading zeros dropped) -> aware UTC datetime."""
    d = int(date_field)
    day, month, year = d // 10000, (d // 100) % 100, d % 100
    year += 2000

    tf = float(time_field)
    whole = int(tf)
    frac = tf - whole
    hour, minute, sec = whole // 10000, (whole // 100) % 100, whole % 100
    micro = int(round(frac * 1_000_000))
    return datetime(year, month, day, hour, minute, sec, micro, tzinfo=timezone.utc)


def _strip_checksum(field: str) -> str:
    """Remove a trailing NMEA-style '*XX' checksum from a field."""
    return field.split("*", 1)[0].strip()


def parse_csv(path: Path) -> list[GpsSample]:
    samples: list[GpsSample] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        for lineno, row in enumerate(csv.reader(fh), start=1):
            if not row or not row[0].startswith("$ASVSM"):
                continue
            if len(row) < 18:
                print(f"  [warn] line {lineno}: only {len(row)} fields, skipped")
                continue
            try:
                t = _parse_gps_datetime(row[2], row[3])
                lon = float(row[4])
                lat = float(row[5])
                heading = float(row[8])
                depth = float(row[10])
                mission = row[16].strip()
                waypoint = int(_strip_checksum(row[17]))
            except (ValueError, IndexError) as exc:
                print(f"  [warn] line {lineno}: parse error ({exc}), skipped")
                continue
            samples.append(GpsSample(t, lon, lat, heading, depth, waypoint, mission))

    samples.sort(key=lambda s: s.t)
    return samples


def mission_window(samples: list[GpsSample]) -> tuple[datetime, datetime, str]:
    """Return (start, end, mission_name) covering waypoint >= 0 samples."""
    on_mission = [s for s in samples if s.waypoint >= 0]
    if not on_mission:
        raise SystemExit("No samples with waypoint >= 0 found - nothing to process.")
    start = min(s.t for s in on_mission)
    end = max(s.t for s in on_mission)
    mission = on_mission[0].mission
    return start, end, mission


# ---------------------------------------------------------------------------
# Image EXIF timestamps
# ---------------------------------------------------------------------------
def image_utc_time(path: Path) -> datetime | None:
    """UTC capture time from EXIF DateTimeOriginal + OffsetTimeOriginal (+SubSec)."""
    try:
        with Image.open(path) as img:
            exif = img._getexif() or {}
    except Exception as exc:  # noqa: BLE001 - corrupt/odd files shouldn't kill the run
        print(f"  [warn] {path.name}: cannot read EXIF ({exc})")
        return None

    dto = exif.get(TAG_DATETIME_ORIGINAL)
    if not dto:
        return None

    try:
        naive = datetime.strptime(dto, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None

    # Sub-second fraction, e.g. "1370" -> 0.1370 s
    subsec = exif.get(TAG_SUBSEC_ORIGINAL)
    if subsec:
        try:
            naive += timedelta(seconds=float(f"0.{str(subsec).strip()}"))
        except ValueError:
            pass

    # Camera UTC offset, e.g. "+10:00". Fall back to UTC if absent.
    offset = exif.get(TAG_OFFSET_ORIGINAL)
    tz = _parse_offset(offset) if offset else timezone.utc
    return naive.replace(tzinfo=tz).astimezone(timezone.utc)


def _parse_offset(offset: str) -> timezone:
    offset = offset.strip()
    sign = 1
    if offset[0] in "+-":
        sign = -1 if offset[0] == "-" else 1
        offset = offset[1:]
    hh, mm = offset.split(":")
    return timezone(sign * timedelta(hours=int(hh), minutes=int(mm)))


def find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------
def _lerp(a: float, b: float, f: float) -> float:
    return a + (b - a) * f


def _lerp_heading(a: float, b: float, f: float) -> float:
    """Circular interpolation of compass degrees along the shortest arc."""
    diff = ((b - a + 180.0) % 360.0) - 180.0
    return (a + diff * f) % 360.0


@dataclass
class InterpResult:
    lon: float
    lat: float
    heading: float
    depth: float


def interpolate_at(samples: list[GpsSample], times: list[datetime],
                   t: datetime) -> InterpResult:
    """Interpolate GPS state at time t. `times` is the sorted key list."""
    idx = bisect.bisect_left(times, t)

    # Clamp to the ends if outside coverage (shouldn't happen for usable images).
    if idx <= 0:
        s = samples[0]
        return InterpResult(s.lon, s.lat, s.heading, s.depth)
    if idx >= len(samples):
        s = samples[-1]
        return InterpResult(s.lon, s.lat, s.heading, s.depth)

    lo, hi = samples[idx - 1], samples[idx]
    span = (hi.t - lo.t).total_seconds()
    f = 0.0 if span == 0 else (t - lo.t).total_seconds() / span

    lon = _lerp(lo.lon, hi.lon, f)
    lat = _lerp(lo.lat, hi.lat, f)
    heading = _lerp_heading(lo.heading, hi.heading, f)

    # Depth: -1 means "no confident reading" - never interpolate through it.
    if lo.depth == -1 or hi.depth == -1:
        depth = -1.0
    else:
        depth = _lerp(lo.depth, hi.depth, f)

    return InterpResult(lon, lat, heading, depth)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=DEFAULT_CSV, type=Path)
    ap.add_argument("--images", default=DEFAULT_IMAGES, type=Path)
    ap.add_argument("--out", default=DEFAULT_OUT, type=Path)
    ap.add_argument("--no-copy", action="store_true",
                    help="Write the CSV only; do not copy image files.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report counts without writing or copying anything.")
    args = ap.parse_args(argv)

    print(f"Reading CSV: {args.csv}")
    samples = parse_csv(args.csv)
    if not samples:
        raise SystemExit("No $ASVSM samples parsed.")
    times = [s.t for s in samples]
    print(f"  parsed {len(samples)} GPS samples "
          f"({times[0].isoformat()} -> {times[-1].isoformat()} UTC)")

    start, end, mission = mission_window(samples)
    print(f"Mission '{mission}' window (waypoint>=0): "
          f"{start.isoformat()} -> {end.isoformat()} UTC "
          f"({(end - start).total_seconds() / 60:.1f} min)")

    print(f"Scanning images under: {args.images}")
    all_images = find_images(args.images)
    print(f"  found {len(all_images)} image files; reading EXIF...")

    usable: list[tuple[Path, datetime, InterpResult]] = []
    no_time = 0
    for i, img in enumerate(all_images, 1):
        if i % 1000 == 0:
            print(f"    ...{i}/{len(all_images)} scanned, {len(usable)} usable")
        t = image_utc_time(img)
        if t is None:
            no_time += 1
            continue
        if start <= t <= end:
            usable.append((img, t, interpolate_at(samples, times, t)))

    usable.sort(key=lambda r: r[1])
    print(f"  {len(usable)} usable images in window "
          f"({no_time} images had no readable timestamp)")

    if args.dry_run:
        print("Dry run - stopping before write/copy.")
        return 0
    if not usable:
        raise SystemExit("No usable images - nothing to write.")

    out_dir = args.out
    img_out_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_copy:
        img_out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "processed_images.csv"
    print(f"Writing {csv_path}"
          + ("" if args.no_copy else f" and copying images to {img_out_dir}"))

    copied = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_name", "image_path", "timestamp_utc",
                    "latitude", "longitude", "heading_deg", "depth_m"])
        for img, t, r in usable:
            if args.no_copy:
                out_path = img.resolve()
            else:
                out_path = img_out_dir / img.name
                # Guard against name collisions across GoPro folders.
                if out_path.exists():
                    out_path = img_out_dir / f"{img.parent.name}_{img.name}"
                shutil.copy2(img, out_path)
                copied += 1
            w.writerow([
                out_path.name,
                str(out_path),
                t.isoformat(),
                f"{r.lat:.8f}",
                f"{r.lon:.8f}",
                f"{r.heading:.2f}",
                f"{r.depth:.2f}",
            ])

    print(f"Done. {len(usable)} rows written"
          + ("" if args.no_copy else f", {copied} images copied") + ".")
    return 0


if __name__ == "__main__":
    sys.exit(main())
