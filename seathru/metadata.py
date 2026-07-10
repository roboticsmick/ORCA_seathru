"""
@file metadata.py
@brief Read a per-image survey CSV into ImageMeta records.

Expected columns:
    image_name,image_path,timestamp_utc,latitude,longitude,heading_deg,depth_m

``depth_m == -1`` denotes a low-confidence reading not reported by the sensor;
it is stored as ``None`` so depth sources can fall back appropriately.

@author Michael Venz
"""
from __future__ import annotations

import csv
from pathlib import Path

from .depth.base import ImageMeta


def _f(row, key):
    """@brief Parse a CSV field as float, tolerating missing/blank/non-numeric values.
    @param row A csv.DictReader row.
    @param key Column name to read.
    @return float value, or None if missing/unparseable."""
    try:
        v = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return v


def load_metadata(csv_path):
    """
    @brief Load the processed-images CSV into a lookup table.
    @param csv_path Path to the CSV (see module docstring for expected columns).
    @return ``{image_name: ImageMeta}`` dict, keyed by file name (basename).
    """
    out = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            name = row.get("image_name") or Path(row.get("image_path", "")).name
            depth = _f(row, "depth_m")
            if depth is not None and depth <= 0:  # -1 sentinel -> unknown
                depth = None
            out[name] = ImageMeta(
                image_name=name,
                depth_m=depth,
                heading_deg=_f(row, "heading_deg"),
                latitude=_f(row, "latitude"),
                longitude=_f(row, "longitude"),
            )
    return out
