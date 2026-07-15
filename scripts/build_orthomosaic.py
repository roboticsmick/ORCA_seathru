"""
@file build_orthomosaic.py
@brief True-orthomosaic GeoTIFF straight from Sea-thru-corrected images +
       COLMAP metric depth. No re-matching, no second photogrammetry run.

Why this exists
---------------
An orthomosaic pipeline (MicMac / ODM / Metashape) spends nearly all of its
runtime computing camera poses and a dense surface — but a survey processed
through this library already has both: a georegistered COLMAP model in local
ENU metres and a per-pixel metric depth map for every frame, with the
colour-corrected images pixel-aligned to them. This script just finishes the
job:

  for each corrected frame (streamed, one at a time — bounded RAM):
      back-project every pixel through the PINHOLE camera with its own
      depth  ->  3D point in ENU metres  ->  ground-grid cell
      keep, per cell, the sample with the HIGHEST elevation
      (top-of-coral wins: correct occlusion handling for nadir imagery)
  write the grid as a tiled, compressed GeoTIFF in the local UTM zone.

Because the images are survey-locked (one radiometric calibration for the
whole survey), simple best-sample selection produces seam-free colour without
feather blending. Heading never enters: the poses encode full orientation.

RAM: one frame + the output grids (~0.5 GB at 4 mm GSD for a 30x30 m site).
CPU: single process, ~1 s/frame. Both bounded by construction.

Usage
-----
    python scripts/build_orthomosaic.py \
        --corrected-dir /path/seathru_out \
        --colmap-workspace survey/colmap/dense \
        --csv survey/processed_images.csv \
        --out survey/orthomosaic.tif \
        --gsd 0.004

    # quick preview: every 4th frame, coarser grid
    ... --subsample 4 --gsd 0.01

Open the result directly in QGIS: CRS and transform are embedded (UTM zone
auto-detected from the survey's mean GPS position).
"""
from __future__ import annotations

import argparse
import csv
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seathru.depth import ColmapDepthSource            # noqa: E402
from seathru.depth.base import ImageMeta               # noqa: E402

M_PER_DEG_LAT = 111_320.0


# --------------------------------------------------------------------------- #
# Minimal COLMAP binary model readers (no external deps)                      #
# --------------------------------------------------------------------------- #
def read_cameras_bin(path):
    """@brief Parse cameras.bin -> {camera_id: (model_id, w, h, params[])}."""
    cams = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
            nparams = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12}.get(model, 4)
            params = struct.unpack("<%dd" % nparams, f.read(8 * nparams))
            cams[cid] = (model, w, h, np.array(params))
    return cams


def read_images_bin(path):
    """@brief Parse images.bin -> {name: (qvec, tvec, camera_id)} (poses only)."""
    images = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            _iid = struct.unpack("<I", f.read(4))[0]
            q = struct.unpack("<4d", f.read(32))
            t = struct.unpack("<3d", f.read(24))
            cid = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            npts = struct.unpack("<Q", f.read(8))[0]
            f.seek(24 * npts, 1)                      # skip 2D points
            images[name.decode()] = (np.array(q), np.array(t), cid)
    return images


def qvec_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])


def survey_origin(csv_path):
    """@brief Mean lat/lon of the survey CSV (the ENU origin used by
    colmap_geo_from_csv.py, so the model's XY are metres about this point)."""
    lats, lons = [], []
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                lats.append(float(r["latitude"]))
                lons.append(float(r["longitude"]))
            except (KeyError, ValueError):
                continue
    return sum(lats) / len(lats), sum(lons) / len(lons)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corrected-dir", required=True,
                    help="Sea-thru output dir (*_seathru.png or renamed .JPG)")
    ap.add_argument("--colmap-workspace", required=True,
                    help="COLMAP dense workspace (georegistered, metres)")
    ap.add_argument("--csv", required=True, help="Survey CSV (for the ENU->UTM origin)")
    ap.add_argument("--out", required=True, help="Output GeoTIFF path")
    ap.add_argument("--gsd", type=float, default=0.004,
                    help="Ground sample distance, metres/pixel (default 4 mm)")
    ap.add_argument("--margin", type=float, default=5.0,
                    help="Grid margin around the camera track, metres")
    ap.add_argument("--subsample", type=int, default=1,
                    help="Use every Nth frame (quick previews)")
    ap.add_argument("--pixel-stride", type=int, default=1,
                    help="Sample every Nth pixel of each frame (previews)")
    ap.add_argument("--depth-kind", choices=["geometric", "photometric"],
                    default="photometric")
    ap.add_argument("--max-view-z", type=float, default=8.0,
                    help="Skip samples farther than this range (m) from the camera "
                         "(kills 'radial spike' depth outliers; 0 disables)")
    ap.add_argument("--border-trim", type=int, default=15,
                    help="Ignore this many pixels around each frame's border — "
                         "MVS depth is least reliable there and border junk "
                         "splashes off-surface at strip edges (0 disables)")
    ap.add_argument("--elev-min", type=float, default=-15.0,
                    help="Reject samples below this ENU elevation (m; cameras ~0)")
    ap.add_argument("--elev-max", type=float, default=0.3,
                    help="Reject samples above this ENU elevation (above water!)")
    args = ap.parse_args(argv)

    ws = Path(args.colmap_workspace)
    cams = read_cameras_bin(ws / "sparse" / "cameras.bin")
    poses = read_images_bin(ws / "sparse" / "images.bin")
    src = ColmapDepthSource(ws, kind=args.depth_kind, clip_low_percentile=2.0,
                            fill_holes_max_frac=0.02, fill_border=False)

    # corrected frames present on disk, matched to poses
    cor = {}
    for p in sorted(Path(args.corrected_dir).iterdir()):
        stem = p.stem.replace("_seathru", "")
        cor[stem] = p
    frames = [(n, cor[Path(n).stem]) for n in sorted(poses)
              if Path(n).stem in cor][::max(1, args.subsample)]
    if not frames:
        raise SystemExit("no corrected frames match the COLMAP model")
    print(f"{len(frames)} frames (of {len(poses)} posed, {len(cor)} corrected)")

    # ---- grid bounds from camera track ----------------------------------- #
    centres = np.array([-qvec_to_R(q).T @ t for q, t, _ in poses.values()])
    xmin, ymin = centres[:, 0].min() - args.margin, centres[:, 1].min() - args.margin
    xmax, ymax = centres[:, 0].max() + args.margin, centres[:, 1].max() + args.margin
    W = int(math.ceil((xmax - xmin) / args.gsd))
    H = int(math.ceil((ymax - ymin) / args.gsd))
    print(f"grid {W} x {H} px at {args.gsd*1000:.0f} mm "
          f"({xmax-xmin:.1f} x {ymax-ymin:.1f} m) "
          f"~{(W*H*7)/1e9:.2f} GB in RAM")
    rgb = np.zeros((H, W, 3), np.uint8)
    elev = np.full((H, W), -np.inf, np.float32)

    t0 = time.time()
    for k, (name, cpath) in enumerate(frames, 1):
        q, t, cid = poses[name]
        model, cw, ch, prm = cams[cid]
        fx, fy, cx, cy = (prm[0], prm[0], prm[1], prm[2]) if model == 0 else prm[:4]
        img = np.asarray(Image.open(cpath).convert("RGB"))
        if img.shape[0] != ch or img.shape[1] != cw:
            img = np.asarray(Image.open(cpath).convert("RGB").resize((cw, ch)))
        z = src.get_depth(img.astype(np.float32) / 255.0,
                          ImageMeta(image_name=name))
        if args.border_trim:
            bt = args.border_trim
            z[:bt, :] = 0
            z[-bt:, :] = 0
            z[:, :bt] = 0
            z[:, -bt:] = 0
        s = args.pixel_stride
        zz = z[::s, ::s]
        valid = zz > 0
        if args.max_view_z:
            valid &= zz <= args.max_view_z
        if valid.sum() < 100:
            continue
        vv, uu = np.nonzero(valid)
        zs = zz[vv, uu]
        u_pix, v_pix = uu * s, vv * s
        xc = (u_pix - cx) / fx * zs
        yc = (v_pix - cy) / fy * zs
        R = qvec_to_R(q)
        Xw = (R.T @ (np.stack([xc, yc, zs]) - t[:, None]))
        gx = ((Xw[0] - xmin) / args.gsd).astype(np.int32)
        gy = ((ymax - Xw[1]) / args.gsd).astype(np.int32)
        inb = ((gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
               & (Xw[2] > args.elev_min) & (Xw[2] < args.elev_max))
        if not inb.any():
            continue
        gx, gy = gx[inb], gy[inb]
        el = Xw[2][inb].astype(np.float32)
        col = img[v_pix[inb], u_pix[inb]]
        cell = gy.astype(np.int64) * W + gx
        # best sample per cell within this frame (highest elevation)
        order = np.lexsort((-el, cell))
        cell, el, col = cell[order], el[order], col[order]
        first = np.ones(len(cell), bool)
        first[1:] = cell[1:] != cell[:-1]
        cell, el, col = cell[first], el[first], col[first]
        # merge into global grid
        cy_, cx_ = np.divmod(cell, W)
        better = el > elev[cy_, cx_]
        elev[cy_[better], cx_[better]] = el[better]
        rgb[cy_[better], cx_[better]] = col[better]
        if k % 100 == 0 or k == len(frames):
            done = (elev > -np.inf).mean()
            print(f"  [{k}/{len(frames)}] coverage {100*done:.1f}%  "
                  f"({(time.time()-t0)/k:.2f} s/frame)")

    # ---- georeference and write ------------------------------------------ #
    import rasterio
    from rasterio.transform import from_origin
    from pyproj import Transformer
    lat0, lon0 = survey_origin(args.csv)
    zone = int((lon0 + 180) // 6) + 1
    epsg = (32700 if lat0 < 0 else 32600) + zone
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    E0, N0 = tr.transform(lon0, lat0)
    transform = from_origin(E0 + xmin, N0 + ymax, args.gsd, args.gsd)
    alpha = ((elev > -np.inf) * 255).astype(np.uint8)
    with rasterio.open(
            args.out, "w", driver="GTiff", width=W, height=H, count=4,
            dtype="uint8", crs=f"EPSG:{epsg}", transform=transform,
            compress="deflate", tiled=True, photometric="RGB") as dst:
        for b in range(3):
            dst.write(rgb[..., b], b + 1)
        dst.write(alpha, 4)
        dst.colorinterp = [rasterio.enums.ColorInterp.red,
                           rasterio.enums.ColorInterp.green,
                           rasterio.enums.ColorInterp.blue,
                           rasterio.enums.ColorInterp.alpha]
    print(f"\nwrote {args.out}  (EPSG:{epsg}, {W}x{H}, "
          f"coverage {100*(elev>-np.inf).mean():.1f}%)")
    print("open in QGIS: the CRS/transform are embedded.")


if __name__ == "__main__":
    main()
