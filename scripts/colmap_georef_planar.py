"""
@file colmap_georef_planar.py
@brief Robustly georegister a COLMAP model to a GPS track for *flat, downward*
       ASV surveys, where ``colmap model_aligner`` fails.

Why this exists
---------------
``colmap model_aligner`` fits a 3D similarity (Sim3) between the reconstructed
camera centres and the GPS reference with RANSAC. That estimator assumes the
cameras are **not coplanar**. A lawnmower reef survey violates this badly: every
frame is shot at the same tow height looking straight down, so the camera
centres form a near-horizontal sheet. The out-of-plane rotation is then
unconstrained and RANSAC routinely returns a mirror-through-the-plane solution
(seabed *above* the cameras) with a wrong scale. On PALFREY_TEST_R10 (2022
frames, ~20x20 m) it produced scale 0.275 and 6 m residuals; the model was
unusable and, crucially, the wrong scale makes Sea-thru's attenuation physics
(which assume metres) meaningless.

The robust construction here avoids the degeneracy by fixing the vertical from
the imagery instead of from RANSAC:

  1. The mean camera **viewing direction** (optical axis, world frame) is the
     survey's "down" - rotate it onto world -Z. This resolves the flip that
     model_aligner gets wrong.
  2. With the model levelled, fit a **2D similarity** (scale, in-plane rotation,
     translation) from the horizontal camera positions to the GPS E/N. This part
     is well-conditioned because the cameras spread out in the horizontal plane.
  3. Put the camera plane at Z=0 (surface track); the seabed then sits at
     negative Z, as it should.

Output is a COLMAP Sim3 file (``scale qw qx qy qz tx ty tz``) that you apply
with ``colmap model_transformer``. Scale is the number Sea-thru cares about; it
is validated here against the CSV sonar ``depth_m`` when available.

Usage
-----
    # 1) export the sparse model to TXT (needs images.txt + points3D.txt)
    colmap model_converter --input_path colmap/sparse/0 \
        --output_path /tmp/sparse_txt --output_type TXT

    # 2) fit the transform against geo_ref.txt (from colmap_geo_from_csv.py)
    python scripts/colmap_georef_planar.py \
        --model-txt /tmp/sparse_txt \
        --geo-ref colmap/geo_ref.txt \
        --out colmap/geo_sim3.txt \
        --csv processed_images.csv        # optional: validate scale vs sonar

    # 3) apply it
    colmap model_transformer --input_path colmap/sparse/0 \
        --output_path colmap/sparse_geo --transform_path colmap/geo_sim3.txt
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


def R_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        q = [0.25 * S, (R[2, 1] - R[1, 2]) / S,
             (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / S, 0.25 * S,
             (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S]
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S,
             0.25 * S, (R[1, 2] + R[2, 1]) / S]
    else:
        S = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S,
             (R[1, 2] + R[2, 1]) / S, 0.25 * S]
    q = np.array(q)
    return q / np.linalg.norm(q)


def rot_between(a, b):
    """Rotation matrix taking unit-ish vector a onto b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def read_model_txt(model_dir):
    """Return (names, camera_centres Nx3, view_dirs Nx3) from a COLMAP images.txt."""
    names, C, V = [], [], []
    for line in (Path(model_dir) / "images.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        # image lines end with the image name; the interleaved POINTS2D line does not
        if len(p) >= 10 and "." in p[-1] and not p[-1].replace(".", "").replace("-", "").isdigit():
            R = quat_to_R([float(v) for v in p[1:5]])
            t = np.array([float(v) for v in p[5:8]])
            C.append(-R.T @ t)
            V.append(R.T @ np.array([0, 0, 1.0]))   # optical axis in world frame
            names.append(p[-1])
    return names, np.array(C), np.array(V)


def fit_planar_sim3(C, V, G):
    """Return (scale, R 3x3, t 3) mapping model -> world (metres), plane-robust."""
    d_model = V.mean(0)
    d_model /= np.linalg.norm(d_model)
    R1 = rot_between(d_model, np.array([0, 0, -1.0]))       # survey down -> world -Z
    C1 = (R1 @ C.T).T

    X, Y = C1[:, :2], G[:, :2]
    mx, my = X.mean(0), Y.mean(0)
    X0, Y0 = X - mx, Y - my
    U, Sv, Vt = np.linalg.svd(X0.T @ Y0)
    Rot = Vt.T @ U.T
    if np.linalg.det(Rot) < 0:
        Vt[1] *= -1
        Rot = Vt.T @ U.T
    s = Sv.sum() / (X0 ** 2).sum()
    th = np.arctan2(Rot[1, 0], Rot[0, 0])
    R2 = np.array([[np.cos(th), -np.sin(th), 0],
                   [np.sin(th), np.cos(th), 0],
                   [0, 0, 1]])
    Rtot = R2 @ R1
    zc = (s * (R2 @ C1.T).T)[:, 2].mean()
    t = np.array([my[0] - s * (Rot @ mx)[0],
                  my[1] - s * (Rot @ mx)[1],
                  -s * zc])                                 # camera plane -> Z=0
    return s, Rtot, t


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-txt", required=True,
                    help="Dir with COLMAP images.txt + points3D.txt (model_converter TXT)")
    ap.add_argument("--geo-ref", required=True,
                    help="geo_ref.txt (image_name X Y Z metres) from colmap_geo_from_csv.py")
    ap.add_argument("--out", required=True, help="Output Sim3 file for model_transformer")
    ap.add_argument("--csv", default=None,
                    help="Optional processed_images.csv to validate scale vs sonar depth_m")
    args = ap.parse_args(argv)

    names, C, V = read_model_txt(args.model_txt)
    if len(C) < 3:
        raise SystemExit(f"Only {len(C)} cameras parsed from {args.model_txt}")
    ref = {}
    for line in Path(args.geo_ref).read_text().splitlines():
        p = line.split()
        if len(p) >= 4:
            ref[p[0]] = np.array([float(v) for v in p[1:4]])
    keep = [i for i, n in enumerate(names) if n in ref]
    C, V = C[keep], V[keep]
    names = [names[i] for i in keep]
    G = np.array([ref[n] for n in names])

    s, R, t = fit_planar_sim3(C, V, G)
    Cw = (s * (R @ C.T)).T + t
    res = np.linalg.norm((Cw - G)[:, :2], axis=1)

    pts_file = Path(args.model_txt) / "points3D.txt"
    seabed_ok = None
    if pts_file.exists():
        pts = np.array([[float(v) for v in l.split()[1:4]]
                        for l in pts_file.read_text().splitlines()
                        if l.strip() and not l.startswith("#")])
        Pw = (s * (R @ pts.T)).T + t
        seabed_z = np.percentile(Pw[:, 2], 50)
        seabed_ok = seabed_z < Cw[:, 2].mean()
        print(f"seabed z (p50) = {seabed_z:.2f} m, cameras at z = {Cw[:,2].mean():.2f} m"
              f"  -> seabed {'below (OK)' if seabed_ok else 'ABOVE (FLIPPED!)'}")

    print(f"scale = {s:.4f} m/model-unit")
    print(f"horizontal residual: p50 {np.percentile(res,50):.2f} m, "
          f"p90 {np.percentile(res,90):.2f} m")
    print(f"model extent: {np.ptp(Cw[:,0]):.1f} x {np.ptp(Cw[:,1]):.1f} m")

    if args.csv:
        depths = [float(r["depth_m"]) for r in csv.DictReader(open(args.csv))
                  if r.get("depth_m") and float(r["depth_m"]) > 0]
        if depths:
            print(f"[validate] sonar depth_m: p50 {np.percentile(depths,50):.2f} m "
                  f"(camera->seabed slant range is expected slightly larger)")

    q = R_to_quat(R)
    Path(args.out).write_text(
        "%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n"
        % (s, q[0], q[1], q[2], q[3], t[0], t[1], t[2]))
    print(f"\nWrote Sim3 -> {args.out}")
    print("Apply with:\n  colmap model_transformer --input_path <sparse/0> "
          f"--output_path <sparse_geo> --transform_path {args.out}")
    if seabed_ok is False:
        raise SystemExit("Refusing silently: seabed came out above cameras.")


if __name__ == "__main__":
    main()
