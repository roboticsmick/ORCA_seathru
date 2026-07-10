"""
@file synthetic_validation.py
@brief End-to-end synthetic validation of the full Sea-thru pipeline.

Builds a synthetic seafloor scene J with six embedded gray patches, degrades it
with the paper's forward model (Equation 2) over a spatially varying range map,
then runs the *full* pipeline (backscatter fit, illuminant, beta_D refinement,
recovery) and scores raw vs recovered with the paper's evaluation metric
(Equation 18: mean RGB angular error of gray patches, in degrees).

Pass criterion: recovered angular error is a large improvement over raw and in
the ballpark the paper reports for Sea-thru (~4-7 deg on real data).

Usage:
    python scripts/synthetic_validation.py [--out-dir out] [--seed 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seathru.core import SeathruParams, run_seathru  # noqa: E402


# --------------------------------------------------------------------------- #
# Scene construction                                                           #
# --------------------------------------------------------------------------- #
GRAY_LEVELS = (0.05, 0.15, 0.30, 0.50, 0.70, 0.90)   # six achromatic patches


def make_scene(size=384, seed=0):
    """Return (J, z, patch_slices): reflectance image in [0,1], range map (m)."""
    rng = np.random.default_rng(seed)
    H = W = size

    # Reef-like reflectance: smooth colour blobs + texture + dark shadows.
    def smooth_noise(scale):
        small = rng.random((scale, scale, 3))
        img = np.kron(small, np.ones((H // scale, W // scale, 1)))
        return img[:H, :W]

    J = 0.45 * smooth_noise(8) + 0.35 * smooth_noise(24) + 0.20 * rng.random((H, W, 3))
    J = 0.08 + 0.8 * J                       # keep off pure black/white
    # Shadowed regions (needed by the dark-pixel backscatter search).
    for _ in range(12):
        cy, cx = rng.integers(0, H), rng.integers(0, W)
        r = rng.integers(8, 22)
        yy, xx = np.ogrid[:H, :W]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
        J[mask] *= 0.06

    # Six gray patches along the mid row (the "colour chart").
    patch = size // 16
    gap = size // 10
    y0 = H // 2 - patch // 2
    patch_slices = []
    for i, g in enumerate(GRAY_LEVELS):
        x0 = gap // 2 + i * (patch + gap)
        sl = (slice(y0, y0 + patch), slice(x0, x0 + patch))
        J[sl] = g
        patch_slices.append(sl)

    # Range map: sloping seafloor 1.5 -> 8 m with mild relief.
    xs = np.linspace(0, 1, W)[None, :]
    ys = np.linspace(0, 1, H)[:, None]
    z = 1.5 + 6.5 * xs + 0.4 * np.sin(6.28 * 3 * ys) * xs
    z = np.broadcast_to(z, (H, W)).copy()
    return J, z, patch_slices


def forward_model(J, z):
    """Equation 2 with plausible clear-reef wideband coefficients."""
    # beta_D(z) per channel as two-term exponentials (Eq. 11 form).
    bD = np.stack([
        0.60 * np.exp(-0.20 * z) + 0.25 * np.exp(-0.02 * z),   # R attenuates hard
        0.14 * np.exp(-0.10 * z) + 0.07 * np.exp(-0.01 * z),   # G
        0.16 * np.exp(-0.10 * z) + 0.09 * np.exp(-0.01 * z),   # B
    ], axis=2)
    B_inf = np.array([0.07, 0.20, 0.28])
    beta_B = np.array([0.9, 1.2, 1.4])
    direct = J * np.exp(-bD * z[..., None])
    backscatter = B_inf * (1.0 - np.exp(-beta_B * z[..., None]))
    return np.clip(direct + backscatter, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Equation 18 metric                                                           #
# --------------------------------------------------------------------------- #
def angular_error_deg(img, patch_slices):
    """Mean angle (deg) between gray-patch RGB triplets and the gray direction."""
    errs = []
    for sl in patch_slices:
        rgb = img[sl].reshape(-1, 3).mean(axis=0)
        cos = rgb.sum() / (np.sqrt(3.0) * np.linalg.norm(rgb) + 1e-12)
        errs.append(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    return float(np.mean(errs))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None, help="Save a comparison panel here")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    np.random.seed(args.seed)
    J, z, patches = make_scene(seed=args.seed)
    I = forward_model(J, z)

    params = SeathruParams(return_debug=True)
    result = run_seathru(I, z, params)

    err_raw = angular_error_deg(I, patches)
    err_rec = angular_error_deg(result.recovered, patches)
    err_truth = angular_error_deg(J, patches)   # ~0 by construction

    print(f"Angular error (Eq. 18), mean over {len(patches)} gray patches:")
    print(f"  ground truth J : {err_truth:6.2f} deg  (sanity: ~0)")
    print(f"  raw underwater : {err_raw:6.2f} deg")
    print(f"  Sea-thru       : {err_rec:6.2f} deg")
    improved = err_rec < 0.5 * err_raw
    print(f"\n{'PASS' if improved else 'FAIL'}: recovered error "
          f"{'<' if improved else '>='} 50% of raw "
          f"(paper reports ~4-7 deg for Sea-thru on real data)")

    if args.out_dir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, (title, im) in zip(axes, [
                (f"Ground truth J ({err_truth:.1f} deg)", J),
                (f"Underwater I, Eq. 2 ({err_raw:.1f} deg)", I),
                (f"Sea-thru recovered ({err_rec:.1f} deg)", result.recovered)]):
            ax.imshow(im)
            ax.set_title(title)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / "synthetic_validation.png", dpi=100)
        print(f"Panel saved to {out / 'synthetic_validation.png'}")

    return 0 if improved else 1


if __name__ == "__main__":
    sys.exit(main())
