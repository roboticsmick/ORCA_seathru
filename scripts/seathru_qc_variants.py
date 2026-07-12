"""
@file seathru_qc_variants.py
@brief Sanity-check Sea-thru settings on a depth-spanning sample BEFORE running a
       survey of thousands of images.

Why this exists
---------------
Sea-thru's quality depends on range, and a survey with real depth relief (a reef
dropoff) can look perfect in the shallows while leaving the deep half full of
water. Tuning on one shallow frame - or judging by eye on a handful of images -
hides that completely. A full survey run is hours; this is minutes.

So this script:

  1. **Selects frames that span depth**, and prefers frames whose *own* depth map
     has a wide spread (shallow AND deep in one image). Those are the frames that
     expose a range-dependent failure, because both regimes are under identical
     illumination and exposure.
  2. Runs several **named variants** (parameter sets) over that same sample.
  3. Reports a **quantitative depth-consistency metric**, not just pictures:
     the recovered red/blue ratio in the deepest vs the shallowest pixels of each
     image. If the correction is working, deep R/B ~= shallow R/B. If the deep
     water is under-corrected, deep R/B collapses toward 0 (cyan).
  4. Writes a contact sheet per image so you can eyeball the trade-offs.

Read the table first, then the sheets. A variant that looks pretty but has
deep R/B << shallow R/B is leaving water in the deep parts of your survey.

Usage
-----
    python scripts/seathru_qc_variants.py \
        --input-dir  $T/colmap/dense/images \
        --colmap-workspace $T/colmap/dense --colmap-depth-kind photometric \
        --csv $T/processed_images.csv \
        --out-dir $T/seathru_qc \
        --n 6
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seathru.core import SeathruParams, run_seathru          # noqa: E402
from seathru.depth import ColmapDepthSource                  # noqa: E402
from seathru.depth.base import ImageMeta                     # noqa: E402
from seathru.io_images import _linear_to_srgb, load_image    # noqa: E402


# --------------------------------------------------------------------------- #
# The variants to compare. Add your own; keep "current" as the baseline.
# --------------------------------------------------------------------------- #
VARIANTS = {
    # paper/default: two-term decaying beta_D fit, raw depth
    "current":              dict(params=dict(l=1.0), clip_low=0.0),
    # remove spurious near-camera MVS depth outliers (they explode beta=-ln(E)/z
    # and drag the two-term fit into a decay)
    "depthclean":           dict(params=dict(l=1.0), clip_low=2.0),
    # skip the decaying two-term fit; use the illuminant-derived beta_D directly
    "coarse":               dict(params=dict(l=1.0, attenuation_mode="coarse"), clip_low=0.0),
    # both fixes together
    "depthclean+coarse":    dict(params=dict(l=1.0, attenuation_mode="coarse"), clip_low=2.0),
    "depthclean+coarse l0.7": dict(params=dict(l=0.7, attenuation_mode="coarse"), clip_low=2.0),

    # --- vibrancy tuning on top of the winning base (depthclean+coarse) -------
    # The base fixes the deep water but looks flat. `l` adds range-correction
    # strength; a wider low stretch percentile deepens shadows and adds punch.
    "dc+coarse l1.3":       dict(params=dict(l=1.3, attenuation_mode="coarse"), clip_low=2.0),
    "dc+coarse l1.6":       dict(params=dict(l=1.6, attenuation_mode="coarse"), clip_low=2.0),
    "dc+coarse stretch2":   dict(params=dict(l=1.0, attenuation_mode="coarse",
                                             stretch_pct=(2.0, 99.5)), clip_low=2.0),
    "dc+coarse l1.3 str2":  dict(params=dict(l=1.3, attenuation_mode="coarse",
                                             stretch_pct=(2.0, 99.5)), clip_low=2.0),
}

LABEL_H = 30
PAD = 6


def _font(size=13):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def pick_samples(depth_src, names, n):
    """Choose frames spanning the survey depth range, preferring wide within-image spread."""
    rows = []
    for nm in names:
        f = depth_src._find(nm)
        if f is None:
            continue
        from seathru.depth.colmap_source import read_colmap_array
        d = read_colmap_array(f).astype(np.float32)
        v = d[d > 0]
        if v.size < 1000:
            continue
        lo, hi = np.percentile(v, 10), np.percentile(v, 90)
        rows.append((nm, np.median(v), hi - lo))
    if not rows:
        raise SystemExit("no depth maps found for the given images")
    # half the budget: widest within-image depth spread (the dropoff frames)
    by_spread = sorted(rows, key=lambda r: -r[2])
    picked = [r[0] for r in by_spread[: max(1, n // 2)]]
    # other half: spread evenly across the survey's median-depth range
    rest = sorted([r for r in rows if r[0] not in picked], key=lambda r: r[1])
    if rest:
        idx = np.linspace(0, len(rest) - 1, n - len(picked)).astype(int)
        picked += [rest[i][0] for i in idx]
    info = {r[0]: (r[1], r[2]) for r in rows}
    return picked, info


def metrics(rec, z):
    """Depth-consistency: recovered R/B in deepest vs shallowest pixels."""
    m = z > 0
    if m.sum() < 100:
        return None
    deep = m & (z > np.percentile(z[m], 85))
    shal = m & (z < np.percentile(z[m], 25))
    if deep.sum() < 50 or shal.sum() < 50:
        return None
    dRB = rec[..., 0][deep].mean() / max(rec[..., 2][deep].mean(), 1e-6)
    sRB = rec[..., 0][shal].mean() / max(rec[..., 2][shal].mean(), 1e-6)
    sat = float((rec.max(-1) - rec.min(-1))[m].mean())
    return dict(deep_rb=float(dRB), shallow_rb=float(sRB),
                consistency=float(dRB / max(sRB, 1e-6)), saturation=sat)


def to_u8(x):
    return (_linear_to_srgb(x) * 255 + 0.5).astype(np.uint8)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, help="Undistorted images (dense/images)")
    ap.add_argument("--colmap-workspace", required=True)
    ap.add_argument("--colmap-depth-kind", choices=["geometric", "photometric"],
                    default="geometric")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=6, help="Number of sample frames")
    ap.add_argument("--max-size", type=int, default=700)
    ap.add_argument("--thumb-width", type=int, default=300)
    ap.add_argument("--variants", default=None,
                    help="Comma-separated subset of: " + ", ".join(VARIANTS))
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in Path(args.input_dir).iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    base_src = ColmapDepthSource(args.colmap_workspace, kind=args.colmap_depth_kind)
    picked, info = pick_samples(base_src, names, args.n)

    chosen = ([v.strip() for v in args.variants.split(",")]
              if args.variants else list(VARIANTS))
    print(f"sample ({len(picked)} frames; * = wide within-image depth spread):")
    for nm in picked:
        med, spread = info[nm]
        print(f"  {nm}  median {med:.2f} m, spread {spread:.2f} m"
              + ("  *" if spread > 1.0 else ""))

    results = {v: [] for v in chosen}
    for nm in picked:
        img, _ = load_image(Path(args.input_dir) / nm, max_size=args.max_size)
        tiles, labels = [to_u8(img)], ["original"]
        for vname in chosen:
            cfg = VARIANTS[vname]
            src = ColmapDepthSource(args.colmap_workspace, kind=args.colmap_depth_kind,
                                    clip_low_percentile=cfg["clip_low"])
            z = src.get_depth(img, ImageMeta(image_name=nm))
            rec = run_seathru(img, z, SeathruParams(**cfg["params"])).recovered
            mt = metrics(rec, z)
            if mt:
                results[vname].append(mt)
            tiles.append(to_u8(rec))
            labels.append(vname + (f"\ndeepR/B {mt['deep_rb']:.2f} vs {mt['shallow_rb']:.2f}"
                                   if mt else ""))

        # contact sheet
        w = args.thumb_width
        ths = [Image.fromarray(t).resize((w, int(t.shape[0] * w / t.shape[1])),
                                         Image.LANCZOS) for t in tiles]
        h = max(t.height for t in ths)
        sheet = Image.new("RGB", (len(ths) * (w + PAD), h + LABEL_H * 2), "white")
        d = ImageDraw.Draw(sheet)
        for i, (t, lab) in enumerate(zip(ths, labels)):
            x = i * (w + PAD)
            sheet.paste(t, (x, LABEL_H))
            d.text((x + 2, 2), lab.split("\n")[0], fill="black", font=_font(13))
            if "\n" in lab:
                d.text((x + 2, h + LABEL_H + 2), lab.split("\n")[1],
                       fill=(150, 0, 0), font=_font(11))
        sheet.save(out_dir / f"{Path(nm).stem}_variants.png")

    print("\n=== depth-consistency (deep R/B  /  shallow R/B; 1.00 = deep as corrected "
          "as shallow, ->0 = deep still full of water) ===")
    print("%-26s %10s %10s %12s %10s" % ("variant", "deep R/B", "shal R/B",
                                         "CONSISTENCY", "satur."))
    for v in chosen:
        rs = results[v]
        if not rs:
            continue
        print("%-26s %10.2f %10.2f %12.2f %10.3f" % (
            v,
            np.mean([r["deep_rb"] for r in rs]),
            np.mean([r["shallow_rb"] for r in rs]),
            np.mean([r["consistency"] for r in rs]),
            np.mean([r["saturation"] for r in rs])))
    print(f"\ncontact sheets -> {out_dir}")


if __name__ == "__main__":
    main()
