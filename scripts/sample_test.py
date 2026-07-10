"""
@file sample_test.py
@brief Run Sea-thru on a FIXED sample of images under a given parameter set.

Designed for iterative tuning: the sample of images is chosen once and cached
(`sample_list.txt` in --out-dir), so every subsequent run - regardless of what
parameters you change - processes the *exact same* images. Each run is stored
under its own `--tag` subfolder, and the parameters used are recorded alongside
it, so `compare_grid.py` can line runs up for a side-by-side look.

Usage
-----
First run locks the sample and also becomes your baseline:

    python scripts/sample_test.py \\
        --input-dir ../2024_02_PALFREY/images \\
        --csv ../2024_02_PALFREY/processed_images.csv \\
        --out-dir ../2024_02_PALFREY/seathru_sweep \\
        --n 10 --seed 42 --tag baseline

Change a parameter, rerun with a new tag - same 10 images automatically:

    python scripts/sample_test.py \\
        --input-dir ../2024_02_PALFREY/images \\
        --csv ../2024_02_PALFREY/processed_images.csv \\
        --out-dir ../2024_02_PALFREY/seathru_sweep \\
        --tag f2.5 --f 2.5

    python scripts/sample_test.py --input-dir ... --csv ... \\
        --out-dir ../2024_02_PALFREY/seathru_sweep \\
        --tag l0.5_stretch_wide --l 0.5 --stretch-low 0.1 --stretch-high 99.9

Then build a comparison grid:

    python scripts/compare_grid.py --out-dir ../2024_02_PALFREY/seathru_sweep
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seathru.core import SeathruParams                       # noqa: E402
from seathru.depth import PlaneDepthSource                   # noqa: E402
from seathru.depth.base import ImageMeta                     # noqa: E402
from seathru.io_images import _linear_to_srgb, save_image    # noqa: E402
from seathru.metadata import load_metadata                   # noqa: E402
from seathru.pipeline import IMAGE_EXTS, process_image       # noqa: E402

SAMPLE_LIST_NAME = "sample_list.txt"
ORIGINALS_DIR = "originals"


def get_or_create_sample(input_dir, out_dir, n, seed):
    """Return the fixed sample image-name list, creating it on first call.

    On every later call (any tag, any parameters) the cached list in
    `out_dir/sample_list.txt` is reused so runs stay directly comparable, even
    if `--n`/`--seed` differ from what was originally used.
    """
    list_path = out_dir / SAMPLE_LIST_NAME
    if list_path.exists():
        sample = [ln.strip() for ln in list_path.read_text().splitlines() if ln.strip()]
        print(f"Reusing cached sample of {len(sample)} images from {list_path}")
        return sample

    names = sorted(p.name for p in input_dir.iterdir()
                   if p.suffix.lower() in IMAGE_EXTS)
    random.seed(seed)
    sample = sorted(random.sample(names, min(n, len(names))))
    out_dir.mkdir(parents=True, exist_ok=True)
    list_path.write_text("\n".join(sample) + "\n")
    print(f"Locked new sample of {len(sample)} of {len(names)} images "
          f"(seed {seed}) -> {list_path}")
    return sample


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="default", help="Name for this parameter set / run")
    ap.add_argument("--n", type=int, default=10, help="Sample size (only used the first time)")
    ap.add_argument("--seed", type=int, default=42, help="Sample seed (only used the first time)")
    ap.add_argument("--max-size", type=int, default=1024)
    ap.add_argument("--plane-default", type=float, default=5.0)

    # SeathruParams overrides - same names/defaults as seathru.cli
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--f", type=float, default=2.0)
    ap.add_argument("--l", type=float, default=1.0)
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--stretch-low", type=float, default=0.5)
    ap.add_argument("--stretch-high", type=float, default=99.5)
    ap.add_argument("--no-protect-red", action="store_true")
    args = ap.parse_args(argv)

    input_dir, out_dir = Path(args.input_dir), Path(args.out_dir)
    run_dir = out_dir / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    orig_dir = out_dir / ORIGINALS_DIR
    meta_map = load_metadata(args.csv) if args.csv else {}

    sample = get_or_create_sample(input_dir, out_dir, args.n, args.seed)

    params = SeathruParams(
        p=args.p, f=args.f, l=args.l, epsilon=args.epsilon,
        protect_red=not args.no_protect_red,
        stretch_pct=(args.stretch_low, args.stretch_high),
    )
    (run_dir / "params.json").write_text(json.dumps({
        "p": params.p, "f": params.f, "l": params.l,
        "epsilon": params.epsilon, "protect_red": params.protect_red,
        "stretch_pct": list(params.stretch_pct),
        "plane_default": args.plane_default, "max_size": args.max_size,
    }, indent=2))

    source = PlaneDepthSource(default_m=args.plane_default)
    print(f"Tag '{args.tag}': p={params.p} f={params.f} l={params.l} "
          f"eps={params.epsilon} stretch={params.stretch_pct} "
          f"protect_red={params.protect_red}")

    for i, name in enumerate(sample, 1):
        t0 = time.time()
        meta = meta_map.get(name, ImageMeta(image_name=name))
        result, img = process_image(input_dir / name, source, meta,
                                    params, max_size=args.max_size)
        save_image(run_dir / f"{Path(name).stem}.png", result.recovered)

        orig_path = orig_dir / f"{Path(name).stem}.png"
        if not orig_path.exists():   # save once, shared across all tags
            orig_dir.mkdir(parents=True, exist_ok=True)
            save_image(orig_path, img)

        print(f"  [{i}/{len(sample)}] {name}  ({time.time() - t0:.1f}s)")

    print(f"\nRun '{args.tag}' -> {run_dir}")
    print("Compare all tags so far with: "
          f"python scripts/compare_grid.py --out-dir {out_dir}")


if __name__ == "__main__":
    main()
