"""
@file colmap_make_chunks.py
@brief Split a capture-order image list into overlapping chunks for COLMAP mapping.

Chunked mapping keeps each ``colmap mapper`` job small (bounded RAM/time), so a
huge survey can be reconstructed one chunk per evening and resumed across
power-offs. Consecutive chunks share ``--overlap`` images so the resulting
sub-models can be merged (they see the same scene in the overlap region).

Usage:
    python scripts/colmap_make_chunks.py \
        --image-list ../2024_02_PALFREY/colmap/image_list.txt \
        --out-dir   ../2024_02_PALFREY/colmap/chunks \
        --chunk 900 --overlap 150
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-list", required=True,
                    help="image_list.txt (one image name per line, capture order)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--chunk", type=int, default=900, help="Images per chunk")
    ap.add_argument("--overlap", type=int, default=150,
                    help="Shared images between consecutive chunks")
    args = ap.parse_args(argv)

    if args.overlap >= args.chunk:
        raise SystemExit("--overlap must be smaller than --chunk")

    names = [ln.strip() for ln in Path(args.image_list).read_text().splitlines()
             if ln.strip()]
    if not names:
        raise SystemExit("Empty image list.")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    step = args.chunk - args.overlap
    chunks = []
    start = 0
    idx = 0
    while start < len(names):
        part = names[start:start + args.chunk]
        path = out / f"chunk_{idx:03d}.txt"
        path.write_text("\n".join(part) + "\n")
        chunks.append((path.name, len(part)))
        idx += 1
        if start + args.chunk >= len(names):
            break
        start += step

    print(f"Wrote {len(chunks)} chunks to {out} "
          f"(chunk={args.chunk}, overlap={args.overlap}, total={len(names)} images):")
    for name, count in chunks:
        print(f"  {name}: {count} images")


if __name__ == "__main__":
    main()
