"""
@file process_test_set.py
@brief Convenience runner for the 2024_02_TEST set.

Edit DEPTH below to switch between 'plane' (works now, no torch), 'file'
(SfM range maps), or 'mono' (needs torch). Then:

    python scripts/process_test_set.py
"""
from pathlib import Path

from seathru.core import SeathruParams
from seathru.pipeline import process_folder

ROOT = Path(__file__).resolve().parents[2] / "2024_02_TEST"
DEPTH = "plane"   # 'plane' | 'file' | 'mono'


def make_depth_source():
    if DEPTH == "plane":
        from seathru.depth import PlaneDepthSource
        return PlaneDepthSource(default_m=5.0)
    if DEPTH == "file":
        from seathru.depth import FileDepthSource
        return FileDepthSource(ROOT / "depth_maps", scale=1.0)
    if DEPTH == "mono":
        from seathru.depth import MonocularDepthSource
        return MonocularDepthSource(backend="midas")
    raise ValueError(DEPTH)


if __name__ == "__main__":
    results = process_folder(
        input_dir=ROOT / "images",
        out_dir=ROOT / f"seathru_out_{DEPTH}",
        depth_source=make_depth_source(),
        csv_path=ROOT / "test_processed_images.csv",
        params=SeathruParams(f=2.0, l=1.0, p=0.5, return_debug=True),
        max_size=1024,
        debug=True,
    )
    ok = sum(1 for _, s in results if s == "ok")
    print(f"{ok}/{len(results)} recovered")
