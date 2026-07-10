# seathru-orca

A Python implementation of **Sea-thru** — the physically based underwater
colour-restoration method from:

> Derya Akkaynak and Tali Treibitz, **"Sea-thru: A Method for Removing Water
> From Underwater Images,"** *IEEE/CVF Conference on Computer Vision and
> Pattern Recognition (CVPR), 2019, pp. 1682–1691.*
> [Paper (CVF Open Access, PDF)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Akkaynak_Sea-Thru_A_Method_for_Removing_Water_From_Underwater_Images_CVPR_2019_paper.pdf) ·
> [Project page](http://csms.haifa.ac.il/profiles/tTreibitz/webpage/sea-thru.html)

Sea-thru was, as far as we're aware, the first method to treat underwater
image formation as a genuine physical inverse problem — recovering true scene
colour from a range map rather than applying a global dehazing-style
correction. It's an excellent piece of work, and this library exists to make
its ideas usable as an ordinary importable Python package. **This is an
independent, from-scratch re-implementation of the paper's equations and
underlying theory — not a copy of the authors' original code** (which is
research MATLAB, not published as a library). All credit for the method,
the physics, and the underlying research belongs to Akkaynak and Treibitz;
please cite their paper (see [Citing](#citing)) if you use this software.

This port was built for the [ORCA](.) autonomous-survey-vehicle reef-mapping
pipeline, to colour-correct large underwater photo datasets before
photogrammetry / Gaussian-splat reconstruction — but the library itself is
general-purpose and has no ORCA-specific dependency.

It recovers water-free colour from an RGB image **plus a per-pixel range
map**, using the paper's revised underwater image formation model (distinct
backscatter and direct-signal attenuation coefficients, range-dependent
attenuation).

## Contents

- [Why a range map matters](#why-a-range-map-matters)
- [Install](#install)
- [Pipeline: correcting a dataset](#pipeline-correcting-a-dataset)
- [Survey-locked mode](#survey-locked-mode)
- [Configuration / tunable parameters](#configuration--tunable-parameters)
- [Time estimates](#time-estimates)
- [Large datasets (10k+ images) → COLMAP](#large-datasets-10k-images--colmap-on-a-laptop)
- [Tuning tools](#tuning-tools)
- [How it maps to the paper](#how-it-maps-to-the-paper)
- [Deviations from the paper](#deviations-from-the-paper-documented)
- [Citing](#citing)

## Why a range map matters

Sea-thru is an **RGB-D** method: every pixel needs a distance-to-scene in
metres. This library makes the range map a pluggable input (`seathru.depth`):

| Source | Class | Needs | Quality |
| --- | --- | --- | --- |
| **COLMAP dense** | `ColmapDepthSource` | a COLMAP dense workspace (`stereo/depth_maps`) | **best** — metric, matches the paper's own method. See [docs/COLMAP_GUIDE.md](docs/COLMAP_GUIDE.md) |
| **Other SfM** | `FileDepthSource` | per-image depth maps exported from Metashape / ODM (`.npy`/`.tif`/`.png`) | best, if metric |
| **Monocular neural depth** | `MonocularDepthSource` | PyTorch (Depth Anything V2 / MiDaS) | good, approximate; scale anchored to `depth_m` |
| **Image-derived prior** | `EstimatedDepthSource` | nothing (red-attenuation cue) | rough but *spatially varying* — runs today with no torch/SfM; good for previews + parameter tuning |
| **Flat plane** | `PlaneDepthSource` | just a scalar altitude (or CSV `depth_m`) | coarse fallback — backscatter + white balance only |

With a *constant* plane the range term is degenerate, so plane mode does
backscatter removal + white balance only (the `f`/`l`/`p`/`epsilon` knobs have
**no effect**). Use SfM, monocular, or the image-derived prior for the full
range-varying colour recovery.

## Install

Tested on Linux (Ubuntu) and Windows, Python 3.9–3.13 (monocular depth needs
Python ≤3.12, see below).

```bash
mkdir -p ~/seathru && cd ~/seathru
git clone https://github.com/<your-username>/seathru_python.git
cd seathru_python

python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -e .                    # core pipeline (numpy/scipy/scikit-image/pillow)
pip install -e ".[debug]"           # + matplotlib, for --debug intermediate-map montages

# Optional: monocular depth (--depth mono). Needs a Python <=3.12 venv,
# torch has no wheels for newer interpreters yet:
pip install torch torchvision
```

Confirm it's installed:

```bash
python -m seathru.cli --help
```

## Pipeline: correcting a dataset

The general shape of a run is always the same:

```
input images  +  a depth source  +  (optional) survey CSV  -->  seathru  -->  corrected images
```

### 1. Lay out your data

```
my_survey/
├── images/                  # your source photos (JPEG/PNG/TIFF)
└── processed_images.csv     # optional: image_name,latitude,longitude,heading_deg,depth_m
```

The CSV is optional. If you have it, `depth_m` (camera altitude in metres) is
used to anchor/seed several depth sources; `-1` or a blank value is treated as
"unknown" and the source falls back sensibly.

### 2. Pick a depth source and run

No SfM, no GPU — good for a first look and for parameter tuning (uses the
spatially-varying red-attenuation prior, see the table above):

```bash
python -m seathru.cli \
    --input-dir my_survey/images \
    --out-dir   my_survey/seathru_out \
    --csv       my_survey/processed_images.csv \
    --depth estimated --est-near 1 --est-far 10
```

Flat-altitude fallback (fastest, backscatter + white balance only):

```bash
python -m seathru.cli --input-dir my_survey/images --out-dir my_survey/seathru_out \
    --csv my_survey/processed_images.csv --depth plane --plane-default 5
```

Per-image SfM depth maps exported from Metashape/ODM:

```bash
python -m seathru.cli --input-dir my_survey/images --out-dir my_survey/seathru_out \
    --depth file --depth-dir my_survey/sfm_depth --depth-scale 1.0
```

COLMAP dense depth (metric, best quality — see [the COLMAP guide](docs/COLMAP_GUIDE.md)
for taking a raw photo survey through SfM to get this):

```bash
python -m seathru.cli --input-dir my_survey/images --out-dir my_survey/seathru_out \
    --depth colmap --colmap-workspace my_survey/colmap/dense --full-res
```

Monocular neural depth (needs torch, see [Install](#install)):

```bash
python -m seathru.cli --input-dir my_survey/images --out-dir my_survey/seathru_out \
    --csv my_survey/processed_images.csv --depth mono --mono-backend midas
```

Add `--full-res` to any of the above to estimate the correction at
`--max-size` (fast) but write the recovered image at the input's **native
resolution** (so it's 1:1 comparable and safe for downstream photogrammetry).

Each run writes `<name>_seathru.png` per input image, plus (with `--debug`)
`<name>_debug.png` intermediate-map montages for QA.

### 3. Use it as a library instead of the CLI

```python
from seathru import run_seathru, SeathruParams
from seathru.io_images import load_image, save_image
from seathru.depth import FileDepthSource, ImageMeta

img, _ = load_image("frame.jpg", max_size=1024)
depths = FileDepthSource("sfm_depth").get_depth(img, ImageMeta("frame.jpg"))
result = run_seathru(img, depths, SeathruParams(f=2.0, l=1.0))
save_image("frame_seathru.png", result.recovered)
```

## Survey-locked mode

By default (matching the paper), Sea-thru fits its backscatter, illuminant,
attenuation, and white-balance statistics **independently for every image**.
That's the right choice for a handful of photos, but across a
multi-thousand-frame survey it lets ambient light, turbidity, and colour
balance drift frame to frame — which shows up as visible seams when frames
are blended into an orthomosaic, and as view-dependent colour flicker when
training a NeRF/3D Gaussian Splat on the corrected frames.

**Survey-locked mode** freezes the per-image adaptive statistics across the
whole batch: it fits the backscatter (Eq. 10) and attenuation (Eq. 11)
coefficients and the white-balance gains (Eq. 9) *once*, from a handful of
frames spread evenly across the survey, then reuses that single frozen fit
for every image — each frame's own range map still drives where the
correction is strongest, only the water-column physics and colour balance
are shared.

```bash
python -m seathru.cli \
    --input-dir my_survey/images --out-dir my_survey/seathru_out \
    --csv my_survey/processed_images.csv \
    --depth colmap --colmap-workspace my_survey/colmap/dense --full-res \
    --survey-locked --calib-sample-size 20
```

What happens:
1. `seathru` samples 20 images spread evenly across the sorted file list
   (start/middle/end of the survey — different lighting, altitude, turbidity).
2. It fits the full per-image adaptive model on each of those 20 frames, then
   takes the **median** of the fitted coefficients across the sample (robust
   to one bad frame — e.g. texture-poor or badly exposed).
3. It saves the result to `my_survey/seathru_out/survey_stats.json` and
   reuses it for every image in the batch. **Re-running the same command
   later loads the saved JSON instead of recalibrating** — delete the file
   (or pass a different `--stats-file`) to force a fresh calibration.
4. Every image is then processed using the frozen coefficients, evaluated
   against *that image's own* depth map — this is also **faster** than
   adaptive mode, since the expensive nonlinear curve fits only run on the
   calibration sample, not on every image (see [Time estimates](#time-estimates)).

Notes:
- **Exposure/contrast is *not* locked by default** — each frame keeps its own
  percentile contrast stretch, because scene brightness legitimately varies
  with altitude and sun angle across a survey. Pass `--lock-exposure` to also
  freeze the output contrast-stretch bounds from calibration, if you need
  every frame on an identical absolute scale.
- With `--depth plane` (a constant range map), only the white-balance gain
  can be usefully locked — the code detects this automatically and leaves
  the backscatter/attenuation coefficients per-image (there's no spatial
  information to lock them from). No action needed on your part.
- `--calib-sample-size` trades calibration robustness for calibration time:
  each sampled frame costs one full adaptive-mode fit. 12–20 frames is a good
  default for a multi-thousand-image survey; smaller/simpler surveys can use
  fewer.
- Programmatically: `seathru.survey.calibrate_survey_stats(...)` returns a
  `seathru.core.SurveyStats`, and `SeathruParams(locked_stats=stats)` applies
  it to any `run_seathru(...)` call.

## Configuration / tunable parameters

All tunable knobs live in one place: `seathru.core.SeathruParams` (mirrored
1:1 by `seathru.cli`'s flags). The dataclass docstring in
[`seathru/core.py`](seathru/core.py) is the canonical reference; summary:

| Param | CLI flag | Default | Effect |
| --- | --- | --- | --- |
| `p` | `--p` | `0.5` | Illuminant locality (Eq. 14 support weight, 0–1). Higher trusts the local pixel over its neighbourhood average. |
| `f` | `--f` | `2.0` | Illuminant geometry factor (paper uses 2). Raises overall brightness. |
| `l` | `--l` | `1.0` | Attenuation/brightness balance — **the main strength dial**. Lower if far/deep areas over-brighten. |
| `epsilon` | `--epsilon` | `0.05` | Iso-range neighbourhood band width (Eq. 15), fraction of the scene's depth span. |
| `protect_red` | `--no-protect-red` to disable | `True` | White-balance red gently instead of pure Gray-World (avoids pink cast on red-starved deep frames). |
| `stretch_pct` | `--stretch-low` / `--stretch-high` | `(0.5, 99.5)` | Output contrast-stretch percentiles. Widen toward `(0.1, 99.9)` for a flatter, safer result; tighten for more punch. |
| `backscatter_restarts` | `--backscatter-restarts` | `25` | Random-restart count for the Eq. 10 fit. Dominates per-image runtime — lower for speed, raise for a harder scene. |
| `attenuation_restarts` | `--attenuation-restarts` | `10` | Random-restart count for the Eq. 11 fit. Also runtime-dominant. |
| `min_neighborhood` | — (library only) | `50` | Minimum neighbourhood pixel count before it's merged into its nearest survivor. |
| `spread_fraction` | — (library only) | `0.01` | Sample-thinning window for the attenuation fit. |

> `p`, `f`, `l`, `epsilon` only take effect with a **spatially varying** range
> map (SfM, monocular, or the image-derived prior). On a flat plane they do
> nothing — only `stretch_pct` and `protect_red` are active.

Use the [tuning tools](#tuning-tools) below to pick working values on your
own imagery before committing to a full run.

## Time estimates

Sea-thru is CPU-bound pure NumPy/SciPy — the dominant cost is the nonlinear
curve fits (`backscatter_restarts` × 3 channels + `attenuation_restarts` × 3
channels) at your `--max-size` working resolution; everything is
single-threaded per image and needs no GPU.

Measured on a laptop CPU (**AMD Ryzen 7 5800H, 8C/16T, single-threaded per
image**), default parameters, spatially-varying depth (the full-cost path —
`--depth plane` is noticeably cheaper, see below). These are real
`run_seathru` timings from this repo's own benchmark, not estimates:

| Working resolution (`--max-size`, long edge) | Adaptive mode (per image) | Survey-locked mode (per image, after calibration) | Speedup |
| --- | --- | --- | --- |
| 512 px  | 8.1 s  | 0.15 s | 54× |
| 1024 px (default) | 28.4 s | 0.62 s | 46× |
| 1600 px | 70.2 s | 1.8 s  | 39× |
| 2048 px | 118.5 s | 2.8 s | 42× |

Survey-locked mode is faster as well as more consistent: it skips the
nonlinear backscatter/attenuation curve fits entirely for every image except
the calibration sample, evaluating the already-fitted coefficients against
each frame's own depth map instead.

`--full-res` adds a fixed upsample+recover pass at native resolution on top
of the `--max-size` estimate above. Measured estimate@1024 (adaptive) →
apply@5568×4872 (a full-resolution GoPro frame): 29.2 s + 4.8 s = **34.1 s**
per image total; with survey-locked mode the estimate step drops to ~0.6 s,
so the full-res total becomes **~5.4 s per image** (upsample+recover is the
same regardless of locked/adaptive).

`--depth plane` skips the spatial backscatter/attenuation fit entirely
(flat-plane recovery is backscatter-percentile + white balance only), so it
is roughly as fast as survey-locked mode regardless of resolution.

### Estimating a full dataset

```
total_time ≈ images × per_image_time / parallel_workers
```

**Worked example — 10,000 images, `--max-size 1024` (the default), single core:**

- Adaptive mode: 10,000 × 28.4 s ≈ 284,000 s ≈ **~79 hours (~3.3 days)**
- Survey-locked mode: 10,000 × 0.62 s ≈ 6,200 s ≈ **~1.7 hours**
  (plus a one-off calibration pass — 12–20 images at the adaptive-mode cost,
  a few minutes — not repeated on re-runs since the stats JSON is cached)

**Same 10,000 images with `--full-res` (native-resolution output):**

- Adaptive mode: 10,000 × 34.1 s ≈ **~95 hours (~4 days)**
- Survey-locked mode: 10,000 × ~5.4 s ≈ **~15 hours**

`seathru` itself processes one image at a time, but nothing about the design
is stateful across images (survey-locked mode's shared state is just the
saved JSON), so you can trivially parallelise across CPU cores by splitting
`--input-dir` into N chunks and running N processes (e.g. with GNU `parallel`
or a simple shell loop), or by writing a small driver that calls
`seathru.pipeline.process_folder` per chunk with `multiprocessing`. On an
8-core laptop this divides the wall-clock estimates above by roughly 6–8×.

For reference, `hw_profile.py` in `scripts/` detects this machine's CPU/RAM/GPU
(used for tuning the COLMAP stage, see below) and is a reasonable place to
add a similar `seathru`-specific worker-count heuristic if you want one.

## Large datasets (10k+ images) → COLMAP on a laptop

For a full survey (e.g. 10,000+ images, tens of GB), see
**[docs/COLMAP_GUIDE.md](docs/COLMAP_GUIDE.md)**. It covers install, folder
setup, GPS/time-limited matching (so it fits in laptop RAM), **chunked mapping
you can pause and resume across nights / power-offs**, georegistration to
metres, dense depth, and feeding the result back into `--depth colmap`. Helper
scripts in `scripts/`:

- `hw_profile.py` — detect CPU/RAM/GPU (or `--profile laptop|workstation|hpc`) and
  emit a `pipeline.env` of COLMAP tuning params, so the *same* workflow scales
  from a laptop to an HPC node.
- `colmap_geo_from_csv.py` — CSV GPS → georegistration file + image list.
- `colmap_make_pairs.py` — GPS/time neighbour match-pairs (bounds CPU/RAM).
- `colmap_make_chunks.py` — split into overlapping chunks for resumable mapping.

The guide is hardware-adaptive and includes a **SLURM section** (Apptainer +
job-array chunked mapping) for running COLMAP on university HPC.

## Tuning tools

Three scripts in `scripts/` help you pick a working point before committing to a
full run. All default to the **image-derived depth prior** so the `f`/`l`/`p`/`epsilon`
knobs are active (a flat plane leaves them inert), and all cache a fixed image
sample so runs stay comparable.

| Script | Answers | Output |
| --- | --- | --- |
| `param_effect_grid.py` | *What does each knob do?* — every parameter swept low→rec→high, one row each | one sheet per image (`<img>_params.png`) |
| `param_grid_2d.py` | *How do two knobs interact?* — one parameter across columns, another down rows | `<img>_<x>_x_<y>.png` |
| `sample_test.py` + `compare_grid.py` | *Is one setting consistent across scenes?* — a tagged run over the same N images, compared side-by-side | `comparison_grid.png` |

```bash
# 1. Learn the knobs on one scene (every parameter, isolated)
python scripts/param_effect_grid.py --input-dir my_survey/images \
    --csv my_survey/processed_images.csv \
    --out-dir my_survey/seathru_param_effects --image example.jpg

# 2. Find the sweet spot between two knobs (e.g. f vs l)
python scripts/param_grid_2d.py --input-dir my_survey/images \
    --csv my_survey/processed_images.csv \
    --out-dir my_survey/seathru_param_effects --image example.jpg \
    --x-param f --x-values 1.5,2,2.5,3 --y-param l --y-values 0.5,1,1.5

# 3. Confirm your chosen setting holds across a random sample, then iterate tags
python scripts/sample_test.py --input-dir my_survey/images \
    --csv my_survey/processed_images.csv \
    --out-dir my_survey/seathru_sweep --n 10 --seed 42 --tag baseline
python scripts/sample_test.py --input-dir my_survey/images --csv my_survey/processed_images.csv \
    --out-dir my_survey/seathru_sweep --tag l0.5 --l 0.5
python scripts/compare_grid.py --out-dir my_survey/seathru_sweep
```

Note: the estimated prior tends to exaggerate the near/far range spread, so the
best `l` under it is usually **lower** than under metric COLMAP depth — re-check
`l` once real depth maps are in. Validate the pipeline quantitatively any time
with `python scripts/synthetic_validation.py` (paper's Eq. 18 angular-error
metric on a synthetic scene).

## How it maps to the paper

| Stage | Paper | Code |
| --- | --- | --- |
| Dark-pixel candidates in 10 range bins | §4.3 | `find_backscatter_points` |
| Backscatter fit `B = B∞(1−e^−βz) + J'e^−β'z` | Eq. 10 | `estimate_backscatter` |
| Iso-range neighbourhoods | Eq. 15 | `construct_neighborhood_map` |
| Local-space-average-colour illuminant | Eq. 13–14 | `estimate_illumination` |
| Coarse `β_D = −log(illuminant)/z` | Eq. 12 | `coarse_attenuation` |
| Two-term-exponential `β_D(z)` refined to range map | Eq. 11, 16–17 | `refine_attenuation` |
| Recover `J = (I−B)·e^{β_D·z}` + white balance | Eq. 8–9 | `recover_image` |

## Deviations from the paper (documented)

- **Inputs are 8-bit sRGB JPEG**, inverse-gamma'd to approximate linear
  radiance. The paper uses linear RAW; a RAW path is stubbed in `io_images`.
- **Illuminant** uses the fast neighbourhood-mean form of local space average
  colour (as in reference implementations of the paper) rather than the full
  per-pixel iterative diffusion — same intent, much faster.
- Photofinishing (§4.5, camera-pipeline colour-space conversion) is out of
  scope; output is a display-ready sRGB PNG.
- **Survey-locked mode** (see above) has no equivalent in the paper — it's an
  addition for processing large, consistently-lit photo surveys where
  per-image adaptivity is a liability rather than a feature. It's opt-in
  (`--survey-locked`); the default behaviour matches the paper.

## Citing

If you use this software, please cite the original paper:

```bibtex
@InProceedings{Akkaynak_2019_CVPR,
    author    = {Akkaynak, Derya and Treibitz, Tali},
    title     = {Sea-Thru: A Method for Removing Water From Underwater Images},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2019},
    pages     = {1682-1691}
}
```

This repository is an independent re-implementation and is not affiliated
with or endorsed by the paper's authors.

## License

MIT — see [LICENSE](LICENSE). This covers this codebase only; see above for
citing the underlying research.
