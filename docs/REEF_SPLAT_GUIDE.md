# Water-free reef splats: end-to-end workflow

Goal: from a raw ASV photo survey to (a) a **clean, water-free Gaussian splat**
that loads fast on the web, (b) a **photogrammetry-grade mesh/orthomosaic**
from the same data — with no fog, floaters, or water colour in the splat.

```
raw images ──► COLMAP (poses + dense metric depth)          [docs/COLMAP_GUIDE.md]
                    │
                    ▼
            Sea-thru, survey-locked ──► water-free, colour-consistent images
                    │                                        [seathru_python]
        ┌───────────┴─────────────┐
        ▼                         ▼
  3DGS training                2DGS / DN-Splatter or OpenMVS
  (+ depth supervision)        (mesh, DEM, orthomosaic)
        │
        ▼
  cleanup (bbox crop, point-cloud prune, SuperSplat)
        │
        ▼
  compress (SOG/SPZ) ──► web viewer
```

Why this order works — and why SeaSplat alone still shows water/floaters:

- **SeaSplat learns the medium jointly** with the splat, but for its first
  `--seathru_from_iter` iterations (default 10,000) the splat trains on *raw
  hazy images*. The optimizer explains backscatter the only way it can — by
  placing semi-transparent Gaussians in the water column. Densification then
  multiplies them. When the medium model finally switches on, those floaters
  already exist and the optimizer has little pressure to delete them.
- **Pre-correcting with Sea-thru means the training images contain no water
  from iteration 0.** There is nothing for floater Gaussians to explain, so
  far fewer are created in the first place.
- **Survey-locked correction** guarantees the same coral is the same colour
  in every overlapping view. Per-image adaptive correction (or none) forces
  the splat to absorb frame-to-frame colour drift as view-dependent SH junk
  → colour flicker when orbiting.

Keep SeaSplat as the comparison baseline (Track B, end of this guide) — it's
the best joint method, but pre-correction + vanilla 3DGS attacks the floater
problem earlier in the pipeline.

---

## Step 0 — Prerequisites

| Piece | Where | Notes |
| --- | --- | --- |
| COLMAP workflow | [seathru_python/docs/COLMAP_GUIDE.md](seathru_python/docs/COLMAP_GUIDE.md) | GPS-limited matching, chunked mapping, georegistration to metres, dense depth |
| Sea-thru library | [seathru_python/](seathru_python/) | `pip install -e .` in its venv |
| SeaSplat / 3DGS trainer | [seasplat/](seasplat/) | CUDA GPU needed. Conda py3.10 + torch cu121 per its README |
| SuperSplat (cleanup + web) | <https://superspl.at/editor> | browser-based, free |

**GPU reality check:** 3DGS training on a 4 GB RTX 3050 works for *chunks*
(a few hundred images at `-r 2`…`-r 4` downscale), not for a 10k-image scene
in one go. For full-survey splats, train per-COLMAP-chunk locally or use the
HPC/SLURM route from the COLMAP guide. Splats from overlapping chunks can be
merged in SuperSplat afterwards (both are just point sets in world metres —
they align because every chunk is georegistered to the same GPS frame).

---

## Step 1 — COLMAP: poses + metric dense depth (on the *raw* images)

Follow [COLMAP_GUIDE.md](seathru_python/docs/COLMAP_GUIDE.md) through the
dense stage. Do **not** colour-correct before matching:

- SIFT matching works fine on raw underwater images (it's gradient-based),
  and Sea-thru needs COLMAP's depth as *input*, so the order is forced anyway.
- Critically, **georegister** (`colmap model_aligner` with the CSV GPS) before
  dense stereo, so depth maps are in metres — Sea-thru's physics assumes metres.

You end with a dense workspace:

```
colmap/dense/
├── images/                  # undistorted images  ← everything downstream uses THESE
├── sparse/                  # undistorted (PINHOLE) cameras + poses
└── stereo/depth_maps/       # <name>.JPG.geometric.bin  (metres)
```

Two products matter from here on: `dense/images` + `dense/sparse` (the 3DGS
dataset skeleton) and `stereo/depth_maps` (Sea-thru input + splat depth
supervision).

## Step 2 — Sea-thru: survey-locked water removal

Run on the **undistorted** images so the output aligns 1:1 with the poses:

```bash
cd seathru_python
python -m seathru.cli \
    --input-dir  ../2024_02_PALFREY/colmap/dense/images \
    --out-dir    ../2024_02_PALFREY/seathru_out \
    --csv        ../2024_02_PALFREY/processed_images.csv \
    --depth colmap --colmap-workspace ../2024_02_PALFREY/colmap/dense \
    --full-res \
    --survey-locked --calib-sample-size 20 --lock-exposure
```

Flag rationale (see the seathru_python README for details):

- `--survey-locked` — one median backscatter/attenuation/white-balance fit,
  frozen across the survey → cross-view colour consistency, the single most
  important setting for splat quality. Also ~46× faster per image.
- `--lock-exposure` — freezes the contrast stretch too. Per-frame exposure
  wobble is exactly what becomes flicker in a splat. Always use for splatting.
- `--full-res` — output at native resolution; you'll downscale in the trainer
  (`-r`), which keeps the option of high-res retraining later.
- Timing (Ryzen 7 5800H, 1 core): ~5.4 s/image locked+full-res → ~15 h for
  10k images; parallelise over cores by splitting the input dir if needed.

**Rename to match COLMAP names** (the 3DGS loader looks images up by the
exact name recorded in the COLMAP model):

```bash
cd ../2024_02_PALFREY/seathru_out
for f in *_seathru.png; do mv "$f" "${f%_seathru.png}.JPG"; done
```

(PNG bytes under a `.JPG` name are fine — PIL sniffs content, not extension.)

**QC before training** (10 minutes that saves a day): flip through ~20
corrected frames spread across the survey. You want no green/blue haze, no
pink cast, and — most importantly — the *same* coral patch in two overlapping
frames should look the same. If far areas over-brighten, lower `--l` (e.g.
0.7), delete `seathru_out/survey_stats.json`, and re-run.

## Step 3 — Assemble the 3DGS dataset

```bash
DATASET=../2024_02_PALFREY/splat_dataset
mkdir -p $DATASET
ln -s "$(realpath ../2024_02_PALFREY/seathru_out)"      $DATASET/images
ln -s "$(realpath ../2024_02_PALFREY/colmap/dense/sparse)" $DATASET/sparse/0
```

Optional but recommended — **depth supervision** files. SeaSplat reads ground
-truth depth from `images/depth/<image_name>.npy` when `--use_depth_l1_loss`
is set. Convert COLMAP's `.geometric.bin` maps (seathru_python already has the
reader):

```python
# save as make_gt_depth.py; run once
from pathlib import Path
import numpy as np
from seathru.depth.colmap_source import read_colmap_array

dense = Path("../2024_02_PALFREY/colmap/dense")
out = Path("../2024_02_PALFREY/seathru_out/depth"); out.mkdir(exist_ok=True)
for f in sorted((dense / "stereo/depth_maps").glob("*.geometric.bin")):
    name = f.name.replace(".geometric.bin", "")        # e.g. G0022406.JPG
    d = read_colmap_array(f).astype(np.float32)
    d[d < 0] = 0
    np.save(out / f"{name}.npy", d)
```

## Step 4 — Train the splat (on corrected images, water model OFF)

Using the SeaSplat codebase as a plain 3DGS trainer (it *is* the INRIA
implementation underneath) — do **not** pass `--do_seathru`; the water is
already gone:

```bash
cd seasplat
python train.py \
    -s $DATASET --exp palfrey_waterfree \
    --use_depth_l1_loss \
    -r 2
```

- `--use_depth_l1_loss` — L1 between rendered depth and the COLMAP depth from
  Step 3 (weight 0.1 in `train.py`). This is the main *training-time* floater
  killer: a Gaussian floating mid-water renders depth far from the measured
  bottom depth and gets pushed down onto the reef.
- `-r 2` / `-r 4` — downscale factor; pick so `(width/r)` ≈ 1600–2000 px for a
  4 GB GPU chunk, higher res on HPC.
- Scene bounding box: SeaSplat has `--do_scene_bb --bb_xlo … --bb_zhi` (world
  metres — your model is georegistered, so these are real coordinates). Set Z
  bounds from seafloor depth ± a couple of metres to *hard-prevent* Gaussians
  in the water column above the reef. Get the numbers from the dense point
  cloud (open `dense/fused.ply` in CloudCompare/MeshLab, note the Z range).
- SH degree: SeaSplat defaults `sh_degree 0`. **Keep it** for this use case —
  view-dependent colour underwater was mostly the water (now removed), coral
  is close to matte, and SH0 makes the output ~3.5× smaller (SH coefficients
  are 48 of the 59 floats per Gaussian). This directly serves the
  fast-web-loading goal.
- Chunking: `--start_cam / --end_cam / --subsample` exist in the loader if you
  want to train on a camera subrange without rebuilding datasets.

Sanity mid-training: open the TensorBoard depth images — rendered depth
should look like the COLMAP depth, not speckled.

## Step 5 — Cleanup: your masking idea, systematised

Even with all of the above a few artifacts survive. Three passes, cheapest
first:

1. **Bounding-box crop** — if you didn't crop at train time, crop now in
   SuperSplat (or a 10-line PLY filter script) using the same Z bounds.
2. **Prune against the COLMAP dense cloud** — this is exactly the
   "mask the model using the colmap data" idea, and it's sound: any Gaussian
   whose centre is farther than ~10–20 cm from the nearest `fused.ply` point
   is not reef; delete it. (A KD-tree query over the PLY — ~20 lines with
   scipy. Worth scripting if this becomes routine.)
3. **SuperSplat manual pass** — <https://superspl.at/editor>: load the PLY,
   sphere-select the reef, invert, delete; then brush/lasso stragglers; also
   filter by opacity (very low-opacity Gaussians are haze remnants).
   Unlimited undo, runs locally in the browser.

## Step 6 — Compress + publish for the web

From SuperSplat, export compressed; or convert the PLY directly:

| Format | Size vs PLY | Use |
| --- | --- | --- |
| PLY | 1× | archive / editing master |
| Compressed PLY (SuperSplat) | ~4× smaller | quick shares |
| **SOG** | ~10–20× smaller | web, PlayCanvas/SuperSplat ecosystem |
| **SPZ** (Niantic) | ~10× smaller | web, becoming the de-facto delivery format; Spark/three.js |

With SH0 + SOG/SPZ, a few-hundred-thousand-Gaussian reef chunk lands in the
tens of MB — genuinely fast web loading. Viewers: SuperSplat's own hosted
viewer (one click from the editor), PlayCanvas engine, or
[Spark](https://sparkjs.dev) if you're embedding in a three.js page.

## Step 7 — The photogrammetry-grade twin (mesh / DEM / orthomosaic)

Same corrected images + same poses, second product line — pick either:

- **DN-Splatter / 2DGS** (nerfstudio ecosystem): splat variants built for
  *surfaces* — they take the depth/normal priors you already have and extract
  a TSDF mesh directly from the splat. Best when you want splat + mesh from
  one training run.
- **Classic MVS** (COLMAP `stereo_fusion` → `poisson_mesher`, or OpenMVS):
  you already have the dense workspace; mesh it and texture with the
  *corrected* images for a water-free textured model; orthomosaic via ODM
  with `--fast-orthophoto` on the corrected frames.

The Sea-thru pass improves every one of these products simultaneously — one
correction, three outputs (splat, mesh, orthomosaic).

---

## Track B — SeaSplat as the baseline comparison

Worth running on one chunk to compare against Track A:

```bash
python train.py -s $DATASET_RAW --exp palfrey_seasplat \
    --do_seathru --seathru_from_iter 10000 \
    --use_depth_l1_loss --do_scene_bb --bb_zlo ... --bb_zhi ...
python render_uw.py -m output/palfrey_seasplat   # writes no_water/ renders + J-only splat
```

Two tweaks that should reduce the floaters/water you've been seeing in stock
SeaSplat, even without pre-correction: `--use_depth_l1_loss` (feed it the
Step-3 depth files) and the scene bounding box. And `--seathru_from_iter
5000` (earlier medium onset = less time for the splat to build haze Gaussians)
is worth one experiment.

Expected outcome: Track A wins on floaters and colour stability (consistent
input images + geometric constraints from iteration 0); SeaSplat remains the
reference for how well joint optimisation can do. If SeaSplat's *restored*
renders look better in some regions, its learned medium parameters are also a
useful cross-check on the survey-locked Sea-thru fit.

## Suggested first experiment (one weekend, one chunk)

1. Pick one COLMAP chunk (~300–500 images) with good coverage.
2. Step 2–4 at `-r 2` on the 3050 (or `-r 1` on HPC).
3. Train three variants: (a) corrected + depth loss + bbox (Track A),
   (b) corrected, no depth loss (ablate), (c) raw + SeaSplat (Track B).
4. Compare: orbit each in SuperSplat — look for floaters, colour flicker,
   coral edge sharpness; compare rendered depth vs COLMAP depth.
5. Whichever wins becomes the per-chunk recipe for the full survey.
