# COLMAP + Sea-thru on Ubuntu — a from-scratch, field-tested guide

End-to-end recipe for turning a raw ASV reef survey into **water-free, colour-consistent
images** using COLMAP (poses + metric dense depth) and this Sea-thru library.

Everything here was verified on a full run of `2024_02_PALFREY_TEST_R10`
(2022 GoPro frames, 5568×4872, 6.2 GB, ~20×20 m reef patch, 10 m grid) on:

> Ubuntu 24.04.4 · Ryzen 7 5800H (16 threads) · 31 GB RAM · RTX 3050 Laptop **4 GB VRAM**
> (compute 8.6) · NVIDIA driver 580

Numbers quoted below are **measured on that machine**, not estimates. Where the
official guides were wrong or a step blew up, it says so.

---

## 0. Read this first — the two things that will bite you

These are not theoretical. Both happened on the reference run and one of them
**hard-froze the laptop**.

### 0.1 Disk: budget ~40 GB per 2000 images, and watch `normal_maps`

`patch_match_stereo` writes a **normal map for every depth map**, and the normal
maps are ~3× larger than the depth maps. For 1939 images:

| Artefact | Size | Needed by Sea-thru? |
| --- | --- | --- |
| `dense/stereo/depth_maps/` | 6.4 GB | ✅ **yes** |
| `dense/stereo/normal_maps/` | **19 GB** | ❌ no — only `stereo_fusion` |
| `dense/images/` (undistorted) | 4.1 GB | ✅ yes |
| `database.db` | 3.5 GB | ❌ no, once `sparse/` exists |
| **`colmap/` total** | **35 GB** | |

The reference run filled a 484 GB drive to **100%**, which froze the machine.
**Check free space before the dense stage** and keep ≥ 40 GB headroom. If you
never need the fused point cloud, you can delete `normal_maps/` afterwards to
reclaim 19 GB (but you cannot then run `stereo_fusion` without recomputing).

### 0.2 RAM: `stereo_fusion` loads *everything* at once

`colmap stereo_fusion` reads all depth **and** normal maps into memory
simultaneously (`Loading workspace data with 16 threads...`). At ~2000 images
that OOM'd 31 GB of RAM. It is **not needed for Sea-thru** — it only produces
`fused.ply` for QA and for the splat bbox/pruning steps.

Also: `PMS_CACHE_GB` from `hw_profile.py` (15 GB here) is a **system-RAM** cache
held for the whole dense run. Do not run other heavy jobs alongside it.

> **Rule of thumb:** run the dense stage *alone*. Do not run parameter sweeps,
> Sea-thru, or fusion concurrently with `patch_match_stereo`.

### 0.3 Keep the workspace on ext4 — never NTFS

COLMAP hammers `database.db` (SQLite) with random writes. On an NTFS mount an
earlier attempt produced features for all 2022 images but **only 242 images in
any verified match pair** — a silently corrupt database. Images may live
anywhere; `database.db`, `sparse*/` and `dense/` must be on ext4:

```bash
findmnt -no FSTYPE -T /path/to/workspace     # must print ext4, not ntfs3/fuseblk
```

---

## 1. Install COLMAP with CUDA (build from source — the apt package won't do)

**The Ubuntu `colmap` package is built without CUDA.** Sparse SfM works, but
`patch_match_stereo` — the dense metric depth Sea-thru needs — refuses to run.
We build the *same version apt ships* (3.9.1) so every CLI flag in these guides
matches.

### 1.1 System packages

```bash
sudo apt update
sudo apt install -y \
    nvidia-cuda-toolkit gcc-12 g++-12 \
    git cmake ninja-build build-essential ccache \
    libboost-program-options-dev libboost-graph-dev libboost-system-dev \
    libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev \
    libgoogle-glog-dev libgtest-dev libsqlite3-dev libglew-dev \
    qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev
```

Two non-obvious entries:

- **`nvidia-cuda-toolkit`** — Ubuntu 24.04 ships CUDA **12.0**. Your driver
  reporting a newer CUDA (e.g. 13.0 in `nvidia-smi`) is fine; drivers are
  backward-compatible with older toolkits.
- **`gcc-12 g++-12`** — nvcc 12.0 **rejects Ubuntu 24.04's default GCC 13** as a
  host compiler. You don't change the system default; you point CMake at g++-12.

### 1.2 Build (needs one patch)

```bash
mkdir -p colmap && cd colmap

wget http://archive.ubuntu.com/ubuntu/pool/universe/c/colmap/colmap_3.9.1.orig.tar.gz
tar xzf colmap_3.9.1.orig.tar.gz && mv colmap-3.9.1 src

# REQUIRED with GCC 13: two files miss `#include <memory>` and fail to compile with
#   error: 'unique_ptr' is not a member of 'std'
# This is upstream PR 2338; Ubuntu ships it as a package patch:
wget http://archive.ubuntu.com/ubuntu/pool/universe/c/colmap/colmap_3.9.1-2build2.debian.tar.xz
tar xJf colmap_3.9.1-2build2.debian.tar.xz
(cd src && patch -p1 < ../debian/patches/gh-pr-2338)

mkdir build && cd build
cmake ../src -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
    -DCMAKE_INSTALL_PREFIX="$(pwd)/../install"
ninja && ninja install          # ~15 min on 16 threads; no sudo with a local prefix

mkdir -p ~/.local/bin && ln -sf "$(pwd)/../install/bin/colmap" ~/.local/bin/colmap
```

`CMAKE_CUDA_ARCHITECTURES=86` is RTX 30-series. Find yours with
`nvidia-smi --query-gpu=compute_cap --format=csv,noheader` and drop the dot
(8.6 → 86).

**Verify — both must pass:**

```bash
colmap -h | head -2                        # must read "... with CUDA"
colmap patch_match_stereo -h >/dev/null && echo dense-ok
```

### 1.3 Sea-thru library

```bash
cd seathru_python
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[debug]"
python -m seathru.cli --help
```

No torch needed for the COLMAP-depth path — the GPU is used by COLMAP, not by
Sea-thru (only `--depth mono` needs torch).

---

## 2. Prepare the workspace

```bash
T=/path/to/2024_02_PALFREY_TEST_R10          # dataset root (images/ + processed_images.csv)
mkdir -p $T/colmap/logs

python scripts/hw_profile.py --out $T/colmap/pipeline.env
python scripts/colmap_geo_from_csv.py --csv $T/processed_images.csv --out-dir $T/colmap
python scripts/colmap_make_pairs.py --csv $T/processed_images.csv \
    --out $T/colmap/pairs.txt --seq 10 --radius 2.0 --max-neighbors 40
```

**Do not use `exhaustive_matcher` above ~500 images.** 2022 images is ~2 M pairs.
GPS/time pair matching gives **53,182 pairs (52.6/image)** — a ~38× cut — with no
loss of reconstruction quality on a dense lawnmower survey. Set `--radius` to
roughly 2–3× your survey line spacing (here lines are ~0.27 m apart).

---

## 3. Sparse reconstruction

```bash
cd $T/colmap && source pipeline.env

colmap feature_extractor --database_path database.db --image_path ../images \
    --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV \
    --SiftExtraction.max_image_size "$MAX_IMAGE_SIZE" \
    --SiftExtraction.max_num_features "$MAX_NUM_FEATURES" \
    --SiftExtraction.use_gpu "$USE_GPU"

colmap matches_importer --database_path database.db \
    --match_list_path pairs.txt --match_type pairs --SiftMatching.use_gpu "$USE_GPU"

colmap mapper --database_path database.db --image_path ../images \
    --output_path sparse --Mapper.num_threads "$COLMAP_THREADS" \
    --Mapper.ba_global_function_tolerance 1e-5
```

**Measured (2022 images):** extraction 7 min (GPU) · matching 1 h 44 (GPU) ·
mapper **5 h 16** (CPU, the long pole). Result: **1939/2022 images registered**,
1.24 M points, mean reprojection error **1.30 px**. All three stages are
resumable — re-running skips completed work.

---

## 4. Georegister to metres ⚠️ *this is where COLMAP fails on flat surveys*

Depths must be in **metres** — Sea-thru's physics assumes it. Georegistration is
what sets that scale.

**`colmap model_aligner` fits the Sim3 with a RANSAC that assumes the cameras are
not coplanar.** A lawnmower reef survey breaks that assumption completely: every
frame is at the same tow height looking straight down, so the camera centres form
a near-horizontal sheet and the out-of-plane rotation is unconstrained. On the
reference run it returned a **mirror-flipped model (seabed *above* the cameras)
at scale 0.275 with 6.2 m residuals** — silently unusable, and the wrong scale
would have made every Sea-thru correction meaningless.

Use the robust fallback in this repo, which fixes the vertical from the imagery
(mean camera view direction → world down) and then fits only the well-conditioned
2D horizontal similarity to GPS:

```bash
colmap model_converter --input_path sparse/0 \
    --output_path /tmp/sparse_txt --output_type TXT

python scripts/colmap_georef_planar.py \
    --model-txt /tmp/sparse_txt \
    --geo-ref geo_ref.txt \
    --out geo_sim3.txt \
    --csv ../processed_images.csv        # cross-checks scale against sonar depth_m

colmap model_transformer --input_path sparse/0 \
    --output_path sparse_geo --transform_path geo_sim3.txt
```

**Measured result:** scale **1.743 m/unit**, horizontal residual **1.38 m** (p50,
≈ GPS noise), seabed **2.28 m below** the cameras, model extent **20.6 × 20.0 m**
— matching the real survey. The script **refuses to write a flipped model**, so a
bad fit cannot silently poison your depth.

**Always sanity-check the scale:** `colmap model_analyzer --path sparse_geo` —
the extent must match your real survey size.

> COLMAP 3.9.1 also **renamed the aligner flags**: use `--alignment_max_error`,
> not `--robust_alignment` / `--robust_alignment_max_error`.

---

## 5. Dense depth — choose your time/quality point

```bash
colmap image_undistorter --image_path ../images --input_path sparse_geo \
    --output_path dense --output_type COLMAP \
    --max_image_size "$DENSE_MAX_IMAGE_SIZE"

colmap patch_match_stereo --workspace_path dense --workspace_format COLMAP \
    --PatchMatchStereo.gpu_index 0 \
    --PatchMatchStereo.max_image_size 1000 \
    --PatchMatchStereo.geom_consistency 0 \
    --PatchMatchStereo.cache_size "$PMS_CACHE_GB"
```

This is by far the longest stage. Two knobs dominate the cost, and on a 4 GB card
the default settings are punishing:

| Setting | Cost per image | 1939 images | Notes |
| --- | --- | --- | --- |
| 1600 px, `geom_consistency 1` | ~22 s × 2 passes | **~24 h** | COLMAP default-ish |
| 1000 px, `geom_consistency 0` | ~29 s × 1 pass | **~15 h** | what the reference run used |

- **`geom_consistency 1` doubles the work** — it runs a full photometric pass over
  every image, then a full geometric pass. It produces multi-view-consistent depth
  (better edges, fewer outliers) and is required for `stereo_fusion --input_type
  geometric`. **Sea-thru does not need it**: it estimates its model at 1024 px and
  fits a smooth backscatter/attenuation model, so single-pass photometric depth is
  effectively quality-neutral for colour recovery.
- **Resolution is the other lever.** Depth is upsampled to the image anyway, so
  ~1000 px costs little for Sea-thru.
- VRAM is *not* the constraint here: 1000 px used only **395 MiB** of the 4 GB.

Read photometric maps with the Sea-thru flag `--colmap-depth-kind photometric`.

**Validate the depth is really metric before you commit hours to correction:**

```python
from seathru.depth.colmap_source import read_colmap_array
import numpy as np
d = read_colmap_array("dense/stereo/depth_maps/G0025219.JPG.photometric.bin")
v = d[d > 0]
print(np.percentile(v, [10, 50, 90]))     # reference run: 1.86 / 2.22 / 4.27 m
```

Those must be plausible reef ranges. On the reference run the median camera→seabed
range of **2.22 m** cross-checked against the sonar `depth_m` median of **1.98 m**
(slant range is legitimately a little longer than vertical depth) — independent
confirmation that the georegistration scale was right.

`stereo_fusion` → `fused.ply` is **optional** (see §0.2 — it is the RAM bomb). Skip
it unless you need the cloud for splat bbox/pruning.

---

## 6. Tune the parameters *before* the full run

Sea-thru's defaults produced visibly **over-bright, washed-out** images on this
reef. Tune on a handful of frames first — a full 1939-image run is ~3.5 h, a
9-tile sweep is ~2 minutes.

`param_grid_2d.py` sweeps two parameters against real COLMAP depth:

```bash
python scripts/param_grid_2d.py \
    --input-dir $T/colmap/dense/images \
    --csv $T/processed_images.csv \
    --out-dir $T/seathru_tuning \
    --image G0022092.JPG \
    --depth colmap --colmap-workspace $T/colmap/dense --colmap-depth-kind photometric \
    --x-param stretch_high --x-values 99.5,99.9,99.99 \
    --y-param l            --y-values 0.4,0.7,1.0
```

**What the sweep showed on this dataset:**

- **`l` is the brightness dial that matters.** Default `l=1.0` and `l=0.7` were both
  too bright; **`l=0.4`** gave natural coral tone. (`l` balances the range-dependent
  attenuation correction — too high and it over-brightens everything.)
- **`stretch_high` made almost no visible difference** (99.5 / 99.9 / 99.99 were
  near-identical) — this reef has few bright outlier pixels to clip, so the upper
  percentile barely moves.

Do this on *your* data; the right `l` is scene-dependent.

---

## 7. Run Sea-thru

Run on the **undistorted** images so output aligns 1:1 with the COLMAP poses:

```bash
python -m seathru.cli \
    --input-dir $T/colmap/dense/images \
    --out-dir   $T/seathru_out \
    --csv       $T/processed_images.csv \
    --depth colmap --colmap-workspace $T/colmap/dense \
    --colmap-depth-kind photometric \
    --l 0.4 \
    --full-res --survey-locked --calib-sample-size 20 --lock-exposure

# rename to the exact names COLMAP recorded (3DGS loaders look images up by name)
cd $T/seathru_out
for f in *_seathru.png; do mv "$f" "${f%_seathru.png}.JPG"; done
```

- `--survey-locked` — one frozen backscatter/attenuation/white-balance fit for the
  whole survey → cross-view colour consistency. Essential for splatting/orthomosaics.
- `--lock-exposure` — freezes the contrast stretch too; per-frame exposure wobble is
  exactly what becomes flicker in a splat.
- **Measured: 6.7 s/image** (full-res, survey-locked) → **~3.5 h for 1939 images**.

> **Output resolution:** the corrected images come out at the *undistorted* size,
> set by `image_undistorter --max_image_size` (1600 px here → 1600×1401), **not** the
> original 27 MP. Fine for `-r 2` splat training; re-undistort larger if you need
> full-resolution deliverables.

---

## 8. Gotchas quick-reference

| Symptom | Cause | Fix |
| --- | --- | --- |
| Machine freezes / hangs | **Disk full** — `normal_maps` is 19 GB per 2000 images | §0.1; keep 40 GB free |
| OOM during `stereo_fusion` | It loads all depth+normal maps at once | §0.2; skip it — Sea-thru doesn't need it |
| Features extracted but almost no verified matches | `database.db` on NTFS | §0.3; move workspace to ext4 |
| `patch_match_stereo` "not supported without CUDA" | apt COLMAP has no CUDA | §1 — build from source |
| `error: 'unique_ptr' is not a member of 'std'` | GCC 13 vs COLMAP 3.9.1 | apply `gh-pr-2338` (§1.2) |
| nvcc rejects the host compiler | nvcc 12.0 vs GCC 13 | `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12` |
| Georegistered extent is nonsense; seabed inverted | `model_aligner` RANSAC on a coplanar survey | §4 — `colmap_georef_planar.py` |
| `--robust_alignment` flag rejected | renamed in 3.9.1 | use `--alignment_max_error` |
| Corrected images look over-bright / washed out | `l` too high | §6 — sweep `l`; 0.4 worked here |
| Dense stage projected at 24 h | 1600 px + `geom_consistency 1` | §5 — 1000 px, photometric |

---

## 9. Total wall-clock for the reference run (2022 images, RTX 3050 4 GB)

| Stage | Time |
| --- | --- |
| COLMAP build | ~15 min |
| Feature extraction (GPU) | 7 min |
| Matching, 53 k GPS pairs (GPU) | 1 h 44 |
| Mapper (CPU) | 5 h 16 |
| Georegistration | seconds |
| Undistort + dense depth, 1000 px photometric (GPU) | ~15 h |
| Parameter tuning sweep | ~2 min |
| Sea-thru, survey-locked full-res (CPU) | ~3.5 h |

Budget **about a day** end-to-end for ~2000 images on a 4 GB laptop GPU. The dense
stage dominates and is the one to parallelise or move to HPC (see
[COLMAP_GUIDE.md §12](COLMAP_GUIDE.md)).
