# Reef splat pipeline — Windows test run (2×2 m subset)

Temporary Windows companion to [REEF_SPLAT_GUIDE.md](REEF_SPLAT_GUIDE.md),
written for a one-day end-to-end test of the small dataset in
`2024_02_PALFREY_TEST_R2\` (116 images, ~4×4 m patch around `G0023380.JPG`,
105/116 frames with valid sonar depth). All commands are **PowerShell**, run
from:

```powershell
cd C:\Users\mickv\OneDrive\Documents\ORCA_mapping
```

> **OneDrive tip:** COLMAP hammers its `database.db` and writes thousands of
> small dense-depth files — OneDrive sync fights it. Either **pause syncing**
> (system tray → OneDrive → Pause) for the session, or copy
> `2024_02_PALFREY_TEST_R2` to a local non-synced folder (e.g. `C:\reefwork\`)
> and work there. Commands below assume the OneDrive path; substitute if you
> relocate.

## 0. Prerequisites on the work PC

| Need | For | Get |
| --- | --- | --- |
| **COLMAP** Windows binary (CUDA build) | steps 2–4 | <https://github.com/colmap/colmap/releases> → `colmap-x64-windows-cuda.zip`, unzip anywhere, use full path to `COLMAP.bat` (or add to PATH) |
| **Python 3.10–3.13** | Sea-thru + helper scripts | python.org installer is fine |
| **NVIDIA GPU + driver** | dense depth (step 4) + splat training (step 8) | `nvidia-smi` must work |
| **Conda + VS 2022 Build Tools (C++)** | SeaSplat only (compiles two CUDA extensions) | only needed for step 8 |

**No NVIDIA GPU?** Steps 1–3 (sparse SfM) and Sea-thru still run on CPU —
do the colour-correction test with `--depth estimated` instead of COLMAP
dense depth, and skip splat training. Everything else below assumes CUDA.

If the dataset needs recreating (or you want a different spot / bigger patch):

```powershell
cd C:\Users\mickv\OneDrive\Documents\ORCA_mapping\seathru_python
python make_test_subset.py --image G0021080.JPG --radius 2 --out-dir 2024_02_PALFREY_TEST_R3
```

## 1. Python env + Sea-thru install (once)

Put the venv **outside** OneDrive so it doesn't sync:

```powershell
python -m venv C:\venvs\seathru
C:\venvs\seathru\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .\seathru_python
pip install -e ".\seathru_python[debug]"
python -m seathru.cli --help    # sanity check
```

(If activation is blocked: `Set-ExecutionPolicy -Scope Process RemoteSigned`.)

## 2. COLMAP sparse (poses) — ~10–20 min

116 images → plain exhaustive matching, none of the big-survey machinery:

```powershell
$T = "C:\Users\mickv\OneDrive\Documents\ORCA_mapping\2024_02_PALFREY_TEST_R10"
New-Item -ItemType Directory -Force "$T\colmap\sparse", "$T\colmap\sparse_geo" | Out-Null

colmap feature_extractor `
    --database_path $T\colmap\database.db `
    --image_path $T\images `
    --ImageReader.camera_model OPENCV `
    --ImageReader.single_camera 1

colmap exhaustive_matcher --database_path $T\colmap\database.db

colmap mapper `
    --database_path $T\colmap\database.db `
    --image_path $T\images `
    --output_path $T\colmap\sparse
```

Check: `$T\colmap\sparse\0\` should exist and register (nearly) all 116 images
(`colmap model_analyzer --path $T\colmap\sparse\0`).

## 3. Georegister to metres (critical — Sea-thru assumes metres)

```powershell
python seathru_python\scripts\colmap_geo_from_csv.py `
    --csv $T\processed_images.csv --out-dir $T\colmap

colmap model_aligner `
    --input_path $T\colmap\sparse\0 `
    --output_path $T\colmap\sparse_geo `
    --ref_images_path $T\colmap\geo_ref.txt `
    --ref_is_gps 0 `
    --alignment_type custom `
    --robust_alignment 1 `
    --robust_alignment_max_error 3.0
```

> If your COLMAP version rejects a flag, run `colmap model_aligner -h` and
> match (the robust_alignment flags changed names across releases).

Check scale: `colmap model_analyzer --path $T\colmap\sparse_geo` — extent
should read roughly **4 × 4 m**.

## 4. Dense depth — ~20–60 min depending on GPU

```powershell
colmap image_undistorter `
    --image_path $T\images `
    --input_path $T\colmap\sparse_geo `
    --output_path $T\colmap\dense `
    --output_type COLMAP --max_image_size 1600

colmap patch_match_stereo `
    --workspace_path $T\colmap\dense `
    --workspace_format COLMAP `
    --PatchMatchStereo.max_image_size 1600 `
    --PatchMatchStereo.geom_consistency 1

colmap stereo_fusion `
    --workspace_path $T\colmap\dense `
    --workspace_format COLMAP `
    --input_type geometric `
    --output_path $T\colmap\dense\fused.ply
```

(CUDA out-of-memory → drop `max_image_size` to 1200 and re-run; it skips
depth maps already done.) `fused.ply` is your QA point cloud **and** the
cleanup mask for step 9 — open it in CloudCompare/MeshLab and note the Z
range of the reef while you're there.

## 5. Sea-thru: survey-locked water removal — ~15 min

On the **undistorted** images, with COLMAP metric depth:

```powershell
python -m seathru.cli `
    --input-dir $T\colmap\dense\images `
    --out-dir   $T\seathru_out `
    --csv       $T\processed_images.csv `
    --depth colmap --colmap-workspace $T\colmap\dense `
    --full-res `
    --survey-locked --calib-sample-size 10 --lock-exposure
```

Rename outputs to the original COLMAP image names:

```powershell
Get-ChildItem $T\seathru_out -Filter *_seathru.png |
    Rename-Item -NewName { $_.Name -replace '_seathru\.png$', '.JPG' }
```

**QC:** open a handful of corrected frames — no haze, no pink cast, and the
same coral identical in overlapping frames. Too bright in the far corners →
re-run with `--l 0.7` after deleting `$T\seathru_out\survey_stats.json`.

## 6. Ground-truth depth files for the splat trainer

SeaSplat reads per-image depth from `images\depth\<name>.JPG.npy`. The
converter script is [make_gt_depth.py](make_gt_depth.py) in this folder:

```powershell
python make_gt_depth.py --dense $T\colmap\dense --out-dir $T\seathru_out\depth
```

## 7. Assemble the splat dataset

Plain copies — it's small:

```powershell
$D = "$T\splat_dataset"
New-Item -ItemType Directory -Force "$D\sparse" | Out-Null
Copy-Item $T\seathru_out "$D\images" -Recurse          # corrected imgs + depth\
Copy-Item $T\colmap\dense\sparse "$D\sparse\0" -Recurse # undistorted poses
```

## 8. Train the splat (SeaSplat codebase, water model OFF) — ~30–60 min

One-time env (conda, per `seasplat\README.md`, plus VS Build Tools present):

```powershell
conda create -n seasplat_py310 -y python=3.10
conda activate seasplat_py310
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
cd seasplat
pip install -r requirements.txt
pip install .\submodules\diff-gaussian-rasterization .\submodules\simple-knn
```

Train — **no `--do_seathru`** (the water is already removed):

```powershell
python train.py -s $D --exp test_r2_waterfree --use_depth_l1_loss -r 2
```

Optional hard mask against water-column Gaussians (Z numbers from the
`fused.ply` you inspected in step 4):

```powershell
python train.py -s $D --exp test_r2_waterfree_bb --use_depth_l1_loss -r 2 `
    --do_scene_bb --bb_xlo -4 --bb_xhi 4 --bb_ylo -4 --bb_yhi 4 --bb_zlo <zmin> --bb_zhi <zmax>
```

Comparison baseline (Track B — raw images, SeaSplat's joint water model): build
a second dataset from `$T\colmap\dense\images` (uncorrected) and run
`python train.py -s $D_RAW --exp test_r2_seasplat --do_seathru --seathru_from_iter 5000 --use_depth_l1_loss -r 2`.

## 9. Cleanup + web export

1. Find the output PLY under `seasplat\output\test_r2_waterfree\point_cloud\iteration_30000\point_cloud.ply`.
2. Open <https://superspl.at/editor> (browser, local processing): sphere-select
   the reef → invert → delete; brush stragglers; delete very-low-opacity
   splats.
3. Export **Compressed PLY** or **SOG** for the web (tens of MB at SH0);
   SuperSplat can publish a shareable viewer link directly.

## Expected wall-clock for the 116-image test

| Step | Time |
| --- | --- |
| COLMAP sparse + georegister | 10–20 min (CPU) |
| COLMAP dense | 20–60 min (GPU-dependent) |
| Sea-thru survey-locked, full-res | ~15 min (1 core; calibration ~5 min of that) |
| Splat training (30k iters, `-r 2`) | 30–60 min |
| Cleanup + export | 15 min interactive |

**Success criteria for the day:** corrected frames show consistent water-free
colour; splat orbits without fog/floaters; SuperSplat export loads instantly
in the browser. Then the same recipe scales to full-survey chunks per
[REEF_SPLAT_GUIDE.md](REEF_SPLAT_GUIDE.md).
