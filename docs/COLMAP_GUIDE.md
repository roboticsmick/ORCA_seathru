# COLMAP → Sea-thru guide (large ASV reef datasets on a laptop)

This guide takes a full ASV survey (e.g. `2024_02_PALFREY`: **10,335 images,
30 GB**) through COLMAP structure-from-motion to produce the per-pixel **range
maps** Sea-thru needs, in a way that:

- **bounds CPU/RAM** by matching only GPS/time neighbours (not all 53 M pairs);
- runs in **batches** so it never tries to fit all images at once;
- can be **paused, powered off overnight, and resumed** the next evening.

It ends by feeding the result straight into the `seathru` pipeline in this repo.

> **Why COLMAP at all?** Sea-thru is an RGB-**D** method — every pixel needs a
> distance-to-scene in metres. Your CSV `depth_m` is one scalar per image (and
> `-1` here), which is not a range map. SfM gives a true per-pixel range map,
> and the GPS track fixes real-world scale. This is the same data you need for
> orthomosaics / 3DGS anyway, so it is not wasted work.

---

## 0. What you need and where things go

### Hardware
- **CPU + RAM:** works on a laptop. 16 GB RAM is enough with the neighbour-only
  matching below; 8 GB works if you also decimate frames (Step 7 notes).
- **GPU (NVIDIA/CUDA):** *optional but strongly recommended for dense depth*
  (`patch_match_stereo`). Without it, sparse SfM still runs on CPU; see
  **Step 8B** for the no-GPU depth path.
- **Disk:** budget ~3× the image size free (database + sparse + dense). For
  PALFREY (30 GB images) keep ~100 GB free for the dense stage.

### Folder layout (create this once)

```
2024_02_PALFREY/
├── images/                     # the 10,335 GoPro JPEGs (already here)
├── processed_images.csv        # ASV metadata (already here)
└── colmap/                     # <- everything COLMAP writes lives here
    ├── geo_ref.txt             # image -> X Y Z metres   (Step 1)
    ├── image_list.txt          # capture order            (Step 1)
    ├── pairs.txt               # GPS/time match pairs      (Step 2)
    ├── database.db             # features + matches (RESUMABLE)
    ├── sparse/                 # sparse models (poses + 3D points)
    │   └── 0/
    ├── sparse_geo/             # georegistered (metric) sparse model
    ├── dense/                  # undistorted images + depth maps
    │   └── stereo/depth_maps/  # <- what Sea-thru reads
    └── logs/
```

Create it:

```bash
cd ~/ORCA_mapping/2024_02_PALFREY        # adjust to your path
mkdir -p colmap/{sparse,sparse_geo,dense,logs}
```

---

## 1. Install COLMAP (Ubuntu)

Three options, easiest first.

### A. APT (quickest; CPU + CUDA if your driver supports it)
```bash
sudo apt update
sudo apt install colmap
colmap -h        # sanity check
```

### B. Docker (isolated, GPU passthrough)
```bash
# needs nvidia-container-toolkit for --gpus
docker run --gpus all -w /working -v $(pwd):/working \
    colmap/colmap:latest colmap -h
```

### C. Build from source (newest features / guaranteed CUDA)
```bash
sudo apt install git cmake ninja-build build-essential \
    libboost-program-options-dev libboost-graph-dev libboost-system-dev \
    libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev \
    libgoogle-glog-dev libgtest-dev libsqlite3-dev libglew-dev \
    qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev

git clone https://github.com/colmap/colmap.git && cd colmap
mkdir build && cd build
cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=native   # drop CUDA flag if CPU-only
ninja && sudo ninja install
```

> **Check GPU support:** `colmap patch_match_stereo -h` should exist and
> `nvidia-smi` should list your GPU. If COLMAP was built without CUDA,
> `patch_match_stereo` will refuse to run — use the **Step 8B** fallback.

### Also install this Sea-thru library (any OS)
```bash
cd ~/ORCA_mapping/seathru_orca
python -m pip install -e .          # core (numpy/scipy/scikit-image/pillow)
python -m pip install -e ".[debug]" # + matplotlib for --debug montages
```
The helper scripts below (`scripts/colmap_*.py`) only need Python stdlib.

---

## 1.5 Tune the pipeline to your hardware  *(do this once)*

The pipeline's cost knobs are not hard-coded — they live in a
`colmap/pipeline.env` file that every command below reads. Generate it for your
machine and **source it in each terminal** before running COLMAP:

```bash
cd ~/ORCA_mapping/seathru_orca
python scripts/hw_profile.py --out ../2024_02_PALFREY/colmap/pipeline.env
cd ~/ORCA_mapping/2024_02_PALFREY/colmap
source pipeline.env         # now $COLMAP_THREADS, $USE_GPU, ... are set
```

`hw_profile.py` auto-detects CPU / RAM / GPU (or takes `--profile
{laptop,workstation,hpc}` and explicit `--threads/--ram-gb/--vram-gb`
overrides), then derives the parameters. For **this laptop** (Ryzen 7 5800H, 16
threads, 32 GB RAM, RTX 3050 Laptop = **4 GB VRAM**) it produces:

| Variable | Value | Controls |
| --- | --- | --- |
| `COLMAP_THREADS` | 14 | CPU threads (leaves 2 for the OS while you work) |
| `USE_GPU` / `GPU_INDEX` | 1 / 0 | CUDA on for SIFT + dense |
| `MAX_IMAGE_SIZE` | 1600 | feature/match downscale |
| `MAX_NUM_FEATURES` | 8192 | SIFT features per image |
| `DENSE_MAX_IMAGE_SIZE` | 1600 | dense stereo image size — **VRAM-bound**; drop to 1200 if CUDA OOM |
| `PMS_CACHE_GB` | 16 | PatchMatch image cache in **system RAM** (not VRAM) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 900 / 153 | resumable mapping chunk (Step 5A) |
| `PAIR_SEQ` / `PAIR_RADIUS` / `PAIR_MAX_NEIGHBORS` | 10 / 3.0 / 40 | match density (Step 2) |

**Scaling to other machines** — the *same commands* adapt by regenerating the
file:
- **Upgrade GPU (8–12 GB VRAM):** `DENSE_MAX_IMAGE_SIZE` rises to 2400 → sharper
  depth maps.
- **Big desktop (128 GB RAM):** `CHUNK_SIZE` → 3000 (fewer merges), bigger images.
- **HPC node (256 GB, 64 cores):** `--profile hpc` sets `CHUNK_SIZE=0` (skip
  chunking — do one hierarchical run) and denser matching. See **Step 12**.

> 4 GB VRAM is the one real limit on this laptop. Sparse SfM is unaffected, but
> dense `patch_match_stereo` is VRAM-bound: if it throws a CUDA out-of-memory
> error, lower `DENSE_MAX_IMAGE_SIZE` (1600 → 1200 → 1000) and re-run — it skips
> depth maps already computed.

---

## 2. Prepare GPS reference and match pairs (from the CSV)

These two commands (run on any machine, no COLMAP needed) turn the CSV into the
files that make COLMAP metric *and* cheap.

```bash
cd ~/ORCA_mapping/seathru_orca

# (1) image -> local ENU metres, for georegistration + capture-order list
python scripts/colmap_geo_from_csv.py \
    --csv ../2024_02_PALFREY/processed_images.csv \
    --out-dir ../2024_02_PALFREY/colmap

# (2) GPS + time neighbour pairs (THE key to bounding CPU/RAM)
source ../2024_02_PALFREY/colmap/pipeline.env    # PAIR_* values for your machine
python scripts/colmap_make_pairs.py \
    --csv ../2024_02_PALFREY/processed_images.csv \
    --out ../2024_02_PALFREY/colmap/pairs.txt \
    --seq "$PAIR_SEQ" --radius "$PAIR_RADIUS" --max-neighbors "$PAIR_MAX_NEIGHBORS"
```

`colmap_make_pairs.py` matches each frame only to (a) its ±`seq` temporal
neighbours along the survey line and (b) up to `max-neighbors` frames within
`radius` metres (adjacent survey lines). On PALFREY this is **~268 k pairs (52
per image)** instead of **53 M** exhaustive — a ~200× cut in matching work.

Tuning:
- Laptop struggling / 8 GB RAM → lower `--radius 2.0 --max-neighbors 25`.
- Sparse survey, want more loop closures → raise `--radius`/`--max-neighbors`.
- **Downward camera:** do **not** filter by heading (footprints overlap
  regardless of heading). `--max-heading-diff` exists only for forward/oblique
  rigs.

---

## 3. Feature extraction  *(resumable)*

One physical camera (GoPro) → share intrinsics across all frames. Downscale for
speed/RAM. GoPro wide FOV → `OPENCV` handles the radial distortion (use
`OPENCV_FISHEYE` if you shot in the widest/fisheye mode).

```bash
cd ~/ORCA_mapping/2024_02_PALFREY/colmap

source pipeline.env      # once per terminal, from the colmap/ dir
colmap feature_extractor \
    --database_path database.db \
    --image_path ../images \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model OPENCV \
    --SiftExtraction.max_image_size "$MAX_IMAGE_SIZE" \
    --SiftExtraction.max_num_features "$MAX_NUM_FEATURES" \
    --SiftExtraction.estimate_affine_shape 0 \
    --SiftExtraction.use_gpu "$USE_GPU"
```

> **Resumable for free:** features are written to `database.db` per image. If
> interrupted, just re-run the same command — already-processed images are
> skipped. The database is the checkpoint for Steps 3–4.

If you have the Basalt VIO intrinsics for the IMX477, you can instead pass known
intrinsics with `--ImageReader.camera_params "fx,fy,cx,cy,..."` and later fix
them in the mapper (`--Mapper.ba_refine_focal_length 0`).

---

## 4. Match only the neighbour pairs  *(resumable)*

This is the custom-matching entry point — it matches **exactly** the pairs from
Step 2 and nothing else:

```bash
colmap matches_importer \
    --database_path database.db \
    --match_list_path pairs.txt \
    --match_type pairs \
    --SiftMatching.use_gpu "$USE_GPU"
```

Matches also land in `database.db` and are skipped on re-run, so this step is
resumable too. After it finishes, **extraction + matching are done once and for
all** — the rest of the pipeline reads from `database.db`.

---

## 5. Sparse reconstruction (mapping) — the long pole

Incremental SfM is the expensive, long-running stage. Two ways to run it on a
laptop; **pick 5A for true power-off-overnight resume.**

### 5A. Chunked mapping (recommended: one bounded job per evening)

Reconstruct overlapping windows of the survey, one chunk at a time, then merge.
Each chunk is a few hundred images → finishes in one session and is saved before
you stop.

Split the capture-order list into overlapping chunks:

```bash
cd ~/ORCA_mapping/seathru_orca
source ../2024_02_PALFREY/colmap/pipeline.env    # CHUNK_SIZE / CHUNK_OVERLAP
python scripts/colmap_make_chunks.py \
    --image-list ../2024_02_PALFREY/colmap/image_list.txt \
    --out-dir   ../2024_02_PALFREY/colmap/chunks \
    --chunk "$CHUNK_SIZE" --overlap "$CHUNK_OVERLAP"
```

> If `hw_profile.py` set `CHUNK_SIZE=0` (big HPC node), skip chunking and use the
> single hierarchical run in **5B** instead.

Map each chunk into its own model (run one per evening; the database already has
all features/matches, so chunks are independent and cheap on RAM):

```bash
cd ~/ORCA_mapping/2024_02_PALFREY/colmap
mkdir -p chunks/models
for f in chunks/chunk_*.txt; do
    name=$(basename "$f" .txt)
    out="chunks/models/$name"
    [ -d "$out/0" ] && { echo "skip $name (done)"; continue; }   # resume-safe
    mkdir -p "$out"
    colmap mapper \
        --database_path database.db \
        --image_path ../images \
        --image_list_path "$f" \
        --output_path "$out" \
        --Mapper.num_threads "$COLMAP_THREADS" \
        --Mapper.ba_global_function_tolerance 1e-5
done
```

Because of the `[ -d "$out/0" ]` guard, re-running the loop **skips finished
chunks** — so “do chunk 1 tonight, chunks 2–3 tomorrow” just works: rerun the
same loop each evening and it continues where it stopped.

Merge the chunk models pairwise (they share the overlap region):

```bash
colmap model_merger --input_path1 chunks/models/chunk_000/0 \
    --input_path2 chunks/models/chunk_001/0 --output_path sparse/merged
colmap model_merger --input_path1 sparse/merged \
    --input_path2 chunks/models/chunk_002/0 --output_path sparse/merged
# ...repeat for each subsequent chunk, then bundle-adjust the union:
colmap bundle_adjuster --input_path sparse/merged --output_path sparse/0
```

### 5B. Single hierarchical run (simpler, but one long job)

`hierarchical_mapper` auto-partitions the scene and merges, keeping RAM bounded:

```bash
colmap hierarchical_mapper \
    --database_path database.db \
    --image_path ../images \
    --output_path sparse \
    --Mapper.ba_global_function_tolerance 1e-5
# -> sparse/0
```

This runs to completion in one go (hours to overnight). To resume it after a
crash, continue registering the remaining images into the partial model:

```bash
colmap mapper --database_path database.db --image_path ../images \
    --input_path sparse/0 --output_path sparse/0
```
(Re-running `mapper` with `--input_path == --output_path` extends the existing
reconstruction rather than starting over.)

---

## 6. Georegister to metres (so depths are in metres)

Align the sparse model to the GPS track. `geo_ref.txt` is already local ENU
**metres**, so this is a similarity (scale+rotation+translation) fit — it sets
real-world scale, which is what makes the eventual depth maps metric.

```bash
colmap model_aligner \
    --input_path sparse/0 \
    --output_path sparse_geo \
    --ref_images_path geo_ref.txt \
    --ref_is_gps 0 \
    --alignment_type custom \
    --robust_alignment 1 \
    --robust_alignment_max_error 3.0
```

> Flag names vary slightly by COLMAP version (older builds use
> `--transform_path` / `--ref_is_gps 1` with lat/lon). If `--ref_is_gps 0`
> is rejected, run `colmap model_aligner -h` and match the flags. The GPS z is 0
> for every frame (surface track), which is fine — the horizontal spacing fixes
> scale.

Quick check that scale is sane:
```bash
colmap model_analyzer --path sparse_geo   # extent should read ~57 x 46 m
```

---

## 7. Notes on this survey’s density (optional decimation)

PALFREY packs 10,335 frames into ~57 × 46 m (~4 images/m²) — huge overlap. If
mapping is slow or you only need colour recovery, **decimating frames** speeds
everything up with little quality loss:

```bash
# keep every 2nd frame for the SfM list (Sea-thru still runs on ALL images later)
awk 'NR % 2 == 1' colmap/image_list.txt > colmap/image_list_half.txt
```
Use the decimated list for extraction/mapping; you can still produce depth for
every image because neighbouring poses interpolate. Start with full frames only
if you have the time budget.

---

## 8. Dense depth maps (what Sea-thru consumes)

### 8A. With a CUDA GPU (preferred)

```bash
colmap image_undistorter \
    --image_path ../images \
    --input_path sparse_geo \
    --output_path dense \
    --output_type COLMAP --max_image_size "$DENSE_MAX_IMAGE_SIZE"

colmap patch_match_stereo \
    --workspace_path dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.gpu_index "$GPU_INDEX" \
    --PatchMatchStereo.max_image_size "$DENSE_MAX_IMAGE_SIZE" \
    --PatchMatchStereo.geom_consistency 1 \
    --PatchMatchStereo.cache_size "$PMS_CACHE_GB"   # system-RAM cache (GB), not VRAM
# optional fused point cloud for QA / orthomosaic:
colmap stereo_fusion --workspace_path dense --workspace_format COLMAP \
    --input_type geometric --output_path dense/fused.ply
```

This writes `dense/stereo/depth_maps/<image>.geometric.bin` — metric range per
pixel. **That is exactly what `ColmapDepthSource` reads.**

### 8B. No CUDA GPU (CPU-only laptop)

`patch_match_stereo` needs CUDA, so on a CPU-only machine choose one:

- **Monocular depth anchored to SfM scale (in this repo).** Use the sparse model
  for metric scale and a neural net for the dense shape:
  ```bash
  # (in a Python <=3.12 env with torch)
  python -m seathru.cli --input-dir ../images \
      --csv ../processed_images.csv --out-dir ../seathru_out \
      --depth mono --mono-backend midas --full-res
  ```
  The mono source anchors each frame’s mean range to `depth_m` when present;
  for PALFREY (`depth_m == -1`) pass `--mono-near/--mono-far` estimated from the
  georegistered model’s camera height (see `model_analyzer`).
- **OpenMVS** (CPU dense) as a drop-in after COLMAP sparse, exporting depth maps.
- **Agisoft Metashape / OpenDroneMap** — CPU-capable dense + orthomosaic; export
  per-image depth and feed via `--depth file` (`FileDepthSource`).

---

## 9. Run Sea-thru with the COLMAP depth

Once `dense/stereo/depth_maps/` exists:

```bash
cd ~/ORCA_mapping/seathru_orca
python -m seathru.cli \
    --input-dir ../2024_02_PALFREY/images \
    --csv       ../2024_02_PALFREY/processed_images.csv \
    --out-dir   ../2024_02_PALFREY/seathru_out \
    --depth colmap \
    --colmap-workspace ../2024_02_PALFREY/colmap/dense \
    --full-res --debug
```

- `--depth colmap` reads `*.geometric.bin` directly (metric, no scaling needed).
- `--full-res` estimates the model at 1024 px but writes output at native
  resolution so it’s 1:1 comparable to the input.
- `--debug` saves a montage (input / backscatter / illuminant / β_D / recovered)
  per image for QA.

Images without a depth map (COLMAP couldn’t reconstruct them) are logged and
skipped; you can re-run them later through `--depth mono` or `--depth plane`.

---

## 10. Pause / resume / power-off cheat-sheet

| Stage | Checkpoint | Resume how |
| --- | --- | --- |
| Feature extraction (3) | `database.db` per image | re-run same command (skips done) |
| Matching (4) | `database.db` per pair | re-run same command (skips done) |
| Chunked mapping (5A) | `chunks/models/chunk_k/0` | re-run the loop (skips finished chunks) |
| Single mapping (5B) | `sparse/0` | `colmap mapper --input_path sparse/0 --output_path sparse/0` |
| Dense (8A) | per-image `.bin` in `dense/stereo/depth_maps` | re-run `patch_match_stereo` (skips existing) |

**Keep a job alive across a lid-close / SSH drop** — run it in `tmux` and stop
Ubuntu from sleeping:
```bash
sudo apt install tmux
tmux new -s colmap
# inside tmux, prevent sleep while the job runs:
systemd-inhibit --what=sleep:idle --why="colmap" bash run_stage.sh
# detach: Ctrl-b then d      reattach later: tmux attach -t colmap
```

**Freeze/thaw a running job without losing state** (same session, laptop stays
on):
```bash
pgrep -f colmap                 # find PID
kill -STOP <pid>                # pause (0% CPU, RAM held)
kill -CONT <pid>                # resume
```
`STOP/CONT` cannot survive a power-off (RAM is lost). To power off overnight,
stop at a **checkpoint boundary** above (finish the current chunk, then quit) —
that is the whole reason for the chunked workflow in 5A.

**Suggested cadence for “work laptop by day, COLMAP by night”:**
1. Night 1: Steps 3–4 (extraction + matching) — resumable, leave in tmux.
2. Night 2+: one or two chunks from 5A per night; rerun the loop each evening.
3. A later night: merge (6) + georegister (6) + dense (8) once mapping is done.
4. Finally: Step 9 Sea-thru (fast; ~seconds–minutes per image, also resumable
   since it skips images that already have an output).

---

## 11. Troubleshooting

- **`patch_match_stereo` errors about CUDA** → COLMAP built without GPU; use 8B.
- **Mapper makes several disconnected models** (`sparse/0`, `sparse/1`, …) →
  overlap between survey lines is thin; raise `--radius`/`--max-neighbors` in
  Step 2 and re-match, or merge the sub-models in Step 6.
- **Georegistration fails / huge error** → check `geo_ref.txt` names match image
  filenames exactly; lower `--robust_alignment_max_error`.
- **Out of RAM during matching** → lower `--max-neighbors`, `--radius`, and
  `--SiftExtraction.max_image_size`; decimate frames (Step 7).
- **Depths look inverted / wrong scale in Sea-thru** → confirm you pointed
  `--colmap-workspace` at the **dense** folder and that Step 6 georegistration
  succeeded (`model_analyzer` extent ≈ real survey size).

---

## 11.5 Running in WSL, in the background, while you use Windows

You can grind through the whole **sparse** reconstruction in a WSL terminal at
low priority while Fusion 360 (etc.) keeps running, then do the GPU **dense**
step when the machine is free. This works well because COLMAP's long pole — the
incremental **mapper** — is **CPU-only**, so during the multi-day part there is
no GPU contention with your CAD work at all.

Helper files in this repo: `scripts/run_colmap_wsl.sh` (the low-priority
resumable runner) and `docs/wslconfig-example.txt`.

### Three WSL gotchas to handle first

1. **Keep the workspace on ext4, not `/mnt/c`.** COLMAP pounds a SQLite
   database with random I/O; on the `/mnt/c` (DrvFs) bridge that is slow and can
   lock/corrupt. **Read** images from `/mnt/c/...`, but put `database.db` and all
   models under `$HOME` (ext4). The runner script defaults to
   `~/palfrey_colmap`.
2. **OneDrive: make the images local.** Your images are in a OneDrive folder. In
   Windows Explorer, right-click `2024_02_PALFREY` → **"Always keep on this
   device"**, or COLMAP will stall trying to hydrate cloud-only placeholders.
3. **Windows sleep freezes WSL2.** Your safe stop points for a real
   shutdown/reboot are the **chunk boundaries** (finished chunks are on disk).
   Consider keeping the PC awake while a chunk is running.

### Cap WSL's resources so Windows stays responsive (once)

Copy `docs/wslconfig-example.txt` to `C:\Users\mickv\.wslconfig`, then in
PowerShell:

```powershell
wsl --shutdown        # reopen your WSL terminal afterwards
```

It caps WSL to 6 of 16 cores and 12 GB RAM — a hard guarantee Fusion keeps the
rest, independent of `nice`. Raise the numbers when you are not using the PC.

### Install COLMAP in WSL

```bash
sudo apt update && sudo apt install colmap tmux
colmap -h
nvidia-smi            # should list your RTX 3050 (WSL2 GPU passthrough)
```

The Ubuntu package runs CPU feature-extraction, matching, and mapping out of the
box — enough for the entire sparse pipeline. GPU **dense** (`patch_match_stereo`,
Step 8A) needs a CUDA-enabled build; if the apt build refuses, build from source
with CUDA (Step 1C) or run the dense step later. You do **not** need the GPU for
the background sparse work.

### Prep (fast, pure Python)

```bash
cd /mnt/c/Users/mickv/OneDrive/Documents/ORCA_mapping/seathru_orca
WORK=$HOME/palfrey_colmap
CSV="/mnt/c/Users/mickv/OneDrive/Documents/ORCA_mapping/2024_02_PALFREY/processed_images.csv"
python scripts/colmap_geo_from_csv.py --csv "$CSV" --out-dir "$WORK"
python scripts/colmap_make_pairs.py   --csv "$CSV" --out "$WORK/pairs.txt" \
    --seq 10 --radius 3.0 --max-neighbors 40
# smaller chunks = you lose less if you stop mid-chunk:
python scripts/colmap_make_chunks.py  --image-list "$WORK/image_list.txt" \
    --out-dir "$WORK/chunks" --chunk 500 --overlap 100
```

### Run it in the background

```bash
tmux new -s colmap                       # survives closing the terminal
./scripts/run_colmap_wsl.sh              # Ctrl-b then d to detach
# reattach anytime:  tmux attach -t colmap
```

The runner wraps every COLMAP call in `nice -n 19 ionice -c3` (lowest CPU + idle
I/O), uses 4 threads and **CPU SIFT** by default so it never touches the GPU, and
logs to `~/palfrey_colmap/logs/`. Defaults (paths, `THREADS`, `USE_GPU`) are env-
overridable at the top of the script.

### Pause / stop / resume

| Want to… | Do |
| --- | --- |
| Free up cores for Fusion *now* (keep progress) | `pkill -STOP -f colmap` … `pkill -CONT -f colmap` to resume |
| Detach the terminal, keep running | `Ctrl-b` then `d` |
| Stop to reboot / shut down | `Ctrl-C` **between chunks**, then re-run the script later — it skips finished chunks |
| See progress | `ls ~/palfrey_colmap/chunks/models` (one folder per finished chunk) |

Feature extraction and matching are resumable at the database level (re-running
skips done work); mapping is resumable at chunk granularity. Nothing is lost on
a clean stop except an in-progress chunk — which is why 500-image chunks are a
good "stop often" size.

### Finish: merge → georegister → dense → Sea-thru

When all chunks exist, run **Step 6** (merge + `model_aligner`) and **Step 8A**
(`patch_match_stereo`, GPU — do this when Fusion is closed) from `$HOME`, then
point Sea-thru at the dense workspace (**Step 9**), e.g.:

```bash
cd /mnt/c/Users/mickv/OneDrive/Documents/ORCA_mapping/seathru_orca
python -m seathru.cli \
    --input-dir "/mnt/c/Users/mickv/OneDrive/Documents/ORCA_mapping/2024_02_PALFREY/images" \
    --csv "/mnt/c/Users/mickv/OneDrive/Documents/ORCA_mapping/2024_02_PALFREY/processed_images.csv" \
    --out-dir "/mnt/c/Users/mickv/OneDrive/Documents/ORCA_mapping/2024_02_PALFREY/seathru_out" \
    --depth colmap --colmap-workspace "$HOME/palfrey_colmap/dense" --full-res
```

### Then: corrected images → photogrammetry / orthomosaic

Colour correction does not change camera geometry, so you have two options for
the final map:

- **Reuse the COLMAP poses** you already computed and just re-texture/ortho with
  the Sea-thru-corrected images (fastest — geometry is done).
- Or run your photogrammetry tool (Metashape / ODM / a fresh COLMAP) on the
  corrected images from scratch. Corrected images often also give *slightly
  better* features (more contrast/colour), so a from-scratch run is a reasonable
  choice if you want the cleanest final orthomosaic.

---

## 12. Running on university HPC (SLURM)

The workflow is **fully transferable** to an HPC cluster, and two properties make
it a natural fit:

1. **Extraction + matching + dense** are single GPU jobs — request one GPU node,
   regenerate `pipeline.env` with `--profile hpc` (or explicit
   `--threads/--ram-gb/--vram-gb`), and run the exact same commands.
2. **Chunked mapping is embarrassingly parallel** — each `chunk_*.txt` is
   independent, so a **SLURM job array** reconstructs all chunks *at once*
   instead of one-per-night. This is the big speed-up over the laptop.

### Containerise once (HPC usually has no Docker — use Apptainer/Singularity)

```bash
# on a machine with Docker/internet, or pull directly on the cluster:
apptainer pull colmap.sif docker://colmap/colmap:latest
# run any command as:  apptainer exec --nv colmap.sif colmap <args>
#   --nv  exposes the GPU inside the container
```

If the cluster instead provides COLMAP as a module, use `module load colmap`
(and `module load cuda`) in place of the container, and drop `apptainer exec
--nv colmap.sif` from the commands below.

### Stage the data to fast scratch

```bash
# copy images + colmap/ to node-local or parallel scratch (NOT your home dir)
export WORK=$SCRATCH/palfrey
mkdir -p "$WORK" && cp -r ~/ORCA_mapping/2024_02_PALFREY/{images,colmap} "$WORK"
cd "$WORK/colmap"
python ~/ORCA_mapping/seathru_orca/scripts/hw_profile.py --profile hpc --out pipeline.env
```

### Job 1 — extraction + matching (one GPU job)

`extract_match.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=colmap_feat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
set -e
cd "$SCRATCH/palfrey/colmap"; source pipeline.env
SIF="apptainer exec --nv $HOME/colmap.sif"
$SIF colmap feature_extractor --database_path database.db --image_path ../images \
    --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV \
    --SiftExtraction.max_image_size "$MAX_IMAGE_SIZE" \
    --SiftExtraction.max_num_features "$MAX_NUM_FEATURES" --SiftExtraction.use_gpu 1
$SIF colmap matches_importer --database_path database.db \
    --match_list_path pairs.txt --match_type pairs --SiftMatching.use_gpu 1
```

### Job 2 — chunked mapping as a job array (the parallel win)

Generate chunks first (`colmap_make_chunks.py` as in Step 5A). If there are 14
chunks, submit `--array=0-13`; each task maps one chunk on CPUs:

`map_array.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=colmap_map
#SBATCH --array=0-13%6          # 14 chunks, at most 6 running at once
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
set -e
cd "$SCRATCH/palfrey/colmap"; source pipeline.env
f=$(printf "chunks/chunk_%03d.txt" "$SLURM_ARRAY_TASK_ID")
out="chunks/models/$(basename "$f" .txt)"
[ -d "$out/0" ] && exit 0        # resumable: skip finished chunk
mkdir -p "$out"
apptainer exec $HOME/colmap.sif colmap mapper \
    --database_path database.db --image_path ../images \
    --image_list_path "$f" --output_path "$out" \
    --Mapper.num_threads "$SLURM_CPUS_PER_TASK"
```

Submit with a dependency so mapping waits for matching:

```bash
jid=$(sbatch --parsable extract_match.sbatch)
sbatch --dependency=afterok:$jid map_array.sbatch
```

Then run merge + georegister + dense (Steps 6, 8A) as a final single GPU job.
Because the chunk-skip guard (`[ -d "$out/0" ]`) is identical to the laptop
loop, a requeued or timed-out array task just resumes.

### Bring results back and run Sea-thru anywhere

Copy `colmap/dense/stereo/depth_maps/` back (or run Sea-thru on the node — it is
light). Sea-thru itself does not need the HPC:

```bash
python -m seathru.cli --input-dir images --csv processed_images.csv \
    --out-dir seathru_out --depth colmap --colmap-workspace colmap/dense --full-res
```

> **Right-sizing the request:** match `--cpus-per-task` to `COLMAP_THREADS`,
> `--mem` to your `pipeline.env` RAM assumption, and keep `%N` in `--array` so
> you don't flood the scheduler. Dense stereo wants the GPU node; sparse mapping
> does not — splitting them (Jobs 1/3 GPU, Job 2 CPU array) uses the allocation
> efficiently.
