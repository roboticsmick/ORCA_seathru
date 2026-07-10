#!/bin/bash
# ---------------------------------------------------------------------------
# run_colmap_wsl.sh — low-priority, resumable COLMAP sparse pipeline for WSL.
#
# Runs feature extraction -> matching -> chunked mapping at the LOWEST CPU/IO
# priority so it can grind through the dataset in the background while you use
# Windows (Fusion 360 etc). Every stage is resumable: re-run this script any
# time and it continues where it stopped (features/matches skip done items,
# mapping skips finished chunks).
#
# KEY DESIGN CHOICES for WSL + OneDrive:
#   * Images are READ from /mnt/c (fine, mostly sequential).
#   * The WORKSPACE (database.db, models) lives on ext4 ($HOME) — DrvFs/SQLite
#     on /mnt/c is slow and can corrupt/lock. Do not put database.db on /mnt/c.
#   * CPU SIFT + CPU mapper by default (USE_GPU=0) so it never fights Fusion for
#     the GPU. Flip USE_GPU=1 for the one-off feature/match step when the GPU is
#     free (the long mapping stage is CPU-only either way).
#
# PREP (run once, fast, pure Python — from the seathru_orca dir):
#   python scripts/colmap_geo_from_csv.py  --csv "$CSV" --out-dir "$WORK"
#   python scripts/colmap_make_pairs.py    --csv "$CSV" --out "$WORK/pairs.txt" \
#          --seq 10 --radius 3.0 --max-neighbors 40
#   python scripts/colmap_make_chunks.py   --image-list "$WORK/image_list.txt" \
#          --out-dir "$WORK/chunks" --chunk 500 --overlap 100
#   (smaller chunks = you lose less if you stop mid-chunk; 500/100 is a good
#    "stop often" size. Finished chunks are never redone.)
#
# USAGE:
#   tmux new -s colmap            # so closing the terminal doesn't kill it
#   ./scripts/run_colmap_wsl.sh   # Ctrl-b d to detach; tmux attach -t colmap
#
# STOP / PAUSE / RESUME:  see the block at the bottom of this file.
# ---------------------------------------------------------------------------
set -uo pipefail

# ---- EDIT THESE (defaults match your machine) -----------------------------
BASE="/mnt/c/Users/mickv/OneDrive/Documents/ORCA_mapping/2024_02_PALFREY"
IMAGES="${IMAGES:-$BASE/images}"
CSV="${CSV:-$BASE/processed_images.csv}"
WORK="${WORK:-$HOME/palfrey_colmap}"     # <-- ext4, NOT /mnt/c
THREADS="${THREADS:-4}"                    # COLMAP cores (of 16; leaves 12 for Windows)
USE_GPU="${USE_GPU:-0}"                    # 0 = CPU (background-friendly), 1 = GPU
CAMERA_MODEL="${CAMERA_MODEL:-OPENCV}"     # OPENCV_FISHEYE if shot very wide
MAX_IMAGE_SIZE="${MAX_IMAGE_SIZE:-1600}"
# ---------------------------------------------------------------------------

LOWPRIO="nice -n 19 ionice -c3"            # lowest CPU + idle IO priority
mkdir -p "$WORK"/{sparse,chunks/models,logs}
LOG="$WORK/logs/run_$(date +%Y%m%d_%H%M%S).log"
echo "Workspace: $WORK   Images: $IMAGES   THREADS=$THREADS USE_GPU=$USE_GPU"
echo "Logging to $LOG"

run() { echo ">>> $*" | tee -a "$LOG"; $LOWPRIO "$@" 2>&1 | tee -a "$LOG"; }

cd "$WORK"

# Sanity: prep files must exist.
for f in pairs.txt chunks; do
    if [ ! -e "$WORK/$f" ]; then
        echo "MISSING $WORK/$f — run the PREP steps in this script's header first."
        exit 1
    fi
done

# --- 1. Feature extraction (resumable: skips already-extracted images) ------
run colmap feature_extractor \
    --database_path database.db \
    --image_path "$IMAGES" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model "$CAMERA_MODEL" \
    --SiftExtraction.max_image_size "$MAX_IMAGE_SIZE" \
    --SiftExtraction.max_num_features 8192 \
    --SiftExtraction.num_threads "$THREADS" \
    --SiftExtraction.use_gpu "$USE_GPU"

# --- 2. Match only the GPS/time neighbour pairs (resumable) -----------------
run colmap matches_importer \
    --database_path database.db \
    --match_list_path pairs.txt \
    --match_type pairs \
    --SiftMatching.num_threads "$THREADS" \
    --SiftMatching.use_gpu "$USE_GPU"

# --- 3. Chunked incremental mapping (resumable per chunk) -------------------
# CPU-only regardless of USE_GPU. Each finished chunk is saved before the next
# starts, so stopping between chunks never loses work.
for f in chunks/chunk_*.txt; do
    name=$(basename "$f" .txt)
    out="chunks/models/$name"
    if [ -d "$out/0" ]; then
        echo "skip $name (already reconstructed)" | tee -a "$LOG"
        continue
    fi
    mkdir -p "$out"
    run colmap mapper \
        --database_path database.db \
        --image_path "$IMAGES" \
        --image_list_path "$f" \
        --output_path "$out" \
        --Mapper.num_threads "$THREADS"
done

echo "=== Sparse chunks complete. Next (see COLMAP_GUIDE Steps 6/8): ===" | tee -a "$LOG"
echo "  merge chunks -> georegister (model_aligner) -> dense (patch_match_stereo, GPU)" | tee -a "$LOG"

# ---------------------------------------------------------------------------
# STOP / PAUSE / RESUME cheat-sheet
# ---------------------------------------------------------------------------
# Instant pause (keep RAM, 0% CPU — e.g. you need cores for Fusion right now):
#     pkill -STOP -f colmap        # freeze
#     pkill -CONT -f colmap        # thaw
#
# Stop to reboot / shut down Windows (do it between chunks — watch the log for
# "skip"/next-chunk lines, then):
#     Ctrl-C   (or: tmux attach then Ctrl-C)
#   ...later, just run this script again — it resumes at the next unfinished
#   chunk. Nothing is lost except an in-progress chunk (keep chunks small).
#
# Detach without stopping:  Ctrl-b then d   (reattach: tmux attach -t colmap)
# Check progress:           ls chunks/models   (one folder per finished chunk)
# ---------------------------------------------------------------------------
