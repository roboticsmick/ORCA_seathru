"""
@file hw_profile.py
@brief Detect (or declare) hardware and emit COLMAP pipeline tuning parameters.

The COLMAP -> Sea-thru workflow is one pipeline whose cost knobs (thread count,
image sizes, GPU use, dense-stereo cache, chunk size, match density) should
scale with the machine it runs on. This script picks sensible values so the
*same* commands run well on:

  * this laptop      (Ryzen 7 5800H, 16 threads, 32 GB RAM, RTX 3050 ~4 GB VRAM)
  * a bigger desktop (more RAM/VRAM -> bigger chunks, larger images, denser match)
  * an HPC node      (many cores + big RAM -> few/no chunks; array-parallel)

It writes a shell-sourceable ``pipeline.env`` (used by the COLMAP commands in
docs/COLMAP_GUIDE.md) and prints a summary. Auto-detects by default; override
any value explicitly for reproducibility / HPC batch scripts.

Examples
--------
    # auto-detect this machine
    python scripts/hw_profile.py --out ../2024_02_PALFREY/colmap/pipeline.env

    # declare an HPC node (no detection), no chunking
    python scripts/hw_profile.py --threads 64 --ram-gb 256 --vram-gb 40 \
        --no-chunk --out colmap/pipeline.env

    # force a named preset
    python scripts/hw_profile.py --profile laptop --out colmap/pipeline.env
"""
from __future__ import annotations

import argparse
import ctypes
import os
import platform
import shutil
import subprocess
import sys


# --------------------------------------------------------------------------- #
# Detection                                                                   #
# --------------------------------------------------------------------------- #
def detect_threads():
    return os.cpu_count() or 4


def detect_ram_gb():
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / (1024 ** 2)  # kB -> GB
        elif system == "Windows":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / (1024 ** 3)
        elif system == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"])
            return int(out) / (1024 ** 3)
    except Exception:
        pass
    return 16.0  # conservative default


def detect_gpu():
    """Return (name, vram_gb) or (None, 0.0) if no NVIDIA GPU / no nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return None, 0.0
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"], text=True, timeout=10)
        name, mem = out.strip().splitlines()[0].split(",")
        return name.strip(), float(mem) / 1024.0  # MiB -> GiB
    except Exception:
        return None, 0.0


# --------------------------------------------------------------------------- #
# Parameter derivation                                                        #
# --------------------------------------------------------------------------- #
PRESETS = {
    "laptop":      dict(threads=16, ram_gb=32,  vram_gb=4),
    "workstation": dict(threads=32, ram_gb=128, vram_gb=12),
    "hpc":         dict(threads=64, ram_gb=256, vram_gb=40),
}


def derive(threads, ram_gb, vram_gb, has_gpu, no_chunk=False):
    """Map raw resources to COLMAP tuning parameters."""
    # Leave 1-2 cores for the OS on small machines, use all on big ones.
    colmap_threads = threads if threads >= 32 else max(2, threads - 2)

    # Feature/match image size: bigger machines can afford more detail.
    if ram_gb >= 96:
        max_image_size = 3200
    elif ram_gb >= 48:
        max_image_size = 2400
    else:
        max_image_size = 1600

    # Dense stereo is VRAM-bound.
    if vram_gb >= 20:
        dense_max_image_size = 3200
    elif vram_gb >= 10:
        dense_max_image_size = 2400
    elif vram_gb >= 6:
        dense_max_image_size = 2000
    else:                       # ~4 GB (laptop RTX 3050)
        dense_max_image_size = 1600

    # PatchMatch image cache lives in system RAM (GB). Use ~half, capped.
    pms_cache_gb = int(max(8, min(32, ram_gb * 0.5)))

    # Chunk size for incremental mapping scales with RAM. More RAM -> bigger
    # chunks -> fewer merges. Very big nodes can skip chunking entirely.
    if no_chunk or ram_gb >= 192:
        chunk_size = 0          # 0 => single hierarchical_mapper run, no chunks
    elif ram_gb >= 96:
        chunk_size = 3000
    elif ram_gb >= 48:
        chunk_size = 1500
    else:
        chunk_size = 900
    chunk_overlap = int(chunk_size * 0.17) if chunk_size else 0

    # Match density: more compute budget -> match more neighbours (better loop
    # closure / robustness) since it is no longer the bottleneck.
    if threads >= 48:
        seq, radius, max_neighbors = 15, 6.0, 100
    elif threads >= 24:
        seq, radius, max_neighbors = 12, 4.0, 60
    else:
        seq, radius, max_neighbors = 10, 3.0, 40

    return {
        "COLMAP_THREADS": colmap_threads,
        "USE_GPU": 1 if has_gpu else 0,
        "GPU_INDEX": 0,
        "MAX_IMAGE_SIZE": max_image_size,
        "MAX_NUM_FEATURES": 8192,
        "DENSE_MAX_IMAGE_SIZE": dense_max_image_size,
        "PMS_CACHE_GB": pms_cache_gb,
        "CHUNK_SIZE": chunk_size,
        "CHUNK_OVERLAP": chunk_overlap,
        "PAIR_SEQ": seq,
        "PAIR_RADIUS": radius,
        "PAIR_MAX_NEIGHBORS": max_neighbors,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=list(PRESETS),
                    help="Use a named preset instead of auto-detecting")
    ap.add_argument("--threads", type=int, help="Override logical thread count")
    ap.add_argument("--ram-gb", type=float, help="Override total RAM (GB)")
    ap.add_argument("--vram-gb", type=float, help="Override GPU VRAM (GB); 0 = no GPU")
    ap.add_argument("--no-chunk", action="store_true",
                    help="Force a single (unchunked) mapping run")
    ap.add_argument("--out", help="Write a sourceable pipeline.env here")
    args = ap.parse_args(argv)

    if args.profile:
        base = PRESETS[args.profile]
        threads, ram_gb, vram_gb = base["threads"], base["ram_gb"], base["vram_gb"]
        gpu_name = f"(preset {args.profile})"
        has_gpu = vram_gb > 0
    else:
        threads = detect_threads()
        ram_gb = detect_ram_gb()
        gpu_name, vram_gb = detect_gpu()
        has_gpu = vram_gb > 0

    if args.threads is not None:
        threads = args.threads
    if args.ram_gb is not None:
        ram_gb = args.ram_gb
    if args.vram_gb is not None:
        vram_gb = args.vram_gb
        has_gpu = vram_gb > 0

    params = derive(threads, ram_gb, vram_gb, has_gpu, args.no_chunk)

    print("Detected / declared hardware")
    print(f"  threads : {threads}")
    print(f"  RAM     : {ram_gb:.0f} GB")
    print(f"  GPU     : {gpu_name or 'none'}"
          + (f" (~{vram_gb:.0f} GB VRAM)" if has_gpu else " -> CPU-only "
             "(dense MVS needs the Step 8B fallback)"))
    print("\nDerived COLMAP pipeline parameters")
    for k, v in params.items():
        print(f"  {k:<22} {v}")
    if params["CHUNK_SIZE"] == 0:
        print("\n  CHUNK_SIZE=0 -> use the single hierarchical_mapper run (5B), "
              "no chunking needed.")

    if args.out:
        lines = ["# Generated by hw_profile.py - source before COLMAP commands.",
                 f"# threads={threads} ram={ram_gb:.0f}GB "
                 f"gpu={gpu_name or 'none'} vram={vram_gb:.0f}GB"]
        lines += [f'export {k}="{v}"' for k, v in params.items()]
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"\nWrote {args.out}  (source it: `source {os.path.basename(args.out)}`)")

    return params


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
