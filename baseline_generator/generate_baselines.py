#!/usr/bin/env python3
"""
Baseline width-map creator (dataset-wide), with pandarallel parallelism.

INPUT:
  --in_dir : directory of mask images (grayscale or RGB). Masks assumed "crack=white".
OUTPUT:
  --out_dir/<image_stem>/<method>/
      width_map.npz   (width_map float32 HxW, support_mask uint8 HxW, meta dict)
  --out_dir/<image_stem>/baselines_overview.png   (optional)

Also writes:
  --out_dir/baseline_timings.csv
  --out_dir/baseline_timings_summary.csv

GPU:
  Uses cucim + cupy medial_axis, serialized by a lock so only one worker uses GPU at a time.
"""

import os
import sys
import time
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd

from pandarallel import pandarallel

from skimage.io import imread
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import distance_transform_edt
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from baseline_plots import *

# ----------------------------
# Optional GPU deps (cucim/cupy)
# ----------------------------
GPU_OK = True
try:
    import cupy as cp
    from cucim.skimage.morphology import medial_axis as cucim_medial_axis
except Exception:
    GPU_OK = False

# DSE pruning (your local module)
DSE_OK = True
try:
    from dsepruning.dsepruning import skel_pruning_DSE
except Exception:
    DSE_OK = False

os.environ["QT_LOGGING_RULES"] = "qt.qpa.*=false"

# ----------------------------
# GPU lock (serialize GPU use)
# ----------------------------
def _gpu_lock_ctx(lock_path: str):
    """
    Simple Linux file lock context for serializing GPU work across processes.
    Works in WSL.
    """
    import contextlib
    import fcntl

    @contextlib.contextmanager
    def _ctx():
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        with open(lock_path, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    return _ctx()


# ----------------------------
# Helpers
# ----------------------------
def _read_mask(path: str, threshold: float) -> np.ndarray:
    """
    Read image and binarize to bool mask (True=crack).
    Supports grayscale or RGB.
    """
    img = imread(path)
    if img.ndim == 3:
        # naive luminance
        img = img[..., :3].astype(np.float32)
        gray = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        # normalize if looks like uint8
        if gray.max() > 1.5:
            gray = gray / 255.0
    else:
        gray = img.astype(np.float32)
        if gray.max() > 1.5:
            gray = gray / 255.0

    bw = (gray > float(threshold))
    return bw


def _safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _save_npz(out_path: str, width_map: np.ndarray, support_mask: np.ndarray, meta: Dict[str, Any]) -> None:
    _safe_mkdir(os.path.dirname(out_path))
    np.savez_compressed(
        out_path,
        width_map=width_map.astype(np.float32),
        support_mask=support_mask.astype(np.uint8),
        meta=json.dumps(meta),
    )


def _plot_overview(
    out_path: str,
    bw: np.ndarray,
    panels: List[Tuple[str, np.ndarray, np.ndarray]],
) -> None:
    """
    panels: list of (label, width_map, support_mask)
    """
    _safe_mkdir(os.path.dirname(out_path))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6), dpi=200)
    if n == 1:
        axes = [axes]

    for ax, (label, wmap, supp) in zip(axes, panels):
        ax.imshow(bw, cmap="gray", alpha=0.50)
        ys, xs = np.nonzero(supp)
        if len(xs) > 0:
            sc = ax.scatter(xs, ys, c=wmap[ys, xs], s=6, cmap="plasma")
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label="Width (px)")
        ax.set_title(label)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


# ----------------------------
# Baseline methods
# ----------------------------

def width_medial_gpu(
    bw: np.ndarray,
    min_area_px: int,
    gpu_lock_path: str,
) -> Dict[str, Any]:
    """
    GPU medial axis (cucim), returning BOTH:
      - raw medial axis (no DSE)
      - pruned medial axis (with DSE, if available)

    Also returns fine-grained timing:
      - medial_gpu_s : cucim medial axis + distance
      - dse_cpu_s    : DSE pruning only
    """
    if not GPU_OK:
        raise RuntimeError("GPU deps not available (cupy/cucim import failed).")

    timings = {}

    # --------------------------------------------------
    # (1) GPU medial axis + distance
    # --------------------------------------------------
    t0 = time.perf_counter()
    with _gpu_lock_ctx(gpu_lock_path):
        bw_gpu = cp.asarray(bw.astype(cp.bool_))
        sk_gpu, dist_gpu = cucim_medial_axis(bw_gpu, return_distance=True)

        sk = cp.asnumpy(sk_gpu).astype(bool)
        dist = cp.asnumpy(dist_gpu).astype(np.float32)
    t1 = time.perf_counter()

    timings["medial_gpu_s"] = float(t1 - t0)

    # -------------------------
    # Raw medial (no DSE)
    # -------------------------
    w_raw = np.zeros_like(dist, dtype=np.float32)
    w_raw[sk] = dist[sk] * 2.0
    supp_raw = sk.astype(np.uint8)

    results = {
        "medial": (w_raw, supp_raw),
    }

    # --------------------------------------------------
    # (2) DSE pruning (CPU, optional)
    # --------------------------------------------------
    if DSE_OK:
        t0 = time.perf_counter()
        pruned = skel_pruning_DSE(
            sk,
            dist,
            min_area_px=int(min_area_px),
            return_graph=False,
        ).astype(bool)
        t1 = time.perf_counter()

        timings["dse_cpu_s"] = float(t1 - t0)

        w_dse = np.zeros_like(dist, dtype=np.float32)
        w_dse[pruned] = dist[pruned] * 2.0
        supp_dse = pruned.astype(np.uint8)
    else:
        timings["dse_cpu_s"] = 0.0
        w_dse = w_raw.copy()
        supp_dse = supp_raw.copy()

    results["medial_dse"] = (w_dse, supp_dse)

    return {
        "results": results,
        "timings": timings,
    }


def _local_tangent_normal(y: int, x: int, skel: np.ndarray, window: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    y0, y1 = max(0, y - window), min(skel.shape[0], y + window + 1)
    x0, x1 = max(0, x - window), min(skel.shape[1], x + window + 1)
    local_points = np.column_stack(np.nonzero(skel[y0:y1, x0:x1]))
    if len(local_points) < 3:
        t = np.array([1.0, 0.0], dtype=np.float32)
        n = np.array([0.0, 1.0], dtype=np.float32)
        return t, n

    local_points = local_points + np.array([y0, x0])
    cov = np.cov(local_points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    tangent = eigvecs[:, int(np.argmax(eigvals))].astype(np.float32)
    tangent /= (np.linalg.norm(tangent) + 1e-12)
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
    normal /= (np.linalg.norm(normal) + 1e-12)
    return tangent, normal


def width_profile_normal_2023(
    bw: np.ndarray,
    skel: np.ndarray,
    window: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Profile-normal width estimation using a PRECOMPUTED skeleton.

    Width is estimated only at skeleton pixels by sampling along
    the local normal direction until mask exit.
    """
    supp = skel.astype(np.uint8)
    wmap = np.zeros_like(bw, dtype=np.float32)

    ys, xs = np.nonzero(skel)
    if len(xs) == 0:
        return wmap, supp

    profile_half_length = int(max(bw.shape))

    for (y, x) in zip(ys, xs):
        _, normal = _local_tangent_normal(
            int(y), int(x), skel, window=window
        )

        found = []
        for sign in (-1.0, 1.0):
            for i in range(1, profile_half_length):
                p = np.array([y, x], dtype=np.float32) + normal * (float(i) * sign)
                pi = np.round(p).astype(int)

                if not (0 <= pi[0] < bw.shape[0] and 0 <= pi[1] < bw.shape[1]):
                    break

                if not bw[pi[0], pi[1]]:
                    edge = (
                        np.array([y, x], dtype=np.float32)
                        + normal * (float(i - 1) * sign)
                    )
                    found.append(np.round(edge).astype(int))
                    break

        if len(found) == 2:
            wmap[y, x] = float(np.linalg.norm(found[0] - found[1]))

    return wmap, supp


def _edge_points_in_patch(patch: np.ndarray) -> np.ndarray:
    """
    Edge points inside a binary patch: pixels True that have a 4-neighbor False.
    Returns coords (row, col) within patch coordinates.
    """
    P = patch.astype(bool)
    Pp = np.pad(P, 1, mode="constant", constant_values=False)
    up    = Pp[:-2,  1:-1]
    down  = Pp[ 2:,  1:-1]
    left  = Pp[1:-1, :-2]
    right = Pp[1:-1,  2:]
    edge = P & (~up | ~down | ~left | ~right)
    return np.argwhere(edge)


def width_pca_local(bw: np.ndarray, skel: np.ndarray, dist_map: np.ndarray, patch_scale: float = 1.5, min_points: int = 4) -> np.ndarray:
    wmap = np.zeros_like(bw, dtype=np.float32)
    ys, xs = np.nonzero(skel)
    for y, x in zip(ys, xs):
        r = int(max(4, float(dist_map[y, x]) * float(patch_scale)))
        y0, y1 = max(0, y - r), min(bw.shape[0], y + r + 1)
        x0, x1 = max(0, x - r), min(bw.shape[1], x + r + 1)
        patch = bw[y0:y1, x0:x1]
        edges = _edge_points_in_patch(patch)
        if edges.shape[0] < int(min_points):
            continue

        edges = edges + np.array([y0, x0])
        pca = PCA(n_components=2)
        pca.fit(edges)
        minor_axis = pca.components_[1]
        proj = (edges - np.array([y, x])) @ minor_axis
        wmap[y, x] = float(proj.max() - proj.min())
    return wmap


def width_adaptive_pca(bw: np.ndarray, skel: np.ndarray, dist_map: np.ndarray, base_patch: int = 8, min_points: int = 4) -> np.ndarray:
    wmap = np.zeros_like(bw, dtype=np.float32)
    ys, xs = np.nonzero(skel)
    for y, x in zip(ys, xs):
        r = int(max(int(base_patch), float(dist_map[y, x]) * 2.0))
        y0, y1 = max(0, y - r), min(bw.shape[0], y + r + 1)
        x0, x1 = max(0, x - r), min(bw.shape[1], x + r + 1)
        patch = bw[y0:y1, x0:x1]
        edges = _edge_points_in_patch(patch)
        if edges.shape[0] < int(min_points):
            continue

        edges = edges + np.array([y0, x0])
        pca = PCA(n_components=2)
        pca.fit(edges)
        minor_axis = pca.components_[1]
        proj = (edges - np.array([y, x])) @ minor_axis
        wmap[y, x] = float(proj.max() - proj.min())
    return wmap


def plot_geometry_comparison(
    out_path: str,
    bw: np.ndarray,
    geoms: List[Tuple[str, np.ndarray]],
):
    """
    geoms: list of (label, binary_mask)
    """
    _safe_mkdir(os.path.dirname(out_path))

    n = len(geoms)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6), dpi=200)
    if n == 1:
        axes = [axes]

    for ax, (label, mask) in zip(axes, geoms):
        ax.imshow(bw, cmap="gray", alpha=0.35)
        ys, xs = np.nonzero(mask)
        ax.scatter(xs, ys, s=3, c="red")
        ax.set_title(label)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


# ----------------------------
# Per-image worker
# ----------------------------
def process_one_image(row: pd.Series, cfg: Dict[str, Any]) -> Dict[str, Any]:
    path = row["path"]
    stem = row["stem"]

    threshold = float(cfg["threshold"])
    min_area_px = int(cfg["min_area_px"])
    out_dir = cfg["out_dir"]
    methods = cfg["methods"]
    make_plots = bool(cfg["make_plots"])
    gpu_lock_path = cfg["gpu_lock_path"]

    t0_total = time.perf_counter()

    # --------------------------------------------------
    # Load + clean mask
    # --------------------------------------------------
    bw0 = _read_mask(path, threshold=threshold)
    bw = remove_small_objects(bw0.astype(bool), min_size=min_area_px).astype(bool)

    crack_px = int(np.count_nonzero(bw))
    H, W = bw.shape[:2]

    img_out_root = os.path.join(out_dir, stem)
    _safe_mkdir(img_out_root)

    timings = {}
    panels = []

    # --------------------------------------------------
    # Shared CPU geometry
    # --------------------------------------------------
    skel = skeletonize(bw).astype(bool)
    dist_map = distance_transform_edt(bw).astype(np.float32)

    medial_mask = None
    medial_dse_mask = None

    # --------------------------------------------------
    # (A) Medial axis (GPU, single call → raw + DSE, timed)
    # --------------------------------------------------
    if "medial_dse" in methods:
        try:
            medial_out = width_medial_gpu(
                bw,
                min_area_px=min_area_px,
                gpu_lock_path=gpu_lock_path,
            )
            medial_results = medial_out["results"]
            timings.update(medial_out["timings"])
            ok = True
        except Exception as e:
            medial_results = {}
            ok = False
            timings["medial_error"] = repr(e)

        # ---- raw medial (no DSE) ----
        if "medial" in medial_results:
            wmap, supp = medial_results["medial"]
            medial_mask = supp.astype(bool)

            outp = os.path.join(img_out_root, "medial", "width_map.npz")
            meta = {
                "method": "medial",
                "ok": ok,
                "H": H,
                "W": W,
                "crack_px": crack_px,
            }
            _save_npz(outp, wmap, supp, meta)

            if make_plots:
                panels.append(("Medial (no DSE)", wmap, medial_mask))

        # ---- medial + DSE ----
        if "medial_dse" in medial_results:
            wmap, supp = medial_results["medial_dse"]
            medial_dse_mask = supp.astype(bool)

            outp = os.path.join(img_out_root, "medial_dse", "width_map.npz")
            meta = {
                "method": "medial_dse",
                "ok": ok,
                "H": H,
                "W": W,
                "crack_px": crack_px,
            }
            _save_npz(outp, wmap, supp, meta)

            if make_plots:
                panels.append(("Medial + DSE", wmap, medial_dse_mask))

    # --------------------------------------------------
    # (B) Profile-normal (CPU)
    # --------------------------------------------------
    if "profile_normal" in methods:
        t0 = time.perf_counter()
        wmap, supp = width_profile_normal_2023(
            bw,
            skel=skel,
            window=int(cfg["profile_window"]),
        )
        t1 = time.perf_counter()
        timings["profile_normal_s"] = float(t1 - t0)

        outp = os.path.join(img_out_root, "profile_normal", "width_map.npz")
        meta = {
            "method": "profile_normal",
            "ok": True,
            "H": H,
            "W": W,
            "crack_px": crack_px,
        }
        _save_npz(outp, wmap, supp, meta)

        if make_plots:
            panels.append(("Profile-Normal", wmap, supp.astype(bool)))

    # --------------------------------------------------
    # (C) PCA-local (CPU)
    # --------------------------------------------------
    if "pca_local" in methods:
        t0 = time.perf_counter()
        wmap = width_pca_local(
            bw,
            skel=skel,
            dist_map=dist_map,
            patch_scale=float(cfg["pca_patch_scale"]),
            min_points=int(cfg["pca_min_points"]),
        )
        t1 = time.perf_counter()
        timings["pca_local_s"] = float(t1 - t0)

        outp = os.path.join(img_out_root, "pca_local", "width_map.npz")
        meta = {
            "method": "pca_local",
            "ok": True,
            "H": H,
            "W": W,
            "crack_px": crack_px,
        }
        _save_npz(outp, wmap, skel.astype(np.uint8), meta)

        if make_plots:
            panels.append(("PCA-Local", wmap, skel))

    # --------------------------------------------------
    # (D) Adaptive PCA (CPU)
    # --------------------------------------------------
    if "adaptive_pca" in methods:
        t0 = time.perf_counter()
        wmap = width_adaptive_pca(
            bw,
            skel=skel,
            dist_map=dist_map,
            base_patch=int(cfg["adaptive_base_patch"]),
            min_points=int(cfg["pca_min_points"]),
        )
        t1 = time.perf_counter()
        timings["adaptive_pca_s"] = float(t1 - t0)

        outp = os.path.join(img_out_root, "adaptive_pca", "width_map.npz")
        meta = {
            "method": "adaptive_pca",
            "ok": True,
            "H": H,
            "W": W,
            "crack_px": crack_px,
        }
        _save_npz(outp, wmap, skel.astype(np.uint8), meta)

        if make_plots:
            panels.append(("Adaptive PCA", wmap, skel))

    # --------------------------------------------------
    # Geometry-only comparison plot (skeleton vs medial)
    # --------------------------------------------------
    if make_plots:
        geoms = [("Skeletonize", skel)]
        if medial_mask is not None:
            geoms.append(("Medial (no DSE)", medial_mask))
        if medial_dse_mask is not None:
            geoms.append(("Medial + DSE", medial_dse_mask))

        plot_geometry_comparison(
            os.path.join(img_out_root, "geometry_comparison.png"),
            bw=bw,
            geoms=geoms,
        )

    # --------------------------------------------------
    # Width overview plot
    # --------------------------------------------------
    if make_plots and panels:
        plot_path = os.path.join(img_out_root, "baselines_overview.png")
        _plot_overview(plot_path, bw=bw, panels=panels)

    t1_total = time.perf_counter()

    return {
        "path": path,
        "stem": stem,
        "H": H,
        "W": W,
        "crack_px": crack_px,
        "total_s": float(t1_total - t0_total),
        **timings,
    }


# ----------------------------
# Main (HARDCODED CONFIG)
# ----------------------------
import wslPath

def normalize_path(p):
    """
    Accepts Windows or Linux paths and returns a WSL-safe POSIX path.
    """
    if not isinstance(p, str):
        return p

    if wslPath.is_windows_path(p):
        return wslPath.to_posix(p)

    return p


def main():
    import os
    import pandas as pd
    from pandarallel import pandarallel
    import wslPath

    # ============================================================
    # HARD-CODED USER CONFIG (WINDOWS PATHS OK)
    # ============================================================
    WIN_IN_DIR  = r"C:\Users\13144\Documents\Masters_Thesis\datasets\SUT_Compressed\Ground Truth"
    WIN_OUT_DIR = r"C:\Users\13144\Documents\Masters_Thesis\datasets\SUT_Compressed\Outputs\baselines"

    # --- convert to WSL paths once ---
    IN_DIR  = wslPath.to_posix(WIN_IN_DIR)  if wslPath.is_windows_path(WIN_IN_DIR)  else WIN_IN_DIR
    OUT_DIR = wslPath.to_posix(WIN_OUT_DIR) if wslPath.is_windows_path(WIN_OUT_DIR) else WIN_OUT_DIR

    THRESHOLD   = 0.25
    MIN_AREA_PX = 1000

    METHODS = [
        "medial_dse",
        "profile_normal",
        "pca_local",
        "adaptive_pca",
        # "distance_ridge",  # optional
    ]

    MAKE_PLOTS = True

    PROFILE_WINDOW      = 5
    PCA_PATCH_SCALE     = 1.5
    PCA_MIN_POINTS      = 4
    ADAPTIVE_BASE_PATCH = 8

    GPU_LOCK_PATH = "/tmp/pandarallel_gpu_lock/lock.txt"
    
    import os
    import subprocess
    import re

    def physical_cores_linux():
        try:
            out = subprocess.check_output(["lscpu"], text=True)
            cores = int(re.search(r"Core\(s\) per socket:\s+(\d+)", out).group(1))
            sockets = int(re.search(r"Socket\(s\):\s+(\d+)", out).group(1))
            return cores * sockets
        except Exception:
            return os.cpu_count() or 4


    # pandarallel
    PHYSICAL_CORES = physical_cores_linux()
    NB_WORKERS = max(1, PHYSICAL_CORES - 1)
    print(PHYSICAL_CORES)
    PROGRESS_BAR = False

    EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    # ============================================================
    # Setup
    # ============================================================
    _safe_mkdir(OUT_DIR)

    pandarallel.initialize(
        nb_workers=NB_WORKERS,
        progress_bar=PROGRESS_BAR,
        verbose=0,
    )

    # ============================================================
    # Collect images
    # ============================================================
    paths = []

    for root, _, files in os.walk(IN_DIR):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in EXTS:
                paths.append(os.path.join(root, fn))

    # hard cap
    paths = paths[:4]

    paths = sorted(paths)

    if not paths:
        print(f"[baseline_creator] No images found under: {IN_DIR}")
        return 2

    df = pd.DataFrame({
        "path": paths,
        "stem": [os.path.splitext(os.path.basename(p))[0] for p in paths],
    })

    cfg = {
        "threshold": THRESHOLD,
        "min_area_px": MIN_AREA_PX,
        "out_dir": OUT_DIR,
        "methods": METHODS,
        "make_plots": MAKE_PLOTS,
        "profile_window": PROFILE_WINDOW,
        "pca_patch_scale": PCA_PATCH_SCALE,
        "pca_min_points": PCA_MIN_POINTS,
        "adaptive_base_patch": ADAPTIVE_BASE_PATCH,
        "gpu_lock_path": GPU_LOCK_PATH,
    }

    # ============================================================
    # Sanity info
    # ============================================================
    print("[baseline_creator] images:", len(df))
    print("[baseline_creator] workers:", NB_WORKERS)
    print("[baseline_creator] methods:", METHODS)
    print("[baseline_creator] GPU_OK:", GPU_OK, " DSE_OK:", DSE_OK)
    print("[baseline_creator] IN_DIR :", IN_DIR)
    print("[baseline_creator] OUT_DIR:", OUT_DIR)

    if "medial_dse" in METHODS and not GPU_OK:
        print("[baseline_creator] WARNING: medial_dse requested but GPU deps unavailable.")

    # ============================================================
    # Parallel processing
    # ============================================================
    results = df.parallel_apply(
        lambda r: process_one_image(r, cfg),
        axis=1,
    )

    res_df = pd.DataFrame(list(results.values))

    # ============================================================
    # Timing CSVs
    # ============================================================
    timings_csv = os.path.join(OUT_DIR, "baseline_timings.csv")
    res_df.to_csv(timings_csv, index=False)
    print("[baseline_creator] wrote:", timings_csv)

    # ---- summary stats (committee-friendly) ----
    weight = res_df["crack_px"].fillna(0).astype(float)
    wsum = float(weight.sum()) if float(weight.sum()) > 0 else 1.0

    method_cols = [c for c in res_df.columns if c.endswith("_s") and c != "total_s"]
    summary_rows = []

    summary_rows.extend([
        {"metric": "total_images", "value": int(len(res_df))},
        {"metric": "total_crack_px", "value": float(weight.sum())},
        {"metric": "total_wall_s_sum", "value": float(res_df["total_s"].sum())},
        {"metric": "total_wall_s_mean", "value": float(res_df["total_s"].mean())},
        {
            "metric": "total_wall_s_weighted_mean_by_crack_px",
            "value": float((res_df["total_s"] * weight).sum() / wsum),
        },
    ])

    for c in method_cols:
        summary_rows.extend([
            {"metric": f"{c}_sum", "value": float(res_df[c].sum())},
            {"metric": f"{c}_mean", "value": float(res_df[c].mean())},
            {
                "metric": f"{c}_weighted_mean_by_crack_px",
                "value": float((res_df[c] * weight).sum() / wsum),
            },
        ])

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(OUT_DIR, "baseline_timings_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print("[baseline_creator] wrote:", summary_csv)
    
    # ============================================================
    # Thesis committee plots
    # ============================================================
    if MAKE_PLOTS:
        print("[baseline_creator] generating summary plots...")
        plot_baseline_summary(res_df, summary_df, OUT_DIR)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

