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
  --out_dir/width_baseline_timings.csv
  --out_dir/width_baseline_timings_summary.csv

GPU:
  Uses cucim + cupy medial_axis, serialized by a lock so only one worker uses GPU at a time.
"""

import os
import sys
import time
import json
import math
import re
import heapq
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from itertools import combinations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd

from pandarallel import pandarallel

from skimage.io import imread
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import distance_transform_edt, convolve, label as ndi_label, binary_dilation
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from width_baseline_plots import *

# Hardcoded switch:
# True  -> run full recomputation pipeline
# False -> only reload existing .npz outputs and regenerate plots
RECOMPUTE_BASELINES = True
RUN_DISTRIBUTION_ANALYSIS = True
REGENERATE_PER_IMAGE_PLOTS_FROM_NPZ = False
DIST_PLOTS_FROM_EXISTING_CSV_ONLY = False

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


def _natural_sort_key(path: str):
    """
    Sort by leading filename number first (e.g., 2.png, 2_modified.png),
    then by full name as a tie-breaker.
    """
    name = os.path.basename(path)
    stem, _ = os.path.splitext(name)
    m = re.match(r"^\s*(\d+)", stem)
    lead_num = int(m.group(1)) if m else float("inf")
    return (lead_num, stem.lower())


def _save_npz(
    out_path: str,
    *,
    width_map: np.ndarray,
    support_mask: np.ndarray,
    skel: np.ndarray,
    meta: Dict[str, Any],
) -> None:
    """
    Save baseline artifacts.

    REQUIRED fields:
      - width_map      (float32)
      - support_mask   (uint8 / bool)
      - skel           (uint8 / bool)  ← MANDATORY
      - meta           (JSON)

    This function MUST NOT silently omit geometry.
    """
    _safe_mkdir(os.path.dirname(out_path))

    if skel is None:
        raise ValueError(f"[SAVE_NPZ] skel is None for {out_path}")

    np.savez_compressed(
        out_path,
        width_map=width_map.astype(np.float32),
        support_mask=support_mask.astype(np.uint8),
        skel=skel.astype(np.uint8),
        meta=json.dumps(meta),
    )


def _plot_overview(
    out_path: str,
    bw: np.ndarray,
    panels: List[Tuple[str, np.ndarray, np.ndarray, str]],
    *,
    min_area_px: Optional[int] = None,
    mode: str = "grouped",
) -> None:
    """
    panels: list of (label, width_map, support_mask, method_key)
    mode:
      - "grouped": related method families share color scale.
      - "global_zero_to_max": all panels share [0, global_max] scale.
    """
    _safe_mkdir(os.path.dirname(out_path))

    def _vals(wmap: np.ndarray, supp: np.ndarray) -> np.ndarray:
        v = np.asarray(wmap)[np.asarray(supp).astype(bool)]
        if v.size == 0:
            return np.asarray([0.0], dtype=np.float32)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return np.asarray([0.0], dtype=np.float32)
        return v.astype(np.float32)

    def _group(method_key: str) -> str:
        if method_key in ("mat_width_raw", "mat_width_dse"):
            return "mat_family"
        if method_key in ("pca_width_dse", "esd_width_dse", "eob_width_dse"):
            return "dse_family"
        return method_key

    panel_limits: List[Tuple[float, float]] = []
    if mode == "grouped":
        grouped = {}
        for _, wmap, supp, method_key in panels:
            g = _group(method_key)
            vals = _vals(wmap, supp)
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
            if g not in grouped:
                grouped[g] = [vmin, vmax]
            else:
                grouped[g][0] = min(grouped[g][0], vmin)
                grouped[g][1] = max(grouped[g][1], vmax)
        for _, _, _, method_key in panels:
            vmin, vmax = grouped[_group(method_key)]
            if not np.isfinite(vmin):
                vmin = 0.0
            if not np.isfinite(vmax) or vmax <= vmin:
                vmax = vmin + 1.0
            panel_limits.append((float(vmin), float(vmax)))
    elif mode == "global_zero_to_max":
        all_max = 0.0
        for _, wmap, supp, _ in panels:
            vals = _vals(wmap, supp)
            vmax = float(np.max(vals))
            if np.isfinite(vmax):
                all_max = max(all_max, vmax)
        if all_max <= 0.0:
            all_max = 1.0
        panel_limits = [(0.0, float(all_max)) for _ in panels]
    else:
        raise ValueError(f"Unknown overview mode: {mode}")

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6), dpi=200)
    if n == 1:
        axes = [axes]

    for idx, (ax, (label, wmap, supp, _), (vmin, vmax)) in enumerate(zip(axes, panels, panel_limits)):
        ax.imshow(bw, cmap="gray", alpha=0.50)
        ys, xs = np.nonzero(supp)
        if len(xs) > 0:
            vals_local = np.asarray(wmap[ys, xs], dtype=np.float32)
            vals_local = vals_local[np.isfinite(vals_local)]
            local_max = float(np.max(vals_local)) if vals_local.size > 0 else float(vmax)
            sc = ax.scatter(
                xs,
                ys,
                c=wmap[ys, xs],
                s=6,
                cmap="plasma",
                vmin=vmin,
                vmax=vmax,
            )
            is_last = (idx == len(panels) - 1)
            cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            if is_last:
                cbar.set_label("Width (px)", fontsize=12)
            ticks = [float(vmin)]
            labels = [f"{vmin:.2f}"]
            # Show panel-local max only when clearly separated from global/family max.
            # This avoids visual overlap with the vmax tick label.
            local_max_gap_px = float(vmax) - float(local_max)
            if (
                float(vmin) < float(local_max) < float(vmax)
                and local_max_gap_px > 2.0
            ):
                ticks.append(float(local_max))
                labels.append(f"{local_max:.2f} (local max)")
            ticks.append(float(vmax))
            labels.append(f"{vmax:.2f}")
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(labels, fontsize=10)
            cbar.ax.tick_params(labelsize=10)
        ax.set_title(label, fontsize=12)
        ax.axis("off")

    fig.suptitle("Width Baseline Comparison", y=1.02)
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

    ALSO RETURNS:
      - raw skeleton (bool)
      - DSE-pruned skeleton (bool, if available)

    Timing:
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
        "medial": {
            "width_map": w_raw,
            "support_mask": supp_raw,
            "skel": sk,
        },
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

        results["medial_dse"] = {
            "width_map": w_dse,
            "support_mask": supp_dse,
            "skel": pruned,
        }
    else:
        timings["dse_cpu_s"] = 0.0
        results["medial_dse"] = {
            "width_map": w_raw.copy(),
            "support_mask": supp_raw.copy(),
            "skel": sk.copy(),
        }

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

#unused
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


'''def width_pca_local(bw: np.ndarray, skel: np.ndarray, dist_map: np.ndarray, patch_scale: float = 1.5, min_points: int = 4) -> np.ndarray:
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

#unused function
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
    return wmap'''
    
def width_pca_proj(
    bw: np.ndarray,
    skel: Optional[np.ndarray] = None,
    patch_radius: int = 7,
    *,
    dist_map: Optional[np.ndarray] = None,
    patch_scale: float = 1.5,
    min_radius: int = 5,
    max_radius: int = 100,
    min_points: int = 5,
) -> np.ndarray:
    import numpy as np

    if skel is None:
        raise ValueError("width_pca_proj requires `skel`.")

    wmap = np.zeros_like(bw, dtype=np.float32)
    edge_mask = (bw > 0) & (
        (np.roll(bw,1,0)==0) | (np.roll(bw,-1,0)==0) |
        (np.roll(bw,1,1)==0) | (np.roll(bw,-1,1)==0)
    )

    ys, xs = np.nonzero(skel)
    H, W = bw.shape

    for y, x in zip(ys, xs):
        if dist_map is not None:
            r = int(np.clip(float(dist_map[y, x]) * float(patch_scale), float(min_radius), float(max_radius)))
        else:
            r = int(max(1, patch_radius))

        y0, y1 = max(0, y - r), min(H, y + r + 1)
        x0, x1 = max(0, x - r), min(W, x + r + 1)

        patch = edge_mask[y0:y1, x0:x1]
        pts = np.column_stack(np.nonzero(patch))

        if len(pts) < int(min_points):
            continue

        pts = pts + np.array([y0, x0])

        # PCA
        mu = pts.mean(axis=0)
        X = pts - mu
        cov = X.T @ X / len(pts)

        eigvals, eigvecs = np.linalg.eig(cov)

        # minor axis (width direction)
        v = eigvecs[:, np.argmin(eigvals)]
        v = v / (np.linalg.norm(v) + 1e-8)

        # projection span
        proj = (pts - np.array([y, x])) @ v
        wmap[y, x] = proj.max() - proj.min()

    return wmap

def compute_shared_precompute(
    bw: np.ndarray,
    *,
    gpu_lock_path: str,
) -> Dict[str, Any]:
    """
    Stage A shared precomputation:
      - bw_mask
      - mat_skel (GPU medial axis when available)
      - dist_map from the same medial axis pass
    """
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()
    if GPU_OK:
        with _gpu_lock_ctx(gpu_lock_path):
            bw_gpu = cp.asarray(bw.astype(cp.bool_))
            sk_gpu, dist_gpu = cucim_medial_axis(bw_gpu, return_distance=True)
            mat_skel = cp.asnumpy(sk_gpu).astype(bool)
            dist_map = cp.asnumpy(dist_gpu).astype(np.float32)
        timings["shared_mat_gpu_s"] = float(time.perf_counter() - t0)
    else:
        # Fallback keeps pipeline runnable when GPU deps are unavailable.
        mat_skel = skeletonize(bw).astype(bool)
        dist_map = distance_transform_edt(bw).astype(np.float32)
        timings["shared_mat_gpu_s"] = 0.0
    return {
        "bw_mask": bw.astype(bool),
        "mat_skel": mat_skel,
        "dist_map": dist_map,
        "timings": timings,
    }


def _step_len_eq6(dy: int, dx: int) -> float:
    adx = abs(int(dx))
    ady = abs(int(dy))
    if adx > ady:
        return float(adx + (math.sqrt(2.0) - 1.0) * ady)
    return float(ady + (math.sqrt(2.0) - 1.0) * adx)


def _dijkstra_farthest(
    adj: Dict[int, List[Tuple[int, float, int]]],
    start: int,
) -> Tuple[int, Dict[int, float], Dict[int, int]]:
    dist: Dict[int, float] = {start: 0.0}
    prev: Dict[int, int] = {}
    pq: List[Tuple[float, int]] = [(0.0, int(start))]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        for v, w, _ in adj.get(u, []):
            nd = float(d + w)
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    far = start
    far_d = -1.0
    for n, d in dist.items():
        if d > far_d:
            far = int(n)
            far_d = float(d)
    return far, dist, prev


def graph_tree_diameter_prune_paperfaithful(skel: np.ndarray) -> np.ndarray:
    """
    Paper-faithful pruning:
      1) build reduced tree (endpoints/branchpoints as nodes)
      2) collapse degree-2 chains to weighted edges (Eq. 6 step metric)
      3) keep only maximum-diameter path on reduced weighted graph
      4) rasterize kept edge chains
    """
    sk = np.asarray(skel).astype(bool)
    out = np.zeros_like(sk, dtype=bool)
    if not np.any(sk):
        return out

    deg = skeleton_degree_8n(sk).astype(np.int32)
    node_mask = sk & ((deg == 1) | (deg >= 3))
    comp_ids, n_comp = ndi_label(sk.astype(np.uint8), structure=np.ones((3, 3), dtype=np.uint8))
    neigh8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for cid in range(1, int(n_comp) + 1):
        comp = (comp_ids == cid)
        if np.count_nonzero(comp) <= 1:
            out |= comp
            continue

        ys, xs = np.nonzero(comp & node_mask)
        node_coords = [(int(y), int(x)) for y, x in zip(ys, xs)]
        if len(node_coords) < 2:
            # loop-like component with no branch/end nodes: keep as-is
            out |= comp
            continue

        node_id = {c: i for i, c in enumerate(node_coords)}
        adj: Dict[int, List[Tuple[int, float, int]]] = {i: [] for i in range(len(node_coords))}
        edge_pixels: Dict[int, List[Tuple[int, int]]] = {}
        edge_pair_to_id: Dict[Tuple[int, int], int] = {}
        visited_dir = set()
        next_edge_id = 0

        for c0 in node_coords:
            u = node_id[c0]
            y0, x0 = c0
            for dy, dx in neigh8:
                y1, x1 = y0 + dy, x0 + dx
                if not (0 <= y1 < sk.shape[0] and 0 <= x1 < sk.shape[1]):
                    continue
                if not comp[y1, x1]:
                    continue
                key_dir = (y0, x0, y1, x1)
                if key_dir in visited_dir:
                    continue
                visited_dir.add(key_dir)

                prev = (y0, x0)
                cur = (y1, x1)
                chain = [c0, cur]
                w = _step_len_eq6(cur[0] - prev[0], cur[1] - prev[1])

                while True:
                    if cur in node_id and cur != c0:
                        v = node_id[cur]
                        a, b = (u, v) if u <= v else (v, u)
                        if (a, b) not in edge_pair_to_id:
                            eid = next_edge_id
                            next_edge_id += 1
                            edge_pair_to_id[(a, b)] = eid
                            edge_pixels[eid] = list(chain)
                            adj[u].append((v, w, eid))
                            adj[v].append((u, w, eid))
                        break

                    nbs = []
                    cy, cx = cur
                    for ddy, ddx in neigh8:
                        ny, nx = cy + ddy, cx + ddx
                        if not (0 <= ny < sk.shape[0] and 0 <= nx < sk.shape[1]):
                            continue
                        if not comp[ny, nx]:
                            continue
                        if (ny, nx) == prev:
                            continue
                        nbs.append((ny, nx))

                    if len(nbs) == 0:
                        break
                    if len(nbs) > 1:
                        # safety on malformed component
                        break
                    nxt = nbs[0]
                    visited_dir.add((cur[0], cur[1], nxt[0], nxt[1]))
                    w += _step_len_eq6(nxt[0] - cur[0], nxt[1] - cur[1])
                    chain.append(nxt)
                    prev = cur
                    cur = nxt

        if len(edge_pixels) == 0:
            out |= comp
            continue

        start = 0
        u, _, _ = _dijkstra_farthest(adj, start)
        v, _, prev = _dijkstra_farthest(adj, u)

        path_nodes = [v]
        cur = v
        while cur != u:
            if cur not in prev:
                break
            cur = prev[cur]
            path_nodes.append(cur)
        path_nodes = path_nodes[::-1]

        for i in range(len(path_nodes) - 1):
            a = int(path_nodes[i])
            b = int(path_nodes[i + 1])
            key = (a, b) if a <= b else (b, a)
            eid = edge_pair_to_id.get(key, None)
            if eid is None:
                continue
            for yy, xx in edge_pixels[eid]:
                out[int(yy), int(xx)] = True

    return out


def graph_longest_path_prune(skel: np.ndarray) -> np.ndarray:
    """
    Strict fast baseline:
      keep only a single longest (hop-count) trunk per connected component.
    """
    sk = np.asarray(skel).astype(bool)
    if not np.any(sk):
        return sk.copy()

    comp_ids, n_comp = ndi_label(sk.astype(np.uint8), structure=np.ones((3, 3), dtype=np.uint8))
    out = np.zeros_like(sk, dtype=bool)
    neigh8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for cid in range(1, int(n_comp) + 1):
        ys, xs = np.nonzero(comp_ids == cid)
        n = len(xs)
        if n == 0:
            continue
        if n == 1:
            out[ys[0], xs[0]] = True
            continue

        nodes = np.column_stack([ys, xs]).astype(np.int32)
        index = {(int(y), int(x)): i for i, (y, x) in enumerate(nodes)}
        adj: List[List[int]] = [[] for _ in range(n)]
        for i, (y, x) in enumerate(nodes):
            yi, xi = int(y), int(x)
            for dy, dx in neigh8:
                nb = (yi + dy, xi + dx)
                j = index.get(nb, -1)
                if j >= 0:
                    adj[i].append(j)

        def _farthest(start: int) -> Tuple[int, np.ndarray]:
            parent = np.full((n,), -1, dtype=np.int32)
            dist = np.full((n,), -1, dtype=np.int32)
            q = deque([int(start)])
            dist[int(start)] = 0
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] >= 0:
                        continue
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
            far = int(np.argmax(dist))
            return far, parent

        u, _ = _farthest(0)
        v, parent = _farthest(u)
        cur = v
        while cur >= 0:
            yy, xx = nodes[cur]
            out[int(yy), int(xx)] = True
            if cur == u:
                break
            cur = int(parent[cur])

    return out


def skeleton_degree_8n(skel: np.ndarray) -> np.ndarray:
    """
    8-neighborhood degree for each skeleton pixel.
    """
    kernel = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=np.uint8,
    )
    return convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)


def remove_intersections(skel: np.ndarray, dilate_iter: int = 1) -> np.ndarray:
    """
    Paper-style simplification:
      - detect junctions (degree > 2)
      - optional light dilation
      - remove them to split into linear segments
    """
    deg = skeleton_degree_8n(skel)
    junctions = (deg > 2) & skel.astype(bool)
    if int(dilate_iter) > 0:
        junctions = binary_dilation(junctions, iterations=int(dilate_iter))
    cut_skel = skel.astype(bool).copy()
    cut_skel[junctions] = False
    return cut_skel


def prune_skeleton_paper_style(mat_skel: np.ndarray, dilate_iter: int = 1) -> np.ndarray:
    """
    MAT -> remove intersections -> per-segment longest-path prune.
    """
    cut = remove_intersections(mat_skel, dilate_iter=int(dilate_iter))
    comp, n = ndi_label(cut.astype(np.uint8), structure=np.ones((3, 3), dtype=np.uint8))
    out = np.zeros_like(mat_skel, dtype=bool)
    for cid in range(1, int(n) + 1):
        seg = (comp == cid)
        if np.count_nonzero(seg) < 2:
            out |= seg
            continue
        trunk = graph_longest_path_prune(seg)
        out |= trunk
    return out


def _boundary_mask_4n(bw: np.ndarray) -> np.ndarray:
    P = bw.astype(bool)
    Pp = np.pad(P, 1, mode="constant", constant_values=False)
    up = Pp[:-2, 1:-1]
    down = Pp[2:, 1:-1]
    left = Pp[1:-1, :-2]
    right = Pp[1:-1, 2:]
    return P & (~up | ~down | ~left | ~right)


def _closest_opposite_side_pair(
    edges: np.ndarray,
    y: int,
    x: int,
    normal: np.ndarray,
    *,
    topk: int = 6,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    rel = edges.astype(np.float32) - np.array([float(y), float(x)], dtype=np.float32)
    proj = rel @ normal.astype(np.float32)

    pos_idx = np.where(proj > 0)[0]
    neg_idx = np.where(proj < 0)[0]
    if pos_idx.size == 0 or neg_idx.size == 0:
        return None, None, 0.0

    k_pos = int(min(topk, pos_idx.size))
    k_neg = int(min(topk, neg_idx.size))
    pos_sel = pos_idx[np.argpartition(proj[pos_idx], -k_pos)[-k_pos:]]
    neg_sel = neg_idx[np.argpartition(proj[neg_idx], k_neg - 1)[:k_neg]]

    pos_pts = edges[pos_sel].astype(np.float32)
    neg_pts = edges[neg_sel].astype(np.float32)
    d2 = np.sum((pos_pts[:, None, :] - neg_pts[None, :, :]) ** 2, axis=2)
    iy, ix = np.unravel_index(int(np.argmin(d2)), d2.shape)
    p1 = pos_pts[int(iy)]
    p2 = neg_pts[int(ix)]
    return p1, p2, float(np.sqrt(max(float(d2[iy, ix]), 0.0)))


def _min_pairwise_dist(A, B):
    # A: (Na, 2), B: (Nb, 2)
    # returns scalar min ||Ai - Bj||
    diff = A[:, None, :] - B[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    return float(np.sqrt(d2.min()))

def width_esd_local(
    bw: np.ndarray,
    *,
    skel: np.ndarray,
    dist_map: np.ndarray,
    patch_scale: float = 1.5,
    min_radius: int = 5,
    max_radius: int = 100,
    min_points: int = 4,
    proj_thresh: float = 0.9,
) -> np.ndarray:

    wmap = np.zeros_like(bw, dtype=np.float32)
    ys, xs = np.nonzero(skel)

    for y, x in zip(ys, xs):

        r = int(np.clip(float(dist_map[y, x]) * float(patch_scale), float(min_radius), float(max_radius)))
        y0, y1 = max(0, y - r), min(bw.shape[0], y + r + 1)
        x0, x1 = max(0, x - r), min(bw.shape[1], x + r + 1)

        patch = bw[y0:y1, x0:x1]
        edges = _edge_points_in_patch(patch)

        if edges.shape[0] < int(min_points):
            continue

        edges = edges + np.array([y0, x0], dtype=np.float32)

        center = np.array([y, x], dtype=np.float32)

        # --- vectorized ---
        rel = edges - center
        norms = np.linalg.norm(rel, axis=1, keepdims=True)
        dirs = rel / (norms + 1e-8)

        tangent, _ = _local_tangent_normal(int(y), int(x), skel, window=5)
        O = tangent / (np.linalg.norm(tangent) + 1e-8)

        proj = dirs @ O

        # --- boolean masks ---
        pos_mask = proj > proj_thresh
        neg_mask = proj < -proj_thresh

        if not np.any(pos_mask) or not np.any(neg_mask):
            continue

        pos = edges[pos_mask]
        neg = edges[neg_mask]

        # --- vectorized distance ---
        wmap[y, x] = _min_pairwise_dist(pos, neg)

    return wmap


def width_eob_hybrid_cpu(
    bw: np.ndarray,
    *,
    skel: np.ndarray,
    dist_map: np.ndarray,
    patch_scale: float = 1.5,
    min_radius: int = 5,
    max_radius: int = 100,
    min_points: int = 4,
    proj_thresh: float = 0.9,
) -> np.ndarray:

    wmap = np.zeros_like(bw, dtype=np.float32)

    def _angle(v):
        return np.arctan2(v[0], v[1])

    def _avg_angle(a, b, c):
        s = np.sin(a) + np.sin(b) + np.sin(c)
        c_ = np.cos(a) + np.cos(b) + np.cos(c)
        return np.arctan2(s, c_)

    ys, xs = np.nonzero(skel)

    for y, x in zip(ys, xs):

        r = int(np.clip(float(dist_map[y, x]) * float(patch_scale), float(min_radius), float(max_radius)))

        y0, y1 = max(0, y - r), min(bw.shape[0], y + r + 1)
        x0, x1 = max(0, x - r), min(bw.shape[1], x + r + 1)

        patch = bw[y0:y1, x0:x1]
        edges = _edge_points_in_patch(patch)

        if edges.shape[0] < int(min_points):
            continue

        edges = edges + np.array([y0, x0], dtype=np.float32)
        center = np.array([y, x], dtype=np.float32)

        rel = edges - center
        norms = np.linalg.norm(rel, axis=1, keepdims=True)
        dirs = rel / (norms + 1e-8)

        tangent, _ = _local_tangent_normal(int(y), int(x), skel, window=5)
        O = tangent / (np.linalg.norm(tangent) + 1e-8)

        proj = dirs @ O

        pos_mask = proj > proj_thresh
        neg_mask = proj < -proj_thresh

        if not np.any(pos_mask) or not np.any(neg_mask):
            continue

        pos = edges[pos_mask]
        neg = edges[neg_mask]

        # --- first pass ---
        w1 = _min_pairwise_dist(pos, neg)

        # get closest pair indices (vectorized)
        diff = pos[:, None, :] - neg[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        idx = np.unravel_index(np.argmin(d2), d2.shape)

        p1 = pos[idx[0]]
        p2 = neg[idx[1]]

        # --- PCA tangents ---
        def _pca_tangent(pt):
            yy, xx = int(pt[0]), int(pt[1])
            y0b, y1b = max(0, yy - 2), min(bw.shape[0], yy + 3)
            x0b, x1b = max(0, xx - 2), min(bw.shape[1], xx + 3)

            sub = bw[y0b:y1b, x0b:x1b]
            pts = _edge_points_in_patch(sub)

            if pts.shape[0] < 3:
                return tangent

            pts = pts + np.array([y0b, x0b])
            X = pts - pts.mean(axis=0)
            cov = X.T @ X / len(pts)

            eigvals, eigvecs = np.linalg.eig(cov)
            t = eigvecs[:, np.argmax(eigvals)]
            return t / (np.linalg.norm(t) + 1e-8)

        t1 = _pca_tangent(p1)
        t2 = _pca_tangent(p2)

        th_corr = _avg_angle(_angle(tangent), _angle(t1), _angle(t2))

        O_corr = np.array([np.sin(th_corr), np.cos(th_corr)], dtype=np.float32)
        O_corr /= (np.linalg.norm(O_corr) + 1e-8)

        # --- second pass ---
        proj2 = dirs @ O_corr

        pos_mask2 = proj2 > proj_thresh
        neg_mask2 = proj2 < -proj_thresh

        if not np.any(pos_mask2) or not np.any(neg_mask2):
            wmap[y, x] = w1
            continue

        pos2 = edges[pos_mask2]
        neg2 = edges[neg_mask2]

        w2 = _min_pairwise_dist(pos2, neg2)

        wmap[y, x] = w2 if w2 > 0 else w1

    return wmap


def plot_geometry_comparison(
    out_path: str,
    bw: np.ndarray,
    geoms: List[Tuple[str, np.ndarray]],
    *,
    min_area_px: Optional[int] = None,
):
    """
    geoms: list of (label, binary_mask)
    """
    _safe_mkdir(os.path.dirname(out_path))

    n = len(geoms)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6), dpi=200)
    if n == 1:
        axes = [axes]

    geom_counts = [int(np.count_nonzero(mask)) for _, mask in geoms]
    max_count = max(geom_counts) if geom_counts else 1
    if max_count <= 0:
        max_count = 1

    for ax, (label, mask), count in zip(axes, geoms, geom_counts):
        ax.imshow(bw, cmap="gray", alpha=0.35)
        ys, xs = np.nonzero(mask)
        ax.scatter(xs, ys, s=3, c="red")
        ax.set_title(label)
        # Mini relative-size tick: panel skeleton px vs figure max skeleton px.
        x0, x1, yb = 0.04, 0.34, 0.96
        ax.plot([x0, x1], [yb, yb], transform=ax.transAxes, color="white", lw=1.0, alpha=0.9)
        r = float(count) / float(max_count)
        xt = x0 + (x1 - x0) * max(0.0, min(1.0, r))
        ax.plot([xt, xt], [yb - 0.02, yb + 0.02], transform=ax.transAxes, color="yellow", lw=1.2, alpha=0.95)
        ax.text(
            x1 + 0.01,
            yb,
            f"{count}/{max_count}",
            transform=ax.transAxes,
            va="center",
            ha="left",
            fontsize=7,
            color="white",
        )
        ax.axis("off")

    fig.suptitle("Skeleton Geometry Comparison", y=1.02)
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

    #img_out_root = os.path.join(out_dir, stem)
    img_out_root = os.path.join(out_dir, stem.replace("_modified", ""))
    _safe_mkdir(img_out_root)

    timings = {}
    panels = []

    # --------------------------------------------------
    # Stage A: shared precompute
    # --------------------------------------------------
    pre = compute_shared_precompute(bw, gpu_lock_path=gpu_lock_path)
    timings.update(pre["timings"])
    mat_skel = pre["mat_skel"].astype(bool)
    dist_map = pre["dist_map"].astype(np.float32)

    # --------------------------------------------------
    # Stage B: skeleton variants
    # --------------------------------------------------
    skel_raw = mat_skel

    t0 = time.perf_counter()
    if DSE_OK:
        try:
            skel_dse = skel_pruning_DSE(
                skel_raw,
                dist_map,
                min_area_px=int(min_area_px),
                return_graph=False,
            ).astype(bool)
            timings["skeleton_dse_s"] = float(time.perf_counter() - t0)
        except Exception as e:
            skel_dse = skel_raw.copy()
            timings["skeleton_dse_s"] = 0.0
            timings["skeleton_dse_error"] = repr(e)
    else:
        skel_dse = skel_raw.copy()
        timings["skeleton_dse_s"] = 0.0

    t0 = time.perf_counter()
    skel_graph = graph_longest_path_prune(skel_raw).astype(bool)
    timings["skeleton_graph_s"] = float(time.perf_counter() - t0)

    geometry_baselines: Dict[str, Dict[str, np.ndarray]] = {}
    width_baselines: Dict[str, Dict[str, np.ndarray]] = {}

    def _add_geometry(name: str, skel_mask: np.ndarray) -> None:
        geometry_baselines[name] = {
            "skeleton": np.asarray(skel_mask, dtype=bool),
            "support_mask": np.asarray(skel_mask, dtype=np.uint8),
        }

    def _add_width(name: str, width_map: np.ndarray, skel_mask: np.ndarray) -> None:
        width_baselines[name] = {
            "width_map": np.asarray(width_map, dtype=np.float32),
            "support_mask": np.asarray(skel_mask, dtype=np.uint8),
            "skeleton": np.asarray(skel_mask, dtype=bool),
        }

    # Skeleton-only baselines
    _add_geometry("skel_mat_raw", skel_raw)
    _add_geometry("skel_mat_dse", skel_dse)
    _add_geometry("skel_graph_longest_path", skel_graph)

    # --------------------------------------------------
    # Stage C: width baselines (adaptive, hardcoded)
    # --------------------------------------------------
    PATCH_SCALE = 1.5
    MIN_RADIUS = 4
    MAX_RADIUS = 128
    PROJ_THRESH = 0.9
    MIN_POINTS = int(cfg["pca_min_points"])

    # 1) MAT width (raw MAT skeleton, DT*2)
    w_mat = np.zeros_like(dist_map, dtype=np.float32)
    w_mat[skel_raw] = dist_map[skel_raw] * 2.0
    _add_width("mat_width_raw", w_mat, skel_raw)

    # 2) MAT width on DSE skeleton (DT*2)
    w_mat_dse = np.zeros_like(dist_map, dtype=np.float32)
    w_mat_dse[skel_dse] = dist_map[skel_dse] * 2.0
    _add_width("mat_width_dse", w_mat_dse, skel_dse)

    # 3) PCA width (DSE skeleton)
    t0 = time.perf_counter()
    w_pca_dse = width_pca_proj(
        bw,
        skel=skel_dse,
        dist_map=dist_map,
        patch_scale=PATCH_SCALE,
        min_radius=MIN_RADIUS,
        max_radius=MAX_RADIUS,
        min_points=MIN_POINTS,
    )
    timings["pca_width_dse_s"] = float(time.perf_counter() - t0)
    _add_width("pca_width_dse", w_pca_dse, skel_dse)

    # 4) ESD width (DSE skeleton)
    t0 = time.perf_counter()
    w_esd_dse = width_esd_local(
        bw,
        skel=skel_dse,
        dist_map=dist_map,
        patch_scale=PATCH_SCALE,
        min_radius=MIN_RADIUS,
        max_radius=MAX_RADIUS,
        min_points=MIN_POINTS,
        proj_thresh=PROJ_THRESH,
    )
    timings["esd_width_dse_s"] = float(time.perf_counter() - t0)
    _add_width("esd_width_dse", w_esd_dse, skel_dse)

    # 5) EOB width (DSE skeleton)
    t0 = time.perf_counter()
    w_eob_dse = width_eob_hybrid_cpu(
        bw,
        skel=skel_dse,
        dist_map=dist_map,
        patch_scale=PATCH_SCALE,
        min_radius=MIN_RADIUS,
        max_radius=MAX_RADIUS,
        min_points=MIN_POINTS,
        proj_thresh=PROJ_THRESH,
    )
    timings["eob_width_dse_s"] = float(time.perf_counter() - t0)
    _add_width("eob_width_dse", w_eob_dse, skel_dse)

    # Optional stress baseline intentionally disabled:
    # strict graph skeleton is retained for geometry-only comparison.
    # t0 = time.perf_counter()
    # w_eob_graph = width_eob_hybrid_cpu(
    #     bw,
    #     skel=skel_graph,
    #     dist_map=dist_map,
    #     patch_scale=PATCH_SCALE,
    #     min_radius=MIN_RADIUS,
    #     max_radius=MAX_RADIUS,
    #     min_points=MIN_POINTS,
    #     proj_thresh=PROJ_THRESH,
    # )
    # timings["eob_width_graph_strict_s"] = float(time.perf_counter() - t0)
    # _add_baseline("eob_width_graph_strict", w_eob_graph, skel_graph)

    method_label = {
        "skel_mat_raw": "MAT Raw",
        "skel_mat_dse": f"MAT + DSE ({int(min_area_px)}px)",
        "skel_graph_longest_path": "MAT + Graph Longest-Path",
        "mat_width_raw": "MAT Width (Raw MAT)",
        "mat_width_dse": f"MAT Width (DSE, {int(min_area_px)}px)",
        "pca_width_dse": f"PCA Width (DSE, {int(min_area_px)}px)",
        "esd_width_dse": f"ESD Width (DSE, {int(min_area_px)}px)",
        "eob_width_dse": f"EOB Width (DSE, {int(min_area_px)}px)",
    }

    saved_geometry_methods: List[str] = []
    saved_width_methods: List[str] = []

    for method in methods:
        if method in width_baselines:
            rec = width_baselines[method]
            outp = os.path.join(img_out_root, method, "width_map.npz")
            meta = {
                "method": method,
                "ok": True,
                "H": H,
                "W": W,
                "crack_px": crack_px,
                "min_area_px": int(min_area_px),
                "baseline_type": "width",
            }
            _save_npz(
                outp,
                width_map=rec["width_map"],
                support_mask=rec["support_mask"],
                skel=rec["skeleton"],
                meta=meta,
            )
            saved_width_methods.append(str(method))
            if make_plots and method in ("mat_width_raw", "mat_width_dse", "pca_width_dse", "esd_width_dse", "eob_width_dse"):
                panels.append((method_label.get(method, method), rec["width_map"], rec["support_mask"], method))

        elif method in geometry_baselines:
            rec = geometry_baselines[method]
            outp = os.path.join(img_out_root, method, "geometry.npz")
            meta = {
                "method": method,
                "ok": True,
                "H": H,
                "W": W,
                "crack_px": crack_px,
                "min_area_px": int(min_area_px),
                "geometry_only": True,
                "baseline_type": "geometry",
            }
            # Keep payload schema stable: geometry files still carry width_map (all zeros).
            _save_npz(
                outp,
                width_map=np.zeros((H, W), dtype=np.float32),
                support_mask=rec["support_mask"],
                skel=rec["skeleton"],
                meta=meta,
            )
            saved_geometry_methods.append(str(method))

    # --------------------------------------------------
    # Geometry-only and width overview plots
    # --------------------------------------------------
    if make_plots:
        geoms = []
        for method in ("skel_mat_raw", "skel_mat_dse", "skel_graph_longest_path"):
            if method in geometry_baselines:
                geoms.append((method_label[method], geometry_baselines[method]["skeleton"]))
        if geoms:
            plot_geometry_comparison(
                os.path.join(img_out_root, "geometry_comparison.png"),
                bw=bw,
                geoms=geoms,
                min_area_px=min_area_px,
            )

    if make_plots and panels:
        plot_path_grouped = os.path.join(img_out_root, "baselines_overview.png")
        _plot_overview(plot_path_grouped, bw=bw, panels=panels, min_area_px=min_area_px, mode="grouped")

        plot_path_global = os.path.join(img_out_root, "baselines_overview_global_scale.png")
        _plot_overview(plot_path_global, bw=bw, panels=panels, min_area_px=min_area_px, mode="global_zero_to_max")

    t1_total = time.perf_counter()

    return {
        "path": path,
        "stem": stem,
        "H": H,
        "W": W,
        "crack_px": crack_px,
        "n_geometry_methods_saved": int(len(saved_geometry_methods)),
        "n_width_methods_saved": int(len(saved_width_methods)),
        "geometry_methods_saved": json.dumps(saved_geometry_methods),
        "width_methods_saved": json.dumps(saved_width_methods),
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


def _collect_image_paths(in_dir: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    numbered_best: Dict[int, str] = {}
    other_paths: List[str] = []
    for root, _, files in os.walk(in_dir):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in exts:
                full = os.path.join(root, fn)
                stem = os.path.splitext(fn)[0]
                m = re.match(r"^\s*(\d+)", stem)
                if m is None:
                    other_paths.append(full)
                    continue

                # Group by leading number across the whole input tree.
                # Prefer *_modified variant when both exist.
                key = int(m.group(1))
                is_modified = "_modified" in stem.lower()
                prev = numbered_best.get(key, None)
                if prev is None:
                    numbered_best[key] = full
                    continue

                prev_stem = os.path.splitext(os.path.basename(prev))[0]
                prev_is_modified = "_modified" in prev_stem.lower()
                if is_modified and not prev_is_modified:
                    numbered_best[key] = full
                elif is_modified == prev_is_modified:
                    # Stable deterministic tie-breaker on full path.
                    if full.lower() < prev.lower():
                        numbered_best[key] = full

    paths = list(numbered_best.values()) + other_paths
    return sorted(paths, key=_natural_sort_key)


def replot_from_saved_npz(
    *,
    in_dir: str,
    out_dir: str,
    threshold: float,
    min_area_px: int,
    methods: List[str],
) -> int:
    """
    Mini alternate flow:
      - load per-image saved width_map.npz files
      - regenerate geometry + overview plots without recomputing baselines
    """
    paths = _collect_image_paths(in_dir)
    if not paths:
        print(f"[width_baseline_creator] No images found under: {in_dir}")
        return 2

    image_by_stem = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in paths
    }
    load_methods = list(methods)

    method_label = {
        "skel_mat_raw": "MAT Raw",
        "skel_mat_dse": f"MAT + DSE ({int(min_area_px)}px)",
        "skel_graph_longest_path": "MAT + Graph Longest-Path",
        "mat_width_raw": "MAT Width (Raw MAT)",
        "mat_width_dse": f"MAT Width (DSE, {int(min_area_px)}px)",
        "pca_width_dse": f"PCA Width (DSE, {int(min_area_px)}px)",
        "esd_width_dse": f"ESD Width (DSE, {int(min_area_px)}px)",
        "eob_width_dse": f"EOB Width (DSE, {int(min_area_px)}px)",
    }

    stems = sorted(
        d for d in os.listdir(out_dir)
        if os.path.isdir(os.path.join(out_dir, d))
    )
    if not stems:
        print(f"[width_baseline_creator] No baseline folders found in: {out_dir}")
        return 2

    tasks = []
    for stem in stems:
        img_path = image_by_stem.get(stem, None)
        if img_path is None:
            continue
        tasks.append((stem, img_path))

    if not tasks:
        print("[replot] no overlapping image stems between input masks and baseline folders")
        return 2

    def _replot_one(stem_img: Tuple[str, str]) -> Tuple[str, bool, str]:
        stem, img_path = stem_img
        try:
            bw0 = _read_mask(img_path, threshold=threshold)
            bw = remove_small_objects(bw0.astype(bool), min_size=min_area_px).astype(bool)
            img_out_root = os.path.join(out_dir, stem)

            panels: List[Tuple[str, np.ndarray, np.ndarray, str]] = []
            geoms: List[Tuple[str, np.ndarray]] = []
            skeletonize_geom = skeletonize(bw).astype(bool)

            for method in load_methods:
                npz_path_width = os.path.join(img_out_root, method, "width_map.npz")
                npz_path_geom = os.path.join(img_out_root, method, "geometry.npz")
                npz_path = npz_path_width if os.path.isfile(npz_path_width) else npz_path_geom
                if not os.path.isfile(npz_path):
                    continue
                try:
                    data = np.load(npz_path, allow_pickle=True)
                    wmap = np.asarray(data["width_map"], dtype=np.float32)
                    supp = np.asarray(data["support_mask"], dtype=np.uint8)
                    skel = np.asarray(data["skel"], dtype=np.uint8).astype(bool)
                except Exception as e:
                    print(f"[replot] failed to load {npz_path}: {e}")
                    continue

                if npz_path == npz_path_width:
                    panels.append((method_label.get(method, method), wmap, supp, method))

                if method == "skel_mat_raw":
                    geoms.append(("MAT Raw", skel))
                elif method == "skel_mat_dse":
                    geoms.append(("MAT + DSE", skel))
                elif method == "skel_graph_longest_path":
                    geoms.append(("MAT + Graph Longest-Path", skel))

            if not geoms:
                geoms.append(("Skeletonize", skeletonize_geom))

            if geoms:
                plot_geometry_comparison(
                    os.path.join(img_out_root, "geometry_comparison.png"),
                    bw=bw,
                    geoms=geoms,
                    min_area_px=min_area_px,
                )
            if panels:
                width_panels = [
                    p for p in panels
                    if p[3] in ("mat_width_raw", "mat_width_dse", "pca_width_dse", "esd_width_dse", "eob_width_dse")
                ]
                _plot_overview(
                    os.path.join(img_out_root, "baselines_overview.png"),
                    bw=bw,
                    panels=width_panels if width_panels else panels,
                    min_area_px=min_area_px,
                    mode="grouped",
                )
                _plot_overview(
                    os.path.join(img_out_root, "baselines_overview_global_scale.png"),
                    bw=bw,
                    panels=width_panels if width_panels else panels,
                    min_area_px=min_area_px,
                    mode="global_zero_to_max",
                )
            return stem, True, "ok"
        except Exception as e:
            return stem, False, repr(e)

    max_workers = max(1, min(os.cpu_count() or 4, len(tasks)))
    print(f"[replot] parallel workers={max_workers} tasks={len(tasks)}")
    failures = 0

    with ThreadPoolExecutor(max_workers=13) as ex:
        futs = [ex.submit(_replot_one, t) for t in tasks]
        for fut in as_completed(futs):
            stem, ok, msg = fut.result()
            if ok:
                print(f"[replot] refreshed plots for {stem}")
            else:
                failures += 1
                print(f"[replot] failed for {stem}: {msg}")

    if failures > 0:
        print(f"[replot] completed with failures: {failures}/{len(tasks)}")
    else:
        print(f"[replot] completed successfully: {len(tasks)}/{len(tasks)}")

    return 0


def _extract_width_values(npz_path: str) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    wmap = np.asarray(data["width_map"], dtype=np.float32)
    supp = np.asarray(data["support_mask"], dtype=np.uint8).astype(bool)
    vals = wmap[supp]
    vals = vals[np.isfinite(vals)]
    vals = vals[vals >= 0]
    return vals.astype(np.float32, copy=False)


def _distribution_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    stem = str(task["stem"])
    stem_dir = str(task["stem_dir"])
    methods = list(task["methods"])
    bins = np.asarray(task["bins"], dtype=np.float32)

    method_values: Dict[str, np.ndarray] = {}
    method_rows: List[Dict[str, Any]] = []
    hist_rows: List[Dict[str, Any]] = []

    for method in methods:
        npz_path = os.path.join(stem_dir, method, "width_map.npz")
        if not os.path.isfile(npz_path):
            continue
        try:
            vals = _extract_width_values(npz_path)
        except Exception:
            continue
        if vals.size == 0:
            continue

        method_values[method] = vals
        q = np.quantile(vals, [0.1, 0.25, 0.5, 0.75, 0.9])
        q25 = float(q[1])
        q50 = float(q[2])
        q75 = float(q[3])
        iqr = float(q75 - q25)
        lower_fence = float(q25 - 1.5 * iqr)
        upper_fence = float(q75 + 1.5 * iqr)
        outlier_mask = (vals < lower_fence) | (vals > upper_fence)
        outlier_count = int(np.count_nonzero(outlier_mask))
        outlier_frac = float(outlier_count / max(int(vals.size), 1))
        method_rows.append(
            {
                "stem": stem,
                "method": method,
                "count": int(vals.size),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "q10": float(q[0]),
                "q25": q25,
                "q50": q50,
                "q75": q75,
                "q90": float(q[4]),
                "iqr": iqr,
                "lower_fence": lower_fence,
                "upper_fence": upper_fence,
                "outlier_count": outlier_count,
                "outlier_frac": outlier_frac,
            }
        )

        counts, _ = np.histogram(vals, bins=bins)
        hist_rows.append({"stem": stem, "method": method, "counts": counts.astype(np.int64)})

    pair_rows: List[Dict[str, Any]] = []
    present = sorted(method_values.keys())
    for m1, m2 in combinations(present, 2):
        v1 = method_values[m1]
        v2 = method_values[m2]
        if v1.size == 0 or v2.size == 0:
            continue
        pair_rows.append(
            {
                "stem": stem,
                "method_a": m1,
                "method_b": m2,
                "wasserstein": float(wasserstein_distance(v1, v2)),
                "ks_stat": float(ks_2samp(v1, v2, mode="auto").statistic),
                "median_abs_diff": float(abs(np.median(v1) - np.median(v2))),
            }
        )

    return {
        "stem": stem,
        "method_rows": method_rows,
        "pair_rows": pair_rows,
        "hist_rows": hist_rows,
    }


def _hist_worker(task: Dict[str, Any]) -> Tuple[str, np.ndarray]:
    method = str(task["method"])
    npz_path = str(task["npz_path"])
    bins = np.asarray(task["bins"], dtype=np.float32)
    try:
        vals = _extract_width_values(npz_path)
    except Exception:
        return method, np.zeros(len(bins) - 1, dtype=np.int64)
    if vals.size == 0:
        return method, np.zeros(len(bins) - 1, dtype=np.int64)
    counts, _ = np.histogram(vals, bins=bins)
    return method, counts.astype(np.int64)


def _plot_matrix(
    out_path: str,
    mat: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    cbar_label: str,
) -> None:
    if mat.size == 0:
        return
    _safe_mkdir(os.path.dirname(out_path))
    fig_w = max(8, 0.35 * max(1, len(col_labels)))
    fig_h = max(6, 0.25 * max(1, len(row_labels)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(cbar_label)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def _plot_per_image_quantile_iqr_bars(df_method: pd.DataFrame, out_path: str) -> None:
    if df_method.empty:
        return
    _safe_mkdir(os.path.dirname(out_path))
    methods = sorted(df_method["method"].dropna().unique().tolist())
    if not methods:
        return

    stems = sorted(df_method["stem"].dropna().unique().tolist())
    n_methods = len(methods)
    fig, axes = plt.subplots(n_methods, 1, figsize=(max(12, 0.28 * len(stems)), 3.2 * n_methods), dpi=200, sharex=True)
    if n_methods == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        sub = df_method[df_method["method"] == method].set_index("stem").reindex(stems)
        x = np.arange(len(stems))
        q25 = sub["q25"].to_numpy(dtype=float)
        q50 = sub["q50"].to_numpy(dtype=float)
        q75 = sub["q75"].to_numpy(dtype=float)
        lo = np.clip(q50 - q25, 0.0, None)
        hi = np.clip(q75 - q50, 0.0, None)
        err = np.vstack([lo, hi])
        ax.bar(x, q50, alpha=0.70, color="#4c78a8", width=0.80, label="q50")
        ax.errorbar(x, q50, yerr=err, fmt="none", ecolor="#f58518", elinewidth=1.2, capsize=2, label="IQR (q25-q75)")
        ax.set_ylabel("Width (px)")
        ax.set_title(f"{method}: per-image median with IQR")
        ax.grid(axis="y", alpha=0.20)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xticks(np.arange(len(stems)))
    axes[-1].set_xticklabels(stems, rotation=45, ha="right", fontsize=7)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def _plot_method_quantile_outlier_bars(df_method: pd.DataFrame, out_path: str) -> None:
    if df_method.empty:
        return
    _safe_mkdir(os.path.dirname(out_path))
    if "iqr" not in df_method.columns:
        if "q25" in df_method.columns and "q75" in df_method.columns:
            df_method = df_method.copy()
            df_method["iqr"] = pd.to_numeric(df_method["q75"], errors="coerce") - pd.to_numeric(df_method["q25"], errors="coerce")
        else:
            df_method = df_method.copy()
            df_method["iqr"] = np.nan
    if "outlier_frac" not in df_method.columns:
        df_method = df_method.copy()
        df_method["outlier_frac"] = 0.0

    g = (
        df_method.groupby("method", as_index=False)[["q25", "q50", "q75", "iqr", "outlier_frac"]]
        .mean()
        .sort_values("method")
    )
    if g.empty:
        return

    methods = g["method"].tolist()
    x = np.arange(len(methods))
    w = 0.24

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(9, 1.8 * len(methods)), 7.5), dpi=200, sharex=True)
    ax1.bar(x - w, g["q25"].to_numpy(float), width=w, alpha=0.75, label="q25")
    ax1.bar(x, g["q50"].to_numpy(float), width=w, alpha=0.75, label="q50")
    ax1.bar(x + w, g["q75"].to_numpy(float), width=w, alpha=0.75, label="q75")
    ax1.set_ylabel("Width (px)")
    ax1.set_title("Dataset-Level Quantiles by Method (mean across images)")
    ax1.grid(axis="y", alpha=0.20)
    ax1.legend()

    ax2.bar(x - 0.18, g["iqr"].to_numpy(float), width=0.36, alpha=0.75, label="IQR")
    ax2.bar(x + 0.18, g["outlier_frac"].to_numpy(float), width=0.36, alpha=0.75, label="Outlier fraction")
    ax2.set_ylabel("Value")
    ax2.set_title("Dataset-Level IQR and Outlier Fraction")
    ax2.grid(axis="y", alpha=0.20)
    ax2.legend()
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=30, ha="right")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def _plot_per_baseline_metric_boxplots(df_method: pd.DataFrame, out_dir: str) -> None:
    if df_method.empty:
        return
    _safe_mkdir(out_dir)

    metric_cols = [
        "min",
        "q25",
        "q50",
        "q75",
        "max",
        "iqr",
        "outlier_frac",
    ]
    for c in metric_cols:
        if c not in df_method.columns:
            if c == "iqr" and "q25" in df_method.columns and "q75" in df_method.columns:
                df_method = df_method.copy()
                df_method["iqr"] = pd.to_numeric(df_method["q75"], errors="coerce") - pd.to_numeric(df_method["q25"], errors="coerce")
            elif c == "outlier_frac":
                df_method = df_method.copy()
                df_method["outlier_frac"] = 0.0
            else:
                df_method = df_method.copy()
                df_method[c] = np.nan

    methods = sorted(df_method["method"].dropna().unique().tolist())
    for method in methods:
        sub = df_method[df_method["method"] == method].copy()
        if sub.empty:
            continue

        series_list = []
        labels = []
        for c in metric_cols:
            vals = pd.to_numeric(sub[c], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            series_list.append(vals)
            labels.append(c)

        if not series_list:
            continue

        fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
        ax.boxplot(series_list, tick_labels=labels, showfliers=True, patch_artist=False)
        ax.set_title(f"{method} — Distribution of Summary Metrics Across Images")
        ax.set_ylabel("value")
        ax.grid(axis="y", alpha=0.20)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{method}_metrics_boxplot.png")
        fig.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close(fig)


def _plot_dataset_concatenated_subplots(
    out_path: str,
    method_prob: Dict[str, np.ndarray],
    bin_centers: np.ndarray,
    method_total_counts: Optional[Dict[str, int]] = None,
) -> None:
    _safe_mkdir(os.path.dirname(out_path))

    preferred_order = [
        "mat_width_raw",
        "mat_width_dse",
        "pca_width_dse",
        "esd_width_dse",
        "eob_width_dse",
    ]
    methods = [m for m in preferred_order if m in method_prob and np.any(np.asarray(method_prob[m]) > 0)]
    methods += [m for m in sorted(method_prob.keys()) if m not in methods and np.any(np.asarray(method_prob[m]) > 0)]
    if not methods:
        return

    max_y = 0.0
    max_x = 0.0
    for m in methods:
        arr = np.asarray(method_prob[m], dtype=float)
        vmax = float(np.max(arr)) if arr.size else 0.0
        if np.isfinite(vmax):
            max_y = max(max_y, vmax)
        nz = np.nonzero(arr > 0)[0]
        if nz.size > 0:
            x_m = float(bin_centers[int(nz[-1])])
            if np.isfinite(x_m):
                max_x = max(max_x, x_m)
    if max_y <= 0:
        max_y = 1.0
    if max_x <= 0:
        max_x = float(bin_centers[-1])

    # Build sampled values from histogram probabilities for boxplot.
    rng = np.random.default_rng(0)
    bin_step = float(np.median(np.diff(bin_centers))) if len(bin_centers) > 1 else 1.0
    box_data = []
    box_labels = []
    for m in methods:
        p = np.asarray(method_prob[m], dtype=float)
        s = float(np.sum(p))
        if s <= 0:
            continue
        p = p / s
        total = None
        if method_total_counts is not None:
            total = int(method_total_counts.get(m, 0))
        n = int(min(50000, total if total and total > 0 else 50000))
        if n < 200:
            n = 200
        idx = rng.choice(len(bin_centers), size=n, p=p)
        vals = bin_centers[idx] + rng.uniform(-0.5 * bin_step, 0.5 * bin_step, size=n)
        vals = np.clip(vals, float(bin_centers[0]), float(max_x))
        box_data.append(vals.astype(np.float32))
        box_labels.append(m)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=200)
    ax_h, ax_b = axes

    for m in methods:
        p = np.asarray(method_prob[m], dtype=float)
        if p.size == 0:
            continue
        ax_h.plot(bin_centers, p, lw=2.0, alpha=0.60, label=m)
    ax_h.set_xlim(float(bin_centers[0]), float(max_x))
    ax_h.set_ylim(0.0, max_y * 1.02)
    ax_h.set_xlabel("Width (px)")
    ax_h.set_ylabel("Normalized Frequency")
    ax_h.set_title("Dataset-Concat Histograms")
    ax_h.grid(alpha=0.20)
    ax_h.legend(fontsize=8)

    if box_data:
        ax_b.boxplot(box_data, tick_labels=box_labels, showfliers=True, patch_artist=False)
    ax_b.set_ylabel("value")
    ax_b.set_title("Baseline Width Distribution Boxplots")
    ax_b.grid(axis="y", alpha=0.20)
    ax_b.tick_params(axis="x", rotation=20)

    fig.suptitle("Baseline Distribution Comparison", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def _rebuild_dataset_histogram_curves_from_npz(
    out_dir: str,
    methods: List[str],
    max_workers: Optional[int] = None,
) -> int:
    dist_dir = os.path.join(out_dir, "distribution_analysis")
    _safe_mkdir(dist_dir)
    out_csv = os.path.join(dist_dir, "dataset_histogram_curves.csv")

    stems = sorted(
        d for d in os.listdir(out_dir)
        if os.path.isdir(os.path.join(out_dir, d))
    )
    if not stems:
        return 2

    use_methods = list(methods)

    global_max = 0.0
    for stem in stems:
        stem_dir = os.path.join(out_dir, stem)
        for method in use_methods:
            npz_path = os.path.join(stem_dir, method, "width_map.npz")
            if not os.path.isfile(npz_path):
                continue
            try:
                vals = _extract_width_values(npz_path)
            except Exception:
                continue
            if vals.size:
                vmax = float(np.max(vals))
                if np.isfinite(vmax):
                    global_max = max(global_max, vmax)
    if global_max <= 0:
        global_max = 1.0

    bins = np.linspace(0.0, float(global_max), 257, dtype=np.float32)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    tasks = []
    for stem in stems:
        for method in use_methods:
            npz_path = os.path.join(out_dir, stem, method, "width_map.npz")
            if os.path.isfile(npz_path):
                tasks.append(
                    {
                        "method": method,
                        "npz_path": npz_path,
                        "bins": bins,
                    }
                )
    if not tasks:
        return 2

    method_counts = {m: np.zeros(len(bins) - 1, dtype=np.int64) for m in use_methods}
    workers = max_workers if max_workers is not None else (os.cpu_count() or 4)
    workers = max(1, min(int(workers), len(tasks)))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_hist_worker, t) for t in tasks]
        for fut in as_completed(futs):
            method, counts = fut.result()
            if method in method_counts:
                method_counts[method] += counts

    rows = []
    for method in sorted(method_counts.keys()):
        counts = method_counts[method].astype(np.float64)
        total = float(np.sum(counts))
        if total <= 0:
            continue
        prob = counts / total
        for bc, p, c in zip(bin_centers, prob, counts):
            rows.append(
                {
                    "method": method,
                    "bin_center": float(bc),
                    "probability": float(p),
                    "count": int(c),
                }
            )
    if not rows:
        return 2
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return 0


def rerender_distribution_plots_from_csv(
    out_dir: str,
    methods: Optional[List[str]] = None,
    max_workers: Optional[int] = None,
) -> int:
    dist_dir = os.path.join(out_dir, "distribution_analysis")
    method_csv = os.path.join(dist_dir, "per_image_method_stats.csv")
    pair_csv = os.path.join(dist_dir, "per_image_pairwise_stats.csv")
    dataset_pair_csv = os.path.join(dist_dir, "dataset_pairwise_distribution_stats.csv")
    hist_csv = os.path.join(dist_dir, "dataset_histogram_curves.csv")

    if not os.path.isfile(method_csv):
        print(f"[dist-csv] missing: {method_csv}")
        return 2

    df_method = pd.read_csv(method_csv)
    if df_method.empty:
        print("[dist-csv] per_image_method_stats.csv is empty")
        return 2

    _plot_per_image_quantile_iqr_bars(
        df_method,
        os.path.join(dist_dir, "per_image_quantile_iqr_bars.png"),
    )
    _plot_method_quantile_outlier_bars(
        df_method,
        os.path.join(dist_dir, "dataset_method_quantile_iqr_outlier_bars.png"),
    )

    if os.path.isfile(dataset_pair_csv):
        df_dataset_pair = pd.read_csv(dataset_pair_csv)
        if not df_dataset_pair.empty:
            pair_label = df_dataset_pair["method_a"].astype(str) + " vs " + df_dataset_pair["method_b"].astype(str)
            x = np.arange(len(df_dataset_pair))
            plt.figure(figsize=(max(10, 0.9 * len(df_dataset_pair)), 4.8), dpi=200)
            plt.bar(x, df_dataset_pair["wasserstein_hist"].to_numpy(float), alpha=0.75)
            plt.xticks(x, pair_label.tolist(), rotation=35, ha="right")
            plt.ylabel("Wasserstein Distance")
            plt.title("Dataset Pairwise Distribution Distance (Wasserstein)")
            plt.grid(axis="y", alpha=0.20)
            plt.tight_layout()
            plt.savefig(os.path.join(dist_dir, "dataset_pairwise_wasserstein_bars.png"), bbox_inches="tight", dpi=200)
            plt.close()

    if os.path.isfile(hist_csv):
        df_hist = pd.read_csv(hist_csv)
        if not df_hist.empty:
            bin_centers = np.sort(df_hist["bin_center"].unique().astype(float))
            method_prob = {}
            method_total_counts = {}
            for method, g in df_hist.groupby("method"):
                g2 = g.sort_values("bin_center")
                method_prob[str(method)] = g2["probability"].to_numpy(dtype=np.float64)
                method_total_counts[str(method)] = int(pd.to_numeric(g2["count"], errors="coerce").fillna(0).sum())
            _plot_dataset_concatenated_subplots(
                os.path.join(dist_dir, "dataset_concatenated_distributions.png"),
                method_prob=method_prob,
                bin_centers=bin_centers.astype(np.float64),
                method_total_counts=method_total_counts,
            )
    else:
        print(f"[dist-csv] missing histogram curves CSV, rebuilding from npz: {hist_csv}")
        if methods is None:
            methods = ["mat_width_raw", "mat_width_dse", "pca_width_dse", "esd_width_dse", "eob_width_dse"]
        rc = _rebuild_dataset_histogram_curves_from_npz(
            out_dir=out_dir,
            methods=list(methods),
            max_workers=max_workers,
        )
        if rc == 0 and os.path.isfile(hist_csv):
            df_hist = pd.read_csv(hist_csv)
            if not df_hist.empty:
                bin_centers = np.sort(df_hist["bin_center"].unique().astype(float))
                method_prob = {}
                method_total_counts = {}
                for method, g in df_hist.groupby("method"):
                    g2 = g.sort_values("bin_center")
                    method_prob[str(method)] = g2["probability"].to_numpy(dtype=np.float64)
                    method_total_counts[str(method)] = int(pd.to_numeric(g2["count"], errors="coerce").fillna(0).sum())
                _plot_dataset_concatenated_subplots(
                    os.path.join(dist_dir, "dataset_concatenated_distributions.png"),
                    method_prob=method_prob,
                    bin_centers=bin_centers.astype(np.float64),
                    method_total_counts=method_total_counts,
                )
        else:
            print(f"[dist-csv] unable to rebuild histogram curves from npz under: {out_dir}")

    print(f"[dist-csv] refreshed plots from CSV in: {dist_dir}")
    return 0


def run_distribution_analysis_from_npz(
    *,
    out_dir: str,
    methods: List[str],
    max_workers: Optional[int] = None,
) -> int:
    stems = sorted(
        d for d in os.listdir(out_dir)
        if os.path.isdir(os.path.join(out_dir, d))
    )
    if not stems:
        print(f"[dist] no baseline folders found in: {out_dir}")
        return 2

    use_methods = list(methods)

    # Pass 1: find global max width so all histograms share one binning.
    global_max = 0.0
    for stem in stems:
        stem_dir = os.path.join(out_dir, stem)
        for method in use_methods:
            npz_path = os.path.join(stem_dir, method, "width_map.npz")
            if not os.path.isfile(npz_path):
                continue
            try:
                vals = _extract_width_values(npz_path)
            except Exception:
                continue
            if vals.size:
                vmax = float(np.max(vals))
                if np.isfinite(vmax):
                    global_max = max(global_max, vmax)
    if global_max <= 0:
        global_max = 1.0

    bins = np.linspace(0.0, float(global_max), 257, dtype=np.float32)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width = float(bins[1] - bins[0])

    tasks = [
        {"stem": stem, "stem_dir": os.path.join(out_dir, stem), "methods": use_methods, "bins": bins}
        for stem in stems
    ]
    workers = max_workers if max_workers is not None else (os.cpu_count() or 4)
    workers = max(1, min(int(workers), len(tasks)))

    print(f"[dist] analyzing distributions with ProcessPoolExecutor workers={workers} images={len(tasks)}")
    out_rows_method: List[Dict[str, Any]] = []
    out_rows_pair: List[Dict[str, Any]] = []
    dataset_counts = {m: np.zeros(len(bins) - 1, dtype=np.int64) for m in use_methods}

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_distribution_worker, t) for t in tasks]
        for fut in as_completed(futs):
            try:
                result = fut.result()
            except Exception as e:
                print(f"[dist] worker failed: {e}")
                continue
            out_rows_method.extend(result.get("method_rows", []))
            out_rows_pair.extend(result.get("pair_rows", []))
            for hrow in result.get("hist_rows", []):
                m = hrow["method"]
                if m in dataset_counts:
                    dataset_counts[m] += np.asarray(hrow["counts"], dtype=np.int64)

    if not out_rows_method:
        print("[dist] no valid width data found in npz files")
        return 2

    dist_dir = os.path.join(out_dir, "distribution_analysis")
    _safe_mkdir(dist_dir)

    df_method = pd.DataFrame(out_rows_method).sort_values(["stem", "method"])
    df_pair = pd.DataFrame(out_rows_pair).sort_values(["stem", "method_a", "method_b"]) if out_rows_pair else pd.DataFrame()

    df_method.to_csv(os.path.join(dist_dir, "per_image_method_stats.csv"), index=False)
    if not df_pair.empty:
        df_pair.to_csv(os.path.join(dist_dir, "per_image_pairwise_stats.csv"), index=False)

    _plot_per_image_quantile_iqr_bars(
        df_method,
        os.path.join(dist_dir, "per_image_quantile_iqr_bars.png"),
    )
    _plot_method_quantile_outlier_bars(
        df_method,
        os.path.join(dist_dir, "dataset_method_quantile_iqr_outlier_bars.png"),
    )

    # Dataset-concatenated histograms
    hist_curve_rows = []
    dataset_summary_rows = []
    method_prob: Dict[str, np.ndarray] = {}
    for method in sorted(dataset_counts.keys()):
        counts = dataset_counts[method].astype(np.float64)
        total = float(np.sum(counts))
        if total <= 0:
            continue
        prob = counts / total
        method_prob[method] = prob
        for bc, p, c in zip(bin_centers, prob, counts):
            hist_curve_rows.append(
                {
                    "method": method,
                    "bin_center": float(bc),
                    "probability": float(p),
                    "count": int(c),
                }
            )
        mean = float(np.sum(prob * bin_centers))
        var = float(np.sum(prob * (bin_centers - mean) ** 2))
        cdf = np.cumsum(prob)
        q25 = float(bin_centers[np.searchsorted(cdf, 0.25)])
        q50 = float(bin_centers[np.searchsorted(cdf, 0.50)])
        q75 = float(bin_centers[np.searchsorted(cdf, 0.75)])
        dataset_summary_rows.append(
            {
                "method": method,
                "count_total": int(total),
                "mean": mean,
                "std": float(np.sqrt(max(var, 0.0))),
                "q25": q25,
                "q50": q50,
                "q75": q75,
                "min_bin": float(bin_centers[np.argmax(counts > 0)]),
                "max_bin": float(bin_centers[len(counts) - 1 - np.argmax((counts > 0)[::-1])]),
            }
        )
    method_total_counts = {m: int(np.sum(dataset_counts[m])) for m in method_prob.keys()}
    _plot_dataset_concatenated_subplots(
        os.path.join(dist_dir, "dataset_concatenated_distributions.png"),
        method_prob=method_prob,
        bin_centers=bin_centers.astype(np.float64),
        method_total_counts=method_total_counts,
    )

    if hist_curve_rows:
        pd.DataFrame(hist_curve_rows).to_csv(
            os.path.join(dist_dir, "dataset_histogram_curves.csv"),
            index=False,
        )

    df_dataset = pd.DataFrame(dataset_summary_rows).sort_values("method")
    df_dataset.to_csv(os.path.join(dist_dir, "dataset_method_distribution_stats.csv"), index=False)

    # Dataset pairwise distances from concatenated histograms
    pair_rows = []
    methods_present = [m for m in sorted(dataset_counts.keys()) if np.sum(dataset_counts[m]) > 0]
    for m1, m2 in combinations(methods_present, 2):
        p1 = dataset_counts[m1].astype(np.float64)
        p2 = dataset_counts[m2].astype(np.float64)
        p1 /= max(np.sum(p1), 1.0)
        p2 /= max(np.sum(p2), 1.0)
        cdf1 = np.cumsum(p1)
        cdf2 = np.cumsum(p2)
        wdist = float(np.sum(np.abs(cdf1 - cdf2)) * bin_width)
        js = 0.5 * (
            np.nansum(p1 * np.log((p1 + 1e-12) / (0.5 * (p1 + p2) + 1e-12)))
            + np.nansum(p2 * np.log((p2 + 1e-12) / (0.5 * (p1 + p2) + 1e-12)))
        )
        pair_rows.append(
            {
                "method_a": m1,
                "method_b": m2,
                "wasserstein_hist": wdist,
                "js_divergence": float(js),
            }
        )

    df_dataset_pair = pd.DataFrame(pair_rows).sort_values(["method_a", "method_b"]) if pair_rows else pd.DataFrame()
    if not df_dataset_pair.empty:
        df_dataset_pair.to_csv(os.path.join(dist_dir, "dataset_pairwise_distribution_stats.csv"), index=False)
        pair_label = df_dataset_pair["method_a"] + " vs " + df_dataset_pair["method_b"]
        x = np.arange(len(df_dataset_pair))
        plt.figure(figsize=(max(10, 0.9 * len(df_dataset_pair)), 4.8), dpi=200)
        plt.bar(x, df_dataset_pair["wasserstein_hist"].to_numpy(float), alpha=0.75)
        plt.xticks(x, pair_label.tolist(), rotation=35, ha="right")
        plt.ylabel("Wasserstein Distance")
        plt.title("Dataset Pairwise Distribution Distance (Wasserstein)")
        plt.grid(axis="y", alpha=0.20)
        plt.tight_layout()
        plt.savefig(os.path.join(dist_dir, "dataset_pairwise_wasserstein_bars.png"), bbox_inches="tight", dpi=200)
        plt.close()

    print(f"[dist] wrote analysis outputs to: {dist_dir}")
    return 0


def main():
    import os
    import pandas as pd
    from pandarallel import pandarallel
    import wslPath

    # ============================================================
    # HARD-CODED USER CONFIG (WINDOWS PATHS OK)
    # ============================================================
    WIN_IN_DIR  = r"C:\Users\13144\Documents\Masters_Thesis\datasets\SUT_Compressed\Ground Truth"
    WIN_OUT_DIR = r"C:\Users\13144\Documents\Masters_Thesis\datasets\SUT_Compressed\width_baselines"

    # --- convert to WSL paths once ---
    IN_DIR  = wslPath.to_posix(WIN_IN_DIR)  if wslPath.is_windows_path(WIN_IN_DIR)  else WIN_IN_DIR
    OUT_DIR = wslPath.to_posix(WIN_OUT_DIR) if wslPath.is_windows_path(WIN_OUT_DIR) else WIN_OUT_DIR

    THRESHOLD   = 0.25
    MIN_AREA_PX = 1000

    METHODS = [
        "skel_mat_raw",
        "skel_mat_dse",
        "skel_graph_longest_path",
        "mat_width_raw",
        "mat_width_dse",
        "pca_width_dse",
        "esd_width_dse",
        "eob_width_dse",
    ]
    WIDTH_METHODS = ["mat_width_raw", "mat_width_dse", "pca_width_dse", "esd_width_dse", "eob_width_dse"]

    MAKE_PLOTS = True

    PCA_MIN_POINTS      = 4
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
    NB_WORKERS = max(1, PHYSICAL_CORES)
    print(PHYSICAL_CORES)
    PROGRESS_BAR = True

    # ============================================================
    # Setup
    # ============================================================
    _safe_mkdir(OUT_DIR)

    if DIST_PLOTS_FROM_EXISTING_CSV_ONLY:
        print("[width_baseline_creator] DIST_PLOTS_FROM_EXISTING_CSV_ONLY=True -> rendering dist plots from existing CSVs")
        return rerender_distribution_plots_from_csv(
            OUT_DIR,
            methods=WIDTH_METHODS,
            max_workers=NB_WORKERS,
        )

    pandarallel.initialize(
        nb_workers=NB_WORKERS,
        progress_bar=PROGRESS_BAR,
        verbose=0,
    )

    # ============================================================
    # Collect images
    # ============================================================
    paths = _collect_image_paths(IN_DIR)

    paths = sorted(paths, key=_natural_sort_key)
    #paths = paths[41:42]
    #print(paths)
    #return

    if not paths:
        print(f"[width_baseline_creator] No images found under: {IN_DIR}")
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
        "pca_min_points": PCA_MIN_POINTS,
        "gpu_lock_path": GPU_LOCK_PATH,
    }

    if not RECOMPUTE_BASELINES:
        rc = 0
        if REGENERATE_PER_IMAGE_PLOTS_FROM_NPZ:
            print("[width_baseline_creator] RECOMPUTE_BASELINES=False + REGENERATE_PER_IMAGE_PLOTS_FROM_NPZ=True -> replaying per-image plots from npz")
            rc = replot_from_saved_npz(
                in_dir=IN_DIR,
                out_dir=OUT_DIR,
                threshold=THRESHOLD,
                min_area_px=MIN_AREA_PX,
                methods=METHODS,
            )
        else:
            print("[width_baseline_creator] RECOMPUTE_BASELINES=False + REGENERATE_PER_IMAGE_PLOTS_FROM_NPZ=False -> skipping per-image npz replay")
        if RUN_DISTRIBUTION_ANALYSIS:
            _ = run_distribution_analysis_from_npz(
                out_dir=OUT_DIR,
                methods=WIDTH_METHODS,
                max_workers=NB_WORKERS,
            )
        return rc

    # ============================================================
    # Sanity info
    # ============================================================
    print("[width_baseline_creator] images:", len(df))
    print("[width_baseline_creator] workers:", NB_WORKERS)
    print("[width_baseline_creator] methods:", METHODS)
    print("[width_baseline_creator] GPU_OK:", GPU_OK, " DSE_OK:", DSE_OK)
    print("[width_baseline_creator] IN_DIR :", IN_DIR)
    print("[width_baseline_creator] OUT_DIR:", OUT_DIR)

    if ("skel_mat_raw" in METHODS or "mat_width_raw" in METHODS) and not GPU_OK:
        print("[width_baseline_creator] WARNING: GPU medial-axis deps unavailable, using CPU fallback.")

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
    timings_csv = os.path.join(OUT_DIR, "width_baseline_timings.csv")
    res_df.to_csv(timings_csv, index=False)
    print("[width_baseline_creator] wrote:", timings_csv)

    # Separate inventory CSVs (easy to inspect / ablate).
    inv_cols = [
        "stem",
        "path",
        "H",
        "W",
        "crack_px",
        "n_geometry_methods_saved",
        "n_width_methods_saved",
        "geometry_methods_saved",
        "width_methods_saved",
    ]
    inv_cols = [c for c in inv_cols if c in res_df.columns]
    if inv_cols:
        inv_df = res_df[inv_cols].copy()
        geom_csv = os.path.join(OUT_DIR, "width_baseline_geometry_inventory.csv")
        width_csv = os.path.join(OUT_DIR, "width_baseline_width_inventory.csv")
        inv_df[
            [c for c in ("stem", "path", "H", "W", "crack_px", "n_geometry_methods_saved", "geometry_methods_saved") if c in inv_df.columns]
        ].to_csv(geom_csv, index=False)
        inv_df[
            [c for c in ("stem", "path", "H", "W", "crack_px", "n_width_methods_saved", "width_methods_saved") if c in inv_df.columns]
        ].to_csv(width_csv, index=False)
        print("[width_baseline_creator] wrote:", geom_csv)
        print("[width_baseline_creator] wrote:", width_csv)

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
    summary_csv = os.path.join(OUT_DIR, "width_baseline_timings_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print("[width_baseline_creator] wrote:", summary_csv)
    
    # ============================================================
    # Thesis committee plots
    # ============================================================
    if MAKE_PLOTS:
        print("[width_baseline_creator] generating summary plots...")
        plot_width_baseline_summary(res_df, summary_df, OUT_DIR)

    if RUN_DISTRIBUTION_ANALYSIS:
        _ = run_distribution_analysis_from_npz(
            out_dir=OUT_DIR,
            methods=WIDTH_METHODS,
            max_workers=NB_WORKERS,
        )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
