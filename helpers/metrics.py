#!/usr/bin/env python3
import cracktools as ct
#from helpers.crackhelpers import *
# helpers/metrics/metrics.py
from .plot_metrics import *
from .save_load_files import *
from .legacy import *


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import matplotlib
matplotlib.use("Agg")   # fastest non-GUI backend for PNG/PDF/SVG export
matplotlib.rcParams.update({
    "figure.max_open_warning": 0,
    "text.kerning_factor": 0,
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
})

import matplotlib.pyplot as plt
plt.ioff()   # no interactive figure updates

import numpy as np
from math import hypot, atan2, pi
from skimage.morphology import skeletonize
import hashlib
import time

ROUNDING_DIGITS=6

# ============================================
# SINGLE CID CORRESPONDENCE DEBUG CONFIG
# ============================================
DEBUG_CORRESPONDENCE_CID = "0"   # set to worst CID
DEBUG_CORRESPONDENCE_ON  = True

#################################################################################
#Basic Utilities
import matplotlib.pyplot as plt

import numpy as np
from math import hypot, atan2, pi
#from skimage.morphology import skeletonize
import hashlib
import time

# ---------- small utils ----------

def _finite_xy(arr):
    if arr is None: return np.empty((0,2), float)
    a = np.asarray(arr, float)
    if a.ndim != 2 or a.shape[1] != 2: return np.empty((0,2), float)
    m = np.all(np.isfinite(a), axis=1)
    a = a[m]
    # drop exact duplicates in sequence
    if len(a) > 1:
        keep = [0]
        for i in range(1, len(a)):
            if not (abs(a[i,0]-a[i-1,0]) < 1e-12 and abs(a[i,1]-a[i-1,1]) < 1e-12):
                keep.append(i)
        a = a[keep]
    return a


def _split_nan_none(arr):
    """Split polyline on [None,None] or NaNs into contiguous segments."""
    a = np.asarray(arr, float)
    if a.ndim != 2 or a.shape[1] != 2: return []
    bad = ~np.isfinite(a).all(axis=1)
    idx = np.where(bad)[0]
    pieces, start = [], 0
    for k in idx:
        if k - start >= 2:
            pieces.append(a[start:k])
        start = k+1
    if len(a) - start >= 2:
        pieces.append(a[start:])
    return pieces


def _resample_by_arclen(xy, N=200):
    """Uniform arclength resample (handles multiple segments)."""
    segs = _split_nan_none(xy) if np.any(~np.isfinite(xy)) else [xy]
    out = []
    for s in segs:
        s = _finite_xy(s)
        if len(s) < 2: continue
        d = np.sqrt(((s[1:]-s[:-1])**2).sum(1))
        L = np.concatenate([[0], np.cumsum(d)])
        if L[-1] <= 1e-9:
            continue
        t = np.linspace(0, L[-1], max(2, int(N * (L[-1]/sum(max(1e-9, _len_seg(_finite_xy(u))) for u in segs)))))
        xi = np.interp(t, L, s[:,0])
        yi = np.interp(t, L, s[:,1])
        out.append(np.column_stack([xi, yi]))
    if not out:
        return np.empty((0,2), float)
    return np.vstack(out)


def _len_seg(xy):
    if xy is None or len(xy) < 2: return 0.0
    return float(np.sqrt(((xy[1:]-xy[:-1])**2).sum(1)).sum())

###############################################################################################
# Midline Metrics
###############################################################################################

# --- DROP-IN REPLACEMENT in py ---

def _nn_dists(A, B):
    """
    Compute nearest-neighbor distances from each point in A to the closest point in B.
    Automatically uses GPU (CuPy + cupyx.scipy.spatial.cKDTree) if available,
    otherwise falls back to SciPy's CPU cKDTree.

    Returns
    -------
    dists : np.ndarray of shape (len(A),)
        Euclidean distances.
    """
    import numpy as np
    try:
        import cupy as cp
        from cupyx.scipy.spatial import cKDTree as GPU_KDTree
        CUPY_AVAILABLE = True
        gpu = True
        try:
            # detect CUDA presence
            _ = cp.cuda.runtime.getDeviceCount()
            if _ <= 0:
                gpu = False
        except Exception:
            gpu = False
    except ImportError:
        CUPY_AVAILABLE = False
        gpu = False

    if A is None or B is None or len(A) == 0 or len(B) == 0:
        return np.zeros((len(A),), dtype=float)

    if gpu:
        try:
            A_gpu = cp.asarray(A, dtype=cp.float32)
            B_gpu = cp.asarray(B, dtype=cp.float32)
            tree = GPU_KDTree(B_gpu)
            dists, _ = tree.query(A_gpu, k=1)
            return cp.asnumpy(dists)
        except Exception as e:
            print(f"[nn_dists][warn] GPU KDTree failed → falling back to CPU: {e}")

    # CPU fallback (SciPy)
    try:
        from scipy.spatial import cKDTree as CPU_KDTree
        tree = CPU_KDTree(B)
        dists, _ = tree.query(A, k=1)
        return dists
    except Exception as e:
        print(f"[nn_dists][warn] CPU KDTree failed, using brute force: {e}")
        A = np.asarray(A, float)
        B = np.asarray(B, float)
        diff = A[:, None, :] - B[None, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        return np.min(dists, axis=1)


def nn_mean_bidirectional(A, B):
    """Mean NN distance both directions."""
    A = _finite_xy(A); B = _finite_xy(B)
    return float(np.mean(_nn_dists(A,B))) + float(np.mean(_nn_dists(B,A)))


def hausdorff_max(A, B):
    """Max directed NN both ways."""
    A = _finite_xy(A); B = _finite_xy(B)
    da = _nn_dists(A, B); db = _nn_dists(B, A)
    if da.size == 0 or db.size == 0:
        return float('inf')
    return float(max(da.max(), db.max()))

def hausdorff_p95(A, B, q=95):
    """
    Robust symmetric Hausdorff distance (q-th percentile of NN distances).

    Parameters
    ----------
    A, B : (N,2) and (M,2) arrays
        Polylines / point sets.
    q : float, optional
        Percentile to use (default: 95).

    Returns
    -------
    float
        max( percentile_q(A→B), percentile_q(B→A) )
    """
    import numpy as np

    A = _finite_xy(A)
    B = _finite_xy(B)

    if len(A) == 0 or len(B) == 0:
        return float("inf")

    dAB = _nn_dists(A, B)
    dBA = _nn_dists(B, A)

    # filter non-finite just in case
    dAB = dAB[np.isfinite(dAB)]
    dBA = dBA[np.isfinite(dBA)]

    if dAB.size == 0 or dBA.size == 0:
        return float("inf")

    return float(max(
        np.percentile(dAB, q),
        np.percentile(dBA, q)
    ))

# --- DROP-IN REPLACEMENT in py ---


def frechet_discrete_ds(A, B, max_points=800):
    """
    Iterative Eiterâ€“Mannila discrete FrÃ©chet distance.
    - No recursion (avoids RecursionError)
    - Resamples long polylines to <= max_points for robustness
    """
    A = _finite_xy(A)
    B = _finite_xy(B)
    if len(A) == 0 or len(B) == 0:
        return float('inf')

    # Optional safety downsampling by arclength (keeps geometry)
    if len(A) > max_points:
        A = _resample_by_arclen(A, N=max_points)
    if len(B) > max_points:
        B = _resample_by_arclen(B, N=max_points)

    n, m = len(A), len(B)
    # DP table of size (n x m)
    ca = np.full((n, m), np.inf, dtype=float)

    # helper to compute Euclidean distance quickly
    def dist(i, j):
        dx = A[i, 0] - B[j, 0]
        dy = A[i, 1] - B[j, 1]
        return np.hypot(dx, dy)

    ca[0, 0] = dist(0, 0)
    # first column
    for i in range(1, n):
        ca[i, 0] = max(ca[i-1, 0], dist(i, 0))
    # first row
    for j in range(1, m):
        ca[0, j] = max(ca[0, j-1], dist(0, j))

    # fill DP
    for i in range(1, n):
        Ai = A[i]  # small locality win
        for j in range(1, m):
            d = np.hypot(Ai[0] - B[j, 0], Ai[1] - B[j, 1])
            ca[i, j] = max(min(ca[i-1, j], ca[i-1, j-1], ca[i, j-1]), d)

    return float(ca[n-1, m-1])


def tangent_angles(xy):
    xy = _finite_xy(xy)
    if len(xy) < 2: return np.array([])
    d = np.gradient(xy, axis=0)
    ang = np.arctan2(d[:,1], d[:,0])
    return ang


def mean_tangent_angle_error_degs(A, B):
    # resample to same count for angle comparison
    Ar = _resample_by_arclen(A, N=400)
    Br = _resample_by_arclen(B, N=len(Ar))
    if len(Ar)==0 or len(Br)==0: return np.nan
    aA = tangent_angles(Ar); aB = tangent_angles(Br)
    n = min(len(aA), len(aB))
    if n == 0: return np.nan
    da = np.abs(np.unwrap(aA[:n]) - np.unwrap(aB[:n]))
    da = np.mod(da + pi, 2*pi) - pi
    return float(np.degrees(np.mean(np.abs(da))))


def orthogonal_deviation(manual_xy, auto_xy, N=400, robust='median'):
    """Signed distance from manual to nearest auto, measured along manual normal."""
    M = _resample_by_arclen(manual_xy, N=N)
    A = _finite_xy(auto_xy)
    if len(M)==0 or len(A)==0:
        return dict(mean=np.nan, median=np.nan, rmse=np.nan, p95=np.nan)
    # manual normals
    d = np.gradient(M, axis=0)  # tangents
    norm = np.column_stack([-d[:,1], d[:,0]])
    nlen = np.maximum(1e-9, np.sqrt((norm**2).sum(1)))
    n = norm / nlen[:,None]
    # nearest auto → signed projection
    d2 = ((M[:,None,:] - A[None,:,:])**2).sum(2)
    idx = d2.argmin(1)
    v = A[idx] - M
    signed = (v * n).sum(1)
    absd = np.abs(signed)
    out = dict(
        mean=float(np.mean(signed)),
        median=float(np.median(signed)),
        rmse=float(np.sqrt(np.mean(absd**2))),
        p95=float(np.percentile(absd, 95))
    )
    return out


def coverage_at_tau(A, B, tau_px=3.0):
    A = _finite_xy(A); B = _finite_xy(B)
    if len(A)==0 or len(B)==0: return dict(A_to_B=0.0, B_to_A=0.0)
    da = _nn_dists(A,B); db = _nn_dists(B,A)
    return dict(
        A_to_B=float(np.mean(da <= tau_px)),
        B_to_A=float(np.mean(db <= tau_px))
    )


def relative_length_error(A, B):
    La = _len_seg(_finite_xy(A)); Lb = _len_seg(_finite_xy(B))
    if La < 1e-9: return np.nan
    return float(abs(Lb - La) / La)


def mask_iou(m1, m2):
    if m1 is None or m2 is None: return np.nan
    a = (np.asarray(m1)>0).astype(np.uint8)
    b = (np.asarray(m2)>0).astype(np.uint8)
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return (inter / union) if union else float('nan')

# === helpers/metrics.py ===

import os, json, numpy as np, cv2

def _safe_poly_fill(H, W, e1, e2):
    """Return (H,W) binary mask by filling polygon between two global-edge polylines."""
    e1 = np.asarray(e1, float); e2 = np.asarray(e2, float)
    if e1.ndim != 2 or e2.ndim != 2 or len(e1) < 2 or len(e2) < 2:
        return np.zeros((H, W), np.uint8)
    e1 = e1[np.isfinite(e1).all(axis=1)]
    e2 = e2[np.isfinite(e2).all(axis=1)]
    if len(e1) < 2 or len(e2) < 2:
        return np.zeros((H, W), np.uint8)
    poly = np.vstack([e1, e2[::-1]]).astype(np.int32)
    poly[:,0] = np.clip(poly[:,0], 0, W-1)
    poly[:,1] = np.clip(poly[:,1], 0, H-1)
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [poly], 1)
    return m








#####################################################################################################
# Edge metric specific functions
#####################################################################################################

# ==== Boundary / Surface metrics (scale-free, pixel domain) ====
import numpy as _np, cv2 as _cv2

def boundary_fscore(gt_mask, pred_mask, tau=2.0):
    """
    Boundary precision/recall/F1 (BSDS-style) with a distance tolerance tau (px).
    """
    gt = (gt_mask > 0).astype(_np.uint8)
    pr = (pred_mask > 0).astype(_np.uint8)

    # 1-px boundaries
    gt_edge = _cv2.Canny(gt*255, 0, 1)
    pr_edge = _cv2.Canny(pr*255, 0, 1)

    # distance transforms for nearest-boundary lookup
    gt_dt = _cv2.distanceTransform(1 - (gt_edge//255), _cv2.DIST_L2, 3)
    pr_dt = _cv2.distanceTransform(1 - (pr_edge//255), _cv2.DIST_L2, 3)

    gt_match = (pr_edge > 0) & (gt_dt <= tau)
    pr_match = (gt_edge > 0) & (pr_dt <= tau)

    # guard against zero division
    gt_cnt = max(1, int((gt_edge > 0).sum()))
    pr_cnt = max(1, int((pr_edge > 0).sum()))

    recall = gt_match.sum() / gt_cnt
    precision = pr_match.sum() / pr_cnt
    f1 = 2*precision*recall / (precision + recall + 1e-9)

    return dict(boundary_precision=float(precision),
                boundary_recall=float(recall),
                boundary_f1=float(f1))

def assd_hd95(gt_mask, pred_mask):
    """
    Average Symmetric Surface Distance (ASSD) and HD95 between mask boundaries.
    """
    gt = (gt_mask > 0).astype(_np.uint8)
    pr = (pred_mask > 0).astype(_np.uint8)
    gt_edge = _cv2.Canny(gt*255, 0, 1)
    pr_edge = _cv2.Canny(pr*255, 0, 1)

    # distance transforms
    gt_dt = _cv2.distanceTransform(1 - (gt_edge//255), _cv2.DIST_L2, 3)
    pr_dt = _cv2.distanceTransform(1 - (pr_edge//255), _cv2.DIST_L2, 3)

    d_gt = gt_dt[pr_edge > 0]
    d_pr = pr_dt[gt_edge > 0]
    if d_gt.size + d_pr.size == 0:
        return dict(ASSD=_np.nan, HD95=_np.nan)

    all_d = _np.concatenate([d_gt, d_pr]) if d_gt.size and d_pr.size else (d_gt if d_gt.size else d_pr)
    assd = float(_np.mean(all_d))
    hd95 = float(_np.percentile(all_d, 95))
    return dict(ASSD=assd, HD95=hd95)

def width_stats(gt_widths, pred_widths):
    """
    One-to-one numeric width comparison in pixels (no physical scale).
    """
    gt = _np.asarray(gt_widths, float)
    pr = _np.asarray(pred_widths, float)
    n = min(len(gt), len(pr))
    if n == 0:
        return dict(width_mae=_np.nan, width_rmse=_np.nan, width_bias=_np.nan, width_corr=_np.nan)
    diff = pr[:n] - gt[:n]
    mae  = float(_np.mean(_np.abs(diff)))
    rmse = float(_np.sqrt(_np.mean(diff**2)))
    bias = float(_np.mean(diff))
    corr = float(_np.corrcoef(gt[:n], pr[:n])[0,1]) if n > 1 else _np.nan
    return dict(width_mae=mae, width_rmse=rmse, width_bias=bias, width_corr=corr)







#################################################################################
# Metrics calculations functions
#################################################################################

# ---- local helpers
def compute_mask_metrics(gt_mask, pred_mask):
    """
    Region metrics + confusion counts + useful derived scalars
    (underfill/overfill, size proxies).
    """
    import numpy as np

    gt = gt_mask.astype(bool)
    pr = pred_mask.astype(bool)

    tp = int(np.logical_and(gt, pr).sum())
    fp = int(np.logical_and(~gt, pr).sum())
    fn = int(np.logical_and(gt, ~pr).sum())
    tn = int(np.logical_and(~gt, ~pr).sum())

    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    iou       = tp / (tp + fp + fn + eps)

    # --- size proxies ---
    gt_area   = tp + fn
    pred_area = tp + fp
    union     = tp + fp + fn

    # --- fill diagnostics ---
    underfill_rate = fn / (gt_area + eps)     # fraction of GT missed
    overfill_rate  = fp / (pred_area + eps)   # fraction of prediction that is wrong
    fill_ratio     = pred_area / (gt_area + eps)

    return {
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
        "iou":       float(iou),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,

        # extra context columns (very useful for plots)
        "gt_area_px": int(gt_area),
        "pred_area_px": int(pred_area),
        "union_area_px": int(union),
        "underfill_rate": float(underfill_rate),
        "overfill_rate":  float(overfill_rate),
        "fill_ratio":     float(fill_ratio),
    }

def save_mask_comparison_plot(gt_mask, pred_mask, out_path, show=False):
    gt = gt_mask.astype(bool)
    pr = pred_mask.astype(bool)
    iou = np.logical_and(gt, pr)   # intersection
    oou = np.logical_and(gt, ~pr)  # missed crack
    cou = np.logical_and(~gt, pr)  # false positive
    vis = np.zeros((*gt.shape, 3), dtype=np.uint8)
    vis[iou] = [255, 255, 255]
    vis[oou] = [255,   0,   0]
    vis[cou] = [  0,   0, 255]
    if show:
        plt.figure(figsize=(8, 6))
        plt.imshow(vis); plt.title("Mask Comparison Overlay"); plt.axis("off"); plt.show()
    else:
        plt.imsave(out_path, vis)
        
def _reconstruct_full_mask(crack, H, W):
    """
    Strict version with detailed debug:
    Verifies crop size, bbox validity, and paste location before constructing full mask.
    Does NOT silently repair or fallback — if data is inconsistent, returns zeros + message.
    """
    import numpy as np

    src = crack.get("src") or crack.get("source")
    mc = crack.get("mask_crop", None)
    bb = crack.get("mask_bbox", None)
    mid = crack.get("midline", [])
    print(f"\n[DEBUG MASK] src={src}")
    print(f"  mask_crop type={type(mc)}, len={len(mc) if mc is not None else 'None'}")
    print(f"  mask_bbox={bb}")
    print(f"  midline len={len(mid)}")

    if len(mid) > 0:
        arr = np.array(mid, float)
        print(f"  midline x-range=({arr[:,0].min():.1f},{arr[:,0].max():.1f}), "
              f"y-range=({arr[:,1].min():.1f},{arr[:,1].max():.1f})")

    # sanity
    if mc is None or bb is None:
        print("[DEBUG MASK] âŒ missing mask_crop or mask_bbox")
        return np.zeros((H, W), dtype=np.uint8)

    crop = np.array(mc, dtype=np.uint8)
    if crop.ndim != 2:
        print(f"[DEBUG MASK] âŒ mask_crop ndim={crop.ndim}, expected 2")
        return np.zeros((H, W), dtype=np.uint8)

    if not isinstance(bb, (list, tuple)) or len(bb) != 4:
        print("[DEBUG MASK] âŒ invalid bbox format")
        return np.zeros((H, W), dtype=np.uint8)

    x, y, w, h = [int(v) for v in bb]
    print(f"[DEBUG MASK] bbox parsed → x={x}, y={y}, w={w}, h={h}")

    # check if bbox is within image
    if x < 0 or y < 0 or x >= W or y >= H:
        print("[DEBUG MASK] âŒ bbox origin outside image bounds")
        return np.zeros((H, W), dtype=np.uint8)

    # check crop consistency
    print(f"[DEBUG MASK] crop shape={crop.shape}, target area=({y}:{y+h}, {x}:{x+w})")

    if h <= 0 or w <= 0:
        print("[DEBUG MASK] âŒ non-positive bbox dimensions")
        return np.zeros((H, W), dtype=np.uint8)

    # safe paste within limits
    x2, y2 = min(x + w, W), min(y + h, H)
    w_eff, h_eff = max(0, x2 - x), max(0, y2 - y)
    if w_eff == 0 or h_eff == 0:
        print("[DEBUG MASK] âŒ effective bbox has zero area after clipping")
        return np.zeros((H, W), dtype=np.uint8)

    crop = (crop > 0).astype(np.uint8)
    crop = crop[:h_eff, :w_eff]

    m = np.zeros((H, W), dtype=np.uint8)
    m[y:y+h_eff, x:x+w_eff] = crop
    print(f"[DEBUG MASK] âœ… pasted crop ({crop.shape}) into full mask at ({x},{y})")
    return m
    
# helpers/metrics.py

import os, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm

# import your existing primitives
#   normals_from_mask_for_midline(midline_xy, mask, max_radius)
#   (and any others you already have here)
#from helpers.metrics import normals_from_mask_for_midline  # if this file IS helpers.metrics, remove this line

def normals_from_mask_for_midline(midline_xy, mask, max_radius=50):
    """
    Pixel-accurate version:
    - Polygonizes the mask into exact pixel-boundary polygons using rasterio.
    - Shifts coords by -0.5 so edges align with imshow pixel grid.
    - Intersects midline normals with those polygons so endpoints lie exactly on the mask edge.
    Robustified to avoid NaNs / zero-length edges.
    """
    import numpy as np
    import shapely
    from shapely.geometry import shape, LineString, Point, MultiPoint
    import rasterio.features

    H, W = mask.shape
    midline_xy = np.asarray(midline_xy, float)
    if midline_xy.ndim != 2 or midline_xy.shape[1] != 2 or len(midline_xy) < 2:
        n = len(midline_xy) if midline_xy.ndim > 0 else 0
        return (np.full(n, np.nan),) * 5, []

    # ---- tangent + normals ----
    try:
        from cracktools.segmentation import (
            compute_smooth_tangent_normals,
            resolve_normal_pair_with_fallback,
        )
        _, nor = compute_smooth_tangent_normals(midline_xy[:, 0], midline_xy[:, 1])
    except Exception:
        resolve_normal_pair_with_fallback = None
        dx, dy = np.gradient(midline_xy[:, 0]), np.gradient(midline_xy[:, 1])
        nrm = np.hypot(dx, dy) + 1e-12
        tan = np.stack([dx / nrm, dy / nrm], axis=1)
        nor = np.stack([-tan[:, 1], tan[:, 0]], axis=1)

    # ---- polygonize mask -> shapely polygons ----
    mask_bin = (mask > 0).astype(np.uint8)
    polygons = []
    for geom, val in rasterio.features.shapes(mask_bin, mask=mask_bin):
        if val == 1:
            poly = shape(geom)
            # shift by -0.5 so edges align with imshow pixel grid
            poly = shapely.affinity.translate(poly, xoff=-0.5, yoff=-0.5)
            polygons.append(poly)

    if not polygons:
        # empty mask: nothing we can do safely
        N = len(midline_xy)
        return (np.full(N, np.nan),) * 5, []

    edges = [poly.boundary for poly in polygons]

    # ---- helper: clamp midline point to nearest polygon boundary if outside ----
    from math import inf

    def clamp_to_polygon(p):
        """Ensure p lies on/in the mask: returns closest point on any polygon boundary."""
        P = Point(p[0], p[1])
        # already inside some polygon
        for poly in polygons:
            if poly.contains(P) or poly.touches(P):
                return np.asarray(p, float)

        best = None
        best_d = inf
        for edge in edges:
            # nearest point on this boundary
            proj = edge.interpolate(edge.project(P))
            d = proj.distance(P)
            if d < best_d:
                best_d = d
                best = (proj.x, proj.y)

        if best is None:
            # fall back to original point (will be skipped later if invalid)
            return np.asarray(p, float)
        return np.asarray(best, float)

    N = len(midline_xy)
    e1x = np.full(N, np.nan); e1y = np.full(N, np.nan)
    e2x = np.full(N, np.nan); e2y = np.full(N, np.nan)
    widths_mask = np.full(N, np.nan)

    eps = 1e-6

    for i, (p_raw, nvec) in enumerate(zip(midline_xy, nor)):
        # basic sanity checks
        if not np.all(np.isfinite(p_raw)) or not np.all(np.isfinite(nvec)):
            continue

        nlen = float(np.hypot(nvec[0], nvec[1]))
        if nlen < eps:
            # direction is undefined here; skip
            continue
        nvec = nvec / nlen

        # clamp midline point to polygon if it's outside
        p = clamp_to_polygon(p_raw)
        if not np.all(np.isfinite(p)):
            continue

        # build long ray
        A = (p[0] - max_radius * nvec[0], p[1] - max_radius * nvec[1])
        B = (p[0] + max_radius * nvec[0], p[1] + max_radius * nvec[1])
        if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))):
            continue

        ray = LineString([A, B])

        hits = []
        for edge in edges:
            inter = edge.intersection(ray)
            if inter.is_empty:
                continue
            if isinstance(inter, Point):
                hits.append((inter.x, inter.y))
            elif isinstance(inter, MultiPoint):
                for g in inter.geoms:
                    hits.append((g.x, g.y))
            elif inter.geom_type == "LineString":
                coords = np.asarray(inter.coords, float)
                if len(coords) >= 1:
                    hits.append(tuple(coords[0]))
                if len(coords) >= 2:
                    hits.append(tuple(coords[-1]))

        if len(hits) < 2:
            # cannot define a width here
            continue

        # remove exact duplicates among hits
        hits_arr = np.asarray(hits, float)
        hits_arr = np.unique(hits_arr, axis=0)
        if len(hits_arr) < 2:
            continue
        hits = [tuple(h) for h in hits_arr]

        # project hits along the normal direction
        dists = [np.dot([hx - p[0], hy - p[1]], nvec) for (hx, hy) in hits]

        # classify as "left" (negative side) / "right" (positive side)
        left_pts  = [(hx, hy, d) for (hx, hy), d in zip(hits, dists) if d < -eps]
        right_pts = [(hx, hy, d) for (hx, hy), d in zip(hits, dists) if d > eps]

        left_cands = [(hx, hy) for (hx, hy, _) in left_pts]
        right_cands = [(hx, hy) for (hx, hy, _) in right_pts]

        def _nearest_boundary_on_side(side_sign):
            P = Point(p[0], p[1])
            best = None
            best_dist = np.inf
            for edge in edges:
                proj = edge.interpolate(edge.project(P))
                q = (float(proj.x), float(proj.y))
                dside = float(np.dot([q[0] - p[0], q[1] - p[1]], nvec))
                if side_sign < 0 and dside >= -eps:
                    continue
                if side_sign > 0 and dside <= eps:
                    continue
                dist = float(np.hypot(q[0] - p[0], q[1] - p[1]))
                if np.isfinite(dist) and dist < best_dist:
                    best_dist = dist
                    best = q
            return best

        if resolve_normal_pair_with_fallback is not None:
            pair = resolve_normal_pair_with_fallback(
                p=p,
                nvec=nvec,
                cand_a=left_cands,
                cand_b=right_cands,
                fallback_a=lambda: _nearest_boundary_on_side(-1),
                fallback_b=lambda: _nearest_boundary_on_side(+1),
                score_a=lambda q: abs(float(np.dot([q[0] - p[0], q[1] - p[1]], nvec))),
                score_b=lambda q: abs(float(np.dot([q[0] - p[0], q[1] - p[1]], nvec))),
                max_dist=float(max_radius),
                max_dist_mult=2.0,
                scale_ref=float(np.sqrt(max(1.0, H * W))),
                fallback_min_px=3.0,
                fallback_max_px=5.0,
                fallback_scale=0.003,
                normal_align_min=0.30,
                span_max_mult=2.0,
            )
            if pair is None:
                continue
            (lp, rp, _, _, w) = pair
        else:
            if not left_pts or not right_pts:
                continue
            lp, _ = min((((hx, hy), abs(d)) for (hx, hy, d) in left_pts), key=lambda t: t[1])
            rp, _ = min((((hx, hy), abs(d)) for (hx, hy, d) in right_pts), key=lambda t: t[1])
            w = float(np.hypot(rp[0] - lp[0], rp[1] - lp[1]))
            if not np.isfinite(w) or w < eps:
                continue

        e1x[i], e1y[i] = lp
        e2x[i], e2y[i] = rp
        widths_mask[i] = w

    return (e1x, e1y, e2x, e2y, widths_mask), polygons

# -------------------- lightweight cache for mask-normals ----------------------
_NORMALS_CACHE = {}  # key: (mask_sha1, midline_sha1) -> (e1x,e1y,e2x,e2y,w_mask)

def _sha1_ndarray(a: np.ndarray) -> str:
    if a is None:
        return "none"
    a = np.ascontiguousarray(a)
    h = hashlib.sha1()
    h.update(a.view(np.uint8))
    return h.hexdigest()

def _mask_midline_cache_key(mask_bin: np.ndarray, midline_xy: np.ndarray) -> str:
    # Reduce midline precision to avoid tiny float diffs busting the cache
    ml = np.round(np.asarray(midline_xy, dtype=np.float32), 3)
    return (_sha1_ndarray(mask_bin.astype(np.uint8)), _sha1_ndarray(ml))

def _plot_gt_normals(mask_bin, mid_xy, e1_xy, e2_xy, out_png, title):
    """
    Plot GT normals sparsely for visualization only.
    (No effect on exported data.)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    H, W = mask_bin.shape
    img = np.zeros((H, W, 3), np.uint8)
    img[mask_bin > 0] = (255, 255, 255)

    plt.figure(figsize=(8, 8))
    plt.imshow(img)

    if len(mid_xy) >= 2:
        plt.plot(mid_xy[:,0], mid_xy[:,1], 'k-', lw=3)
        plt.plot(mid_xy[:,0], mid_xy[:,1], 'w-', lw=1.5)

    n = min(len(e1_xy), len(e2_xy), len(mid_xy))
    print(n)
    if n >= 2:
        # --- only subsample for drawing ---
        stride = 100
        e1_draw = e1_xy[::stride]
        e2_draw = e2_xy[::stride]

        segs = np.stack([e1_draw, e2_draw], axis=1)
        lc = LineCollection(segs, colors='C0', linewidths=1.5, alpha=0.85)
        plt.gca().add_collection(lc)

    plt.title(title)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

import base64
import numpy as np

def _decode_packbits_b64_to_mask(packbits_b64, shape):
    raw = base64.b64decode(packbits_b64.encode("utf-8"))
    bits = np.frombuffer(raw, dtype=np.uint8)
    arr = np.unpackbits(bits)

    H, W = int(shape[0]), int(shape[1])
    n = H * W
    if arr.size < n:
        arr = np.pad(arr, (0, n - arr.size), constant_values=0)

    return arr[:n].reshape((H, W)).astype(np.uint8)


def bite_blob_to_fullmask(bite_blob, H, W, base_bbox=None, assume_local=None):
    """
    Decode bite blob into a GLOBAL fullmask (H,W).

    bite_blob["bbox"] is *supposed* to be global xywh, but some exporters store it
    relative to the combined crack's mask_bbox (local coords). This function supports both.

    Args:
        base_bbox: optional [bx,by,bw,bh] of the combined crack. If provided, we can
                   interpret bite bbox as local-to-base when needed.
        assume_local:
            - None  : auto-detect (recommended)
            - True  : always treat bite bbox as local to base_bbox
            - False : always treat bite bbox as global
    """
    import numpy as np

    full = np.zeros((H, W), np.uint8)
    if not bite_blob:
        return full

    bb = bite_blob.get("bbox")
    shape = bite_blob.get("shape")
    b64 = bite_blob.get("packbits_b64")

    if not (bb and shape and b64):
        return full

    x0, y0, w, h = map(int, bb)
    mask = _decode_packbits_b64_to_mask(b64, shape)
    if mask is None or mask.ndim != 2:
        return full

    # -------------------------
    # Decide whether bbox is local
    # -------------------------
    local = False
    if base_bbox is not None:
        bx, by, bw, bh = map(int, base_bbox)

        if assume_local is True:
            local = True
        elif assume_local is False:
            local = False
        else:
            # Heuristic auto-detect:
            # If bbox appears to fit *inside* base bbox dimensions, it's likely local.
            # (This is extremely common for bite blobs packed relative to the combined crop.)
            if 0 <= x0 <= bw and 0 <= y0 <= bh and (x0 + w) <= (bw + 2) and (y0 + h) <= (bh + 2):
                local = True

        if local:
            x0 += bx
            y0 += by

    # -------------------------
    # Paste into full global mask
    # -------------------------
    hh = min(h, mask.shape[0], H - y0)
    ww = min(w, mask.shape[1], W - x0)
    if hh > 0 and ww > 0 and x0 >= 0 and y0 >= 0:
        full[y0:y0+hh, x0:x0+ww] = mask[:hh, :ww]

    return full

def split_polyline_by_mask(xy, invalid_mask, min_pts=2):
    """
    xy: (N,2)
    invalid_mask: (N,) bool — True = REMOVE
    returns: list of (M,2) arrays
    """
    out = []
    buf = []

    for p, bad in zip(xy, invalid_mask):
        if bad:
            if len(buf) >= min_pts:
                out.append(np.asarray(buf, float))
            buf = []
        else:
            buf.append(p)

    if len(buf) >= min_pts:
        out.append(np.asarray(buf, float))

    return out

def _polyline_points_keyset(segs, ndigits=3):
    """
    Build a set of rounded (x,y) keys from a list of polylines.

    Parameters
    ----------
    segs : list of (N,2) arrays
    ndigits : int
        Rounding precision for float stability

    Returns
    -------
    set of (float, float)
    """
    pts = set()
    if not segs:
        return pts

    for S in segs:
        if S is None:
            continue
        S = np.asarray(S, float)
        if S.ndim != 2 or S.shape[1] != 2:
            continue
        for x, y in S:
            if np.isfinite(x) and np.isfinite(y):
                pts.add((
                    round(float(x), ndigits),
                    round(float(y), ndigits),
                ))
    return pts

def subtract_segments_by_pointset(
    orig_segs,
    keep_point_set,
    min_pts=2,
    ndigits=3,
):
    """
    Subtract kept geometry from original segments using a point keyset.

    Parameters
    ----------
    orig_segs : list of (N,2) arrays
        Original (topology-pruned) geometry
    keep_point_set : set of (x,y)
        Points that survived later stages (post-bite, post-width)
    min_pts : int
        Minimum points per returned segment
    ndigits : int
        Rounding precision (must match keyset)

    Returns
    -------
    list of (M,2) arrays
        Geometry removed BEFORE bite (pure topology prune)
    """
    removed = []

    for S in orig_segs:
        if S is None:
            continue
        S = np.asarray(S, float)
        if S.ndim != 2 or len(S) < 2:
            continue

        buf = []
        for x, y in S:
            key = (
                round(float(x), ndigits),
                round(float(y), ndigits),
            )
            if key in keep_point_set:
                if len(buf) >= min_pts:
                    removed.append(np.asarray(buf, float))
                buf = []
            else:
                buf.append([x, y])

        if len(buf) >= min_pts:
            removed.append(np.asarray(buf, float))

    return removed

def _qpts_keyset(segs, q=0.25):
    """
    Quantized point keyset for provenance comparisons.
    q in pixels; smaller is stricter.
    """
    pts = set()
    if not segs:
        return pts
    q = float(q) if q is not None else 0.25
    if q <= 0:
        q = 0.25

    for S in segs:
        if S is None:
            continue
        S = np.asarray(S, float)
        if S.ndim != 2 or S.shape[1] != 2:
            continue
        P = np.round(S / q).astype(np.int32)
        for x, y in P:
            pts.add((int(x), int(y)))
    return pts

def _qpts_keys_for_seg(S, q=0.25):
    S = np.asarray(S, float)
    if S.ndim != 2 or S.shape[1] != 2 or len(S) < 1:
        return []
    q = float(q) if q is not None else 0.25
    if q <= 0:
        q = 0.25
    P = np.round(S / q).astype(np.int32)
    return [(int(x), int(y)) for x, y in P]

def _seg_key_meta(S, m):
    mm = m if isinstance(m, dict) else {}
    return {
        "seg_idx": mm.get("seg_idx", None),
        "branch_id": mm.get("branch_id", None),
        "atomic_id": mm.get("atomic_id", None),
        "npts": int(len(S)) if S is not None else 0,
    }

def _write_csv(path, rows, header=None):
    import csv
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if header:
                w.writerow(header)
            for r in (rows or []):
                w.writerow(r)
    except Exception as e:
        print(f"[TOPO DBG] CSV write failed: {path}: {e}")

def _append_csv_row(path, row, header=None):
    import csv
    try:
        need_header = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if need_header and header:
                w.writerow(header)
            w.writerow(row)
    except Exception as e:
        print(f"[TOPO DBG] CSV append failed: {path}: {e}")

def _plot_seg_provenance(
    *,
    out_png,
    S_full,
    keep_mask_bool,
    bite_mask_bool=None,
    title="",
):
    """
    Segment-local provenance view:
      - green: kept points
      - red: missing points (not in keep set)
      - orange: bite-marked points (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[TOPO DBG] matplotlib unavailable; skipping provenance plot")
        return

    S = np.asarray(S_full, float)
    n = int(len(S))
    if S.ndim != 2 or S.shape[1] != 2 or n < 2:
        return

    keep = np.asarray(keep_mask_bool, bool).reshape(-1)[:n]
    if keep.size < n:
        pad = np.zeros((n - keep.size,), dtype=bool)
        keep = np.concatenate([keep, pad], axis=0)
    miss = ~keep

    fig, ax = plt.subplots(figsize=(7, 3), dpi=220)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    ax.plot(S[:, 0], S[:, 1], lw=1.0, alpha=0.25, color="black")

    def _plot_runs(mask, color, lw):
        mask = np.asarray(mask, bool).reshape(-1)[:n]
        if mask.size < n:
            mask = np.concatenate([mask, np.zeros((n - mask.size,), dtype=bool)], axis=0)
        buf = []
        for i in range(n):
            if mask[i]:
                buf.append(i)
            else:
                if len(buf) >= 2:
                    pts = S[buf]
                    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw)
                buf = []
        if len(buf) >= 2:
            pts = S[buf]
            ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw)

    _plot_runs(miss, color=(0.75, 0.1, 0.1), lw=2.5)
    _plot_runs(keep, color=(0.1, 0.6, 0.2), lw=2.8)

    if bite_mask_bool is not None:
        bite = np.asarray(bite_mask_bool, bool).reshape(-1)[:n]
        if bite.size < n:
            bite = np.concatenate([bite, np.zeros((n - bite.size,), dtype=bool)], axis=0)
        if np.any(bite):
            _plot_runs(bite, color=(0.95, 0.6, 0.05), lw=2.8)

    try:
        fig.savefig(out_png, bbox_inches="tight")
    finally:
        plt.close(fig)

def build_branch_bite_masks(dominance_meta, H, W):
    """
    Returns:
        dict: branch_id -> full_mask (uint8 HxW)
    """
    out = {}

    if not dominance_meta:
        return out

    branches = dominance_meta.get("branches", {})
    bite = dominance_meta.get("bite", {})

    by_cause = bite.get("by_cause", {})

    for bid, bmeta in branches.items():
        bid = int(bid)
        lost = bmeta.get("lost_to", [])  # â† THIS is critical

        masks = []
        for cause in lost:
            blob = by_cause.get(cause)
            if blob:
                masks.append(bite_blob_to_fullmask(blob, H, W))

        if masks:
            out[bid] = np.logical_or.reduce(masks)
        else:
            out[bid] = np.zeros((H, W), bool)

    return out


#############################
#   Stage 4 helper functions
#############################

def _safe_int(x, default=None):
            try:
                return int(x)
            except Exception:
                return default

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ----------------------------
# helpers
# ----------------------------
'''def _safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default'''

def _dump_json(path, obj):
    try:
        import json
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
    except Exception:
        pass

def _decode_by_losing_branch(dom_meta, H, W, base_bbox=None, assume_local=None):
    """
    Returns: dict[int bid] -> bool fullmask

    If bite blobs are stored local-to-combined mask_bbox, pass base_bbox so we can
    lift them into GLOBAL coords correctly.
    """
    out = {}
    if not isinstance(dom_meta, dict):
        return out

    bite = dom_meta.get("bite")
    if not isinstance(bite, dict):
        return out

    by_branch = bite.get("by_losing_branch")
    if not isinstance(by_branch, dict):
        return out

    fallback_bbox = bite.get("bbox")

    for bid_str, blob in by_branch.items():
        bid = _safe_int(bid_str)
        if bid is None:
            continue

        if "bbox" not in blob and fallback_bbox is not None:
            blob = dict(blob)
            blob["bbox"] = fallback_bbox

        try:
            m = bite_blob_to_fullmask(blob, H, W, base_bbox=base_bbox, assume_local=assume_local)
            if m is not None and np.any(m):
                out[bid] = (m > 0)
        except Exception:
            continue

    return out


#############################
# Stage 4.5 helper functions
#############################

# ------------------------------------------------------------
# Polyline clipping with MIDPOINT sampling (important!)
# ------------------------------------------------------------
def _clip_polyline_into_runs(S, remove_mask, H, W, min_pts=2):
    """
    Split S into (kept_runs, removed_runs) based on remove_mask (True = remove).
    Uses midpoint sampling per segment to avoid missed bites.
    """
    S = np.asarray(S, float)
    if S is None or len(S) < 2 or remove_mask is None:
        return [S], []

    kept, removed = [], []
    buf_k, buf_r = [], []

    for i in range(len(S) - 1):
        p0 = S[i]
        p1 = S[i + 1]

        mx = 0.5 * (p0[0] + p1[0])
        my = 0.5 * (p0[1] + p1[1])

        ix = int(round(mx))
        iy = int(round(my))

        in_remove = False
        if 0 <= ix < W and 0 <= iy < H:
            in_remove = bool(remove_mask[iy, ix])

        if not in_remove:
            if not buf_k:
                buf_k.append(p0)
            buf_k.append(p1)
            if len(buf_r) >= min_pts:
                removed.append(np.asarray(buf_r, float))
            buf_r = []
        else:
            if not buf_r:
                buf_r.append(p0)
            buf_r.append(p1)
            if len(buf_k) >= min_pts:
                kept.append(np.asarray(buf_k, float))
            buf_k = []

    if len(buf_k) >= min_pts:
        kept.append(np.asarray(buf_k, float))
    if len(buf_r) >= min_pts:
        removed.append(np.asarray(buf_r, float))

    return kept, removed


import numpy as np
import math

# ============================================================
# Arc-length parameterization
# ============================================================

def arclen_s(xy):
    """
    Cumulative arc-length parameterization.
    Returns s with s[0]=0 and s[i]=sum ||p_k - p_{k-1}||.
    """
    xy = np.asarray(xy, float)
    if xy.ndim != 2 or len(xy) < 2:
        return np.asarray([], float)

    ds = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    return np.concatenate([[0.0], np.cumsum(ds)])


def local_step_sizes(xy):
    """
    Local step sizes Î”s_i = ||p_{i+1} - p_i||.
    """
    xy = np.asarray(xy, float)
    if xy.ndim != 2 or len(xy) < 2:
        return np.asarray([], float)
    return np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))


def _already_uniform_enough(xy, ds_target=1.0, mean_tol=0.02, cv_tol=0.05):
    """
    Very strict fast-path: treat as already-uniform only if:
      - mean(Î”s) is close to ds_target
      - coefficient of variation of Î”s is small
    """
    ds = local_step_sizes(xy)
    if ds.size == 0:
        return False

    mu = float(np.mean(ds))
    if (not np.isfinite(mu)) or mu <= 0:
        return False

    cv = float(np.std(ds) / mu) if mu > 0 else float("inf")
    return (abs(mu - ds_target) <= mean_tol * ds_target) and (cv <= cv_tol)


def resample_by_arclength(xy, *signals, ds_target=1.0, min_pts=2,
                          preserve_endpoints=True, fastpath=True):
    """
    Resample polyline and aligned signals onto uniform arc-length spacing.

    IMPORTANT DESIGN:
      - Call this unconditionally at the callsite for consistency.
      - Internally it may "fast-path" return inputs if already uniform enough.

    Args:
      xy: (N,2)
      signals: optional aligned 1D arrays length N (or None)
      ds_target: spacing in px
      preserve_endpoints: keep exact first/last point values
      fastpath: if True, returns inputs when already uniform enough

    Returns:
      xy_rs, sig1_rs, sig2_rs, ...
    """
    xy = np.asarray(xy, float)
    if xy.ndim != 2 or len(xy) < min_pts:
        return (None,) * (1 + len(signals))

    # Trim all signals to a common length
    n = len(xy)
    sigs = []
    for s in signals:
        if s is None:
            sigs.append(None)
            continue
        s = np.asarray(s, float)
        n = min(n, len(s))
        sigs.append(s)

    if n < min_pts:
        return (None,) * (1 + len(signals))

    xy = xy[:n]
    sigs = [None if s is None else s[:n] for s in sigs]

    # Optional strict fast-path
    ds_target = float(max(ds_target, 1e-6))
    if fastpath and _already_uniform_enough(xy, ds_target=ds_target):
        # Still return copies to avoid accidental in-place surprises downstream
        xy_out = np.array(xy, copy=True)
        out_sigs = []
        for s in sigs:
            out_sigs.append(None if s is None else np.array(s, copy=True))
        return (xy_out, *out_sigs)

    s = arclen_s(xy)
    if len(s) < 2:
        return (None,) * (1 + len(signals))

    L = float(s[-1])
    if (not np.isfinite(L)) or L <= 0:
        return (None,) * (1 + len(signals))

    n_out = max(int(math.floor(L / ds_target)) + 1, 2)
    s_new = np.linspace(0.0, L, n_out)

    x_new = np.interp(s_new, s, xy[:, 0])
    y_new = np.interp(s_new, s, xy[:, 1])
    xy_rs = np.column_stack([x_new, y_new]).astype(float)

    out_sigs = []
    for sig in sigs:
        if sig is None:
            out_sigs.append(None)
        else:
            out_sigs.append(np.interp(s_new, s, sig.astype(float)).astype(float))

    if preserve_endpoints and len(xy_rs) >= 2 and len(xy) >= 2:
        xy_rs[0]  = xy[0]
        xy_rs[-1] = xy[-1]
        for j, sig in enumerate(sigs):
            if sig is None:
                continue
            out_sigs[j][0]  = sig[0]
            out_sigs[j][-1] = sig[-1]

    return (xy_rs, *out_sigs)


# ---------------- helpers ----------------
def _split_on_nans(arr):
    arr = np.asarray(arr, float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return []
    good = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    segs, s = [], None
    for i, g in enumerate(good):
        if g and s is None:
            s = i
        elif not g and s is not None:
            if i - s >= 2:
                segs.append(arr[s:i])
            s = None
    if s is not None and len(arr) - s >= 2:
        segs.append(arr[s:])
    return segs

def _get_edges(crack):
    geo = crack.get("normal_edge_points") or crack.get("geodesic_edges") or {}
    e1 = np.asarray(geo.get("edge1", []), float)
    e2 = np.asarray(geo.get("edge2", []), float)
    return e1, e2

def _union_bboxes(bboxes):
    xs0, ys0, xs1, ys1 = [], [], [], []
    for b in bboxes:
        if not b or len(b) != 4:
            continue
        x, y, w, h = b
        if w <= 0 or h <= 0:
            continue
        xs0.append(x); ys0.append(y)
        xs1.append(x + w); ys1.append(y + h)
    if not xs0:
        return None
    return [min(xs0), min(ys0), max(xs1) - min(xs0), max(ys1) - min(ys0)]

def _decode_packbits_mask(blob):
    """
    blob = {"shape":[h,w], "packbits_b64": "..."}
    returns uint8 mask (h,w)
    """
    if not blob:
        return None
    shape = blob.get("shape") or [0, 0]
    h, w = int(shape[0]), int(shape[1])
    s = blob.get("packbits_b64") or ""
    if h <= 0 or w <= 0 or not s:
        return np.zeros((h, w), np.uint8)
    raw = base64.b64decode(s.encode("ascii"))
    packed = np.frombuffer(raw, dtype=np.uint8)
    # packed is (h * ceil(w/8),) flatten
    row_bytes = int((w + 7) // 8)
    if packed.size < h * row_bytes:
        # corrupted / mismatch: fail closed to empty
        return np.zeros((h, w), np.uint8)
    packed = packed[: h * row_bytes].reshape((h, row_bytes))
    unpacked = np.unpackbits(packed, axis=1)[:, :w]
    return unpacked.astype(np.uint8)

def _points_hit_bite(points_xy, bite_obj):
    """
    bite_obj:
        {"bbox":[x0,y0,w,h], "shape":[h,w], "packbits_b64":"..."}
    returns boolean mask len(points)
    """
    if bite_obj is None:
        return np.zeros((len(points_xy),), bool)

    bb = bite_obj.get("bbox")
    if not bb or len(bb) != 4:
        return np.zeros((len(points_xy),), bool)

    x0, y0, bw, bh = map(int, bb)
    if bw <= 0 or bh <= 0:
        return np.zeros((len(points_xy),), bool)

    m = _decode_packbits_mask({
        "shape": bite_obj.get("shape"),
        "packbits_b64": bite_obj.get("packbits_b64"),
    })
    if m is None or m.size == 0:
        return np.zeros((len(points_xy),), bool)

    pts = np.asarray(points_xy, float)
    xs = np.round(pts[:, 0]).astype(int) - x0
    ys = np.round(pts[:, 1]).astype(int) - y0
    ok = (xs >= 0) & (xs < bw) & (ys >= 0) & (ys < bh)
    hit = np.zeros((len(pts),), bool)
    if np.any(ok):
        hit[ok] = (m[ys[ok], xs[ok]] > 0)
    return hit

def _split_by_keep_mask(points, keep_mask):
    """
    points: (N,2)
    keep_mask: (N,) bool
    returns list of (run_points, run_indices_in_original)
    """
    pts = np.asarray(points, float)
    keep = np.asarray(keep_mask, bool)
    out = []
    s = None
    for i, k in enumerate(keep):
        if k and s is None:
            s = i
        elif (not k) and (s is not None):
            if i - s >= 2:
                idx = np.arange(s, i, dtype=int)
                out.append((pts[s:i], idx))
            s = None
    if s is not None and len(pts) - s >= 2:
        idx = np.arange(s, len(pts), dtype=int)
        out.append((pts[s:], idx))
    return out

def _linestring_length(S):
    S = np.asarray(S, float)
    if S.ndim != 2 or len(S) < 2:
        return 0.0
    d = np.linalg.norm(S[1:] - S[:-1], axis=1)
    d = d[np.isfinite(d)]
    return float(d.sum()) if len(d) else 0.0

def _build_branch_table(segs, seg_meta, shared_members=None):
    """
    Builds branch dict:
        branch_id -> {"members":set(str), "seg_idxs":[int], "length":float}
    If seg_meta has atomic_id, that's used for branch membership.
    """
    if shared_members is None:
        shared_members = None
    branches = {}
    for i, (S, m) in enumerate(zip(segs, seg_meta)):
        if S is None or len(S) < 2:
            continue
        bid = int(m.get("branch_id", i))
        aid = m.get("atomic_id")
        if bid not in branches:
            branches[bid] = {"members": set(), "seg_idxs": [], "length": 0.0}
        branches[bid]["seg_idxs"].append(i)
        branches[bid]["length"] += _linestring_length(S)
        if aid is not None:
            aid = str(aid)
            if shared_members is None or aid in shared_members:
                branches[bid]["members"].add(aid)
    return branches

def _greedy_match_branches(gt_br, pr_br):
    """
    gt_br, pr_br: dict branch_id -> {"members", "length", ...}
    Returns:
        list of (gt_bid, pr_bid, shared_count, shared_len)
    """
    pairs = []
    for g_id, g in gt_br.items():
        for p_id, p in pr_br.items():
            inter = g["members"] & p["members"]
            sc = len(inter)
            if sc <= 0:
                continue
            # tie-break with min length (stable, doesn't depend on float dominance noise)
            sl = float(min(g.get("length", 0.0), p.get("length", 0.0)))
            pairs.append((g_id, p_id, sc, sl))
    # sort by (shared_count, shared_len) desc
    pairs.sort(key=lambda t: (t[2], t[3]), reverse=True)

    gt_used = set()
    pr_used = set()
    out = []
    for g_id, p_id, sc, sl in pairs:
        if g_id in gt_used or p_id in pr_used:
            continue
        gt_used.add(g_id)
        pr_used.add(p_id)
        out.append((g_id, p_id, sc, sl))
    return out


# ============================================================
# WIDTH EVAL HELPERS (drop once, outside any function)
# ============================================================

import os, json
import numpy as np

def _safe_json_load(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def _polyline_length_px(pts):
    pts = np.asarray(pts, float)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return 0.0
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    d = d[np.isfinite(d)]
    return float(np.sum(d)) if d.size else 0.0

def _sample_width_map_at_pts(width_map_2d, pts_xy, *, bbox=None):
    """
    width_map_2d: (h,w) float/uint
    pts_xy: (N,2) in FULL-IMAGE coords
    bbox: (x,y,w,h) describing where width_map_2d lives in full image coords.
          If bbox is None, assume map is full image.
    """
    wm = np.asarray(width_map_2d)
    if wm.ndim != 2 or wm.size == 0:
        return None

    pts = np.asarray(pts_xy, float)
    if pts.ndim != 2 or pts.shape[0] < 1:
        return None

    if bbox is None:
        x0 = 0
        y0 = 0
    else:
        x0 = int(bbox[0])
        y0 = int(bbox[1])

    xi = np.rint(pts[:, 0]).astype(int) - x0
    yi = np.rint(pts[:, 1]).astype(int) - y0
    xi = np.clip(xi, 0, wm.shape[1] - 1)
    yi = np.clip(yi, 0, wm.shape[0] - 1)
    return wm[yi, xi].astype(float, copy=False)

def load_width_baseline_widthmaps_for_image(width_baseline_img_dir: str):
    """
    Loads baseline NPZ artifacts.

    REQUIRED per method:
      - width_map
      - support_mask
      - skel

    Any missing field is a HARD ERROR.
    """
    import os
    import numpy as np
    import json

    if not width_baseline_img_dir or not os.path.isdir(width_baseline_img_dir):
        return {}

    out = {}

    for method in os.listdir(width_baseline_img_dir):
        method_dir = os.path.join(width_baseline_img_dir, method)
        if not os.path.isdir(method_dir):
            continue

        p = os.path.join(method_dir, "width_map.npz")
        if not os.path.isfile(p):
            continue

        z = np.load(p, allow_pickle=True)

        for k in ("width_map", "support_mask", "skel"):
            if k not in z:
                raise KeyError(f"[BASELINE LOAD] missing '{k}' in {p}")

        record = {
            "width_map": z["width_map"].astype(np.float32),
            "support_mask": z["support_mask"].astype(bool),
            "skel": z["skel"].astype(bool),
        }

        if "meta" in z:
            try:
                record["meta"] = json.loads(z["meta"])
            except Exception:
                record["meta"] = {}
        else:
            record["meta"] = {}

        out[method] = record

    return out

def augment_combined_with_orphan_atomics(
    *,
    combined_src: dict,
    atomic_src: dict,
):
    """
    Returns a NEW combined_cracks dict that includes:
      - all existing combined cracks
      - PLUS atomic cracks that are not members of any combined crack,
        injected as singleton-combined cracks.

    This is AUTHORITATIVE for width evaluation.
    """
    if not combined_src:
        combined_src = {}

    if not atomic_src:
        return dict(combined_src)

    combined_members = {
        str(m)
        for cmb in combined_src.values()
        for m in (cmb.get("members") or [])
    }

    print(f"[COMBINED AUGMENT] combined members = {sorted(combined_members)}")

    out = {}

    # Copy real combined cracks verbatim
    for ccid, cmb in combined_src.items():
        out[str(ccid)] = cmb

    # Inject orphan atomics
    for aid, acr in atomic_src.items():
        aid_s = str(aid)
        if aid_s in combined_members:
            continue

        mid = acr.get("midline")
        if mid is None or len(mid) < 2:
            continue

        print(f"[COMBINED AUGMENT] injecting orphan atomic {aid_s}")

        # Treat atomic midline as BOTH:
        #  - midline_segments (topology stream)
        #  - derived_midline_segments (width stream)
        # For singleton cracks, derived == midline is the correct, minimal invariant.
        seg = mid  # already Nx2-ish list/array

        seg_meta = {
            "branch_id": 0,
            "atomic_id": aid_s,
        }

        out[f"atomic_{aid_s}"] = {
            "members": [aid_s],

            "midline_segments": [seg],
            "midline_segments_meta": [dict(seg_meta)],

            # âœ… REQUIRED by your combined extractor now
            "derived_midline_segments": [seg],
            "derived_midline_segments_meta": [dict(seg_meta)],

            # width / geometry sources (pass through)
            "normal_edge_points": acr.get("normal_edge_points"),
            "geodesic_edges": acr.get("geodesic_edges"),
            "mask_bbox": acr.get("mask_bbox"),
            "mask_crop": acr.get("mask_crop"),

            # no dominance pruning for singleton
            "dominance_meta": None,

            # optional passthrough
            "timing": acr.get("timing", {}),
        }

    print(
        f"[COMBINED AUGMENT] final combined count = {len(out)} "
        f"(original={len(combined_src)}, injected={len(out) - len(combined_src)})"
    )

    return out

def project_widths_to_support(
    wmap,
    supp,
    mid_xy,
    *,
    max_nn_dist_px=6.0,
    use_support_mask=True,
    debug=False,
):
    """
    Strategy 1 (nearest-support projection):

    For each GT midline point:
        - Find nearest baseline support pixel (typically skeleton pixel)
        - Take width_map value at that support pixel
        - Reject if nearest distance > max_nn_dist_px

    This respects how baseline methods define widths:
        widths exist only at skeleton/support pixels.

    Args:
        wmap: HxW float width map
        supp: HxW uint8/bool support mask (usually skeleton)
        mid_xy: Nx2 float array of GT midline points (x, y)
        max_nn_dist_px: maximum allowed projection distance
        use_support_mask: if False, uses (wmap > 0) as support
        debug: prints coverage diagnostics

    Returns:
        (N,) float32 array of projected widths (NaN where invalid)
    """
    import numpy as np

    if wmap is None:
        return np.array([], dtype=np.float32)

    wmap = np.asarray(wmap)
    H, W = wmap.shape[:2]

    xy = np.asarray(mid_xy, float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        return np.array([], dtype=np.float32)

    finite = np.isfinite(xy).all(axis=1)
    out = np.full((len(xy),), np.nan, dtype=np.float32)
    if not np.any(finite):
        return out

    # --------------------------------------------
    # Build support mask
    # --------------------------------------------
    if use_support_mask and supp is not None:
        supp_m = np.asarray(supp).astype(bool)
    else:
        supp_m = np.isfinite(wmap) & (wmap > 0)

    ys, xs = np.nonzero(supp_m)
    if len(xs) == 0:
        if debug:
            print("[B1 PROJ] support empty")
        return out

    supp_xy = np.column_stack([
        xs.astype(np.float32),
        ys.astype(np.float32),
    ])

    # --------------------------------------------
    # Nearest neighbor projection
    # --------------------------------------------
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(supp_xy)
        d, idx = tree.query(xy[finite], k=1)
        d = np.asarray(d, float)
        idx = np.asarray(idx, int)
    except Exception:
        # fallback (slow but safe)
        d = np.full((np.sum(finite),), np.inf, float)
        idx = np.full((np.sum(finite),), -1, int)
        for i, p in enumerate(xy[finite]):
            dd = np.sum((supp_xy - p) ** 2, axis=1)
            j = int(np.argmin(dd))
            idx[i] = j
            d[i] = float(np.sqrt(dd[j]))

    ok = d <= float(max_nn_dist_px)

    if not np.any(ok):
        if debug:
            print(
                f"[B1 PROJ] 0/{len(d)} within max_nn_dist_px={max_nn_dist_px}"
            )
        return out

    nn_x = supp_xy[idx[ok], 0].astype(int)
    nn_y = supp_xy[idx[ok], 1].astype(int)
    nn_x = np.clip(nn_x, 0, W - 1)
    nn_y = np.clip(nn_y, 0, H - 1)

    vals = wmap[nn_y, nn_x].astype(np.float32)

    out_idx = np.flatnonzero(finite)[ok]
    out[out_idx] = vals

    if debug:
        print(
            f"[B1 PROJ] total={len(xy)} "
            f"finite={np.sum(finite)} "
            f"valid_proj={np.sum(ok)} "
            f"nan_out={np.sum(~np.isfinite(out))}"
        )

    return out

# baseline width comparison function (extremely simple)
def compute_projected_width_diffs(
    *,
    gt_payload,
    gt_full,
    width_baseline_maps,
    base_name,
    midline_type,
    crack_type,
    metrics_dir_local=None,
):
    """
    Regime B1: GT-conditioned width evaluation (projection-based).

    Semantics:
      - Geometry source: GT midline ONLY
      - Width source: baseline width map sampled along GT midline
      - Skeleton disagreement is NOT penalized here
    """
    import numpy as np

    if not isinstance(gt_payload, dict):
        raise TypeError(f"[B1] gt_payload must be dict, got {type(gt_payload)}")

    if crack_type == "combined":
        if "combined_cracks" not in gt_payload:
            raise KeyError("[B1] gt_payload missing 'combined_cracks'")
        cracks = gt_payload["combined_cracks"]
    else:
        if "atomic_cracks" not in gt_payload:
            raise KeyError("[B1] gt_payload missing 'atomic_cracks'")
        cracks = gt_payload["atomic_cracks"]

    if not isinstance(cracks, dict):
        raise TypeError(f"[B1] cracks must be dict, got {type(cracks)}")
    if gt_full is None:
        raise ValueError("[B1] gt_full is required")

    width_rows = []

    # -----------------------------
    # Baseline debug counters
    # -----------------------------
    dbg = {
        "cr_total": 0,
        "cr_not_dict": 0,
        "no_midline_segments": 0,
        "no_valid_midline_pts": 0,
        "no_gt_widths": 0,
        "gt_widths_too_short": 0,
        "pred_widths_too_short": 0,
        "rows_emitted": 0,
    }
    dbg_examples = {
        "no_midline_segments": [],
        "no_valid_midline_pts": [],
        "no_gt_widths": [],
        "pred_widths_too_short": [],
    }
    MAX_EX = 5

    # ------------------------------------------------------------
    # Local inline helpers (no external dependency required)
    # ------------------------------------------------------------

    def _finite_xy(arr):
        arr = np.asarray(arr, float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return np.empty((0, 2), float)
        m = np.isfinite(arr).all(axis=1)
        return arr[m]

    def _coerce_width_vector(raw):
        """
        Accepts:
          - flat list
          - packed list with None
          - nested lists
          - numpy arrays
        Returns clean 1D finite float array or None
        """
        if raw is None:
            return None

        vals = []

        def _flatten(x):
            if x is None:
                return
            if isinstance(x, (list, tuple)):
                for v in x:
                    _flatten(v)
                return
            try:
                v = float(x)
            except Exception:
                return
            if np.isfinite(v):
                vals.append(v)

        _flatten(raw)

        if len(vals) < 2:
            return None

        return np.asarray(vals, float)

    # ============================================================
    # Main evaluation
    # ============================================================

    for method, (wmap, supp) in width_baseline_maps.items():
        dbg_m = dict(dbg)
        overlay_coords = []
        overlay_diffs = []

        for cid, cr in cracks.items():
            dbg_m["cr_total"] += 1

            if not isinstance(cr, dict):
                dbg_m["cr_not_dict"] += 1
                continue

            # --------------------------------------------------
            # GT midline segments (authoritative)
            # --------------------------------------------------
            segs = cr.get("midline_segments") or []
            if not segs:
                dbg_m["no_midline_segments"] += 1
                if len(dbg_examples["no_midline_segments"]) < MAX_EX:
                    dbg_examples["no_midline_segments"].append(str(cid))
                continue

            gt_mid_parts = []
            for s in segs:
                if s is None:
                    continue
                s = _finite_xy(s)
                if len(s) >= 2:
                    gt_mid_parts.append(s)

            if not gt_mid_parts:
                dbg_m["no_valid_midline_pts"] += 1
                if len(dbg_examples["no_valid_midline_pts"]) < MAX_EX:
                    dbg_examples["no_valid_midline_pts"].append(str(cid))
                continue

            gt_mid = np.vstack(gt_mid_parts)
            if len(gt_mid) < 2:
                dbg_m["no_valid_midline_pts"] += 1
                continue

            # --------------------------------------------------
            # GT widths (robust extraction)
            # --------------------------------------------------
            gt_widths = None

            # Preferred explicit GT vector
            for k in (
                "gt_widths_auto_centered",
                "gt_widths",
                "widths",
            ):
                gt_widths = _coerce_width_vector(cr.get(k))
                if gt_widths is not None:
                    break

            # Fallback to normals dict
            if gt_widths is None:
                gtn = cr.get("gt_normals") or {}
                if isinstance(gtn, dict):
                    gt_widths = _coerce_width_vector(gtn.get("width_px"))

            if gt_widths is None:
                dbg_m["no_gt_widths"] += 1
                if len(dbg_examples["no_gt_widths"]) < MAX_EX:
                    dbg_examples["no_gt_widths"].append(str(cid))
                continue

            if gt_widths.size < 2:
                dbg_m["gt_widths_too_short"] += 1
                continue

            # --------------------------------------------------
            # Baseline widths sampled along GT geometry
            # --------------------------------------------------
            pred_widths = project_widths_to_support(
                wmap,
                supp,
                gt_mid,
                max_nn_dist_px=6.0,   # adjust 4–8 depending on density
                debug=False,
            )


            if pred_widths is None:
                dbg_m["pred_widths_too_short"] += 1
                continue

            pred_widths = np.asarray(pred_widths, float)
            if pred_widths.size < 2:
                dbg_m["pred_widths_too_short"] += 1
                if len(dbg_examples["pred_widths_too_short"]) < MAX_EX:
                    dbg_examples["pred_widths_too_short"].append(str(cid))
                continue

            # --------------------------------------------------
            # Align lengths
            # --------------------------------------------------
            n = min(len(gt_widths), len(pred_widths))
            if n < 2:
                dbg_m["pred_widths_too_short"] += 1
                continue

            gt_widths   = gt_widths[:n]
            pred_widths = pred_widths[:n]
            diff        = pred_widths - gt_widths

            # Keep method-local geometry + diffs for projected spatial overlay.
            # IMPORTANT: preserve segment boundaries to avoid visual stitching
            # lines between disjoint segments.
            off_seg = 0
            for sseg in gt_mid_parts:
                sseg = np.asarray(sseg, float)
                if sseg.ndim != 2 or len(sseg) < 2:
                    continue
                if off_seg >= n:
                    break
                mseg = min(len(sseg), n - off_seg)
                if mseg < 2:
                    off_seg += len(sseg)
                    continue
                overlay_coords.append(np.asarray(sseg[:mseg], float))
                overlay_diffs.append(np.asarray(diff[off_seg:off_seg + mseg], float))
                off_seg += len(sseg)

            for i in range(n):
                width_rows.append({
                    "image": base_name,
                    "cid": str(cid),
                    "method": method,
                    "gt_width_px": float(gt_widths[i]),
                    "pred_width_px": float(pred_widths[i]),
                    "diff_px": float(diff[i]),
                    "s_idx": int(i),
                })

            dbg_m["rows_emitted"] += int(n)

        # --------------------------------------------------
        # Baseline projected-width spatial overlay (B1)
        # --------------------------------------------------
        if dbg_m["rows_emitted"] > 0 and metrics_dir_local is not None:
            try:
                import matplotlib.pyplot as plt
                from matplotlib.colors import TwoSlopeNorm
                import numpy as np
                import os

                coords = overlay_coords
                diffs = overlay_diffs

                if not coords:
                    raise RuntimeError("No valid projected segments for overlay")

                # --------------------------------------------------
                # Compute tight bounds from geometry (+ fixed 5px pad)
                # --------------------------------------------------
                all_pts = np.vstack(coords)
                x0, y0 = np.min(all_pts, axis=0)
                x1, y1 = np.max(all_pts, axis=0)
                pad = 5.0

                x0p = int(np.floor(x0 - pad))
                y0p = int(np.floor(y0 - pad))
                x1p = int(np.ceil(x1 + pad))
                y1p = int(np.ceil(y1 + pad))

                # --------------------------------------------------
                # Clip to GT image bounds and crop GT mask
                # --------------------------------------------------
                H, W = gt_full.shape[:2]
                x0c = max(0, x0p)
                y0c = max(0, y0p)
                x1c = min(W, x1p)
                y1c = min(H, y1p)
                if x1c <= x0c or y1c <= y0c:
                    raise RuntimeError("Invalid overlay crop bounds after clipping")

                mask_crop = gt_full[y0c:y1c, x0c:x1c]

                fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
                ax.set_facecolor("white")
                ax.imshow(
                    mask_crop,
                    cmap="gray",
                    extent=[x0c, x1c, y1c, y0c],
                    interpolation="nearest",
                )
                for s in coords:
                    ax.plot(s[:, 0], s[:, 1], color="#333333", lw=1.0)

                all_d = np.concatenate(diffs)
                all_d = all_d[np.isfinite(all_d)]

                if all_d.size > 0:
                    absmax = float(np.percentile(np.abs(all_d), 95))
                    if absmax < 1e-6:
                        absmax = 1e-6
                    norm = TwoSlopeNorm(vmin=-absmax, vcenter=0.0, vmax=absmax)
                    cmap = plt.get_cmap("coolwarm")

                    for s, d in zip(coords, diffs):
                        n = min(len(s), len(d))
                        for i in range(n - 1):
                            if not np.isfinite(d[i]):
                                continue
                            ax.plot(
                                [s[i, 0], s[i + 1, 0]],
                                [s[i, 1], s[i + 1, 1]],
                                color=cmap(norm(d[i])),
                                lw=2.5,
                                solid_capstyle="round",
                            )

                    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
                    sm.set_array([])
                    cb = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
                    cb.set_label("Projected width - GT width (px)")

                ax.set_xlim(x0c, x1c)
                ax.set_ylim(y1c, y0c)
                ax.set_aspect("equal")
                ax.axis("off")
                ax.set_title(f"{method} - Baseline projected widths")

                out_dir = os.path.join(
                    metrics_dir_local,
                    midline_type or "unknown",
                    crack_type,
                )
                os.makedirs(out_dir, exist_ok=True)

                out_path = os.path.join(
                    out_dir,
                    f"{base_name}_{method}_width_baseline_projected_overlay.png",
                )

                fig.savefig(out_path, dpi=200, bbox_inches="tight")
                plt.close(fig)

                print(f"[BASELINE B1] wrote overlay: {out_path}")

            except Exception as e:
                print(f"[BASELINE B1] overlay failed: {e}")

        # --------------------------------------------------
        # Debug if empty
        # --------------------------------------------------
        if dbg_m["rows_emitted"] == 0:
            print(f"[BASELINE B1 DEBUG] method='{method}' produced 0 rows.")
            print(f"  total_cracks={dbg_m['cr_total']}")
            print(f"  not_dict={dbg_m['cr_not_dict']}")
            print(f"  no_midline_segments={dbg_m['no_midline_segments']}")
            print(f"  no_valid_midline_pts={dbg_m['no_valid_midline_pts']}")
            print(f"  no_gt_widths={dbg_m['no_gt_widths']}")
            print(f"  gt_widths_too_short={dbg_m['gt_widths_too_short']}")
            print(f"  pred_widths_too_short={dbg_m['pred_widths_too_short']}")
            if dbg_examples["no_midline_segments"]:
                print(f"  example cids (no_midline_segments): {dbg_examples['no_midline_segments']}")
            if dbg_examples["no_valid_midline_pts"]:
                print(f"  example cids (no_valid_midline_pts): {dbg_examples['no_valid_midline_pts']}")
            if dbg_examples["no_gt_widths"]:
                print(f"  example cids (no_gt_widths): {dbg_examples['no_gt_widths']}")
            if dbg_examples["pred_widths_too_short"]:
                print(f"  example cids (pred_widths_too_short): {dbg_examples['pred_widths_too_short']}")

    return width_rows

# ------------------------------------------------------------
# Part 2 plots
#   (A) TopK metrics: RMSE + MAE + Bias + finite_len_px (weight)
#   (B) Resampling explainers:
#       - worst / median / best by RMSE
#       - for each: show ALL finite runs (original + resampled)
#       - show d(s) curves per-run
#       - if predw/gtruthw available, also show predw(s), gtruthw(s) before/after
# ------------------------------------------------------------


# ======================================================================
# Part 2 PLOT HELPERS (MUST BE DEFINED BEFORE USE)
# ======================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def draw_colored_polyline(ax, pts, vals, x0, y0, lw, cmap, norm, alpha=1.0):
    if pts is None or vals is None or len(pts) < 2:
        return
    segs = [
        [(pts[i,0]-x0, pts[i,1]-y0),
        (pts[i+1,0]-x0, pts[i+1,1]-y0)]
        for i in range(len(pts)-1)
    ]
    lc = LineCollection(
        segs, array=vals, cmap=cmap, norm=norm,
        linewidths=lw, alpha=alpha,
        capstyle="round", joinstyle="round", zorder=3
    )
    ax.add_collection(lc)

def _arclen_param(xy):
    xy = np.asarray(xy, float)
    d = np.sqrt(((xy[1:] - xy[:-1])**2).sum(1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    return s

def _debug_plot_correspondence_single(
    pts,
    predw,
    gtw,
    cid,
    branch_id,
    seg_idx,
    out_dir,
    stride=15,
    gt_pts=None,
):
    def _to_pts_segments(x):
        if isinstance(x, (list, tuple)) and len(x) > 0:
            segs = []
            for a in x:
                aa = np.asarray(a, float)
                if aa.ndim == 2 and aa.shape[1] == 2 and len(aa) >= 2:
                    segs.append(aa)
            if segs:
                return segs
        aa = np.asarray(x, float)
        if aa.ndim == 2 and aa.shape[1] == 2 and len(aa) >= 2:
            return [aa]
        return []

    def _to_w_segments(x):
        if isinstance(x, (list, tuple)) and len(x) > 0:
            segs = []
            for a in x:
                aa = np.asarray(a, float).reshape(-1)
                if aa.size >= 2:
                    segs.append(aa)
            if segs:
                return segs
        aa = np.asarray(x, float).reshape(-1)
        if aa.size >= 2:
            return [aa]
        return []

    pts_segs = _to_pts_segments(pts)
    predw_segs = _to_w_segments(predw)
    gtw_segs = _to_w_segments(gtw)
    gt_pts_segs = _to_pts_segments(gt_pts) if gt_pts is not None else []

    nseg = min(len(pts_segs), len(predw_segs), len(gtw_segs))
    if nseg <= 0:
        print("[CORRESP DEBUG] skipped - no valid segments")
        return
    if gt_pts is not None and gt_pts_segs:
        nseg = min(nseg, len(gt_pts_segs))

    pts_segs = pts_segs[:nseg]
    predw_segs = predw_segs[:nseg]
    gtw_segs = gtw_segs[:nseg]
    if gt_pts_segs:
        gt_pts_segs = gt_pts_segs[:nseg]

    finite_pred = int(np.sum([np.sum(np.isfinite(w)) for w in predw_segs]))
    finite_gt = int(np.sum([np.sum(np.isfinite(w)) for w in gtw_segs]))
    total_samples = int(np.sum([min(len(a), len(b), len(c)) for a, b, c in zip(pts_segs, predw_segs, gtw_segs)]))

    print("\n==============================")
    print(f"[CORRESP DEBUG] CID={cid} branch={branch_id} seg={seg_idx}")
    print(f"Segments: {nseg}")
    print(f"Total samples: {total_samples}")
    print(f"Finite pred: {finite_pred}")
    print(f"Finite gt  : {finite_gt}")
    print("==============================\n")

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2)

    ax_geom = fig.add_subplot(gs[:, 0])
    ax_w = fig.add_subplot(gs[0, 1])
    ax_diff = fig.add_subplot(gs[1, 1])

    # Geometry plot: pred + gt + explicit correspondence links
    ax_geom.set_title("Geometry + Correspondence Links")
    first_pred = True
    first_gt = True
    step = max(1, int(stride))
    for i_seg in range(nseg):
        p = np.asarray(pts_segs[i_seg], float)
        pw = np.asarray(predw_segs[i_seg], float).reshape(-1)
        gw = np.asarray(gtw_segs[i_seg], float).reshape(-1)
        n = min(len(p), len(pw), len(gw))
        if n < 2:
            continue
        p = p[:n]
        pw = pw[:n]
        gw = gw[:n]

        ax_geom.plot(
            p[:, 0], p[:, 1],
            color="blue", lw=2,
            label="Pred midline" if first_pred else None
        )
        first_pred = False

        if gt_pts_segs:
            gxy = np.asarray(gt_pts_segs[i_seg], float)
            if gxy.ndim == 2 and gxy.shape[1] == 2 and len(gxy) >= 2:
                ng = min(n, len(gxy))
                p2 = p[:ng]
                g2 = gxy[:ng]
                gw2 = gw[:ng]
                ax_geom.plot(
                    g2[:, 0], g2[:, 1],
                    color="green", lw=1.6, alpha=0.9,
                    label="GT matched geometry" if first_gt else None
                )
                first_gt = False
                for j in range(0, ng, step):
                    if np.isfinite(gw2[j]):
                        ax_geom.plot([p2[j, 0], g2[j, 0]], [p2[j, 1], g2[j, 1]],
                                     color="red", lw=0.9, alpha=0.8)
                ax_geom.scatter(p2[::step, 0], p2[::step, 1], s=8, c="blue", alpha=0.9)
                ax_geom.scatter(g2[::step, 0], g2[::step, 1], s=8, c="green", alpha=0.9)
            else:
                for j in range(0, n, step):
                    c = "red" if np.isfinite(gw[j]) else "black"
                    ax_geom.plot(p[j, 0], p[j, 1], "o", color=c, markersize=3)
        else:
            for j in range(0, n, step):
                c = "red" if np.isfinite(gw[j]) else "black"
                ax_geom.plot(p[j, 0], p[j, 1], "o", color=c, markersize=3)

    ax_geom.set_aspect("equal")
    ax_geom.invert_yaxis()
    ax_geom.legend()

    # Width vs arclength (concatenated by segment, gapless)
    ax_w.set_title("Width vs Arc-Length")
    s_off = 0.0
    first_pw = True
    first_gw = True
    for i_seg in range(nseg):
        p = np.asarray(pts_segs[i_seg], float)
        pw = np.asarray(predw_segs[i_seg], float).reshape(-1)
        gw = np.asarray(gtw_segs[i_seg], float).reshape(-1)
        n = min(len(p), len(pw), len(gw))
        if n < 2:
            continue
        p = p[:n]
        pw = pw[:n]
        gw = gw[:n]
        s = _arclen_param(p)
        s = s + s_off
        s_off = float(s[-1])
        ax_w.plot(s, pw, color="blue", label="Pred width" if first_pw else None)
        ax_w.plot(s, gw, color="red", alpha=0.75, label="GT width" if first_gw else None)
        first_pw = False
        first_gw = False
    ax_w.legend()
    ax_w.grid(True)

    # Width difference
    ax_diff.set_title("Width Difference (Pred - GT)")
    s_off = 0.0
    for i_seg in range(nseg):
        p = np.asarray(pts_segs[i_seg], float)
        pw = np.asarray(predw_segs[i_seg], float).reshape(-1)
        gw = np.asarray(gtw_segs[i_seg], float).reshape(-1)
        n = min(len(p), len(pw), len(gw))
        if n < 2:
            continue
        p = p[:n]
        d = pw[:n] - gw[:n]
        s = _arclen_param(p)
        s = s + s_off
        s_off = float(s[-1])
        ax_diff.plot(s, d, color="purple")
    ax_diff.axhline(0, color="black", lw=1)
    ax_diff.grid(True)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f"correspondence_single_cid{cid}_b{branch_id}_s{seg_idx}.png",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[CORRESP DEBUG] wrote {out_path}")

def plot_sampling_consistency(
    *,
    pts_list,
    ptsr_list,
    mask_bin,
    crop,
    title,
    out_path,
):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    print("[SAMPLE CONSISTENCY] enter")

    # ------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------
    if not pts_list or not ptsr_list:
        print("[SAMPLE CONSISTENCY] skipped: empty pts_list or ptsr_list")
        return

    # KEEP POLYLINES SEPARATE (CRITICAL)
    pts_list  = [np.asarray(p, float) for p in pts_list  if p is not None and len(p) >= 2]
    ptsr_list = [np.asarray(p, float) for p in ptsr_list if p is not None and len(p) >= 2]

    if not pts_list or not ptsr_list:
        print("[SAMPLE CONSISTENCY] skipped: no valid polylines after filtering")
        return

    # ------------------------------------------------------------
    # Collect segment lengths ONLY for color normalization
    # ------------------------------------------------------------
    ds_all = []
    for p in pts_list:
        ds = np.linalg.norm(np.diff(p, axis=0), axis=1)
        ds = ds[np.isfinite(ds)]
        if ds.size:
            ds_all.append(ds)

    if not ds_all:
        print("[SAMPLE CONSISTENCY] skipped: no finite segment lengths")
        return

    ds_all = np.concatenate(ds_all)

    # ------------------------------------------------------------
    # Color normalization
    # ------------------------------------------------------------
    vmin = np.percentile(ds_all, 10)
    vmax = np.percentile(ds_all, 90)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.min(ds_all))
        vmax = float(np.max(ds_all))
        if vmax <= vmin:
            vmax = vmin + 1e-6

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin, vmax)

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------
    fig, (axO, axR) = plt.subplots(1, 2, figsize=(13.8, 5.2), dpi=200)

    # ------------------------------------------------------------
    # Background mask (safe crop)
    # ------------------------------------------------------------
    x0 = y0 = 0
    if mask_bin is not None and crop is not None:
        h, w = mask_bin.shape[:2]
        x0, y0, x1, y1 = crop
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)

        if x1 > x0 and y1 > y0:
            for ax in (axO, axR):
                ax.imshow(mask_bin[y0:y1, x0:x1], cmap="gray", zorder=0)

    for ax in (axO, axR):
        ax.axis("off")

    # ------------------------------------------------------------
    # Draw ORIGINAL polylines (NO cross-run connections)
    # ------------------------------------------------------------
    for p in pts_list:
        ds = np.linalg.norm(np.diff(p, axis=0), axis=1)
        ds = ds[np.isfinite(ds)]
        if ds.size < 1:
            continue

        ds_p = np.r_[ds[0], ds]  # per-point values
        draw_colored_polyline(axO, p, ds_p, x0, y0, 2, cmap, norm, 0.85)

    # ------------------------------------------------------------
    # Draw RESAMPLED polylines (NO cross-run connections)
    # ------------------------------------------------------------
    for p in ptsr_list:
        ds = np.linalg.norm(np.diff(p, axis=0), axis=1)
        ds = ds[np.isfinite(ds)]
        if ds.size < 1:
            continue

        ds_p = np.r_[ds[0], ds]
        draw_colored_polyline(axR, p, ds_p, x0, y0, 2, cmap, norm, 0.90)

    # ------------------------------------------------------------
    # Colorbar
    # ------------------------------------------------------------
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=[axO, axR], fraction=0.03, pad=0.03)
    cbar.set_label("Segment Length (px)")

    fig.text(
        0.5, 0.985, title,
        ha="center", va="top",
        fontsize=11, fontweight="bold"
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"[SAMPLE CONSISTENCY] wrote: {out_path}")


def plot_width_error_distribution(*, runs, title, out_path, bins=25):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    d_all = []
    for r in runs:
        d1 = np.asarray(r.get("d_rs", []), float)
        if len(d1) >= 2:
            d_all.append(d1[:-1])

    if not d_all:
        print(f"[PART2 WIDTH DIST] skipped (no valid samples): {out_path}")
        return

    d_all = np.concatenate(d_all)
    d_all = d_all[np.isfinite(d_all)]
    if d_all.size < 10:
        print(f"[PART2 WIDTH DIST] skipped (too few samples): {out_path}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8.8, 4.8), dpi=200)

    ax.hist(d_all, bins=bins, density=True, alpha=0.55, label="resampled")

    ax.axvline(0.0, lw=1.4, color="black", alpha=0.8)

    rmse = float(np.sqrt(np.mean(d_all ** 2)))
    mean = float(np.mean(d_all))

    ax.axvline(mean, lw=1.2, linestyle="--", alpha=0.8)

    ax.set_xlabel("Width error (pred − gt) [px]", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.grid(True, alpha=0.25)
    #ax.legend(fontsize=8)

    ax.text(
        0.02, 0.96,
        f"RMSE={rmse:.3f}px",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
    )

    fig.suptitle(title, fontsize=11, fontweight="bold")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"[PART2 WIDTH DIST] wrote: {out_path}")

    
def plot_part2_width_signals_preservation(
    *,
    run,
    title,
    out_path,
):
    """
    Plot GT width and predicted width vs arclength for a SINGLE run,
    showing ORIGINAL vs RESAMPLED signals (shape preservation).

    Expects run dict keys (from Part-2 cache):
    ORIGINAL (plot-only):
        - s
        - gruthw
        - predw

    RESAMPLED (metrics domain):
        - s_rs
        - gtruthw_rs
        - predw_rs
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # ------------------------------------------------------------
    # Pull ORIGINAL signals (optional but preferred)
    # ------------------------------------------------------------
    s0   = np.asarray(run.get("s", []), float)
    gtw0 = np.asarray(run.get("gruthw", []), float)
    pw0  = np.asarray(run.get("predw", []), float)

    have_orig = (s0.size >= 2 and gtw0.size >= 2 and pw0.size >= 2)

    # ------------------------------------------------------------
    # Pull RESAMPLED signals (required)
    # ------------------------------------------------------------
    s1   = np.asarray(run.get("s_rs", []), float)
    gtw1 = np.asarray(run.get("gtruthw_rs", []), float)
    pw1  = np.asarray(run.get("predw_rs", []), float)

    if s1.size < 2 or gtw1.size < 2 or pw1.size < 2:
        print("[PART2 SIGNAL] skipped (missing/short resampled signals)")
        return

    # ------------------------------------------------------------
    # Align lengths safely (resampled)
    # ------------------------------------------------------------
    n1 = min(len(s1), len(gtw1), len(pw1))
    s1, gtw1, pw1 = s1[:n1], gtw1[:n1], pw1[:n1]

    ok1 = np.isfinite(s1) & np.isfinite(gtw1) & np.isfinite(pw1)
    if not np.any(ok1):
        print("[PART2 SIGNAL] skipped (no finite resampled samples)")
        return

    # ------------------------------------------------------------
    # Align lengths safely (original)
    # ------------------------------------------------------------
    if have_orig:
        n0 = min(len(s0), len(gtw0), len(pw0))
        s0, gtw0, pw0 = s0[:n0], gtw0[:n0], pw0[:n0]
        ok0 = np.isfinite(s0) & np.isfinite(gtw0) & np.isfinite(pw0)
        have_orig = np.any(ok0)

    # ------------------------------------------------------------
    # Means (visual reference only)
    # ------------------------------------------------------------
    gtw1_mean = float(np.mean(gtw1[ok1]))
    pw1_mean  = float(np.mean(pw1[ok1]))

    if have_orig:
        gtw0_mean = float(np.mean(gtw0[ok0]))
        pw0_mean  = float(np.mean(pw0[ok0]))

    # ------------------------------------------------------------
    # Shared axis limits
    # ------------------------------------------------------------
    s_max = float(np.max(s1[ok1]))
    w_max = float(max(np.max(gtw1[ok1]), np.max(pw1[ok1])))

    if have_orig:
        s_max = max(s_max, float(np.max(s0[ok0])))
        w_max = max(w_max, float(np.max(gtw0[ok0])), float(np.max(pw0[ok0])))

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, (axG, axP) = plt.subplots(
        1, 2, figsize=(13.8, 5.2), dpi=200, sharex=True, sharey=True
    )

    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ---------------- GT panel ----------------
    if have_orig:
        axG.plot(
            s0[ok0], gtw0[ok0],
            lw=2.2, color="tab:blue", alpha=0.85,
            label="gt (orig)"
        )
        axG.axhline(gtw0_mean, lw=1.3, alpha=0.35, color="tab:blue")

    axG.plot(
        s1[ok1], gtw1[ok1],
        lw=2.6, ls="--", color="tab:orange",
        label="gt (resampled)"
    )
    axG.axhline(gtw1_mean, lw=1.3, alpha=0.35, ls="--", color="tab:orange")

    axG.set_title("GT width vs arclength", fontsize=12)
    axG.set_xlabel("arclength s (px)", fontsize=11)
    axG.set_ylabel("width (px)", fontsize=11)
    axG.grid(True, alpha=0.25)
    axG.legend(fontsize=10)

    # ---------------- Pred panel ----------------
    if have_orig:
        axP.plot(
            s0[ok0], pw0[ok0],
            lw=2.2, color="darkgreen", alpha=0.85,
            label="pred (orig)"
        )
        axP.axhline(pw0_mean, lw=1.3, alpha=0.35, color="darkgreen")

    axP.plot(
        s1[ok1], pw1[ok1],
        lw=2.6, ls="--", color="red",
        label="pred (resampled)"
    )
    axP.axhline(pw1_mean, lw=1.3, alpha=0.35, ls="--", color="red")

    axP.set_title("Predicted width vs arclength", fontsize=12)
    axP.set_xlabel("arclength s (px)", fontsize=11)
    axP.set_ylabel("width (px)", fontsize=11)
    axP.grid(True, alpha=0.25)
    axP.legend(fontsize=10)

    # ---------------- Limits ----------------
    axG.set_xlim(0.0, s_max * 1.02)
    axG.set_ylim(0.0, w_max * 1.05)

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"[PART2 SIGNAL] wrote: {out_path}")
        
def part2_plot_worst_and_all(
        *,
        worst_cid_runs,
        all_runs_global,
        pred_mask_worst,
        pred_mask_all,
        crop_worst,
        crop_all,
        part2_resample_dir,
        worst_cid,
    ):
    import os
    import numpy as np

    # ------------------------------------------------------------
    # WORST CID — ALL segments
    # ------------------------------------------------------------
    plot_sampling_consistency(
        pts_list=[r.get("pts") for r in worst_cid_runs if r.get("pts") is not None],
        ptsr_list=[r.get("pts_rs") for r in worst_cid_runs if r.get("pts_rs") is not None],
        mask_bin=pred_mask_worst,
        crop=crop_worst,
        title=f"Part 2 sampling consistency — WORST CID={worst_cid}",
        out_path=os.path.join(
            part2_resample_dir,
            f"part2_sampling_WORST_cid{worst_cid}.png",
        ),
    )

    plot_width_error_distribution(
        runs=worst_cid_runs,
        title=f"Part 2 width error distribution — WORST CID={worst_cid}",
        out_path=os.path.join(part2_resample_dir, f"part2_width_dist_WORST_cid{worst_cid}.png"),
    )

    # Worst representative run (longest finite length)
    worst_run = None
    bestL = -1.0
    for r in worst_cid_runs:
        L = float(r.get("run_len_px", 0.0))
        if np.isfinite(L) and L > bestL:
            bestL = L
            worst_run = r

    if worst_run is not None:
        plot_part2_width_signals_preservation(
            run=worst_run,
            title=f"Part 2 worst-run width signals — cid={worst_cid}",
            out_path=os.path.join(part2_resample_dir, f"part2_width_signals_WORST_cid{worst_cid}.png"),
        )

    # ------------------------------------------------------------
    # GLOBAL — ALL CIDs
    # ------------------------------------------------------------
    plot_sampling_consistency(
        pts_list=[r.get("pts") for r in all_runs_global if r.get("pts") is not None],
        ptsr_list=[r.get("pts_rs") for r in all_runs_global if r.get("pts_rs") is not None],
        mask_bin=pred_mask_all,
        crop=crop_all,
        title="Part 2 sampling consistency — ALL CIDs",
        out_path=os.path.join(
            part2_resample_dir,
            "part2_sampling_ALL_CIDS.png",
        ),
    )

    plot_width_error_distribution(
        runs=all_runs_global,
        title="Part 2 width error distribution — ALL CIDs",
        out_path=os.path.join(part2_resample_dir, "part2_width_dist_ALL_CIDS.png"),
    )

    # Representative GLOBAL signal (longest run across all CIDs)
    global_run = None
    bestL = -1.0
    for r in all_runs_global:
        L = float(r.get("run_len_px", 0.0))
        if np.isfinite(L) and L > bestL:
            bestL = L
            global_run = r

    if global_run is not None:
        plot_part2_width_signals_preservation(
            run=global_run,
            title="Part 2 width signals — ALL CIDs (representative)",
            out_path=os.path.join(part2_resample_dir, "part2_width_signals_ALL_CIDS.png"),
        )

def export_width_distribution_summary(
    *,
    pred_widths,
    gt_widths,
    out_dir,
    image_name,

    # ---- identity ----
    variant,                  # e.g. "medial", "auto", "manual"
    midline_type,              # REQUIRED: baseline / auto / manual
    crack_type=None,           # "atomic" | "combined"

    # ---- crack identity (NEW) ----
    cid=None,                  # atomic crack id (string or int)
    group_id=None,             # combined branch / group id (string or int)

    # ---- GT semantics ----
    gt_tier,                   # "atomic" | "combined_unfiltered" | "combined_filtered"
    gt_pairing="none",         # "none" | "manual" | "auto"
    filtered=False,            # True if mutual filtering applied

    # ---- bookkeeping ----
    method_family=None,        # optional grouping tag ("baseline" | "model")
):
    """
    Distributional width comparison (Regime A).

    Properties:
      - Geometry-agnostic
      - No alignment assumed
      - No resampling
      - No clipping
      - Descriptive statistics ONLY

    Identifiers:
      - cid:      atomic crack id (diagnostic)
      - group_id: combined branch / group id (diagnostic)

    Output:
      Appends ONE row to:
        <out_dir>/width_distribution_summary.csv
    """

    import os
    import numpy as np
    import pandas as pd
    from scipy.stats import wasserstein_distance, ks_2samp

    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "width_distribution_summary.csv")

    # -------------------------------------------------
    # Sanitize inputs
    # -------------------------------------------------
    pw = np.asarray(pred_widths, float).reshape(-1)
    gw = np.asarray(gt_widths, float).reshape(-1)

    mask = (
        np.isfinite(pw) &
        np.isfinite(gw) &
        (pw > 0) &
        (gw > 0)
    )

    pw = pw[mask]
    gw = gw[mask]

    if pw.size < 5 or gw.size < 5:
        print("[DIST] skipped: insufficient valid samples")
        return

    # -------------------------------------------------
    # Distributional differences (NOT error)
    # -------------------------------------------------
    median_diff = float(np.median(pw) - np.median(gw))
    mean_diff   = float(np.mean(pw)   - np.mean(gw))

    iqr_pred = float(np.percentile(pw, 75) - np.percentile(pw, 25))
    iqr_gt   = float(np.percentile(gw, 75) - np.percentile(gw, 25))
    iqr_ratio = float(iqr_pred / (iqr_gt + 1e-12))

    std_ratio = float(np.std(pw) / (np.std(gw) + 1e-12))

    try:
        w_dist = float(wasserstein_distance(pw, gw))
    except Exception:
        w_dist = np.nan

    try:
        ks_stat = float(ks_2samp(pw, gw).statistic)
    except Exception:
        ks_stat = np.nan

    # -------------------------------------------------
    # Row
    # -------------------------------------------------
    row = {
        # ---- identifiers ----
        "image": image_name,
        "variant": str(variant),
        "midline_type": str(midline_type),
        "crack_type": crack_type,

        # ---- NEW diagnostic identifiers ----
        "cid": None if cid is None else str(cid),
        "member_id": None if group_id is None else str(group_id),

        "method_family": method_family,

        # ---- GT semantics ----
        "gt_tier": gt_tier,
        "gt_pairing": gt_pairing,
        "filtered": bool(filtered),

        # ---- sample counts ----
        "n_samples": int(min(pw.size, gw.size)),

        # ---- GT distribution ----
        "gt_mean": float(np.mean(gw)),
        "gt_median": float(np.median(gw)),
        "gt_iqr": float(iqr_gt),
        "gt_std": float(np.std(gw)),

        # ---- Pred distribution ----
        "pred_mean": float(np.mean(pw)),
        "pred_median": float(np.median(pw)),
        "pred_iqr": float(iqr_pred),
        "pred_std": float(np.std(pw)),

        # ---- Distributional divergence ----
        "median_diff": median_diff,
        "mean_diff": mean_diff,
        "iqr_ratio": iqr_ratio,
        "std_ratio": std_ratio,
        "wasserstein_dist": w_dist,
        "ks_stat": ks_stat,
    }

    df = pd.DataFrame([row])

    # -------------------------------------------------
    # Append safely
    # -------------------------------------------------
    if os.path.exists(out_csv):
        df.to_csv(out_csv, mode="a", header=False, index=False)
    else:
        df.to_csv(out_csv, index=False)

    print(
        f"[DIST] wrote distribution row | "
        f"variant={variant} | gt={gt_tier} | "
        f"cid={cid} | group={group_id} | "
        f"n={row['n_samples']}"
    )
    
def _atomic_pred_matches_combined_gt(cid, gt_sup):
    """
    Returns True if this atomic prediction ID appears inside ANY combined GT crack.
    gt_sup keys for combined are frozensets of member IDs.
    """
    if not gt_sup:
        return False
    cid_s = str(cid)
    for k in gt_sup.keys():
        if isinstance(k, frozenset) and cid_s in k:
            return True
    return False

import base64
import numpy as np


def _decode_packbits_mask(blob):
    """
    Decode a packed mask blob:
      {
        "shape": [H, W],
        "packbits_b64": "..."
      }

    Returns:
        np.ndarray bool (H, W) or None
    """
    if not isinstance(blob, dict):
        return None

    shape = blob.get("shape")
    b64 = blob.get("packbits_b64")

    if (
        not shape
        or len(shape) != 2
        or not b64
        or shape[0] <= 0
        or shape[1] <= 0
    ):
        return None

    H, W = map(int, shape)

    raw = base64.b64decode(b64.encode("ascii"))
    packed = np.frombuffer(raw, dtype=np.uint8)

    row_bytes = (W + 7) // 8
    expected = H * row_bytes
    if packed.size != expected:
        print(
            f"[PACKBITS ERROR] size mismatch: "
            f"got={packed.size}, expected={expected}"
        )
        return None

    packed = packed.reshape(H, row_bytes)
    mask = np.unpackbits(packed, axis=1)[:, :W]

    return mask.astype(bool)

def compare_widths_for_aligned_cracks(
    ann,
    crack_mask,
    base_name,
    metrics_dir,
    display=False,
    midline_type=None,
    crack_type=None,
    return_normals=False,
    normals_plot=False,
    normals_dir=None,
    max_radius=50,
    gt_sup_root=None,
    *,
    variant_id="main",
    write_part2_tables=True,
):
    """
    WIDTH COMPARISON — ALIGNED CRACKS ONLY (ATOMIC OR COMBINED)

    Contract:
      - This function MUST be called with exactly one of:
          { "atomic_cracks": {...} }  OR
          { "combined_cracks": {...} }

    Scope:
      - This function assumes **geometrically alignable cracks**
      - Used for:
          * GT vs MANUAL
          * GT vs AUTO
          * MANUAL vs AUTO (after opsec pruning)
      - NOT used for baseline evaluation

    Guarantees:
      - Segment-safe (no flattening)
      - Geodesic fallback if normals missing
      - Zoom uses ONLY union of provided mask_bbox values
      - Solid blue bbox overlay
      - TwoSlopeNorm always monotonic (0 included)

    Combined-specific behavior:
      - Stage 0: match combined crack to GT supervision by member overlap
      - Stage 1: prune segments to shared atomic IDs
      - Stage 2: optional branch matching
      - Stage 3: symmetric bite-union clipping
      - GT widths computed along FINAL clipped polyline
    """

    import os, json
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    #from helpers.metrics import normals_from_mask_for_midline

    os.makedirs(metrics_dir, exist_ok=True)

    # ---------------- variant tag (output isolation only) ----------------
    variant_id = str(variant_id or "main").strip()
    file_tag = "main" if variant_id in ("", "main") else variant_id

    H, W = crack_mask.shape
    mask_bin = (crack_mask > 0).astype(np.uint8)

    atomic   = ann.get("atomic_cracks")
    combined = ann.get("combined_cracks")

    print(
        f"[WIDTH DEBUG] ann keys: "
        f"atomic={None if atomic is None else list(atomic.keys())[:5]}, "
        f"combined={None if combined is None else list(combined.keys())[:5]}"
    )

    # ---------------- enforce SINGLE MODE ----------------
    if (atomic is None) == (combined is None):
        raise RuntimeError(
            "compare_widths_for_aligned_cracks expects EXACTLY ONE of "
            "'atomic_cracks' or 'combined_cracks'"
        )

    mode = "atomic" if atomic is not None else "combined"
    cracks = atomic if mode == "atomic" else combined

    print(f"\n[WIDTH DEBUG] === RUN MODE: {mode.upper()} ===")

    # ---------------- load GT supervision (ATOMIC + COMBINED) ----------------
    gt_sup_combined = {}   # frozenset(members) -> entry
    gt_sup_atomic = {}     # str(id) -> entry

    if gt_sup_root:
        p = os.path.join(gt_sup_root, "gt_supervision.json")
        if os.path.exists(p):
            with open(p, "r") as f:
                data = json.load(f)

            for e in data.get("cracks", []):
                kind = str(e.get("kind") or "").lower()
                if kind == "combined":
                    members = e.get("members") or []
                    if members:
                        key = frozenset(map(str, members))
                        gt_sup_combined[key] = e
                elif kind == "atomic":
                    aid = e.get("id")
                    if aid is not None:
                        gt_sup_atomic[str(aid)] = e

    print(f"[GT SUP] loaded {len(gt_sup_combined)} combined GT entries")
    print(f"[GT SUP] loaded {len(gt_sup_atomic)} atomic GT entries")
    if gt_sup_atomic:
        print(f"[GT SUP] atomic ids sample: {list(gt_sup_atomic.keys())[:10]}")
    else:
        print("[GT SUP] WARNING: no atomic GT entries found. Atomic width eval will skip.")

    ORIENT_DEBUG = True
    from helpers.geometry_canonical import assert_direction_consistency as _assert_direction_consistency

    def _orient_cost(A, B):
        A = np.asarray(A, float)
        B = np.asarray(B, float)
        if A.ndim != 2 or B.ndim != 2 or len(A) < 2 or len(B) < 2:
            return np.nan, np.nan, "invalid"
        a0, a1 = A[0], A[-1]
        b0, b1 = B[0], B[-1]
        d_fwd = float(np.linalg.norm(a0 - b0) + np.linalg.norm(a1 - b1))
        d_rev = float(np.linalg.norm(a0 - b1) + np.linalg.norm(a1 - b0))
        return d_fwd, d_rev, ("reversed_candidate" if d_rev < d_fwd else "forward_candidate")

    def _log_branch_orientation(segs, meta, *, tag, cid_dbg):
        if not ORIENT_DEBUG:
            return
        if not segs:
            return

        by_branch = {}
        for i, (S, m) in enumerate(zip(segs, meta)):
            if S is None or len(S) < 2:
                continue
            mm = m if isinstance(m, dict) else {}
            bid = _safe_int(mm.get("branch_id"), -1)
            sid = _safe_int(mm.get("seg_idx"), i)
            by_branch.setdefault(int(bid), []).append((int(sid), i, np.asarray(S, float), mm))

        for bid, items in by_branch.items():
            items = sorted(items, key=lambda t: (t[0], t[1]))
            if len(items) < 2:
                continue
            for j in range(1, len(items)):
                sid0, i0, s0, _ = items[j - 1]
                sid1, i1, s1, _ = items[j]
                d_fwd, d_rev, flag = _orient_cost(s0, s1)
                print(
                    f"[ORIENT DBG] cid={cid_dbg} tag={tag} branch={bid} "
                    f"prev(seg_idx={sid0},i={i0},n={len(s0)}) -> "
                    f"curr(seg_idx={sid1},i={i1},n={len(s1)}) "
                    f"d_fwd={d_fwd:.4f} d_rev={d_rev:.4f} flag={flag}"
                )

        # Duplicate metadata keys are a common source of ambiguous matching.
        key_counts = {}
        for i, m in enumerate(meta):
            mm = m if isinstance(m, dict) else {}
            k = (str(mm.get("atomic_id", "None")), _safe_int(mm.get("branch_id"), -1), _safe_int(mm.get("seg_idx"), -1))
            key_counts[k] = key_counts.get(k, 0) + 1
        dup = [(k, c) for (k, c) in key_counts.items() if c > 1]
        for k, c in dup[:20]:
            print(f"[ORIENT DBG] cid={cid_dbg} tag={tag} duplicate_key={k} count={c}")

    def _check_branch_direction_consistency(segs, meta, *, tag, cid_dbg):
        if not ORIENT_DEBUG:
            return
        by_branch = {}
        for i, (S, m) in enumerate(zip(segs or [], meta or [])):
            if S is None or len(S) < 2:
                continue
            mm = m if isinstance(m, dict) else {}
            bid = _safe_int(mm.get("branch_id"), -1)
            sid = _safe_int(mm.get("seg_idx"), i)
            by_branch.setdefault(int(bid), []).append((int(sid), i, np.asarray(S, float)))

        for bid, items in by_branch.items():
            items = sorted(items, key=lambda t: (t[0], t[1]))
            if len(items) < 2:
                continue
            segs_b = [it[2] for it in items]
            try:
                _assert_direction_consistency(segs_b)
                print(f"[ORIENT CHECK] cid={cid_dbg} tag={tag} branch={bid} status=PASS")
            except Exception as e:
                print(f"[ORIENT CHECK] cid={cid_dbg} tag={tag} branch={bid} status=FAIL err={e}")


    def _extract_segments_and_meta(crack, cid_dbg=None):
        """
        Returns:
            mid_segs
            mid_meta
            derived_segs
            derived_meta
            bite_obj
            members_set
        """

        if mode == "atomic":
            mid_segs = _split_on_nans(crack.get("midline", []))
            if not mid_segs:
                raise ValueError("Atomic missing midline")

            derived = crack.get("derived_midline")
            if not isinstance(derived, list) or not derived:
                raise ValueError("Atomic missing derived_midline")

            derived_segs = _split_on_nans(derived)
            if not derived_segs:
                raise ValueError("Atomic derived_midline empty after split")

            # Atomic has no multi-branch dominance
            mid_meta = [{"branch_id": 0, "atomic_id": str(crack.get("id"))} for _ in mid_segs]
            derived_meta = [{"branch_id": 0, "atomic_id": str(crack.get("id"))} for _ in derived_segs]

            return (
                mid_segs,
                mid_meta,
                derived_segs,
                derived_meta,
                None,
                {str(crack.get("id"))},
            )

        # ===============================
        # COMBINED
        # ===============================

        # --- Midline geometry ---
        mid_segs = [np.asarray(s, float) for s in (crack.get("midline_segments") or [])]
        if not mid_segs:
            raise ValueError("Combined missing midline_segments")

        mid_meta = crack.get("midline_segments_meta") or crack.get("segments_meta") or []
        if not isinstance(mid_meta, list):
            mid_meta = []
        if len(mid_meta) != len(mid_segs):
            tmp = []
            for i in range(len(mid_segs)):
                d = mid_meta[i] if i < len(mid_meta) and isinstance(mid_meta[i], dict) else {}
                tmp.append(d)
            mid_meta = tmp
        for i in range(len(mid_meta)):
            if not isinstance(mid_meta[i], dict):
                mid_meta[i] = {}
            mid_meta[i].setdefault("branch_id", int(_safe_int(mid_meta[i].get("branch_id"), i)))
            mid_meta[i].setdefault("seg_idx", int(i))

        # --- Derived geometry (explicit per-segment representation only) ---
        derived_segs_raw = crack.get("derived_midline_segments")
        derived_meta = crack.get("derived_midline_segments_meta")
        if not isinstance(derived_segs_raw, list) or not derived_segs_raw:
            raise ValueError("Combined missing derived_midline_segments")
        if not isinstance(derived_meta, list) or not derived_meta:
            raise ValueError("Combined missing derived_midline_segments_meta")

        derived_segs = [
            np.asarray(s, float)
            for s in derived_segs_raw
            if s is not None and len(s) >= 2
        ]
        if len(derived_segs) != len(derived_meta):
            raise ValueError(
                f"derived segments mismatch derived_midline_segments_meta: "
                f"{len(derived_segs)} segs vs {len(derived_meta)} meta"
            )

        for i in range(len(derived_meta)):
            if not isinstance(derived_meta[i], dict):
                derived_meta[i] = {}
            derived_meta[i].setdefault("branch_id", int(_safe_int(derived_meta[i].get("branch_id"), i)))
            derived_meta[i].setdefault("seg_idx", int(i))

        _log_branch_orientation(mid_segs, mid_meta, tag="combined_mid_extract", cid_dbg=cid_dbg)
        _log_branch_orientation(derived_segs, derived_meta, tag="combined_derived_extract", cid_dbg=cid_dbg)
        _check_branch_direction_consistency(mid_segs, mid_meta, tag="combined_mid_extract", cid_dbg=cid_dbg)
        _check_branch_direction_consistency(derived_segs, derived_meta, tag="combined_derived_extract", cid_dbg=cid_dbg)

        # --- Dominance bite stays dict-shaped ---
        bite_obj = None
        dom = crack.get("dominance_meta") or crack.get("dominance") or crack.get("dominance_info") or {}
        if isinstance(dom, dict) and "bite" in dom and isinstance(dom["bite"], dict):
            bite_obj = dom["bite"]
        else:
            b = crack.get("bite")
            if isinstance(b, dict) and "bbox" in b:
                bite_obj = b

        members = crack.get("members") or []
        members_set = set(map(str, members))

        return mid_segs, mid_meta, derived_segs, derived_meta, bite_obj, members_set

    # ---------------- accumulators ----------------
    coords, diffs, bboxes = [], [], []
    rows = []
    midline_metric_rows = []   # for combined midline diagnostics

    width_pairs = []
    width_metric_rows = []

    def _should_midline_metrics(*, run_mode, run_midline_type, geometry_type):
        """
        Gate which midline metric variants are emitted.
        geometry_type is expected to be "orig" or "derived".
        """
        g = str(geometry_type).lower()
        m = str(run_mode).lower()
        t = str(run_midline_type).lower()

        if t == "manual" and g == "orig":
            return False
        if m == "atomic" and g == "orig":
            return False
        return True

    # debug dir for opsec artifacts
    opsec_dir = os.path.join(metrics_dir, midline_type or "unknown", "opsec_debug")
    os.makedirs(opsec_dir, exist_ok=True)

    # Debug-only forensic trace for Stage 4.5 -> Stage 5 provenance.
    DEBUG_TOPOLOGY_TRACE = True
    topo_dbg_dir = os.path.join(opsec_dir, "topology_trace")
    print(topo_dbg_dir)
    if DEBUG_TOPOLOGY_TRACE:
        os.makedirs(topo_dbg_dir, exist_ok=True)

    # ---------------- local: predicted width trace extraction ----------------
    def _get_pred_width_full(crack_obj, midline_concat_pts, widths_geo_fallback):
        """
        Returns a per-point width vector aligned to CONCATENATED midline points.
        Preference order:
          1) crack_obj["width_px"] / ["midline_width_px"] / ["pred_width_px"]
          2) edge distance (widths_geo_fallback)
          3) sample width_map / width_map_crop at points
        """
        for k in ("pred_width_px", "midline_width_px", "width_px", "widths_px"):
            w = crack_obj.get(k, None)
            if w is not None:
                w = np.asarray(w, float).reshape(-1)
                if w.size >= 2:
                    return w

        if widths_geo_fallback is not None:
            w = np.asarray(widths_geo_fallback, float).reshape(-1)
            if w.size >= 2:
                return w

        wm = crack_obj.get("width_map", None)
        if wm is not None:
            return _sample_width_map_at_pts(wm, midline_concat_pts, bbox=None)

        wm_crop = crack_obj.get("width_map_crop", None)
        bb = crack_obj.get("mask_bbox", None)
        if wm_crop is not None and bb is not None:
            return _sample_width_map_at_pts(wm_crop, midline_concat_pts, bbox=bb)

        return None

    def _flatten_numeric_1d(x, out):
        if x is None:
            return
        if isinstance(x, np.ndarray):
            if x.ndim == 0:
                try:
                    v = float(x.item())
                    if np.isfinite(v):
                        out.append(v)
                except Exception:
                    pass
                return
            for vv in x.reshape(-1):
                _flatten_numeric_1d(vv, out)
            return
        if isinstance(x, (list, tuple)):
            for vv in x:
                _flatten_numeric_1d(vv, out)
            return
        try:
            v = float(x)
        except Exception:
            return
        if np.isfinite(v):
            out.append(v)

    def _coerce_gt_width_vec(raw):
        vals = []
        _flatten_numeric_1d(raw, vals)
        if len(vals) < 2:
            return None
        return np.asarray(vals, float).reshape(-1)

    def _get_gt_width_full(crack_obj, gt_entry_obj=None):
        """
        Returns GT width vector from already-loaded payloads/supervision.
        No mask-based recomputation here.
        """
        for src in (crack_obj, gt_entry_obj):
            if not isinstance(src, dict):
                continue

            for k in ("gt_widths", "gt_widths_auto_centered", "gt_width_px", "gt_width_px_auto_centered", "gruthw"):
                w = _coerce_gt_width_vec(src.get(k, None))
                if w is not None:
                    return w

            gtn = src.get("gt_normals", None)
            if isinstance(gtn, dict):
                w = _coerce_gt_width_vec(gtn.get("width_px", None))
                if w is not None:
                    return w

        return None

    def _lookup_atomic_gt_entry(cid_val):
        cid_s = str(cid_val)
        cands = [cid_s]
        if cid_s.startswith("atomic_"):
            cands.append(cid_s.split("atomic_", 1)[1])
        for c in cands:
            if c in gt_sup_atomic:
                return gt_sup_atomic[c]
        return None

    def _seg_endpoints(S):
        S = np.asarray(S, float)
        if S.ndim != 2 or len(S) < 2:
            return None
        return S[0].astype(float), S[-1].astype(float)

    def _endpoint_pair_key(S, snap=5.0):
        """
        Order-invariant snapped endpoint key.
        """
        ep = _seg_endpoints(S)
        if ep is None:
            return None
        a, b = ep
        a = tuple((np.round(a / snap) * snap).tolist())
        b = tuple((np.round(b / snap) * snap).tolist())
        return tuple(sorted([a, b]))

    def _poly_centroid(S):
        return np.nanmean(np.asarray(S, float), axis=0)

    def _poly_length(S):
        try:
            return float(_linestring_length(np.asarray(S, float)))
        except Exception:
            S = np.asarray(S, float)
            if len(S) < 2:
                return 0.0
            d = np.diff(S, axis=0)
            return float(np.nansum(np.sqrt(np.sum(d * d, axis=1))))

    def _build_branch_table_geom(segs_in, meta_in, *, scope_members=None):
        groups = {}
        for S, m in zip(segs_in, meta_in):
            if S is None or len(S) < 2:
                continue
            m = m if isinstance(m, dict) else {}

            if scope_members is not None:
                aid = m.get("atomic_id", None)
                if aid is not None and str(aid) not in scope_members:
                    continue

            k = _endpoint_pair_key(S, snap=5.0)
            if k is None:
                continue
            groups.setdefault(k, []).append((np.asarray(S, float), dict(m)))

        out = []
        for bi, (_, items) in enumerate(groups.items()):
            segs_k = [it[0] for it in items]
            meta_k = [it[1] for it in items]

            lens = [_poly_length(S) for S in segs_k]
            j = int(np.argmax(lens)) if lens else 0
            rep = segs_k[j]
            ep = _seg_endpoints(rep)
            if ep is None:
                continue
            a, b = ep

            all_pts = np.vstack(segs_k) if segs_k else rep
            c = _poly_centroid(all_pts)
            L = float(np.sum([_poly_length(S) for S in segs_k]))

            aids = set()
            for mm in meta_k:
                aid = mm.get("atomic_id", None)
                if aid is not None:
                    aids.add(str(aid))

            out.append(
                {
                    "branch_id": int(bi),
                    "segs": segs_k,
                    "meta": meta_k,
                    "endpoints": (np.asarray(a, float), np.asarray(b, float)),
                    "centroid": np.asarray(c, float),
                    "length": float(L),
                    "atomic_ids": aids,
                }
            )

        return out

    def _branch_geom_cost(gtb, prb):
        ga, gb = gtb["endpoints"]
        pa, pb = prb["endpoints"]

        d1 = float(np.linalg.norm(ga - pa) + np.linalg.norm(gb - pb))
        d2 = float(np.linalg.norm(ga - pb) + np.linalg.norm(gb - pa))
        d_end = min(d1, d2)

        d_cent = float(np.linalg.norm(gtb["centroid"] - prb["centroid"]))

        Lg = max(1e-6, float(gtb["length"]))
        Lp = max(1e-6, float(prb["length"]))
        ratio = max(Lg / Lp, Lp / Lg)
        d_len = float((ratio - 1.0) * 50.0)

        return d_end + 0.25 * d_cent + d_len

    def _greedy_match_branches_geom(gt_br, pr_br, *, max_cost=250.0):
        pairs = []
        for gi, g in enumerate(gt_br):
            for pi, p in enumerate(pr_br):
                c = _branch_geom_cost(g, p)
                pairs.append((c, gi, pi))
        pairs.sort(key=lambda t: t[0])

        used_g = set()
        used_p = set()
        matches = []
        for c, gi, pi in pairs:
            if c > max_cost:
                break
            if gi in used_g or pi in used_p:
                continue
            used_g.add(gi)
            used_p.add(pi)
            matches.append((gt_br[gi]["branch_id"], pr_br[pi]["branch_id"], float(c)))
        return matches

    def _assign_synth_branch_ids(segs_in, meta_in, scope_members=None):
        br = _build_branch_table_geom(segs_in, meta_in, scope_members=scope_members)
        seg_to_bid = {}

        for b in br:
            bid = int(b["branch_id"])
            for S in b["segs"]:
                S = np.asarray(S, float)
                if len(S) < 2:
                    continue
                k = (
                    tuple(np.round(S[0], 3)),
                    tuple(np.round(S[-1], 3)),
                    int(len(S)),
                )
                seg_to_bid[k] = bid
        return seg_to_bid

    def _extract_gt_stream_segments_and_meta(gt_entry_obj, geom_name):
        if not isinstance(gt_entry_obj, dict):
            return [], []

        # ------------------------------------------
        # GT has NO derived geometry.
        # For geom_name == "derived", reuse midline.
        # ------------------------------------------

        segs = [np.asarray(s, float) for s in (gt_entry_obj.get("midline_segments") or [])]

        meta = (
            gt_entry_obj.get("midline_segments_meta")
            or gt_entry_obj.get("segments_meta")
            or ((gt_entry_obj.get("dominance_meta") or {}).get("segments_meta") or [])
        )

        if not isinstance(meta, list):
            meta = []

        # Ensure 1:1 seg/meta alignment
        if len(meta) != len(segs):
            tmp = []
            for i in range(len(segs)):
                d = meta[i] if i < len(meta) and isinstance(meta[i], dict) else {}
                tmp.append(d)
            meta = tmp

        for i in range(len(meta)):
            if not isinstance(meta[i], dict):
                meta[i] = {}
            meta[i].setdefault("branch_id", int(i))

        return segs, [dict(m) for m in meta]


    def _inflate_local_to_full(bbox_xywh, m_local, H_full, W_full):
        if bbox_xywh is None or m_local is None:
            return None

        bx, by, _, _ = map(int, bbox_xywh)
        m = np.asarray(m_local).astype(bool)
        if m.ndim != 2 or m.size == 0:
            return None

        full = np.zeros((H_full, W_full), bool)
        mh, mw = m.shape
        x0, y0 = max(0, bx), max(0, by)
        x1, y1 = min(W_full, bx + mw), min(H_full, by + mh)
        if x1 <= x0 or y1 <= y0:
            return full

        sx0, sy0 = max(0, -bx), max(0, -by)
        sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)
        full[y0:y1, x0:x1] = m[sy0:sy1, sx0:sx1]
        return full

    def _decode_bite_loss_masks_full(dom, H_full, W_full):
        if not isinstance(dom, dict):
            return {}
        bite = dom.get("bite")
        if not isinstance(bite, dict):
            return {}
        bb = bite.get("bbox")
        by_branch = bite.get("by_losing_branch")
        if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
            return {}

        out = {}
        for bid, info in (by_branch or {}).items():
            m_local = _decode_packbits_mask(info)
            if m_local is None:
                continue
            m_full = _inflate_local_to_full(bb, m_local, H_full, W_full)
            if m_full is None:
                continue
            try:
                out[int(bid)] = m_full.astype(bool)
            except Exception:
                continue
        return out

    def _apply_union_dominance(
        segs_in,
        meta_in,
        *,
        loss_masks_pred_by_branch,
        loss_masks_gt_by_branch,
        H_full,
        W_full,
    ):
        def _union_bite_for_branch(bid):
            if bid is None:
                return None
            try:
                bid_i = int(bid)
            except Exception:
                return None
            mg = loss_masks_gt_by_branch.get(bid_i)
            mp = loss_masks_pred_by_branch.get(bid_i)
            if mg is None and mp is None:
                return None
            if mg is None:
                return mp
            if mp is None:
                return mg
            return (mg | mp)

        kept_segs = []
        kept_meta = []
        removed_segs = []
        for S, m in zip(segs_in or [], meta_in or []):
            if S is None or len(S) < 2:
                continue
            mm = m if isinstance(m, dict) else {}
            bid = _safe_int(mm.get("branch_id"), None)
            rm = _union_bite_for_branch(bid)
            if rm is None:
                kept_segs.append(np.asarray(S, float))
                kept_meta.append(dict(mm))
                continue

            runs_keep, runs_removed = _clip_polyline_into_runs(
                S, rm, H_full, W_full, min_pts=2
            )
            removed_segs.extend(runs_removed)
            for k in runs_keep:
                kept_segs.append(np.asarray(k, float))
                kept_meta.append(dict(mm))

        return kept_segs, kept_meta, removed_segs

    def _k_ab(m):
        m = m if isinstance(m, dict) else {}
        aid = m.get("atomic_id", None)
        bid = _safe_int(m.get("branch_id"), -1)
        return (str(aid) if aid is not None else None, int(bid))

    def _match_midline_to_derived(mid_keep_segs, mid_keep_meta, dsegs_in, dmeta_in, *, cid):
        """
        Enforce 1:1 mapping from kept MIDLINE segments to DERIVED segments by (atomic_id, branch_id).
        """
        if len(dsegs_in) != len(dmeta_in):
            raise RuntimeError(
                f"[FATAL] derived seg/meta mismatch in cid={cid}: {len(dsegs_in)} vs {len(dmeta_in)}"
            )

        buckets = {}
        for S, m in zip(dsegs_in, dmeta_in):
            if S is None or len(S) < 2:
                continue
            k = _k_ab(m)
            buckets.setdefault(k, []).append((np.asarray(S, float), dict(m if isinstance(m, dict) else {})))

        out_segs, out_meta = [], []
        for _Sm, mm in zip(mid_keep_segs, mid_keep_meta):
            k = _k_ab(mm)
            if k not in buckets or not buckets[k]:
                raise RuntimeError(f"[FATAL] Missing derived segment for key={k} in cid={cid}")
            Sd, md = buckets[k].pop(0)
            out_segs.append(Sd)
            out_meta.append(md)

        return out_segs, out_meta

    # ---------------- iterate cracks (NO width_baseline_mode; baseline is injected via pred_widths) ----------------
    crack_iter = list(cracks.items())
    print(f"cracks iterating through: {len(crack_iter)}")

    for cid, crack in crack_iter:
        print(f"\n[WIDTH DEBUG] {mode.upper()} cid={cid}")

        mid_segs, mid_meta, derived_segs, derived_meta, bite_pred, pred_members = _extract_segments_and_meta(crack, cid_dbg=cid)
        if not derived_segs:
            raise RuntimeError(f"Missing derived geometry (no fallback allowed) for cid={cid}")

        segs = list(mid_segs or [])
        seg_meta = list(mid_meta or [])
        dsegs = list(derived_segs or [])
        dmeta = list(derived_meta or [])

        if mode == "atomic" or str(cid).startswith("atomic_"):
            segs = dsegs
            seg_meta = dmeta

        e1, e2 = _get_edges(crack)
        if len(e1) < 2 or len(e2) < 2:
            continue
        m_edge = min(len(e1), len(e2))
        widths_geo = np.linalg.norm(e1[:m_edge] - e2[:m_edge], axis=1)

        derived_concat = (
            np.vstack([np.asarray(s, float) for s in dsegs if s is not None and len(s) >= 2])
            if dsegs else None
        )

        if isinstance(crack, dict) and "pred_widths" in crack:
            predw_full_any = np.asarray(crack["pred_widths"], float).reshape(-1)
        else:
            predw_full_any = _get_pred_width_full(crack, derived_concat, widths_geo)


        ##############################################
        # Part 1 (atomic): push width_pairs only
        #   - pred widths come from precomputed widths / edges / width-map
        #   - GT widths MUST come from loaded payload (no mask-based recompute here)
        ##############################################
        if mode == "atomic" or cid.startswith("atomic_"):

            # --- detect GT semantic mismatch (atomic pred vs combined GT) ---
            atomic_vs_combined_gt = _atomic_pred_matches_combined_gt(cid, gt_sup_combined)

            if predw_full_any is None or predw_full_any.size < 2:
                print(f"[WIDTH DEBUG] atomic cid={cid} has no usable pred width trace -> skip")
                continue

            predw_full_any = np.asarray(predw_full_any, float).reshape(-1)
            gt_entry_atomic = _lookup_atomic_gt_entry(cid)
            gtw_full_any = _get_gt_width_full(crack, gt_entry_atomic)
            if gtw_full_any is None or gtw_full_any.size < 2:
                print(f"[WIDTH DEBUG] atomic cid={cid} has no usable GT width trace (payload + supervision) -> skip")
                if gt_entry_atomic is None:
                    print(f"[WIDTH DEBUG] atomic cid={cid} missing from gt_sup_atomic")
                else:
                    print(f"[WIDTH DEBUG] atomic cid={cid} gt_sup keys: {list(gt_entry_atomic.keys())}")
                print(f"[WIDTH DEBUG] atomic cid={cid} has keys: {list(crack.keys())[:20]}")
                continue
            gtw_full_any = np.asarray(gtw_full_any, float).reshape(-1)

            def _resample_1d_to_len_atomic(arr, L):
                arr = np.asarray(arr, float).reshape(-1)
                L = int(L)
                if L <= 0:
                    return np.asarray([], float)
                if arr.size == 0:
                    out = np.empty((L,), float)
                    out[:] = np.nan
                    return out
                if arr.size == 1:
                    out = np.empty((L,), float)
                    out[:] = float(arr[0])
                    return out
                if arr.size == L:
                    return arr.astype(float, copy=False)
                x_old = np.linspace(0.0, 1.0, num=arr.size)
                x_new = np.linspace(0.0, 1.0, num=L)
                return np.interp(x_new, x_old, arr).astype(float, copy=False)

            total_geom = int(sum(len(s) for s in segs if s is not None and len(s) >= 2))
            if total_geom < 2:
                print(f"[WIDTH DEBUG] atomic cid={cid} has <2 derived geometry samples -> skip")
                continue

            predw_full_aligned = _resample_1d_to_len_atomic(predw_full_any, total_geom)
            gtw_full_aligned = _resample_1d_to_len_atomic(gtw_full_any, total_geom)

            print(
                f"[WIDTH DEBUG] atomic cid={cid} aligned streams: "
                f"geom={total_geom} pred_raw={len(predw_full_any)} gt_raw={len(gtw_full_any)}"
            )

            off = 0
            for s in segs:
                if s is None or len(s) < 2:
                    continue

                pts = np.asarray(s, float)
                L = int(len(pts))
                if L < 2:
                    off += L
                    continue

                predw = predw_full_aligned[off:off + L].astype(float, copy=False)
                gtw = gtw_full_aligned[off:off + L].astype(float, copy=False)

                width_pairs.append({
                    "image": base_name,
                    "cid": str(cid),
                    "crack_type": "atomic",
                    "midline_type": midline_type,
                    "geometry_type": "derived",
                    "member_id": None,
                    "bbox": crack.get("mask_bbox"),
                    "pred_mask_bbox": crack.get("mask_bbox"),
                    "pred_mask_crop": crack.get("mask_crop"),
                    "pts": pts,
                    "predw": predw,
                    "gruthw": gtw,
                    "gt_source": "payload_precomputed",

                    # -------- NEW FLAGS --------
                    "gt_mismatch": atomic_vs_combined_gt,
                    "gt_relation": (
                        "atomic_vs_combined"
                        if atomic_vs_combined_gt
                        else "atomic_vs_atomic"
                    ),
                })

                off += L

            continue

        # --------------------------------------------
        # COMBINED MODE OPSEC (HEAVY DEBUG)
        # --------------------------------------------

        # Stage 0: find best GT entry by overlap
        pred_key = frozenset(map(str, crack.get("members", []) or []))
        gt_entry = gt_sup_combined.get(pred_key)

        if gt_entry is None and gt_sup_combined:
            pm = set(map(str, crack.get("members", []) or []))
            best = None
            for k, e in gt_sup_combined.items():
                gm = set(map(str, e.get("members", []) or []))
                inter = len(pm & gm)
                denom = max(1, max(len(pm), len(gm)))
                u = inter / denom
                if best is None or u > best[0]:
                    best = (u, e)
            if best is not None and best[0] >= 0.60:
                gt_entry = best[1]

        if gt_entry is None:
            gt_members = set(map(str, crack.get("members", []) or []))
            bite_gt = None
        else:
            gt_members = set(map(str, gt_entry.get("members", []) or []))
            bite_gt = None
            dom_gt = gt_entry.get("dominance_meta") or gt_entry.get("dominance") or {}
            if isinstance(dom_gt, dict) and "bite" in dom_gt:
                bite_gt = dom_gt["bite"]

        shared = pred_members & gt_members
        overlap = len(shared) / max(1, max(len(pred_members), len(gt_members)))
        if overlap < 0.60:
            print(f"[WIDTH DEBUG] combined cid={cid} overlap={overlap:.3f} -> SKIP")
            continue

        print(f"[WIDTH DEBUG] cid={cid} shared_members={sorted(shared)}")
        
        # ============================================================
        # STAGE 0.5 — DECODE GT DOMINANCE BITE (AUTHORITATIVE)
        # ============================================================

        loss_masks_gt_by_branch = {}

        if gt_entry and isinstance(gt_entry, dict):
            dom_gt = gt_entry.get("dominance_meta")
            if isinstance(dom_gt, dict):
                bite = dom_gt.get("bite")
                if isinstance(bite, dict):
                    bb = bite.get("bbox")
                    by_branch = bite.get("by_losing_branch")

                    if bb and isinstance(by_branch, dict):
                        bx, by, bw, bh = map(int, bb)

                        for bid, info in by_branch.items():
                            m_local = _decode_packbits_mask(info)
                            if m_local is None:
                                print(
                                    f"[GT BITE WARN] cid={cid} bid={bid} decode failed"
                                )
                                continue

                            # Place into GLOBAL canvas
                            full = np.zeros((H, W), dtype=bool)

                            y1 = min(H, by + m_local.shape[0])
                            x1 = min(W, bx + m_local.shape[1])

                            full[by:y1, bx:x1] = m_local[: y1 - by, : x1 - bx]

                            loss_masks_gt_by_branch[int(bid)] = full

        print(
            f"[GT BITE OK] cid={cid} "
            f"branches={sorted(loss_masks_gt_by_branch.keys())}"
        )
        
        for bid, m in loss_masks_gt_by_branch.items():
            assert m.shape == (H, W), (
                f"[ASSERT FAIL] GT bite shape wrong for bid={bid}: {m.shape}"
            )
            assert m.any(), (
                f"[ASSERT FAIL] GT bite EMPTY for bid={bid}"
            )

        for bid, m in loss_masks_gt_by_branch.items():
            assert m.shape == (H, W), (
                f"[ASSERT FAIL] GT bite shape wrong for bid={bid}: {m.shape}"
            )
            assert m.any(), (
                f"[ASSERT FAIL] GT bite EMPTY for bid={bid}"
            )

        def debug_plot_gt_sup_dominance_raw(
            cid,
            gt_entry,
            out_dir,
        ):
            """
            Plot GT dominance_meta.bite EXACTLY as stored.

            - Proper packbits decode
            - Proper bite-local frame
            - No rebasing hacks
            """

            import os
            import matplotlib.pyplot as plt

            if not isinstance(gt_entry, dict):
                return

            dom = gt_entry.get("dominance_meta")
            if not isinstance(dom, dict):
                return

            bite = dom.get("bite")
            if not isinstance(bite, dict):
                return

            bb = bite.get("bbox")
            by_branch = bite.get("by_losing_branch")

            if not bb or not isinstance(by_branch, dict):
                return

            bx, by, bw, bh = map(int, bb)

            union = None

            for bid, info in by_branch.items():
                m = _decode_packbits_mask(info)
                if m is None:
                    continue
                union = m if union is None else (union | m)

            if union is None or not union.any():
                print(f"[GT_SUP DEBUG] cid={cid} bite EMPTY")
                return

            fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
            ax.set_title(f"GT SUP — RAW DOMINANCE (cid={cid})")
            ax.axis("off")

            ax.imshow(union, cmap="hot", interpolation="nearest", alpha=0.9)

            # Overlay stored midlines (GLOBAL → LOCAL)
            segs = gt_entry.get("midline_segments") or []
            for S in segs:
                if S is None or len(S) < 2:
                    continue
                S = np.asarray(S, float)
                ax.plot(S[:, 0] - bx, S[:, 1] - by, color="cyan", lw=2)

            ax.add_patch(
                plt.Rectangle(
                    (0, 0), union.shape[1], union.shape[0],
                    fill=False, edgecolor="lime", linewidth=2
                )
            )

            os.makedirs(out_dir, exist_ok=True)
            out = os.path.join(out_dir, f"gt_sup_dom_raw_{cid}.png")
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)

            print(f"[GT_SUP DEBUG] wrote {out}")

        debug_plot_gt_sup_dominance_raw(
            cid=cid,
            gt_entry=gt_entry,
            out_dir=opsec_dir,
        )

        # --------------------------------------------
        # Stage 1: prune segments by shared atomic IDs
        # --------------------------------------------
        pruned_segs = []
        pruned_meta = []

        for i, (S, m) in enumerate(zip(segs, seg_meta)):
            if S is None or len(S) < 2:
                continue
            aid = m.get("atomic_id")
            if aid is not None and str(aid) not in shared:
                print(f"[WIDTH DEBUG] DROP seg#{i} atomic={aid} (not shared)")
                continue
            pruned_segs.append(np.asarray(S, float))
            pruned_meta.append(dict(m))

        if not pruned_segs:
            print(f"[WIDTH DEBUG] cid={cid} -> NO SEGMENTS AFTER PRUNE")
            continue

        print(f"[WIDTH DEBUG] cid={cid} kept {len(pruned_segs)} segments after atomic prune")

        # --------------------------------------------
        # Stage 2: optional branch matching (SYMMETRIC)
        #   + GT-local pruning
        #   + compute GT-local decode frame (bbox) for Stage-4/5
        # --------------------------------------------
        matched_pred_branch_ids = None
        matched_gt_branch_ids   = None

        gt_pruned_segs = []
        gt_pruned_meta = []

        # ============================================================
        # Stage-2 helpers: branch signatures & matching (GEOMETRY-BASED)
        # ============================================================
        def _seg_endpoints(S):
            S = np.asarray(S, float)
            if S.ndim != 2 or len(S) < 2:
                return None
            a = S[0].astype(float)
            b = S[-1].astype(float)
            return a, b

        def _endpoint_pair_key(S, snap=5.0):
            """
            Order-invariant snapped endpoint key. Used to group segs into branches.
            snap is in pixels (5 px default).
            """
            ep = _seg_endpoints(S)
            if ep is None:
                return None
            a, b = ep
            a = tuple((np.round(a / snap) * snap).tolist())
            b = tuple((np.round(b / snap) * snap).tolist())
            return tuple(sorted([a, b]))

        def _poly_centroid(S):
            S = np.asarray(S, float)
            return np.nanmean(S, axis=0)

        def _poly_length(S):
            try:
                return float(_linestring_length(np.asarray(S, float)))
            except Exception:
                S = np.asarray(S, float)
                if len(S) < 2:
                    return 0.0
                d = np.diff(S, axis=0)
                return float(np.nansum(np.sqrt(np.sum(d * d, axis=1))))

        def _build_branch_table_geom(segs_in, meta_in, *, scope_members=None):
            """
            Build branches by shared endpoints key (snapped).
            Returns list of dicts:
            {
                "branch_id": int (synthetic),
                "segs": [S...],
                "meta": [m...],
                "endpoints": (a,b) from the longest seg,
                "centroid": (cx,cy),
                "length": total length,
                "atomic_ids": set(str)
            }
            """
            groups = {}
            for S, m in zip(segs_in, meta_in):
                if S is None or len(S) < 2:
                    continue
                m = m if isinstance(m, dict) else {}

                # optional: drop segments that are out of scope
                if scope_members is not None:
                    aid = m.get("atomic_id", None)
                    if aid is not None and str(aid) not in scope_members:
                        continue

                k = _endpoint_pair_key(S, snap=5.0)
                if k is None:
                    continue
                groups.setdefault(k, []).append((np.asarray(S, float), dict(m)))

            out = []
            for bi, (k, items) in enumerate(groups.items()):
                segs_k = [it[0] for it in items]
                meta_k = [it[1] for it in items]

                # choose representative endpoints from the longest seg
                lens = [ _poly_length(S) for S in segs_k ]
                j = int(np.argmax(lens)) if lens else 0
                rep = segs_k[j]
                ep = _seg_endpoints(rep)
                if ep is None:
                    continue
                a, b = ep

                # centroid over all points
                all_pts = np.vstack(segs_k) if segs_k else rep
                c = _poly_centroid(all_pts)

                # total length
                L = float(np.sum([_poly_length(S) for S in segs_k]))

                aids = set()
                for mm in meta_k:
                    aid = mm.get("atomic_id", None)
                    if aid is not None:
                        aids.add(str(aid))

                out.append({
                    "branch_id": int(bi),   # synthetic, stable only inside this call
                    "segs": segs_k,
                    "meta": meta_k,
                    "endpoints": (np.asarray(a, float), np.asarray(b, float)),
                    "centroid": np.asarray(c, float),
                    "length": float(L),
                    "atomic_ids": aids,
                })

            return out

        def _branch_geom_cost(gtb, prb):
            """
            Lower is better. Cost uses:
            - min endpoint pairing distance (order invariant)
            - centroid distance
            - length ratio penalty
            """
            ga, gb = gtb["endpoints"]
            pa, pb = prb["endpoints"]

            # endpoint distance (order invariant)
            d1 = float(np.linalg.norm(ga - pa) + np.linalg.norm(gb - pb))
            d2 = float(np.linalg.norm(ga - pb) + np.linalg.norm(gb - pa))
            d_end = min(d1, d2)

            d_cent = float(np.linalg.norm(gtb["centroid"] - prb["centroid"]))

            # length ratio penalty
            Lg = max(1e-6, float(gtb["length"]))
            Lp = max(1e-6, float(prb["length"]))
            ratio = max(Lg / Lp, Lp / Lg)   # >= 1
            d_len = float((ratio - 1.0) * 50.0)  # 50px penalty per 1x mismatch

            return d_end + 0.25 * d_cent + d_len

        def _greedy_match_branches_geom(gt_br, pr_br, *, max_cost=250.0):
            """
            Greedy one-to-one matching by geometry cost.
            Returns list of tuples: (gt_branch_id, pr_branch_id, cost)
            """
            pairs = []
            for gi, g in enumerate(gt_br):
                for pi, p in enumerate(pr_br):
                    c = _branch_geom_cost(g, p)
                    pairs.append((c, gi, pi))

            pairs.sort(key=lambda t: t[0])

            used_g = set()
            used_p = set()
            matches = []
            for c, gi, pi in pairs:
                if c > max_cost:
                    break
                if gi in used_g or pi in used_p:
                    continue
                used_g.add(gi)
                used_p.add(pi)
                matches.append((gt_br[gi]["branch_id"], pr_br[pi]["branch_id"], float(c)))

            return matches


        # ============================================================
        # (A) GT prune — IMPORTANT FIX:
        #   1) scope to pred_members first
        #   2) then prune to shared
        # ============================================================
        if gt_entry is not None:
            gt_segs_all = gt_entry.get("midline_segments") or []
            gt_meta_all = (gt_entry.get("dominance_meta", {}).get("segments_meta") or [])

            print(
                f"[STAGE2 DBG] cid={cid} GT segs={len(gt_segs_all)} "
                f"GT meta={len(gt_meta_all)} shared={sorted(shared)}"
            )

            if len(gt_segs_all) == len(gt_meta_all) and len(gt_segs_all) > 0:
                for i, (Sg, mg) in enumerate(zip(gt_segs_all, gt_meta_all)):
                    if Sg is None or len(Sg) < 2:
                        continue
                    mg = mg if isinstance(mg, dict) else {}
                    aid = mg.get("atomic_id")

                    # --- scope gate: drop GT segments not in THIS predicted crack ---
                    if aid is not None and str(aid) not in pred_members:
                        print(f"[STAGE2 DBG] SKIP GT seg#{i} atomic={aid} (out-of-scope)")
                        continue

                    # --- shared gate: drop GT segments not shared with prediction ---
                    if aid is not None and str(aid) not in shared:
                        print(f"[STAGE2 DBG] DROP GT seg#{i} atomic={aid} (not shared)")
                        continue

                    gt_pruned_segs.append(np.asarray(Sg, float))
                    gt_pruned_meta.append(dict(mg))
            else:
                # meta mismatch → keep geometry but you lose ability to do atomic/shared gating reliably
                for i, Sg in enumerate(gt_segs_all):
                    if Sg is None or len(Sg) < 2:
                        continue
                    gt_pruned_segs.append(np.asarray(Sg, float))
                    gt_pruned_meta.append({})

            print(f"[STAGE2 DBG] cid={cid} GT kept {len(gt_pruned_segs)} segs after scoped+shared prune")


        # ============================================================
        # (B) symmetric branch matching (GEOMETRY-BASED)
        #   NOTE: do NOT trust GT/PRED branch_id numbering!
        # ============================================================
        if gt_pruned_segs and pruned_segs:
            gt_br = _build_branch_table_geom(gt_pruned_segs, gt_pruned_meta, scope_members=shared)
            pr_br = _build_branch_table_geom(pruned_segs, pruned_meta, scope_members=shared)

            if gt_br and pr_br:
                matches = _greedy_match_branches_geom(gt_br, pr_br, max_cost=250.0)

                if matches:
                    matched_gt_branch_ids   = {g for (g, p, c) in matches}
                    matched_pred_branch_ids = {p for (g, p, c) in matches}

                    print(
                        f"[STAGE2 DBG] cid={cid} branch matches (geom): "
                        f"GT={sorted(matched_gt_branch_ids)} "
                        f"PRED={sorted(matched_pred_branch_ids)} "
                        f"costs={[round(c,1) for (_,_,c) in matches]}"
                    )


        # ============================================================
        # (C) apply branch prune symmetrically
        #   Here: branch_id is SYNTHETIC from our geom tables.
        #   So we must rebuild a map from seg->synthetic branch_id.
        # ============================================================
        def _assign_synth_branch_ids(segs_in, meta_in, scope_members=None):
            br = _build_branch_table_geom(segs_in, meta_in, scope_members=scope_members)
            seg_to_bid = {}

            def _key(S):
                S = np.asarray(S, float)
                a = tuple(np.round(S[0], 3))
                b = tuple(np.round(S[-1], 3))
                n = int(len(S))
                return (a, b, n)

            for b in br:
                bid = int(b["branch_id"])
                for S in b["segs"]:
                    seg_to_bid[_key(S)] = bid
            return seg_to_bid

        if matched_pred_branch_ids is not None:
            seg2bid = _assign_synth_branch_ids(pruned_segs, pruned_meta, scope_members=shared)
            keep_s, keep_m = [], []

            for S, m in zip(pruned_segs, pruned_meta):
                if S is None or len(S) < 2:
                    continue
                k = (tuple(np.round(np.asarray(S, float)[0], 3)),
                    tuple(np.round(np.asarray(S, float)[-1], 3)),
                    int(len(S)))
                bid = seg2bid.get(k, None)

                if bid in matched_pred_branch_ids:
                    keep_s.append(S)
                    keep_m.append(m)
                else:
                    print(f"[STAGE2 DBG] DROP PRED synth_branch={bid} (unmatched)")

            pruned_segs, pruned_meta = keep_s, keep_m

        if matched_gt_branch_ids is not None:
            seg2bid = _assign_synth_branch_ids(gt_pruned_segs, gt_pruned_meta, scope_members=shared)
            keep_s, keep_m = [], []

            for S, m in zip(gt_pruned_segs, gt_pruned_meta):
                if S is None or len(S) < 2:
                    continue
                k = (tuple(np.round(np.asarray(S, float)[0], 3)),
                    tuple(np.round(np.asarray(S, float)[-1], 3)),
                    int(len(S)))
                bid = seg2bid.get(k, None)

                if bid in matched_gt_branch_ids:
                    keep_s.append(S)
                    keep_m.append(m)
                else:
                    print(f"[STAGE2 DBG] DROP GT synth_branch={bid} (unmatched)")

            gt_pruned_segs, gt_pruned_meta = keep_s, keep_m


        if not pruned_segs:
            print(f"[WIDTH DEBUG] cid={cid} -> NO PRED SEGMENTS AFTER BRANCH MATCH")
            continue

        if not gt_pruned_segs:
            print(f"[WIDTH DEBUG] cid={cid} -> NO GT SEGMENTS AFTER BRANCH MATCH")
            # allowed; Stage-4 should just show GT empty

        # ------------------------------------------------------------
        # Stage 2.25: derive corresponding kept DERIVED segments
        # ------------------------------------------------------------
        pred_mid_stage2_segs = pruned_segs
        pred_mid_stage2_meta = pruned_meta
        pred_der_stage2_segs, pred_der_stage2_meta = _match_midline_to_derived(
            pred_mid_stage2_segs,
            pred_mid_stage2_meta,
            dsegs,
            dmeta,
            cid=cid,
        )

        # ------------------------------------------------------------
        # ORIENTATION ROOT DEBUG:
        # Compare Stage-2 PRED derived segments vs Stage-2/4 GT segments
        # by strict (branch_id, seg_idx) key before Stage 4.5/5 processing.
        # ------------------------------------------------------------
        if ORIENT_DEBUG and mode == "combined":
            def _norm_keys_with_branch_seq(segs_in, meta_in):
                out = []
                seq = {}
                for i, (Sx, mx) in enumerate(zip(segs_in or [], meta_in or [])):
                    if Sx is None or len(Sx) < 2:
                        continue
                    mm = mx if isinstance(mx, dict) else {}
                    b = _safe_int(mm.get("branch_id"), None)
                    s = _safe_int(mm.get("seg_idx"), None)
                    if b is None:
                        b = -1
                    if s is None:
                        s = int(seq.get(int(b), 0))
                        seq[int(b)] = int(s) + 1
                    out.append(((int(b), int(s)), np.asarray(Sx, float)))
                return out

            gt_by_key = {}
            for k, Sg in _norm_keys_with_branch_seq(gt_pruned_segs, gt_pruned_meta):
                gt_by_key.setdefault(k, []).append(Sg)

            pred_by_key = {}
            for k, Sp in _norm_keys_with_branch_seq(pred_der_stage2_segs, pred_der_stage2_meta):
                pred_by_key.setdefault(k, []).append(Sp)

            all_keys = sorted(set(gt_by_key.keys()) | set(pred_by_key.keys()))
            if not all_keys:
                print(f"[ORIENT ROOT DBG] cid={cid} no strict (branch_id, seg_idx) keys available")
            for k in all_keys:
                g_list = gt_by_key.get(k, [])
                p_list = pred_by_key.get(k, [])
                print(
                    f"[ORIENT ROOT DBG] cid={cid} key={k} "
                    f"pred_count={len(p_list)} gt_count={len(g_list)}"
                )
                n = min(len(p_list), len(g_list))
                for j in range(n):
                    d_fwd, d_rev, flag = _orient_cost(p_list[j], g_list[j])
                    print(
                        f"[ORIENT ROOT DBG] cid={cid} key={k} pair_idx={j} "
                        f"n_pred={len(p_list[j])} n_gt={len(g_list[j])} "
                        f"d_fwd={d_fwd:.4f} d_rev={d_rev:.4f} flag={flag}"
                    )


        # ============================================================
        # (D) compute GT-local bbox for Stage-4/5
        #   IMPORTANT FIX: use kept GT segs ONLY (already scoped+shared)
        # ============================================================
        gt_bite_bbox_local = None
        if gt_pruned_segs:
            xs = np.concatenate([np.asarray(S)[:, 0] for S in gt_pruned_segs])
            ys = np.concatenate([np.asarray(S)[:, 1] for S in gt_pruned_segs])

            x0 = int(max(0, np.floor(xs.min())))
            y0 = int(max(0, np.floor(ys.min())))
            x1 = int(min(W, np.ceil(xs.max())))
            y1 = int(min(H, np.ceil(ys.max())))

            if x1 > x0 and y1 > y0:
                gt_bite_bbox_local = [x0, y0, x1 - x0, y1 - y0]
                print(f"[STAGE2 DBG] cid={cid} GT-local bite bbox={gt_bite_bbox_local}")
            else:
                print(f"[STAGE2 DBG] cid={cid} GT-local bbox collapsed; leaving None")

        gt_bite_reframe = {
            "bbox": gt_bite_bbox_local,     # xywh in FULL IMAGE coords
            "segments": gt_pruned_segs,
            "segments_meta": gt_pruned_meta,
        }
        
        def plot_stage2_prune_opsec(
            *,
            cid,
            crack,
            H,
            W,
            mask_bin,
            segs,
            seg_meta,
            pruned_segs,
            pruned_meta,
            gt_entry,
            gt_pruned_segs,
            gt_pruned_meta,
            pred_members,
            shared,
            out_dir,
        ):
            """
            Stage-2 OPSEC plot (CID-local):

            LEFT  : GT supervision (kept vs dropped for THIS predicted crack)
            RIGHT : Prediction (kept vs dropped after Stage-2)

            - Geometry only (no dominance masks yet)
            - CID-local crop based on predicted mask_bbox
            """

            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D

            os.makedirs(out_dir, exist_ok=True)

            # ------------------------------------------------------------
            # Reconstruct predicted FULL mask
            # ------------------------------------------------------------
            pred_mask_full = np.zeros((H, W), np.uint8)

            bb = crack.get("mask_bbox")
            crop_list = crack.get("mask_crop")

            if bb and crop_list is not None:
                x, y, w, h = map(int, bb)
                crop_u8 = np.asarray(crop_list, dtype=np.uint8)

                hh = min(h, crop_u8.shape[0]) if crop_u8.ndim >= 2 else 0
                ww = min(w, crop_u8.shape[1]) if crop_u8.ndim >= 2 else 0

                if hh > 0 and ww > 0:
                    pred_mask_full[y:y+hh, x:x+ww] = (crop_u8[:hh, :ww] > 0).astype(np.uint8)

            # ------------------------------------------------------------
            # CID-local crop window (from PRED bbox)
            # ------------------------------------------------------------
            if bb:
                x, y, w, h = map(int, bb)
                pad = 25
                x0 = max(0, x - pad)
                y0 = max(0, y - pad)
                x1 = min(W, x + w + pad)
                y1 = min(H, y + h + pad)
            else:
                x0, y0, x1, y1 = 0, 0, W, H

            # ------------------------------------------------------------
            # Helper: stable segment key
            # ------------------------------------------------------------
            def _seg_key(S):
                S = np.asarray(S, float)
                if S.ndim != 2 or len(S) < 2:
                    return None
                a = tuple(np.round(S[0], 3))
                b = tuple(np.round(S[-1], 3))
                n = int(len(S))
                return (a, b, n)

            # ------------------------------------------------------------
            # PRED kept vs dropped
            # ------------------------------------------------------------
            kept_keys = set()
            for S in pruned_segs or []:
                k = _seg_key(S)
                if k is not None:
                    kept_keys.add(k)

            pred_kept, pred_dropped = [], []
            for S, m in zip(segs, seg_meta):
                if S is None or len(S) < 2:
                    continue
                k = _seg_key(S)
                if k in kept_keys:
                    pred_kept.append(np.asarray(S, float))
                else:
                    pred_dropped.append(np.asarray(S, float))

            # ------------------------------------------------------------
            # GT kept vs dropped (SCOPED + SHARED semantics)
            # ------------------------------------------------------------
            gt_kept = [np.asarray(S, float) for S in (gt_pruned_segs or [])]

            gt_dropped = []
            if gt_entry is not None:
                gt_all = gt_entry.get("midline_segments") or []
                gt_meta_all = (gt_entry.get("dominance_meta", {}) or {}).get("segments_meta") or []

                if len(gt_all) == len(gt_meta_all):
                    for Sg, mg in zip(gt_all, gt_meta_all):
                        if Sg is None or len(Sg) < 2:
                            continue
                        mg = mg if isinstance(mg, dict) else {}
                        aid = mg.get("atomic_id")

                        # out-of-scope GT never plotted
                        if aid is not None and str(aid) not in pred_members:
                            continue

                        Sg = np.asarray(Sg, float)
                        k = _seg_key(Sg)
                        if k not in {_seg_key(S) for S in gt_kept}:
                            gt_dropped.append(Sg)

            # ------------------------------------------------------------
            # Plot
            # ------------------------------------------------------------
            fig, axes = plt.subplots(
                1, 2,
                figsize=(10, 5),
                dpi=200,
                sharex=True,
                sharey=True
            )

            axes[0].set_title("GT supervision (Stage-2)", fontsize=10)
            axes[1].set_title("Prediction (Stage-2)", fontsize=10)

            for ax in axes:
                ax.axis("off")

            axes[0].imshow(mask_bin[y0:y1, x0:x1], cmap="gray", zorder=0)
            axes[1].imshow(pred_mask_full[y0:y1, x0:x1], cmap="gray", zorder=0)

            col_keep = (0.2, 0.4, 0.8)   # blue
            col_drop = (0.5, 0.0, 0.0)   # dark red

            # ---- GT ----
            for S in gt_dropped:
                S2 = S - np.array([x0, y0])
                axes[0].plot(S2[:, 0], S2[:, 1], color=col_drop, lw=2.0, alpha=0.8)

            for S in gt_kept:
                S2 = S - np.array([x0, y0])
                axes[0].plot(S2[:, 0], S2[:, 1], color=col_keep, lw=2.5)

            # ---- PRED ----
            for S in pred_dropped:
                S2 = S - np.array([x0, y0])
                axes[1].plot(S2[:, 0], S2[:, 1], color=col_drop, lw=2.0, alpha=0.8)

            for S in pred_kept:
                S2 = S - np.array([x0, y0])
                axes[1].plot(S2[:, 0], S2[:, 1], color=col_keep, lw=2.5)

            # ---- bbox ----
            if bb:
                for ax in axes:
                    ax.add_patch(
                        plt.Rectangle(
                            (x - x0, y - y0),
                            w, h,
                            fill=False,
                            edgecolor="dodgerblue",
                            lw=1.5,
                        )
                    )

            axes[1].legend(
                handles=[
                    Line2D([0], [0], color=col_keep, lw=3, label="Kept segments"),
                    Line2D([0], [0], color=col_drop, lw=3, label="Dropped segments"),
                    Line2D([0], [0], color="dodgerblue", lw=1.5, label="BBox"),
                ],
                loc="lower right",
                fontsize=8,
                framealpha=0.9,
            )

            member_str = ", ".join(sorted(shared)) if shared else ", ".join(sorted(pred_members))
            fig.suptitle(
                f"Stage-2 prune — cid={cid}\nAtomic members: [{member_str}]",
                fontsize=11,
                fontweight="bold",
            )

            out = os.path.join(out_dir, f"stage2_prune_{cid}.png")
            fig.savefig(out, bbox_inches="tight", dpi=200)
            plt.close(fig)

            print(f"[STAGE2 OPSEC] wrote {out}")

        # Stage-2 prediction plot stream:
        # Use derived geometry so kept/dropped keying is internally consistent.
        plot_pred_segs = pred_der_stage2_segs
        plot_pred_meta = pred_der_stage2_meta

        plot_stage2_prune_opsec(
            cid=cid,
            crack=crack,
            H=H,
            W=W,
            mask_bin=mask_bin,
            segs=dsegs,
            seg_meta=dmeta,
            pruned_segs=plot_pred_segs,
            pruned_meta=plot_pred_meta,
            gt_entry=gt_entry,
            gt_pruned_segs=gt_pruned_segs,
            gt_pruned_meta=gt_pruned_meta,
            pred_members=pred_members,
            shared=shared,
            out_dir=opsec_dir,
        )

        # --------------------------------------------
        # Stage 3: build ORIGINAL segment offsets
        # --------------------------------------------
        orig_segs = [np.asarray(s, float) for s in dsegs if s is not None and len(s) >= 2]
        seg_start = {}
        off = 0
        for i, S in enumerate(orig_segs):
            seg_start[i] = off
            off += len(S)

        print(f"[WIDTH DEBUG] cid={cid} widths_geo={len(widths_geo)}")

        have_valid_seg_idx = any(
            isinstance(m.get("seg_idx"), int) and m["seg_idx"] in seg_start
            for m in pred_der_stage2_meta
        )

        off_fallback = 0
        
        
        
        def _mask_bbox_bool(m):
            ys, xs = np.where(m)
            if xs.size == 0:
                return None
            return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

        def _mask_centroid_bool(m):
            ys, xs = np.where(m)
            if xs.size == 0:
                return None
            return float(xs.mean()), float(ys.mean())

        def _shift_mask_bool(fullmask_bool, dx, dy):
            """
            Shift an HxW bool mask by (dx,dy) in pixel coords.
            Positive dx moves right; positive dy moves down.
            """
            H, W = fullmask_bool.shape
            out = np.zeros((H, W), dtype=bool)

            dx = int(round(dx))
            dy = int(round(dy))
            if dx == 0 and dy == 0:
                return fullmask_bool.copy()

            # Source and destination slices
            x0s = max(0, -dx)
            x1s = min(W, W - dx)
            y0s = max(0, -dy)
            y1s = min(H, H - dy)

            x0d = max(0, dx)
            x1d = min(W, W + dx)
            y0d = max(0, dy)
            y1d = min(H, H + dy)

            if x1s <= x0s or y1s <= y0s or x1d <= x0d or y1d <= y0d:
                return out

            out[y0d:y1d, x0d:x1d] = fullmask_bool[y0s:y1s, x0s:x1s]
            return out

        def _pts_centroid_xy(pts):
            if pts is None:
                return None
            pts = np.asarray(pts, float)
            if pts.ndim != 2 or pts.shape[0] == 0:
                return None
            return float(pts[:, 0].mean()), float(pts[:, 1].mean())

        # ============================================================
        # Stage 4: DOMINANCE-AWARE BITE — READ-ONLY (STAGE0-STYLE)
        #   - decode packbits in BITE-LOCAL frame
        #   - union in local frame
        #   - optional clip to (x0,y0,x1,y1) view window
        #   - NO SHIFT
        #   - NO rebuilding into full HxW
        # ============================================================

        # ----------------------------
        # SAFETY + GT GEOMETRY SOURCE (Stage-4 authoritative)
        # ----------------------------

        # Prefer Stage-2-pruned GT geometry (correct + scoped)
        if "gt_pruned_segs" in locals() and gt_pruned_segs:
            gt_plot_segs = gt_pruned_segs
            gt_plot_meta = gt_pruned_meta if "gt_pruned_meta" in locals() else [{}] * len(gt_pruned_segs)
            print(f"[STAGE4] using gt_pruned_segs ({len(gt_plot_segs)}) for GT plot")

        # Fallback: raw GT midlines (only if Stage-2 GT missing)
        elif isinstance(gt_entry, dict) and gt_entry.get("midline_segments"):
            gt_plot_segs = gt_entry.get("midline_segments")
            gt_plot_meta = (
                (gt_entry.get("dominance_meta", {}) or {}).get("segments_meta")
                or [{}] * len(gt_plot_segs)
            )
            print(f"[STAGE4 WARN] falling back to RAW GT midlines ({len(gt_plot_segs)})")

        # Last-resort safety (should not happen)
        else:
            gt_plot_segs = []
            gt_plot_meta = []
            print("[STAGE4 WARN] no GT geometry available for plotting")

        import os
        import numpy as np
        import matplotlib.pyplot as plt

        dom_pred = crack.get("dominance_meta") if isinstance(crack, dict) else None
        dom_gt   = gt_entry.get("dominance_meta") if isinstance(gt_entry, dict) else None

        os.makedirs(opsec_dir, exist_ok=True)
        _dump_json(os.path.join(opsec_dir, f"dom_pred_{cid}.json"), dom_pred)
        _dump_json(os.path.join(opsec_dir, f"dom_gt_{cid}.json"), dom_gt)

        def _get_bite_union_local(dom):
            """
            Returns:
            bbox_xywh (list[int] or None)
            union_bool (Hloc, Wloc) bool or None
            by_branch (dict[str]->bool mask) in local coords (optional)
            """
            if not isinstance(dom, dict):
                return None, None, {}

            bite = dom.get("bite")
            if not isinstance(bite, dict):
                return None, None, {}

            bb = bite.get("bbox")
            by_branch = bite.get("by_losing_branch")

            if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                return None, None, {}

            bx, by, bw, bh = map(int, bb)

            if not isinstance(by_branch, dict) or not by_branch:
                return [bx, by, bw, bh], None, {}

            union = None
            out_by_branch = {}

            for bid, info in by_branch.items():
                # IMPORTANT: this must decode into BITE-LOCAL coordinates
                m = _decode_packbits_mask(info)  # expected shape (bh, bw) or similar
                if m is None:
                    continue
                m = np.asarray(m).astype(bool)
                if m.ndim != 2 or m.size == 0:
                    continue

                out_by_branch[str(bid)] = m
                union = m if union is None else (union | m)

            return [bx, by, bw, bh], union, out_by_branch

        def _plot_bite_local_union(*, title, bbox_xywh, union, segs_global, out_png, clip_global_xyxy=None):
            """
            Plot union in bite-local frame, overlay global segs shifted by -bbox origin.
            Optional clip to a global xyxy window (x0,y0,x1,y1):
            - clip is applied in LOCAL coords by intersecting with bbox.
            """
            if bbox_xywh is None or union is None or (not np.any(union)):
                print(f"[STAGE4] {title}: EMPTY (bbox={bbox_xywh})")
                return

            bx, by, bw, bh = map(int, bbox_xywh)

            # Optional clip (global window -> local window)
            U = union
            lx0 = ly0 = 0
            if clip_global_xyxy is not None:
                gx0, gy0, gx1, gy1 = map(int, clip_global_xyxy)

                # intersection in GLOBAL
                ix0 = max(gx0, bx)
                iy0 = max(gy0, by)
                ix1 = min(gx1, bx + bw)
                iy1 = min(gy1, by + bh)

                if ix1 > ix0 and iy1 > iy0:
                    # convert to LOCAL slice
                    lx0 = ix0 - bx
                    ly0 = iy0 - by
                    lx1 = ix1 - bx
                    ly1 = iy1 - by
                    U = U[ly0:ly1, lx0:lx1]
                else:
                    # no overlap -> treat as empty plot
                    print(f"[STAGE4] {title}: clip window does not intersect bite bbox -> skip")
                    return

            fig, ax = plt.subplots(figsize=(6, 6), dpi=220)
            ax.set_title(title, fontsize=12)
            ax.axis("off")

            ax.imshow(U.astype(np.uint8), cmap="hot", interpolation="nearest", alpha=0.95)

            # overlay segs (global -> bite-local -> optionally clipped window offset)
            for S in segs_global or []:
                if S is None or len(S) < 2:
                    continue
                S = np.asarray(S, float)
                X = S[:, 0] - bx - float(lx0)
                Y = S[:, 1] - by - float(ly0)
                ax.plot(X, Y, color="cyan", lw=2)

            # draw frame boundary (local)
            ax.add_patch(
                plt.Rectangle(
                    (0, 0),
                    U.shape[1],
                    U.shape[0],
                    fill=False,
                    edgecolor="lime",
                    linewidth=2,
                )
            )

            fig.savefig(out_png, bbox_inches="tight")
            plt.close(fig)
            print(f"[STAGE4] wrote {out_png}")

        def _paste_pred_into_gt_frame(*, gt_bbox, gt_union, pred_bbox, pred_union):
            """
            Create two masks in GT bite-local frame:
            - gt_local = gt_union (as-is)
            - pred_in_gt = pred_union pasted into GT frame using bbox offset
            Returns: (gt_local_bool, pred_in_gt_bool)
            """
            if gt_bbox is None or gt_union is None:
                return None, None
            if pred_bbox is None or pred_union is None:
                return gt_union.astype(bool), np.zeros_like(gt_union, dtype=bool)

            gbx, gby, gbw, gbh = map(int, gt_bbox)
            pbx, pby, pbw, pbh = map(int, pred_bbox)

            gt_local = gt_union.astype(bool)

            # pred pixel at global (pbx + x, pby + y)
            # maps to GT-local (x + (pbx-gbx), y + (pby-gby))
            dx = int(pbx - gbx)
            dy = int(pby - gby)

            canvas = np.zeros((gt_local.shape[0], gt_local.shape[1]), dtype=bool)

            # paste with clipping to canvas bounds
            y0 = max(0, dy)
            x0 = max(0, dx)
            y1 = min(canvas.shape[0], dy + pred_union.shape[0])
            x1 = min(canvas.shape[1], dx + pred_union.shape[1])

            sy0 = max(0, -dy)
            sx0 = max(0, -dx)
            sy1 = sy0 + (y1 - y0)
            sx1 = sx0 + (x1 - x0)

            if (y1 > y0) and (x1 > x0) and (sy1 > sy0) and (sx1 > sx0):
                canvas[y0:y1, x0:x1] = pred_union.astype(bool)[sy0:sy1, sx0:sx1]

            return gt_local, canvas

        def _plot_overlay_in_gt_frame(*, gt_bbox, gt_union, pred_bbox, pred_union, out_png, clip_global_xyxy=None):
            """
            Overlay GT (red) and Pred (blue) in GT bite-local frame, optionally clipped to a global window.
            """
            if gt_bbox is None or gt_union is None or not np.any(gt_union):
                print("[STAGE4] overlay: GT union empty -> skip")
                return

            gbx, gby, gbw, gbh = map(int, gt_bbox)

            gt_local, pred_in_gt = _paste_pred_into_gt_frame(
                gt_bbox=gt_bbox,
                gt_union=gt_union,
                pred_bbox=pred_bbox,
                pred_union=pred_union,
            )

            # Optional clip window (global -> gt-local)
            lx0 = ly0 = 0
            GT = gt_local
            PR = pred_in_gt
            if clip_global_xyxy is not None:
                gx0, gy0, gx1, gy1 = map(int, clip_global_xyxy)

                ix0 = max(gx0, gbx)
                iy0 = max(gy0, gby)
                ix1 = min(gx1, gbx + gbw)
                iy1 = min(gy1, gby + gbh)

                if ix1 <= ix0 or iy1 <= iy0:
                    print("[STAGE4] overlay: clip does not intersect GT bite bbox -> skip")
                    return

                lx0 = ix0 - gbx
                ly0 = iy0 - gby
                lx1 = ix1 - gbx
                ly1 = iy1 - gby

                GT = GT[ly0:ly1, lx0:lx1]
                PR = PR[ly0:ly1, lx0:lx1]

            overlay = np.zeros((GT.shape[0], GT.shape[1], 3), dtype=np.float32)
            overlay[..., 0] = GT.astype(np.float32)  # R
            overlay[..., 2] = PR.astype(np.float32)  # B

            fig, ax = plt.subplots(figsize=(7, 6), dpi=220)
            ax.set_title("Stage4 overlay in GT bite-local frame (GT=R, Pred=B)", fontsize=11)
            ax.imshow(overlay, interpolation="nearest")
            ax.axis("off")

            ax.add_patch(
                plt.Rectangle(
                    (0, 0),
                    overlay.shape[1],
                    overlay.shape[0],
                    fill=False,
                    edgecolor="lime",
                    linewidth=2,
                )
            )

            fig.savefig(out_png, bbox_inches="tight")
            plt.close(fig)
            print(f"[STAGE4] wrote {out_png}")

        # ----------------------------
        # Decode unions in bite-local coordinates (Stage0 style)
        # ----------------------------
        gt_bbox, gt_union_local, _ = _get_bite_union_local(dom_gt)
        pr_bbox, pr_union_local, _ = _get_bite_union_local(dom_pred)

        print(f"[STAGE4] cid={cid} GT bite bbox={gt_bbox} union_px={(0 if gt_union_local is None else int(gt_union_local.sum()))}")
        print(f"[STAGE4] cid={cid} PR bite bbox={pr_bbox} union_px={(0 if pr_union_local is None else int(pr_union_local.sum()))}")

        # Build seg lists for overlay:
        gt_segs_for_plot = gt_entry.get("midline_segments") if isinstance(gt_entry, dict) else []
        pr_segs_for_plot = pruned_segs if ("pruned_segs" in locals()) else []

        # ============================================================
        # Stage-4 VIEW WINDOW (MATCHES STAGE 2 / STAGE 5)
        #   - ALWAYS pred mask_bbox + pad
        # ============================================================

        pad = 25

        bb = crack.get("mask_bbox") if isinstance(crack, dict) else None
        if bb:
            x, y, w, h = map(int, bb)
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(W, x + w + pad)
            y1 = min(H, y + h + pad)
        else:
            x0, y0, x1, y1 = 0, 0, W, H

        clip_xyxy = (x0, y0, x1, y1)

        print(f"[STAGE4 VIEW] cid={cid} view bbox = {(x0, y0, x1, y1)}")

        # ----------------------------
        # PLOT 1: GT bite-local raw (like Stage 0)
        # ----------------------------
        '''_plot_bite_local_union(
            title=f"Stage4 GT — RAW bite-local union (cid={cid})",
            bbox_xywh=gt_bbox,
            union=gt_union_local,
            segs_global=gt_segs_for_plot,
            out_png=os.path.join(opsec_dir, f"stage4_gt_dom_raw_local_{cid}.png"),
            clip_global_xyxy=None,  # full bite-local
        )

        # ----------------------------
        # PLOT 2: PRED bite-local raw
        # ----------------------------
        _plot_bite_local_union(
            title=f"Stage4 PRED — RAW bite-local union (cid={cid})",
            bbox_xywh=pr_bbox,
            union=pr_union_local,
            segs_global=pr_segs_for_plot,
            out_png=os.path.join(opsec_dir, f"stage4_pred_dom_raw_local_{cid}.png"),
            clip_global_xyxy=None,
        )

        # ----------------------------
        # PLOT 3: Overlay pred pasted into GT bite-local frame (no modification, just bbox mapping)
        # ----------------------------
        _plot_overlay_in_gt_frame(
            gt_bbox=gt_bbox,
            gt_union=gt_union_local,
            pred_bbox=pr_bbox,
            pred_union=pr_union_local,
            out_png=os.path.join(opsec_dir, f"stage4_dom_overlay_in_gt_frame_{cid}.png"),
            clip_global_xyxy=None,
        )

        # ----------------------------
        # CLIPPED VERSIONS (Stage-2/5-style view window)
        # ----------------------------

        _plot_bite_local_union(
            title=f"Stage4 GT — RAW bite-local union (CLIPPED view) (cid={cid})",
            bbox_xywh=gt_bbox,
            union=gt_union_local,
            segs_global=gt_segs_for_plot,
            out_png=os.path.join(opsec_dir, f"stage4_gt_dom_raw_local_clipped_{cid}.png"),
            clip_global_xyxy=clip_xyxy,
        )

        _plot_bite_local_union(
            title=f"Stage4 PRED — RAW bite-local union (CLIPPED view) (cid={cid})",
            bbox_xywh=pr_bbox,
            union=pr_union_local,
            segs_global=pr_segs_for_plot,
            out_png=os.path.join(opsec_dir, f"stage4_pred_dom_raw_local_clipped_{cid}.png"),
            clip_global_xyxy=clip_xyxy,
        )

        _plot_overlay_in_gt_frame(
            gt_bbox=gt_bbox,
            gt_union=gt_union_local,
            pred_bbox=pr_bbox,
            pred_union=pr_union_local,
            out_png=os.path.join(opsec_dir, f"stage4_dom_overlay_in_gt_frame_clipped_{cid}.png"),
            clip_global_xyxy=clip_xyxy,
        )'''
        
        # ============================================================
        # Stage 4 — PLOT B (crop-local visualization, Stage-0 truthful)
        #   - dominance comes ONLY from bite-local packbits
        #   - rasterized into crop-local frame for plotting
        #   - NO shift, NO rebase, NO metric side effects
        # ============================================================

        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.colors import ListedColormap

        # ----------------------------
        # SAFETY
        # ----------------------------
        if gt_plot_segs is None:
            gt_plot_segs = []
        if gt_plot_meta is None:
            gt_plot_meta = []

        # ----------------------------
        # Crop geometry
        # ----------------------------
        Hc = int(y1 - y0)
        Wc = int(x1 - x0)

        # ----------------------------
        # Build dom_label *only for Plot B*
        #   0 = background
        #   1 = GT-only
        #   2 = Pred-only
        #   3 = both
        # ----------------------------
        dom_label = np.zeros((Hc, Wc), dtype=np.uint8)

        def _raster_bite_into_crop(*, bbox, union, value):
            """
            bbox: [bx,by,bw,bh] in GLOBAL coords
            union: (bh,bw) bool, BITE-LOCAL
            value: 1 (GT) or 2 (Pred)
            """
            if bbox is None or union is None or not np.any(union):
                return

            bx, by, bw, bh = map(int, bbox)

            # intersection in GLOBAL coords
            ix0 = max(x0, bx)
            iy0 = max(y0, by)
            ix1 = min(x1, bx + bw)
            iy1 = min(y1, by + bh)

            if ix1 <= ix0 or iy1 <= iy0:
                return

            # crop-local indices
            cx0 = ix0 - x0
            cy0 = iy0 - y0
            cx1 = ix1 - x0
            cy1 = iy1 - y0

            # bite-local indices
            ux0 = ix0 - bx
            uy0 = iy0 - by
            ux1 = ux0 + (cx1 - cx0)
            uy1 = uy0 + (cy1 - cy0)

            dom_label[cy0:cy1, cx0:cx1] |= (
                union[uy0:uy1, ux0:ux1].astype(np.uint8) * value
            )

        # rasterize GT and Pred
        _raster_bite_into_crop(
            bbox=gt_bbox,
            union=gt_union_local,
            value=1,
        )

        _raster_bite_into_crop(
            bbox=pr_bbox,
            union=pr_union_local,
            value=2,
        )

        # ----------------------------
        # Prepare masked dominance
        # ----------------------------
        dom_masked = np.ma.array(dom_label, mask=(dom_label == 0))

        DOM_CMAP = ListedColormap([
            "#000000",  # 0 background (masked)
            "#e41a1c",  # 1 GT-only
            "#377eb8",  # 2 Pred-only
            "#984ea3",  # 3 GT ∩ Pred
        ])

        # ----------------------------
        # Rebuild prediction mask (background only)
        # ----------------------------
        def _rebuild_pred_mask(crack, H, W):
            pm = np.zeros((H, W), np.uint8)
            bb = crack.get("mask_bbox")
            crop = crack.get("mask_crop")
            if bb and crop is not None:
                x, y, w, h = map(int, bb)
                crop = np.asarray(crop, np.uint8)
                if crop.ndim >= 2:
                    hh = min(h, crop.shape[0])
                    ww = min(w, crop.shape[1])
                    if hh > 0 and ww > 0:
                        pm[y:y+hh, x:x+ww] = (crop[:hh, :ww] > 0).astype(np.uint8)
            return pm

        pred_mask_full = _rebuild_pred_mask(crack, H, W)

        # ----------------------------
        # Plot
        # ----------------------------
        fig, axes = plt.subplots(
            1, 2, figsize=(12, 6), dpi=240, sharex=True, sharey=True
        )

        axes[0].set_title("Stage 4 — GT supervision", fontsize=10)
        axes[1].set_title("Stage 4 — Prediction", fontsize=10)

        for ax in axes:
            ax.axis("off")

        # backgrounds
        axes[0].imshow(
            (mask_bin[y0:y1, x0:x1] > 0).astype(np.uint8),
            cmap="gray", vmin=0, vmax=1,
            interpolation="nearest", zorder=0
        )

        axes[1].imshow(
            pred_mask_full[y0:y1, x0:x1] if np.any(pred_mask_full)
            else np.zeros((Hc, Wc), np.uint8),
            cmap="gray", vmin=0, vmax=1,
            interpolation="nearest", zorder=0
        )

        # dominance overlay
        extent = [0, Wc, Hc, 0]
        for ax in axes:
            ax.imshow(
                dom_masked,
                cmap=DOM_CMAP,
                interpolation="nearest",
                vmin=0, vmax=3,
                alpha=0.9,
                zorder=1,
                extent=extent,
            )
            ax.set_xlim(0, Wc)
            ax.set_ylim(Hc, 0)

        # ----------------------------
        # Midlines
        # ----------------------------
        color_cycle = [
            (0.95, 0.90, 0.25),
            (0.25, 0.85, 0.35),
            (0.25, 0.55, 0.95),
            (0.95, 0.35, 0.35),
        ]

        # atomic -> branch map (best effort)
        atomic_to_branch = {}
        for mg in gt_plot_meta or []:
            if isinstance(mg, dict):
                aid = mg.get("atomic_id")
                bid = mg.get("branch_id")
                if aid is not None and bid is not None:
                    atomic_to_branch[str(aid)] = int(bid)

        legend_handles = []
        seen = set()

        # GT midlines (left)
        for Sg, mg in zip(gt_plot_segs or [], gt_plot_meta or []):
            if Sg is None or len(Sg) < 2:
                continue
            mg = mg if isinstance(mg, dict) else {}
            aid = mg.get("atomic_id")

            bid = None
            if aid is not None:
                bid = atomic_to_branch.get(str(aid))
            if bid is None:
                bid = _safe_int(mg.get("branch_id"), 0)

            col = color_cycle[int(bid) % len(color_cycle)]
            S2 = np.asarray(Sg, float) - np.array([x0, y0], float)

            axes[0].plot(S2[:, 0], S2[:, 1], color=col, lw=2.3, zorder=5)

            if bid not in seen:
                legend_handles.append(
                    Line2D([0], [0], color=col, lw=3, label=f"branch {bid}")
                )
                seen.add(bid)

        # Pred midlines (right) — derived stage-2 geometry
        for S, m in zip(pred_der_stage2_segs or [], pred_der_stage2_meta or []):
            if S is None or len(S) < 2:
                continue
            m = m if isinstance(m, dict) else {}
            bid = _safe_int(m.get("branch_id"), 0)
            col = color_cycle[int(bid) % len(color_cycle)]
            S2 = np.asarray(S, float) - np.array([x0, y0], float)
            axes[1].plot(S2[:, 0], S2[:, 1], color=col, lw=2.3, zorder=5)

        # dominance legend
        legend_handles += [
            Line2D([0], [0], color="#e41a1c", lw=6, label="GT-only loss"),
            Line2D([0], [0], color="#377eb8", lw=6, label="Pred-only loss"),
            Line2D([0], [0], color="#984ea3", lw=6, label="GT ∩ Pred"),
        ]

        axes[1].legend(
            handles=legend_handles,
            loc="lower right",
            fontsize=8,
            framealpha=0.9,
        )

        outB = os.path.join(
            opsec_dir,
            f"stage4_dominance_bite_{cid}_{midline_type}_{mode}.png",
        )
        fig.savefig(outB, bbox_inches="tight")
        plt.close(fig)

        print(f"[OPSEC] Stage-4 dominance plot written: {outB}")
            
        # ============================================================
        # Stage 4.5 + Stage 5 — STRICT DOMINANCE APPLICATION
        #   HARD RULES:
        #     - MUST start from Stage-4 pruned geometry
        #     - GT MUST come from Stage-2/4-scoped GT (gt_pruned_segs)
        #     - If those do not exist → FAIL
        # ============================================================

        import numpy as np

        # ------------------------------------------------------------
        # HARD FAILS — NO FALLBACKS
        # ------------------------------------------------------------
        if not pred_mid_stage2_segs:
            raise RuntimeError("[STAGE4.5 FATAL] pred_mid_stage2_segs missing")
        if not pred_der_stage2_segs:
            raise RuntimeError("[STAGE4.5 FATAL] pred_der_stage2_segs missing")
        if "gt_pruned_segs" not in locals() or not gt_pruned_segs:
            raise RuntimeError("[STAGE4.5 FATAL] gt_pruned_segs missing — must start from Stage-2/4 GT geometry")

        gt_stage5_source_segs = gt_pruned_segs
        gt_stage5_source_meta = (
            gt_pruned_meta
            if ("gt_pruned_meta" in locals() and gt_pruned_meta)
            else [{}] * len(gt_stage5_source_segs)
        )

        print(f"[STAGE4.5] using Stage-2 PRED mid segs: {len(pred_mid_stage2_segs)}")
        print(f"[STAGE4.5] using Stage-2 PRED derived segs: {len(pred_der_stage2_segs)}")
        print(f"[STAGE4.5] using Stage-2/4 GT segs: {len(gt_stage5_source_segs)}")

        # ------------------------------------------------------------
        # Decode dominance loss masks (FULL FRAME)
        # ------------------------------------------------------------
        dom_pred = crack.get("dominance_meta") if isinstance(crack, dict) else None
        dom_gt   = gt_entry.get("dominance_meta") if isinstance(gt_entry, dict) else None

        def _inflate_local_to_full(bbox_xywh, m_local, H, W):
            if bbox_xywh is None or m_local is None:
                return None
            bx, by, bw, bh = map(int, bbox_xywh)
            m = np.asarray(m_local).astype(bool)
            if m.ndim != 2 or m.size == 0:
                return None

            full = np.zeros((H, W), bool)
            mh, mw = m.shape
            x0, y0 = max(0, bx), max(0, by)
            x1, y1 = min(W, bx + mw), min(H, by + mh)
            if x1 <= x0 or y1 <= y0:
                return full

            sx0, sy0 = max(0, -bx), max(0, -by)
            sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)
            full[y0:y1, x0:x1] = m[sy0:sy1, sx0:sx1]
            return full

        def _decode_bite_loss_masks_full(dom, H, W):
            if not isinstance(dom, dict):
                return {}
            bite = dom.get("bite")
            if not isinstance(bite, dict):
                return {}
            bb = bite.get("bbox")
            by_branch = bite.get("by_losing_branch")
            if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                return {}
            out = {}
            for bid, info in (by_branch or {}).items():
                m_local = _decode_packbits_mask(info)
                if m_local is None:
                    continue
                m_full = _inflate_local_to_full(bb, m_local, H, W)
                if m_full is None:
                    continue
                try:
                    out[int(bid)] = m_full.astype(bool)
                except Exception:
                    pass
            return out

        loss_masks_pred_by_branch = _decode_bite_loss_masks_full(dom_pred, H, W)
        loss_masks_gt_by_branch   = _decode_bite_loss_masks_full(dom_gt,   H, W)

        # ------------------------------------------------------------
        # Snapshot BEFORE dominance
        # ------------------------------------------------------------
        pred_pre45_segs = [np.asarray(S, float) for S in pred_der_stage2_segs if S is not None and len(S) >= 2]

        # ============================================================
        # Stage 4.5 — APPLY UNION DOMINANCE (PRED MID + PRED DERIVED + GT)
        # ============================================================
        pred_mid_dom_segs, pred_mid_dom_meta, bite_pruned_pred_mid = _apply_union_dominance(
            pred_mid_stage2_segs,
            pred_mid_stage2_meta,
            loss_masks_pred_by_branch=loss_masks_pred_by_branch,
            loss_masks_gt_by_branch=loss_masks_gt_by_branch,
            H_full=H,
            W_full=W,
        )
        pred_der_dom_segs, pred_der_dom_meta, bite_pruned_pred_der = _apply_union_dominance(
            pred_der_stage2_segs,
            pred_der_stage2_meta,
            loss_masks_pred_by_branch=loss_masks_pred_by_branch,
            loss_masks_gt_by_branch=loss_masks_gt_by_branch,
            H_full=H,
            W_full=W,
        )
        gt_dom_segs, gt_dom_meta, bite_pruned_gt_segs = _apply_union_dominance(
            gt_stage5_source_segs,
            gt_stage5_source_meta,
            loss_masks_pred_by_branch=loss_masks_pred_by_branch,
            loss_masks_gt_by_branch=loss_masks_gt_by_branch,
            H_full=H,
            W_full=W,
        )

        bite_pruned_pred_segs = bite_pruned_pred_der
        gt_stage5_segs = gt_dom_segs
        gt_stage5_meta = gt_dom_meta

        pruned_segs = pred_mid_dom_segs
        pruned_meta = pred_mid_dom_meta

        if DEBUG_TOPOLOGY_TRACE:
            _dump_json(
                os.path.join(topo_dbg_dir, f"cid_{cid}_stage45_counts.json"),
                {
                    "cid": str(cid),
                    "pred_der_stage2_n": int(len(pred_der_stage2_segs)),
                    "pred_der_dom_n": int(len(pred_der_dom_segs)),
                    "bite_pruned_pred_n": int(len(bite_pruned_pred_der) if bite_pruned_pred_der else 0),
                    "gt_stage5_source_n": int(len(gt_stage5_source_segs)),
                    "gt_dom_n": int(len(gt_dom_segs)),
                },
            )

            rows_stage45 = []
            for i, (S2, m2) in enumerate(zip(pred_der_stage2_segs, pred_der_stage2_meta)):
                mm2 = m2 if isinstance(m2, dict) else {}
                rows_stage45.append(
                    ["stage2", int(i), mm2.get("branch_id"), mm2.get("seg_idx"), int(len(S2))]
                )
            for i, (S3, m3) in enumerate(zip(pred_der_dom_segs, pred_der_dom_meta)):
                mm3 = m3 if isinstance(m3, dict) else {}
                rows_stage45.append(
                    ["stage45_pred_dom", int(i), mm3.get("branch_id"), mm3.get("seg_idx"), int(len(S3))]
                )
            for i, Sx in enumerate(bite_pruned_pred_der or []):
                rows_stage45.append(["stage45_bite_pruned", int(i), None, None, int(len(Sx))])

            _write_csv(
                os.path.join(topo_dbg_dir, f"cid_{cid}_stage45_seg_lengths.csv"),
                rows_stage45,
                header=["where", "i", "branch_id", "seg_idx", "npts"],
            )

        print(f"[STAGE4.5] PRED MID kept {len(pred_mid_dom_segs)} runs")
        print(f"[STAGE4.5] PRED DERIVED kept {len(pred_der_dom_segs)} runs")
        print(f"[STAGE4.5] GT kept {len(gt_stage5_segs)} runs")
        print("\n====================")
        print(f"[TOPO TRACE] CID={cid}")
        print("====================")
        print(f"[TRACE] Stage2 derived seg count: {len(pred_der_stage2_segs)}")
        print(f"[TRACE] Stage4.5 derived seg count: {len(pred_der_dom_segs)}")
        print(f"[TRACE] Bite-pruned seg count: {len(bite_pruned_pred_der) if bite_pruned_pred_der else 0}")
        for i, (S2, S3) in enumerate(zip(pred_der_stage2_segs, pred_der_dom_segs)):
            n2 = len(S2) if S2 is not None else 0
            n3 = len(S3) if S3 is not None else 0
            print(f"[TRACE] seg{i} Stage2 pts={n2} -> Stage4.5 pts={n3}  Delta={n2 - n3}")

        _log_branch_orientation(pred_der_stage2_segs, pred_der_stage2_meta, tag="stage2_pred_derived", cid_dbg=cid)
        _log_branch_orientation(pred_der_dom_segs, pred_der_dom_meta, tag="stage45_pred_derived", cid_dbg=cid)
        _log_branch_orientation(gt_stage5_segs, gt_stage5_meta, tag="stage45_gt", cid_dbg=cid)
        _check_branch_direction_consistency(pred_der_stage2_segs, pred_der_stage2_meta, tag="stage2_pred_derived", cid_dbg=cid)
        _check_branch_direction_consistency(pred_der_dom_segs, pred_der_dom_meta, tag="stage45_pred_derived", cid_dbg=cid)
        _check_branch_direction_consistency(gt_stage5_segs, gt_stage5_meta, tag="stage45_gt", cid_dbg=cid)

        # ============================================================
        # Stage 5 - WIDTH ATTACHMENT (STRICT, NO GEOMETRY TRUNCATION)
        # ============================================================
        if not pred_der_dom_segs:
            raise RuntimeError("[STAGE5 FATAL] no derived prediction geometry after dominance")
        if not gt_stage5_segs:
            raise RuntimeError("[STAGE5 FATAL] no GT geometry after dominance")

        final_pred_segs = []
        stage4_pairs = []

        pred_source = predw_full_any
        if pred_source is None or np.asarray(pred_source).size < 2:
            pred_source = widths_geo
        if pred_source is None or np.asarray(pred_source).size < 2:
            print(f"[STAGE5] cid={cid} no usable predicted width source")
            continue
        pred_source = np.asarray(pred_source, float).reshape(-1)

        gtw_source = _get_gt_width_full(crack, gt_entry)
        if gtw_source is None or np.asarray(gtw_source).size < 2:
            print(f"[STAGE5] cid={cid} no usable GT width source in payload/supervision")
            continue
        gtw_source = np.asarray(gtw_source, float).reshape(-1)

        def _pad_to_len(arr, L, pad_value=np.nan):
            arr = np.asarray(arr, float).reshape(-1)
            if arr.size >= L:
                return arr[:L]
            out = np.empty((L,), dtype=float)
            out[:] = pad_value
            if arr.size > 0:
                out[:arr.size] = arr
            return out

        def _safe_take(arr, s0, L, pad_value=np.nan):
            """
            Take arr[s0:s0+L], but if it runs out, pad with pad_value to length L.
            Never returns shorter than L.
            """
            arr = np.asarray(arr, float).reshape(-1)
            if L <= 0:
                return np.asarray([], float)
            s0 = int(max(0, s0))
            s1 = int(s0 + L)
            if s0 >= arr.size:
                out = np.empty((L,), dtype=float)
                out[:] = pad_value
                return out
            sl = arr[s0:min(s1, arr.size)]
            return _pad_to_len(sl, L, pad_value=pad_value)

        def _resample_1d_to_len(arr, L):
            """
            Resample 1D signal to length L using linear interpolation.
            Falls back to NaN padding for degenerate/empty inputs.
            """
            arr = np.asarray(arr, float).reshape(-1)
            L = int(L)
            if L <= 0:
                return np.asarray([], float)
            if arr.size == 0:
                return _pad_to_len(arr, L, pad_value=np.nan)
            if arr.size == 1:
                return _pad_to_len(arr, L, pad_value=np.nan)
            if arr.size == L:
                return arr.astype(float, copy=False)
            x_old = np.linspace(0.0, 1.0, num=arr.size)
            x_new = np.linspace(0.0, 1.0, num=L)
            return np.interp(x_new, x_old, arr).astype(float, copy=False)

        stage5_slice_csv = os.path.join(topo_dbg_dir, f"cid_{cid}_stage5_slices.csv")

        print(
            f"[STAGE5 PRECHECK] cid={cid} "
            f"pred_width_len={len(pred_source)} gt_width_len={len(gtw_source)} "
            f"geom_pts_total={int(sum(len(S) for S in (pred_der_dom_segs or []) if S is not None))}"
        )
        if len(pred_source) != len(gtw_source):
            print(
                f"[STAGE5 WARN] cid={cid} width vector length mismatch "
                f"(pred={len(pred_source)} gt={len(gtw_source)})"
            )

        # Build GT per-(branch_id, seg_idx) buckets in GT-local concatenation order.
        # This avoids using prediction-side global offsets to read GT width streams.
        def _norm_stage5_key(meta_obj):
            if not isinstance(meta_obj, dict):
                return None
            b = _safe_int(meta_obj.get("branch_id"), None)
            s = _safe_int(meta_obj.get("seg_idx"), None)
            if b is None or s is None:
                return None
            return (int(b), int(s))

        gt_bucket = {}
        gt_bucket_by_branch = {}
        gt_branch_seq = {}
        gt_off = 0
        for S_gt, m_gt in zip(gt_stage5_segs or [], gt_stage5_meta or []):
            if S_gt is None or len(S_gt) < 2:
                continue
            mm_gt = m_gt if isinstance(m_gt, dict) else {}
            key = _norm_stage5_key(mm_gt)
            b_only = _safe_int(mm_gt.get("branch_id"), None)
            if key is None and b_only is not None:
                kseq = int(gt_branch_seq.get(int(b_only), 0))
                key = (int(b_only), kseq)
                gt_branch_seq[int(b_only)] = kseq + 1
            Lgt = int(len(S_gt))
            gtw_local = _safe_take(gtw_source, gt_off, Lgt, pad_value=np.nan)
            gt_off += Lgt
            rec = (
                np.asarray(S_gt, float),
                np.asarray(gtw_local, float),
                int(Lgt),
                dict(mm_gt),
            )
            if key is not None:
                gt_bucket.setdefault(key, []).append(rec)
            if b_only is not None:
                gt_bucket_by_branch.setdefault(int(b_only), []).append(rec)

        print("\n[DEBUG KEY INSPECTION]")
        print("GT KEYS:")
        for k in gt_bucket.keys():
            print("   ", k, type(k[0]), type(k[1]))
        print("GT BRANCH KEYS:")
        for bk, arr in gt_bucket_by_branch.items():
            print("   ", bk, "count=", len(arr))
        print("PRED KEYS:")
        for m in (pred_der_dom_meta or []):
            if isinstance(m, dict):
                kb = m.get("branch_id")
                ks = m.get("seg_idx")
                print("   ", (kb, ks), type(kb), type(ks))
        print("[END DEBUG]\n")

        pred_off = 0
        for S, m in zip(pred_der_dom_segs, pred_der_dom_meta):
            if S is None or len(S) < 2:
                continue

            pts = np.asarray(S, float)
            L = int(len(pts))

            mm_pred = m if isinstance(m, dict) else {}
            seg_idx_dbg = mm_pred.get("seg_idx")
            branch_dbg = mm_pred.get("branch_id")
            key = _norm_stage5_key(mm_pred)
            if key is None:
                raise RuntimeError(
                    f"[STAGE5 FATAL] invalid pred key metadata for cid={cid}: "
                    f"branch_id={branch_dbg}, seg_idx={seg_idx_dbg}"
                )

            # Pred width stream: local sequential attachment over surviving pred segments.
            predw = _safe_take(pred_source, pred_off, L, pad_value=np.nan)
            pred_off += L

            # GT width stream: local by (branch_id, seg_idx), no pred-global indexing.
            gt_list = gt_bucket.get(key, [])
            if gt_list:
                gt_match_seg, gtw_local, gt_seg_len, _ = gt_list.pop(0)
                gt_match_mode = "segment_local_match_strict"
            else:
                # Fallback: branch-only match if GT seg_idx is missing/inconsistent.
                b_only = key[0]
                cand = gt_bucket_by_branch.get(int(b_only), [])
                if cand:
                    # Pick length-closest remaining segment in this branch.
                    j_best = min(range(len(cand)), key=lambda j: abs(int(cand[j][2]) - int(L)))
                    gt_match_seg, gtw_local, gt_seg_len, _ = cand.pop(j_best)
                    # Keep strict bucket consistent: remove this chosen record if present there.
                    k_strict = _norm_stage5_key({"branch_id": b_only, "seg_idx": seg_idx_dbg})
                    if k_strict in gt_bucket and gt_bucket[k_strict]:
                        # best-effort pop one; strict set may be empty anyway
                        pass
                    gt_match_mode = "segment_local_match_branch_fallback"
                    print(
                        f"[STAGE5 WARN] strict key miss for {key}, "
                        f"used branch fallback branch={b_only} len_pred={L} len_gt={gt_seg_len}"
                    )
                else:
                    raise RuntimeError(
                        f"[STAGE5 FATAL] no GT match for normalized key={key} "
                        f"(raw branch={branch_dbg}, seg_idx={seg_idx_dbg}) cid={cid}"
                    )

            # Orientation diagnostic (debug-only): detect local segment reversal.
            orient_flag = "unknown"
            d_forward = np.nan
            d_reverse = np.nan
            try:
                p0 = np.asarray(pts[0], float)
                p1 = np.asarray(pts[-1], float)
                g0 = np.asarray(gt_match_seg[0], float)
                g1 = np.asarray(gt_match_seg[-1], float)
                d_forward = float(np.linalg.norm(p0 - g0) + np.linalg.norm(p1 - g1))
                d_reverse = float(np.linalg.norm(p0 - g1) + np.linalg.norm(p1 - g0))
                orient_flag = "reversed_candidate" if d_reverse < d_forward else "forward_candidate"
            except Exception:
                orient_flag = "orientation_check_failed"

            print(
                f"[STAGE5 ORIENT] cid={cid} branch_id={branch_dbg} seg_idx={seg_idx_dbg} "
                f"mode={gt_match_mode} d_forward={d_forward:.4f} d_reverse={d_reverse:.4f} "
                f"flag={orient_flag}"
            )

            gtw = _resample_1d_to_len(gtw_local, L)

            # ----------------------------
            # DEBUG: classify nonfinite causes (segment-local)
            # ----------------------------
            _gt_padded_mask = np.zeros_like(np.asarray(gtw, float), dtype=bool)
            _gt_nonfinite_mask = ~np.isfinite(gtw)
            _pred_nonfinite_mask = ~np.isfinite(predw)

            _gt_padded_nonfinite = int(np.sum(_gt_padded_mask & _gt_nonfinite_mask))
            _gt_real_nonfinite = int(np.sum((~_gt_padded_mask) & _gt_nonfinite_mask))
            _pred_real_nonfinite = int(np.sum(_pred_nonfinite_mask))

            if (_gt_real_nonfinite > 0) or (_pred_real_nonfinite > 0):
                print(f"[STAGE5 NONFINITE DETAIL] cid={cid} seg_idx={seg_idx_dbg} branch_id={branch_dbg}")
                if _gt_real_nonfinite > 0:
                    bad = np.where((~_gt_padded_mask) & _gt_nonfinite_mask)[0][:10]
                    print(f"  GT NONFINITE (NOT padding): count={_gt_real_nonfinite} first_idx={bad.tolist()}")
                    for j in bad:
                        print(f"    j={int(j)} pred_global={int(max(0, pred_off - L) + j)} gtw={gtw[j]} predw={predw[j]}")
                if _pred_real_nonfinite > 0:
                    bad = np.where(_pred_nonfinite_mask)[0][:10]
                    print(f"  PRED NONFINITE: count={_pred_real_nonfinite} first_idx={bad.tolist()}")
                    for j in bad:
                        print(
                            f"    j={int(j)} pred_global={int(max(0, pred_off - L) + j)} predw={predw[j]} "
                            f"gtw={gtw[j]} padded_gt={bool(_gt_padded_mask[j])}"
                        )

            print(
                f"[STAGE5 TRACE] cid={cid} branch_id={branch_dbg} seg_idx={seg_idx_dbg} "
                f"source=local_segmatch pred_s0={int(max(0, pred_off - L))} L={L} "
                f"gt_match={gt_match_mode} gt_seg_len={gt_seg_len} "
                f"pred_nonfinite={_pred_real_nonfinite} "
                f"gt_nonfinite={int(np.sum(_gt_nonfinite_mask))} "
                f"(gt_padded={_gt_padded_nonfinite} gt_real={_gt_real_nonfinite})"
            )

            if DEBUG_TOPOLOGY_TRACE:
                _append_csv_row(
                    stage5_slice_csv,
                    [
                        str(cid),
                        str(branch_dbg),
                        str(seg_idx_dbg),
                        int(L),
                        int(max(0, pred_off - L)),
                        int(len(predw)),
                        int(len(gtw)),
                        int(_pred_real_nonfinite),
                        int(np.sum(_gt_nonfinite_mask)),
                        int(_gt_padded_nonfinite),
                        int(_gt_real_nonfinite),
                        int(len(pred_source)),
                        int(len(gtw_source)),
                        str(gt_match_mode),
                        int(gt_seg_len),
                    ],
                    header=[
                        "cid",
                        "branch_id",
                        "seg_idx",
                        "L_geom",
                        "pred_s0",
                        "predw_len",
                        "gtw_len",
                        "pred_nonfinite",
                        "gt_nonfinite",
                        "gt_padded_nonfinite",
                        "gt_real_nonfinite",
                        "pred_len",
                        "gt_len",
                        "gt_match_mode",
                        "gt_seg_len",
                    ],
                )

            if np.sum(np.isfinite(predw)) < 2:
                print(f"[STAGE5 TRACE] cid={cid} seg_idx={seg_idx_dbg} drop: predw has <2 finite samples")
                continue

            predw = np.asarray(predw, float)
            gtw = np.asarray(gtw, float)

            d = predw - gtw
            stage4_pairs.append((pts, d))
            final_pred_segs.append(pts)

            width_pairs.append({
                "image": base_name,
                "cid": str(cid),
                "member_id": str(m.get("atomic_id")) if isinstance(m, dict) else None,
                "crack_type": "combined",
                "midline_type": midline_type,
                "geometry_type": "derived",
                "bbox": crack.get("mask_bbox"),
                "pred_mask_bbox": crack.get("mask_bbox"),
                "pred_mask_crop": crack.get("mask_crop"),
                "pts": pts,
                "predw": predw,
                "gruthw": gtw,
                "gt_source": gt_match_mode,
                "gt_match_seg": np.asarray(gt_match_seg, float) if gt_match_seg is not None else None,
                "branch_id": m.get("branch_id"),
                "seg_idx": m.get("seg_idx"),
                "gt_mismatch": False,
                "gt_relation": "combined_vs_combined",
            })

        print(f"[STAGE5] width attachment complete - {len(stage4_pairs)} segments")

        # ============================================================
        # OPSEC PLOT — STAGE 5 FINAL GEOMETRY (DOMINANCE-RESOLVED)
        #   - NO dominance pruning logic here
        #   - Overlay can be categorical (GT-only / PRED-only / BOTH) like Stage 4
        # ============================================================
        assert bite_pruned_pred_segs is not None, "bite_pruned_pred_segs lost"
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
            from matplotlib.colors import ListedColormap
            import numpy as np
            import os

            bb = crack.get("mask_bbox")
            if bb:
                x, y, w, h = map(int, bb)
                pad = 25
                x0 = max(0, x - pad)
                y0 = max(0, y - pad)
                x1 = min(W, x + w + pad)
                y1 = min(H, y + h + pad)
            else:
                x, y, w, h = 0, 0, W, H
                x0, y0, x1, y1 = 0, 0, W, H

            def _split_pred_nonfinite(pts, predw, min_pts=2):
                """
                Split by predicted-width validity only.
                Geometry provenance must not depend on GT availability.
                """
                pts = np.asarray(pts, float)
                predw = np.asarray(predw, float).reshape(-1)
                n = min(len(pts), len(predw))
                if n < 2:
                    return [], []

                pts = pts[:n]
                predw = predw[:n]

                kept, undef_other = [], []
                bk, bo = [], []

                def _flush(buf, out):
                    if len(buf) >= min_pts:
                        out.append(np.asarray(buf, float))

                for i in range(n - 1):
                    p0, p1 = pts[i], pts[i + 1]
                    pf = np.isfinite(predw[i])

                    if pf:
                        if not bk:
                            bk.append(p0)
                        bk.append(p1)
                        _flush(bo, undef_other)
                        bo = []
                    else:
                        if not bo:
                            bo.append(p0)
                        bo.append(p1)
                        _flush(bk, kept)
                        bk = []

                _flush(bk, kept)
                _flush(bo, undef_other)
                return kept, undef_other

            # --------------------------------------------------
            # Build prediction plot segments (kept vs undef)
            # --------------------------------------------------
            pred_kept_segs = []
            pred_undef_other_segs = []
            pred_full_segs = []

            for wp in (width_pairs or []):
                if str(wp.get("cid", "")) != str(cid):
                    continue
                pts_ok = wp.get("pts", None)
                pw_ok = wp.get("predw", None)
                if pts_ok is None or pw_ok is None:
                    continue

                k, uo = _split_pred_nonfinite(pts_ok, pw_ok, min_pts=2)
                pred_kept_segs.extend(k)
                pred_undef_other_segs.extend(uo)
                pred_full_segs.append(np.asarray(pts_ok, float))
            # NO topology pruning allowed:
            # Stage 2 + Stage 4.5 fully define geometry survival.

            # --------------------------------------------------
            # prediction mask
            # --------------------------------------------------
            pred_mask_full = np.zeros((H, W), np.uint8)
            if "mask_crop" in crack and crack["mask_crop"] is not None and bb:
                mc = np.asarray(crack["mask_crop"], np.uint8)
                hh = min(h, mc.shape[0])
                ww = min(w, mc.shape[1])
                if hh > 0 and ww > 0:
                    pred_mask_full[y:y+hh, x:x+ww] = mc[:hh, :ww]

            # --------------------------------------------------
            # Dominance categorical overlay like Stage 4
            #   0 = none
            #   1 = GT-only
            #   2 = PRED-only
            #   3 = BOTH
            # --------------------------------------------------
            dom_label = np.zeros((H, W), dtype=np.uint8)
            all_bids_local = sorted(set(loss_masks_pred_by_branch.keys()) | set(loss_masks_gt_by_branch.keys()))
            for bid in all_bids_local:
                mg = loss_masks_gt_by_branch.get(bid)
                mp = loss_masks_pred_by_branch.get(bid)
                if mg is not None:
                    dom_label[mg.astype(bool)] |= 1
                if mp is not None:
                    dom_label[mp.astype(bool)] |= 2

            dom_crop = dom_label[y0:y1, x0:x1].astype(np.uint8)
            dom_masked = np.ma.array(dom_crop, mask=(dom_crop == 0))

            DOM_CMAP = ListedColormap([
                "#000000",  # masked
                "#e41a1c",  # GT-only
                "#377eb8",  # Pred-only
                "#984ea3",  # GT ∩ Pred
            ])

            # --------------------------------------------------
            # figure
            # --------------------------------------------------
            fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=200, sharex=True, sharey=True)

            axes[0].set_title("GT supervision (Stage 5, dominance-resolved)", fontsize=10)
            axes[1].set_title("Prediction (Stage 5, dominance-resolved)", fontsize=10)

            axes[0].imshow(mask_bin[y0:y1, x0:x1], cmap="gray", zorder=0)
            axes[1].imshow(pred_mask_full[y0:y1, x0:x1], cmap="gray", zorder=0)

            for ax in axes:
                ax.imshow(
                    dom_masked,
                    cmap=DOM_CMAP,
                    interpolation="nearest",
                    vmin=0,
                    vmax=3,
                    alpha=0.65,
                    zorder=1,
                )
                ax.axis("off")

            col_keep  = (0.2, 0.4, 0.8)
            col_undef = (0.6, 0.6, 0.6)
            col_bite  = (0.9, 0.6, 0.0)

            # --------------------------------------------------
            # GT plot (LEFT)
            # --------------------------------------------------
            for S in gt_stage5_segs:
                if S is None or len(S) < 2:
                    continue
                axes[0].plot(S[:, 0] - x0, S[:, 1] - y0, color=col_keep, lw=2.2, zorder=4)

            for S in bite_pruned_gt_segs:
                if S is None or len(S) < 2:
                    continue
                axes[0].plot(S[:, 0] - x0, S[:, 1] - y0, color=col_bite, lw=2.2, zorder=3)

            # --------------------------------------------------
            # PRED plot (RIGHT)
            # --------------------------------------------------
            # Geometry provenance baseline: draw full surviving Stage-5 geometry first.
            for S in pred_full_segs:
                if S is None or len(S) < 2:
                    continue
                axes[1].plot(S[:, 0] - x0, S[:, 1] - y0, color=col_keep, lw=2.5, zorder=4)

            for S in bite_pruned_pred_segs:
                if S is None or len(S) < 2:
                    continue
                axes[1].plot(S[:, 0] - x0, S[:, 1] - y0, color=col_bite, lw=2.0, zorder=3)

            for S in pred_undef_other_segs:
                if S is None or len(S) < 2:
                    continue
                axes[1].plot(S[:, 0] - x0, S[:, 1] - y0, color=col_undef, lw=2.2, zorder=4)

            for S in pred_kept_segs:
                if S is None or len(S) < 2:
                    continue
                axes[1].plot(S[:, 0] - x0, S[:, 1] - y0, color=col_keep, lw=1.0, alpha=0.35, zorder=5)

            for ax in axes:
                ax.add_patch(
                    plt.Rectangle((x - x0, y - y0), w, h, fill=False,
                                edgecolor="dodgerblue", lw=1.5)
                )

            legend_items = [
                Line2D([0],[0], color=col_keep,  lw=2.5, label="Stage-5 geometry (kept)"),
                Line2D([0],[0], color=col_undef, lw=2.2, label="Pred undef / other nonfinite"),
                Line2D([0],[0], color=col_bite,  lw=2.0, label="Dominance-bite (union)"),
                Line2D([0],[0], color="#e41a1c", lw=6, label="GT-only loss (overlay)"),
                Line2D([0],[0], color="#377eb8", lw=6, label="Pred-only loss (overlay)"),
                Line2D([0],[0], color="#984ea3", lw=6, label="GT ∩ Pred (overlay)"),
            ]
            axes[1].legend(handles=legend_items, loc="lower right", fontsize=7, framealpha=0.9)

            fig.suptitle(
                f"Stage-5 Geometry Provenance (Dominance-resolved @ 4.5) — cid={cid}",
                fontsize=11,
                fontweight="bold",
            )

            os.makedirs(opsec_dir, exist_ok=True)
            out = os.path.join(opsec_dir, f"stage5_geom_provenance_{cid}.png")
            fig.savefig(out, bbox_inches="tight", dpi=200)
            plt.close(fig)

            print(f"[OPSEC STAGE5 PLOT] wrote: {out}")

        except Exception as e:
            print(f"[OPSEC STAGE5 PLOT] skipped cid={cid}: {e}")
                                            
        # ============================================================
        # MIDLINE METRICS (COMBINED MODE)
        #   Compare BOTH:
        #     - pred MIDLINE stream vs GT midline
        #     - pred DERIVED stream vs GT midline
        # ============================================================
        if mode == "combined" and gt_entry is not None:
            try:
                from helpers.metrics import compute_midline_metrics
                import math
                import numpy as np

                if (
                    _should_midline_metrics(
                        run_mode=mode,
                        run_midline_type=midline_type,
                        geometry_type="orig",
                    )
                    and pred_mid_dom_segs
                    and gt_dom_segs
                ):
                    pred_mid = np.vstack([np.asarray(s, float) for s in pred_mid_dom_segs if s is not None and len(s) >= 2])
                    gt_mid   = np.vstack([np.asarray(s, float) for s in gt_dom_segs if s is not None and len(s) >= 2])

                    if len(pred_mid) >= 2 and len(gt_mid) >= 2:
                        mm = compute_midline_metrics(pred_mid, gt_mid)
                        ch  = float(mm.get("nn_mean_bidirectional", np.inf))
                        hd  = float(mm.get("hausdorff_max", np.inf))
                        cov = float(mm.get("coverage_min", 0.0))
                        score_mid = (math.log1p(max(ch, 0.0)) + 0.5 * math.log1p(max(hd, 0.0)) + (1.0 - float(np.clip(cov, 0.0, 1.0))))

                        bbox0 = crack.get("mask_bbox")
                        midline_metric_rows.append({
                            "image": base_name,
                            "crack_id": str(cid),
                            "crack_type": str(crack_type or mode),
                            "midline_type": str(midline_type or ""),
                            "variant_id": str(variant_id or "main"),
                            "geometry_type": "orig",
                            "variant_global_id": -1,
                            "os_mode": "combined",
                            "g11": np.nan, "g22": np.nan, "g33": np.nan,
                            "length_px": _linestring_length(gt_mid),
                            "bbox_area": float(bbox0[2] * bbox0[3]) if bbox0 else np.nan,
                            "nn_mean_bidirectional": ch,
                            "hausdorff_max": hd,
                            "coverage_min": cov,
                            "score_mid": score_mid,
                            "frechet_discrete_ds": mm.get("frechet_discrete_ds"),
                            "mean_tan_angle_error_deg": mm.get("mean_tan_angle_error_deg"),
                            "relative_length_error": mm.get("relative_length_error"),
                            "orth_mean": mm.get("orth_mean"),
                            "orth_std": mm.get("orth_std"),
                            "signed_bias_z": mm.get("signed_bias_z"),
                            "curvature_rms_auto": mm.get("curvature_rms_auto"),
                            "curvature_rms_manual": mm.get("curvature_rms_manual"),
                            "curvature_rms_ratio": mm.get("curvature_rms_ratio"),
                        })

                if (
                    _should_midline_metrics(
                        run_mode=mode,
                        run_midline_type=midline_type,
                        geometry_type="derived",
                    )
                    and pred_der_dom_segs
                    and gt_dom_segs
                ):
                    pred_der = np.vstack([np.asarray(s, float) for s in pred_der_dom_segs if s is not None and len(s) >= 2])
                    gt_mid   = np.vstack([np.asarray(s, float) for s in gt_dom_segs if s is not None and len(s) >= 2])

                    if len(pred_der) >= 2 and len(gt_mid) >= 2:
                        mm = compute_midline_metrics(pred_der, gt_mid)
                        ch  = float(mm.get("nn_mean_bidirectional", np.inf))
                        hd  = float(mm.get("hausdorff_max", np.inf))
                        cov = float(mm.get("coverage_min", 0.0))
                        score_mid = (math.log1p(max(ch, 0.0)) + 0.5 * math.log1p(max(hd, 0.0)) + (1.0 - float(np.clip(cov, 0.0, 1.0))))

                        bbox0 = crack.get("mask_bbox")
                        midline_metric_rows.append({
                            "image": base_name,
                            "crack_id": str(cid),
                            "crack_type": str(crack_type or mode),
                            "midline_type": str(midline_type or ""),
                            "variant_id": str(variant_id or "main"),
                            "geometry_type": "derived",
                            "variant_global_id": -1,
                            "os_mode": "combined",
                            "g11": np.nan, "g22": np.nan, "g33": np.nan,
                            "length_px": _linestring_length(gt_mid),
                            "bbox_area": float(bbox0[2] * bbox0[3]) if bbox0 else np.nan,
                            "nn_mean_bidirectional": ch,
                            "hausdorff_max": hd,
                            "coverage_min": cov,
                            "score_mid": score_mid,
                            "frechet_discrete_ds": mm.get("frechet_discrete_ds"),
                            "mean_tan_angle_error_deg": mm.get("mean_tan_angle_error_deg"),
                            "relative_length_error": mm.get("relative_length_error"),
                            "orth_mean": mm.get("orth_mean"),
                            "orth_std": mm.get("orth_std"),
                            "signed_bias_z": mm.get("signed_bias_z"),
                            "curvature_rms_auto": mm.get("curvature_rms_auto"),
                            "curvature_rms_manual": mm.get("curvature_rms_manual"),
                            "curvature_rms_ratio": mm.get("curvature_rms_ratio"),
                        })

            except Exception as e:
                print(f"[MIDLINE METRICS] skipped cid={cid}: {e}")

    # ============================================================
    # Part 2: FAIR WIDTH METRICS (ARCLENGTH RESAMPLING + LENGTH-WEIGHTED STATS)
    #   - Postprocess Stage-5 outputs for BOTH atomic + combined
    #   - Computes length-weighted RMSE/MAE/Bias per (image,cid,crack_type,midline_type)
    #   - Produces committee-friendly plots in:
    #       <metrics_dir>/<midline_type>/compare_widths_debug/<mode>/part2/...
    #   - Swaps final compare-width plotting inputs to the Stage-6 resampled geometry
    #
    # NOTES:
    #   - Resampling is applied to the *measurement samples* (d(s)=pred-gt), not to GT geometry itself.
    #   - If you also pass per-sample pred/gt widths into width_pairs as "predw" and "gruthw", Part 2 will
    #     resample and plot those too (so you can show â€œeffect on GT vs predâ€ explicitly).
    # ============================================================
    
    try:
        import os, math
        import numpy as np

        # ------------------------------------------------------------
        # Ensure containers exist (you should also define these in your accumulators)
        # ------------------------------------------------------------
        if "width_pairs" not in locals():
            width_pairs = []
        if "width_metric_rows" not in locals():
            width_metric_rows = []

        # ------------------------------------------------------------
        # Debug folder structure (split by mode)
        # ------------------------------------------------------------
        debug_root = os.path.join(metrics_dir, midline_type or "unknown", "compare_widths_debug")
        os.makedirs(debug_root, exist_ok=True)

        debug_mode_dir = os.path.join(debug_root, str(mode), file_tag)
        os.makedirs(debug_mode_dir, exist_ok=True)

        part2_dir = os.path.join(debug_mode_dir, "part2")
        os.makedirs(part2_dir, exist_ok=True)

        part2_metrics_dir  = os.path.join(part2_dir, "metrics")
        part2_resample_dir = os.path.join(part2_dir, "resample")
        os.makedirs(part2_metrics_dir, exist_ok=True)
        os.makedirs(part2_resample_dir, exist_ok=True)

        if not width_pairs:
            raise RuntimeError("[PART2 FATAL] width_pairs is empty. Stage-5 must populate derived width_pairs.")

        def _rebuild_pred_mask_from_wp(wp_obj, H_full, W_full):
            pm = np.zeros((H_full, W_full), np.uint8)
            bb = wp_obj.get("pred_mask_bbox") or wp_obj.get("bbox")
            crop = wp_obj.get("pred_mask_crop")
            if bb is None or crop is None:
                raise RuntimeError(
                    f"[PART2 FATAL] missing pred mask info for plotting: cid={wp_obj.get('cid','')}"
                )
            x, y, w, h = map(int, bb)
            crop = np.asarray(crop, np.uint8)
            if crop.ndim != 2:
                raise RuntimeError(
                    f"[PART2 FATAL] pred_mask_crop invalid ndim={crop.ndim} for cid={wp_obj.get('cid','')}"
                )
            hh = min(h, crop.shape[0])
            ww = min(w, crop.shape[1])
            if hh <= 0 or ww <= 0:
                return pm
            pm[y:y + hh, x:x + ww] = (crop[:hh, :ww] > 0).astype(np.uint8)
            return pm

        # ------------------------------------------------------------
        # Helpers (local, safe)
        # ------------------------------------------------------------
        def _contiguous_true_runs(mask_bool):
            mask_bool = np.asarray(mask_bool, bool)
            if mask_bool.size == 0:
                return []
            runs = []
            in_run = False
            start = 0
            for i, v in enumerate(mask_bool):
                if v and not in_run:
                    in_run = True
                    start = i
                elif (not v) and in_run:
                    runs.append((start, i - 1))
                    in_run = False
            if in_run:
                runs.append((start, len(mask_bool) - 1))
            return runs

        def _length_weighted_err_stats(d_vals, ds_w):
            """
            d_vals: length N
            ds_w:   length N (weights). In our usage: d_rs[:-1] with ds = diff(s_rs).
            """
            d_vals = np.asarray(d_vals, float)
            ds_w   = np.asarray(ds_w, float)
            n = min(len(d_vals), len(ds_w))
            if n <= 0:
                return {"bias": np.nan, "mae": np.nan, "rmse": np.nan, "p95_abs": np.nan, "median_abs": np.nan}

            d_vals = d_vals[:n]
            ds_w   = ds_w[:n]
            ok = np.isfinite(d_vals) & np.isfinite(ds_w) & (ds_w > 0)
            if not np.any(ok):
                return {"bias": np.nan, "mae": np.nan, "rmse": np.nan, "p95_abs": np.nan, "median_abs": np.nan}

            d = d_vals[ok]
            w = ds_w[ok]
            W = float(np.sum(w) + 1e-12)

            bias = float(np.sum(d * w) / W)
            mae  = float(np.sum(np.abs(d) * w) / W)
            mse  = float(np.sum((d ** 2) * w) / W)
            rmse = float(math.sqrt(max(mse, 0.0)))

            absd = np.abs(d[np.isfinite(d)])
            p95  = float(np.percentile(absd, 95)) if absd.size else np.nan
            med  = float(np.median(absd)) if absd.size else np.nan

            return {"bias": bias, "mae": mae, "rmse": rmse, "p95_abs": p95, "median_abs": med}

      
        # ============================================================
        # Part 2 main: resample segments, then compute widths + diffs,
        # and compute length-weighted stats (metrics == visualization)
        # ============================================================
        coords_part2, diffs_part2, bboxes_part2 = [], [], []
        part2_cache = []  # list of cache_item dicts for explainers

        ds_target_px = 1.0  # knob later
        per_crack = {}

        print("\n[PART2 DEBUG] ===============================")
        print("[PART2 DEBUG] ENTER Part 2")
        print(f"[PART2 DEBUG] width_pairs count = {len(width_pairs or [])}")
        print(f"[PART2 DEBUG] ds_target_px = {ds_target_px}")
        print("[PART2 DEBUG] ===============================")

        part2_branch_debug = {}

        def _resample_polyline_to_len_part2(xy, L):
            xy = np.asarray(xy, float)
            L = int(L)
            if L <= 0:
                return np.empty((0, 2), float)
            if xy.ndim != 2 or xy.shape[1] != 2:
                out = np.empty((L, 2), float)
                out[:] = np.nan
                return out
            m = np.isfinite(xy).all(axis=1)
            xy = xy[m]
            if len(xy) == 0:
                out = np.empty((L, 2), float)
                out[:] = np.nan
                return out
            if len(xy) == 1:
                return np.repeat(xy[:1], L, axis=0)
            if len(xy) == L:
                return xy.astype(float, copy=False)
            d = np.sqrt(((xy[1:] - xy[:-1]) ** 2).sum(axis=1))
            s = np.concatenate([[0.0], np.cumsum(d)])
            if not np.isfinite(s[-1]) or s[-1] <= 1e-12:
                return np.repeat(xy[:1], L, axis=0)
            t = np.linspace(0.0, s[-1], num=L)
            x = np.interp(t, s, xy[:, 0])
            y = np.interp(t, s, xy[:, 1])
            return np.column_stack([x, y]).astype(float, copy=False)

        # ============================================================
        # Part 2: per-width-pair processing
        #   - Build ORIGINAL-domain signals (plot-only)
        #   - Resample into authoritative domain
        #   - Compute d ONLY in resampled domain
        #   - Cache ORIGINAL using arclength windowing (no index clipping)
        # ============================================================
        for wp in (width_pairs or []):
            pts     = wp.get("pts", None)
            predw   = wp.get("predw", None)
            gtruthw = wp.get("gruthw", None)
            image = str(wp.get("image", base_name if "base_name" in locals() else ""))
            cid_s = str(wp.get("cid", ""))
            ctype = str(wp.get("crack_type", mode))
            mtype = str(wp.get("midline_type", midline_type))
            gtype = str(wp.get("geometry_type", "derived"))
            bbox  = wp.get("bbox", None)

            print(
                f"[PART2 DEBUG] ▶ wp: "
                f"cid={wp.get('cid','')}, "
                f"type={wp.get('crack_type',mode)}, "
                f"geom={wp.get('geometry_type','derived')}, "
                f"midline={wp.get('midline_type',midline_type)}, "
                f"pts={None if pts is None else len(pts)}, "
                f"predw={None if predw is None else len(predw)}, "
                f"gtw={'None' if gtruthw is None else len(gtruthw)}"
            )

            if gtype != "derived":
                raise RuntimeError(
                    f"[PART2 FATAL] non-derived geometry slipped into Part2: "
                    f"cid={cid_s} crack_type={ctype} midline_type={mtype} geometry_type={gtype}"
                )
            if pts is None or predw is None:
                raise RuntimeError(
                    f"[PART2 FATAL] missing pts/predw: cid={cid_s} "
                    f"pts={None if pts is None else len(pts)} "
                    f"predw={None if predw is None else len(predw)}"
                )
            if gtruthw is None:
                raise RuntimeError(
                    f"[PART2 FATAL] gtruthw is None (should be produced in Stage5): cid={cid_s}"
                )

            pts = np.asarray(pts, float)
            predw = np.asarray(predw, float)
            gtruthw = np.asarray(gtruthw, float)

            # Geometry + predw must align 1:1 (Stage5 ensures this)
            n_geom = min(len(pts), len(predw))
            if n_geom < 2:
                raise RuntimeError(
                    f"[PART2 FATAL] <2 samples after geom/pred trim: cid={cid_s} "
                    f"(pts={len(pts)}, predw={len(predw)}, gtw={len(gtruthw)})"
                )

            pts = pts[:n_geom]
            predw = predw[:n_geom]

            # GT stream does not control geometry length.
            # If shorter, pad with NaN; if longer, clip to geometry.
            if len(gtruthw) < n_geom:
                gtmp = np.empty((n_geom,), float)
                gtmp[:] = np.nan
                if len(gtruthw) > 0:
                    gtmp[:len(gtruthw)] = gtruthw
                gtruthw = gtmp
            else:
                gtruthw = gtruthw[:n_geom]

            n = n_geom

            s_full = arclen_s(pts)
            if len(s_full) < 2:
                raise RuntimeError(f"[PART2 FATAL] invalid arclength for cid={cid_s}")

            total_len = float(s_full[-1] - s_full[0])
            if not np.isfinite(total_len) or total_len <= 0:
                raise RuntimeError(f"[PART2 FATAL] non-finite total_len for cid={cid_s}: {total_len}")

            # ------------------------------------------------------------
            # ORIGINAL DOMAIN (plot-only)
            # ------------------------------------------------------------
            s_orig = arclen_s(pts)
            predw_orig = np.asarray(predw, float)

            gtruthw_orig = np.asarray(gtruthw, float)

            m0 = min(len(s_orig), len(predw_orig), len(gtruthw_orig))
            if m0 < 2:
                raise RuntimeError(f"[PART2 FATAL] too few aligned ORIGINAL samples for cid={cid_s}")

            s_orig       = np.asarray(s_orig[:m0], float)
            predw_orig   = np.asarray(predw_orig[:m0], float)
            gtruthw_orig = np.asarray(gtruthw_orig[:m0], float)
            
            # ============================================================
            # DISTRIBUTIONAL WIDTH SUMMARY (Regime A)
            #   - Geometry-agnostic
            #   - MUST be called BEFORE resampling
            #   - Uses ORIGINAL-domain widths only
            # ============================================================
            export_width_distribution_summary(
                pred_widths = predw_orig,
                gt_widths   = gtruthw_orig,
                out_dir     = metrics_dir,   # image-level metrics root (NOT part2)

                # ---- identity ----
                image_name  = image,
                variant     = wp.get("variant", variant_id),
                midline_type= mtype,
                crack_type  = ctype,

                # ---- NEW: crack identity ----
                cid         = cid_s,                 # atomic crack id
                group_id    = wp.get("member_id", None),  # combined branch id (None for atomic)

                # ---- GT semantics ----
                gt_tier     = (
                    "atomic"
                    if mode == "atomic"
                    else ("combined_filtered" if write_part2_tables else "combined_unfiltered")
                ),
                gt_pairing  = wp.get("variant", variant_id),
                filtered    = (mode == "combined"),

                # ---- bookkeeping ----
                method_family = (
                    "baseline"
                    if str(wp.get("variant", "")).startswith("medial")
                    else "model"
                ),
            )


            # ------------------------------------------------------------
            # RESAMPLE GEOMETRY + WIDTHS (authoritative domain)
            # ------------------------------------------------------------
            pts_rs, predw_rs, gtruthw_rs = resample_by_arclength(
                pts, predw, gtruthw,
                ds_target=ds_target_px,
                min_pts=2,
                preserve_endpoints=True,
                fastpath=True,
            )

            if pts_rs is None or len(pts_rs) < 2:
                raise RuntimeError(f"[PART2 FATAL] resample failed for cid={cid_s}")

            mrs = min(len(pts_rs), len(predw_rs), len(gtruthw_rs))
            if mrs < 2:
                raise RuntimeError(f"[PART2 FATAL] resampled arrays too short for cid={cid_s}")

            pts_rs     = np.asarray(pts_rs[:mrs], float)
            predw_rs   = np.asarray(predw_rs[:mrs], float)
            gtruthw_rs = np.asarray(gtruthw_rs[:mrs], float)

            if (
                DEBUG_CORRESPONDENCE_ON
                and str(cid_s) == str(DEBUG_CORRESPONDENCE_CID)
            ):
                bkey = str(wp.get("branch_id", "NA"))
                bd = part2_branch_debug.setdefault(
                    bkey,
                    {
                        "pts_list": [],
                        "gt_pts_list": [],
                        "predw_list": [],
                        "gtw_list": [],
                    },
                )
                bd["pts_list"].append(np.asarray(pts_rs, float))
                bd["predw_list"].append(np.asarray(predw_rs, float))
                bd["gtw_list"].append(np.asarray(gtruthw_rs, float))
                gt_match_seg = wp.get("gt_match_seg", None)
                gt_pts_rs = _resample_polyline_to_len_part2(gt_match_seg, mrs)
                bd["gt_pts_list"].append(gt_pts_rs)
                d_fwd, d_rev, flag = _orient_cost(np.asarray(pts_rs, float), np.asarray(gt_pts_rs, float))
                print(
                    f"[ORIENT DBG] cid={cid_s} tag=part2_resampled branch={wp.get('branch_id', 'NA')} "
                    f"seg_idx={wp.get('seg_idx', 'NA')} n={mrs} "
                    f"d_fwd={d_fwd:.4f} d_rev={d_rev:.4f} flag={flag}"
                )

            # ------------------------------------------------------------
            # Width error ONLY defined here
            # ------------------------------------------------------------
            d_rs = predw_rs - gtruthw_rs

            # ============================================================
            # DEBUG: diagnose non-finite + clipping behavior
            # ============================================================
            def _count_invalid(arr):
                arr = np.asarray(arr)
                return {
                    "nan": int(np.sum(np.isnan(arr))),
                    "posinf": int(np.sum(arr == np.inf)),
                    "neginf": int(np.sum(arr == -np.inf)),
                    "finite": int(np.sum(np.isfinite(arr))),
                    "total": int(arr.size),
                }

            pred_stats = _count_invalid(predw_rs)
            gt_stats   = _count_invalid(gtruthw_rs)
            d_stats    = _count_invalid(d_rs)

            print(f"[PART2 VALIDITY] cid={cid_s}")
            print(f"  predw_rs: {pred_stats}")
            print(f"  gtruthw_rs: {gt_stats}")
            print(f"  d_rs: {d_stats}")

            if (
                pred_stats["nan"] > 0 or pred_stats["posinf"] > 0 or pred_stats["neginf"] > 0 or
                gt_stats["nan"] > 0 or gt_stats["posinf"] > 0 or gt_stats["neginf"] > 0
            ):
                invalid_mask = (
                    ~np.isfinite(predw_rs) |
                    ~np.isfinite(gtruthw_rs) |
                    ~np.isfinite(d_rs)
                )
                bad_indices = np.where(invalid_mask)[0]

                print(f"[PART2 VALIDITY]   -> invalid sample count = {len(bad_indices)}")

                for idx in bad_indices[:10]:
                    print(
                        f"    idx={idx} | "
                        f"xy={pts_rs[idx]} | "
                        f"pred={predw_rs[idx]} | "
                        f"gt={gtruthw_rs[idx]} | "
                        f"d={d_rs[idx]}"
                    )

                try:
                    s_dbg = arclen_s(pts_rs)
                    for idx in bad_indices[:5]:
                        print(f"    idx={idx} | s={s_dbg[idx]:.3f}px")
                except Exception:
                    pass

            finite_mask = (
                np.isfinite(d_rs) &
                np.isfinite(predw_rs) &
                np.isfinite(gtruthw_rs)
            )
            if np.sum(finite_mask) < 2:
                print(
                    f"[PART2 WARN] cid={cid_s} no finite overlap after resample "
                    f"(finite={int(np.sum(finite_mask))}/{len(finite_mask)})"
                )
            runs = _contiguous_true_runs(finite_mask)

            print(
                f"[PART2 DEBUG]   runs found = {len(runs)} "
                f"(finite samples = {int(np.sum(finite_mask))})"
            )

            finite_len = 0.0
            run_stats = []

            cache_item = {
                "image": image,
                "cid": cid_s,
                "crack_type": ctype,
                "midline_type": mtype,
                "geometry_type": gtype,
                "bbox": bbox,
                "pred_mask_bbox": wp.get("pred_mask_bbox") or wp.get("bbox"),
                "pred_mask_crop": wp.get("pred_mask_crop"),
                "pred_mask_full": _rebuild_pred_mask_from_wp(wp, H, W),
                "runs": [],
            }

            # ------------------------------------------------------------
            # Process each finite run
            # ------------------------------------------------------------
            for (i0, i1) in runs:
                if i1 - i0 + 1 < 2:
                    continue

                # -------- RESAMPLED DOMAIN (authoritative) --------
                pts_run   = np.asarray(pts_rs[i0:i1 + 1], float)
                d_run     = np.asarray(d_rs[i0:i1 + 1], float)
                predw_run = np.asarray(predw_rs[i0:i1 + 1], float)
                gtw_run   = np.asarray(gtruthw_rs[i0:i1 + 1], float)

                s_run = arclen_s(pts_run)
                if len(s_run) < 2:
                    continue

                ds_w = np.diff(s_run)
                runL = float(np.sum(ds_w))
                if not np.isfinite(runL) or runL <= 0:
                    continue

                finite_len += runL

                coords_part2.append(pts_run)
                diffs_part2.append(d_run)
                if bbox is not None:
                    bboxes_part2.append(bbox)

                st = _length_weighted_err_stats(d_run[:-1], ds_w)
                st["run_len_px"] = runL
                run_stats.append(st)

                # -------- ORIGINAL DOMAIN (normalized arclength window) --------
                # Normalize arclengths so ORIGINAL and RESAMPLED align parametrically
                s_orig_norm = s_orig / max(s_orig[-1], 1e-9)
                s_run_norm  = s_run  / max(s_run[-1],  1e-9)

                lo = float(s_run_norm[0])
                hi = float(s_run_norm[-1])

                m_orig = (s_orig_norm >= lo - 1e-6) & (s_orig_norm <= hi + 1e-6)
                if np.count_nonzero(m_orig) < 2:
                    m_orig = np.ones_like(s_orig, dtype=bool)

                pts_orig_run   = np.asarray(pts[m_orig], float)
                predw_orig_run = np.asarray(predw_orig[m_orig], float)
                gtw_orig_run   = np.asarray(gtruthw_orig[m_orig], float)
                s_orig_run     = np.asarray(s_orig[m_orig], float)

                # Fallback (should rarely trigger, but keeps plots alive)
                if pts_orig_run.shape[0] < 2:
                    pts_orig_run   = np.asarray(pts, float)
                    predw_orig_run = np.asarray(predw_orig, float)
                    gtw_orig_run   = np.asarray(gtruthw_orig, float)
                    s_orig_run     = np.asarray(s_orig, float)

                # -------- CACHE --------
                cache_item["runs"].append({
                    # ORIGINAL (plot + sampling diagnostics)
                    "pts":    pts_orig_run,
                    "s":      s_orig_run,
                    "predw":  predw_orig_run,
                    "gruthw": gtw_orig_run,

                    # RESAMPLED (metrics domain)
                    "pts_rs":     pts_run,
                    "s_rs":       s_run,
                    "predw_rs":   predw_run,
                    "gtruthw_rs": gtw_run,
                    "d_rs":       d_run,

                    "run_len_px": runL,
                })

                # -------- per-point rows (resampled only) --------
                for (x, y), dw, gtw_i, pw_i in zip(pts_run, d_run, gtw_run, predw_run):
                    if not (np.isfinite(dw) and np.isfinite(gtw_i) and np.isfinite(pw_i)):
                        continue
                    rows.append({
                        "x": float(x),
                        "y": float(y),
                        "gt_width_px": float(gtw_i),
                        "pred_width_px": float(pw_i),
                        "width_diff_px": float(dw),
                        "cid": cid_s,
                        "crack_type": ctype,
                        "midline_type": mtype,
                        "geometry_type": gtype,
                    })

            if cache_item["runs"]:
                part2_cache.append(cache_item)

            # ------------------------------------------------------------
            # Per-crack aggregation
            # ------------------------------------------------------------
            #key = (image, cid_s, ctype, mtype)
            vtag = str(wp.get("variant", variant_id))
            key = (image, vtag, cid_s, ctype, mtype, gtype)

            '''if key not in per_crack:
                per_crack[key] = {
                    "total_len_px": total_len,
                    "finite_len_px": 0.0,
                    "sum_bias_L": 0.0,
                    "sum_mae_L": 0.0,
                    "sum_mse_L": 0.0,
                    "bbox": bbox,
                }'''
            if key not in per_crack:
                per_crack[key] = {
                    "total_pred_len_px": 0.0,   # SUM of all candidate midline length
                    "finite_len_px": 0.0,       # SUM of finite-evaluable length
                    "sum_bias_L": 0.0,
                    "sum_mae_L": 0.0,
                    "sum_mse_L": 0.0,
                    "bbox": bbox,
                }

            bin_ = per_crack[key]
            bin_["total_pred_len_px"] += max(total_len, 0.0)
            bin_["finite_len_px"] += finite_len


#            bin_ = per_crack[key]
#            bin_["total_len_px"] = max(bin_["total_len_px"], total_len)
#            bin_["finite_len_px"] += finite_len

            for st in run_stats:
                L = st["run_len_px"]
                if not np.isfinite(L) or L <= 0:
                    continue
                bin_["sum_bias_L"] += st["bias"] * L
                bin_["sum_mae_L"]  += st["mae"]  * L
                bin_["sum_mse_L"]  += (st["rmse"] ** 2) * L

        if DEBUG_CORRESPONDENCE_ON and part2_branch_debug:
            for bkey, bd in part2_branch_debug.items():
                _debug_plot_correspondence_single(
                    bd["pts_list"],
                    bd["predw_list"],
                    bd["gtw_list"],
                    cid=DEBUG_CORRESPONDENCE_CID,
                    branch_id=bkey,
                    seg_idx="part2_resampled",
                    out_dir=part2_resample_dir,
                    stride=20,
                    gt_pts=bd["gt_pts_list"],
                )

        # ------------------------------------------------------------
        # Emit per-crack metric rows
        # ------------------------------------------------------------
        
        for (image, vtag, cid_s, ctype, mtype, gtype), bin_ in per_crack.items():
            finL = float(bin_["finite_len_px"])
            totL = float(bin_["total_pred_len_px"])
            if finL <= 0 or totL <= 0:
                continue

            width_metric_rows.append({
                "image": image,
                "variant": vtag,
                "crack_id": cid_s,
                "crack_type": ctype,
                "midline_type": mtype,
                "geometry_type": gtype,

                # lengths / coverage
                "total_pred_len_px": totL,
                "finite_len_px": finL,
                "coverage_pred_len_frac": finL / (totL + 1e-12),

                # legacy key for older plotting (keep it)
                "finite_len_frac": finL / (totL + 1e-12),

                "bbox_area": (
                    bin_["bbox"][2] * bin_["bbox"][3]
                    if bin_["bbox"] is not None else np.nan
                ),

                # length-weighted error
                "width_bias_L": bin_["sum_bias_L"] / (finL + 1e-12),
                "width_mae_L":  bin_["sum_mae_L"]  / (finL + 1e-12),
                "width_rmse_L": np.sqrt(bin_["sum_mse_L"] / (finL + 1e-12)),
            })
            
        # ------------------------------------------------------------
        # Write per-crack metric table (accuracy + coverage)
        # ------------------------------------------------------------
        if write_part2_tables and width_metric_rows:
            try:
                import pandas as pd
                out_tbl = os.path.join(part2_metrics_dir, f"{base_name}__{variant_id}__width_metrics_{mode}_{midline_type}.csv")
                pd.DataFrame(width_metric_rows).to_csv(out_tbl, index=False)
                print(f"[PART2] wrote width metric table: {out_tbl}")
            except Exception as e:
                print(f"[PART2] failed to write width metric table: {e}")

            

        # ------------------------------------------------------------
        # IMPORTANT: overwrite legacy plot buffers
        # ------------------------------------------------------------
        coords = coords_part2
        diffs  = diffs_part2
        bboxes = bboxes_part2

                                        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import TwoSlopeNorm

            # ------------------------------------------------------------
            # Select rows for current scope
            # ------------------------------------------------------------
            rows_here = [
                r for r in width_metric_rows
                if str(r.get("crack_type", "")) == str(mode)
                and str(r.get("midline_type", "")) == str(midline_type)
                and str(r.get("geometry_type", "derived")) == "derived"
            ]
            if not rows_here:
                rows_here = list(width_metric_rows)

            # keep finite rows
            rows_here = [
                r for r in rows_here
                if np.isfinite(r.get("width_rmse_L", np.nan))
                and np.isfinite(r.get("width_mae_L", np.nan))
                and np.isfinite(r.get("width_bias_L", np.nan))
                and float(r.get("finite_len_px", 0.0)) > 0.0
            ]

            if rows_here:
                Ls   = np.asarray([float(r["finite_len_px"]) for r in rows_here], float)
                rmse = np.asarray([float(r["width_rmse_L"]) for r in rows_here], float)
                mae  = np.asarray([float(r["width_mae_L"])  for r in rows_here], float)
                bias = np.asarray([float(r["width_bias_L"]) for r in rows_here], float)

                Lsum = float(np.sum(Ls) + 1e-12)

                global_rmse = float(np.sqrt(np.sum((rmse ** 2) * Ls) / Lsum))
                global_mae  = float(np.sum(mae * Ls) / Lsum)
                global_bias = float(np.sum(bias * Ls) / Lsum)

                # TopK by RMSE
                rows_here_sorted = sorted(
                    rows_here,
                    key=lambda r: float(r.get("width_rmse_L", -1e9)),
                    reverse=True
                )
                topK = rows_here_sorted[:15]

                labels, rmse_v, mae_v, bias_v, len_v = [], [], [], [], []
                for r in topK:
                    cid0 = r["crack_id"]
                    gtp  = str(r.get("geometry_type", "derived"))
                    Lf   = float(r["finite_len_px"])
                    cov  = float(r.get("finite_len_frac", np.nan))
                    labels.append(f"cid {cid0}/{gtp}  (L={Lf:.0f}px, cov={cov:.2f})")
                    rmse_v.append(float(r["width_rmse_L"]))
                    mae_v.append(float(r["width_mae_L"]))
                    bias_v.append(float(r["width_bias_L"]))
                    len_v.append(Lf)

                y = np.arange(len(labels))

                # (A) 4-panel metrics figure
                fig, axes = plt.subplots(1, 4, figsize=(17.5, 5.0), dpi=200, sharey=True)

                axes[0].barh(y, rmse_v)
                axes[0].axvline(global_rmse, lw=2, linestyle="--")
                axes[0].set_title("Width RMSE (length-weighted)", fontsize=10, fontweight="bold")
                axes[0].set_xlabel("px", fontsize=9)
                axes[0].grid(True, axis="x", alpha=0.25)

                axes[1].barh(y, mae_v)
                axes[1].axvline(global_mae, lw=2, linestyle="--")
                axes[1].set_title("Width MAE (length-weighted)", fontsize=10, fontweight="bold")
                axes[1].set_xlabel("px", fontsize=9)
                axes[1].grid(True, axis="x", alpha=0.25)

                axes[2].barh(y, bias_v)
                axes[2].axvline(0.0, lw=1.5, linestyle="-")
                axes[2].axvline(global_bias, lw=2, linestyle="--")
                axes[2].set_title("Width Bias (length-weighted)", fontsize=10, fontweight="bold")
                axes[2].set_xlabel("px (pred − gt)", fontsize=9)
                axes[2].grid(True, axis="x", alpha=0.25)

                axes[3].barh(y, len_v)
                axes[3].set_title("Finite length used (weight)", fontsize=10, fontweight="bold")
                axes[3].set_xlabel("px", fontsize=9)
                axes[3].grid(True, axis="x", alpha=0.25)

                axes[0].set_yticks(y)
                axes[0].set_yticklabels(labels, fontsize=8)
                axes[0].invert_yaxis()

                fig.suptitle(
                    f"Part 2 — Width error metrics (fair arclength sampling) — {mode} / {midline_type}\n"
                    f"Global length-weighted means: RMSE={global_rmse:.3f}px, MAE={global_mae:.3f}px, Bias={global_bias:.3f}px",
                    fontsize=11,
                    fontweight="bold",
                )

                out = os.path.join(part2_metrics_dir, f"part2_topK_width_metrics_{mode}_{midline_type}.png")
                fig.savefig(out, bbox_inches="tight", dpi=200)
                plt.close(fig)
                print(f"[PART2] wrote: {out}")

                # ------------------------------------------------------------
                # (B) Resampling explainers corrected semantics
                # ------------------------------------------------------------

                rows_here_sorted = sorted(
                    rows_here,
                    key=lambda r: float(r.get("width_rmse_L", 0.0)),
                    reverse=True
                )
                if not rows_here_sorted:
                    raise RuntimeError("[PART2] no rows for resampling explainers")

                # ------------------------------------------------------------
                # Identify WORST CID (by RMSE)
                # ------------------------------------------------------------
                worst_row = rows_here_sorted[0]
                worst_cid = str(worst_row["crack_id"])
                worst_ct  = str(worst_row["crack_type"])
                worst_mt  = str(worst_row["midline_type"])
                worst_gt  = str(worst_row.get("geometry_type", "derived"))

                def _cache_for_cid(cid0, ctype0, mtype0, gtype0):
                    return [
                        it for it in part2_cache
                        if str(it.get("cid","")) == str(cid0)
                        and str(it.get("crack_type","")) == str(ctype0)
                        and str(it.get("midline_type","")) == str(mtype0)
                        and str(it.get("geometry_type","derived")) == str(gtype0)
                        and it.get("runs")
                    ]

                # ------------------------------------------------------------
                # Collect runs for WORST CID
                # ------------------------------------------------------------
                items = _cache_for_cid(worst_cid, worst_ct, worst_mt, worst_gt)
                if not items:
                    raise RuntimeError(f"[PART2] no cache items for worst cid={worst_cid}")

                worst_cid_runs = []
                bbox0 = None
                for it in items:
                    bbox0 = bbox0 or it.get("bbox", None)
                    worst_cid_runs.extend(it["runs"])

                if not worst_cid_runs:
                    raise RuntimeError(f"[PART2] no runs for worst cid={worst_cid}")

                # ------------------------------------------------------------
                # Crop for WORST CID
                # ------------------------------------------------------------
                if bbox0 and len(bbox0) >= 4:
                    x, y, w, h = map(int, bbox0[:4])
                    pad = 25
                    crop_worst = (
                        max(0, x - pad),
                        max(0, y - pad),
                        min(W, x + w + pad),
                        min(H, y + h + pad),
                    )
                else:
                    crop_worst = (0, 0, W, H)

                # ------------------------------------------------------------
                # Crop for ALL CIDs (GLOBAL)
                # ------------------------------------------------------------
                xs, ys, xe, ye = [], [], [], []
                for it in part2_cache:
                    bb = it.get("bbox", None)
                    if bb and len(bb) >= 4:
                        x, y, w, h = map(int, bb[:4])
                        xs.append(x)
                        ys.append(y)
                        xe.append(x + w)
                        ye.append(y + h)

                if xs:
                    pad = 25
                    crop_all = (
                        max(0, min(xs) - pad),
                        max(0, min(ys) - pad),
                        min(W, max(xe) + pad),
                        min(H, max(ye) + pad),
                    )
                else:
                    crop_all = (0, 0, W, H)


                # ------------------------------------------------------------
                # (B1) COMBINED AGGREGATED DIAGNOSTIC (UNCHANGED)
                # ------------------------------------------------------------
                if mode == "combined":
                    d_all = []
                    run_rmse = []

                    for rr in worst_cid_runs:
                        d_rs = np.asarray(rr.get("d_rs", []), float)
                        s_rs = np.asarray(rr.get("s_rs", []), float)
                        if len(d_rs) >= 2 and len(s_rs) >= 2:
                            ds = np.diff(s_rs)
                            d_all.append(d_rs[:-1])
                            st = _length_weighted_err_stats(d_rs[:-1], ds)
                            if np.isfinite(st.get("rmse", np.nan)):
                                run_rmse.append(st["rmse"])

                    if d_all:
                        d_all = np.concatenate(d_all)

                        fig, (axH, axB) = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=200)

                        axH.hist(d_all, bins=25, density=True, alpha=0.85)
                        axH.axvline(0.0, lw=1.4)
                        axH.set_title("Width error distribution after resampling", fontsize=10, fontweight="bold")
                        axH.set_xlabel("d = pred − gt (px)")
                        axH.set_ylabel("density")
                        axH.grid(True, alpha=0.25)

                        axB.boxplot(run_rmse, vert=True)
                        axB.set_title("Per-run width RMSE (resampled)", fontsize=10, fontweight="bold")
                        axB.set_ylabel("RMSE (px)")
                        axB.grid(True, alpha=0.25)

                        fig.suptitle(
                            f"Part 2 aggregated diagnostic — WORST CID={worst_cid} — combined/{worst_mt}",
                            fontsize=11,
                            fontweight="bold"
                        )

                        out = os.path.join(
                            part2_resample_dir,
                            f"part2_resample_aggregated_WORST_cid{worst_cid}_{worst_ct}_{worst_mt}.png"
                        )
                        fig.savefig(out, bbox_inches="tight", dpi=200)
                        plt.close(fig)
                        print(f"[PART2] wrote: {out}")

                # ------------------------------------------------------------
                # Collect GLOBAL ALL runs (ALL CIDs)
                # ------------------------------------------------------------
                all_runs_global = []
                for it in part2_cache:
                    if it.get("runs"):
                        all_runs_global.extend(it["runs"])

                if not all_runs_global:
                    raise RuntimeError("[PART2] no global runs found")

                worst_items = _cache_for_cid(worst_cid, worst_ct, worst_mt, worst_gt)
                if not worst_items:
                    raise RuntimeError(f"[PART2 FATAL] no cache items for worst cid={worst_cid}")

                pred_mask_worst = None
                for it in worst_items:
                    pm = it.get("pred_mask_full", None)
                    if pm is None:
                        continue
                    pm_u8 = (np.asarray(pm, np.uint8) > 0).astype(np.uint8)
                    if pred_mask_worst is None:
                        pred_mask_worst = pm_u8
                    else:
                        pred_mask_worst = ((pred_mask_worst > 0) | (pm_u8 > 0)).astype(np.uint8)
                if pred_mask_worst is None:
                    raise RuntimeError(f"[PART2 FATAL] missing predicted mask for worst cid={worst_cid}")

                pred_mask_all = None
                for it in part2_cache:
                    pm = it.get("pred_mask_full", None)
                    if pm is None:
                        continue
                    pm_u8 = (np.asarray(pm, np.uint8) > 0).astype(np.uint8)
                    if pred_mask_all is None:
                        pred_mask_all = pm_u8
                    else:
                        pred_mask_all = ((pred_mask_all > 0) | (pm_u8 > 0)).astype(np.uint8)
                if pred_mask_all is None:
                    raise RuntimeError("[PART2 FATAL] missing predicted mask for ALL CIDs plot")

                # ------------------------------------------------------------
                # (B2/B3) FINAL PLOTS
                # ------------------------------------------------------------
                part2_plot_worst_and_all(
                    worst_cid_runs=worst_cid_runs,
                    all_runs_global=all_runs_global,
                    pred_mask_worst=pred_mask_worst,
                    pred_mask_all=pred_mask_all,
                    crop_worst=crop_worst,
                    crop_all=crop_all,
                    part2_resample_dir=part2_resample_dir,
                    worst_cid=worst_cid,
                )


        except Exception as e:
            print(f"[PART2]1 plots skipped: {e}")

        # ------------------------------------------------------------
        # Swap final compare-width plotting inputs to Stage-6 resampled segments
        # ------------------------------------------------------------
        if coords_part2 and diffs_part2:
            coords = coords_part2
            diffs  = diffs_part2
            if bboxes_part2:
                bboxes = bboxes_part2
            print(f"[PART2] using resampled plotting inputs: {len(coords)} segs")
        else:
            print("[PART2] no resampled segs produced; keeping Stage-5 plotting inputs")

    except Exception as e:
        print(f"[PART2] skipped (fatal): {e}")


    # ---------------- plotting ----------------
    if not coords:
        print("[WIDTH DEBUG] nothing to plot")
        return [], []

    bbox = _union_bboxes(bboxes)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    gt_mask_u8 = (np.asarray(crack_mask) > 0).astype(np.uint8)
    bg = np.stack([gt_mask_u8 * 255] * 3, axis=-1)
    ax.imshow(bg)

    # ---- prediction mask overlay (semi-transparent green) ----
    pred_mask_union = np.zeros((H, W), np.uint8)
    for wp in (width_pairs or []):
        bbm = wp.get("pred_mask_bbox") or wp.get("bbox")
        cropm = wp.get("pred_mask_crop")
        if bbm is None or cropm is None:
            continue
        try:
            px, py, pw, ph = map(int, bbm)
        except Exception:
            continue
        cm = np.asarray(cropm, np.uint8)
        if cm.ndim != 2:
            continue
        hh = min(max(ph, 0), cm.shape[0], H - max(py, 0))
        ww = min(max(pw, 0), cm.shape[1], W - max(px, 0))
        if hh <= 0 or ww <= 0:
            continue
        y0 = max(py, 0)
        x0 = max(px, 0)
        pred_mask_union[y0:y0 + hh, x0:x0 + ww] |= (cm[:hh, :ww] > 0).astype(np.uint8)

    if np.any(pred_mask_union):
        pred_rgba = np.zeros((H, W, 4), float)
        pred_rgba[..., 1] = pred_mask_union.astype(float) * 0.55  # darker green channel
        pred_rgba[..., 3] = pred_mask_union.astype(float) * 0.35  # alpha
        ax.imshow(pred_rgba, zorder=0.5)

    all_d = np.concatenate([d for d in diffs if d is not None and len(d) > 0])
    all_d = all_d[np.isfinite(all_d)]
    if all_d.size == 0:
        print("[WIDTH DEBUG] no finite diffs")
        return rows, midline_metric_rows

    vmin, vmax = np.percentile(all_d, [5, 95])
    vmin = min(float(vmin), 0.0)
    vmax = max(float(vmax), 0.0)
    if vmax <= vmin:
        vmax = vmin + 1e-6

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("coolwarm")

    # ---- plot colored width-diff segments ----
    for s, d in zip(coords, diffs):
        s = np.asarray(s, float)
        d = np.asarray(d, float)
        n = min(len(s), len(d))
        if n < 2:
            continue
        for i in range(n - 1):
            if not np.isfinite(d[i]):
                continue
            ax.plot(
                [s[i, 0], s[i + 1, 0]],
                [s[i, 1], s[i + 1, 1]],
                color=cmap(norm(d[i])),
                lw=2,
                solid_capstyle="round",
            )

    # ---- zoom to union bbox ----
    if bbox:
        x, y, w, h = bbox
        pad = 0.15 * max(w, h)
        ax.set_xlim(max(0, x - pad), min(W, x + w + pad))
        ax.set_ylim(min(H, y + h + pad), max(0, y - pad))
        ax.add_patch(
            plt.Rectangle(
                (x, y), w, h,
                edgecolor="dodgerblue",
                facecolor="none",
                lw=2,
            )
        )
    else:
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)

    ax.axis("off")
    ax.set_aspect("equal")

    # ---- colorbar with label + endpoint ticks ----
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Estimated width - GT width (px)", fontsize=10, fontweight="bold")

    ticks = list(cb.get_ticks())
    if len(ticks) >= 2:
        tol = 0.3
        ticks[0] = vmin
        ticks[-1] = vmax
        cleaned = []
        for i, t in enumerate(ticks):
            if i == 0 or i == len(ticks) - 1:
                cleaned.append(t)
            else:
                if abs(t - vmin) > tol and abs(t - vmax) > tol:
                    cleaned.append(t)
        cb.set_ticks(cleaned)
        cb.set_ticklabels([f"{t:.1f}" for t in cleaned])

    # Variant-isolated output dir + filename
    out_dir = os.path.join(metrics_dir, midline_type or "unknown", crack_type)
    out = os.path.join(out_dir, f"{midline_type}_{crack_type}_width_diffs.png")

    # ---- mask legend (GT vs Pred) ----
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=(1, 1, 1, 0.95), edgecolor="black", label="GT mask"),
        Patch(facecolor=(0, 1, 0, 0.35), edgecolor="black", label="Pred mask"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=6, framealpha=0.8)

    os.makedirs(out_dir, exist_ok=True)

    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    
    # ============================================================
    # MIDLINE DIAGNOSTIC PLOTS (COMBINED / AUTO)
    # ============================================================
    if midline_metric_rows:
        try:
            import pandas as pd
            from helpers.present_plots import plot_rs3_midline_diagnostics

            df_mid = pd.DataFrame(midline_metric_rows)

            diag_dir = os.path.join(metrics_dir, midline_type or "unknown", "midline_diagnostics", crack_type)
            os.makedirs(diag_dir, exist_ok=True)

            plot_rs3_midline_diagnostics(
                df_all=df_mid,
                out_dir=diag_dir,
                selected_family=None,   # GENERALIZED
            )

            print(f"[MIDLINE METRICS] plotted {len(df_mid)} combined diagnostics")

        except Exception as e:
            print(f"[MIDLINE METRICS] plotting failed: {e}")


    return rows, midline_metric_rows

# ============================================================================
# WIDTH EXPORT: raw diffs + summary + histogram
# ============================================================================

def export_width_metrics_all(
    metrics_dir,
    base_name,
    width_rows,
    midline_type,
    crack_type,
):
    """
    ALWAYS does:
      1) raw per-point width diffs CSV
      2) summary CSV (MAE, RMSE, bias, corr)
      3) histogram PNG

    Paths:
      metrics_dir/<midline_type>/
    """
    import os
    import numpy as np
    import pandas as pd

    if not width_rows:
        print("[WIDTH EXPORT] no data")
        return

    out_dir = os.path.join(metrics_dir, midline_type)
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) RAW WIDTH DIFFS CSV
    # ------------------------------------------------------------------
    diffs_csv = os.path.join(
        out_dir,
        f"{base_name}_width_diffs_{crack_type}.csv"
    )
    df = pd.DataFrame(width_rows)
    df.to_csv(diffs_csv, index=False)
    print("[WIDTH DIFFS] wrote:", diffs_csv)

    # ------------------------------------------------------------------
    # 2) SUMMARY CSV
    # ------------------------------------------------------------------
    # permissive column detection
    cols = {c.lower(): c for c in df.columns}

    gt_col = next(
        (cols[k] for k in cols if k in ("gt", "gt_width", "gt_width_px")),
        None,
    )
    pred_col = next(
        (cols[k] for k in cols if k in ("geodesic", "pred", "pred_width", "pred_width_px")),
        None,
    )
    diff_col = cols.get("diff_px", None) or next(
        (cols[k] for k in cols if "diff" in k),
        None,
    )

    if gt_col is None or pred_col is None:
        print("[WIDTH SUMMARY] missing gt/pred columns")
        return

    gt   = df[gt_col].astype(float).values
    pred = df[pred_col].astype(float).values
    diff = (
        df[diff_col].astype(float).values
        if diff_col is not None
        else pred - gt
    )

    keep = np.isfinite(gt) & np.isfinite(pred) & np.isfinite(diff)
    gt, pred, diff = gt[keep], pred[keep], diff[keep]

    if diff.size == 0:
        print("[WIDTH SUMMARY] no valid samples")
        return

    mae  = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    bias = float(np.mean(diff))
    corr = float(np.corrcoef(gt, pred)[0, 1]) if gt.size > 1 else np.nan

    summary_csv = os.path.join(
        out_dir,
        f"{base_name}_width_summary_{crack_type}.csv"
    )

    pd.DataFrame([{
        "method": crack_type,
        "n_samples": int(diff.size),
        "mae_px": mae,
        "rmse_px": rmse,
        "bias_px": bias,
        "corr": corr,
    }]).to_csv(summary_csv, index=False)

    print("[WIDTH SUMMARY] wrote:", summary_csv)

    # ------------------------------------------------------------------
    # 3) HISTOGRAM
    # ------------------------------------------------------------------
    try:
        from helpers.present_plots import plot_width_diff_histogram

        hist_png = os.path.join(
            out_dir, crack_type,
            f"{base_name}_width_diff_hist_{crack_type}.png"
        )
        os.makedirs(os.path.dirname(hist_png), exist_ok=True)

        plot_width_diff_histogram(
            diffs_csv,
            hist_png,
            title=f"{midline_type} {base_name} width diffs",
            bins=30,
            vlim=None,
        )
    except Exception as e:
        print(f"[WIDTH HIST] failed: {e}")

def export_midline_metrics_all(
    metrics_dir,
    base_name,
    midline_rows,
    midline_type,
    crack_type,
    variant_id="main",
):
    """
    Writes midline geometry metrics CSVs split by geometry_type.

    Output dir:
      metrics_dir/<midline_type>/midline_metrics/<crack_type>/<variant_id>/
    """
    import os
    import pandas as pd

    if not midline_rows:
        print("[MIDLINE EXPORT] no data")
        return

    out_dir = os.path.join(
        metrics_dir,
        midline_type,
        "midline_metrics",
        crack_type,
    )
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(midline_rows)
    if "geometry_type" not in df.columns:
        df["geometry_type"] = "unknown"
    df["geometry_type"] = df["geometry_type"].astype(str).str.lower()

    for geom, sub in df.groupby("geometry_type", dropna=False):
        csv_path = os.path.join(
            out_dir,
            f"{base_name}_midline_metrics_{crack_type}_{geom}.csv",
        )
        sub.to_csv(csv_path, index=False)
        print("[MIDLINE METRICS] wrote:", csv_path)

def write_width_diff_overlay(H, W, rows, out_png, vlim=8.0, radius=2):
    """
    Image-sized dot overlay colored by (pred - gt) width diff in px.
    Robust to key naming differences in `rows`.
    """
    import numpy as np, cv2

    if not rows:
        return

    # --- find candidate keys from first row ---
    sample = rows[0] if isinstance(rows[0], dict) else {}
    keys = set(sample.keys())

    def pick_key(cands):
        for c in cands:
            for k in keys:
                if k.lower() == c.lower():
                    return k
        return None

    xk = pick_key(["x", "mid_x", "mx"])
    yk = pick_key(["y", "mid_y", "my"])

    gtk   = pick_key(["gt", "gt_width", "gt_width_px"])
    predk = pick_key(["geodesic", "pred", "pred_width", "pred_width_px"])
    diffk = next((k for k in keys if "diff" in k.lower()), None)

    # --- accumulate points ---
    xs, ys, diffs = [], [], []
    for r in rows:
        if not isinstance(r, dict):
            continue

        x = r.get(xk) if xk else r.get("x", r.get("mid_x"))
        y = r.get(yk) if yk else r.get("y", r.get("mid_y"))
        if x is None or y is None:
            continue

        if diffk is not None and r.get(diffk) is not None:
            d = r.get(diffk)
        else:
            gt   = r.get(gtk)   if gtk   else r.get("gt")
            pred = r.get(predk) if predk else r.get("geodesic")
            if gt is None or pred is None:
                continue
            d = float(pred) - float(gt)

        try:
            xs.append(float(x)); ys.append(float(y)); diffs.append(float(d))
        except Exception:
            continue

    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    diffs = np.asarray(diffs, float)

    keep = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(diffs)
    xs, ys, diffs = xs[keep], ys[keep], diffs[keep]
    if len(xs) == 0:
        print("[WIDTH OVERLAY] no valid samples")
        return

    # --- render dots ---
    canvas = np.zeros((H, W, 3), np.uint8)

    # normalize to [0,255]
    diffs_clip = np.clip(diffs, -vlim, vlim)
    m = (diffs_clip + vlim) / (2.0 * vlim)
    m8 = (m * 255.0).astype(np.uint8)

    # OpenCV colormap expects 2D uint8
    colors = cv2.applyColorMap(m8.reshape(-1, 1), cv2.COLORMAP_COOL).reshape(-1, 3)

    for (x, y, col) in zip(xs, ys, colors):
        xi = int(round(x)); yi = int(round(y))
        if 0 <= xi < W and 0 <= yi < H:
            cv2.circle(canvas, (xi, yi), radius, tuple(int(c) for c in col), thickness=-1)

    cv2.imwrite(out_png, canvas)
    print(f"[DEBUG WIDTH] wrote diff dot overlay → {out_png}")
    
'''def compute_midline_metrics_for_image(app):
    """
    Compute and save per-CID midline metrics into midline_metrics.csv.
    Always writes a file, even if empty, and logs skipped CIDs.
    """
    import os, numpy as np, pandas as pd
    from .metrics import compute_midline_metrics

    base = app._image_base()
    out_dir = os.path.join(app.save_folder, "metrics", base)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "midline_metrics.csv")

    atomic = app._metric_atomic() or {}
    rows = []

    print(f"[DEBUG MIDLINE] evaluating {len(atomic)} cracks ...")

    for cid, crack in atomic.items():
        ge = crack.get("geodesic_edges", {}) or {}
        if not ("edge1" in ge and "edge2" in ge):
            print(f"[DEBUG MIDLINE] cid{cid}: missing geodesic_edges → skip")
            continue

        e1 = np.asarray(ge["edge1"], float)
        e2 = np.asarray(ge["edge2"], float)
        if len(e1) < 2 or len(e2) < 2:
            print(f"[DEBUG MIDLINE] cid{cid}: invalid edge arrays ({len(e1)}, {len(e2)}) → skip")
            continue

        auto_midline = 0.5 * (e1 + e2)
        man_midline = np.asarray(crack.get("midline", []), float)
        if man_midline.ndim != 2 or len(man_midline) < 2:
            print(f"[DEBUG MIDLINE] cid{cid}: no valid manual midline → skip")
            continue

        try:
            m = compute_midline_metrics(auto_midline, man_midline, tau=3.0)
            m["image"] = base
            m["crack_id"] = cid
            rows.append(m)
            print(f"[DEBUG MIDLINE] cid{cid}: metrics computed OK")
        except Exception as e:
            print(f"[DEBUG MIDLINE] cid{cid}: failed ({e})")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[DEBUG MIDLINE] wrote {len(df)} rows → {out_csv}")'''

# ---------- NEW helpers ----------
def compute_midline_metrics_baseline(pred_xy, gt_xy, tau=3.0):
    """
    Baseline midline geometry metrics.

    Robust against:
      - NaNs
      - inf
      - empty inputs
      - accidental packed midline contamination
    """
    import numpy as np

    # --------------------------------------------------
    # Sanitize inputs
    # --------------------------------------------------
    def _finite_xy(arr, name):
        if arr is None:
            print(f"[B2 DEBUG] {name} is None")
            return np.empty((0, 2), float)

        arr = np.asarray(arr, float)

        if arr.ndim != 2 or arr.shape[1] != 2:
            print(f"[B2 DEBUG] {name} invalid shape: {arr.shape}")
            return np.empty((0, 2), float)

        finite_mask = np.isfinite(arr).all(axis=1)
        n_bad = int(np.sum(~finite_mask))

        if n_bad > 0:
            print(f"[B2 DEBUG] {name} dropped {n_bad} non-finite rows")

        arr = arr[finite_mask]

        if len(arr) == 0:
            print(f"[B2 DEBUG] {name} empty after filtering")

        return arr

    pred_xy = _finite_xy(pred_xy, "pred_xy")
    gt_xy   = _finite_xy(gt_xy,   "gt_xy")

    # --------------------------------------------------
    # Early exit if degenerate
    # --------------------------------------------------
    if len(pred_xy) < 2 or len(gt_xy) < 2:
        print(
            f"[B2 DEBUG] degenerate input after filtering: "
            f"pred={len(pred_xy)} pts, gt={len(gt_xy)} pts"
        )

        return {
            "nn_mean_bidirectional": np.nan,
            "hausdorff_max": np.nan,
            "coverage_min": np.nan,
            "precision_tau": np.nan,
            "recall_tau": np.nan,
            "f1_tau": np.nan,
            "mean_tan_angle_error_deg": np.nan,
            "relative_length_error": np.nan,
            "orth_mean": np.nan,
            "orth_std": np.nan,
            "signed_bias_z": np.nan,
            "frechet_discrete_ds": np.nan,
        }

    # --------------------------------------------------
    # Core midline metrics
    # --------------------------------------------------
    mm = compute_midline_metrics(pred_xy, gt_xy, tau=tau)

    # Ordering-based metric meaningless for baseline
    mm["frechet_discrete_ds"] = np.nan

    # --------------------------------------------------
    # Explicit Ï„-precision (spurious geometry penalty)
    # --------------------------------------------------
    try:
        from scipy.spatial import cKDTree

        gt_tree = cKDTree(gt_xy)

        # distance from pred → nearest GT
        d_pred_to_gt, _ = gt_tree.query(pred_xy, k=1)

        precision_tau = float(np.mean(d_pred_to_gt <= tau))
        recall_tau = mm.get("coverage_min", np.nan)

        mm["precision_tau"] = precision_tau
        mm["recall_tau"] = recall_tau

        if np.isfinite(precision_tau) and np.isfinite(recall_tau):
            mm["f1_tau"] = (
                2 * precision_tau * recall_tau
                / max(1e-6, precision_tau + recall_tau)
            )
        else:
            mm["f1_tau"] = np.nan

    except Exception as e:
        print(f"[B2 DEBUG] KDTree failure: {e}")
        mm["precision_tau"] = np.nan
        mm["recall_tau"] = np.nan
        mm["f1_tau"] = np.nan

    return mm

def compute_midline_metrics(auto_xy, man_xy, tau=3.0):
    """
    Core and diagnostic midline metrics (robust to dict returns).

    Outputs:
    nn_mean_bidirectional, hausdorff, frechet_discrete_ds, mean_tan_angle_error_deg,
    relative_length_error, coverage, orth_mean, orth_std, signed_bias_z,
    curvature_rms_[auto|manual|ratio]
    """
    import numpy as np

    def _unwrap(v):
        """extract numeric value if helper returns dict"""
        if isinstance(v, dict):
            for k in ("value", "mean", "coverage_min", "score", "dist"):
                if k in v and np.isscalar(v[k]):
                    return float(v[k])
            # try first numeric entry
            for val in v.values():
                if np.isscalar(val):
                    return float(val)
            return float("nan")
        try:
            return float(v)
        except Exception:
            return float("nan")

    A = _finite_xy(man_xy)
    B = _finite_xy(auto_xy)
    if len(A) < 2 or len(B) < 2:
        return {k: np.nan for k in
                ("nn_mean_bidirectional","hausdorff_max","frechet_discrete_ds",
                "mean_tan_angle_error_deg","relative_length_error","coverage_min",
                "orth_mean","orth_std","signed_bias_z",
                "curvature_rms_auto","curvature_rms_manual","curvature_rms_ratio")}

    # --- Light resampling for expensive ops ---
    A_ds = _resample_by_arclen(A, N=min(600, len(A)))
    B_ds = _resample_by_arclen(B, N=min(600, len(B)))

    out = {
        "nn_mean_bidirectional": _unwrap(nn_mean_bidirectional(A, B)),
        "hausdorff_max":    _unwrap(hausdorff_max(A, B)),
        "frechet_discrete_ds": float("nan"),
        "mean_tan_angle_error_deg": _unwrap(mean_tangent_angle_error_degs(A, B)),
        "relative_length_error":  _unwrap(relative_length_error(A, B)),
    }
    
    cov = coverage_at_tau(A, B, tau_px=tau)
    out["coverage_A_to_B"] = float(cov["A_to_B"])
    out["coverage_B_to_A"] = float(cov["B_to_A"])
    out["coverage_min"]    = float(min(cov["A_to_B"], cov["B_to_A"]))
    out["hausdorff_p95"] = hausdorff_p95(A, B)

    # --- FrÃ©chet (optional but standard) ---
    try:
        if len(A_ds) >= 2 and len(B_ds) >= 2:
            out["frechet_discrete_ds"] = _unwrap(
                frechet_discrete_ds(A_ds, B_ds, max_points=800)
            )
    except Exception as e:
        print(f"[metrics][warn] FrÃ©chet failed: {e}")

    # --- Orthogonal deviation stats ---
    orth = orthogonal_deviation(A, B, N=400)

    # --- unwrap dict structures safely ---
    import numpy as np
    def _extract_orth_array(obj):
        if isinstance(obj, dict):
            # direct key
            for k in ("orth_dev", "values", "array", "data"):
                if k in obj:
                    return _extract_orth_array(obj[k])
            # nested numeric arrays
            for v in obj.values():
                arr = _extract_orth_array(v)
                if arr is not None:
                    return arr
            return None
        if isinstance(obj, (list, tuple, np.ndarray)):
            return np.asarray(obj, float)
        try:
            return np.asarray([float(obj)], float)
        except Exception:
            return None

    a = _extract_orth_array(orth)
    if a is None:
        a = np.empty(0)
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]

    if a.size:
        mu = float(np.mean(a))
        sd = float(np.std(a) + 1e-12)
        out["orth_mean"] = mu
        out["orth_std"]  = sd
        out["signed_bias_z"] = float(np.sign(mu) * abs(mu) / sd)
    else:
        out["orth_mean"] = out["orth_std"] = out["signed_bias_z"] = np.nan

    # --- Curvature RMS ratio (smoothness diagnostic) ---
    def _rms_curvature(xy):
        xy = np.asarray(xy, float)
        n = len(xy)
        if n < 3: return float("nan")
        ds = np.hypot(np.diff(xy[:,0]), np.diff(xy[:,1]))
        s  = np.concatenate([[0], np.cumsum(ds)])
        if s[-1] <= 0: return float("nan")
        dx = np.gradient(xy[:,0], s, edge_order=2)
        dy = np.gradient(xy[:,1], s, edge_order=2)
        ddx = np.gradient(dx, s, edge_order=2)
        ddy = np.gradient(dy, s, edge_order=2)
        num = np.abs(dx*ddy - dy*ddx)
        den = (dx*dx + dy*dy)**1.5 + 1e-12
        kappa = num / den
        return float(np.sqrt(np.nanmean(kappa**2)))

    kA = _rms_curvature(B_ds)
    kM = _rms_curvature(A_ds)
    out["curvature_rms_auto"]   = kA
    out["curvature_rms_manual"] = kM
    out["curvature_rms_ratio"]  = (
        kA / (kM + 1e-12)
        if np.isfinite(kA) and np.isfinite(kM) else np.nan
    )

    return out


def widths_from_normals(n1_xy, n2_xy):
    """Take Nx2 arrays; return width vector (min-aligned length)."""
    n1 = _finite_xy(n1_xy); n2 = _finite_xy(n2_xy)
    m = min(len(n1), len(n2))
    if m < 2: return np.array([])
    d = np.sqrt(((n1[:m] - n2[:m])**2).sum(1))
    return d[np.isfinite(d)]


def compare_widths(w_ref, w_pred):
    if w_ref.size == 0 or w_pred.size == 0: 
        return dict(MAE=np.nan, RMSE=np.nan, corr=np.nan)
    m = min(len(w_ref), len(w_pred))
    wr = w_ref[:m]; wp = w_pred[:m]
    mae = float(np.mean(np.abs(wr-wp)))
    rmse = float(np.sqrt(np.mean((wr-wp)**2)))
    corr = float(np.corrcoef(wr, wp)[0,1]) if m>2 else np.nan
    return dict(MAE=mae, RMSE=rmse, corr=corr)


def _auto_cache_key(self):
    # Include any params that affect auto generation
    parts = [
        "v1",
        f"down={getattr(self, 'downsample_factor_box', None).value() if hasattr(self,'downsample_factor_box') else 'na'}",
        f"mu={getattr(self,'mu_box',None).value() if hasattr(self,'mu_box') else 'na'}",
        f"l={getattr(self,'l_box',None).value() if hasattr(self,'l_box') else 'na'}",
        f"p={getattr(self,'p_box',None).value() if hasattr(self,'p_box') else 'na'}",
        f"color={getattr(self, 'edge_track_color_box', None).currentText() if hasattr(self,'edge_track_color_box') else 'G'}"
    ]
    return "|".join(map(str, parts))


##########################################################
# Helpers
###########################################################

# -------------------- NORMALS CACHE KEY --------------------
# If you keep compare_widths_for_aligned_cracks here, make sure this is visible:
import hashlib
def _mask_midline_cache_key(mask_bin, midline):
    h = hashlib.blake2b(digest_size=16)
    h.update(mask_bin.tobytes())
    h.update(np.asarray(midline, np.float32).tobytes())
    return h.hexdigest()

def reconstruct_manual_mask_from_edges(crack: dict, H: int, W: int) -> "np.ndarray":
    """
    Strictly reconstruct a full-image binary mask (H,W) from geodesic_edges only.
    If edges are missing or invalid → returns all zeros.
    """
    import numpy as np, cv2

    ge = crack.get("geodesic_edges") or {}
    e1 = np.asarray(ge.get("edge1", []), float)
    e2 = np.asarray(ge.get("edge2", []), float)

    if e1.ndim != 2 or e2.ndim != 2 or len(e1) < 2 or len(e2) < 2:
        return np.zeros((H, W), np.uint8)

    e1 = e1[np.isfinite(e1).all(axis=1)]
    e2 = e2[np.isfinite(e2).all(axis=1)]
    if len(e1) < 2 or len(e2) < 2:
        return np.zeros((H, W), np.uint8)

    poly = np.vstack([e1, e2[::-1]])
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 255)
    return mask
   
def merged_metric_atomic(authoring_atomic: dict, save_folder: str, image_base: str) -> dict:
    """
    Merge authoring entries with per-cid snapshots.
    Supports nested canonical path:
        metrics/<image_base>/cid{cid}/cid{cid}.json
    And legacy fallback paths.
    """
    import os, json

    merged = {}
    base_dir = os.path.join(save_folder, "metrics", image_base)

    for cid, cr in (authoring_atomic or {}).items():
        merged[cid] = dict(cr)

        # preferred new path
        canonical_dir  = os.path.join(base_dir, f"cid{cid}")
        canonical_json = os.path.join(canonical_dir, f"cid{cid}.json")

        # legacy paths
        legacy_a = os.path.join(base_dir, "cid", f"{cid}.json")
        legacy_b = os.path.join(base_dir, f"cid{cid}.json")

        for p in (canonical_json, legacy_a, legacy_b):
            if os.path.exists(p):
                try:
                    with open(p, "r") as f:
                        snap = json.load(f) or {}
                    merged[cid].update(snap)
                except Exception:
                    print(f"[merge] warning: failed reading {p}")
                break
        else:
            # nothing found — ensure canonical folder exists
            try:
                os.makedirs(canonical_dir, exist_ok=True)
            except Exception:
                print(f"[merge] warning: could not create {canonical_dir}")

    return merged

def has_valid_mask(crack: dict) -> bool:
    """
    Returns True if a crack dictionary has a valid mask (crop+bbox) or midline.
    Protects combine/metrics from ghost cracks.
    """
    try:
        mc, bb = crack.get("mask_crop"), crack.get("mask_bbox")
        ml = crack.get("midline")
        if mc is not None and bb is not None and len(mc) and len(bb) == 4:
            return True
        if isinstance(ml, (list, tuple)) and len(ml) >= 2:
            return True
        return False
    except Exception:
        return False
