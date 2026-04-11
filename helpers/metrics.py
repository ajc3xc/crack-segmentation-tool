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
    "path.simplify": True,
    "path.simplify_threshold": 1.0,
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


def _split_valid_runs(mask):
    """
    Split a 1D boolean mask into contiguous True runs.
    Returns list of (start, end) with end-exclusive indexing.
    """
    m = np.asarray(mask, bool).reshape(-1)
    if m.size == 0:
        return []
    runs = []
    in_run = False
    i0 = 0
    for i, v in enumerate(m):
        if v and not in_run:
            in_run = True
            i0 = i
        elif (not v) and in_run:
            runs.append((int(i0), int(i)))
            in_run = False
    if in_run:
        runs.append((int(i0), int(m.size)))
    return runs


def length_weighted_err_stats(d_vals, ds_w):
    """
    Primitive length-weighted error stats.
    Keeps exact behavior/math used in compare_widths_for_aligned_cracks.
    """
    d_vals = np.asarray(d_vals, float)
    ds_w = np.asarray(ds_w, float)
    n = min(len(d_vals), len(ds_w))
    if n <= 0:
        return {"bias": np.nan, "mae": np.nan, "rmse": np.nan, "p95_abs": np.nan, "median_abs": np.nan}

    d_vals = d_vals[:n]
    ds_w = ds_w[:n]
    ok = np.isfinite(d_vals) & np.isfinite(ds_w) & (ds_w > 0)
    if not np.any(ok):
        return {"bias": np.nan, "mae": np.nan, "rmse": np.nan, "p95_abs": np.nan, "median_abs": np.nan}

    d = d_vals[ok]
    w = ds_w[ok]
    W = float(np.sum(w) + 1e-12)

    bias = float(np.sum(d * w) / W)
    mae = float(np.sum(np.abs(d) * w) / W)
    mse = float(np.sum((d ** 2) * w) / W)
    rmse = float(np.sqrt(max(mse, 0.0)))

    absd = np.abs(d[np.isfinite(d)])
    p95 = float(np.percentile(absd, 95)) if absd.size else np.nan
    med = float(np.median(absd)) if absd.size else np.nan

    return {"bias": bias, "mae": mae, "rmse": rmse, "p95_abs": p95, "median_abs": med}


def compute_length_weighted_metrics(d_vals, s_vals, *, debug=False):
    """
    Robust wrapper:
    - builds ds and d_mid from (d_vals, s_vals)
    - splits contiguous valid runs
    - computes per-segment and crack-level length-weighted stats
    - returns coverage and segment diagnostics
    """
    d_vals = np.asarray(d_vals, float).reshape(-1)
    s_vals = np.asarray(s_vals, float).reshape(-1)
    n = min(len(d_vals), len(s_vals))
    if n < 2:
        return {"bias": np.nan, "mae": np.nan, "rmse": np.nan, "coverage": 0.0, "segments": []}

    d_vals = d_vals[:n]
    s_vals = s_vals[:n]
    ds = np.diff(s_vals)
    d_mid = d_vals[:-1]

    if ds.size == 0 or d_mid.size == 0:
        return {"bias": np.nan, "mae": np.nan, "rmse": np.nan, "coverage": 0.0, "segments": []}

    valid = np.isfinite(d_mid) & np.isfinite(ds) & (ds > 0)
    runs = _split_valid_runs(valid)

    L_total = float(np.sum(ds))
    if (not np.isfinite(L_total)) or L_total <= 0:
        return {"bias": np.nan, "mae": np.nan, "rmse": np.nan, "coverage": 0.0, "segments": []}

    segments = []
    for i0, i1 in runs:
        if i1 - i0 <= 0:
            continue
        d_r = np.asarray(d_mid[i0:i1], float)
        ds_r = np.asarray(ds[i0:i1], float)
        stats_r = length_weighted_err_stats(d_r, ds_r)
        L_r = float(np.sum(ds_r[np.isfinite(ds_r) & (ds_r > 0)]))
        if (not np.isfinite(L_r)) or L_r <= 0:
            continue
        segments.append(
            {
                "L": L_r,
                "bias": stats_r.get("bias", np.nan),
                "mae": stats_r.get("mae", np.nan),
                "rmse": stats_r.get("rmse", np.nan),
                "p95_abs": stats_r.get("p95_abs", np.nan),
                "median_abs": stats_r.get("median_abs", np.nan),
                "i0": int(i0),
                "i1": int(i1),
            }
        )

    if not segments:
        out = {"bias": np.nan, "mae": np.nan, "rmse": np.nan, "coverage": 0.0, "segments": []}
        if debug:
            print("[METRICS] N_total=0 N_valid=0 num_segments=0 coverage=0.0000")
        return out

    L_total_valid = float(np.sum([seg["L"] for seg in segments]))
    sum_bias = 0.0
    sum_mae = 0.0
    sum_mse = 0.0
    for seg in segments:
        L = float(seg["L"])
        b = float(seg["bias"]) if np.isfinite(seg["bias"]) else np.nan
        a = float(seg["mae"]) if np.isfinite(seg["mae"]) else np.nan
        r = float(seg["rmse"]) if np.isfinite(seg["rmse"]) else np.nan
        if np.isfinite(b):
            sum_bias += b * L
        if np.isfinite(a):
            sum_mae += a * L
        if np.isfinite(r):
            sum_mse += (r ** 2) * L

    denom = float(L_total_valid + 1e-12)
    out = {
        "bias": float(sum_bias / denom),
        "mae": float(sum_mae / denom),
        "rmse": float(np.sqrt(max(sum_mse / denom, 0.0))),
        "coverage": float(L_total_valid / max(L_total, 1e-12)),
        "segments": segments,
    }
    if debug:
        n_valid = int(np.sum(valid))
        print(
            f"[METRICS] N_total={int(len(d_mid))} N_valid={n_valid} "
            f"num_segments={len(segments)} coverage={out['coverage']:.4f}"
        )
    return out

###############################################################################################
# Midline Metrics
###############################################################################################

# --- DROP-IN REPLACEMENT in py ---

def _nn_dists(A, B):
    """
    Compute nearest-neighbor distances from each point in A to the closest point in B.
    CPU cKDTree only (lower overhead and more stable for these metric sizes).
    """
    import numpy as np
    B1_DIAGNOSTIC_DEBUG = True
    if A is None or B is None or len(A) == 0 or len(B) == 0:
        return np.zeros((len(A),), dtype=float)
    try:
        from scipy.spatial import cKDTree as CPU_KDTree
        tree = CPU_KDTree(np.asarray(B, float))
        dists, _ = tree.query(np.asarray(A, float), k=1)
        return np.asarray(dists, float)
    except Exception as e:
        print(f"[nn_dists][warn] CPU KDTree failed, using brute force: {e}")
        A = np.asarray(A, float)
        B = np.asarray(B, float)
        diff = A[:, None, :] - B[None, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        return np.min(dists, axis=1)


def _nn_bidirectional(A, B):
    """
    One-pass bidirectional nearest-neighbor query with shared KDTree builds.
    Returns dAB, dBA, idxAB, idxBA.
    """
    import numpy as np
    A = _finite_xy(A)
    B = _finite_xy(B)
    if len(A) == 0 or len(B) == 0:
        return (
            np.zeros((len(A),), float),
            np.zeros((len(B),), float),
            np.full((len(A),), -1, int),
            np.full((len(B),), -1, int),
        )
    try:
        from scipy.spatial import cKDTree
        treeB = cKDTree(B)
        dAB, idxAB = treeB.query(A, k=1)
        treeA = cKDTree(A)
        dBA, idxBA = treeA.query(B, k=1)
        return np.asarray(dAB, float), np.asarray(dBA, float), np.asarray(idxAB, int), np.asarray(idxBA, int)
    except Exception as e:
        print(f"[_nn_bidirectional][warn] cKDTree failed, using brute force: {e}")
        diffAB = A[:, None, :] - B[None, :, :]
        dmatAB = np.sqrt(np.sum(diffAB ** 2, axis=2))
        idxAB = np.argmin(dmatAB, axis=1)
        dAB = dmatAB[np.arange(len(A)), idxAB]
        diffBA = B[:, None, :] - A[None, :, :]
        dmatBA = np.sqrt(np.sum(diffBA ** 2, axis=2))
        idxBA = np.argmin(dmatBA, axis=1)
        dBA = dmatBA[np.arange(len(B)), idxBA]
        return np.asarray(dAB, float), np.asarray(dBA, float), np.asarray(idxAB, int), np.asarray(idxBA, int)


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


def frechet_discrete_ds(A, B):
    """
    Iterative EiterMannila discrete Fréchet distance.
    - No recursion (avoids RecursionError)
    - Expects caller-provided sampling
    """
    A = _finite_xy(A)
    B = _finite_xy(B)
    if len(A) == 0 or len(B) == 0:
        return float('inf')

    n, m = len(A), len(B)
    # DP table of size (n x m)
    ca = np.full((n, m), np.inf, dtype=float)

    ca[0, 0] = np.hypot(A[0, 0] - B[0, 0], A[0, 1] - B[0, 1])
    # first column
    for i in range(1, n):
        ca[i, 0] = max(ca[i-1, 0], np.hypot(A[i, 0] - B[0, 0], A[i, 1] - B[0, 1]))
    # first row
    for j in range(1, m):
        ca[0, j] = max(ca[0, j-1], np.hypot(A[0, 0] - B[j, 0], A[0, 1] - B[j, 1]))

    # fill DP
    for i in range(1, n):
        Ai = A[i]  # small locality win
        drow = np.hypot(Ai[0] - B[:, 0], Ai[1] - B[:, 1])
        for j in range(1, m):
            d = drow[j]
            ca[i, j] = max(min(ca[i-1, j], ca[i-1, j-1], ca[i, j-1]), d)

    return float(ca[n-1, m-1])


def tangent_angles(xy):
    xy = _finite_xy(xy)
    if len(xy) < 2: return np.array([])
    d = np.gradient(xy, axis=0)
    ang = np.arctan2(d[:,1], d[:,0])
    return ang


def mean_tangent_angle_error_degs(A_resampled, B_resampled):
    """
    Mean absolute tangent-angle error (degrees) on caller-provided resampled curves.
    """
    Ar = _finite_xy(A_resampled)
    Br = _finite_xy(B_resampled)
    if len(Ar)==0 or len(Br)==0: return np.nan
    aA = tangent_angles(Ar); aB = tangent_angles(Br)
    n = min(len(aA), len(aB))
    if n == 0: return np.nan
    da = np.abs(np.unwrap(aA[:n]) - np.unwrap(aB[:n]))
    da = np.mod(da + pi, 2*pi) - pi
    return float(np.degrees(np.mean(np.abs(da))))


def orthogonal_deviation(manual_xy_resampled, auto_xy):
    """
    Signed distance from manual (reference) to nearest auto, measured
    along manual normals. Sign convention is with respect to manual normals.
    """
    M = _finite_xy(manual_xy_resampled)
    A = _finite_xy(auto_xy)
    if len(M)==0 or len(A)==0:
        return dict(mean=np.nan, median=np.nan, rmse=np.nan, p95=np.nan)
    # manual normals
    d = np.gradient(M, axis=0)  # tangents
    norm = np.column_stack([-d[:,1], d[:,0]])
    nlen = np.maximum(1e-9, np.sqrt((norm**2).sum(1)))
    n = norm / nlen[:,None]
    # nearest auto → signed projection (KDTree, no O(n*m) dense matrix)
    try:
        from scipy.spatial import cKDTree
        treeA = cKDTree(A)
        _, idx = treeA.query(M, k=1)
    except Exception:
        d2 = ((M[:, None, :] - A[None, :, :]) ** 2).sum(2)
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
        print("[DEBUG MASK] ❌ missing mask_crop or mask_bbox")
        return np.zeros((H, W), dtype=np.uint8)

    crop = np.array(mc, dtype=np.uint8)
    if crop.ndim != 2:
        print(f"[DEBUG MASK] ❌ mask_crop ndim={crop.ndim}, expected 2")
        return np.zeros((H, W), dtype=np.uint8)

    if not isinstance(bb, (list, tuple)) or len(bb) != 4:
        print("[DEBUG MASK] ❌ invalid bbox format")
        return np.zeros((H, W), dtype=np.uint8)

    x, y, w, h = [int(v) for v in bb]
    print(f"[DEBUG MASK] bbox parsed → x={x}, y={y}, w={w}, h={h}")

    # check if bbox is within image
    if x < 0 or y < 0 or x >= W or y >= H:
        print("[DEBUG MASK] ❌ bbox origin outside image bounds")
        return np.zeros((H, W), dtype=np.uint8)

    # check crop consistency
    print(f"[DEBUG MASK] crop shape={crop.shape}, target area=({y}:{y+h}, {x}:{x+w})")

    if h <= 0 or w <= 0:
        print("[DEBUG MASK] ❌ non-positive bbox dimensions")
        return np.zeros((H, W), dtype=np.uint8)

    # safe paste within limits
    x2, y2 = min(x + w, W), min(y + h, H)
    w_eff, h_eff = max(0, x2 - x), max(0, y2 - y)
    if w_eff == 0 or h_eff == 0:
        print("[DEBUG MASK] ❌ effective bbox has zero area after clipping")
        return np.zeros((H, W), dtype=np.uint8)

    crop = (crop > 0).astype(np.uint8)
    crop = crop[:h_eff, :w_eff]

    m = np.zeros((H, W), dtype=np.uint8)
    m[y:y+h_eff, x:x+w_eff] = crop
    print(f"[DEBUG MASK] ✅ pasted crop ({crop.shape}) into full mask at ({x},{y})")
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

def normals_from_mask_for_midline(
    midline_xy,
    mask,
    max_radius=50,
    diagnostics=None,
    debug_max_examples=40,
    image_hw=None,
    radius_frac=0.10,
    align_thresh=0.90,
    ratio_thresh=2.0,
    dt_radius_map=None,
    radius_scale=2.2,
    endpoint_frac=0.03,
    align_reject_min=None,
    endpoint_mode="atomic",
):
    """
    Boundary-clipped normal extraction by vectorized ray marching on a binary mask.
    For each point, march along +/- normal until first mask exit (or max radius).
    """
    import numpy as np

    def _diag_init():
        return {
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "invalid_frac": 0.0,
            "reasons": {},
            "examples": [],
            "clamp_moved_count": 0,
            "clamp_moved_mean_px": 0.0,
            "clamp_moved_max_px": 0.0,
        }

    diag = _diag_init()

    def _bump(reason):
        diag["reasons"][reason] = int(diag["reasons"].get(reason, 0)) + 1

    def _add_example(i, reason, **extra):
        if int(debug_max_examples) <= 0:
            return
        if len(diag["examples"]) >= int(debug_max_examples):
            return
        item = {"idx": int(i), "reason": str(reason)}
        for k, v in extra.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                item[k] = v
        diag["examples"].append(item)

    def _finalize_diag():
        diag["total"] = int(len(midline_xy) if hasattr(midline_xy, "__len__") else 0)
        diag["valid"] = int(np.sum(np.isfinite(widths_mask))) if "widths_mask" in locals() else 0
        diag["invalid"] = int(max(0, diag["total"] - diag["valid"]))
        diag["invalid_frac"] = float(diag["invalid"]) / float(diag["total"]) if diag["total"] > 0 else 0.0
        if isinstance(diagnostics, dict):
            diagnostics.clear()
            diagnostics.update(diag)

    H, W = mask.shape[:2]
    mask_bin = (np.asarray(mask) > 0)

    midline_xy = np.asarray(midline_xy, float)
    if midline_xy.ndim != 2 or midline_xy.shape[1] != 2 or len(midline_xy) < 2:
        n = len(midline_xy) if midline_xy.ndim > 0 else 0
        widths_mask = np.full(n, np.nan, float)
        _bump("bad_midline_shape")
        _finalize_diag()
        return (np.full(n, np.nan),) * 5, []

    if not np.any(mask_bin):
        N = len(midline_xy)
        widths_mask = np.full(N, np.nan, float)
        _bump("empty_mask_polygons")
        _finalize_diag()
        return (np.full(N, np.nan),) * 5, []

    if image_hw is not None and len(image_hw) == 2:
        H_img, W_img = int(image_hw[0]), int(image_hw[1])
    else:
        H_img, W_img = int(H), int(W)

    base_radius = float(np.ceil(max(
        float(max_radius),
        float(radius_frac) * float(min(H_img, W_img)),
    )))

    # ---- tangent + normals ----
    try:
        from cracktools.segmentation import compute_smooth_tangent_normals
        _, nor = compute_smooth_tangent_normals(midline_xy[:, 0], midline_xy[:, 1])
        nor = np.asarray(nor, float)
    except Exception:
        dx, dy = np.gradient(midline_xy[:, 0]), np.gradient(midline_xy[:, 1])
        nrm = np.hypot(dx, dy) + 1e-12
        tan = np.stack([dx / nrm, dy / nrm], axis=1)
        nor = np.stack([-tan[:, 1], tan[:, 0]], axis=1)

    N = len(midline_xy)
    e1x = np.full(N, np.nan, float); e1y = np.full(N, np.nan, float)
    e2x = np.full(N, np.nan, float); e2y = np.full(N, np.nan, float)
    widths_mask = np.full(N, np.nan, float)

    eps = 1e-9
    nlen = np.hypot(nor[:, 0], nor[:, 1])
    good_n = np.isfinite(nlen) & (nlen > eps) & np.all(np.isfinite(nor), axis=1)
    nor_u = np.zeros_like(nor, float)
    nor_u[good_n] = nor[good_n] / nlen[good_n, None]

    # Clamp starts to nearest in-mask pixel if needed.
    pts = np.asarray(midline_xy, float).copy()
    xi0 = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
    yi0 = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
    starts_inside = mask_bin[yi0, xi0]
    clamp_dists = []
    if not np.all(starts_inside):
        try:
            from scipy.ndimage import distance_transform_edt
            outside = ~mask_bin
            _, inds = distance_transform_edt(outside, return_indices=True)
            bad = np.where(~starts_inside)[0]
            ny = inds[0, yi0[bad], xi0[bad]]
            nx = inds[1, yi0[bad], xi0[bad]]
            moved = np.hypot(pts[bad, 0] - nx.astype(float), pts[bad, 1] - ny.astype(float))
            pts[bad, 0] = nx.astype(float)
            pts[bad, 1] = ny.astype(float)
            clamp_dists = moved[np.isfinite(moved) & (moved > 1e-9)].tolist()
            if bad.size > 0:
                diag["start_clamp_count"] = int(bad.size)
        except Exception:
            _bump("clamp_failed")

    # Per-point radius (atomic-only endpoint taper on true tips).
    radius_i = np.full(N, base_radius, float)
    endpoint_mode = str(endpoint_mode or "atomic").strip().lower()
    if endpoint_mode not in ("atomic", "combined", "none"):
        endpoint_mode = "atomic"
    if endpoint_mode == "atomic" and N >= 2:
        taper_idx = [0, N - 1]
        for ti in taper_idx:
            if radius_i[ti] > 50.0:
                radius_i[ti] *= 0.5
            elif radius_i[ti] > (0.10 * float(min(H_img, W_img))):
                radius_i[ti] *= 0.7
        radius_i = np.maximum(radius_i, 8.0)

    Rmax = int(max(1, int(np.ceil(float(np.nanmax(radius_i))))))
    r = np.arange(0.0, float(Rmax) + 1.0, 1.0, dtype=float)  # (R,)

    P = pts[:, None, :]          # (N,1,2)
    Nrm = nor_u[:, None, :]      # (N,1,2)
    rr = r[None, :, None]        # (1,R,1)

    def _march(side_sign):
        ray = P + float(side_sign) * rr * Nrm  # (N,R,2)
        x = np.round(ray[:, :, 0]).astype(int)
        y = np.round(ray[:, :, 1]).astype(int)
        inb = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        xc = np.clip(x, 0, W - 1)
        yc = np.clip(y, 0, H - 1)
        inside = inb & mask_bin[yc, xc]
        inside &= (r[None, :] <= radius_i[:, None] + 1e-9)

        first_out = ~inside
        exit_idx = np.argmax(first_out, axis=1)
        never_exit = ~np.any(first_out, axis=1)
        exit_idx[never_exit] = len(r) - 1
        edge_idx = np.maximum(exit_idx - 1, 0)

        has_inside = np.any(inside, axis=1)
        ex = ray[np.arange(N), edge_idx, 0]
        ey = ray[np.arange(N), edge_idx, 1]
        return np.stack([ex, ey], axis=1), has_inside

    e_pos, ok_pos = _march(+1.0)
    e_neg, ok_neg = _march(-1.0)

    valid = good_n & ok_pos & ok_neg

    if not np.all(good_n):
        for i in np.where(~good_n)[0][:int(debug_max_examples)]:
            _bump("nonfinite_point_or_normal")
            _add_example(int(i), "nonfinite_point_or_normal")
    miss_side = (~ok_pos) | (~ok_neg)
    if np.any(miss_side):
        for i in np.where(miss_side & good_n)[0][:int(debug_max_examples)]:
            _bump("side_march_failed")
            _add_example(int(i), "side_march_failed", pos_ok=bool(ok_pos[i]), neg_ok=bool(ok_neg[i]))

    widths = np.linalg.norm(e_pos - e_neg, axis=1)
    valid &= np.isfinite(widths) & (widths > 1e-6)

    # Optional DT-based guard.
    if dt_radius_map is not None:
        try:
            dtr = np.asarray(dt_radius_map, float)
            xi = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
            yi = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
            rloc = dtr[yi, xi]
            bad_terr = valid & np.isfinite(rloc) & (rloc > 0) & (widths > (float(radius_scale) * (2.0 * rloc)))
            if np.any(bad_terr):
                valid[bad_terr] = False
                for i in np.where(bad_terr)[0][:int(debug_max_examples)]:
                    _bump("territory_radius_exceeded")
                    _add_example(int(i), "territory_radius_exceeded", width=float(widths[i]), rloc=float(rloc[i]))
        except Exception:
            _bump("dt_radius_guard_failed")

    # Optional alignment guard.
    if align_reject_min is not None:
        span = e_pos - e_neg
        sl = np.linalg.norm(span, axis=1)
        align = np.full(N, np.nan, float)
        g = np.isfinite(sl) & (sl > 1e-9) & np.all(np.isfinite(Nrm[:, 0, :]), axis=1)
        align[g] = np.abs(np.sum((span[g] / sl[g, None]) * Nrm[g, 0, :], axis=1))
        bad_align = valid & ((~np.isfinite(align)) | (align < float(align_reject_min)))
        if np.any(bad_align):
            valid[bad_align] = False
            for i in np.where(bad_align)[0][:int(debug_max_examples)]:
                _bump("low_alignment_reject")
                _add_example(int(i), "low_alignment_reject", align=float(align[i]) if np.isfinite(align[i]) else None)

    e1x[valid], e1y[valid] = e_pos[valid, 0], e_pos[valid, 1]
    e2x[valid], e2y[valid] = e_neg[valid, 0], e_neg[valid, 1]
    widths_mask[valid] = widths[valid]

    if clamp_dists:
        dd = np.asarray(clamp_dists, float)
        diag["clamp_moved_count"] = int(len(dd))
        diag["clamp_moved_mean_px"] = float(np.mean(dd))
        diag["clamp_moved_max_px"] = float(np.max(dd))

    # Lightweight compatibility diagnostics with prior schema.
    if int(np.sum(np.isfinite(widths_mask))) == 0 and not diag["reasons"]:
        _bump("all_invalid")

    _finalize_diag()
    return (e1x, e1y, e2x, e2y, widths_mask), []

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
    plt.savefig(out_png, dpi=100)
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

    fig, ax = plt.subplots(figsize=(7, 3), dpi=100)
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
        fig.savefig(out_png)
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
        lost = bmeta.get("lost_to", [])  # ← THIS is critical

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
    Local step sizes Δs_i = ||p_{i+1} - p_i||.
    """
    xy = np.asarray(xy, float)
    if xy.ndim != 2 or len(xy) < 2:
        return np.asarray([], float)
    return np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))


def _already_uniform_enough(xy, ds_target=1.0, mean_tol=0.02, cv_tol=0.05):
    """
    Very strict fast-path: treat as already-uniform only if:
      - mean(Δs) is close to ds_target
      - coefficient of variation of Δs is small
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

            # ✅ REQUIRED by your combined extractor now
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

def _project_support_indices_core(
    wmap,
    supp,
    mid_xy,
    *,
    max_nn_dist_px=15.0,
    use_support_mask=True,
    domain_mask=None,
    bbox=None,
    debug=False,
):
    """
    Projection-only stage:
      - build filtered support cloud
      - nearest-neighbor map GT points -> support indices
      - apply radius rejection
    """
    import numpy as np

    if wmap is None:
        return {
            "valid": np.zeros((0,), dtype=bool),
            "indices": np.zeros((0,), dtype=int),
            "dists": np.zeros((0,), dtype=float),
            "skel_xy": np.zeros((0, 2), dtype=np.float32),
            "finite_mask": np.zeros((0,), dtype=bool),
        }

    wmap = np.asarray(wmap)
    H, W = wmap.shape[:2]
    xy = np.asarray(mid_xy, float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        return {
            "valid": np.zeros((0,), dtype=bool),
            "indices": np.zeros((0,), dtype=int),
            "dists": np.zeros((0,), dtype=float),
            "skel_xy": np.zeros((0, 2), dtype=np.float32),
            "finite_mask": np.zeros((0,), dtype=bool),
        }

    bbox_clip = None
    if bbox is not None:
        try:
            x0, y0, bw, bh = [int(v) for v in bbox]
            x1 = x0 + bw
            y1 = y0 + bh
            x0 = int(np.clip(x0, 0, W))
            y0 = int(np.clip(y0, 0, H))
            x1 = int(np.clip(x1, 0, W))
            y1 = int(np.clip(y1, 0, H))
            if x1 > x0 and y1 > y0:
                bbox_clip = (x0, y0, x1, y1)
        except Exception:
            bbox_clip = None

    finite = np.isfinite(xy).all(axis=1)
    if bbox_clip is not None:
        x0, y0, x1, y1 = bbox_clip
        inside = (
            (xy[:, 0] >= x0) & (xy[:, 0] < x1) &
            (xy[:, 1] >= y0) & (xy[:, 1] < y1)
        )
        finite &= inside

    if use_support_mask and supp is not None:
        supp_m = np.asarray(supp).astype(bool)
    else:
        supp_m = np.isfinite(wmap) & (wmap > 0)

    if domain_mask is not None:
        dm = np.asarray(domain_mask).astype(bool)
        if dm.shape == supp_m.shape:
            supp_m &= dm
        elif debug:
            print(f"[B1 PROJ] domain_mask shape mismatch: supp={supp_m.shape} domain={dm.shape}")

    if bbox_clip is not None:
        x0, y0, x1, y1 = bbox_clip
        bbox_mask = np.zeros_like(supp_m, dtype=bool)
        bbox_mask[y0:y1, x0:x1] = True
        supp_m &= bbox_mask

    ys, xs = np.nonzero(supp_m)
    if len(xs) == 0:
        if debug:
            print("[B1 PROJ] support empty")
        return {
            "valid": np.zeros((len(xy),), dtype=bool),
            "indices": np.full((len(xy),), -1, dtype=int),
            "dists": np.full((len(xy),), np.inf, dtype=float),
            "skel_xy": np.zeros((0, 2), dtype=np.float32),
            "finite_mask": finite,
        }

    supp_xy = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])

    d_all = np.full((len(xy),), np.inf, dtype=float)
    idx_all = np.full((len(xy),), -1, dtype=int)

    if np.any(finite):
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(supp_xy)
            d, idx = tree.query(xy[finite], k=1)
            d = np.asarray(d, float)
            idx = np.asarray(idx, int)
        except Exception:
            d = np.full((np.sum(finite),), np.inf, float)
            idx = np.full((np.sum(finite),), -1, int)
            for i, p in enumerate(xy[finite]):
                dd = np.sum((supp_xy - p) ** 2, axis=1)
                j = int(np.argmin(dd))
                idx[i] = j
                d[i] = float(np.sqrt(dd[j]))

        finite_idx = np.flatnonzero(finite)
        d_all[finite_idx] = d
        idx_all[finite_idx] = idx

    valid = finite & np.isfinite(d_all) & (idx_all >= 0) & (d_all <= float(max_nn_dist_px))
    return {
        "valid": valid,
        "indices": idx_all,
        "dists": d_all,
        "skel_xy": supp_xy,
        "finite_mask": finite,
    }


def project_indices_to_support(
    wmap,
    supp,
    mid_xy,
    *,
    max_nn_dist_px=10.0,
    use_support_mask=True,
    domain_mask=None,
    bbox=None,
    debug=False,
):
    """
    Public projection stage returning indices/distances/support cloud.
    """
    return _project_support_indices_core(
        wmap,
        supp,
        mid_xy,
        max_nn_dist_px=max_nn_dist_px,
        use_support_mask=use_support_mask,
        domain_mask=domain_mask,
        bbox=bbox,
        debug=debug,
    )


def sample_widths_from_projected_indices(wmap, proj):
    """
    Width-lookup stage:
      - uses projected support indices
      - preserves per-method NaNs independently
    """
    import numpy as np

    if wmap is None:
        return np.array([], dtype=np.float32)
    wmap = np.asarray(wmap)
    H, W = wmap.shape[:2]

    valid = np.asarray((proj or {}).get("valid", []), bool).reshape(-1)
    idx_all = np.asarray((proj or {}).get("indices", []), int).reshape(-1)
    skel_xy = np.asarray((proj or {}).get("skel_xy", []), float)
    if valid.size == 0 or idx_all.size != valid.size or skel_xy.ndim != 2 or skel_xy.shape[1] != 2:
        return np.array([], dtype=np.float32)

    out = np.full((len(valid),), np.nan, dtype=np.float32)
    if not np.any(valid):
        return out

    idx_v = idx_all[valid]
    ok_idx = (idx_v >= 0) & (idx_v < len(skel_xy))
    if not np.any(ok_idx):
        return out

    take_rows = np.flatnonzero(valid)[ok_idx]
    pxy = skel_xy[idx_v[ok_idx]]
    xx = np.clip(pxy[:, 0].astype(int), 0, W - 1)
    yy = np.clip(pxy[:, 1].astype(int), 0, H - 1)
    vals = np.asarray(wmap[yy, xx], np.float32)
    vals[~np.isfinite(vals)] = np.nan
    out[take_rows] = vals
    return out


def project_widths_to_support(
    wmap,
    supp,
    mid_xy,
    *,
    max_nn_dist_px=15.0,
    use_support_mask=True,
    domain_mask=None,
    bbox=None,
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
        domain_mask: optional HxW bool ownership mask for projection domain
        bbox: optional hard clip bbox (x, y, w, h)
        debug: prints coverage diagnostics

    Returns:
        (N,) float32 array of projected widths (NaN where invalid)
    """
    import numpy as np

    proj = _project_support_indices_core(
        wmap,
        supp,
        mid_xy,
        max_nn_dist_px=max_nn_dist_px,
        use_support_mask=use_support_mask,
        domain_mask=domain_mask,
        bbox=bbox,
        debug=debug,
    )
    out = sample_widths_from_projected_indices(wmap, proj)

    if debug and out.size > 0:
        valid_proj = int(np.sum(np.asarray(proj.get("valid", []), bool)))
        finite = np.asarray(proj.get("finite_mask", []), bool)
        print(
            f"[B1 PROJ] total={len(out)} "
            f"finite={int(np.sum(finite))} "
            f"valid_proj={valid_proj} "
            f"nan_out={int(np.sum(~np.isfinite(out)))}"
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
    B1_DIAGNOSTIC_DEBUG = True
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
    dbg_template = {
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

    def _normalize_meta_list(meta_raw, n):
        meta_raw = meta_raw if isinstance(meta_raw, list) else []
        out = []
        for i in range(int(n)):
            m = meta_raw[i] if i < len(meta_raw) and isinstance(meta_raw[i], dict) else {}
            out.append(dict(m))
        return out

    def _safe_bbox_xywh(bb, H, W):
        if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
            return None
        try:
            x, y, w, h = [int(v) for v in bb]
        except Exception:
            return None
        if w <= 0 or h <= 0:
            return None
        x0 = int(np.clip(x, 0, W))
        y0 = int(np.clip(y, 0, H))
        x1 = int(np.clip(x + w, 0, W))
        y1 = int(np.clip(y + h, 0, H))
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1 - x0, y1 - y0)

    def _tight_bbox_from_seg(seg_xy, H, W, pad=4):
        seg_xy = _finite_xy(seg_xy)
        if len(seg_xy) < 2:
            return None
        x0 = int(np.floor(np.min(seg_xy[:, 0])) - int(pad))
        y0 = int(np.floor(np.min(seg_xy[:, 1])) - int(pad))
        x1 = int(np.ceil(np.max(seg_xy[:, 0])) + int(pad))
        y1 = int(np.ceil(np.max(seg_xy[:, 1])) + int(pad))
        x0 = int(np.clip(x0, 0, W))
        y0 = int(np.clip(y0, 0, H))
        x1 = int(np.clip(x1, 0, W))
        y1 = int(np.clip(y1, 0, H))
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1 - x0, y1 - y0)

    def _segment_seed_mask(seg_xy, H, W):
        seg_xy = _finite_xy(seg_xy)
        m = np.zeros((H, W), dtype=bool)
        if len(seg_xy) < 2:
            return m
        for i in range(len(seg_xy) - 1):
            x0, y0 = seg_xy[i]
            x1, y1 = seg_xy[i + 1]
            if not (np.isfinite(x0) and np.isfinite(y0) and np.isfinite(x1) and np.isfinite(y1)):
                continue
            steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
            if steps < 2:
                steps = 2
            xs = np.rint(np.linspace(x0, x1, num=steps)).astype(int)
            ys = np.rint(np.linspace(y0, y1, num=steps)).astype(int)
            ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
            if np.any(ok):
                m[ys[ok], xs[ok]] = True
        return m

    def _rebuild_crack_mask_full(cr, H, W):
        if not isinstance(cr, dict):
            return None
        pm = cr.get("pred_mask", None)
        if pm is not None:
            pm = np.asarray(pm).astype(bool)
            if pm.shape == (H, W):
                return pm
        bb = _safe_bbox_xywh(cr.get("mask_bbox"), H, W)
        crop = cr.get("mask_crop", None)
        if bb is None or crop is None:
            return None
        x, y, w, h = bb
        crop = np.asarray(crop)
        if crop.ndim < 2:
            return None
        hh = min(h, crop.shape[0])
        ww = min(w, crop.shape[1])
        if hh <= 0 or ww <= 0:
            return None
        out = np.zeros((H, W), dtype=bool)
        out[y:y + hh, x:x + ww] = (crop[:hh, :ww] > 0)
        return out

    def _build_projection_domain_mask(
        *,
        crack_type,
        cid,
        cr,
        gt_seg,
        seg_meta,
        gt_mid_parts,
        gt_mid_meta,
        supp,
        wmap,
        gt_full,
    ):
        H, W = np.asarray(wmap).shape[:2]
        if supp is not None:
            support_base = np.asarray(supp).astype(bool)
        else:
            support_base = np.isfinite(wmap) & (np.asarray(wmap) > 0)

        crack_mask = _rebuild_crack_mask_full(cr, H, W)
        if crack_mask is not None and crack_mask.shape == support_base.shape and np.any(crack_mask):
            domain = crack_mask.copy()
        else:
            gtb = (np.asarray(gt_full) > 0)
            if gtb.shape == support_base.shape and np.any(gtb):
                domain = gtb.copy()
            else:
                domain = support_base.copy()

        if str(crack_type).lower() == "atomic":
            seed = _segment_seed_mask(gt_seg, H, W)
            if np.any(seed):
                try:
                    from scipy.ndimage import label
                    lbl, _n = label(domain.astype(np.uint8))
                    ids = np.unique(lbl[(seed) & (lbl > 0)])
                    if ids.size > 0:
                        domain = np.isin(lbl, ids)
                except Exception:
                    pass
            return domain.astype(bool)

        # Combined: use pre-computed branch territory from dominance_meta.
        # This was built by dominant_segments_from_group with radius = max(4, min(1.2 * half_width, 50))
        # and is the authoritative spatial extent for each branch.
        branch_id = None
        if isinstance(seg_meta, dict):
            branch_id = _safe_int(seg_meta.get("branch_id"), None)
        territory_applied = False
        if branch_id is None:
            print(f"[DOMAIN DBG] cid={cid} branch={branch_id} territory_applied={territory_applied} domain_nnz={int(np.sum(domain))}")
            return domain.astype(bool)

        try:
            dom_meta = cr.get("dominance_meta") if isinstance(cr, dict) else None
            if isinstance(dom_meta, dict):
                terr_by_branch = dom_meta.get("branch_territory") or {}
                terr_entry = terr_by_branch.get(str(int(branch_id)))
                bite_bbox = (dom_meta.get("bite") or {}).get("bbox")  # [bx0, by0, bw, bh] global xywh
                if (
                    isinstance(terr_entry, dict)
                    and terr_entry.get("packbits_b64")
                    and isinstance(bite_bbox, (list, tuple))
                    and len(bite_bbox) == 4
                ):
                    terr_crop = _decode_packbits_b64_to_mask(
                        terr_entry["packbits_b64"],
                        terr_entry["shape"],
                    )
                    bx0, by0, bw, bh = [int(v) for v in bite_bbox]
                    terr_full = np.zeros((H, W), dtype=bool)
                    bx1 = min(bx0 + bw, W)
                    by1 = min(by0 + bh, H)
                    ch = min(terr_crop.shape[0], by1 - by0)
                    cw = min(terr_crop.shape[1], bx1 - bx0)
                    if ch > 0 and cw > 0:
                        terr_full[by0:by0 + ch, bx0:bx0 + cw] = (terr_crop[:ch, :cw] > 0)
                    if np.any(terr_full):
                        domain &= terr_full
                        territory_applied = True
        except Exception:
            pass

        if not territory_applied:
            # Fallback: width-adaptive dilation from wmap, matching combiner's scale
            seg_arr = _finite_xy(gt_seg) if gt_seg is not None else np.empty((0, 2), float)
            dil_px = 30  # safe default
            if len(seg_arr) > 0:
                ys_s = np.clip(np.round(seg_arr[:, 1]).astype(int), 0, H - 1)
                xs_s = np.clip(np.round(seg_arr[:, 0]).astype(int), 0, W - 1)
                wvals = np.asarray(wmap)[ys_s, xs_s]
                wvals = wvals[np.isfinite(wvals) & (wvals > 0)]
                if len(wvals) > 0:
                    half_w = float(np.median(wvals)) / 2.0
                    dil_px = int(np.clip(int(1.2 * half_w), 8, 60))
            branch_seed = np.zeros_like(domain, dtype=bool)
            for Sx, mx in zip(gt_mid_parts, gt_mid_meta):
                b = _safe_int((mx or {}).get("branch_id"), None) if isinstance(mx, dict) else None
                if b is None or int(b) != int(branch_id):
                    continue
                branch_seed |= _segment_seed_mask(Sx, H, W)
            if np.any(branch_seed):
                from scipy.ndimage import binary_dilation
                domain &= binary_dilation(branch_seed, iterations=dil_px).astype(bool)

        print(f"[DOMAIN DBG] cid={cid} branch={branch_id} territory_applied={territory_applied} domain_nnz={int(np.sum(domain))}")
        return domain.astype(bool)

    def _expand_bbox_xywh(bb, H, W, pad=0):
        bb = _safe_bbox_xywh(bb, H, W)
        if bb is None:
            return None
        try:
            pad_i = int(max(0, int(pad)))
        except Exception:
            pad_i = 0
        if pad_i <= 0:
            return bb
        x, y, w, h = bb
        x0 = int(np.clip(x - pad_i, 0, W))
        y0 = int(np.clip(y - pad_i, 0, H))
        x1 = int(np.clip(x + w + pad_i, 0, W))
        y1 = int(np.clip(y + h + pad_i, 0, H))
        if x1 <= x0 or y1 <= y0:
            return bb
        return (x0, y0, x1 - x0, y1 - y0)

    def _get_segment_bbox(crack_type, cr, seg_xy, seg_meta, H, W, pad_override=None):
        ctype = str(crack_type).lower()
        if ctype == "atomic":
            bb = _safe_bbox_xywh((cr or {}).get("mask_bbox"), H, W)
            if bb is not None:
                return _expand_bbox_xywh(bb, H, W, pad=pad_override or 0)
            pad_seg = int(pad_override) if pad_override is not None else 4
            return _tight_bbox_from_seg(seg_xy, H, W, pad=pad_seg)

        if isinstance(seg_meta, dict):
            for k in ("branch_bbox", "bbox", "mask_bbox", "pred_mask_bbox"):
                bb = _safe_bbox_xywh(seg_meta.get(k), H, W)
                if bb is not None:
                    return _expand_bbox_xywh(bb, H, W, pad=pad_override or 0)
        pad_seg = int(pad_override) if pad_override is not None else 4
        return _tight_bbox_from_seg(seg_xy, H, W, pad=pad_seg)

    def _normalize_baseline_record(method_name, rec_obj):
        method_s = str(method_name)
        if isinstance(rec_obj, dict):
            wmap = rec_obj.get("width_map")
            supp = rec_obj.get("support_mask")
            skel = rec_obj.get("skel")
            meta = rec_obj.get("meta", {})
        elif isinstance(rec_obj, (tuple, list)) and len(rec_obj) >= 2:
            wmap, supp = rec_obj[0], rec_obj[1]
            skel = supp
            meta = {}
        else:
            return None

        if wmap is None or supp is None:
            return None

        wmap = np.asarray(wmap)
        supp = np.asarray(supp).astype(bool)
        if wmap.ndim != 2 or supp.ndim != 2 or wmap.shape[:2] != supp.shape[:2]:
            return None

        if skel is None:
            skel = supp
        skel = np.asarray(skel).astype(bool)
        if skel.ndim != 2 or skel.shape[:2] != wmap.shape[:2]:
            skel = supp

        return {
            "method": method_s,
            "width_map": wmap,
            "support_mask": supp,
            "skel": skel,
            "meta": meta if isinstance(meta, dict) else {},
            "is_skeleton_method": method_s.lower().startswith("skel_"),
        }

    def _infer_family_key(method_name, rec_obj):
        m = str(method_name).lower()
        if m in {"skel_mat_raw", "mat_width_raw"}:
            return "skel_mat_raw"
        if m in {"skel_mat_dse", "mat_width_dse", "pca_width_dse", "esd_width_dse", "eob_width_dse"}:
            return "skel_mat_dse"

        meta = rec_obj.get("meta", {}) if isinstance(rec_obj, dict) else {}
        skel_method = str(meta.get("skeleton_method", "")).strip()
        if skel_method:
            return skel_method

        supp = np.asarray(rec_obj.get("support_mask"), bool)
        sig = hashlib.md5(np.ascontiguousarray(supp.astype(np.uint8)).tobytes()).hexdigest()[:12]
        return f"support::{supp.shape[0]}x{supp.shape[1]}::{sig}"

    # Normalize maps (support both dict records and legacy tuples).
    norm_maps = {}
    for method, rec in (width_baseline_maps or {}).items():
        nrec = _normalize_baseline_record(method, rec)
        if nrec is None:
            print(f"[BASELINE B1] skip invalid baseline record for method='{method}'")
            continue
        norm_maps[str(method)] = nrec

    if not norm_maps:
        return width_rows

    families = {}
    for method, rec in norm_maps.items():
        fam_key = _infer_family_key(method, rec)
        fam = families.setdefault(
            fam_key,
            {
                "family_key": fam_key,
                "support_method": None,
                "width_methods": [],
                "all_methods": [],
            },
        )
        fam["all_methods"].append(method)
        if rec["is_skeleton_method"]:
            fam["support_method"] = method
        if fam["support_method"] is None:
            fam["support_method"] = method
        _m = str(method).lower()
        _allow_skel_width_method = False
        if (not _m.startswith("skel_")) or _allow_skel_width_method:
            fam["width_methods"].append(method)

    # Keep only families that have width methods to evaluate.
    families = {k: v for k, v in families.items() if v.get("width_methods")}
    if not families:
        return width_rows

    width_methods = sorted({m for fam in families.values() for m in fam["width_methods"]})
    dbg_by_method = {m: dict(dbg_template) for m in width_methods}
    overlays_by_method = {m: {"coords": [], "diffs": []} for m in width_methods}
    diagnostics_by_family = {
        fk: {
            "coords": [],
            "radii": [],
            "pts_success": [],
            "pts_radius_rejected": [],
            "pts_wmap_nan": [],
            "pts_pre_projection": [],
            "pts_no_support": [],
            "skel_available_segs": [],
            "pts_dists": [],
        }
        for fk in families.keys()
    }

    fam_dbg = {
        k: {
            "support_method": v["support_method"],
            "width_methods": sorted(v["width_methods"]),
        }
        for k, v in families.items()
    }
    print(f"[BASELINE B1] family projection plan: {fam_dbg}")

    # ============================================================
    # Main evaluation (family-shared projection + per-method sampling)
    # ============================================================
    for cid, cr in cracks.items():
        for method in width_methods:
            dbg_by_method[method]["cr_total"] += 1

        if not isinstance(cr, dict):
            for method in width_methods:
                dbg_by_method[method]["cr_not_dict"] += 1
            continue

        segs = cr.get("midline_segments") or []
        if not segs:
            for method in width_methods:
                dbg_by_method[method]["no_midline_segments"] += 1
            if len(dbg_examples["no_midline_segments"]) < MAX_EX:
                dbg_examples["no_midline_segments"].append(str(cid))
            continue

        seg_meta = _normalize_meta_list(
            (cr.get("midline_segments_meta") or cr.get("segments_meta") or []),
            len(segs),
        )

        gt_mid_parts = []
        gt_mid_meta = []
        for s, mm in zip(segs, seg_meta):
            if s is None:
                continue
            s = _finite_xy(s)
            if len(s) >= 2:
                gt_mid_parts.append(s)
                gt_mid_meta.append(mm if isinstance(mm, dict) else {})

        if not gt_mid_parts:
            for method in width_methods:
                dbg_by_method[method]["no_valid_midline_pts"] += 1
            if len(dbg_examples["no_valid_midline_pts"]) < MAX_EX:
                dbg_examples["no_valid_midline_pts"].append(str(cid))
            continue

        gt_widths = None
        for k in ("gt_widths_auto_centered", "gt_widths", "widths"):
            gt_widths = _coerce_width_vector(cr.get(k))
            if gt_widths is not None:
                break

        if gt_widths is None:
            gtn = cr.get("gt_normals") or {}
            if isinstance(gtn, dict):
                gt_widths = _coerce_width_vector(gtn.get("width_px"))

        if gt_widths is None:
            for method in width_methods:
                dbg_by_method[method]["no_gt_widths"] += 1
            if len(dbg_examples["no_gt_widths"]) < MAX_EX:
                dbg_examples["no_gt_widths"].append(str(cid))
            continue

        if gt_widths.size < 2:
            for method in width_methods:
                dbg_by_method[method]["gt_widths_too_short"] += 1
            continue

        gt_off = 0
        crack_s_idx_by_method = {m: 0 for m in width_methods}
        crack_rows_by_method = {m: 0 for m in width_methods}

        for seg_idx, (gt_seg, seg_meta_i) in enumerate(zip(gt_mid_parts, gt_mid_meta)):
            gt_seg = np.asarray(gt_seg, float)
            Lseg = int(len(gt_seg))
            if Lseg < 2:
                gt_off += Lseg
                continue

            gt_widths_seg = np.asarray(gt_widths[gt_off:gt_off + Lseg], float)
            gt_off += Lseg
            if gt_widths_seg.size < 2:
                continue

            # Per-segment adaptive projection radius based on local GT width.
            _finite_gtw = gt_widths_seg[np.isfinite(gt_widths_seg)]
            if _finite_gtw.size >= 1:
                _median_gtw = float(np.median(_finite_gtw))
                _seg_radius = float(np.clip(0.6 * _median_gtw, 6.0, 60.0))
            else:
                _seg_radius = 6.0
                print(f"[B1 PROJ] seg={seg_idx} cid={cid}: no finite GT widths, using fallback radius=6.0")

            # Keep bbox pad coupled to segment projection radius.
            _seg_bbox_pad = int(np.ceil(_seg_radius)) + 2
            if B1_DIAGNOSTIC_DEBUG:
                _finite_gtw_dbg = gt_widths_seg[np.isfinite(gt_widths_seg)]
                if _finite_gtw_dbg.size > 0:
                    print(
                        f"[B1 DBG] cid={cid} seg={seg_idx} "
                        f"Lseg={len(gt_seg)} "
                        f"gtw_n_finite={_finite_gtw_dbg.size} "
                        f"gtw_median={float(np.median(_finite_gtw_dbg)):.2f} "
                        f"gtw_min={float(np.min(_finite_gtw_dbg)):.2f} "
                        f"gtw_max={float(np.max(_finite_gtw_dbg)):.2f} "
                        f"radius={_seg_radius:.2f} "
                        f"bbox_pad={_seg_bbox_pad}"
                    )

            # Segment-wise projection is computed once per skeleton family and then
            # reused for all width methods in that family.
            for fam_key, fam in families.items():
                support_method = fam["support_method"]
                support_rec = norm_maps.get(support_method)
                if support_rec is None:
                    continue

                wmap_support = np.asarray(support_rec["width_map"])
                supp_support = np.asarray(support_rec["support_mask"]).astype(bool)
                Hm, Wm = wmap_support.shape[:2]

                domain_mask = _build_projection_domain_mask(
                    crack_type=crack_type,
                    cid=cid,
                    cr=cr,
                    gt_seg=gt_seg,
                    seg_meta=seg_meta_i,
                    gt_mid_parts=gt_mid_parts,
                    gt_mid_meta=gt_mid_meta,
                    supp=supp_support,
                    wmap=wmap_support,
                    gt_full=gt_full,
                )
                bbox_seg = _get_segment_bbox(
                    crack_type=crack_type,
                    cr=cr,
                    seg_xy=gt_seg,
                    seg_meta=seg_meta_i,
                    H=Hm,
                    W=Wm,
                    pad_override=_seg_bbox_pad,
                )
                if B1_DIAGNOSTIC_DEBUG:
                    supp_support_arr = np.asarray(supp_support).astype(bool)
                    dom_arr = np.asarray(domain_mask).astype(bool)
                    n_skel_raw = int(np.count_nonzero(supp_support_arr))
                    n_skel_in_dom = int(np.count_nonzero(supp_support_arr & dom_arr))
                    n_dom_total = int(np.count_nonzero(dom_arr))

                    bbox_mask_dbg = np.zeros_like(supp_support_arr, dtype=bool)
                    if bbox_seg is not None:
                        _bx, _by, _bw, _bh = [int(v) for v in bbox_seg]
                        _bx0 = max(0, _bx)
                        _by0 = max(0, _by)
                        _bx1 = min(supp_support_arr.shape[1], _bx + _bw)
                        _by1 = min(supp_support_arr.shape[0], _by + _bh)
                        if _bx1 > _bx0 and _by1 > _by0:
                            bbox_mask_dbg[_by0:_by1, _bx0:_bx1] = True
                    n_skel_after_bbox = (
                        int(np.count_nonzero(supp_support_arr & dom_arr & bbox_mask_dbg))
                        if bbox_seg is not None else n_skel_in_dom
                    )
                    print(
                        f"[B1 DBG] cid={cid} seg={seg_idx} fam={fam_key} "
                        f"skel_raw={n_skel_raw} "
                        f"skel_after_domain={n_skel_in_dom} "
                        f"skel_after_bbox={n_skel_after_bbox} "
                        f"domain_total={n_dom_total} "
                        f"bbox_seg={bbox_seg}"
                    )

                    _gt_head = gt_seg[:3].tolist() if len(gt_seg) >= 3 else gt_seg.tolist()
                    _gt_bbox = (
                        float(np.min(gt_seg[:, 0])),
                        float(np.min(gt_seg[:, 1])),
                        float(np.max(gt_seg[:, 0])),
                        float(np.max(gt_seg[:, 1])),
                    ) if len(gt_seg) >= 1 else None

                    eff_support = supp_support_arr & dom_arr
                    if bbox_seg is not None and bbox_mask_dbg.any():
                        eff_support &= bbox_mask_dbg
                    _ys_dbg, _xs_dbg = np.nonzero(eff_support)
                    if len(_xs_dbg) >= 3:
                        _supp_head = list(zip(
                            _xs_dbg[:3].tolist(),
                            _ys_dbg[:3].tolist(),
                        ))
                    else:
                        _supp_head = list(zip(_xs_dbg.tolist(), _ys_dbg.tolist()))
                    _supp_bbox = (
                        int(np.min(_xs_dbg)), int(np.min(_ys_dbg)),
                        int(np.max(_xs_dbg)), int(np.max(_ys_dbg)),
                    ) if len(_xs_dbg) > 0 else None

                    print(
                        f"[B1 DBG] cid={cid} seg={seg_idx} fam={fam_key} "
                        f"gt_head_xy={_gt_head} "
                        f"gt_bbox_xyxy={_gt_bbox}"
                    )
                    print(
                        f"[B1 DBG] cid={cid} seg={seg_idx} fam={fam_key} "
                        f"supp_head_xy={_supp_head} "
                        f"supp_bbox_xyxy={_supp_bbox}"
                    )

                proj = project_indices_to_support(
                    wmap_support,
                    supp_support,
                    gt_seg,
                    max_nn_dist_px=float(_seg_radius),
                    use_support_mask=True,
                    domain_mask=domain_mask,
                    bbox=bbox_seg,
                    debug=False,
                )
                if B1_DIAGNOSTIC_DEBUG:
                    _valid = np.asarray(proj.get("valid", []), bool)
                    _finite_mask = np.asarray(proj.get("finite_mask", []), bool)
                    _dists = np.asarray(proj.get("dists", []), float)
                    _dists_finite = _dists[np.isfinite(_dists)]

                    if _dists_finite.size > 0:
                        d_min = float(np.min(_dists_finite))
                        d_median = float(np.median(_dists_finite))
                        d_max = float(np.max(_dists_finite))
                        d_p90 = float(np.percentile(_dists_finite, 90))
                    else:
                        d_min = d_median = d_max = d_p90 = float("nan")

                    print(
                        f"[B1 DBG] cid={cid} seg={seg_idx} fam={fam_key} "
                        f"n_total={len(gt_seg)} "
                        f"n_finite_pre={int(np.sum(_finite_mask))} "
                        f"n_valid={int(np.sum(_valid))} "
                        f"used_radius={_seg_radius:.2f} "
                        f"dist_min={d_min:.2f} "
                        f"dist_median={d_median:.2f} "
                        f"dist_p90={d_p90:.2f} "
                        f"dist_max={d_max:.2f}"
                    )

                diagnostics_by_family[fam_key]["coords"].append(np.asarray(gt_seg, float))
                diagnostics_by_family[fam_key]["radii"].append(float(_seg_radius))
                diagnostics_by_family[fam_key]["skel_available_segs"].append(
                    np.asarray(proj.get("skel_xy", np.empty((0, 2), float)), float)
                )
                diagnostics_by_family[fam_key]["pts_dists"].append((
                    np.asarray(gt_seg, float),
                    np.asarray(proj.get("dists", np.full(len(gt_seg), np.inf)), float),
                    float(_seg_radius),
                ))
                # Per-point attribution is intentionally computed against the
                # family's support-method width map (not downstream width methods).
                finite_pre = np.asarray(proj.get("finite_mask", []), bool)
                valid_mask = np.asarray(proj.get("valid", []), bool)
                if finite_pre.size == len(gt_seg) and valid_mask.size == len(gt_seg):
                    support_widths = sample_widths_from_projected_indices(
                        np.asarray(support_rec["width_map"]),
                        proj,
                    )
                    finite_out = np.isfinite(np.asarray(support_widths, float))
                    if finite_out.size != len(gt_seg):
                        finite_out = np.zeros((len(gt_seg),), dtype=bool)

                    no_support_segment = (
                        int(np.sum(valid_mask)) == 0 and int(np.sum(finite_pre)) > 0
                    )

                    for i in range(len(gt_seg)):
                        pt = (float(gt_seg[i, 0]), float(gt_seg[i, 1]))
                        if not finite_pre[i]:
                            diagnostics_by_family[fam_key]["pts_pre_projection"].append(pt)
                        elif no_support_segment:
                            diagnostics_by_family[fam_key]["pts_no_support"].append(pt)
                        elif not valid_mask[i]:
                            diagnostics_by_family[fam_key]["pts_radius_rejected"].append(pt)
                        elif not finite_out[i]:
                            diagnostics_by_family[fam_key]["pts_wmap_nan"].append(pt)
                        else:
                            diagnostics_by_family[fam_key]["pts_success"].append(pt)

                for method in fam["width_methods"]:
                    rec_method = norm_maps.get(method)
                    if rec_method is None:
                        continue

                    pred_widths_seg = sample_widths_from_projected_indices(
                        np.asarray(rec_method["width_map"]),
                        proj,
                    )
                    if pred_widths_seg is None:
                        continue

                    pred_widths_seg = np.asarray(pred_widths_seg, float)
                    n = min(len(gt_widths_seg), len(pred_widths_seg))
                    if n < 2:
                        continue

                    gt_w = np.asarray(gt_widths_seg[:n], float)
                    pr_w = np.asarray(pred_widths_seg[:n], float)
                    diff = pr_w - gt_w
                    seg_xy_eval = np.asarray(gt_seg[:n], float)

                    overlays_by_method[method]["coords"].append(seg_xy_eval)
                    overlays_by_method[method]["diffs"].append(np.asarray(diff, float))

                    s_idx = int(crack_s_idx_by_method[method])
                    for i in range(n):
                        width_rows.append(
                            {
                                "image": base_name,
                                "cid": str(cid),
                                "method": method,
                                "gt_width_px": float(gt_w[i]),
                                "pred_width_px": float(pr_w[i]),
                                "diff_px": float(diff[i]),
                                "s_idx": int(s_idx + i),
                            }
                        )
                    crack_s_idx_by_method[method] = s_idx + int(n)
                    crack_rows_by_method[method] += int(n)

        for method in width_methods:
            if crack_rows_by_method[method] <= 0:
                dbg_by_method[method]["pred_widths_too_short"] += 1
                if len(dbg_examples["pred_widths_too_short"]) < MAX_EX:
                    dbg_examples["pred_widths_too_short"].append(str(cid))
            else:
                dbg_by_method[method]["rows_emitted"] += int(crack_rows_by_method[method])

    # --------------------------------------------------
    # Baseline projected-width spatial overlay (B1), per method
    # --------------------------------------------------
    for method in width_methods:
        dbg_m = dbg_by_method[method]
        coords = overlays_by_method[method]["coords"]
        diffs = overlays_by_method[method]["diffs"]

        if dbg_m["rows_emitted"] > 0 and metrics_dir_local is not None:
            try:
                import matplotlib.pyplot as plt
                from matplotlib.colors import TwoSlopeNorm
                import numpy as np
                import os

                if not coords:
                    raise RuntimeError("No valid projected segments for overlay")

                all_pts = np.vstack(coords)
                x0, y0 = np.min(all_pts, axis=0)
                x1, y1 = np.max(all_pts, axis=0)
                pad = 5.0

                x0p = int(np.floor(x0 - pad))
                y0p = int(np.floor(y0 - pad))
                x1p = int(np.ceil(x1 + pad))
                y1p = int(np.ceil(y1 + pad))

                H, W = gt_full.shape[:2]
                x0c = max(0, x0p)
                y0c = max(0, y0p)
                x1c = min(W, x1p)
                y1c = min(H, y1p)
                if x1c <= x0c or y1c <= y0c:
                    raise RuntimeError("Invalid overlay crop bounds after clipping")

                mask_crop = gt_full[y0c:y1c, x0c:x1c]

                fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
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

                method_root = metrics_dir_local
                if len(width_methods) > 1:
                    method_root = os.path.join(metrics_dir_local, method)

                out_dir = os.path.join(
                    method_root,
                    midline_type or "unknown",
                    crack_type,
                )
                os.makedirs(out_dir, exist_ok=True)

                out_path = os.path.join(
                    out_dir,
                    f"{base_name}_{method}_width_baseline_projected_overlay.png",
                )

                fig.savefig(out_path, dpi=150)
                plt.close(fig)

                print(f"[BASELINE B1] wrote overlay: {out_path}")

            except Exception as e:
                print(f"[BASELINE B1] overlay failed for method='{method}': {e}")

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

    # --------------------------------------------------
    # Baseline projection diagnostics (B1), per skeleton family
    # --------------------------------------------------
    for fam_key, fam in families.items():
        support_method = fam.get("support_method")
        if support_method is None:
            continue
        diag = diagnostics_by_family.get(fam_key, {})
        coords = diag.get("coords", []) or []
        radii = diag.get("radii", []) or []
        if not coords:
            continue
        if metrics_dir_local is None:
            continue

        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
            import numpy as np
            import os

            support_rec = norm_maps.get(str(support_method), None)
            if support_rec is None:
                continue

            wmap_support = np.asarray(support_rec.get("width_map"))
            skel_support = support_rec.get("skel", None)
            if skel_support is None:
                skel_support = support_rec.get("support_mask", None)
            skel_support = np.asarray(skel_support).astype(bool)
            if wmap_support.ndim != 2 or skel_support.ndim != 2 or wmap_support.shape[:2] != skel_support.shape[:2]:
                continue

            all_pts = np.vstack(coords)
            x0, y0 = np.min(all_pts, axis=0)
            x1, y1 = np.max(all_pts, axis=0)
            pad = 5.0

            x0p = int(np.floor(x0 - pad))
            y0p = int(np.floor(y0 - pad))
            x1p = int(np.ceil(x1 + pad))
            y1p = int(np.ceil(y1 + pad))

            H, W = gt_full.shape[:2]
            x0c = max(0, x0p)
            y0c = max(0, y0p)
            x1c = min(W, x1p)
            y1c = min(H, y1p)
            if x1c <= x0c or y1c <= y0c:
                continue

            gt_crop = gt_full[y0c:y1c, x0c:x1c]

            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
            ax.set_facecolor("white")
            ax.imshow(
                gt_crop,
                cmap="gray",
                extent=[x0c, x1c, y1c, y0c],
                interpolation="nearest",
                alpha=0.4,
            )

            for S in coords:
                S = np.asarray(S, float)
                if S.ndim == 2 and S.shape[1] == 2 and len(S) >= 2:
                    ax.plot(S[:, 0], S[:, 1], color="black", lw=1.5)

            for S, r in zip(coords, radii):
                S = np.asarray(S, float)
                if S.ndim != 2 or S.shape[1] != 2 or len(S) == 0:
                    continue
                rr = float(r) if np.isfinite(r) else 0.0
                if rr <= 0:
                    continue
                for i in range(0, len(S), 20):
                    cxy = S[i]
                    if not (np.isfinite(cxy[0]) and np.isfinite(cxy[1])):
                        continue
                    ax.add_patch(
                        Circle(
                            (float(cxy[0]), float(cxy[1])),
                            radius=rr,
                            fill=False,
                            edgecolor="gray",
                            linewidth=0.6,
                            alpha=0.15,
                        )
                    )
            # --- Skeleton: gray = filtered out, teal = survived domain+bbox ---
            ys_s, xs_s = np.where(skel_support)
            keep_crop = (xs_s >= x0c) & (xs_s < x1c) & (ys_s >= y0c) & (ys_s < y1c)
            if np.any(keep_crop):
                ax.scatter(
                    xs_s[keep_crop],
                    ys_s[keep_crop],
                    s=1,
                    c="#bbbbbb",
                    alpha=0.5,
                    zorder=2,
                    label="skel (filtered out)",
                )

            avail_segs = diag.get("skel_available_segs", [])
            if avail_segs:
                valid_avail = [s for s in avail_segs if len(np.asarray(s, float)) > 0]
                avail_all = np.vstack(valid_avail) if valid_avail else np.empty((0, 2), float)
                if len(avail_all) > 0:
                    ax_keep = (
                        (avail_all[:, 0] >= x0c) & (avail_all[:, 0] < x1c) &
                        (avail_all[:, 1] >= y0c) & (avail_all[:, 1] < y1c)
                    )
                    if np.any(ax_keep):
                        ax.scatter(
                            avail_all[ax_keep, 0],
                            avail_all[ax_keep, 1],
                            s=4,
                            c="#00cc88",
                            alpha=0.85,
                            zorder=3,
                            label="skel (available)",
                        )

            # --- GT midline colored by NN distance (0..2x radius -> green..red) ---
            from matplotlib.collections import LineCollection
            import matplotlib.cm as cm

            pts_dists = diag.get("pts_dists", [])
            for S, dists, seg_r in pts_dists:
                S = np.asarray(S, float)
                dists = np.asarray(dists, float)
                if len(S) < 2 or len(dists) != len(S):
                    continue
                norm_d = np.clip(dists / max(float(seg_r), 1.0), 0.0, 2.0) / 2.0
                norm_d = np.where(np.isfinite(norm_d), norm_d, 1.0)
                colors = cm.RdYlGn_r(norm_d)
                points = S.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, colors=colors[:-1], linewidths=2.0, zorder=6)
                ax.add_collection(lc)

            sm = plt.cm.ScalarMappable(cmap=cm.RdYlGn_r, norm=plt.Normalize(vmin=0, vmax=2))
            sm.set_array([])
            plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.01, label="NN dist / radius (capped 2x)")

            ax.set_xlim(x0c, x1c)
            ax.set_ylim(y1c, y0c)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"{support_method} - B1 projection diagnostics")

            out_dir = os.path.join(
                metrics_dir_local,
                str(support_method),
                midline_type or "unknown",
                crack_type,
            )
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(
                out_dir,
                f"{base_name}_{support_method}_b1_projection_diagnostics.png",
            )
            fig.savefig(out_path, dpi=100)
            plt.close(fig)
            print(f"[BASELINE B1] wrote projection diagnostics: {out_path}")
        except Exception as e:
            print(f"[BASELINE B1] projection diagnostics failed for support_method='{support_method}': {e}")

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
    stride=10,
    gt_pts=None,
    gt_mask=None,
    pred_mask=None,
    zoom_bbox=None,
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

    print(f"[CORRESP DEBUG EXT] masks={'yes' if gt_mask is not None else 'no'}")
    print(f"[CORRESP DEBUG EXT] zoom_bbox={zoom_bbox}")

    fig = plt.figure(figsize=(16, 8.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1.0])

    ax_geom = fig.add_subplot(gs[:, 0])
    ax_w = fig.add_subplot(gs[0, 1])
    ax_drift = fig.add_subplot(gs[1, 1])

    def _valid_gt_geom(xy):
        xy = np.asarray(xy, float)
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2:
            return False
        return bool(np.isfinite(xy).all())

    # Geometry plot: pred + gt + explicit correspondence links
    ax_geom.set_title("Geometry + Correspondence Links")
    first_pred = True
    first_gt = True
    gt_geom_present_any = False
    step = max(1, int(stride))
    drift_curves = []

    # Decide zoom region in global coordinates
    all_pts = []
    for p in pts_segs:
        pp = np.asarray(p, float)
        if pp.ndim == 2 and pp.shape[1] == 2 and len(pp) >= 2:
            pp = pp[np.isfinite(pp).all(axis=1)]
            if len(pp) >= 2:
                all_pts.append(pp)
    if gt_pts_segs:
        for p in gt_pts_segs:
            pp = np.asarray(p, float)
            if pp.ndim == 2 and pp.shape[1] == 2 and len(pp) >= 2:
                pp = pp[np.isfinite(pp).all(axis=1)]
                if len(pp) >= 2:
                    all_pts.append(pp)

    x0 = y0 = x1 = y1 = None
    if zoom_bbox is not None and len(zoom_bbox) == 4:
        zx, zy, zw, zh = [int(v) for v in zoom_bbox]
        pad = 15
        x0 = max(0, zx - pad)
        y0 = max(0, zy - pad)
        x1 = zx + max(1, zw) + pad
        y1 = zy + max(1, zh) + pad
    elif all_pts:
        all_cat = np.vstack(all_pts)
        pad = 15
        x0 = max(0, int(np.floor(np.min(all_cat[:, 0]))) - pad)
        y0 = max(0, int(np.floor(np.min(all_cat[:, 1]))) - pad)
        x1 = int(np.ceil(np.max(all_cat[:, 0]))) + pad
        y1 = int(np.ceil(np.max(all_cat[:, 1]))) + pad

    # Clamp zoom to available mask bounds when possible
    Hm = Wm = None
    if gt_mask is not None:
        gm = np.asarray(gt_mask)
        if gm.ndim == 2:
            Hm, Wm = gm.shape
    if pred_mask is not None:
        pm = np.asarray(pred_mask)
        if pm.ndim == 2:
            if Hm is None or Wm is None:
                Hm, Wm = pm.shape
            else:
                Hm = min(Hm, pm.shape[0])
                Wm = min(Wm, pm.shape[1])

    if x0 is not None:
        if Hm is not None and Wm is not None:
            x0 = int(np.clip(x0, 0, max(0, Wm - 1)))
            y0 = int(np.clip(y0, 0, max(0, Hm - 1)))
            x1 = int(np.clip(x1, x0 + 1, Wm))
            y1 = int(np.clip(y1, y0 + 1, Hm))
        else:
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

    # Overlay masks (global coordinates via extent) with categorical colors:
    # 0 none (transparent), 1 GT-only, 2 Pred-only, 3 overlap.
    if x0 is not None and y0 is not None and x1 is not None and y1 is not None:
        gm_crop = None
        pm_crop = None
        if gt_mask is not None:
            gm = np.asarray(gt_mask)
            if gm.ndim == 2 and y1 > y0 and x1 > x0:
                gm_crop = (gm[y0:y1, x0:x1] > 0)
        if pred_mask is not None:
            pm = np.asarray(pred_mask)
            if pm.ndim == 2 and y1 > y0 and x1 > x0:
                pm_crop = (pm[y0:y1, x0:x1] > 0)

        if gm_crop is not None or pm_crop is not None:
            if gm_crop is None:
                gm_crop = np.zeros_like(pm_crop, dtype=bool)
            if pm_crop is None:
                pm_crop = np.zeros_like(gm_crop, dtype=bool)
            cls = np.zeros(gm_crop.shape, dtype=np.uint8)
            cls[np.logical_and(gm_crop, ~pm_crop)] = 1
            cls[np.logical_and(~gm_crop, pm_crop)] = 2
            cls[np.logical_and(gm_crop, pm_crop)] = 3
            from matplotlib.colors import ListedColormap
            cmap_cls = ListedColormap([
                (0.0, 0.0, 0.0, 0.0),   # none
                (1.00, 1.00, 1.00, 0.30),  # GT-only (white)
                (0.80, 0.25, 1.00, 0.34),  # Pred-only (pink/purple)
                (1.00, 0.75, 0.95, 0.44),  # overlap
            ])
            ax_geom.imshow(
                cls,
                cmap=cmap_cls,
                origin="upper",
                interpolation="nearest",
                extent=[x0, x1, y1, y0],
                zorder=0,
            )

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
            color="blue", lw=2.0, alpha=1.0,
            label="Pred midline" if first_pred else None
        )
        # Segment start/end markers (orange) to visualize local direction.
        ax_geom.scatter([p[0, 0]], [p[0, 1]], s=30, c="orange", edgecolors="black", linewidths=0.5, zorder=6)
        ax_geom.scatter([p[-1, 0]], [p[-1, 1]], s=30, c="orange", marker="s", edgecolors="black", linewidths=0.5, zorder=6)
        first_pred = False

        if gt_pts_segs:
            gxy = np.asarray(gt_pts_segs[i_seg], float)
            if _valid_gt_geom(gxy):
                ng = min(n, len(gxy))
                p2 = p[:ng]
                g2 = gxy[:ng]
                gw2 = gw[:ng]
                gt_geom_present_any = True
                ax_geom.plot(
                    g2[:, 0], g2[:, 1],
                    color="green", lw=1.6, alpha=0.9,
                    label="GT matched geometry" if first_gt else None
                )
                ax_geom.scatter([g2[0, 0]], [g2[0, 1]], s=26, c="orange", edgecolors="black", linewidths=0.5, zorder=6)
                ax_geom.scatter([g2[-1, 0]], [g2[-1, 1]], s=26, c="orange", marker="s", edgecolors="black", linewidths=0.5, zorder=6)
                first_gt = False
                for j in range(0, ng, step):
                    if np.isfinite(gw2[j]) and np.isfinite(p2[j]).all() and np.isfinite(g2[j]).all():
                        ax_geom.plot([p2[j, 0], g2[j, 0]], [p2[j, 1], g2[j, 1]],
                                     color="red", lw=0.9, alpha=0.8)
                ax_geom.scatter(p2[::step, 0], p2[::step, 1], s=8, c="blue", alpha=0.9)
                ax_geom.scatter(g2[::step, 0], g2[::step, 1], s=8, c="green", alpha=0.9)
                # 1D progression lag drift:
                # compare normalized arclength progress of pred vs GT at matched indices.
                sp = _arclen_param(p2)
                sg = _arclen_param(g2)
                if len(sp) >= 2 and len(sg) >= 2 and len(sp) == len(sg):
                    Lp = max(float(sp[-1]), 1e-9)
                    Lg = max(float(sg[-1]), 1e-9)
                    up = sp / Lp
                    ug = sg / Lg
                    lag_frac = up - ug
                    # Report lag in pixels along pred segment length for interpretability.
                    lag_px = lag_frac * Lp
                    drift_curves.append((sp, lag_px))
            else:
                for j in range(0, n, step):
                    c = "red" if np.isfinite(gw[j]) else "black"
                    ax_geom.plot(p[j, 0], p[j, 1], "o", color=c, markersize=3)
        else:
            for j in range(0, n, step):
                c = "red" if np.isfinite(gw[j]) else "black"
                ax_geom.plot(p[j, 0], p[j, 1], "o", color=c, markersize=3)

    ax_geom.set_aspect("equal")
    if x0 is not None and y0 is not None and x1 is not None and y1 is not None:
        ax_geom.set_xlim(x0, x1)
        ax_geom.set_ylim(y1, y0)
    else:
        ax_geom.invert_yaxis()
    if gt_pts is not None and not gt_geom_present_any:
        ax_geom.text(
            0.02, 0.98,
            "GT geometry unavailable (width-only stream)",
            transform=ax_geom.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="darkred",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="darkred", alpha=0.8),
        )
    ax_geom.set_facecolor("black")
    lg = ax_geom.legend()
    if lg is not None:
        frame = lg.get_frame()
        frame.set_facecolor("black")
        frame.set_edgecolor("white")
        frame.set_alpha(0.8)
        for txt in lg.get_texts():
            txt.set_color("white")

    # Width vs arclength (concatenated by segment, gapless)
    ax_w.set_title("Width vs Arc-Length")
    ax_w.set_ylabel("Width (px)")
    ax_w.set_xlabel("Arc length (px)")
    s_off = 0.0
    first_pw = True
    first_gw = True
    width_boundaries = []
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
        # Keep per-subsegment boundaries so starts/ends are visible in concatenated domain.
        width_boundaries.append({
            "s0": float(s[0]),
            "s1": float(s[-1]),
            "pw0": float(pw[0]),
            "pw1": float(pw[-1]),
            "gw0": float(gw[0]),
            "gw1": float(gw[-1]),
        })
        first_pw = False
        first_gw = False
    if width_boundaries:
        # Start markers (circle) and end markers (square) for both pred/gt.
        s_start = [b["s0"] for b in width_boundaries]
        s_end = [b["s1"] for b in width_boundaries]
        pw_start = [b["pw0"] for b in width_boundaries]
        pw_end = [b["pw1"] for b in width_boundaries]
        gw_start = [b["gw0"] for b in width_boundaries]
        gw_end = [b["gw1"] for b in width_boundaries]

        ax_w.scatter(s_start, pw_start, s=20, c="orange", edgecolors="black", linewidths=0.4, zorder=5)
        ax_w.scatter(s_end, pw_end, s=20, c="orange", marker="s", edgecolors="black", linewidths=0.4, zorder=5)
        ax_w.scatter(s_start, gw_start, s=20, c="orange", edgecolors="black", linewidths=0.4, zorder=5)
        ax_w.scatter(s_end, gw_end, s=20, c="orange", marker="s", edgecolors="black", linewidths=0.4, zorder=5)

        # Faint separators at subsegment starts (except first) like drift panel.
        for xs in s_start[1:]:
            ax_w.axvline(xs, color="orange", alpha=0.18, lw=0.8)
    ax_w.legend()
    ax_w.grid(True)

    ax_drift.set_title("Arc-Length Lag Drift (Pred - GT)")
    ax_drift.set_ylabel("Lag (px along pred)")
    ax_drift.set_xlabel("Arc length (px)")
    drift_means = []
    if drift_curves:
        s_cat = []
        d_cat = []
        s_off = 0.0
        drift_boundaries = []
        for s_drift, drift in drift_curves:
            s_now = np.asarray(s_drift, float)
            d_now = np.asarray(drift, float)
            if len(s_now) < 2 or len(d_now) < 2:
                continue
            n = min(len(s_now), len(d_now))
            s_now = s_now[:n] + s_off
            d_now = d_now[:n]
            s_cat.append(s_now)
            d_cat.append(d_now)
            # Per-subsegment boundaries in concatenated arclength domain.
            drift_boundaries.append((float(s_now[0]), float(d_now[0]), "start"))
            drift_boundaries.append((float(s_now[-1]), float(d_now[-1]), "end"))
            s_off = float(s_now[-1])
        if s_cat:
            s_cat = np.concatenate(s_cat)
            d_cat = np.concatenate(d_cat)
            ax_drift.plot(s_cat, d_cat, color="limegreen", alpha=0.95, lw=1.6)
            # Marker each subsegment start/end (not just global endpoints).
            if drift_boundaries:
                s_start = [x for (x, _, k) in drift_boundaries if k == "start"]
                d_start = [y for (_, y, k) in drift_boundaries if k == "start"]
                s_end = [x for (x, _, k) in drift_boundaries if k == "end"]
                d_end = [y for (_, y, k) in drift_boundaries if k == "end"]
                if s_start:
                    ax_drift.scatter(
                        s_start, d_start,
                        s=24, c="orange", edgecolors="black", linewidths=0.5, zorder=5
                    )
                if s_end:
                    ax_drift.scatter(
                        s_end, d_end,
                        s=24, c="orange", marker="s", edgecolors="black", linewidths=0.5, zorder=5
                    )
                # Optional faint separators at subsegment starts to aid reading.
                for xs in s_start[1:]:
                    ax_drift.axvline(xs, color="orange", alpha=0.18, lw=0.8)
            drift_f = d_cat[np.isfinite(d_cat)]
            if drift_f.size:
                drift_means.append(float(np.mean(np.abs(drift_f))))
    ax_drift.axhline(0.0, color="black", lw=1)
    ax_drift.grid(True)
    if drift_means:
        print(f"[CORRESP DEBUG EXT] mean |arc-lag| = {float(np.mean(drift_means)):.3f}px")
    else:
        print("[CORRESP DEBUG EXT] mean |arc-lag| = n/a (no GT geometry pair)")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f"correspondence_single_cid{cid}_b{branch_id}_s{seg_idx}.png",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
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
      - Stage 1: prune segments to effective atomic IDs (shared-only for weak matches)
      - Stage 2: optional branch matching
      - Stage 3: symmetric bite-union clipping
      - GT widths computed along FINAL clipped polyline
    """

    import os, json
    import concurrent.futures as _cf
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    #from helpers.metrics import normals_from_mask_for_midline

    os.makedirs(metrics_dir, exist_ok=True)
    _plot_executor = _cf.ThreadPoolExecutor(max_workers=4)
    _plot_futures = []

    def _async_savefig(fig, path, dpi=100):
        """Submit fig.savefig to thread pool; close fig after save."""
        def _do(f, p, d):
            try:
                f.savefig(p, dpi=d, bbox_inches="tight")
            finally:
                import matplotlib.pyplot as _plt
                _plt.close(f)
        _plot_futures.append(_plot_executor.submit(_do, fig, path, dpi))

    def _drain_async_savefigs():
        # Drain all deferred savefigs before returning
        for _f in _plot_futures:
            try:
                _f.result(timeout=60)
            except Exception as _e:
                print(f"[ASYNC SAVEFIG] failed: {_e}")
        _plot_executor.shutdown(wait=False)

    # ---------------- variant tag (output isolation only) ----------------
    variant_id = str(variant_id or "main").strip()
    file_tag = "main"
    gt_variant_key = "manual_gt"
    print(f"[WIDTH] variant_id={variant_id} -> gt_variant_key={gt_variant_key}")

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
    output_dir = os.path.join(metrics_dir, midline_type or "unknown", crack_type or mode)
    os.makedirs(output_dir, exist_ok=True)

    def _collect_member_ids(raw_members):
        out = []
        for m in (raw_members or []):
            if isinstance(m, dict):
                aid = m.get("atomic_id", m.get("id", None))
            else:
                aid = m
            if aid is None:
                continue
            out.append(str(aid))
        return out

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
                    members = _collect_member_ids(e.get("members") or [])
                    if members:
                        key = frozenset(members)
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
        members_set = set(_collect_member_ids(members))

        return mid_segs, mid_meta, derived_segs, derived_meta, bite_obj, members_set

    # ---------------- accumulators ----------------
    coords, diffs, bboxes = [], [], []
    rows = []
    midline_metric_rows = []   # for combined midline diagnostics

    width_pairs = []
    width_metric_rows = []
    invalid_logs = []
    failed_segment_csv = os.path.join(output_dir, "stage5_strict_skips.csv")

    def _log_failed_segment_row(
        *,
        image,
        cid,
        reason,
        L_geom,
        pred_len_total=0,
        gt_len_total=0,
        branch_id="atomic",
        seg_idx="group",
        num_gt_candidates_same_branch=0,
    ):
        # Reuse the existing failed-segment CSV mechanism/schema.
        _append_csv_row(
            failed_segment_csv,
            [
                str(image),
                str(cid),
                str(branch_id),
                str(seg_idx),
                int(max(0, L_geom)),
                str(reason),
                int(max(0, pred_len_total)),
                int(max(0, gt_len_total)),
                int(max(0, num_gt_candidates_same_branch)),
            ],
            header=[
                "image",
                "cid",
                "branch_id",
                "seg_idx",
                "L_geom",
                "reason",
                "pred_len_total",
                "gt_len_total",
                "num_gt_candidates_same_branch",
            ],
        )

    def _serialize_member_list(v):
        if v is None:
            return None
        try:
            vals = sorted([str(x) for x in list(v)])
        except Exception:
            vals = [str(v)]
        return json.dumps(vals)

    def log_invalid(
        *,
        image,
        cid,
        level,
        reason,
        length,
        n_segments,
        pred_members=None,
        gt_members=None,
        overlap=None,
        entity_id=None,
        branch_id=None,
        extra_info=None,
    ):
        invalid_logs.append(
            {
                "image": str(image),
                "cid": str(cid),
                "level": str(level),
                "entity_id": None if entity_id is None else str(entity_id),
                "reason": str(reason),
                "length": (
                    float(length)
                    if length is not None and np.isfinite(length)
                    else np.nan
                ),
                "n_segments": int(n_segments) if n_segments is not None else np.nan,
                "pred_members": _serialize_member_list(pred_members),
                "gt_members": _serialize_member_list(gt_members),
                "overlap": (
                    float(overlap)
                    if overlap is not None and np.isfinite(overlap)
                    else np.nan
                ),
                "branch_id": None if branch_id is None else str(branch_id),
                "extra_info": None if extra_info is None else str(extra_info),
            }
        )

    def _should_midline_metrics(*, run_mode, run_midline_type, geometry_type):
        """
        Gate which midline metric variants are emitted.
        geometry_type is expected to be "orig" or "derived".
        """
        g = str(geometry_type).lower()
        m = str(run_mode).lower()
        t = str(run_midline_type).lower()

        if t in {"manual", "et"} and g == "orig":
            return False
        if m == "atomic" and g == "orig":
            return False
        return True

    # debug dir for opsec artifacts
    opsec_dir = os.path.join(metrics_dir, midline_type or "unknown", "opsec_debug")
    os.makedirs(opsec_dir, exist_ok=True)

    # Debug-only forensic trace for Stage 4.5 -> Stage 5 provenance.
    DEBUG_TOPOLOGY_TRACE = True

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
        Returns GT width vector from payload/supervision with variant-aware preference.
        """
        for src in [gt_entry_obj, crack_obj]:
            if not isinstance(src, dict):
                continue

            if "gt_widths" in src:
                w = _coerce_gt_width_vec(src.get("gt_widths", None))
                if w is not None:
                    return w

            gtn = src.get("gt_normals")
            if isinstance(gtn, dict):
                w = _coerce_gt_width_vec(gtn.get("width_px", None))
                if w is not None:
                    return w
        return None

    def _polyline_length(xy):
        xy = np.asarray(xy, float)
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2:
            return 0.0
        d = np.sqrt(((xy[1:] - xy[:-1]) ** 2).sum(axis=1))
        if d.size == 0:
            return 0.0
        d = d[np.isfinite(d)]
        if d.size == 0:
            return 0.0
        return float(np.sum(d))

    def _get_gt_geom_full_atomic(gt_entry_obj):
        """
        Atomic GT geometry stream (single polyline), variant-aware.
        """
        if not isinstance(gt_entry_obj, dict):
            return None

        for k in ("midline_auto_centered", "midline"):
            xy = _coerce_polyline_xy(gt_entry_obj.get(k, None))
            if xy is not None and len(xy) >= 2:
                return np.asarray(xy, float)
        return None

    def _align_atomic_gt_geom_to_pred(gt_xy_raw, pred_concat_xy, cid_dbg):
        """
        Validate/normalize atomic GT geometry for optional diagnostics only.
        No resampling is allowed here: Stage 1 must not create index-space
        correspondence. We only allow endpoint-direction flip, then require
        sample-count compatibility with predicted support.
        Returns (gt_xy_kept, strategy_label) or (None, reason).
        """
        gt = np.asarray(gt_xy_raw, float)
        pr = np.asarray(pred_concat_xy, float)

        if gt.ndim != 2 or gt.shape[1] != 2 or len(gt) < 2:
            return None, "invalid_gt_polyline"
        if pr.ndim != 2 or pr.shape[1] != 2 or len(pr) < 2:
            return None, "invalid_pred_polyline"

        gt = gt[np.isfinite(gt).all(axis=1)]
        pr = pr[np.isfinite(pr).all(axis=1)]
        if len(gt) < 2:
            return None, "gt_nonfinite_after_filter"
        if len(pr) < 2:
            return None, "pred_nonfinite_after_filter"

        pr_len = _polyline_length(pr)
        gt_len = _polyline_length(gt)
        if pr_len <= 1e-9 or gt_len <= 1e-9:
            return None, "degenerate_length"

        # Enforce direction relative to pred endpoints.
        d_fwd = float(np.linalg.norm(pr[0] - gt[0]) + np.linalg.norm(pr[-1] - gt[-1]))
        d_rev = float(np.linalg.norm(pr[0] - gt[-1]) + np.linalg.norm(pr[-1] - gt[0]))
        flipped = bool(d_rev < d_fwd)
        if flipped:
            gt = gt[::-1].copy()
            d_fwd, d_rev = d_rev, d_fwd

        # No geometry resampling in atomic Stage 1.
        # Require native sample-count compatibility for optional gt_match_seg slicing.
        if len(gt) != len(pr):
            return None, f"sample_count_mismatch(gt={len(gt)},pred={len(pr)})"

        end_err = float(np.linalg.norm(pr[0] - gt[0]) + np.linalg.norm(pr[-1] - gt[-1]))
        len_ratio = float(gt_len / max(pr_len, 1e-9))
        len_mismatch = abs(len_ratio - 1.0)

        # Conservative acceptance guardrail.
        end_tol = max(20.0, 0.35 * pr_len)
        len_tol = 0.85
        if (not np.isfinite(end_err)) or end_err > end_tol:
            return None, f"endpoint_error_too_high({end_err:.3f}>{end_tol:.3f})"
        if (not np.isfinite(len_ratio)) or len_mismatch > len_tol:
            return None, f"length_ratio_out_of_range({len_ratio:.3f})"

        strategy = "flip_only" if flipped else "native_noop"
        print(
            f"[ATOMIC GT ALIGN] cid={cid_dbg} pred_total={len(pr)} gt_raw={len(gt_xy_raw)} "
            f"strategy={strategy} endpoint_error={end_err:.3f} len_ratio={len_ratio:.3f}"
        )
        return gt, strategy

    def _lookup_atomic_gt_entry(cid_val):
        cid_s = str(cid_val)
        cands = [cid_s]
        if cid_s.startswith("atomic_"):
            cands.append(cid_s.split("atomic_", 1)[1])
        for c in cands:
            if c in gt_sup_atomic:
                return gt_sup_atomic[c]
        return None

    def _coerce_polyline_xy(raw):
        if raw is None:
            return None
        try:
            arr = np.asarray(raw, float)
        except Exception:
            return None

        if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) >= 2:
            arr = arr[np.isfinite(arr).all(axis=1)]
            return arr if len(arr) >= 2 else None

        # Handle flattened [ [x,y], [None,None], ... ] style by taking first valid segment.
        try:
            segs = _split_on_nans(raw)
        except Exception:
            segs = []
        for s in segs:
            s = np.asarray(s, float)
            if s.ndim == 2 and s.shape[1] == 2 and len(s) >= 2:
                s = s[np.isfinite(s).all(axis=1)]
                if len(s) >= 2:
                    return s
        return None

    def _resample_polyline_to_len(xy, L):
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

            def _meta_atomic_ids(mm):
                out_ids = []
                aid0 = mm.get("atomic_id", None)
                if aid0 is not None:
                    out_ids.append(str(aid0))
                aid_list = mm.get("atomic_ids", None)
                if isinstance(aid_list, (list, tuple, set)):
                    for a in aid_list:
                        if a is not None:
                            out_ids.append(str(a))
                # preserve order while removing duplicates
                return list(dict.fromkeys(out_ids))

            if scope_members is not None:
                a_ids = _meta_atomic_ids(m)
                if a_ids and not any(a in scope_members for a in a_ids):
                    continue

            # Prefer explicit branch namespace when present (critical for ET split segments);
            # fallback to endpoint pairing only when branch_id is unavailable.
            bid_raw = m.get("branch_id", None)
            if bid_raw is not None:
                try:
                    k = ("branch_id", int(bid_raw))
                except Exception:
                    k = ("branch_id", str(bid_raw))
            else:
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

            L = float(np.sum([_poly_length(S) for S in segs_k]))

            aids = set()
            for mm in meta_k:
                aid = mm.get("atomic_id", None)
                if aid is not None:
                    aids.add(str(aid))
                aid_list = mm.get("atomic_ids", None)
                if isinstance(aid_list, (list, tuple, set)):
                    for a in aid_list:
                        if a is not None:
                            aids.add(str(a))

            out.append(
                {
                    "branch_id": int(bi),
                    "segs": segs_k,
                    "meta": meta_k,
                    "endpoints": (np.asarray(a, float), np.asarray(b, float)),
                    "length": float(L),
                    "atomic_ids": aids,
                }
            )

        return out

    def _branch_shared_length(branch_obj, shared_ids, *, allow_none_atomic=False):
        L = 0.0
        for Sx, mx in zip(branch_obj.get("segs", []), branch_obj.get("meta", [])):
            if Sx is None or len(Sx) < 2:
                continue
            mmx = mx if isinstance(mx, dict) else {}
            aid = mmx.get("atomic_id", None)
            if aid is None:
                if not allow_none_atomic:
                    continue
            elif str(aid) not in shared_ids:
                continue
            L += _poly_length(np.asarray(Sx, float))
        return float(L)

    def _greedy_match_branches_geom(
        gt_br,
        pr_br,
        *,
        lambda_seg=0.20,
        return_diag=False,
        allow_multi_pred_per_gt=False,
    ):
        """
        Shared-support objective:
        final_score = min(Lp_shared, Lg_shared) * (1 + lambda_seg * ns/max(np,ng,1))
        where shared-support uses only atomics present in BOTH candidate branches.
        """
        scored_candidates = []
        shared_candidates = []
        pred_branch_ids = {int(p["branch_id"]) for p in (pr_br or [])}
        gt_all_no_atomics = bool(gt_br) and all(
            len({str(a) for a in (g.get("atomic_ids") or set())}) == 0
            for g in (gt_br or [])
        )

        for g in (gt_br or []):
            g_id = int(g["branch_id"])
            Ag = {str(a) for a in (g.get("atomic_ids") or set())}
            for p in (pr_br or []):
                p_id = int(p["branch_id"])
                Ap = {str(a) for a in (p.get("atomic_ids") or set())}
                As = sorted(Ap) if gt_all_no_atomics else sorted(Ap & Ag)
                if not As:
                    continue

                Lp_shared = _branch_shared_length(p, set(As))
                Lg_shared = (
                    float(g.get("length", 0.0))
                    if gt_all_no_atomics
                    else _branch_shared_length(g, set(As))
                )
                L_shared = float(min(Lp_shared, Lg_shared))

                ns = int(len(As))
                np_seg = int(max(len(Ap), 1))
                ng_seg = int(max(len(Ag), 1))

                cand = {
                    "gt_branch_id": g_id,
                    "pr_branch_id": p_id,
                    "shared_ids": As,
                    "ns": ns,
                    "np": np_seg,
                    "ng": ng_seg,
                    "Lp_shared": float(Lp_shared),
                    "Lg_shared": float(Lg_shared),
                    "L_shared": float(L_shared),
                    "gt_atomic_ids": sorted(list(Ag)),
                    "pred_atomic_ids": sorted(list(Ap)),
                }
                shared_candidates.append(cand)

                if not np.isfinite(L_shared) or L_shared <= 0.0:
                    continue

                # New Stage-2 objective: shared-support length + weak multiplicative shared-segment bias.
                score_len = L_shared
                score_count = 1.0 + float(lambda_seg) * (float(ns) / float(max(np_seg, ng_seg, 1)))
                final_score = float(score_len * score_count)
                cand["score"] = final_score
                scored_candidates.append(cand)

        scored_candidates = sorted(scored_candidates, key=lambda d: d["score"], reverse=True)
        used_g = set()
        used_p = set()
        matches = []
        for c in scored_candidates:
            gb = int(c["gt_branch_id"])
            pb = int(c["pr_branch_id"])
            if pb in used_p:
                continue
            if (not allow_multi_pred_per_gt) and (gb in used_g):
                continue
            used_g.add(gb)
            used_p.add(pb)
            matches.append((gb, pb, float(c["score"])))

        if not return_diag:
            return matches

        pred_with_shared = {int(c["pr_branch_id"]) for c in shared_candidates}
        pred_with_scored = {int(c["pr_branch_id"]) for c in scored_candidates}
        diag = {
            "scored_candidates": scored_candidates,
            "shared_candidates": shared_candidates,
            "pred_no_shared": set(pred_branch_ids - pred_with_shared),
            "pred_zero_shared_length": set(pred_with_shared - pred_with_scored),
            "gt_all_no_atomics": bool(gt_all_no_atomics),
        }
        return matches, diag

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

        segs = []
        meta = []

        def _read_segs(seg_key, packed_key=None):
            out = [np.asarray(s, float) for s in (gt_entry_obj.get(seg_key) or []) if s is not None and len(s) >= 2]
            if out:
                return out
            if packed_key is not None and gt_entry_obj.get(packed_key) is not None:
                return [np.asarray(s, float) for s in _split_on_nans(gt_entry_obj.get(packed_key)) if s is not None and len(s) >= 2]
            return []

        segs = _read_segs("midline_segments", "midline")
        meta = (
            gt_entry_obj.get("midline_segments_meta")
            or gt_entry_obj.get("segments_meta")
            or ((gt_entry_obj.get("dominance_meta") or {}).get("segments_meta") or [])
        )

        # === DIAG: trace what meta source and atomic_id values were loaded ===
        _dom_seg_meta_diag = ((gt_entry_obj.get("dominance_meta") or {}).get("segments_meta") or [])
        print(f"[EXTRACT META DIAG] kind={gt_entry_obj.get('kind','?')} id={gt_entry_obj.get('id','?')}")
        print(f"[EXTRACT META DIAG]   midline_segments_meta present={gt_entry_obj.get('midline_segments_meta') is not None}")
        print(f"[EXTRACT META DIAG]   dominance_meta present={gt_entry_obj.get('dominance_meta') is not None}")
        print(f"[EXTRACT META DIAG]   dominance_meta.segments_meta len={len(_dom_seg_meta_diag)}")
        for _i, _dsm in enumerate(_dom_seg_meta_diag[:4]):
            print(f"[EXTRACT META DIAG]   dom_seg_meta[{_i}] = {_dsm}")
        print(f"[EXTRACT META DIAG]   resolved meta source len={len(meta) if isinstance(meta, list) else 'NOT_LIST'}")
        for _i, _m in enumerate((meta or [])[:4]):
            print(f"[EXTRACT META DIAG]   meta[{_i}] atomic_id={(_m or {}).get('atomic_id','MISSING')} branch_id={(_m or {}).get('branch_id','MISSING')}")
        # === END DIAG ===

        # Backfill atomic_id from dominance_meta.segments_meta when the primary
        # meta source (often midline_segments_meta) does not carry atomic tags.
        if isinstance(meta, list):
            dom_seg_meta = (
                (gt_entry_obj.get("dominance_meta") or {}).get("segments_meta") or []
            )
            for i, m in enumerate(meta):
                if isinstance(m, dict) and m.get("atomic_id") is None and i < len(dom_seg_meta):
                    dsm = dom_seg_meta[i]
                    if isinstance(dsm, dict):
                        if dsm.get("atomic_id") is not None:
                            m["atomic_id"] = str(dsm["atomic_id"])
                        elif isinstance(dsm.get("atomic_ids"), (list, tuple)) and dsm.get("atomic_ids"):
                            aid0 = dsm.get("atomic_ids")[0]
                            if aid0 is not None:
                                m["atomic_id"] = str(aid0)
                                m.setdefault("atomic_ids", [str(a) for a in dsm.get("atomic_ids") if a is not None])

                # Secondary fallback: infer primary atomic_id from atomic_ids list already on meta.
                if isinstance(m, dict) and m.get("atomic_id") is None:
                    aid_list = m.get("atomic_ids")
                    if isinstance(aid_list, (list, tuple)) and aid_list:
                        aid0 = aid_list[0]
                        if aid0 is not None:
                            m["atomic_id"] = str(aid0)

        # === DIAG: confirm what atomic_ids look like after backfill ===
        print(f"[EXTRACT META POST-BACKFILL]   meta atomic_ids after backfill: {[(_m or {}).get('atomic_id') for _m in (meta or [])[:4]]}")
        # === END DIAG ===

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

    def _clip_polyline_into_run_indices(S, remove_mask, H, W, min_pts=2):
        """
        Index-preserving variant of _clip_polyline_into_runs.
        Returns (kept_idx_runs, removed_idx_runs) where each run is a 1D int array.
        """
        S = np.asarray(S, float)
        if S is None or len(S) < 2 or remove_mask is None:
            idx = np.arange(len(S), dtype=int) if S is not None else np.asarray([], int)
            return ([idx] if idx.size >= min_pts else []), []

        kept_idx_runs = []
        removed_idx_runs = []
        buf_k = []
        buf_r = []

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
                    buf_k.append(i)
                buf_k.append(i + 1)
                if len(buf_r) >= min_pts:
                    removed_idx_runs.append(np.asarray(buf_r, int))
                buf_r = []
            else:
                if not buf_r:
                    buf_r.append(i)
                buf_r.append(i + 1)
                if len(buf_k) >= min_pts:
                    kept_idx_runs.append(np.asarray(buf_k, int))
                buf_k = []

        if len(buf_k) >= min_pts:
            kept_idx_runs.append(np.asarray(buf_k, int))
        if len(buf_r) >= min_pts:
            removed_idx_runs.append(np.asarray(buf_r, int))

        return kept_idx_runs, removed_idx_runs

    def _build_segment_records(segs_in, meta_in, width_full=None, *, fill_nan=False):
        segs_valid = []
        meta_valid = []
        for i, (S, m) in enumerate(zip(segs_in or [], meta_in or [])):
            if S is None or len(S) < 2:
                continue
            mm = m if isinstance(m, dict) else {}
            segs_valid.append(np.asarray(S, float))
            meta_valid.append(dict(mm))

        total_pts = int(np.sum([len(S) for S in segs_valid]))
        if total_pts <= 0:
            return []

        width_stream = None
        if (not fill_nan) and width_full is not None:
            width_stream = np.asarray(width_full, float).reshape(-1)

        records = []
        off = 0
        for i, (S, mm) in enumerate(zip(segs_valid, meta_valid)):
            L = int(len(S))
            if fill_nan or width_stream is None:
                w = np.full((L,), np.nan, float)
            else:
                s0 = int(off)
                s1 = int(off + L)
                if s0 >= width_stream.size:
                    w = np.full((L,), np.nan, float)
                else:
                    sl = np.asarray(width_stream[s0:min(s1, width_stream.size)], float)
                    if sl.size < L:
                        w = np.full((L,), np.nan, float)
                        if sl.size > 0:
                            w[:sl.size] = sl
                    else:
                        w = sl[:L]
            off += L
            bid = _safe_int(mm.get("branch_id"), None)
            sid = _safe_int(mm.get("seg_idx"), None)
            if bid is None:
                bid = int(i)
            if sid is None:
                sid = int(i)
            aid = mm.get("atomic_id", None)
            rec = {
                "pts": np.asarray(S, float),
                "width": np.asarray(w, float),
                "branch_id": int(bid),
                "seg_idx": int(sid),
                "atomic_id": None if aid is None else str(aid),
                "meta": dict(mm),
            }
            rec["meta"]["branch_id"] = int(rec["branch_id"])
            rec["meta"]["seg_idx"] = int(rec["seg_idx"])
            if rec["atomic_id"] is not None:
                rec["meta"]["atomic_id"] = rec["atomic_id"]
            records.append(rec)
        return records

    def _records_to_segs_meta(records_in):
        segs_out, meta_out = [], []
        for r in records_in or []:
            pts = np.asarray(r.get("pts", None), float) if r is not None else None
            if pts is None or pts.ndim != 2 or len(pts) < 2:
                continue
            mm = dict(r.get("meta", {}) if isinstance(r.get("meta", {}), dict) else {})
            mm["branch_id"] = int(_safe_int(r.get("branch_id"), mm.get("branch_id", 0)))
            mm["seg_idx"] = int(_safe_int(r.get("seg_idx"), mm.get("seg_idx", 0)))
            if r.get("atomic_id", None) is not None:
                mm["atomic_id"] = str(r.get("atomic_id"))
            segs_out.append(pts)
            meta_out.append(mm)
        return segs_out, meta_out

    def _apply_union_dominance_to_records(
        records_in,
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

        kept_records = []
        removed_records = []
        for rec in (records_in or []):
            pts = np.asarray(rec.get("pts", None), float)
            w = np.asarray(rec.get("width", None), float).reshape(-1)
            if pts is None or pts.ndim != 2 or len(pts) < 2:
                continue
            if len(w) != len(pts):
                if len(w) < len(pts):
                    ww = np.full((len(pts),), np.nan, float)
                    if len(w) > 0:
                        ww[:len(w)] = w
                    w = ww
                else:
                    w = np.asarray(w[:len(pts)], float)

            rm = _union_bite_for_branch(rec.get("branch_id", None))
            if rm is None:
                out = dict(rec)
                out["pts"] = np.asarray(pts, float)
                out["width"] = np.asarray(w, float)
                kept_records.append(out)
                continue

            keep_idx_runs, rem_idx_runs = _clip_polyline_into_run_indices(
                pts, rm, H_full, W_full, min_pts=2
            )

            for idx in keep_idx_runs:
                out = dict(rec)
                out["pts"] = np.asarray(pts[idx], float)
                out["width"] = np.asarray(w[idx], float)
                kept_records.append(out)
            for idx in rem_idx_runs:
                out = dict(rec)
                out["pts"] = np.asarray(pts[idx], float)
                out["width"] = np.asarray(w[idx], float)
                removed_records.append(out)

        return kept_records, removed_records

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
                # DEBUG: inspect expected vs available derived keys before failing
                try:
                    print(f"[DERIVED MATCH DBG] cid={cid} MISSING key={k}")
                    print(f"[DERIVED MATCH DBG] midline_meta count={len(mid_keep_meta or [])} derived_meta count={len(dmeta_in or [])}")

                    exp = []
                    for _mm in (mid_keep_meta or []):
                        if not isinstance(_mm, dict):
                            continue
                        exp.append(_k_ab(_mm))
                    print(f"[DERIVED MATCH DBG] expected(midline) keys sample={exp[:20]}")

                    avail = []
                    for _dm in (dmeta_in or []):
                        if not isinstance(_dm, dict):
                            continue
                        avail.append(_k_ab(_dm))
                    print(f"[DERIVED MATCH DBG] available(derived) keys sample={avail[:40]}")
                    print(f"[DERIVED MATCH DBG] derived atomic_ids={sorted({a for (a, _) in avail if a is not None})}")
                except Exception as _e:
                    print(f"[DERIVED MATCH DBG] failed: {_e}")
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
        cid_opsec_dir = os.path.join(opsec_dir, f"cid{cid}") if mode == "combined" else None
        if DEBUG_TOPOLOGY_TRACE and mode == "combined":
            topo_dbg_dir = cid_opsec_dir

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

        derived_concat = (
            np.vstack([np.asarray(s, float) for s in dsegs if s is not None and len(s) >= 2])
            if dsegs else None
        )

        e1, e2 = _get_edges(crack)
        widths_geo = None
        m_edge = min(len(e1), len(e2))
        if m_edge >= 2:
            widths_geo = np.linalg.norm(e1[:m_edge] - e2[:m_edge], axis=1)

        if isinstance(crack, dict) and "pred_widths" in crack:
            predw_full_any = np.asarray(crack["pred_widths"], float).reshape(-1)
        else:
            predw_full_any = _get_pred_width_full(crack, derived_concat, widths_geo)

        if (len(e1) < 2 or len(e2) < 2) and (predw_full_any is None or np.asarray(predw_full_any).size < 2):
            if mode == "atomic" or str(cid).startswith("atomic_"):
                _log_failed_segment_row(
                    image=base_name,
                    cid=cid,
                    reason="atomic_no_usable_pred_width_trace",
                    L_geom=int(pred_group_n_segments),
                    pred_len_total=0,
                    gt_len_total=0,
                    branch_id="atomic",
                    seg_idx="group",
                )
            continue

        pred_group_len = float(
            np.sum(
                [
                    _polyline_length(np.asarray(s, float))
                    for s in (segs or [])
                    if s is not None and len(s) >= 2
                ]
            )
        )
        pred_group_n_segments = int(
            np.sum([1 for s in (segs or []) if s is not None and len(s) >= 2])
        )

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
                _log_failed_segment_row(
                    image=base_name,
                    cid=cid,
                    reason="atomic_no_usable_pred_width_trace",
                    L_geom=int(pred_group_n_segments),
                    pred_len_total=0,
                    gt_len_total=0,
                    branch_id="atomic",
                    seg_idx="group",
                )
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
                _log_failed_segment_row(
                    image=base_name,
                    cid=cid,
                    reason="atomic_no_usable_gt_width_trace",
                    L_geom=int(pred_group_n_segments),
                    pred_len_total=int(len(predw_full_any)),
                    gt_len_total=0,
                    branch_id="atomic",
                    seg_idx="group",
                )
                continue
            gtw_full_any = np.asarray(gtw_full_any, float).reshape(-1)

            total_geom = int(sum(len(s) for s in segs if s is not None and len(s) >= 2))
            if total_geom < 2:
                print(f"[WIDTH DEBUG] atomic cid={cid} has <2 derived geometry samples -> skip")
                _log_failed_segment_row(
                    image=base_name,
                    cid=cid,
                    reason="atomic_invalid_geometry",
                    L_geom=int(total_geom),
                    pred_len_total=int(len(predw_full_any)),
                    gt_len_total=int(len(gtw_full_any)),
                    branch_id="atomic",
                    seg_idx="group",
                )
                continue

            # Part 1 keeps native correspondence identity but must keep traces usable.
            # If stream lengths drift, resample widths to geometry length instead of skipping.
            if (len(predw_full_any) != total_geom) or (len(gtw_full_any) != total_geom):
                def _resample_width_trace_to_len(trace_in, L_out):
                    arr = np.asarray(trace_in, float).reshape(-1)
                    L_out = int(L_out)
                    if L_out <= 0:
                        return np.asarray([], float)
                    if arr.size == 0:
                        out = np.empty((L_out,), float)
                        out[:] = np.nan
                        return out
                    m = np.isfinite(arr)
                    arr = arr[m]
                    if arr.size == 0:
                        out = np.empty((L_out,), float)
                        out[:] = np.nan
                        return out
                    if arr.size == 1:
                        out = np.empty((L_out,), float)
                        out[:] = float(arr[0])
                        return out
                    if arr.size == L_out:
                        return arr.astype(float, copy=False)
                    u_src = np.linspace(0.0, 1.0, num=int(arr.size))
                    u_dst = np.linspace(0.0, 1.0, num=L_out)
                    return np.interp(u_dst, u_src, arr).astype(float, copy=False)

                print(
                    f"[WIDTH DEBUG] atomic cid={cid} -> align (atomic width length mismatch) "
                    f"pred_len={len(predw_full_any)} gt_len={len(gtw_full_any)} total_geom={total_geom}"
                )
                log_invalid(
                    image=base_name,
                    cid=cid,
                    level="group",
                    reason="atomic_width_length_mismatch_resampled",
                    length=float(total_geom),
                    n_segments=int(pred_group_n_segments),
                    pred_members=[str(cid)],
                    gt_members=[str(cid)] if gt_entry_atomic is not None else None,
                    overlap=np.nan,
                    entity_id=cid,
                    branch_id=None,
                    extra_info=(
                        f"pred_len={len(predw_full_any)} "
                        f"gt_len={len(gtw_full_any)} total_geom={total_geom}"
                    ),
                )
                _log_failed_segment_row(
                    image=base_name,
                    cid=cid,
                    reason="atomic_width_length_mismatch_resampled",
                    L_geom=int(total_geom),
                    pred_len_total=int(len(predw_full_any)),
                    gt_len_total=int(len(gtw_full_any)),
                    branch_id="atomic",
                    seg_idx="group",
                )
                predw_full_any = _resample_width_trace_to_len(predw_full_any, total_geom)
                gtw_full_any = _resample_width_trace_to_len(gtw_full_any, total_geom)

            pred_concat_xy = np.vstack([np.asarray(s, float) for s in segs if s is not None and len(s) >= 2])
            gt_geom_aligned = None
            gt_geom_strategy = "not_attempted"
            gt_geom_raw = _get_gt_geom_full_atomic(gt_entry_atomic)
            if gt_geom_raw is None:
                gt_geom_strategy = "missing_gt_geometry_stream"
                print(f"[ATOMIC GT WARN] cid={cid} reason={gt_geom_strategy} GT geometry not attached")
                _log_failed_segment_row(
                    image=base_name,
                    cid=cid,
                    reason=str(gt_geom_strategy),
                    L_geom=int(total_geom),
                    pred_len_total=int(len(predw_full_any)),
                    gt_len_total=int(len(gtw_full_any)),
                    branch_id="atomic",
                    seg_idx="group",
                )
            else:
                gt_geom_aligned, gt_geom_strategy = _align_atomic_gt_geom_to_pred(
                    gt_geom_raw,
                    pred_concat_xy,
                    str(cid),
                )
                if gt_geom_aligned is None:
                    print(f"[ATOMIC GT WARN] cid={cid} reason={gt_geom_strategy} GT geometry not attached")
                    _log_failed_segment_row(
                        image=base_name,
                        cid=cid,
                        reason=str(gt_geom_strategy),
                        L_geom=int(total_geom),
                        pred_len_total=int(len(predw_full_any)),
                        gt_len_total=int(len(gtw_full_any)),
                        branch_id="atomic",
                        seg_idx="group",
                    )

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

                predw = predw_full_any[off:off + L].astype(float, copy=False)
                gtw = gtw_full_any[off:off + L].astype(float, copy=False)
                gt_match_seg = None
                if gt_geom_aligned is not None:
                    gt_match_seg = np.asarray(gt_geom_aligned[off:off + L], float)
                    if gt_match_seg.ndim != 2 or gt_match_seg.shape[1] != 2 or len(gt_match_seg) < 2:
                        print(
                            f"[ATOMIC GT WARN] cid={cid} reason=invalid_segment_slice(off={off},L={L}) "
                            f"GT geometry not attached for this segment"
                        )
                        gt_match_seg = None

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
                    "gt_match_seg": gt_match_seg,
                    "gt_geom_strategy": gt_geom_strategy,

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
        pred_key = frozenset(pred_members)
        gt_entry = gt_sup_combined.get(pred_key)

        if gt_entry is None and gt_sup_combined:
            pm = set(pred_members)
            best = None
            for k, e in gt_sup_combined.items():
                gm = set(_collect_member_ids(e.get("members", []) or []))
                inter = len(pm & gm)
                denom = max(1, max(len(pm), len(gm)))
                u = inter / denom
                if best is None or u > best[0]:
                    best = (u, e)
            if best is not None:
                gt_entry = best[1]

        if gt_entry is None:
            log_invalid(
                image=base_name,
                cid=cid,
                level="group",
                reason="no_gt_match",
                entity_id=cid,
                length=pred_group_len,
                n_segments=pred_group_n_segments,
                pred_members=pred_members,
                gt_members=None,
                overlap=np.nan,
            )

        if gt_entry is None:
            gt_members = set()
            bite_gt = None
        else:
            gt_members = set(_collect_member_ids(gt_entry.get("members", []) or []))
            bite_gt = None
            dom_gt = gt_entry.get("dominance_meta") or gt_entry.get("dominance") or {}
            if isinstance(dom_gt, dict) and "bite" in dom_gt:
                bite_gt = dom_gt["bite"]

        shared = pred_members & gt_members
        overlap = len(shared) / max(1, max(len(pred_members), len(gt_members)))

        if overlap >= 0.70:
            match_quality = "strong"
        elif overlap >= 0.50:
            match_quality = "weak"
        else:
            match_quality = "invalid"

        if match_quality == "invalid":
            log_invalid(
                image=base_name,
                cid=cid,
                level="group",
                reason="overlap_below_threshold",
                entity_id=cid,
                length=pred_group_len,
                n_segments=pred_group_n_segments,
                pred_members=pred_members,
                gt_members=gt_members,
                overlap=overlap,
                extra_info=f"n_shared={len(shared)}",
            )
            print(
                f"[WIDTH DEBUG] combined cid={cid} overlap={overlap:.3f} "
                f"quality=invalid -> SKIP"
            )
            continue

        if abs(float(overlap) - 1.0) < 1e-9:
            effective_members = set(pred_members)
        else:
            effective_members = set(shared)

        raw_members = crack.get("members", []) or []
        if abs(float(overlap) - 1.0) >= 1e-9:
            members_iter = []
            for m in raw_members:
                aid = m.get("atomic_id", m.get("id", None)) if isinstance(m, dict) else m
                if aid is not None and str(aid) in effective_members:
                    members_iter.append(m)
        else:
            members_iter = list(raw_members)
        members_iter_ids = set(_collect_member_ids(members_iter))
        if not members_iter_ids:
            members_iter_ids = set(effective_members)

        print(
            f"[WIDTH DEBUG] cid={cid} quality={match_quality} overlap={overlap:.3f} "
            f"shared_members={sorted(shared)} effective_members={sorted(effective_members)}"
        )
        
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

            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
            ax.set_title(f"GT SUP — RAW DOMINANCE (cid={cid})")
            ax.axis("off")

            ax.imshow(union, cmap="hot", interpolation="nearest", alpha=0.9)

            # Overlay stored midlines (GLOBAL → LOCAL)
            segs, _ = _extract_gt_stream_segments_and_meta(gt_entry, "midline")
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
            out = os.path.join(out_dir, "gt_sup_dom_raw.png")
            _async_savefig(fig, out, dpi=100)

            print(f"[GT_SUP DEBUG] wrote {out}")

        debug_plot_gt_sup_dominance_raw(
            cid=cid,
            gt_entry=gt_entry,
            out_dir=cid_opsec_dir,
        )

        # --------------------------------------------
        # Stage 1: prune segments by effective atomic IDs
        # --------------------------------------------
        pruned_segs = []
        pruned_meta = []

        for i, (S, m) in enumerate(zip(segs, seg_meta)):
            if S is None or len(S) < 2:
                continue
            aid = m.get("atomic_id")
            if aid is not None and str(aid) not in effective_members:
                print(f"[STAGE2 EXCL] cid={cid} level=atomic atomic={aid} reason=not_in_effective_members")
                log_invalid(
                    image=base_name,
                    cid=cid,
                    level="atomic",
                    reason="not_in_effective_members",
                    entity_id=aid,
                    length=_poly_length(np.asarray(S, float)),
                    n_segments=1,
                    pred_members=pred_members,
                    gt_members=gt_members,
                    overlap=overlap,
                    branch_id=m.get("branch_id") if isinstance(m, dict) else None,
                )
                continue
            pruned_segs.append(np.asarray(S, float))
            pruned_meta.append(dict(m))

        if not pruned_segs:
            print(f"[WIDTH DEBUG] cid={cid} -> NO SEGMENTS AFTER PRUNE")
            continue

        print(f"[WIDTH DEBUG] cid={cid} kept {len(pruned_segs)} segments after effective-member prune")

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
        # Stage-2 helpers: branch signatures + shared-support branch scoring
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
                "length": total length,
                "atomic_ids": set(str)
            }
            """
            groups = {}
            for S, m in zip(segs_in, meta_in):
                if S is None or len(S) < 2:
                    continue
                m = m if isinstance(m, dict) else {}

                def _meta_atomic_ids(mm):
                    out_ids = []
                    aid0 = mm.get("atomic_id", None)
                    if aid0 is not None:
                        out_ids.append(str(aid0))
                    aid_list = mm.get("atomic_ids", None)
                    if isinstance(aid_list, (list, tuple, set)):
                        for a in aid_list:
                            if a is not None:
                                out_ids.append(str(a))
                    return list(dict.fromkeys(out_ids))

                # optional: drop segments that are out of scope
                if scope_members is not None:
                    a_ids = _meta_atomic_ids(m)
                    if a_ids and not any(a in scope_members for a in a_ids):
                        continue

                bid_raw = m.get("branch_id", None)
                if bid_raw is not None:
                    try:
                        k = ("branch_id", int(bid_raw))
                    except Exception:
                        k = ("branch_id", str(bid_raw))
                else:
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

                # total length
                L = float(np.sum([_poly_length(S) for S in segs_k]))

                aids = set()
                for mm in meta_k:
                    aid = mm.get("atomic_id", None)
                    if aid is not None:
                        aids.add(str(aid))
                    aid_list = mm.get("atomic_ids", None)
                    if isinstance(aid_list, (list, tuple, set)):
                        for a in aid_list:
                            if a is not None:
                                aids.add(str(a))

                out.append({
                    "branch_id": int(bi),   # synthetic, stable only inside this call
                    "segs": segs_k,
                    "meta": meta_k,
                    "endpoints": (np.asarray(a, float), np.asarray(b, float)),
                    "length": float(L),
                    "atomic_ids": aids,
                })

            return out

        def _branch_shared_length(branch_obj, shared_ids, *, allow_none_atomic=False):
            L = 0.0
            for Sx, mx in zip(branch_obj.get("segs", []), branch_obj.get("meta", [])):
                if Sx is None or len(Sx) < 2:
                    continue
                mmx = mx if isinstance(mx, dict) else {}
                aid = mmx.get("atomic_id", None)
                if aid is None:
                    if not allow_none_atomic:
                        continue
                elif str(aid) not in shared_ids:
                    continue
                L += _poly_length(np.asarray(Sx, float))
            return float(L)

        def _greedy_match_branches_geom(
            gt_br,
            pr_br,
            *,
            lambda_seg=0.20,
            return_diag=False,
            allow_multi_pred_per_gt=False,
        ):
            """
            Shared-support objective:
            final_score = min(Lp_shared, Lg_shared) * (1 + lambda_seg * ns/max(np,ng,1))
            where shared-support uses only atomics present in BOTH candidate branches.
            """
            scored_candidates = []
            shared_candidates = []
            pred_branch_ids = {int(p["branch_id"]) for p in (pr_br or [])}
            gt_all_no_atomics = bool(gt_br) and all(
                len({str(a) for a in (g.get("atomic_ids") or set())}) == 0
                for g in (gt_br or [])
            )

            for g in (gt_br or []):
                g_id = int(g["branch_id"])
                Ag = {str(a) for a in (g.get("atomic_ids") or set())}
                for p in (pr_br or []):
                    p_id = int(p["branch_id"])
                    Ap = {str(a) for a in (p.get("atomic_ids") or set())}
                    As = sorted(Ap) if gt_all_no_atomics else sorted(Ap & Ag)
                    if not As:
                        continue

                    Lp_shared = _branch_shared_length(p, set(As))
                    Lg_shared = (
                        float(g.get("length", 0.0))
                        if gt_all_no_atomics
                        else _branch_shared_length(g, set(As))
                    )
                    L_shared = float(min(Lp_shared, Lg_shared))

                    ns = int(len(As))
                    np_seg = int(max(len(Ap), 1))
                    ng_seg = int(max(len(Ag), 1))

                    cand = {
                        "gt_branch_id": g_id,
                        "pr_branch_id": p_id,
                        "shared_ids": As,
                        "ns": ns,
                        "np": np_seg,
                        "ng": ng_seg,
                        "Lp_shared": float(Lp_shared),
                        "Lg_shared": float(Lg_shared),
                        "L_shared": float(L_shared),
                        "gt_atomic_ids": sorted(list(Ag)),
                        "pred_atomic_ids": sorted(list(Ap)),
                    }
                    shared_candidates.append(cand)

                    if not np.isfinite(L_shared) or L_shared <= 0.0:
                        continue

                    # New Stage-2 objective: shared-support length + weak multiplicative shared-segment bias.
                    score_len = L_shared
                    score_count = 1.0 + float(lambda_seg) * (float(ns) / float(max(np_seg, ng_seg, 1)))
                    final_score = float(score_len * score_count)
                    cand["score"] = final_score
                    scored_candidates.append(cand)

            scored_candidates = sorted(scored_candidates, key=lambda d: d["score"], reverse=True)
            used_g = set()
            used_p = set()
            matches = []
            for c in scored_candidates:
                gb = int(c["gt_branch_id"])
                pb = int(c["pr_branch_id"])
                if pb in used_p:
                    continue
                if (not allow_multi_pred_per_gt) and (gb in used_g):
                    continue
                used_g.add(gb)
                used_p.add(pb)
                matches.append((gb, pb, float(c["score"])))

            if not return_diag:
                return matches

            pred_with_shared = {int(c["pr_branch_id"]) for c in shared_candidates}
            pred_with_scored = {int(c["pr_branch_id"]) for c in scored_candidates}
            diag = {
                "scored_candidates": scored_candidates,
                "shared_candidates": shared_candidates,
                "pred_no_shared": set(pred_branch_ids - pred_with_shared),
                "pred_zero_shared_length": set(pred_with_shared - pred_with_scored),
                "gt_all_no_atomics": bool(gt_all_no_atomics),
            }
            return matches, diag


        # ============================================================
        # (A) GT prune:
        #   1) scope to members_iter first
        #   2) then apply effective-members gate (weak => shared-only)
        # ============================================================
        if gt_entry is not None:
            gt_segs_all, gt_meta_all = _extract_gt_stream_segments_and_meta(gt_entry, "midline")
            gt_missing_atomic_segs = []
            gt_missing_atomic_meta = []

            print(
                f"[STAGE2 DBG] cid={cid} GT segs={len(gt_segs_all)} "
                f"GT meta={len(gt_meta_all)} effective_members={sorted(effective_members)}"
            )

            if len(gt_segs_all) == len(gt_meta_all) and len(gt_segs_all) > 0:
                for i, (Sg, mg) in enumerate(zip(gt_segs_all, gt_meta_all)):
                    if Sg is None or len(Sg) < 2:
                        continue
                    mg = mg if isinstance(mg, dict) else {}
                    aid = mg.get("atomic_id")

                    # --- scope gate: drop GT segments not in THIS predicted crack ---
                    if aid is not None and str(aid) not in members_iter_ids:
                        print(f"[STAGE2 DBG] SKIP GT seg#{i} atomic={aid} (out-of-scope)")
                        continue

                    if aid is None:
                        gt_missing_atomic_segs.append(np.asarray(Sg, float))
                        gt_missing_atomic_meta.append(dict(mg))
                        print(f"[STAGE2 META DIAG] seg#{i} full meta dict = {mg}")
                        print(f"[STAGE2 DBG] HOLD GT seg#{i} atomic=None (fallback candidate)")
                        continue

                    if str(aid) not in effective_members:
                        print(f"[STAGE2 DBG] DROP GT seg#{i} atomic={aid} (not in effective set)")
                        continue

                    gt_pruned_segs.append(np.asarray(Sg, float))
                    gt_pruned_meta.append(dict(mg))
            else:
                # For partial overlap we cannot trust GT meta-less segments to be in shared support.
                if abs(float(overlap) - 1.0) >= 1e-9:
                    print(
                        f"[STAGE2 DBG] cid={cid} partial-overlap with GT meta mismatch; "
                        f"dropping GT segments to avoid non-shared evaluation"
                    )
                else:
                    for i, Sg in enumerate(gt_segs_all):
                        if Sg is None or len(Sg) < 2:
                            continue
                        gt_pruned_segs.append(np.asarray(Sg, float))
                        gt_pruned_meta.append({})

            if (not gt_pruned_segs) and gt_missing_atomic_segs:
                gt_pruned_segs.extend(gt_missing_atomic_segs)
                gt_pruned_meta.extend(gt_missing_atomic_meta)
                print(
                    f"[STAGE2 DBG] cid={cid} GT fallback -> kept {len(gt_missing_atomic_segs)} "
                    f"segments with atomic=None (strict effective-members prune yielded zero)"
                )
                log_invalid(
                    image=base_name,
                    cid=cid,
                    level="group",
                    reason="gt_missing_atomic_id_fallback",
                    length=float(np.sum([_poly_length(np.asarray(s, float)) for s in gt_missing_atomic_segs])),
                    n_segments=int(len(gt_missing_atomic_segs)),
                    pred_members=pred_members,
                    gt_members=gt_members,
                    overlap=overlap,
                    entity_id=cid,
                    branch_id=None,
                )

            print(f"[STAGE2 DBG] cid={cid} GT kept {len(gt_pruned_segs)} segs after scoped/effective prune")


        # ============================================================
        # (B) symmetric branch matching (GEOMETRY-BASED)
        #   NOTE: do NOT trust GT/PRED branch_id numbering!
        # ============================================================
        gt_br = _build_branch_table_geom(gt_pruned_segs, gt_pruned_meta, scope_members=effective_members)
        pr_br = _build_branch_table_geom(pruned_segs, pruned_meta, scope_members=effective_members)
        branch_diag = {
            "scored_candidates": [],
            "shared_candidates": [],
            "pred_no_shared": set(),
            "pred_zero_shared_length": set(),
        }

        if pr_br:
            if gt_br:
                gt_all_no_atomics_for_match = bool(gt_br) and all(
                    len(g.get("atomic_ids") or set()) == 0
                    for g in gt_br
                )
                et_multi_match = str(variant_id or "").strip().lower().startswith("et")
                matches, branch_diag = _greedy_match_branches_geom(
                    gt_br,
                    pr_br,
                    lambda_seg=0.20,
                    return_diag=True,
                    allow_multi_pred_per_gt=(gt_all_no_atomics_for_match or et_multi_match),
                )
            else:
                matches = []
                branch_diag["pred_no_shared"] = {int(p["branch_id"]) for p in pr_br}

            matched_gt_branch_ids = {g for (g, p, s) in matches}
            matched_pred_branch_ids = {p for (g, p, s) in matches}

            for c in branch_diag.get("scored_candidates", []):
                print(
                    f"[STAGE2 SCORE] cid={cid} pred_branch={c['pr_branch_id']} gt_branch={c['gt_branch_id']} "
                    f"score={c['score']:.3f} L_shared={c['L_shared']:.3f} "
                    f"ns={c['ns']} np={c['np']} ng={c['ng']}"
                )

            if matches:
                print(
                    f"[STAGE2 DBG] cid={cid} branch matches (shared-support): "
                    f"GT={sorted(matched_gt_branch_ids)} "
                    f"PRED={sorted(matched_pred_branch_ids)} "
                    f"scores={[round(s,3) for (_, _, s) in matches]}"
                )
            else:
                print(f"[STAGE2 DBG] cid={cid} no matched branches under shared-support scoring")

            best_scored_by_pred = {}
            for c in branch_diag.get("scored_candidates", []):
                pb = int(c["pr_branch_id"])
                if pb not in best_scored_by_pred or float(c["score"]) > float(best_scored_by_pred[pb]["score"]):
                    best_scored_by_pred[pb] = c

            best_shared_by_pred = {}
            for c in branch_diag.get("shared_candidates", []):
                pb = int(c["pr_branch_id"])
                if pb not in best_shared_by_pred or float(c["L_shared"]) > float(best_shared_by_pred[pb]["L_shared"]):
                    best_shared_by_pred[pb] = c

            for pb in pr_br:
                bid = int(pb.get("branch_id", -1))
                if bid in matched_pred_branch_ids:
                    continue

                if bid in branch_diag.get("pred_no_shared", set()):
                    reason = "no_shared_atomics"
                    cand = None
                elif bid in branch_diag.get("pred_zero_shared_length", set()):
                    reason = "zero_shared_length"
                    cand = best_shared_by_pred.get(bid, None)
                else:
                    reason = "lost_in_matching"
                    cand = best_scored_by_pred.get(bid, None)

                gt_members_branch = None
                extra = None
                if cand is not None:
                    gt_members_branch = cand.get("gt_atomic_ids", None)
                    if reason == "lost_in_matching":
                        extra = (
                            f"best_score={float(cand.get('score', np.nan)):.3f}; "
                            f"best_gt_branch={cand.get('gt_branch_id')}; "
                            f"shared={cand.get('shared_ids', [])}"
                        )
                    else:
                        extra = (
                            f"L_shared={float(cand.get('L_shared', np.nan)):.3f}; "
                            f"best_gt_branch={cand.get('gt_branch_id')}; "
                            f"shared={cand.get('shared_ids', [])}"
                        )

                print(f"[STAGE2 EXCL] cid={cid} level=branch branch={bid} reason={reason}")
                log_invalid(
                    image=base_name,
                    cid=cid,
                    level="branch",
                    reason=reason,
                    entity_id=bid,
                    branch_id=bid,
                    length=float(pb.get("length", np.nan)),
                    n_segments=len(pb.get("segs", []) or []),
                    pred_members=pb.get("atomic_ids", None),
                    gt_members=gt_members_branch,
                    overlap=overlap,
                    extra_info=extra,
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
            seg2bid = _assign_synth_branch_ids(pruned_segs, pruned_meta, scope_members=effective_members)
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
                    aid = m.get("atomic_id", None) if isinstance(m, dict) else None
                    print(
                        f"[STAGE2 EXCL] cid={cid} level=atomic atomic={aid} "
                        f"reason=unused_after_branch_matching branch={bid}"
                    )
                    log_invalid(
                        image=base_name,
                        cid=cid,
                        level="atomic",
                        reason="unused_after_branch_matching",
                        entity_id=aid,
                        branch_id=bid,
                        length=_poly_length(np.asarray(S, float)),
                        n_segments=1,
                        pred_members=pred_members,
                        gt_members=gt_members,
                        overlap=overlap,
                    )

            pruned_segs, pruned_meta = keep_s, keep_m

        if matched_gt_branch_ids is not None:
            seg2bid = _assign_synth_branch_ids(gt_pruned_segs, gt_pruned_meta, scope_members=effective_members)
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
            effective_members,
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
                gt_all, gt_meta_all = _extract_gt_stream_segments_and_meta(gt_entry, "midline")

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
                dpi=100,
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

            legend_items = []
            has_kept = any(S is not None and len(S) >= 2 for S in (gt_kept or [])) or any(
                S is not None and len(S) >= 2 for S in (pred_kept or [])
            )
            has_drop = any(S is not None and len(S) >= 2 for S in (gt_dropped or [])) or any(
                S is not None and len(S) >= 2 for S in (pred_dropped or [])
            )
            has_bbox = bool(bb)

            if has_kept:
                legend_items.append(Line2D([0], [0], color=col_keep, lw=3, label="Kept segments"))
            if has_drop:
                legend_items.append(Line2D([0], [0], color=col_drop, lw=3, label="Dropped segments"))
            if has_bbox:
                legend_items.append(Line2D([0], [0], color="dodgerblue", lw=1.5, label="BBox"))

            if legend_items:
                axes[1].legend(
                    handles=legend_items,
                    loc="lower right",
                    fontsize=6,
                    framealpha=0.8,
                    markerscale=0.7,
                    handlelength=1.5,
                    borderpad=0.5,
                )

            member_str = ", ".join(sorted(effective_members)) if effective_members else ", ".join(sorted(pred_members))
            fig.suptitle(
                f"Stage-2 prune — cid={cid}\nAtomic members: [{member_str}]",
                fontsize=11,
                fontweight="bold",
            )

            out = os.path.join(out_dir, "stage2_prune.png")
            _async_savefig(fig, out, dpi=100)

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
            pred_members=members_iter_ids,
            effective_members=effective_members,
            out_dir=cid_opsec_dir,
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
        elif isinstance(gt_entry, dict):
            gt_plot_segs, gt_plot_meta = _extract_gt_stream_segments_and_meta(gt_entry, "midline")
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

            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
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

            _async_savefig(fig, out_png, dpi=100)
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

            fig, ax = plt.subplots(figsize=(7, 6), dpi=100)
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

            _async_savefig(fig, out_png, dpi=100)
            print(f"[STAGE4] wrote {out_png}")

        # ----------------------------
        # Decode unions in bite-local coordinates (Stage0 style)
        # ----------------------------
        gt_bbox, gt_union_local, _ = _get_bite_union_local(dom_gt)
        pr_bbox, pr_union_local, _ = _get_bite_union_local(dom_pred)

        print(f"[STAGE4] cid={cid} GT bite bbox={gt_bbox} union_px={(0 if gt_union_local is None else int(gt_union_local.sum()))}")
        print(f"[STAGE4] cid={cid} PR bite bbox={pr_bbox} union_px={(0 if pr_union_local is None else int(pr_union_local.sum()))}")

        # Build seg lists for overlay:
        gt_segs_for_plot, _ = _extract_gt_stream_segments_and_meta(gt_entry, "midline")
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
            1, 2, figsize=(12, 6), dpi=100, sharex=True, sharey=True
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

        # dominance legend (dynamic: only include present overlays)
        if np.any(dom_label == 1):
            legend_handles.append(Line2D([0], [0], color="#e41a1c", lw=6, label="GT-only loss"))
        if np.any(dom_label == 2):
            legend_handles.append(Line2D([0], [0], color="#377eb8", lw=6, label="Pred-only loss"))
        if np.any(dom_label == 3):
            legend_handles.append(Line2D([0], [0], color="#984ea3", lw=6, label="GT ∩ Pred"))

        if legend_handles:
            axes[1].legend(
                handles=legend_handles,
                loc="lower right",
                fontsize=6,
                framealpha=0.8,
                markerscale=0.7,
                handlelength=1.5,
                borderpad=0.5,
            )

        outB = os.path.join(
            cid_opsec_dir,
            f"stage4_dominance_bite_{midline_type}_{mode}.png",
        )
        os.makedirs(cid_opsec_dir, exist_ok=True)
        _async_savefig(fig, outB, dpi=100)

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

        # Build authoritative width streams BEFORE dominance clipping.
        # Offsets are only allowed here while converting flat width vectors into segment records.
        pred_source_stage45 = predw_full_any
        if pred_source_stage45 is None or np.asarray(pred_source_stage45).size < 2:
            pred_source_stage45 = widths_geo
        if pred_source_stage45 is None or np.asarray(pred_source_stage45).size < 2:
            print(f"[STAGE4.5] cid={cid} no usable predicted width source")
            continue
        pred_source_stage45 = np.asarray(pred_source_stage45, float).reshape(-1)

        gtw_source_stage45 = _get_gt_width_full(crack, gt_entry)
        if gtw_source_stage45 is None or np.asarray(gtw_source_stage45).size < 2:
            print(f"[STAGE4.5] cid={cid} no usable GT width source in payload/supervision")
            continue
        gtw_source_stage45 = np.asarray(gtw_source_stage45, float).reshape(-1)

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

        # Build symmetric records (pts + width + keys) for dominance clipping.
        pred_mid_stage2_records = _build_segment_records(
            pred_mid_stage2_segs,
            pred_mid_stage2_meta,
            width_full=None,
            fill_nan=True,
        )
        pred_der_stage2_records = _build_segment_records(
            pred_der_stage2_segs,
            pred_der_stage2_meta,
            width_full=pred_source_stage45,
            fill_nan=False,
        )
        gt_stage5_source_records = _build_segment_records(
            gt_stage5_source_segs,
            gt_stage5_source_meta,
            width_full=gtw_source_stage45,
            fill_nan=False,
        )

        # ============================================================
        # Stage 4.5 — APPLY UNION DOMINANCE (PRED MID + PRED DERIVED + GT)
        # ============================================================
        pred_mid_dom_records, bite_pruned_pred_mid_records = _apply_union_dominance_to_records(
            pred_mid_stage2_records,
            loss_masks_pred_by_branch=loss_masks_pred_by_branch,
            loss_masks_gt_by_branch=loss_masks_gt_by_branch,
            H_full=H,
            W_full=W,
        )
        pred_der_dom_records, bite_pruned_pred_der_records = _apply_union_dominance_to_records(
            pred_der_stage2_records,
            loss_masks_pred_by_branch=loss_masks_pred_by_branch,
            loss_masks_gt_by_branch=loss_masks_gt_by_branch,
            H_full=H,
            W_full=W,
        )
        gt_dom_records, bite_pruned_gt_records = _apply_union_dominance_to_records(
            gt_stage5_source_records,
            loss_masks_pred_by_branch=loss_masks_pred_by_branch,
            loss_masks_gt_by_branch=loss_masks_gt_by_branch,
            H_full=H,
            W_full=W,
        )

        pred_mid_dom_segs, pred_mid_dom_meta = _records_to_segs_meta(pred_mid_dom_records)
        pred_der_dom_segs, pred_der_dom_meta = _records_to_segs_meta(pred_der_dom_records)
        gt_dom_segs, gt_dom_meta = _records_to_segs_meta(gt_dom_records)
        bite_pruned_pred_mid = [np.asarray(r["pts"], float) for r in bite_pruned_pred_mid_records if r is not None and len(r.get("pts", [])) >= 2]
        bite_pruned_pred_der = [np.asarray(r["pts"], float) for r in bite_pruned_pred_der_records if r is not None and len(r.get("pts", [])) >= 2]
        bite_pruned_gt_segs = [np.asarray(r["pts"], float) for r in bite_pruned_gt_records if r is not None and len(r.get("pts", [])) >= 2]

        bite_pruned_pred_segs = bite_pruned_pred_der
        gt_stage5_segs = gt_dom_segs
        gt_stage5_meta = gt_dom_meta

        pruned_segs = pred_mid_dom_segs
        pruned_meta = pred_mid_dom_meta

        if DEBUG_TOPOLOGY_TRACE:
            _dump_json(
                os.path.join(topo_dbg_dir, "stage45_counts.json"),
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
                os.path.join(topo_dbg_dir, "stage45_seg_lengths.csv"),
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

        # ------------------------------------------------------------
        # Stage 4.75 — ALIGN GT branch_id namespace to PRED namespace
        # Root cause of Stage5 key misses: GT/PRED branch_id labels may differ
        # after dominance even when geometry matches.
        # ------------------------------------------------------------
        if mode == "combined" and gt_stage5_segs and pred_der_dom_segs:
            try:
                gt_br_45 = _build_branch_table_geom(gt_stage5_segs, gt_stage5_meta, scope_members=effective_members)
                pr_br_45 = _build_branch_table_geom(pred_der_dom_segs, pred_der_dom_meta, scope_members=effective_members)
                m_45 = _greedy_match_branches_geom(gt_br_45, pr_br_45, lambda_seg=0.20)
            except Exception as e:
                m_45 = []
                print(f"[STAGE4.75 WARN] branch remap failed to build: {e}")

            if m_45:
                gt_to_pred_bid = {int(g): int(p) for (g, p, _c) in m_45}
                n_remap = 0
                for mm in (gt_stage5_meta or []):
                    if not isinstance(mm, dict):
                        continue
                    b_old = _safe_int(mm.get("branch_id"), None)
                    if b_old is None:
                        continue
                    b_new = gt_to_pred_bid.get(int(b_old), None)
                    if b_new is None:
                        continue
                    if int(b_new) != int(b_old):
                        mm["branch_id"] = int(b_new)
                        n_remap += 1

                # Keep record namespace in sync with metadata for strict Stage-5 key joins.
                for rec in (gt_dom_records or []):
                    b_old = _safe_int(rec.get("branch_id"), None)
                    if b_old is None:
                        continue
                    b_new = gt_to_pred_bid.get(int(b_old), None)
                    if b_new is None:
                        continue
                    rec["branch_id"] = int(b_new)
                    mmr = rec.get("meta", {})
                    if isinstance(mmr, dict):
                        mmr["branch_id"] = int(b_new)

                print(
                    f"[STAGE4.75] GT->PRED branch remap applied: {gt_to_pred_bid} "
                    f"(meta_updates={n_remap})"
                )
            else:
                print("[STAGE4.75 WARN] no GT/PRED branch remap pairs found; keeping raw branch IDs")

        # ============================================================
        # Stage 5 - WIDTH ATTACHMENT (STRICT, NO GEOMETRY TRUNCATION)
        # ============================================================
        if not pred_der_dom_segs:
            raise RuntimeError("[STAGE5 FATAL] no derived prediction geometry after dominance")
        if not gt_stage5_segs:
            raise RuntimeError("[STAGE5 FATAL] no GT geometry after dominance")

        final_pred_segs = []
        stage4_pairs = []

        def _norm_stage5_key_from_record(rec_obj):
            if not isinstance(rec_obj, dict):
                return None
            b = _safe_int(rec_obj.get("branch_id"), None)
            s = _safe_int(rec_obj.get("seg_idx"), None)
            mm = rec_obj.get("meta", {})
            if b is None and isinstance(mm, dict):
                b = _safe_int(mm.get("branch_id"), None)
            if s is None and isinstance(mm, dict):
                s = _safe_int(mm.get("seg_idx"), None)
            if b is None or s is None:
                return None
            return (int(b), int(s))

        def _iter_valid_records(records_in):
            for rec in (records_in or []):
                if not isinstance(rec, dict):
                    continue
                pts = np.asarray(rec.get("pts", None), float)
                if pts.ndim != 2 or len(pts) < 2:
                    continue
                yield rec

        stage5_slice_csv = os.path.join(topo_dbg_dir, "stage5_slices.csv")
        if DEBUG_TOPOLOGY_TRACE:
            os.makedirs(topo_dbg_dir, exist_ok=True)

        pred_total_pts = int(
            np.sum([len(np.asarray(rec.get("pts", []), float)) for rec in _iter_valid_records(pred_der_dom_records)])
        )
        gt_total_pts = int(
            np.sum([len(np.asarray(rec.get("pts", []), float)) for rec in _iter_valid_records(gt_dom_records)])
        )

        print(
            f"[STAGE5 PRECHECK] cid={cid} "
            f"pred_width_len={pred_total_pts} gt_width_len={gt_total_pts} "
            f"geom_pts_total={int(sum(len(S) for S in (pred_der_dom_segs or []) if S is not None))}"
        )
        if pred_total_pts != gt_total_pts:
            print(
                f"[STAGE5 WARN] cid={cid} width vector length mismatch "
                f"(pred={pred_total_pts} gt={gt_total_pts})"
            )

        gt_bucket = {}
        for gt_rec in _iter_valid_records(gt_dom_records):
            key = _norm_stage5_key_from_record(gt_rec)
            if key is None:
                continue
            gt_bucket.setdefault(key, []).append(gt_rec)

        print("\n[DEBUG KEY INSPECTION]")
        print("GT KEYS:")
        for k in gt_bucket.keys():
            print("   ", k, type(k[0]), type(k[1]))
        print("PRED KEYS:")
        for pr_rec in _iter_valid_records(pred_der_dom_records):
            kb = pr_rec.get("branch_id")
            ks = pr_rec.get("seg_idx")
            print("   ", (kb, ks), type(kb), type(ks))
        print("[END DEBUG]\n")

        stage5_strict_skip_csv = os.path.join(topo_dbg_dir, "stage5_strict_skips.csv")
        used_gt_ids = set()

        stage5_unmatched_skips = 0
        for pred_rec in _iter_valid_records(pred_der_dom_records):
            pts = np.asarray(pred_rec.get("pts", None), float)
            L = int(len(pts))

            mm_pred = pred_rec.get("meta", {}) if isinstance(pred_rec.get("meta", {}), dict) else {}
            seg_idx_dbg = pred_rec.get("seg_idx", mm_pred.get("seg_idx"))
            branch_dbg = pred_rec.get("branch_id", mm_pred.get("branch_id"))
            atomic_dbg = pred_rec.get("atomic_id", mm_pred.get("atomic_id", None))
            key = _norm_stage5_key_from_record(pred_rec)
            if key is None:
                stage5_unmatched_skips += 1
                print(
                    f"[STAGE5 STRICT SKIP] cid={cid} branch_id={branch_dbg} seg_idx={seg_idx_dbg} "
                    f"reason=invalid_key_metadata"
                )
                log_invalid(
                    image=base_name,
                    cid=cid,
                    level="atomic",
                    reason="invalid_key_metadata",
                    length=_polyline_length(pts),
                    n_segments=1,
                    pred_members=pred_members,
                    gt_members=gt_members,
                    overlap=overlap,
                    entity_id=atomic_dbg,
                    branch_id=branch_dbg,
                    extra_info=f"seg_idx={seg_idx_dbg}",
                )
                if DEBUG_TOPOLOGY_TRACE:
                    _append_csv_row(
                        stage5_strict_skip_csv,
                        [
                            str(base_name),
                            str(cid),
                            str(branch_dbg),
                            str(seg_idx_dbg),
                            int(L),
                            "invalid_key_metadata",
                            int(pred_total_pts),
                            int(gt_total_pts),
                            0,
                        ],
                        header=[
                            "image",
                            "cid",
                            "branch_id",
                            "seg_idx",
                            "L_geom",
                            "reason",
                            "pred_len_total",
                            "gt_len_total",
                            "num_gt_candidates_same_branch",
                        ],
                    )
                continue

            predw = np.asarray(pred_rec.get("width", np.full((L,), np.nan, float)), float).reshape(-1)
            if predw.size != L:
                stage5_unmatched_skips += 1
                print(
                    f"[STAGE5 STRICT SKIP] cid={cid} branch_id={branch_dbg} seg_idx={seg_idx_dbg} "
                    f"reason=invalid_width_alignment_pred predw_len={predw.size} geom_len={L}"
                )
                log_invalid(
                    image=base_name,
                    cid=cid,
                    level="atomic",
                    reason="invalid_width_alignment_pred",
                    length=_polyline_length(pts),
                    n_segments=1,
                    pred_members=pred_members,
                    gt_members=gt_members,
                    overlap=overlap,
                    entity_id=atomic_dbg,
                    branch_id=branch_dbg,
                    extra_info=f"predw_len={predw.size}; geom_len={L}",
                )
                if DEBUG_TOPOLOGY_TRACE:
                    _append_csv_row(
                        stage5_strict_skip_csv,
                        [
                            str(base_name),
                            str(cid),
                            str(branch_dbg),
                            str(seg_idx_dbg),
                            int(L),
                            "invalid_width_alignment_pred",
                            int(pred_total_pts),
                            int(gt_total_pts),
                            0,
                        ],
                        header=[
                            "image",
                            "cid",
                            "branch_id",
                            "seg_idx",
                            "L_geom",
                            "reason",
                            "pred_len_total",
                            "gt_len_total",
                            "num_gt_candidates_same_branch",
                        ],
                    )
                continue

            # GT width stream: STRICT local by (branch_id, seg_idx); no fallback matching.
            gt_list = gt_bucket.get(key, [])
            gt_list_avail = [rec for rec in gt_list if id(rec) not in used_gt_ids]
            if gt_list_avail:
                rec0 = gt_list_avail[0]
                used_gt_ids.add(id(rec0))
                gt_match_seg = np.asarray(rec0.get("pts", None), float)
                gtw = np.asarray(rec0.get("width", np.full((len(gt_match_seg),), np.nan, float)), float).reshape(-1)
                if gtw.size != len(gt_match_seg):
                    stage5_unmatched_skips += 1
                    reason = "invalid_width_alignment_gt"
                    print(
                        f"[STAGE5 STRICT SKIP] cid={cid} branch_id={branch_dbg} seg_idx={seg_idx_dbg} "
                        f"reason={reason} gtw_len={gtw.size} gt_geom_len={len(gt_match_seg)}"
                    )
                    log_invalid(
                        image=base_name,
                        cid=cid,
                        level="atomic",
                        reason=reason,
                        length=_polyline_length(pts),
                        n_segments=1,
                        pred_members=pred_members,
                        gt_members=gt_members,
                        overlap=overlap,
                        entity_id=atomic_dbg,
                        branch_id=branch_dbg,
                        extra_info=f"gtw_len={gtw.size}; gt_geom_len={len(gt_match_seg)}; key={key}",
                    )
                    if DEBUG_TOPOLOGY_TRACE:
                        _append_csv_row(
                            stage5_strict_skip_csv,
                            [
                                str(base_name),
                                str(cid),
                                str(branch_dbg),
                                str(seg_idx_dbg),
                                int(L),
                                str(reason),
                                int(pred_total_pts),
                                int(gt_total_pts),
                                0,
                            ],
                            header=[
                                "image",
                                "cid",
                                "branch_id",
                                "seg_idx",
                                "L_geom",
                                "reason",
                                "pred_len_total",
                                "gt_len_total",
                                "num_gt_candidates_same_branch",
                            ],
                        )
                    continue
                gt_seg_len = int(len(gt_match_seg))
                gt_match_mode = "segment_local_match_strict"
            else:
                # Fallback: for secondary sub-segments, try (branch_id, 0) and allow reuse.
                fallback_processed = False
                fallback_key = (key[0], 0)
                if key[1] != 0 and fallback_key != key:
                    fb_list = gt_bucket.get(fallback_key, [])
                    fb_list_avail = [rec for rec in fb_list if id(rec) not in used_gt_ids]
                    if fb_list_avail:
                        rec0 = fb_list_avail[0]
                        # Intentionally do NOT mark as used: GT seg_idx=0 may absorb multiple sub-segments.
                        gt_match_seg = np.asarray(rec0.get("pts", None), float)
                        gtw = np.asarray(
                            rec0.get("width", np.full((len(gt_match_seg),), np.nan, float)),
                            float,
                        ).reshape(-1)
                        if gtw.size == len(gt_match_seg):
                            gt_seg_len = int(len(gt_match_seg))
                            gt_match_mode = "branch_fallback_seg0"
                            fallback_processed = True

                if not fallback_processed:
                    stage5_unmatched_skips += 1
                    b_only = key[0]
                    n_same_branch = int(
                        np.sum(
                            [
                                1
                                for k_stage5, arr_stage5 in gt_bucket.items()
                                if isinstance(k_stage5, tuple) and len(k_stage5) == 2 and int(k_stage5[0]) == int(b_only)
                                for _ in arr_stage5
                            ]
                        )
                    )
                    if len(gt_list) > 0:
                        reason = "gt_already_consumed"
                    else:
                        reason = "no_gt_match_strict"

                    print(
                        f"[STAGE5 STRICT SKIP] cid={cid} branch_id={branch_dbg} seg_idx={seg_idx_dbg} "
                        f"reason={reason} key={key}"
                    )
                    log_invalid(
                        image=base_name,
                        cid=cid,
                        level="atomic",
                        reason=reason,
                        length=_polyline_length(pts),
                        n_segments=1,
                        pred_members=pred_members,
                        gt_members=gt_members,
                        overlap=overlap,
                        entity_id=atomic_dbg,
                        branch_id=branch_dbg,
                        extra_info=f"key={key}",
                    )
                    if DEBUG_TOPOLOGY_TRACE:
                        _append_csv_row(
                            stage5_strict_skip_csv,
                            [
                                str(base_name),
                                str(cid),
                                str(branch_dbg),
                                str(seg_idx_dbg),
                                int(L),
                                str(reason),
                                int(pred_total_pts),
                                int(gt_total_pts),
                                int(n_same_branch),
                            ],
                            header=[
                                "image",
                                "cid",
                                "branch_id",
                                "seg_idx",
                                "L_geom",
                                "reason",
                                "pred_len_total",
                                "gt_len_total",
                                "num_gt_candidates_same_branch",
                            ],
                        )
                    continue

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
                        pv = predw[j] if int(j) < len(predw) else np.nan
                        print(f"    j={int(j)} gtw={gtw[j]} predw={pv}")
                if _pred_real_nonfinite > 0:
                    bad = np.where(_pred_nonfinite_mask)[0][:10]
                    print(f"  PRED NONFINITE: count={_pred_real_nonfinite} first_idx={bad.tolist()}")
                    for j in bad:
                        gv = gtw[j] if int(j) < len(gtw) else np.nan
                        padded_flag = bool(_gt_padded_mask[int(j)]) if int(j) < len(_gt_padded_mask) else False
                        print(
                            f"    j={int(j)} predw={predw[j]} "
                            f"gtw={gv} padded_gt={padded_flag}"
                        )

            print(
                f"[STAGE5 TRACE] cid={cid} branch_id={branch_dbg} seg_idx={seg_idx_dbg} "
                f"source=record_strict_match L={L} "
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
                        int(len(predw)),
                        int(len(gtw)),
                        int(_pred_real_nonfinite),
                        int(np.sum(_gt_nonfinite_mask)),
                        int(_gt_padded_nonfinite),
                        int(_gt_real_nonfinite),
                        int(pred_total_pts),
                        int(gt_total_pts),
                        str(gt_match_mode),
                        int(gt_seg_len),
                    ],
                    header=[
                        "cid",
                        "branch_id",
                        "seg_idx",
                        "L_geom",
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

            nd = int(min(len(pts), len(predw), len(gtw)))
            if nd >= 2:
                pts_eval = np.asarray(pts[:nd], float)
                predw_eval = np.asarray(predw[:nd], float)
                gtw_eval = np.asarray(gtw[:nd], float)
                d = predw_eval - gtw_eval
                stage4_pairs.append((pts_eval, d))
            else:
                print(
                    f"[STAGE5 TRACE] cid={cid} seg_idx={seg_idx_dbg} "
                    f"no overlap for stage4_pairs (n={nd}); keeping strict joined record"
                )
            final_pred_segs.append(pts)

            width_pairs.append({
                "image": base_name,
                "cid": str(cid),
                "member_id": (None if atomic_dbg is None else str(atomic_dbg)),
                "crack_type": "combined",
                "midline_type": midline_type,
                "geometry_type": "derived",
                "bbox": crack.get("mask_bbox"),
                "pred_mask_bbox": crack.get("mask_bbox"),
                "pred_mask_crop": crack.get("mask_crop"),
                "pts": np.asarray(pts, float),
                "predw": np.asarray(predw, float),
                "gruthw": np.asarray(gtw, float),
                "gt_source": gt_match_mode,
                "gt_match_seg": np.asarray(gt_match_seg, float) if gt_match_seg is not None else None,
                "branch_id": branch_dbg,
                "seg_idx": seg_idx_dbg,
                "gt_mismatch": False,
                "gt_relation": "combined_vs_combined",
                "match_quality": match_quality,
                "is_weak_alignment": (match_quality == "weak"),
                "is_valid_match": True,
                "overlap_score": float(overlap),
            })

        print(f"[STAGE5] width attachment complete - {len(stage4_pairs)} segments")
        if stage5_unmatched_skips > 0:
            print(f"[STAGE5] skipped unmatched segments: {stage5_unmatched_skips}")

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

            # Use pred_der_dom_segs directly (same Stage-5 source family as GT side).
            for S_seg, m_seg in zip(pred_der_dom_segs or [], pred_der_dom_meta or []):
                if S_seg is None or len(S_seg) < 2:
                    continue
                pred_full_segs.append(np.asarray(S_seg, float))

            # Also collect nonfinite runs from width_pairs for overlay detail.
            for wp in (width_pairs or []):
                if str(wp.get("cid", "")) != str(cid):
                    continue
                if str(wp.get("crack_type", "")) != "combined":
                    continue
                pts_ok = wp.get("pts", None)
                pw_ok = wp.get("predw", None)
                if pts_ok is None or pw_ok is None:
                    continue

                k, uo = _split_pred_nonfinite(pts_ok, pw_ok, min_pts=2)
                pred_undef_other_segs.extend(uo)
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
            fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=100, sharex=True, sharey=True)

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

            legend_items = []
            has_kept = any(S is not None and len(S) >= 2 for S in pred_full_segs)
            has_undef = any(S is not None and len(S) >= 2 for S in pred_undef_other_segs)
            has_bite = any(S is not None and len(S) >= 2 for S in bite_pruned_pred_segs)
            has_gt_only = bool(np.any(dom_crop == 1))
            has_pred_only = bool(np.any(dom_crop == 2))
            has_both = bool(np.any(dom_crop == 3))

            if has_kept:
                legend_items.append(Line2D([0], [0], color=col_keep, lw=2.5, label="Stage-5 geometry (kept)"))
            if has_undef:
                legend_items.append(Line2D([0], [0], color=col_undef, lw=2.2, label="Pred undef / other nonfinite"))
            if has_bite:
                legend_items.append(Line2D([0], [0], color=col_bite, lw=2.0, label="Dominance-bite (union)"))
            if has_gt_only:
                legend_items.append(Line2D([0], [0], color="#e41a1c", lw=6, label="GT-only loss (overlay)"))
            if has_pred_only:
                legend_items.append(Line2D([0], [0], color="#377eb8", lw=6, label="Pred-only loss (overlay)"))
            if has_both:
                legend_items.append(Line2D([0], [0], color="#984ea3", lw=6, label="GT ∩ Pred (overlay)"))

            if legend_items:
                axes[1].legend(
                    handles=legend_items,
                    loc="lower right",
                    fontsize=6,
                    framealpha=0.8,
                    markerscale=0.7,
                    handlelength=1.5,
                    borderpad=0.5,
                )

            fig.suptitle(
                f"Stage-5 Geometry Provenance (Dominance-resolved @ 4.5) — cid={cid}",
                fontsize=11,
                fontweight="bold",
            )

            os.makedirs(cid_opsec_dir, exist_ok=True)
            out = os.path.join(cid_opsec_dir, "stage5_geom_provenance.png")
            _async_savefig(fig, out, dpi=100)

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
    #     resample and plot those too (so you can show effect on GT vs pred" explicitly).
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
            print("[PART2] skipped: width_pairs is empty. Stage-5 produced no derived width pairs.")

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

        def _normalized_arclen_u_part2(xy):
            xy = np.asarray(xy, float)
            if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2:
                return None
            d = np.sqrt(np.sum((xy[1:] - xy[:-1]) ** 2, axis=1))
            if d.size == 0:
                return None
            s = np.concatenate([[0.0], np.cumsum(d)])
            total = float(s[-1]) if len(s) > 0 else 0.0
            if (not np.isfinite(total)) or total <= 1e-12:
                return None
            return (s / total).astype(float, copy=False)

        def _interp_1d_on_u(u_src, y_src, u_common):
            u_src = np.asarray(u_src, float).reshape(-1)
            y_src = np.asarray(y_src, float).reshape(-1)
            u_common = np.asarray(u_common, float).reshape(-1)
            if len(u_src) != len(y_src) or len(u_src) < 2:
                out = np.empty((len(u_common),), float)
                out[:] = np.nan
                return out
            m = np.isfinite(u_src) & np.isfinite(y_src)
            u = u_src[m]
            y = y_src[m]
            if len(u) == 0:
                out = np.empty((len(u_common),), float)
                out[:] = np.nan
                return out
            if len(u) == 1:
                out = np.empty((len(u_common),), float)
                out[:] = float(y[0])
                return out
            u_uni, idx = np.unique(u, return_index=True)
            y_uni = y[idx]
            if len(u_uni) == 1:
                out = np.empty((len(u_common),), float)
                out[:] = float(y_uni[0])
                return out
            return np.interp(u_common, u_uni, y_uni).astype(float, copy=False)

        def _interp_xy_on_u(u_src, pts_src, u_common):
            pts_src = np.asarray(pts_src, float)
            if pts_src.ndim != 2 or pts_src.shape[1] != 2 or len(pts_src) < 2:
                out = np.empty((len(u_common), 2), float)
                out[:] = np.nan
                return out
            x = _interp_1d_on_u(u_src, pts_src[:, 0], u_common)
            y = _interp_1d_on_u(u_src, pts_src[:, 1], u_common)
            return np.column_stack([x, y]).astype(float, copy=False)

        def resample_width_pair_relative(pred_pts, predw, gt_pts_support, gtw, N=None):
            """
            Compare widths at matched relative positions along the segment:
            - predicted geometry defines pair identity (reference support)
            - GT support comes from gt_match_seg when available/valid
            - both widths are interpolated to one shared normalized arclength grid
            """
            pred_pts = np.asarray(pred_pts, float)
            predw = np.asarray(predw, float).reshape(-1)
            gt_pts_support = np.asarray(gt_pts_support, float)
            gtw = np.asarray(gtw, float).reshape(-1)

            if pred_pts.ndim != 2 or pred_pts.shape[1] != 2 or len(pred_pts) < 2:
                return None, None, None
            if gt_pts_support.ndim != 2 or gt_pts_support.shape[1] != 2 or len(gt_pts_support) < 2:
                return None, None, None
            if len(predw) != len(pred_pts) or len(gtw) != len(gt_pts_support):
                return None, None, None

            u_pred = _normalized_arclen_u_part2(pred_pts)
            u_gt = _normalized_arclen_u_part2(gt_pts_support)
            if u_pred is None or u_gt is None:
                return None, None, None

            if N is None:
                N = int(len(predw))
            N = int(N)
            if N < 2:
                return None, None, None

            u_common = np.linspace(0.0, 1.0, num=N)
            predw_rs = _interp_1d_on_u(u_pred, predw, u_common)
            gtw_rs = _interp_1d_on_u(u_gt, gtw, u_common)
            pts_rs = _interp_xy_on_u(u_pred, pred_pts, u_common)
            return pts_rs, predw_rs, gtw_rs

        # ============================================================
        # Part 2: per-width-pair processing
        #   - Build ORIGINAL-domain signals (plot-only)
        #   - Compare widths on one shared relative-arclength domain
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
            entity_part2 = wp.get("member_id", None)
            if entity_part2 is None:
                entity_part2 = cid_s

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

            # Strict support validation:
            # malformed support is skipped and logged instead of repaired.
            if pts is None or predw is None:
                log_invalid(
                    image=image,
                    cid=cid_s,
                    level="atomic",
                    reason="part2_pred_support_mismatch",
                    length=np.nan,
                    n_segments=1,
                    entity_id=entity_part2,
                    branch_id=wp.get("branch_id", None),
                    extra_info=(
                        f"pred_pts_len={None if pts is None else len(pts)} "
                        f"predw_len={None if predw is None else len(predw)} "
                        f"gt_pts_len={None if wp.get('gt_match_seg', None) is None else len(wp.get('gt_match_seg', []))} "
                        f"gtw_len={None if gtruthw is None else len(gtruthw)}"
                    ),
                )
                print(f"[PART2 SKIP] cid={cid_s} reason=part2_pred_support_mismatch (missing pts/predw)")
                continue
            if gtruthw is None:
                log_invalid(
                    image=image,
                    cid=cid_s,
                    level="atomic",
                    reason="part2_gt_support_mismatch",
                    length=float(_polyline_length(np.asarray(pts, float))) if pts is not None else np.nan,
                    n_segments=1,
                    entity_id=entity_part2,
                    branch_id=wp.get("branch_id", None),
                    extra_info=(
                        f"pred_pts_len={len(pts)} predw_len={len(predw)} "
                        f"gt_pts_len={None if wp.get('gt_match_seg', None) is None else len(wp.get('gt_match_seg', []))} "
                        f"gtw_len=None"
                    ),
                )
                print(f"[PART2 SKIP] cid={cid_s} reason=part2_gt_support_mismatch (missing gtruthw)")
                continue

            pred_pts = np.asarray(pts, float)
            predw_raw = np.asarray(predw, float).reshape(-1)
            gtruthw_raw = np.asarray(gtruthw, float).reshape(-1)

            if pred_pts.ndim != 2 or pred_pts.shape[1] != 2 or len(pred_pts) < 2 or len(predw_raw) != len(pred_pts):
                log_invalid(
                    image=image,
                    cid=cid_s,
                    level="atomic",
                    reason="part2_pred_support_mismatch",
                    length=float(_polyline_length(pred_pts)) if pred_pts.ndim == 2 else np.nan,
                    n_segments=1,
                    entity_id=entity_part2,
                    branch_id=wp.get("branch_id", None),
                    extra_info=(
                        f"pred_pts_len={len(pred_pts) if pred_pts.ndim == 2 else None} "
                        f"predw_len={len(predw_raw)} "
                        f"gt_pts_len={None if wp.get('gt_match_seg', None) is None else len(wp.get('gt_match_seg', []))} "
                        f"gtw_len={len(gtruthw_raw)}"
                    ),
                )
                print(
                    f"[PART2 SKIP] cid={cid_s} reason=part2_pred_support_mismatch "
                    f"pred_pts_len={len(pred_pts) if pred_pts.ndim == 2 else None} predw_len={len(predw_raw)}"
                )
                continue

            # Keep local aliases used by downstream cache/debug blocks.
            pts = np.asarray(pred_pts, float)

            gt_match_seg_raw = wp.get("gt_match_seg", None)
            gt_pts_candidate = None
            if gt_match_seg_raw is not None:
                gt_tmp = np.asarray(gt_match_seg_raw, float)
                if gt_tmp.ndim == 2 and gt_tmp.shape[1] == 2:
                    gt_pts_candidate = gt_tmp

            # Part 2 compares widths at matched relative positions along the segment.
            # Predicted geometry defines the reference pair identity.
            # GT support may come from gt_match_seg if available and aligned to gtruthw.
            if gt_pts_candidate is not None and len(gt_pts_candidate) == len(gtruthw_raw):
                gt_pts_support = gt_pts_candidate
            elif len(gtruthw_raw) == len(pred_pts):
                gt_pts_support = pred_pts
            else:
                log_invalid(
                    image=image,
                    cid=cid_s,
                    level="atomic",
                    reason="part2_gt_support_mismatch",
                    length=float(_polyline_length(pred_pts)),
                    n_segments=1,
                    entity_id=entity_part2,
                    branch_id=wp.get("branch_id", None),
                    extra_info=(
                        f"pred_pts_len={len(pred_pts)} predw_len={len(predw_raw)} "
                        f"gt_pts_len={None if gt_pts_candidate is None else len(gt_pts_candidate)} "
                        f"gtw_len={len(gtruthw_raw)}"
                    ),
                )
                print(
                    f"[PART2 SKIP] cid={cid_s} reason=part2_gt_support_mismatch "
                    f"pred_pts_len={len(pred_pts)} gtw_len={len(gtruthw_raw)} "
                    f"gt_pts_len={None if gt_pts_candidate is None else len(gt_pts_candidate)}"
                )
                continue

            s_full = arclen_s(pred_pts)
            if len(s_full) < 2:
                log_invalid(
                    image=image,
                    cid=cid_s,
                    level="atomic",
                    reason="part2_pred_support_mismatch",
                    length=float(_polyline_length(pred_pts)),
                    n_segments=1,
                    entity_id=entity_part2,
                    branch_id=wp.get("branch_id", None),
                    extra_info=f"invalid_pred_arclength pred_pts_len={len(pred_pts)}",
                )
                print(f"[PART2 SKIP] cid={cid_s} reason=part2_pred_support_mismatch (invalid pred arclength)")
                continue

            total_len = float(s_full[-1] - s_full[0])
            if not np.isfinite(total_len) or total_len <= 0:
                log_invalid(
                    image=image,
                    cid=cid_s,
                    level="atomic",
                    reason="part2_pred_support_mismatch",
                    length=np.nan,
                    n_segments=1,
                    entity_id=entity_part2,
                    branch_id=wp.get("branch_id", None),
                    extra_info=f"nonfinite_total_len={total_len}",
                )
                print(f"[PART2 SKIP] cid={cid_s} reason=part2_pred_support_mismatch (non-finite total_len)")
                continue

            # ------------------------------------------------------------
            # ORIGINAL DOMAIN (plot-only)
            # ------------------------------------------------------------
            s_orig = arclen_s(pred_pts)
            predw_orig = np.asarray(predw_raw, float)
            gtruthw_orig = np.asarray(gtruthw_raw, float)

            m0 = min(len(s_orig), len(predw_orig), len(gtruthw_orig))
            if m0 < 2:
                log_invalid(
                    image=image,
                    cid=cid_s,
                    level="atomic",
                    reason="part2_gt_support_mismatch",
                    length=float(_polyline_length(pred_pts)),
                    n_segments=1,
                    entity_id=entity_part2,
                    branch_id=wp.get("branch_id", None),
                    extra_info=(
                        f"too_few_original_samples m0={m0} "
                        f"pred_pts_len={len(pred_pts)} predw_len={len(predw_orig)} gtw_len={len(gtruthw_orig)}"
                    ),
                )
                print(f"[PART2 SKIP] cid={cid_s} reason=part2_gt_support_mismatch (too few original samples)")
                continue

            s_orig       = np.asarray(s_orig[:m0], float)
            predw_orig   = np.asarray(predw_orig[:m0], float)
            gtruthw_orig = np.asarray(gtruthw_orig[:m0], float)
            pts_orig = np.asarray(pred_pts[:m0], float)
            
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
            # RESAMPLE WIDTHS IN A SINGLE RELATIVE-ARCLENGTH DOMAIN
            #   - no padding/clipping repairs
            #   - both streams evaluated at matched normalized positions
            # ------------------------------------------------------------
            pts_rs, predw_rs, gtruthw_rs = resample_width_pair_relative(
                pred_pts,
                predw_raw,
                gt_pts_support,
                gtruthw_raw,
                N=len(predw_raw),
            )
            if (
                pts_rs is None or predw_rs is None or gtruthw_rs is None
                or len(pts_rs) < 2
                or len(predw_rs) != len(pts_rs)
                or len(gtruthw_rs) != len(pts_rs)
            ):
                log_invalid(
                    image=image,
                    cid=cid_s,
                    level="atomic",
                    reason="part2_gt_support_mismatch",
                    length=float(_polyline_length(pred_pts)),
                    n_segments=1,
                    entity_id=entity_part2,
                    branch_id=wp.get("branch_id", None),
                    extra_info=(
                        f"pred_pts_len={len(pred_pts)} predw_len={len(predw_raw)} "
                        f"gt_pts_len={len(gt_pts_support)} gtw_len={len(gtruthw_raw)} "
                        f"rs_pts_len={None if pts_rs is None else len(pts_rs)} "
                        f"rs_predw_len={None if predw_rs is None else len(predw_rs)} "
                        f"rs_gtw_len={None if gtruthw_rs is None else len(gtruthw_rs)}"
                    ),
                )
                print(f"[PART2 SKIP] cid={cid_s} reason=part2_gt_support_mismatch (relative resample failed)")
                continue

            pts_rs = np.asarray(pts_rs, float)
            predw_rs = np.asarray(predw_rs, float)
            gtruthw_rs = np.asarray(gtruthw_rs, float)
            mrs = int(len(pts_rs))

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
                        "pred_mask_full": None,
                        "zoom_bbox": None,
                    },
                )
                bd["pts_list"].append(np.asarray(pts_rs, float))
                bd["predw_list"].append(np.asarray(predw_rs, float))
                bd["gtw_list"].append(np.asarray(gtruthw_rs, float))
                gt_pts_rs = _resample_polyline_to_len_part2(gt_pts_support, mrs)
                bd["gt_pts_list"].append(gt_pts_rs)
                try:
                    pm = _rebuild_pred_mask_from_wp(wp, H, W)
                    if pm is not None:
                        pm = (np.asarray(pm) > 0).astype(np.uint8)
                        if bd["pred_mask_full"] is None:
                            bd["pred_mask_full"] = pm
                        else:
                            bd["pred_mask_full"] = (
                                (np.asarray(bd["pred_mask_full"]) > 0) | (pm > 0)
                            ).astype(np.uint8)
                except Exception:
                    pass

                bb = wp.get("pred_mask_bbox") or wp.get("bbox")
                if isinstance(bb, (list, tuple)) and len(bb) == 4:
                    bx, by, bw, bh = [int(v) for v in bb]
                    if bw > 0 and bh > 0:
                        if bd["zoom_bbox"] is None:
                            bd["zoom_bbox"] = [bx, by, bw, bh]
                        else:
                            ux, uy, uw, uh = bd["zoom_bbox"]
                            x0 = min(ux, bx)
                            y0 = min(uy, by)
                            x1 = max(ux + uw, bx + bw)
                            y1 = max(uy + uh, by + bh)
                            bd["zoom_bbox"] = [x0, y0, x1 - x0, y1 - y0]

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

                mw = compute_length_weighted_metrics(d_run, s_run)
                st0 = length_weighted_err_stats(d_run[:-1], ds_w)
                st = {
                    "bias": mw.get("bias", np.nan),
                    "mae": mw.get("mae", np.nan),
                    "rmse": mw.get("rmse", np.nan),
                    "p95_abs": st0.get("p95_abs", np.nan),
                    "median_abs": st0.get("median_abs", np.nan),
                }
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

                pts_orig_run   = np.asarray(pts_orig[m_orig], float)
                predw_orig_run = np.asarray(predw_orig[m_orig], float)
                gtw_orig_run   = np.asarray(gtruthw_orig[m_orig], float)
                s_orig_run     = np.asarray(s_orig[m_orig], float)

                # Fallback (should rarely trigger, but keeps plots alive)
                if pts_orig_run.shape[0] < 2:
                    pts_orig_run   = np.asarray(pts_orig, float)
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
                    stride=10,
                    gt_pts=bd["gt_pts_list"],
                    gt_mask=(np.asarray(crack_mask) > 0).astype(np.uint8),
                    pred_mask=bd.get("pred_mask_full"),
                    zoom_bbox=bd.get("zoom_bbox"),
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
                _async_savefig(fig, out, dpi=200)
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
                            st = compute_length_weighted_metrics(d_rs, s_rs)
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
                        _async_savefig(fig, out, dpi=200)
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

    if invalid_logs:
        try:
            import pandas as pd
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, "invalid_matches.csv")
            df_new = pd.DataFrame(
                invalid_logs,
                columns=[
                    "image",
                    "cid",
                    "level",
                    "entity_id",
                    "reason",
                    "length",
                    "n_segments",
                    "pred_members",
                    "gt_members",
                    "overlap",
                    "branch_id",
                    "extra_info",
                ],
            )
            if os.path.exists(out_path):
                try:
                    df_old = pd.read_csv(out_path)
                except Exception:
                    df_old = pd.DataFrame(columns=df_new.columns)
                df_out = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df_out = df_new
            df_out.to_csv(out_path, index=False)
            print(f"[WIDTH DEBUG] saved invalid/exclusion rows → {out_path}")
        except Exception as e:
            print(f"[WIDTH DEBUG] failed to save invalid/exclusion rows: {e}")

    # ---------------- plotting ----------------
    if not coords:
        print("[WIDTH DEBUG] nothing to plot")
        _drain_async_savefigs()
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
        _drain_async_savefigs()
        return rows, midline_metric_rows

    '''vmin, vmax = np.percentile(all_d, [5, 95])
    vmin = min(float(vmin), 0.0)
    vmax = max(float(vmax), 0.0)
    if vmax <= vmin:
        vmax = vmin + 1e-6

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("coolwarm")'''
    from matplotlib.colors import TwoSlopeNorm, Normalize

    p5, p95 = np.percentile(all_d, [5, 95])
    vmin = float(p5)
    vmax = float(p95)

    # Expand slightly if degenerate
    if vmax - vmin < 1e-9:
        vmax = vmin + 1e-6

    # Decide normalization type
    if vmin < 0.0 and vmax > 0.0:
        # Proper diverging distribution
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        # One-sided distribution → linear scale
        norm = Normalize(vmin=vmin, vmax=vmax)

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

    _async_savefig(fig, out, dpi=200)
    
    # ============================================================
    # MIDLINE DIAGNOSTIC PLOTS (COMBINED / AUTO)
    # ============================================================
    if midline_metric_rows:
        try:
            import pandas as pd
            from pathlib import Path
            from helpers.present_plots import (
                plot_rs3_midline_diagnostics,
                plot_midline_length_score_relationship,
            )

            df_mid = pd.DataFrame(midline_metric_rows)

            diag_dir = os.path.join(metrics_dir, midline_type or "unknown", "midline_diagnostics", crack_type)
            os.makedirs(diag_dir, exist_ok=True)

            plot_rs3_midline_diagnostics(
                df_all=df_mid,
                out_dir=diag_dir,
                selected_family=None,   # GENERALIZED
            )

            # Per-image length-score diagnostics (continuous + binned + summary CSVs).
            plot_midline_length_score_relationship(
                df_all=df_mid,
                out_dir=diag_dir,
                prefix=f"{base_name}_length_score",
                length_col="length_px",
                score_col="score_mid",
                group_cols=("midline_type", "geometry_type", "variant_id"),
                bins=10,
                max_groups=8,
            )

            # Aggregate all images into metrics/_summary and refresh global plots.
            def _metrics_root_from_any_path(pth):
                try:
                    p = Path(pth).resolve()
                    parts = list(p.parts)
                    idx = None
                    for i, part in enumerate(parts):
                        if str(part).lower() == "metrics":
                            idx = i
                    if idx is None:
                        return str(p.parent)
                    return str(Path(*parts[: idx + 1]))
                except Exception:
                    return os.path.dirname(os.path.abspath(str(pth)))

            def _upsert_by_keys(csv_path, df_new, key_cols):
                if df_new is None or df_new.empty:
                    return pd.DataFrame()
                df_new = df_new.copy()
                if os.path.exists(csv_path):
                    try:
                        old = pd.read_csv(csv_path)
                    except Exception:
                        old = pd.DataFrame()
                else:
                    old = pd.DataFrame()

                if old.empty:
                    merged = df_new
                elif all(c in old.columns for c in key_cols) and all(c in df_new.columns for c in key_cols):
                    merged = old.copy()
                    new_keys = df_new[key_cols].drop_duplicates()
                    for tup in new_keys.itertuples(index=False, name=None):
                        mask = np.ones(len(merged), dtype=bool)
                        for c, v in zip(key_cols, tup):
                            mask &= (merged[c].astype(str) == str(v))
                        merged = merged[~mask]
                    merged = pd.concat([merged, df_new], ignore_index=True)
                else:
                    merged = pd.concat([old, df_new], ignore_index=True)

                merged.to_csv(csv_path, index=False)
                return merged

            metrics_root = _metrics_root_from_any_path(metrics_dir)
            global_dir = os.path.join(metrics_root, "_summary", "compare_midline_length_score")
            os.makedirs(global_dir, exist_ok=True)

            global_csv = os.path.join(global_dir, "compare_midline_length_score_all_images.csv")
            df_global = _upsert_by_keys(
                global_csv,
                df_mid,
                key_cols=("image", "midline_type", "crack_type", "variant_id"),
            )
            if not df_global.empty:
                plot_midline_length_score_relationship(
                    df_all=df_global,
                    out_dir=global_dir,
                    prefix="compare_midline_length_score_all_images",
                    length_col="length_px",
                    score_col="score_mid",
                    group_cols=("midline_type", "geometry_type", "variant_id"),
                    bins=12,
                    max_groups=10,
                )

            print(f"[MIDLINE METRICS] plotted {len(df_mid)} combined diagnostics")

        except Exception as e:
            print(f"[MIDLINE METRICS] plotting failed: {e}")


    _drain_async_savefigs()
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
    # Explicit τ-precision (spurious geometry penalty)
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

    # --- Caller-owned resampling for sampling-dependent ops ---
    A_dense = _resample_by_arclen(A, N=400)
    B_dense = _resample_by_arclen(B, N=400)
    # Reuse dense resampling for Fréchet to avoid redundant resample passes.
    A_ds = A_dense
    B_ds = B_dense

    dAB, dBA, _idxAB, _idxBA = _nn_bidirectional(A, B)
    if dAB.size and dBA.size:
        nn_bi = float(np.mean(dAB) + np.mean(dBA))
        hd_max = float(max(np.max(dAB), np.max(dBA)))
        hd_p95 = float(max(np.percentile(dAB, 95), np.percentile(dBA, 95)))
        cov_a = float(np.mean(dAB <= float(tau)))
        cov_b = float(np.mean(dBA <= float(tau)))
    else:
        nn_bi = np.nan
        hd_max = np.inf
        hd_p95 = np.inf
        cov_a = 0.0
        cov_b = 0.0

    out = {
        "nn_mean_bidirectional": nn_bi,
        "hausdorff_max":    hd_max,
        "frechet_discrete_ds": float("nan"),
        "mean_tan_angle_error_deg": _unwrap(mean_tangent_angle_error_degs(A_dense, B_dense)),
        "relative_length_error":  _unwrap(relative_length_error(A, B)),
    }

    out["coverage_A_to_B"] = cov_a
    out["coverage_B_to_A"] = cov_b
    out["coverage_min"] = float(min(cov_a, cov_b))
    out["hausdorff_p95"] = hd_p95

    # --- Fréchet (optional but standard) ---
    try:
        if len(A_ds) >= 2 and len(B_ds) >= 2:
            out["frechet_discrete_ds"] = _unwrap(
                frechet_discrete_ds(A_ds, B_ds)
            )
    except Exception as e:
        print(f"[metrics][warn] Fréchet failed: {e}")

    # --- Orthogonal deviation stats ---
    orth = orthogonal_deviation(A_dense, B)

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

    def _mask_dbg(tag, cid, cr_obj):
        try:
            bb = (cr_obj or {}).get("mask_bbox", None)
            mc = (cr_obj or {}).get("mask_crop", None)
            shp = None if mc is None else tuple(np.asarray(mc).shape)
            src = (cr_obj or {}).get("source", None)
            print(f"[SNAPSHOT LOAD DEBUG] {tag} cid={cid} source={src}")
            print(f"  bbox: {bb}")
            print(f"  crop shape: {shp}")
        except Exception as e:
            print(f"[SNAPSHOT LOAD DEBUG] {tag} cid={cid} debug failed: {e}")

    for cid, cr in (authoring_atomic or {}).items():
        merged[cid] = dict(cr)
        _mask_dbg("authoring", cid, merged[cid])

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
                    _mask_dbg(f"snapshot_file={os.path.basename(p)}", cid, snap)
                    merged[cid].update(snap)
                    _mask_dbg("merged_after_update", cid, merged[cid])
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
