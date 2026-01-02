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


def chamfer_symmetric(A, B):
    """Mean NN distance both directions."""
    A = _finite_xy(A); B = _finite_xy(B)
    return float(np.mean(_nn_dists(A,B))) + float(np.mean(_nn_dists(B,A)))


def hausdorff_symmetric(A, B):
    """Max directed NN both ways."""
    A = _finite_xy(A); B = _finite_xy(B)
    da = _nn_dists(A, B); db = _nn_dists(B, A)
    if da.size == 0 or db.size == 0:
        return float('inf')
    return float(max(da.max(), db.max()))

# --- DROP-IN REPLACEMENT in py ---


def frechet_discrete(A, B, max_points=800):
    """
    Iterative Eiter–Mannila discrete Fréchet distance.
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


def angle_error_degrees(A, B):
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


def length_ratio(A, B):
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
from helpers.metrics import normals_from_mask_for_midline  # if this file IS helpers.metrics, remove this line

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

def compare_widths_for_cracks(
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
):
    """
    WIDTH COMPARISON — SINGLE-MODE (ATOMIC OR COMBINED)

    Contract:
      - This function MUST be called with exactly one of:
          { "atomic_cracks": {...} }  OR
          { "combined_cracks": {...} }

    Guarantees (retained):
      - Segment-safe (no flattening)
      - Geodesic fallback if normals missing
      - Zoom uses ONLY union of provided mask_bbox values
      - Solid blue bbox overlay
      - TwoSlopeNorm always monotonic (0 included)

    NEW (combined opsec):
      - Stage 0: match combined crack to GT supervision by member overlap (fallback to best-overlap)
      - Stage 1: prune segments to shared atomic IDs (when segment meta exists)
      - Stage 2: optional branch matching using branch membership (when both sides provide metadata)
      - Stage 3: symmetric bite-union clipping (when bite masks exist)
      - GT widths for combined are computed along the FINAL clipped polyline (mask-based),
        so segment ordering differences cannot corrupt alignment.
    """

    import os, json, base64
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from helpers.metrics import normals_from_mask_for_midline

    os.makedirs(metrics_dir, exist_ok=True)

    H, W = crack_mask.shape
    mask_bin = (crack_mask > 0).astype(np.uint8)

    atomic   = ann.get("atomic_cracks")
    combined = ann.get("combined_cracks")

    # ---------------- enforce SINGLE MODE ----------------
    if (atomic is None) == (combined is None):
        raise RuntimeError(
            "compare_widths_for_cracks expects EXACTLY ONE of "
            "'atomic_cracks' or 'combined_cracks'"
        )

    mode = "atomic" if atomic is not None else "combined"
    cracks = atomic if mode == "atomic" else combined

    print(f"\n[WIDTH DEBUG] === RUN MODE: {mode.upper()} ===")

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

    def _extract_segments_and_meta(crack):
        """
        Returns:
          segs: list[np.ndarray (Ni,2)]
          seg_meta: list[dict] same length as segs (best effort)
          bite_obj: dict or None
          members_set: set(str)
        """
        if mode == "atomic":
            segs = _split_on_nans(crack.get("midline", []))
            seg_meta = [{"branch_id": 0, "atomic_id": str(crack.get("id", ""))} for _ in segs]
            return segs, seg_meta, None, {str(crack.get("id", ""))}
        else:
            print(f"\n------------{crack.keys()}-----------\n")
            segs = [np.asarray(s, float) for s in (crack.get("midline_segments", []) or [])]
            seg_meta = crack.get("midline_segments_meta") or crack.get("segments_meta") or []
            if not isinstance(seg_meta, list):
                seg_meta = []
            if len(seg_meta) != len(segs):
                # best effort padding
                tmp = []
                for i in range(len(segs)):
                    d = seg_meta[i] if i < len(seg_meta) and isinstance(seg_meta[i], dict) else {}
                    tmp.append(d)
                seg_meta = tmp
                for i in range(len(seg_meta)):
                    if "branch_id" not in seg_meta[i]:
                        seg_meta[i]["branch_id"] = int(seg_meta[i].get("branch_id", i))
            bite_obj = None
            dom = crack.get("dominance_meta") or crack.get("dominance") or crack.get("dominance_info") or {}
            if isinstance(dom, dict) and "bite" in dom and isinstance(dom["bite"], dict):
                bite_obj = dom["bite"]
            else:
                # some pipelines store bite directly
                b = crack.get("bite")
                if isinstance(b, dict) and "bbox" in b:
                    bite_obj = b
            members = crack.get("members") or []
            members_set = set(map(str, members))
            return segs, seg_meta, bite_obj, members_set

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

    # ---------------- load GT supervision ----------------
    gt_sup = {}
    if gt_sup_root:
        p = os.path.join(gt_sup_root, "gt_supervision.json")
        if os.path.exists(p):
            with open(p, "r") as f:
                data = json.load(f)
            for e in data.get("cracks", []):
                if e.get("kind") == mode:
                    key = str(e["id"]) if mode == "atomic" else frozenset(map(str, e.get("members", [])))
                    gt_sup[key] = e

    # ---------------- accumulators ----------------
    coords, diffs, bboxes = [], [], []
    rows = []
    midline_metric_rows = []   # NEW: for combined midline diagnostics

    # debug dir for opsec artifacts
    opsec_dir = os.path.join(metrics_dir, midline_type or "unknown", "opsec_debug")
    os.makedirs(opsec_dir, exist_ok=True)

    # ---------------- iterate cracks ----------------
    for cid, crack in cracks.items():
        print(f"\n[WIDTH DEBUG] {mode.upper()} cid={cid}")

        segs, seg_meta, bite_pred, pred_members = _extract_segments_and_meta(crack)
        if not segs:
            continue

        e1, e2 = _get_edges(crack)
        if len(e1) < 2 or len(e2) < 2:
            continue
        m_edge = min(len(e1), len(e2))
        widths_geo = np.linalg.norm(e1[:m_edge] - e2[:m_edge], axis=1)

        # --------------------------------------------
        # ATOMIC MODE (keep prior logic, minimal)
        # --------------------------------------------
        if mode == "atomic":
            gt_widths = []
            if gt_sup_root and str(cid) in gt_sup:
                try:
                    gt_widths = [np.asarray(gt_sup[str(cid)]["gt_normals"]["width_px"], float)]
                except Exception:
                    gt_widths = []

            if not gt_widths:
                for s in segs:
                    (_, _, _, _, w), _ = normals_from_mask_for_midline(s, mask_bin, max_radius)
                    gt_widths.append(np.asarray(w, float))

            off = 0
            for s, gtw in zip(segs, gt_widths):
                m = min(len(s), len(gtw), len(widths_geo) - off)
                if m < 2:
                    off += max(m, 0)
                    continue
                d = widths_geo[off:off + m] - gtw[:m]
                pts = s[:m]
                diffs.append(d)
                coords.append(pts)
                bboxes.append(crack.get("mask_bbox"))

                for (x, y), dw, gw, pw in zip(pts, d, gtw[:m], widths_geo[off:off + m]):
                    if not np.isfinite(dw):
                        continue
                    rows.append({
                        "x": float(x), "y": float(y),
                        "gt_width_px": float(gw),
                        "pred_width_px": float(pw),
                        "width_diff_px": float(dw),
                        "cid": str(cid),
                        "crack_type": mode,
                        "midline_type": midline_type,
                    })
                off += m
            continue

        # --------------------------------------------
        # COMBINED MODE OPSEC (HEAVY DEBUG)
        # --------------------------------------------

        # Stage 0: find best GT entry by overlap
        pred_key = frozenset(map(str, crack.get("members", []) or []))
        gt_entry = gt_sup.get(pred_key)

        if gt_entry is None and gt_sup:
            pm = set(map(str, crack.get("members", []) or []))
            best = None
            for k, e in gt_sup.items():
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
        
        # ============================================================
        # OPSEC PLOT — STAGE 1 ATOMIC PRUNING (GT vs PRED)
        #   - LEFT : GT mask (gt_full -> mask_bin) + GT final geometry
        #   - RIGHT: PRED mask (from crack["mask_crop"] + crack["mask_bbox"])
        #           + kept/pruned pred segments (Stage 1 rule)
        # ============================================================
        try:
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
            import numpy as np

            # ---- reconstruct PRED mask from mask_crop + mask_bbox ----
            pred_mask_full = np.zeros((H, W), np.uint8)
            bb = crack.get("mask_bbox")
            crop_list = crack.get("mask_crop")

            if bb and crop_list is not None:
                x, y, w, h = map(int, bb)
                crop_u8 = np.asarray(crop_list, dtype=np.uint8)

                # safety: handle mismatched shapes gracefully
                hh = min(h, crop_u8.shape[0]) if crop_u8.ndim >= 2 else 0
                ww = min(w, crop_u8.shape[1]) if crop_u8.ndim >= 2 else 0
                if hh > 0 and ww > 0:
                    pred_mask_full[y:y+hh, x:x+ww] = (crop_u8[:hh, :ww] > 0).astype(np.uint8)

            # ---- bbox crop window ----
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

            # ---- classify PRED segments (Stage 1 rule) ----
            pred_kept = []
            pred_pruned = []
            for S, m in zip(segs, seg_meta):
                if S is None or len(S) < 2:
                    continue
                aid = m.get("atomic_id", None)
                if aid is not None and str(aid) not in shared:
                    pred_pruned.append(np.asarray(S, float))
                else:
                    pred_kept.append(np.asarray(S, float))

            # ---- GT segments (final geometry) ----
            gt_kept = []
            gt_pruned = []  # will remain empty unless GT provides per-seg atomic IDs
            if gt_entry is not None:
                gt_segs_all = gt_entry.get("midline_segments") or []
                for Sg in gt_segs_all:
                    if Sg is None or len(Sg) < 2:
                        continue
                    gt_kept.append(np.asarray(Sg, float))

            # ---- figure ----
            fig, axes = plt.subplots(
                1, 2, figsize=(10, 5), dpi=200, sharex=True, sharey=True
            )

            for ax, title in zip(
                axes,
                ["GT supervision (final geometry)", "Prediction (Stage 1 pruning)"],
            ):
                ax.set_title(title, fontsize=10)
                ax.axis("off")

            # backgrounds
            axes[0].imshow(mask_bin[y0:y1, x0:x1], cmap="gray", zorder=0)
            axes[1].imshow(pred_mask_full[y0:y1, x0:x1], cmap="gray", zorder=0)

            # ---- colors ----
            col_keep = (0.2, 0.4, 0.8)   # muted blue
            col_drop = (0.5, 0.0, 0.0)   # dark red

            # ---- GT plot ----
            for S in gt_pruned:
                S2 = S - np.array([x0, y0])
                axes[0].plot(S2[:, 0], S2[:, 1], color=col_drop, lw=2.0, alpha=0.8)

            for S in gt_kept:
                S2 = S - np.array([x0, y0])
                axes[0].plot(S2[:, 0], S2[:, 1], color=col_keep, lw=2.5)

            # ---- Pred plot ----
            for S in pred_pruned:
                S2 = S - np.array([x0, y0])
                axes[1].plot(S2[:, 0], S2[:, 1], color=col_drop, lw=2.0, alpha=0.8)

            for S in pred_kept:
                S2 = S - np.array([x0, y0])
                axes[1].plot(S2[:, 0], S2[:, 1], color=col_keep, lw=2.5)

            # ---- bbox overlay (same bbox coords for both panels) ----
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

            # ---- legend ----
            legend_items = [
                Line2D([0], [0], color=col_keep, lw=3, label="Kept segments"),
                Line2D([0], [0], color=col_drop, lw=3, label="Pruned segments"),
                Line2D([0], [0], color="dodgerblue", lw=1.5, label="BBox"),
            ]
            axes[1].legend(handles=legend_items, loc="lower right", fontsize=8, framealpha=0.9)

            fig.suptitle(
                f"Stage 1 Atomic Pruning — cid={cid}",
                fontsize=11,
                fontweight="bold",
            )

            out = os.path.join(opsec_dir, f"stage1_prune_{cid}.png")
            fig.savefig(out, bbox_inches="tight", dpi=200)
            plt.close(fig)

        except Exception as e:
            print(f"[OPSEC STAGE1 PLOT] skipped cid={cid}: {e}")

        # --------------------------------------------
        # Stage 2: optional branch matching
        # --------------------------------------------
        matched_pred_branch_ids = None
        if gt_entry is not None:
            gt_mid_segs = gt_entry.get("midline_segments") or []
            gt_meta = gt_entry.get("midline_segments_meta") or []
            if len(gt_mid_segs) == len(gt_meta) and gt_mid_segs:
                gt_segs = [np.asarray(s, float) for s in gt_mid_segs]
                gt_br = _build_branch_table(gt_segs, gt_meta, shared_members=shared)
                pr_br = _build_branch_table(pruned_segs, pruned_meta, shared_members=shared)
                if gt_br and pr_br:
                    matches = _greedy_match_branches(gt_br, pr_br)
                    if matches:
                        matched_pred_branch_ids = {p for (_, p, _, _) in matches}

        if matched_pred_branch_ids is not None:
            keep_s, keep_m = [], []
            for S, m in zip(pruned_segs, pruned_meta):
                bid = int(m.get("branch_id", -1))
                if bid in matched_pred_branch_ids:
                    keep_s.append(S)
                    keep_m.append(m)
                else:
                    print(f"[WIDTH DEBUG] DROP branch={bid} (unmatched)")
            pruned_segs, pruned_meta = keep_s, keep_m

        if not pruned_segs:
            print(f"[WIDTH DEBUG] cid={cid} -> NO SEGMENTS AFTER BRANCH MATCH")
            continue

        # --------------------------------------------
        # Stage 3: build ORIGINAL segment offsets
        # --------------------------------------------
        orig_segs = [np.asarray(s, float) for s in crack.get("midline_segments", [])]
        seg_start = {}
        off = 0
        for i, S in enumerate(orig_segs):
            seg_start[i] = off
            off += len(S)

        print(f"[WIDTH DEBUG] cid={cid} widths_geo={len(widths_geo)}")

        have_valid_seg_idx = any(
            isinstance(m.get("seg_idx"), int) and m["seg_idx"] in seg_start
            for m in pruned_meta
        )

        off_fallback = 0

        # --------------------------------------------
        # Stage 4: width slicing + SANITY PLOTS
        # --------------------------------------------
        for si, (S, m) in enumerate(zip(pruned_segs, pruned_meta)):
            if len(S) < 2:
                continue

            if have_valid_seg_idx and isinstance(m.get("seg_idx"), int) and m["seg_idx"] in seg_start:
                s0 = seg_start[m["seg_idx"]]
                src = "seg_idx"
            else:
                s0 = off_fallback
                src = "fallback"

            L = len(S)
            s1 = min(s0 + L, len(widths_geo))

            pw_full = widths_geo[s0:s1]
            pts_full = S[:len(pw_full)]

            print(
                f"[WIDTH DEBUG] cid={cid} seg#{si} "
                f"branch={m.get('branch_id')} src={src} "
                f"geom_pts={len(S)} pw_pts={len(pw_full)} "
                f"s0={s0} s1={s1}"
            )

            off_fallback += L

            if len(pts_full) < 2:
                continue

            # ---------------- SANITY PLOT (PER SEGMENT)
            dbg_seg = os.path.join(opsec_dir, f"combined_opsec_{cid}_seg{si}.png")
            if not os.path.exists(dbg_seg):
                bb = crack.get("mask_bbox")
                if bb:
                    x, y, w, h = map(int, bb)
                    x0 = max(0, x - 20); y0 = max(0, y - 20)
                    x1 = min(W, x + w + 20); y1 = min(H, y + h + 20)
                else:
                    x0, y0, x1, y1 = 0, 0, W, H

                fig, ax = plt.subplots(figsize=(6,6), dpi=200)
                ax.imshow(mask_bin[y0:y1, x0:x1], cmap="gray")

                pr = pts_full - np.array([x0, y0])
                ax.plot(pr[:,0], pr[:,1], color="cyan", lw=1.5, label="width coverage")

                ax.set_title(
                    f"cid={cid} seg#{si}\n"
                    f"geom_len={_linestring_length(S):.1f}px  "
                    f"pw_pts={len(pw_full)}"
                )
                ax.legend()
                ax.axis("off")
                fig.savefig(dbg_seg, bbox_inches="tight", dpi=200)
                plt.close(fig)

            # ---------------- GT WIDTHS
            (_, _, _, _, gw), _ = normals_from_mask_for_midline(pts_full, mask_bin, max_radius)
            gw = np.asarray(gw, float)

            mlen = min(len(pts_full), len(pw_full), len(gw))
            if mlen < 2:
                continue

            pts_ok = pts_full[:mlen]
            pw_ok = pw_full[:mlen]
            gw_ok = gw[:mlen]
            d = pw_ok - gw_ok
            
            # ============================================================
            # MIDLINE METRICS (COMBINED + AUTO ONLY)
            # ============================================================
            if mode == "combined" and midline_type == "auto" and gt_entry is not None:
                try:
                    from helpers.metrics import compute_midline_metrics
                    import math

                    # ---- build GT midline (same pruning rules as pred) ----
                    gt_segs_all = gt_entry.get("midline_segments") or []
                    gt_meta_all = gt_entry.get("midline_segments_meta") or []

                    gt_keep = []
                    for Sg, mg in zip(gt_segs_all, gt_meta_all):
                        if Sg is None or len(Sg) < 2:
                            continue
                        aid = mg.get("atomic_id")
                        if aid is not None and str(aid) not in shared:
                            continue
                        bid = int(mg.get("branch_id", -1))
                        if matched_pred_branch_ids is not None and bid not in matched_pred_branch_ids:
                            continue
                        gt_keep.append(np.asarray(Sg, float))

                    if gt_keep:
                        gt_mid = np.vstack(gt_keep)

                        # ---- compute midline metrics ----
                        mm = compute_midline_metrics(pts_ok, gt_mid)

                        ch  = float(mm.get("chamfer_mean", np.inf))
                        hd  = float(mm.get("hausdorff", np.inf))
                        cov = float(mm.get("coverage", 0.0))

                        score_mid = (
                            math.log1p(max(ch, 0.0)) +
                            0.5 * math.log1p(max(hd, 0.0)) +
                            (1.0 - float(np.clip(cov, 0.0, 1.0)))
                        )

                        midline_metric_rows.append({
                            "image": base_name,
                            "crack_id": str(cid),
                            "variant_global_id": -1,   # sentinel (not RS3)
                            "os_mode": "combined",
                            "g11": np.nan,
                            "g22": np.nan,
                            "g33": np.nan,

                            "length_px": _linestring_length(gt_mid),
                            "bbox_area": float(bbox0[2] * bbox0[3]) if bbox0 else np.nan,

                            # --- selection metrics ---
                            "chamfer_mean": ch,
                            "hausdorff": hd,
                            "coverage": cov,
                            "score_mid": score_mid,

                            # --- diagnostics ---
                            "frechet_discrete": mm.get("frechet_discrete"),
                            "angle_err_deg": mm.get("angle_err_deg"),
                            "length_ratio": mm.get("length_ratio"),
                            "orth_mean": mm.get("orth_mean"),
                            "orth_std": mm.get("orth_std"),
                            "directional_bias": mm.get("directional_bias"),
                            "curvature_rms_auto": mm.get("curvature_rms_auto"),
                            "curvature_rms_manual": mm.get("curvature_rms_manual"),
                            "curvature_rms_ratio": mm.get("curvature_rms_ratio"),
                        })

                except Exception as e:
                    print(f"[MIDLINE METRICS] skipped cid={cid} seg#{si}: {e}")


            # ---- bbox bookkeeping (FIX) ----
            bbox0 = crack.get("mask_bbox")
            if bbox0 is not None:
                bboxes.append(bbox0)

            diffs.append(d)
            coords.append(pts_ok)

            for (x, y), dw, gwi, pwi in zip(pts_ok, d, gw_ok, pw_ok):
                if not np.isfinite(dw):
                    continue
                rows.append({
                    "x": float(x),
                    "y": float(y),
                    "gt_width_px": float(gwi),
                    "pred_width_px": float(pwi),
                    "width_diff_px": float(dw),
                    "cid": str(cid),
                    "crack_type": "combined",
                    "midline_type": midline_type,
                })



    # ---------------- plotting ----------------
    if not coords:
        print("[WIDTH DEBUG] nothing to plot")
        return [], []

    bbox = _union_bboxes(bboxes)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    bg = np.stack([(crack_mask > 0) * 255] * 3, axis=-1)
    ax.imshow(bg)

    all_d = np.concatenate([d for d in diffs if d is not None and len(d) > 0])
    all_d = all_d[np.isfinite(all_d)]
    if all_d.size == 0:
        print("[WIDTH DEBUG] no finite diffs")
        return rows, None

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
    cb.set_label("Estimated width − GT width (px)", fontsize=10, fontweight="bold")

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

    out_dir = os.path.join(metrics_dir, midline_type or "unknown")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{midline_type}_{crack_type}_width_diffs.png")
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

            diag_dir = os.path.join(metrics_dir, midline_type or "unknown", "midline_diagnostics")
            os.makedirs(diag_dir, exist_ok=True)

            plot_rs3_midline_diagnostics(
                df_all=df_mid,
                out_dir=diag_dir,
                selected_family=None,   # GENERALIZED
            )

            print(f"[MIDLINE METRICS] plotted {len(df_mid)} combined diagnostics")

        except Exception as e:
            print(f"[MIDLINE METRICS] plotting failed: {e}")


    return rows, None

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
    diff_col = next(
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
            out_dir,
            f"{base_name}_width_diff_hist_{crack_type}.png"
        )

        plot_width_diff_histogram(
            diffs_csv,
            hist_png,
            title=f"{midline_type} {base_name} width diffs",
            bins=30,
            vlim=None,
        )
    except Exception as e:
        print(f"[WIDTH HIST] failed: {e}")

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
    
def compute_midline_metrics_for_image(app):
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
    print(f"[DEBUG MIDLINE] wrote {len(df)} rows → {out_csv}")

# ---------- NEW helpers ----------

def compute_midline_metrics(auto_xy, man_xy, tau=3.0):
    """
    Core and diagnostic midline metrics (robust to dict returns).

    Outputs:
    chamfer_mean, hausdorff, frechet_discrete, angle_err_deg,
    length_ratio, coverage, orth_mean, orth_std, directional_bias,
    curvature_rms_[auto|manual|ratio]
    """
    import numpy as np

    def _unwrap(v):
        """extract numeric value if helper returns dict"""
        if isinstance(v, dict):
            for k in ("value", "mean", "coverage", "score", "dist"):
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
                ("chamfer_mean","hausdorff","frechet_discrete",
                "angle_err_deg","length_ratio","coverage",
                "orth_mean","orth_std","directional_bias",
                "curvature_rms_auto","curvature_rms_manual","curvature_rms_ratio")}

    # --- Light resampling for expensive ops ---
    A_ds = _resample_by_arclen(A, N=min(600, len(A)))
    B_ds = _resample_by_arclen(B, N=min(600, len(B)))

    out = {
        "chamfer_mean": _unwrap(chamfer_symmetric(A, B)),
        "hausdorff":    _unwrap(hausdorff_symmetric(A, B)),
        "frechet_discrete": float("nan"),
        "angle_err_deg": _unwrap(angle_error_degrees(A, B)),
        "length_ratio":  _unwrap(length_ratio(A, B)),
        "coverage":      _unwrap(coverage_at_tau(A, B, tau_px=tau)),
    }

    # --- Fréchet (optional but standard) ---
    try:
        if len(A_ds) >= 2 and len(B_ds) >= 2:
            out["frechet_discrete"] = _unwrap(
                frechet_discrete(A_ds, B_ds, max_points=800)
            )
    except Exception as e:
        print(f"[metrics][warn] Fréchet failed: {e}")

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
        out["directional_bias"] = float(np.sign(mu) * abs(mu) / sd)
    else:
        out["orth_mean"] = out["orth_std"] = out["directional_bias"] = np.nan

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
# If you keep compare_widths_for_cracks here, make sure this is visible:
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