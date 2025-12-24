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
    gt = gt_mask.astype(bool); pr = pred_mask.astype(bool)
    tp = np.logical_and(gt, pr).sum()
    fp = np.logical_and(~gt, pr).sum()
    fn = np.logical_and(gt, ~pr).sum()
    tn = np.logical_and(~gt, ~pr).sum()
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    iou       = tp / (tp + fp + fn + 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}

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

'''def compare_widths_for_cracks(
    ann,
    crack_mask,
    base_name,
    metrics_dir,
    display=False,
    tag=None,
    return_normals=False,
    normals_plot=False,
    normals_dir=None,
    max_radius=50
):
    """
    Width-difference evaluator with:
      - GT normals from mask
      - Geodesic normals (manual)
      - Smooth width-colored midline plots (global + zoom)
      - Per-sample diffs CSV and summary CSV with mae / rmse / bias / corr
    """
    import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    os.makedirs(metrics_dir, exist_ok=True)
    if normals_plot and normals_dir:
        os.makedirs(normals_dir, exist_ok=True)

    # --- cache of GT normals to avoid recomputation across cracks
    global _NORMALS_CACHE
    try:
        _NORMALS_CACHE
    except NameError:
        _NORMALS_CACHE = {}

    from helpers.metrics import normals_from_mask_for_midline

    H, W = crack_mask.shape
    mask_bin = (crack_mask > 0).astype(np.uint8)

    atomic   = ann.get("atomic_cracks", {}) or {}
    combined = ann.get("combined_cracks", {}) or {}

    atomics_in_combined = {
        str(m)
        for cmb in combined.values()
        for m in (cmb.get("members", []) or [])
    }

    agg = {"atomic": {"coords": [], "diff": []},
           "combined": {"coords": [], "diff": []}}

    width_rows, diffs_rows = [], []
    normals_dict = {} if return_normals else None

    def _to_xy(v):
        import numpy as np
        if isinstance(v, list) and len(v) == 2 and isinstance(v[0], (list, tuple, np.ndarray)):
            return np.column_stack([v[0], v[1]]).astype(float)
        return np.asarray(v, float)

    # ---------------------------------------------------------------
    # Collect width curves for all cracks
    # ---------------------------------------------------------------
    all_cracks = [
        ("atomic", str(cid), c)
        for cid, c in atomic.items()
        if str(cid) not in atomics_in_combined
    ]
    all_cracks += [
        ("combined", str(cid), c)
        for cid, c in combined.items()
    ]

    for ctype, cid, crack in all_cracks:

        # ------------------------------
        # MIDLINE
        # ------------------------------
        midline = np.asarray(crack.get("midline", []), float)
        if midline.ndim != 2 or midline.shape[1] != 2 or len(midline) < 3:
            continue

        # ------------------------------
        # NORMALS FROM GT MASK (cached)
        # ------------------------------
        key = (id(mask_bin), midline.shape[0], float(midline[0, 0]), float(midline[0, 1]))

        if key in _NORMALS_CACHE:
            e1x, e1y, e2x, e2y, w_mask = _NORMALS_CACHE[key]
        else:
            (e1x, e1y, e2x, e2y, w_mask), _ = normals_from_mask_for_midline(
                midline, mask_bin, max_radius=max_radius
            )
            _NORMALS_CACHE[key] = (e1x, e1y, e2x, e2y, w_mask)

        if return_normals:
            normals_dict[f"{ctype}:{cid}"] = {
                "edge1": np.column_stack([e1x, e1y]),
                "edge2": np.column_stack([e2x, e2y]),
                "widths": np.asarray(w_mask, float),
                "midline": midline
            }

        # ------------------------------
        # GEODESIC NORMALS (manual)
        # ------------------------------
        geo = crack.get("normal_edge_points_full") or crack.get("normal_edge_points")
        if not geo:
            continue

        if isinstance(geo, dict):
            ge1 = _to_xy(geo.get("edge1", []))
            ge2 = _to_xy(geo.get("edge2", []))
        elif isinstance(geo, (list, tuple)) and len(geo) == 2:
            ge1, ge2 = _to_xy(geo[0]), _to_xy(geo[1])
        else:
            continue

        n = min(len(ge1), len(ge2))
        if n < 2:
            continue

        ge_width = np.linalg.norm(ge1[:n] - ge2[:n], axis=1)
        m = min(n, len(w_mask))
        if m < 2:
            continue

        # ------------------------------
        # WIDTHS
        # ------------------------------
        gtw = np.asarray(w_mask[:m], float)   # GT width from mask normals
        gew = ge_width[:m]                    # Geodesic width

        diffs = gew - gtw                     # auto/manual – gt

        # ------------------------------
        # STORE FOR PLOTTING ACCUMULATOR
        # ------------------------------
        agg[ctype]["coords"].append(midline[:m])
        agg[ctype]["diff"].append(diffs)

        # ------------------------------
        # CORRELATION (per crack, optional)
        # ------------------------------
        if np.std(gew) > 1e-6 and np.std(gtw) > 1e-6:
            crack_corr = float(np.corrcoef(gew, gtw)[0, 1])
        else:
            crack_corr = np.nan

        # ------------------------------
        # PER-CRACK SUMMARY ROW
        # ------------------------------
        width_rows.append({
            "image": base_name,
            "crack_type": ctype,
            "crack_id": cid,
            "n_samples": int(m),
            "diff_mean": float(np.nanmean(diffs)),
            "diff_std":  float(np.nanstd(diffs)),
            "mae_px":    float(np.nanmean(np.abs(diffs))),
            "rmse_px":   float(np.sqrt(np.nanmean(diffs**2))),
            "bias_px":   float(np.nanmean(diffs)),
            "corr":      crack_corr,
        })

        # ------------------------------
        # PER-SAMPLE DIFF ROWS
        # ------------------------------
        for k, dv in enumerate(diffs):
            diffs_rows.append({
                "image": base_name,
                "crack_type": ctype,
                "crack_id": cid,
                "sample_idx": int(k),
                "width_diff_px": float(dv),
                "gt_width_px": float(gtw[k]),
                "geo_width_px": float(gew[k]),
            })

    # ---------------------------------------------------------------
    # SMOOTH MIDLINE PLOTS — robust, no crashes
    # ---------------------------------------------------------------
    from scipy.ndimage import gaussian_filter1d  # kept if you later want smoothing

    for ctype in ("atomic", "combined"):

        raw_coords = agg[ctype]["coords"]
        raw_diffs  = agg[ctype]["diff"]
        if not raw_coords:
            continue

        # -------- sanitize bad entries ----------
        coords_list, diffs_list = [], []
        for c, d in zip(raw_coords, raw_diffs):
            c = np.asarray(c, float)
            d = np.asarray(d, float)
            if c.ndim != 2 or c.shape[1] != 2:
                continue
            if len(c) < 2 or len(d) < 2:
                continue
            m = min(len(c), len(d))
            if m < 2:
                continue
            coords_list.append(c[:m])
            diffs_list.append(d[:m])

        if not coords_list:
            continue

        all_diffs = np.concatenate(diffs_list, axis=0)
        vmin = float(np.nanpercentile(all_diffs, 5))
        vcenter = 0.0
        vmax = float(np.nanpercentile(all_diffs, 95))
        if vmax <= vmin:
            vmax = vmin + 1.0

        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        cmap = plt.get_cmap("coolwarm")

        # -------- plot function ----------            
        def _plot_on_gt_background(coords_list, diffs_list, out_path, zoom=False):
            import numpy as np
            import matplotlib.pyplot as plt
            import matplotlib as mpl
            from matplotlib.colors import TwoSlopeNorm

            bg = (crack_mask > 0).astype(np.uint8) * 255
            bg = np.stack([bg, bg, bg], axis=-1)

            H, W = bg.shape[:2]

            coords = []
            diffs  = []
            for c, d in zip(coords_list, diffs_list):
                c = np.asarray(c, float)
                d = np.asarray(d, float)
                m = min(len(c), len(d))
                if m > 1:
                    coords.append(c[:m])
                    diffs.append(d[:m])

            if not coords:
                print("[DEBUG WIDTH] nothing to plot")
                return

            coords = np.concatenate(coords, axis=0)
            diffs  = np.concatenate(diffs,  axis=0)

            # Deduplicate
            _, uniq = np.unique(coords, axis=0, return_index=True)
            order_u = np.sort(uniq)
            coords = coords[order_u]
            diffs  = diffs[order_u]

            if len(coords) < 2:
                print("[DEBUG WIDTH] degenerate coords")
                return

            # Arc-length ordering
            d = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
            s = np.concatenate([[0], np.cumsum(d)])
            order = np.argsort(s)
            coords = coords[order]
            diffs  = diffs[order]

            # Color normalization around 0
            norm = TwoSlopeNorm(vmin=-8, vcenter=0, vmax=8)
            cmap = plt.get_cmap("coolwarm")

            fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
            ax.imshow(bg, interpolation="nearest")

            for i in range(len(coords) - 1):
                x0, y0 = coords[i]
                x1, y1 = coords[i+1]
                ax.plot(
                    [x0, x1], [y0, y1],
                    color=cmap(norm(diffs[i])),
                    linewidth=2.5,
                    solid_capstyle="round"
                )

            if zoom:
                x0, x1 = np.nanpercentile(coords[:,0], [1, 99])
                y0, y1 = np.nanpercentile(coords[:,1], [1, 99])
                pad = 0.05 * max(x1 - x0, y1 - y0)
                ax.set_xlim(x0 - pad, x1 + pad)
                ax.set_ylim(y1 + pad, y0 - pad)
            else:
                ax.set_xlim(0, W)
                ax.set_ylim(H, 0)

            ax.axis("off")
            ax.set_aspect("equal")

            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cb = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
            cb.set_label("geodesic width – GT width (px)")

            plt.tight_layout()
            fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

        _plot_on_gt_background(
            coords_list,
            diffs_list,
            os.path.join(metrics_dir, f"{base_name}_{ctype}_all{('_'+tag) if tag else ''}_width_diffs_global.png"),
            zoom=False
        )

        _plot_on_gt_background(
            coords_list,
            diffs_list,
            os.path.join(metrics_dir, f"{base_name}_{ctype}_all{('_'+tag) if tag else ''}_width_diffs_zoom.png"),
            zoom=True
        )

    # ---------------------------------------------------------------
    # WRITE CSVs
    # ---------------------------------------------------------------
    df_w = pd.DataFrame(width_rows)
    df_d = pd.DataFrame(diffs_rows)

    # per-crack stats (keep original info, but use a different filename)
    if not df_w.empty:
        df_w.to_csv(
            os.path.join(
                metrics_dir,
                f"{base_name}_width_percrack{('_'+str(tag)) if tag else ''}.csv"
            ),
            index=False
        )

    # per-sample diffs file used for summary + any later analysis
    if not df_d.empty and tag is not None:
        diffs_path = os.path.join(metrics_dir, f"{base_name}_width_diffs_{tag}.csv")
        df_d.to_csv(diffs_path, index=False)

        # --- SUMMARY STATS FOR TRIPLET PLOT ---
        diffs = df_d["width_diff_px"].to_numpy(dtype=float)
        gt    = df_d["gt_width_px"].to_numpy(dtype=float)
        geo   = df_d["geo_width_px"].to_numpy(dtype=float)

        mae  = float(np.nanmean(np.abs(diffs)))
        rmse = float(np.sqrt(np.nanmean(diffs**2)))
        bias = float(np.nanmean(diffs))

        valid = np.isfinite(gt) & np.isfinite(geo)
        if valid.sum() > 1 and np.std(gt[valid]) > 1e-6 and np.std(geo[valid]) > 1e-6:
            corr = float(np.corrcoef(gt[valid], geo[valid])[0, 1])
        else:
            corr = np.nan

        summary_df = pd.DataFrame([{
            "method": tag,
            "mae_px": mae,
            "rmse_px": rmse,
            "bias_px": bias,
            "corr": corr
        }])

        summary_df.to_csv(
            os.path.join(metrics_dir, f"{base_name}_width_summary_{tag}.csv"),
            index=False
        )
        print(f"[WIDTH SUMMARY] {base_name} tag={tag} N={len(diffs)} mae={mae:.3f} rmse={rmse:.3f} bias={bias:.3f} corr={corr}")

    if return_normals:
        return width_rows, diffs_rows, normals_dict
    return width_rows, diffs_rows'''
    
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

    Guarantees:
      - Segment-safe (no flattening)
      - Combined supervision unpacked by segment length
      - Geodesic fallback if normals missing
      - Zoom uses ONLY union of provided mask_bbox values
      - Solid blue bbox overlay
      - TwoSlopeNorm always monotonic (0 included)
    """

    import os, json
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

    def _unpack_packed(packed, seg_lens):
        out, i, n = [], 0, len(packed)
        def skip(j):
            while j < n and packed[j] is None:
                j += 1
            return j
        i = skip(i)
        for L in seg_lens:
            cur = []
            while len(cur) < L and i < n:
                v = packed[i]; i += 1
                cur.append(np.nan if v is None else float(v))
            i = skip(i)
            out.append(np.asarray(cur, float))
        return out

    def _union_bboxes(bboxes):
        xs0, ys0, xs1, ys1 = [], [], [], []
        for b in bboxes:
            if not b or len(b) != 4:
                continue
            x,y,w,h = b
            if w <= 0 or h <= 0:
                continue
            xs0.append(x); ys0.append(y)
            xs1.append(x+w); ys1.append(y+h)
        if not xs0:
            return None
        return [min(xs0), min(ys0), max(xs1)-min(xs0), max(ys1)-min(ys0)]

    # ---------------- load GT supervision ----------------
    gt_sup = {}
    if gt_sup_root:
        p = os.path.join(gt_sup_root, "gt_supervision.json")
        if os.path.exists(p):
            with open(p, "r") as f:
                data = json.load(f)
            for e in data.get("cracks", []):
                if e.get("kind") == mode:
                    key = str(e["id"]) if mode == "atomic" else frozenset(map(str, e["members"]))
                    gt_sup[key] = e

    # ---------------- accumulators ----------------
    coords, diffs, bboxes = [], [], []
    rows = []

    # ---------------- iterate cracks ----------------
    for cid, crack in cracks.items():

        print(f"\n[WIDTH DEBUG] {mode.upper()} cid={cid}")

        if mode == "atomic":
            mid_segs = _split_on_nans(crack.get("midline", []))
        else:
            mid_segs = [np.asarray(s, float) for s in crack.get("midline_segments", [])]

        if not mid_segs:
            continue

        e1, e2 = _get_edges(crack)
        if len(e1) < 2 or len(e2) < 2:
            continue

        m_edge = min(len(e1), len(e2))
        widths_geo = np.linalg.norm(e1[:m_edge] - e2[:m_edge], axis=1)

        # ---- GT widths ----
        gt_widths = []

        if mode == "atomic" and cid in gt_sup:
            gt_widths = [np.asarray(gt_sup[cid]["gt_normals"]["width_px"], float)]

        elif mode == "combined":
            key = frozenset(map(str, crack.get("members", [])))
            if key in gt_sup:
                packed = gt_sup[key]["gt_normals"]["width_px"]
                seg_lens = [len(s) for s in mid_segs]
                gt_widths = _unpack_packed(packed, seg_lens)

        if not gt_widths:
            for s in mid_segs:
                (_,_,_,_,w),_ = normals_from_mask_for_midline(s, mask_bin, max_radius)
                gt_widths.append(np.asarray(w, float))

        # ---- compare ----
        off = 0
        for s, gtw in zip(mid_segs, gt_widths):
            m = min(len(s), len(gtw), len(widths_geo)-off)
            if m < 2:
                off += max(m,0)
                continue
            '''diffs.append(widths_geo[off:off+m] - gtw[:m])
            coords.append(s[:m])
            bboxes.append(crack.get("mask_bbox"))
            off += m'''
            d = widths_geo[off:off+m] - gtw[:m]
            pts = s[:m]

            diffs.append(d)
            coords.append(pts)
            bboxes.append(crack.get("mask_bbox"))

            for (x, y), dw, gw, pw in zip(
                    pts,
                    d,
                    gtw[:m],
                    widths_geo[off:off+m],
                ):
                if not np.isfinite(dw):
                    continue

                rows.append({
                    "x": float(x),
                    "y": float(y),
                    "gt_width_px": float(gw),
                    "pred_width_px": float(pw),
                    "width_diff_px": float(dw),
                    "cid": str(cid),
                    "crack_type": mode,
                    "midline_type": midline_type,
                })

            off += m

    # ---------------- plotting ----------------
    if not coords:
        print("[WIDTH DEBUG] nothing to plot")
        return [], []

    bbox = _union_bboxes(bboxes)

    fig, ax = plt.subplots(figsize=(6,6), dpi=200)
    bg = np.stack([(crack_mask > 0) * 255] * 3, axis=-1)
    ax.imshow(bg)

    all_d = np.concatenate(diffs)
    all_d = all_d[np.isfinite(all_d)]
    vmin, vmax = np.percentile(all_d, [5, 95])
    vmin = min(vmin, 0.0)
    vmax = max(vmax, 0.0)
    if vmax <= vmin:
        vmax = vmin + 1e-6

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("coolwarm")

    # ---- plot colored width-diff segments ----
    for s, d in zip(coords, diffs):
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

        # Replace endpoints
        ticks[0]  = vmin
        ticks[-1] = vmax

        # Remove ticks too close to endpoints (except endpoints themselves)
        cleaned = []
        for i, t in enumerate(ticks):
            if i == 0 or i == len(ticks) - 1:
                cleaned.append(t)
            else:
                if abs(t - vmin) > tol and abs(t - vmax) > tol:
                    cleaned.append(t)

        cb.set_ticks(cleaned)
        cb.set_ticklabels([f"{t:.1f}" for t in cleaned])

    out_dir = os.path.join(metrics_dir, midline_type)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{midline_type}_{crack_type}_width_diffs.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return rows, None


# ==== WIDTH SUMMARY + IMAGE-SIZED OVERLAY (no matplotlib) ======================

'''def width_diffs_to_csv(metrics_dir, base_name, width_rows, midline_type, crack_type):
    """
    Writes per-point raw width diffs.
    Path: metrics_dir/<midline_type>/<base_name>_width_diffs_<crack_type>.csv
    """
    import os
    import pandas as pd

    if not width_rows:
        print("[WIDTH DIFFS] no data to write")
        return

    out_dir = os.path.join(metrics_dir, midline_type)
    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(out_dir, f"{base_name}_width_diffs_{crack_type}.csv")
    pd.DataFrame(width_rows).to_csv(out_csv, index=False)
    print("[WIDTH DIFFS] wrote:", out_csv)'''

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