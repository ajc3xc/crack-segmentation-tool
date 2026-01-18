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


def bite_blob_to_fullmask(bite_blob, H, W):
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

    hh = min(h, mask.shape[0], H - y0)
    ww = min(w, mask.shape[1], W - x0)
    if hh > 0 and ww > 0:
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

def _decode_by_losing_branch(dom_meta, H, W):
    """
    Returns: dict[int bid] -> bool fullmask
    AUTHORITATIVE decoder.
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
            m = bite_blob_to_fullmask(blob, H, W)
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
            #print(f"\n------------{crack.keys()}-----------\n")
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
    
    width_pairs = []
    width_metric_rows = []

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
        # ATOMIC MODE (Stage-6 compatible)
        # --------------------------------------------
        if mode == "atomic":
            gt_widths = []

            # ---- Prefer GT supervision if available ----
            if gt_sup_root and str(cid) in gt_sup:
                try:
                    gt_widths = [
                        np.asarray(
                            gt_sup[str(cid)]["gt_normals"]["width_px"],
                            float
                        )
                    ]
                except Exception:
                    gt_widths = []

            # ---- Fallback: compute GT widths from mask ----
            if not gt_widths:
                for s in segs:
                    (_, _, _, _, w), _ = normals_from_mask_for_midline(
                        s, mask_bin, max_radius
                    )
                    gt_widths.append(np.asarray(w, float))

            off = 0  # running offset into widths_geo

            for s, gtw in zip(segs, gt_widths):
                if s is None or len(s) < 2:
                    continue

                # ---- determine safe aligned length ----
                m = min(
                    len(s),
                    len(gtw),
                    max(0, len(widths_geo) - off)
                )
                if m < 2:
                    off += max(m, 0)
                    continue

                # ---- aligned geometry + widths ----
                pts = np.asarray(s[:m], float)
                gw  = np.asarray(gtw[:m], float)
                pw  = np.asarray(widths_geo[off:off + m], float)

                # ---- width error (pred − gt) ----
                d = pw - gw

                # ---- legacy plotting accumulators (unchanged) ----
                coords.append(pts)
                diffs.append(d)
                bboxes.append(crack.get("mask_bbox"))

                # ---- NEW: Stage-6 compatible width_pairs entry ----
                width_pairs.append({
                    "image": base_name,
                    "cid": str(cid),
                    "crack_type": mode,
                    "midline_type": midline_type,
                    "bbox": crack.get("mask_bbox"),
                    "pts": pts,
                    "d": d,
                    "pw": pw,   # REQUIRED for Stage 6 GT vs pred plots
                    "gw": gw,   # REQUIRED for Stage 6 GT vs pred plots
                })

                # ---- per-point export rows (unchanged semantics) ----
                for (x, y), dw, gwi, pwi in zip(pts, d, gw, pw):
                    if not np.isfinite(dw):
                        continue
                    rows.append({
                        "x": float(x),
                        "y": float(y),
                        "gt_width_px": float(gwi),
                        "pred_width_px": float(pwi),
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

            # ---- GT segments (Stage-1 pruning — symmetric when possible) ----
            gt_kept = []
            gt_pruned = []

            if gt_entry is not None:
                gt_segs_all  = gt_entry.get("midline_segments") or []
                gt_meta_all  = gt_entry.get("midline_segments_meta") or []

                if len(gt_segs_all) != len(gt_meta_all):
                    # No reliable atomic metadata → do NOT prune GT
                    for Sg in gt_segs_all:
                        if Sg is None or len(Sg) < 2:
                            continue
                        gt_kept.append(np.asarray(Sg, float))
                else:
                    for Sg, mg in zip(gt_segs_all, gt_meta_all):
                        if Sg is None or len(Sg) < 2:
                            continue

                        aid = mg.get("atomic_id", None)
                        if aid is not None and str(aid) not in shared:
                            gt_pruned.append(np.asarray(Sg, float))
                        else:
                            gt_kept.append(np.asarray(Sg, float))

            # ---- figure ----
            fig, axes = plt.subplots(
                1, 2, figsize=(10, 5), dpi=200, sharex=True, sharey=True
            )

            # ---- descriptor for titles ----
            member_list = sorted(shared) if shared else sorted(pred_members)
            member_str = ", ".join(member_list)

            combo_label = f"Combined crack cid={cid}"
            members_label = f"Atomic members: [{member_str}]"

            
            axes[0].set_title(
                "GT supervision (final geometry)",
                fontsize=10,
            )

            axes[1].set_title(
                "Prediction (atomic pruning)",
                fontsize=10,
            )

            for ax in axes:
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
                f"Stage 1 Atomic Pruning - "
                f"{combo_label}  "
                f"{members_label}",
                fontsize=11,
                fontweight="bold",
            )

            out = os.path.join(opsec_dir, f"stage1_prune_{cid}.png")
            fig.savefig(out, bbox_inches="tight", dpi=200)
            plt.close(fig)

        except Exception as e:
            print(f"[OPSEC STAGE1 PLOT] skipped cid={cid}: {e}")

        # --------------------------------------------
        # Stage 2: optional branch matching (SYMMETRIC)
        # --------------------------------------------
        matched_pred_branch_ids = None

        if gt_entry is not None:
            gt_segs_all = gt_entry.get("midline_segments") or []
            gt_meta_all = (gt_entry.get("dominance_meta", {}).get("segments_meta") or [])

            gt_pruned_segs = []
            gt_pruned_meta = []

            if len(gt_segs_all) == len(gt_meta_all) and len(gt_segs_all) > 0:
                for Sg, mg in zip(gt_segs_all, gt_meta_all):
                    if Sg is None or len(Sg) < 2:
                        continue
                    # allow None branch_id; branch matcher can still use geometry
                    gt_pruned_segs.append(np.asarray(Sg, float))
                    gt_pruned_meta.append(mg if isinstance(mg, dict) else {})
            else:
                for Sg in gt_segs_all:
                    if Sg is None or len(Sg) < 2:
                        continue
                    gt_pruned_segs.append(np.asarray(Sg, float))
                    gt_pruned_meta.append({})  # dummy meta

            if gt_pruned_segs and pruned_segs:
                gt_br = _build_branch_table(gt_pruned_segs, gt_pruned_meta, shared_members=shared)
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
        
        # ============================================================
        # Stage 4: DOMINANCE-AWARE BITE (AUTHORITATIVE, EXPLANATORY)
        #
        # Purpose (READ-ONLY):
        #   - Visualize WHERE each branch loses dominance
        #   - Separate GT vs PRED vs OR(expanded) loss regions
        #   - Overlay segments to show what geometry is at risk
        #
        # Semantics:
        #   - RED    : GT-only loss
        #   - BLUE   : PRED-only loss (OR expansion)
        #   - PURPLE : GT ∩ PRED (agreement)
        #
        # NO GEOMETRY IS REMOVED HERE
        # ============================================================

        # ============================================================
        # Stage 4: DOMINANCE-AWARE BITE — LOGIC ONLY
        #   (no plotting, no matplotlib)
        # ============================================================

        # ----------------------------
        # decode dominance
        # ----------------------------
        dom_pred = crack.get("dominance_meta")
        dom_gt   = gt_entry.get("dominance_meta") if gt_entry else None

        os.makedirs(opsec_dir, exist_ok=True)
        _dump_json(os.path.join(opsec_dir, f"dom_pred_{cid}.json"), dom_pred)
        _dump_json(os.path.join(opsec_dir, f"dom_gt_{cid}.json"), dom_gt)

        loss_pred = _decode_by_losing_branch(dom_pred, H, W)
        loss_gt   = _decode_by_losing_branch(dom_gt,   H, W)

        # ------------------------------------------------------------
        # Dominance loss masks (AUTHORITATIVE)
        #   - pred loss masks clip prediction geometry
        #   - gt   loss masks clip GT geometry
        # ------------------------------------------------------------
        loss_masks_pred_by_branch = loss_pred if isinstance(loss_pred, dict) else {}
        loss_masks_gt_by_branch   = loss_gt   if isinstance(loss_gt, dict) else {}

        # Backwards-compat aliases (so older Stage-5 code doesn’t explode)
        loss_masks_by_branch = loss_masks_pred_by_branch     # historical meaning: PRED side
        loss_gt_by_branch    = loss_masks_gt_by_branch       # if any code uses this name

        if loss_masks_pred_by_branch:
            print(
                f"[BITE DOM] cid={cid} "
                f"loss_masks_pred_by_branch (PRED) = {sorted(loss_masks_pred_by_branch.keys())}"
            )
        else:
            print(
                f"[BITE DOM] cid={cid} "
                f"no prediction-side dominance loss (no pruning will occur)"
            )

        if loss_masks_gt_by_branch:
            print(
                f"[BITE DOM] cid={cid} "
                f"loss_masks_gt_by_branch (GT) = {sorted(loss_masks_gt_by_branch.keys())}"
            )

        # ----------------------------
        # union of branch IDs involved
        # ----------------------------
        all_bids = sorted(set(loss_pred.keys()) | set(loss_gt.keys()))

        if not all_bids:
            print(f"[STAGE4] cid={cid} no losing branches in GT or PRED")
            
        # ----------------------------
        # build GT segments to plot (robust)
        #   Priority:
        #     (A) keep GT segments whose atomic_id is in this combined crack's members
        #     (B) else fallback to branch_id matching (if present)
        #     (C) else fallback: plot ALL GT segments (better than blank)
        # ----------------------------
        gt_plot_segs = []
        gt_plot_meta = []

        # members of this combined crack (strings)
        _members = crack.get("members", []) or []
        members_set = set(str(m) for m in _members)

        if gt_entry is not None:
            gt_segs_all = gt_entry.get("midline_segments") or []

            # AUTHORITATIVE: GT segment metadata lives ONLY here
            gt_meta_all = (
                gt_entry.get("dominance_meta", {}).get("segments_meta") or []
            )


            print(f"[STAGE4 DBG] cid={cid} GT segs={len(gt_segs_all)} GT meta={len(gt_meta_all)} members={sorted(members_set) if members_set else 'NONE'}")

            # ---- Case 1: metadata aligned ----
            if len(gt_segs_all) == len(gt_meta_all) and len(gt_segs_all) > 0:
                kept_by_atomic = 0
                kept_by_branch = 0
                kept_all       = 0

                for Sg, mg in zip(gt_segs_all, gt_meta_all):
                    if Sg is None or len(Sg) < 2:
                        continue
                    mg = mg if isinstance(mg, dict) else {}

                    # (A) atomic_id membership filter (preferred)
                    aid = mg.get("atomic_id", None)
                    if members_set and aid is not None and str(aid) in members_set:
                        gt_plot_segs.append(np.asarray(Sg, float))
                        gt_plot_meta.append(mg)
                        kept_by_atomic += 1
                        continue

                    # (B) branch_id fallback (only if it exists)
                    bid = _safe_int(mg.get("branch_id"), None)
                    if bid is not None and bid in all_bids:
                        gt_plot_segs.append(np.asarray(Sg, float))
                        gt_plot_meta.append(mg)
                        kept_by_branch += 1
                        continue

                    # (C) if neither key exists AND we have no members_set, keep-all fallback
                    if not members_set and bid is None and aid is None:
                        gt_plot_segs.append(np.asarray(Sg, float))
                        gt_plot_meta.append(mg)
                        kept_all += 1

                print(
                    f"[STAGE4 DBG] cid={cid} GT kept: "
                    f"by_atomic={kept_by_atomic} by_branch={kept_by_branch} keepall={kept_all} "
                    f"TOTAL={len(gt_plot_segs)}"
                )

                # If we filtered everything out (common when keys don't match), do NOT go blank:
                if len(gt_plot_segs) == 0:
                    print(f"[STAGE4 DBG] cid={cid} GT kept=0 after filtering → fallback to plotting ALL GT segs")
                    for Sg, mg in zip(gt_segs_all, gt_meta_all):
                        if Sg is None or len(Sg) < 2:
                            continue
                        gt_plot_segs.append(np.asarray(Sg, float))
                        gt_plot_meta.append(mg if isinstance(mg, dict) else {})

            # ---- Case 2: metadata not aligned or missing → plot all GT segments ----
            else:
                print(f"[STAGE4 DBG] cid={cid} GT meta mismatch/missing → plotting ALL GT segs")
                for Sg in gt_segs_all:
                    if Sg is None or len(Sg) < 2:
                        continue
                    gt_plot_segs.append(np.asarray(Sg, float))
                    gt_plot_meta.append({})
        else:
            print(f"[STAGE4 DBG] cid={cid} gt_entry is None → no GT midlines available")

        # ----------------------------
        # crop window (shared by plots)
        # ----------------------------
        bb = crack.get("mask_bbox")
        if bb:
            x, y, w, h = map(int, bb)
            pad = 25
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(W, x + w + pad)
            y1 = min(H, y + h + pad)
        else:
            x0, y0, x1, y1 = 0, 0, W, H

        # ----------------------------
        # categorical dominance map
        #   0 = background
        #   1 = GT-only
        #   2 = PRED-only
        #   3 = BOTH (GT ∩ PRED)
        # ----------------------------
        dom_label = np.zeros((H, W), dtype=np.uint8)

        for bid in all_bids:
            m_gt   = loss_gt.get(bid)
            m_pred = loss_pred.get(bid)

            if m_gt is None and m_pred is None:
                continue

            if m_gt is not None:
                dom_label[m_gt.astype(bool)] |= 1

            if m_pred is not None:
                dom_label[m_pred.astype(bool)] |= 2

        # cropped + masked version (used only for plotting)
        dom_crop   = dom_label[y0:y1, x0:x1]
        dom_masked = np.ma.array(dom_crop, mask=(dom_crop == 0))
        dom_crop = dom_crop.astype(np.uint8)
        dom_masked = np.ma.array(dom_crop, mask=(dom_crop == 0))
        
        # --------------------------------------------------
        # Build atomic_id → branch_id map from prediction
        # --------------------------------------------------
        atomic_to_branch = {}

        for m in pruned_meta:
            if not isinstance(m, dict):
                continue
            aid = m.get("atomic_id", None)
            bid = m.get("branch_id", None)
            if aid is not None and bid is not None:
                atomic_to_branch[str(aid)] = int(bid)

        print(
            f"[STAGE4 DBG] cid={cid} atomic→branch map: {atomic_to_branch}"
        )


        
        # ============================================================
        # Stage 4: DOMINANCE-AWARE BITE — PLOTTING ONLY
        # ============================================================

        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.colors import ListedColormap

        # ---- helper: rebuild pred mask exactly like Stage 1 ----
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

        # ---- figure ----
        fig, axes = plt.subplots(
            1, 2, figsize=(12, 6), dpi=240, sharex=True, sharey=True
        )

        axes[0].set_title("Stage 4 — GT supervision", fontsize=10)
        axes[1].set_title("Stage 4 — Prediction", fontsize=10)

        for ax in axes:
            ax.axis("off")

        # ---- backgrounds ----
        # LEFT: GT mask
        axes[0].imshow(
            (mask_bin[y0:y1, x0:x1] > 0).astype(np.uint8),
            cmap="gray",
            vmin=0, vmax=1,
            interpolation="nearest",
            zorder=0,
        )

        # RIGHT: pred mask if AUTO, otherwise black (MANUAL)
        pred_mask_full = _rebuild_pred_mask(crack, H, W)
        if np.any(pred_mask_full):
            axes[1].imshow(
                pred_mask_full[y0:y1, x0:x1],
                cmap="gray",
                vmin=0, vmax=1,
                interpolation="nearest",
                zorder=0,
            )
        else:
            axes[1].imshow(
                np.zeros((y1 - y0, x1 - x0), np.uint8),
                cmap="gray",
                vmin=0, vmax=1,
                interpolation="nearest",
                zorder=0,
            )

        # ---- dominance overlay (UNCHANGED semantics) ----
        DOM_CMAP = ListedColormap([
            "#000000",  # 0 unused (masked)
            "#e41a1c",  # GT-only
            "#377eb8",  # Pred-only
            "#984ea3",  # GT ∩ Pred
        ])

        for ax in axes:
            ax.imshow(
                dom_masked,
                cmap=DOM_CMAP,
                interpolation="nearest",
                vmin=0,
                vmax=3,
                alpha=0.9,
                zorder=1,
            )

        # ---- overlay segments ----
        color_cycle = [
            (0.95, 0.90, 0.25),
            (0.25, 0.85, 0.35),
            (0.25, 0.55, 0.95),
            (0.95, 0.35, 0.35),
        ]

        legend_handles = []
        seen = set()


        # ----------------------------
        # LEFT: GT supervision midlines (PRUNED)
        # ----------------------------
        for Sg, mg in zip(gt_plot_segs, gt_plot_meta):
            if Sg is None or len(Sg) < 2:
                continue

            # Resolve GT segment → branch via atomic_id
            aid = mg.get("atomic_id", None)
            bid = None

            if aid is not None:
                bid = atomic_to_branch.get(str(aid), None)

            # Fallbacks (never collapse everything silently)
            if bid is None:
                bid = _safe_int(mg.get("branch_id"), None)

            if bid is None:
                bid = 0  # last-resort fallback, explicit

            col = color_cycle[bid % len(color_cycle)]

            S2 = Sg - np.array([x0, y0], float)

            axes[0].plot(
                S2[:, 0],
                S2[:, 1],
                color=col,
                lw=2.3,
                zorder=5,
            )

            if bid not in seen:
                legend_handles.append(
                    Line2D([0], [0], color=col, lw=3, label=f"branch {bid}")
                )
                seen.add(bid)

        # ----------------------------
        # RIGHT: prediction midlines
        # ----------------------------
        for S, m in zip(pruned_segs, pruned_meta):
            if S is None or len(S) < 2:
                continue

            bid = _safe_int(m.get("branch_id"), 0)
            col = color_cycle[bid % len(color_cycle)]

            S = np.asarray(S, float)
            S2 = S - np.array([x0, y0], float)

            axes[1].plot(
                S2[:, 0],
                S2[:, 1],
                color=col,
                lw=2.3,
                zorder=5,
            )

        # ---- legend ----
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

        out = os.path.join(opsec_dir, f"stage4_dominance_bite_{cid}.png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)

        print(f"[OPSEC] Stage-4 dominance plot written: {out}")

        # ============================================================
        # Stage 4.5: APPLY DOMINANCE BITE (SYMMETRIC, DISPOSABLE)
        #
        # Key rule (per branch):
        #   - BOTH GT and PRED are clipped by the UNION of:
        #       (GT loss mask) ∪ (PRED loss mask)
        #
        # Dominance defines INVALID TERRITORY, not ownership.
        #
        # OUTPUTS (used by Stage 5 + OPSEC):
        #   - pruned_segs / pruned_meta        : PRED geometry after union-bite clipping
        #   - bite_pruned_pred_segs           : removed PRED runs (viz)
        #   - gt_stage5_segs / gt_stage5_meta : GT geometry after union-bite clipping
        #   - bite_pruned_gt_segs             : removed GT runs (viz)
        #   - pred_pre45_segs                 : snapshot of PRED geometry BEFORE dominance
        # ============================================================
        
        # ------------------------------------------------------------
        # Union-bite helper (GT ∪ PRED)
        # ------------------------------------------------------------
        def _union_bite_for_branch(bid):
            mg = loss_masks_gt_by_branch.get(bid)
            mp = loss_masks_pred_by_branch.get(bid)

            if mg is None and mp is None:
                return None
            if mg is None:
                return mp
            if mp is None:
                return mg
            return (mg.astype(bool) | mp.astype(bool))


        # ------------------------------------------------------------
        # Snapshot BEFORE dominance (for provenance / topology plot)
        # ------------------------------------------------------------
        pred_pre45_segs = [
            np.asarray(S, float)
            for S in pruned_segs
            if S is not None and len(S) >= 2
        ]

        # ============================================================
        # PRED: clip by UNION dominance bite
        # ============================================================
        pred_dom_clipped_segs = []
        pred_dom_clipped_meta = []
        bite_pruned_pred_segs = []   # viz only

        for S, m in zip(pruned_segs, pruned_meta):
            if S is None or len(S) < 2:
                continue

            m = m if isinstance(m, dict) else {}
            bid = _safe_int(m.get("branch_id"), None)
            rm = _union_bite_for_branch(bid)

            if rm is None:
                pred_dom_clipped_segs.append(np.asarray(S, float))
                pred_dom_clipped_meta.append(m)
                continue

            kept_runs, removed_runs = _clip_polyline_into_runs(S, rm, H, W, min_pts=2)
            bite_pruned_pred_segs.extend(removed_runs)

            for kr in kept_runs:
                pred_dom_clipped_segs.append(np.asarray(kr, float))
                pred_dom_clipped_meta.append(m)

        print(
            f"[DOM CLIP UNION | PRED] cid={cid} kept {len(pred_dom_clipped_segs)} runs "
            f"(from {len(pruned_segs)} segs)"
        )

        # overwrite authoritative prediction geometry
        pruned_segs = pred_dom_clipped_segs
        pruned_meta = pred_dom_clipped_meta

        # ============================================================
        # GT: clip by SAME UNION dominance bite
        # ============================================================
        gt_stage5_segs = []
        gt_stage5_meta = []
        bite_pruned_gt_segs = []   # viz only

        gt_segs_all = gt_entry.get("midline_segments", []) if gt_entry else []
        gt_meta_all = (
            gt_entry.get("dominance_meta", {}).get("segments_meta") or []
        ) if gt_entry else []

        if gt_entry and len(gt_segs_all) == len(gt_meta_all) and len(gt_segs_all) > 0:
            for Sg, mg in zip(gt_segs_all, gt_meta_all):
                if Sg is None or len(Sg) < 2:
                    continue

                mg = mg if isinstance(mg, dict) else {}
                bid = _safe_int(mg.get("branch_id"), None)
                rm = _union_bite_for_branch(bid)

                if rm is None:
                    gt_stage5_segs.append(np.asarray(Sg, float))
                    gt_stage5_meta.append(mg)
                    continue

                kept_runs, removed_runs = _clip_polyline_into_runs(Sg, rm, H, W, min_pts=2)
                bite_pruned_gt_segs.extend(removed_runs)

                for kr in kept_runs:
                    gt_stage5_segs.append(np.asarray(kr, float))
                    gt_stage5_meta.append(mg)

            print(
                f"[DOM CLIP UNION | GT]   cid={cid} kept {len(gt_stage5_segs)} runs "
                f"(from {len(gt_segs_all)} segs)"
            )
        else:
            # fallback: cannot align GT meta → keep GT uncut
            gt_stage5_segs = [
                np.asarray(Sg, float)
                for Sg in gt_segs_all
                if Sg is not None and len(Sg) >= 2
            ]
            gt_stage5_meta = [{} for _ in gt_stage5_segs]
            bite_pruned_gt_segs = []
            print(
                f"[DOM CLIP UNION | GT]   cid={cid} GT meta missing/misaligned → "
                f"kept {len(gt_stage5_segs)} uncut segs"
            )
                                        
        # ============================================================
        # Stage 5: width slicing (DOMINANCE-APPLIED GEOMETRY)
        # ============================================================

        final_pred_segs = []
        stage4_pairs = []             # (pts_ok, width_diff)

        # ---- GT plot segments (dominance-clipped, symmetric) ----
        gt_plot_segs = [np.asarray(s, float) for s in (gt_stage5_segs or []) if s is not None and len(s) >= 2]
        gt_plot_meta = gt_stage5_meta if isinstance(gt_stage5_meta, list) else []


        # ============================================================
        # Consume DOMINANCE-CLIPPED prediction geometry
        # ============================================================        
        for si, (S, m) in enumerate(zip(pruned_segs, pruned_meta)):
            if S is None or len(S) < 2:
                continue

            # ---- determine width index source ----
            if (
                have_valid_seg_idx
                and isinstance(m.get("seg_idx"), int)
                and m["seg_idx"] in seg_start
            ):
                s0 = seg_start[m["seg_idx"]]
                src = "seg_idx"
            else:
                s0 = off_fallback
                src = "fallback"

            L = len(S)
            s1 = min(s0 + L, len(widths_geo))

            pw_full  = widths_geo[s0:s1]
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

            # ---- GT widths (authoritative) ----
            (_, _, _, _, gw), _ = normals_from_mask_for_midline(
                pts_full, mask_bin, max_radius
            )
            gw = np.asarray(gw, float)

            mlen = min(len(pts_full), len(pw_full), len(gw))
            if mlen < 2:
                continue

            pts_raw = np.asarray(pts_full[:mlen], float)
            pw_raw  = np.asarray(pw_full[:mlen], float)
            gw_raw  = np.asarray(gw[:mlen], float)

            # ---- no dominance logic here (already clipped) ----
            d_ok = pw_raw - gw_raw

            final_pred_segs.append(pts_raw)
            stage4_pairs.append((pts_raw, d_ok))

            diffs.append(d_ok)
            coords.append(pts_raw)

            bbox0 = crack.get("mask_bbox")
            if bbox0 is not None:
                bboxes.append(bbox0)
                
            pts = pts_raw
            d   = d_ok
            # IMPORTANT: include pw/gw so Stage 6 can resample BOTH and plot the effect
            width_pairs.append({
                "image": base_name,
                "cid": str(cid),
                "crack_type": mode,               # "atomic" or "combined"
                "midline_type": midline_type,     # "auto"/"manual"/...
                "bbox": crack.get("mask_bbox"),
                "pts": np.asarray(pts_raw, float),
                "pw":  np.asarray(pw_raw, float),  # pred widths along pts
                "gw":  np.asarray(gw_raw, float),  # gt widths along pts
                "d":   np.asarray(d_ok, float),    # pw - gw
                # optional, if you have them:
                "branch_id": m.get("branch_id") if isinstance(m, dict) else None,
                "seg_idx":   m.get("seg_idx")   if isinstance(m, dict) else None,
            })



            for (x, y), dw, gwi, pwi in zip(pts_raw, d_ok, gw_raw, pw_raw):
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

        # ============================================================
        # OPSEC PLOT — STAGE 5 FINAL GEOMETRY (DOMINANCE-RESOLVED)
        #   - NO dominance pruning logic here
        #   - Overlay can be categorical (GT-only / PRED-only / BOTH) like Stage 4
        # ============================================================
        assert pred_pre45_segs, "pred_pre45_segs empty — snapshot timing broken"
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

            def _split_finite_undef(pts, d, min_pts=2):
                pts = np.asarray(pts, float)
                d = np.asarray(d, float)
                n = min(len(pts), len(d))
                if n < 2:
                    return [], []

                kept, undef = [], []
                buf_k, buf_u = [], []

                for i in range(n - 1):
                    p0, p1 = pts[i], pts[i + 1]
                    if np.isfinite(d[i]):
                        if not buf_k:
                            buf_k.append(p0)
                        buf_k.append(p1)
                        if len(buf_u) >= min_pts:
                            undef.append(np.asarray(buf_u, float))
                        buf_u = []
                    else:
                        if not buf_u:
                            buf_u.append(p0)
                        buf_u.append(p1)
                        if len(buf_k) >= min_pts:
                            kept.append(np.asarray(buf_k, float))
                        buf_k = []

                if len(buf_k) >= min_pts:
                    kept.append(np.asarray(buf_k, float))
                if len(buf_u) >= min_pts:
                    undef.append(np.asarray(buf_u, float))
                return kept, undef

            # --------------------------------------------------
            # Build prediction plot segments (kept vs undef)
            # --------------------------------------------------
            pred_kept_segs = []
            pred_undef_segs = []
            pred_stage5_support = []

            for pts_ok, d_ok in stage4_pairs:
                k, u = _split_finite_undef(pts_ok, d_ok, min_pts=2)
                pred_kept_segs.extend(k)
                pred_undef_segs.extend(u)
                pred_stage5_support.append(pts_ok)

            # --------------------------------------------------
            # TOPOLOGY-PRUNED geometry (compute against pre-4.5 snapshot)
            # --------------------------------------------------
            keep_set = _polyline_points_keyset(pred_stage5_support)
            if bite_pruned_pred_segs:
                keep_set |= _polyline_points_keyset(bite_pruned_pred_segs)
            
            topology_pruned_pred_segs = subtract_segments_by_pointset(
                pred_pre45_segs,
                keep_set,
                min_pts=2,
            )

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
            col_topo  = (0.7, 0.1, 0.1)
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
            for S in topology_pruned_pred_segs:
                if S is None or len(S) < 2:
                    continue
                axes[1].plot(S[:, 0] - x0, S[:, 1] - y0, color=col_topo, lw=1.4, zorder=2)

            for S in bite_pruned_pred_segs:
                if S is None or len(S) < 2:
                    continue
                axes[1].plot(S[:, 0] - x0, S[:, 1] - y0, color=col_bite, lw=2.0, zorder=3)

            for S in pred_undef_segs:
                if S is None or len(S) < 2:
                    continue
                axes[1].plot(S[:, 0] - x0, S[:, 1] - y0, color=col_undef, lw=2.2, zorder=4)

            for S in pred_kept_segs:
                if S is None or len(S) < 2:
                    continue
                axes[1].plot(S[:, 0] - x0, S[:, 1] - y0, color=col_keep, lw=2.5, zorder=5)

            for ax in axes:
                ax.add_patch(
                    plt.Rectangle((x - x0, y - y0), w, h, fill=False,
                                edgecolor="dodgerblue", lw=1.5)
                )

            legend_items = [
                Line2D([0],[0], color=col_keep,  lw=2.5, label="Finite dist (kept)"),
                Line2D([0],[0], color=col_undef, lw=2.2, label="Infinite/undef"),
                Line2D([0],[0], color=col_bite,  lw=2.0, label="Dominance-bite (union)"),
                Line2D([0],[0], color=col_topo,  lw=1.4, label="Topology-pruned"),
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
        # MIDLINE METRICS (COMBINED + AUTO ONLY)
        # ============================================================
        if mode == "combined" and midline_type == "auto" and gt_entry is not None:
            try:
                from helpers.metrics import compute_midline_metrics
                import math
                import numpy as np

                # ---- build GT midline (same pruning rules as pred) ----
                gt_segs_all = gt_entry.get("midline_segments") or []

                # AUTHORITATIVE: GT segment metadata lives ONLY here
                gt_meta_all = (
                    gt_entry.get("dominance_meta", {}).get("segments_meta") or []
                )


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

                if final_pred_segs and gt_keep:
                    pred_mid = np.vstack(final_pred_segs)
                    gt_mid   = np.vstack(gt_keep)

                    mm = compute_midline_metrics(pred_mid, gt_mid)

                    ch  = float(mm.get("nn_mean_bidirectional", np.inf))
                    hd  = float(mm.get("hausdorff_max", np.inf))
                    cov = float(mm.get("coverage_min", 0.0))

                    score_mid = (
                        math.log1p(max(ch, 0.0)) +
                        0.5 * math.log1p(max(hd, 0.0)) +
                        (1.0 - float(np.clip(cov, 0.0, 1.0)))
                    )

                    bbox0 = crack.get("mask_bbox")

                    midline_metric_rows.append({
                        "image": base_name,
                        "crack_id": str(cid),
                        "variant_global_id": -1,
                        "os_mode": "combined",
                        "g11": np.nan,
                        "g22": np.nan,
                        "g33": np.nan,

                        "length_px": _linestring_length(gt_mid),
                        "bbox_area": float(bbox0[2] * bbox0[3]) if bbox0 else np.nan,

                        # --- selection metrics ---
                        "nn_mean_bidirectional": ch,
                        "hausdorff_max": hd,
                        "coverage_min": cov,
                        "score_mid": score_mid,

                        # --- diagnostics ---
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
    # Stage 6: FAIR WIDTH METRICS (ARCLENGTH RESAMPLING + LENGTH-WEIGHTED STATS)
    #   - Postprocess Stage-5 outputs for BOTH atomic + combined
    #   - Computes length-weighted RMSE/MAE/Bias per (image,cid,crack_type,midline_type)
    #   - Produces committee-friendly plots in:
    #       <metrics_dir>/<midline_type>/compare_widths_debug/<mode>/stage6/...
    #   - Swaps final compare-width plotting inputs to the Stage-6 resampled geometry
    #
    # NOTES:
    #   - Resampling is applied to the *measurement samples* (d(s)=pred-gt), not to GT geometry itself.
    #   - If you also pass per-sample pred/gt widths into width_pairs as "pw" and "gw", Stage 6 will
    #     resample and plot those too (so you can show “effect on GT vs pred” explicitly).
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

        debug_mode_dir = os.path.join(debug_root, str(mode))
        os.makedirs(debug_mode_dir, exist_ok=True)

        stage6_dir = os.path.join(debug_mode_dir, "stage6")
        os.makedirs(stage6_dir, exist_ok=True)

        stage6_metrics_dir  = os.path.join(stage6_dir, "metrics")
        stage6_resample_dir = os.path.join(stage6_dir, "resample")
        os.makedirs(stage6_metrics_dir, exist_ok=True)
        os.makedirs(stage6_resample_dir, exist_ok=True)

        # ------------------------------------------------------------
        # If nothing pushed into width_pairs yet, fallback to Stage-5 plot inputs
        # (Better: explicitly push width_pairs during Stage 5; this fallback is “do not crash”.)
        # ------------------------------------------------------------
        if (not width_pairs) and ("coords" in locals()) and ("diffs" in locals()) and coords and diffs:
            for pts, d in zip(coords, diffs):
                if pts is None or d is None:
                    continue
                width_pairs.append({
                    "image": base_name if "base_name" in locals() else "",
                    "cid": "",  # unknown in fallback
                    "crack_type": mode,
                    "midline_type": midline_type,
                    "bbox": None,
                    "pts": np.asarray(pts, float),
                    "d": np.asarray(d, float),
                    # Optional (not available here): "pw", "gw"
                })

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

        # ------------------------------------------------------------
        # Stage 6 main: resample finite runs and compute length-weighted stats
        # ------------------------------------------------------------
        coords_stage6, diffs_stage6, bboxes_stage6 = [], [], []
        # store resampling artifacts for explainers (small, for selected cids only)
        stage6_cache = []  # each item: dict with keys about runs + resampled versions

        ds_target_px = 1.0  # knob later

        # per-crack aggregation: key=(image,cid,crack_type,midline_type)
        per_crack = {}

        print("\n[STAGE6 DEBUG] ===============================")
        print("[STAGE6 DEBUG] ENTER Stage 6")
        print(f"[STAGE6 DEBUG] width_pairs count = {len(width_pairs or [])}")
        print(f"[STAGE6 DEBUG] ds_target_px = {ds_target_px}")
        print("[STAGE6 DEBUG] ===============================")

        for wp in (width_pairs or []):
            pts = wp.get("pts", None)
            d   = wp.get("d", None)

            print(
                f"[STAGE6 DEBUG] ▶ wp: "
                f"cid={wp.get('cid','')}, "
                f"type={wp.get('crack_type',mode)}, "
                f"midline={wp.get('midline_type',midline_type)}, "
                f"pts={None if pts is None else len(pts)}, "
                f"d={None if d is None else len(d)}"
            )

            if pts is None or d is None:
                print("[STAGE6 DEBUG]   ⛔ skipped: missing pts or d")
                continue

            pts = np.asarray(pts, float)
            d   = np.asarray(d, float)
            n = min(len(pts), len(d))
            if n < 2:
                print("[STAGE6 DEBUG]   ⛔ skipped: <2 samples")
                continue
            pts = pts[:n]
            d   = d[:n]

            # optional: per-sample pred width and gt width (for “effect on GT vs pred” plots)
            pw = wp.get("pw", None)
            gw = wp.get("gw", None)
            pw = None if pw is None else np.asarray(pw, float)[:n]
            gw = None if gw is None else np.asarray(gw, float)[:n]

            image = str(wp.get("image", base_name if "base_name" in locals() else ""))
            cid_s = str(wp.get("cid", ""))
            ctype = str(wp.get("crack_type", mode))
            mtype = str(wp.get("midline_type", midline_type))
            bbox  = wp.get("bbox", None)

            s_full = arclen_s(pts)
            if len(s_full) < 2:
                print("[STAGE6 DEBUG]   ⛔ skipped: invalid arclength")
                continue
            total_len = float(s_full[-1] - s_full[0])
            if (not np.isfinite(total_len)) or total_len <= 0:
                print("[STAGE6 DEBUG]   ⛔ skipped: non-finite total_len")
                continue

            finite_mask = np.isfinite(d)
            runs = _contiguous_true_runs(finite_mask)

            print(
                f"[STAGE6 DEBUG]   runs found = {len(runs)} "
                f"(finite samples = {int(np.sum(finite_mask))})"
            )

            # IMPORTANT: this MUST be accumulated, otherwise finL stays 0 and plots are skipped
            finite_len = 0.0
            run_stats = []

            # cache entry for explainers
            cache_item = {
                "image": image, "cid": cid_s, "crack_type": ctype, "midline_type": mtype,
                "bbox": bbox,
                "runs": [],  # list of dicts: original + resampled
            }

            for (i0, i1) in runs:
                if i1 - i0 + 1 < 2:
                    print(f"[STAGE6 DEBUG]     ⛔ run [{i0}:{i1}] too short")
                    continue

                pts_run = pts[i0:i1 + 1]
                d_run   = d[i0:i1 + 1]
                pw_run  = None if pw is None else pw[i0:i1 + 1]
                gw_run  = None if gw is None else gw[i0:i1 + 1]

                # Mandatory resampling call (consistent contract for ALL modes)
                pts_rs, d_rs, pw_rs, gw_rs = resample_by_arclength(
                    pts_run, d_run, pw_run, gw_run,
                    ds_target=ds_target_px,
                    min_pts=2,
                    preserve_endpoints=True,
                    fastpath=True,   # internal identity when already uniform enough
                )
                if pts_rs is None or d_rs is None or len(pts_rs) < 2:
                    print(f"[STAGE6 DEBUG]     ⛔ resample failed for run [{i0}:{i1}]")
                    continue

                s_rs = arclen_s(pts_rs)
                if len(s_rs) < 2:
                    print(f"[STAGE6 DEBUG]     ⛔ invalid s_rs for run [{i0}:{i1}]")
                    continue

                ds_w = np.diff(s_rs)
                runL = float(np.sum(ds_w))
                if (not np.isfinite(runL)) or runL <= 0:
                    print(f"[STAGE6 DEBUG]     ⛔ non-finite run length for run [{i0}:{i1}]")
                    continue

                # ✅ CRITICAL FIX: accumulate finite length, otherwise per-crack finL remains 0
                finite_len += runL

                print(
                    f"[STAGE6 DEBUG]     run [{i0}:{i1}] "
                    f"→ resampled pts={len(pts_rs)}, run_len_px={runL:.2f} "
                    f"(finite_len_px now {finite_len:.2f})"
                )

                # Save for final compare-width plot
                coords_stage6.append(np.asarray(pts_rs, float))
                diffs_stage6.append(np.asarray(d_rs, float))
                if bbox is not None:
                    bboxes_stage6.append(bbox)

                st = _length_weighted_err_stats(d_rs[:-1], ds_w)
                st["run_len_px"] = runL
                run_stats.append(st)

                # cache for explainers
                cache_item["runs"].append({
                    "i0": int(i0), "i1": int(i1),
                    "pts": np.asarray(pts_run, float),
                    "d":   np.asarray(d_run, float),
                    "pw":  None if pw_run is None else np.asarray(pw_run, float),
                    "gw":  None if gw_run is None else np.asarray(gw_run, float),
                    "pts_rs": np.asarray(pts_rs, float),
                    "d_rs":   np.asarray(d_rs, float),
                    "pw_rs":  None if pw_rs is None else np.asarray(pw_rs, float),
                    "gw_rs":  None if gw_rs is None else np.asarray(gw_rs, float),
                    "s":   arclen_s(pts_run),
                    "s_rs": s_rs,
                    "run_len_px": runL,
                })

        # Only keep cache items that actually have runs; reduces noise + makes debug clearer
            if cache_item["runs"]:
                stage6_cache.append(cache_item)
                print(
                    f"[STAGE6 DEBUG] cache_item added: "
                    f"cid={cid_s}, runs_cached={len(cache_item['runs'])}, "
                    f"finite_len_px={finite_len:.2f}"
                )
            else:
                print(f"[STAGE6 DEBUG] cache_item skipped (no valid runs): cid={cid_s}")

            # Per-crack aggregation (length-weighted over runs)
            key = (image, cid_s, ctype, mtype)
            if key not in per_crack:
                per_crack[key] = {
                    "total_len_px": total_len,
                    "finite_len_px": 0.0,
                    "sum_bias_L": 0.0,
                    "sum_mae_L": 0.0,
                    "sum_mse_L": 0.0,
                    "bbox": bbox,
                }

            bin_ = per_crack[key]
            bin_["total_len_px"] = max(float(bin_.get("total_len_px", 0.0)), total_len)
            bin_["finite_len_px"] += float(finite_len)

            for st in run_stats:
                L = float(st.get("run_len_px", 0.0))
                if (not np.isfinite(L)) or L <= 0:
                    continue

                b = float(st.get("bias", np.nan))
                a = float(st.get("mae", np.nan))
                r = float(st.get("rmse", np.nan))

                if np.isfinite(b):
                    bin_["sum_bias_L"] += b * L
                if np.isfinite(a):
                    bin_["sum_mae_L"] += a * L
                if np.isfinite(r):
                    bin_["sum_mse_L"] += (r ** 2) * L  # pool via MSE

        print("\n[STAGE6 DEBUG] -------- SUMMARY AFTER RESAMPLING --------")
        print(f"[STAGE6 DEBUG] coords_stage6 count = {len(coords_stage6)}")
        print(f"[STAGE6 DEBUG] stage6_cache items = {len(stage6_cache)}")
        print(
            "[STAGE6 DEBUG] cache keys =",
            {(it['cid'], it['crack_type'], it['midline_type']) for it in stage6_cache}
        )
        print(f"[STAGE6 DEBUG] per_crack bins = {len(per_crack)}")
        if per_crack:
            # print a tiny sample for sanity
            samp = next(iter(per_crack.items()))
            print(f"[STAGE6 DEBUG] per_crack sample key={samp[0]} finL={samp[1].get('finite_len_px', None)} totL={samp[1].get('total_len_px', None)}")
        print("[STAGE6 DEBUG] -------------------------------------------\n")

        # ------------------------------------------------------------
        # Emit per-crack metric rows
        # ------------------------------------------------------------
        rows_added = 0
        for (image, cid_s, ctype, mtype), bin_ in per_crack.items():
            totL = float(bin_.get("total_len_px", 0.0))
            finL = float(bin_.get("finite_len_px", 0.0))
            if (not np.isfinite(totL)) or totL <= 0:
                continue

            coverage = float(np.clip(finL / (totL + 1e-12), 0.0, 1.0))

            bias = float(bin_["sum_bias_L"] / (finL + 1e-12)) if finL > 0 else np.nan
            mae  = float(bin_["sum_mae_L"]  / (finL + 1e-12)) if finL > 0 else np.nan
            rmse = float(math.sqrt(bin_["sum_mse_L"] / (finL + 1e-12))) if finL > 0 else np.nan

            bbox = bin_.get("bbox", None)
            bbox_area = float(bbox[2] * bbox[3]) if bbox and len(bbox) >= 4 else np.nan

            width_metric_rows.append({
                "image": image,
                "crack_id": str(cid_s),
                "crack_type": ctype,
                "midline_type": mtype,

                # weights / geometry
                "total_len_px": totL,
                "finite_len_px": finL,
                "finite_len_frac": coverage,
                "bbox_area": bbox_area,

                # core width error metrics (length-weighted)
                "width_bias_L": bias,     # pred − gt
                "width_mae_L": mae,
                "width_rmse_L": rmse,
            })
            rows_added += 1

        print(f"[STAGE6 DEBUG] emitted metric rows added = {rows_added} (width_metric_rows total now {len(width_metric_rows)})")


        # ------------------------------------------------------------
        # Stage 6 plots
        #   (A) TopK metrics: RMSE + MAE + Bias + finite_len_px (weight)
        #   (B) Resampling explainers:
        #       - worst / median / best by RMSE
        #       - for each: show ALL finite runs (original + resampled)
        #       - show d(s) curves per-run
        #       - if pw/gw available, also show pw(s), gw(s) before/after
        # ------------------------------------------------------------


        # ======================================================================
        # STAGE 6 PLOT HELPERS (MUST BE DEFINED BEFORE USE)
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


        def plot_sampling_consistency(
            *,
            pts_list,
            ptsr_list,
            mask_bin,
            crop,
            title,
            out_path,
        ):
            x0, y0, x1, y1 = crop

            if not pts_list or not ptsr_list:
                return

            pts_all  = np.vstack([p for p in pts_list if p is not None and len(p) >= 2])
            ptsr_all = np.vstack([p for p in ptsr_list if p is not None and len(p) >= 2])

            if len(pts_all) < 2 or len(ptsr_all) < 2:
                return

            ds0 = np.linalg.norm(np.diff(pts_all, axis=0), axis=1)
            ds1 = np.linalg.norm(np.diff(ptsr_all, axis=0), axis=1)
            ds0 = ds0[np.isfinite(ds0)]
            ds1 = ds1[np.isfinite(ds1)]

            if ds0.size < 1 or ds1.size < 1:
                return

            vmin = np.percentile(ds0, 10)
            vmax = np.percentile(ds0, 90)
            if vmax <= vmin:
                vmax = vmin + 1e-6

            cmap = plt.get_cmap("viridis")
            norm = plt.Normalize(vmin, vmax)

            fig, (axO, axR) = plt.subplots(1, 2, figsize=(13.8, 5.2), dpi=200)

            for ax in (axO, axR):
                try:
                    ax.imshow(mask_bin[y0:y1, x0:x1], cmap="gray", zorder=0)
                except Exception:
                    pass
                ax.axis("off")

            draw_colored_polyline(axO, pts_all,  ds0, x0, y0, 2, cmap, norm, 0.85)
            draw_colored_polyline(axR, ptsr_all, ds1, x0, y0, 2, cmap, norm, 0.90)

            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            plt.colorbar(sm, ax=[axO, axR], fraction=0.03, pad=0.03)

            fig.text(0.5, 0.985, title, ha="center", va="top", fontsize=11, fontweight="bold")
            fig.savefig(out_path, bbox_inches="tight", dpi=200)
            plt.close(fig)


        def plot_width_error_distribution(
            *,
            runs,
            title,
            out_path,
            bins=25,   # ⬅️ increased
        ):
            """
            Width error distribution comparison:
            - ORIGINAL vs RESAMPLED
            - SAME axes
            - Normalized histogram (honest density)
            """

            import numpy as np
            import matplotlib.pyplot as plt
            import os

            d_orig_all = []
            d_rs_all   = []

            for r in runs:
                d0 = np.asarray(r.get("d", []), float)
                d1 = np.asarray(r.get("d_rs", []), float)

                if len(d0) >= 2:
                    d_orig_all.append(d0[:-1])
                if len(d1) >= 2:
                    d_rs_all.append(d1[:-1])

            if not d_orig_all or not d_rs_all:
                print(f"[STAGE6 WIDTH DIST] skipped (no valid samples): {out_path}")
                return

            d_orig_all = np.concatenate(d_orig_all)
            d_rs_all   = np.concatenate(d_rs_all)

            d_orig_all = d_orig_all[np.isfinite(d_orig_all)]
            d_rs_all   = d_rs_all[np.isfinite(d_rs_all)]

            if d_orig_all.size < 10 or d_rs_all.size < 10:
                print(f"[STAGE6 WIDTH DIST] skipped (too few samples): {out_path}")
                return

            fig, ax = plt.subplots(1, 1, figsize=(8.8, 4.8), dpi=200)

            ax.hist(
                d_orig_all,
                bins=bins,
                density=True,
                alpha=0.45,
                label="original",
            )
            ax.hist(
                d_rs_all,
                bins=bins,
                density=True,
                alpha=0.45,
                label="resampled",
            )

            ax.axvline(0.0, lw=1.4, color="black", alpha=0.8)

            rmse_orig = float(np.sqrt(np.mean(d_orig_all ** 2)))
            rmse_rs   = float(np.sqrt(np.mean(d_rs_all ** 2)))

            mean_orig = float(np.mean(d_orig_all))
            mean_rs   = float(np.mean(d_rs_all))

            ax.axvline(mean_orig, lw=1.2, linestyle="--", alpha=0.8)
            ax.axvline(mean_rs,   lw=1.2, linestyle="--", alpha=0.8)

            ax.set_xlabel("Width error (pred − gt) [px]", fontsize=9)
            ax.set_ylabel("Density", fontsize=9)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)

            ax.text(
                0.02, 0.96,
                f"RMSE (orig) = {rmse_orig:.3f}px\nRMSE (resampled) = {rmse_rs:.3f}px",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
            )

            fig.suptitle(title, fontsize=11, fontweight="bold")
            fig.savefig(out_path, bbox_inches="tight", dpi=200)
            plt.close(fig)

            print(f"[STAGE6 WIDTH DIST] wrote: {out_path}")

        def plot_stage6_width_signals_preservation(
            *,
            run,
            title,
            out_path,
        ):
            """
            Plot GT width and predicted width vs arclength for a SINGLE run,
            showing original vs resampled on the same axes (shape preservation).

            Expects run dict keys from your Stage-6 cache:
            - s, gw, pw
            - s_rs, gw_rs, pw_rs
            """
            import numpy as np
            import matplotlib.pyplot as plt
            import os

            # ------------------------------------------------------------
            # Pull signals
            # ------------------------------------------------------------
            s0  = np.asarray(run.get("s", []), float)    if run.get("s")    is not None else None
            s1  = np.asarray(run.get("s_rs", []), float) if run.get("s_rs") is not None else None
            gw0 = np.asarray(run.get("gw", []), float)   if run.get("gw")   is not None else None
            gw1 = np.asarray(run.get("gw_rs", []), float)if run.get("gw_rs")is not None else None
            pw0 = np.asarray(run.get("pw", []), float)   if run.get("pw")   is not None else None
            pw1 = np.asarray(run.get("pw_rs", []), float)if run.get("pw_rs")is not None else None

            missing = []
            if s0  is None or len(s0)  < 2: missing.append("s")
            if s1  is None or len(s1)  < 2: missing.append("s_rs")
            if gw0 is None or len(gw0) < 2: missing.append("gw")
            if gw1 is None or len(gw1) < 2: missing.append("gw_rs")
            if pw0 is None or len(pw0) < 2: missing.append("pw")
            if pw1 is None or len(pw1) < 2: missing.append("pw_rs")

            if missing:
                print(f"[STAGE6 SIGNAL] skipped (missing/short: {', '.join(missing)})")
                return

            # ------------------------------------------------------------
            # Align lengths safely
            # ------------------------------------------------------------
            n_g0 = min(len(s0), len(gw0))
            n_g1 = min(len(s1), len(gw1))
            n_p0 = min(len(s0), len(pw0))
            n_p1 = min(len(s1), len(pw1))

            s0g, gw0 = s0[:n_g0], gw0[:n_g0]
            s1g, gw1 = s1[:n_g1], gw1[:n_g1]
            s0p, pw0 = s0[:n_p0], pw0[:n_p0]
            s1p, pw1 = s1[:n_p1], pw1[:n_p1]

            g0_ok = np.isfinite(s0g) & np.isfinite(gw0)
            g1_ok = np.isfinite(s1g) & np.isfinite(gw1)
            p0_ok = np.isfinite(s0p) & np.isfinite(pw0)
            p1_ok = np.isfinite(s1p) & np.isfinite(pw1)

            if not (np.any(g0_ok) and np.any(g1_ok) and np.any(p0_ok) and np.any(p1_ok)):
                print("[STAGE6 SIGNAL] skipped (no finite width samples)")
                return

            # ------------------------------------------------------------
            # Means
            # ------------------------------------------------------------
            gw0_mean = float(np.mean(gw0[g0_ok]))
            gw1_mean = float(np.mean(gw1[g1_ok]))
            pw0_mean = float(np.mean(pw0[p0_ok]))
            pw1_mean = float(np.mean(pw1[p1_ok]))

            # ------------------------------------------------------------
            # Shared axis limits (start at 0)
            # ------------------------------------------------------------
            s_max = max(
                np.max(s0g[g0_ok]),
                np.max(s1g[g1_ok]),
                np.max(s0p[p0_ok]),
                np.max(s1p[p1_ok]),
            )

            w_max = max(
                np.max(gw0[g0_ok]),
                np.max(gw1[g1_ok]),
                np.max(pw0[p0_ok]),
                np.max(pw1[p1_ok]),
            )

            xlim = (0.0, float(s_max) * 1.02)
            ylim = (0.0, float(w_max) * 1.05)

            # ------------------------------------------------------------
            # Plot
            # ------------------------------------------------------------
            fig, (axG, axP) = plt.subplots(
                1, 2, figsize=(13.8, 5.2), dpi=200, sharex=True, sharey=True
            )

            fig.suptitle(title, fontsize=14, fontweight="bold")

            # --- GT panel ---
            axG.plot(s0g[g0_ok], gw0[g0_ok], lw=2.2, color="tab:blue", label="gt (orig)")
            axG.plot(s1g[g1_ok], gw1[g1_ok], lw=2.6, ls="--", color="tab:orange", label="gt (resampled)")
            axG.axhline(gw0_mean, lw=1.6, alpha=0.35, color="tab:blue")
            axG.axhline(gw1_mean, lw=1.6, alpha=0.35, ls="--", color="tab:orange")
            axG.set_title("GT width vs arclength", fontsize=12)
            axG.set_xlabel("arclength s (px)", fontsize=11)
            axG.set_ylabel("width (px)", fontsize=11)
            axG.set_xlim(*xlim)
            axG.set_ylim(*ylim)
            axG.grid(True, alpha=0.25)
            axG.legend(loc="upper right", fontsize=10)
            axG.text(0.02, 0.02, "Horizontal lines = mean width",
                    transform=axG.transAxes, fontsize=10)

            # --- Pred panel ---
            axP.plot(s0p[p0_ok], pw0[p0_ok], lw=2.2, color="darkgreen", label="pred (orig)")
            axP.plot(s1p[p1_ok], pw1[p1_ok], lw=2.6, ls="--", color="red", label="pred (resampled)")
            axP.axhline(pw0_mean, lw=1.6, alpha=0.35, color="darkgreen")
            axP.axhline(pw1_mean, lw=1.6, alpha=0.35, ls="--", color="red")
            axP.set_title("Predicted width vs arclength", fontsize=12)
            axP.set_xlabel("arclength s (px)", fontsize=11)
            axP.set_ylabel("width (px)", fontsize=11)
            axP.grid(True, alpha=0.25)
            axP.legend(loc="upper right", fontsize=10)
            axP.text(0.02, 0.02, "Horizontal lines = mean width",
                    transform=axP.transAxes, fontsize=10)

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            fig.savefig(out_path, bbox_inches="tight", dpi=200)
            plt.close(fig)

            print(f"[STAGE6 SIGNAL] wrote: {out_path}")
        
        def stage6_plot_worst_and_all(
                *,
                worst_cid_runs,
                all_runs_global,
                mask_bin,
                crop_worst,
                crop_all,
                stage6_resample_dir,
                worst_cid,
            ):
            import os
            import numpy as np

            # ------------------------------------------------------------
            # WORST CID — ALL segments
            # ------------------------------------------------------------
            plot_sampling_consistency(
                pts_list=[r["pts"] for r in worst_cid_runs if r.get("pts") is not None],
                ptsr_list=[r["pts_rs"] for r in worst_cid_runs if r.get("pts_rs") is not None],
                mask_bin=mask_bin,
                crop=crop_worst,
                title=f"Stage 6 sampling consistency — WORST CID={worst_cid}",
                out_path=os.path.join(stage6_resample_dir, f"stage6_sampling_WORST_cid{worst_cid}.png"),
            )

            plot_width_error_distribution(
                runs=worst_cid_runs,
                title=f"Stage 6 width error distribution — WORST CID={worst_cid}",
                out_path=os.path.join(stage6_resample_dir, f"stage6_width_dist_WORST_cid{worst_cid}.png"),
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
                plot_stage6_width_signals_preservation(
                    run=worst_run,
                    title=f"Stage 6 worst-run width signals — cid={worst_cid}",
                    out_path=os.path.join(stage6_resample_dir, f"stage6_width_signals_WORST_cid{worst_cid}.png"),
                )

            # ------------------------------------------------------------
            # GLOBAL — ALL CIDs
            # ------------------------------------------------------------
            plot_sampling_consistency(
                pts_list=[r["pts"] for r in all_runs_global if r.get("pts") is not None],
                ptsr_list=[r["pts_rs"] for r in all_runs_global if r.get("pts_rs") is not None],
                mask_bin=mask_bin,
                crop=crop_all,
                title="Stage 6 sampling consistency — ALL CIDs",
                out_path=os.path.join(stage6_resample_dir, "stage6_sampling_ALL_CIDS.png"),
            )

            plot_width_error_distribution(
                runs=all_runs_global,
                title="Stage 6 width error distribution — ALL CIDs",
                out_path=os.path.join(stage6_resample_dir, "stage6_width_dist_ALL_CIDS.png"),
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
                plot_stage6_width_signals_preservation(
                    run=global_run,
                    title="Stage 6 width signals — ALL CIDs (representative)",
                    out_path=os.path.join(stage6_resample_dir, "stage6_width_signals_ALL_CIDS.png"),
                )
                                        
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
                    Lf   = float(r["finite_len_px"])
                    cov  = float(r.get("finite_len_frac", np.nan))
                    labels.append(f"cid {cid0}  (L={Lf:.0f}px, cov={cov:.2f})")
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
                    f"Stage 6 — Width error metrics (fair arclength sampling) — {mode} / {midline_type}\n"
                    f"Global length-weighted means: RMSE={global_rmse:.3f}px, MAE={global_mae:.3f}px, Bias={global_bias:.3f}px",
                    fontsize=11,
                    fontweight="bold",
                )

                out = os.path.join(stage6_metrics_dir, f"stage6_topK_width_metrics_{mode}_{midline_type}.png")
                fig.savefig(out, bbox_inches="tight", dpi=200)
                plt.close(fig)
                print(f"[STAGE6] wrote: {out}")

                # ------------------------------------------------------------
                # (B) Resampling explainers — corrected semantics
                # ------------------------------------------------------------

                rows_here_sorted = sorted(
                    rows_here,
                    key=lambda r: float(r.get("width_rmse_L", 0.0)),
                    reverse=True
                )
                if not rows_here_sorted:
                    raise RuntimeError("[STAGE6] no rows for resampling explainers")

                # ------------------------------------------------------------
                # Identify WORST CID (by RMSE)
                # ------------------------------------------------------------
                worst_row = rows_here_sorted[0]
                worst_cid = str(worst_row["crack_id"])
                worst_ct  = str(worst_row["crack_type"])
                worst_mt  = str(worst_row["midline_type"])

                def _cache_for_cid(cid0, ctype0, mtype0):
                    return [
                        it for it in stage6_cache
                        if str(it.get("cid","")) == str(cid0)
                        and str(it.get("crack_type","")) == str(ctype0)
                        and str(it.get("midline_type","")) == str(mtype0)
                        and it.get("runs")
                    ]

                # ------------------------------------------------------------
                # Collect runs for WORST CID
                # ------------------------------------------------------------
                items = _cache_for_cid(worst_cid, worst_ct, worst_mt)
                if not items:
                    raise RuntimeError(f"[STAGE6] no cache items for worst cid={worst_cid}")

                worst_cid_runs = []
                bbox0 = None
                for it in items:
                    bbox0 = bbox0 or it.get("bbox", None)
                    worst_cid_runs.extend(it["runs"])

                if not worst_cid_runs:
                    raise RuntimeError(f"[STAGE6] no runs for worst cid={worst_cid}")

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
                for it in stage6_cache:
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
                # (B1) COMBINED — AGGREGATED DIAGNOSTIC (UNCHANGED)
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
                            f"Stage 6 aggregated diagnostic — WORST CID={worst_cid} — combined/{worst_mt}",
                            fontsize=11,
                            fontweight="bold"
                        )

                        out = os.path.join(
                            stage6_resample_dir,
                            f"stage6_resample_aggregated_WORST_cid{worst_cid}_{worst_ct}_{worst_mt}.png"
                        )
                        fig.savefig(out, bbox_inches="tight", dpi=200)
                        plt.close(fig)
                        print(f"[STAGE6] wrote: {out}")

                # ------------------------------------------------------------
                # Collect GLOBAL ALL runs (ALL CIDs)
                # ------------------------------------------------------------
                all_runs_global = []
                for it in stage6_cache:
                    if it.get("runs"):
                        all_runs_global.extend(it["runs"])

                if not all_runs_global:
                    raise RuntimeError("[STAGE6] no global runs found")

                # ------------------------------------------------------------
                # (B2/B3) FINAL PLOTS
                # ------------------------------------------------------------
                stage6_plot_worst_and_all(
                    worst_cid_runs=worst_cid_runs,
                    all_runs_global=all_runs_global,
                    mask_bin=mask_bin,
                    crop_worst=crop_worst,
                    crop_all=crop_all,
                    stage6_resample_dir=stage6_resample_dir,
                    worst_cid=worst_cid,
                )


        except Exception as e:
            print(f"[STAGE6] plots skipped: {e}")

        # ------------------------------------------------------------
        # Swap final compare-width plotting inputs to Stage-6 resampled segments
        # ------------------------------------------------------------
        if coords_stage6 and diffs_stage6:
            coords = coords_stage6
            diffs  = diffs_stage6
            if bboxes_stage6:
                bboxes = bboxes_stage6
            print(f"[STAGE6] using resampled plotting inputs: {len(coords)} segs")
        else:
            print("[STAGE6] no resampled segs produced; keeping Stage-5 plotting inputs")

    except Exception as e:
        print(f"[STAGE6] skipped (fatal): {e}")


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

    out_dir = os.path.join(metrics_dir, midline_type or "unknown", crack_type)
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
            out_dir, crack_type,
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

    # --- Fréchet (optional but standard) ---
    try:
        if len(A_ds) >= 2 and len(B_ds) >= 2:
            out["frechet_discrete_ds"] = _unwrap(
                frechet_discrete_ds(A_ds, B_ds, max_points=800)
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