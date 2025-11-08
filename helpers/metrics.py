#!/usr/bin/env python3
import cracktools as ct
from helpers.crackhelpers import *

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

import matplotlib.pyplot as plt

import numpy as np
from math import hypot, atan2, pi
from skimage.morphology import skeletonize
import hashlib
import time

ROUNDING_DIGITS=6

'''def _json_has_manual_midlines(ann_path: str) -> bool:
    """Lightweight on-disk check: does JSON have any manual midline with ≥2 points?"""
    import json
    try:
        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except UnicodeDecodeError:
        # Fallback for any legacy files accidentally saved in cp1252, etc.
        with open(ann_path, "r", encoding="cp1252", errors="strict") as f:
            data = json.load(f)
    except Exception:
        return False
'''

def _json_has_manual_midlines(ann_path: str) -> bool:
    """Lightweight on-disk check: does JSON have any manual midline with ≥2 points?
       Also purges stale annotations that contain no valid atomic cracks.
    """
    import json, os
    try:
        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except UnicodeDecodeError:
        with open(ann_path, "r", encoding="cp1252", errors="strict") as f:
            data = json.load(f)
    except Exception:
        return False

    ann = (data or {}).get("annotations", {}) or {}
    atomic = ann.get("atomic_cracks", {}) or {}

    # --- purge safety: no atomic cracks or all midlines empty ---
    if not atomic or all(len(cr.get("midline", [])) < 2 for cr in atomic.values()):
        print(f"[global-metrics] 🧹 purging stale JSON with no manual midlines: {ann_path}")
        try:
            os.remove(ann_path)
        except Exception as e:
            print(f"[global-metrics] ⚠ failed to remove {ann_path}: {e}")
        return False

    # --- otherwise: check for ≥2-pt manual midlines ---
    for crack in atomic.values():
        src = (crack.get("source") or "").lower()
        if src.startswith("auto") or src == "combined":
            continue
        mid = crack.get("midline", [])
        if isinstance(mid, list) and len(mid) >= 2:
            return True
    return False

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

'''def _reconstruct_full_mask(crack, H, W):
    print(f"\n[DEBUG SUP]src={crack.get('src')}")

    mc = crack.get('mask_crop', None)
    bb = crack.get('mask_bbox', None)
    mid = crack.get('midline', [])
    print(f"  mask_crop type={type(mc)}, len={len(mc) if mc is not None else 'None'}")
    print(f"  mask_bbox={bb}")
    print(f"  midline len={len(mid)}")

    if len(mid) > 0:
        arr = np.array(mid, float)
        print(f"  midline x-range=({arr[:,0].min():.1f},{arr[:,0].max():.1f}), "
            f"y-range=({arr[:,1].min():.1f},{arr[:,1].max():.1f})")

    try:
        return reconstruct_full_mask_from_crack(crack, H, W)
    except Exception:
        mc = crack.get("mask_crop"); bb = crack.get("mask_bbox")
        if mc is not None and bb is not None:
            crop = np.array(mc, dtype=np.uint8)
            x, y, w, h = [int(v) for v in bb]
            x2, y2 = min(x+w, W), min(y+h, H)
            w_eff, h_eff = max(0, x2-x), max(0, y2-y)
            if h_eff > 0 and w_eff > 0:
                crop = (crop > 0).astype(np.uint8)[:h_eff, :w_eff]
                m = np.zeros((H, W), dtype=np.uint8)
                m[y:y+h_eff, x:x+w_eff] = crop
                return m
        full = np.array(crack.get("mask", []), dtype=np.uint8)
        if full.size == H*W and full.shape == (H, W):
            return (full > 0).astype(np.uint8)
        return np.zeros((H, W), dtype=np.uint8)'''
        
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
    
def normals_from_mask_for_midline(midline_xy, mask, max_radius=50):
    """
    Pixel-accurate version:
    - Polygonizes the mask into exact pixel-boundary polygons using rasterio.
    - Shifts coords by -0.5 so edges align with imshow pixel grid.
    - Intersects midline normals with those polygons so endpoints lie exactly on the mask edge.
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

    # tangent + normals
    try:
        from cracktools.segmentation import compute_smooth_tangent_normals
        _, nor = compute_smooth_tangent_normals(midline_xy[:, 0], midline_xy[:, 1])
    except Exception:
        dx, dy = np.gradient(midline_xy[:, 0]), np.gradient(midline_xy[:, 1])
        nrm = np.hypot(dx, dy) + 1e-12
        tan = np.stack([dx/nrm, dy/nrm], axis=1)
        nor = np.stack([-tan[:, 1], tan[:, 0]], axis=1)

    # polygonize mask -> shapely polygons
    mask_bin = (mask > 0).astype(np.uint8)
    polygons = []
    for geom, val in rasterio.features.shapes(mask_bin, mask=mask_bin):
        if val == 1:
            poly = shape(geom)
            # shift by -0.5 in both x and y
            poly = shapely.affinity.translate(poly, xoff=-0.5, yoff=-0.5)
            polygons.append(poly)
    edges = [poly.boundary for poly in polygons]

    N = len(midline_xy)
    e1x = np.full(N, np.nan); e1y = np.full(N, np.nan)
    e2x = np.full(N, np.nan); e2y = np.full(N, np.nan)
    widths_mask = np.full(N, np.nan)

    for i, (p, nvec) in enumerate(zip(midline_xy, nor)):
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(nvec)):
            continue

        # build long ray
        A = (p[0] - max_radius * nvec[0], p[1] - max_radius * nvec[1])
        B = (p[0] + max_radius * nvec[0], p[1] + max_radius * nvec[1])
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
                hits.append(tuple(coords[0])); hits.append(tuple(coords[-1]))

        if len(hits) >= 2:
            dists = [np.dot([hx - p[0], hy - p[1]], nvec) for (hx, hy) in hits]
            left_pts = [(hx, hy) for (hx, hy), d in zip(hits, dists) if d < 0]
            right_pts = [(hx, hy) for (hx, hy), d in zip(hits, dists) if d > 0]
            if left_pts and right_pts:
                lp = max(left_pts, key=lambda q: np.dot([q[0]-p[0], q[1]-p[1]], nvec))
                rp = min(right_pts, key=lambda q: np.dot([q[0]-p[0], q[1]-p[1]], nvec))
                e1x[i], e1y[i] = lp
                e2x[i], e2y[i] = rp
                widths_mask[i] = np.hypot(rp[0]-lp[0], rp[1]-lp[1])

    return (e1x, e1y, e2x, e2y, widths_mask), polygons

def plot_mask_normals(midline, e1x, e1y, e2x, e2y, mask, contours=None,
                    spacing_px=20, show=True, out_path=None, crack_label=""):
    """
    Plot normals + crack contours (polygons) for visualization.
    - contours: list of Shapely Polygons (from rasterio.features.shapes)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    H, W = mask.shape
    plt.figure(figsize=(8, 8))

    # Force 0 = black, 255 = white
    mask_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    mask_rgb[mask > 0] = [255, 255, 255]

    plt.imshow(mask_rgb, alpha=1.0)  # alpha=1 for full opaque b/w

    # plot polygon contours
    if contours:
        for poly in contours:
            if poly.is_empty:
                continue
            # in plot_mask_normals when drawing contours
            if poly.geom_type == "Polygon":
                x, y = poly.exterior.xy
                plt.plot(np.array(x), np.array(y),
                        color="orange", lw=1.5, alpha=0.8)
                for interior in poly.interiors:
                    xi, yi = interior.xy
                    plt.plot(np.array(xi), np.array(yi),
                            color="orange", lw=1.5, alpha=0.5)
            elif poly.geom_type == "MultiPolygon":
                for sub in poly.geoms:
                    x, y = sub.exterior.xy
                    plt.plot(x, y, color="orange", lw=1.0, alpha=0.8)

    # plot midline
    if midline is not None and len(midline) > 1:
        plt.plot(midline[:,0], midline[:,1], 'g-', lw=1.0, label="midline")

    # plot normals
    N = len(midline)
    for i in range(0, N, spacing_px):
        if np.isfinite(e1x[i]) and np.isfinite(e2x[i]):
            plt.plot([e1x[i], e2x[i]], [e1y[i], e2y[i]],
                    color="cyan", lw=0.5, alpha=0.8)
            plt.scatter([e1x[i], e2x[i]], [e1y[i], e2y[i]],
                        c=["red","blue"], s=8, marker="o", alpha=0.7)

    plt.title(f"Mask normals — {crack_label}")
    plt.axis("equal"); plt.legend(); plt.tight_layout()

    if show:
        plt.show()
    elif out_path:
        plt.savefig(out_path, dpi=200); plt.close()
        
def plot_width_differences(midline, w_mask, w_edge, mask, contours=None,
                        spacing_px=20, show=True, out_path=None, crack_label=""):
    """
    Visualize width differences along the midline:
    - Background mask (0=black, 255=white)
    - Midline (green)
    - Points colored by relative error (red=mask wider, blue=edge wider)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    H, W = mask.shape
    # force black/white background
    mask_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    mask_rgb[mask > 0] = [255, 255, 255]

    plt.figure(figsize=(8, 8))
    plt.imshow(mask_rgb, origin="upper")

    # plot contours if available
    if contours:
        for poly in contours:
            if poly.geom_type == "Polygon":
                x, y = poly.exterior.xy
                plt.plot(x, y, color="orange", lw=0.8, alpha=0.7)
                for interior in poly.interiors:
                    xi, yi = interior.xy
                    plt.plot(xi, yi, color="orange", lw=0.5, alpha=0.5)

    if midline is not None and len(midline) > 1:
        plt.plot(midline[:, 0], midline[:, 1], 'g-', lw=1.0, label="midline")

    # compute diffs
    valid = np.isfinite(w_mask) & np.isfinite(w_edge)
    diffs = w_edge - w_mask
    diffs = np.where(valid, diffs, np.nan)

    # color map: red (mask larger), blue (edge larger)
    colors = []
    for d in diffs:
        if np.isnan(d):
            colors.append("gray")
        elif d > 0:
            colors.append("blue")   # edge wider
        else:
            colors.append("red")    # mask wider

    # sample points along midline
    N = len(midline)
    for i in range(0, N, spacing_px):
        if np.isfinite(diffs[i]):
            plt.scatter(midline[i, 0], midline[i, 1],
                        c=colors[i], s=20, marker="o", alpha=0.8)

    plt.title(f"Width comparison — {crack_label}")
    plt.axis("equal"); plt.legend(); plt.tight_layout()

    if show:
        plt.show()
    elif out_path:
        plt.savefig(out_path, dpi=200)
        plt.close()

    return diffs

'''def compare_widths_for_cracks(ann, crack_mask, base_name, metrics_dir, display=True):
    """
    Compare mask-derived vs edge-tracking widths for all cracks.
    - Plots midlines color-coded by signed width difference (edge - mask).
    - Saves summary stats + per-point diffs.
    """
    import numpy as np, matplotlib.pyplot as plt, os, pandas as pd
    from matplotlib.collections import LineCollection

    H, W = crack_mask.shape
    width_rows = []
    diffs_rows = []

    atomic = ann.get("atomic_cracks", {}) or {}
    combined = ann.get("combined_cracks", {}) or {}

    # skip atomics already absorbed in combined
    atomics_in_combined = {m for cmb in combined.values() for m in cmb.get("members", [])}
    all_cracks = [("atomic", cid, crack) for cid, crack in atomic.items() if cid not in atomics_in_combined]
    all_cracks += [("combined", cid, crack) for cid, crack in combined.items()]

    for ctype, cid, crack in all_cracks:
        midline = np.asarray(crack.get("midline", []), float)
        if midline.ndim != 2 or midline.shape[1] != 2 or len(midline) < 3:
            continue

        # mask-based widths
        (_, _, _, _, w_mask), _ = normals_from_mask_for_midline(
            midline, crack_mask, max_radius=50
        )

        # edge-tracking widths
        ne = crack.get("normal_edge_points")
        w_edge = None
        if ne and isinstance(ne, dict):
            def _to_array(v):
                if isinstance(v, list) and len(v) == 2 and isinstance(v[0], (list, tuple)):
                    return np.column_stack([v[0], v[1]]).astype(float)
                return np.array(v, float)
            e1 = _to_array(ne.get("edge1", []))
            e2 = _to_array(ne.get("edge2", []))
            if e1.ndim == 2 and e2.ndim == 2 and len(e1) and len(e2):
                m = min(len(e1), len(e2), len(w_mask), len(midline))
                w_edge = np.full(m, np.nan)
                for i in range(m):
                    if np.all(np.isfinite(e1[i])) and np.all(np.isfinite(e2[i])):
                        w_edge[i] = np.hypot(e1[i,0] - e2[i,0], e1[i,1] - e2[i,1])
                # trim everything consistently
                w_mask = w_mask[:m]
                midline = midline[:m]

        if w_edge is None:
            continue

        valid = np.isfinite(w_mask) & np.isfinite(w_edge)
        n_valid = int(valid.sum())
        if n_valid < 3:
            continue

        diff = w_edge[valid] - w_mask[valid]
        coords = midline[valid]

        # --- add stats row
        width_rows.append({
            "image": base_name, "crack_type": ctype, "crack_id": cid,
            "n_valid": n_valid,
            "mask_width_mean": float(np.mean(w_mask[valid])),
            "edge_width_mean": float(np.mean(w_edge[valid])),
            "width_diff_mae": float(np.mean(np.abs(diff))),
            "width_diff_rmse": float(np.sqrt(np.mean(diff**2))),
            "width_diff_mean": float(np.mean(diff)),
            "width_diff_std": float(np.std(diff)),
            "width_diff_min": float(np.min(diff)),
            "width_diff_max": float(np.max(diff))
        })

        # --- save raw diffs
        diffs_out = os.path.join(metrics_dir, f"{base_name}_{ctype}{cid}_width_diffs.csv")
        pd.DataFrame({
            "mid_x": coords[:,0], "mid_y": coords[:,1],
            "mask_width": w_mask[valid],
            "edge_width": w_edge[valid],
            "width_diff": diff
        }).to_csv(diffs_out, index=False)

        from matplotlib.colors import TwoSlopeNorm

        # --- plot crack with midline color-coded
        if len(coords) > 1:
            segments = np.stack([coords[:-1], coords[1:]], axis=1)

            # get actual min/max
            vmin = np.min(diff)
            vmax = np.max(diff)

            # symmetric scale around 0 so colors are proportional
            max_abs = max(abs(vmin), abs(vmax))
            norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=max_abs)

            mask_rgb = np.zeros((H, W, 3), dtype=np.uint8)
            mask_rgb[crack_mask > 0] = [255, 255, 255]

            plt.figure(figsize=(8, 8))
            plt.imshow(mask_rgb, origin="upper")

            lc = LineCollection(
                segments, cmap="coolwarm", norm=norm,
                linewidth=3.0, alpha=0.9
            )
            lc.set_array(diff[:-1])  # color from diffs
            plt.gca().add_collection(lc)

            # colorbar with explicit min/max ticks
            cbar = plt.colorbar(lc, ax=plt.gca(), shrink=0.7)
            cbar.set_label("Width difference (edge - mask) [px]")
            cbar.set_ticks([vmin, 0, vmax])
            cbar.ax.set_yticklabels([f"{vmin:.2f}", "0", f"{vmax:.2f}"])

            plt.title(f"Width diffs — {ctype} {cid}")
            plt.axis("equal"); plt.tight_layout()

            out_plot = os.path.join(metrics_dir, f"{base_name}_{ctype}{cid}_width_diffs.png")
            if display: 
                plt.show()
            else: 
                plt.savefig(out_plot, dpi=200); plt.close()

    return width_rows, diffs_rows'''
    
'''def compare_widths_for_cracks(ann, crack_mask, base_name, metrics_dir, display=True, tag=None, **kwargs):
    """
    Compare mask-derived vs edge-tracking widths for all cracks.
    - Plots midlines color-coded by signed width difference (edge - mask).
    - Saves summary stats + per-point diffs.
    - Optional `tag` argument lets caller distinguish manual vs auto outputs.
    """
    import numpy as np, matplotlib.pyplot as plt, os, pandas as pd
    from matplotlib.collections import LineCollection
    from matplotlib.colors import TwoSlopeNorm

    H, W = crack_mask.shape
    width_rows = []
    diffs_rows = []

    atomic = ann.get("atomic_cracks", {}) or {}
    combined = ann.get("combined_cracks", {}) or {}

    # skip atomics already absorbed in combined
    atomics_in_combined = {m for cmb in combined.values() for m in cmb.get("members", [])}
    all_cracks = [("atomic", cid, crack) for cid, crack in atomic.items() if cid not in atomics_in_combined]
    all_cracks += [("combined", cid, crack) for cid, crack in combined.items()]

    for ctype, cid, crack in all_cracks:
        midline = np.asarray(crack.get("midline", []), float)
        if midline.ndim != 2 or midline.shape[1] != 2 or len(midline) < 3:
            continue

        # mask-based widths
        (_, _, _, _, w_mask), _ = normals_from_mask_for_midline(
            midline, crack_mask, max_radius=50
        )

        # edge-tracking widths
        ne = crack.get("normal_edge_points")
        w_edge = None
        if ne and isinstance(ne, dict):
            def _to_array(v):
                if isinstance(v, list) and len(v) == 2 and isinstance(v[0], (list, tuple)):
                    return np.column_stack([v[0], v[1]]).astype(float)
                return np.array(v, float)
            e1 = _to_array(ne.get("edge1", []))
            e2 = _to_array(ne.get("edge2", []))
            if e1.ndim == 2 and e2.ndim == 2 and len(e1) and len(e2):
                m = min(len(e1), len(e2), len(w_mask), len(midline))
                w_edge = np.full(m, np.nan)
                for i in range(m):
                    if np.all(np.isfinite(e1[i])) and np.all(np.isfinite(e2[i])):
                        w_edge[i] = np.hypot(e1[i, 0] - e2[i, 0], e1[i, 1] - e2[i, 1])
                w_mask = w_mask[:m]
                midline = midline[:m]

        if w_edge is None:
            continue

        valid = np.isfinite(w_mask) & np.isfinite(w_edge)
        n_valid = int(valid.sum())
        if n_valid < 3:
            continue

        diff = w_edge[valid] - w_mask[valid]
        coords = midline[valid]

        # --- add stats row
        width_rows.append({
            "image": base_name, "crack_type": ctype, "crack_id": cid,
            "n_valid": n_valid,
            "mask_width_mean": float(np.mean(w_mask[valid])),
            "edge_width_mean": float(np.mean(w_edge[valid])),
            "width_diff_mae": float(np.mean(np.abs(diff))),
            "width_diff_rmse": float(np.sqrt(np.mean(diff ** 2))),
            "width_diff_mean": float(np.mean(diff)),
            "width_diff_std": float(np.std(diff)),
            "width_diff_min": float(np.min(diff)),
            "width_diff_max": float(np.max(diff))
        })

        # --- save raw diffs
        suffix = f"_{tag}" if tag else ""
        diffs_out = os.path.join(metrics_dir, f"{base_name}_{ctype}{cid}{suffix}_width_diffs.csv")
        pd.DataFrame({
            "mid_x": coords[:, 0],
            "mid_y": coords[:, 1],
            "mask_width": w_mask[valid],
            "edge_width": w_edge[valid],
            "width_diff": diff
        }).to_csv(diffs_out, index=False)

        # --- plot crack with midline color-coded
        if len(coords) > 1:
            segments = np.stack([coords[:-1], coords[1:]], axis=1)
            vmin, vmax = np.min(diff), np.max(diff)
            max_abs = max(abs(vmin), abs(vmax))
            norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=max_abs)

            mask_rgb = np.zeros((H, W, 3), dtype=np.uint8)
            mask_rgb[crack_mask > 0] = [255, 255, 255]

            plt.figure(figsize=(8, 8))
            plt.imshow(mask_rgb, origin="upper")

            lc = LineCollection(segments, cmap="coolwarm", norm=norm, linewidth=3.0, alpha=0.9)
            lc.set_array(diff[:-1])
            plt.gca().add_collection(lc)

            cbar = plt.colorbar(lc, ax=plt.gca(), shrink=0.7)
            cbar.set_label("Width difference (edge - mask) [px]")
            cbar.set_ticks([vmin, 0, vmax])
            cbar.ax.set_yticklabels([f"{vmin:.2f}", "0", f"{vmax:.2f}"])

            tag_title = f" ({tag})" if tag else ""
            plt.title(f"Width diffs — {ctype} {cid}{tag_title}")
            plt.axis("equal"); plt.tight_layout()

            out_plot = os.path.join(metrics_dir, f"{base_name}_{ctype}{cid}{suffix}_width_diffs.png")
            if display:
                plt.show()
            else:
                plt.savefig(out_plot, dpi=200)
                plt.close()

    return width_rows, diffs_rows'''
    
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
    ann, crack_mask, base_name, metrics_dir,
    display=False, tag=None,
    return_normals=False,
    normals_plot=False,
    normals_dir=None,
    max_radius=50
):
    """
    One-figure per category (atomic_all, combined_all) width-diff visualization.
    Aggregates all subcracks into a single plot per category, plus stats CSVs.

    Files written (if data exists):
      <base>_atomic_all[_tag]_width_diffs_global.png
      <base>_atomic_all[_tag]_width_diffs_zoom.png
      <base>_combined_all[_tag]_width_diffs_global.png
      <base>_combined_all[_tag]_width_diffs_zoom.png
    """
    import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import TwoSlopeNorm

    os.makedirs(metrics_dir, exist_ok=True)
    if normals_plot and normals_dir:
        os.makedirs(normals_dir, exist_ok=True)

    H, W = crack_mask.shape
    mask_bin = (crack_mask > 0).astype(np.uint8)

    atomic = ann.get("atomic_cracks", {}) or {}
    combined = ann.get("combined_cracks", {}) or {}
    # atomics that are NOT members of any combined
    atomics_in_combined = {str(m) for cmb in combined.values() for m in (cmb.get("members", []) or [])}

    # containers for aggregated plotting
    agg = {
        "atomic":  {"coords": [], "diff": []},
        "combined": {"coords": [], "diff": []},
    }

    width_rows, diffs_rows = [], []
    normals_dict = {} if return_normals else None

    def _to_xy(v):
        # supports {"edge1":[xs,ys]} or [[x,y],...]
        import numpy as np
        if isinstance(v, list) and len(v) == 2 and isinstance(v[0], (list, tuple, np.ndarray)):
            return np.column_stack([v[0], v[1]]).astype(float)
        return np.asarray(v, float)

    # iterate atomic then combined in one loop
    all_cracks = [("atomic", str(cid), c) for cid, c in atomic.items() if str(cid) not in atomics_in_combined]
    all_cracks += [("combined", str(cid), c) for cid, c in combined.items()]

    for ctype, cid, crack in all_cracks:
        midline = np.asarray(crack.get("midline", []), float)
        if midline.ndim != 2 or midline.shape[1] != 2 or len(midline) < 3:
            continue

        # ----- GT normals/widths -----
        key = _mask_midline_cache_key(mask_bin, midline)
        if key in _NORMALS_CACHE:
            e1x, e1y, e2x, e2y, w_mask = _NORMALS_CACHE[key]
        else:
            (e1x, e1y, e2x, e2y, w_mask), _ = normals_from_mask_for_midline(midline, mask_bin, max_radius=max_radius)
            _NORMALS_CACHE[key] = (e1x, e1y, e2x, e2y, w_mask)

        gt_e1 = np.column_stack([e1x, e1y]); gt_e2 = np.column_stack([e2x, e2y])

        # ----- Edge normals / widths (read only; no recompute) -----
        ne = (
            crack.get("normal_edge_points_full")
            or crack.get("normal_edge_points_clipped")
            or crack.get("normal_edge_points")
        )
        e1 = e2 = np.empty((0, 2), float)
        if isinstance(ne, dict):
            e1 = _to_xy(ne.get("edge1", [])); e2 = _to_xy(ne.get("edge2", []))
        elif isinstance(ne, (list, tuple)) and len(ne) == 2:
            e1, e2 = _to_xy(ne[0]), _to_xy(ne[1])

        m = min(len(e1), len(e2), len(w_mask), len(midline))
        if m < 3:
            # still return GT normals if requested
            if return_normals:
                normals_dict[(ctype, cid)] = {
                    "midline": midline.astype(np.float32),
                    "mask_width": np.asarray(w_mask, np.float32),
                    "gt_e1": gt_e1.astype(np.float32),
                    "gt_e2": gt_e2.astype(np.float32),
                    "edge_e1": None, "edge_e2": None,
                }
            continue

        e1, e2 = e1[:m], e2[:m]
        w_edge = np.hypot(e1[:, 0] - e2[:, 0], e1[:, 1] - e2[:, 1])
        w_mask = w_mask[:m]
        coords = midline[:m]
        valid = np.isfinite(w_mask) & np.isfinite(w_edge)
        if valid.sum() < 3:
            continue

        diff = w_edge[valid] - w_mask[valid]
        coords = coords[valid]

        # ---- per-crack stats (CSV rows) ----
        width_rows.append({
            "image": base_name, "crack_type": ctype, "crack_id": cid,
            "n_valid": int(valid.sum()),
            "mask_width_mean": float(np.mean(w_mask[valid])),
            "edge_width_mean": float(np.mean(w_edge[valid])),
            "width_diff_mae": float(np.mean(np.abs(diff))),
            "width_diff_rmse": float(np.sqrt(np.mean(diff ** 2))),
            "width_diff_mean": float(np.mean(diff)),
            "width_diff_std": float(np.std(diff)),
            "width_diff_min": float(np.min(diff)),
            "width_diff_max": float(np.max(diff))
        })

        # ---- aggregate for one-figure plot ----
        agg[ctype]["coords"].append(coords)
        agg[ctype]["diff"].append(diff)

        # normals payload
        if return_normals:
            normals_dict[(ctype, cid)] = {
                "midline": coords.astype(np.float32),
                "mask_width": w_mask[valid].astype(np.float32),
                "gt_e1": gt_e1[:m][valid].astype(np.float32),
                "gt_e2": gt_e2[:m][valid].astype(np.float32),
                "edge_e1": e1.astype(np.float32),
                "edge_e2": e2.astype(np.float32),
            }

    # ---- helper to render one aggregated figure (global + zoom) ----
    def _render_all(label):
        if not agg[label]["coords"]:
            return
        # concat
        C = np.concatenate(agg[label]["coords"], axis=0)
        D = np.concatenate(agg[label]["diff"], axis=0)

        vmin, vmax = float(np.min(D)), float(np.max(D))
        max_abs = max(abs(vmin), abs(vmax))
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=max_abs)

        # segments along midline
        if len(C) <= 1:
            return
        segs = np.stack([C[:-1], C[1:]], axis=1)

        mask_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        mask_rgb[mask_bin > 0] = [255, 255, 255]
        suffix = f"_{tag}" if tag else ""

        # GLOBAL
        fig, ax = plt.subplots(figsize=(12, 12), dpi=250)
        ax.imshow(mask_rgb, origin="upper")
        lc = LineCollection(segs, cmap="coolwarm", norm=norm, linewidth=2.0, alpha=0.95)
        lc.set_array(D[:-1])
        ax.add_collection(lc)
        cbar = plt.colorbar(lc, ax=ax, shrink=0.75, pad=0.02)
        cbar.set_label("Width difference (edge - mask) [px]", fontsize=12)
        ax.set_title(f"Width diffs — {label} (all)", fontsize=14)
        ax.axis("equal"); plt.tight_layout()
        out_g = os.path.join(metrics_dir, f"{base_name}_{label}_all{suffix}_width_diffs_global.png")
        plt.savefig(out_g, dpi=250); plt.close()

        # ZOOM : tight bounds around all coords
        x0, x1 = np.clip([np.min(C[:, 0]) - 20, np.max(C[:, 0]) + 20], 0, W)
        y0, y1 = np.clip([np.min(C[:, 1]) - 20, np.max(C[:, 1]) + 20], 0, H)

        fig, ax = plt.subplots(figsize=(8, 8), dpi=350)
        ax.imshow(mask_rgb, origin="upper")
        lc = LineCollection(segs, cmap="coolwarm", norm=norm, linewidth=3.0, alpha=0.95)
        lc.set_array(D[:-1])
        ax.add_collection(lc)
        cbar = plt.colorbar(lc, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("Width difference (edge - mask) [px]", fontsize=10)
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
        ax.set_title(f"Width diffs — {label} (zoom)", fontsize=11)
        ax.axis("equal"); plt.tight_layout()
        out_z = os.path.join(metrics_dir, f"{base_name}_{label}_all{suffix}_width_diffs_zoom.png")
        plt.savefig(out_z, dpi=350); plt.close()

        print(f"[DEBUG VIS] wrote → {out_g} and {out_z}")

    _render_all("atomic")
    _render_all("combined")

    if return_normals:
        return width_rows, diffs_rows, normals_dict
    return width_rows, diffs_rows

###############################################################################################
# Midline Metrics
###############################################################################################
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

# ---------- NEW helpers ----------

def _resample_xy_by_arclen(xy, N=400):
    xy = np.asarray(xy, float)
    if len(xy) < 2: return xy
    d = np.hypot(np.diff(xy[:,0]), np.diff(xy[:,1]))
    s = np.concatenate([[0], np.cumsum(d)])
    if s[-1] <= 0: return xy
    t = np.linspace(0, s[-1], min(N, len(xy)))
    x = np.interp(t, s, xy[:,0]); y = np.interp(t, s, xy[:,1])
    return np.column_stack([x, y])


def _rms_curvature(xy):
    import numpy as np
    xy = np.asarray(xy, float)
    n = len(xy)
    if n < 3:
        return float('nan')

    # arc-length parameterization
    ds = np.hypot(np.diff(xy[:,0]), np.diff(xy[:,1]))
    s = np.concatenate([[0], np.cumsum(ds)])   # len = n
    if s[-1] <= 0:
        return float('nan')

    dx = np.gradient(xy[:,0], s, edge_order=2)
    dy = np.gradient(xy[:,1], s, edge_order=2)
    ddx = np.gradient(dx, s, edge_order=2)
    ddy = np.gradient(dy, s, edge_order=2)

    num = np.abs(dx*ddy - dy*ddx)
    den = (dx*dx + dy*dy)**1.5 + 1e-12
    kappa = num / den
    return float(np.sqrt(np.nanmean(kappa**2)))


def _orth_stats(orth_dev_arr):
    """
    Accepts either a NumPy array of orthogonal deviations or a dict
    returned by orthogonal_deviation(). Extracts numeric values
    robustly and returns summary stats including directional_bias.
    """
    import numpy as np

    # unwrap dict form (e.g. {"orth_dev": [...]} or {"array": [...]} etc.)
    if isinstance(orth_dev_arr, dict):
        # try common keys
        for k in ("orth_dev", "values", "array", "data"):
            if k in orth_dev_arr:
                orth_dev_arr = orth_dev_arr[k]
                break
        # if dict values are numeric scalars, flatten them
        if isinstance(orth_dev_arr, dict):
            orth_dev_arr = list(orth_dev_arr.values())

    # convert to array
    a = np.asarray(orth_dev_arr, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return dict(
            orth_mean=np.nan,
            orth_mean_abs=np.nan,
            orth_std=np.nan,
            directional_bias=np.nan
        )

    mu = float(np.mean(a))
    sd = float(np.std(a) + 1e-12)
    return {
        "orth_mean": mu,
        "orth_mean_abs": float(np.mean(np.abs(a))),
        "orth_std": sd,
        # signed normalized bias
        "directional_bias": float(np.sign(mu) * (abs(mu)/sd))
    }


def _split_bins_by_arclen(xy, n_bins=5):
    xy = np.asarray(xy, float)
    if len(xy) < 2: return [xy]
    d = np.hypot(np.diff(xy[:,0]), np.diff(xy[:,1]))
    s = np.concatenate([[0], np.cumsum(d)])
    edges = np.linspace(0, s[-1], n_bins+1)
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        idx = np.where((s >= lo) & (s <= hi))[0]
        if len(idx) >= 2:
            bins.append(xy[idx])
        else:
            j = np.searchsorted(s, (lo+hi)/2)
            j0 = max(0, j-1); j1 = min(len(xy)-1, j+1)
            if j1 > j0:
                bins.append(xy[j0:j1+1])
    return [b for b in bins if len(b) >= 2]


def _widths_from_normal_pairs(normals):
    """
    normals (crop or full coords): [[e1x,e1y],[e2x,e2y]]
    returns width array (NaN where missing)
    """
    if normals is None: return np.array([])
    (e1x, e1y), (e2x, e2y) = normals
    e1 = np.column_stack([np.asarray(e1x,float), np.asarray(e1y,float)])
    e2 = np.column_stack([np.asarray(e2x,float), np.asarray(e2y,float)])
    ok = np.isfinite(e1).all(axis=1) & np.isfinite(e2).all(axis=1)
    w = np.full(len(e1), np.nan)
    if ok.any():
        w[ok] = np.hypot(e1[ok,0]-e2[ok,0], e1[ok,1]-e2[ok,1])
    return w


def _pearson_nan(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = min(a.size, b.size)
    if n == 0: return float('nan')
    a = a[:n]; b = b[:n]
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3: return float('nan')
    a = (a[ok] - a[ok].mean())/(a[ok].std()+1e-12)
    b = (b[ok] - b[ok].mean())/(b[ok].std()+1e-12)
    return float(np.mean(a*b))


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

def debug_plot_midlines(self, crack_id, cache_key=None, show_gt=True, save_path=None):
    """
    Plot overlay of manual vs auto midline (and GT mask if available).
    """
    ann = self.annotation.get("annotations", {})
    atomic = ann.get("atomic_cracks", {})
    crack = atomic.get(str(crack_id))
    if crack is None:
        print(f"⚠️ No crack {crack_id} found")
        return

    cache_key = cache_key or _auto_cache_key(self)
    auto_var = crack.get("variants", {}).get("auto", {}).get(cache_key)

    man_xy = np.array(crack.get("midline", []), float)
    auto_xy = np.array(auto_var["midline"], float) if auto_var else np.empty((0,2))

    # Start with image
    im = self.original_image.copy()

    # Optional GT mask outline
    if show_gt and getattr(self, "current_mask", None) is not None:
        contours, _ = cv2.findContours(self.current_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, contours, -1, (0,0,255), 1)  # blue outline

    # Plot with matplotlib
    plt.figure(figsize=(8,8))
    plt.imshow(im)
    if len(man_xy) > 1:
        plt.plot(man_xy[:,0], man_xy[:,1], 'g-', linewidth=2, label="Manual (GT midline)")
    if len(auto_xy) > 1:
        plt.plot(auto_xy[:,0], auto_xy[:,1], 'r-', linewidth=2, label="Auto midline")
    plt.legend()
    plt.title(f"Crack {crack_id} - cache_key={cache_key}")
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✅ Saved debug plot → {save_path}")
    else:
        plt.show()

# ---------------- JSON helpers (static) ----------------

def _to_py(obj, ndigits=6):
    """
    Recursively converts NumPy / CuPy arrays, pandas, etc. to plain
    Python lists, rounding floats to `ndigits` decimals for compact JSON.
    """
    import numpy as np

    if obj is None:
        return None
    if isinstance(obj, (int, bool, str)):
        return obj
    if isinstance(obj, float):
        # round floats directly
        return round(obj, ndigits)
    if isinstance(obj, (list, tuple)):
        return [_to_py(x, ndigits) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_py(v, ndigits) for k, v in obj.items()}
    if hasattr(obj, "tolist"):  # numpy / cupy array
        return _to_py(obj.tolist(), ndigits)
    return obj


def safe_json_dump(data, path, compact=True):
    """Atomic JSON writer — supports compact (semi-human) or fully minified mode."""
    import os, tempfile, json
    d = _to_py(data)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path)+".", suffix=".tmp",
                            dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            if compact:
                # minimal spacing, arrays inline
                json.dump(d, f, ensure_ascii=False, indent=None, separators=(',', ':'))
            else:
                # readable multi-line, smaller indent
                json.dump(d, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try: os.remove(tmp)
        except Exception: pass
        raise


def _normals_to_json(normals, xmin, ymin, ndigits=ROUNDING_DIGITS):
    import numpy as np

    def to_xy2(arr):
        a = np.asarray(arr, float)
        if a.ndim == 2 and a.shape[1] == 2:       # already Nx2
            x, y = a[:, 0], a[:, 1]
        elif a.ndim == 2 and a.shape[0] == 2:     # 2xN ( [xlist, ylist] )
            x, y = a[0], a[1]
        elif a.ndim == 1:                         # degenerate 1-D
            x, y = a, np.full_like(a, np.nan, dtype=float)
        else:
            x = y = np.array([], float)

        x = x + float(xmin)
        y = y + float(ymin)

        out = np.stack([x, y], axis=1)
        # round
        if np.isfinite(out).any():
            out = np.round(out, ndigits=ndigits, where=np.isfinite(out))
        # JSON-safe NaNs
        out[~np.isfinite(out)] = None
        return out.tolist()

    # dict form: {"edge1":[xlist,ylist], "edge2":[xlist,ylist]} or {"edge1":Nx2,...}
    if isinstance(normals, dict):
        e1 = normals.get("edge1", [])
        e2 = normals.get("edge2", [])
        # accept either [xlist,ylist] or Nx2
        e1 = e1 if isinstance(e1, (list, tuple)) else []
        e2 = e2 if isinstance(e2, (list, tuple)) else []
        e1 = to_xy2(e1)
        e2 = to_xy2(e2)
        return {"edge1": e1, "edge2": e2}

    # tuple/list form: ((e1x,e1y), (e2x,e2y))
    try:
        (e1x, e1y), (e2x, e2y) = normals
        return {"edge1": to_xy2([e1x, e1y]), "edge2": to_xy2([e2x, e2y])}
    except Exception:
        return {"edge1": [], "edge2": []}


##########################################################
# Helpers
###########################################################
# ======================== metrics.py (snapshot helpers) ========================
import os, json, hashlib, math
import numpy as np

# --- small filesystem helpers -------------------------------------------------
def _ensure_dir(p): os.makedirs(p, exist_ok=True); return p

def metric_snapshot_root(save_folder, image_base):
    """metrics/<image>/snapshot/"""
    return _ensure_dir(os.path.join(save_folder, "metrics", image_base, "snapshot"))

def metric_atomic_dir(save_folder, image_base):
    """metrics/<image>/snapshot/atomic/"""
    return _ensure_dir(os.path.join(metric_snapshot_root(save_folder, image_base), "atomic"))

'''def metric_atomic_path_for(save_folder, image_base, crack_id):
    return os.path.join(metric_atomic_dir(save_folder, image_base), f"cid{crack_id}_metrics.json")'''
    
# -------------------- SNAPSHOT PATHS --------------------
def metric_image_dir(save_folder, base_name):
    return os.path.join(save_folder, "metrics", base_name)

def metric_atomic_path_for(save_folder, base_name, crack_id):
    return os.path.join(save_folder, "metrics", base_name, f"cid{crack_id}.json")

def metric_combined_path(save_folder, image_base):
    return os.path.join(metric_snapshot_root(save_folder, image_base), "combined.json")

def safe_read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

# in helpers/metrics.py
from helpers.metrics import _to_py   # if not already in the same file

def safe_write_json(path, data):
    """Drop-in replacement that auto-converts NumPy arrays."""
    import json, os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_to_py(data), f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

# --- snapshot assembly / persistence ------------------------------------------
def snapshot_pick_crack_fields(cr):
    """Keep only fields metrics care about (read-only in snapshot)."""
    keep = {}
    for k in ("source","midline","mask_bbox","geodesic_edges",
              "normal_edge_points_full","normal_edge_points","mask_crop"):
        if k in cr: keep[k] = cr[k]
    # do NOT keep bulky 'variants' tree in authoring snapshot; we will store per-crack auto artifacts separately
    return keep

def snapshot_from_authoring(authoring_ann, cache_key=None):
    """
    Build an in-memory snapshot dict *without touching disk*.
    You will later persist per-crack files with split_snapshot_to_files().
    authoring_ann: {'atomic_cracks':{...}, 'combined_cracks':{...}}  (the .get('annotations', {}) block)
    """
    atomic_src   = (authoring_ann or {}).get("atomic_cracks", {}) or {}
    combined_src = (authoring_ann or {}).get("combined_cracks", {}) or {}

    atomic = { cid: snapshot_pick_crack_fields(cr) for cid, cr in atomic_src.items() }
    combined = {}
    for k, cmb in combined_src.items():
        cc = {}
        for fld in ("members","midline","mask_bbox","normal_edge_points_full","normal_edge_points"):
            if fld in cmb: cc[fld] = cmb[fld]
        # store optional 'auto' minimal fields, if present on authoring (not required)
        if "auto" in cmb and isinstance(cmb["auto"], dict):
            a = {}
            for fld in ("midline","mask_bbox","normal_edge_points_full","normal_edge_points"):
                if fld in cmb["auto"]: a[fld] = cmb["auto"][fld]
            cc["auto"] = a
        combined[k] = cc

    # auto_best is not taken from authoring variants here; it’s populated later from per-crack files
    return {"atomic_cracks": atomic, "combined_cracks": combined, "auto_best_atomic_cracks": {}}

def split_snapshot_to_files(snapshot, save_folder, image_base, merge_if_exists=True):
    """
    Persist per-crack snapshot: one JSON per crack + one combined.json.
    If merge_if_exists=True, keep previously computed fields (e.g., tracked edges, auto_best) already on disk.
    """
    atomic = snapshot.get("atomic_cracks", {}) or {}
    for cid, cr in atomic.items():
        p = metric_atomic_path_for(save_folder, image_base, cid)
        if merge_if_exists:
            old = safe_read_json(p, {})
            # merge old computed stuff (e.g., autotrack, auto_best) into new minimal authoring view
            for k in ("geodesic_edges","normal_edge_points_full","normal_edge_points",
                      "mask_crop","auto_best"):
                if k in old and k not in cr:
                    cr[k] = old[k]
        safe_write_json(p, cr)
        #safe_json_dump(cnew, cpath)

    # combined
    cpath = metric_combined_path(save_folder, image_base)
    cold  = safe_read_json(cpath, {}) if merge_if_exists else {}
    cnew  = snapshot.get("combined_cracks", {}) or {}
    # merge optional 'auto' sub if it already lived on disk
    for k, v in (cold or {}).items():
        if k in cnew and isinstance(v, dict) and "auto" in v and "auto" not in cnew[k]:
            cnew[k]["auto"] = v["auto"]
    safe_write_json(cpath, cnew)
    #safe_json_dump(cnew, cpath)

'''def load_snapshot_from_files(save_folder, image_base):
    """Reassemble an in-memory snapshot dict by reading per-crack JSON + combined.json files."""
    # atomic
    adir = metric_atomic_dir(save_folder, image_base)
    atomic = {}
    try:
        for fn in os.listdir(adir):
            if not fn.endswith("_metrics.json") or not fn.startswith("cid"): continue
            cid = fn[len("cid"):-len("_metrics.json")]
            atomic[cid] = safe_read_json(os.path.join(adir, fn), {}) or {}
    except Exception:
        pass

    # combined
    combined = safe_read_json(metric_combined_path(save_folder, image_base), {}) or {}

    # inject auto_best view (if per-crack 'auto_best' exists)
    auto_best = {}
    for cid, cr in atomic.items():
        if isinstance(cr.get("auto_best"), dict):
            auto_best[cid] = cr["auto_best"]

    return {"atomic_cracks": atomic, "combined_cracks": combined, "auto_best_atomic_cracks": auto_best}'''
    
def load_snapshot_from_files(save_folder, base_name):
    """Load all cid*.json under metrics/<base> into a dict."""
    root = metric_image_dir(save_folder, base_name)
    out = {"atomic_cracks": {}, "combined_cracks": {}, "auto_best": {}}
    if not os.path.isdir(root):
        return out
    for fn in os.listdir(root):
        if not fn.startswith("cid") or not fn.endswith(".json"): 
            continue
        p = os.path.join(root, fn)
        rec = safe_read_json(p, {})
        if not rec: 
            continue
        cid = rec.get("crack_id")
        if cid is None:
            # fallback from filename
            try: cid = int(fn[3:-5])
            except: continue
        out["atomic_cracks"][cid] = rec
        if "auto_best" in rec and rec["auto_best"]:
            out["auto_best"][cid] = rec["auto_best"]
    return out

def snapshot_fingerprint(snapshot):
    j = json.dumps(snapshot or {}, sort_keys=True, separators=(",",":"))
    return hashlib.sha1(j.encode("utf-8")).hexdigest()

# --- auto-best helpers in per-crack files -------------------------------------
'''def set_auto_variant_for_crack(save_folder, image_base, crack_id, variant_record, params=None, is_best=False):
    """
    Store one auto variant into the per-crack file; if is_best=True also update 'auto_best'.
    variant_record must at least contain {"midline":[[x,y],...]} and may include
    "normal_edge_points_full"/"normal_edge_points".
    """
    p = metric_atomic_path_for(save_folder, image_base, crack_id)
    rec = safe_read_json(p, {}) or {}
    # store whole variant list under 'auto_variants' list (optional)
    av = rec.setdefault("auto_variants", [])
    vstore = {"midline": variant_record.get("midline", [])}
    for k in ("normal_edge_points_full","normal_edge_points","mask_bbox","params"):
        if k in variant_record:
            vstore[k] = variant_record[k]
    if params and "params" not in vstore:
        vstore["params"] = params
    av.append(vstore)

    if is_best:
        rec["auto_best"] = vstore  # compact best copy

    safe_write_json(p, rec)'''
    
def _flatten_variant_record(vid: int, vrec: dict, params: dict, scores: dict = None):
    """
    Keep bottom-level clean and single-line-ish:
    {
      "variant_id": 2, "midline": [[...],[...]], 
      "params": {"g11":1,"g22":35,"g33":25,"win":45,"mu":0,"ell":5,"p":14},
      "scores": {"chamfer_mean":..., "hausdorff":..., "coverage":...},
      "normal_edge_points": {"edge1":[[x,y],...], "edge2":[[x,y],...]}
    }
    """
    out = {
        "variant_id": int(vid),
        "midline": vrec.get("midline", []),
        "params": params or {},
        "scores": scores or {},
    }
    for k in ("normal_edge_points_full", "normal_edge_points"):
        if k in vrec and vrec[k]:
            out["normal_edge_points"] = vrec[k]
            break
    return out
    
def set_auto_variant_for_crack(save_folder, base_name, crack_id, vrec, params=None, is_best=False, scores=None):
    """
    Persist a variant under atomic crack snapshot (flat).
    Ensures keys: auto_variants (dict of vN -> {...}), auto_best (single flat dict).
    """
    p = metric_atomic_path_for(save_folder, base_name, crack_id)
    rec = safe_read_json(p, {})
    if not rec: 
        rec = {"crack_id": crack_id}

    # Normalize container
    av = rec.get("auto_variants")
    if not isinstance(av, dict):
        av = {}
    vid = None
    if isinstance(vrec, dict):
        # best effort: try to extract known id from params/desc
        vid = vrec.get("params", {}).get("variant_id", None)
    # fallback from caller
    if vid is None:
        # caller usually passes 'v<id>' externally; we won't rely on that here
        vid = len(av)

    flat = _flatten_variant_record(vid, vrec, params or vrec.get("params", {}), scores=scores)
    av[f"v{vid}"] = flat
    rec["auto_variants"] = av

    if is_best:
        rec["auto_best"] = dict(flat)  # copy

    safe_write_json(p, rec)
    return rec

def set_tracked_edges_for_crack(save_folder: str, base: str, cid, payload: dict, mask_crop=None):
    p = metric_atomic_path_for(save_folder, base, cid)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    data = dict(payload or {})
    if mask_crop is not None:
        data["mask_crop"] = mask_crop
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return p
    
def set_geodesic_edges_for_crack(save_folder, base_name, crack_id, ge_dict):
    """Store geodesic edges (edge1/edge2 lists) under atomic crack snapshot."""
    p = metric_atomic_path_for(save_folder, base_name, crack_id)
    rec = safe_read_json(p, {})
    if not rec:
        rec = {"crack_id": crack_id}
    rec["geodesic_edges"] = ge_dict or {}
    safe_write_json(p, rec)
    return rec

# -------------------- PLOTTING (GT QUICK LOOK) --------------------
def debug_plot_gt_preview(mask_bin, mid_xy, e1_xy=None, e2_xy=None, out_png=None, title="GT preview"):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    H, W = mask_bin.shape
    import numpy as np, cv2
    img = np.zeros((H, W, 3), np.uint8); img[mask_bin > 0] = (255,255,255)

    plt.figure(figsize=(6,6))
    plt.imshow(img, origin="upper")
    if isinstance(mid_xy, np.ndarray) and len(mid_xy) >= 2:
        plt.plot(mid_xy[:,0], mid_xy[:,1], 'k-', lw=3)
        plt.plot(mid_xy[:,0], mid_xy[:,1], 'w-', lw=1.5)
    if (e1_xy is not None) and (e2_xy is not None):
        n = min(len(e1_xy), len(e2_xy))
        stride = max(1, n // max(1, n // 8))  # ~8x sparser just for display
        segs = np.stack([e1_xy[:n:stride], e2_xy[:n:stride]], axis=1)
        lc = LineCollection(segs, colors='C0', linewidths=1.5, alpha=0.85)
        plt.gca().add_collection(lc)
    plt.title(title); plt.axis("equal"); plt.tight_layout()
    if out_png:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=180)
        plt.close()
    else:
        plt.show()

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

def merged_metric_atomic(authoring_atomic, save_folder, image_base):
    """
    Merge authoring cracks with any per-cid metric snapshots.
    Supports either .../metrics/<image_base>/cid/<cid>.json
    or .../metrics/<image_base>/cid<cid>.json
    """
    import os, json
    merged = {}
    base_dir = os.path.join(save_folder, "metrics", image_base)

    for cid, cr in (authoring_atomic or {}).items():
        merged[cid] = dict(cr)
        path_opts = [
            os.path.join(base_dir, "cid", f"{cid}.json"),
            os.path.join(base_dir, f"cid{cid}.json"),
        ]
        for path in path_opts:
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        snap = json.load(f) or {}
                    merged[cid].update(snap)
                    print(f"[merge] ✅ merged edges from {path}")
                    break
                except Exception as e:
                    print(f"[merge] ⚠ failed {path}: {e}")
    return merged

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

def manual_mask_from_crack(crack: dict, H: int, W: int) -> np.ndarray:
    """
    Build a binary manual mask for a single crack.
    Priority: geodesic_edges polygon → (optional) mask_crop+mask_bbox → zeros.
    """
    ge = (crack or {}).get("geodesic_edges", {}) or {}
    if ("edge1" in ge) and ("edge2" in ge):
        m = _safe_poly_fill(H, W, ge.get("edge1", []), ge.get("edge2", []))
        if m.any():
            return m.astype(np.uint8)

    # Fallback ONLY if a (legacy) crop exists
    mc = crack.get("mask_crop", None)
    bb = crack.get("mask_bbox", None)
    if mc is not None and isinstance(bb, (list, tuple)) and len(bb) == 4:
        x, y, w, h = [int(v) for v in bb]
        if w > 0 and h > 0:
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(W, x + w), min(H, y + h)
            if x1 > x0 and y1 > y0:
                crop = (np.asarray(mc) > 0).astype(np.uint8)
                crop = crop[:(y1-y0), ::(x1-x0)]
                m = np.zeros((H, W), np.uint8)
                m[y0:y1, x0:x1] = crop[:(y1-y0), :(x1-x0)]
                if m.any():
                    return m
    return np.zeros((H, W), np.uint8)

def merged_metric_atomic(authoring_atomic: dict, save_folder: str, image_base: str) -> dict:
    """
    Merge authoring entries with per-cid snapshots (supports both layouts):
      .../metrics/<base>/cid/<cid>.json  OR  .../metrics/<base>/cid<cid>.json
    """
    merged = {}
    base_dir = os.path.join(save_folder, "metrics", image_base)
    for cid, cr in (authoring_atomic or {}).items():
        merged[cid] = dict(cr)
        path_opts = [
            os.path.join(base_dir, "cid", f"{cid}.json"),
            os.path.join(base_dir, f"cid{cid}.json"),
        ]
        for p in path_opts:
            if os.path.exists(p):
                try:
                    with open(p, "r") as f:
                        snap = json.load(f) or {}
                    merged[cid].update(snap)
                    print(f"[merge] ✅ merged edges from {p}")
                    break
                except Exception as e:
                    print(f"[merge] ⚠ failed {p}: {e}")
    return merged

# --- minimal geometry utils (existing from your metrics) ----------------------
# Expect these to already exist in your codebase:
#   - _reconstruct_full_mask(obj, H, W)
#   - mask_iou(m1, m2)
#   - normals_from_mask_for_midline(midline_xy, mask, max_radius=50)
#   - compute_midline_metrics(auto_xy, man_xy, tau)
#   - compare_widths_for_cracks(ann_like, crack_mask, base_name, metrics_dir, display=False, tag=None)

# For clarity: ann_like for compare_widths_for_cracks is {"atomic_cracks": {cid: crackdict, ...}}
