import os, json, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import csv
import numpy as np
import cv2
from helpers.plot_metrics import plot_edges_and_normals


# --------------------------------------------
# UTIL
# --------------------------------------------
def _ensure(p):
    os.makedirs(p, exist_ok=True)
    return p


def _rebuild_segs(flat):
    """
    Convert a flattened [ [x,y], [x,y], [None,None], ... ]
    into a list of contiguous Nx2 numpy arrays.
    """
    segs = []
    curr = []
    for x, y in flat:
        if x is None or y is None:
            if curr:
                segs.append(np.array(curr))
                curr = []
        else:
            curr.append([x, y])
    if curr:
        segs.append(np.array(curr))
    return segs


# --------------------------------------------
# MAIN ENTRYPOINT
# --------------------------------------------
def export_all_supervision(*, atomic, combined, metrics_dir, original_image):
    """
    Writes:
        metrics/<image>/supervision/supervision.json
        metrics/<image>/supervision/preview/*.png
    """

    sup_root = _ensure(os.path.join(metrics_dir, "supervision"))
    sup_prev = _ensure(os.path.join(sup_root, "preview"))

    H, W = original_image.shape[:2]

    sup_list = []

    # -------------------------------------------------------
    # FLATTEN combined membership so we avoid double-exporting atomics
    # -------------------------------------------------------
    combined_members_flat = {
        m for g in (combined or {}).values()
        for m in (g.get("members") or [])
    }

    # =======================================================
    # ATOMIC (standalone only)
    # =======================================================
    for cid, cr in atomic.items():
        if str(cid) in combined_members_flat:
            continue

        entry = {
            "type": "atomic",
            "id": str(cid),
            "source": cr.get("source"),
            "members": [],
            "midline": cr.get("midline", []),
            "normals": cr.get("normal_edge_points", {}),
            "mask_bbox": cr.get("mask_bbox"),
            "mask_crop": cr.get("mask_crop"),
        }
        sup_list.append(entry)

        _supervision_preview(entry, original_image, os.path.join(sup_prev, f"cid{cid}.png"))

    # =======================================================
    # COMBINED
    # =======================================================
    for ccid, cmb in (combined or {}).items():
        entry = {
            "type": "combined",
            "id": str(ccid),
            "source": "combined",
            "members": cmb.get("members", []),
            "midline": cmb.get("midline", []),
            "normals": cmb.get("normal_edge_points", {}),
            "mask_bbox": cmb.get("mask_bbox"),
            "mask_crop": cmb.get("mask_crop"),
        }
        sup_list.append(entry)

        tag = f"combined_{'_'.join(entry['members'])}"
        _supervision_preview(entry, original_image, os.path.join(sup_prev, f"{tag}.png"))

    # =======================================================
    # WRITE ONE UNIFIED JSON
    # =======================================================
    out_json = os.path.join(sup_root, "supervision.json")
    with open(out_json, "w") as f:
        json.dump(sup_list, f)

    print(f"[SUPERVISION] âœ“ wrote → {out_json}")


# --------------------------------------------
# PREVIEW GENERATOR
# --------------------------------------------
def _supervision_preview(entry, original_image, out_path):
    """
    Preview:
        - Overlay GT mask (from mask_bbox + mask_crop) blended 50/50 with raw
        - Plot midline + normals (no geodesic edges)
        - Uses unified plot_edges_and_normals
    """
    H, W = original_image.shape[:2]

    # ---- rebuild segments from flattened format ----
    mid_segs = _rebuild_segs(entry.get("midline") or [])
    normals = entry.get("normals") or {}
    n1 = _rebuild_segs(normals.get("edge1") or [])
    n2 = _rebuild_segs(normals.get("edge2") or [])

    # ---- reconstruct ground truth mask (full image size) ----
    full_mask = np.zeros((H, W), np.uint8)
    bbox = entry.get("mask_bbox")

    if bbox and entry.get("mask_crop") is not None:
        x, y, w, h = map(int, bbox)
        crop_arr = np.asarray(entry["mask_crop"], np.uint8)
        full_mask[y:y+h, x:x+w] = crop_arr > 0

    # ---- build overlay (50/50 raw image + GT mask) ----
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mask_f = full_mask.astype(np.float32)

    # Bright highlights for GT mask
    gt_vis = gray * 0.25 + mask_f * 0.75
    gt_vis = np.clip(gt_vis, 0, 1.0)

    gt_rgb = np.stack([gt_vis]*3, axis=-1)

    # ---- call unified plot function ----
    title = f"Ground truth crack normals â€” {entry['type']} {entry['id']}"

    plot_edges_and_normals(
        base_image=(gt_rgb * 255).astype(np.uint8),
        midline_segs=mid_segs,
        edge1_segs=[],     # GT has no geodesic edges
        edge2_segs=[],
        norm1_segs=n1,
        norm2_segs=n2,
        bbox=bbox,
        out_png=out_path,
        title=title,
    )

    print(f"[SUPERVISION] preview → {out_path}")
    
    
    
    
    
    
    
    
    
    

# ============================================================
#  GT SUPERVISION EXPORT (CLEAN CROPS ONLY + GLOBAL OVERVIEW)
# ============================================================

import os, json
import numpy as np
import cv2
from matplotlib import pyplot as plt
from helpers.geometry_canonical import (
    canonicalize_segment_direction,
    enforce_branch_continuity,
    canonicalize_branch_direction,
    assert_direction_consistency,
)

from helpers.metrics import normals_from_mask_for_midline, resample_by_arclength
#from combiner import _stitch_lines_by_user
from helpers.plot_metrics import plot_edges_and_normals


# ============================================================
# UTILS
# ============================================================
def _arr_to_list(a):
    if a is None:
        return []
    return np.asarray(a).tolist()


def _cc_label_for_midline(mid_xy: np.ndarray, cc_labels: np.ndarray):
    """
    Returns CC index most frequently hit by round(midline).
    """
    if mid_xy.ndim != 2 or mid_xy.shape[1] != 2:
        return None

    H, W = cc_labels.shape
    xs = np.clip(np.round(mid_xy[:, 0]).astype(int), 0, W - 1)
    ys = np.clip(np.round(mid_xy[:, 1]).astype(int), 0, H - 1)

    lbls = cc_labels[ys, xs]
    lbls = lbls[lbls > 0]

    if lbls.size == 0:
        return None

    vals, counts = np.unique(lbls, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox_from_coords(coords, H, W, pad=2):
    """Safe bounding-box for arbitrary xy coords."""

    coords = np.asarray(coords, float)
    coords = coords[np.isfinite(coords).all(axis=1)]
    if coords.size == 0:
        return None

    xs, ys = coords[:, 0], coords[:, 1]

    x0 = max(0, int(np.floor(xs.min() - pad)))
    x1 = min(W - 1, int(np.ceil(xs.max() + pad)))
    y0 = max(0, int(np.floor(ys.min() - pad)))
    y1 = min(H - 1, int(np.ceil(ys.max() + pad)))

    if x1 <= x0 or y1 <= y0:
        return None

    return (x0, y0, x1, y1)


def _clip_mask_to_xywh(mask_u8, bbox_xywh):
    """
    Hard-clip mask support to bbox domain.
    """
    m = (np.asarray(mask_u8) > 0).astype(np.uint8)
    if bbox_xywh is None or len(bbox_xywh) != 4:
        return m
    H, W = m.shape[:2]
    x, y, w, h = map(int, bbox_xywh)
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(W, x + max(0, w))
    y1 = min(H, y + max(0, h))
    if x1 <= x0 or y1 <= y0:
        return np.zeros_like(m, np.uint8)
    bb = np.zeros_like(m, np.uint8)
    bb[y0:y1, x0:x1] = 1
    return (m & bb).astype(np.uint8)





def _split_midline_packed(mid_packed):
    """
    mid_packed: list like [[x,y], [x,y], [None,None], [x,y], ...]
    returns: list of (N,2) float arrays
    """
    segs = []
    cur = []
    for pt in (mid_packed or []):
        if pt is None or len(pt) != 2 or pt[0] is None or pt[1] is None:
            if len(cur) >= 2:
                segs.append(np.asarray(cur, float))
            cur = []
            continue
        cur.append([float(pt[0]), float(pt[1])])
    if len(cur) >= 2:
        segs.append(np.asarray(cur, float))
    return segs


def _split_xy_none_seps(xs, ys):
    """
    xs,ys: lists like [x,x,x,None,x,x,...] and [y,y,y,None,y,y,...]
    returns: list of (N,2) float arrays
    """
    segs = []
    cur = []
    n = min(len(xs or []), len(ys or []))
    for i in range(n):
        x = xs[i]
        y = ys[i]
        if x is None or y is None:
            if len(cur) >= 2:
                segs.append(np.asarray(cur, float))
            cur = []
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        cur.append([float(x), float(y)])
    if len(cur) >= 2:
        segs.append(np.asarray(cur, float))
    return segs




def _cropped_preview(entry, gt_mask_u8, original_image, out_dir):
    """
    Generates:
        1) Canonical GT preview (manual only)
        2) Comparison preview (manual vs centered) if available
        3) Comparison preview (manual vs depth_distridge) if available

    Manual GT remains authoritative.
    Centered GT is diagnostic only.
    """
    import os
    import numpy as np
    import cv2
    from helpers.plot_metrics import plot_edges_and_normals
    from combiner import bbox_xywh_to_xyxy

    os.makedirs(out_dir, exist_ok=True)

    H, W = gt_mask_u8.shape[:2]
    crack_id = entry.get("id", "UNKNOWN")
    kind = entry.get("kind", "UNKNOWN")

    # 1) Manual midline segments
    if entry.get("midline_segments"):
        manual_mid_segs = [
            np.asarray(S, float)
            for S in entry["midline_segments"]
            if S is not None and len(S) >= 2
        ]
    else:
        mid = np.asarray(entry.get("midline", []), float)
        manual_mid_segs = [mid] if (mid.ndim == 2 and len(mid) >= 2) else []

    if not manual_mid_segs:
        return

    # 2) Manual normals
    normals = entry.get("gt_normals") or {}
    e1_segs = _split_xy_none_seps(normals.get("edge1_x", []), normals.get("edge1_y", []))
    e2_segs = _split_xy_none_seps(normals.get("edge2_x", []), normals.get("edge2_y", []))

    # 3) BBox
    bb = entry.get("mask_bbox")
    if bb is None:
        raise ValueError(f"[CROP_DBG] {kind}:{crack_id} missing mask_bbox")

    x0, y0, x1, y1 = bbox_xywh_to_xyxy(bb, H, W, pad=5)

    # 4) Expand crop for normals (visual only)
    all_pts = []
    for S in e1_segs + e2_segs:
        if S is not None and len(S):
            all_pts.append(np.asarray(S, float))

    if all_pts:
        P = np.vstack(all_pts)
        x0 = int(max(0, min(x0, np.floor(P[:, 0].min()) - 5)))
        y0 = int(max(0, min(y0, np.floor(P[:, 1].min()) - 5)))
        x1 = int(min(W - 1, max(x1, np.ceil(P[:, 0].max()) + 5)))
        y1 = int(min(H - 1, max(y1, np.ceil(P[:, 1].max()) + 5)))

    # 5) Build overlay image
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mask_f = (gt_mask_u8 > 0).astype(np.float32)
    overlay = np.clip(gray * 0.25 + mask_f * 0.75, 0, 1)
    overlay_rgb = (np.stack([overlay] * 3, axis=-1) * 255).astype(np.uint8)

    crop_img = overlay_rgb[y0:y1, x0:x1]

    shift = np.array([x0, y0], float)
    manual_mid_crop = [S - shift for S in manual_mid_segs]
    e1_crop = [S - shift for S in e1_segs]
    e2_crop = [S - shift for S in e2_segs]

    bx, by, bw, bh = bb
    bbox_plot = [int(bx - x0), int(by - y0), int(bw), int(bh)]

    # A) Canonical GT preview (manual only)
    out_gt = os.path.join(out_dir, f"{kind}_{crack_id}_crop_gt.png")

    plot_edges_and_normals(
        base_image=crop_img,
        midline_segs=manual_mid_crop,
        edge1_segs=[],
        edge2_segs=[],
        norm1_segs=e1_crop,
        norm2_segs=e2_crop,
        sparsity=5,
        gt_plot=True,
        bbox=bbox_plot,
        out_png=out_gt,
        title=f"{kind} {crack_id} - Manual GT",
    )

    # B) Comparison preview (manual + centered)
    centered_mid_segs = []
    if entry.get("midline_segments_auto_centered"):
        centered_mid_segs = [
            np.asarray(S, float)
            for S in (entry.get("midline_segments_auto_centered") or [])
            if S is not None and len(S) >= 2
        ]
    else:
        cm = entry.get("midline_auto_centered")
        if cm is not None:
            arr = np.asarray(cm, float)
            if arr.ndim == 2 and len(arr) >= 2:
                centered_mid_segs = [arr]
            else:
                centered_mid_segs = _split_midline_packed(cm)

    centered_normals = entry.get("gt_normals_auto_centered")

    if centered_mid_segs and isinstance(centered_normals, dict):
        centered_mid_crop = [S - shift for S in centered_mid_segs]

        ce1 = _split_xy_none_seps(
            centered_normals.get("edge1_x", []),
            centered_normals.get("edge1_y", []),
        )
        ce2 = _split_xy_none_seps(
            centered_normals.get("edge2_x", []),
            centered_normals.get("edge2_y", []),
        )

        ce1_crop = [S - shift for S in ce1]
        ce2_crop = [S - shift for S in ce2]

        out_cmp = os.path.join(out_dir, f"{kind}_{crack_id}_crop_compare.png")

        plot_edges_and_normals(
            base_image=crop_img,
            midline_segs=manual_mid_crop,
            derived_midline_segs=centered_mid_crop,
            edge1_segs=[],
            edge2_segs=[],
            norm1_segs=ce1_crop,
            norm2_segs=ce2_crop,
            sparsity=5,
            gt_plot=False,
            bbox=bbox_plot,
            out_png=out_cmp,
            title=f"{kind} {crack_id} - Manual vs Centered",
        )

    # C) Comparison preview (manual + depth_distridge)
    depth_mid_segs = []
    if entry.get("depth_midline_segments"):
        depth_mid_segs = [
            np.asarray(S, float)
            for S in (entry.get("depth_midline_segments") or [])
            if S is not None and len(S) >= 2
        ]
    else:
        dm = entry.get("depth_midline", entry.get("midline_depth_centered"))
        if dm is not None:
            arr = np.asarray(dm, float)
            if arr.ndim == 2 and len(arr) >= 2:
                depth_mid_segs = [arr]
            else:
                depth_mid_segs = _split_midline_packed(dm)

    if depth_mid_segs:
        depth_mid_crop = [S - shift for S in depth_mid_segs]
        depth_normals = entry.get("depth_normals")
        if not isinstance(depth_normals, dict) or not depth_normals:
            depth_normals = entry.get("gt_normals_depth_centered")
        if not isinstance(depth_normals, dict):
            depth_normals = {}

        de1 = _split_xy_none_seps(
            depth_normals.get("edge1_x", []),
            depth_normals.get("edge1_y", []),
        )
        de2 = _split_xy_none_seps(
            depth_normals.get("edge2_x", []),
            depth_normals.get("edge2_y", []),
        )
        de1_crop = [S - shift for S in de1]
        de2_crop = [S - shift for S in de2]

        out_depth_cmp = os.path.join(out_dir, f"{kind}_{crack_id}_crop_compare_depth_distridge.png")
        plot_edges_and_normals(
            base_image=crop_img,
            midline_segs=manual_mid_crop,
            derived_midline_segs=depth_mid_crop,
            edge1_segs=[],
            edge2_segs=[],
            norm1_segs=de1_crop,
            norm2_segs=de2_crop,
            sparsity=5,
            gt_plot=False,
            bbox=bbox_plot,
            out_png=out_depth_cmp,
            title=f"{kind} {crack_id} - Manual vs DepthDistridge",
        )
# ============================================================
# GLOBAL OVERVIEW (with legend + title)
# ============================================================
def _global_overview(entries, gt_mask, out_png, title="Global GT Overview"):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    H, W = gt_mask.shape[:2]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=320)
    ax.imshow(gt_mask, cmap="gray", interpolation="nearest")

    # ---------------------------
    # Draw all midlines
    # ---------------------------
    for e in entries:
        col = "red" if e["kind"] == "atomic" else "lime"

        if e.get("midline_segments"):
            segs = [np.asarray(S, float) for S in e["midline_segments"] if S is not None and len(S) >= 2]
        else:
            mid_raw = e.get("midline", [])
            # atomic style
            try:
                mid = np.asarray(mid_raw, float)
                segs = [mid] if (mid.ndim == 2 and len(mid) >= 2) else []
            except Exception:
                # packed style fallback
                segs = _split_midline_packed(mid_raw)

        for S in segs:
            if len(S) >= 2:
                ax.plot(S[:, 0], S[:, 1], lw=1.3, color=col, alpha=0.9)

    # ---------------------------
    # Legend
    # ---------------------------
    handles = [
        Line2D([], [], color="red", lw=2, label="Atomic crack"),
        Line2D([], [], color="lime", lw=2, label="Combined crack"),
    ]

    leg = ax.legend(
        handles=handles,
        fontsize=11,
        loc="lower right",
        framealpha=0.85,
        title="Crack Types",
        title_fontsize=12
    )
    # Make title blue + bold
    plt.setp(leg.get_title(), color="blue", fontweight="bold")
    for t in leg.get_texts():
        t.set_fontweight("bold")

    # ---------------------------
    # Title
    # ---------------------------
    if title:
        ax.set_title(title, fontsize=15, fontweight="bold", color="blue")

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=320, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def _cc_label_for_members(members, atomic, cc_labels):
    """Robust CC label selection for a group: vote over all member midline points."""
    H, W = cc_labels.shape[:2]
    labels = []
    for m in members:
        cr = atomic.get(str(m), {}) or {}
        mid = np.asarray(cr.get("midline", []), float)
        if mid.ndim == 2 and len(mid) >= 1:
            ys = np.clip(np.round(mid[:, 1]).astype(int), 0, H - 1)
            xs = np.clip(np.round(mid[:, 0]).astype(int), 0, W - 1)
            labs = cc_labels[ys, xs]
            labs = labs[labs > 0]
            if len(labs):
                labels.append(labs)
    if not labels:
        return None
    labs = np.concatenate(labels, axis=0)
    vals, cnts = np.unique(labs, return_counts=True)
    return int(vals[np.argmax(cnts)]) if len(vals) else None




import numpy as np
import cv2
import os

def _pack_segs_with_separators(segs):
    """Flatten [N_i x 2] segments into one list with [None,None] separators."""
    out = []
    for k, S in enumerate(segs):
        if S is None or len(S) < 2:
            continue
        if k > 0:
            out.append([None, None])
        out.extend([[float(x), float(y)] for x, y in np.asarray(S, float)])
    return out

def _pack_arrs_with_none_separators(arr_list):
    """Flatten list of 1D arrays into one list with None separators."""
    out = []
    for k, a in enumerate(arr_list):
        a = list(a) if a is not None else []
        if k > 0:
            out.append(None)
        out.extend([float(v) if np.isfinite(v) else None for v in a])
    return out

def _polyline_mask(mid_xy, H, W):
    """Rasterize a polyline (Nx2 xy) into a uint8 mask."""
    S = np.asarray(mid_xy, float)
    out = np.zeros((H, W), np.uint8)
    if S.ndim != 2 or S.shape[1] != 2 or len(S) < 2:
        return out
    pts = np.round(S).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
    cv2.polylines(out, [pts.reshape(-1, 1, 2)], isClosed=False, color=1, thickness=1)
    return out

def _shift_stats(manual_xy, centered_xy):
    """Per-point displacement stats in px for equal-length polylines."""
    m = np.asarray(manual_xy, float)
    c = np.asarray(centered_xy, float)
    n = min(len(m), len(c))
    if n <= 0:
        return {"mean_shift_px": 0.0, "p95_shift_px": 0.0, "max_shift_px": 0.0}
    d = np.linalg.norm(c[:n] - m[:n], axis=1)
    if d.size == 0:
        return {"mean_shift_px": 0.0, "p95_shift_px": 0.0, "max_shift_px": 0.0}
    return {
        "mean_shift_px": float(np.mean(d)),
        "p95_shift_px": float(np.percentile(d, 95)),
        "max_shift_px": float(np.max(d)),
    }

def _width_stability_stats(manual_widths, centered_widths):
    """
    Compare width stability and invalid-rate between manual and centered traces.
    """
    wm = np.asarray(manual_widths, float).reshape(-1)
    wc = np.asarray(centered_widths, float).reshape(-1)
    if wm.size == 0:
        wm = np.array([np.nan], float)
    if wc.size == 0:
        wc = np.array([np.nan], float)

    vm = np.isfinite(wm)
    vc = np.isfinite(wc)

    return {
        "manual_width_mean": float(np.nanmean(wm)),
        "centered_width_mean": float(np.nanmean(wc)),
        "manual_width_std": float(np.nanstd(wm)),
        "centered_width_std": float(np.nanstd(wc)),
        "manual_invalid_frac": float(1.0 - np.mean(vm)),
        "centered_invalid_frac": float(1.0 - np.mean(vc)),
    }


def _normals_diag_summary(diag, topk=4):
    d = dict(diag) if isinstance(diag, dict) else {}
    reasons = d.get("reasons", {}) if isinstance(d.get("reasons", {}), dict) else {}
    reasons_sorted = sorted(reasons.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
    return {
        "total": int(d.get("total", 0) or 0),
        "valid": int(d.get("valid", 0) or 0),
        "invalid": int(d.get("invalid", 0) or 0),
        "invalid_frac": float(d.get("invalid_frac", 0.0) or 0.0),
        "top_reasons": reasons_sorted[:max(1, int(topk))],
    }

def _geometry_disagreement_stats(manual_xy, centered_xy):
    """
    Sampling-agnostic geometric disagreement (bidirectional NN + robust Hausdorff).
    """
    from helpers.metrics import nn_mean_bidirectional, hausdorff_p95

    m = np.asarray(manual_xy, float)
    c = np.asarray(centered_xy, float)
    if m.ndim != 2 or c.ndim != 2 or len(m) < 2 or len(c) < 2:
        return {
            "nn_mean_bidirectional_px": float("nan"),
            "hausdorff95_px": float("nan"),
        }

    return {
        "nn_mean_bidirectional_px": float(nn_mean_bidirectional(m, c)),
        "hausdorff95_px": float(hausdorff_p95(m, c)),
    }

def _dt_radius_from_polyline(dt, S, *, window_half_size=50, min_r=3.0, fallback_frac=0.30):
    """
    Estimate representative crack radius by sampling DT along polyline S.
    """
    H, W = dt.shape[:2]
    S = np.asarray(S, float)
    if S.ndim != 2 or S.shape[1] != 2 or len(S) < 2:
        return None

    ys = np.clip(np.round(S[:, 1]).astype(int), 0, H - 1)
    xs = np.clip(np.round(S[:, 0]).astype(int), 0, W - 1)
    d = dt[ys, xs]
    d = d[np.isfinite(d)]

    if d.size:
        r = float(np.median(d))
    else:
        r = float(fallback_frac * float(window_half_size))

    r = max(float(min_r), min(r, float(window_half_size)))
    return r

def build_territory_mask_from_polyline(
    *,
    mid_xy,
    crack_mask_u8,
    window_half_size=50,
    dt_domain_u8=None,
    min_r=3.0,
    fallback_frac=0.30,
    rad_scale=1.20,
    min_rad_px=4,
):
    """
    Build territory corridor around one polyline.
    DT is measured on dt_domain_u8 (if provided), always constrained to crack mask.
    """
    H, W = crack_mask_u8.shape[:2]
    crack = (np.asarray(crack_mask_u8) > 0).astype(np.uint8)
    if not np.any(crack):
        return np.zeros((H, W), np.uint8)

    S = np.asarray(mid_xy, float)
    if S.ndim != 2 or S.shape[1] != 2 or len(S) < 2:
        return np.zeros((H, W), np.uint8)

    if dt_domain_u8 is None:
        dt_domain = crack
    else:
        dt_domain = (np.asarray(dt_domain_u8) > 0).astype(np.uint8)
        dt_domain = (dt_domain & crack).astype(np.uint8)
        if not np.any(dt_domain):
            dt_domain = crack

    dt = cv2.distanceTransform(dt_domain, cv2.DIST_L2, 5).astype(np.float32)
    r = _dt_radius_from_polyline(
        dt, S,
        window_half_size=window_half_size,
        min_r=min_r,
        fallback_frac=fallback_frac,
    )
    if r is None:
        return np.zeros((H, W), np.uint8)

    rad = int(max(int(min_rad_px), min(float(window_half_size), float(rad_scale) * r)))
    line = _polyline_mask(S, H, W).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
    terr = cv2.dilate(line, kernel, iterations=1).astype(np.uint8)
    # Keep territory strictly inside crack support.
    terr = (terr & crack).astype(np.uint8)
    return terr

def build_territory_mask_for_segments(
    *,
    segs,
    crack_mask_u8,
    window_half_size=50,
    dt_domain_u8=None,
    **kwargs,
):
    """
    Build union territory for multiple segments with shared DT policy.
    """
    H, W = crack_mask_u8.shape[:2]
    terr = np.zeros((H, W), np.uint8)
    for S in (segs or []):
        S = np.asarray(S, float)
        if S.ndim != 2 or S.shape[1] != 2 or len(S) < 2:
            continue
        terr |= build_territory_mask_from_polyline(
            mid_xy=S,
            crack_mask_u8=crack_mask_u8,
            window_half_size=window_half_size,
            dt_domain_u8=dt_domain_u8,
            **kwargs,
        )
    return terr

def build_centering_domain_mask(*, crack_mask_u8, territory_u8=None, mode="soft"):
    """
    Build allowed domain for center snapping.
    mode:
      - soft: crack mask only
      - terr_or_mask: crack | territory
      - terr_and_mask: crack & territory
    """
    m = (np.asarray(crack_mask_u8) > 0).astype(np.uint8)
    if territory_u8 is None:
        return m
    t = (np.asarray(territory_u8) > 0).astype(np.uint8)

    if mode == "terr_and_mask":
        return (m & t).astype(np.uint8)
    if mode == "terr_or_mask":
        return (m | t).astype(np.uint8)
    return m
from helpers.supervision_midline_helpers import (
    METHOD_SPECS,
    compute_midline_method_variants_and_normals,
    compute_centered_midline_and_normals,
    snap_polyline_to_dt_ridge,
)

def plot_midline_centering_debug(
    *,
    out_path,
    crack_mask_u8,
    manual_segs,
    centered_segs,
    territory_u8=None,
    bbox_xywh=None,
    title="GT supervision auto-centering",
    invalid_manual_masks=None,
    invalid_center_masks=None,
    show_dt_panel=True,
    show_territory=True,
    territory_alpha=0.25,
    compare_label="Centered Midline",
    compare_color="cyan",
    left_panel_title="manual vs centered (mask)",
    right_panel_title="DT ridge view (domain)",
):
    """
    Plot manual vs centered midlines over crack mask (crop around bbox).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    M = (np.asarray(crack_mask_u8) > 0).astype(np.uint8)
    H, W = M.shape[:2]

    if not centered_segs:
        raise RuntimeError("No centered segments provided - cyan should exist but is empty.")
    for i, S in enumerate(centered_segs or []):
        S = np.asarray(S, float)
        if S.ndim != 2 or len(S) < 2:
            raise RuntimeError(f"Centered segment {i} is invalid shape {S.shape}")

    # Geometry-debug: report overlap likelihood (centered hidden under manual).
    pair_n = min(len(manual_segs or []), len(centered_segs or []))
    for i in range(pair_n):
        m = np.asarray((manual_segs or [])[i], float)
        c = np.asarray((centered_segs or [])[i], float)
        if m.shape == c.shape and m.ndim == 2 and len(m) >= 2:
            max_diff = float(np.max(np.abs(m - c)))
            print(f"[AUTO CENTER DEBUG] seg={i} max_shift_abs={max_diff:.6f}")

    if bbox_xywh is not None and len(bbox_xywh) == 4:
        x, y, w, h = map(int, bbox_xywh)
        pad = 25
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(W, x + w + pad)
        y1 = min(H, y + h + pad)
    else:
        ys, xs = np.where(M > 0)
        if xs.size:
            pad = 25
            x0 = max(0, int(xs.min()) - pad)
            y0 = max(0, int(ys.min()) - pad)
            x1 = min(W, int(xs.max()) + 1 + pad)
            y1 = min(H, int(ys.max()) + 1 + pad)
        else:
            x0, y0, x1, y1 = 0, 0, W, H

    T = None
    if territory_u8 is not None:
        T = (np.asarray(territory_u8) > 0).astype(np.uint8)

    if show_dt_panel:
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.0), dpi=220, sharex=True, sharey=True)
        ax0, ax1 = axes
    else:
        fig, ax0 = plt.subplots(1, 1, figsize=(6.4, 6.4), dpi=220)
        ax1 = ax0

    # True RGB render (no colormap interpolation): black bg, white crack mask.
    crop_mask = M[y0:y1, x0:x1]
    rgb0 = np.zeros((crop_mask.shape[0], crop_mask.shape[1], 3), np.uint8)
    rgb0[crop_mask > 0] = (255, 255, 255)
    if show_territory and T is not None:
        crop_T = (T[y0:y1, x0:x1] > 0)
        # Explicit RGB blend for stable, artifact-free overlay.
        overlay = rgb0.copy()
        overlay[crop_T] = (120, 255, 120)
        rgb0 = (0.75 * rgb0 + 0.25 * overlay).astype(np.uint8)
    ax0.imshow(rgb0)
    ax0.axis("off")
    ax0.set_title(left_panel_title, fontsize=10)

    if show_dt_panel:
        if T is not None:
            dom = (M & T).astype(np.uint8)
            if not np.any(dom):
                dom = M
        else:
            dom = M

        dt = cv2.distanceTransform(dom, cv2.DIST_L2, 5).astype(np.float32)
        dt_crop = dt[y0:y1, x0:x1]

        # Pure magma for DT
        ax1.imshow(dt_crop, cmap="magma")
        ax1.axis("off")
        ax1.set_title(right_panel_title, fontsize=10)

        # Territory overlay as pure white alpha mask (no colormap)
        if show_territory and T is not None:
            crop_T = (T[y0:y1, x0:x1] > 0)

            overlay = np.zeros((*crop_T.shape, 4), dtype=np.float32)
            overlay[..., 0] = 1.0  # R
            overlay[..., 1] = 1.0  # G
            overlay[..., 2] = 1.0  # B
            overlay[..., 3] = crop_T.astype(np.float32) * 0.20  # alpha only where territory

            ax1.imshow(overlay)

    # -------------------------------------------------
    # Plot CENTERED first (solid, underneath)
    # -------------------------------------------------
    for i, S in enumerate(centered_segs or []):
        S = np.asarray(S, float)
        if S.ndim == 2 and len(S) >= 2:
            for ax in (ax0, ax1):
                ax.plot(
                    S[:, 0] - x0,
                    S[:, 1] - y0,
                    color=compare_color,
                    lw=2.5,
                    alpha=0.95,
                    zorder=2,
                )
            if invalid_center_masks and i < len(invalid_center_masks):
                bad = np.asarray(invalid_center_masks[i], bool)
                if bad.size == len(S) and np.any(bad):
                    for ax in (ax0, ax1):
                        ax.scatter(
                            S[bad, 0] - x0,
                            S[bad, 1] - y0,
                            s=14,
                            color="magenta",
                            zorder=3,
                        )

    # -------------------------------------------------
    # Plot MANUAL second (dashed, on top)
    # -------------------------------------------------
    for i, S in enumerate(manual_segs or []):
        S = np.asarray(S, float)
        if S.ndim == 2 and len(S) >= 2:
            for ax in (ax0, ax1):
                ax.plot(
                    S[:, 0] - x0,
                    S[:, 1] - y0,
                    color="yellow",
                    lw=2.0,
                    linestyle="--",
                    alpha=0.9,
                    zorder=5,
                )
            if invalid_manual_masks and i < len(invalid_manual_masks):
                bad = np.asarray(invalid_manual_masks[i], bool)
                if bad.size == len(S) and np.any(bad):
                    for ax in (ax0, ax1):
                        ax.scatter(
                            S[bad, 0] - x0,
                            S[bad, 1] - y0,
                            s=14,
                            color="red",
                            zorder=6,
                        )

    if bbox_xywh is not None and len(bbox_xywh) == 4:
        x, y, w, h = map(int, bbox_xywh)
        for ax in (ax0, ax1):
            ax.add_patch(plt.Rectangle((x - x0, y - y0), w, h, fill=False, edgecolor="dodgerblue", linewidth=1.3))

    # Figure-level legend on the right (applies to both panels).
    handles = [
        Line2D([], [], color=compare_color, lw=2.5, linestyle="-", label=str(compare_label)),
        Line2D([], [], color="yellow", lw=2.0, linestyle="--", label="Manual Midline"),
    ]
    if invalid_center_masks is not None:
        handles.append(Line2D([], [], marker="o", linestyle="None", color="magenta", markersize=5, label=f"{compare_label} Invalid Width"))
    if invalid_manual_masks is not None:
        handles.append(Line2D([], [], marker="o", linestyle="None", color="red", markersize=5, label="Manual Invalid Width"))
    fig.subplots_adjust(right=0.83)
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.845, 0.5),
        framealpha=0.9,
        fontsize=9,
        title="Legend",
        title_fontsize=10,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_depth_cost_diagnostic(
    *,
    out_path,
    crack_mask_u8,
    dt_norm,
    depth_norm=None,
    recess_norm=None,
    depth_score=None,
    refine_score=None,
    costmap=None,
    costmaps=None,
    bbox_xywh=None,
    title="Cost diagnostics",
    method_label="",
    depth_label=None,
):
    import matplotlib.pyplot as plt

    M = (np.asarray(crack_mask_u8) > 0).astype(np.uint8)
    H, W = M.shape[:2]

    if bbox_xywh is not None and len(bbox_xywh) == 4:
        x, y, w, h = map(int, bbox_xywh)
        pad = 25
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(W, x + w + pad)
        y1 = min(H, y + h + pad)
    else:
        ys, xs = np.where(M > 0)
        if xs.size:
            pad = 25
            x0 = max(0, int(xs.min()) - pad)
            y0 = max(0, int(ys.min()) - pad)
            x1 = min(W, int(xs.max()) + 1 + pad)
            y1 = min(H, int(ys.max()) + 1 + pad)
        else:
            x0, y0, x1, y1 = 0, 0, W, H

    def _masked_cost(arr):
        if arr is None:
            return None
        a = np.asarray(arr, np.float32)
        valid = (M > 0) & np.isfinite(a) & (a < np.float32(1e8))
        out = np.full_like(a, np.nan, dtype=np.float32)
        vals = a[valid]
        if vals.size == 0:
            return out
        lo = float(np.percentile(vals, 2))
        hi = float(np.percentile(vals, 98))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-9:
            out[valid] = 0.5
            return out
        out[valid] = (a[valid] - lo) / (hi - lo)
        return out

    def _unit_as_is(arr):
        if arr is None:
            return None
        a = np.asarray(arr, np.float32)
        out = np.zeros_like(a, dtype=np.float32)
        valid = np.isfinite(a)
        if not np.any(valid):
            return out
        out[valid] = np.clip(a[valid], 0.0, 1.0)
        return out.astype(np.float32, copy=False)

    panels = []
    costmaps_dict = costmaps if isinstance(costmaps, dict) else None
    if costmaps_dict is None and costmap is not None:
        costmaps_dict = {"selected": costmap}

    if isinstance(costmaps_dict, dict):
        selected = np.asarray(costmaps_dict.get("selected"), np.float32) if "selected" in costmaps_dict else None
        selected_key = str(costmaps_dict.get("selected_key", ""))
        if not selected_key:
            selected_key = "dt"
            for key in ("dt_ridge_valley_depth", "dt_ridge_valley", "dt_depth", "dt"):
                if key in costmaps_dict and selected is not None:
                    try:
                        if np.allclose(
                            np.asarray(costmaps_dict[key], np.float32),
                            selected,
                            atol=1e-6,
                            equal_nan=True,
                        ):
                            selected_key = key
                            break
                    except Exception:
                        pass
        if selected is None:
            selected_key = "dt"
            selected = np.asarray(costmaps_dict.get("dt"), np.float32) if "dt" in costmaps_dict else None

        if "dt" in costmaps_dict:
            panels.append(("DT cost", _masked_cost(costmaps_dict["dt"]), "inferno"))
        if selected_key in {"dt_ridge_valley", "dt_ridge_valley_depth"} and "rgb_cue" in costmaps_dict:
            panels.append(("RGB cost", _masked_cost(costmaps_dict["rgb_cue"]), "inferno"))
        if selected_key in {"dt_depth", "dt_ridge_valley_depth"} and depth_norm is not None:
            panels.append(("Depth map", _unit_as_is(depth_norm), "viridis"))
        if selected_key in {"dt_depth", "dt_ridge_valley_depth"} and recess_norm is not None:
            panels.append(("Depth signal", _unit_as_is(recess_norm), "magma"))
        if selected is not None:
            if "dt" in costmaps_dict:
                try:
                    if not np.allclose(
                        np.asarray(costmaps_dict["dt"], np.float32),
                        np.asarray(selected, np.float32),
                        atol=1e-6,
                        equal_nan=True,
                    ):
                        panels.append(("Final cost (used)", _masked_cost(selected), "inferno"))
                except Exception:
                    panels.append(("Final cost (used)", _masked_cost(selected), "inferno"))
            else:
                panels.append(("Final cost (used)", _masked_cost(selected), "inferno"))
    elif dt_norm is not None:
        dt_arr = np.asarray(dt_norm, np.float32)
        dt_cost = np.full_like(dt_arr, np.float32(1e9), dtype=np.float32)
        valid = (M > 0) & np.isfinite(dt_arr)
        dt_cost[valid] = 1.0 - np.clip(dt_arr[valid], 0.0, 1.0)
        panels.append(("DT cost", _masked_cost(dt_cost), "inferno"))
        panels.append(("Final cost (used)", _masked_cost(dt_cost), "inferno"))

    panels = [(lbl, arr, cmap) for (lbl, arr, cmap) in panels if arr is not None]

    n = max(1, len(panels))
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 4.6), dpi=220, sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (label, arr, cmap) in zip(axes, panels):
        ax.imshow(np.asarray(arr, np.float32)[y0:y1, x0:x1], cmap=cmap)
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    supt = str(title)
    if method_label:
        supt = f"{supt} [{method_label}]"
    fig.suptitle(supt, fontsize=11)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _dom_mask_to_local_array(m, bw, bh):
    """
    Convert a stored dominance bite mask to a (bh, bw) bool array, best-effort.

    Supported:
      - list[list[int/bool]] shape ~= (bh,bw)
      - flat list length == bw*bh
      - numpy array any of the above
    Returns:
      (arr_bool, info_str) or (None, reason_str)
    """
    import numpy as np

    if m is None:
        return None, "mask=None"

    arr = np.asarray(m)

    if arr.size == 0:
        return None, f"mask empty array shape={arr.shape}"

    # list-of-lists (2D)
    if arr.ndim == 2:
        if arr.shape[0] == bh and arr.shape[1] == bw:
            return (arr.astype(bool), f"ndim2 ok shape={arr.shape}")
        # sometimes transposed
        if arr.shape[0] == bw and arr.shape[1] == bh:
            return (arr.T.astype(bool), f"ndim2 transposed -> {arr.T.shape}")
        return None, f"ndim2 wrong shape={arr.shape} expected {(bh,bw)}"

    # flat
    if arr.ndim == 1:
        if arr.size == bw * bh:
            arr2 = arr.reshape((bh, bw))
            return (arr2.astype(bool), f"flat reshape -> {arr2.shape}")
        return None, f"flat wrong size={arr.size} expected {bw*bh}"

    return None, f"unsupported ndim={arr.ndim} shape={arr.shape}"

def _unpack_mask_b64(blob):
    """
    blob = {"shape":[h,w], "packbits_b64":"..."}
    Returns uint8 mask of shape (h,w) with values {0,1}.
    """
    import base64
    import numpy as np

    if not isinstance(blob, dict):
        return None

    shape = blob.get("shape", None)
    b64 = blob.get("packbits_b64", "")

    if not shape or len(shape) != 2:
        return None

    h, w = int(shape[0]), int(shape[1])
    if h <= 0 or w <= 0:
        return np.zeros((max(h, 0), max(w, 0)), np.uint8)

    if not isinstance(b64, str) or len(b64) == 0:
        return np.zeros((h, w), np.uint8)

    raw = base64.b64decode(b64.encode("ascii"))
    packed = np.frombuffer(raw, dtype=np.uint8)

    # packed is (h, ceil(w/8)) when created with np.packbits(..., axis=1)
    row_bytes = (w + 7) // 8
    need = h * row_bytes
    if packed.size < need:
        # corrupted blob
        return None

    packed = packed[:need].reshape(h, row_bytes)
    bits = np.unpackbits(packed, axis=1)[:, :w]
    return (bits > 0).astype(np.uint8)


def debug_plot_gt_sup_dominance_bite_packed(
    *,
    base_name,
    ccid,
    members,
    dom_meta,
    segs,
    gt_mask,
    out_dir,
    zoom_pad=10,
):
    """
    Plots dominance_meta["bite"] exactly as stored by dominant_segments_from_group(),
    which uses packbits_b64 + shape for masks.

    Panels:
      (1) RAW LOCAL union (bite frame)
      (2) FULL placement on GT mask (global canvas)
      (3) ZOOM crop (union bbox)

    Writes:
      out_dir/gt_sup_dom_bite_debug_{base_name}_ccid{ccid}_<members>.png
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    if not isinstance(dom_meta, dict):
        print(f"[GT_SUP DOMDBG] base={base_name} ccid={ccid} dominance_meta missing")
        return

    bite = dom_meta.get("bite", None)
    if not isinstance(bite, dict):
        print(f"[GT_SUP DOMDBG] base={base_name} ccid={ccid} bite missing")
        return

    bb = bite.get("bbox", None)
    if not bb or len(bb) != 4:
        print(f"[GT_SUP DOMDBG] base={base_name} ccid={ccid} bite bbox missing/invalid: {bb}")
        return

    bx, by, bw, bh = map(int, bb)
    if bw <= 0 or bh <= 0:
        print(f"[GT_SUP DOMDBG] base={base_name} ccid={ccid} bite bbox non-positive: {bb}")
        return

    print(f"[GT_SUP DOMDBG] base={base_name} ccid={ccid} RAW bite bbox={bb} members={members}")

    H, W = gt_mask.shape[:2]

    # -----------------------------
    # 1) RAW LOCAL union (bite frame)
    # -----------------------------
    local_union = np.zeros((bh, bw), np.uint8)

    # First try union blob (backward compatible)
    if "packbits_b64" in bite and "shape" in bite:
        u = _unpack_mask_b64({"shape": bite.get("shape"), "packbits_b64": bite.get("packbits_b64")})
        if u is None:
            print(f"[GT_SUP DOMDBG] base={base_name} ccid={ccid} union blob decode failed")
        else:
            if u.shape == (bh, bw):
                local_union |= (u > 0).astype(np.uint8)

    # Also try per-losing-branch union blobs (new format)
    by_lb = bite.get("by_losing_branch", {}) or {}
    if isinstance(by_lb, dict):
        for bid, entry in by_lb.items():
            if not isinstance(entry, dict):
                continue
            # entry itself is a blob: {"shape":..,"packbits_b64":..,"by_cause":..}
            u = _unpack_mask_b64(entry)
            if u is None:
                print(f"[GT_SUP DOMDBG]  bid={bid}: BAD packed blob decode (entry keys={list(entry.keys())})")
                continue
            if u.shape != (bh, bw):
                print(f"[GT_SUP DOMDBG]  bid={bid}: shape mismatch {u.shape} vs {(bh,bw)}")
                continue
            if np.any(u):
                local_union |= (u > 0).astype(np.uint8)

    if not np.any(local_union):
        print(f"[GT_SUP DOMDBG] base={base_name} ccid={ccid} RAW LOCAL union is EMPTY")
        # still write a figure so you see placement context
        # (this helps diagnose â€œbbox exists but union emptyâ€)
    # -----------------------------
    # 2) FULL placement on GT mask
    # -----------------------------
    placed = np.zeros((H, W), np.uint8)
    y1 = min(H, by + bh)
    x1 = min(W, bx + bw)
    yy = max(0, by)
    xx = max(0, bx)
    ph = y1 - yy
    pw = x1 - xx
    if ph > 0 and pw > 0:
        placed[yy:y1, xx:x1] = local_union[(yy - by):(yy - by + ph), (xx - bx):(xx - bx + pw)]

    # -----------------------------
    # 3) ZOOM crop around bite bbox (or nonzero union)
    # -----------------------------
    if np.any(placed):
        ys, xs = np.where(placed > 0)
        zx0, zx1 = int(xs.min()), int(xs.max())
        zy0, zy1 = int(ys.min()), int(ys.max())
    else:
        # fallback: zoom to bite bbox
        zx0, zy0 = bx, by
        zx1, zy1 = bx + bw - 1, by + bh - 1

    zx0 = max(0, zx0 - zoom_pad)
    zy0 = max(0, zy0 - zoom_pad)
    zx1 = min(W - 1, zx1 + zoom_pad)
    zy1 = min(H - 1, zy1 + zoom_pad)

    # -----------------------------
    # Plot
    # -----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=220)
    for ax in axes:
        ax.axis("off")

    axes[0].set_title("RAW LOCAL union (bite frame)", fontsize=9)
    axes[1].set_title("FULL placement on GT mask", fontsize=9)
    axes[2].set_title("ZOOM crop (union or bbox)", fontsize=9)

    # panel 0: local
    axes[0].imshow(local_union, cmap="hot", interpolation="nearest", alpha=0.9)
    axes[0].add_patch(plt.Rectangle((0, 0), bw, bh, fill=False, edgecolor="lime", linewidth=2))

    # overlay segs projected into bite frame
    for S in (segs or []):
        S = np.asarray(S, float)
        if S.ndim == 2 and len(S) >= 2:
            axes[0].plot(S[:, 0] - bx, S[:, 1] - by, color="cyan", lw=2)

    # panel 1: full placement on gt mask (use gt_mask as context)
    axes[1].imshow((gt_mask > 0).astype(np.uint8), cmap="gray", interpolation="nearest")
    axes[1].imshow(placed, cmap="hot", interpolation="nearest", alpha=0.9)
    axes[1].add_patch(plt.Rectangle((bx, by), bw, bh, fill=False, edgecolor="lime", linewidth=2))
    for S in (segs or []):
        S = np.asarray(S, float)
        if S.ndim == 2 and len(S) >= 2:
            axes[1].plot(S[:, 0], S[:, 1], color="cyan", lw=1.2)

    # panel 2: zoom
    zoom_gt = (gt_mask[zy0:zy1+1, zx0:zx1+1] > 0).astype(np.uint8)
    zoom_pl = placed[zy0:zy1+1, zx0:zx1+1]
    axes[2].imshow(zoom_gt, cmap="gray", interpolation="nearest")
    axes[2].imshow(zoom_pl, cmap="hot", interpolation="nearest", alpha=0.9)
    axes[2].add_patch(
        plt.Rectangle((bx - zx0, by - zy0), bw, bh, fill=False, edgecolor="lime", linewidth=2)
    )
    for S in (segs or []):
        S = np.asarray(S, float)
        if S.ndim == 2 and len(S) >= 2:
            axes[2].plot(S[:, 0] - zx0, S[:, 1] - zy0, color="cyan", lw=1.6)

    fig.suptitle(f"GT dominance bite debug â€” base={base_name} ccid={ccid}", fontsize=11)

    os.makedirs(out_dir, exist_ok=True)
    tag = f"ccid{ccid}_" + "_".join([str(m) for m in members])
    out = os.path.join(out_dir, f"gt_sup_dom_bite_debug_{base_name}_{tag}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print(f"[GT_SUP DOMDBG] wrote {out}")


def export_gt_centering_metrics(
    *,
    base_name: str,
    save_root: str,
    final_entries: list,
    combined_member_ids=None,
):
    """
    Compute per-crack GT alignment metrics:
      - manual vs centered (DT ridge)
      - manual vs depth_distridge

    Outputs:
      - backward-compatible files under supervision/<image>/analysis
      - separated files under:
          supervision/<image>/gt_centering
          supervision/<image>/depth_distridge
    """
    import os
    import numpy as np
    import pandas as pd
    from helpers.metrics import compute_midline_metrics
    from helpers.present_plots import plot_rs3_midline_diagnostics

    sup_img_dir = os.path.join(save_root, "supervision", base_name)
    analysis_dir = os.path.join(sup_img_dir, "analysis")
    gt_centering_dir = os.path.join(sup_img_dir, "gt_centering")
    depth_distridge_dir = os.path.join(sup_img_dir, "depth_distridge")
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(gt_centering_dir, exist_ok=True)
    os.makedirs(depth_distridge_dir, exist_ok=True)

    rows = []
    combined_member_ids = {str(x) for x in (combined_member_ids or [])}

    def _coerce_seg_list(entry, *, seg_keys=(), packed_keys=()):
        for k in seg_keys:
            val = entry.get(k, None)
            if isinstance(val, list):
                out = [np.asarray(s, float) for s in val if s is not None and len(s) >= 2]
                if out:
                    return out
        for k in packed_keys:
            val = entry.get(k, None)
            if val is None:
                continue
            arr = np.asarray(val, float)
            if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) >= 2:
                return [arr]
            segs = _split_midline_packed(val)
            segs = [np.asarray(s, float) for s in segs if s is not None and len(s) >= 2]
            if segs:
                return segs
        return []

    for entry in (final_entries or []):
        cid = str(entry.get("id", ""))
        kind = str(entry.get("kind", "unknown"))
        method_variants = entry.get("method_variants", {}) if isinstance(entry.get("method_variants", {}), dict) else {}

        method_cmp_specs = [
            ("manual_vs_dt", "dt"),
            ("manual_vs_dt_depth", "dt_depth"),
            ("manual_vs_dt_ridge_valley", "dt_ridge_valley"),
            ("manual_vs_dt_ridge_valley_depth", "dt_ridge_valley_depth"),
        ]

        if kind == "atomic":
            manual_mid = np.asarray(entry.get("midline", []), float)
            candidate_pairs = []
            if method_variants:
                for cmp_label, mkey in method_cmp_specs:
                    mm = method_variants.get(mkey, {}) if isinstance(method_variants.get(mkey, {}), dict) else {}
                    m_mid = mm.get("midline", None)
                    m_arr = np.asarray(m_mid, float) if m_mid is not None else np.empty((0, 2), float)
                    candidate_pairs.append((cmp_label, mkey, manual_mid, m_arr))
            else:
                centered_mid = np.asarray(
                    entry.get("centered_midline", entry.get("midline_auto_centered", [])),
                    float,
                )
                depth_mid = np.asarray(
                    entry.get("depth_midline", entry.get("midline_depth_centered", [])),
                    float,
                )
                candidate_pairs = [
                    ("manual_vs_dt", "dt", manual_mid, centered_mid),
                    ("manual_vs_dt_ridge_valley_depth", "dt_ridge_valley_depth", manual_mid, depth_mid),
                ]
        elif kind == "combined":
            manual_parts = _coerce_seg_list(
                entry,
                seg_keys=("midline_segments",),
                packed_keys=("midline",),
            )
            manual_mid = np.vstack(manual_parts) if manual_parts else np.empty((0, 2), float)
            candidate_pairs = []
            if method_variants:
                for cmp_label, mkey in method_cmp_specs:
                    mm = method_variants.get(mkey, {}) if isinstance(method_variants.get(mkey, {}), dict) else {}
                    parts = _coerce_seg_list(
                        mm,
                        seg_keys=("midline_segments",),
                        packed_keys=("midline",),
                    )
                    m_mid = np.vstack(parts) if parts else np.empty((0, 2), float)
                    candidate_pairs.append((cmp_label, mkey, manual_mid, m_mid))
            else:
                centered_parts = _coerce_seg_list(
                    entry,
                    seg_keys=("centered_midline_segments", "midline_segments_auto_centered"),
                    packed_keys=("centered_midline", "midline_auto_centered"),
                )
                depth_parts = _coerce_seg_list(
                    entry,
                    seg_keys=("depth_midline_segments",),
                    packed_keys=("depth_midline", "midline_depth_centered"),
                )
                centered_mid = np.vstack(centered_parts) if centered_parts else np.empty((0, 2), float)
                depth_mid = np.vstack(depth_parts) if depth_parts else np.empty((0, 2), float)
                candidate_pairs = [
                    ("manual_vs_dt", "dt", manual_mid, centered_mid),
                    ("manual_vs_dt_ridge_valley_depth", "dt_ridge_valley_depth", manual_mid, depth_mid),
                ]
        else:
            continue

        for cmp_label, variant_id, manual_mid, other_mid in candidate_pairs:
            if (
                manual_mid is None or other_mid is None
                or len(manual_mid) < 2
                or len(other_mid) < 2
            ):
                continue

            mm = compute_midline_metrics(
                auto_xy=other_mid,
                man_xy=manual_mid,
                tau=3.0,
            )

            nn = float(mm.get("nn_mean_bidirectional", np.nan))
            hd = float(mm.get("hausdorff_max", np.nan))
            cov = float(mm.get("coverage_min", np.nan))
            score_mid = np.nan
            if np.isfinite(nn) and np.isfinite(hd) and np.isfinite(cov):
                score_mid = float(
                    np.log1p(max(nn, 0.0))
                    + 0.5 * np.log1p(max(hd, 0.0))
                    + (1.0 - float(np.clip(cov, 0.0, 1.0)))
                )

            row = {
                "image": base_name,
                "crack_id": cid,
                "crack_kind": kind,
                "comparison_label": cmp_label,
                "variant_id": variant_id,
                "is_atomic": int(kind == "atomic"),
                "is_combined": int(kind == "combined"),
                "is_noncombined_atomic": int(kind == "atomic" and cid not in combined_member_ids),
                "geometry_type": "gt_alignment",
                "length_px": float(np.sum(np.hypot(np.diff(manual_mid[:, 0]), np.diff(manual_mid[:, 1])))),
                "os_mode": variant_id,
                "g11": np.nan,
                "g22": np.nan,
                "g33": np.nan,
                "score_mid": score_mid,
                **mm,
            }
            rows.append(row)

            # backward-compatible aliases
            if str(cmp_label) == "manual_vs_dt":
                row2 = dict(row)
                row2["comparison_label"] = "manual_vs_centered"
                row2["variant_id"] = "centered_gt"
                rows.append(row2)
            if str(cmp_label) == "manual_vs_dt_ridge_valley_depth":
                row2 = dict(row)
                row2["comparison_label"] = "manual_vs_depth_distridge"
                row2["variant_id"] = "centered_depth_gt"
                rows.append(row2)

    df_all = pd.DataFrame(rows)
    if df_all.empty:
        out_csv = os.path.join(analysis_dir, "gt_alignment_metrics.csv")
        df_all.to_csv(out_csv, index=False)
        centered_csv = os.path.join(analysis_dir, "gt_centering_metrics.csv")
        df_all.to_csv(centered_csv, index=False)
        depth_csv = os.path.join(analysis_dir, "gt_depth_distridge_metrics.csv")
        df_all.to_csv(depth_csv, index=False)
        df_all.to_csv(os.path.join(gt_centering_dir, "midline_metrics.csv"), index=False)
        df_all.to_csv(os.path.join(gt_centering_dir, "weighted_summary.csv"), index=False)
        df_all.to_csv(os.path.join(depth_distridge_dir, "midline_metrics.csv"), index=False)
        df_all.to_csv(os.path.join(depth_distridge_dir, "weighted_summary.csv"), index=False)
        print(f"[GT_SUP] wrote alignment metrics -> {out_csv}")
        print(f"[GT_SUP] wrote centering metrics -> {centered_csv}")
        print(f"[GT_SUP] wrote depth_distridge metrics -> {depth_csv}")
        return df_all

    def _length_weighted_means(sub_df, metric_cols):
        if sub_df is None or sub_df.empty:
            return {c: np.nan for c in metric_cols}
        w_all = np.asarray(sub_df["length_px"], float)
        out = {}
        for c in metric_cols:
            x = np.asarray(sub_df[c], float)
            ok = np.isfinite(x) & np.isfinite(w_all) & (w_all > 0)
            if np.any(ok):
                out[c] = float(np.sum(x[ok] * w_all[ok]) / np.sum(w_all[ok]))
            else:
                out[c] = np.nan
        return out

    metric_cols = [
        c for c in [
            "score_mid",
            "nn_mean_bidirectional",
            "hausdorff_max",
            "coverage_min",
            "hausdorff_p95",
            "frechet_discrete_ds",
            "mean_tan_angle_error_deg",
        ]
        if c in df_all.columns
    ]

    summary_rows_all = []
    for cmp_label, dcmp in df_all.groupby("comparison_label", dropna=False):
        dcmp = dcmp.copy()
        total_count_atomic = int((dcmp["is_atomic"] == 1).sum())
        total_count_combined_plus_noncombined_atomic = int(
            ((dcmp["is_combined"] == 1) | (dcmp["is_noncombined_atomic"] == 1)).sum()
        )
        total_length_atomic_px = float(dcmp.loc[dcmp["is_atomic"] == 1, "length_px"].sum())
        total_length_combined_plus_noncombined_atomic_px = float(
            dcmp.loc[(dcmp["is_combined"] == 1) | (dcmp["is_noncombined_atomic"] == 1), "length_px"].sum()
        )
        dcmp["total_count_atomic"] = total_count_atomic
        dcmp["total_count_combined_plus_noncombined_atomic"] = total_count_combined_plus_noncombined_atomic
        dcmp["total_length_atomic_px"] = total_length_atomic_px
        dcmp["total_length_combined_plus_noncombined_atomic_px"] = (
            total_length_combined_plus_noncombined_atomic_px
        )

        df_atomic = dcmp[dcmp["is_atomic"] == 1].copy()
        df_combo_plus_noncombo = dcmp[(dcmp["is_combined"] == 1) | (dcmp["is_noncombined_atomic"] == 1)].copy()

        w_atomic = _length_weighted_means(df_atomic, metric_cols)
        w_combo = _length_weighted_means(df_combo_plus_noncombo, metric_cols)

        for c in metric_cols:
            dcmp[f"lwmean_atomic_{c}"] = float(w_atomic.get(c, np.nan))
            dcmp[f"lwmean_combined_plus_noncombined_atomic_{c}"] = float(w_combo.get(c, np.nan))

        summary_rows_all.append({
            "image": base_name,
            "comparison_label": str(cmp_label),
            "group": "atomic",
            "count": int(len(df_atomic)),
            "total_length_px": float(df_atomic["length_px"].sum()) if not df_atomic.empty else 0.0,
            **{f"lwmean_{c}": float(w_atomic.get(c, np.nan)) for c in metric_cols},
        })
        summary_rows_all.append({
            "image": base_name,
            "comparison_label": str(cmp_label),
            "group": "combined_plus_noncombined_atomic",
            "count": int(len(df_combo_plus_noncombo)),
            "total_length_px": float(df_combo_plus_noncombo["length_px"].sum()) if not df_combo_plus_noncombo.empty else 0.0,
            **{f"lwmean_{c}": float(w_combo.get(c, np.nan)) for c in metric_cols},
        })

        # Write diagnostics per comparison label (analysis + separated tracks).
        plot_rs3_midline_diagnostics(
            df_all=dcmp,
            out_dir=os.path.join(analysis_dir, "diagnostics", str(cmp_label)),
            selected_family=None,
            title_suffix=f"GT Alignment Audit ({cmp_label})",
        )
        if str(cmp_label) in {"manual_vs_centered", "manual_vs_dt"}:
            plot_rs3_midline_diagnostics(
                df_all=dcmp,
                out_dir=os.path.join(gt_centering_dir, "diagnostics"),
                selected_family=None,
                title_suffix="GT Centering Audit",
            )
        elif str(cmp_label) in {
            "manual_vs_depth_distridge",
            "manual_vs_depth",
            "manual_vs_dt_depth",
            "manual_vs_dt_ridge_valley_depth",
        }:
            plot_rs3_midline_diagnostics(
                df_all=dcmp,
                out_dir=os.path.join(depth_distridge_dir, "diagnostics"),
                selected_family=None,
                title_suffix="Depth DistRidge Audit",
            )

    # New all-comparison files.
    out_csv_all = os.path.join(analysis_dir, "gt_alignment_metrics.csv")
    df_all.to_csv(out_csv_all, index=False)
    print(f"[GT_SUP] wrote alignment metrics -> {out_csv_all}")

    summary_csv_all = os.path.join(analysis_dir, "gt_alignment_metrics_weighted_summary.csv")
    pd.DataFrame(summary_rows_all).to_csv(summary_csv_all, index=False)
    print(f"[GT_SUP] wrote alignment weighted summary -> {summary_csv_all}")

    # Backward-compatible centered-only files under analysis/.
    df_centered = df_all[df_all["comparison_label"].astype(str) == "manual_vs_centered"].copy()
    out_csv_centered = os.path.join(analysis_dir, "gt_centering_metrics.csv")
    df_centered.to_csv(out_csv_centered, index=False)
    print(f"[GT_SUP] wrote centering metrics -> {out_csv_centered}")

    centered_summary_rows = [
        r for r in summary_rows_all
        if str(r.get("comparison_label", "")) == "manual_vs_centered"
    ]
    summary_csv_centered = os.path.join(analysis_dir, "gt_centering_metrics_weighted_summary.csv")
    pd.DataFrame(centered_summary_rows).to_csv(summary_csv_centered, index=False)
    print(f"[GT_SUP] wrote weighted centering summary -> {summary_csv_centered}")

    # Depth-distridge explicit files under analysis/.
    df_depth = df_all[
        df_all["comparison_label"].astype(str).isin(["manual_vs_depth_distridge", "manual_vs_depth"])
    ].copy()
    out_csv_depth = os.path.join(analysis_dir, "gt_depth_distridge_metrics.csv")
    df_depth.to_csv(out_csv_depth, index=False)
    print(f"[GT_SUP] wrote depth_distridge metrics -> {out_csv_depth}")

    depth_summary_rows = [
        r for r in summary_rows_all
        if str(r.get("comparison_label", "")) in {"manual_vs_depth_distridge", "manual_vs_depth"}
    ]
    summary_csv_depth = os.path.join(analysis_dir, "gt_depth_distridge_metrics_weighted_summary.csv")
    pd.DataFrame(depth_summary_rows).to_csv(summary_csv_depth, index=False)
    print(f"[GT_SUP] wrote weighted depth_distridge summary -> {summary_csv_depth}")

    # Separated track folders.
    df_centered.to_csv(os.path.join(gt_centering_dir, "midline_metrics.csv"), index=False)
    pd.DataFrame(centered_summary_rows).to_csv(
        os.path.join(gt_centering_dir, "weighted_summary.csv"), index=False
    )
    df_depth.to_csv(os.path.join(depth_distridge_dir, "midline_metrics.csv"), index=False)
    pd.DataFrame(depth_summary_rows).to_csv(
        os.path.join(depth_distridge_dir, "weighted_summary.csv"), index=False
    )

    return df_all


# ============================================================
# MAIN EXPORT FUNCTION
# ============================================================
def export_gt_supervision_for_image(
    *,
    base_name: str,
    save_root: str,
    original_image: np.ndarray,
    H: int,
    W: int,
    atomic: dict,
    combined_groups: dict | None,
    gt_mask: np.ndarray,
    depth_full: np.ndarray | None = None,
    depth_local_crops: dict | None = None,
    enable_auto_centering: bool = True,
    auto_centering_debug: bool = True,
    auto_centering_window_half_size: int = 75,
    auto_centering_iters: int = 30,
    auto_centering_step_px: float = 0.3,
    auto_centering_domain_atomic: str = "terr_and_mask",
    auto_centering_domain_combined: str = "terr_and_mask",
):
    sup_root = os.path.join(save_root, "supervision", base_name)
    if os.path.isdir(sup_root):
        print(f"[GT_SUP] wiping existing supervision folder: {sup_root}")
        shutil.rmtree(sup_root)
    os.makedirs(sup_root, exist_ok=True)

    #mask_root = os.path.join(sup_root, "masks")
    atomic_crop_root = os.path.join(sup_root, "atomic_crops")
    combined_crop_root = os.path.join(sup_root, "combined_crops")
    auto_center_root = os.path.join(sup_root, "auto_center_debug")
    #os.makedirs(mask_root, exist_ok=True)
    os.makedirs(atomic_crop_root, exist_ok=True)
    os.makedirs(combined_crop_root, exist_ok=True)
    if enable_auto_centering and auto_centering_debug:
        os.makedirs(auto_center_root, exist_ok=True)

    gt_bin = (gt_mask > 0).astype(np.uint8)
    num_cc, cc_labels = cv2.connectedComponents(gt_bin, 8)
    print(f"[GT_SUP] GT connected components: {num_cc-1}")

    combined_groups = combined_groups or {}
    combined_flat = {str(m) for g in combined_groups.values() for m in g.get("members", [])}

    final_entries = []
    gt_sup_diag = {
        "atomic_total": 0,
        "atomic_added": 0,
        "atomic_skip_bad_midline": 0,
        "atomic_skip_no_cc_label": 0,
        "combined_total": 0,
        "combined_added": 0,
        "combined_skip_no_members": 0,
        "combined_skip_no_cc_label": 0,
        "combined_skip_no_dominant_segs": 0,
        "combined_skip_empty_after_canon": 0,
    }
    timing_totals = {
        "atomic_compute_sec": 0.0,
        "noncombined_atomic_compute_sec": 0.0,
        "atomic_centering_sec": 0.0,
        "noncombined_atomic_centering_sec": 0.0,
        "combined_compute_sec": 0.0,
        "combined_centering_sec": 0.0,
        "dt_compute_s": 0.0,
        "centered_snap_s": 0.0,
        "depth_align_s": 0.0,
        "depth_recess_s": 0.0,
        "depth_costmap_s": 0.0,
        "depth_dijkstra_s": 0.0,
        "depth_postprocess_s": 0.0,
        "normals_centered_s": 0.0,
        "normals_depth_s": 0.0,
    }

    def _sum_centering_seconds(timing_blob):
        if not isinstance(timing_blob, dict):
            return 0.0
        methods_blob = timing_blob.get("methods", {}) if isinstance(timing_blob.get("methods", {}), dict) else {}
        if methods_blob:
            s = 0.0
            for _k, rec in methods_blob.items():
                if not isinstance(rec, dict):
                    continue
                s += float(rec.get("total_s", 0.0) or 0.0)
            return float(s)
        dt = timing_blob.get("dt", {}) if isinstance(timing_blob.get("dt", {}), dict) else {}
        ctr = timing_blob.get("centered", {}) if isinstance(timing_blob.get("centered", {}), dict) else {}
        dep = timing_blob.get("depth", {}) if isinstance(timing_blob.get("depth", {}), dict) else {}
        nrm = timing_blob.get("normals", {}) if isinstance(timing_blob.get("normals", {}), dict) else {}
        return float(
            float(dt.get("compute_s", 0.0) or 0.0)
            + float(ctr.get("snap_s", 0.0) or 0.0)
            + float(dep.get("depth_align_s", 0.0) or 0.0)
            + float(dep.get("recess_s", 0.0) or 0.0)
            + float(dep.get("costmap_s", 0.0) or 0.0)
            + float(dep.get("dijkstra_s", 0.0) or 0.0)
            + float(dep.get("postprocess_s", 0.0) or 0.0)
            + float(nrm.get("centered_s", 0.0) or 0.0)
            + float(nrm.get("depth_s", 0.0) or 0.0)
        )

    def _accumulate_timing_blob(timing_blob):
        if not isinstance(timing_blob, dict):
            return
        methods_blob = timing_blob.get("methods", {}) if isinstance(timing_blob.get("methods", {}), dict) else {}
        if methods_blob:
            m_dt = methods_blob.get("dt", {}) if isinstance(methods_blob.get("dt", {}), dict) else {}
            m_depth = methods_blob.get("dt_ridge_valley_depth", {}) if isinstance(methods_blob.get("dt_ridge_valley_depth", {}), dict) else {}
            if not m_depth:
                m_depth = methods_blob.get("dt_depth", {}) if isinstance(methods_blob.get("dt_depth", {}), dict) else {}
            timing_totals["dt_compute_s"] += float(m_dt.get("dt_compute_s", 0.0) or 0.0)
            timing_totals["centered_snap_s"] += float(m_dt.get("dijkstra_s", 0.0) or 0.0)
            timing_totals["depth_align_s"] += float(m_depth.get("depth_align_s", 0.0) or 0.0)
            timing_totals["depth_recess_s"] += float(m_depth.get("depth_recess_s", 0.0) or 0.0)
            timing_totals["depth_costmap_s"] += float(m_depth.get("costmap_s", 0.0) or 0.0)
            timing_totals["depth_dijkstra_s"] += float(m_depth.get("dijkstra_s", 0.0) or 0.0)
            timing_totals["depth_postprocess_s"] += float(m_depth.get("postprocess_s", 0.0) or 0.0)
            timing_totals["normals_centered_s"] += float(m_dt.get("normals_s", 0.0) or 0.0)
            timing_totals["normals_depth_s"] += float(m_depth.get("normals_s", 0.0) or 0.0)
            return
        dt = timing_blob.get("dt", {}) if isinstance(timing_blob.get("dt", {}), dict) else {}
        ctr = timing_blob.get("centered", {}) if isinstance(timing_blob.get("centered", {}), dict) else {}
        dep = timing_blob.get("depth", {}) if isinstance(timing_blob.get("depth", {}), dict) else {}
        nrm = timing_blob.get("normals", {}) if isinstance(timing_blob.get("normals", {}), dict) else {}
        timing_totals["dt_compute_s"] += float(dt.get("compute_s", 0.0) or 0.0)
        timing_totals["centered_snap_s"] += float(ctr.get("snap_s", 0.0) or 0.0)
        timing_totals["depth_align_s"] += float(dep.get("depth_align_s", 0.0) or 0.0)
        timing_totals["depth_recess_s"] += float(dep.get("recess_s", 0.0) or 0.0)
        timing_totals["depth_costmap_s"] += float(dep.get("costmap_s", 0.0) or 0.0)
        timing_totals["depth_dijkstra_s"] += float(dep.get("dijkstra_s", 0.0) or 0.0)
        timing_totals["depth_postprocess_s"] += float(dep.get("postprocess_s", 0.0) or 0.0)
        timing_totals["normals_centered_s"] += float(nrm.get("centered_s", 0.0) or 0.0)
        timing_totals["normals_depth_s"] += float(nrm.get("depth_s", 0.0) or 0.0)

    method_style = {
        "dt": {"slug": "dt", "label": "DT", "compare_label": "DT Midline", "color": "cyan"},
        "dt_depth": {"slug": "dt_depth", "label": "DT + Depth", "compare_label": "DT + Depth Midline", "color": "magenta"},
        "dt_ridge_valley": {"slug": "dt_ridge_valley", "label": "DT + Ridge/Valley", "compare_label": "DT + Ridge/Valley Midline", "color": "deepskyblue"},
        "dt_ridge_valley_depth": {"slug": "dt_ridge_valley_depth", "label": "DT + Ridge/Valley + Depth", "compare_label": "DT + Ridge/Valley + Depth Midline", "color": "orange"},
    }

    def _variant_to_json(m):
        if not isinstance(m, dict):
            return {"midline": None, "normals": None, "normals_diag": {}, "timing": {}, "meta": {"reason": "missing_method"}}
        mid = m.get("midline", None)
        mid_list = np.asarray(mid, float).tolist() if mid is not None else None
        return {
            "midline": mid_list,
            "normals": m.get("normals", None),
            "normals_diag": m.get("normals_diag", {}) if isinstance(m.get("normals_diag", {}), dict) else {},
            "timing": m.get("timing", {}) if isinstance(m.get("timing", {}), dict) else {},
            "meta": m.get("meta", {}) if isinstance(m.get("meta", {}), dict) else {},
        }

    def _legacy_timing_from_methods(methods):
        m1 = methods.get("dt", {}) if isinstance(methods.get("dt", {}), dict) else {}
        m4 = methods.get("dt_ridge_valley_depth", {}) if isinstance(methods.get("dt_ridge_valley_depth", {}), dict) else {}
        if not m4:
            m4 = methods.get("dt_depth", {}) if isinstance(methods.get("dt_depth", {}), dict) else {}
        return {
            "dt": {"compute_s": float((m1.get("timing", {}) or {}).get("dt_compute_s", 0.0))},
            "centered": {"snap_s": float((m1.get("timing", {}) or {}).get("dijkstra_s", 0.0))},
            "depth": {
                "depth_align_s": float((m4.get("timing", {}) or {}).get("depth_align_s", 0.0)),
                "recess_s": float((m4.get("timing", {}) or {}).get("depth_recess_s", 0.0)),
                "costmap_s": float((m4.get("timing", {}) or {}).get("costmap_s", 0.0)),
                "dijkstra_s": float((m4.get("timing", {}) or {}).get("dijkstra_s", 0.0)),
                "refine_s": float((m4.get("timing", {}) or {}).get("refine_s", 0.0)),
                "postprocess_s": float((m4.get("timing", {}) or {}).get("postprocess_s", 0.0)),
            },
            "normals": {
                "centered_s": float((m1.get("timing", {}) or {}).get("normals_s", 0.0)),
                "depth_s": float((m4.get("timing", {}) or {}).get("normals_s", 0.0)),
            },
            "methods": {
                k: (v.get("timing", {}) if isinstance(v, dict) else {})
                for k, v in (methods or {}).items()
            },
        }

    def _lookup_local_depth_for_atomic(atomic_id):
        if not isinstance(depth_local_crops, dict):
            return None
        keys = [atomic_id, str(atomic_id)]
        if str(atomic_id).isdigit():
            keys.append(int(str(atomic_id)))
        for k in keys:
            if k in depth_local_crops:
                return depth_local_crops.get(k)
        return None

    def _method_has_required_priors(method_key, mv):
        if not isinstance(mv, dict):
            return False
        mxy = mv.get("midline", None)
        if mxy is None:
            return False
        dbg = mv.get("debug", {}) if isinstance(mv.get("debug", {}), dict) else {}
        has_depth = dbg.get("recess_norm") is not None
        has_rgb = (dbg.get("ridge_norm") is not None) or (dbg.get("rgb_cue_norm") is not None)
        mk = str(method_key)
        if mk == "dt_depth" and not has_depth:
            return False
        if mk == "dt_ridge_valley" and not has_rgb:
            return False
        if mk == "dt_ridge_valley_depth" and (not has_rgb or not has_depth):
            return False
        return True

    def _extract_selected_costmap(mv):
        dbg = mv.get("debug", {}) if isinstance(mv.get("debug", {}), dict) else {}
        costmaps_dbg = dbg.get("costmaps", {}) if isinstance(dbg.get("costmaps", {}), dict) else {}
        meta = mv.get("meta", {}) if isinstance(mv.get("meta", {}), dict) else {}
        cost_meta = meta.get("costmap", {}) if isinstance(meta.get("costmap", {}), dict) else {}
        sel_key = str(cost_meta.get("selected_cost_key", "dt"))
        sel_cost = costmaps_dbg.get(sel_key, None)
        if sel_cost is None:
            sel_cost = dbg.get("costmap", None)
        if sel_cost is None and isinstance(costmaps_dbg, dict) and "dt" in costmaps_dbg:
            sel_key = "dt"
            sel_cost = costmaps_dbg.get("dt")
        return sel_key, sel_cost, dbg, costmaps_dbg

    def _viz_territory_from_costmap(costmap, crack_mask_u8, pct=60.0):
        cm = np.asarray(costmap, np.float32)
        m = (np.asarray(crack_mask_u8) > 0)
        out = np.zeros_like(m, dtype=np.uint8)
        valid = m & np.isfinite(cm) & (cm < np.float32(1e9))
        out[valid] = 1
        return out.astype(np.uint8)

    def _dump_compare_arrays(dbg_tag, *, sel_cost, crack_mask_local, territory_local):
        if sel_cost is None:
            return
        try:
            os.makedirs("/tmp", exist_ok=True)
            sc = np.asarray(sel_cost, np.float32)
            np.save(f"/tmp/{dbg_tag}_COST_PANEL_cost.npy", sc)
            np.save(
                f"/tmp/{dbg_tag}_COST_PANEL_domain.npy",
                (np.asarray(crack_mask_local) > 0).astype(np.uint8),
            )
            np.save(f"/tmp/{dbg_tag}_MANUAL_cost.npy", sc)
            if territory_local is not None:
                np.save(
                    f"/tmp/{dbg_tag}_MANUAL_territory.npy",
                    np.asarray(territory_local, np.uint8),
                )
            print(f"[{dbg_tag}] cost sum: {float(np.sum(sc))}")
            print(f"[{dbg_tag}] cost nonzero: {int(np.sum(sc > 0))}")
            if territory_local is not None:
                print(f"[{dbg_tag}] territory sum: {int(np.sum(np.asarray(territory_local) > 0))}")
        except Exception as _e:
            print(f"[{dbg_tag}] dump failed: {_e}")

    right_title_by_cost_key = {
        "dt": "DT-preferred region",
        "dt_depth": "DT + depth preferred region",
        "dt_ridge_valley": "DT + ridge/valley preferred region",
        "dt_ridge_valley_depth": "DT + ridge/valley + depth preferred region",
    }

    def _canonicalize_segments_with_meta(segs_in, meta_in, *, label):
        segs_valid = [np.asarray(s, float) for s in (segs_in or []) if s is not None and len(s) >= 2]
        if not segs_valid:
            return [], []

        meta = list(meta_in or [])
        if len(meta) != len(segs_valid):
            mm = []
            for i in range(len(segs_valid)):
                d = meta[i] if i < len(meta) and isinstance(meta[i], dict) else {}
                mm.append(d)
            meta = mm
        for i in range(len(meta)):
            if not isinstance(meta[i], dict):
                meta[i] = {}
            meta[i].setdefault("branch_id", int(meta[i].get("branch_id", 0)))
            meta[i].setdefault("seg_idx", int(meta[i].get("seg_idx", i)))

        out_segs, out_meta = [], []
        branch_ids = sorted({int(m.get("branch_id", 0)) for m in meta})
        for bid in branch_ids:
            pairs = [
                (int((m if isinstance(m, dict) else {}).get("seg_idx", i)), i, np.asarray(S, float), dict(m if isinstance(m, dict) else {}))
                for i, (S, m) in enumerate(zip(segs_valid, meta))
                if int((m if isinstance(m, dict) else {}).get("branch_id", 0)) == bid
            ]
            if not pairs:
                continue
            pairs.sort(key=lambda t: (t[0], t[1]))
            segs_b = [p[2] for p in pairs]
            meta_b = [p[3] for p in pairs]

            segs_b, meta_b = enforce_branch_continuity(segs_b, associated_data=meta_b)
            segs_b, meta_b, flipped_branch = canonicalize_branch_direction(segs_b, associated_data=meta_b)
            if flipped_branch:
                print(f"[CANON][GT_SUP] {label} branch={bid} flipped whole branch orientation")
            assert_direction_consistency(segs_b)

            for j, (S, m) in enumerate(zip(segs_b, meta_b)):
                m2 = dict(m if isinstance(m, dict) else {})
                m2["branch_id"] = int(bid)
                m2["seg_idx"] = int(j)
                out_segs.append(np.asarray(S, float))
                out_meta.append(m2)

        return out_segs, out_meta

    # =====================================================
    # 1) ATOMIC BEFORE MERGE  (USE USER mask_bbox ONLY)
    # =====================================================
    atomic_jobs = []
    for order_i, (cid, cr) in enumerate((atomic or {}).items()):
        gt_sup_diag["atomic_total"] += 1
        scid = str(cid)

        mid_xy = np.asarray(cr.get("midline", []), float)
        if mid_xy.ndim != 2 or len(mid_xy) < 2:
            gt_sup_diag["atomic_skip_bad_midline"] += 1
            continue
        mid_xy, _n, _w, _e1, _e2, cinfo = canonicalize_segment_direction(mid_xy)
        if cinfo.get("flipped", False):
            print(f"[CANON][GT_SUP] atomic {scid} midline flipped to canonical direction")

        bb = cr.get("mask_bbox", None)
        if bb is None or not isinstance(bb, (list, tuple)) or len(bb) != 4:
            raise ValueError(
                f"[GT_SUP] atomic {scid} missing or invalid user mask_bbox: {bb}"
            )
        x, y, w, h = map(int, bb)
        if w <= 0 or h <= 0:
            raise ValueError(
                f"[GT_SUP] atomic {scid} has non-positive mask_bbox: {bb}"
            )

        lbl = _cc_label_for_midline(mid_xy, cc_labels)
        if lbl is None or lbl <= 0:
            gt_sup_diag["atomic_skip_no_cc_label"] += 1
            continue

        crack_mask = (cc_labels == lbl).astype(np.uint8)
        crack_mask_clipped = _clip_mask_to_xywh(crack_mask, [int(x), int(y), int(w), int(h)])

        atomic_jobs.append({
            "order": int(order_i),
            "scid": scid,
            "mid_xy": np.asarray(mid_xy, float),
            "bbox": [int(x), int(y), int(w), int(h)],
            "crack_mask": crack_mask,
            "crack_mask_clipped": crack_mask_clipped,
        })

    def _atomic_job_worker(job):
        scid = str(job["scid"])
        mid_xy = np.asarray(job["mid_xy"], float)
        bb = list(job["bbox"])
        crack_mask = np.asarray(job["crack_mask"], np.uint8)
        crack_mask_clipped = np.asarray(job["crack_mask_clipped"], np.uint8)
        t_atomic_compute0 = time.perf_counter()
        atomic_center_sec = 0.0

        t_manual_normals0 = time.perf_counter()
        manual_normals_diag = {}
        (e1x, e1y, e2x, e2y, widths), _ = normals_from_mask_for_midline(
            mid_xy,
            crack_mask_clipped > 0,
            60,
            diagnostics=manual_normals_diag,
            image_hw=crack_mask_clipped.shape[:2],
            endpoint_mode="atomic",
        )
        manual_normals_s = float(time.perf_counter() - t_manual_normals0)
        diag_brief = _normals_diag_summary(manual_normals_diag)
        print(
            f"[GT_SUP NORMDBG] atomic {scid} manual total={diag_brief['total']} "
            f"valid={diag_brief['valid']} invalid={diag_brief['invalid']} "
            f"invalid_frac={diag_brief['invalid_frac']:.4f} top_reasons={diag_brief['top_reasons']}"
        )

        atomic_entry = {
            "id": scid,
            "kind": "atomic",
            "members": [],
            "mask_bbox": [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])],
            "midline": mid_xy.tolist(),
            "gt_normals": {
                "edge1_x": _arr_to_list(e1x),
                "edge1_y": _arr_to_list(e1y),
                "edge2_x": _arr_to_list(e2x),
                "edge2_y": _arr_to_list(e2y),
                "width_px": _arr_to_list(widths),
            },
            "gt_widths": _arr_to_list(widths),
            "gt_normals_diag": manual_normals_diag,
            "timing": {
                "manual": {
                    "normals_s": float(manual_normals_s),
                }
            },
        }
        atomic_compute_sec = float(time.perf_counter() - t_atomic_compute0)

        if enable_auto_centering:
            terr = build_territory_mask_from_polyline(
                mid_xy=mid_xy,
                crack_mask_u8=crack_mask_clipped,
                window_half_size=int(auto_centering_window_half_size),
                dt_domain_u8=None,
            )
            local_depth_crop = _lookup_local_depth_for_atomic(scid)
            method_res = compute_midline_method_variants_and_normals(
                mid_xy=mid_xy,
                crack_mask_u8=crack_mask_clipped,
                domain_u8=None,
                image_rgb=original_image,
                depth_full=depth_full,
                depth_crop=local_depth_crop,
                depth_bbox_xywh=bb,
                full_image_hw=(int(H), int(W)),
                max_radius=50,
                snap_kwargs={
                    "n_iters": int(auto_centering_iters),
                    "step_px": float(auto_centering_step_px),
                    "keep_endpoints": True,
                },
                endpoint_mode="atomic",
            )
            methods = method_res.get("methods", {}) if isinstance(method_res.get("methods", {}), dict) else {}
            m1 = methods.get("dt", {}) if isinstance(methods.get("dt", {}), dict) else {}
            m3 = methods.get("dt_depth", {}) if isinstance(methods.get("dt_depth", {}), dict) else {}
            m4 = methods.get("dt_ridge_valley_depth", {}) if isinstance(methods.get("dt_ridge_valley_depth", {}), dict) else {}
            depth_pick = m4 if m4.get("midline", None) is not None else m3
            timing_blob = _legacy_timing_from_methods(methods)
            atomic_center_sec += _sum_centering_seconds({"methods": {k: (v.get("timing", {}) if isinstance(v, dict) else {}) for k, v in methods.items()}})

            atomic_entry["method_variants"] = {
                k: _variant_to_json(v)
                for k, v in methods.items()
            }

            centered_xy = np.asarray(m1.get("midline", mid_xy), float)
            centered_normals = m1.get("normals", {}) if isinstance(m1.get("normals", {}), dict) else {}
            centered_normals_diag = m1.get("normals_diag", {}) if isinstance(m1.get("normals_diag", {}), dict) else {}

            atomic_entry["centered_midline"] = np.asarray(centered_xy, float).tolist()
            atomic_entry["centered_normals"] = centered_normals
            atomic_entry["midline_auto_centered"] = np.asarray(centered_xy, float).tolist()
            atomic_entry["gt_normals_auto_centered"] = centered_normals
            atomic_entry["gt_widths_auto_centered"] = centered_normals.get("width_px", [])
            atomic_entry["centered_normals_diag"] = centered_normals_diag
            atomic_entry["gt_normals_auto_centered_diag"] = centered_normals_diag

            atomic_entry["depth_midline_local"] = None
            atomic_entry["depth_midline_global"] = None
            if m3.get("midline", None) is not None:
                atomic_entry["depth_midline_local"] = np.asarray(m3.get("midline"), float).tolist()
            if m4.get("midline", None) is not None:
                atomic_entry["depth_midline_global"] = np.asarray(m4.get("midline"), float).tolist()

            if depth_pick.get("midline", None) is not None:
                dxy = np.asarray(depth_pick.get("midline"), float)
                depth_normals = depth_pick.get("normals", {}) if isinstance(depth_pick.get("normals", {}), dict) else {}
                depth_normals_diag = depth_pick.get("normals_diag", {}) if isinstance(depth_pick.get("normals_diag", {}), dict) else {}
                atomic_entry["depth_midline"] = dxy.tolist()
                atomic_entry["depth_normals"] = depth_normals
                atomic_entry["depth_normals_diag"] = depth_normals_diag
                atomic_entry["midline_depth_centered"] = dxy.tolist()
                atomic_entry["gt_normals_depth_centered"] = depth_normals
                atomic_entry["gt_widths_depth_centered"] = depth_normals.get("width_px", [])

            atomic_entry["depth_cost_meta"] = {
                "dt_depth": m3.get("meta", {}) if isinstance(m3.get("meta", {}), dict) else {},
                "dt_ridge_valley_depth": m4.get("meta", {}) if isinstance(m4.get("meta", {}), dict) else {},
            }
            atomic_entry["timing"]["dt"] = timing_blob.get("dt", {})
            atomic_entry["timing"]["centered"] = timing_blob.get("centered", {})
            atomic_entry["timing"]["depth"] = timing_blob.get("depth", {})
            atomic_entry["timing"]["normals"] = timing_blob.get("normals", {})
            atomic_entry["timing"]["methods"] = timing_blob.get("methods", {})

            auto_meta = {
                "enabled": True,
                "domain_mode": str(auto_centering_domain_atomic),
                "snap": {
                    "n_iters": int(auto_centering_iters),
                    "step_px": float(auto_centering_step_px),
                },
                "comparisons": {},
            }
            for mk in METHOD_SPECS.keys():
                mv = methods.get(mk, {}) if isinstance(methods.get(mk, {}), dict) else {}
                mxy = mv.get("midline", None)
                mn = mv.get("normals", {}) if isinstance(mv.get("normals", {}), dict) else {}
                auto_meta["comparisons"][mk] = {
                    **_shift_stats(mid_xy, mxy),
                    **_geometry_disagreement_stats(mid_xy, mxy),
                    **_width_stability_stats(widths, mn.get("width_px", [])),
                    "available": bool(mxy is not None),
                    "reason": (mv.get("meta", {}) or {}).get("reason"),
                }
            auto_meta["manual_vs_centered"] = dict(auto_meta["comparisons"].get("dt", {}))
            auto_meta["manual_vs_depth_distridge"] = dict(auto_meta["comparisons"].get("dt_ridge_valley_depth", {}))
            auto_meta["depth_available"] = bool(
                methods.get("dt_depth", {}).get("midline", None) is not None
                or methods.get("dt_ridge_valley_depth", {}).get("midline", None) is not None
            )
            atomic_entry["auto_centering_meta"] = auto_meta

            if auto_centering_debug:
                atomic_dbg_dir = os.path.join(auto_center_root, "atomic", f"cid_{scid}")
                os.makedirs(atomic_dbg_dir, exist_ok=True)
                manual_invalid = [~np.isfinite(np.asarray(widths, float))]
                for mk in METHOD_SPECS.keys():
                    mv = methods.get(mk, {}) if isinstance(methods.get(mk, {}), dict) else {}
                    if not _method_has_required_priors(mk, mv):
                        continue
                    mxy = mv.get("midline", None)
                    mn = mv.get("normals", {}) if isinstance(mv.get("normals", {}), dict) else {}
                    style = method_style.get(mk, {})
                    mslug = str(style.get("slug", mk))
                    dbg_tag = f"cid{scid}_{mslug}"
                    sel_key, sel_cost, dbg, _costmaps_dbg = _extract_selected_costmap(mv)
                    territory_plot = terr
                    if sel_cost is not None:
                        territory_plot = _viz_territory_from_costmap(sel_cost, crack_mask_clipped, pct=60.0)
                    _dump_compare_arrays(
                        dbg_tag,
                        sel_cost=sel_cost,
                        crack_mask_local=crack_mask_clipped,
                        territory_local=territory_plot,
                    )
                    out_dbg = os.path.join(atomic_dbg_dir, f"manual_vs_{mslug}.png")
                    plot_midline_centering_debug(
                        out_path=out_dbg,
                        crack_mask_u8=crack_mask_clipped,
                        manual_segs=[mid_xy],
                        centered_segs=[np.asarray(mxy, float)],
                        territory_u8=territory_plot,
                        bbox_xywh=atomic_entry.get("mask_bbox"),
                        title=f"atomic {scid}: manual vs {style.get('label', mk)}",
                        invalid_manual_masks=manual_invalid,
                        invalid_center_masks=[~np.isfinite(np.asarray(mn.get("width_px", []), float))],
                        compare_label=style.get("compare_label", mk),
                        compare_color=style.get("color", "cyan"),
                        left_panel_title=f"manual vs {style.get('label', mk)} (mask)",
                        right_panel_title=right_title_by_cost_key.get(sel_key, "Preferred region"),
                    )

                    depth_lbl = "global" if mk in ("dt_depth", "dt_ridge_valley_depth") else None
                    mslug = str(style.get("slug", mk))
                    out_cost = os.path.join(atomic_dbg_dir, f"{mslug}_cost_panel.png")
                    plot_depth_cost_diagnostic(
                        out_path=out_cost,
                        crack_mask_u8=crack_mask_clipped,
                        dt_norm=np.asarray(dbg.get("dt_norm"), np.float32) if dbg.get("dt_norm") is not None else np.zeros_like(crack_mask_clipped, np.float32),
                        depth_norm=np.asarray(dbg.get("depth_norm"), np.float32) if dbg.get("depth_norm") is not None else None,
                        recess_norm=dbg.get("recess_norm"),
                        depth_score=dbg.get("score_for_refine"),
                        costmaps=dbg.get("costmaps"),
                        bbox_xywh=atomic_entry.get("mask_bbox"),
                        title=f"atomic {scid}: cost cues",
                        method_label=style.get("label", mk),
                        depth_label=depth_lbl,
                    )

        return int(job["order"]), atomic_entry, float(atomic_compute_sec), float(atomic_center_sec)

    max_workers = max(1, min(8, os.cpu_count() or 1))
    atomic_results = {}
    if atomic_jobs:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_atomic_job_worker, j) for j in atomic_jobs]
            for fut in as_completed(futs):
                oi, entry, atomic_compute_sec, atomic_center_sec = fut.result()
                atomic_results[int(oi)] = entry
                timing_totals["atomic_compute_sec"] += float(atomic_compute_sec)
                timing_totals["atomic_centering_sec"] += float(atomic_center_sec)
                _accumulate_timing_blob(entry.get("timing", {}))
                if str(entry.get("id", "")) not in combined_flat:
                    timing_totals["noncombined_atomic_compute_sec"] += float(atomic_compute_sec)
                    timing_totals["noncombined_atomic_centering_sec"] += float(atomic_center_sec)

    for oi in sorted(atomic_results.keys()):
        atomic_entry = atomic_results[oi]
        final_entries.append(atomic_entry)
        gt_sup_diag["atomic_added"] += 1
        _cropped_preview(atomic_entry, gt_mask, original_image, atomic_crop_root)

    # =====================================================
    # 2) COMBINED  (USE UNION OF USER mask_bbox ONLY)
    # =====================================================
    for ccid, grp in (combined_groups or {}).items():
        gt_sup_diag["combined_total"] += 1
        members = [str(m) for m in grp.get("members", [])]
        if not members:
            gt_sup_diag["combined_skip_no_members"] += 1
            continue
        t_combined_compute0 = time.perf_counter()
        combined_plot_excluded_sec = 0.0

        # -------------------------------------------------
        # REQUIRED: union of USER-authored atomic bboxes (xywh)
        # -------------------------------------------------
        boxes = []
        for m in members:
            cr = atomic.get(str(m))
            if cr is None:
                raise ValueError(f"[GT_SUP] combined {ccid} missing atomic {m}")

            bb = cr.get("mask_bbox")
            if bb is None or not isinstance(bb, (list, tuple)) or len(bb) != 4:
                raise ValueError(
                    f"[GT_SUP] combined {ccid} atomic {m} has invalid mask_bbox: {bb}"
                )

            x, y, w, h = map(int, bb)
            if w <= 0 or h <= 0:
                raise ValueError(
                    f"[GT_SUP] combined {ccid} atomic {m} has non-positive bbox: {bb}"
                )

            boxes.append((x, y, x + w, y + h))

        # Union in xyXY space
        ux0 = max(0, min(b[0] for b in boxes))
        uy0 = max(0, min(b[1] for b in boxes))
        ux1 = min(W, max(b[2] for b in boxes))
        uy1 = min(H, max(b[3] for b in boxes))

        if ux1 <= ux0 or uy1 <= uy0:
            raise ValueError(
                f"[GT_SUP] combined {ccid} bbox collapses after union: {boxes}"
            )

        # Convert BACK to canonical xywh for storage
        ux = int(ux0)
        uy = int(uy0)
        uw = int(ux1 - ux0)
        uh = int(uy1 - uy0)

        # -------------------------------------------------
        # GT CC label ONLY for normals + dominance logic
        # -------------------------------------------------
        lbl = _cc_label_for_members(members, atomic, cc_labels)
        if lbl is None or lbl <= 0:
            gt_sup_diag["combined_skip_no_cc_label"] += 1
            continue

        crack_mask = (cc_labels == lbl).astype(np.uint8)
        crack_mask_clipped = _clip_mask_to_xywh(crack_mask, [ux, uy, uw, uh])

        # -------------------------------------------------
        # Dominance-selected sub-midlines
        # -------------------------------------------------
        debug_dir = os.path.join(sup_root, "combined_debug")
        tag = f"ccid{ccid}_" + "_".join(members)

        from combiner import dominant_segments_from_group
        segs, dom_meta = dominant_segments_from_group(
            members=members,
            atomic=atomic,
            crack_mask_u8=crack_mask,
            window_half_size=50,
            debug_dir=debug_dir,
            debug_tag=tag,
        )

        if not segs:
            gt_sup_diag["combined_skip_no_dominant_segs"] += 1
            continue

        dom_meta = dom_meta if isinstance(dom_meta, dict) else {}
        seg_meta = dom_meta.get("segments_meta", [])
        segs, seg_meta = _canonicalize_segments_with_meta(segs, seg_meta, label=f"combined {ccid}")
        if not segs:
            gt_sup_diag["combined_skip_empty_after_canon"] += 1
            continue
        dom_meta["segments_meta"] = seg_meta

        # -------------------------------------------------
        # Build per-branch allowed GT masks from:
        #   allowed_i = crack_mask - union(territory_j - bite_j) for j != i
        # GT has no per-branch mask ownership; this only restricts where
        # normals are allowed to exist.
        # -------------------------------------------------
        Hm, Wm = crack_mask_clipped.shape[:2]
        global_mask = (crack_mask_clipped > 0).astype(np.uint8)

        branch_order = dom_meta.get("order", []) or []
        if not branch_order:
            branch_order = sorted({
                int(sm.get("branch_id", -1))
                for sm in (seg_meta or [])
                if int(sm.get("branch_id", -1)) >= 0
            })

        def _decode_packed_mask_local(blob, H, W):
            """
            Decode a packed cropped mask blob using the combined crack bbox.
            Returns full-image uint8 mask.
            """
            full = np.zeros((H, W), np.uint8)
            if not isinstance(blob, dict):
                return full

            bite_meta_local = dom_meta.get("bite", {}) if isinstance(dom_meta.get("bite", {}), dict) else {}
            bbox = bite_meta_local.get("bbox", None)
            if bbox is None or len(bbox) != 4:
                return full

            x0, y0, bw, bh = map(int, bbox)
            shape = blob.get("shape", None)
            b64 = blob.get("packbits_b64", None)
            if shape is None or b64 is None:
                return full

            try:
                import base64
                raw = base64.b64decode(b64.encode("utf-8"))
                bits = np.frombuffer(raw, dtype=np.uint8)
                arr = np.unpackbits(bits)

                hh = int(shape[0])
                ww = int(shape[1])
                n = hh * ww
                if arr.size < n:
                    arr = np.pad(arr, (0, n - arr.size), constant_values=0)

                crop = arr[:n].reshape((hh, ww)).astype(np.uint8)

                y1 = min(H, y0 + hh)
                x1 = min(W, x0 + ww)
                ch = max(0, y1 - y0)
                cw = max(0, x1 - x0)

                if ch > 0 and cw > 0:
                    full[y0:y1, x0:x1] = crop[:ch, :cw]
            except Exception:
                pass

            return full

        # Territory per branch (full-image masks)
        terr_meta = dom_meta.get("branch_territory", {}) if isinstance(dom_meta.get("branch_territory", {}), dict) else {}
        branch_terr_masks = {}
        for bi in branch_order:
            bi_str = str(int(bi))
            terr_blob = terr_meta.get(bi_str, {})
            branch_terr_masks[int(bi)] = _decode_packed_mask_local(terr_blob, Hm, Wm)

        # Bite per branch (full-image masks)
        bite_meta = dom_meta.get("bite", {}) if isinstance(dom_meta.get("bite", {}), dict) else {}
        bite_by_branch = bite_meta.get("by_losing_branch", {}) if isinstance(bite_meta.get("by_losing_branch", {}), dict) else {}

        branch_bite_masks = {}
        for bi in branch_order:
            bi_str = str(int(bi))
            bb = bite_by_branch.get(bi_str, {})
            branch_bite_masks[int(bi)] = _decode_packed_mask_local(bb, Hm, Wm)

        # Allowed mask per branch:
        # remove only the surviving territory of OTHER branches
        branch_allowed_masks = {}
        for bi in branch_order:
            bi = int(bi)
            forbidden = np.zeros((Hm, Wm), np.uint8)
            for sj in branch_order:
                sj = int(sj)
                if sj == bi:
                    continue

                terr_j = (branch_terr_masks.get(sj, np.zeros((Hm, Wm), np.uint8)) > 0).astype(np.uint8)
                bite_j = (branch_bite_masks.get(sj, np.zeros((Hm, Wm), np.uint8)) > 0).astype(np.uint8)

                # only forbid territory that branch sj still effectively owns
                own_j = terr_j & (~bite_j.astype(bool))
                forbidden |= own_j.astype(np.uint8)

            allowed = global_mask & (~forbidden.astype(bool))
            branch_allowed_masks[bi] = allowed.astype(np.uint8)

        # Temporary sanity print for mask sizes
        try:
            for bi in branch_order:
                bi = int(bi)
                terr_px = int(np.count_nonzero(branch_terr_masks.get(bi, 0)))
                bite_px = int(np.count_nonzero(branch_bite_masks.get(bi, 0)))
                allow_px = int(np.count_nonzero(branch_allowed_masks.get(bi, 0)))
                print(f"[GT_SUP MASKDBG] branch={bi} terr_px={terr_px} bite_px={bite_px} allow_px={allow_px}")
        except Exception as _e:
            print(f"[GT_SUP MASKDBG] failed: {_e}")
        
        # -------------------------------------------------
        # DEBUG: dominance bite as-written (RAW, no decode)
        # -------------------------------------------------
        try:
            t_plot0 = time.perf_counter()
            debug_plot_gt_sup_dominance_bite_packed(
                base_name=base_name,
                ccid=ccid,
                members=members,
                dom_meta=dom_meta,
                segs=segs,
                gt_mask=gt_mask,
                out_dir=debug_dir,  # or sup_root, up to you
            )
            combined_plot_excluded_sec += float(time.perf_counter() - t_plot0)

        except Exception as e:
            print(f"[GT_SUP DOMDBG] plot failed: {e}")


        # -------------------------------------------------
        # Compute GT normals per segment
        # -------------------------------------------------
        e1x_list = [None] * len(segs)
        e1y_list = [None] * len(segs)
        e2x_list = [None] * len(segs)
        e2y_list = [None] * len(segs)
        w_list = [None] * len(segs)
        per_seg_manual_normals_diag = [None] * len(segs)

        def _combined_seg_worker(si, S):
            seg_diag = {}
            bi = int(seg_meta[si].get("branch_id", -1)) if si < len(seg_meta) else -1
            mask_use = branch_allowed_masks.get(
                bi,
                (crack_mask_clipped > 0).astype(np.uint8),
            )
            (e1x, e1y, e2x, e2y, widths), _ = normals_from_mask_for_midline(
                S,
                mask_use > 0,
                max_radius=50,
                diagnostics=seg_diag,
                image_hw=mask_use.shape[:2],
                endpoint_mode="combined",
            )
            sdiag_brief = _normals_diag_summary(seg_diag)
            return si, e1x, e1y, e2x, e2y, widths, seg_diag, sdiag_brief

        if segs:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(_combined_seg_worker, si, S) for si, S in enumerate(segs)]
                for fut in as_completed(futs):
                    si, e1x, e1y, e2x, e2y, widths, seg_diag, sdiag_brief = fut.result()
                    print(
                        f"[GT_SUP NORMDBG] combined {ccid} seg={si} manual total={sdiag_brief['total']} "
                        f"valid={sdiag_brief['valid']} invalid={sdiag_brief['invalid']} "
                        f"invalid_frac={sdiag_brief['invalid_frac']:.4f} top_reasons={sdiag_brief['top_reasons']}"
                    )
                    e1x_list[si] = e1x
                    e1y_list[si] = e1y
                    e2x_list[si] = e2x
                    e2y_list[si] = e2y
                    w_list[si] = widths
                    per_seg_manual_normals_diag[si] = seg_diag

        packed_mid = _pack_segs_with_separators(segs)

        tag_name = "_".join(members)
        combined_entry = {
            "id": tag_name,
            "kind": "combined",
            "members": members,
            # ðŸ”´ AUTHORITATIVE bbox = UNION OF USER BOXES (xywh)
            "mask_bbox": [ux, uy, uw, uh],
            "midline": packed_mid,
            "gt_normals": {
                "edge1_x": _pack_arrs_with_none_separators(
                    [_arr_to_list(a) for a in e1x_list]
                ),
                "edge1_y": _pack_arrs_with_none_separators(
                    [_arr_to_list(a) for a in e1y_list]
                ),
                "edge2_x": _pack_arrs_with_none_separators(
                    [_arr_to_list(a) for a in e2x_list]
                ),
                "edge2_y": _pack_arrs_with_none_separators(
                    [_arr_to_list(a) for a in e2y_list]
                ),
                "width_px": _pack_arrs_with_none_separators(
                    [_arr_to_list(a) for a in w_list]
                ),
            },
            "gt_widths": [float(v) for arr in w_list for v in np.asarray(arr, float)],
            "midline_segments": [np.asarray(S, float).tolist() for S in segs],
            "midline_segments_meta": [dict(m) for m in (seg_meta or [])],
            "dominance_meta": dom_meta,
            "gt_normals_diag_per_segment": per_seg_manual_normals_diag,
            "timing": {
                "manual": {},
            },
        }
        timing_totals["combined_compute_sec"] += float(
            time.perf_counter() - t_combined_compute0 - combined_plot_excluded_sec
        )

        if enable_auto_centering:
            centered_segs = []
            ce1x_list, ce1y_list, ce2x_list, ce2y_list, cw_list = [], [], [], [], []
            depth_segs = []
            de1x_list, de1y_list, de2x_list, de2y_list, dw_list = [], [], [], [], []
            shift_all = []
            depth_shift_all = []
            invalid_manual_masks = []
            invalid_center_masks = []
            invalid_depth_masks = []

            per_seg_centered_normals_diag = []
            per_seg_depth_normals_diag = []
            depth_cost_meta_per_segment = []
            method_variant_segments = {
                k: {
                    "midline_segments": [],
                    "midline_segments_meta": [],
                    "normals_diag_per_segment": [],
                    "edge1_x": [],
                    "edge1_y": [],
                    "edge2_x": [],
                    "edge2_y": [],
                    "width_px": [],
                    "timing_sum": {},
                    "meta": METHOD_SPECS.get(k, {}),
                }
                for k in METHOD_SPECS.keys()
            }
            method_cost_debug_blobs = {k: [] for k in METHOD_SPECS.keys()}

            combined_timing_blob = {
                "dt": {"compute_s": 0.0},
                "centered": {"snap_s": 0.0},
                "depth": {
                    "depth_align_s": 0.0,
                    "recess_s": 0.0,
                    "costmap_s": 0.0,
                    "dijkstra_s": 0.0,
                    "postprocess_s": 0.0,
                },
                "normals": {
                    "centered_s": 0.0,
                    "depth_s": 0.0,
                },
            }

            def _acc_nested_timing(dst, src):
                if not (isinstance(dst, dict) and isinstance(src, dict)):
                    return
                for gk, gv in src.items():
                    if isinstance(gv, dict):
                        dst.setdefault(gk, {})
                        for sk, sv in gv.items():
                            try:
                                dst[gk][sk] = float(dst[gk].get(sk, 0.0)) + float(sv or 0.0)
                            except Exception:
                                continue

            for seg_i, (S, w_manual) in enumerate(zip(segs, w_list)):
                S = np.asarray(S, float)
                bi = int(seg_meta[seg_i].get("branch_id", -1)) if seg_i < len(seg_meta) else -1
                mask_use = branch_allowed_masks.get(
                    bi,
                    (crack_mask_clipped > 0).astype(np.uint8),
                )
                terr_i = build_territory_mask_from_polyline(
                    mid_xy=S,
                    crack_mask_u8=mask_use,
                    window_half_size=int(auto_centering_window_half_size),
                    dt_domain_u8=None,
                )
                center_res = compute_midline_method_variants_and_normals(
                    mid_xy=S,
                    crack_mask_u8=mask_use,
                    domain_u8=None,
                    image_rgb=original_image,
                    depth_full=depth_full,
                    depth_bbox_xywh=[ux, uy, uw, uh],
                    full_image_hw=(int(H), int(W)),
                    max_radius=50,
                    snap_kwargs={
                        "n_iters": int(auto_centering_iters),
                        "step_px": float(auto_centering_step_px),
                        "keep_endpoints": True,
                    },
                    endpoint_mode="combined",
                )

                # Build backward-compatible timing accumulator from method family.
                _acc_nested_timing(
                    combined_timing_blob,
                    _legacy_timing_from_methods(
                        center_res.get("methods", {}) if isinstance(center_res.get("methods", {}), dict) else {}
                    ),
                )
                methods_local = center_res.get("methods", {}) if isinstance(center_res.get("methods", {}), dict) else {}
                for mk in METHOD_SPECS.keys():
                    mv = methods_local.get(mk, {}) if isinstance(methods_local.get(mk, {}), dict) else {}
                    mseg = mv.get("midline", None)
                    mnorm = mv.get("normals", {}) if isinstance(mv.get("normals", {}), dict) else {}
                    mdiag = mv.get("normals_diag", {}) if isinstance(mv.get("normals_diag", {}), dict) else {}
                    mtime = mv.get("timing", {}) if isinstance(mv.get("timing", {}), dict) else {}
                    mdebug = mv.get("debug", {}) if isinstance(mv.get("debug", {}), dict) else {}
                    mm = method_variant_segments.get(mk, {})
                    if mseg is not None:
                        mseg = np.asarray(mseg, float)
                    if mseg is not None and mseg.ndim == 2 and len(mseg) >= 2:
                        mm["midline_segments"].append(np.asarray(mseg, float))
                        mm["midline_segments_meta"].append(
                            dict(seg_meta[seg_i]) if seg_i < len(seg_meta) else {"seg_idx": int(seg_i)}
                        )
                        mm["edge1_x"].append(np.asarray(mnorm.get("edge1_x", []), float))
                        mm["edge1_y"].append(np.asarray(mnorm.get("edge1_y", []), float))
                        mm["edge2_x"].append(np.asarray(mnorm.get("edge2_x", []), float))
                        mm["edge2_y"].append(np.asarray(mnorm.get("edge2_y", []), float))
                        mm["width_px"].append(np.asarray(mnorm.get("width_px", []), float))
                        mm["normals_diag_per_segment"].append(mdiag)
                    for tk, tv in mtime.items():
                        try:
                            mm["timing_sum"][tk] = float(mm["timing_sum"].get(tk, 0.0)) + float(tv or 0.0)
                        except Exception:
                            continue
                    if mk in method_cost_debug_blobs and mdebug.get("dt_norm") is not None and _method_has_required_priors(mk, mv):
                        sel_key, sel_cost, _, _ = _extract_selected_costmap(mv)
                        terr_plot_local = mask_use
                        if sel_cost is not None:
                            terr_plot_local = _viz_territory_from_costmap(sel_cost, mask_use, pct=60.0)
                        method_cost_debug_blobs[mk].append((int(seg_i), mdebug, mask_use, str(sel_key), terr_plot_local))

                m1_local = methods_local.get("dt", {}) if isinstance(methods_local.get("dt", {}), dict) else {}
                m4_local = methods_local.get("dt_ridge_valley_depth", {}) if isinstance(methods_local.get("dt_ridge_valley_depth", {}), dict) else {}
                if not m4_local:
                    m4_local = methods_local.get("dt_depth", {}) if isinstance(methods_local.get("dt_depth", {}), dict) else {}

                centered_S = np.asarray(m1_local.get("midline", S), float)
                centered_normals = m1_local.get("normals", {})
                if not isinstance(centered_normals, dict):
                    centered_normals = {}
                cseg_diag = m1_local.get("normals_diag", {})
                if not isinstance(cseg_diag, dict):
                    cseg_diag = {}
                per_seg_centered_normals_diag.append(cseg_diag)
                csdiag_brief = _normals_diag_summary(cseg_diag)
                print(
                    f"[GT_SUP NORMDBG] combined {ccid} seg={seg_i} centered total={csdiag_brief['total']} "
                    f"valid={csdiag_brief['valid']} invalid={csdiag_brief['invalid']} "
                    f"invalid_frac={csdiag_brief['invalid_frac']:.4f} top_reasons={csdiag_brief['top_reasons']}"
                )

                centered_segs.append(centered_S)
                ce1x_list.append(np.asarray(centered_normals.get("edge1_x", []), float))
                ce1y_list.append(np.asarray(centered_normals.get("edge1_y", []), float))
                ce2x_list.append(np.asarray(centered_normals.get("edge2_x", []), float))
                ce2y_list.append(np.asarray(centered_normals.get("edge2_y", []), float))
                cw_list.append(np.asarray(centered_normals.get("width_px", []), float))

                dseg = m4_local.get("midline", None)
                dnorm = m4_local.get("normals", {})
                if not isinstance(dnorm, dict):
                    dnorm = {}
                dseg_diag = m4_local.get("normals_diag", {})
                if not isinstance(dseg_diag, dict):
                    dseg_diag = {}
                per_seg_depth_normals_diag.append(dseg_diag)
                depth_cost_meta_per_segment.append(m4_local.get("meta", {}) if isinstance(m4_local.get("meta", {}), dict) else {})
                ddebug = m4_local.get("debug", {}) if isinstance(m4_local.get("debug", {}), dict) else {}

                if dseg is not None:
                    dseg = np.asarray(dseg, float)
                if dseg is not None and dseg.ndim == 2 and len(dseg) >= 2:
                    depth_segs.append(dseg)
                    de1x_list.append(np.asarray(dnorm.get("edge1_x", []), float))
                    de1y_list.append(np.asarray(dnorm.get("edge1_y", []), float))
                    de2x_list.append(np.asarray(dnorm.get("edge2_x", []), float))
                    de2y_list.append(np.asarray(dnorm.get("edge2_y", []), float))
                    dw_list.append(np.asarray(dnorm.get("width_px", []), float))
                    n_depth = min(len(S), len(dseg))
                    if n_depth > 0:
                        depth_shift_all.append(np.linalg.norm(dseg[:n_depth] - S[:n_depth], axis=1))
                    invalid_depth_masks.append(~np.isfinite(np.asarray(dnorm.get("width_px", []), float)))
                    dsdiag_brief = _normals_diag_summary(dseg_diag)
                    print(
                        f"[GT_SUP NORMDBG] combined {ccid} seg={seg_i} depth total={dsdiag_brief['total']} "
                        f"valid={dsdiag_brief['valid']} invalid={dsdiag_brief['invalid']} "
                        f"invalid_frac={dsdiag_brief['invalid_frac']:.4f} top_reasons={dsdiag_brief['top_reasons']}"
                    )
                else:
                    depth_segs.append(None)
                    de1x_list.append(np.asarray([], float))
                    de1y_list.append(np.asarray([], float))
                    de2x_list.append(np.asarray([], float))
                    de2y_list.append(np.asarray([], float))
                    dw_list.append(np.asarray([], float))
                    invalid_depth_masks.append(np.asarray([], bool))

                n = min(len(S), len(centered_S))
                if n > 0:
                    shift_all.append(np.linalg.norm(centered_S[:n] - S[:n], axis=1))

                invalid_manual_masks.append(~np.isfinite(np.asarray(w_manual, float)))
                invalid_center_masks.append(~np.isfinite(np.asarray(centered_normals.get("width_px", []), float)))

            if shift_all:
                d = np.concatenate(shift_all)
                shift_meta = {
                    "mean_shift_px": float(np.mean(d)),
                    "p95_shift_px": float(np.percentile(d, 95)),
                    "max_shift_px": float(np.max(d)),
                }
            else:
                shift_meta = {"mean_shift_px": 0.0, "p95_shift_px": 0.0, "max_shift_px": 0.0}

            if depth_shift_all:
                d2 = np.concatenate(depth_shift_all)
                depth_shift_meta = {
                    "mean_shift_px": float(np.mean(d2)),
                    "p95_shift_px": float(np.percentile(d2, 95)),
                    "max_shift_px": float(np.max(d2)),
                }
            else:
                depth_shift_meta = {"mean_shift_px": 0.0, "p95_shift_px": 0.0, "max_shift_px": 0.0}

            combined_entry["centered_midline_segments"] = [np.asarray(S, float).tolist() for S in centered_segs]
            combined_entry["centered_midline_segments_meta"] = [dict(m) for m in (seg_meta or [])]
            combined_entry["centered_midline"] = _pack_segs_with_separators(centered_segs)
            combined_entry["centered_normals"] = {
                "edge1_x": _pack_arrs_with_none_separators([_arr_to_list(a) for a in ce1x_list]),
                "edge1_y": _pack_arrs_with_none_separators([_arr_to_list(a) for a in ce1y_list]),
                "edge2_x": _pack_arrs_with_none_separators([_arr_to_list(a) for a in ce2x_list]),
                "edge2_y": _pack_arrs_with_none_separators([_arr_to_list(a) for a in ce2y_list]),
                "width_px": _pack_arrs_with_none_separators([_arr_to_list(a) for a in cw_list]),
            }
            combined_entry["midline_segments_auto_centered"] = [np.asarray(S, float).tolist() for S in centered_segs]
            combined_entry["midline_auto_centered"] = _pack_segs_with_separators(centered_segs)
            combined_entry["gt_normals_auto_centered"] = {
                "edge1_x": _pack_arrs_with_none_separators([_arr_to_list(a) for a in ce1x_list]),
                "edge1_y": _pack_arrs_with_none_separators([_arr_to_list(a) for a in ce1y_list]),
                "edge2_x": _pack_arrs_with_none_separators([_arr_to_list(a) for a in ce2x_list]),
                "edge2_y": _pack_arrs_with_none_separators([_arr_to_list(a) for a in ce2y_list]),
                "width_px": _pack_arrs_with_none_separators([_arr_to_list(a) for a in cw_list]),
            }
            combined_entry["centered_normals_diag_per_segment"] = per_seg_centered_normals_diag
            combined_entry["gt_normals_auto_centered_diag_per_segment"] = per_seg_centered_normals_diag
            combined_entry["gt_widths_auto_centered"] = [
                float(v) for arr in cw_list for v in np.asarray(arr, float)
            ]

            valid_depth_idx = [
                i for i, Sdep in enumerate(depth_segs)
                if Sdep is not None and np.asarray(Sdep, float).ndim == 2 and len(np.asarray(Sdep, float)) >= 2
            ]
            if valid_depth_idx:
                depth_valid_segs = [np.asarray(depth_segs[i], float) for i in valid_depth_idx]
                depth_meta_valid = [dict(seg_meta[i]) if i < len(seg_meta) else {} for i in valid_depth_idx]
                de1_valid = [de1x_list[i] for i in valid_depth_idx]
                de1y_valid = [de1y_list[i] for i in valid_depth_idx]
                de2_valid = [de2x_list[i] for i in valid_depth_idx]
                de2y_valid = [de2y_list[i] for i in valid_depth_idx]
                dw_valid = [dw_list[i] for i in valid_depth_idx]
                combined_entry["depth_midline_segments"] = [np.asarray(S, float).tolist() for S in depth_valid_segs]
                combined_entry["depth_midline_segments_meta"] = depth_meta_valid
                combined_entry["depth_midline"] = _pack_segs_with_separators(depth_valid_segs)
                combined_entry["depth_normals"] = {
                    "edge1_x": _pack_arrs_with_none_separators([_arr_to_list(a) for a in de1_valid]),
                    "edge1_y": _pack_arrs_with_none_separators([_arr_to_list(a) for a in de1y_valid]),
                    "edge2_x": _pack_arrs_with_none_separators([_arr_to_list(a) for a in de2_valid]),
                    "edge2_y": _pack_arrs_with_none_separators([_arr_to_list(a) for a in de2y_valid]),
                    "width_px": _pack_arrs_with_none_separators([_arr_to_list(a) for a in dw_valid]),
                }
                combined_entry["depth_normals_diag_per_segment"] = [
                    per_seg_depth_normals_diag[i] if i < len(per_seg_depth_normals_diag) else {}
                    for i in valid_depth_idx
                ]
                combined_entry["midline_depth_centered"] = combined_entry["depth_midline"]
                combined_entry["gt_normals_depth_centered"] = combined_entry["depth_normals"]
                combined_entry["gt_widths_depth_centered"] = [
                    float(v) for arr in dw_valid for v in np.asarray(arr, float)
                ]

            combined_entry["method_variants"] = {}
            for mk in METHOD_SPECS.keys():
                mpack = method_variant_segments.get(mk, {}) if isinstance(method_variant_segments.get(mk, {}), dict) else {}
                m_segs = [np.asarray(Sm, float) for Sm in (mpack.get("midline_segments", []) or []) if Sm is not None and len(np.asarray(Sm, float)) >= 2]
                m_meta = [dict(mm) for mm in (mpack.get("midline_segments_meta", []) or [])]
                e1x = [np.asarray(a, float) for a in (mpack.get("edge1_x", []) or [])]
                e1y = [np.asarray(a, float) for a in (mpack.get("edge1_y", []) or [])]
                e2x = [np.asarray(a, float) for a in (mpack.get("edge2_x", []) or [])]
                e2y = [np.asarray(a, float) for a in (mpack.get("edge2_y", []) or [])]
                ww = [np.asarray(a, float) for a in (mpack.get("width_px", []) or [])]
                combined_entry["method_variants"][mk] = {
                    "midline_segments": [np.asarray(Sm, float).tolist() for Sm in m_segs],
                    "midline_segments_meta": m_meta,
                    "midline": _pack_segs_with_separators(m_segs) if m_segs else [],
                    "normals": {
                        "edge1_x": _pack_arrs_with_none_separators([_arr_to_list(a) for a in e1x]),
                        "edge1_y": _pack_arrs_with_none_separators([_arr_to_list(a) for a in e1y]),
                        "edge2_x": _pack_arrs_with_none_separators([_arr_to_list(a) for a in e2x]),
                        "edge2_y": _pack_arrs_with_none_separators([_arr_to_list(a) for a in e2y]),
                        "width_px": _pack_arrs_with_none_separators([_arr_to_list(a) for a in ww]),
                    },
                    "normals_diag_per_segment": mpack.get("normals_diag_per_segment", []),
                    "timing": mpack.get("timing_sum", {}),
                    "meta": {
                        "label": METHOD_SPECS.get(mk, {}).get("label"),
                        "use_rgb": bool(METHOD_SPECS.get(mk, {}).get("use_rgb", False)),
                        "use_depth": bool(METHOD_SPECS.get(mk, {}).get("use_depth", False)),
                        "reason": None if m_segs else "no_valid_segments",
                    },
                }

            combined_entry["depth_cost_meta"] = {
                "per_segment": depth_cost_meta_per_segment,
            }

            combined_entry["timing"]["dt"] = combined_timing_blob.get("dt", {})
            combined_entry["timing"]["centered"] = combined_timing_blob.get("centered", {})
            combined_entry["timing"]["depth"] = combined_timing_blob.get("depth", {})
            combined_entry["timing"]["normals"] = combined_timing_blob.get("normals", {})
            timing_totals["combined_centering_sec"] += _sum_centering_seconds(combined_timing_blob)
            _accumulate_timing_blob(combined_timing_blob)

            manual_geom_parts = [np.asarray(S, float) for S in segs if S is not None and len(S) >= 2]
            center_geom_parts = [np.asarray(S, float) for S in centered_segs if S is not None and len(S) >= 2]
            depth_geom_parts = [
                np.asarray(S, float)
                for S in (combined_entry.get("depth_midline_segments", []) or [])
                if S is not None and len(S) >= 2
            ]
            manual_geom_all = np.vstack(manual_geom_parts) if manual_geom_parts else np.empty((0, 2), float)
            center_geom_all = np.vstack(center_geom_parts) if center_geom_parts else np.empty((0, 2), float)
            depth_geom_all = np.vstack(depth_geom_parts) if depth_geom_parts else np.empty((0, 2), float)

            combined_entry["auto_centering_meta"] = {
                "enabled": True,
                "domain_mode": str(auto_centering_domain_combined),
                "snap": {
                    "n_iters": int(auto_centering_iters),
                    "step_px": float(auto_centering_step_px),
                },
                "manual_vs_centered": {
                    **shift_meta,
                    **_geometry_disagreement_stats(manual_geom_all, center_geom_all),
                    **_width_stability_stats(
                        np.concatenate([np.asarray(a, float).reshape(-1) for a in w_list]) if w_list else [],
                        np.concatenate([np.asarray(a, float).reshape(-1) for a in cw_list]) if cw_list else [],
                    ),
                },
                "manual_vs_depth_distridge": (
                    {
                        **depth_shift_meta,
                        **_geometry_disagreement_stats(manual_geom_all, depth_geom_all),
                        **_width_stability_stats(
                            np.concatenate([np.asarray(a, float).reshape(-1) for a in w_list]) if w_list else [],
                            np.concatenate([np.asarray(a, float).reshape(-1) for a in dw_list]) if dw_list else [],
                        ),
                    }
                    if depth_geom_parts else None
                ),
                "depth_available": bool(depth_geom_parts),
            }
            combined_entry["auto_centering_meta"]["comparisons"] = {}
            for mk in METHOD_SPECS.keys():
                mv = (combined_entry.get("method_variants", {}) or {}).get(mk, {})
                mv_parts = [np.asarray(Sm, float) for Sm in (mv.get("midline_segments", []) or []) if Sm is not None and len(Sm) >= 2]
                mv_all = np.vstack(mv_parts) if mv_parts else np.empty((0, 2), float)
                mv_w = []
                if isinstance(mv.get("normals", {}), dict):
                    ww_pack = mv["normals"].get("width_px", [])
                    if isinstance(ww_pack, list):
                        for v in ww_pack:
                            if v is None:
                                continue
                            try:
                                mv_w.append(float(v))
                            except Exception:
                                continue
                combined_entry["auto_centering_meta"]["comparisons"][mk] = {
                    **_geometry_disagreement_stats(manual_geom_all, mv_all),
                    **_width_stability_stats(
                        np.concatenate([np.asarray(a, float).reshape(-1) for a in w_list]) if w_list else [],
                        mv_w,
                    ),
                    "available": bool(mv_parts),
                    "reason": ((mv.get("meta", {}) or {}).get("reason") if isinstance(mv.get("meta", {}), dict) else None),
                }

            if auto_centering_debug:
                combined_dbg_dir = os.path.join(auto_center_root, "combined", f"ccid_{tag_name}")
                os.makedirs(combined_dbg_dir, exist_ok=True)
                terr_vis = build_territory_mask_for_segments(
                    segs=segs,
                    crack_mask_u8=crack_mask_clipped,
                    window_half_size=int(auto_centering_window_half_size),
                )
                out_dbg = os.path.join(combined_dbg_dir, "manual_vs_centered.png")
                plot_midline_centering_debug(
                    out_path=out_dbg,
                    crack_mask_u8=crack_mask,
                    manual_segs=[np.asarray(S, float) for S in segs],
                    centered_segs=centered_segs,
                    territory_u8=terr_vis,
                    bbox_xywh=combined_entry.get("mask_bbox"),
                    title=f"combined {tag_name}: manual (yellow) vs centered (cyan)",
                    invalid_manual_masks=invalid_manual_masks,
                    invalid_center_masks=invalid_center_masks,
                )

                for mk, blobs in method_cost_debug_blobs.items():
                    for seg_i, dbg_blob, dbg_mask_use, _sel_key, _terr_plot in blobs:
                        style = method_style.get(mk, {})
                        mslug = str(style.get("slug", mk))
                        out_cost = os.path.join(
                            combined_dbg_dir,
                            f"seg{int(seg_i)}_{mslug}_cost_panel.png",
                        )
                        depth_lbl = "global" if mk in ("dt_depth", "dt_ridge_valley_depth") else None
                        plot_depth_cost_diagnostic(
                            out_path=out_cost,
                            crack_mask_u8=dbg_mask_use,
                            dt_norm=np.asarray(dbg_blob.get("dt_norm"), np.float32),
                            depth_norm=np.asarray(dbg_blob.get("depth_norm"), np.float32) if dbg_blob.get("depth_norm") is not None else None,
                            recess_norm=np.asarray(dbg_blob.get("recess_norm"), np.float32) if dbg_blob.get("recess_norm") is not None else None,
                            depth_score=np.asarray(dbg_blob.get("score_for_refine"), np.float32) if dbg_blob.get("score_for_refine") is not None else None,
                            costmaps=dbg_blob.get("costmaps"),
                            bbox_xywh=combined_entry.get("mask_bbox"),
                            title=f"combined {tag_name} seg{int(seg_i)}: cost cues",
                            method_label=style.get("label", mk),
                            depth_label=depth_lbl,
                        )

                if combined_entry.get("depth_midline_segments"):
                    out_overlay = os.path.join(
                        combined_dbg_dir,
                        "manual_vs_depth_distridge.png",
                    )
                    plot_midline_centering_debug(
                        out_path=out_overlay,
                        crack_mask_u8=crack_mask_clipped,
                        manual_segs=[np.asarray(S, float) for S in segs],
                        centered_segs=[np.asarray(S, float) for S in (combined_entry.get("depth_midline_segments") or []) if S is not None and len(S) >= 2],
                        territory_u8=terr_vis,
                        bbox_xywh=combined_entry.get("mask_bbox"),
                        title=f"combined {tag_name}: manual vs depth_distridge",
                        compare_label="Depth Distridge Midline",
                        compare_color="magenta",
                        left_panel_title="manual vs depth_distridge (mask)",
                        right_panel_title="DT ridge view (depth domain)",
                    )

                # Method-family overlays and cost diagnostics (M1-M4).
                mv_combined = combined_entry.get("method_variants", {}) if isinstance(combined_entry.get("method_variants", {}), dict) else {}
                for mk in METHOD_SPECS.keys():
                    rec = mv_combined.get(mk, {}) if isinstance(mv_combined.get(mk, {}), dict) else {}
                    mparts = [np.asarray(Sm, float) for Sm in (rec.get("midline_segments", []) or []) if Sm is not None and len(Sm) >= 2]
                    if mparts:
                        dbg_blobs_mk = method_cost_debug_blobs.get(mk, [])
                        if mk in ("dt_depth", "dt_ridge_valley", "dt_ridge_valley_depth") and not dbg_blobs_mk:
                            continue
                        sel_key_mk = "dt"
                        territory_method = np.zeros_like(crack_mask_clipped, dtype=np.uint8)
                        if dbg_blobs_mk:
                            sel_key_mk = str(dbg_blobs_mk[0][3])
                            for _seg_i, _dbg, _mask_use, _sk, terr_local in dbg_blobs_mk:
                                try:
                                    territory_method = territory_method | (np.asarray(terr_local) > 0).astype(np.uint8)
                                except Exception:
                                    continue
                        if not np.any(territory_method):
                            territory_method = terr_vis
                        style = method_style.get(mk, {})
                        mslug = str(style.get("slug", mk))
                        out_overlay = os.path.join(combined_dbg_dir, f"manual_vs_{mslug}.png")
                        plot_midline_centering_debug(
                            out_path=out_overlay,
                            crack_mask_u8=crack_mask_clipped,
                            manual_segs=[np.asarray(S, float) for S in segs],
                            centered_segs=mparts,
                            territory_u8=territory_method,
                            bbox_xywh=combined_entry.get("mask_bbox"),
                            title=f"combined {tag_name}: manual vs {style.get('label', mk)}",
                            compare_label=style.get("compare_label", mk),
                            compare_color=style.get("color", "cyan"),
                            left_panel_title=f"manual vs {style.get('label', mk)} (mask)",
                            right_panel_title=right_title_by_cost_key.get(sel_key_mk, "Preferred region"),
                        )

        final_entries.append(combined_entry)
        gt_sup_diag["combined_added"] += 1
        _cropped_preview(combined_entry, gt_mask, original_image, combined_crop_root)

    # =====================================================
    # 3) GLOBAL OVERVIEW
    # =====================================================
    analysis_dir = os.path.join(sup_root, "analysis")
    gt_centering_dir = os.path.join(sup_root, "gt_centering")
    depth_distridge_dir = os.path.join(sup_root, "depth_distridge")
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(gt_centering_dir, exist_ok=True)
    os.makedirs(depth_distridge_dir, exist_ok=True)

    compute_csv = os.path.join(analysis_dir, "gt_compute_timing.csv")
    centering_csv = os.path.join(analysis_dir, "gt_centering_timing.csv")
    combined_plus_noncombined_atomics_sec = float(
        timing_totals["combined_compute_sec"] + timing_totals["noncombined_atomic_compute_sec"]
    )
    centering_total_sec = float(timing_totals["atomic_centering_sec"] + timing_totals["combined_centering_sec"])
    combined_plus_noncombined_atomics_centering_sec = float(
        timing_totals["combined_centering_sec"] + timing_totals["noncombined_atomic_centering_sec"]
    )

    with open(compute_csv, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow([
            "image",
            "atomic_compute_sec",
            "noncombined_atomic_compute_sec",
            "combined_compute_sec",
            "combined_plus_noncombined_atomics_sec",
        ])
        wcsv.writerow([
            base_name,
            float(timing_totals["atomic_compute_sec"]),
            float(timing_totals["noncombined_atomic_compute_sec"]),
            float(timing_totals["combined_compute_sec"]),
            float(combined_plus_noncombined_atomics_sec),
        ])

    with open(centering_csv, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow([
            "image",
            "atomic_centering_sec",
            "noncombined_atomic_centering_sec",
            "combined_centering_sec",
            "combined_plus_noncombined_atomics_centering_sec",
            "centering_total_sec",
            "dt_compute_s",
            "centered_snap_s",
            "depth_align_s",
            "depth_recess_s",
            "depth_costmap_s",
            "depth_dijkstra_s",
            "depth_postprocess_s",
            "normals_centered_s",
            "normals_depth_s",
        ])
        wcsv.writerow([
            base_name,
            float(timing_totals["atomic_centering_sec"]),
            float(timing_totals["noncombined_atomic_centering_sec"]),
            float(timing_totals["combined_centering_sec"]),
            float(combined_plus_noncombined_atomics_centering_sec),
            float(centering_total_sec),
            float(timing_totals["dt_compute_s"]),
            float(timing_totals["centered_snap_s"]),
            float(timing_totals["depth_align_s"]),
            float(timing_totals["depth_recess_s"]),
            float(timing_totals["depth_costmap_s"]),
            float(timing_totals["depth_dijkstra_s"]),
            float(timing_totals["depth_postprocess_s"]),
            float(timing_totals["normals_centered_s"]),
            float(timing_totals["normals_depth_s"]),
        ])

    # Split timing outputs by track for clearer interpretation.
    centered_timing_csv = os.path.join(gt_centering_dir, "timing.csv")
    with open(centered_timing_csv, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow([
            "image",
            "dt_compute_s",
            "centered_snap_s",
            "normals_centered_s",
            "atomic_centering_sec",
            "noncombined_atomic_centering_sec",
            "combined_centering_sec",
            "combined_plus_noncombined_atomics_centering_sec",
            "centering_total_sec",
        ])
        wcsv.writerow([
            base_name,
            float(timing_totals["dt_compute_s"]),
            float(timing_totals["centered_snap_s"]),
            float(timing_totals["normals_centered_s"]),
            float(timing_totals["atomic_centering_sec"]),
            float(timing_totals["noncombined_atomic_centering_sec"]),
            float(timing_totals["combined_centering_sec"]),
            float(combined_plus_noncombined_atomics_centering_sec),
            float(centering_total_sec),
        ])

    depth_timing_csv = os.path.join(depth_distridge_dir, "timing.csv")
    with open(depth_timing_csv, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow([
            "image",
            "dt_compute_s",
            "depth_align_s",
            "depth_recess_s",
            "depth_costmap_s",
            "depth_dijkstra_s",
            "depth_postprocess_s",
            "normals_depth_s",
            "atomic_centering_sec",
            "noncombined_atomic_centering_sec",
            "combined_centering_sec",
            "combined_plus_noncombined_atomics_centering_sec",
            "centering_total_sec",
        ])
        wcsv.writerow([
            base_name,
            float(timing_totals["dt_compute_s"]),
            float(timing_totals["depth_align_s"]),
            float(timing_totals["depth_recess_s"]),
            float(timing_totals["depth_costmap_s"]),
            float(timing_totals["depth_dijkstra_s"]),
            float(timing_totals["depth_postprocess_s"]),
            float(timing_totals["normals_depth_s"]),
            float(timing_totals["atomic_centering_sec"]),
            float(timing_totals["noncombined_atomic_centering_sec"]),
            float(timing_totals["combined_centering_sec"]),
            float(combined_plus_noncombined_atomics_centering_sec),
            float(centering_total_sec),
        ])

    print(
        f"[GT_SUP TIMING] atomic_compute_sec={timing_totals['atomic_compute_sec']:.4f} "
        f"noncombined_atomic_compute_sec={timing_totals['noncombined_atomic_compute_sec']:.4f} "
        f"combined_compute_sec={timing_totals['combined_compute_sec']:.4f} "
        f"combined_plus_noncombined_atomics_sec={combined_plus_noncombined_atomics_sec:.4f} "
        f"centering_total_sec={centering_total_sec:.4f} "
        f"dt_compute_s={timing_totals['dt_compute_s']:.4f} "
        f"depth_dijkstra_s={timing_totals['depth_dijkstra_s']:.4f}"
    )
    print(f"[GT_SUP TIMING] wrote {compute_csv}")
    print(f"[GT_SUP TIMING] wrote {centering_csv}")
    print(f"[GT_SUP TIMING] wrote {centered_timing_csv}")
    print(f"[GT_SUP TIMING] wrote {depth_timing_csv}")

    if not final_entries:
        print(f"[GT_SUP DIAG] no final_entries produced: {gt_sup_diag}")
    else:
        print(f"[GT_SUP DIAG] entries={len(final_entries)} summary={gt_sup_diag}")
    global_png = os.path.join(sup_root, "global_overview.png")
    _global_overview(final_entries, gt_mask, global_png)

    # =====================================================
    # 4) WRITE JSON
    # =====================================================
    out_json = os.path.join(sup_root, "gt_supervision.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"cracks": final_entries}, f, indent=2)

    # =====================================================
    # 5) GT CENTERING METRICS (Supervision Audit)
    # =====================================================
    if enable_auto_centering:
        export_gt_centering_metrics(
            base_name=base_name,
            save_root=save_root,
            final_entries=final_entries,
            combined_member_ids=combined_flat,
        )

    print(f"[GT_SUP] wrote JSON → {out_json}")
    print(f"[GT_SUP] global overview → {global_png}")
