import os, json, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import csv
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "path.simplify": True,
    "path.simplify_threshold": 1.0,
                "savefig.dpi": 200,
})
from matplotlib import pyplot as plt
plt.ioff()
from helpers.plot_metrics import plot_edges_and_normals

# Set to False to suppress verbose per-image debug prints during batch runs
_VERBOSE_DEBUG = True

DEBUG_LEVEL = 1
# 0 = silent
# 1 = important only
# 2 = per-crack summaries
# 3 = full debug (rare)
DEBUG_TARGET = "cid1"   # set failing image key
DEBUG_SPLIT = True      # branch -> segment split diagnostics
DEBUG_SUPPRESS = True   # suppression diagnostics
ENABLE_ABLATION_DEBUG_PLOTS = True
ENABLE_COMBINED_METHOD_DEBUG_PLOTS = True
DEBUG_LIGHT = True      # minimal high-level logs
PER_ATOMIC_METHOD_DEBUG = False  # keep False: use aggregated atomic-all method plots instead
METHODS_RS3 = [
    "dt",
    "dt_depth",
    "dt_ridge_valley",
    "dt_ridge_valley_depth",
    "dt_ridge_color_depth",
]
ISOLATE_GT_BRANCH_GEOMETRY = False
ISOLATE_GT_IMAGE = None
ISOLATE_GT_BRANCH_ID = 0
HARD_ISOLATION_DISABLE_CENTERING = False
GT_BRANCH_LOOP_KILL_CHECK = False
GT_BRANCH_LOOP_KILL_IMAGE = None
GT_BRANCH_LOOP_KILL_BRANCH = None


def _dlog(level, msg):
    if int(DEBUG_LEVEL) >= int(level):
        print(msg)


def _dbg(base_name):
    return str(base_name) == str(DEBUG_TARGET)


def _isolate_gt_dbg(base_name, branch_id):
    if not bool(ISOLATE_GT_BRANCH_GEOMETRY):
        return False
    return str(base_name) == str(ISOLATE_GT_IMAGE) and int(branch_id) == int(ISOLATE_GT_BRANCH_ID)


def _branch_kill_dbg(base_name, branch_id):
    if not bool(GT_BRANCH_LOOP_KILL_CHECK):
        return False
    if str(base_name) != str(GT_BRANCH_LOOP_KILL_IMAGE):
        return False
    if GT_BRANCH_LOOP_KILL_BRANCH is None:
        return True
    try:
        return int(branch_id) == int(GT_BRANCH_LOOP_KILL_BRANCH)
    except Exception:
        return False


# --------------------------------------------
# UTIL
# --------------------------------------------
def _ensure(p):
    os.makedirs(p, exist_ok=True)
    return p


def _save_debug_plot(fig, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    from matplotlib import pyplot as plt
    plt.close(fig)


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
        metrics/<image>/supervision/preview/*.jpg
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

        _supervision_preview(entry, original_image, os.path.join(sup_prev, f"cid{cid}.jpg"))

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
        _supervision_preview(entry, original_image, os.path.join(sup_prev, f"{tag}.jpg"))

    # =======================================================
    # WRITE ONE UNIFIED JSON
    # =======================================================
    out_json = os.path.join(sup_root, "supervision.json")
    with open(out_json, "w") as f:
        json.dump(sup_list, f)

    _dlog(1, f"[SUPERVISION] wrote {out_json}")


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
    title = f"Ground truth crack normals - {entry['type']} {entry['id']}"

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

    _dlog(2, f"[PREVIEW] {out_path}")
    
    
    
    
    
    
    
    
    
    

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
from helpers.branch_stitching import stitch_branch_segments
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


def _crop_local(arr, bbox_xywh):
    """
    Crop array in image coordinates using bbox=(x,y,w,h).
    Returns None if arr is None.
    """
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.ndim < 2:
        return None
    x, y, w, h = map(int, bbox_xywh)
    H, W = a.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(W, x + max(0, w))
    y1 = min(H, y + max(0, h))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((max(0, h), max(0, w)), dtype=a.dtype)
    return a[y0:y1, x0:x1]





def _split_midline_packed(mid_packed):
    """
    mid_packed: list like [[x,y], [x,y], [None,None], [x,y], ...]
    returns: list of (N,2) float arrays
    """
    segs = []
    cur = []
    if mid_packed is None:
        return segs
    if isinstance(mid_packed, (float, int, np.floating, np.integer)):
        return segs
    try:
        iterator = iter(mid_packed)
    except TypeError:
        return segs

    for pt in iterator:
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




def _plot_single_debug(entry, gt_mask_u8, original_image, out_path):
    """
    Lightweight ablation debug plot: one crop, manual + all available methods.
    """
    import os
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from combiner import bbox_xywh_to_xyxy

    H, W = original_image.shape[:2]
    crack_id = entry.get("id", "UNKNOWN")
    kind = entry.get("kind", "UNKNOWN")
    bb = entry.get("mask_bbox")
    if bb is None:
        return

    if entry.get("midline_segments"):
        manual_mid_segs = [np.asarray(S, float) for S in entry["midline_segments"] if S is not None and len(S) >= 2]
    else:
        mid = np.asarray(entry.get("midline", []), float)
        manual_mid_segs = [mid] if (mid.ndim == 2 and len(mid) >= 2) else []
    if not manual_mid_segs:
        return

    method_variants = entry.get("method_variants", {}) if isinstance(entry.get("method_variants", {}), dict) else {}
    method_order = [
        ("dt", "DT", "white"),
        ("dt_depth", "Depth", "cyan"),
        ("dt_ridge_valley_depth", "DT+Depth (ridge)", "magenta"),
        ("dt_ridge_color_depth", "Final (multi-cue)", "yellow"),
    ]
    method_segs = []
    for mkey, label, color in method_order:
        mv = method_variants.get(mkey, {}) if isinstance(method_variants.get(mkey, {}), dict) else {}
        segs = [np.asarray(S, float) for S in (mv.get("midline_segments", []) or []) if S is not None and len(S) >= 2]
        if not segs:
            mline = mv.get("midline")
            if mline is not None:
                arr = np.asarray(mline, float)
                if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) >= 2:
                    segs = [arr]
                else:
                    segs = _split_midline_packed(mline)
        if segs:
            method_segs.append((label, color, segs))
    if not method_segs:
        return

    x0, y0, x1, y1 = bbox_xywh_to_xyxy(bb, H, W, pad=8)
    all_pts = []
    for S in manual_mid_segs:
        all_pts.append(np.asarray(S, float))
    for _, _, segs in method_segs:
        for S in segs:
            all_pts.append(np.asarray(S, float))
    if all_pts:
        P = np.vstack(all_pts)
        x0 = int(max(0, min(x0, np.floor(P[:, 0].min()) - 8)))
        y0 = int(max(0, min(y0, np.floor(P[:, 1].min()) - 8)))
        x1 = int(min(W, max(x1, np.ceil(P[:, 0].max()) + 8)))
        y1 = int(min(H, max(y1, np.ceil(P[:, 1].max()) + 8)))
    if x1 <= x0 or y1 <= y0:
        return

    full_mask = np.zeros((H, W), np.uint8)
    mask_crop = entry.get("mask_crop", None)
    if bb is not None and mask_crop is not None:
        bx, by, bw, bh = map(int, bb)
        crop_arr = np.asarray(mask_crop, np.uint8)
        if crop_arr.ndim >= 2 and crop_arr.shape[:2] == (bh, bw):
            y0m, x0m = max(0, by), max(0, bx)
            y1m, x1m = min(H, by + bh), min(W, bx + bw)
            ch, cw = max(0, y1m - y0m), max(0, x1m - x0m)
            if ch > 0 and cw > 0:
                full_mask[y0m:y1m, x0m:x1m] = (crop_arr[:ch, :cw] > 0).astype(np.uint8)
    elif gt_mask_u8 is not None:
        arr = (np.asarray(gt_mask_u8) > 0).astype(np.uint8)
        if arr.ndim >= 2 and arr.shape[:2] == (H, W):
            full_mask = arr

    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    overlay = np.clip(gray * 0.25 + full_mask.astype(np.float32) * 0.75, 0, 1)
    crop_img = (np.stack([overlay] * 3, axis=-1) * 255).astype(np.uint8)[y0:y1, x0:x1]
    shift = np.array([x0, y0], float)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(crop_img[:, :, ::-1])
    manual_color = "darkorange"
    for i, S in enumerate(manual_mid_segs):
        SS = np.asarray(S, float) - shift
        if SS.ndim == 2 and SS.shape[1] == 2 and len(SS) >= 2:
            ax.plot(SS[:, 0], SS[:, 1], color=manual_color, linestyle="--", lw=2.2, alpha=0.95, label="Manual" if i == 0 else None)
    for label, color, segs in method_segs:
        for i, S in enumerate(segs):
            SS = np.asarray(S, float) - shift
            if SS.ndim == 2 and SS.shape[1] == 2 and len(SS) >= 2:
                ax.plot(SS[:, 0], SS[:, 1], color=color, lw=1.8, alpha=0.95, label=label if i == 0 else None)

    ax.set_title(f"{kind} {crack_id} - Ablation Comparison")
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
# ============================================================
# GLOBAL OVERVIEW (with legend + title)
# ============================================================
def _global_overview(entries, gt_mask, out_png, title="Global GT Overview"):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    H, W = gt_mask.shape[:2]

    fig, ax = plt.subplots(figsize=(8, 8))
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
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _debug_plot_all_atomics_for_export(*, image_rgb, atomic_map, out_png, title="RAW JSON DEBUG"):
    import matplotlib.pyplot as plt

    if image_rgb is None:
        return
    img = np.asarray(image_rgb)
    if img.ndim < 2:
        return
    H, W = img.shape[:2]

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.imshow(img)

    for cid, cr in (atomic_map or {}).items():
        if not isinstance(cr, dict):
            continue
        scid = str(cid)
        bbox = cr.get("mask_bbox", None)
        mid = cr.get("midline", None)
        color_mid = plt.cm.tab10(int(scid) % 10) if scid.isdigit() else "lime"

        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                x, y, bw, bh = [int(v) for v in bbox]
                rect = plt.Rectangle((x, y), bw, bh, edgecolor="cyan", facecolor="none", linewidth=2)
                ax.add_patch(rect)
                ax.text(x, y - 5, f"ID {scid}", color="cyan", fontsize=9, weight="bold")
                ax.scatter(x + bw / 2.0, y + bh / 2.0, c="yellow", s=20)
            except Exception:
                pass

        try:
            m = np.asarray(mid, float)
            if m.ndim == 2 and m.shape[1] == 2 and len(m) > 0:
                ax.plot(m[:, 0], m[:, 1], color=color_mid, linewidth=2)
                ax.scatter(m[0, 0], m[0, 1], c="red", s=24)
                ax.scatter(m[-1, 0], m[-1, 1], c="red", s=24)
                ax.scatter(float(np.mean(m[:, 0])), float(np.mean(m[:, 1])), c="magenta", s=20)
        except Exception:
            pass

    ax.set_title(str(title))
    ax.set_xlim([0, W])
    ax.set_ylim([H, 0])
    ax.axis("off")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    _dlog(3, f"[GT_SUP DEBUG] wrote atomic truth debug -> {out_png}")

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


def _cc_labels_for_members(members, atomic, cc_labels):
    """Return all GT CC labels touched by any member midline point."""
    H, W = cc_labels.shape[:2]
    labels_found = set()
    for m in (members or []):
        cr = (atomic or {}).get(str(m), {}) or {}
        mid = np.asarray(cr.get("midline", []), float)
        if mid.ndim != 2 or mid.shape[1] != 2 or len(mid) < 1:
            continue
        ys = np.clip(np.round(mid[:, 1]).astype(int), 0, H - 1)
        xs = np.clip(np.round(mid[:, 0]).astype(int), 0, W - 1)
        labs = cc_labels[ys, xs]
        labs = labs[labs > 0]
        for l in labs:
            labels_found.add(int(l))
    return labels_found




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
    if m.ndim != 2 or c.ndim != 2 or m.shape[1] != 2 or c.shape[1] != 2:
        return {"mean_shift_px": 0.0, "p95_shift_px": 0.0, "max_shift_px": 0.0}
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

    def _safe_mean(a):
        a = np.asarray(a, float).reshape(-1)
        v = a[np.isfinite(a)]
        return float(np.mean(v)) if v.size else float("nan")

    def _safe_std(a):
        a = np.asarray(a, float).reshape(-1)
        v = a[np.isfinite(a)]
        return float(np.std(v)) if v.size else float("nan")

    return {
        "manual_width_mean": _safe_mean(wm),
        "centered_width_mean": _safe_mean(wc),
        "manual_width_std": _safe_std(wm),
        "centered_width_std": _safe_std(wc),
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
    _dbg_coord,
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
    left_panel_title="ET vs centered (mask)",
    right_panel_title="DT ridge view (domain)",
    normals=None,
):
    """
    Plot ET vs centered midlines over crack mask (crop around bbox).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    M = (np.asarray(crack_mask_u8) > 0).astype(np.uint8)
    H, W = M.shape[:2]

    centered_valid = []
    for i, S in enumerate(centered_segs or []):
        S = np.asarray(S, float)
        if S.ndim == 2 and S.shape[1] == 2 and len(S) >= 2:
            centered_valid.append(S)
        else:
            _dlog(2, f"[PLOT DIAG][SKIP CENTER] invalid centered seg idx={i} shape={S.shape}")
    centered_segs = centered_valid

    # Geometry-debug: report overlap likelihood (centered hidden under manual).
    pair_n = min(len(manual_segs or []), len(centered_segs or []))
    for i in range(pair_n):
        m = np.asarray((manual_segs or [])[i], float)
        c = np.asarray((centered_segs or [])[i], float)
        if m.shape == c.shape and m.ndim == 2 and len(m) >= 2:
            max_diff = float(np.max(np.abs(m - c)))
            _dlog(3, f"[AUTO CENTER DEBUG] seg={i} max_shift_abs={max_diff:.6f}")

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
        if M.shape[:2] != T.shape[:2]:
            raise ValueError(
                f"[plot_midline_centering_debug] mask/territory shape mismatch: "
                f"M={M.shape[:2]} T={T.shape[:2]}"
            )

    if show_dt_panel:
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.0), sharex=True, sharey=True)
        ax0, ax1 = axes
    else:
        fig, ax0 = plt.subplots(1, 1, figsize=(6.4, 6.4))
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
                    color="orange",
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

    # -------------------------------------------------
    # Plot normals (optional)
    # -------------------------------------------------
    normal_color = "blue"
    normals_iter = []
    if isinstance(normals, dict):
        normals_iter = [normals]
    elif isinstance(normals, (list, tuple)):
        normals_iter = [n for n in normals if isinstance(n, dict)]
    for nrm in normals_iter:
        e1 = _split_xy_none_seps(nrm.get("edge1_x", []), nrm.get("edge1_y", []))
        e2 = _split_xy_none_seps(nrm.get("edge2_x", []), nrm.get("edge2_y", []))
        for a, b in zip(e1, e2):
            aa = np.asarray(a, float)
            bb = np.asarray(b, float)
            n = min(len(aa), len(bb))
            if n <= 0:
                continue
            step = max(1, n // 50)
            for i in range(0, n, step):
                p1 = aa[i]
                p2 = bb[i]
                for ax in (ax0, ax1):
                    ax.plot(
                        [p1[0] - x0, p2[0] - x0],
                        [p1[1] - y0, p2[1] - y0],
                        color=normal_color,
                        lw=0.9,
                        alpha=0.55,
                        zorder=4,
                    )

    if bbox_xywh is not None and len(bbox_xywh) == 4:
        x, y, w, h = map(int, bbox_xywh)
        for ax in (ax0, ax1):
            ax.add_patch(plt.Rectangle((x - x0, y - y0), w, h, fill=False, edgecolor="dodgerblue", linewidth=1.3))

    # Figure-level legend on the right (applies to both panels).
    handles = [
        Line2D([], [], color=compare_color, lw=2.5, linestyle="-", label=str(compare_label)),
        Line2D([], [], color="orange", lw=2.0, linestyle="--", label="Manual"),
        Line2D([], [], color=normal_color, lw=1.2, linestyle="-", label="Normals"),
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


def plot_costmap_debug(
    *,
    out_path,
    crack_mask_u8,
    manual_segs,
    pred_segs,
    costmap=None,
    normals=None,
    normals_offset_xy=None,
    title="",
    pred_color="magenta",
):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    M = (np.asarray(crack_mask_u8) > 0).astype(np.uint8)
    H, W = M.shape[:2]
    ys, xs = np.where(M > 0)
    if xs.size > 0 and ys.size > 0:
        pad = 25
        x0 = max(0, int(xs.min()) - pad)
        y0 = max(0, int(ys.min()) - pad)
        x1 = min(W, int(xs.max()) + 1 + pad)
        y1 = min(H, int(ys.max()) + 1 + pad)
    else:
        x0, y0, x1, y1 = 0, 0, W, H

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axL, axR = axes

    axL.imshow(M[y0:y1, x0:x1], cmap="gray")
    axL.axis("off")
    axL.set_title("Geometry", fontsize=10)
    for S in (manual_segs or []):
        S = np.asarray(S, float)
        if S.ndim == 2 and S.shape[1] == 2 and len(S) >= 2:
            axL.plot(S[:, 0] - x0, S[:, 1] - y0, "--", color="orange", linewidth=2.0)
    for S in (pred_segs or []):
        S = np.asarray(S, float)
        if S.ndim == 2 and S.shape[1] == 2 and len(S) >= 2:
            axL.plot(S[:, 0] - x0, S[:, 1] - y0, "-", color=pred_color, linewidth=2.0)

    if costmap is not None:
        C = np.asarray(costmap, np.float32)
        if C.ndim >= 2:
            axR.imshow(C[y0:y1, x0:x1], cmap="inferno")
        else:
            axR.imshow(np.zeros((max(1, y1 - y0), max(1, x1 - x0)), np.float32), cmap="inferno")
    else:
        axR.imshow(np.zeros((max(1, y1 - y0), max(1, x1 - x0)), np.float32), cmap="inferno")
    for S in (pred_segs or []):
        S = np.asarray(S, float)
        if S.ndim == 2 and S.shape[1] == 2 and len(S) >= 2:
            axR.plot(S[:, 0] - x0, S[:, 1] - y0, "-", color="cyan", linewidth=2.0)
    axR.axis("off")
    axR.set_title("Costmap", fontsize=10)

    normals_iter = []
    if isinstance(normals, dict):
        normals_iter = [normals]
    elif isinstance(normals, (list, tuple)):
        normals_iter = [n for n in normals if isinstance(n, dict)]
    if normals_offset_xy is not None and len(normals_offset_xy) == 2:
        nox, noy = float(normals_offset_xy[0]), float(normals_offset_xy[1])
    else:
        nox, noy = 0.0, 0.0
    for nrm in normals_iter:
        e1 = _split_xy_none_seps(nrm.get("edge1_x", []), nrm.get("edge1_y", []))
        e2 = _split_xy_none_seps(nrm.get("edge2_x", []), nrm.get("edge2_y", []))
        for a, b in zip(e1, e2):
            aa = np.asarray(a, float)
            bb = np.asarray(b, float)
            n = min(len(aa), len(bb))
            if n <= 0:
                continue
            step = max(1, n // 50)
            for i in range(0, n, step):
                p1 = aa[i]
                p2 = bb[i]
                axL.plot(
                    [p1[0] - nox - x0, p2[0] - nox - x0],
                    [p1[1] - noy - y0, p2[1] - noy - y0],
                    color="blue",
                    lw=0.9,
                    alpha=0.7,
                )

    handles = [
        Line2D([], [], color="yellow", lw=2.0, linestyle="--", label="Manual"),
        Line2D([], [], color=pred_color, lw=2.0, linestyle="-", label="Prediction"),
        Line2D([], [], color="blue", lw=1.2, linestyle="-", label="Normals"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, framealpha=0.9, fontsize=9)
    fig.suptitle(str(title), fontsize=11)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_gt_normals_debug_global(
    *,
    out_path,
    original_image,
    crack_mask_u8,
    gt_midline_segs,
    gt_normals,
    title="GT normals (global)",
):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    M = (np.asarray(crack_mask_u8) > 0).astype(np.uint8)
    H, W = M.shape[:2]
    ys, xs = np.where(M > 0)
    if xs.size:
        pad = 25
        x0 = max(0, int(xs.min()) - pad)
        y0 = max(0, int(ys.min()) - pad)
        x1 = min(W, int(xs.max()) + 1 + pad)
        y1 = min(H, int(ys.max()) + 1 + pad)
    else:
        x0, y0, x1, y1 = 0, 0, W, H

    gray = cv2.cvtColor(np.asarray(original_image), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    overlay = np.clip(gray * 0.25 + M.astype(np.float32) * 0.75, 0, 1)
    rgb = (np.stack([overlay] * 3, axis=-1) * 255).astype(np.uint8)
    crop = rgb[y0:y1, x0:x1].copy()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(crop)

    for S in (gt_midline_segs or []):
        S = np.asarray(S, float)
        if S.ndim == 2 and S.shape[1] == 2 and len(S) >= 2:
            ax.plot(
                S[:, 0] - x0,
                S[:, 1] - y0,
                "-",
                color="orange",
                lw=2.0,
                alpha=0.95,
            )

    normals_iter = []
    if isinstance(gt_normals, dict):
        normals_iter = [gt_normals]
    elif isinstance(gt_normals, (list, tuple)):
        normals_iter = [n for n in gt_normals if isinstance(n, dict)]

    for nrm in normals_iter:
        e1 = _split_xy_none_seps(nrm.get("edge1_x", []), nrm.get("edge1_y", []))
        e2 = _split_xy_none_seps(nrm.get("edge2_x", []), nrm.get("edge2_y", []))
        for a, b in zip(e1, e2):
            aa = np.asarray(a, float)
            bb = np.asarray(b, float)
            n = min(len(aa), len(bb))
            if n <= 0:
                continue
            _step = max(1, n // 35)
            for i in range(0, n, _step):
                p1 = aa[i]
                p2 = bb[i]
                if (
                    not np.isfinite(p1[0]) or not np.isfinite(p1[1]) or
                    not np.isfinite(p2[0]) or not np.isfinite(p2[1])
                ):
                    continue
                if np.linalg.norm(p2 - p1) < 2.0:
                    continue
                ax.plot(
                    [p1[0] - x0, p2[0] - x0],
                    [p1[1] - y0, p2[1] - y0],
                    color="blue",
                    lw=0.9,
                    alpha=0.30,
                )

    handles = [
        Line2D([], [], color="orange", lw=2.0, linestyle="-", label="GT midline"),
        Line2D([], [], color="blue", lw=1.0, linestyle="-", label="GT normals"),
    ]
    ax.legend(handles=handles, loc="lower right", framealpha=0.9, fontsize=9)
    ax.set_title(str(title), fontsize=10)
    ax.axis("off")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_combined_overlay_debug(
    *,
    out_path,
    original_image,
    crack_mask_u8,
    manual_segs,
    pred_segs,
    normals=None,
    territory_u8=None,
    bbox_xywh=None,
    title="",
    compare_label="Prediction",
    compare_color="magenta",
):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

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

    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    overlay = np.clip(gray * 0.25 + M.astype(np.float32) * 0.75, 0, 1)
    rgb = (np.stack([overlay] * 3, axis=-1) * 255).astype(np.uint8)
    crop = rgb[y0:y1, x0:x1].copy()

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(crop)

    if territory_u8 is not None:
        T = (np.asarray(territory_u8) > 0).astype(np.uint8)
        if T.shape[:2] == M.shape[:2]:
            crop_T = T[y0:y1, x0:x1] > 0
            terr_rgba = np.zeros((crop_T.shape[0], crop_T.shape[1], 4), dtype=np.float32)
            terr_rgba[..., 1] = 1.0
            terr_rgba[..., 3] = crop_T.astype(np.float32) * 0.18
            ax.imshow(terr_rgba)

    for S in manual_segs or []:
        S = np.asarray(S, float)
        if S.ndim == 2 and S.shape[1] == 2 and len(S) >= 2:
            ax.plot(S[:, 0] - x0, S[:, 1] - y0, "--", color="orange", lw=2.0)

    for S in pred_segs or []:
        S = np.asarray(S, float)
        if S.ndim == 2 and S.shape[1] == 2 and len(S) >= 2:
            ax.plot(S[:, 0] - x0, S[:, 1] - y0, "-", color=compare_color, lw=2.4)

    normals_iter = []
    if isinstance(normals, dict):
        normals_iter = [normals]
    elif isinstance(normals, (list, tuple)):
        normals_iter = [n for n in normals if isinstance(n, dict)]
    for nrm in normals_iter:
        e1 = _split_xy_none_seps(nrm.get("edge1_x", []), nrm.get("edge1_y", []))
        e2 = _split_xy_none_seps(nrm.get("edge2_x", []), nrm.get("edge2_y", []))
        for a, b in zip(e1, e2):
            aa = np.asarray(a, float)
            bb = np.asarray(b, float)
            n = min(len(aa), len(bb))
            if n <= 0:
                continue
            step = max(1, n // 35)
            for i in range(0, n, step):
                p1 = aa[i]
                p2 = bb[i]
                if np.linalg.norm(p2 - p1) < 2.0:
                    continue
                ax.plot(
                    [p1[0] - x0, p2[0] - x0],
                    [p1[1] - y0, p2[1] - y0],
                    color="blue",
                    lw=0.9,
                    alpha=0.55,
                )

    handles = [
        Line2D([], [], color=compare_color, lw=2.4, linestyle="-", label=str(compare_label)),
        Line2D([], [], color="orange", lw=2.0, linestyle="--", label="Manual"),
        Line2D([], [], color="blue", lw=1.0, linestyle="-", label="Normals"),
    ]
    ax.legend(handles=handles, loc="lower right", framealpha=0.9, fontsize=9)
    ax.set_title(str(title))
    ax.axis("off")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_combined_overlay_and_costmap(
    *,
    out_path,
    original_image,
    crack_mask_u8,
    manual_segs,
    pred_segs,
    selected_costmap,
    territory_u8=None,
    bbox_xywh=None,
    title="",
    compare_label="Prediction",
    compare_color="magenta",
):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    M = (np.asarray(crack_mask_u8) > 0).astype(np.uint8)
    H, W = M.shape[:2]
    C = np.asarray(selected_costmap, np.float32)
    if C.shape[:2] != M.shape[:2]:
        return

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

    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    overlay = np.clip(gray * 0.25 + M.astype(np.float32) * 0.75, 0, 1)
    rgb = (np.stack([overlay] * 3, axis=-1) * 255).astype(np.uint8)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.4, 6.0), sharex=True, sharey=True)
    ax0.imshow(rgb[y0:y1, x0:x1])
    if territory_u8 is not None:
        T = (np.asarray(territory_u8) > 0).astype(np.uint8)
        if T.shape[:2] == M.shape[:2]:
            crop_T = T[y0:y1, x0:x1] > 0
            terr_rgba = np.zeros((crop_T.shape[0], crop_T.shape[1], 4), dtype=np.float32)
            terr_rgba[..., 1] = 1.0
            terr_rgba[..., 3] = crop_T.astype(np.float32) * 0.18
            ax0.imshow(terr_rgba)

    cropC = C[y0:y1, x0:x1]
    valid = np.isfinite(cropC) & (cropC < np.float32(1e8))
    show = np.full_like(cropC, np.nan, dtype=np.float32)
    if np.any(valid):
        vals = cropC[valid]
        lo = float(np.percentile(vals, 2))
        hi = float(np.percentile(vals, 98))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo + 1e-9:
            show[valid] = (cropC[valid] - lo) / (hi - lo)
        else:
            show[valid] = 0.5
    cmap_cost = plt.cm.get_cmap("viridis").copy()
    cmap_cost.set_bad(color="black")
    cmap_cost.set_under(color="black")
    ax1.imshow(show, cmap=cmap_cost, vmin=1e-6, vmax=1.0)

    for ax in (ax0, ax1):
        for S in manual_segs or []:
            S = np.asarray(S, float)
            if S.ndim == 2 and S.shape[1] == 2 and len(S) >= 2:
                ax.plot(S[:, 0] - x0, S[:, 1] - y0, "--", color="orange", lw=2.0)
        for S in pred_segs or []:
            S = np.asarray(S, float)
            if S.ndim == 2 and S.shape[1] == 2 and len(S) >= 2:
                ax.plot(S[:, 0] - x0, S[:, 1] - y0, "-", color=compare_color, lw=2.4)

    ax0.set_title("Territory + manual/prediction", fontsize=10)
    ax1.set_title("Selected costmap + manual/prediction", fontsize=10)
    ax0.axis("off")
    ax1.axis("off")

    handles = [
        Line2D([], [], color=compare_color, lw=2.4, linestyle="-", label=str(compare_label)),
        Line2D([], [], color="orange", lw=2.0, linestyle="--", label="Manual"),
    ]
    fig.subplots_adjust(right=0.83)
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.845, 0.5), framealpha=0.9, fontsize=9)
    fig.suptitle(str(title), fontsize=11)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_atomic_ablation_debug(
    *,
    out_path,
    crack_mask_u8,
    all_manual,
    all_pred,
    normals=None,
    title="",
    pred_color="magenta",
    original_image=None,
    territory_mask=None,
):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    M = (np.asarray(crack_mask_u8) > 0).astype(np.uint8)
    H, W = M.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Old-style baseline: gray image with dominant GT mask overlay.
    gray = None
    if original_image is not None:
        img = np.asarray(original_image)
        if img.ndim == 3 and img.shape[:2] == M.shape[:2]:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    if gray is None:
        gray = np.zeros((H, W), np.float32)
    overlay = np.clip(gray * 0.25 + M.astype(np.float32) * 0.75, 0, 1)
    overlay_rgb = np.stack([overlay] * 3, axis=-1)

    # Expand crop using all geometry + normals so nothing gets cut.
    all_pts = []
    for S in (all_manual or []):
        S = np.asarray(S, float)
        if S.ndim == 2 and S.shape[1] == 2 and len(S) >= 2:
            all_pts.append(S)
    for S in (all_pred or []):
        S = np.asarray(S, float)
        if S.ndim == 2 and S.shape[1] == 2 and len(S) >= 2:
            all_pts.append(S)
    normals_iter = []
    if isinstance(normals, dict):
        normals_iter = [normals]
    elif isinstance(normals, (list, tuple)):
        normals_iter = [n for n in normals if isinstance(n, dict)]
    for nrm in normals_iter:
        e1 = _split_xy_none_seps(nrm.get("edge1_x", []), nrm.get("edge1_y", []))
        e2 = _split_xy_none_seps(nrm.get("edge2_x", []), nrm.get("edge2_y", []))
        for S in (e1 + e2):
            S = np.asarray(S, float)
            if S.ndim == 2 and S.shape[1] == 2 and len(S) >= 2:
                all_pts.append(S)

    x0, y0, x1, y1 = 0, 0, W, H
    if all_pts:
        P = np.vstack(all_pts)
        x0 = int(max(0, np.floor(P[:, 0].min()) - 5))
        y0 = int(max(0, np.floor(P[:, 1].min()) - 5))
        x1 = int(min(W, np.ceil(P[:, 0].max()) + 5))
        y1 = int(min(H, np.ceil(P[:, 1].max()) + 5))
        if x1 <= x0 or y1 <= y0:
            x0, y0, x1, y1 = 0, 0, W, H

    ax.imshow(overlay_rgb[y0:y1, x0:x1])

    if territory_mask is not None:
        T = (np.asarray(territory_mask) > 0).astype(np.uint8)
        if T.shape[:2] == M.shape[:2]:
            ax.imshow(T[y0:y1, x0:x1], cmap="Greens", alpha=0.25)

    for S in (all_manual or []):
        S = np.asarray(S, float)
        if S.ndim == 2 and S.shape[1] == 2 and len(S) >= 2:
            ax.plot(S[:, 0] - x0, S[:, 1] - y0, "--", color="yellow", linewidth=2.0)
    for S in (all_pred or []):
        S = np.asarray(S, float)
        if S.ndim == 2 and S.shape[1] == 2 and len(S) >= 2:
            ax.plot(S[:, 0] - x0, S[:, 1] - y0, "-", color=pred_color, linewidth=2.0)

    normals_iter = []
    if isinstance(normals, dict):
        normals_iter = [normals]
    elif isinstance(normals, (list, tuple)):
        normals_iter = [n for n in normals if isinstance(n, dict)]
    for nrm in normals_iter:
        e1 = _split_xy_none_seps(nrm.get("edge1_x", []), nrm.get("edge1_y", []))
        e2 = _split_xy_none_seps(nrm.get("edge2_x", []), nrm.get("edge2_y", []))
        for a, b in zip(e1, e2):
            aa = np.asarray(a, float)
            bb = np.asarray(b, float)
            n = min(len(aa), len(bb))
            if n <= 0:
                continue
            step = max(1, n // 50)
            for i in range(0, n, step):
                p1 = aa[i]
                p2 = bb[i]
                ax.plot([p1[0] - x0, p2[0] - x0], [p1[1] - y0, p2[1] - y0], color="blue", lw=0.9, alpha=0.7)

    handles = [
        Line2D([], [], color="orange", lw=2.0, linestyle="--", label="Manual"),
        Line2D([], [], color=pred_color, lw=2.0, linestyle="-", label="Prediction"),
        Line2D([], [], color="blue", lw=1.2, linestyle="-", label="Normals"),
    ]
    ax.legend(handles=handles, loc="lower right", framealpha=0.9, fontsize=9)
    ax.set_title(str(title))
    ax.axis("off")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def debug_plot_branch_midlines(
    img,
    branch_bbox,        # (x, y, w, h) in global image coordinates
    gt_midline,         # (N,2) global xy or None
    pred_midline,       # (M,2) global xy or None
    save_path,
    title="",
):
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    if img is None or branch_bbox is None:
        return

    x, y, w, h = map(int, branch_bbox)
    arr = np.asarray(img)
    if arr.ndim < 2:
        return
    H, W = arr.shape[:2]

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(W, x + max(1, w))
    y1 = min(H, y + max(1, h))
    if x1 <= x0 or y1 <= y0:
        return

    crop = arr[y0:y1, x0:x1]

    plt.figure(figsize=(5, 5))
    if crop.ndim == 2:
        plt.imshow(crop, cmap="gray")
    else:
        plt.imshow(crop)

    if gt_midline is not None:
        gt = np.asarray(gt_midline, float)
        if gt.ndim == 2 and gt.shape[1] == 2 and len(gt) > 0:
            plt.plot(gt[:, 0] - x0, gt[:, 1] - y0, "g-", linewidth=2, label="GT")

    if pred_midline is not None:
        pr = np.asarray(pred_midline, float)
        if pr.ndim == 2 and pr.shape[1] == 2 and len(pr) > 0:
            plt.plot(pr[:, 0] - x0, pr[:, 1] - y0, "r-", linewidth=2, label="Pred")

    plt.title(title)
    if (
        (gt_midline is not None and len(np.asarray(gt_midline)) > 0)
        or (pred_midline is not None and len(np.asarray(pred_midline)) > 0)
    ):
        plt.legend()
    plt.axis("off")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def debug_plot_cc_domain(mask_u8, cc_debug, save_path, title=""):
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    dom = (np.asarray(mask_u8) > 0).astype(np.uint8)
    if dom.ndim != 2:
        return

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(dom, cmap="gray")

    def _pt(v):
        try:
            a = np.asarray(v, float).reshape(-1)
            if a.size >= 2 and np.all(np.isfinite(a[:2])):
                return float(a[0]), float(a[1])
        except Exception:
            return None
        return None

    sb = _pt((cc_debug or {}).get("start_before"))
    eb = _pt((cc_debug or {}).get("end_before"))
    sa = _pt((cc_debug or {}).get("start_after"))
    ea = _pt((cc_debug or {}).get("end_after"))
    if sb is not None:
        ax.plot(sb[0], sb[1], marker="x", color="red", markersize=8, mew=2)
    if eb is not None:
        ax.plot(eb[0], eb[1], marker="x", color="orange", markersize=8, mew=2)
    if sa is not None:
        ax.plot(sa[0], sa[1], marker="o", color="cyan", markersize=5)
    if ea is not None:
        ax.plot(ea[0], ea[1], marker="o", color="lime", markersize=5)

    cc_count = int((cc_debug or {}).get("cc_count", 0) or 0)
    same_cc = bool((cc_debug or {}).get("same_cc_after", False))
    chosen_cc = int((cc_debug or {}).get("chosen_cc", 0) or 0)
    ax.set_title(f"{title}\ncc={cc_count} same_cc={same_cc} chosen_cc={chosen_cc}")
    ax.axis("off")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_gt_branch_segments_only(original_image, ux, uy, bbox_local, segs_branch, out_png, title):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    if bbox_local is None:
        return
    bx, by, bw, bh = [int(v) for v in bbox_local]
    gx, gy = int(ux + bx), int(uy + by)
    if bw <= 0 or bh <= 0:
        return

    arr = np.asarray(original_image)
    H, W = arr.shape[:2]
    x0, y0 = max(0, gx), max(0, gy)
    x1, y1 = min(W, gx + bw), min(H, gy + bh)
    if x1 <= x0 or y1 <= y0:
        return
    crop = arr[y0:y1, x0:x1]

    plt.figure(figsize=(6, 6))
    plt.imshow(crop)
    first = True
    for S in (segs_branch or []):
        S = np.asarray(S, float)
        if S.ndim != 2 or S.shape[1] != 2 or len(S) < 2:
            continue
        S_local = S.copy()
        S_local[:, 0] -= float(bx)
        S_local[:, 1] -= float(by)
        plt.plot(S_local[:, 0], S_local[:, 1], linewidth=2, label="GT segments" if first else None)
        first = False

    plt.title(title)
    if not first:
        plt.legend()
    plt.axis("off")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def _plot_branch_audit(
    *,
    original_image,
    gt_mask,
    atomic,
    members,
    kept_segs,
    segments_meta,
    dom_meta,
    out_png,
    tag_name,
):
    """Visualize raw per-branch input segments vs dominance survivors."""
    try:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        img = np.asarray(original_image)
        if img.ndim != 3:
            return
        canvas = img.copy()
        mask_u8 = (np.asarray(gt_mask) > 0).astype(np.uint8)
        if mask_u8.shape[:2] == canvas.shape[:2]:
            overlay = canvas.copy()
            overlay[mask_u8 > 0] = np.array([255, 255, 255], dtype=np.uint8)
            canvas = cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0.0)

        branches = dom_meta.get("branches", []) if isinstance(dom_meta, dict) else []
        branch_order = dom_meta.get("order", []) if isinstance(dom_meta, dict) else []
        if not branch_order and branches:
            branch_order = [int((b or {}).get("branch_id", -1)) for b in branches if int((b or {}).get("branch_id", -1)) >= 0]

        branch_to_atomic = {}
        for b in branches:
            if not isinstance(b, dict):
                continue
            bi = int(b.get("branch_id", -1))
            aids = [str(a) for a in (b.get("atomic_ids", []) or [])]
            if bi >= 0:
                branch_to_atomic[bi] = aids

        # Fallback if branch stats missing: infer from segment meta.
        if not branch_to_atomic:
            for m in (segments_meta or []):
                if not isinstance(m, dict):
                    continue
                bi = int(m.get("branch_id", -1))
                aid = m.get("atomic_id", None)
                if bi < 0 or aid is None:
                    continue
                branch_to_atomic.setdefault(bi, [])
                a = str(aid)
                if a not in branch_to_atomic[bi]:
                    branch_to_atomic[bi].append(a)

        kept_by_branch_atomic = {}
        for m in (segments_meta or []):
            if not isinstance(m, dict):
                continue
            bi = int(m.get("branch_id", -1))
            aid = m.get("atomic_id", None)
            if bi < 0 or aid is None:
                continue
            kept_by_branch_atomic.setdefault(bi, set()).add(str(aid))

        # Build raw per-branch user segments from atomic ids.
        raw_branch_segs = {}
        for bi, aids in branch_to_atomic.items():
            segs = []
            for aid in aids:
                cr = (atomic or {}).get(str(aid), {}) or {}
                S = np.asarray(cr.get("midline", []), float)
                if S.ndim == 2 and S.shape[1] == 2 and len(S) >= 2:
                    segs.append((str(aid), S))
            if segs:
                raw_branch_segs[int(bi)] = segs

        # Draw raw branch segments with branch colors; dropped segments in red.
        from matplotlib.lines import Line2D
        cmap = plt.get_cmap("tab10", max(1, len(raw_branch_segs)))
        fig, ax = plt.subplots(figsize=(9, 9), dpi=180)
        ax.imshow(canvas[:, :, ::-1])  # BGR->RGB

        order_use = [int(b) for b in branch_order if int(b) in raw_branch_segs]
        order_use += [b for b in sorted(raw_branch_segs.keys()) if b not in order_use]
        branch_colors = {int(bi): cmap(i) for i, bi in enumerate(order_use)}

        branch_handles = []
        for rank, bi in enumerate(order_use):
            base_color = branch_colors.get(int(bi), cmap(rank))
            kept_ids = kept_by_branch_atomic.get(int(bi), set())
            for aid, S in raw_branch_segs.get(int(bi), []):
                dropped = str(aid) not in kept_ids
                col = "#d62728" if dropped else base_color
                lw = 2.0 if dropped else 1.0
                alpha = 0.9 if dropped else 0.65
                ax.plot(S[:, 0], S[:, 1], color=col, linewidth=lw, alpha=alpha)
            aids = branch_to_atomic.get(int(bi), [])
            branch_handles.append(
                Line2D([0], [0], color=base_color, lw=3, label=f"branch {int(bi)}: {aids}")
            )

        # Overlay kept output segments by branch color, thicker.
        for i, S in enumerate(kept_segs or []):
            A = np.asarray(S, float)
            if A.ndim == 2 and A.shape[1] == 2 and len(A) >= 2:
                m = segments_meta[i] if i < len(segments_meta) and isinstance(segments_meta[i], dict) else {}
                bi = int(m.get("branch_id", -1))
                col = branch_colors.get(bi, (0.0, 1.0, 0.4, 1.0))
                ax.plot(A[:, 0], A[:, 1], color=col, linewidth=3.0, alpha=0.95)

        extra_handles = [
            Line2D([0], [0], color="#d62728", lw=2, label="dropped/clipped"),
            Line2D([0], [0], color="#111111", lw=1, label="raw input (thin)"),
            Line2D([0], [0], color="#111111", lw=3, label="kept output (thick)"),
        ]
        ax.legend(
            handles=(branch_handles + extra_handles),
            loc="upper left",
            fontsize=7,
            framealpha=0.9,
        )
        ax.set_title(f"Branch Audit - {tag_name}")
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
    except Exception as _e:
        print(f"[BRANCH AUDIT] failed for {tag_name}: {_e}")


def _plot_centering_audit_entry(
    *,
    entry,
    original_image,
    out_png,
):
    """Plot GT combined branches vs method centered outputs for one combined entry."""
    try:
        if not isinstance(entry, dict) or str(entry.get("type", "")) != "combined":
            return
        gt_segs = [np.asarray(S, float) for S in (entry.get("midline_segments", []) or [])]
        gt_meta = entry.get("midline_segments_meta", []) if isinstance(entry.get("midline_segments_meta", []), list) else []
        gt_valid = []
        for i, S in enumerate(gt_segs):
            if S.ndim == 2 and S.shape[1] == 2 and len(S) >= 2:
                gt_valid.append((i, S))
        if not gt_valid:
            return

        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        img = np.asarray(original_image)
        if img.ndim != 3:
            return

        from matplotlib.lines import Line2D
        branch_ids = sorted({
            int((gt_meta[i] if i < len(gt_meta) and isinstance(gt_meta[i], dict) else {}).get("branch_id", -1))
            for i, _ in gt_valid
        })
        branch_ids = [b for b in branch_ids if b >= 0]
        bmap = plt.get_cmap("tab10", max(1, len(branch_ids)))
        branch_colors = {int(b): bmap(i) for i, b in enumerate(branch_ids)}

        methods_blob = entry.get("method_variants", {}) if isinstance(entry.get("method_variants", {}), dict) else {}
        method_keys = [k for k in methods_blob.keys()]
        mmap = plt.get_cmap("tab20", max(1, len(method_keys)))
        method_colors = {str(k): mmap(i) for i, k in enumerate(method_keys)}

        fig, ax = plt.subplots(figsize=(10, 10), dpi=180)
        ax.imshow(img[:, :, ::-1])

        for i, S in gt_valid:
            m = gt_meta[i] if i < len(gt_meta) and isinstance(gt_meta[i], dict) else {}
            bi = int(m.get("branch_id", -1))
            col = branch_colors.get(bi, (0.2, 0.9, 0.2, 1.0))
            ax.plot(S[:, 0], S[:, 1], color=col, linewidth=3.0, alpha=0.95)

        for mk in method_keys:
            mv = methods_blob.get(mk, {}) if isinstance(methods_blob.get(mk, {}), dict) else {}
            m_segs = [np.asarray(S, float) for S in (mv.get("midline_segments", []) or [])]
            col = method_colors.get(str(mk), (1.0, 0.6, 0.0, 1.0))
            for S in m_segs:
                if S.ndim == 2 and S.shape[1] == 2 and len(S) >= 2:
                    ax.plot(S[:, 0], S[:, 1], color=col, linewidth=1.7, alpha=0.80)

        branch_handles = [Line2D([0], [0], color=branch_colors[b], lw=3, label=f"GT branch {int(b)}") for b in branch_ids]
        method_handles = [Line2D([0], [0], color=method_colors[k], lw=2, label=f"{str(k)}") for k in method_keys]
        ax.legend(handles=(branch_handles + method_handles), loc="upper left", fontsize=7, framealpha=0.9, ncol=2)
        ax.set_title(f"Centering Audit - combined {entry.get('id', '')}")
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
    except Exception as _e:
        print(f"[CENTERING AUDIT] failed for entry={entry.get('id', '?') if isinstance(entry, dict) else '?'}: {_e}")


def _plot_territory_debug(
    *,
    original_image,
    gt_mask_full,
    branch_terr_masks,
    branch_bite_masks,
    branch_allowed_masks,
    branch_order,
    dom_meta,
    ux,
    uy,
    uw,
    uh,
    out_png,
):
    import matplotlib.cm as cm
    from matplotlib.patches import Patch

    try:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        img = np.asarray(original_image)
        if img.ndim != 3:
            return
        H, W = img.shape[:2]
        gt_full = (np.asarray(gt_mask_full) > 0)
        _ys, _xs = np.where(gt_full)
        if _xs.size > 0 and _ys.size > 0:
            _pad = 25
            x0 = max(0, int(_xs.min()) - _pad)
            y0 = max(0, int(_ys.min()) - _pad)
            x1 = min(W, int(_xs.max()) + 1 + _pad)
            y1 = min(H, int(_ys.max()) + 1 + _pad)
        else:
            x0 = max(0, int(ux))
            y0 = max(0, int(uy))
            x1 = min(W, int(ux + uw))
            y1 = min(H, int(uy + uh))
        if x1 <= x0 or y1 <= y0:
            return

        img_crop = img[y0:y1, x0:x1]
        fig_w = max(18.0, float(x1 - x0) / 60.0)
        fig_h = max(10.0, float(y1 - y0) / 60.0)
        fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h), dpi=120)
        titles = ["Territory per branch", "Bite per branch", "Allowed mask per branch"]

        order_use = [int(b) for b in (branch_order or [])]
        cmap = cm.get_cmap("tab10", max(len(order_use), 1))
        branch_colors = {int(bi): cmap(i) for i, bi in enumerate(order_use)}

        gt_crop = gt_full[y0:y1, x0:x1]
        crop_h, crop_w = gt_crop.shape[:2]

        def _local_mask_to_crop(mask_local):
            """Project a local-frame mask (origin at ux,uy) into current global crop coords."""
            if mask_local is None:
                return None
            m = (np.asarray(mask_local) > 0)
            if m.ndim < 2:
                return None
            mh, mw = m.shape[:2]
            out = np.zeros((crop_h, crop_w), dtype=bool)

            # Overlap in global coordinates between crop box and local-mask box.
            gx0 = max(int(x0), int(ux))
            gy0 = max(int(y0), int(uy))
            gx1 = min(int(x1), int(ux + mw))
            gy1 = min(int(y1), int(uy + mh))
            if gx1 <= gx0 or gy1 <= gy0:
                return out

            # Source indices in local mask.
            sx0 = int(gx0 - ux)
            sy0 = int(gy0 - uy)
            sx1 = int(gx1 - ux)
            sy1 = int(gy1 - uy)

            # Destination indices in crop-local coordinates.
            dx0 = int(gx0 - x0)
            dy0 = int(gy0 - y0)
            dx1 = int(gx1 - x0)
            dy1 = int(gy1 - y0)

            out[dy0:dy1, dx0:dx1] = m[sy0:sy1, sx0:sx1]
            return out

        for ax, title in zip(axes, titles):
            ax.imshow(img_crop[:, :, ::-1])  # BGR -> RGB
            gt_overlay = np.zeros((gt_crop.shape[0], gt_crop.shape[1], 4), float)
            gt_overlay[..., :3] = (1.0, 1.0, 1.0)
            gt_overlay[..., 3] = gt_crop.astype(float) * 0.12
            ax.imshow(gt_overlay)
            ax.contour(gt_crop.astype(float), levels=[0.5], colors=["white"], linewidths=0.5, alpha=0.5)
            ax.set_title(title, fontsize=8)
            ax.axis("off")

        branch_stats = {
            int((b or {}).get("branch_id", -1)): (b if isinstance(b, dict) else {})
            for b in (dom_meta.get("branches", []) if isinstance(dom_meta, dict) else [])
            if int((b or {}).get("branch_id", -1)) >= 0
        }

        for bi in order_use:
            bi = int(bi)
            color = branch_colors.get(bi, (1, 1, 1, 1))

            terr = branch_terr_masks.get(bi, None)
            if terr is not None:
                tc = _local_mask_to_crop(terr).astype(float)
                if np.any(tc > 0):
                    overlay = np.zeros((*tc.shape, 4), float)
                    overlay[..., :3] = color[:3]
                    overlay[..., 3] = tc * 0.35
                    axes[0].imshow(overlay)
                    ys, xs = np.where(tc > 0)
                    if len(xs):
                        aids = branch_stats.get(bi, {}).get("atomic_ids", [])
                        axes[0].text(
                            float(np.mean(xs)),
                            float(np.mean(ys)),
                            f"b{bi}\n{aids}",
                            fontsize=5,
                            color="white",
                            ha="center",
                            va="center",
                            bbox=dict(boxstyle="round,pad=0.1", facecolor=color[:3], alpha=0.6),
                        )

            bite = branch_bite_masks.get(bi, None)
            if bite is not None:
                bc = _local_mask_to_crop(bite).astype(float)
                if np.any(bc > 0):
                    red_overlay = np.zeros((*bc.shape, 4), float)
                    red_overlay[..., 0] = 1.0
                    red_overlay[..., 3] = bc * 0.55
                    axes[1].imshow(red_overlay)

            allowed = branch_allowed_masks.get(bi, None)
            if allowed is not None:
                ac = _local_mask_to_crop(allowed).astype(float)
                if np.any(ac > 0):
                    axes[2].contour(ac, levels=[0.5], colors=[color[:3]], linewidths=1.0)

        handles = [
            Patch(
                facecolor=branch_colors[int(bi)][:3],
                label=f"b{int(bi)}: {branch_stats.get(int(bi), {}).get('atomic_ids', [])}",
            )
            for bi in order_use
        ]
        if handles:
            axes[0].legend(handles=handles, fontsize=5, loc="lower right", framealpha=0.7)

        plt.suptitle(f"Territory debug - {os.path.basename(out_png)}", fontsize=8)
        plt.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
    except Exception as _e:
        print(f"[TERRITORY DEBUG] failed: {_e}")


def _plot_gt_branch_isolation(segs_branch, S_branch, out_dir, base_name, branch_id):
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    os.makedirs(out_dir, exist_ok=True)
    raw_png = os.path.join(out_dir, f"{base_name}_branch{int(branch_id)}_raw_segments.jpg")
    stacked_png = os.path.join(out_dir, f"{base_name}_branch{int(branch_id)}_stacked_vs_raw.jpg")

    plt.figure(figsize=(6, 6))
    for j, Sseg in enumerate(segs_branch):
        arr = np.asarray(Sseg, float)
        if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) >= 2:
            plt.plot(arr[:, 0], arr[:, 1], linewidth=2, label=f"seg{int(j)}")
    plt.gca().invert_yaxis()
    plt.title(f"RAW SEGMENTS (branch {int(branch_id)})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(raw_png, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 6))
    arr_stacked = np.asarray(S_branch, float)
    if arr_stacked.ndim == 2 and arr_stacked.shape[1] == 2 and len(arr_stacked) >= 2:
        plt.plot(arr_stacked[:, 0], arr_stacked[:, 1], "r-", linewidth=2, label="stacked")
    for Sseg in segs_branch:
        arr = np.asarray(Sseg, float)
        if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) >= 2:
            plt.plot(arr[:, 0], arr[:, 1], "--", linewidth=1)
    plt.gca().invert_yaxis()
    plt.title(f"STACKED vs RAW (branch {int(branch_id)})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(stacked_png, bbox_inches="tight")
    plt.close()
    return raw_png, stacked_png


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

    def _pick_local_mask_for_cost(a, crack_mask_u8=None, domain_u8=None, territory=None):
        Hc, Wc = np.asarray(a).shape[:2]
        for cand in (territory, domain_u8, crack_mask_u8):
            if cand is None:
                continue
            c = np.asarray(cand)
            if c.ndim >= 2 and c.shape[:2] == (Hc, Wc):
                return (c > 0)
        return None

    def _masked_cost(arr):
        if arr is None:
            return None
        a = np.asarray(arr, np.float32)
        local_mask = _pick_local_mask_for_cost(a, crack_mask_u8=crack_mask_u8)
        if local_mask is None:
            print(f"[PLOT DIAG][SKIP MASK] no shape-matched mask for costmap {a.shape}")
            return a
        valid = local_mask & np.isfinite(a) & (a < np.float32(1e8))
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
        out = np.full_like(a, np.nan, dtype=np.float32)
        # Kill sentinels first — before any mask check — so 1e6 background
        # never contaminates normalization even when mask shape doesn't match
        valid = np.isfinite(a) & (a < np.float32(1e5))
        if not np.any(valid):
            return out
        local_mask = _pick_local_mask_for_cost(a, crack_mask_u8=crack_mask_u8)
        if local_mask is not None:
            valid &= local_mask
        # Robust percentile normalize over valid pixels instead of blind clip to [0,1]
        # This handles both dt_norm (already 0-1) and raw DT (0-42px) correctly
        vals = a[valid]
        lo = float(np.percentile(vals, 2))
        hi = float(np.percentile(vals, 98))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-9:
            out[valid] = np.clip(a[valid], 0.0, 1.0)
        else:
            out[valid] = np.clip((a[valid] - lo) / (hi - lo), 0.0, 1.0)
        return out.astype(np.float32, copy=False)

    panels = []
    costmaps_dict = costmaps if isinstance(costmaps, dict) else None
    if costmaps_dict is None and costmap is not None:
        costmaps_dict = {"selected": costmap}

    if isinstance(costmaps_dict, dict):
        selected = np.asarray(costmaps_dict.get("selected"), np.float32) if "selected" in costmaps_dict else None
        selected_key = str(costmaps_dict.get("selected_key", ""))

        if selected is None:
            for k in (
                "dt_ridge_color_depth",
                "dt_ridge_valley_depth",
                "dt_ridge_valley",
                "dt_depth",
                "dt",
            ):
                if k in costmaps_dict:
                    selected = np.asarray(costmaps_dict[k], np.float32)
                    if not selected_key:
                        selected_key = k
                    break

        # Core signals only.
        if dt_norm is not None:
            panels.append(("DT", _unit_as_is(dt_norm), "viridis"))

        if "ridge_valley" in costmaps_dict:
            panels.append(("Ridge/Valley", _unit_as_is(costmaps_dict["ridge_valley"]), "magma"))
        elif "rgb_cue" in costmaps_dict:
            panels.append(("RGB cue", _unit_as_is(costmaps_dict["rgb_cue"]), "magma"))

        if depth_norm is not None:
            panels.append(("Depth map", _unit_as_is(depth_norm), "viridis"))

        if recess_norm is not None:
            panels.append(("Depth signal", _unit_as_is(recess_norm), "plasma"))

        if selected is not None:
            panels.append((f"Final cost ({selected_key or 'selected'})", _masked_cost(selected), "inferno"))
    elif dt_norm is not None:
        panels.append(("DT", _unit_as_is(dt_norm), "viridis"))

    panels = [(lbl, arr, cmap) for (lbl, arr, cmap) in panels if arr is not None]

    n = max(1, len(panels))
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 4.6), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (label, arr, cmap) in zip(axes, panels):
        arr_crop = np.asarray(arr, np.float32)[y0:y1, x0:x1]
        cmap_obj = plt.cm.get_cmap(cmap).copy()
        cmap_obj.set_bad(color="black")
        cmap_obj.set_under(color="black")
        ax.imshow(arr_crop, cmap=cmap_obj, vmin=0.0, vmax=1.0)
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    supt = str(title)
    if method_label:
        supt = f"{supt} [{method_label}]"
    fig.suptitle(supt, fontsize=11)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_global_cost_maps(branches, full_shape):
    H, W = full_shape
    global_cost = np.full((H, W), np.nan, dtype=np.float32)
    global_dt = np.full((H, W), np.nan, dtype=np.float32)
    global_rgb = np.full((H, W), np.nan, dtype=np.float32)
    global_ridge = np.full((H, W), np.nan, dtype=np.float32)
    global_depth = np.full((H, W), np.nan, dtype=np.float32)
    global_recess = np.full((H, W), np.nan, dtype=np.float32)
    global_count = np.zeros((H, W), np.float32)
    winner_id = np.full((H, W), -1, np.int32)

    def _ks(a):
        if a is None:
            return None
        a = np.asarray(a, np.float32)
        return np.where(np.isfinite(a) & (a < np.float32(1e5)), a, np.float32(np.nan))

    def _write(dst, src, y0, y1, x0, x1, mode="max"):
        if src is None:
            return
        s = np.asarray(src, np.float32)
        if s.ndim < 2:
            return
        sub = dst[y0:y1, x0:x1]
        s = s[:sub.shape[0], :sub.shape[1]]
        valid = np.isfinite(s)
        if not np.any(valid):
            return
        if mode == "min":
            write = valid & (~np.isfinite(sub) | (s < sub))
        else:
            write = valid & (~np.isfinite(sub) | (s > sub))
        if np.any(write):
            sub[write] = s[write]
            dst[y0:y1, x0:x1] = sub

    ordered = sorted(
        enumerate(branches or []),
        key=lambda t: -float((t[1] or {}).get("area", 0.0) or 0.0),
    )

    for bi, b in ordered:
        dt_src = _ks(b.get("dt_norm", None))
        rgb_src = _ks(b.get("rgb", None))
        ridge_src = _ks(b.get("ridge", None))
        depth_src = _ks(b.get("depth", None))
        recess_src = _ks(b.get("recess_norm", None))
        cost_src = _ks(b.get("final_cost", None))

        bbox = b.get("bbox", None)
        if bbox is None or len(bbox) != 4:
            continue
        y0, y1, x0, x1 = [int(v) for v in bbox]
        if y1 <= y0 or x1 <= x0:
            continue

        # Compute clip extents from whichever map exists first.
        ref = None
        for v in (cost_src, dt_src, rgb_src, ridge_src, depth_src, recess_src):
            if v is not None:
                a = np.asarray(v)
                if a.ndim >= 2:
                    ref = a
                    break
        if ref is None:
            continue
        yy1 = min(H, y0 + int(ref.shape[0]))
        xx1 = min(W, x0 + int(ref.shape[1]))
        if yy1 <= y0 or xx1 <= x0:
            continue

        if cost_src is not None:
            c = np.asarray(cost_src, np.float32)[: (yy1 - y0), : (xx1 - x0)]
            valid_c = np.isfinite(c)
            if np.any(valid_c):
                sub = global_cost[y0:yy1, x0:xx1]
                write_c = valid_c & (~np.isfinite(sub) | (c < sub))
                if np.any(write_c):
                    sub[write_c] = c[write_c]
                    global_cost[y0:yy1, x0:xx1] = sub
                    winners = winner_id[y0:yy1, x0:xx1]
                    winners[write_c] = int(bi)
                    winner_id[y0:yy1, x0:xx1] = winners
                global_count[y0:yy1, x0:xx1][valid_c] += 1.0

        _write(global_dt, dt_src, y0, yy1, x0, xx1, mode="max")
        _write(global_rgb, rgb_src, y0, yy1, x0, xx1, mode="max")
        _write(global_ridge, ridge_src, y0, yy1, x0, xx1, mode="max")
        _write(global_depth, depth_src, y0, yy1, x0, xx1, mode="max")
        _write(global_recess, recess_src, y0, yy1, x0, xx1, mode="max")

    return {
        "dt": global_dt.astype(np.float32, copy=False),
        "rgb": global_rgb.astype(np.float32, copy=False),
        "ridge": global_ridge.astype(np.float32, copy=False),
        "depth": global_depth.astype(np.float32, copy=False),
        "recess": global_recess.astype(np.float32, copy=False),
        "cost": global_cost.astype(np.float32, copy=False),
        "count": global_count.astype(np.float32, copy=False),
        "winner_id": winner_id.astype(np.int32, copy=False),
    }


def save_global_cost_panel(
    maps,
    out_path,
    *,
    crack_mask_u8=None,
    bbox_xywh=None,
    title="Global Atomic Cost Panel",
):
    import matplotlib.pyplot as plt

    def _ks(a):
        if a is None:
            return None
        a = np.asarray(a, np.float32)
        return np.where(np.isfinite(a) & (a < np.float32(1e5)), a, np.float32(np.nan))

    def _norm_cost(arr, mask=None):
        if arr is None:
            return None
        a = np.asarray(arr, np.float32)
        out = np.full_like(a, np.nan, dtype=np.float32)

        valid = np.isfinite(a) & (a < np.float32(1e8))
        if mask is not None:
            m = np.asarray(mask)
            if m.ndim >= 2 and m.shape[:2] == a.shape[:2]:
                valid &= (m > 0)

        vals = a[valid]
        if vals.size == 0:
            return None

        lo = float(np.percentile(vals, 2))
        hi = float(np.percentile(vals, 98))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-9:
            out[valid] = 0.5
            return out
        out[valid] = (a[valid] - lo) / (hi - lo)
        return out

    def _norm_rgb(arr, mask=None):
        if arr is None:
            return None
        a = np.asarray(arr, np.float32)
        out = np.full_like(a, np.nan, dtype=np.float32)
        valid = np.isfinite(a)
        if mask is not None:
            m = np.asarray(mask)
            if m.ndim >= 2 and m.shape[:2] == a.shape[:2]:
                valid &= (m > 0)
        vals = a[valid]
        if vals.size == 0:
            return None
        lo = float(np.percentile(vals, 5))
        hi = float(np.percentile(vals, 99.5))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-9:
            out[valid] = 0.5
            return out
        out[valid] = (a[valid] - lo) / (hi - lo)
        return out

    def _unit(arr, mask=None):
        if arr is None:
            return None
        a = np.asarray(arr, np.float32)
        out = np.full_like(a, np.nan, dtype=np.float32)
        valid = np.isfinite(a)
        if mask is not None:
            m = np.asarray(mask)
            if m.ndim >= 2 and m.shape[:2] == a.shape[:2]:
                valid &= (m > 0)
        if not np.any(valid):
            return None
        out[valid] = np.clip(a[valid], 0.0, 1.0)
        return out

    maps = maps if isinstance(maps, dict) else {}
    dt = _ks(maps.get("dt", None))
    depth = _ks(maps.get("depth", None))
    recess = _ks(maps.get("recess", None))
    cost = _ks(maps.get("cost", None))
    rgb = _ks(maps.get("rgb", None))
    ridge = _ks(maps.get("ridge", None))
    if ridge is None:
        ridge = _ks(maps.get("ridge_valley", None))

    H = W = None
    M = None
    if crack_mask_u8 is not None:
        M = (np.asarray(crack_mask_u8) > 0)
        H, W = M.shape[:2]
    else:
        for v in (dt, rgb, ridge, depth, recess, cost):
            if v is None:
                continue
            a = np.asarray(v)
            if a.ndim >= 2:
                H, W = a.shape[:2]
                break
    if H is None or W is None:
        H, W = 1, 1

    if bbox_xywh is not None and len(bbox_xywh) == 4:
        x, y, w, h = map(int, bbox_xywh)
        pad = 25
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(W, x + w + pad)
        y1 = min(H, y + h + pad)
    elif M is not None:
        ys, xs = np.where(M > 0)
        if xs.size:
            pad = 25
            x0 = max(0, int(xs.min()) - pad)
            y0 = max(0, int(ys.min()) - pad)
            x1 = min(W, int(xs.max()) + 1 + pad)
            y1 = min(H, int(ys.max()) + 1 + pad)
        else:
            x0, y0, x1, y1 = 0, 0, W, H
    else:
        x0, y0, x1, y1 = 0, 0, W, H

    dt_v = _norm_cost(dt, mask=crack_mask_u8)
    rgb_v = _norm_rgb(rgb, mask=crack_mask_u8)
    ridge_v = _norm_rgb(ridge, mask=crack_mask_u8)
    depth_v = _unit(depth, mask=crack_mask_u8)
    recess_v = _unit(recess, mask=crack_mask_u8)
    cost_v = _norm_cost(cost, mask=crack_mask_u8)

    def _has_signal(arr):
        if arr is None:
            return False
        a = np.asarray(arr, np.float32)
        if not np.any(np.isfinite(a)):
            return False
        vals = a[np.isfinite(a)]
        return bool((vals.size > 10) and (float(np.nanstd(vals)) > 1e-6))

    panels = []
    if _has_signal(dt_v):
        panels.append(("DT cost", dt_v, "inferno"))
    if _has_signal(rgb_v):
        panels.append(("RGB cue", rgb_v, "magma"))
    if _has_signal(ridge_v):
        panels.append(("Ridge / Valley", ridge_v, "cividis"))
    if _has_signal(depth_v):
        panels.append(("Depth map", depth_v, "viridis"))
    if _has_signal(recess_v):
        panels.append(("Depth signal", recess_v, "plasma"))
    if cost_v is not None and np.any(np.isfinite(cost_v)):
        panels.append(("Final cost", cost_v, "viridis"))

    if not panels:
        return

    n = len(panels)
    fig, axs = plt.subplots(1, n, figsize=(4.5 * n, 4.5), sharex=True, sharey=True)
    if n == 1:
        axs = [axs]
    for ax, (label, arr, cmap) in zip(axs, panels):
        arr_crop = np.asarray(arr, np.float32)[y0:y1, x0:x1]
        cmap_obj = plt.cm.get_cmap(cmap).copy()
        cmap_obj.set_bad(color="black")
        cmap_obj.set_under(color="black")
        ax.imshow(arr_crop, cmap=cmap_obj, vmin=1e-6, vmax=1.0)
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    fig.suptitle(str(title))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_overlap_panel(count_map, out_path, title="Overlap Count"):
    fig = plt.figure(figsize=(6, 5))
    plt.imshow(np.asarray(count_map, np.float32), cmap="viridis")
    plt.title(str(title))
    plt.colorbar()
    plt.axis("off")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# def save_winner_panel(winner_id_map, out_path, title="Winner Branch Id"):
#     fig = plt.figure(figsize=(6, 5))
#     arr = np.asarray(winner_id_map, np.int32)
#     masked = np.ma.masked_where(arr < 0, arr)
#     cmap = plt.cm.get_cmap("tab20").copy()
#     cmap.set_bad(color="black")
#     plt.imshow(masked, cmap=cmap)
#     plt.title(str(title))
#     plt.colorbar()
#     plt.axis("off")
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     fig.savefig(out_path, bbox_inches="tight")
#     plt.close(fig)


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
      out_dir/gt_sup_dom_bite_debug_{base_name}_ccid{ccid}_<members>.jpg
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    if not isinstance(dom_meta, dict):
        _dlog(3, f"[GT_SUP DOMDBG] base={base_name} ccid={ccid} dominance_meta missing")
        return

    bite = dom_meta.get("bite", None)
    if not isinstance(bite, dict):
        _dlog(3, f"[GT_SUP DOMDBG] base={base_name} ccid={ccid} bite missing")
        return

    bb = bite.get("bbox", None)
    if not bb or len(bb) != 4:
        _dlog(3, f"[GT_SUP DOMDBG] base={base_name} ccid={ccid} bite bbox missing/invalid: {bb}")
        return

    bx, by, bw, bh = map(int, bb)
    if bw <= 0 or bh <= 0:
        _dlog(3, f"[GT_SUP DOMDBG] base={base_name} ccid={ccid} bite bbox non-positive: {bb}")
        return

    _dlog(3, f"[GT_SUP DOMDBG] base={base_name} ccid={ccid} RAW bite bbox={bb} members={members}")

    H, W = gt_mask.shape[:2]

    # -----------------------------
    # 1) RAW LOCAL union (bite frame)
    # -----------------------------
    local_union = np.zeros((bh, bw), np.uint8)

    # First try union blob (backward compatible)
    if "packbits_b64" in bite and "shape" in bite:
        u = _unpack_mask_b64({"shape": bite.get("shape"), "packbits_b64": bite.get("packbits_b64")})
        if u is None:
            _dlog(3, f"[GT_SUP DOMDBG] base={base_name} ccid={ccid} union blob decode failed")
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
                _dlog(3, f"[GT_SUP DOMDBG]  bid={bid}: BAD packed blob decode (entry keys={list(entry.keys())})")
                continue
            if u.shape != (bh, bw):
                _dlog(3, f"[GT_SUP DOMDBG]  bid={bid}: shape mismatch {u.shape} vs {(bh,bw)}")
                continue
            if np.any(u):
                local_union |= (u > 0).astype(np.uint8)

    if not np.any(local_union):
        _dlog(3, f"[GT_SUP DOMDBG] base={base_name} ccid={ccid} RAW LOCAL union is EMPTY")
        # still write a figure so you see placement context
        # (this helps diagnose �bbox exists but union empty�)
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
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
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

    fig.suptitle(f"GT dominance bite debug - base={base_name} ccid={ccid}", fontsize=11)

    os.makedirs(out_dir, exist_ok=True)
    tag = f"ccid{ccid}_" + "_".join([str(m) for m in members])
    out = os.path.join(out_dir, f"gt_sup_dom_bite_debug_{base_name}_{tag}.jpg")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    _dlog(3, f"[GT_SUP DOMDBG] wrote {out}")


def _compute_midline_metrics_task(payload):
    """
    ProcessPool worker: compute full metric dict + score for one (manual,pred) pair.
    """
    import numpy as np
    from helpers.metrics import compute_midline_metrics

    man_xy, pred_xy, tau = payload
    mm = compute_midline_metrics(auto_xy=pred_xy, man_xy=man_xy, tau=float(tau))
    if not isinstance(mm, dict):
        mm = {}
    nn = float(mm.get("nn_mean_bidirectional", np.nan))
    hd = float(mm.get("hausdorff_p95", np.nan))
    cov = float(mm.get("coverage_min", np.nan))
    score_mid = np.nan
    if np.isfinite(nn) and np.isfinite(hd) and np.isfinite(cov):
        score_mid = float(
            np.log1p(max(nn, 0.0))
            + 0.5 * np.log1p(max(hd, 0.0))
            + (1.0 - float(np.clip(cov, 0.0, 1.0)))
        )
    return {"mm": mm, "score_mid": score_mid}


def export_gt_centering_metrics(
    *,
    base_name: str,
    save_root: str,
    final_entries: list,
    combined_member_ids=None,
    profile_accumulator: dict | None = None,
    batch_plots_only: bool = False,
):
    """
    Compute per-method GT alignment metrics for the 5-method ablation family.

    Primary outputs:
      - supervision/<image>/analysis/gt_ablation_midline_metrics.csv
      - supervision/<image>/analysis/gt_ablation_midline_weighted_summary.csv
      - supervision/<image>/analysis/diagnostics/<method_key>/...
    """
    import os
    import numpy as np
    import pandas as pd
    from helpers.present_plots import plot_rs3_midline_diagnostics, plot_rs3_midline_decomposition

    sup_img_dir = os.path.join(save_root, "supervision", base_name)
    analysis_dir = os.path.join(sup_img_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    rows = []
    metric_jobs = []
    combined_member_ids = {str(x) for x in (combined_member_ids or [])}

    method_cmp_specs = [
        ("et_vs_dt", "dt"),
        ("et_vs_dt_depth", "dt_depth"),
        ("et_vs_dt_ridge_valley", "dt_ridge_valley"),
        ("et_vs_dt_ridge_valley_depth", "dt_ridge_valley_depth"),
        ("et_vs_dt_ridge_color_depth", "dt_ridge_color_depth"),
    ]

    def _coerce_seg_list(entry, *, seg_keys=(), packed_keys=()):
        for k in seg_keys:
            val = entry.get(k, None) if isinstance(entry, dict) else None
            if isinstance(val, list):
                out = [np.asarray(s, float) for s in val if s is not None and len(s) >= 2]
                if out:
                    return out
        for k in packed_keys:
            val = entry.get(k, None) if isinstance(entry, dict) else None
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

    def _queue_metric_row(meta_row, man_xy, pred_xy):
        metric_jobs.append((dict(meta_row), np.asarray(man_xy, float), np.asarray(pred_xy, float)))

    for entry in (final_entries or []):
        cid = str(entry.get("id", ""))
        kind = str(entry.get("kind", "unknown"))
        bb = entry.get("mask_bbox", None)
        bbox_area = np.nan
        if isinstance(bb, (list, tuple)) and len(bb) == 4:
            try:
                bbox_area = float(max(0, int(bb[2])) * max(0, int(bb[3])))
            except Exception:
                bbox_area = np.nan

        method_variants = entry.get("method_variants", {}) if isinstance(entry.get("method_variants", {}), dict) else {}
        if not method_variants:
            continue

        if kind == "atomic":
            et_mid = np.asarray(entry.get("midline", []), float)
            if et_mid.ndim != 2 or et_mid.shape[1] != 2 or len(et_mid) < 2:
                continue
            seg_idx = 0
            branch_id = 0

            for cmp_label, variant_id in method_cmp_specs:
                mv = method_variants.get(variant_id, {}) if isinstance(method_variants.get(variant_id, {}), dict) else {}
                pred_mid = np.asarray(mv.get("midline", []), float)
                if pred_mid.ndim != 2 or pred_mid.shape[1] != 2 or len(pred_mid) < 2:
                    continue
                _queue_metric_row({
                    "image": base_name,
                    "crack_id": cid,
                    "crack_kind": kind,
                    "comparison_label": cmp_label,
                    "variant_id": variant_id,
                    "segment_index": int(seg_idx),
                    "branch_id": int(branch_id),
                    "is_atomic": 1,
                    "is_combined": 0,
                    "is_noncombined_atomic": int(cid not in combined_member_ids),
                    "geometry_type": "gt_alignment",
                    "length_px": float(np.sum(np.hypot(np.diff(et_mid[:, 0]), np.diff(et_mid[:, 1])))),
                    "bbox_area": bbox_area,
                    "os_mode": variant_id,
                    "g11": np.nan,
                    "g22": np.nan,
                    "g33": np.nan,
                }, et_mid, pred_mid)
            continue

        if kind == "combined":
            et_parts = _coerce_seg_list(entry, seg_keys=("midline_segments",), packed_keys=("midline",))
            et_meta = entry.get("midline_segments_meta", []) if isinstance(entry.get("midline_segments_meta", []), list) else []

            for cmp_label, variant_id in method_cmp_specs:
                mv = method_variants.get(variant_id, {}) if isinstance(method_variants.get(variant_id, {}), dict) else {}
                pred_parts = _coerce_seg_list(mv, seg_keys=("midline_segments",), packed_keys=("midline",))
                pred_meta = mv.get("midline_segments_meta", []) if isinstance(mv.get("midline_segments_meta", []), list) else []

                # Prefer segment-wise rows when topology aligns; otherwise fallback to concatenated row.
                aligned = len(et_parts) > 0 and len(et_parts) == len(pred_parts)
                if aligned:
                    for si, (et_mid, pred_mid) in enumerate(zip(et_parts, pred_parts)):
                        et_mid = np.asarray(et_mid, float)
                        pred_mid = np.asarray(pred_mid, float)
                        if et_mid.ndim != 2 or pred_mid.ndim != 2 or len(et_mid) < 2 or len(pred_mid) < 2:
                            continue
                        bm = et_meta[si] if si < len(et_meta) and isinstance(et_meta[si], dict) else {}
                        branch_id = int(bm.get("branch_id", si))
                        _queue_metric_row({
                            "image": base_name,
                            "crack_id": cid,
                            "crack_kind": kind,
                            "comparison_label": cmp_label,
                            "variant_id": variant_id,
                            "segment_index": int(si),
                            "branch_id": int(branch_id),
                            "is_atomic": 0,
                            "is_combined": 1,
                            "is_noncombined_atomic": 0,
                            "geometry_type": "gt_alignment",
                            "length_px": float(np.sum(np.hypot(np.diff(et_mid[:, 0]), np.diff(et_mid[:, 1])))),
                            "bbox_area": bbox_area,
                            "os_mode": variant_id,
                            "g11": np.nan,
                            "g22": np.nan,
                            "g33": np.nan,
                        }, et_mid, pred_mid)
                else:
                    et_all = np.vstack(et_parts) if et_parts else np.empty((0, 2), float)
                    pred_all = np.vstack(pred_parts) if pred_parts else np.empty((0, 2), float)
                    if len(et_all) < 2 or len(pred_all) < 2:
                        continue
                    _queue_metric_row({
                        "image": base_name,
                        "crack_id": cid,
                        "crack_kind": kind,
                        "comparison_label": cmp_label,
                        "variant_id": variant_id,
                        "segment_index": -1,
                        "branch_id": -1,
                        "is_atomic": 0,
                        "is_combined": 1,
                        "is_noncombined_atomic": 0,
                        "geometry_type": "gt_alignment_concat_fallback",
                        "length_px": float(np.sum(np.hypot(np.diff(et_all[:, 0]), np.diff(et_all[:, 1])))),
                        "bbox_area": bbox_area,
                        "os_mode": variant_id,
                        "g11": np.nan,
                        "g22": np.nan,
                        "g33": np.nan,
                    }, et_all, pred_all)

    if metric_jobs:
        t_met0 = time.perf_counter()
        metric_out = []
        for _meta_row, man_xy, pred_xy in metric_jobs:
            metric_out.append(
                _compute_midline_metrics_task((man_xy, pred_xy, 3.0))
            )

        for (meta_row, _man, _pred), mout in zip(metric_jobs, metric_out):
            row = dict(meta_row)
            row["score_mid"] = float(mout.get("score_mid", np.nan))
            mm = mout.get("mm", {})
            if isinstance(mm, dict):
                row.update(mm)
            rows.append(row)

        if isinstance(profile_accumulator, dict):
            profile_accumulator["met"] = float(profile_accumulator.get("met", 0.0) or 0.0) + float(time.perf_counter() - t_met0)
            profile_accumulator["n"] = int(profile_accumulator.get("n", 0) or 0) + int(len(metric_out))

    df_all = pd.DataFrame(rows)
    out_csv_all = os.path.join(analysis_dir, "gt_ablation_midline_metrics.csv")

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

    def _length_weighted_means(sub_df, metric_cols_):
        if sub_df is None or sub_df.empty:
            return {c: np.nan for c in metric_cols_}
        w_all = np.asarray(sub_df["length_px"], float)
        out = {}
        for c in metric_cols_:
            x = np.asarray(sub_df[c], float)
            ok = np.isfinite(x) & np.isfinite(w_all) & (w_all > 0)
            out[c] = float(np.sum(x[ok] * w_all[ok]) / np.sum(w_all[ok])) if np.any(ok) else np.nan
        return out

    if df_all.empty:
        df_all.to_csv(out_csv_all, index=False)
        pd.DataFrame([]).to_csv(os.path.join(analysis_dir, "gt_ablation_midline_weighted_summary.csv"), index=False)
        print(f"[GT_SUP] wrote ablation metrics -> {out_csv_all}")
        return df_all

    df_all.to_csv(out_csv_all, index=False)
    print(f"[GT_SUP] wrote ablation metrics -> {out_csv_all}")

    summary_rows = []
    for (cmp_label, variant_id), dcmp in df_all.groupby(["comparison_label", "variant_id"], dropna=False):
        df_atomic = dcmp[dcmp["is_atomic"] == 1].copy()
        df_combo_plus_noncombo = dcmp[(dcmp["is_combined"] == 1) | (dcmp["is_noncombined_atomic"] == 1)].copy()

        w_atomic = _length_weighted_means(df_atomic, metric_cols)
        w_combo = _length_weighted_means(df_combo_plus_noncombo, metric_cols)

        summary_rows.append({
            "image": base_name,
            "comparison_label": str(cmp_label),
            "variant_id": str(variant_id),
            "group": "atomic",
            "count": int(len(df_atomic)),
            "total_length_px": float(df_atomic["length_px"].sum()) if not df_atomic.empty else 0.0,
            **{f"lwmean_{c}": float(w_atomic.get(c, np.nan)) for c in metric_cols},
        })
        summary_rows.append({
            "image": base_name,
            "comparison_label": str(cmp_label),
            "variant_id": str(variant_id),
            "group": "combined_plus_noncombined_atomic",
            "count": int(len(df_combo_plus_noncombo)),
            "total_length_px": float(df_combo_plus_noncombo["length_px"].sum()) if not df_combo_plus_noncombo.empty else 0.0,
            **{f"lwmean_{c}": float(w_combo.get(c, np.nan)) for c in metric_cols},
        })


    summary_csv = os.path.join(analysis_dir, "gt_ablation_midline_weighted_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"[GT_SUP] wrote ablation weighted summary -> {summary_csv}")

    # Fast path: one grouped figure across all methods instead of per-method plotting.
    plot_rs3_midline_diagnostics(
        df_all=df_all,
        out_dir=os.path.join(analysis_dir, "diagnostics", "all_methods"),
        selected_family=None,
        title_suffix="GT Ablation Audit (All Methods)",
        compare_mode="grouped",
        group_cols=("variant_id",),
        include_diagnostic_metrics=False,
    )
    if not batch_plots_only:
        plot_rs3_midline_decomposition(
            df_all=df_all,
            out_path=os.path.join(analysis_dir, "diagnostics", "all_methods", "midline_all_metrics_decomposition.jpg"),
            group_col="variant_id",
            title="Midline Metrics - Full Decomposition (All Methods)",
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
    ann_path: str | None = None,
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
    t0_total = time.perf_counter()
    FAST_EXPORT_MODE = True  # Keep thesis-critical outputs; skip heavy debug visuals.
    BATCH_PLOTS_ONLY = not bool(DEBUG_TARGET and DEBUG_TARGET != "cid1")  # True in batch, False when debugging single image
    t_plot = 0.0
    t_preview = 0.0
    t_json = 0.0
    t_metrics_export = 0.0
    t_overview = 0.0
    accP = {"mid": 0.0, "rs": 0.0, "met": 0.0, "n": 0}

    # ================================
    # HARD DEBUG MODE (TEMP)
    # ================================
    DEBUG_GT_BRANCH_ONLY = False

    # Re-enable full centering pipeline for holistic evaluation.
    enable_auto_centering = True
    auto_centering_debug = bool(DEBUG_TARGET and DEBUG_TARGET != "cid1")
    # Always keep summary centering outputs (atomic_all/combined overlays) in batch.
    auto_centering_outputs = True

    if bool(HARD_ISOLATION_DISABLE_CENTERING):
        enable_auto_centering = False
        auto_centering_debug = False
        auto_centering_outputs = False

    DEBUG_MODE = False
    DEBUG_CROP_AUDIT = bool(DEBUG_MODE and DEBUG_LEVEL >= 3)
    DEBUG_SKIP_COMBINED = bool(DEBUG_MODE and DEBUG_LEVEL >= 3)

    sup_root = os.path.join(save_root, "supervision", base_name)
    if os.path.isdir(sup_root):
        shutil.rmtree(sup_root)
    os.makedirs(sup_root, exist_ok=True)

    # -------------------------------------------------
    # Bbox repair (runs after rmtree so plots survive)
    # -------------------------------------------------
    if ann_path and os.path.isfile(str(ann_path)):
        try:
            from pathlib import Path
            from tools.repair_atomic_bboxes import repair_bboxes as _repair_atomic_bboxes

            _repair_plot_dir = os.path.join(sup_root, "analysis")
            os.makedirs(_repair_plot_dir, exist_ok=True)
            before_png = os.path.join(_repair_plot_dir, "bbox_repair_before.png")
            after_png = os.path.join(_repair_plot_dir, "bbox_repair_after.png")

            _image_path = None
            repaired_count, repaired_data = _repair_atomic_bboxes(
                json_path=Path(str(ann_path)),
                contain_thresh=0.8,
                weak_thresh=0.3,
                image_path=_image_path,
                before_png=Path(before_png),
                after_png=Path(after_png),
                return_data=True,
            )
            print(f"[GT_SUP] bbox repair: changed={int(repaired_count)}")
            if isinstance(repaired_data, dict):
                _repaired_atomics = (
                    repaired_data.get("annotations", {}) or {}
                ).get("atomic_cracks", {}) or {}
                for _cid, _cr in _repaired_atomics.items():
                    if str(_cid) in atomic and isinstance(_cr, dict):
                        _bb = _cr.get("mask_bbox")
                        if _bb is not None:
                            atomic[str(_cid)]["mask_bbox"] = _bb
        except Exception as _e_repair:
            print(f"[GT_SUP] bbox repair skipped: {_e_repair}")

    #mask_root = os.path.join(sup_root, "masks")
    atomic_crop_root = os.path.join(sup_root, "atomic_crops")
    combined_crop_root = os.path.join(sup_root, "combined_crops")
    auto_center_root = os.path.join(sup_root, "auto_center_debug")
    #os.makedirs(mask_root, exist_ok=True)
    os.makedirs(atomic_crop_root, exist_ok=True)
    os.makedirs(combined_crop_root, exist_ok=True)
    if enable_auto_centering:
        os.makedirs(auto_center_root, exist_ok=True)

    gt_bin = (gt_mask > 0).astype(np.uint8)
    num_cc, cc_labels = cv2.connectedComponents(gt_bin, 8)
    _dlog(2, f"[GT_SUP] GT connected components: {num_cc-1}")

    try:
        raw_dbg_png = os.path.join(sup_root, "raw_json_atomics_debug.jpg")
        _debug_plot_all_atomics_for_export(
            image_rgb=original_image,
            atomic_map=(atomic or {}),
            out_png=raw_dbg_png,
            title=f"RAW JSON DEBUG - {base_name}",
        )
    except Exception as _e:
        _dlog(3, f"[GT_SUP DEBUG] failed raw atomic truth debug plot: {_e}")

    combined_groups = combined_groups or {}
    if DEBUG_SKIP_COMBINED:
        _dlog(2, "[GT_SUP] debug mode: skipping combined processing")
        combined_groups = {}
    _dlog(1, f"[GT_SUP] {base_name} | atomics={len(atomic or {})} | combined={len(combined_groups or {})}")
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
        "multi_cue_align_s": 0.0,
        "depth_recess_s": 0.0,
        "multi_cue_costmap_s": 0.0,
        "multi_cue_dijkstra_s": 0.0,
        "multi_cue_postprocess_s": 0.0,
        "normals_centered_s": 0.0,
        "normals_depth_s": 0.0,
    }

    def _run_timed_plot(fn, *args, **kwargs):
        nonlocal t_plot
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        t_plot += float(time.perf_counter() - t0)
        return out

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
            + float(dep.get("multi_cue_align_s", 0.0) or 0.0)
            + float(dep.get("recess_s", 0.0) or 0.0)
            + float(dep.get("costmap_s", 0.0) or 0.0)
            + float(dep.get("dijkstra_s", 0.0) or 0.0)
            + float(dep.get("postprocess_s", 0.0) or 0.0)
            + float(nrm.get("centered_s", 0.0) or 0.0)
            + float(nrm.get("multi_cue_s", 0.0) or 0.0)
        )

    def _accumulate_timing_blob(timing_blob):
        if not isinstance(timing_blob, dict):
            return
        methods_blob = timing_blob.get("methods", {}) if isinstance(timing_blob.get("methods", {}), dict) else {}
        if methods_blob:
            m_dt = methods_blob.get("dt", {}) if isinstance(methods_blob.get("dt", {}), dict) else {}
            m_depth = methods_blob.get("dt_ridge_color_depth", {}) if isinstance(methods_blob.get("dt_ridge_color_depth", {}), dict) else {}
            if not m_depth:
                m_depth = methods_blob.get("dt_ridge_valley_depth", {}) if isinstance(methods_blob.get("dt_ridge_valley_depth", {}), dict) else {}
            if not m_depth:
                m_depth = methods_blob.get("dt_depth", {}) if isinstance(methods_blob.get("dt_depth", {}), dict) else {}
            timing_totals["dt_compute_s"] += float(m_dt.get("dt_compute_s", 0.0) or 0.0)
            timing_totals["centered_snap_s"] += float(m_dt.get("dijkstra_s", 0.0) or 0.0)
            timing_totals["multi_cue_align_s"] += float(m_depth.get("multi_cue_align_s", 0.0) or 0.0)
            timing_totals["depth_recess_s"] += float(m_depth.get("depth_recess_s", 0.0) or 0.0)
            timing_totals["multi_cue_costmap_s"] += float(m_depth.get("costmap_s", 0.0) or 0.0)
            timing_totals["multi_cue_dijkstra_s"] += float(m_depth.get("dijkstra_s", 0.0) or 0.0)
            timing_totals["multi_cue_postprocess_s"] += float(m_depth.get("postprocess_s", 0.0) or 0.0)
            timing_totals["normals_centered_s"] += float(m_dt.get("normals_s", 0.0) or 0.0)
            timing_totals["normals_depth_s"] += float(m_depth.get("normals_s", 0.0) or 0.0)
            return
        dt = timing_blob.get("dt", {}) if isinstance(timing_blob.get("dt", {}), dict) else {}
        ctr = timing_blob.get("centered", {}) if isinstance(timing_blob.get("centered", {}), dict) else {}
        dep = timing_blob.get("depth", {}) if isinstance(timing_blob.get("depth", {}), dict) else {}
        nrm = timing_blob.get("normals", {}) if isinstance(timing_blob.get("normals", {}), dict) else {}
        timing_totals["dt_compute_s"] += float(dt.get("compute_s", 0.0) or 0.0)
        timing_totals["centered_snap_s"] += float(ctr.get("snap_s", 0.0) or 0.0)
        timing_totals["multi_cue_align_s"] += float(dep.get("multi_cue_align_s", 0.0) or 0.0)
        timing_totals["depth_recess_s"] += float(dep.get("recess_s", 0.0) or 0.0)
        timing_totals["multi_cue_costmap_s"] += float(dep.get("costmap_s", 0.0) or 0.0)
        timing_totals["multi_cue_dijkstra_s"] += float(dep.get("dijkstra_s", 0.0) or 0.0)
        timing_totals["multi_cue_postprocess_s"] += float(dep.get("postprocess_s", 0.0) or 0.0)
        timing_totals["normals_centered_s"] += float(nrm.get("centered_s", 0.0) or 0.0)
        timing_totals["normals_depth_s"] += float(nrm.get("multi_cue_s", 0.0) or 0.0)

    method_style = {
        "dt": {"slug": "dt", "label": "DT", "compare_label": "DT Midline", "color": "cyan"},
        "dt_depth": {"slug": "dt_depth", "label": "DT + Depth", "compare_label": "DT + Depth Midline", "color": "magenta"},
        "dt_ridge_valley": {"slug": "dt_ridge_valley", "label": "DT + Ridge/Valley", "compare_label": "DT + Ridge/Valley Midline", "color": "deepskyblue"},
        "dt_ridge_valley_depth": {"slug": "dt_ridge_valley_depth", "label": "DT + Ridge/Valley + Depth", "compare_label": "DT + Ridge/Valley + Depth Midline", "color": "orange"},
        "dt_ridge_color_depth": {"slug": "dt_ridge_color_depth", "label": "DT + Ridge/Valley + RGB + Depth", "compare_label": "DT + Ridge/Valley + RGB + Depth Midline", "color": "lime"},
    }
    ATOMIC_PER_SEG_DEBUG = False  # temporary: disable per-atomic seg_### debug folders

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

    def _midline_to_segments(mid):
        if mid is None:
            return []
        arr = np.asarray(mid, float)
        if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) >= 2:
            return [arr]
        if isinstance(mid, (list, tuple)):
            segs = []
            for s in mid:
                sa = np.asarray(s, float)
                if sa.ndim == 2 and sa.shape[1] == 2 and len(sa) >= 2:
                    segs.append(sa)
            if segs:
                return segs
        return _split_midline_packed(mid)

    def _timing_by_method(methods):
        out = {}
        for k, v in (methods or {}).items():
            if isinstance(v, dict):
                out[str(k)] = dict(v.get("timing", {}) or {})
        return out

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
        if mk in {"dt_ridge_valley_depth", "dt_ridge_color_depth"} and (not has_rgb or not has_depth):
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

    def _assert_same_hw(tag, a, b):
        if a is None or b is None:
            return
        aa = np.asarray(a)
        bb = np.asarray(b)
        if aa.ndim < 2 or bb.ndim < 2:
            return
        if aa.shape[:2] != bb.shape[:2]:
            raise ValueError(f"[{tag}] shape mismatch: {aa.shape[:2]} vs {bb.shape[:2]}")

    def _viz_territory_from_costmap(costmap, crack_mask_u8, pct=60.0):
        cm = np.asarray(costmap, np.float32)
        m = (np.asarray(crack_mask_u8) > 0)
        if cm.shape[:2] != m.shape[:2]:
            print(f"[TERRITORY VIZ][MISMATCH] costmap={cm.shape} mask={m.shape}")
            return None
        valid = m & np.isfinite(cm) & (cm < np.float32(1e9))
        if not np.any(valid):
            return None
        thr = float(np.percentile(cm[valid], float(pct)))
        out = np.zeros_like(cm, dtype=np.uint8)
        out[valid & (cm <= thr)] = 255
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
        "dt_ridge_color_depth": "DT + ridge/valley + RGB + depth preferred region",
    }

    def _canonicalize_segments_with_meta(
        segs_in,
        meta_in,
        *,
        label,
        save_root=None,
        base_name=None,
    ):
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
                _dlog(2, f"[CANON][GT_SUP] {label} branch={bid} flipped whole branch orientation")
            assert_direction_consistency(segs_b)

            S_chain_dbg, ok_chain_dbg, reason_chain_dbg = stitch_branch_segments(segs_b)
            if ok_chain_dbg:
                print(f"[DEBUG] stitch OK branch={bid}")
            else:
                print(f"[DEBUG] stitch FAIL branch={bid}: {reason_chain_dbg}")

            # DEBUG: visualize canonicalized per-branch segments and naive concatenation.
            if DEBUG_GT_BRANCH_ONLY and save_root is not None and base_name is not None:
                import matplotlib.pyplot as plt
                debug_dir = os.path.join(
                    save_root,
                    "supervision",
                    str(base_name),
                    "analysis",
                    "canonicalized_segments",
                )
                fig = plt.figure(figsize=(6, 6))
                for k, S in enumerate(segs_b):
                    S = np.asarray(S)
                    if S.ndim != 2 or S.shape[1] != 2 or len(S) < 2:
                        continue
                    plt.plot(S[:, 0], S[:, 1], linewidth=2, label=f"seg{k}")
                    plt.scatter(S[0, 0], S[0, 1], s=30, marker="o")
                    plt.scatter(S[-1, 0], S[-1, 1], s=30, marker="x")
                plt.title(f"{label} branch {bid} canonicalized")
                plt.legend()
                plt.gca().invert_yaxis()
                plt.axis("equal")
                out_png = os.path.join(debug_dir, f"{base_name}_branch{int(bid)}_canonicalized.jpg")
                _save_debug_plot(fig, out_png)
                print(f"[DEBUG] saved canonicalized -> {out_png}")

                # DEBUG: stitched branch visualization (matches intended pipeline geometry).
                S_chain, ok_chain, reason = stitch_branch_segments(
                    segs_b,
                    max_jump=10.0,
                    allow_teleport=False,
                )
                fig2 = plt.figure(figsize=(6, 6))
                if ok_chain and S_chain is not None:
                    plt.plot(S_chain[:, 0], S_chain[:, 1], "g-", linewidth=2)
                    title = f"{label} branch {bid} STITCHED (OK)"
                else:
                    for S in segs_b:
                        S = np.asarray(S)
                        if S.ndim != 2 or S.shape[1] != 2 or len(S) < 2:
                            continue
                        plt.plot(S[:, 0], S[:, 1], linewidth=2)
                    title = f"{label} branch {bid} STITCH FAIL: {reason}"
                plt.title(title)
                plt.gca().invert_yaxis()
                plt.axis("equal")
                out_png2 = os.path.join(debug_dir, f"{base_name}_branch{int(bid)}_stitched.jpg")
                _save_debug_plot(fig2, out_png2)
                print(f"[DEBUG] saved stitched plot -> {out_png2}")

            for j, (S, m) in enumerate(zip(segs_b, meta_b)):
                m2 = dict(m if isinstance(m, dict) else {})
                m2["branch_id"] = int(bid)
                m2["seg_idx"] = int(j)
                out_segs.append(np.asarray(S, float))
                out_meta.append(m2)

        # === FORCE STITCHED OUTPUT PER BRANCH ===
        from collections import defaultdict

        branch_groups = defaultdict(list)
        branch_meta = defaultdict(list)
        for S, m in zip(out_segs, out_meta):
            bid = int((m or {}).get("branch_id", 0))
            arr = np.asarray(S, float)
            if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) >= 2:
                branch_groups[bid].append(arr)
                branch_meta[bid].append(dict(m or {}))

        stitched_segs = []
        stitched_meta = []
        for bid in sorted(branch_groups.keys()):
            segs_b = branch_groups[bid]
            meta_b = branch_meta.get(bid, [])
            S_chain, ok_chain, reason_chain = stitch_branch_segments(
                segs_b,
                max_jump=10.0,
                allow_teleport=False,
            )
            if not ok_chain or S_chain is None or len(np.asarray(S_chain, float)) < 2:
                print(
                    f"[WARN] stitch failed for branch {int(bid)}: {reason_chain} "
                    f"- emitting {len(segs_b)} segments individually"
                )
                for si, S_seg in enumerate(segs_b):
                    S_arr = np.asarray(S_seg, float)
                    if S_arr.ndim != 2 or S_arr.shape[1] != 2 or len(S_arr) < 2:
                        continue
                    m_out = dict(meta_b[si] or {}) if si < len(meta_b) else {"branch_id": int(bid)}
                    m_out["branch_id"] = int((m_out or {}).get("branch_id", int(bid)))
                    m_out["stitched"] = False
                    m_out["stitch_reason"] = str(reason_chain)
                    stitched_segs.append(S_arr)
                    stitched_meta.append(m_out)
                continue

            # Preserve atomic membership provenance on stitched GT branches.
            atomic_ids = []
            seen_atomic = set()
            for mm in meta_b:
                mm = mm if isinstance(mm, dict) else {}
                # Prefer explicit atomic_ids list when present.
                aid_list = mm.get("atomic_ids")
                if isinstance(aid_list, (list, tuple)):
                    for a in aid_list:
                        if a is None:
                            continue
                        a_s = str(a)
                        if a_s not in seen_atomic:
                            atomic_ids.append(a_s)
                            seen_atomic.add(a_s)
                aid = mm.get("atomic_id")
                if aid is not None:
                    a_s = str(aid)
                    if a_s not in seen_atomic:
                        atomic_ids.append(a_s)
                        seen_atomic.add(a_s)

            stitched_segs.append(np.asarray(S_chain, float))
            meta_rec = {
                "branch_id": int(bid),
                "seg_idx": 0,
                "stitched": True,
            }
            if atomic_ids:
                meta_rec["atomic_ids"] = list(atomic_ids)
                meta_rec["atomic_id"] = str(atomic_ids[0])
            stitched_meta.append(meta_rec)

        print(f"[DEBUG] returning {len(stitched_segs)} stitched branches")
        return stitched_segs, stitched_meta

    # =====================================================
    # 1) ATOMIC BEFORE MERGE  (USE USER mask_bbox ONLY)
    # =====================================================
    atomic_jobs = []
    if DEBUG_CROP_AUDIT:
        print("\n[JOB_BUILD_DEBUG] atomic_cracks BEFORE job creation")
        for _cid, _cr in (atomic or {}).items():
            _bbox = (_cr or {}).get("mask_bbox", None) if isinstance(_cr, dict) else None
            print(
                f"[JOB_BUILD_DEBUG] scid={str(_cid)} bbox={_bbox} "
                f"bbox_id={id(_bbox)} dict_id={id(_cr)}"
            )
    for order_i, (cid, cr) in enumerate((atomic or {}).items()):
        gt_sup_diag["atomic_total"] += 1
        scid = str(cid)

        mid_xy = np.asarray(cr.get("midline", []), float)
        if mid_xy.ndim != 2 or len(mid_xy) < 2:
            gt_sup_diag["atomic_skip_bad_midline"] += 1
            continue
        mid_xy, _n, _w, _e1, _e2, cinfo = canonicalize_segment_direction(mid_xy)
        if cinfo.get("flipped", False):
            _dlog(2, f"[CANON][GT_SUP] atomic {scid} midline flipped to canonical direction")

        bb = cr.get("mask_bbox", None)
        if bb is not None:
            bb = list(map(int, bb))
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
        crack_mask_clipped = (_crop_local(crack_mask, [int(x), int(y), int(w), int(h)]) > 0).astype(np.uint8)

        atomic_jobs.append({
            "order": int(order_i),
            "scid": scid,
            "mid_xy": np.asarray(mid_xy, float),
            "bbox": [int(x), int(y), int(w), int(h)],
            "mask_bbox": [int(x), int(y), int(w), int(h)],
            "cr_dict_id": int(id(cr)),
            "crack_mask": crack_mask,
            "crack_mask_clipped": crack_mask_clipped,
        })
        if DEBUG_CROP_AUDIT:
            print(
                f"[JOB_BUILD] atomic {scid} bbox FROM JSON = {[int(x), int(y), int(w), int(h)]} "
                f"bbox_id={id(bb)} dict_id={id(cr)}"
            )
            print(f"[JOB_BUILD_DEBUG] ADD JOB scid={scid} bbox={[int(x), int(y), int(w), int(h)]}")

    def _atomic_job_worker(job):
        scid = str(job["scid"])
        mid_xy = np.asarray(job["mid_xy"], float)
        orig_bbox = tuple(job.get("mask_bbox", job.get("bbox", [0, 0, 0, 0])))
        bb = list(job.get("mask_bbox", job.get("bbox", [0, 0, 0, 0])))
        x, y, w, h = map(int, bb)
        if DEBUG_CROP_AUDIT:
            print(f"\n[WORKER_DEBUG] scid={scid}")
            print(f"[WORKER_DEBUG] job_bbox={(x, y, w, h)} job_bbox_id={id(job.get('mask_bbox', None))}")
            print(f"[WORKER_DEBUG] job_cr_dict_id={job.get('cr_dict_id')}")
            print(f"[JOB_USE] atomic {scid} bbox USED = {(x, y, w, h)}")
        if w <= 0 or h <= 0 or w > 2000 or h > 2000:
            print(f"[FATAL] atomic {scid} invalid bbox: {(x, y, w, h)}")
        crack_mask = np.asarray(job["crack_mask"], np.uint8)
        crack_mask_clipped = np.asarray(job["crack_mask_clipped"], np.uint8)
        bbox_xywh = (x, y, w, h)
        if DEBUG_CROP_AUDIT:
            print(f"[CROP_DEBUG] scid={scid} USING bbox={bbox_xywh}")
            print(f"[CROP_DEBUG] image_shape={tuple(np.asarray(original_image).shape)}")
        crack_mask_local = _crop_local(crack_mask, bbox_xywh)
        if crack_mask_local is None or np.asarray(crack_mask_local).size == 0:
            crack_mask_local = np.zeros((max(1, h), max(1, w)), np.uint8)
        crack_mask_local = (np.asarray(crack_mask_local) > 0).astype(np.uint8)
        if DEBUG_CROP_AUDIT:
            print(f"\n[OWNERSHIP] atomic {scid}")
            print(f"[OWNERSHIP] bbox_xywh={bbox_xywh}")
            mid = np.asarray(mid_xy, float)
            x0, y0, bw, bh = bbox_xywh
            x1, y1 = x0 + bw, y0 + bh
            # Frame-aware ownership diagnostics:
            # evaluate both global-frame and local-frame containment, then pick best.
            inside_global = (
                (mid[:, 0] >= float(x0)) & (mid[:, 0] <= float(x1)) &
                (mid[:, 1] >= float(y0)) & (mid[:, 1] <= float(y1))
            )
            inside_local = (
                (mid[:, 0] >= 0.0) & (mid[:, 0] <= float(bw)) &
                (mid[:, 1] >= 0.0) & (mid[:, 1] <= float(bh))
            )
            tot_n = int(len(mid))
            frac_g = float(np.mean(inside_global)) if tot_n > 0 else 0.0
            frac_l = float(np.mean(inside_local)) if tot_n > 0 else 0.0
            use_local = bool(frac_l > frac_g)
            frame_used = "local" if use_local else "global"
            inside_bbox = inside_local if use_local else inside_global
            in_n = int(np.sum(inside_bbox))
            in_frac = float(np.mean(inside_bbox)) if tot_n > 0 else 0.0
            print(f"[OWNERSHIP] midline_inside_bbox_global = {frac_g:.3f} ({int(np.sum(inside_global))}/{tot_n})")
            print(f"[OWNERSHIP] midline_inside_bbox_local  = {frac_l:.3f} ({int(np.sum(inside_local))}/{tot_n})")
            print(f"[OWNERSHIP] midline_inside_bbox({frame_used}) = {in_frac:.3f} ({in_n}/{tot_n})")

            Hcc, Wcc = cc_labels.shape[:2]
            bx0 = max(0, int(x0))
            by0 = max(0, int(y0))
            bx1 = min(Wcc, int(x0 + bw))
            by1 = min(Hcc, int(y0 + bh))
            if bx1 > bx0 and by1 > by0:
                bbox_labels = cc_labels[by0:by1, bx0:bx1]
                unique_labels = np.unique(bbox_labels)
                print(f"[OWNERSHIP] labels_in_bbox = {unique_labels.tolist()}")
            else:
                print("[OWNERSHIP] labels_in_bbox = [] (bbox out of bounds)")

            if use_local:
                mid_for_cc = np.asarray(mid, float).copy()
                mid_for_cc[:, 0] += float(x0)
                mid_for_cc[:, 1] += float(y0)
            else:
                mid_for_cc = np.asarray(mid, float)
            xs = np.clip(np.round(mid_for_cc[:, 0]).astype(int), 0, Wcc - 1)
            ys = np.clip(np.round(mid_for_cc[:, 1]).astype(int), 0, Hcc - 1)
            mid_labels = cc_labels[ys, xs]
            unique_mid_labels = np.unique(mid_labels)
            print(f"[OWNERSHIP] labels_on_midline = {unique_mid_labels.tolist()}")

            pos_labels = unique_mid_labels[unique_mid_labels > 0]
            lbl_mid = int(pos_labels[0]) if len(pos_labels) else 0
            if lbl_mid > 0:
                cc_mask = (cc_labels == lbl_mid)
                bbox_mask = np.zeros_like(cc_mask, dtype=np.uint8)
                if bx1 > bx0 and by1 > by0:
                    bbox_mask[by0:by1, bx0:bx1] = 1
                intersection = int(np.sum(cc_mask & (bbox_mask > 0)))
                cc_area = int(np.sum(cc_mask))
                frac = float(intersection / (cc_area + 1e-6))
                print(f"[OWNERSHIP] bbox n CC = {intersection}, CC area = {cc_area}, frac = {frac:.4f}")
            else:
                print("[OWNERSHIP] bbox n CC = 0, CC area = 0, frac = 0.0000 (no positive midline label)")
        t_atomic_compute0 = time.perf_counter()
        atomic_center_sec = 0.0

        t_manual_normals0 = time.perf_counter()
        manual_normals_diag = {}
        mid_xy_local = np.asarray(mid_xy, float).copy()
        mid_xy_local[:, 0] -= float(x)
        mid_xy_local[:, 1] -= float(y)
        t_rs0 = time.perf_counter()
        (e1x, e1y, e2x, e2y, widths), _ = normals_from_mask_for_midline(
            mid_xy_local,
            crack_mask_clipped > 0,
            60,
            diagnostics=manual_normals_diag,
            image_hw=crack_mask_clipped.shape[:2],
            endpoint_mode="atomic",
        )
        # Shift results back to global coordinates.
        e1x = np.asarray(e1x, float) + float(x)
        e1y = np.asarray(e1y, float) + float(y)
        e2x = np.asarray(e2x, float) + float(x)
        e2y = np.asarray(e2y, float) + float(y)
        accP["rs"] += float(time.perf_counter() - t_rs0)
        manual_normals_s = float(time.perf_counter() - t_manual_normals0)
        diag_brief = _normals_diag_summary(manual_normals_diag)
        _dlog(3, (
            f"[GT_SUP NORMDBG] atomic {scid} manual total={diag_brief['total']} "
            f"valid={diag_brief['valid']} invalid={diag_brief['invalid']} "
            f"invalid_frac={diag_brief['invalid_frac']:.4f} top_reasons={diag_brief['top_reasons']}"
        ))

        atomic_entry = {
            "id": scid,
            "kind": "atomic",
            "members": [],
            "mask_bbox": [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])],
            "mask_crop": np.asarray(crack_mask_local, np.uint8).tolist(),
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
        atomic_entry["method_results"] = {
            str(mk): {
                "midline": None,
                "normals": None,
                "normals_diag": {},
                "debug": {},
                "timing": {},
                "meta": {"status": "missing"},
            }
            for mk in METHOD_SPECS.keys()
        }
        atomic_entry["method_variants"] = dict(atomic_entry["method_results"])
        atomic_compute_sec = float(time.perf_counter() - t_atomic_compute0)

        if enable_auto_centering:
            _dbg_coord(
                tag=f"atomic_{scid}",
                mid_xy=mid_xy,
                mask_u8=crack_mask_clipped,
                bbox_xywh=(x, y, w, h),
            )
            S = np.asarray(mid_xy, float)
            bx, by = int(x), int(y)
            S_local = np.asarray(S, float).copy()
            if S_local.ndim == 2 and S_local.shape[1] == 2 and len(S_local) >= 1:
                S_local[:, 0] -= float(bx)
                S_local[:, 1] -= float(by)
                print(
                    f"[ATOMIC_OFFSET_DBG] cid={scid} bx={bx} by={by} "
                    f"S_before=[{S[0].tolist()}] S_local=[{S_local[0].tolist()}]",
                    flush=True,
                )
            terr = build_territory_mask_from_polyline(
                mid_xy=S_local,
                crack_mask_u8=crack_mask_clipped,
                window_half_size=int(auto_centering_window_half_size),
                dt_domain_u8=None,
            )
            local_depth_crop = _lookup_local_depth_for_atomic(scid)
            t_mid0 = time.perf_counter()
            method_res = compute_midline_method_variants_and_normals(
                mid_xy=S_local,
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
            accP["mid"] += float(time.perf_counter() - t_mid0)
            methods = method_res.get("methods", {}) if isinstance(method_res.get("methods", {}), dict) else {}
            m1 = methods.get("dt", {}) if isinstance(methods.get("dt", {}), dict) else {}
            m3 = methods.get("dt_depth", {}) if isinstance(methods.get("dt_depth", {}), dict) else {}
            m4 = methods.get("dt_ridge_valley_depth", {}) if isinstance(methods.get("dt_ridge_valley_depth", {}), dict) else {}
            m5 = methods.get("dt_ridge_color_depth", {}) if isinstance(methods.get("dt_ridge_color_depth", {}), dict) else {}
            fused_pick = m5 if m5.get("midline", None) is not None else (m4 if m4.get("midline", None) is not None else m3)
            timing_by_method = _timing_by_method(methods)
            atomic_center_sec += _sum_centering_seconds({"methods": {k: (v.get("timing", {}) if isinstance(v, dict) else {}) for k, v in methods.items()}})

            atomic_entry["method_variants"] = {
                k: _variant_to_json(v)
                for k, v in methods.items()
            }
            atomic_entry["method_results"] = {
                k: _variant_to_json(v)
                for k, v in methods.items()
            }
            atomic_entry["method_cost_debug"] = {}
            for mk, mv in methods.items():
                if not isinstance(mv, dict):
                    continue
                dbg_mv = mv.get("debug", {}) if isinstance(mv.get("debug", {}), dict) else {}
                cmaps_mv = dbg_mv.get("costmaps", {}) if isinstance(dbg_mv.get("costmaps", {}), dict) else {}
                sel_mv = cmaps_mv.get("selected", None)
                if sel_mv is None:
                    sk_mv = str(cmaps_mv.get("selected_key", "dt"))
                    sel_mv = cmaps_mv.get(sk_mv, cmaps_mv.get("dt", None))
                atomic_entry["method_cost_debug"][str(mk)] = {
                    "bbox_xywh": [int(x), int(y), int(w), int(h)],
                    "dt_norm": dbg_mv.get("dt_norm", None),
                    "depth_norm": dbg_mv.get("depth_norm", None),
                    "recess_norm": dbg_mv.get("recess_norm", None),
                    "costmaps": cmaps_mv,
                    "selected": sel_mv,
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

            atomic_entry["fused_midline_local"] = None
            atomic_entry["fused_midline_global"] = None
            if m3.get("midline", None) is not None:
                atomic_entry["fused_midline_local"] = np.asarray(m3.get("midline"), float).tolist()
            if m4.get("midline", None) is not None:
                atomic_entry["fused_midline_global"] = np.asarray(m4.get("midline"), float).tolist()
            if m5.get("midline", None) is not None:
                atomic_entry["fused_midline_color_global"] = np.asarray(m5.get("midline"), float).tolist()

            if fused_pick.get("midline", None) is not None:
                fxy = np.asarray(fused_pick.get("midline"), float)
                fused_normals = fused_pick.get("normals", {}) if isinstance(fused_pick.get("normals", {}), dict) else {}
                fused_normals_diag = fused_pick.get("normals_diag", {}) if isinstance(fused_pick.get("normals_diag", {}), dict) else {}
                atomic_entry["fused_midline"] = fxy.tolist()
                atomic_entry["fused_normals"] = fused_normals
                atomic_entry["fused_normals_diag"] = fused_normals_diag
                atomic_entry["midline_fused"] = fxy.tolist()
                atomic_entry["gt_normals_fused"] = fused_normals
                atomic_entry["gt_widths_fused"] = fused_normals.get("width_px", [])

            atomic_entry["multi_cue_cost_meta"] = {
                "dt_depth": m3.get("meta", {}) if isinstance(m3.get("meta", {}), dict) else {},
                "dt_ridge_valley_depth": m4.get("meta", {}) if isinstance(m4.get("meta", {}), dict) else {},
                "dt_ridge_color_depth": m5.get("meta", {}) if isinstance(m5.get("meta", {}), dict) else {},
            }
            atomic_entry["timing"]["methods"] = timing_by_method

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
            auto_meta["et_vs_dt"] = dict(auto_meta["comparisons"].get("dt", {}))
            auto_meta["available_methods"] = [
                mk for mk in METHOD_SPECS.keys()
                if (methods.get(mk, {}) if isinstance(methods.get(mk, {}), dict) else {}).get("midline", None) is not None
            ]
            atomic_entry["auto_centering_meta"] = auto_meta

            if auto_centering_debug and ATOMIC_PER_SEG_DEBUG:
                atomic_dbg_dir = os.path.join(auto_center_root, "atomic", f"seg_{str(scid).zfill(3)}")
                os.makedirs(atomic_dbg_dir, exist_ok=True)
                manual_invalid = [~np.isfinite(np.asarray(widths, float))]
                for mk in METHODS_RS3:
                    print(f"[RUN] image={base_name} seg={scid} method={mk}")
                    mv = methods.get(mk, {}) if isinstance(methods.get(mk, {}), dict) else {}
                    mxy = mv.get("midline", None)
                    mn = mv.get("normals", {}) if isinstance(mv.get("normals", {}), dict) else {}
                    style = method_style.get(mk, {})
                    mslug = str(style.get("slug", mk))
                    dbg_tag = f"cid{scid}_{mslug}"
                    sel_key, sel_cost, dbg, _costmaps_dbg = _extract_selected_costmap(mv)
                    territory_plot = terr
                    if sel_cost is not None:
                        Hc, Wc = np.asarray(sel_cost).shape[:2]
                        local_dom = None
                        if isinstance(dbg, dict) and dbg.get("domain_u8") is not None:
                            try:
                                local_dom = (np.asarray(dbg.get("domain_u8")) > 0).astype(np.uint8)
                            except Exception:
                                local_dom = None
                        mask_for_viz_local = None
                        if crack_mask_clipped is not None and np.asarray(crack_mask_clipped).shape[:2] == (Hc, Wc):
                            mask_for_viz_local = (np.asarray(crack_mask_clipped) > 0)
                        if crack_mask_local is not None and np.asarray(crack_mask_local).shape[:2] == (Hc, Wc):
                            mask_for_viz_local = (np.asarray(crack_mask_local) > 0)
                        elif local_dom is not None and np.asarray(local_dom).shape[:2] == (Hc, Wc):
                            mask_for_viz_local = (np.asarray(local_dom) > 0)
                        elif terr is not None and np.asarray(terr).shape[:2] == (Hc, Wc):
                            mask_for_viz_local = (np.asarray(terr) > 0)
                        else:
                            print(
                                f"[TERRITORY VIZ][SKIP] shape mismatch: "
                                f"costmap={np.asarray(sel_cost).shape}, "
                                f"crack_mask_local={None if crack_mask_local is None else np.asarray(crack_mask_local).shape}, "
                                f"crack_mask_clipped={None if crack_mask_clipped is None else np.asarray(crack_mask_clipped).shape}, "
                                f"domain_u8={None if local_dom is None else np.asarray(local_dom).shape}, "
                                f"territory={None if terr is None else np.asarray(terr).shape}"
                            )
                        #print("[DBG_SHAPES]", flush=True)
                        #print(f"costmap: {np.asarray(sel_cost).shape}", flush=True)
                        #print(f"mask   : {None if mask_for_viz_local is None else np.asarray(mask_for_viz_local).shape}", flush=True)
                        if mask_for_viz_local is not None:
                            _assert_same_hw("atomic territory viz", sel_cost, mask_for_viz_local)
                            territory_plot = _viz_territory_from_costmap(sel_cost, mask_for_viz_local, pct=60.0)
                    '''_dump_compare_arrays(
                        dbg_tag,
                        sel_cost=sel_cost,
                        crack_mask_local=crack_mask_clipped,
                        territory_local=territory_plot,
                    )'''
                    out_dbg = os.path.join(atomic_dbg_dir, f"{mslug}_overlay.jpg")
                    mid_local = np.asarray(mid_xy, float).copy()
                    mid_local[:, 0] -= float(x)
                    mid_local[:, 1] -= float(y)
                    pred_local = []
                    if mxy is not None:
                        mxy_arr = np.asarray(mxy, float)
                        if mxy_arr.ndim == 2 and mxy_arr.shape[1] == 2 and len(mxy_arr) >= 2:
                            mxy_local = mxy_arr.copy()
                            mxy_local[:, 0] -= float(x)
                            mxy_local[:, 1] -= float(y)
                            pred_local = [mxy_local]
                    if not FAST_EXPORT_MODE:
                        _run_timed_plot(
                            plot_costmap_debug,
                            out_path=out_dbg,
                            crack_mask_u8=crack_mask_local,
                            manual_segs=[mid_local],
                            pred_segs=pred_local,
                            costmap=np.asarray(sel_cost, np.float32) if sel_cost is not None else None,
                            normals=mn if isinstance(mn, dict) else None,
                            title=f"atomic {scid} - {style.get('label', mk)}",
                            pred_color="magenta",
                        )

                    depth_lbl = "global" if mk in ("dt_depth", "dt_ridge_valley_depth", "dt_ridge_color_depth") else None
                    mslug = str(style.get("slug", mk))
                    out_cost = os.path.join(atomic_dbg_dir, f"{mslug}_cost.png")
                    _run_timed_plot(
                        plot_depth_cost_diagnostic,
                        out_path=out_cost,
                        crack_mask_u8=crack_mask_local,
                        dt_norm=np.asarray(dbg.get("dt_norm"), np.float32) if dbg.get("dt_norm") is not None else np.zeros_like(crack_mask_local, np.float32),
                        depth_norm=np.asarray(dbg.get("depth_norm"), np.float32) if dbg.get("depth_norm") is not None else None,
                        recess_norm=dbg.get("recess_norm"),
                        depth_score=dbg.get("score_for_refine"),
                        costmaps=dbg.get("costmaps"),
                        bbox_xywh=None,
                        title=f"atomic {scid}: cost cues",
                        method_label=style.get("label", mk),
                        depth_label=depth_lbl,
                    )

        if tuple(job.get("mask_bbox", job.get("bbox", [0, 0, 0, 0]))) != orig_bbox:
            print(
                f"[FATAL] atomic {scid} bbox mutated! "
                f"{orig_bbox} -> {tuple(job.get('mask_bbox', job.get('bbox', [0, 0, 0, 0])))}"
            )

        return int(job["order"]), atomic_entry, float(atomic_compute_sec), float(atomic_center_sec)

    max_workers = max(1, min(8, os.cpu_count() or 1))
    atomic_results = {}
    if DEBUG_CROP_AUDIT and atomic_jobs:
        seen = {}
        for j in atomic_jobs:
            key = tuple(j.get("mask_bbox", j.get("bbox", [0, 0, 0, 0])))
            seen.setdefault(key, []).append(str(j.get("scid")))
        for key, ids in seen.items():
            if len(ids) > 1:
                print(f"[WARN] multiple atomics share bbox {key}: {ids}")

    if DEBUG_CROP_AUDIT and atomic_jobs:
        print("\n[OWNERSHIP] bbox overlap matrix")
        for i, a in enumerate(atomic_jobs):
            ax, ay, aw, ah = [int(v) for v in a.get("bbox", [0, 0, 0, 0])]
            ax1, ay1 = ax + aw, ay + ah
            for j in range(i + 1, len(atomic_jobs)):
                b = atomic_jobs[j]
                bx, by, bw, bh = [int(v) for v in b.get("bbox", [0, 0, 0, 0])]
                bx1, by1 = bx + bw, by + bh
                ix0 = max(ax, bx)
                iy0 = max(ay, by)
                ix1 = min(ax1, bx1)
                iy1 = min(ay1, by1)
                if ix1 > ix0 and iy1 > iy0:
                    print(
                        f"[OWNERSHIP] atomic {a.get('scid')} overlaps atomic {b.get('scid')} "
                        f"(overlap={ix1 - ix0}x{iy1 - iy0})"
                    )
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

    atomic_debug_done = False
    combined_debug_done = False

    for oi in sorted(atomic_results.keys()):
        atomic_entry = atomic_results[oi]
        final_entries.append(atomic_entry)
        gt_sup_diag["atomic_added"] += 1
        if not atomic_debug_done and False:
            t0 = time.perf_counter()
            _plot_single_debug(
                atomic_entry,
                gt_mask,
                original_image,
                os.path.join(atomic_crop_root, "atomic_ablation_debug.jpg"),
            )
            t_preview += float(time.perf_counter() - t0)
            atomic_debug_done = True

    if auto_centering_outputs and atomic_results:
        atomic_all_dir = os.path.join(auto_center_root, "atomic_all")
        os.makedirs(atomic_all_dir, exist_ok=True)
        target_methods = list(METHODS_RS3)
        all_manual = []
        all_methods = {mk: [] for mk in target_methods}
        all_normals = {mk: [] for mk in target_methods}
        for oi in sorted(atomic_results.keys()):
            ae = atomic_results[oi]
            all_manual.extend(_midline_to_segments(ae.get("midline")))
            mres = ae.get("method_results", {})
            if not isinstance(mres, dict):
                mres = {}
            for mk in target_methods:
                mv = mres.get(mk, {}) if isinstance(mres.get(mk, {}), dict) else {}
                all_methods[mk].extend(_midline_to_segments(mv.get("midline")))
                nrm = mv.get("normals")
                if isinstance(nrm, dict):
                    all_normals[mk].append(nrm)
        for mk in target_methods:
            pred = all_methods.get(mk, [])
            if not pred:
                _dlog(2, f"[SKIP] atomic_all {mk} (no midline)")
                continue
            terr_all = build_territory_mask_for_segments(
                segs=all_manual,
                crack_mask_u8=(np.asarray(gt_mask) > 0).astype(np.uint8),
                window_half_size=int(auto_centering_window_half_size),
            )
            style = method_style.get(mk, {})
            out_atomic = os.path.join(atomic_all_dir, f"atomic_all_{str(style.get('slug', mk))}.jpg")
            t0 = time.perf_counter()
            _run_timed_plot(
                plot_atomic_ablation_debug,
                out_path=out_atomic,
                crack_mask_u8=gt_mask,
                all_manual=all_manual,
                all_pred=pred,
                title=f"Atomic (ALL) - {style.get('label', mk)}",
                pred_color="magenta",
                normals=all_normals.get(mk, []),
                original_image=original_image,
                territory_mask=terr_all,
            )
            t_preview += float(time.perf_counter() - t0)

        if auto_centering_debug:
            atomic_cost_global_dir = os.path.join(auto_center_root, "atomic_global_cost")
            os.makedirs(atomic_cost_global_dir, exist_ok=True)

            def _first_non_none(*vals):
                for v in vals:
                    if v is not None:
                        return v
                return None

            for mk in METHODS_RS3:
                style = method_style.get(mk, {})
                mslug = str(style.get("slug", mk))
                use_ridge = mk in {"dt_ridge_valley", "dt_ridge_valley_depth", "dt_ridge_color_depth"}
                use_depth = mk in {"dt_depth", "dt_ridge_valley_depth", "dt_ridge_color_depth"}
                use_rgb = mk == "dt_ridge_color_depth"
                branches_for_global = []
                for oi in sorted(atomic_results.keys()):
                    ae = atomic_results.get(oi, {})
                    dbg_by_method = ae.get("method_cost_debug", {}) if isinstance(ae.get("method_cost_debug", {}), dict) else {}
                    md = dbg_by_method.get(mk, {}) if isinstance(dbg_by_method.get(mk, {}), dict) else {}
                    cmaps = md.get("costmaps", {}) if isinstance(md.get("costmaps", {}), dict) else {}
                    selected = md.get("selected", None)
                    if selected is None:
                        continue
                    a_sel = np.asarray(selected, np.float32)
                    if a_sel.ndim < 2:
                        continue
                    bb = md.get("bbox_xywh", ae.get("mask_bbox", None))
                    if not isinstance(bb, (list, tuple)) or len(bb) != 4:
                        continue
                    x0, y0, bw, bh = [int(v) for v in bb]
                    rgb_src = _first_non_none(cmaps.get("rgb_cue", None), cmaps.get("rgb", None))
                    ridge_src = _first_non_none(cmaps.get("ridge_valley", None), cmaps.get("dt_ridge_valley", None))
                    branches_for_global.append({
                        "bbox": [int(y0), int(y0 + a_sel.shape[0]), int(x0), int(x0 + a_sel.shape[1])],
                        "dt_norm": md.get("dt_norm", None),
                        "rgb": rgb_src if use_rgb else None,
                        "ridge": ridge_src if use_ridge else None,
                        "depth": md.get("depth_norm", None) if use_depth else None,
                        "recess_norm": md.get("recess_norm", None) if use_depth else None,
                        "final_cost": a_sel,
                        "area": float(max(0, bw) * max(0, bh)),
                    })
                if not branches_for_global:
                    continue
                maps_global = build_global_cost_maps(branches_for_global, (int(H), int(W)))
                method_dir = os.path.join(atomic_cost_global_dir, mslug)
                os.makedirs(method_dir, exist_ok=True)
                _run_timed_plot(
                    save_global_cost_panel,
                    maps=maps_global,
                    out_path=os.path.join(method_dir, "atomic_global_cost_panel.jpg"),
                    crack_mask_u8=gt_mask,
                    bbox_xywh=None,
                    title=f"atomic global [{style.get('label', mk)}]",
                )
                # Temporary: disable overlap-count plot emission.

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
        lbls = _cc_labels_for_members(members, atomic, cc_labels)
        if not lbls:
            gt_sup_diag["combined_skip_no_cc_label"] += 1
            continue

        crack_mask = np.zeros_like(gt_bin, dtype=np.uint8)
        for lbl in sorted(lbls):
            if int(lbl) > 0:
                crack_mask |= (cc_labels == int(lbl)).astype(np.uint8)
        crack_mask_clipped = (_crop_local(crack_mask, [ux, uy, uw, uh]) > 0).astype(np.uint8)

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
        dom_meta = dom_meta if isinstance(dom_meta, dict) else {}
        tag_name = str(tag)
        if not BATCH_PLOTS_ONLY:
            branch_audit_png = os.path.join(sup_root, "auto_center_debug", f"branch_audit_{tag_name}.jpg")
            _plot_branch_audit(
                original_image=original_image,
                gt_mask=crack_mask,
                atomic=atomic,
                members=members,
                kept_segs=segs,
                segments_meta=dom_meta.get("segments_meta", []),
                dom_meta=dom_meta,
                out_png=branch_audit_png,
                tag_name=tag_name,
            )
        _branches = dom_meta.get("branches", []) if isinstance(dom_meta.get("branches", []), list) else []
        _order = dom_meta.get("order", []) if isinstance(dom_meta.get("order", []), list) else []
        _segs_in = int(sum(len((b or {}).get("atomic_ids", []) or []) for b in _branches))
        print(
            f"[BRANCH AUDIT] {tag_name} branch_order={_order} "
            f"branches_in={len(_branches)} segs_in={_segs_in} segs_out={len(segs)}"
        )
        for bs in _branches:
            if not isinstance(bs, dict):
                continue
            print(
                f"[BRANCH AUDIT]   branch={int(bs.get('branch_id', -1))} rank={int(bs.get('rank', -1))} "
                f"user_len={float(bs.get('user_len', 0.0)):.1f} kept_len={float(bs.get('kept_len', 0.0)):.1f} "
                f"suppressed={bool(bs.get('suppressed', False))} atomic_ids={list(bs.get('atomic_ids', []) or [])}"
            )

        if not segs:
            gt_sup_diag["combined_skip_no_dominant_segs"] += 1
            continue

        seg_meta = dom_meta.get("segments_meta", [])
        segs_canon, seg_meta_canon = _canonicalize_segments_with_meta(
            segs,
            seg_meta,
            label=f"combined_{ccid}",
            save_root=save_root,
            base_name=base_name,
        )
        # HARD OVERRIDE: only use canonicalized + stitched geometry downstream.
        segs = segs_canon
        seg_meta = seg_meta_canon
        print(f"[PIPELINE] using canonicalized+stitched segments: {len(segs)}")
        if not segs:
            gt_sup_diag["combined_skip_empty_after_canon"] += 1
            continue
        dom_meta["segments_meta"] = seg_meta

        # ================================
        # DEBUG: PLOT GT SEGMENTS PER BRANCH
        # ================================
        if DEBUG_GT_BRANCH_ONLY:
            import matplotlib.pyplot as plt

            branch_gt_dir = os.path.join(sup_root, "analysis", "branch_gt_only")
            os.makedirs(branch_gt_dir, exist_ok=True)

            # Build branch index directly from canonicalized segment meta.
            branch_seg_indices_dbg = {}
            for si, _S in enumerate(segs):
                bi = int(seg_meta[si].get("branch_id", -1)) if si < len(seg_meta) and isinstance(seg_meta[si], dict) else -1
                if bi < 0:
                    continue
                branch_seg_indices_dbg.setdefault(int(bi), []).append(int(si))

            for bi, idxs in branch_seg_indices_dbg.items():
                segs_branch = [np.asarray(segs[i], float) for i in idxs if i < len(segs)]
                pts = [S for S in segs_branch if S.ndim == 2 and S.shape[1] == 2 and len(S) >= 2]
                if not pts:
                    continue
                P = np.vstack(pts)
                bx = int(np.floor(np.min(P[:, 0])) - 5)
                by = int(np.floor(np.min(P[:, 1])) - 5)
                ex = int(np.ceil(np.max(P[:, 0])) + 5)
                ey = int(np.ceil(np.max(P[:, 1])) + 5)
                bx = max(0, min(int(uw) - 1, bx))
                by = max(0, min(int(uh) - 1, by))
                ex = max(0, min(int(uw), ex))
                ey = max(0, min(int(uh), ey))
                bw = int(max(1, ex - bx))
                bh = int(max(1, ey - by))

                gx, gy = int(ux + bx), int(uy + by)
                gex, gey = int(min(W, gx + bw)), int(min(H, gy + bh))
                if gex <= gx or gey <= gy:
                    continue

                crop = np.asarray(original_image)[gy:gey, gx:gex]
                plt.figure(figsize=(6, 6))
                plt.imshow(crop)
                first = True
                for S in segs_branch:
                    S = np.asarray(S, float)
                    if S.ndim != 2 or S.shape[1] != 2 or len(S) < 2:
                        continue
                    S_local = S.copy()
                    S_local[:, 0] -= float(bx)
                    S_local[:, 1] -= float(by)
                    plt.plot(
                        S_local[:, 0],
                        S_local[:, 1],
                        linewidth=2,
                        label="GT segments" if first else None,
                    )
                    first = False
                plt.title(f"{base_name} branch {int(bi)} GT segments")
                if not first:
                    plt.legend()
                plt.axis("off")
                out_png = os.path.join(branch_gt_dir, f"{base_name}_branch{int(bi)}_gt_segments.jpg")
                plt.savefig(out_png, bbox_inches="tight")
                plt.close()

            print("[DEBUG] GT branch plots written ->", branch_gt_dir)
            raise RuntimeError("STOP: GT branch debug complete")

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

            # Use blob bbox when present (territory blobs), fallback to bite bbox for legacy blobs.
            bbox = blob.get("bbox", None)
            if bbox is None:
                bite_meta_local = dom_meta.get("bite", {}) if isinstance(dom_meta.get("bite", {}), dict) else {}
                bbox = bite_meta_local.get("bbox", None)
            if bbox is None or len(bbox) != 4:
                return full

            x0, y0, bw, bh = map(int, bbox)
            # bite bbox is in global coords; shift to crack_mask_clipped local coords
            x0 -= int(ux)
            y0 -= int(uy)
            if x0 >= W or y0 >= H or x0 + bw <= 0 or y0 + bh <= 0:
                return full
            x0 = max(0, x0)
            y0 = max(0, y0)
            shape = blob.get("shape", None)
            b64 = blob.get("packbits_b64", None)
            if shape is None or b64 is None:
                return full

            try:
                import base64
                raw = base64.b64decode(b64.encode("utf-8"))
                bits = np.frombuffer(raw, dtype=np.uint8)

                hh = int(shape[0])
                ww = int(shape[1])
                # packbits pads each row to a multiple of 8 — unpack row-aware to avoid stride scramble
                stride = (ww + 7) // 8
                expected_bytes = hh * stride
                if bits.size < expected_bytes:
                    bits = np.pad(bits, (0, expected_bytes - bits.size), constant_values=0)
                packed_2d = bits[:expected_bytes].reshape((hh, stride))
                crop = np.unpackbits(packed_2d, axis=1)[:, :ww].astype(np.uint8)

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
            decoded = _decode_packed_mask_local(terr_blob, Hm, Wm)
            branch_terr_masks[int(bi)] = decoded
            ys, xs = np.where(decoded > 0)
            if len(xs):
                print(
                    f"[TERR_BBOX_DBG] branch={bi_str} nnz={len(xs)} "
                    f"x=[{int(xs.min())},{int(xs.max())}] y=[{int(ys.min())},{int(ys.max())}] "
                    f"frame=local({int(Hm)}x{int(Wm)})"
                )
            else:
                print(f"[TERR_BBOX_DBG] branch={bi_str} nnz=0")

        # Bite per branch (full-image masks)
        bite_meta = dom_meta.get("bite", {}) if isinstance(dom_meta.get("bite", {}), dict) else {}
        bite_by_branch = bite_meta.get("by_losing_branch", {}) if isinstance(bite_meta.get("by_losing_branch", {}), dict) else {}

        branch_bite_masks = {}
        for bi in branch_order:
            bi_str = str(int(bi))
            bb = bite_by_branch.get(bi_str, {})
            decoded_b = _decode_packed_mask_local(bb, Hm, Wm)
            branch_bite_masks[int(bi)] = decoded_b
            ys_b, xs_b = np.where(decoded_b > 0)
            if len(xs_b):
                print(
                    f"[TERR_BBOX_DBG] bite branch={bi_str} nnz={len(xs_b)} "
                    f"x=[{int(xs_b.min())},{int(xs_b.max())}] y=[{int(ys_b.min())},{int(ys_b.max())}] "
                    f"frame=local({int(Hm)}x{int(Wm)})"
                )
            else:
                print(f"[TERR_BBOX_DBG] bite branch={bi_str} nnz=0")

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
                forbidden_px = int(np.count_nonzero(own_j))
                print(f"[ALLOWED_DBG] branch={int(bi)} excluding sj={int(sj)} own_j_px={forbidden_px}")
                forbidden |= own_j.astype(np.uint8)

            allowed = global_mask & (~forbidden.astype(bool))
            branch_allowed_masks[bi] = allowed.astype(np.uint8)
            allowed_px = int(np.count_nonzero(allowed))
            total_px = int(np.count_nonzero(global_mask))
            print(
                f"[ALLOWED_DBG] branch={int(bi)} forbidden_px={int(np.count_nonzero(forbidden))} "
                f"allowed_px={allowed_px} total_px={total_px}"
            )

        try:
            out_terr_dbg = os.path.join(auto_center_root, f"territory_debug_{tag_name}.jpg")
            _plot_territory_debug(
                original_image=original_image,
                gt_mask_full=crack_mask,
                branch_terr_masks=branch_terr_masks,
                branch_bite_masks=branch_bite_masks,
                branch_allowed_masks=branch_allowed_masks,
                branch_order=branch_order,
                dom_meta=dom_meta,
                ux=int(ux),
                uy=int(uy),
                uw=int(uw),
                uh=int(uh),
                out_png=out_terr_dbg,
            )
        except Exception as _e_terr_dbg:
            _dlog(3, f"[TERRITORY DEBUG] failed call: {_e_terr_dbg}")

        # Temporary sanity print for mask sizes
        try:
            for bi in branch_order:
                bi = int(bi)
                terr_px = int(np.count_nonzero(branch_terr_masks.get(bi, 0)))
                bite_px = int(np.count_nonzero(branch_bite_masks.get(bi, 0)))
                allow_px = int(np.count_nonzero(branch_allowed_masks.get(bi, 0)))
                if _VERBOSE_DEBUG:
                    print(f"[GT_SUP MASKDBG] branch={bi} terr_px={terr_px} bite_px={bite_px} allow_px={allow_px}")
        except Exception as _e:
            _dlog(3, f"[GT_SUP MASKDBG] failed: {_e}")
        
        # -------------------------------------------------
        # DEBUG: dominance bite as-written (RAW, no decode)
        # -------------------------------------------------
        if not FAST_EXPORT_MODE:
            try:
                t_plot0 = time.perf_counter()
                _run_timed_plot(
                    debug_plot_gt_sup_dominance_bite_packed,
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
                _dlog(3, f"[GT_SUP DOMDBG] plot failed: {e}")


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
            _bi_allowed = branch_allowed_masks.get(bi, None)
            mask_use = (
                np.asarray(_bi_allowed, np.uint8)
                if _bi_allowed is not None and np.any(_bi_allowed)
                else (crack_mask_clipped > 0).astype(np.uint8)
            )
            S_arr = np.asarray(S, float)
            S_local = S_arr.copy()
            _DBG_NORMALS = True  # flip to False to silence
            # Combined branch masks are in crack_mask_clipped-local coords (origin ux,uy).
            # Convert segment to local for normal solving, then convert solved edges back.
            if S_local.ndim == 2 and S_local.shape[1] == 2 and len(S_local) >= 1:
                S_local[:, 0] -= float(ux)
                S_local[:, 1] -= float(uy)
            # ADD right before normals_from_mask_for_midline call.
            if _DBG_NORMALS:
                _s_xmin = float(np.nanmin(S_local[:, 0])) if len(S_local) else float("nan")
                _s_xmax = float(np.nanmax(S_local[:, 0])) if len(S_local) else float("nan")
                _s_ymin = float(np.nanmin(S_local[:, 1])) if len(S_local) else float("nan")
                _s_ymax = float(np.nanmax(S_local[:, 1])) if len(S_local) else float("nan")
                _mH, _mW = mask_use.shape[:2]
                _mnnz = int(np.count_nonzero(mask_use))
                print(
                    f"[NORMALS_DBG] si={si} bi={bi} "
                    f"ux={int(ux)} uy={int(uy)} "
                    f"S_global_x=[{_s_xmin + ux:.1f},{_s_xmax + ux:.1f}] "
                    f"S_global_y=[{_s_ymin + uy:.1f},{_s_ymax + uy:.1f}] "
                    f"S_local_x=[{_s_xmin:.1f},{_s_xmax:.1f}] "
                    f"S_local_y=[{_s_ymin:.1f},{_s_ymax:.1f}] "
                    f"mask_shape=({_mH},{_mW}) mask_nnz={_mnnz} "
                    f"mid_in_mask_x_ok={0 <= _s_xmin and _s_xmax < _mW} "
                    f"mid_in_mask_y_ok={0 <= _s_ymin and _s_ymax < _mH}",
                    flush=True,
                )
            # CC-filter: keep only crack components touched by this branch's midline.
            # This preserves full local crack width while removing distant disconnected blobs.
            try:
                num_labels, labels = cv2.connectedComponents(mask_use.astype(np.uint8), 8)
            except Exception:
                num_labels, labels = 0, None
            if int(num_labels) > 2 and labels is not None and S_local.ndim == 2 and len(S_local) >= 1:
                _xi = np.clip(np.round(S_local[:, 0]).astype(int), 0, int(mask_use.shape[1]) - 1)
                _yi = np.clip(np.round(S_local[:, 1]).astype(int), 0, int(mask_use.shape[0]) - 1)
                _touched = {int(labels[y, x]) for x, y in zip(_xi, _yi) if int(labels[y, x]) > 0}
                if _touched:
                    _keep = np.zeros_like(mask_use, dtype=np.uint8)
                    for _lbl in _touched:
                        _keep[labels == int(_lbl)] = 1
                    mask_use = _keep
            t_rs0 = time.perf_counter()
            (e1x, e1y, e2x, e2y, widths), _ = normals_from_mask_for_midline(
                S_local,
                mask_use > 0,
                max_radius=50,
                diagnostics=seg_diag,
                image_hw=mask_use.shape[:2],
                endpoint_mode="combined",
            )
            accP["rs"] += float(time.perf_counter() - t_rs0)
            try:
                e1x = np.asarray(e1x, float) + float(ux)
                e1y = np.asarray(e1y, float) + float(uy)
                e2x = np.asarray(e2x, float) + float(ux)
                e2y = np.asarray(e2y, float) + float(uy)
            except Exception:
                pass
            if bi in (5, 6, 7, 3, 4):  # junction branches
                _e1x_a = np.asarray(e1x, float)
                _e1y_a = np.asarray(e1y, float)
                _fin = np.isfinite(_e1x_a)
                _w_arr = np.asarray(widths, float)
                _w_fin = np.isfinite(_w_arr)
                if np.any(_fin):
                    _mxw = float(np.nanmax(_w_arr[_w_fin])) if np.any(_w_fin) else float("nan")
                    print(
                        f"[NORMALS_JUNCTION] bi={bi} si={si} "
                        f"midline_x=[{float(np.nanmin(S_local[:,0]+ux)):.0f},{float(np.nanmax(S_local[:,0]+ux)):.0f}] "
                        f"midline_y=[{float(np.nanmin(S_local[:,1]+uy)):.0f},{float(np.nanmax(S_local[:,1]+uy)):.0f}] "
                        f"e1x=[{float(np.nanmin(_e1x_a[_fin])):.0f},{float(np.nanmax(_e1x_a[_fin])):.0f}] "
                        f"e1y=[{float(np.nanmin(_e1y_a[_fin])):.0f},{float(np.nanmax(_e1y_a[_fin])):.0f}] "
                        f"max_width={_mxw:.1f}px "
                        f"mask_nnz={int(np.count_nonzero(mask_use))}",
                        flush=True
                    )
            if _DBG_NORMALS:
                _e1x_arr = np.asarray(e1x, float)
                _e1y_arr = np.asarray(e1y, float)
                _fin = np.isfinite(_e1x_arr)
                if np.any(_fin):
                    print(
                        f"[NORMALS_DBG_OUT] si={si} bi={bi} "
                        f"e1x_global=[{float(np.nanmin(_e1x_arr)):.1f},{float(np.nanmax(_e1x_arr)):.1f}] "
                        f"e1y_global=[{float(np.nanmin(_e1y_arr)):.1f},{float(np.nanmax(_e1y_arr)):.1f}] "
                        f"valid_frac={float(np.mean(_fin)):.3f}",
                        flush=True,
                    )
                else:
                    print(f"[NORMALS_DBG_OUT] si={si} bi={bi} ALL NaN", flush=True)
            sdiag_brief = _normals_diag_summary(seg_diag)
            return si, e1x, e1y, e2x, e2y, widths, seg_diag, sdiag_brief

        if segs:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(_combined_seg_worker, si, S) for si, S in enumerate(segs)]
                for fut in as_completed(futs):
                    si, e1x, e1y, e2x, e2y, widths, seg_diag, sdiag_brief = fut.result()
                    _dlog(3, (
                        f"[GT_SUP NORMDBG] combined {ccid} seg={si} manual total={sdiag_brief['total']} "
                        f"valid={sdiag_brief['valid']} invalid={sdiag_brief['invalid']} "
                        f"invalid_frac={sdiag_brief['invalid_frac']:.4f} top_reasons={sdiag_brief['top_reasons']}"
                    ))
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
            # ?? AUTHORITATIVE bbox = UNION OF USER BOXES (xywh)
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
        combined_entry["method_results"] = {
            str(mk): {
                "midline": None,
                "normals": None,
                "normals_diag": {},
                "debug": {},
                "timing": {},
                "meta": {"status": "missing"},
            }
            for mk in METHOD_SPECS.keys()
        }
        combined_entry["method_variants"] = dict(combined_entry["method_results"])
        timing_totals["combined_compute_sec"] += float(
            time.perf_counter() - t_combined_compute0 - combined_plot_excluded_sec
        )

        if bool(HARD_ISOLATION_DISABLE_CENTERING):
            final_entries.append(combined_entry)
            gt_sup_diag["combined_added"] += 1
            if not combined_debug_done and False:
                t0 = time.perf_counter()
                _plot_single_debug(
                    combined_entry,
                    gt_mask,
                    original_image,
                    os.path.join(combined_crop_root, "combined_ablation_debug.jpg"),
                )
                t_preview += float(time.perf_counter() - t0)
                combined_debug_done = True
            continue

        if enable_auto_centering and not DEBUG_GT_BRANCH_ONLY:
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
                    "plot_debug": None,
                    "plot_debug_branches": [],
                }
                for k in METHOD_SPECS.keys()
            }

            combined_timing_blob = {"methods": {}}

            def _acc_nested_timing(dst, src):
                if not (isinstance(dst, dict) and isinstance(src, dict)):
                    return
                dst.setdefault("methods", {})
                for mk, rec in src.items():
                    if not isinstance(rec, dict):
                        continue
                    dst["methods"].setdefault(str(mk), {})
                    for sk, sv in rec.items():
                        try:
                            dst["methods"][str(mk)][sk] = float(dst["methods"][str(mk)].get(sk, 0.0)) + float(sv or 0.0)
                        except Exception:
                            continue

            def _bbox_from_segments(segs_local, pad, Hh, Ww):
                xs = []
                ys = []
                for Ss in segs_local:
                    Ss = np.asarray(Ss, float)
                    if Ss.ndim != 2 or Ss.shape[1] != 2 or len(Ss) < 2:
                        continue
                    xs.append(Ss[:, 0])
                    ys.append(Ss[:, 1])
                if not xs:
                    return None
                x_all = np.concatenate(xs)
                y_all = np.concatenate(ys)
                xmin = int(np.floor(np.min(x_all))) - int(pad)
                xmax = int(np.ceil(np.max(x_all))) + int(pad)
                ymin = int(np.floor(np.min(y_all))) - int(pad)
                ymax = int(np.ceil(np.max(y_all))) + int(pad)
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(Ww, xmax)
                ymax = min(Hh, ymax)
                if xmax <= xmin or ymax <= ymin:
                    return None
                return (xmin, ymin, xmax - xmin, ymax - ymin)

            # Per-branch local context cache (within crack_mask_clipped coords).
            branch_seg_indices = {}
            for si, Sm in enumerate(segs):
                bi = int(seg_meta[si].get("branch_id", -1)) if si < len(seg_meta) else -1
                if bi < 0:
                    continue
                branch_seg_indices.setdefault(bi, []).append(int(si))

            branch_local_cache = {}
            for bi, idxs in branch_seg_indices.items():
                segs_branch = [np.asarray(segs[i], float) for i in idxs]
                if not segs_branch:
                    continue
                # Build branch bbox from territory-mask support (local crack_mask_clipped frame),
                # then convert to global image coordinates.
                terr_full = np.asarray(
                    branch_terr_masks.get(int(bi), np.zeros((Hm, Wm), np.uint8)),
                    np.uint8,
                )
                _ty, _tx = np.where(terr_full > 0)
                bb_local = None
                if _tx.size > 0 and _ty.size > 0:
                    _pad = 20
                    _lx0 = max(0, int(_tx.min()) - _pad)
                    _ly0 = max(0, int(_ty.min()) - _pad)
                    _lx1 = min(int(Wm), int(_tx.max()) + 1 + _pad)
                    _ly1 = min(int(Hm), int(_ty.max()) + 1 + _pad)
                    if _lx1 > _lx0 and _ly1 > _ly0:
                        bb_local = (
                            int(ux + _lx0),
                            int(uy + _ly0),
                            int(_lx1 - _lx0),
                            int(_ly1 - _ly0),
                        )
                if bb_local is None:
                    # Fallback for malformed/missing territory blobs.
                    bb_local = _bbox_from_segments(segs_branch, pad=5, Hh=H, Ww=W)
                if bb_local is None:
                    if _VERBOSE_DEBUG:
                        print(f"[CACHE SKIP] branch={int(bi)} reason=bb_local_none n_segs={len(segs_branch)}")
                    continue
                bx, by, bw, bh = [int(v) for v in bb_local]

                mask_branch_full = branch_allowed_masks.get(
                    int(bi),
                    (crack_mask_clipped > 0).astype(np.uint8),
                )
                # branch_allowed_masks / branch_terr_masks are in crack_mask_clipped space,
                # which starts at (ux, uy) in the global image.  The bbox (bx, by) from
                # _bbox_from_segments is in global coords, so we must subtract the origin
                # before indexing the local mask arrays.
                bx_in_mask = int(bx) - int(ux)
                by_in_mask = int(by) - int(uy)
                mask_local = np.asarray(
                    mask_branch_full[by_in_mask:by_in_mask + bh, bx_in_mask:bx_in_mask + bw],
                    np.uint8,
                )
                if mask_local.size == 0 or not np.any(mask_local):
                    if _VERBOSE_DEBUG:
                        print(
                            f"[CACHE SKIP] branch={int(bi)} reason=empty_mask_local "
                            f"size={mask_local.size} any={bool(np.any(mask_local))}"
                        )
                    continue
                print(
                    f"[MASK_SHAPE_DBG] branch={int(bi)} "
                    f"allowed_mask.shape={np.asarray(mask_local).shape} "
                    f"bbox=(bx={int(bx)},by={int(by)},bw={int(bw)},bh={int(bh)}) "
                    f"expected=({int(bh)},{int(bw)})"
                )
                # Keep only connected components that this branch's midline samples touch.
                # This avoids discarding valid crack width just because territory is narrow.
                try:
                    num_labels, labels = cv2.connectedComponents((mask_local > 0).astype(np.uint8), 8)
                    if int(num_labels) > 2:
                        _mid_mask = np.zeros_like(mask_local, np.uint8)
                        for _si in idxs:
                            _Sm = np.asarray(segs[_si], float)
                            if _Sm.ndim != 2 or _Sm.shape[1] != 2 or len(_Sm) < 1:
                                continue
                            _xi = np.clip(np.round(_Sm[:, 0]).astype(int) - int(bx), 0, int(bw) - 1)
                            _yi = np.clip(np.round(_Sm[:, 1]).astype(int) - int(by), 0, int(bh) - 1)
                            _mid_mask[_yi, _xi] = 1
                        keep = np.zeros_like(mask_local, np.uint8)
                        for lab in range(1, int(num_labels)):
                            comp = (labels == lab)
                            if np.any(comp & (_mid_mask > 0)):
                                keep[comp] = 1
                        if np.any(keep):
                            mask_local = keep
                except Exception:
                    pass
                if auto_centering_outputs and not BATCH_PLOTS_ONLY:
                    _mask_dbg_dir = os.path.join(auto_center_root, "branch_masks")
                    os.makedirs(_mask_dbg_dir, exist_ok=True)
                    _mask_vis = (mask_local * 255).astype(np.uint8)
                    cv2.imwrite(
                        os.path.join(_mask_dbg_dir, f"branch_{int(bi):03d}_mask_local.png"),
                        _mask_vis,
                    )

                if _VERBOSE_DEBUG:
                    print(
                        f"[CACHE BUILD] branch={int(bi)} bbox=({bx},{by},{bw},{bh}) "
                        f"mask_nnz={int(np.count_nonzero(mask_local))} "
                        f"terr_available={branch_terr_masks.get(int(bi)) is not None}"
                    )

                gx0 = int(bx)
                gy0 = int(by)
                gx1 = int(min(W, bx + bw))
                gy1 = int(min(H, by + bh))
                if gx1 <= gx0 or gy1 <= gy0:
                    if _VERBOSE_DEBUG:
                        print(
                            f"[CACHE SKIP] branch={int(bi)} reason=zero_size_global_crop "
                            f"gx=({gx0},{gx1}) gy=({gy0},{gy1})"
                        )
                    continue

                rgb_local = np.asarray(original_image[gy0:gy1, gx0:gx1])
                depth_local = None
                if depth_full is not None:
                    try:
                        depth_arr = np.asarray(depth_full)
                        if depth_arr.ndim >= 2:
                            Hdf, Wdf = depth_arr.shape[:2]
                            sx = float(Wdf) / float(max(1, W))
                            sy = float(Hdf) / float(max(1, H))
                            dx0 = int(np.floor(gx0 * sx))
                            dy0 = int(np.floor(gy0 * sy))
                            dx1 = int(np.ceil(gx1 * sx))
                            dy1 = int(np.ceil(gy1 * sy))
                            dx0 = max(0, min(Wdf, dx0))
                            dy0 = max(0, min(Hdf, dy0))
                            dx1 = max(0, min(Wdf, dx1))
                            dy1 = max(0, min(Hdf, dy1))
                            if dx1 > dx0 and dy1 > dy0:
                                depth_local = np.asarray(depth_arr[dy0:dy1, dx0:dx1])
                    except Exception:
                        depth_local = None

                branch_local_cache[int(bi)] = {
                    "bbox_local": [bx, by, bw, bh],  # global coords
                    "mask_local": mask_local,
                    "rgb_local": rgb_local,
                    "depth_local": depth_local,
                }

            # GT-only branch plots (outside centered/method solver path).
            if bool(HARD_ISOLATION_DISABLE_CENTERING):
                branch_gt_dir = os.path.join(sup_root, "analysis", "branch_gt_only")
                os.makedirs(branch_gt_dir, exist_ok=True)
                for bi, idxs in branch_seg_indices.items():
                    cache = branch_local_cache.get(int(bi), None)
                    if cache is None:
                        continue
                    bbox_local = cache.get("bbox_local", None)
                    if bbox_local is None:
                        continue
                    segs_branch = [np.asarray(segs[i], float) for i in idxs if i < len(segs)]
                    out_png = os.path.join(
                        branch_gt_dir,
                        f"{base_name}_branch{int(bi)}_gt_segments.jpg",
                    )
                    plot_gt_branch_segments_only(
                        original_image=original_image,
                        ux=ux,
                        uy=uy,
                        bbox_local=bbox_local,
                        segs_branch=segs_branch,
                        out_png=out_png,
                        title=f"{base_name} branch {int(bi)} GT segments",
                    )

            combined_endpoint_mode = "combined"
            if combined_endpoint_mode == "combined":
                seg_work_items = []
                branch_specs = []
                for bi, idxs in branch_seg_indices.items():
                    if _branch_kill_dbg(base_name, int(bi)):
                        print(f"[KILL CHECK] entering branch {int(bi)} for image={base_name}", flush=True)
                        raise RuntimeError("STOP HERE - combined branch loop reached")
                    cache = branch_local_cache.get(int(bi), None)
                    if cache is None:
                        continue
                    bx, by, bw, bh = [int(v) for v in cache.get("bbox_local", [0, 0, 0, 0])]
                    mask_use = np.asarray(cache.get("mask_local"), np.uint8)
                    rgb_use = np.asarray(cache.get("rgb_local"))
                    depth_use = cache.get("depth_local", None)
                    segs_branch = [np.asarray(segs[i], float) for i in idxs]
                    if not segs_branch:
                        continue
                    S_branch = np.vstack(segs_branch)
                    if _isolate_gt_dbg(base_name, int(bi)):
                        print("\n" + "=" * 60)
                        print(f"[GT_DEBUG] branch={int(bi)} | num_segments={len(segs_branch)}")
                        for j, Sseg in enumerate(segs_branch):
                            Sseg = np.asarray(Sseg, float)
                            if Sseg.ndim == 2 and Sseg.shape[1] == 2 and len(Sseg) >= 2:
                                print(f"[SEG {int(j)}] len={len(Sseg)}")
                                print(f"   start={Sseg[0].tolist()} end={Sseg[-1].tolist()}")
                        d = np.linalg.norm(np.diff(S_branch, axis=0), axis=1) if len(S_branch) >= 2 else np.asarray([], float)
                        jumps = np.where(d > 20.0)[0] if d.size else np.asarray([], int)
                        print(f"[STACKED] len={len(S_branch)}")
                        print(f"[JUMPS] count={int(len(jumps))} idx={jumps.tolist()}")
                        if d.size:
                            print(f"[MAX STEP] {float(np.max(d)):.2f}")
                        iso_dir = os.path.join(sup_root, "analysis", "gt_branch_isolation")
                        raw_png, stacked_png = _plot_gt_branch_isolation(
                            segs_branch=segs_branch,
                            S_branch=S_branch,
                            out_dir=iso_dir,
                            base_name=base_name,
                            branch_id=int(bi),
                        )
                        print(f"[GT_DEBUG] raw plot -> {raw_png}")
                        print(f"[GT_DEBUG] stacked plot -> {stacked_png}")
                        raise RuntimeError("[HALT] stopping after GT branch geometry inspection")
                    if DEBUG_SPLIT and _dbg(base_name):
                        _dlog(
                            1,
                            f"[SPLIT_IN] branch={int(bi)} mk=ALL | "
                            f"mid_len={len(S_branch)} | n_segs={len(segs_branch)}",
                        )
                        for _j, _seg in enumerate(segs_branch):
                            _dlog(1, f"[SPLIT_TARGET] branch={int(bi)} seg_local={int(_j)} seg_len={len(np.asarray(_seg, float))}")
                    S_local = S_branch.copy()
                    S_local[:, 0] -= float(bx)
                    S_local[:, 1] -= float(by)
                    branch_specs.append((int(bi), idxs, bx, by, mask_use, rgb_use, depth_use, S_local, S_branch))

                def _run_branch_centering(spec):
                    bi, idxs, bx, by, mask_use, rgb_use, depth_use, S_local, S_branch = spec
                    t_mid0 = time.perf_counter()
                    center_res = compute_midline_method_variants_and_normals(
                        mid_xy=S_local,
                        crack_mask_u8=mask_use,
                        domain_u8=None,
                        image_rgb=rgb_use,
                        depth_full=None,
                        depth_crop=depth_use,
                        depth_bbox_xywh=None,
                        full_image_hw=(int(mask_use.shape[0]), int(mask_use.shape[1])),
                        max_radius=50,
                        snap_kwargs={
                            "n_iters": int(auto_centering_iters),
                            "step_px": float(auto_centering_step_px),
                            "keep_endpoints": True,
                        },
                        endpoint_mode=combined_endpoint_mode,
                        debug_base_name=base_name,
                        debug_branch_id=int(bi),
                    )
                    elapsed = float(time.perf_counter() - t_mid0)
                    return bi, idxs, bx, by, mask_use, S_branch, center_res, elapsed

                n_branches = len(branch_specs)
                branch_results = [None] * n_branches
                if n_branches > 1:
                    with ThreadPoolExecutor(max_workers=n_branches) as ex:
                        fut_map = {ex.submit(_run_branch_centering, spec): i for i, spec in enumerate(branch_specs)}
                        for fut in as_completed(fut_map):
                            i = fut_map[fut]
                            branch_results[i] = fut.result()
                else:
                    for i, spec in enumerate(branch_specs):
                        branch_results[i] = _run_branch_centering(spec)

                for result in branch_results:
                    if result is None:
                        continue
                    bi, idxs, bx, by, mask_use, S_branch, center_res, elapsed = result
                    accP["mid"] += elapsed
                    _acc_nested_timing(
                        combined_timing_blob,
                        _timing_by_method(
                            center_res.get("methods", {}) if isinstance(center_res.get("methods", {}), dict) else {}
                        ),
                    )
                    methods_local = center_res.get("methods", {}) if isinstance(center_res.get("methods", {}), dict) else {}
                    if DEBUG_SPLIT and _dbg(base_name):
                        for _mk in METHOD_SPECS.keys():
                            _mv = methods_local.get(_mk, {}) if isinstance(methods_local.get(_mk, {}), dict) else {}
                            _mseg = _mv.get("midline", None)
                            _mlen = 0
                            try:
                                _marr = np.asarray(_mseg, float)
                                if _marr.ndim == 2 and _marr.shape[1] == 2:
                                    _mlen = int(len(_marr))
                            except Exception:
                                _mlen = 0
                            _dlog(
                                1,
                                f"[SPLIT_IN] branch={int(bi)} mk={str(_mk)} "
                                f"| mid_len={_mlen} | n_segs={len(np.asarray(S_branch, float))}",
                            )

                    for _mk, _mv in methods_local.items():
                        if not isinstance(_mv, dict):
                            continue
                        _mseg = _mv.get("midline", None)
                        if _mseg is not None:
                            try:
                                _mseg = np.asarray(_mseg, float)
                                if _mseg.ndim == 2 and _mseg.shape[1] == 2 and len(_mseg) >= 2:
                                    _mseg[:, 0] += float(bx)
                                    _mseg[:, 1] += float(by)
                                    _mv["midline"] = _mseg
                            except Exception:
                                pass
                        _mn = _mv.get("normals", {})
                        if isinstance(_mn, dict):
                            for ex_key, ey_key in (("edge1_x", "edge1_y"), ("edge2_x", "edge2_y")):
                                try:
                                    ex = np.asarray(_mn.get(ex_key, []), float)
                                    ey = np.asarray(_mn.get(ey_key, []), float)
                                    if ex.size:
                                        _mn[ex_key] = (ex + float(bx)).tolist()
                                    if ey.size:
                                        _mn[ey_key] = (ey + float(by)).tolist()
                                except Exception:
                                    continue

                    for j, seg_i in enumerate(idxs):
                        if seg_i < len(segs) and seg_i < len(w_list):
                            if DEBUG_SPLIT and _dbg(base_name):
                                _dlog(
                                    1,
                                    f"[SPLIT_OUT] branch={int(bi)} mk=ALL seg_local={int(j)} "
                                    f"out_len={len(np.asarray(segs[seg_i], float))}",
                                )
                            seg_work_items.append((
                                int(seg_i),
                                np.asarray(segs[seg_i], float),
                                w_list[seg_i],
                                methods_local,
                                mask_use,
                                bx,
                                by,
                                f"branch_{int(bi)}",
                                np.asarray(S_branch, float),
                                bool(j == 0),
                            ))
            else:
                seg_work_items = []
                for seg_i, (S, w_manual) in enumerate(zip(segs, w_list)):
                    S = np.asarray(S, float)
                    bi = int(seg_meta[seg_i].get("branch_id", -1)) if seg_i < len(seg_meta) else -1
                    cache = branch_local_cache.get(int(bi), None)
                    if cache is not None:
                        bx, by, bw, bh = [int(v) for v in cache.get("bbox_local", [0, 0, 0, 0])]
                        mask_use = np.asarray(cache.get("mask_local"), np.uint8)
                        rgb_use = np.asarray(cache.get("rgb_local"))
                        depth_use = cache.get("depth_local", None)
                        S_local = np.asarray(S, float).copy()
                        S_local[:, 0] -= float(bx)
                        S_local[:, 1] -= float(by)
                    else:
                        bx, by, bw, bh = 0, 0, int(Wm), int(Hm)
                        mask_use = branch_allowed_masks.get(
                            bi,
                            (crack_mask_clipped > 0).astype(np.uint8),
                        )
                        rgb_use = np.asarray(original_image)
                        depth_use = depth_full
                        S_local = np.asarray(S, float)
                    t_mid0 = time.perf_counter()
                    center_res = compute_midline_method_variants_and_normals(
                        mid_xy=S_local,
                        crack_mask_u8=mask_use,
                        domain_u8=None,
                        image_rgb=rgb_use,
                        depth_full=None,
                        depth_crop=depth_use,
                        depth_bbox_xywh=None,
                        full_image_hw=(int(mask_use.shape[0]), int(mask_use.shape[1])),
                        max_radius=50,
                        snap_kwargs={
                            "n_iters": int(auto_centering_iters),
                            "step_px": float(auto_centering_step_px),
                            "keep_endpoints": True,
                        },
                        endpoint_mode=combined_endpoint_mode,
                        debug_base_name=base_name,
                        debug_branch_id=int(bi),
                    )
                    accP["mid"] += float(time.perf_counter() - t_mid0)
                    _acc_nested_timing(
                        combined_timing_blob,
                        _timing_by_method(
                            center_res.get("methods", {}) if isinstance(center_res.get("methods", {}), dict) else {}
                        ),
                    )
                    methods_local = center_res.get("methods", {}) if isinstance(center_res.get("methods", {}), dict) else {}
                    if cache is not None:
                        for _mk, _mv in methods_local.items():
                            if not isinstance(_mv, dict):
                                continue
                            _mseg = _mv.get("midline", None)
                            if _mseg is not None:
                                try:
                                    _mseg = np.asarray(_mseg, float)
                                    if _mseg.ndim == 2 and _mseg.shape[1] == 2 and len(_mseg) >= 2:
                                        _mseg[:, 0] += float(bx)
                                        _mseg[:, 1] += float(by)
                                        _mv["midline"] = _mseg
                                except Exception:
                                    pass
                            _mn = _mv.get("normals", {})
                            if isinstance(_mn, dict):
                                for ex_key, ey_key in (("edge1_x", "edge1_y"), ("edge2_x", "edge2_y")):
                                    try:
                                        ex = np.asarray(_mn.get(ex_key, []), float)
                                        ey = np.asarray(_mn.get(ey_key, []), float)
                                        if ex.size:
                                            _mn[ex_key] = (ex + float(bx)).tolist()
                                        if ey.size:
                                            _mn[ey_key] = (ey + float(by)).tolist()
                                    except Exception:
                                        continue
                    seg_work_items.append((
                        int(seg_i),
                        S,
                        w_manual,
                        methods_local,
                        mask_use,
                        bx,
                        by,
                        f"seg_{int(seg_i)}",
                        np.asarray(S, float),
                        True,
                    ))

            split_expected_per_branch = {int(bi): int(len(idxs)) for bi, idxs in (branch_seg_indices or {}).items()}
            split_nonzero_counts = {}
            for seg_i, S, w_manual, methods_local, mask_use, bx, by, blob_id, blob_manual_seg, emit_cost_debug in seg_work_items:
                suppress_branch_reuse = False
                methods_effective = methods_local
                if DEBUG_SUPPRESS and _dbg(base_name):
                    _dlog(
                        1,
                        f"[SUPPRESS] blob_id={str(blob_id)} emit_cost_debug={bool(emit_cost_debug)} "
                        f"suppress_branch_reuse={bool(suppress_branch_reuse)}",
                    )
                branch_id = int(seg_meta[seg_i].get("branch_id", -1)) if seg_i < len(seg_meta) and isinstance(seg_meta[seg_i], dict) else -1
                for mk in METHOD_SPECS.keys():
                    mv = methods_effective.get(mk, {}) if isinstance(methods_effective.get(mk, {}), dict) else {}
                    mseg = mv.get("midline", None)
                    mnorm = mv.get("normals", {}) if isinstance(mv.get("normals", {}), dict) else {}
                    mdiag = mv.get("normals_diag", {}) if isinstance(mv.get("normals_diag", {}), dict) else {}
                    mtime = mv.get("timing", {}) if isinstance(mv.get("timing", {}), dict) else {}
                    mdebug = mv.get("debug", {}) if isinstance(mv.get("debug", {}), dict) else {}
                    mm = method_variant_segments.get(mk, {})
                    if mseg is not None:
                        mseg = np.asarray(mseg, float)
                    if DEBUG_SPLIT and _dbg(base_name):
                        _in_len = 0
                        if mseg is not None and np.asarray(mseg).ndim == 2 and np.asarray(mseg).shape[1] == 2:
                            _in_len = int(len(mseg))
                        _dlog(
                            1,
                            f"[SPLIT_OUT] branch={branch_id} mk={str(mk)} seg_local={int(seg_i)} out_len={_in_len}",
                        )
                        _k = (int(branch_id), str(mk))
                        split_nonzero_counts[_k] = int(split_nonzero_counts.get(_k, 0)) + (1 if _in_len > 0 else 0)
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
                    if mm.get("plot_debug") is None and isinstance(mdebug, dict):
                        cm = mdebug.get("costmap", None)
                        if cm is None and isinstance(mdebug.get("costmaps", {}), dict):
                            sk = str(mdebug.get("selected_cost_key", "dt"))
                            cm = mdebug.get("costmaps", {}).get(sk, None)
                        if cm is not None:
                            try:
                                cm_arr = np.asarray(cm, np.float32)
                            except Exception:
                                cm_arr = None
                            if cm_arr is not None and cm_arr.ndim >= 2:
                                manual_local = np.asarray(blob_manual_seg, float).copy()
                                pred_local = np.asarray(mseg, float).copy() if mseg is not None else None
                                try:
                                    manual_local[:, 0] -= float(bx)
                                    manual_local[:, 1] -= float(by)
                                except Exception:
                                    pass
                                if pred_local is not None:
                                    try:
                                        pred_local[:, 0] -= float(bx)
                                        pred_local[:, 1] -= float(by)
                                    except Exception:
                                        pred_local = None
                                mm["plot_debug"] = {
                                    "costmap": cm_arr,
                                    "mask_use": np.asarray(mask_use, np.uint8),
                                    "manual_local": manual_local,
                                    "pred_local": pred_local,
                                    "normals": mnorm if isinstance(mnorm, dict) else None,
                                    "bx": int(bx),
                                    "by": int(by),
                                }
                    if bool(emit_cost_debug) and isinstance(mdebug, dict):
                        cm = mdebug.get("costmap", None)
                        if cm is None and isinstance(mdebug.get("costmaps", {}), dict):
                            sk = str(mdebug.get("selected_cost_key", "dt"))
                            cm = mdebug.get("costmaps", {}).get(sk, None)
                        if cm is not None:
                            try:
                                cm_arr = np.asarray(cm, np.float32)
                            except Exception:
                                cm_arr = None
                            if cm_arr is not None and cm_arr.ndim >= 2:
                                mm.setdefault("plot_debug_branches", [])
                                print(
                                    f"[BRANCH_DBG_BXY] blob_id={blob_id} bx={int(bx)} by={int(by)} "
                                    f"mask_shape={np.asarray(mask_use).shape}"
                                )
                                mm["plot_debug_branches"].append({
                                    "blob_id": str(blob_id),
                                    "mask_use": np.asarray(mask_use, np.uint8),
                                    "costmaps": mdebug.get("costmaps", {}),
                                    "dt_norm": mdebug.get("dt_norm"),
                                    "depth_norm": mdebug.get("depth_norm"),
                                    "recess_norm": mdebug.get("recess_norm"),
                                    "multi_cue_score": mdebug.get("score_for_refine"),
                                    "bx": int(bx),
                                    "by": int(by),
                                })
                    if DEBUG_SPLIT and _dbg(base_name):
                        _assigned = int(len(mm.get("midline_segments", []) or []))
                        _dlog(1, f"[ASSIGN] mk={str(mk)} assigned_segments={_assigned}")
                    for tk, tv in mtime.items():
                        try:
                            mm["timing_sum"][tk] = float(mm["timing_sum"].get(tk, 0.0)) + float(tv or 0.0)
                        except Exception:
                            continue
                    # Combined debug plotting now reads from combined_entry["method_results"]
                    # after aggregation, so per-blob caches are no longer needed here.

                m1_local = methods_local.get("dt", {}) if isinstance(methods_local.get("dt", {}), dict) else {}
                fused_local = methods_local.get("dt_ridge_color_depth", {}) if isinstance(methods_local.get("dt_ridge_color_depth", {}), dict) else {}
                if not fused_local:
                    fused_local = methods_local.get("dt_ridge_valley_depth", {}) if isinstance(methods_local.get("dt_ridge_valley_depth", {}), dict) else {}
                if not fused_local:
                    fused_local = methods_local.get("dt_depth", {}) if isinstance(methods_local.get("dt_depth", {}), dict) else {}
                if suppress_branch_reuse:
                    m1_local = {}
                    fused_local = {}

            if DEBUG_SPLIT and _dbg(base_name):
                for (bid, mk), nz in split_nonzero_counts.items():
                    exp = int(split_expected_per_branch.get(int(bid), 0))
                    if exp > 0 and nz < exp:
                        _dlog(1, f"[SPLIT_WARN] branch={int(bid)} mk={str(mk)} {int(nz)}/{int(exp)} segments received midline")

            for seg_i, S, w_manual, methods_local, mask_use, bx, by, blob_id, blob_manual_seg, emit_cost_debug in seg_work_items:
                m1_local = methods_local.get("dt", {}) if isinstance(methods_local.get("dt", {}), dict) else {}
                fused_local = methods_local.get("dt_ridge_color_depth", {}) if isinstance(methods_local.get("dt_ridge_color_depth", {}), dict) else {}
                if not fused_local:
                    fused_local = methods_local.get("dt_ridge_valley_depth", {}) if isinstance(methods_local.get("dt_ridge_valley_depth", {}), dict) else {}
                if not fused_local:
                    fused_local = methods_local.get("dt_depth", {}) if isinstance(methods_local.get("dt_depth", {}), dict) else {}

                centered_S = np.asarray(m1_local.get("midline", S), float)
                centered_normals = m1_local.get("normals", {})
                if not isinstance(centered_normals, dict):
                    centered_normals = {}
                cseg_diag = m1_local.get("normals_diag", {})
                if not isinstance(cseg_diag, dict):
                    cseg_diag = {}
                per_seg_centered_normals_diag.append(cseg_diag)
                csdiag_brief = _normals_diag_summary(cseg_diag)
                _dlog(3, (
                    f"[GT_SUP NORMDBG] combined {ccid} seg={seg_i} centered total={csdiag_brief['total']} "
                    f"valid={csdiag_brief['valid']} invalid={csdiag_brief['invalid']} "
                    f"invalid_frac={csdiag_brief['invalid_frac']:.4f} top_reasons={csdiag_brief['top_reasons']}"
                ))

                centered_segs.append(centered_S)
                ce1x_list.append(np.asarray(centered_normals.get("edge1_x", []), float))
                ce1y_list.append(np.asarray(centered_normals.get("edge1_y", []), float))
                ce2x_list.append(np.asarray(centered_normals.get("edge2_x", []), float))
                ce2y_list.append(np.asarray(centered_normals.get("edge2_y", []), float))
                cw_list.append(np.asarray(centered_normals.get("width_px", []), float))

                dseg = fused_local.get("midline", None)
                dnorm = fused_local.get("normals", {})
                if not isinstance(dnorm, dict):
                    dnorm = {}
                dseg_diag = fused_local.get("normals_diag", {})
                if not isinstance(dseg_diag, dict):
                    dseg_diag = {}
                per_seg_depth_normals_diag.append(dseg_diag)
                depth_cost_meta_per_segment.append(fused_local.get("meta", {}) if isinstance(fused_local.get("meta", {}), dict) else {})

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
                    _dlog(3, (
                        f"[GT_SUP NORMDBG] combined {ccid} seg={seg_i} depth total={dsdiag_brief['total']} "
                        f"valid={dsdiag_brief['valid']} invalid={dsdiag_brief['invalid']} "
                        f"invalid_frac={dsdiag_brief['invalid_frac']:.4f} top_reasons={dsdiag_brief['top_reasons']}"
                    ))
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
                combined_entry["multi_cue_midline_segments"] = [np.asarray(S, float).tolist() for S in depth_valid_segs]
                combined_entry["multi_cue_midline_segments_meta"] = depth_meta_valid
                combined_entry["multi_cue_midline"] = _pack_segs_with_separators(depth_valid_segs)
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
                combined_entry["fused_midline_segments"] = combined_entry["multi_cue_midline_segments"]
                combined_entry["fused_midline_segments_meta"] = combined_entry["multi_cue_midline_segments_meta"]
                combined_entry["fused_midline"] = combined_entry["multi_cue_midline"]
                combined_entry["fused_normals"] = combined_entry["depth_normals"]
                combined_entry["fused_normals_diag_per_segment"] = combined_entry["depth_normals_diag_per_segment"]
                combined_entry["midline_fused"] = combined_entry["fused_midline"]
                combined_entry["gt_normals_fused"] = combined_entry["fused_normals"]
                combined_entry["gt_widths_fused"] = [
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
                    "midline": _pack_segs_with_separators(m_segs) if m_segs else None,
                    "normals": (
                        {
                            "edge1_x": _pack_arrs_with_none_separators([_arr_to_list(a) for a in e1x]),
                            "edge1_y": _pack_arrs_with_none_separators([_arr_to_list(a) for a in e1y]),
                            "edge2_x": _pack_arrs_with_none_separators([_arr_to_list(a) for a in e2x]),
                            "edge2_y": _pack_arrs_with_none_separators([_arr_to_list(a) for a in e2y]),
                            "width_px": _pack_arrs_with_none_separators([_arr_to_list(a) for a in ww]),
                        }
                        if m_segs else None
                    ),
                    "normals_diag_per_segment": mpack.get("normals_diag_per_segment", []),
                    "timing": mpack.get("timing_sum", {}),
                    "meta": {
                        "label": METHOD_SPECS.get(mk, {}).get("label"),
                        "use_rgb": bool(METHOD_SPECS.get(mk, {}).get("use_rgb", False)),
                        "use_depth": bool(METHOD_SPECS.get(mk, {}).get("use_depth", False)),
                        "status": "ok" if m_segs else "failed",
                        "reason": None if m_segs else "no_valid_segments",
                    },
                }
            combined_entry["method_results"] = dict(combined_entry.get("method_variants", {}))

            combined_entry["multi_cue_cost_meta"] = {
                "per_segment": depth_cost_meta_per_segment,
            }

            combined_entry["timing"]["methods"] = combined_timing_blob.get("methods", {})
            timing_totals["combined_centering_sec"] += _sum_centering_seconds(combined_timing_blob)
            _accumulate_timing_blob(combined_timing_blob)

            manual_geom_parts = [np.asarray(S, float) for S in segs if S is not None and len(S) >= 2]
            center_geom_parts = [np.asarray(S, float) for S in centered_segs if S is not None and len(S) >= 2]
            depth_geom_parts = [
                np.asarray(S, float)
                for S in (combined_entry.get("fused_midline_segments", combined_entry.get("multi_cue_midline_segments", [])) or [])
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
                "et_vs_dt": {
                    **shift_meta,
                    **_geometry_disagreement_stats(manual_geom_all, center_geom_all),
                    **_width_stability_stats(
                        np.concatenate([np.asarray(a, float).reshape(-1) for a in w_list]) if w_list else [],
                        np.concatenate([np.asarray(a, float).reshape(-1) for a in cw_list]) if cw_list else [],
                    ),
                },
                "et_vs_fused": (
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
                "available_methods": [
                    mk for mk in METHOD_SPECS.keys()
                    if (
                        (combined_entry.get("method_variants", {}) or {}).get(mk, {})
                        if isinstance((combined_entry.get("method_variants", {}) or {}).get(mk, {}), dict) else {}
                    ).get("midline_segments")
                ],
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

            if DEBUG_LIGHT and _dbg(base_name):
                _branch_ids_present = sorted({
                    int((m or {}).get("branch_id", -1))
                    for m in (seg_meta or [])
                    if isinstance(m, dict)
                })
                _dlog(1, f"[FINAL] combined branches present: {_branch_ids_present}")

            if auto_centering_outputs:
                combined_dbg_dir = os.path.join(auto_center_root, "combined", f"ccid_{tag_name}")
                os.makedirs(combined_dbg_dir, exist_ok=True)
                terr_vis = build_territory_mask_for_segments(
                    segs=segs,
                    crack_mask_u8=crack_mask_clipped,
                    window_half_size=int(auto_centering_window_half_size),
                )
                target_methods = list(METHODS_RS3)
                methods_for_plot = combined_entry.get("method_results", {})
                if not isinstance(methods_for_plot, dict):
                    methods_for_plot = {}

                def _assemble_full_from_branches(branch_recs, field):
                    shape_hw = np.asarray(crack_mask, np.uint8).shape[:2]
                    full = np.full(shape_hw, np.nan, np.float32)
                    is_cost_field = ("cost" in str(field).lower())

                    def _fetch_arr(_rec):
                        if field == "selected_cost":
                            cmaps = _rec.get("costmaps", {}) if isinstance(_rec.get("costmaps", {}), dict) else {}
                            arr0 = cmaps.get("selected", None)
                            if arr0 is None:
                                skey = str(cmaps.get("selected_key", "dt"))
                                arr0 = cmaps.get(skey, cmaps.get("dt", None))
                            return arr0
                        return _rec.get(field, None)

                    ordered = sorted(
                        (branch_recs or []),
                        key=lambda _b: -float(
                            np.prod(np.asarray(_fetch_arr(_b)).shape[:2])
                            if (_fetch_arr(_b) is not None and np.asarray(_fetch_arr(_b)).ndim >= 2)
                            else 0.0
                        ),
                    )

                    for _r in ordered:
                        arr = _fetch_arr(_r)
                        if arr is None:
                            continue
                        a = np.asarray(arr, np.float32)
                        if a.ndim < 2:
                            continue

                        by = int(_r.get("by", 0) or 0)
                        bx = int(_r.get("bx", 0) or 0)
                        h, w = a.shape[:2]
                        y0 = max(0, by)
                        x0 = max(0, bx)
                        y1 = min(shape_hw[0], by + h)
                        x1 = min(shape_hw[1], bx + w)
                        if y1 <= y0 or x1 <= x0:
                            continue

                        patch = a[:(y1 - y0), :(x1 - x0)]
                        sub = full[y0:y1, x0:x1]
                        valid = np.isfinite(patch)
                        if not np.any(valid):
                            continue

                        if is_cost_field:
                            write = valid & (~np.isfinite(sub) | (patch < sub))
                        else:
                            write = valid & (~np.isfinite(sub) | (patch > sub))
                        if not np.any(write):
                            continue

                        sub[write] = patch[write]
                        full[y0:y1, x0:x1] = sub
                    return full

                def _assemble_costmap_key_from_branches(branch_recs, cmap_key):
                    shape_hw = np.asarray(crack_mask, np.uint8).shape[:2]
                    acc = np.zeros(shape_hw, np.float32)
                    wgt = np.zeros(shape_hw, np.float32)
                    for _r in (branch_recs or []):
                        by = int(_r.get("by", 0) or 0)
                        bx = int(_r.get("bx", 0) or 0)
                        cmaps = _r.get("costmaps", {}) if isinstance(_r.get("costmaps", {}), dict) else {}
                        arr = cmaps.get(str(cmap_key), None)
                        if arr is None:
                            continue
                        a = np.asarray(arr, np.float32)
                        if a.ndim < 2:
                            continue
                        h, w = a.shape[:2]
                        y0 = max(0, by)
                        x0 = max(0, bx)
                        y1 = min(shape_hw[0], by + h)
                        x1 = min(shape_hw[1], bx + w)
                        if y1 <= y0 or x1 <= x0:
                            continue
                        patch = a[:(y1 - y0), :(x1 - x0)]
                        valid = np.isfinite(patch)
                        if not np.any(valid):
                            continue
                        acc_view = acc[y0:y1, x0:x1]
                        wgt_view = wgt[y0:y1, x0:x1]
                        acc_view[valid] += patch[valid]
                        wgt_view[valid] += 1.0
                        acc[y0:y1, x0:x1] = acc_view
                        wgt[y0:y1, x0:x1] = wgt_view
                    full = np.full(shape_hw, np.nan, np.float32)
                    nz = wgt > 0
                    full[nz] = acc[nz] / wgt[nz]
                    return full

                for mk in target_methods:
                    mv = methods_for_plot.get(mk, {}) if isinstance(methods_for_plot.get(mk, {}), dict) else {}
                    style = method_style.get(mk, {})
                    mslug = str(style.get("slug", mk))
                    method_dir = os.path.join(combined_dbg_dir, mslug)
                    os.makedirs(method_dir, exist_ok=True)
                    mpack_dbg = method_variant_segments.get(mk, {}) if isinstance(method_variant_segments.get(mk, {}), dict) else {}
                    pdbg = mpack_dbg.get("plot_debug", None) if isinstance(mpack_dbg.get("plot_debug", None), dict) else None
                    branch_dbg_list = mpack_dbg.get("plot_debug_branches", []) if isinstance(mpack_dbg.get("plot_debug_branches", []), list) else []
                    _mask_for_bbox = np.asarray(crack_mask, np.uint8)
                    _H_bbox, _W_bbox = _mask_for_bbox.shape[:2]
                    _ys_bbox, _xs_bbox = np.where(_mask_for_bbox > 0)
                    if _xs_bbox.size > 0 and _ys_bbox.size > 0:
                        _pad_bbox = 25
                        _x0_bbox = max(0, int(_xs_bbox.min()) - _pad_bbox)
                        _y0_bbox = max(0, int(_ys_bbox.min()) - _pad_bbox)
                        _x1_bbox = min(_W_bbox, int(_xs_bbox.max()) + 1 + _pad_bbox)
                        _y1_bbox = min(_H_bbox, int(_ys_bbox.max()) + 1 + _pad_bbox)
                        _bbox_full = (
                            int(_x0_bbox),
                            int(_y0_bbox),
                            int(max(1, _x1_bbox - _x0_bbox)),
                            int(max(1, _y1_bbox - _y0_bbox)),
                        )
                    else:
                        _bbox_full = None

                    pred_segs = [
                        np.asarray(Sm, float)
                        for Sm in (mv.get("midline_segments", []) or [])
                        if Sm is not None and len(Sm) >= 2
                    ]
                    if not pred_segs:
                        mraw = mv.get("midline")
                        if mraw is not None:
                            marr = np.asarray(mraw, float)
                            if marr.ndim == 2 and marr.shape[1] == 2 and len(marr) >= 2:
                                pred_segs = [marr]
                            else:
                                pred_segs = _split_midline_packed(mraw)

                    if not pred_segs:
                        _dlog(2, f"[SKIP] combined {tag_name} {mk} (no midline)")
                        continue

                    # (0) Clean binary mask + DT diagnostic (white on black background).
                    if not BATCH_PLOTS_ONLY:
                        try:
                            import matplotlib.pyplot as _plt_diag
                            _H_diag, _W_diag = np.asarray(crack_mask, np.uint8).shape[:2]
                            _mask_global = np.zeros((_H_diag, _W_diag), np.float32)
                            _dt_global = np.zeros((_H_diag, _W_diag), np.float32)
                            for _brec in branch_dbg_list:
                                _by = int(_brec.get("by", 0) or 0)
                                _bx = int(_brec.get("bx", 0) or 0)
                                _mu = _brec.get("mask_use", None)
                                _dn = _brec.get("dt_norm", None)
                                if _mu is not None:
                                    _mu_arr = (np.asarray(_mu, np.uint8) > 0).astype(np.float32)
                                    _mh, _mw = _mu_arr.shape[:2]
                                    _y1e = min(_H_diag, _by + _mh)
                                    _x1e = min(_W_diag, _bx + _mw)
                                    _mask_global[_by:_y1e, _bx:_x1e] = np.maximum(
                                        _mask_global[_by:_y1e, _bx:_x1e],
                                        _mu_arr[:(_y1e - _by), :(_x1e - _bx)]
                                    )
                                if _dn is not None:
                                    _dn_arr = np.asarray(_dn, np.float32)
                                    _dn_arr = np.where((_dn_arr < 1e5) & np.isfinite(_dn_arr), _dn_arr, 0.0)
                                    _dh, _dw = _dn_arr.shape[:2]
                                    _y1e = min(_H_diag, _by + _dh)
                                    _x1e = min(_W_diag, _bx + _dw)
                                    _dt_global[_by:_y1e, _bx:_x1e] = np.maximum(
                                        _dt_global[_by:_y1e, _bx:_x1e],
                                        _dn_arr[:(_y1e - _by), :(_x1e - _bx)]
                                    )
                            _bb = _bbox_full
                            if _bb and len(_bb) == 4:
                                _cx, _cy, _cw, _ch = [int(v) for v in _bb]
                                _pad = 20
                                _cx0 = max(0, _cx - _pad); _cy0 = max(0, _cy - _pad)
                                _cx1 = min(_W_diag, _cx + _cw + _pad); _cy1 = min(_H_diag, _cy + _ch + _pad)
                            else:
                                _cy0, _cx0, _cy1, _cx1 = 0, 0, _H_diag, _W_diag
                            _fig_diag, (_ax_m, _ax_d, _ax_c) = _plt_diag.subplots(1, 3, figsize=(12, 10))
                            _ax_m.imshow(_mask_global[_cy0:_cy1, _cx0:_cx1], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                            _ax_m.set_title("Binary mask", fontsize=9)
                            _ax_m.axis("off")
                            _dt_crop = _dt_global[_cy0:_cy1, _cx0:_cx1]
                            _vmax = float(np.max(_dt_crop)) if np.any(_dt_crop > 0) else 1.0
                            # DT: black background for zero pixels, viridis for crack interior
                            _dt_cmap = _plt_diag.cm.get_cmap("viridis").copy()
                            _dt_cmap.set_under(color="black")
                            _ax_d.imshow(_dt_crop, cmap=_dt_cmap, vmin=1e-6, vmax=_vmax, interpolation="nearest")
                            _ax_d.set_title("DT norm (black=0, purple=edge, yellow=center)", fontsize=9)
                            _ax_d.axis("off")
                            # Costmap: assemble from branch_dbg_list same as DT
                            _cost_global = np.zeros((_H_diag, _W_diag), np.float32)
                            for _brec in branch_dbg_list:
                                _by2 = int(_brec.get("by", 0) or 0)
                                _bx2 = int(_brec.get("bx", 0) or 0)
                                _cmaps = _brec.get("costmaps", {}) if isinstance(_brec.get("costmaps", {}), dict) else {}
                                _sel = _cmaps.get("selected", None)
                                if _sel is None:
                                    _sk = str(_cmaps.get("selected_key", "dt"))
                                    _sel = _cmaps.get(_sk, _cmaps.get("dt", None))
                                if _sel is None:
                                    continue
                                _sel_arr = np.asarray(_sel, np.float32)
                                # Kill sentinels, remap so low cost = high brightness
                                _sel_arr = np.where((_sel_arr < 1e5) & np.isfinite(_sel_arr), _sel_arr, 0.0)
                                _ch2, _cw2 = _sel_arr.shape[:2]
                                _y1e2 = min(_H_diag, _by2 + _ch2)
                                _x1e2 = min(_W_diag, _bx2 + _cw2)
                                _cost_global[_by2:_y1e2, _bx2:_x1e2] = np.maximum(
                                    _cost_global[_by2:_y1e2, _bx2:_x1e2],
                                    _sel_arr[:(_y1e2 - _by2), :(_x1e2 - _bx2)]
                                )
                            _cost_crop = _cost_global[_cy0:_cy1, _cx0:_cx1]
                            # Normalize raw cost within crack pixels only (low cost stays dark/purple).
                            _cost_valid = _cost_crop[_cost_crop > 1e-9]
                            if _cost_valid.size > 0:
                                _clo = float(np.percentile(_cost_valid, 2))
                                _chi = float(np.percentile(_cost_valid, 98))
                                _cost_viz = np.where(_cost_crop > 1e-9,
                                    np.clip((_cost_crop - _clo) / max(_chi - _clo, 1e-9), 0.0, 1.0),
                                    0.0).astype(np.float32)
                            else:
                                _cost_viz = np.zeros_like(_cost_crop)
                            _cost_cmap = _plt_diag.cm.get_cmap("viridis").copy()
                            _cost_cmap.set_under(color="black")
                            _ax_c.imshow(_cost_viz, cmap=_cost_cmap, vmin=1e-6, vmax=1.0, interpolation="nearest")
                            _ax_c.set_title("Costmap inverted (black=0, yellow=low cost)", fontsize=9)
                            _ax_c.axis("off")
                            _fig_diag.suptitle(f"{tag_name} {mk} mask+DT", fontsize=8)
                            _fig_diag.tight_layout()
                            _out_diag = os.path.join(method_dir, "mask_costmap.jpg")
                            _fig_diag.savefig(_out_diag, dpi=100, bbox_inches="tight", facecolor="black")
                            _plt_diag.close(_fig_diag)
                        except Exception as _e_diag:
                            _dlog(2, f"[DIAG PLOT FAIL] mask+DT plot failed: {_e_diag}")

                    # (1) Overlay-only combined debug (old visual style).
                    out_overlay = os.path.join(method_dir, "combined_overlay.jpg")
                    _run_timed_plot(
                        plot_combined_overlay_debug,
                        out_path=out_overlay,
                        original_image=original_image,
                        crack_mask_u8=crack_mask,
                        manual_segs=[np.asarray(S, float) for S in segs],
                        pred_segs=pred_segs,
                        normals=mv.get("normals"),
                        territory_u8=terr_vis,
                        bbox_xywh=_bbox_full,
                        title=f"Combined - {style.get('label', mk)}",
                        compare_label=style.get("compare_label", mk),
                        compare_color="magenta",
                    )

                    selected_full = _assemble_full_from_branches(branch_dbg_list, "selected_cost")
                    dt_full = _assemble_full_from_branches(branch_dbg_list, "dt_norm")
                    depth_full = _assemble_full_from_branches(branch_dbg_list, "depth_norm")
                    recess_full = _assemble_full_from_branches(branch_dbg_list, "recess_norm")
                    score_full = _assemble_full_from_branches(branch_dbg_list, "multi_cue_score")

                    if selected_full is not None and np.any(np.isfinite(selected_full)):
                        out_side = os.path.join(method_dir, "combined_side_by_side.jpg")
                        _run_timed_plot(
                            plot_combined_overlay_and_costmap,
                            out_path=out_side,
                            original_image=original_image,
                            crack_mask_u8=crack_mask,
                            manual_segs=[np.asarray(S, float) for S in segs],
                            pred_segs=pred_segs,
                            selected_costmap=selected_full,
                            territory_u8=terr_vis,
                            bbox_xywh=_bbox_full,
                            title=f"Combined - {style.get('label', mk)}",
                            compare_label=style.get("compare_label", mk),
                            compare_color="magenta",
                        )

                    # (3) Global aggregated cost panel (MIN for final cost, AVG for intermediates).
                    use_ridge = mk in {"dt_ridge_valley", "dt_ridge_valley_depth", "dt_ridge_color_depth"}
                    use_depth = mk in {"dt_depth", "dt_ridge_valley_depth", "dt_ridge_color_depth"}
                    use_rgb = mk == "dt_ridge_color_depth"

                    def _first_non_none(*vals):
                        for v in vals:
                            if v is not None:
                                return v
                        return None

                    def _ks(a):
                        """Kill sentinels: zero out any value >= 1e5 or non-finite."""
                        if a is None:
                            return None
                        arr = np.asarray(a, np.float32)
                        arr = np.where(np.isfinite(arr) & (arr < np.float32(1e5)), arr, np.float32(np.nan))
                        return arr

                    branches_for_global = []
                    for brec in branch_dbg_list:
                        cmaps = brec.get("costmaps", {}) if isinstance(brec.get("costmaps", {}), dict) else {}
                        rgb_src = _first_non_none(cmaps.get("rgb_cue", None), cmaps.get("rgb", None))
                        ridge_src = _first_non_none(cmaps.get("ridge_valley", None), cmaps.get("dt_ridge_valley", None))
                        selected_b = cmaps.get("selected", None)
                        if selected_b is None:
                            sk = str(cmaps.get("selected_key", "dt"))
                            selected_b = cmaps.get(sk, cmaps.get("dt", None))
                        if selected_b is None:
                            continue
                        a_sel = np.asarray(selected_b, np.float32)
                        if a_sel.ndim < 2:
                            continue
                        by = int(brec.get("by", 0) or 0)
                        bx = int(brec.get("bx", 0) or 0)
                        branches_for_global.append({
                            "bbox": [by, by + int(a_sel.shape[0]), bx, bx + int(a_sel.shape[1])],
                            "dt_norm": _ks(brec.get("dt_norm")),
                            "rgb": _ks(rgb_src) if use_rgb else None,
                            "ridge": _ks(ridge_src) if use_ridge else None,
                            "depth": _ks(brec.get("depth_norm")) if use_depth else None,
                            "recess_norm": _ks(brec.get("recess_norm")) if use_depth else None,
                            "final_cost": _ks(a_sel),
                        })
                    if branches_for_global and not BATCH_PLOTS_ONLY:
                        maps_global = build_global_cost_maps(branches_for_global, np.asarray(crack_mask, np.uint8).shape[:2])
                        print(
                            f"[GLOBAL SIGNAL CHECK] mk={mk} "
                            f"rgb_finite={np.any(np.isfinite(maps_global.get('rgb')))} "
                            f"ridge_finite={np.any(np.isfinite(maps_global.get('ridge')))} "
                            f"rgb_std={float(np.nanstd(maps_global.get('rgb')) if maps_global.get('rgb') is not None else np.nan):.6f} "
                            f"ridge_std={float(np.nanstd(maps_global.get('ridge')) if maps_global.get('ridge') is not None else np.nan):.6f}"
                        )
                        out_cost_panel = os.path.join(method_dir, "combined_cost_panel.jpg")
                        _run_timed_plot(
                            save_global_cost_panel,
                            maps=maps_global,
                            out_path=out_cost_panel,
                            crack_mask_u8=crack_mask,
                            bbox_xywh=_bbox_full,
                            title=f"combined {tag_name} [{style.get('label', mk)}]",
                        )
                        # Temporary: disable overlap-count plot emission.
                        # out_winner = os.path.join(combined_dbg_dir, f"combined_{mslug}_winner_map.jpg")
                        # _save_winner_fn = globals().get("save_winner_panel", None)
                        # if callable(_save_winner_fn):
                        #     _run_timed_plot(
                        #         _save_winner_fn,
                        #         winner_id_map=maps_global.get("winner_id"),
                        #         out_path=out_winner,
                        #         title=f"combined {tag_name} winner branch id",
                        #     )
                        # else:
                        #     _dlog(1, "[WARN] save_winner_panel unavailable; skipping winner map.")

                    # Keep per-branch cost panels in separate folders.
                    if branch_dbg_list and not BATCH_PLOTS_ONLY:
                        for brec in branch_dbg_list:
                            blob_id = str(brec.get("blob_id", "branch_000"))
                            bidx = 0
                            if blob_id.startswith("branch_"):
                                try:
                                    bidx = int(blob_id.split("_", 1)[1])
                                except Exception:
                                    bidx = 0
                            out_cost = os.path.join(method_dir, f"branch_{int(bidx):03d}_cost.png")
                            _cm = brec.get("costmaps", {}) if isinstance(brec.get("costmaps", {}), dict) else {}
                            _rgb_src = _first_non_none(_cm.get("rgb_cue"), _cm.get("rgb"))
                            _ridge_src = _first_non_none(_cm.get("ridge_valley"), _cm.get("dt_ridge_valley"))
                            print(
                                f"[COSTMAP KEYS] mk={mk} branch={int(bidx):03d} "
                                f"keys={sorted(_cm.keys())} "
                                f"has_depth_norm={brec.get('depth_norm') is not None} "
                                f"has_recess_norm={brec.get('recess_norm') is not None}"
                            )
                            try:
                                import matplotlib.pyplot as _plt_branch

                                def _ks_viz(a):
                                    if a is None:
                                        return None
                                    a = np.asarray(a, np.float32)
                                    a = np.where(np.isfinite(a) & (a < np.float32(1e5)), a, np.float32(0.0))
                                    nz = a > 1e-9
                                    if not np.any(nz):
                                        return None
                                    vals = a[nz]
                                    lo, hi = float(np.percentile(vals, 2)), float(np.percentile(vals, 98))
                                    if hi <= lo + 1e-9:
                                        return None
                                    out = np.zeros_like(a)
                                    out[nz] = np.clip((a[nz] - lo) / (hi - lo), 0.0, 1.0)
                                    return out

                                _selected_b = _cm.get("selected")
                                if _selected_b is None:
                                    _sk = str(_cm.get("selected_key", "dt"))
                                    _selected_b = _cm.get(_sk, _cm.get("dt"))
                                _cost_viz = _ks_viz(_selected_b)
                                if _cost_viz is None:
                                    continue

                                _bh, _bw = _cost_viz.shape[:2]
                                _bx0, _by0, _bx1, _by1 = 0, 0, _bw, _bh
                                _bb_local = brec.get("mask_bbox")
                                if _bb_local is not None and len(_bb_local) == 4:
                                    _x, _y, _w, _h = [int(v) for v in _bb_local]
                                    _pad = 20
                                    _bx0 = max(0, _x - _pad)
                                    _by0 = max(0, _y - _pad)
                                    _bx1 = min(_bw, _x + _w + _pad)
                                    _by1 = min(_bh, _y + _h + _pad)

                                _crop = _cost_viz[_by0:_by1, _bx0:_bx1]
                                _fig_b, _ax_b = _plt_branch.subplots(1, 1, figsize=(5.5, 5.5))
                                _cmap_b = _plt_branch.cm.get_cmap("viridis").copy()
                                _cmap_b.set_under(color="black")
                                _ax_b.imshow(_crop, cmap=_cmap_b, vmin=1e-6, vmax=1.0, interpolation="nearest")
                                _ax_b.set_title(f"combined {tag_name} [{mslug} | branch {int(bidx):03d}]", fontsize=9)
                                _ax_b.axis("off")
                                os.makedirs(os.path.dirname(out_cost), exist_ok=True)
                                _fig_b.savefig(out_cost, dpi=100, bbox_inches="tight", facecolor="black")
                                _plt_branch.close(_fig_b)
                            except Exception as _e_branch:
                                _dlog(2, f"[BRANCH COST PLOT FAIL] branch={int(bidx):03d} err={_e_branch}")
                    elif pdbg is not None:
                        # Backward fallback when branch records are unavailable.
                        pass
                        # out_cost = os.path.join(method_dir, "costmap_fallback.jpg")
                        # _run_timed_plot(
                        #     plot_costmap_debug,
                        #     out_path=out_cost,
                        #     crack_mask_u8=np.asarray(pdbg.get("mask_use"), np.uint8),
                        #     manual_segs=[np.asarray(pdbg.get("manual_local"), float)] if pdbg.get("manual_local") is not None else [],
                        #     pred_segs=[np.asarray(pdbg.get("pred_local"), float)] if pdbg.get("pred_local") is not None else [],
                        #     costmap=np.asarray(pdbg.get("costmap"), np.float32) if pdbg.get("costmap") is not None else None,
                        #     normals=pdbg.get("normals"),
                        #     normals_offset_xy=(pdbg.get("bx", 0), pdbg.get("by", 0)),
                        #     title=f"{tag_name} - {style.get('label', mk)}",
                        #     pred_color="magenta",
                        # )
        for i_seg, seg_arr in enumerate(segs or []):
            try:
                print(f"[FINAL MIDLINE] branch={int(i_seg)} len={int(len(np.asarray(seg_arr, float)))}")
            except Exception:
                print(f"[FINAL MIDLINE] branch={int(i_seg)} len=0")

        final_entries.append(combined_entry)
        gt_sup_diag["combined_added"] += 1
        if not combined_debug_done and False:
            t0 = time.perf_counter()
            _plot_single_debug(
                combined_entry,
                gt_mask,
                original_image,
                os.path.join(combined_crop_root, "combined_ablation_debug.jpg"),
            )
            t_preview += float(time.perf_counter() - t0)
            combined_debug_done = True

    # =====================================================
    # 3) GLOBAL OVERVIEW
    # =====================================================
    analysis_dir = os.path.join(sup_root, "analysis")
    dt_track_dir = os.path.join(sup_root, "dt")
    fused_track_dir = os.path.join(sup_root, "multi_cue_track")
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(dt_track_dir, exist_ok=True)
    os.makedirs(fused_track_dir, exist_ok=True)

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
            "multi_cue_align_s",
            "depth_recess_s",
            "multi_cue_costmap_s",
            "multi_cue_dijkstra_s",
            "multi_cue_postprocess_s",
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
            float(timing_totals["multi_cue_align_s"]),
            float(timing_totals["depth_recess_s"]),
            float(timing_totals["multi_cue_costmap_s"]),
            float(timing_totals["multi_cue_dijkstra_s"]),
            float(timing_totals["multi_cue_postprocess_s"]),
            float(timing_totals["normals_centered_s"]),
            float(timing_totals["normals_depth_s"]),
        ])

    # Split timing outputs by track for clearer interpretation.
    centered_timing_csv = os.path.join(dt_track_dir, "timing.csv")
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

    depth_timing_csv = os.path.join(fused_track_dir, "timing.csv")
    with open(depth_timing_csv, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow([
            "image",
            "dt_compute_s",
            "multi_cue_align_s",
            "depth_recess_s",
            "multi_cue_costmap_s",
            "multi_cue_dijkstra_s",
            "multi_cue_postprocess_s",
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
            float(timing_totals["multi_cue_align_s"]),
            float(timing_totals["depth_recess_s"]),
            float(timing_totals["multi_cue_costmap_s"]),
            float(timing_totals["multi_cue_dijkstra_s"]),
            float(timing_totals["multi_cue_postprocess_s"]),
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
        f"multi_cue_dijkstra_s={timing_totals['multi_cue_dijkstra_s']:.4f}"
    )
    print(f"[GT_SUP TIMING] wrote {compute_csv}")
    print(f"[GT_SUP TIMING] wrote {centering_csv}")
    print(f"[GT_SUP TIMING] wrote {centered_timing_csv}")
    print(f"[GT_SUP TIMING] wrote {depth_timing_csv}")

    if not final_entries:
        print(f"[GT_SUP DIAG] no final_entries produced: {gt_sup_diag}")
    else:
        print(f"[GT_SUP DIAG] entries={len(final_entries)} summary={gt_sup_diag}")
    # Per-combined final centering audit: GT branches vs each method output.
    centering_audit_dir = os.path.join(sup_root, "auto_center_debug")
    os.makedirs(centering_audit_dir, exist_ok=True)
    for _e in (final_entries or []):
        if not isinstance(_e, dict) or str(_e.get("type", "")) != "combined":
            continue
        _cid = str(_e.get("id", "unknown"))
        _out = os.path.join(centering_audit_dir, f"centering_audit_{_cid}.jpg")
        _plot_centering_audit_entry(
            entry=_e,
            original_image=original_image,
            out_png=_out,
        )

    global_png = os.path.join(sup_root, "global_overview.jpg")
    t0 = time.perf_counter()
    _global_overview(final_entries, gt_mask, global_png)
    t_overview += float(time.perf_counter() - t0)
    try:
        _all_segs_global = []
        _all_normals_global = []
        for _e in (final_entries or []):
            if not isinstance(_e, dict):
                continue
            # Skip atomics represented by a combined entry to avoid duplicate/stub overlays.
            if str(_e.get("kind", "")) == "atomic" and str(_e.get("id", "")) in combined_flat:
                continue
            _mid = _e.get("midline_segments", None)
            if not isinstance(_mid, list) or not _mid:
                _mid_raw = _e.get("midline", None)
                _mid = _rebuild_segs(_mid_raw) if _mid_raw is not None else []
            for _s in (_mid or []):
                _sa = np.asarray(_s, float)
                if _sa.ndim == 2 and _sa.shape[1] == 2 and len(_sa) >= 2:
                    _all_segs_global.append(_sa)
            _gtn = _e.get("gt_normals", None)
            if isinstance(_gtn, dict):
                _all_normals_global.append(dict(_gtn))

        _gt_normals_global_png = os.path.join(sup_root, "analysis", "gt_normals_global.png")
        _run_timed_plot(
            plot_gt_normals_debug_global,
            out_path=_gt_normals_global_png,
            original_image=original_image,
            crack_mask_u8=np.asarray(gt_mask, np.uint8),
            gt_midline_segs=_all_segs_global,
            gt_normals=_all_normals_global,
            title=f"{base_name} - all GT normals (atomic + combined)",
        )
    except Exception as _e_gnorm:
        _dlog(2, f"[GT NORMALS GLOBAL] failed: {_e_gnorm}")

    # =====================================================
    # 4) WRITE JSON
    # =====================================================
    out_json = os.path.join(sup_root, "gt_supervision.json")

    def _json_default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    json_entries = []
    for _e in (final_entries or []):
        if isinstance(_e, dict):
            _d = dict(_e)
            # Keep runtime diagnostics in memory, but avoid bloating disk JSON.
            _d.pop("auto_centering_meta", None)
            _d.pop("method_cost_debug", None)
            json_entries.append(_d)
        else:
            json_entries.append(_e)
    t0 = time.perf_counter()
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"cracks": json_entries}, f, indent=2, default=_json_default)
    t_json += float(time.perf_counter() - t0)

    # =====================================================
    # 5) GT CENTERING METRICS (Supervision Audit)
    # =====================================================
    if enable_auto_centering:
        t0 = time.perf_counter()
        export_gt_centering_metrics(
            base_name=base_name,
            save_root=save_root,
            final_entries=final_entries,
            combined_member_ids=combined_flat,
            profile_accumulator=accP,
            batch_plots_only=BATCH_PLOTS_ONLY,
        )
        t_metrics_export += float(time.perf_counter() - t0)

    print(f"[GT_SUP] wrote JSON ? {out_json}")
    print(f"[GT_SUP] global overview ? {global_png}")

    # One-line timing summary (high signal, per image).
    t_total = float(time.perf_counter() - t0_total)
    t_atomic = float(timing_totals.get("atomic_compute_sec", 0.0) or 0.0)
    t_combined = float(timing_totals.get("combined_compute_sec", 0.0) or 0.0)
    t_atomic_center = float(timing_totals.get("atomic_centering_sec", 0.0) or 0.0)
    t_combined_center = float(timing_totals.get("combined_centering_sec", 0.0) or 0.0)
    t_multi_cue = float(
        (timing_totals.get("multi_cue_align_s", 0.0) or 0.0)
        + (timing_totals.get("depth_recess_s", 0.0) or 0.0)
        + (timing_totals.get("multi_cue_costmap_s", 0.0) or 0.0)
        + (timing_totals.get("multi_cue_dijkstra_s", 0.0) or 0.0)
        + (timing_totals.get("multi_cue_postprocess_s", 0.0) or 0.0)
    )
    P_mid = float(accP.get("mid", 0.0) or 0.0)
    P_rs = float(accP.get("rs", 0.0) or 0.0)
    P_met = float(accP.get("met", 0.0) or 0.0)
    P_n = int(accP.get("n", 0) or 0)
    t_P_total = float(P_mid + P_rs + P_met)
    print(
        f"[TIME] {base_name} | "
        f"A:{t_atomic:.2f}s "
        f"C:{t_combined:.2f}s "
        f"Ac:{t_atomic_center:.2f}s "
        f"Cc:{t_combined_center:.2f}s "
        f"MC:{t_multi_cue:.2f}s "
        f"P:{t_P_total:.2f}s "
        f"(MID:{P_mid:.2f}s RS:{P_rs:.2f}s MET:{P_met:.2f}s n:{P_n}) "
        f"Plt:{t_plot:.2f}s "
        f"Pv:{t_preview:.2f}s "
        f"Ov:{t_overview:.2f}s "
        f"J:{t_json:.2f}s "
        f"Mx:{t_metrics_export:.2f}s "
        f"T:{t_total:.2f}s"
    )






