import os, json
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

    print(f"[SUPERVISION] ✓ wrote → {out_json}")


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
    title = f"Ground truth crack normals — {entry['type']} {entry['id']}"

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

from helpers.metrics import normals_from_mask_for_midline
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

def snap_polyline_to_dt_ridge(
    mid_xy,
    domain_mask_u8,
    *,
    n_iters=25,
    step_px=0.35,
    grad_ksize=3,
    keep_endpoints=True,
    freeze_k=3,
    debug=False,
):
    """
    Nudge polyline points toward DT ridge by gradient ascent.
    Includes light Savitzky-Golay smoothing after convergence.
    """

    import numpy as np
    import cv2

    S = np.asarray(mid_xy, float).copy()

    if S.ndim != 2 or S.shape[1] != 2 or len(S) < 2:
        return S

    H, W = domain_mask_u8.shape[:2]
    domain = (np.asarray(domain_mask_u8) > 0).astype(np.uint8)

    if not np.any(domain):
        return S

    dt = cv2.distanceTransform(domain, cv2.DIST_L2, 5).astype(np.float32)
    gx = cv2.Sobel(dt, cv2.CV_32F, 1, 0, ksize=int(grad_ksize))
    gy = cv2.Sobel(dt, cv2.CV_32F, 0, 1, ksize=int(grad_ksize))

    def _bilinear(img, x, y):
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x0 = max(0, min(x0, W - 1))
        y0 = max(0, min(y0, H - 1))
        x1 = min(x0 + 1, W - 1)
        y1 = min(y0 + 1, H - 1)
        dx = float(x - x0)
        dy = float(y - y0)
        v00 = float(img[y0, x0]); v10 = float(img[y0, x1])
        v01 = float(img[y1, x0]); v11 = float(img[y1, x1])
        v0 = v00 * (1 - dx) + v10 * dx
        v1 = v01 * (1 - dx) + v11 * dx
        return v0 * (1 - dy) + v1 * dy

    def _allowed(x, y):
        xi = int(round(x))
        yi = int(round(y))
        if xi < 0 or xi >= W or yi < 0 or yi >= H:
            return False
        return bool(domain[yi, xi])

    freeze_k = int(max(0, freeze_k))
    idx_lo = freeze_k if keep_endpoints else 0
    idx_hi = len(S) - 1 - freeze_k if keep_endpoints else len(S) - 1

    if idx_hi <= idx_lo:
        return S

    for _ in range(int(max(1, n_iters))):
        moved = 0

        for i in range(idx_lo, idx_hi + 1):
            x = float(S[i, 0])
            y = float(S[i, 1])

            if not _allowed(x, y):
                continue

            # tangent
            if i == 0:
                t = S[1] - S[0]
            elif i == len(S) - 1:
                t = S[-1] - S[-2]
            else:
                t = S[i + 1] - S[i - 1]

            tn = float(np.hypot(t[0], t[1])) + 1e-12
            t = t / tn
            nx, ny = -t[1], t[0]

            gxi = _bilinear(gx, x, y)
            gyi = _bilinear(gy, x, y)
            g_proj = gxi * nx + gyi * ny

            if abs(g_proj) < 1e-12:
                continue

            sgn = 1.0 if g_proj >= 0.0 else -1.0
            ux, uy = sgn * nx, sgn * ny

            dt0 = _bilinear(dt, x, y)
            xn = x + float(step_px) * ux
            yn = y + float(step_px) * uy

            if not _allowed(xn, yn):
                continue

            dt1 = _bilinear(dt, xn, yn)

            if dt1 >= dt0 - 1e-6:
                S[i, 0] = xn
                S[i, 1] = yn
                moved += 1

        if moved == 0:
            break

    # ----------------------------------------------------
    # Light Savitzky-Golay smoothing (no resampling)
    # ----------------------------------------------------
    try:
        from scipy.signal import savgol_filter

        n = len(S)
        if n >= 5:
            window = 5
            if n >= 7:
                window = 7
            if window % 2 == 0:
                window += 1

            # preserve endpoints
            S_smooth = S.copy()
            xs = savgol_filter(S[:, 0], window, 2)
            ys = savgol_filter(S[:, 1], window, 2)

            S_smooth[:, 0] = xs
            S_smooth[:, 1] = ys

            if keep_endpoints:
                S_smooth[:freeze_k] = S[:freeze_k]
                S_smooth[-freeze_k:] = S[-freeze_k:]

            S = S_smooth

    except Exception:
        pass

    return S

def compute_centered_midline_and_normals(
    *,
    mid_xy,
    crack_mask_u8,
    territory_u8=None,
    max_radius=50,
    domain_mode="terr_and_mask",
    snap_kwargs=None,
):
    """
    Center a polyline in DT domain and compute normals/widths on GT crack mask.
    """
    if snap_kwargs is None:
        snap_kwargs = {}

    # Hard rule for centering: domain = crack & territory.
    territory = territory_u8
    if territory is None:
        territory = (np.asarray(crack_mask_u8) > 0).astype(np.uint8)

    domain_u8 = build_centering_domain_mask(
        crack_mask_u8=crack_mask_u8,
        territory_u8=territory,
        mode="terr_and_mask",
    )

    centered = snap_polyline_to_dt_ridge(
        np.asarray(mid_xy, float),
        domain_u8,
        **snap_kwargs,
    )

    (e1x, e1y, e2x, e2y, widths), _ = normals_from_mask_for_midline(
        centered,
        (np.asarray(crack_mask_u8) > 0),
        int(max_radius),
    )

    normals = {
        "edge1_x": _arr_to_list(e1x),
        "edge1_y": _arr_to_list(e1y),
        "edge2_x": _arr_to_list(e2x),
        "edge2_y": _arr_to_list(e2y),
        "width_px": _arr_to_list(widths),
    }
    return centered, normals

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
):
    """
    Plot manual vs centered midlines over crack mask (crop around bbox).
    """
    import matplotlib.pyplot as plt

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
    ax0.set_title(f"{title} (mask)", fontsize=10)

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
        ax1.set_title("DT ridge view (domain)", fontsize=10)

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
                    color="cyan",
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
        # (this helps diagnose “bbox exists but union empty”)
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

    fig.suptitle(f"GT dominance bite debug — base={base_name} ccid={ccid}", fontsize=11)

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
):
    """
    Compute midline metrics comparing manual GT midline vs auto-centered GT midline.

    Outputs:
      supervision/<image>/analysis/gt_centering_metrics.csv
      supervision/<image>/analysis/diagnostics/*.png
    """
    import os
    import numpy as np
    import pandas as pd
    from helpers.metrics import compute_midline_metrics
    from helpers.present_plots import plot_rs3_midline_diagnostics

    analysis_dir = os.path.join(save_root, "supervision", base_name, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    rows = []

    for entry in (final_entries or []):
        cid = str(entry.get("id", ""))
        kind = str(entry.get("kind", "unknown"))

        manual_mid = None
        centered_mid = None

        if kind == "atomic":
            manual_mid = np.asarray(entry.get("midline", []), float)
            centered_mid = np.asarray(entry.get("midline_auto_centered", []), float)

        elif kind == "combined":
            segs_manual = entry.get("midline_segments", []) or []
            segs_center = entry.get("midline_segments_auto_centered", []) or []

            manual_parts = [np.asarray(s, float) for s in segs_manual if s is not None and len(s) >= 2]
            center_parts = [np.asarray(s, float) for s in segs_center if s is not None and len(s) >= 2]

            if manual_parts:
                manual_mid = np.vstack(manual_parts)
            if center_parts:
                centered_mid = np.vstack(center_parts)

        if (
            manual_mid is None or centered_mid is None
            or len(manual_mid) < 2
            or len(centered_mid) < 2
        ):
            continue

        mm = compute_midline_metrics(
            auto_xy=centered_mid,
            man_xy=manual_mid,
            tau=3.0,
        )

        nn = float(mm.get("nn_mean_bidirectional", np.nan))
        hd = float(mm.get("hausdorff_max", np.nan))
        cov = float(mm.get("coverage_min", np.nan))
        score_mid = np.nan
        if np.isfinite(nn) and np.isfinite(hd) and np.isfinite(cov):
            score_mid = float(np.log1p(max(nn, 0.0)) + 0.5 * np.log1p(max(hd, 0.0)) + (1.0 - float(np.clip(cov, 0.0, 1.0))))

        row = {
            "image": base_name,
            "crack_id": cid,
            "crack_kind": kind,
            "geometry_type": "gt_centering",
            "length_px": float(np.sum(np.hypot(np.diff(manual_mid[:, 0]), np.diff(manual_mid[:, 1])))),
            "os_mode": "gt_centering",
            "g11": np.nan,
            "g22": np.nan,
            "g33": np.nan,
            "score_mid": score_mid,
            **mm,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(analysis_dir, "gt_centering_metrics.csv")
    df.to_csv(out_csv, index=False)
    print(f"[GT_SUP] wrote centering metrics -> {out_csv}")

    if not df.empty:
        plot_rs3_midline_diagnostics(
            df_all=df,
            out_dir=os.path.join(analysis_dir, "diagnostics"),
            selected_family=None,
            title_suffix="GT Centering Audit",
        )

    return df


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
    enable_auto_centering: bool = True,
    auto_centering_debug: bool = True,
    auto_centering_window_half_size: int = 50,
    auto_centering_iters: int = 30,
    auto_centering_step_px: float = 0.35,
    auto_centering_domain_atomic: str = "terr_and_mask",
    auto_centering_domain_combined: str = "terr_and_mask",
):
    sup_root = os.path.join(save_root, "supervision", base_name)
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

    # =====================================================
    # 1) ATOMIC BEFORE MERGE  (USE USER mask_bbox ONLY)
    # =====================================================
    for cid, cr in (atomic or {}).items():
        scid = str(cid)

        mid_xy = np.asarray(cr.get("midline", []), float)
        if mid_xy.ndim != 2 or len(mid_xy) < 2:
            continue

        # -------------------------------------------------
        # REQUIRED: user-authored mask_bbox (xywh)
        # -------------------------------------------------
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

        # Clamp ONLY for safety — semantics unchanged
        #x = max(0, x)
        #y = max(0, y)
        #w = min(w, W - x)
        #h = min(h, H - y)

        if w <= 0 or h <= 0:
            raise ValueError(
                f"[GT_SUP] atomic {scid} bbox collapses after clamp: {bb}"
            )

        # -------------------------------------------------
        # GT CC label ONLY for normals computation
        # -------------------------------------------------
        lbl = _cc_label_for_midline(mid_xy, cc_labels)
        if lbl is None or lbl <= 0:
            continue

        crack_mask = (cc_labels == lbl).astype(np.uint8)

        (e1x, e1y, e2x, e2y, widths), _ = normals_from_mask_for_midline(
            mid_xy,
            crack_mask > 0,
            50
        )

        atomic_entry = {
            "id": scid,
            "kind": "atomic",
            "members": [],
            # 🔴 STORE EXACTLY AS USER PROVIDED (xywh)
            "mask_bbox": [int(x), int(y), int(w), int(h)],
            "midline": mid_xy.tolist(),
            "gt_normals": {
                "edge1_x": _arr_to_list(e1x),
                "edge1_y": _arr_to_list(e1y),
                "edge2_x": _arr_to_list(e2x),
                "edge2_y": _arr_to_list(e2y),
                "width_px": _arr_to_list(widths),
            },
            "gt_widths": _arr_to_list(widths),
        }

        if enable_auto_centering:
            terr = build_territory_mask_from_polyline(
                mid_xy=mid_xy,
                crack_mask_u8=crack_mask,
                window_half_size=int(auto_centering_window_half_size),
                dt_domain_u8=None,
            )
            centered_xy, centered_normals = compute_centered_midline_and_normals(
                mid_xy=mid_xy,
                crack_mask_u8=crack_mask,
                territory_u8=terr,
                max_radius=50,
                domain_mode=auto_centering_domain_atomic,
                snap_kwargs={
                    "n_iters": int(auto_centering_iters),
                    "step_px": float(auto_centering_step_px),
                    "keep_endpoints": True,
                },
            )

            atomic_entry["midline_auto_centered"] = np.asarray(centered_xy, float).tolist()
            atomic_entry["gt_normals_auto_centered"] = centered_normals
            atomic_entry["gt_widths_auto_centered"] = centered_normals.get("width_px", [])
            atomic_entry["auto_centering_meta"] = {
                "enabled": True,
                "domain_mode": str(auto_centering_domain_atomic),
                "snap": {
                    "n_iters": int(auto_centering_iters),
                    "step_px": float(auto_centering_step_px),
                },
                **_shift_stats(mid_xy, centered_xy),
                **_geometry_disagreement_stats(mid_xy, centered_xy),
                **_width_stability_stats(widths, centered_normals.get("width_px", [])),
            }

            if auto_centering_debug:
                manual_invalid = [~np.isfinite(np.asarray(widths, float))]
                center_invalid = [~np.isfinite(np.asarray(centered_normals.get("width_px", []), float))]
                out_dbg = os.path.join(auto_center_root, f"atomic_{scid}_manual_vs_centered.png")
                plot_midline_centering_debug(
                    out_path=out_dbg,
                    crack_mask_u8=crack_mask,
                    manual_segs=[mid_xy],
                    centered_segs=[np.asarray(centered_xy, float)],
                    territory_u8=terr,
                    bbox_xywh=atomic_entry.get("mask_bbox"),
                    title=f"atomic {scid}: manual (yellow) vs centered (cyan)",
                    invalid_manual_masks=manual_invalid,
                    invalid_center_masks=center_invalid,
                )

        final_entries.append(atomic_entry)
        _cropped_preview(atomic_entry, gt_mask, original_image, atomic_crop_root)

    # =====================================================
    # 2) COMBINED  (USE UNION OF USER mask_bbox ONLY)
    # =====================================================
    for ccid, grp in (combined_groups or {}).items():
        members = [str(m) for m in grp.get("members", [])]
        if not members:
            continue

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
            continue

        crack_mask = (cc_labels == lbl).astype(np.uint8)

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
            continue
        
        # -------------------------------------------------
        # DEBUG: dominance bite as-written (RAW, no decode)
        # -------------------------------------------------
        try:
            debug_plot_gt_sup_dominance_bite_packed(
                base_name=base_name,
                ccid=ccid,
                members=members,
                dom_meta=dom_meta,
                segs=segs,
                gt_mask=gt_mask,
                out_dir=debug_dir,  # or sup_root, up to you
            )

        except Exception as e:
            print(f"[GT_SUP DOMDBG] plot failed: {e}")


        # -------------------------------------------------
        # Compute GT normals per segment
        # -------------------------------------------------
        e1x_list, e1y_list, e2x_list, e2y_list, w_list = [], [], [], [], []
        for S in segs:
            (e1x, e1y, e2x, e2y, widths), _ = normals_from_mask_for_midline(
                S, crack_mask > 0, 50
            )
            e1x_list.append(e1x)
            e1y_list.append(e1y)
            e2x_list.append(e2x)
            e2y_list.append(e2y)
            w_list.append(widths)

        packed_mid = _pack_segs_with_separators(segs)

        tag_name = "_".join(members)
        combined_entry = {
            "id": tag_name,
            "kind": "combined",
            "members": members,
            # 🔴 AUTHORITATIVE bbox = UNION OF USER BOXES (xywh)
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
            "dominance_meta": dom_meta,
        }

        if enable_auto_centering:
            centered_segs = []
            ce1x_list, ce1y_list, ce2x_list, ce2y_list, cw_list = [], [], [], [], []
            shift_all = []
            invalid_manual_masks = []
            invalid_center_masks = []

            for S, w_manual in zip(segs, w_list):
                S = np.asarray(S, float)
                terr_i = build_territory_mask_from_polyline(
                    mid_xy=S,
                    crack_mask_u8=crack_mask,
                    window_half_size=int(auto_centering_window_half_size),
                    dt_domain_u8=None,
                )
                centered_S, centered_normals = compute_centered_midline_and_normals(
                    mid_xy=S,
                    crack_mask_u8=crack_mask,
                    territory_u8=terr_i,
                    max_radius=50,
                    domain_mode="terr_and_mask",
                    snap_kwargs={
                        "n_iters": int(auto_centering_iters),
                        "step_px": float(auto_centering_step_px),
                        "keep_endpoints": True,
                    },
                )

                centered_S = np.asarray(centered_S, float)
                centered_segs.append(centered_S)
                ce1x_list.append(np.asarray(centered_normals.get("edge1_x", []), float))
                ce1y_list.append(np.asarray(centered_normals.get("edge1_y", []), float))
                ce2x_list.append(np.asarray(centered_normals.get("edge2_x", []), float))
                ce2y_list.append(np.asarray(centered_normals.get("edge2_y", []), float))
                cw_list.append(np.asarray(centered_normals.get("width_px", []), float))

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

            combined_entry["midline_segments_auto_centered"] = [np.asarray(S, float).tolist() for S in centered_segs]
            combined_entry["midline_auto_centered"] = _pack_segs_with_separators(centered_segs)
            combined_entry["gt_normals_auto_centered"] = {
                "edge1_x": _pack_arrs_with_none_separators([_arr_to_list(a) for a in ce1x_list]),
                "edge1_y": _pack_arrs_with_none_separators([_arr_to_list(a) for a in ce1y_list]),
                "edge2_x": _pack_arrs_with_none_separators([_arr_to_list(a) for a in ce2x_list]),
                "edge2_y": _pack_arrs_with_none_separators([_arr_to_list(a) for a in ce2y_list]),
                "width_px": _pack_arrs_with_none_separators([_arr_to_list(a) for a in cw_list]),
            }
            combined_entry["gt_widths_auto_centered"] = [
                float(v) for arr in cw_list for v in np.asarray(arr, float)
            ]
            manual_geom_parts = [np.asarray(S, float) for S in segs if S is not None and len(S) >= 2]
            center_geom_parts = [np.asarray(S, float) for S in centered_segs if S is not None and len(S) >= 2]
            manual_geom_all = np.vstack(manual_geom_parts) if manual_geom_parts else np.empty((0, 2), float)
            center_geom_all = np.vstack(center_geom_parts) if center_geom_parts else np.empty((0, 2), float)

            combined_entry["auto_centering_meta"] = {
                "enabled": True,
                "domain_mode": "terr_and_mask",
                "snap": {
                    "n_iters": int(auto_centering_iters),
                    "step_px": float(auto_centering_step_px),
                },
                **shift_meta,
                **_geometry_disagreement_stats(manual_geom_all, center_geom_all),
                **_width_stability_stats(
                    np.concatenate([np.asarray(a, float).reshape(-1) for a in w_list]) if w_list else [],
                    np.concatenate([np.asarray(a, float).reshape(-1) for a in cw_list]) if cw_list else [],
                ),
            }

            if auto_centering_debug:
                terr_vis = build_territory_mask_for_segments(
                    segs=segs,
                    crack_mask_u8=crack_mask,
                    window_half_size=int(auto_centering_window_half_size),
                )
                out_dbg = os.path.join(auto_center_root, f"combined_{tag_name}_manual_vs_centered.png")
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

        final_entries.append(combined_entry)
        _cropped_preview(combined_entry, gt_mask, original_image, combined_crop_root)

    # =====================================================
    # 3) GLOBAL OVERVIEW
    # =====================================================
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
        )

    print(f"[GT_SUP] wrote JSON → {out_json}")
    print(f"[GT_SUP] global overview → {global_png}")
