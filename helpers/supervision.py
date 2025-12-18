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

        tag = f"combined{ccid}_{'_'.join(entry['members'])}"
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
from combiner import _stitch_lines_by_user
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


def _bbox_from_coords(coords, H, W, pad=10):
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


# ============================================================
# CROPPED PREVIEW GENERATOR
# ============================================================
def _cropped_preview(entry, gt_mask_u8, original_image, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    H, W = gt_mask_u8.shape[:2]

    crack_id = entry["id"]
    kind = entry["kind"]

    # ---------
    # Midlines: prefer explicit segments if present (best)
    # ---------
    if entry.get("midline_segments"):
        mid_segs = [np.asarray(S, float) for S in entry["midline_segments"] if S is not None and len(S) >= 2]
    else:
        # atomic: midline is a plain list; combined: may be packed
        mid_raw = entry.get("midline", [])
        if len(mid_raw) and isinstance(mid_raw[0], (list, tuple)) and len(mid_raw[0]) == 2 and mid_raw[0][0] is None:
            mid_segs = _split_midline_packed(mid_raw)
        else:
            mid = np.asarray(mid_raw, float)
            mid_segs = [mid] if (mid.ndim == 2 and len(mid) >= 2) else []

    if not mid_segs:
        return

    # ---------
    # Normals: split None-separated lists into segments
    # ---------
    normals = entry.get("gt_normals") or {}
    e1_segs = _split_xy_none_seps(normals.get("edge1_x", []), normals.get("edge1_y", []))
    e2_segs = _split_xy_none_seps(normals.get("edge2_x", []), normals.get("edge2_y", []))

    # ---------
    # BBox from all coords (midlines + normals)
    # ---------
    coords = []
    for S in mid_segs:
        coords.append(S)
    for S in e1_segs:
        coords.append(S)
    for S in e2_segs:
        coords.append(S)
    coords = np.vstack(coords) if coords else np.empty((0, 2), float)

    bbox = _bbox_from_coords(coords, H, W, pad=10)
    if bbox is None:
        return
    x0, y0, x1, y1 = bbox

    # ---------
    # Overlay GT mask on grayscale original
    # ---------
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mask_f = (gt_mask_u8 > 0).astype(np.float32)
    overlay = np.clip(gray * 0.25 + mask_f * 0.75, 0, 1)
    overlay_rgb = (np.stack([overlay]*3, axis=-1) * 255).astype(np.uint8)

    crop_img = overlay_rgb[y0:y1+1, x0:x1+1]

    # shift into crop coords
    shift = np.array([x0, y0], float)
    mid_crop_segs = [S - shift for S in mid_segs]
    e1_crop_segs = [S - shift for S in e1_segs]
    e2_crop_segs = [S - shift for S in e2_segs]

    out_png = os.path.join(out_dir, f"{kind}_{crack_id}_crop.png")

    plot_edges_and_normals(
        base_image=crop_img,
        midline_segs=mid_crop_segs,
        edge1_segs=[],
        edge2_segs=[],
        norm1_segs=e1_crop_segs,
        norm2_segs=e2_crop_segs,
        sparsity=5,
        gt_plot=True,
        bbox=None,
        out_png=out_png,
        title=f"{kind} {crack_id}"
    )

# ============================================================
# GLOBAL OVERVIEW (simple color-coded)
# ============================================================
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

def _polyline_mask(S, H, W):
    m = np.zeros((H, W), np.uint8)
    pts = np.round(np.asarray(S, float)).astype(np.int32)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        return m
    pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
    cv2.polylines(m, [pts], False, 1, thickness=1, lineType=cv2.LINE_8)
    return m

def _finite_xy(arr):
    if arr is None or len(arr) == 0: return np.empty((0,2), float)
    a = np.asarray(arr, float)
    if a.ndim != 2 or a.shape[1] != 2: return np.empty((0,2), float)
    ok = np.isfinite(a).all(axis=1)
    a = a[ok]
    if len(a) <= 1: return a
    keep = [0]
    for i in range(1, len(a)):
        if not (abs(a[i,0]-a[i-1,0]) < 1e-9 and abs(a[i,1]-a[i-1,1]) < 1e-9):
            keep.append(i)
    return a[keep]

def _clip_polyline_to_mask(S, mask):
    """Keep only points that lie inside mask; then split into contiguous runs."""
    S = np.asarray(S, float)
    if len(S) < 2:
        return []
    H, W = mask.shape[:2]
    ys = np.clip(np.round(S[:, 1]).astype(int), 0, H - 1)
    xs = np.clip(np.round(S[:, 0]).astype(int), 0, W - 1)
    keep = mask[ys, xs].astype(bool)

    segs = []
    start = None
    for i, ok in enumerate(keep):
        if ok and start is None:
            start = i
        elif (not ok) and (start is not None):
            if i - start >= 2:
                segs.append(S[start:i])
            start = None
    if start is not None and (len(S) - start >= 2):
        segs.append(S[start:])

    # final cleanup
    segs = [_finite_xy(s) for s in segs]
    segs = [s for s in segs if len(s) >= 2]
    return segs

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

def _linestring_length(arr):
    try:
        a = np.asarray(arr, float)
        if a.ndim != 2 or a.shape[1] != 2 or len(a) < 2:
            return 0.0
        d = np.diff(a, axis=0)
        return float(np.sqrt((d * d).sum(axis=1)).sum())
    except Exception:
        return 0.0
    
def _endpoints_key(S, eps=2.0):
    """Quantize endpoints so 'almost same' endpoints match."""
    S = np.asarray(S, float)
    a = S[0]; b = S[-1]
    qa = (int(round(a[0] / eps)), int(round(a[1] / eps)))
    qb = (int(round(b[0] / eps)), int(round(b[1] / eps)))
    return qa, qb

def _endpoint_components(segs, eps=2.0):
    """
    Build components where segments are connected if they share an endpoint
    (within eps quantization).
    Returns list of lists of indices.
    """
    # map endpoint -> seg indices
    end_map = {}
    ends = []
    for i, S in enumerate(segs):
        ea, eb = _endpoints_key(S, eps=eps)
        ends.append((ea, eb))
        end_map.setdefault(ea, []).append(i)
        end_map.setdefault(eb, []).append(i)

    # adjacency
    adj = {i: set() for i in range(len(segs))}
    for _, idxs in end_map.items():
        if len(idxs) < 2:
            continue
        for a in idxs:
            for b in idxs:
                if a != b:
                    adj[a].add(b)

    # components
    comps = []
    seen = set()
    for i in range(len(segs)):
        if i in seen:
            continue
        stack = [i]
        comp = []
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.append(u)
            stack.extend(list(adj[u]))
        comps.append(comp)

    return comps

def dominant_segments_from_group(
    *,
    members,
    atomic,
    crack_mask_u8,
    window_half_size,
    debug_dir=None,
    debug_tag="group",
):
    """
    FINAL GT-correct dominance logic.

    Rules:
    - Branches are defined STRICTLY by shared USER endpoints
    - ALL atomics within a branch are preserved
    - Dominance applies ONLY BETWEEN branches
    - Dominance trims TERRITORY, not atomic identity
    """

    import os
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt

    H, W = crack_mask_u8.shape[:2]
    crack_mask = (crack_mask_u8 > 0).astype(np.uint8)

    # ------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------
    def get_user_endpoints(cr):
        ups = cr.get("user_points", []) or []
        ucs = cr.get("user_connections", []) or []
        out = set()
        for pair in ucs:
            for idx in pair:
                if 0 <= idx < len(ups):
                    out.add(tuple(map(float, ups[idx])))
        return out

    # ------------------------------------------------------------
    # 1) collect atomics + endpoints
    # ------------------------------------------------------------
    atomics = []
    endpoints = []

    for m in members:
        cr = atomic.get(str(m), {}) or {}
        ml = np.asarray(cr.get("midline", []), float)
        if ml.ndim == 2 and len(ml) >= 2:
            atomics.append((str(m), _finite_xy(ml)))
            endpoints.append(get_user_endpoints(cr))

    print(f"[DOM] members={members}")
    print(f"[DOM] atomics={len(atomics)}")

    if not atomics:
        return [], []

    # ------------------------------------------------------------
    # 2) build BRANCHES in atomic space
    # ------------------------------------------------------------
    N = len(atomics)
    adj = {i: set() for i in range(N)}

    for i in range(N):
        for j in range(i + 1, N):
            if endpoints[i] & endpoints[j]:
                adj[i].add(j)
                adj[j].add(i)

    branches = []
    seen = set()
    for i in range(N):
        if i in seen:
            continue
        stack = [i]
        comp = []
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.append(u)
            stack.extend(adj[u])
        branches.append(comp)

    print(f"[DOM] branches={len(branches)}")
    for bi, b in enumerate(branches):
        print(f"  [DOM] branch {bi}: atomics={[atomics[i][0] for i in b]}")

    # ------------------------------------------------------------
    # 3) Build BRANCHES with TOTAL USER-SPACE LENGTH
    #    (USER geometry is authoritative; clipping is territory-only)
    # ------------------------------------------------------------
    branch_user_len = []        # total USER length per branch
    branch_user_segs = []       # USER midlines per branch (never clipped)
    branch_clipped_segs = []    # clipped geometry per branch (territory only)

    for bi, atom_ids in enumerate(branches):
        total_len = 0.0
        user_segs = []
        clipped_segs = []

        for ai in atom_ids:
            _, S_user = atomics[ai]   # USER midline
            total_len += _linestring_length(S_user)
            user_segs.append(S_user)

            # clipping ONLY for territory computation
            pieces = _clip_polyline_to_mask(S_user, crack_mask)
            clipped_segs.extend([p for p in pieces if len(p) >= 2])

        branch_user_len.append(total_len)
        branch_user_segs.append(user_segs)
        branch_clipped_segs.append(clipped_segs)

        print(
            f"[DOM] branch {bi}: atomics={atom_ids} "
            f"user_len={total_len:.1f} "
            f"clipped_segs={len(clipped_segs)}"
        )

    # ------------------------------------------------------------
    # 4) Dominance BETWEEN branches (ordered by USER length)
    # ------------------------------------------------------------
    dt = cv2.distanceTransform(crack_mask, cv2.DIST_L2, 5)

    def seg_radius(S):
        ys = np.clip(np.round(S[:, 1]).astype(int), 0, H - 1)
        xs = np.clip(np.round(S[:, 0]).astype(int), 0, W - 1)
        d = dt[ys, xs]
        d = d[np.isfinite(d)]
        if len(d) == 0:
            return 0.3 * window_half_size
        return max(3.0, min(float(np.median(d)), window_half_size))

    # dominance order: PURE USER length
    order = sorted(
        range(len(branches)),
        key=lambda i: branch_user_len[i],
        reverse=True,
    )

    claimed = np.zeros((H, W), np.uint8)
    kept = []   # FINAL midlines (USER space, never fragmented)

    for rank, bi in enumerate(order):
        print(
            f"[DOM] processing branch {bi} "
            f"(rank={rank}, user_len={branch_user_len[bi]:.1f})"
        )

        # --------------------------------------------------------
        # Build TERRITORY from clipped geometry ONLY
        # --------------------------------------------------------
        branch_terr = np.zeros((H, W), np.uint8)

        for S_clip in branch_clipped_segs[bi]:
            r = seg_radius(S_clip)
            rad = int(max(3, 0.8 * r))

            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1)
            )

            line = _polyline_mask(S_clip, H, W)
            terr = cv2.dilate(line, kernel, iterations=1) & crack_mask
            branch_terr |= terr

        unique = branch_terr & (~claimed)

        # --------------------------------------------------------
        # Dominance decision
        # --------------------------------------------------------
        if rank > 0 and unique.sum() < max(10, 0.5 * window_half_size):
            print(
                f"[DOM] suppress branch {bi} "
                f"(unique={int(unique.sum())})"
            )
            continue

        # --------------------------------------------------------
        # KEEP: append USER midlines (never clipped)
        # --------------------------------------------------------
        if rank == 0:
            # PRIMARY branch: keep full USER geometry
            for S_user in branch_user_segs[bi]:
                kept.append(S_user)
            claimed |= branch_terr
            print(f"[DOM] keep PRIMARY branch {bi}")
            continue

        # --------------------------------------------------------
        # SUBORDINATE branch: clip USER midlines to remaining space
        # --------------------------------------------------------
        remaining = branch_terr & (~claimed)

        kept_any = False
        for S_user in branch_user_segs[bi]:
            # Clip USER midline against remaining territory
            pieces = _clip_polyline_to_mask(S_user, remaining)
            for p in pieces:
                if len(p) >= 2:
                    kept.append(p)
                    kept_any = True

        if kept_any:
            claimed |= remaining
            print(f"[DOM] keep SUBORDINATE branch {bi} (clipped)")
        else:
            print(f"[DOM] suppress branch {bi} (no remaining geometry)")

    print(f"[DOM] FINAL kept USER midlines={len(kept)}")

    # ------------------------------------------------------------
    # 5) debug visualization
    # ------------------------------------------------------------
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        coords = np.vstack([s for s in kept if len(s) >= 2])
        bbox = _bbox_from_coords(coords, H, W, pad=20)
        if bbox:
            x0, y0, x1, y1 = bbox
            fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
            ax.imshow(crack_mask[y0:y1, x0:x1], cmap="gray")
            for S in kept:
                S2 = S - np.array([x0, y0])
                ax.plot(S2[:, 0], S2[:, 1], lw=2)
            ax.set_title(debug_tag)
            ax.axis("off")
            fig.savefig(os.path.join(debug_dir, f"{debug_tag}_final.png"))
            plt.close(fig)

    return kept, kept

def _plot_dominance_debug(*, debug_dir, tag, crack_mask, candidates, kept, claimed):
    """
    Cropped dominance debug plots:
      - candidates
      - kept
      - claimed territory

    Cropping is based on all candidate + kept polyline coordinates.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    H, W = crack_mask.shape[:2]

    # -----------------------------
    # Collect coords for bbox
    # -----------------------------
    coords = []
    for segs in (candidates, kept):
        for S in segs:
            S = np.asarray(S, float)
            if S.ndim == 2 and len(S) >= 2:
                coords.append(S)

    if not coords:
        return

    coords = np.vstack(coords)
    coords = coords[np.isfinite(coords).all(axis=1)]
    if coords.size == 0:
        return

    pad = 10
    xs, ys = coords[:, 0], coords[:, 1]
    x0 = max(0, int(np.floor(xs.min() - pad)))
    x1 = min(W - 1, int(np.ceil(xs.max() + pad)))
    y0 = max(0, int(np.floor(ys.min() - pad)))
    y1 = min(H - 1, int(np.ceil(ys.max() + pad)))

    if x1 <= x0 or y1 <= y0:
        return

    # -----------------------------
    # Crop images
    # -----------------------------
    mask_crop = crack_mask[y0:y1+1, x0:x1+1]
    claimed_crop = claimed[y0:y1+1, x0:x1+1]

    def draw_lines(ax, segs, color="yellow"):
        for S in segs:
            S = np.asarray(S, float)
            if len(S) >= 2:
                ax.plot(S[:, 0] - x0, S[:, 1] - y0, color=color, lw=1.5)

    os.makedirs(debug_dir, exist_ok=True)

    # -----------------------------
    # Candidates
    # -----------------------------
    fig, ax = plt.subplots()
    ax.imshow(mask_crop, cmap="gray")
    draw_lines(ax, candidates, color="cyan")
    ax.set_title(f"{tag}: candidates")
    ax.axis("off")
    fig.savefig(
        os.path.join(debug_dir, f"{tag}_candidates.png"),
        dpi=200, bbox_inches="tight"
    )
    plt.close(fig)

    # -----------------------------
    # Kept
    # -----------------------------
    fig, ax = plt.subplots()
    ax.imshow(mask_crop, cmap="gray")
    draw_lines(ax, kept, color="lime")
    ax.set_title(f"{tag}: kept")
    ax.axis("off")
    fig.savefig(
        os.path.join(debug_dir, f"{tag}_kept.png"),
        dpi=200, bbox_inches="tight"
    )
    plt.close(fig)

    # -----------------------------
    # Claimed territory
    # -----------------------------
    fig, ax = plt.subplots()
    ax.imshow(mask_crop, cmap="gray")
    ax.imshow(claimed_crop, alpha=0.35)
    draw_lines(ax, kept, color="lime")
    ax.set_title(f"{tag}: claimed territory")
    ax.axis("off")
    fig.savefig(
        os.path.join(debug_dir, f"{tag}_claimed.png"),
        dpi=200, bbox_inches="tight"
    )
    plt.close(fig)

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
):
    sup_root = os.path.join(save_root, "supervision", base_name)
    #mask_root = os.path.join(sup_root, "masks")
    atomic_crop_root = os.path.join(sup_root, "atomic_crops")
    combined_crop_root = os.path.join(sup_root, "combined_crops")
    #os.makedirs(mask_root, exist_ok=True)
    os.makedirs(atomic_crop_root, exist_ok=True)
    os.makedirs(combined_crop_root, exist_ok=True)

    gt_bin = (gt_mask > 0).astype(np.uint8)
    num_cc, cc_labels = cv2.connectedComponents(gt_bin, 8)
    print(f"[GT_SUP] GT connected components: {num_cc-1}")

    combined_groups = combined_groups or {}
    combined_flat = {str(m) for g in combined_groups.values() for m in g.get("members", [])}

    final_entries = []

    # =====================================================
    # 1) ATOMIC BEFORE MERGE
    # =====================================================
    for cid, cr in (atomic or {}).items():
        scid = str(cid)

        # ALWAYS export atomic preview (before merge)
        mid_xy = np.asarray(cr.get("midline", []), float)
        if len(mid_xy) < 2:
            continue

        lbl = _cc_label_for_midline(mid_xy, cc_labels)
        if lbl is None or lbl <= 0:
            continue

        crack_mask = (cc_labels == lbl).astype(np.uint8)
        ys, xs = np.where(crack_mask > 0)
        if xs.size == 0:
            continue

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        (e1x, e1y, e2x, e2y, widths), _ = normals_from_mask_for_midline(mid_xy, crack_mask > 0, 50)

        atomic_entry = {
            "id": scid,
            "kind": "atomic",
            "members": [],
            "mask_bbox": [int(x0), int(y0), int(x1), int(y1)],
            "midline": mid_xy.tolist(),
            "gt_normals": {
                "edge1_x": _arr_to_list(e1x),
                "edge1_y": _arr_to_list(e1y),
                "edge2_x": _arr_to_list(e2x),
                "edge2_y": _arr_to_list(e2y),
                "width_px": _arr_to_list(widths),
            },
        }

        final_entries.append(atomic_entry)

        # Crop preview
        _cropped_preview(atomic_entry, gt_mask, original_image, atomic_crop_root)

    # =====================================================
    # 2) COMBINED
    # =====================================================
    for ccid, grp in (combined_groups or {}).items():
        members = [str(m) for m in grp.get("members", [])]
        if not members:
            continue

        '''stitched = _stitch_lines_by_user(members, atomic)
        if stitched:
            mid_xy = max(stitched, key=lambda arr: arr.shape[0])
        else:
            all_mid = []
            for m in members:
                ml = atomic.get(m, {}).get("midline", [])
                if len(ml) >= 2:
                    all_mid.append(np.asarray(ml, float))
            if not all_mid:
                continue
            mid_xy = np.vstack(all_mid)

        lbl = _cc_label_for_midline(mid_xy, cc_labels)
        if lbl is None or lbl <= 0:
            continue

        crack_mask = (cc_labels == lbl).astype(np.uint8)
        ys, xs = np.where(crack_mask > 0)
        if xs.size == 0:
            continue

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        (e1x, e1y, e2x, e2y, widths), _ = normals_from_mask_for_midline(mid_xy, crack_mask > 0, 50)

        # Proper combined naming
        tag_name = f"combined_{'_'.join(members)}"

        combined_entry = {
            "id": tag_name,
            "kind": "combined",
            "members": members,
            "mask_bbox": [int(x0), int(y0), int(x1), int(y1)],
            "midline": mid_xy.tolist(),
            "gt_normals": {
                "edge1_x": _arr_to_list(e1x),
                "edge1_y": _arr_to_list(e1y),
                "edge2_x": _arr_to_list(e2x),
                "edge2_y": _arr_to_list(e2y),
                "width_px": _arr_to_list(widths),
            },
        }

        final_entries.append(combined_entry)

        # Crop preview
        _cropped_preview(combined_entry, gt_mask, original_image, combined_crop_root)'''
        
        # robust CC label for the whole group (don’t depend on a single midline)
        lbl = _cc_label_for_members(members, atomic, cc_labels)
        if lbl is None or lbl <= 0:
            continue

        crack_mask = (cc_labels == lbl).astype(np.uint8)
        ys, xs = np.where(crack_mask > 0)
        if xs.size == 0:
            continue

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        # dominance-selected sub-midlines for this ONE crack
        debug_dir = os.path.join(sup_root, "combined_debug")
        tag = f"ccid{ccid}_" + "_".join(members)

        segs, _cands = dominant_segments_from_group(
            members=members,
            atomic=atomic,
            crack_mask_u8=crack_mask,
            window_half_size=50,
            debug_dir=debug_dir,   # comment out if you don’t want plots
            debug_tag=tag,
        )

        if not segs:
            continue

        # compute normals per segment, then pack with separators so JSON stays simple
        e1x_list, e1y_list, e2x_list, e2y_list, w_list = [], [], [], [], []
        for S in segs:
            (e1x, e1y, e2x, e2y, widths), _ = normals_from_mask_for_midline(S, crack_mask > 0, 50)
            e1x_list.append(e1x); e1y_list.append(e1y)
            e2x_list.append(e2x); e2y_list.append(e2y)
            w_list.append(widths)

        packed_mid = _pack_segs_with_separators(segs)

        tag_name = f"combined_{'_'.join(members)}"
        combined_entry = {
            "id": tag_name,
            "kind": "combined",
            "members": members,
            "mask_bbox": [int(x0), int(y0), int(x1), int(y1)],
            "midline": packed_mid,  # now supports multiple sub-midlines cleanly
            "gt_normals": {
                "edge1_x": _pack_arrs_with_none_separators([_arr_to_list(a) for a in e1x_list]),
                "edge1_y": _pack_arrs_with_none_separators([_arr_to_list(a) for a in e1y_list]),
                "edge2_x": _pack_arrs_with_none_separators([_arr_to_list(a) for a in e2x_list]),
                "edge2_y": _pack_arrs_with_none_separators([_arr_to_list(a) for a in e2y_list]),
                "width_px": _pack_arrs_with_none_separators([_arr_to_list(a) for a in w_list]),
            },
            # optional: explicit segments for easier downstream consumption
            "midline_segments": [np.asarray(S, float).tolist() for S in segs],
        }

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

    print(f"[GT_SUP] wrote JSON → {out_json}")
    print(f"[GT_SUP] global overview → {global_png}")
