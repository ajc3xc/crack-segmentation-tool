# helpers/combiner.py
# Stateless combined-crack builder for metrics pipeline
from __future__ import annotations
import os, math
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "figure.max_open_warning": 0,
    "path.simplify": True,
    "path.simplify_threshold": 1.0,
})

# --- imports that exist in your tree; keep both fallbacks ---
from cracktools.segmentation import edge_masks, edges_tracking


from helpers.metrics import bbox_from_mask
from helpers.geometry_canonical import (
    orient_segment_to_reference,
    enforce_branch_continuity,
    canonicalize_branch_direction,
    assert_direction_consistency,
)
from helpers.branch_stitching import stitch_branch_segments as shared_stitch_branch_segments

from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union
from helpers.plot_metrics import *

# Set to False to suppress verbose per-image debug prints during batch runs
_VERBOSE_DEBUG = True

COMBINER_DEBUG_LEVEL = 1


def _c_log(level, msg):
    if _VERBOSE_DEBUG and COMBINER_DEBUG_LEVEL >= int(level):
        print(msg)


def _dbg_pts(tag, pts, bbox=None, level=1):
    try:
        a = np.asarray(pts, float)
        if a.ndim != 2 or a.shape[1] != 2 or len(a) == 0:
            _c_log(level, f"[DBG_PTS] {tag}: invalid shape={getattr(a, 'shape', None)}")
            return
        mn = np.min(a, axis=0)
        mx = np.max(a, axis=0)
        _c_log(level, f"[DBG_PTS] {tag}: min={mn.tolist()}, max={mx.tolist()}, n={len(a)}")
        if bbox is not None and len(bbox) == 4:
            x, y, w, h = [int(v) for v in bbox]
            _c_log(level, f"[DBG_PTS] {tag}: bbox=({x},{y},{w},{h})")
    except Exception as e:
        _c_log(level, f"[DBG_PTS] {tag}: failed ({e})")


def _dbg_frame_mismatch(pts, bbox, level=1):
    try:
        a = np.asarray(pts, float)
        if a.ndim != 2 or a.shape[1] != 2 or len(a) == 0:
            _c_log(level, "[FRAME_CHECK] invalid pts")
            return
        x0, y0, w, h = [float(v) for v in bbox]
        global_like = float(np.mean((a[:, 0] > w) | (a[:, 1] > h)))
        local_like = float(np.mean(
            (a[:, 0] >= 0) & (a[:, 0] <= w) &
            (a[:, 1] >= 0) & (a[:, 1] <= h)
        ))
        _c_log(level, f"[FRAME_CHECK] global_like={global_like:.2f}, local_like={local_like:.2f}")
    except Exception as e:
        _c_log(level, f"[FRAME_CHECK] failed ({e})")


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

def plot_greedy_branch_debug(
    *,
    atomics,
    branches,
    attach_orders,
    endpoint_graph_deg,
    connected_comp_size,
    group_id,
    debug_dir,
    original_image=None,
):
    """
    Optional 2-panel debug plot for greedy branch construction.
    """
    if debug_dir is None:
        return

    has_junction = any(int(d) >= 3 for d in (endpoint_graph_deg or {}).values())
    if not has_junction:
        return

    if int(connected_comp_size or 0) < 4:
        return

    if len(branches or []) < 2:
        return

    import os
    import matplotlib.pyplot as plt

    os.makedirs(debug_dir, exist_ok=True)
    safe_gid = str(group_id).replace(os.sep, "_").replace("/", "_")
    out_path = os.path.join(debug_dir, f"greedy_branches_group_{safe_gid}.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=200)

    # Panel A: input topology
    ax_a = axes[0]
    for a in atomics:
        S = np.asarray(a.get("poly", []), float)
        if S.ndim == 2 and len(S) >= 2:
            ax_a.plot(S[:, 0], S[:, 1], color="lightgray", linewidth=1.0)
            # Label with atomic_id in black at midpoint
            mid = S[len(S) // 2]
            ax_a.text(
                float(mid[0]), float(mid[1]),
                str(a.get("atomic_id", "?")),
                color="black", fontsize=6, ha="center", va="center", zorder=20,
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.75),
            )

    junction_pts = [k for k, d in (endpoint_graph_deg or {}).items() if int(d) >= 3]
    if junction_pts:
        jp = np.asarray(junction_pts, float)
        if jp.ndim == 2 and jp.shape[1] == 2:
            ax_a.scatter(jp[:, 0], jp[:, 1], c="red", s=12, zorder=30)

    ax_a.set_title("A - Atomic Crack Topology")
    ax_a.set_aspect("equal", adjustable="box")

    # Panel B: greedy result
    ax_b = axes[1]
    cmap = plt.get_cmap("tab10")
    for b_idx, branch in enumerate(branches):
        color = cmap(int(b_idx % 10))
        for seg_position_in_branch, atomic_idx in enumerate(branch):
            S = np.asarray(atomics[atomic_idx]["poly"], float)
            if S.ndim != 2 or len(S) < 2:
                continue
            ax_b.plot(S[:, 0], S[:, 1], color=color, linewidth=2.8)
            mid = S[len(S) // 2]
            order_label = int(seg_position_in_branch) + 1
            # Offset text slightly off the centerline so digits don't sit on top of the branch.
            if len(S) >= 3:
                i = len(S) // 2
                v = S[min(i + 1, len(S) - 1)] - S[max(i - 1, 0)]
            else:
                v = S[-1] - S[0]
            n = np.array([-v[1], v[0]], float)
            nn = float(np.linalg.norm(n))
            if nn > 1e-9:
                n /= nn
            else:
                n[:] = 0.0
            off = 4.0 * n
            ax_b.text(
                float(mid[0] + off[0]),
                float(mid[1] + off[1]),
                str(order_label),
                color="black",
                fontsize=7,
                ha="center",
                va="center",
                zorder=50,
                bbox=dict(
                    boxstyle="round,pad=0.1",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.65,
                ),
            )

    ax_b.set_title("B - Greedy Branch Construction (numbers = attachment order)")
    ax_b.set_aspect("equal", adjustable="box")

    # Force image-style coordinates (y grows downward): low y at top, high y at bottom.
    ya0, ya1 = ax_a.get_ylim()
    if ya0 < ya1:
        ax_a.set_ylim(ya1, ya0)
    yb0, yb1 = ax_b.get_ylim()
    if yb0 < yb1:
        ax_b.set_ylim(yb1, yb0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close(fig)

    # -------------------------------------------------------
    # Panel C: image-overlay stitch debug (saved separately)
    # Shows each branch coloured on the real image, with stitch
    # join points coloured by join distance (green=ok, red=gap)
    # and junction nodes marked as stars.
    # -------------------------------------------------------
    if original_image is None:
        return

    try:
        img_rgb = np.asarray(original_image)
        if img_rgb.ndim == 3 and img_rgb.shape[2] == 3:
            # assume BGR from OpenCV
            img_rgb = img_rgb[:, :, ::-1].copy()
        elif img_rgb.ndim != 3:
            return

        H_img, W_img = img_rgb.shape[:2]

        # Compute crop around all atomics + padding
        all_pts = []
        for a in atomics:
            S = np.asarray(a.get("poly", []), float)
            if S.ndim == 2 and len(S) >= 2:
                all_pts.append(S)
        if not all_pts:
            return
        all_xy = np.vstack(all_pts)
        pad_px = 40
        x0 = max(0, int(all_xy[:, 0].min()) - pad_px)
        y0 = max(0, int(all_xy[:, 1].min()) - pad_px)
        x1 = min(W_img, int(all_xy[:, 0].max()) + pad_px + 1)
        y1 = min(H_img, int(all_xy[:, 1].max()) + pad_px + 1)
        if x1 <= x0 or y1 <= y0:
            return
        img_crop = img_rgb[y0:y1, x0:x1]

        fig2, ax_c = plt.subplots(1, 1, figsize=(max(10.0, (x1-x0)/80.0), max(8.0, (y1-y0)/80.0)), dpi=150)
        ax_c.imshow(img_crop)

        cmap2 = plt.get_cmap("tab10")
        junction_keys = {k for k, d in (endpoint_graph_deg or {}).items() if int(d) >= 3}
        TELEPORT_THRESH = 10.0

        for b_idx, branch in enumerate(branches):
            color = cmap2(int(b_idx % 10))
            aid_labels = [atomics[i]["atomic_id"] for i in branch]

            # Draw each atomic midline in branch colour
            for atomic_idx in branch:
                S = np.asarray(atomics[atomic_idx]["poly"], float)
                if S.ndim != 2 or len(S) < 2:
                    continue
                ax_c.plot(S[:, 0] - x0, S[:, 1] - y0, color=color, linewidth=2.0, alpha=0.85, zorder=10)

                # Mark endpoints
                for ep in [S[0], S[-1]]:
                    ax_c.plot(ep[0] - x0, ep[1] - y0, "o", color=color, markersize=5, zorder=15)

            # Greedy stitch simulation: show join distances between consecutive atomics
            # (mirror the stitch algorithm to find actual join points and distances)
            if len(branch) >= 2:
                segs = []
                for atomic_idx in branch:
                    S = np.asarray(atomics[atomic_idx]["poly"], float)
                    if S.ndim == 2 and len(S) >= 2:
                        segs.append(S)

                if len(segs) >= 2:
                    # Replicate nearest-neighbour chain to find join points
                    used = set()
                    chain = [segs[0]]
                    used.add(0)
                    while len(used) < len(segs):
                        cur_end = chain[-1][-1]
                        cur_start = chain[0][0]
                        best_j, best_dist, best_oriented, best_action = None, float("inf"), None, None
                        for j, sj in enumerate(segs):
                            if j in used:
                                continue
                            a_, b_ = sj[0], sj[-1]
                            for dist, oriented, flipped, action in [
                                (np.linalg.norm(cur_end - a_), sj, False, "append"),
                                (np.linalg.norm(cur_end - b_), sj[::-1], True, "append"),
                                (np.linalg.norm(b_ - cur_start), sj, False, "prepend"),
                                (np.linalg.norm(a_ - cur_start), sj[::-1], True, "prepend"),
                            ]:
                                if dist < best_dist:
                                    best_dist, best_j, best_oriented, best_action = dist, j, oriented, action
                        if best_j is None:
                            break
                        # Draw the join gap line
                        if best_action == "append":
                            p1 = chain[-1][-1]
                            p2 = best_oriented[0]
                        else:
                            p1 = best_oriented[-1]
                            p2 = chain[0][0]
                        join_color = "lime" if best_dist <= TELEPORT_THRESH else "red"
                        ax_c.plot(
                            [p1[0] - x0, p2[0] - x0],
                            [p1[1] - y0, p2[1] - y0],
                            color=join_color, linewidth=2.5, linestyle="--", zorder=20,
                        )
                        if best_action == "append":
                            chain.append(best_oriented)
                        else:
                            chain.insert(0, best_oriented)
                        used.add(best_j)

            # Junction nodes (degree >= 3) as yellow stars
        for k in junction_keys:
            ax_c.plot(k[0] - x0, k[1] - y0, "*", color="yellow", markersize=12, zorder=35,
                      markeredgecolor="black", markeredgewidth=0.5)

        ax_c.set_title(
            f"C - ET Stitch Debug  |  group={group_id}\n"
            f"green dashes=clean join (≤{TELEPORT_THRESH}px)  red dashes=teleport gap  ★=junction node",
            fontsize=7,
        )
        ax_c.axis("off")

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color="lime", linestyle="--", linewidth=2, label=f"clean join ≤{TELEPORT_THRESH}px"),
            Line2D([0], [0], color="red", linestyle="--", linewidth=2, label="teleport gap"),
            Line2D([0], [0], marker="*", color="yellow", markersize=8, linestyle="none", label="junction node (deg≥3)"),
        ] + [
            Line2D([0], [0], color=cmap2(i % 10), linewidth=2,
                   label=f"b{i}: {[atomics[j]['atomic_id'] for j in br]}")
            for i, br in enumerate(branches)
        ]
        ax_c.legend(handles=legend_elements, fontsize=5, loc="lower right", framealpha=0.7)

        plt.tight_layout()
        out_path_c = os.path.join(debug_dir, f"et_stitch_debug_group_{safe_gid}.png")
        fig2.savefig(out_path_c, bbox_inches="tight")
        plt.close(fig2)
        print(f"[ET STITCH DBG] wrote: {out_path_c}")

    except Exception as _e:
        print(f"[ET STITCH DBG] failed: {_e}")
        import traceback; traceback.print_exc()


'''def _split_lines(geom):
    if geom.is_empty: return []
    if isinstance(geom, LineString): return [geom]
    if isinstance(geom, MultiLineString): return list(geom.geoms)
    return []


def _linestring_length(arr):
    try:
        return float(LineString(arr).length)
    except Exception:
        return 0.0


def _shoelace_area(xs, ys):
    return 0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))


def _split_on_teleports(arr, max_step=50.0):
    arr = np.asarray(arr, float)
    if len(arr) < 2: return []
    d = np.sqrt(np.sum(np.diff(arr, axis=0)**2, axis=1))
    breaks = np.where(d > max_step)[0]
    segs, start = [], 0
    for b in breaks:
        if b+1 - start >= 2:
            segs.append(arr[start:b+1])
        start = b+1
    if len(arr) - start >= 2:
        segs.append(arr[start:])
    return segs if segs else [arr]


def _endpoints_from_crack(crack):
    ups = crack.get("user_points", []) or []
    ucs = crack.get("user_connections", []) or []
    ends = set()
    for conn in ucs:
        for idx in conn:
            if 0 <= idx < len(ups):
                pt = ups[idx]
                ends.add((float(pt[0]), float(pt[1])))
    return ends


def _stitch_lines_by_user(member_ids, atomic):
    """Greedy, endpoint-aware stitching identical in spirit to your class version."""
    mid2arr, mid2ends = {}, {}
    for mid in member_ids:
        crack = atomic.get(mid)
        #print()
        #print(atomic.keys(), crack.keys(), crack['source'])
        if not crack: continue
        midl = crack.get("midline", []) or []
        if len(midl) < 2: continue
        arr = np.array([[float(x), float(y)] for (x,y) in midl], dtype=float)
        arr = _finite_xy(arr)
        if len(arr) >= 2:
            mid2arr[mid] = arr
            mid2ends[mid] = _endpoints_from_crack(crack)
    if not mid2arr: return []

    # build adjacency by shared endpoints
    end_to_mids = {}
    for mid, ends in mid2ends.items():
        for e in ends:
            end_to_mids.setdefault(e, set()).add(mid)

    adj = {mid: set() for mid in mid2arr}
    for mids in end_to_mids.values():
        mids = list(mids)
        for i in range(len(mids)):
            for j in range(i+1, len(mids)):
                adj[mids[i]].add(mids[j])
                adj[mids[j]].add(mids[i])

    # connected components
    comps, seen = [], set()
    for mid in adj:
        if mid in seen: continue
        stack, comp = [mid], []
        seen.add(mid)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v); stack.append(v)
        comps.append(comp)

    # stitch each component greedily
    stitched = []
    for comp in comps:
        comp_sorted = sorted(comp, key=lambda m: _linestring_length(mid2arr[m]), reverse=True)
        used = set()
        if comp_sorted:
            cur = mid2arr[comp_sorted[0]].copy()
            used.add(comp_sorted[0])
            extended = True
            while extended:
                extended = False
                end_pt = tuple(cur[-1])
                for m in comp_sorted:
                    if m in used: continue
                    arr2 = mid2arr[m]
                    if tuple(arr2[0]) == end_pt:
                        cur = np.vstack([cur, arr2[1:]])
                        used.add(m); extended = True; break
                    elif tuple(arr2[-1]) == end_pt:
                        cur = np.vstack([cur, arr2[-2::-1]])
                        used.add(m); extended = True; break
            stitched.append(cur)
        for m in comp_sorted:
            if m not in used:
                stitched.append(mid2arr[m])
    return [_finite_xy(s) for s in stitched if len(s) >= 2]

def _ribbon_mask_from_midline(H, W, S_xy, thickness_px=4):
    mask = np.zeros((H, W), dtype=np.uint8)
    pts = np.round(S_xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=thickness_px, lineType=cv2.LINE_AA)
    return mask'''
    
def _align_edge_to_midline(S_xy, E_xy):
    d_f = np.linalg.norm(E_xy[0]-S_xy[0]) + np.linalg.norm(E_xy[-1]-S_xy[-1])
    d_r = np.linalg.norm(E_xy[0]-S_xy[-1]) + np.linalg.norm(E_xy[-1]-S_xy[0])
    return (E_xy[::-1] if d_r < d_f else E_xy)
    
'''def build_branches_from_user_endpoints(members, atomic):
    """
    Pure branch construction from USER endpoints.

    Semantics:
    - Branches are defined STRICTLY by shared USER endpoints
    - Geometry is NEVER merged or modified
    - No clipping, no dominance, no masks
    - Output is stable and deterministic

    Returns:
        branches            : list[list[int]]
            Indices into atomics list (branch topology)

        branch_user_segs    : list[list[np.ndarray]]
            USER midlines per branch (unmodified)

        branch_user_len     : list[float]
            Total USER-space length per branch
    """

    import numpy as np

    # -----------------------------
    # helper: extract user endpoints
    # -----------------------------
    def get_user_endpoints(cr):
        ups = cr.get("user_points", []) or []
        ucs = cr.get("user_connections", []) or []
        out = set()
        for pair in ucs:
            for idx in pair:
                if 0 <= idx < len(ups):
                    out.add(tuple(map(float, ups[idx])))
        return out

    # -----------------------------
    # 1) collect valid atomics
    # -----------------------------
    atomics = []    # [(cid_str, midline_array)]
    endpoints = []  # [set((x,y), ...)]

    for m in members:
        cr = atomic.get(str(m), {}) or {}
        midl = np.asarray(cr.get("midline", []), float)

        if midl.ndim != 2 or len(midl) < 2:
            continue

        midl = _finite_xy(midl)
        if len(midl) < 2:
            continue

        atomics.append((str(m), midl))
        endpoints.append(get_user_endpoints(cr))

    if not atomics:
        return [], [], []

    # -----------------------------
    # 2) build adjacency via shared endpoints
    # -----------------------------
    N = len(atomics)
    adj = {i: set() for i in range(N)}

    for i in range(N):
        for j in range(i + 1, N):
            if endpoints[i] & endpoints[j]:
                adj[i].add(j)
                adj[j].add(i)

    # -----------------------------
    # 3) connected components = branches
    # -----------------------------
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

    # -----------------------------
    # 4) per-branch geometry + length
    # -----------------------------
    branch_user_segs = []
    branch_user_len = []

    for comp in branches:
        user_segs = []
        total_len = 0.0

        for ai in comp:
            _, S = atomics[ai]
            user_segs.append(S)
            total_len += _linestring_length(S)

        branch_user_segs.append(user_segs)
        branch_user_len.append(total_len)

    return branches, branch_user_segs, branch_user_len'''





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

def _linestring_length(arr):
    try:
        a = np.asarray(arr, float)
        if a.ndim != 2 or a.shape[1] != 2 or len(a) < 2:
            return 0.0
        d = np.diff(a, axis=0)
        return float(np.sqrt((d * d).sum(axis=1)).sum())
    except Exception:
        return 0.0
    
def bbox_xywh_to_xyxy(bbox, H, W, *, pad=0):
    """
    Convert authoritative bbox from xywh → xyxy (inclusive).
    NO fallback. NO guessing.

    bbox = [x, y, w, h]
    """
    if bbox is None or len(bbox) != 4:
        raise ValueError("bbox_xywh_to_xyxy: missing or invalid bbox")

    x, y, w, h = map(int, bbox)

    if w <= 0 or h <= 0:
        raise ValueError(f"bbox_xywh_to_xyxy: non-positive w/h: {bbox}")

    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W - 1, x + w - 1 + pad)
    y1 = min(H - 1, y + h - 1 + pad)

    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"bbox_xywh_to_xyxy: degenerate bbox after clip: {bbox}")

    return x0, y0, x1, y1

def _opsec_plot_mask_bbox_only(
    *,
    mask_crop,
    mask_bbox,
    H, W,
    out_path,
    title=None,
):
    """
    OPSEC debug plot:
      - mask_crop pasted into global canvas
      - bbox drawn with sub-pixel precision
      - NO geometry, NO edges, NO midlines
    """

    import numpy as np
    import matplotlib.pyplot as plt

    mc = np.asarray(mask_crop, np.uint8)
    x, y, w, h = map(float, mask_bbox)

    canvas = np.zeros((H, W), np.uint8)

    ih, iw = mc.shape
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(W, x0 + iw)
    y1 = min(H, y0 + ih)

    canvas[y0:y1, x0:x1] = mc[: y1 - y0, : x1 - x0]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    ax.imshow(canvas, cmap="gray")

    ax.add_patch(
        plt.Rectangle(
            (x, y), w, h,
            fill=False,
            edgecolor="red",
            linewidth=1.5,
        )
    )

    ax.set_title(title or "MASK / BBOX OPSEC")
    ax.axis("off")

    fig.savefig(out_path)
    plt.close(fig)

def _atomic_mask_global(cr, H, W, debug_dir=None):
    """
    Expand per-atomic mask_crop into global image space.

    STRICT OPSEC MODE:
      - mask_bbox is authoritative
      - mask_crop MUST match bbox size exactly
      - bbox MUST lie fully inside image bounds
      - NO auto-clamp, NO silent fixes
      - any violation -> OPSEC plot + hard failure
    """

    import numpy as np
    import os

    mc = cr.get("mask_crop")
    bb = cr.get("mask_bbox")
    cid = cr.get("id", "unknown")

    if mc is None or bb is None:
        return None

    mc = np.asarray(mc, np.uint8)
    x, y, w, h = map(int, bb)
    mh, mw = mc.shape

    # --------------------------------------------------
    # OPSEC 1: mask_crop vs bbox size mismatch
    # --------------------------------------------------
    if mh != h or mw != w:
        if debug_dir is not None:
            out_path = os.path.join(
                debug_dir, "opsec",
                f"atomic_mask_bbox_size_mismatch_cid{cid}.png"
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            _opsec_plot_mask_bbox_only(
                mask_crop=mc,
                mask_bbox=bb,
                H=H,
                W=W,
                out_path=out_path,
                title=(
                    f"CID {cid} — MASK/BBOX SIZE MISMATCH\n"
                    f"bbox=(w={w}, h={h})  mask_crop=(w={mw}, h={mh})"
                ),
            )

        raise ValueError(
            f"[ATOMIC OPSEC] cid={cid} mask_crop shape={mc.shape} "
            f"does not match mask_bbox (w={w}, h={h})"
        )

    # --------------------------------------------------
    # OPSEC 2: bbox out of image bounds
    # --------------------------------------------------
    if x < 0 or y < 0 or x + w > W or y + h > H:
        if debug_dir is not None:
            out_path = os.path.join(
                debug_dir, "opsec",
                f"atomic_bbox_out_of_bounds_cid{cid}.png"
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            _opsec_plot_mask_bbox_only(
                mask_crop=mc,
                mask_bbox=bb,
                H=H,
                W=W,
                out_path=out_path,
                title=(
                    f"CID {cid} — BBOX OUT OF IMAGE BOUNDS\n"
                    f"bbox=(x={x}, y={y}, w={w}, h={h})\n"
                    f"image=(W={W}, H={H})"
                ),
            )

        raise ValueError(
            f"[ATOMIC OPSEC] cid={cid} bbox out of bounds: "
            f"(x+w={x+w} > W={W} or y+h={y+h} > H={H})"
        )

    # --------------------------------------------------
    # Normal path (authoritative placement)
    # --------------------------------------------------
    m = np.zeros((H, W), np.uint8)
    m[y:y+h, x:x+w] = mc
    return m
   
def dominant_segments_from_group(
    *,
    members,
    atomic,
    crack_mask_u8=None,
    window_half_size,
    debug_dir=None,
    debug_tag="dominance_grouping",
    original_image=None,
):
    """
    FINAL dominance logic (portable version) + OPSEC metadata.

    - Branches defined by shared USER endpoints (atomic space)
    - Branch ordering by total USER length (never clipped)
    - Dominance between branches only
    - Output segments are USER-space polylines:
        * primary: unmodified
        * subordinate: clipped OUT of claimed territory (and also constrained to crack_mask when provided)

    Territory modes:
      A) crack_mask_u8 PROVIDED  -> DT-based territory (dilated midline-within-crack_mask)
      B) crack_mask_u8 is None   -> atomic-mask territory (union of atomic mask crops)

    Returns: (kept_segments, meta)
    """
    import os
    import base64
    import numpy as np
    import cv2

    # -----------------------------
    # local helper: user endpoints
    # -----------------------------
    def get_user_endpoints(cr):
        ups = cr.get("user_points", []) or []
        ucs = cr.get("user_connections", []) or []
        out = set()
        for pair in ucs:
            for idx in pair:
                if 0 <= idx < len(ups):
                    out.add(tuple(map(float, ups[idx])))
        return out

    def _union_member_bbox_xyxy(member_ids):
        """
        Compute intrinsic union bbox (x0, y0, x1, y1)
        from atomic mask_bboxes.

        No clamping. No canvas assumptions.
        """
        boxes = []

        for m in member_ids:
            bb = (atomic.get(str(m), {}) or {}).get("mask_bbox")
            if bb and len(bb) == 4:
                x, y, w, h = map(int, bb)
                if w > 0 and h > 0:
                    boxes.append((x, y, x + w, y + h))

        if not boxes:
            return None

        bx0 = min(b[0] for b in boxes)
        by0 = min(b[1] for b in boxes)
        bx1 = max(b[2] for b in boxes)
        by1 = max(b[3] for b in boxes)

        if bx1 <= bx0 or by1 <= by0:
            return None

        return (bx0, by0, bx1, by1)
    
    def _pack_mask_b64(mask_u8):
        mask_u8 = (mask_u8 > 0).astype(np.uint8)
        if mask_u8.size == 0:
            return {"shape": [0, 0], "packbits_b64": ""}
        packed = np.packbits(mask_u8, axis=1)  # (H, ceil(W/8))
        b64 = base64.b64encode(packed.tobytes()).decode("ascii")
        return {"shape": [int(mask_u8.shape[0]), int(mask_u8.shape[1])], "packbits_b64": b64}

    def _midline_global_from_atomic(cr, S_in):
        """
        Normalize an atomic midline into full-image/global coordinates.

        Some persisted atomics may carry crop-local midlines while mask_bbox stays global.
        Detect that frame mismatch and shift by (bx, by) exactly once.
        """
        S = np.asarray(S_in, float)
        if S.ndim != 2 or S.shape[1] != 2 or len(S) < 2:
            return S

        bb = (cr or {}).get("mask_bbox", None)
        if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
            return S

        try:
            bx, by, bw, bh = [float(v) for v in bb]
        except Exception:
            return S

        if bw <= 0 or bh <= 0:
            return S

        # Heuristic:
        # - local-like if points mostly inside [0..bw]x[0..bh]
        # - global-like if points mostly inside [bx..bx+bw]x[by..by+bh]
        # If local-like and not global-like, shift to global frame.
        local_like = np.mean(
            (S[:, 0] >= -1.0) & (S[:, 0] <= bw + 1.0) &
            (S[:, 1] >= -1.0) & (S[:, 1] <= bh + 1.0)
        )
        global_like = np.mean(
            (S[:, 0] >= bx - 1.0) & (S[:, 0] <= bx + bw + 1.0) &
            (S[:, 1] >= by - 1.0) & (S[:, 1] <= by + bh + 1.0)
        )

        if (bx > 0 or by > 0) and local_like >= 0.95 and global_like < 0.5:
            S = S + np.array([bx, by], float)
            _c_log(
                1,
                f"[FRAME_FIX] atomic midline local->global via mask_bbox "
                f"offset atomic_bbox=({int(bx)},{int(by)},{int(bw)},{int(bh)})",
            )
        return S

    # -----------------------------
    # 1) collect atomic segments
    # -----------------------------
    atomics = []  # list of dicts: {"atomic_id", "poly", "length", "endpoints"}

    for m in members:
        cr = atomic.get(str(m), {}) or {}
        S = np.asarray(cr.get("midline", []), float)
        if S.ndim == 2 and len(S) >= 2:
            S = _finite_xy(S)
            if S is None or len(S) < 2:
                continue
            S = _midline_global_from_atomic(cr, S)
            atomics.append(
                {
                    "atomic_id": str(m),
                    "poly": S,
                    "length": float(_linestring_length(S)),
                    "endpoints": get_user_endpoints(cr),
                }
            )

    for a in atomics:
        S = a["poly"]
        bb = (atomic.get(str(a["atomic_id"]), {}) or {}).get("mask_bbox")
        print(
            f"[ATOMIC_FRAME_DBG] id={a['atomic_id']} "
            f"bbox={bb} "
            f"midline_x=[{S[:,0].min():.1f},{S[:,0].max():.1f}] "
            f"midline_y=[{S[:,1].min():.1f},{S[:,1].max():.1f}] "
            f"n={len(S)}"
        )

    if not atomics:
        return [], {"branches": [], "segments_meta": [], "bite": None}

    # -----------------------------
    # 1.5) mask / canvas init (NOW atomics exists)
    # -----------------------------
    use_dt_territory = (crack_mask_u8 is not None)   # DT territory needs real crack mask
    use_atomic_mask_territory = (crack_mask_u8 is None)

    # --------------------------------------------------
    # Determine global canvas (AUTHORITATIVE)
    # --------------------------------------------------
    if crack_mask_u8 is not None:
        # ----------------------------------------------
        # Primary path: canvas comes from real mask
        # ----------------------------------------------
        H, W = crack_mask_u8.shape[:2]
        crack_mask = (crack_mask_u8 > 0).astype(np.uint8)

        # Union bbox (pure geometry, then clamped to canvas)
        bbox_xyxy = _union_member_bbox_xyxy(members)
        if bbox_xyxy is None:
            bx0, by0, bx1, by1 = 0, 0, W, H
        else:
            bx0, by0, bx1, by1 = bbox_xyxy

            # Clamp ONLY here (mask-backed canvas is authoritative)
            bx0 = max(0, min(bx0, W))
            by0 = max(0, min(by0, H))
            bx1 = max(0, min(bx1, W))
            by1 = max(0, min(by1, H))

            if bx1 <= bx0 or by1 <= by0:
                raise ValueError(
                    f"[OPSEC] Union bbox collapses after clamp: "
                    f"({bx0},{by0})-({bx1},{by1}) with canvas W={W}, H={H}"
                )

    else:
        # ----------------------------------------------
        # STRICT fallback: canvas derived ONLY from union bbox
        # ----------------------------------------------
        bbox_xyxy = _union_member_bbox_xyxy(members)

        if bbox_xyxy is None:
            raise ValueError(
                "[OPSEC] No crack_mask_u8 and no union bbox available — "
                "cannot construct fallback canvas"
            )

        bx0, by0, bx1, by1 = bbox_xyxy

        W = int(np.ceil(bx1))
        H = int(np.ceil(by1))

        if W <= 0 or H <= 0:
            raise ValueError(
                f"[OPSEC] Invalid fallback canvas from bbox: W={W}, H={H}"
            )

        crack_mask = None

    # -----------------------------
    def _dbg_plot_junction(jx, jy, full_branch, incoming, valid_candidates, atomics_ref, save_path, N=30):
        """Debug plot: show incoming direction and candidate outgoing directions at a junction."""
        try:
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as _plt
            fig, ax = _plt.subplots(figsize=(8, 8))
            tail60 = full_branch[-min(60, len(full_branch)):]
            ax.plot(tail60[:,0], tail60[:,1], '-', color='#AAAAAA', lw=1.5, label='branch tail (60 pts)')
            tailN = full_branch[-min(N, len(full_branch)):]
            ax.plot(tailN[:,0], tailN[:,1], '-', color='#E8A020', lw=3, label=f'local tail ({N} pts)')
            ax.plot(jx, jy, 'ko', ms=10, zorder=6, label='junction')
            inc_n = incoming / (np.linalg.norm(incoming) + 1e-9) * 50
            ax.annotate('', xy=(jx+inc_n[0], jy+inc_n[1]), xytext=(jx, jy),
                        arrowprops=dict(arrowstyle='->', color='#E8A020', lw=2.5))
            cols = ['#2A9D5C', '#C83030', '#1166CC', '#AA44AA', '#CC8800']
            for ci, (cj, cmode, capt, cturn) in enumerate(valid_candidates):
                if (round(float(capt[0]),1), round(float(capt[1]),1)) != (jx, jy):
                    continue
                col = cols[ci % len(cols)]
                Sjc = np.asarray(atomics_ref[cj]['poly'], float)
                if cmode[1]: Sjc = Sjc[::-1]
                headN = Sjc[:min(N, len(Sjc))]
                ax.plot(headN[:,0], headN[:,1], '-', color=col, lw=2,
                        label=f'cid{atomics_ref[cj]["atomic_id"]} turn={cturn:.1f}deg')
                ov = headN[-1] - headN[0] if len(headN) > 1 else np.array([1, 0])
                on = ov / (np.linalg.norm(ov) + 1e-9) * 50
                ax.annotate('', xy=(jx+on[0], jy+on[1]), xytext=(jx, jy),
                            arrowprops=dict(arrowstyle='->', color=col, lw=2.2))
                ax.text(jx+on[0]+3, jy+on[1], f'cid{atomics_ref[cj]["atomic_id"]}\n{cturn:.1f}deg', fontsize=8, color=col)
            ax.set_aspect('equal'); ax.invert_yaxis()
            ax.legend(fontsize=8); ax.set_title(f'Junction ({jx},{jy}) N={N}', fontsize=10)
            _plt.tight_layout()
            _plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            _plt.close(fig)
            print(f'[DBG_JUNCTION] saved {save_path}', flush=True)
        except Exception as _de:
            print(f'[DBG_JUNCTION] failed: {_de}', flush=True)

    # 2) build branches greedily in atomic space
    # -----------------------------
    def _endpoints(S):
        return np.asarray(S[0], float), np.asarray(S[-1], float)

    def _pts_close(a, b, tol=.2):
        return float(np.linalg.norm(a - b)) <= tol

    # Pre-compute group bounding box from all atomic endpoints
    _all_eps = []
    for _a in atomics:
        _s, _e = _endpoints(_a["poly"])
        _all_eps.extend([_s, _e])
    _all_eps = np.array(_all_eps)
    _grp_xmin, _grp_ymin = _all_eps[:,0].min(), _all_eps[:,1].min()
    _grp_xmax, _grp_ymax = _all_eps[:,0].max(), _all_eps[:,1].max()
    _EDGE_THRESH = 5.0  # px — far endpoint within this distance of group bbox edge gets priority

    def _dist_to_group_edge(pt):
        """Distance from pt to nearest edge of the group bbox."""
        dx = min(float(pt[0]) - _grp_xmin, _grp_xmax - float(pt[0]))
        dy = min(float(pt[1]) - _grp_ymin, _grp_ymax - float(pt[1]))
        return min(dx, dy)

    # Pre-compute combine_debug dir for junction plots
    _combine_debug_dir = None
    if debug_dir is not None:
        _combine_debug_dir = os.path.join(os.path.dirname(debug_dir), "combine_debug")
        os.makedirs(_combine_debug_dir, exist_ok=True)

    unused = set(range(len(atomics)))
    branches = []  # list[list[atomic indices]]
    attach_orders = {}  # atomic_idx -> order within branch (1-based)

    # Pre-compute endpoint degree across all atomics in this group.
    # Endpoints touched by 3+ segments are junction/loop nodes.
    ep_degree = {}
    for i, a in enumerate(atomics):
        s, e = _endpoints(a["poly"])
        for pt in (s, e):
            key = (round(float(pt[0]), 1), round(float(pt[1]), 1))
            ep_degree[key] = ep_degree.get(key, 0) + 1
    junction_pts = {k for k, v in ep_degree.items() if v >= 3}
    if junction_pts:
        print(f"[LOOP_DETECT] found {len(junction_pts)} junction/loop nodes: {junction_pts}", flush=True)

    while unused:
        loop_rejected = set()  # reset per branch
        # start new branch with longest remaining segment
        start_idx = max(unused, key=lambda i: atomics[i]["length"])
        unused.remove(start_idx)

        branch = [start_idx]
        attach_orders[int(start_idx)] = 1
        order_in_branch = 1

        S0 = atomics[start_idx]["poly"]
        b_start, b_end = _endpoints(S0)

        restarted = True
        while restarted:
            restarted = False
            endpoint_history = []  # reset on each greedy pass (restarted or fresh)
            # Recalculate open endpoints by replaying the current branch.
            # Segments may have attached at EITHER b_end or b_start, so check both.
            b_start, b_end = _endpoints(atomics[branch[0]]["poly"])
            for _seg in branch[1:]:
                _S = atomics[_seg]["poly"]
                _s, _e = _endpoints(_S)
                if _pts_close(_e, b_end) or _pts_close(_s, b_end):
                    # segment attaches at b_end - update b_end
                    _Sf = _S[::-1] if _pts_close(_e, b_end) else _S
                    _, b_end = _endpoints(_Sf)
                elif _pts_close(_e, b_start) or _pts_close(_s, b_start):
                    # segment attaches at b_start - update b_start
                    _Sf = _S[::-1] if _pts_close(_s, b_start) else _S
                    b_start, _ = _endpoints(_Sf)

            def _turn_angle_deg(incoming_dir, outgoing_dir):
                """Compute absolute turn angle in degrees [0, 180] between two direction vectors.
                Uses the minimum of CW and CCW rotation so a slight clockwise turn
                reads as ~24 deg, not ~336 deg."""
                if np.linalg.norm(incoming_dir) < 1e-9 or np.linalg.norm(outgoing_dir) < 1e-9:
                    return 0.0
                a_in  = float(np.arctan2(incoming_dir[1], incoming_dir[0]))
                a_out = float(np.arctan2(outgoing_dir[1], outgoing_dir[0]))
                diff  = (a_out - a_in) % (2 * np.pi)
                # Return the smaller of CCW and CW rotation (0-180 range)
                return float(np.degrees(min(diff, 2 * np.pi - diff)))

            grew = True
            while grew:
                grew = False
                best_j = None
                best_len = -1.0
                best_attach_mode = None  # ("start"/"end", flip_bool)

                # --- Collect all valid candidates first (for per-junction angle filtering) ---
                valid_candidates = []  # list of (j, mode, apt_key, turn_deg)
                for j in list(unused):
                    if j in loop_rejected:
                        print(
                            f"[GREEDY_DBG] branch_so_far={[atomics[i]['atomic_id'] for i in branch]} "
                            f"candidate={atomics[j]['atomic_id']} skipped=loop_rejected",
                            flush=True
                        )
                        continue
                    Sj = atomics[j]["poly"]
                    j_start, j_end = _endpoints(Sj)

                    # try attach to branch end
                    if _pts_close(b_end, j_start):
                        mode = ("end", False)
                    elif _pts_close(b_end, j_end):
                        mode = ("end", True)

                    # try attach to branch start
                    elif _pts_close(b_start, j_end):
                        mode = ("start", False)
                    elif _pts_close(b_start, j_start):
                        mode = ("start", True)
                    else:
                        print(
                            f"[GREEDY_DBG] branch_so_far={[atomics[i]['atomic_id'] for i in branch]} "
                            f"candidate={atomics[j]['atomic_id']} "
                            f"b_start={b_start.tolist()} b_end={b_end.tolist()} "
                            f"j_start={j_start.tolist()} j_end={j_end.tolist()} "
                            f"attach_pt=None in_junction=False skipped=no_endpoint_match",
                            flush=True
                        )
                        continue

                    attach_pt = b_end if mode[0] == "end" else b_start
                    apt_key = (round(float(attach_pt[0]), 1), round(float(attach_pt[1]), 1))

                    # Compute turn angle at this attach point using local tail/head direction
                    # Use last/first N points of the branch polyline for a robust local direction
                    _N_LOCAL = 30  # 20-40 pts for stable local direction at junction
                    def _local_dir(poly, from_end=True):
                        """Local direction at the end (from_end=True) or start of a polyline."""
                        P = np.asarray(poly, float)
                        n = max(2, min(_N_LOCAL, len(P)))
                        if from_end:
                            return P[-1] - P[-n]
                        else:
                            return P[n-1] - P[0]

                    # Build the current branch polyline in order to get local tail direction
                    # Replay branch to get oriented segments
                    _branch_pts_end = None
                    _branch_pts_start = None
                    _cur_b_end = _endpoints(atomics[branch[0]]["poly"])[1]
                    _seg_oriented = [atomics[branch[0]]["poly"]]
                    for _bi in branch[1:]:
                        _Sb = atomics[_bi]["poly"]
                        _sb, _eb = _endpoints(_Sb)
                        if _pts_close(_eb, _cur_b_end):
                            _Sb = _Sb[::-1]
                        _seg_oriented.append(_Sb)
                        _cur_b_end = _endpoints(_Sb)[1]
                    _full_branch = np.concatenate([np.asarray(s, float) for s in _seg_oriented], axis=0)

                    # incoming: local direction of travel arriving AT the attach point.
                    # Scan branch in reverse for the last segment whose endpoint IS
                    # the attach point, oriented so its tail points INTO that point.
                    # This works correctly for BOTH b_end and b_start, and avoids
                    # the _full_branch orientation bug when the branch has grown on
                    # both sides (b_start segments would corrupt the b_end tail).
                    _attach_terminal = b_end if mode[0] == "end" else b_start
                    _incoming_seg = None
                    for _ri in reversed(branch):
                        _Rs = np.asarray(atomics[_ri]["poly"], float)
                        _rs, _re = _endpoints(_Rs)
                        if _pts_close(_re, _attach_terminal):
                            _incoming_seg = _Rs        # end = attach_pt, tail approaches it
                            break
                        elif _pts_close(_rs, _attach_terminal):
                            _incoming_seg = _Rs[::-1]  # flip so end = attach_pt
                            break
                    if _incoming_seg is not None:
                        incoming = _local_dir(_incoming_seg, from_end=True)
                    else:
                        # fallback: use full branch tail/head
                        if mode[0] == "end":
                            incoming = _local_dir(_full_branch, from_end=True)
                        else:
                            incoming = -_local_dir(_full_branch, from_end=False)

                    # outgoing: local direction LEAVING the junction into candidate segment
                    Sj_arr = np.asarray(atomics[j]["poly"], float)
                    if mode[0] == "end":
                        # attaching to b_end
                        if not mode[1]:  # j_start matches b_end → seg departs forward from junction
                            outgoing = _local_dir(Sj_arr, from_end=False)
                        else:           # j_end matches b_end → seg departs reversed from junction
                            outgoing = _local_dir(Sj_arr[::-1], from_end=False)
                    else:
                        # attaching to b_start
                        if not mode[1]:  # j_end matches b_start → seg departs from j_end toward j_start (reversed)
                            outgoing = _local_dir(Sj_arr[::-1], from_end=False)
                        else:           # j_start matches b_start → seg departs naturally from j_start toward j_end
                            outgoing = _local_dir(Sj_arr, from_end=False)
                    turn_deg = _turn_angle_deg(incoming, outgoing)


                    # DEBUG PLOT — save junction plot to combine_debug for all junctions
                    _dbg_jx2 = round(float(attach_pt[0]), 1) if attach_pt is not None else None
                    _dbg_jy2 = round(float(attach_pt[1]), 1) if attach_pt is not None else None
                    if (_dbg_jx2 is not None and apt_key in junction_pts
                            and len(valid_candidates) == 0  # only on first candidate at this junction
                            and _combine_debug_dir is not None):
                        _jplot_path = os.path.join(
                            _combine_debug_dir, f"junction_{_dbg_jx2}_{_dbg_jy2}.png"
                        )
                        _dbg_plot_junction(
                            _dbg_jx2, _dbg_jy2, _full_branch, incoming, valid_candidates, atomics,
                            _jplot_path
                        )

                    valid_candidates.append((j, mode, apt_key, turn_deg))


                # --- Per-junction angle filter: at each junction with multiple candidates,
                #     if any candidate has turn <= 300°, exclude >300° ones from this pass ---
                SHARP_TURN_THRESHOLD = 150.0  # 0-180 range (equiv. 300 deg CW): exclude near-U-turns at junctions
                from collections import defaultdict as _dd
                by_junction = _dd(list)
                for cand in valid_candidates:
                    by_junction[cand[2]].append(cand)

                filtered_candidates = []
                for apt_key, group in by_junction.items():
                    if apt_key in junction_pts and len(group) > 1:
                        has_gentle = any(c[3] <= SHARP_TURN_THRESHOLD for c in group)
                        for c in group:
                            if has_gentle and c[3] > SHARP_TURN_THRESHOLD:
                                print(
                                    f"[SHARP_TURN_FILTER] branch_so_far={[atomics[i]['atomic_id'] for i in branch]} "
                                    f"candidate={atomics[c[0]]['atomic_id']} apt_key={apt_key} "
                                    f"turn={c[3]:.1f}° excluded (alternatives exist at junction)",
                                    flush=True
                                )
                            else:
                                filtered_candidates.append(c)
                    else:
                        filtered_candidates.extend(group)

                # --- Pick best: edge-proximity first, then length ---
                # Compute far endpoint (the end NOT at the attach point) for each candidate
                best_edge_dist = float('inf')
                for j, mode, apt_key, turn_deg in filtered_candidates:
                    Sj_ep = atomics[j]["poly"]
                    js, je = _endpoints(Sj_ep)
                    attach_pt = b_end if mode[0] == "end" else b_start
                    far_pt = je if _pts_close(attach_pt, js) else js
                    edge_dist = _dist_to_group_edge(far_pt)
                    if edge_dist < best_edge_dist:
                        best_edge_dist = edge_dist

                for j, mode, apt_key, turn_deg in filtered_candidates:
                    Sj_ep = atomics[j]["poly"]
                    js, je = _endpoints(Sj_ep)
                    attach_pt = b_end if mode[0] == "end" else b_start
                    far_pt = je if _pts_close(attach_pt, js) else js
                    edge_dist = _dist_to_group_edge(far_pt)
                    print(
                        f"[GREEDY_DBG] branch_so_far={[atomics[i]['atomic_id'] for i in branch]} "
                        f"candidate={atomics[j]['atomic_id']} "
                        f"b_start={b_start.tolist()} b_end={b_end.tolist()} "
                        f"j_start={js.tolist()} j_end={je.tolist()} "
                        f"attach_pt={attach_pt.tolist()} "
                        f"in_junction={apt_key in junction_pts} "
                        f"turn={turn_deg:.1f}° edge_dist={edge_dist:.1f}px",
                        flush=True
                    )
                    # Edge-proximity overrides length: if any candidate is within
                    # _EDGE_THRESH of the group bbox edge, only those compete by length.
                    # If none are within threshold, all compete by length as before.
                    candidate_near_edge = best_edge_dist <= _EDGE_THRESH
                    if candidate_near_edge and edge_dist > _EDGE_THRESH:
                        continue  # skip — a better edge candidate exists
                    if atomics[j]["length"] > best_len:
                        best_len = atomics[j]["length"]
                        best_j = j
                        best_attach_mode = mode

                if best_j is not None:
                    # --- Prospective loop check ---
                    # Would adding best_j land its far endpoint on an interior junction
                    # already in the branch? Same logic as the reactive backstep but
                    # checked BEFORE accepting. Interior = branch[1:-1] endpoints,
                    # excluding the current two terminals (b_start, b_end).
                    _Sj_chk = atomics[best_j]["poly"]
                    _js_chk, _je_chk = _endpoints(_Sj_chk)
                    _attach_chk = b_end if best_attach_mode[0] == "end" else b_start
                    _far_chk = _je_chk if _pts_close(_attach_chk, _js_chk) else _js_chk
                    _far_key = (round(float(_far_chk[0]), 1), round(float(_far_chk[1]), 1))
                    # Interior = all branch endpoints except the two current terminals
                    _b_start_key = (round(float(b_start[0]), 1), round(float(b_start[1]), 1))
                    _b_end_key   = (round(float(b_end[0]),   1), round(float(b_end[1]),   1))
                    _interior_keys = set()
                    for _bi in branch[1:-1]:
                        _bs, _be = _endpoints(atomics[_bi]["poly"])
                        _interior_keys.add((round(float(_bs[0]), 1), round(float(_bs[1]), 1)))
                        _interior_keys.add((round(float(_be[0]), 1), round(float(_be[1]), 1)))
                    _interior_keys.discard(_b_start_key)
                    _interior_keys.discard(_b_end_key)
                    if _far_key in _interior_keys:
                        # Mirror the reactive backstep exactly:
                        # best_j is the would-be loop closer — treat it like the
                        # reactive "popped" segment. The last segment in branch is
                        # its attachment partner — treat it like "pulled".
                        # Save them as a pair (or solo) and restart on trimmed branch.
                        _popped = best_j
                        unused.discard(_popped)  # keep it out of unused — it's going to a new branch
                        if len(branch) >= 1:
                            _pop_s, _pop_e = _endpoints(atomics[_popped]["poly"])
                            _pop_attach = b_end if best_attach_mode[0] == "end" else b_start
                            _pop_other = _pop_e if _pts_close(_pop_attach, _pop_s) else _pop_s
                            _last_s, _last_e = _endpoints(atomics[branch[-1]]["poly"])
                            if _pts_close(_last_s, _pop_other) or _pts_close(_last_e, _pop_other):
                                _pulled = branch.pop()
                                attach_orders.pop(int(_pulled), None)
                                if endpoint_history:
                                    b_start, b_end = endpoint_history.pop()
                                unused.discard(_pulled)
                                branches.append([_pulled, _popped])
                                attach_orders[int(_pulled)] = 1
                                attach_orders[int(_popped)] = 2
                                print(f"[PROSP_LOOP] saved [{atomics[_pulled]['atomic_id']},{atomics[_popped]['atomic_id']}] as pair", flush=True)
                            else:
                                branches.append([_popped])
                                attach_orders[int(_popped)] = 1
                                print(f"[PROSP_LOOP] saved [{atomics[_popped]['atomic_id']}] as solo", flush=True)
                        else:
                            branches.append([_popped])
                            attach_orders[int(_popped)] = 1
                            print(f"[PROSP_LOOP] saved [{atomics[_popped]['atomic_id']}] as solo (empty branch)", flush=True)
                        if branch:
                            restarted = True
                        grew = False
                        break

                    # 2-hop triangle check: if best_j's far endpoint is an exact
                    # endpoint of an unused segment whose OTHER endpoint is in
                    # interior_keys, then best_j + that connector form a closed
                    # 3-segment triangle with a segment already in the branch.
                    # Save best_j + connector as a pair branch without touching
                    # the current branch.
                    # Use ALL branch segment endpoints (not just [1:-1]) minus
                    # the current growth tip — this handles 2-element branches
                    # where branch[1:-1] is empty but the interior junction exists.
                    _all_branch_keys = set()
                    for _bi in branch:
                        _abs, _abe = _endpoints(atomics[_bi]["poly"])
                        _all_branch_keys.add((round(float(_abs[0]), 1), round(float(_abs[1]), 1)))
                        _all_branch_keys.add((round(float(_abe[0]), 1), round(float(_abe[1]), 1)))
                    _all_branch_keys.discard(_b_start_key)  # current growth tip only
                    _hop2_connector = None
                    for _uc in list(unused):
                        if _uc == best_j:
                            continue
                        _uc_s, _uc_e = _endpoints(atomics[_uc]["poly"])
                        _uc_s_key = (round(float(_uc_s[0]), 1), round(float(_uc_s[1]), 1))
                        _uc_e_key = (round(float(_uc_e[0]), 1), round(float(_uc_e[1]), 1))
                        if _pts_close(_uc_s, _far_chk) and _uc_e_key in _all_branch_keys:
                            _hop2_connector = _uc
                            break
                        if _pts_close(_uc_e, _far_chk) and _uc_s_key in _all_branch_keys:
                            _hop2_connector = _uc
                            break
                    if _hop2_connector is not None:
                        unused.discard(best_j)
                        unused.discard(_hop2_connector)
                        branches.append([best_j, _hop2_connector])
                        attach_orders[int(best_j)] = 1
                        attach_orders[int(_hop2_connector)] = 2
                        print(
                            f"[PROSP_LOOP_2HOP] saved "
                            f"[{atomics[best_j]['atomic_id']},{atomics[_hop2_connector]['atomic_id']}]"
                            f" as triangle pair",
                            flush=True
                        )
                        if branch:
                            restarted = True
                        grew = False
                        break

                    side, flip = best_attach_mode

                    endpoint_history.append((b_start.copy(), b_end.copy()))
                    unused.remove(best_j)
                    branch.append(best_j)

                    order_in_branch += 1
                    attach_orders[int(best_j)] = int(order_in_branch)

                    Sj = atomics[best_j]["poly"]
                    if flip:
                        Sj = Sj[::-1].copy()

                    j_start, j_end = _endpoints(Sj)

                    # update branch endpoints
                    if side == "end":
                        b_end = j_end
                    else:
                        b_start = j_start

                    grew = True

        if branch:
            branches.append(branch)

    _c_log(1, "\n[BRANCH SUMMARY]")
    for bi, br in enumerate(branches):
        _c_log(1, f"  branch {bi}: {[atomics[i]['atomic_id'] for i in br]}")

    # --------------------------------------------------
    # Optional greedy-branch construction debug plot
    # --------------------------------------------------
    endpoint_graph_deg = {}
    for i, a in enumerate(atomics):
        for pt in a.get("endpoints", set()) or set():
            k = (float(pt[0]), float(pt[1]))
            endpoint_graph_deg.setdefault(k, set()).add(i)
    endpoint_graph_deg = {k: len(v) for k, v in endpoint_graph_deg.items()}

    seg_adj = {i: set() for i in range(len(atomics))}
    for i in range(len(atomics)):
        ei = atomics[i].get("endpoints", set()) or set()
        for j in range(i + 1, len(atomics)):
            ej = atomics[j].get("endpoints", set()) or set()
            if ei & ej:
                seg_adj[i].add(j)
                seg_adj[j].add(i)

    comps = []
    seen = set()
    for i in range(len(atomics)):
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
            stack.extend(seg_adj[u])
        comps.append(comp)

    connected_comp_size = len(comps[0]) if len(comps) == 1 else 0
    plot_greedy_branch_debug(
        atomics=atomics,
        branches=branches,
        attach_orders=attach_orders,
        endpoint_graph_deg=endpoint_graph_deg,
        connected_comp_size=connected_comp_size,
        group_id=debug_tag,
        debug_dir=debug_dir,
        original_image=original_image,
    )

    # Write branch summary CSV to combine_debug dir (sibling of pairwise_debug.csv)
    if debug_dir:
        try:
            import os, pandas as pd
            _branch_rows = []
            for _bi, _br in enumerate(branches):
                for _order, _idx in enumerate(_br):
                    _branch_rows.append({
                        "branch_id": _bi,
                        "position": _order,
                        "atomic_id": atomics[_idx]["atomic_id"],
                        "n_atomics_in_branch": len(_br),
                    })
            if _branch_rows:
                _bdf = pd.DataFrame(_branch_rows)
                # Put alongside pairwise_debug.csv in the combine_debug subdir
                _combine_debug_dir = os.path.join(os.path.dirname(debug_dir), "combine_debug")
                os.makedirs(_combine_debug_dir, exist_ok=True)
                _bcsv = os.path.join(_combine_debug_dir, "greedy_branches.csv")
                _bdf.to_csv(_bcsv, index=False)
                print(f"[GREEDY_BRANCHES] wrote {_bcsv}", flush=True)
        except Exception as _be:
            print(f"[GREEDY_BRANCHES] CSV write failed: {_be}", flush=True)

    # -----------------------------
    # 3) per-branch user length + seg lists
    # -----------------------------
    branch_user_len = []
    branch_user_segs = []        # list[list[(atomic_id, USER_polyline)]]
    branch_terr_segs = []        # list[list[polyline_piece]] (ONLY used for DT territory mode)

    for br in branches:
        total_len = 0.0
        user_segs = []
        terr_segs = []

        for idx in br:
            atomic_id = atomics[idx]["atomic_id"]
            S_user = atomics[idx]["poly"]
            if S_user is None or len(S_user) < 2:
                continue

            total_len += atomics[idx]["length"]
            user_segs.append((atomic_id, S_user))

            # Territory polylines only needed when we have a crack_mask to clip to
            if crack_mask is not None:
                pieces = _clip_polyline_to_mask(S_user, crack_mask)
                for p in pieces:
                    if p is not None and len(p) >= 2:
                        terr_segs.append(p)
            else:
                # no crack_mask: keep something non-empty so terr_len gate doesn't auto-suppress
                terr_segs.append(S_user)

        branch_user_len.append(float(total_len))
        branch_user_segs.append(user_segs)
        branch_terr_segs.append(terr_segs)

    # -----------------------------
    # 4) dominance between branches (ordered by USER length)
    # -----------------------------

    # -------------------------------------------------
    # Select DT domain mask
    # -------------------------------------------------
    if crack_mask is not None:
        domain_mask = (crack_mask > 0).astype(np.uint8)
    else:
        domain_mask = np.zeros((H, W), np.uint8)
        for m in members:
            cr = atomic.get(str(m), {}) or {}
            am = _atomic_mask_global(cr, H, W, debug_dir)
            if am is not None:
                domain_mask |= (am > 0).astype(np.uint8)

    # -------------------------------------------------
    # Compute DT once over domain
    # -------------------------------------------------
    dt = cv2.distanceTransform(domain_mask, cv2.DIST_L2, 5)

    def seg_radius(S):
        ys = np.clip(np.round(S[:, 1]).astype(int), 0, H - 1)
        xs = np.clip(np.round(S[:, 0]).astype(int), 0, W - 1)
        d = dt[ys, xs]
        d = d[np.isfinite(d)]
        if len(d) == 0:
            return 0.3 * window_half_size
        return max(3.0, min(float(np.median(d)), window_half_size))

    order = sorted(
        range(len(branches)),
        key=lambda i: branch_user_len[i],
        reverse=True,
    )

    claimed = np.zeros((H, W), np.uint8)
    bite_total = {
        "mask": np.zeros((H, W), np.uint8),
        "terr": np.zeros((H, W), np.uint8),
        "both": np.zeros((H, W), np.uint8),
    }


    kept_meta = []
    branch_stats = []
    primary_branch = order[0] if order else None

    branch_terr_masks = {}  # debug only

    # -------------------------------------------------
    # Precompute per-branch atomic masks (hard exclusion)
    # -------------------------------------------------
    branch_atomic_masks = {}
    for bi in range(len(branches)):
        m = np.zeros((H, W), np.uint8)
        for atomic_id, _ in branch_user_segs[bi]:
            cr = atomic.get(str(atomic_id), {}) or {}
            am = _atomic_mask_global(cr, H, W)
            if am is not None:
                m |= (am > 0).astype(np.uint8)
        branch_atomic_masks[bi] = m

    # If no member has a usable mask, dominance suppression is unstable for
    # midline-only annotations. Keep all branch geometry in this mode.
    no_atomic_masks_mode = (
        crack_mask is None
        and not any(np.any(mm > 0) for mm in branch_atomic_masks.values())
    )
    if no_atomic_masks_mode:
        print(
            "[DOMINANCE FALLBACK] no usable atomic masks for members; "
            "keeping all branch segments (no suppression)."
        )

    bite_by_losing_branch = {}

    for rank, bi in enumerate(order):

        # -------------------------------------------------
        # Build DT-based territory
        # -------------------------------------------------
        branch_terr = np.zeros((H, W), np.uint8)

        for atomic_id, S_user in branch_user_segs[bi]:
            if S_user is None or len(S_user) < 2:
                continue

            r = seg_radius(S_user)
            rad = int(
                max(
                    4,                         # hard minimum exclusion
                    min(2.0 * r, window_half_size)
                )
            )

            line = _polyline_mask(S_user, H, W)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1)
            )

            terr = cv2.dilate(line, kernel, iterations=1)
            #terr &= domain_mask
            branch_terr |= terr

        # Constrain territory to crack mask first.
        if crack_mask is not None:
            branch_terr &= crack_mask

        # Keep only territory CCs connected to this branch's own segments.
        if crack_mask is not None and np.any(branch_terr):
            num_labels, labels = cv2.connectedComponents(
                (branch_terr > 0).astype(np.uint8), connectivity=8
            )
            if int(num_labels) > 2:
                seg_mask = np.zeros((H, W), np.uint8)
                for atomic_id, S_user in branch_user_segs[bi]:
                    if S_user is not None and len(S_user) >= 2:
                        seg_mask |= _polyline_mask(S_user, H, W)
                seg_mask = cv2.dilate(seg_mask, np.ones((3, 3), np.uint8), iterations=1)
                keep = np.zeros((H, W), np.uint8)
                for lab in range(1, int(num_labels)):
                    if np.any((labels == lab) & (seg_mask > 0)):
                        keep |= (labels == lab).astype(np.uint8)
                if np.any(keep):
                    branch_terr = keep

        branch_terr_masks[bi] = branch_terr.copy()

        if no_atomic_masks_mode:
            kept_len = 0.0
            for atomic_id, S_user in branch_user_segs[bi]:
                if S_user is None or len(S_user) < 2:
                    continue
                kept_meta.append((bi, atomic_id, S_user, bool(rank == 0)))
                kept_len += _linestring_length(S_user)
            branch_stats.append({
                "branch_id": int(bi),
                "rank": int(rank),
                "atomic_ids": [aid for (aid, _) in branch_user_segs[bi]],
                "user_len": float(branch_user_len[bi]),
                "kept_len": float(kept_len),
                "suppressed": False,
            })
            claimed |= branch_terr
            continue

        # -------------------------------------------------
        # Suppression gate
        # -------------------------------------------------
        terr_len = float(branch_user_len[bi])
        min_len_px = 0.10 * branch_user_len[bi]

        if rank > 0 and terr_len < min_len_px:
            branch_stats.append({
                "branch_id": int(bi),
                "rank": int(rank),
                "atomic_ids": [aid for (aid, _) in branch_user_segs[bi]],
                "user_len": float(branch_user_len[bi]),
                "kept_len": 0.0,
                "suppressed": True,
            })
            claimed |= branch_terr
            continue

        # -------------------------------------------------
        # PRIMARY branch
        # -------------------------------------------------
        if rank == 0:
            kept_len = 0.0
            for atomic_id, S_user in branch_user_segs[bi]:
                if S_user is not None and len(S_user) >= 2:
                    kept_meta.append((bi, atomic_id, S_user, True))
                    kept_len += _linestring_length(S_user)

            branch_stats.append({
                "branch_id": int(bi),
                "rank": int(rank),
                "atomic_ids": [aid for (aid, _) in branch_user_segs[bi]],
                "user_len": float(branch_user_len[bi]),
                "kept_len": float(kept_len),
                "suppressed": False,
            })

            claimed |= branch_terr
            continue

        # -------------------------------------------------
        # SUBORDINATE branches
        # -------------------------------------------------
        
        # -------------------------------------------------
        # Hard exclusion: dominant territory + dominant masks
        # -------------------------------------------------
        terr_forbidden = claimed.copy()

        mask_forbidden = np.zeros((H, W), np.uint8)
        for obi in order[:rank]:
            mask_forbidden |= branch_atomic_masks[obi]

        forbidden = terr_forbidden | mask_forbidden

        # -------------------------------------------------
        # Bite decomposition (DEBUG-ONLY semantics)
        # -------------------------------------------------
        bite_mask_only = branch_terr & mask_forbidden
        bite_terr_only = branch_terr & terr_forbidden & (~mask_forbidden)
        bite_both = branch_terr & terr_forbidden & mask_forbidden

        # accumulate for debug plots
        if bi not in bite_by_losing_branch:
            bite_by_losing_branch[bi] = {
                "mask": np.zeros((H, W), np.uint8),
                "territory": np.zeros((H, W), np.uint8),
                "both": np.zeros((H, W), np.uint8),
            }
            
        # record per-branch losing bite (AUTHORITATIVE for pruning later)
        bite_by_losing_branch[bi]["mask"]      |= bite_mask_only
        bite_by_losing_branch[bi]["territory"] |= bite_terr_only
        bite_by_losing_branch[bi]["both"]      |= bite_both
        
        if np.any(bite_mask_only):
            bite_total["mask"] |= bite_mask_only
        if np.any(bite_terr_only):
            bite_total["terr"] |= bite_terr_only
        if np.any(bite_both):
            bite_total["both"] |= bite_both

        allowed = (forbidden == 0).astype(np.uint8)

        kept_any = False
        kept_len = 0.0

        for atomic_id, S_user in branch_user_segs[bi]:
            if S_user is None or len(S_user) < 2:
                continue

            pieces = _clip_polyline_to_mask(S_user, allowed.astype(np.uint8))
            for p in pieces:
                if p is not None and len(p) >= 2:
                    _plen = _linestring_length(p)
                    if _plen < 3.0:
                        print(f"[DOM_CLIP_FILTER] branch={bi} atomic={atomic_id} "
                              f"piece dropped: len_px={_plen:.1f} < 3px minimum", flush=True)
                        continue
                    kept_meta.append((bi, atomic_id, p, False))
                    kept_any = True
                    kept_len += _plen

        branch_stats.append({
            "branch_id": int(bi),
            "rank": int(rank),
            "atomic_ids": [aid for (aid, _) in branch_user_segs[bi]],
            "user_len": float(branch_user_len[bi]),
            "kept_len": float(kept_len),
            "suppressed": not kept_any,
        })

        claimed |= branch_terr

    # -----------------------------
    # Pack outputs + meta
    # -----------------------------
    branch_to_items = {}
    for (bi, atomic_id, S, is_primary) in kept_meta:
        if S is None or len(S) < 2:
            continue
        branch_to_items.setdefault(int(bi), []).append(
            {
                "atomic_id": str(atomic_id),
                "is_primary": bool(is_primary),
                "seg": np.asarray(S, float),
            }
        )

    def _proximity_subgroups(items, max_jump=10.0):
        """Split branch items into connected components by endpoint proximity."""
        segs = [np.asarray((it or {}).get("seg", []), float) for it in (items or [])]
        n = len(segs)
        if n <= 1:
            return [list(items or [])]
        adj = {i: set() for i in range(n)}
        for i in range(n):
            si = segs[i]
            if si.ndim != 2 or si.shape[1] != 2 or len(si) < 2:
                continue
            for j in range(i + 1, n):
                sj = segs[j]
                if sj.ndim != 2 or sj.shape[1] != 2 or len(sj) < 2:
                    continue
                dmin = min(
                    float(np.linalg.norm(si[0] - sj[0])),
                    float(np.linalg.norm(si[0] - sj[-1])),
                    float(np.linalg.norm(si[-1] - sj[0])),
                    float(np.linalg.norm(si[-1] - sj[-1])),
                )
                if dmin <= float(max_jump):
                    adj[i].add(j)
                    adj[j].add(i)
        visited = set()
        groups = []
        for i in range(n):
            if i in visited:
                continue
            comp = []
            stack = [i]
            while stack:
                u = stack.pop()
                if u in visited:
                    continue
                visited.add(u)
                comp.append(u)
                stack.extend(list(adj[u] - visited))
            groups.append([items[k] for k in comp])
        return groups

    def _build_branch_territory_from_items(items):
        """Rebuild branch territory from kept segment items (global frame)."""
        branch_terr = np.zeros((H, W), np.uint8)
        for it in (items or []):
            S_user = np.asarray((it or {}).get("seg", []), float)
            if S_user.ndim != 2 or S_user.shape[1] != 2 or len(S_user) < 2:
                continue
            r = seg_radius(S_user)
            rad = int(max(4, min(2.0 * r, window_half_size)))
            line = _polyline_mask(S_user, H, W)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1)
            )
            terr = cv2.dilate(line, kernel, iterations=1)
            branch_terr |= terr
        if crack_mask is not None:
            branch_terr &= crack_mask
        if crack_mask is not None and np.any(branch_terr):
            num_labels, labels = cv2.connectedComponents(
                (branch_terr > 0).astype(np.uint8), connectivity=8
            )
            if int(num_labels) > 2:
                seg_mask = np.zeros((H, W), np.uint8)
                for it in (items or []):
                    S_user = np.asarray((it or {}).get("seg", []), float)
                    if S_user.ndim == 2 and S_user.shape[1] == 2 and len(S_user) >= 2:
                        seg_mask |= _polyline_mask(S_user, H, W)
                seg_mask = cv2.dilate(seg_mask, np.ones((3, 3), np.uint8), iterations=1)
                keep = np.zeros((H, W), np.uint8)
                for lab in range(1, int(num_labels)):
                    if np.any((labels == lab) & (seg_mask > 0)):
                        keep |= (labels == lab).astype(np.uint8)
                if np.any(keep):
                    branch_terr = keep
        return branch_terr

    # Validate post-clip stitchability per branch and split disconnected pieces.
    ordered_branch_ids = [int(b) for b in order]
    ordered_branch_ids += [b for b in sorted(branch_to_items.keys()) if b not in ordered_branch_ids]
    next_synth_bid = (max(ordered_branch_ids) + 1) if ordered_branch_ids else 100
    expanded_branch_to_items = {}
    expanded_order = []
    branch_stats_map = {
        int((r or {}).get("branch_id", -1)): (r if isinstance(r, dict) else {})
        for r in (branch_stats or [])
        if int((r or {}).get("branch_id", -1)) >= 0
    }
    for bi in ordered_branch_ids:
        items = branch_to_items.get(int(bi), [])
        if not items:
            continue
        segs_b = [np.asarray((it or {}).get("seg", []), float) for it in items]
        _, ok_chain, _reason_chain = shared_stitch_branch_segments(
            segs_b,
            max_jump=10.0,
            allow_teleport=False,
        )
        if ok_chain or len(items) <= 1:
            expanded_branch_to_items[int(bi)] = items
            expanded_order.append(int(bi))
            continue

        subgroups = _proximity_subgroups(items, max_jump=10.0)
        print(
            f"[DOM_SPLIT] branch={int(bi)} post-clip stitch failed "
            f"(reason={_reason_chain}) - splitting into {len(subgroups)} sub-branches"
        )
        first_used_parent = False
        parent_stat = branch_stats_map.get(int(bi), {})
        stolen_aids = set()   # track atomic_ids claimed by synthetic sub-branches
        for sg in subgroups:
            if not sg:
                continue
            if not first_used_parent:
                out_bid = int(bi)
                first_used_parent = True
            else:
                out_bid = int(next_synth_bid)
                next_synth_bid += 1
                if int(bi) in bite_by_losing_branch:
                    bsrc = bite_by_losing_branch[int(bi)]
                    bite_by_losing_branch[out_bid] = {
                        "mask": np.asarray(bsrc.get("mask", 0), np.uint8).copy(),
                        "territory": np.asarray(bsrc.get("territory", 0), np.uint8).copy(),
                        "both": np.asarray(bsrc.get("both", 0), np.uint8).copy(),
                    }
            expanded_branch_to_items[out_bid] = sg
            expanded_order.append(int(out_bid))
            if out_bid != int(bi):
                aids = []
                seen_a = set()
                kept_len = 0.0
                for it in sg:
                    a = str((it or {}).get("atomic_id", ""))
                    if a and a not in seen_a:
                        aids.append(a)
                        seen_a.add(a)
                        stolen_aids.add(a)
                    seg = np.asarray((it or {}).get("seg", []), float)
                    if seg.ndim == 2 and seg.shape[1] == 2 and len(seg) >= 2:
                        kept_len += float(_linestring_length(seg))
                branch_stats.append(
                    {
                        "branch_id": int(out_bid),
                        "rank": int(parent_stat.get("rank", -1)),
                        "atomic_ids": aids,
                        "user_len": float(parent_stat.get("user_len", kept_len)),
                        "kept_len": float(kept_len),
                        "suppressed": False,
                    }
                )
            else:
                # Update the parent branch's stats to reflect only this subgroup's atomics.
                parent_aids = []
                seen_pa = set()
                parent_kept_len = 0.0
                for it in sg:
                    a = str((it or {}).get("atomic_id", ""))
                    if a and a not in seen_pa:
                        parent_aids.append(a)
                        seen_pa.add(a)
                    seg = np.asarray((it or {}).get("seg", []), float)
                    if seg.ndim == 2 and seg.shape[1] == 2 and len(seg) >= 2:
                        parent_kept_len += float(_linestring_length(seg))
                for r in branch_stats:
                    if isinstance(r, dict) and int(r.get("branch_id", -1)) == int(bi):
                        r["atomic_ids"] = parent_aids
                        r["kept_len"]   = float(parent_kept_len)
                        break
            # Ensure territory exists for both parent-retained and synthetic branches.
            syn_terr = _build_branch_territory_from_items(sg)
            branch_terr_masks[int(out_bid)] = syn_terr
            sg_segs = [np.asarray((it or {}).get("seg", []), float) for it in (sg or [])]
            sg_segs = [a for a in sg_segs if a.ndim == 2 and a.shape[1] == 2 and len(a) >= 2]
            if sg_segs:
                sg_polyline = np.vstack(sg_segs)
                terr_ys, terr_xs = np.where(syn_terr > 0)
                if len(terr_xs):
                    tx0, ty0 = int(terr_xs.min()), int(terr_ys.min())
                    tx1, ty1 = int(terr_xs.max()) + 1, int(terr_ys.max()) + 1
                else:
                    tx0, ty0, tx1, ty1 = 0, 0, 1, 1
                print(
                    f"[SYNTH_TERR_DBG] out_bid={int(out_bid)} H={int(H)} W={int(W)} "
                    f"terr_nnz={int(np.sum(syn_terr > 0))} "
                    f"bbox=[{tx0},{ty0},{int(tx1 - tx0)},{int(ty1 - ty0)}] "
                    f"seg_pts={int(len(sg_polyline))} "
                    f"seg_x=[{float(sg_polyline[:,0].min()):.1f},{float(sg_polyline[:,0].max()):.1f}] "
                    f"seg_y=[{float(sg_polyline[:,1].min()):.1f},{float(sg_polyline[:,1].max()):.1f}]"
                )
    branch_to_items = expanded_branch_to_items

    canonical_kept = []
    ordered_branch_ids = [int(b) for b in expanded_order]
    ordered_branch_ids += [b for b in sorted(branch_to_items.keys()) if b not in ordered_branch_ids]
    order = [int(b) for b in ordered_branch_ids]

    for bi in ordered_branch_ids:
        items = branch_to_items.get(int(bi), [])
        if not items:
            continue

        segs_b = [it["seg"] for it in items]
        assoc_b = [{"atomic_id": it["atomic_id"], "is_primary": it["is_primary"]} for it in items]

        segs_b, assoc_b = enforce_branch_continuity(segs_b, associated_data=assoc_b)
        segs_b, assoc_b, flipped_branch = canonicalize_branch_direction(segs_b, associated_data=assoc_b)
        if flipped_branch:
            if _VERBOSE_DEBUG:
                print(f"[CANON] dominant_segments branch={bi} flipped whole branch orientation")
        try:
            assert_direction_consistency(segs_b)
        except AssertionError as e:
            # Defensive repair: some branches still arrive with a locally reversed segment.
            # Flip only the offending segment(s) based on endpoint continuity, then re-check.
            if _VERBOSE_DEBUG:
                print(f"[CANON] dominant_segments branch={bi} repairing direction inconsistency: {e}")
            segs_b, assoc_b = enforce_branch_continuity(segs_b, associated_data=assoc_b)
            try:
                assert_direction_consistency(segs_b)
            except AssertionError:
                # Final local pairwise pass (handles edge cases after whole-branch canonicalization).
                for j in range(1, len(segs_b)):
                    a_prev = np.asarray(segs_b[j - 1], float)
                    b_cur = np.asarray(segs_b[j], float)
                    if len(a_prev) < 2 or len(b_cur) < 2:
                        continue
                    d_fwd = float(np.linalg.norm(a_prev[-1] - b_cur[0]))
                    d_rev = float(np.linalg.norm(a_prev[-1] - b_cur[-1]))
                    if d_rev < d_fwd:
                        segs_b[j] = b_cur[::-1].copy()
                assert_direction_consistency(segs_b)

        for seg_idx, (S, a) in enumerate(zip(segs_b, assoc_b)):
            canonical_kept.append(
                (int(bi), str(a.get("atomic_id")), np.asarray(S, float), bool(a.get("is_primary", False)), int(seg_idx))
            )

    kept = [S for (_, _, S, _, _) in canonical_kept]

    segments_meta = []
    for (bi, atomic_id, S, is_primary, seg_idx) in canonical_kept:
        segments_meta.append({
            "branch_id": int(bi),
            "atomic_id": str(atomic_id),
            "seg_idx": int(seg_idx),
            "is_primary": bool(is_primary),
            "branch_rank": int(order.index(bi)) if bi in order else -1,
            "length": float(_linestring_length(S)),
        })

    # -------------------------------------------------
    # BITE EXPORT (AUTHORITATIVE)
    # -------------------------------------------------
    # Compose overall bite (union of all dominance causes)
    bite_union = (
        bite_total["mask"] |
        bite_total["terr"] |
        bite_total["both"]
    ).astype(np.uint8)

    bite_crop = bite_union[by0:by1, bx0:bx1]
    bite_blob = _pack_mask_b64(bite_crop)

    # Optional: also export decomposed bite channels (DEBUG / RESEARCH)
    bite_mask_crop = bite_total["mask"][by0:by1, bx0:bx1].astype(np.uint8)
    bite_terr_crop = bite_total["terr"][by0:by1, bx0:bx1].astype(np.uint8)
    bite_both_crop = bite_total["both"][by0:by1, bx0:bx1].astype(np.uint8)

    bite_mask_blob = _pack_mask_b64(bite_mask_crop)
    bite_terr_blob = _pack_mask_b64(bite_terr_crop)
    bite_both_blob = _pack_mask_b64(bite_both_crop)

    bite_by_losing_branch_export = {}

    for bi, d in bite_by_losing_branch.items():
        # union for that losing branch
        bu = (d["mask"] | d["territory"] | d["both"]).astype(np.uint8)
        if not np.any(bu):
            continue

        bu_crop = bu[by0:by1, bx0:bx1].astype(np.uint8)
        blob = _pack_mask_b64(bu_crop)

        bite_by_losing_branch_export[str(int(bi))] = {
            "shape": blob["shape"],
            "packbits_b64": blob["packbits_b64"],
            # optional: export per-cause too (handy for debug)
            "by_cause": {
                "mask": _pack_mask_b64(d["mask"][by0:by1, bx0:bx1].astype(np.uint8)),
                "territory": _pack_mask_b64(d["territory"][by0:by1, bx0:bx1].astype(np.uint8)),
                "both": _pack_mask_b64(d["both"][by0:by1, bx0:bx1].astype(np.uint8)),
            },
        }
    
    if primary_branch is not None:
        assert str(int(primary_branch)) not in bite_by_losing_branch_export

    # -------------------------------------------------
    # Branch territory export (AUTHORITATIVE for GT mask restriction)
    # -------------------------------------------------
    branch_territory_export = {}
    for bi, terr in branch_terr_masks.items():
        terr_u8 = (terr > 0).astype(np.uint8)
        if not np.any(terr_u8):
            continue
        # Tight bbox in GLOBAL coords (full HxW frame), then crop for packing.
        ys, xs = np.where(terr_u8 > 0)
        tx0, ty0 = int(xs.min()), int(ys.min())
        tx1, ty1 = int(xs.max()) + 1, int(ys.max()) + 1
        terr_crop = terr_u8[ty0:ty1, tx0:tx1].astype(np.uint8)
        terr_blob = _pack_mask_b64(terr_crop)
        branch_territory_export[str(int(bi))] = {
            "bbox": [int(tx0), int(ty0), int(tx1 - tx0), int(ty1 - ty0)],
            "shape": terr_blob["shape"],
            "packbits_b64": terr_blob["packbits_b64"],
        }

    
    meta = {
        "members": [str(m) for m in members],
        "primary_branch_id": int(primary_branch) if primary_branch is not None else None,
        "order": [int(b) for b in order],
        "branches": branch_stats,
        "segments_meta": segments_meta,
        "branch_territory": branch_territory_export,

        # -------------------------------------------------
        # Bite metadata
        # -------------------------------------------------
        "bite": {
            # Backward-compatible UNION bite
            "bbox": [int(bx0), int(by0), int(bx1 - bx0), int(by1 - by0)],
            "shape": bite_blob["shape"],
            "packbits_b64": bite_blob["packbits_b64"],

            # Decomposed dominance causes (NEW, OPTIONAL)
            "by_cause": {
                "mask": {
                    "shape": bite_mask_blob["shape"],
                    "packbits_b64": bite_mask_blob["packbits_b64"],
                },
                "territory": {
                    "shape": bite_terr_blob["shape"],
                    "packbits_b64": bite_terr_blob["packbits_b64"],
                },
                "both": {
                    "shape": bite_both_blob["shape"],
                    "packbits_b64": bite_both_blob["packbits_b64"],
                },
            },
            "by_losing_branch": bite_by_losing_branch_export,
        },
    }

    
    # -------------------------------------------------
    # 5 PLOT BASE MASK:
    #   - if crack_mask exists: use it
    #   - else: union of atomic masks (global) for this group
    # -------------------------------------------------
    if crack_mask is not None:
        plot_mask = crack_mask.astype(np.uint8)
    else:
        plot_mask = np.zeros((H, W), np.uint8)
        for m in members:
            cr = atomic.get(str(m), {}) or {}
            am = _atomic_mask_global(cr, H, W)
            if am is not None:
                plot_mask |= (am > 0).astype(np.uint8)


    # -----------------------------
    # DEBUG: PRE (territory + bite + user) via shared helper
    # -----------------------------
    if debug_dir:
        plot_branch_territory_debug_pre(
            debug_dir=debug_dir,
            debug_tag=debug_tag,
            plot_mask=plot_mask,
            branch_terr_masks=branch_terr_masks,
            branch_user_segs=branch_user_segs,
            branch_user_len=branch_user_len,
            bite_total=bite_total,
            bx0=bx0,
            by0=by0,
            bx1=bx1,
            by1=by1,
            H=H,
            W=W,
        )

    # -----------------------------
    # DEBUG: FINAL (user dashed if clipped, kept solid, labels kept/user)
    # -----------------------------
    if debug_dir:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.lines import Line2D
        import numpy as np
        import os

        pad = 20
        x0 = max(0, bx0 - pad)
        y0 = max(0, by0 - pad)
        x1 = min(W, bx1 + pad)
        y1 = min(H, by1 + pad)

        Hc, Wc = y1 - y0, x1 - x0

        fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
        ax.imshow(plot_mask[y0:y1, x0:x1], cmap="gray", zorder=0)

        # same 5 colors as PRE (identity only)
        branch_colors = [
            "#F77F00",  # vivid orange
            "#2DC653",  # bright green
            "#CC00CC",  # magenta
            "#FFD100",  # vivid yellow
            "#00BCD4",  # cyan
            "#7B2FBE",  # deep violet
            "#8BC34A",  # lime
            "#A0522D",  # sienna brown
            "#FF5C8A",  # rose pink
            "#009688",  # teal
        ]

        # --- bite (make it visible) ---
        bm = bite_total["mask"][y0:y1, x0:x1]
        bt = bite_total["terr"][y0:y1, x0:x1]
        bb = bite_total["both"][y0:y1, x0:x1]

        if np.any(bm) or np.any(bt) or np.any(bb):
            bite_rgba = np.zeros((Hc, Wc, 4), float)

            # mask-only → red
            bite_rgba[..., 0] += (bm > 0)

            # territory-only → orange (red + green)
            bite_rgba[..., 0] += 0.9 * (bt > 0)
            bite_rgba[..., 1] += 0.5 * (bt > 0)

            # both → purple (red + blue)
            bite_rgba[..., 0] += (bb > 0)
            bite_rgba[..., 2] += (bb > 0)

            bite_rgba[..., 3] = 0.45 * ((bm | bt | bb) > 0)

            ax.imshow(bite_rgba, zorder=2)

        # --- dashed USER (ONLY if clipped happened) ---
        for bi, user_segs in enumerate(branch_user_segs):
            user_len = branch_user_len[bi]
            kept_len = sum(
                _linestring_length(S)
                for (bii, _, S, _) in kept_meta
                if bii == bi
            )
            if kept_len >= 0.999 * user_len:
                continue

            for _, S in user_segs:
                if S is None or len(S) < 2:
                    continue
                S2 = S - np.array([x0, y0])
                ax.plot(
                    S2[:, 0], S2[:, 1],
                    color="white",
                    lw=1.6,
                    linestyle=(0, (1, 3)),  # dotty
                    alpha=0.9,
                    zorder=3,
                )

        # --- kept geometry (primary on top) ---
        # draw sub first, then primary last so it overlays
        for pass_primary in (False, True):
            for bi, _, S, is_primary in kept_meta:
                if bool(is_primary) != pass_primary:
                    continue
                if S is None or len(S) < 2:
                    continue
                S2 = S - np.array([x0, y0])
                ax.plot(
                    S2[:, 0], S2[:, 1],
                    color=branch_colors[bi % len(branch_colors)],
                    lw=3.8 if is_primary else 2.8,
                    solid_capstyle="round",
                    zorder=6 if is_primary else 4,
                )

        # --- per-branch labels: "user" OR "kept/user" if clipped ---
        for bi, user_segs in enumerate(branch_user_segs):
            if not user_segs:
                continue

            user_len = branch_user_len[bi]
            kept_len = sum(
                _linestring_length(S)
                for (bii, _, S, _) in kept_meta
                if bii == bi
            )

            if kept_len < 0.999 * user_len:
                label = f"{int(kept_len)}/{int(user_len)} px"
            else:
                label = f"{int(user_len)} px"

            longest = max(user_segs, key=lambda t: _linestring_length(t[1]))[1]
            mid = longest[len(longest) // 2]
            if not (x0 <= mid[0] <= x1 and y0 <= mid[1] <= y1):
                continue
            mx, my = mid - np.array([x0, y0])

            ax.text(
                mx, my,
                label,
                fontsize=7,
                color=branch_colors[bi % len(branch_colors)],
                fontweight="bold",
                ha="center",
                va="center",
                zorder=20,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.75,
                ),
            )

        # --- bbox ---
        ax.add_patch(
            Rectangle(
                (bx0 - x0, by0 - y0),
                bx1 - bx0,
                by1 - by0,
                fill=False,
                edgecolor="#0033cc",
                linewidth=1.5,
                zorder=30,
            )
        )

        legend_items = [
            Line2D([0], [0], color="white", lw=2, linestyle=(0, (1, 3)),
                label="Clipped user midlines"),

            Line2D([0], [0], color="black", lw=3,
                label="Kept midlines (branch-colored)"),

            Line2D([0], [0], color="red", lw=6, alpha=0.55,
                label="Bite: mask"),

            Line2D([0], [0], color="#e67e22", lw=6, alpha=0.55,
                label="Bite: territory"),

            Line2D([0], [0], color="#d16ba5", lw=6, alpha=0.55,
                label="Bite: mask + territory"),

            Line2D([0], [0], color="#0033cc", lw=1.5,
                label="BBox"),
        ]
        
        leg = ax.legend(
            handles=legend_items,
            loc="upper right",
            borderaxespad=0.5,
            fontsize=11,
            framealpha=0.85,
        )
        leg.set_zorder(100)

        ax.set_title(f"{debug_tag} — FINAL", fontsize=10)
        ax.axis("off")

        out = os.path.join(debug_dir, f"{debug_tag}_final.png")
        fig.savefig(out, dpi=100, bbox_inches="tight")
        plt.close(fig)

    # -----------------------------
    # INSTRUMENTATION: segment audit
    # -----------------------------
    if debug_dir:
        bad = []
        for i, S in enumerate(kept):
            if S is None:
                bad.append((i, "None"))
            elif not isinstance(S, np.ndarray):
                bad.append((i, f"type={type(S)}"))
            elif S.ndim != 2:
                bad.append((i, f"ndim={S.ndim}"))
            elif S.shape[1] != 2:
                bad.append((i, f"shape={S.shape}"))
            elif len(S) < 3:
                bad.append((i, f"len={len(S)}"))

        if bad:
            print("\n[DOMINANCE AUDIT] ⚠ problematic kept segments:")
            for i, reason in bad:
                m = segments_meta[i] if i < len(segments_meta) else {}
                print(
                    f"  seg#{i}: {reason}, "
                    f"branch={m.get('branch_id')}, "
                    f"rank={m.get('branch_rank')}, "
                    f"atomic={m.get('atomic_id')}, "
                    f"is_primary={m.get('is_primary')}, "
                    f"len_px={m.get('length'):.2f}"
                )
        for i, a in enumerate(atomics):
            _c_log(3, f"[SEG {i}] start={a['poly'][0]} end={a['poly'][-1]} len={a['length']:.1f}")
    
    return kept, meta


def plot_branch_territory_debug_pre(
    *,
    debug_dir,
    debug_tag,
    plot_mask,
    branch_terr_masks,
    branch_user_segs,
    branch_user_len,
    bite_total,
    bx0,
    by0,
    bx1,
    by1,
    H,
    W,
    out_filename=None,
):
    """
    Shared PRE renderer for branch territory + bite + user geometry.
    Intended to be reusable by both dominance and derived-geometry debug flows.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D

    os.makedirs(debug_dir, exist_ok=True)

    pad = 30
    x0 = max(0, int(bx0) - pad)
    y0 = max(0, int(by0) - pad)
    x1 = min(int(W), int(bx1) + pad)
    y1 = min(int(H), int(by1) + pad)

    Hc, Wc = y1 - y0, x1 - x0

    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    ax.imshow(plot_mask[y0:y1, x0:x1], cmap="gray", zorder=0)

    branch_colors = [
        np.array([0.97, 0.50, 0.00]),  # vivid orange
        np.array([0.18, 0.78, 0.33]),  # bright green
        np.array([0.80, 0.00, 0.80]),  # magenta
        np.array([1.00, 0.82, 0.00]),  # vivid yellow
        np.array([0.00, 0.74, 0.83]),  # cyan
        np.array([0.48, 0.18, 0.75]),  # deep violet
        np.array([0.55, 0.76, 0.29]),  # lime
        np.array([0.63, 0.32, 0.18]),  # sienna brown
        np.array([1.00, 0.36, 0.54]),  # rose pink
        np.array([0.00, 0.59, 0.53]),  # teal
    ]

    for bi, terr in branch_terr_masks.items():
        bm = terr[y0:y1, x0:x1]
        if not np.any(bm):
            continue
        rgba = np.zeros((Hc, Wc, 4), float)
        rgba[..., :3] = branch_colors[int(bi) % len(branch_colors)]
        rgba[..., 3] = 0.25 * (bm > 0)
        ax.imshow(rgba, zorder=1)

    order_by_len = sorted(range(len(branch_user_segs)), key=lambda i: branch_user_len[i])
    for rank, bi in enumerate(order_by_len):
        col = branch_colors[int(bi) % len(branch_colors)]
        z = 3 + rank

        for _, S in branch_user_segs[bi]:
            if S is None or len(S) < 2:
                continue
            if (
                S[:, 0].max() < x0 or S[:, 0].min() > x1 or
                S[:, 1].max() < y0 or S[:, 1].min() > y1
            ):
                continue
            S2 = S - np.array([x0, y0])
            ax.plot(
                S2[:, 0], S2[:, 1],
                color=col,
                lw=3.2,
                alpha=0.95,
                solid_capstyle="round",
                zorder=z,
            )

        if branch_user_segs[bi]:
            longest = max(branch_user_segs[bi], key=lambda t: _linestring_length(t[1]))[1]
            mid = longest[len(longest) // 2]
            if x0 <= mid[0] <= x1 and y0 <= mid[1] <= y1:
                mx, my = mid - np.array([x0, y0])
                ax.text(
                    mx, my,
                    f"{int(branch_user_len[bi])} px",
                    fontsize=8,
                    color=branch_colors[bi % len(branch_colors)],
                    fontweight="bold",
                    ha="center",
                    va="center",
                    zorder=z + 0.2,
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.75,
                    ),
                )

    bm = bite_total["mask"][y0:y1, x0:x1]
    bt = bite_total["terr"][y0:y1, x0:x1]
    bb = bite_total["both"][y0:y1, x0:x1]

    if np.any(bm) or np.any(bt) or np.any(bb):
        bite_rgba = np.zeros((Hc, Wc, 4), float)
        bite_rgba[..., 0] += (bm > 0)                     # red
        bite_rgba[..., 0] += 0.9 * (bt > 0)              # orange
        bite_rgba[..., 1] += 0.5 * (bt > 0)
        bite_rgba[..., 0] += (bb > 0)                    # purple
        bite_rgba[..., 2] += (bb > 0)
        bite_rgba[..., 3] = 0.45 * ((bm | bt | bb) > 0)
        ax.imshow(bite_rgba, zorder=2)

    ax.add_patch(
        Rectangle(
            (int(bx0) - x0, int(by0) - y0),
            int(bx1 - bx0),
            int(by1 - by0),
            fill=False,
            edgecolor="#0033cc",
            linewidth=1.5,
            zorder=10,
        )
    )

    legend_items = [
        Line2D([0], [0], color="white", lw=2, linestyle=(0, (1, 3)),
               label="Clipped user midlines"),
        Line2D([0], [0], color="black", lw=3,
               label="Kept midlines (branch-colored)"),
        Line2D([0], [0], color="red", lw=6, alpha=0.55,
               label="Bite: mask"),
        Line2D([0], [0], color="#e67e22", lw=6, alpha=0.55,
               label="Bite: territory"),
        Line2D([0], [0], color="#d16ba5", lw=6, alpha=0.55,
               label="Bite: mask + territory"),
        Line2D([0], [0], color="#0033cc", lw=1.5, label="BBox"),
    ]

    leg = ax.legend(
        handles=legend_items,
        loc="upper right",
        borderaxespad=0.5,
        fontsize=11,
        framealpha=0.85,
    )
    leg.set_zorder(100)

    ax.set_title(f"{debug_tag} — PRE", fontsize=10)
    ax.axis("off")

    out = os.path.join(debug_dir, out_filename or f"{debug_tag}_pre.png")
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)




def gt_groups_from_midlines_and_gtmask(atomic: dict, gt_mask, H, W):
    """
    Pure GT grouping:
    Group atomic cracks if:
      (1) Their midlines fall into the SAME connected component of the GT mask, OR
      (2) They share a start/end user endpoint.

    Does NOT use:
        - manual_mask_from_crack
        - auto grouping heuristics
        - proximity or overlap logic
        - geodesic edges
        - snapshot data
    """

    import numpy as np
    import cv2

    # ------------------------------------
    # 1) Label connected components in GT
    # ------------------------------------
    gt_bin = (gt_mask > 0).astype(np.uint8)
    num_labels, cc_map = cv2.connectedComponents(gt_bin)

    # ------------------------------------
    # 2) Determine for each atomic:
    #     - which GT CC its midline occupies
    #     - which user endpoints it has
    # ------------------------------------
    cid_to_cc = {}
    cid_to_endpoints = {}

    def get_user_endpoints(cr):
        ups = cr.get("user_points", []) or []
        ucs = cr.get("user_connections", []) or []
        endpoints = set()
        for pair in ucs:
            for idx in pair:
                if 0 <= idx < len(ups):
                    endpoints.add(tuple(map(float, ups[idx])))
        return endpoints

    for cid, cr in atomic.items():
        cid_str = str(cid)

        mid = np.asarray(cr.get("midline", []), float)
        if mid.ndim == 2 and len(mid) >= 1:
            ys = np.clip(mid[:, 1].round().astype(int), 0, H-1)
            xs = np.clip(mid[:, 0].round().astype(int), 0, W-1)
            cc_labels = cc_map[ys, xs]
            # pick the most common CC label (ignoring background 0)
            nonzero = cc_labels[cc_labels > 0]
            if len(nonzero):
                # mode of the CC labels
                vals, counts = np.unique(nonzero, return_counts=True)
                cid_to_cc[cid_str] = int(vals[np.argmax(counts)])
            else:
                # midline did not land inside GT mask
                cid_to_cc[cid_str] = None
        else:
            cid_to_cc[cid_str] = None

        cid_to_endpoints[cid_str] = get_user_endpoints(cr)

    # ------------------------------------
    # 3) Build adjacency graph:
    #     A ~ B if:
    #       - cc(A) == cc(B) and != None
    #       - endpoints(A) ∩ endpoints(B) != ∅
    # ------------------------------------
    ids = sorted(cid_to_cc.keys())
    adj = {cid: set() for cid in ids}

    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            a, b = ids[i], ids[j]

            same_cc = (cid_to_cc[a] is not None and
                       cid_to_cc[a] == cid_to_cc[b])

            shared_endpoints = bool(cid_to_endpoints[a] & cid_to_endpoints[b])

            if same_cc or shared_endpoints:
                adj[a].add(b)
                adj[b].add(a)

    # ------------------------------------
    # 4) Connected components of adjacency graph
    # ------------------------------------
    visited = set()
    groups = []
    for cid in ids:
        if cid in visited:
            continue
        comp = []
        stack = [cid]
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            comp.append(u)
            stack.extend(adj[u])
        if len(comp) >= 2:
            groups.append(sorted(comp, key=lambda x: int(x)))

    return {
        str(i): {"members": g}
        for i, g in enumerate(groups)
    }

def plot_combined_debug(
    *,
    original_image,
    segs,
    derived_midline_segs=None,
    edge1_segs,
    edge2_segs,
    norm1_segs,
    norm2_segs,
    mask_bbox,
    member_ids,
    out_dir,
):
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "edges_midlines_normals_pretty.png")

    # Expand bbox to cover all plotted geometry, not just the union mask.
    expanded_bbox = mask_bbox
    if mask_bbox is not None:
        all_pts = []
        for seg_list in [segs, derived_midline_segs, edge1_segs, edge2_segs]:
            for seg in (seg_list or []):
                arr = np.asarray(seg, float)
                if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) >= 2:
                    finite = arr[np.isfinite(arr).all(axis=1)]
                    if len(finite):
                        all_pts.append(finite)
        if all_pts:
            pts = np.vstack(all_pts)
            x0, y0, w, h = mask_bbox
            x_min = min(x0, int(np.floor(pts[:, 0].min())))
            y_min = min(y0, int(np.floor(pts[:, 1].min())))
            x_max = max(x0 + w, int(np.ceil(pts[:, 0].max())))
            y_max = max(y0 + h, int(np.ceil(pts[:, 1].max())))
            expanded_bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

    plot_edges_and_normals(
        base_image=original_image,
        midline_segs=segs,
        derived_midline_segs=derived_midline_segs or [],
        edge1_segs=edge1_segs,
        edge2_segs=edge2_segs,
        norm1_segs=norm1_segs,
        norm2_segs=norm2_segs,
        bbox=expanded_bbox,
        out_png=out_png,
        title=f"Combined Crack (members={', '.join(member_ids)})",
    )


def recompute_dominance_geometry_from_derived(
    *,
    dominance_meta,
    branch_to_derived_runs,
    atomic,
    H,
    W,
    derived_domain_mask,
    window_half_size,
    members=None,
    debug_dir=None,
    debug_tag="derived_geometry_dominance",
):
    """
    Geometry-only recomputation on derived midlines.
    Topology stays fixed to dominance_meta:
      - same order
      - same branch IDs
      - no clipping/suppression decisions here
    """
    import base64
    import numpy as np
    import cv2

    def _pack_mask_b64(mask_u8):
        m = (np.asarray(mask_u8) > 0).astype(np.uint8)
        if m.size == 0:
            return {"shape": [0, 0], "packbits_b64": ""}
        packed = np.packbits(m, axis=1)
        b64 = base64.b64encode(packed.tobytes()).decode("ascii")
        return {"shape": [int(m.shape[0]), int(m.shape[1])], "packbits_b64": b64}

    if not isinstance(dominance_meta, dict):
        raise AssertionError("dominance_meta must be a dict")

    order = [int(x) for x in (dominance_meta.get("order") or [])]
    if not order:
        raise AssertionError("dominance_meta.order missing/empty")

    primary_branch_id = dominance_meta.get("primary_branch_id", None)
    if primary_branch_id is not None:
        primary_branch_id = int(primary_branch_id)
        if primary_branch_id not in order:
            raise AssertionError("primary_branch_id not in dominance order")

    dom_branches = dominance_meta.get("branches") or []
    branch_rows = {}
    for r in dom_branches:
        if isinstance(r, dict) and "branch_id" in r:
            branch_rows[int(r["branch_id"])] = r

    dom_branch_ids = set(branch_rows.keys())
    order_ids = set(order)
    if dom_branch_ids != order_ids:
        raise AssertionError(
            f"Dominance branch IDs and order mismatch: "
            f"branches={sorted(dom_branch_ids)} order={sorted(order_ids)}"
        )

    derived_ids = set(int(k) for k in (branch_to_derived_runs or {}).keys())
    if not derived_ids.issubset(order_ids):
        raise AssertionError(
            f"Derived branch IDs outside dominance order: "
            f"{sorted(derived_ids - order_ids)}"
        )

    domain_mask = (np.asarray(derived_domain_mask) > 0).astype(np.uint8)
    if domain_mask.shape[:2] != (H, W):
        raise AssertionError("[derived dominance] derived_domain_mask shape mismatch")
    if not np.any(domain_mask):
        raise AssertionError("[derived dominance] empty derived_domain_mask")

    dt = cv2.distanceTransform(domain_mask, cv2.DIST_L2, 5)

    def seg_radius(S):
        S = np.asarray(S, float)
        ys = np.clip(np.round(S[:, 1]).astype(int), 0, H - 1)
        xs = np.clip(np.round(S[:, 0]).astype(int), 0, W - 1)
        d = dt[ys, xs]
        d = d[np.isfinite(d)]
        if len(d) == 0:
            return 0.3 * float(window_half_size)
        return max(3.0, min(float(np.median(d)), float(window_half_size)))

    branch_atomic_masks = {}
    for bi in order:
        aids = [str(a) for a in (branch_rows[bi].get("atomic_ids", []) or [])]
        m = np.zeros((H, W), np.uint8)
        for aid in aids:
            cr = atomic.get(str(aid), {}) or {}
            am = _atomic_mask_global(cr, H, W)
            if am is not None:
                m |= (am > 0).astype(np.uint8)
        branch_atomic_masks[int(bi)] = m

    branch_terr_masks = {}
    branch_rows_out = []
    for rank, bi in enumerate(order):
        terr = np.zeros((H, W), np.uint8)
        runs = branch_to_derived_runs.get(int(bi), []) or []
        derived_len = 0.0
        derived_pts = 0

        for S in runs:
            S = np.asarray(S, float)
            if S.ndim != 2 or S.shape[1] != 2 or len(S) < 2:
                continue

            derived_len += float(_linestring_length(S))
            derived_pts += int(len(S))

            r = seg_radius(S)
            rad = int(max(4, min(1.5 * r, float(window_half_size))))
            line = _polyline_mask(S, H, W)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1)
            )
            terr |= cv2.dilate(line, kernel, iterations=1)

        branch_terr_masks[int(bi)] = terr

        src = branch_rows[int(bi)]
        branch_rows_out.append({
            "branch_id": int(bi),
            "rank": int(rank),
            "atomic_ids": [str(a) for a in (src.get("atomic_ids", []) or [])],
            "user_len": float(src.get("user_len", 0.0)),
            "kept_len": float(src.get("kept_len", 0.0)),
            "suppressed": bool(src.get("suppressed", False)),
            "derived_len": float(derived_len),
            "derived_run_count": int(len(runs)),
            "derived_point_count": int(derived_pts),
        })

    claimed = np.zeros((H, W), np.uint8)
    bite_total = {
        "mask": np.zeros((H, W), np.uint8),
        "terr": np.zeros((H, W), np.uint8),
        "both": np.zeros((H, W), np.uint8),
    }
    bite_by_losing_branch = {}

    for rank, bi in enumerate(order):
        bi = int(bi)
        branch_terr = branch_terr_masks.get(bi, np.zeros((H, W), np.uint8))

        if rank == 0:
            claimed |= branch_terr
            continue

        terr_forbidden = claimed.copy()
        mask_forbidden = np.zeros((H, W), np.uint8)
        for obi in order[:rank]:
            mask_forbidden |= branch_atomic_masks.get(int(obi), 0).astype(np.uint8)

        bite_mask_only = branch_terr & mask_forbidden
        bite_terr_only = branch_terr & terr_forbidden & (~mask_forbidden)
        bite_both = branch_terr & terr_forbidden & mask_forbidden

        bite_by_losing_branch[bi] = {
            "mask": bite_mask_only.astype(np.uint8),
            "territory": bite_terr_only.astype(np.uint8),
            "both": bite_both.astype(np.uint8),
        }

        bite_total["mask"] |= bite_mask_only
        bite_total["terr"] |= bite_terr_only
        bite_total["both"] |= bite_both
        claimed |= branch_terr

    bite_old = (dominance_meta.get("bite") or {}) if isinstance(dominance_meta, dict) else {}
    bb = bite_old.get("bbox")
    if isinstance(bb, (list, tuple)) and len(bb) == 4:
        bx0, by0, bw, bh = map(int, bb)
        bx1, by1 = bx0 + bw, by0 + bh
    else:
        bx0, by0, bx1, by1 = 0, 0, W, H

    bx0 = max(0, min(bx0, W)); bx1 = max(0, min(bx1, W))
    by0 = max(0, min(by0, H)); by1 = max(0, min(by1, H))
    if bx1 <= bx0 or by1 <= by0:
        bx0, by0, bx1, by1 = 0, 0, W, H

    bite_union = (bite_total["mask"] | bite_total["terr"] | bite_total["both"]).astype(np.uint8)
    bite_blob = _pack_mask_b64(bite_union[by0:by1, bx0:bx1])

    bite_by_losing_branch_export = {}
    for bi, d in bite_by_losing_branch.items():
        bu = (d["mask"] | d["territory"] | d["both"]).astype(np.uint8)
        if not np.any(bu):
            continue
        bu_crop = bu[by0:by1, bx0:bx1]
        blob = _pack_mask_b64(bu_crop)
        bite_by_losing_branch_export[str(int(bi))] = {
            "shape": blob["shape"],
            "packbits_b64": blob["packbits_b64"],
            "by_cause": {
                "mask": _pack_mask_b64(d["mask"][by0:by1, bx0:bx1]),
                "territory": _pack_mask_b64(d["territory"][by0:by1, bx0:bx1]),
                "both": _pack_mask_b64(d["both"][by0:by1, bx0:bx1]),
            },
        }

    if primary_branch_id is not None:
        assert str(int(primary_branch_id)) not in bite_by_losing_branch_export

    terr_by_branch_export = {}
    for bi in order:
        terr = branch_terr_masks.get(int(bi), np.zeros((H, W), np.uint8))
        terr_crop = terr[by0:by1, bx0:bx1]
        terr_by_branch_export[str(int(bi))] = _pack_mask_b64(terr_crop)

    derived_geometry_meta = {
        "members": [str(m) for m in (members or dominance_meta.get("members") or [])],
        "primary_branch_id": primary_branch_id,
        "order": [int(b) for b in order],
        "branches": branch_rows_out,
        "bite": {
            "bbox": [int(bx0), int(by0), int(bx1 - bx0), int(by1 - by0)],
            "shape": bite_blob["shape"],
            "packbits_b64": bite_blob["packbits_b64"],
            "by_cause": {
                "mask": _pack_mask_b64(bite_total["mask"][by0:by1, bx0:bx1]),
                "territory": _pack_mask_b64(bite_total["terr"][by0:by1, bx0:bx1]),
                "both": _pack_mask_b64(bite_total["both"][by0:by1, bx0:bx1]),
            },
            "by_losing_branch": bite_by_losing_branch_export,
        },
        "territory_by_branch": terr_by_branch_export,
        "diagnostics": {
            "domain": "derived_domain_mask",
            "window_half_size": int(window_half_size),
        },
    }

    assert [int(b) for b in derived_geometry_meta["order"]] == [int(b) for b in dominance_meta.get("order", [])]
    assert {int(b["branch_id"]) for b in derived_geometry_meta["branches"]} == set(order)

    if debug_dir:
        max_bid = max(order) if order else -1
        branch_geom_segs = [[] for _ in range(max_bid + 1)]
        branch_geom_len = [0.0 for _ in range(max_bid + 1)]
        for bi in order:
            runs = branch_to_derived_runs.get(int(bi), []) or []
            branch_geom_segs[int(bi)] = [(None, np.asarray(S, float)) for S in runs if S is not None and len(S) >= 2]
            branch_geom_len[int(bi)] = float(sum(_linestring_length(np.asarray(S, float)) for S in runs if S is not None and len(S) >= 2))

        plot_mask = domain_mask.astype(np.uint8)
        plot_branch_territory_debug_pre(
            debug_dir=debug_dir,
            debug_tag=debug_tag,
            plot_mask=plot_mask,
            branch_terr_masks=branch_terr_masks,
            branch_user_segs=branch_geom_segs,
            branch_user_len=branch_geom_len,
            bite_total=bite_total,
            bx0=bx0,
            by0=by0,
            bx1=bx1,
            by1=by1,
            H=H,
            W=W,
            out_filename="derived_geometry_dominance.png",
        )

    return derived_geometry_meta
    
def union_bboxes(bboxes, *, pad=5):
    """
    bboxes: iterable of [x0, y0, x1, y1] (x1/y1 EXCLUSIVE)
    returns: (x0, y0, x1, y1) clamped to image bounds
    """
    xs0, ys0, xs1, ys1 = [], [], [], []

    for bb in bboxes:
        if not bb or len(bb) != 4:
            continue
        x0, y0, x1, y1 = map(int, bb)
        if x1 > x0 and y1 > y0:
            xs0.append(x0)
            ys0.append(y0)
            xs1.append(x1)
            ys1.append(y1)

    if not xs0:
        return 0, 0, W, H

    x0 = max(min(xs0) - pad, 0)
    y0 = max(min(ys0) - pad, 0)
    x1 = min(max(xs1) + pad, W)
    y1 = min(max(ys1) + pad, H)

    return x0, y0, x1, y1

def debug_bbox_only(
    *,
    debug_dir,
    branch_id,
    run_id,
    gray_full,
    bbox,
    S_run,
):
    if debug_dir is None:
        return

    import os
    import cv2
    import numpy as np

    os.makedirs(debug_dir, exist_ok=True)

    x0, y0, x1, y1 = bbox

    # -------------------------------------------------
    # SAFE visualization conversion (NO side effects)
    # -------------------------------------------------
    if gray_full.dtype != np.uint8:
        g = gray_full
        # robust normalize → uint8
        g = g.astype(np.float32)
        g = g - np.min(g)
        mx = np.max(g)
        if mx > 0:
            g = g / mx
        g = (255.0 * g).clip(0, 255).astype(np.uint8)
    else:
        g = gray_full

    vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    # -------------------------------------------------
    # Draw bbox
    # -------------------------------------------------
    cv2.rectangle(
        vis,
        (int(x0), int(y0)),
        (int(x1), int(y1)),
        (255, 255, 0),   # cyan
        2,
    )

    # -------------------------------------------------
    # Draw midline
    # -------------------------------------------------
    for i in range(len(S_run) - 1):
        p1 = tuple(np.round(S_run[i]).astype(int))
        p2 = tuple(np.round(S_run[i + 1]).astype(int))
        cv2.line(vis, p1, p2, (255, 255, 255), 1)

    # -------------------------------------------------
    # Draw endpoints
    # -------------------------------------------------
    p_start = tuple(np.round(S_run[0]).astype(int))
    p_end   = tuple(np.round(S_run[-1]).astype(int))
    cv2.circle(vis, p_start, 3, (0, 255, 0), -1)
    cv2.circle(vis, p_end,   3, (0, 0, 255), -1)

    out_path = os.path.join(
        debug_dir,
        f"bbox_branch{branch_id}_run{run_id}.png"
    )
    cv2.imwrite(out_path, vis)

import numpy as np

def _safe_float_xy(p):
    p = np.asarray(p, float).reshape(2)
    return np.array([float(p[0]), float(p[1])], dtype=float)

def _pt_key(p, tol):
    p = _safe_float_xy(p)
    return (int(np.round(p[0] / tol)), int(np.round(p[1] / tol)))

def _order_midline_branch_chain(segs, metas, *, tol=2.0):
    """
    Order + orient segs into a single chain using endpoint matching.
    Returns ordered_segs, ordered_metas, chain_start_xy, chain_end_xy
    """
    segs2, metas2 = [], []
    for S, m in zip(segs, metas):
        if S is None:
            continue
        S = np.asarray(S, float)
        if S.ndim != 2 or S.shape[1] != 2 or len(S) < 2:
            continue
        segs2.append(S)
        metas2.append(m if isinstance(m, dict) else {})

    if not segs2:
        return [], [], None, None
    if len(segs2) == 1:
        S0 = segs2[0]
        return segs2, metas2, _safe_float_xy(S0[0]), _safe_float_xy(S0[-1])

    ends = [(_pt_key(S[0], tol), _pt_key(S[-1], tol)) for S in segs2]
    ep_map = {}
    for i, (ka, kb) in enumerate(ends):
        ep_map.setdefault(ka, []).append((i, 0))
        ep_map.setdefault(kb, []).append((i, 1))

    degrees = {k: len(v) for k, v in ep_map.items()}
    start_key = None
    for k, deg in degrees.items():
        if deg == 1:
            start_key = k
            break
    if start_key is None:
        start_key = ends[0][0]

    used = set()
    candidates = ep_map.get(start_key, [])
    if not candidates:
        S0 = segs2[0]
        return segs2, metas2, _safe_float_xy(S0[0]), _safe_float_xy(S0[-1])

    seg_idx, _side = candidates[0]
    S0 = segs2[seg_idx]
    if _pt_key(S0[0], tol) != start_key:
        S0 = S0[::-1].copy()

    chain_segs = [S0]
    chain_metas = [metas2[seg_idx]]
    used.add(seg_idx)
    cur_key = _pt_key(chain_segs[-1][-1], tol)

    while True:
        nxt = None
        for (j, _side_j) in ep_map.get(cur_key, []):
            if j in used:
                continue
            nxt = j
            break
        if nxt is None:
            break

        Sj = segs2[nxt]
        if _pt_key(Sj[0], tol) != cur_key:
            Sj = Sj[::-1].copy()

        chain_segs.append(Sj)
        chain_metas.append(metas2[nxt])
        used.add(nxt)
        cur_key = _pt_key(chain_segs[-1][-1], tol)

    chain_start = _safe_float_xy(chain_segs[0][0])
    chain_end = _safe_float_xy(chain_segs[-1][-1])
    return chain_segs, chain_metas, chain_start, chain_end

def _nearest_index(poly_xy, q_xy):
    poly = np.asarray(poly_xy, float)
    q = _safe_float_xy(q_xy)
    d2 = np.sum((poly - q[None, :]) ** 2, axis=1)
    return int(np.argmin(d2))

def _split_derived_by_atomic_junctions(
    *,
    branch_id: int,
    derived_run_xy: np.ndarray,
    mid_segs: list,
    mid_meta: list,
    tol_shared=2.5,   # kept for signature compatibility; not used in arclen method
):
    """
    Split ONE derived branch polyline into per-atomic segments.

    NEW: split by arclength FRACTIONS along the ordered midline chain,
         NOT by XY proximity to junction points.

    This is robust for AUTO where derived midline may not pass near the
    user junction XY in the middle of the crack.
    """
    import numpy as np

    if derived_run_xy is None:
        raise ValueError(f"[DERIVED SPLIT] branch={branch_id} derived_run_xy is None")

    D = np.asarray(derived_run_xy, float)
    if D.ndim != 2 or D.shape[1] != 2 or len(D) < 2:
        raise ValueError(f"[DERIVED SPLIT] branch={branch_id} derived_run invalid shape/len")

    chain_segs, chain_metas, chain_start, chain_end = _order_midline_branch_chain(
        mid_segs, mid_meta, tol=tol_shared
    )
    if not chain_segs:
        return [D], [{
            "branch_id": int(branch_id),
            "atomic_id": None,
            "seg_idx": 0,
            "source": "derived_unsplit_no_midline",
        }]

    # ------------------------------------------------------------
    # SAFETY REPAIR:
    # If _order_midline_branch_chain loses atomic_id diversity,
    # rebuild the chain deterministically by seg_idx (when present).
    # This avoids "derived_unsplit_single_atomic" when the input
    # mid_meta clearly contains multiple atomics.
    # ------------------------------------------------------------
    try:
        in_aids = []
        for mm in (mid_meta or []):
            if isinstance(mm, dict):
                aid = mm.get("atomic_id", None)
                if aid is not None:
                    in_aids.append(str(aid))
        in_unique = sorted(set(in_aids))

        chain_aids = []
        for mm in (chain_metas or []):
            if isinstance(mm, dict):
                aid = mm.get("atomic_id", None)
                if aid is not None:
                    chain_aids.append(str(aid))
        chain_unique = sorted(set(chain_aids))

        if len(in_unique) >= 2 and len(chain_unique) <= 1:
            print(
                f"[DERIVED SPLIT WARN] branch={branch_id} "
                f"_order_midline_branch_chain collapsed atomic_ids "
                f"in_unique={in_unique} chain_unique={chain_unique} -> rebuilding by seg_idx"
            )

            pairs = []
            for Sx, mx in zip(mid_segs or [], mid_meta or []):
                if Sx is None or len(Sx) < 2:
                    continue
                mm = mx if isinstance(mx, dict) else {}
                sidx = mm.get("seg_idx", None)
                try:
                    sidx = int(sidx) if sidx is not None else None
                except Exception:
                    sidx = None
                pairs.append((sidx, np.asarray(Sx, float), dict(mm)))

            if pairs and all(p[0] is not None for p in pairs):
                pairs.sort(key=lambda t: int(t[0]))
            else:
                pairs.sort(key=lambda t: 10**9 if t[0] is None else int(t[0]))

            chain_segs = [p[1] for p in pairs]
            chain_metas = [p[2] for p in pairs]

            for i in range(1, len(chain_segs)):
                a_prev = np.asarray(chain_segs[i - 1], float)
                b_cur = np.asarray(chain_segs[i], float)
                if len(a_prev) < 2 or len(b_cur) < 2:
                    continue
                d_fwd = float(np.linalg.norm(a_prev[-1] - b_cur[0]))
                d_rev = float(np.linalg.norm(a_prev[-1] - b_cur[-1]))
                if d_rev < d_fwd:
                    chain_segs[i] = b_cur[::-1].copy()

            if chain_segs:
                chain_start = np.asarray(chain_segs[0][0], float)
                chain_end = np.asarray(chain_segs[-1][-1], float)
    except Exception as _e:
        print(f"[DERIVED SPLIT WARN] branch={branch_id} chain repair failed: {_e}")

    # --- Ensure derived direction matches chain direction (same as your current code) ---
    d0 = _safe_float_xy(D[0]); d1 = _safe_float_xy(D[-1])
    cs = _safe_float_xy(chain_start); ce = _safe_float_xy(chain_end)
    forward_score = float(np.linalg.norm(d0 - cs) + np.linalg.norm(d1 - ce))
    reverse_score = float(np.linalg.norm(d0 - ce) + np.linalg.norm(d1 - cs))
    if reverse_score + 1e-6 < forward_score:
        D = D[::-1].copy()

    # --- Extract atomic sequence in chain order ---
    atomic_seq = []
    for m in chain_metas:
        aid = m.get("atomic_id", None) if isinstance(m, dict) else None
        atomic_seq.append(str(aid) if aid is not None else None)

    unique_aids = [a for a in atomic_seq if a is not None]
    if len(set(unique_aids)) <= 1:
        try:
            _c_log(3, f"[DERIVED SPLIT DBG] branch={branch_id} atomic_seq={atomic_seq} unique_aids={sorted(set(unique_aids))}")
            _c_log(3, (
                f"[DERIVED SPLIT DBG] branch={branch_id} chain_metas_atomic="
                f"{[((m if isinstance(m, dict) else {}).get('atomic_id'), (m if isinstance(m, dict) else {}).get('seg_idx')) for m in (chain_metas or [])]}"
            ))
        except Exception as _e:
            _c_log(3, f"[DERIVED SPLIT DBG] branch={branch_id} chain atomic debug failed: {_e}")
        aid0 = unique_aids[0] if unique_aids else None
        return [D], [{
            "branch_id": int(branch_id),
            "atomic_id": aid0,
            "seg_idx": 0,
            "source": "derived_unsplit_single_atomic",
        }]

    # --- Helper: arclength cumulative ---
    def _cumlen(xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, float)
        d = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
        return np.concatenate([[0.0], np.cumsum(d)])

    # --- Compute cut FRACTIONS along the ordered MIDLINE chain ---
    seg_lens = []
    for S in chain_segs:
        S = np.asarray(S, float)
        if len(S) < 2:
            seg_lens.append(0.0)
        else:
            c = _cumlen(S)
            seg_lens.append(float(c[-1]))

    total_L = float(np.sum(seg_lens))
    if total_L <= 1e-9:
        # degenerate midline chain -> cannot fraction-split
        return [D], [{
            "branch_id": int(branch_id),
            "atomic_id": unique_aids[0] if unique_aids else None,
            "seg_idx": 0,
            "source": "derived_unsplit_midline_zero_length",
        }]

    # Boundaries at end of each segment except the last: [L1, L1+L2, ...] / total_L
    cum_end = np.cumsum(seg_lens)
    cut_fracs = [float(c / total_L) for c in cum_end[:-1] if np.isfinite(c)]

    # --- Convert those fractions to indices on derived polyline by derived arclength ---
    cD = _cumlen(D)
    LD = float(cD[-1])
    if LD <= 1e-9:
        raise ValueError(f"[DERIVED SPLIT] branch={branch_id} derived polyline zero length")

    sD = cD / LD  # normalized [0..1]

    # Turn each cut fraction into an index in D (nearest in normalized arclength)
    raw_cut_idx = []
    for frac in cut_fracs:
        # clamp to interior to avoid empty pieces
        frac = float(np.clip(frac, 1e-6, 1.0 - 1e-6))
        idx = int(np.argmin(np.abs(sD - frac)))
        raw_cut_idx.append(idx)

    # --- De-dup / enforce strict increasing cut indices (preserve atomic boundary index) ---
    # raw_cut_idx[k] is the boundary between atomic_seq[k] and atomic_seq[k+1].
    cut_pairs = []  # (derived_idx, seq_pos)
    last = -10**9
    for seq_pos, raw_idx in enumerate(raw_cut_idx):
        if raw_idx <= last + 1:
            continue
        cut_pairs.append((int(raw_idx), int(seq_pos)))
        last = int(raw_idx)

    if not cut_pairs:
        # Nothing usable -> return unsplit (still better than “missing derived segment” fatal)
        return [D], [{
            "branch_id": int(branch_id),
            "atomic_id": unique_aids[0] if unique_aids else None,
            "seg_idx": 0,
            "source": "derived_unsplit_no_valid_cuts",
        }]

    # --- Emit segments with atomic ids aligned to chain_metas order ---
    out_segs = []
    out_meta = []

    start_i = 0
    seg_counter = 0

    for idx, seq_pos in cut_pairs:
        end_i = int(idx) + 1
        if end_i - start_i >= 2:
            piece = D[start_i:end_i]
            aid = atomic_seq[seq_pos] if seq_pos < len(atomic_seq) else atomic_seq[-1]
            out_segs.append(piece)
            out_meta.append({
                "branch_id": int(branch_id),
                "atomic_id": str(aid) if aid is not None else None,
                "seg_idx": int(seg_counter),
                "source": "derived_split_by_midline_arclen_fraction",
            })
            seg_counter += 1
        start_i = int(idx)

    # tail
    if len(D) - start_i >= 2:
        last_seq_pos = (cut_pairs[-1][1] + 1) if cut_pairs else 0
        aid = atomic_seq[last_seq_pos] if last_seq_pos < len(atomic_seq) else atomic_seq[-1]
        out_segs.append(D[start_i:])
        out_meta.append({
            "branch_id": int(branch_id),
            "atomic_id": str(aid) if aid is not None else None,
            "seg_idx": int(seg_counter),
            "source": "derived_split_by_midline_arclen_fraction",
        })

    if not out_segs:
        raise RuntimeError(f"[DERIVED SPLIT] branch={branch_id} produced zero output segments")

    # DEBUG: verify split output (segment count, atomic assignments, endpoints)
    try:
        _c_log(3, f"[DERIVED SPLIT DBG] branch={branch_id} out_segs={len(out_segs)} out_meta={len(out_meta)}")
        for i, (S, m) in enumerate(zip(out_segs, out_meta)):
            S = np.asarray(S, float)
            mm = m if isinstance(m, dict) else {}
            aid = mm.get("atomic_id", None)
            if S.ndim == 2 and S.shape[1] == 2 and len(S) >= 1:
                _c_log(3, (
                    f"[DERIVED SPLIT DBG]  seg{i}: atomic_id={aid} n={len(S)} "
                    f"start=({float(S[0,0]):.3f},{float(S[0,1]):.3f}) "
                    f"end=({float(S[-1,0]):.3f},{float(S[-1,1]):.3f})"
                ))
            else:
                _c_log(3, f"[DERIVED SPLIT DBG]  seg{i}: atomic_id={aid} invalid_shape={getattr(S, 'shape', None)}")
    except Exception as _e:
        _c_log(3, f"[DERIVED SPLIT DBG] branch={branch_id} failed: {_e}")

    return out_segs, out_meta

def build_combined_crack_stateless(
    original_image: np.ndarray,
    authoring_atomic: dict,
    member_ids: list[str],
    *,
    window_half_size: int = 50,
    mu: float = 0.0,
    l: int = 5,
    p: int = 14,
    color_channel: int = 0,
    pad: int = 10,
    prefer_gpu: bool = True,
    debug_dir=None,
    debug_callback=None,
    crack_mask_full: np.ndarray = None,
    mode="new",
    is_auto: bool = False,
):
    """
    Stateless “metrics-safe” combiner with fully instrumented timing.
    """
    import time
    import numpy as np

    t0 = time.perf_counter()

    img = original_image
    H, W = img.shape[:2]

    if img.ndim == 3:
        bgr_idx = {0: 2, 1: 0, 2: 1}.get(color_channel, 2)
        gray_full = img[:, :, bgr_idx].astype(np.float32)
    else:
        gray_full = img.astype(np.float32)

    # ---------------------
    # Timing: stitching
    # ---------------------
    t_stitch0 = time.perf_counter()
        
    H, W = original_image.shape[:2]

    if crack_mask_full is None:
        raise ValueError("build_combined_crack_stateless: crack_mask_full is required")

    if crack_mask_full.shape[:2] != (H, W):
        raise ValueError(
            "build_combined_crack_stateless: crack_mask_full shape mismatch"
        )

    crack_mask_full = (crack_mask_full > 0).astype(np.uint8)

    # --------------------------------------------------------
    # Dominance + branch-aware execution (WITH run-splitting)
    # --------------------------------------------------------
    segs, dom_meta = dominant_segments_from_group(
        members=member_ids,
        atomic=authoring_atomic,
        crack_mask_u8=crack_mask_full,
        window_half_size=window_half_size,
        debug_dir=debug_dir,
        debug_tag="dominance_grouping",
        original_image=original_image,
    )

    midline_segments_meta = dom_meta.get("segments_meta", [])
    dominance_meta = dom_meta

    t_stitch1 = time.perf_counter()
    stitching_sec = float(t_stitch1 - t_stitch0)

    # --------------------------------------------------------
    # Group segments by branch_id
    # --------------------------------------------------------
    from collections import defaultdict

    branch_to_segs = defaultdict(list)
    branch_to_segmeta = defaultdict(list)

    # Be robust if lengths ever diverge
    for i, S in enumerate(segs):
        if S is None or len(S) < 2:
            continue
        if i >= len(midline_segments_meta):
            # fallback: treat as its own "unknown" branch
            bid = -1
            mm = {}
        else:
            mm = midline_segments_meta[i] if isinstance(midline_segments_meta[i], dict) else {}
            bid = int(mm.get("branch_id", -1))
        branch_to_segs[bid].append(np.asarray(S, float))
        branch_to_segmeta[bid].append(dict(mm) if isinstance(mm, dict) else {})

    for bid, seg_list in branch_to_segs.items():
        if len(seg_list) >= 2:
            assert_direction_consistency(seg_list)

    # --------------------------------------------------------
    # Helper: split on discontinuities (CRITICAL)
    # --------------------------------------------------------
    def split_polyline_on_jumps(S, max_step=5.0):
        S = np.asarray(S, float)
        if S.ndim != 2 or S.shape[1] != 2 or len(S) < 2:
            return []
        d = np.sqrt(np.sum(np.diff(S, axis=0) ** 2, axis=1))
        breaks = np.where(d > float(max_step))[0]
        out = []
        s = 0
        for b in breaks:
            if b + 1 - s >= 2:
                out.append(S[s : b + 1])
            s = b + 1
        if len(S) - s >= 2:
            out.append(S[s:])
        return out

    # --------------------------------------------------------
    # Outputs
    # --------------------------------------------------------
    edge1_segs, edge2_segs = [], []
    norm1_segs, norm2_segs = [], []
    derived_midline_segs = []
    branch_to_derived_runs = defaultdict(list)
    union_mask = np.zeros((H, W), np.uint8)
    all_widths = []

    t_masks_total = 0.0
    t_edges_total = 0.0
    t_post_total  = 0.0
    t_loop_total  = 0.0

    _c_log(1, f"[COMBINER] branches = {len(branch_to_segs)}")

    # --------------------------------------------------------
    # Process EACH BRANCH as ONE CHAIN (preferred)
    # --------------------------------------------------------
    for branch_id, seg_list in branch_to_segs.items():
        t_branch0 = time.perf_counter()

        if not seg_list:
            continue

        seg_meta_list = list(branch_to_segmeta.get(branch_id, []))

        def _user_endpoint_points(cr):
            pts = []
            if not isinstance(cr, dict):
                return np.zeros((0, 2), float)
            ups = cr.get("user_points", []) or []
            ucs = cr.get("user_connections", []) or []
            seen = set()
            for pair in ucs:
                try:
                    for idx in pair:
                        ii = int(idx)
                        if 0 <= ii < len(ups):
                            p = ups[ii]
                            key = (float(p[0]), float(p[1]))
                            if key not in seen:
                                seen.add(key)
                                pts.append([key[0], key[1]])
                except Exception:
                    continue
            return np.asarray(pts, float) if pts else np.zeros((0, 2), float)

        def _nearest_pt_info(p, pts):
            try:
                p = np.asarray(p, float)
                pts = np.asarray(pts, float)
                if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
                    return None, None
                d = np.sqrt(np.sum((pts - p[None, :]) ** 2, axis=1))
                j = int(np.argmin(d))
                return j, float(d[j])
            except Exception:
                return None, None

        def _branch_endpoint_degree_diag(segs_in, tol=2.0):
            """
            Cluster segment endpoints and report node degrees to detect non-chain branches.
            """
            endpoints = []
            for si, S in enumerate(segs_in or []):
                S = np.asarray(S, float)
                if S.ndim != 2 or S.shape[1] != 2 or len(S) < 2:
                    continue
                endpoints.append((si, "start", np.asarray(S[0], float)))
                endpoints.append((si, "end", np.asarray(S[-1], float)))

            if len(endpoints) < 2:
                return

            clusters = []
            for si, tag, p in endpoints:
                assigned = False
                for c in clusters:
                    if float(np.linalg.norm(p - c["center"])) <= float(tol):
                        c["items"].append((si, tag, p))
                        P = np.vstack([it[2] for it in c["items"]])
                        c["center"] = np.mean(P, axis=0)
                        assigned = True
                        break
                if not assigned:
                    clusters.append({"center": p.copy(), "items": [(si, tag, p)]})

            deg = [len(c["items"]) for c in clusters]
            odd = int(sum(1 for d in deg if (d % 2) == 1))
            max_deg = int(max(deg)) if deg else 0
            chain_like = bool(max_deg <= 2 and (odd == 0 or odd == 2))
            nodes_preview = [
                (
                    round(float(c["center"][0]), 3),
                    round(float(c["center"][1]), 3),
                    int(len(c["items"])),
                )
                for c in clusters
            ]
            _c_log(
                1,
                f"[TOPO DEGREE] branch={branch_id} nodes={len(clusters)} "
                f"odd={odd} max_deg={max_deg} chain_like={chain_like}",
            )
            if not chain_like:
                if _VERBOSE_DEBUG:
                    print(
                        f"[TOPO DEGREE WARN] branch={branch_id} appears non-chain; "
                        f"stitch may require teleport or fail in manual mode."
                    )

        # Debug endpoint drift vs declared user topology before stitching.
        if len(seg_list) >= 2:
            try:
                _c_log(3, f"[TOPO PRE-STITCH] branch={branch_id} n_segs={len(seg_list)}")
                _branch_endpoint_degree_diag(seg_list, tol=2.0)
                for i, S in enumerate(seg_list):
                    S = np.asarray(S, float)
                    mm = seg_meta_list[i] if i < len(seg_meta_list) and isinstance(seg_meta_list[i], dict) else {}
                    aid = str(mm.get("atomic_id")) if mm.get("atomic_id", None) is not None else None
                    cr = (authoring_atomic.get(aid, {}) or {}) if aid is not None else {}
                    ups = np.asarray(cr.get("user_points", []) or [], float)
                    ueps = _user_endpoint_points(cr)
                    p0 = np.asarray(S[0], float)
                    p1 = np.asarray(S[-1], float)
                    u0_idx, u0_d = _nearest_pt_info(p0, ups)
                    u1_idx, u1_d = _nearest_pt_info(p1, ups)
                    e0_idx, e0_d = _nearest_pt_info(p0, ueps)
                    e1_idx, e1_d = _nearest_pt_info(p1, ueps)
                    _c_log(3, (
                        f"[TOPO PRE-STITCH] seg{i} atomic={aid} n={len(S)} "
                        f"start=({p0[0]:.3f},{p0[1]:.3f}) end=({p1[0]:.3f},{p1[1]:.3f})"
                    ))
                    _c_log(3, (
                        f"[TOPO PRE-STITCH] seg{i} nearest_userpt start=(idx={u0_idx},d={u0_d}) "
                        f"end=(idx={u1_idx},d={u1_d})"
                    ))
                    _c_log(3, (
                        f"[TOPO PRE-STITCH] seg{i} nearest_user_endpoint start=(idx={e0_idx},d={e0_d}) "
                        f"end=(idx={e1_idx},d={e1_d}) n_user_endpoints={len(ueps)}"
                    ))

                for i in range(len(seg_list)):
                    for j in range(i + 1, len(seg_list)):
                        mi = seg_meta_list[i] if i < len(seg_meta_list) and isinstance(seg_meta_list[i], dict) else {}
                        mj = seg_meta_list[j] if j < len(seg_meta_list) and isinstance(seg_meta_list[j], dict) else {}
                        ai = str(mi.get("atomic_id")) if mi.get("atomic_id", None) is not None else None
                        aj = str(mj.get("atomic_id")) if mj.get("atomic_id", None) is not None else None
                        cri = (authoring_atomic.get(ai, {}) or {}) if ai is not None else {}
                        crj = (authoring_atomic.get(aj, {}) or {}) if aj is not None else {}
                        ei = _user_endpoint_points(cri)
                        ej = _user_endpoint_points(crj)
                        shared = []
                        if len(ei) and len(ej):
                            for pi in ei:
                                d = np.sqrt(np.sum((ej - pi[None, :]) ** 2, axis=1))
                                if np.any(d <= 1e-6):
                                    shared.append((float(pi[0]), float(pi[1])))
                        _c_log(3, (
                            f"[TOPO PRE-STITCH] pair i={i}({ai}) j={j}({aj}) "
                            f"shared_user_endpoints={len(shared)} {shared[:4]}"
                        ))
            except Exception as _e:
                _c_log(3, f"[TOPO PRE-STITCH] debug failed branch={branch_id}: {_e}")

        # ====================================================
        # 0) Stitch segments into ONE continuous polyline
        #    - Connectivity is assumed (dominance/user topology already decided)
        #    - Ordering only
        #    - AUTO ONLY: snap/join to nearest declared USER endpoint (within snap_px)
        #    - HARD FAILS: no run-splitting fallback
        # ====================================================

        def _user_endpoint_points(cr):
            """
            Returns unique USER endpoints referenced by user_connections (indices into user_points).
            """
            pts = []
            if not isinstance(cr, dict):
                return np.zeros((0, 2), float)
            ups = cr.get("user_points", []) or []
            ucs = cr.get("user_connections", []) or []
            seen = set()
            for pair in ucs:
                try:
                    for idx in pair:
                        ii = int(idx)
                        if 0 <= ii < len(ups):
                            p = ups[ii]
                            key = (float(p[0]), float(p[1]))
                            if key not in seen:
                                seen.add(key)
                                pts.append([key[0], key[1]])
                except Exception:
                    continue
            return np.asarray(pts, float) if pts else np.zeros((0, 2), float)

        def _nearest_user_endpoint(p, ueps):
            """
            Returns (pt, dist) for nearest endpoint in ueps to p, or (None, inf).
            """
            p = np.asarray(p, float)
            ueps = np.asarray(ueps, float)
            if ueps.ndim != 2 or ueps.shape[1] != 2 or len(ueps) == 0:
                return None, float("inf")
            d = np.sqrt(np.sum((ueps - p[None, :]) ** 2, axis=1))
            j = int(np.argmin(d))
            return np.asarray(ueps[j], float), float(d[j])

        # ====================================================
        # 0) Stitch segments into ONE continuous polyline
        # ====================================================

        def stitch_branch_segments(seg_list, max_jump=10.0, *, allow_teleport=False):
            """
            Order + orient segments into a single chain.

            Assumes connectivity is already guaranteed by topology (branch_id + dominance).
            Does NOT modify geometry (no snapping, no insertion).
            Teleports are either:
            - treated as failure (allow_teleport=False)
            - tolerated with warnings (allow_teleport=True)
            """

            segs = [np.asarray(s, float) for s in seg_list if s is not None and len(s) >= 2]
            if not segs:
                return None, False, "no valid segs"
            for si, ss in enumerate(segs):
                _dbg_pts(f"branch{branch_id}_seg{si}", ss, level=1)

            used = set()
            chain = [segs[0]]
            used.add(0)

            while len(used) < len(segs):
                cur_end = np.asarray(chain[-1][-1], float)
                cur_start = np.asarray(chain[0][0], float)

                best_j = None
                best_dist = float("inf")
                best_oriented = None
                best_flipped = False
                best_action = None  # "append" or "prepend"

                for j, sj in enumerate(segs):
                    if j in used:
                        continue

                    sj = np.asarray(sj, float)
                    a = sj[0]
                    b = sj[-1]

                    # Append candidates: connect chain end -> candidate start
                    d_app_as_is = float(np.linalg.norm(cur_end - a))
                    d_app_rev = float(np.linalg.norm(cur_end - b))

                    if d_app_as_is < best_dist:
                        best_dist = d_app_as_is
                        best_j = j
                        best_oriented = sj
                        best_flipped = False
                        best_action = "append"

                    if d_app_rev < best_dist:
                        best_dist = d_app_rev
                        best_j = j
                        best_oriented = sj[::-1].copy()
                        best_flipped = True
                        best_action = "append"

                    # Prepend candidates: connect candidate end -> chain start
                    d_pre_as_is = float(np.linalg.norm(b - cur_start))
                    d_pre_rev = float(np.linalg.norm(a - cur_start))
                    _c_log(
                        1,
                        f"[DBG_DIST] branch={branch_id} cand={j} "
                        f"app_fwd={d_app_as_is:.3f} app_rev={d_app_rev:.3f} "
                        f"pre_fwd={d_pre_as_is:.3f} pre_rev={d_pre_rev:.3f}",
                    )

                    if d_pre_as_is < best_dist:
                        best_dist = d_pre_as_is
                        best_j = j
                        best_oriented = sj
                        best_flipped = False
                        best_action = "prepend"

                    if d_pre_rev < best_dist:
                        best_dist = d_pre_rev
                        best_j = j
                        best_oriented = sj[::-1].copy()
                        best_flipped = True
                        best_action = "prepend"

                if best_j is None:
                    break

                try:
                    if best_action == "prepend":
                        join_dist = float(
                            np.linalg.norm(np.asarray(best_oriented[-1], float) - np.asarray(chain[0][0], float))
                        )
                    else:
                        join_dist = float(
                            np.linalg.norm(np.asarray(chain[-1][-1], float) - np.asarray(best_oriented[0], float))
                        )
                    _c_log(
                        1,
                        f"[STITCH JOIN] branch={branch_id} {best_action}_j={best_j} "
                        f"dist={join_dist:.3f} flipped={best_flipped}",
                    )
                except Exception as _e:
                    _c_log(3, f"[STITCH JOIN DBG] branch={branch_id} failed: {_e}")

                if best_action == "prepend":
                    chain.insert(0, best_oriented)
                else:
                    chain.append(best_oriented)
                used.add(best_j)

            if len(used) != len(segs):
                print(f"[STITCH WARN] branch={branch_id} ordering incomplete used={len(used)}/{len(segs)}")

            # Concatenate (no geometry modification)
            S_chain = np.vstack(chain)

            # Teleport diagnostics (but do NOT fail when allow_teleport=True)
            d = np.sqrt(np.sum(np.diff(S_chain, axis=0) ** 2, axis=1)) if len(S_chain) >= 2 else np.array([], float)
            if len(d) and np.any(d > max_jump):
                bad = np.where(d > max_jump)[0]
                first_bad = int(bad[0]) if len(bad) else -1
                max_step = float(np.max(d)) if len(d) else 0.0

                try:
                    pA = np.asarray(S_chain[first_bad], float) if 0 <= first_bad < len(S_chain) else None
                    pB = np.asarray(S_chain[first_bad + 1], float) if 0 <= first_bad + 1 < len(S_chain) else None
                    if pA is not None and pB is not None:
                        print(
                            f"[STITCH TELEPORT] branch={branch_id} threshold={float(max_jump):.3f} "
                            f"count={len(bad)} first_idx={first_bad} max_step={max_step:.3f} "
                            f"first_bad_pts=({pA[0]:.3f},{pA[1]:.3f})->({pB[0]:.3f},{pB[1]:.3f})"
                        )
                    else:
                        print(
                            f"[STITCH TELEPORT] branch={branch_id} threshold={float(max_jump):.3f} "
                            f"count={len(bad)} first_idx={first_bad} max_step={max_step:.3f}"
                        )
                except Exception as _e:
                    print(f"[STITCH TELEPORT] branch={branch_id} debug failed: {_e}")

                if not allow_teleport:
                    return None, False, "teleport remains after stitch"
                else:
                    print(f"[STITCH TELEPORT] branch={branch_id} WARNING: teleport tolerated (auto mode)")

            return S_chain, True, "ok"


        # AUTO/MANUAL semantics are passed explicitly by caller; `mode` is edge mode ("old"/"new").
        allow_teleport = bool(is_auto)
        S_branch, ok, reason = shared_stitch_branch_segments(
            seg_list,
            max_jump=10.0,
            allow_teleport=allow_teleport,
        )

        # If stitch failed and topology is non-chain (Y-junction etc.), retry with teleport tolerated.
        # This gracefully handles branching crack tips that can't form a single Eulerian path.
        if not ok and not chain_like:
            print(f"[STITCH RETRY] branch={branch_id} non-chain topology ({reason}) — retrying with allow_teleport=True")
            S_branch, ok, reason = shared_stitch_branch_segments(
                seg_list,
                max_jump=10.0,
                allow_teleport=True,
            )
            if ok:
                print(f"[STITCH RETRY] branch={branch_id} succeeded with teleport tolerance")
        print(
            f"[STITCH RESULT] branch={branch_id} ok={ok} reason={reason} "
            f"n={len(S_branch) if S_branch is not None else 0} "
            f"x_range=[{S_branch[:,0].min():.1f},{S_branch[:,0].max():.1f}] "
            f"y_range=[{S_branch[:,1].min():.1f},{S_branch[:,1].max():.1f}]"
            if S_branch is not None else ""
        )

        if not ok:
            print(f"[STITCH FAIL] branch={branch_id}: {reason}")

        # ====================================================
        # 1) No fallback runs in AUTO. Either proceed or fail.
        # ====================================================
        if not ok:
            # If you want *absolutely no funny business*, keep it a hard fail.
            raise ValueError(f"[COMBINER] branch {branch_id}: stitch failed ({reason})")

        runs = [S_branch]  # ALWAYS one run; never fallback

        # ====================================================
        # 2) Process each run (normally 1 per branch now)
        # ====================================================
        for run_id, S_run in enumerate(runs):
            t_loop0 = time.perf_counter()

            if S_run is None or len(S_run) < 3:
                continue

            # ------------------------------------------------
            # Loop-open guard: near-closed runs can break ET endpoint logic.
            # Open them by trimming a small tail from both ends.
            # ------------------------------------------------
            _s0 = np.asarray(S_run[0], float)
            _s1 = np.asarray(S_run[-1], float)
            _loop_dist = float(np.linalg.norm(_s0 - _s1))
            _arc_len = float(np.sum(np.linalg.norm(np.diff(np.asarray(S_run, float), axis=0), axis=1)))
            _loop_thresh = max(5.0, 0.05 * _arc_len)
            if _loop_dist < _loop_thresh:
                _trim = max(1, len(S_run) // 20)
                if len(S_run) > 2 * _trim + 2:
                    S_run = np.asarray(S_run[_trim:-_trim], float).copy()
                    if len(S_run) < 3:
                        continue
                    print(
                        f"[LOOP OPEN] branch={branch_id} run={run_id} "
                        f"loop_dist={_loop_dist:.1f}px trimmed {_trim} pts from each end",
                        flush=True,
                    )

            # ------------------------------------------------
            # Local Gaussian smoothing at sharp bends (>60°)
            # Only applied in a small window centred on the bend.
            # ------------------------------------------------
            _S = np.asarray(S_run, float).copy()
            if len(_S) >= 4:
                _tan = np.diff(_S, axis=0)
                _tlen = np.linalg.norm(_tan, axis=1, keepdims=True).clip(1e-9)
                _utan = _tan / _tlen
                _dots = np.einsum('ij,ij->i', _utan[:-1], _utan[1:]).clip(-1.0, 1.0)
                _turns = np.degrees(np.arccos(_dots))   # len = len(_S)-2
                _bad = np.where(_turns > 60.0)[0] + 1  # +1 -> index in _S

                if len(_bad) > 0:
                    # Cluster nearby bend indices (within 5 pts = same junction)
                    _clusters, _cur = [], [int(_bad[0])]
                    for _bi in _bad[1:]:
                        if int(_bi) - _cur[-1] <= 5:
                            _cur.append(int(_bi))
                        else:
                            _clusters.append(_cur); _cur = [int(_bi)]
                    _clusters.append(_cur)

                    _W     = max(10, int(2 * window_half_size))
                    _sigma = _W / 3.0

                    from scipy.ndimage import gaussian_filter1d

                    for _cl in _clusters:
                        _center = int(round(float(np.mean(_cl))))
                        _lo = max(0, _center - _W)
                        _hi = min(len(_S) - 1, _center + _W)

                        # Blend weight: Gaussian envelope, 1 at centre, ~0 at edges
                        _idxs  = np.arange(_lo, _hi + 1)
                        _blend = np.exp(-0.5 * ((_idxs - _center) / _sigma) ** 2)

                        # Smooth using padded context so edge artefacts don't leak in
                        _ctx_lo = max(0, _lo - _W)
                        _ctx_hi = min(len(_S) - 1, _hi + _W)
                        _xs = gaussian_filter1d(_S[_ctx_lo:_ctx_hi+1, 0], sigma=_sigma, mode='nearest')
                        _ys = gaussian_filter1d(_S[_ctx_lo:_ctx_hi+1, 1], sigma=_sigma, mode='nearest')

                        _off = _lo - _ctx_lo
                        _xw  = _xs[_off:_off + len(_idxs)]
                        _yw  = _ys[_off:_off + len(_idxs)]

                        _S[_lo:_hi+1, 0] = _blend * _xw + (1.0 - _blend) * _S[_lo:_hi+1, 0]
                        _S[_lo:_hi+1, 1] = _blend * _yw + (1.0 - _blend) * _S[_lo:_hi+1, 1]

                        _peak_turn = float(_turns[np.array(_cl) - 1].max())
                        print(
                            f"[BEND_SMOOTH] branch={branch_id} run={run_id} "
                            f"center={_center} window=[{_lo},{_hi}] W={_W} sigma={_sigma:.1f} "
                            f"n_bad={len(_cl)} peak_turn={_peak_turn:.1f}deg",
                            flush=True
                        )

                    S_run = _S

            # ------------------------------------------------
            # Branch-safe pad
            # ------------------------------------------------
            branch_pad = max(
                pad,
                window_half_size + 5,   # edge_mask window + solver lateral drift
            )

            # ------------------------------------------------
            # Run bbox (pad included)
            # IMPORTANT: bbox from THIS run; if stitched ok, it's the whole branch
            # ------------------------------------------------
            x0 = max(0, int(np.floor(S_run[:, 0].min() - branch_pad)))
            x1 = min(W, int(np.ceil (S_run[:, 0].max() + branch_pad)))
            y0 = max(0, int(np.floor(S_run[:, 1].min() - branch_pad)))
            y1 = min(H, int(np.ceil (S_run[:, 1].max() + branch_pad)))
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            bw, bh = int(x1 - x0), int(y1 - y0)
            _c_log(1, f"[DBG_BBOX] branch={branch_id} run={run_id} bbox=({x0},{y0},{bw},{bh})")
            _dbg_pts(f"branch{branch_id}_run{run_id}_midline BEFORE crop", S_run, bbox=(x0, y0, bw, bh), level=1)
            _dbg_frame_mismatch(S_run, (x0, y0, bw, bh), level=1)

            # 🔴 DEBUG BEFORE ANY SOLVER LOGIC
            '''debug_bbox_only(
                debug_dir=debug_dir,
                branch_id=branch_id,
                run_id=run_id,
                gray_full=gray_full,
                bbox=(x0, y0, x1, y1),
                S_run=S_run,
            )'''

            crop = gray_full[y0:y1, x0:x1]

            # ---- DBG: save raw crop image ----
            if debug_dir is not None:
                _ddir = os.path.join(debug_dir, f"branch_{branch_id}", f"run_{run_id}")
                os.makedirs(_ddir, exist_ok=True)
                cv2.imwrite(os.path.join(_ddir, "crop_raw.png"), crop)
                print(f"[DBG_CROP] branch={branch_id} run={run_id} crop shape={crop.shape} "
                      f"dtype={crop.dtype} min={int(crop.min())} max={int(crop.max())}", flush=True)

            track_local_yx = np.vstack([
                S_run[:, 1] - y0,
                S_run[:, 0] - x0
            ])
            midline_xy_crop_dbg = np.column_stack([track_local_yx[1], track_local_yx[0]])
            _dbg_pts(
                f"branch{branch_id}_run{run_id}_midline AFTER crop",
                midline_xy_crop_dbg,
                bbox=(x0, y0, bw, bh),
                level=1,
            )
            if np.any(midline_xy_crop_dbg < -5) or np.any(midline_xy_crop_dbg[:, 0] > (bw + 5)) or np.any(midline_xy_crop_dbg[:, 1] > (bh + 5)):
                if _VERBOSE_DEBUG:
                    print("[DBG_BBOX][WARN] OUT OF BOUNDS AFTER CROP", flush=True)

            # seeds/tips for THIS run (branch)
            pts_crop = [
                S_run[0]  - [x0, y0],
                S_run[-1] - [x0, y0],
            ]

            # ---- DBG: seed/tip in crop coords + bounds check ----
            _seed_c = np.asarray(pts_crop[0], float)
            _tip_c  = np.asarray(pts_crop[1], float)
            print(f"[DBG_PTS_CROP] branch={branch_id} run={run_id} "
                  f"seed_crop={_seed_c.tolist()} tip_crop={_tip_c.tolist()} "
                  f"crop_WH=({bw},{bh})", flush=True)
            for _lbl, _pt in [("seed", _seed_c), ("tip", _tip_c)]:
                if _pt[0] < 0 or _pt[0] > bw or _pt[1] < 0 or _pt[1] > bh:
                    print(f"[DBG_PTS_CROP][WARN] {_lbl} OUT OF CROP BOUNDS: {_pt.tolist()}", flush=True)

            # ---- DBG: midline tangent angle stats (detect V-bend) ----
            _S = np.asarray(S_run, float)
            if len(_S) >= 4:
                _tangents = np.diff(_S, axis=0)
                _lens = np.linalg.norm(_tangents, axis=1, keepdims=True).clip(1e-9)
                _unit = _tangents / _lens
                _angles_deg = np.degrees(np.arctan2(_unit[:, 1], _unit[:, 0]))
                _n = len(_angles_deg)
                _q1 = int(_n * 0.10); _q2 = int(_n * 0.45); _q3 = int(_n * 0.55); _q4 = int(_n * 0.90)
                _ang_start  = float(np.mean(_angles_deg[:max(1, _q1)]))
                _ang_mid    = float(np.mean(_angles_deg[_q2:max(_q2+1, _q3)]))
                _ang_end    = float(np.mean(_angles_deg[min(_q4, _n-1):]))
                _max_turn   = float(np.max(np.abs(np.diff(_angles_deg))))
                print(f"[DBG_TANGENT] branch={branch_id} run={run_id} "
                      f"angle_start={_ang_start:.1f}° mid={_ang_mid:.1f}° end={_ang_end:.1f}° "
                      f"max_single_turn={_max_turn:.1f}° n_pts={len(_S)}", flush=True)
                if _max_turn > 90:
                    print(f"[DBG_TANGENT][WARN] sharp bend detected: {_max_turn:.1f}° — ET edge masks likely corrupted at junction", flush=True)

            # -------------------------
            # edge_masks
            # -------------------------
            t_em0 = time.perf_counter()

            if track_local_yx.shape[1] < 3:
                continue

            em1, em2 = edge_masks(
                crop.astype(np.uint8),
                track_local_yx,
                window_half_size=window_half_size,
                mode=mode
            )
            t_masks_total += (time.perf_counter() - t_em0)

            # ---- Build corridor domain mask ----
            # Dilate the midline into a binary corridor to prevent the
            # geodesic shortcutting through empty background.
            _corridor_radius = int(3 * window_half_size)
            _mid_mask = np.zeros((bh, bw), dtype=np.uint8)
            _mid_pts = np.round(midline_xy_crop_dbg).astype(np.int32)
            _mid_pts[:, 0] = np.clip(_mid_pts[:, 0], 0, bw - 1)
            _mid_pts[:, 1] = np.clip(_mid_pts[:, 1], 0, bh - 1)
            for _pt in _mid_pts:
                _mid_mask[_pt[1], _pt[0]] = 1
            _kern_r = _corridor_radius * 2 + 1
            _kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_kern_r, _kern_r))
            domain_mask_crop = cv2.dilate(_mid_mask, _kern, iterations=1)
            print(f"[CORRIDOR] branch={branch_id} run={run_id} radius={_corridor_radius}px "
                  f"coverage={100*domain_mask_crop.mean():.1f}%", flush=True)

            # Save corridor mask for debug
            if debug_dir is not None:
                _ddir = os.path.join(debug_dir, f"branch_{branch_id}", f"run_{run_id}")
                cv2.imwrite(os.path.join(_ddir, "domain_mask.png"), domain_mask_crop * 255)
            if debug_dir is not None:
                _ddir = os.path.join(debug_dir, f"branch_{branch_id}", f"run_{run_id}")
                os.makedirs(_ddir, exist_ok=True)
                def _save_em(arr, path):
                    a = np.asarray(arr, float)
                    mn, mx = a.min(), a.max()
                    if mx > mn:
                        a = ((a - mn) / (mx - mn) * 255).astype(np.uint8)
                    else:
                        a = np.zeros_like(a, dtype=np.uint8)
                    cv2.imwrite(path, a)
                _save_em(em1, os.path.join(_ddir, "em1.png"))
                _save_em(em2, os.path.join(_ddir, "em2.png"))
                print(f"[DBG_EM] branch={branch_id} run={run_id} "
                      f"em1 shape={np.shape(em1)} range=[{float(np.min(em1)):.3f},{float(np.max(em1)):.3f}] "
                      f"em2 shape={np.shape(em2)} range=[{float(np.min(em2)):.3f},{float(np.max(em2)):.3f}]",
                      flush=True)

            # -------------------------
            # edges_tracking (PER BRANCH now)
            # -------------------------
            midline_xy_crop = np.column_stack([
                track_local_yx[1],
                track_local_yx[0],
            ])
            ys = np.round(midline_xy_crop[:, 1]).astype(int)
            xs = np.round(midline_xy_crop[:, 0]).astype(int)
            valid = (
                (ys >= 0) & (ys < crop.shape[0]) &
                (xs >= 0) & (xs < crop.shape[1])
            )
            _c_log(1, "[DBG_MASK_ALIGNMENT]")
            _c_log(1, f"midline sample: {midline_xy_crop[:3].tolist()}")
            _c_log(1, f"mask shape: {(crop.shape[0], crop.shape[1])}")
            _c_log(1, f"valid ratio: {float(np.mean(valid)):.3f}")

            t_et0 = time.perf_counter()
            # ------------------------------------------------
            # Create per-branch/run debug directory (no refactor)
            # ------------------------------------------------
            branch_debug_dir = None

            if debug_dir is not None:
                branch_debug_dir = os.path.join(
                    debug_dir,
                    f"branch_{branch_id}",
                    f"run_{run_id}"
                )

                os.makedirs(branch_debug_dir, exist_ok=True)

            # -------------------------
            # edges_tracking (with domain mask; fallback to no mask if solver fails)
            # -------------------------
            res = None
            for _dm_attempt, _dm_label in [
                (domain_mask_crop, "with_corridor"),
                (None,             "no_corridor_fallback"),
            ]:
                try:
                    res = edges_tracking(
                        image_crop=crop,
                        pts_cropp=pts_crop,
                        edge_mask1_cropp=em1,
                        edge_mask2_cropp=em2,
                        midline=midline_xy_crop,
                        mu=int(mu), l=int(l), p=int(p),
                        return_normal_edges=True,
                        prefer_gpu=prefer_gpu,
                        mode=mode,
                        domain_mask=_dm_attempt,
                        debug_dir=branch_debug_dir
                    )
                    print(f"[ET_ATTEMPT] branch={branch_id} run={run_id} "
                          f"attempt={_dm_label} succeeded", flush=True)
                    break
                except Exception as _et_exc:
                    print(f"[ET_ATTEMPT] branch={branch_id} run={run_id} "
                          f"attempt={_dm_label} failed: {_et_exc}", flush=True)
                    res = None

            t_edges_total += (time.perf_counter() - t_et0)

            if not isinstance(res, dict):
                continue

            e1, e2 = res.get("geodesic_edges", [None, None])
            if e1 is None or e2 is None or len(e1) < 2 or len(e2) < 2:
                continue
            
            derived_mid = np.asarray(res.get("derived_midline", []), float)
            print(
                f"[EDGE LENGTHS] branch={branch_id} run={run_id} "
                f"e1={len(e1)} e2={len(e2)} derived_mid={len(derived_mid)}"
            )
            if derived_mid.ndim != 2 or derived_mid.shape[1] != 2 or len(derived_mid) < 2:
                raise ValueError("[COMBINER] edges_tracking returned no derived midline")

            # -------------------------
            # Post-processing
            # -------------------------
            t_post0 = time.perf_counter()

            e1 = _finite_xy(np.asarray(e1, float))
            e2 = _finite_xy(np.asarray(e2, float))
            if len(e1) < 2 or len(e2) < 2:
                continue

            e1_full = _finite_xy(np.column_stack([e1[:, 0] + x0, e1[:, 1] + y0]))
            e2_full = _finite_xy(np.column_stack([e2[:, 0] + x0, e2[:, 1] + y0]))
            if len(e1_full) < 2 or len(e2_full) < 2:
                continue
            
            derived_mid_full = _finite_xy(
                np.column_stack([
                    derived_mid[:, 0] + x0,
                    derived_mid[:, 1] + y0
                ])
            )
            if len(derived_mid_full) < 2:
                raise ValueError("[COMBINER] derived midline collapsed after full-image mapping")

            # ---- normals ----
            normals = res.get("normal_edge_points")
            if normals is not None:
                (e1x, e1y), (e2x, e2y) = normals
                n1_full = _finite_xy(np.column_stack([np.asarray(e1x) + x0, np.asarray(e1y) + y0]))
                n2_full = _finite_xy(np.column_stack([np.asarray(e2x) + x0, np.asarray(e2y) + y0]))

                m = min(len(n1_full), len(n2_full))
                if m >= 2:
                    d = np.sqrt(np.sum((n1_full[:m] - n2_full[:m]) ** 2, axis=1))
                    if d.size:
                        all_widths.append(d[np.isfinite(d)])
            else:
                raise ValueError("[COMBINER] edges_tracking returned no normal_edge_points")

            # Canonicalize this run direction against stitched branch direction.
            (
                derived_mid_full,
                _n_unused,
                _w_unused,
                e1_full,
                e2_full,
                orient_info,
            ) = orient_segment_to_reference(
                derived_mid_full,
                ref_start=np.asarray(S_run[0], float),
                ref_end=np.asarray(S_run[-1], float),
                normals=None,
                widths=None,
                edge1=e1_full,
                edge2=e2_full,
                normals_are_vectors=False,
            )
            try:
                if _VERBOSE_DEBUG:
                    print("\n[DBG_CANON_INPUT]", flush=True)
                    print(f"segA_end: {np.asarray(S_run[-1], float).tolist()}", flush=True)
                    print(f"segB_start: {np.asarray(derived_mid_full[0], float).tolist()}", flush=True)
                    print(f"segB_end: {np.asarray(derived_mid_full[-1], float).tolist()}", flush=True)
                    print(f"segA_bbox: ({int(x0)},{int(y0)},{int(x1 - x0)},{int(y1 - y0)})", flush=True)
                    print(f"segB_bbox: ({int(x0)},{int(y0)},{int(x1 - x0)},{int(y1 - y0)})", flush=True)
            except Exception as _e:
                if _VERBOSE_DEBUG:
                    print(f"[DBG_CANON_INPUT] failed: {_e}", flush=True)
            if orient_info.get("flipped", False):
                n1_full = _finite_xy(np.asarray(n1_full, float)[::-1].copy())
                n2_full = _finite_xy(np.asarray(n2_full, float)[::-1].copy())
                if _VERBOSE_DEBUG:
                    print(
                        f"[CANON] combine branch={branch_id} run={run_id} "
                        f"flipped derived run (d_fwd={orient_info['d_forward']:.4f}, d_rev={orient_info['d_reverse']:.4f})"
                    )

            # ---- normals validity + fallback ----
            from cracktools.segmentation import generate_mask_from_edges
            m_norm = min(len(n1_full), len(n2_full))
            if m_norm < 2:
                # Fallback: estimate normals from derived midline tangents + fixed half-width.
                try:
                    _dm = np.asarray(derived_mid_full, float)
                    _tangents = np.diff(_dm, axis=0)
                    _lens = np.linalg.norm(_tangents, axis=1, keepdims=True).clip(1e-9)
                    _tangents = _tangents / _lens
                    _perp = np.column_stack([-_tangents[:, 1], _tangents[:, 0]])
                    _hw = float(np.median(np.concatenate(all_widths)) / 2.0) if all_widths else 8.0
                    _pts = (_dm[:-1] + _dm[1:]) / 2.0
                    n1_full = _finite_xy(_pts + _perp * _hw)
                    n2_full = _finite_xy(_pts - _perp * _hw)
                    m_norm = min(len(n1_full), len(n2_full))
                    print(
                        f"[COMBINER FALLBACK] branch={branch_id} run={run_id} "
                        f"recomputed normals from tangents n={m_norm} hw={_hw:.1f}px"
                    )
                except Exception as _ef:
                    print(f"[COMBINER FALLBACK] branch={branch_id} run={run_id} failed: {_ef}")
                    m_norm = 0
                if m_norm < 2:
                    print(
                        f"[COMBINER DIAG] branch={branch_id} run={run_id} insufficient normals "
                        f"after mapping/canon: len(S_run)={len(S_run)} len(derived_mid)={len(derived_mid_full)} "
                        f"len(e1)={len(e1_full)} len(e2)={len(e2_full)} "
                        f"len(n1)={len(n1_full)} len(n2)={len(n2_full)} "
                        f"raw_norm_shapes={(np.shape(e1x), np.shape(e1y), np.shape(e2x), np.shape(e2y))}"
                    )
                    continue

            # ---- store geometry (only after normals validated) ----
            derived_midline_segs.append(derived_mid_full)
            branch_to_derived_runs[int(branch_id)].append(derived_mid_full)
            edge1_segs.append(e1_full)
            edge2_segs.append(e2_full)
            norm1_segs.append(n1_full)
            norm2_segs.append(n2_full)

            try:
                _mask_out_dir = branch_debug_dir or (
                    os.path.join(debug_dir, f"branch_{branch_id}", f"run_{run_id}") if debug_dir else None
                )
                mask_run = generate_mask_from_edges(
                    img_gray=gray_full,
                    edge1_xy=e1_full,
                    edge2_xy=e2_full,
                    midline_xy=derived_mid_full,
                    normals_xy=(n1_full, n2_full),
                    out_dir=_mask_out_dir,
                    tag=f"branch{branch_id}_run{run_id}",
                    do_morph=False,
                )
            except ValueError as e:
                if "normals_xy" in str(e):
                    print(
                        f"[COMBINER DIAG] generate_mask_from_edges failed branch={branch_id} run={run_id}: {e} "
                        f"(len derived={len(derived_mid_full)} e1={len(e1_full)} e2={len(e2_full)} "
                        f"n1={len(n1_full)} n2={len(n2_full)})"
                    )
                raise

            if mask_run is not None and mask_run.any():
                union_mask |= (mask_run > 0).astype(np.uint8)

            t_post_total += (time.perf_counter() - t_post0)
            t_loop_total += (time.perf_counter() - t_loop0)

        _ = time.perf_counter() - t_branch0


    if np.any(union_mask):
        bb = bbox_from_mask(union_mask)
        if bb is None:
            raise ValueError("combined crack mask produced no bbox")
        x, y, w, h = bb
        crop = union_mask[y:y+h, x:x+w].astype(np.uint8)
    else:
        x = y = 0
        w = h = 1
        crop = np.zeros((1, 1), np.uint8)

    def _flatten(seg_list):
        out = []
        for i, arr in enumerate(seg_list):
            out.extend([[float(xx), float(yy)] for (xx, yy) in arr])
            if i < len(seg_list) - 1:
                out.append([None, None])
        return out

    def _flatten_with_none(seg_list):
        out = []
        for i, arr in enumerate(seg_list):
            for (xx, yy) in arr:
                out.append([float(xx), float(yy)])
            if i < len(seg_list) - 1:
                out.append([None, None])
        return out

    derived_midline_meta = recompute_dominance_geometry_from_derived(
        dominance_meta=dominance_meta,
        branch_to_derived_runs=branch_to_derived_runs,
        atomic=authoring_atomic,
        H=H,
        W=W,
        derived_domain_mask=union_mask,
        window_half_size=window_half_size,
        members=member_ids,
        debug_dir=debug_dir,
        debug_tag="derived_geometry_dominance",
    )

    # --------------------------------------------------------
    # One stitched derived segment per branch (seg_idx=0), matching GT supervision structure.
    # Per-atomic splitting was removed — it caused key mismatches in compare_widths_for_aligned_cracks.
    # --------------------------------------------------------
    derived_midline_segments = []
    derived_midline_segments_meta = []

    # Build branch_to_mid for atomic_id lookup (used for meta only)
    branch_to_mid = {}
    for S, m in zip(segs, midline_segments_meta):
        if S is None or len(S) < 2:
            continue
        mm = m if isinstance(m, dict) else {}
        bid = int(mm.get("branch_id", -1))
        branch_to_mid.setdefault(bid, {"segs": [], "meta": []})
        branch_to_mid[bid]["segs"].append(np.asarray(S, float))
        branch_to_mid[bid]["meta"].append(dict(mm))

    for bi, runs in (branch_to_derived_runs or {}).items():
        bi = int(bi)
        runs = runs or []
        if not runs:
            continue

        runs_np = [np.asarray(r, float) for r in runs if r is not None and len(r) >= 2]
        if not runs_np:
            continue
        # Use longest derived run as the single stitched segment for this branch
        runs_np.sort(key=lambda a: int(len(a)), reverse=True)
        D = runs_np[0]

        mid_pack = branch_to_mid.get(bi, {"segs": [], "meta": []})
        mid_meta_b = mid_pack["meta"]
        # Collect all atomic_ids for this branch from meta
        atomic_ids = [str((mx if isinstance(mx, dict) else {}).get("atomic_id", "")) for mx in mid_meta_b]
        atomic_ids = [a for a in atomic_ids if a]

        # One segment per branch, seg_idx=0 — matches GT supervision key structure
        seg_meta = {
            "branch_id": int(bi),
            "seg_idx": 0,
            "stitched": True,
            "atomic_ids": atomic_ids,
            "atomic_id": atomic_ids[0] if atomic_ids else None,
        }
        derived_midline_segments.append(D)
        derived_midline_segments_meta.append(seg_meta)

    if len(derived_midline_segments) != len(derived_midline_segments_meta):
        raise RuntimeError(
            f"[COMBINER] derived_midline_segments/meta mismatch: "
            f"{len(derived_midline_segments)} segs vs {len(derived_midline_segments_meta)} meta"
        )

    # Final check: branch-level continuity for user and derived outputs.
    for bi in sorted({int(m.get("branch_id", -1)) for m in (midline_segments_meta or []) if isinstance(m, dict)}):
        bsegs = [np.asarray(S, float) for S, m in zip(segs, midline_segments_meta) if isinstance(m, dict) and int(m.get("branch_id", -1)) == bi and S is not None and len(S) >= 2]
        if len(bsegs) >= 2:
            assert_direction_consistency(bsegs)
    for bi in sorted({int(m.get("branch_id", -1)) for m in (derived_midline_segments_meta or []) if isinstance(m, dict)}):
        pairs = [
            (int((m if isinstance(m, dict) else {}).get("seg_idx", i)), np.asarray(S, float))
            for i, (S, m) in enumerate(zip(derived_midline_segments, derived_midline_segments_meta))
            if isinstance(m, dict) and int(m.get("branch_id", -1)) == bi and S is not None and len(S) >= 2
        ]
        if len(pairs) >= 2:
            pairs.sort(key=lambda t: t[0])
            assert_direction_consistency([p[1] for p in pairs])

    combined_length = float(sum(_linestring_length(s) for s in segs))
    if len(all_widths):
        mean_width = float(np.nanmean(np.concatenate(all_widths)))
    else:
        mean_width = None
    
    # --------------------------------------------------------
    # 🔍 DEBUG CALLBACK FOR COMBINED CRACK OVERLAY
    # --------------------------------------------------------
    if debug_callback:
        try:
            debug_callback(
                segs=segs,
                derived_midline_segs=derived_midline_segs,
                edge1_segs=edge1_segs,
                edge2_segs=edge2_segs,
                norm1_segs=norm1_segs,
                norm2_segs=norm2_segs,
                mask_bbox=[int(x), int(y), int(w), int(h)]
            )
        except Exception as e:
            if _VERBOSE_DEBUG:
                print(f"[COMBINE_DBG] debug_callback failed: {e}")

    elapsed = float(time.perf_counter() - t0)
    
    semantic_id = "_".join(str(m) for m in member_ids)

    ret = {
        "source": "combined",
        "semantic_id": semantic_id,
        "mode": mode,
        "members": [str(m) for m in member_ids],
        "midline_segments": [
            [[float(xx), float(yy)] for (xx, yy) in s] for s in segs
        ],
        "midline": _flatten(segs),
        "derived_midline_segments": [
            [[float(xx), float(yy)] for (xx, yy) in s] for s in derived_midline_segments
        ],
        "derived_midline_segments_meta": derived_midline_segments_meta,
        "derived_midline": _flatten_with_none(derived_midline_segments),
        "geodesic_edges": {"edge1": _flatten(edge1_segs),
                           "edge2": _flatten(edge2_segs)},
        "normal_edge_points": {"edge1": _flatten(norm1_segs),
                               "edge2": _flatten(norm2_segs)},
        "mask_crop": crop.tolist(),
        "mask_bbox": [int(x), int(y), int(w), int(h)],
        "combined_length": combined_length,
        "mean_width": mean_width,
        "timing": {
            "build_combined_sec": elapsed,
            "stitching_sec": stitching_sec,
            "combine_edge_masks_sec": t_masks_total,
            "combine_edge_tracking_sec": t_edges_total,
            "combine_postprocess_sec": t_post_total,
            "combine_loop_total_sec": t_loop_total,
        },
        "midline_segments_meta": midline_segments_meta,
        "dominance_meta": dominance_meta,
        "derived_midline_meta": derived_midline_meta,
    }
    if isinstance(ret["derived_midline_meta"], dict):
        ret["derived_midline_meta"].setdefault("segments_meta", ret["derived_midline_segments_meta"])
    return ret
