# helpers/combiner.py
# Stateless combined-crack builder for metrics pipeline
from __future__ import annotations
import os, math
import numpy as np
import cv2

# --- imports that exist in your tree; keep both fallbacks ---
from cracktools.segmentation import edge_masks, edges_tracking


from helpers.metrics import bbox_from_mask

from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union
from helpers.plot_metrics import *


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


def _split_lines(geom):
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
        ml = crack.get("midline", []) or []
        if len(ml) < 2: continue
        arr = np.array([[float(x), float(y)] for (x,y) in ml], dtype=float)
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

def _align_edge_to_midline(S_xy, E_xy):
    d_f = np.linalg.norm(E_xy[0]-S_xy[0]) + np.linalg.norm(E_xy[-1]-S_xy[-1])
    d_r = np.linalg.norm(E_xy[0]-S_xy[-1]) + np.linalg.norm(E_xy[-1]-S_xy[0])
    return (E_xy[::-1] if d_r < d_f else E_xy)


def _ribbon_mask_from_midline(H, W, S_xy, thickness_px=4):
    mask = np.zeros((H, W), dtype=np.uint8)
    pts = np.round(S_xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=thickness_px, lineType=cv2.LINE_AA)
    return mask

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

    plot_edges_and_normals(
        base_image=original_image,
        midline_segs=segs,
        edge1_segs=edge1_segs,
        edge2_segs=edge2_segs,
        norm1_segs=norm1_segs,
        norm2_segs=norm2_segs,
        bbox=mask_bbox,
        out_png=out_png,
        title=f"Combined Crack (members={', '.join(member_ids)})",
    )
    
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

def build_combined_crack_stateless(
    original_image: np.ndarray,
    authoring_atomic: dict,
    member_ids: list[str],
    *,
    window_half_size: int = 45,
    mu: float = 0.0,
    l: int = 5,
    p: int = 14,
    color_channel: int = 0,
    pad: int = 10,
    prefer_gpu: bool = True,
    debug_callback=None,
):
    """
    Stateless “metrics-safe” combiner with fully instrumented timing.
    """
    import time

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

    stitched = _stitch_lines_by_user(member_ids, authoring_atomic)
    stitched.sort(key=_linestring_length, reverse=True)

    try:
        from shapely.geometry import LineString
        from shapely.ops import unary_union
        use_shapely = True
    except Exception:
        use_shapely = False

    kept_segs, dom_buffer = [], None
    overlap_px = max(6, int(window_half_size * 0.6))
    min_keep_len = max(8.0, 0.6 * window_half_size)

    if use_shapely:
        for S in stitched:
            g = LineString(S)
            if dom_buffer is None:
                kept_segs.append(S)
                dom_buffer = g.buffer(overlap_px, cap_style=2, join_style=2)
            else:
                rem = g.difference(dom_buffer)
                if rem.is_empty:
                    continue
                for piece in _split_lines(rem):
                    if piece.length >= min_keep_len:
                        kept_segs.append(np.asarray(piece.coords, float))
                dom_buffer = unary_union([
                    dom_buffer,
                    g.buffer(overlap_px, cap_style=2, join_style=2)
                ])
        segs = kept_segs if kept_segs else stitched
    else:
        segs = stitched

    t_stitch1 = time.perf_counter()
    stitching_sec = float(t_stitch1 - t_stitch0)

    # --------------------------------------------------------
    # Per-segment processing timing
    # --------------------------------------------------------
    edge1_segs, edge2_segs = [], []
    norm1_segs, norm2_segs = [], []
    union_mask = np.zeros((H, W), np.uint8)
    all_widths = []

    # NEW timers
    t_masks_total = 0.0
    t_edges_total = 0.0
    t_post_total = 0.0
    t_loop_total = 0.0

    for S in segs:
        t_loop0 = time.perf_counter()

        if S is None or len(S) < 2:
            continue

        x0 = max(0, int(np.floor(S[:, 0].min()) - pad))
        x1 = min(W, int(np.ceil(S[:, 0].max()) + pad))
        y0 = max(0, int(np.floor(S[:, 1].min()) - pad))
        y1 = min(H, int(np.ceil(S[:, 1].max()) + pad))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue

        crop = gray_full[y0:y1, x0:x1]
        track_local_yx = np.vstack([S[:, 1] - y0, S[:, 0] - x0])
        pts_crop = [S[0] - [x0, y0], S[-1] - [x0, y0]]

        # -------------------------
        # edge_masks timing
        # -------------------------
        t_em0 = time.perf_counter()
        em1, em2 = edge_masks(
            crop.astype(np.uint8),
            track_local_yx,
            window_half_size=window_half_size
        )
        t_em1 = time.perf_counter()
        t_masks_total += (t_em1 - t_em0)

        # -------------------------
        # edges_tracking timing
        # -------------------------
        midline_xy_crop = np.column_stack([track_local_yx[1], track_local_yx[0]])
        t_et0 = time.perf_counter()
        res = edges_tracking(
            image_crop=crop,
            pts_cropp=pts_crop,
            edge_mask1_cropp=em1, edge_mask2_cropp=em2,
            midline=midline_xy_crop,
            mu=int(mu), l=int(l), p=int(p),
            return_normal_edges=True,
            prefer_gpu=prefer_gpu
        )
        t_et1 = time.perf_counter()
        t_edges_total += (t_et1 - t_et0)

        if not isinstance(res, dict):
            continue

        ge = res.get("geodesic_edges", [None, None])
        if ge is None or len(ge) != 2:
            continue
        e1, e2 = ge
        if e1 is None or e2 is None or len(e1) < 2 or len(e2) < 2:
            continue

        # -------------------------
        # post-processing timing
        # -------------------------
        t_post0 = time.perf_counter()

        e1 = np.asarray(e1, float)
        e2 = np.asarray(e2, float)
        e1_full = _finite_xy(np.column_stack([e1[:, 0] + x0, e1[:, 1] + y0]))
        e2_full = _finite_xy(np.column_stack([e2[:, 0] + x0, e2[:, 1] + y0]))
        if len(e1_full) < 2 or len(e2_full) < 2:
            continue

        e1_full = _align_edge_to_midline(S, e1_full)
        e2_full = _align_edge_to_midline(S, e2_full)

        normals = res.get("normal_edge_points")
        if normals is not None:
            (e1x, e1y), (e2x, e2y) = normals
            n1_full = _finite_xy(np.column_stack([np.asarray(e1x) + x0,
                                                  np.asarray(e1y) + y0]))
            n2_full = _finite_xy(np.column_stack([np.asarray(e2x) + x0,
                                                  np.asarray(e2y) + y0]))
            m = min(len(n1_full), len(n2_full))
            if m >= 2:
                d = np.sqrt(np.sum((n1_full[:m] - n2_full[:m])**2, axis=1))
                if d.size:
                    all_widths.append(d[np.isfinite(d)])
        else:
            n1_full = np.empty((0, 2))
            n2_full = np.empty((0, 2))

        edge1_segs.append(e1_full)
        edge2_segs.append(e2_full)
        norm1_segs.append(n1_full)
        norm2_segs.append(n2_full)

        ex = np.concatenate((e1_full[:, 0][::-1], e2_full[:, 0]))
        ey = np.concatenate((e1_full[:, 1][::-1], e2_full[:, 1]))
        exc, eyc = np.clip(ex, 0, W - 1), np.clip(ey, 0, H - 1)
        area = _shoelace_area(exc, eyc)

        if area > 0.5:
            poly = np.stack([exc, eyc], axis=1).astype(np.int32).reshape(-1, 1, 2)
            mask_seg = np.zeros((H, W), np.uint8)
            cv2.fillPoly(mask_seg, [poly], 255)
        else:
            mask_seg = _ribbon_mask_from_midline(
                H, W, S, thickness_px=max(3, window_half_size // 3)
            )

        union_mask |= (mask_seg > 0).astype(np.uint8)

        t_post1 = time.perf_counter()
        t_post_total += (t_post1 - t_post0)

        t_loop_total += (time.perf_counter() - t_loop0)

    if np.any(union_mask):
        x, y, w, h = bbox_from_mask(union_mask) or [0, 0, W, H]
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
                edge1_segs=edge1_segs,
                edge2_segs=edge2_segs,
                norm1_segs=norm1_segs,
                norm2_segs=norm2_segs,
                mask_bbox=[int(x), int(y), int(w), int(h)]
            )
        except Exception as e:
            print(f"[COMBINE_DBG] debug_callback failed: {e}")

    elapsed = float(time.perf_counter() - t0)

    return {
        "source": "combined",
        "members": [str(m) for m in member_ids],
        "midline_segments": [
            [[float(xx), float(yy)] for (xx, yy) in s] for s in segs
        ],
        "midline": _flatten(segs),
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
    }
