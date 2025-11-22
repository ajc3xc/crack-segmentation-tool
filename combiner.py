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

'''def metrics_combined_debug_plot(
    *,
    original_image,
    metrics_dir,
    combined_id,
    member_ids,
    segs,
    edge1,
    edge2,
    normals1,
    normals2,
    union_mask
):
    """
    Metrics-safe debug plot. Identical style to GUI, but saved in metrics tree.
    """
    import os, numpy as np, matplotlib.pyplot as plt
    H, W = original_image.shape[:2]

    member_str = "_".join(str(m) for m in member_ids)
    out_dir = os.path.join(metrics_dir, f"combined{combined_id}_{member_str}")
    os.makedirs(out_dir, exist_ok=True)

    fname = os.path.join(out_dir, "combined_debug.png")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(original_image)
    ax.set_title(f"Combined Debug (metrics) cid={combined_id} members={member_str}")

    def _split(arr, max_step=50.0):
        arr = np.asarray(arr, float)
        if len(arr) < 2:
            return []
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

    for S in segs:
        for segp in _split(S):
            ax.plot(segp[:,0], segp[:,1], "g-", lw=0.7)

    for e in edge1:
        for segp in _split(e):
            ax.plot(segp[:,0], segp[:,1], "r-", lw=0.6)
    for e in edge2:
        for segp in _split(e):
            ax.plot(segp[:,0], segp[:,1], "b-", lw=0.6)

    for n1, n2 in zip(normals1, normals2):
        if len(n1) == 0 or len(n2) == 0:
            continue
        m = min(len(n1), len(n2))
        step = max(1, m // 70)
        for i in range(0, m, step):
            p1 = n1[i]; p2 = n2[i]
            if np.isfinite(p1).all() and np.isfinite(p2).all():
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        color="cyan", lw=0.3, alpha=0.7)

    if union_mask is not None:
        ys, xs = np.where(union_mask > 0)
        if xs.size > 0:
            ax.scatter(xs, ys, s=1, c="yellow", alpha=0.15)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("equal")

    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"[METRICS_COMBINED_DEBUG] wrote → {fname}")'''

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
    out_dir
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, "combined_debug.png")

    H, W = original_image.shape[:2]
    x0, y0, w, h = mask_bbox
    x1, y1 = x0 + w, y0 + h

    pad = 40
    x0p = max(0, x0 - pad)
    x1p = min(W, x1 + pad)
    y0p = max(0, y0 - pad)
    y1p = min(H, y1 + pad)

    crop = original_image[y0p:y1p, x0p:x1p]
    crop = crop[:, :, ::-1] if crop.ndim == 3 else np.stack([crop]*3, -1)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(crop)

    def split(arr, max_step=50):
        arr = np.asarray(arr)
        if len(arr) < 2: return []
        d = np.sqrt(np.sum(np.diff(arr, axis=0)**2, axis=1))
        breaks = np.where(d > max_step)[0]
        out=[]; s=0
        for b in breaks:
            if b+1-s >= 2: out.append(arr[s:b+1])
            s = b+1
        if len(arr)-s >= 2: out.append(arr[s:])
        return out or [arr]

    # midline
    for S in segs:
        for segp in split(S):
            ax.plot(segp[:,0]-x0p, segp[:,1]-y0p, "w-", lw=1)

    # edges
    for E in edge1_segs:
        for segp in split(E):
            ax.plot(segp[:,0]-x0p, segp[:,1]-y0p, "r-", lw=1)

    for E in edge2_segs:
        for segp in split(E):
            ax.plot(segp[:,0]-x0p, segp[:,1]-y0p, "g-", lw=1)

    # normals
    STRIDE = 10
    for n1, n2 in zip(norm1_segs, norm2_segs):
        m = min(len(n1), len(n2))
        for i in range(0, m, STRIDE):
            p1, p2 = n1[i], n2[i]
            ax.plot([p1[0]-x0p, p2[0]-x0p],
                    [p1[1]-y0p, p2[1]-y0p],
                    color="cyan", lw=1)

    ax.set_title(f"Combined Crack (members={', '.join(member_ids)})")
    ax.axis("off")

    fig.savefig(fname, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"[COMBINED_DEBUG] wrote → {fname}")

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
    Stateless “metrics-safe” combine.
    """

    img = original_image
    H, W = img.shape[:2]

    if img.ndim == 3:
        bgr_idx = {0:2, 1:0, 2:1}.get(color_channel, 2)
        gray_full = img[:, :, bgr_idx].astype(np.float32)
    else:
        gray_full = img.astype(np.float32)

    # stitch member midlines
    stitched = _stitch_lines_by_user(member_ids, authoring_atomic)
    stitched.sort(key=_linestring_length, reverse=True)

    # shapely prune
    try:
        from shapely.geometry import LineString
        from shapely.ops import unary_union
        use_shapely = True
    except Exception:
        use_shapely = False

    kept_segs, dom_buffer = [], None
    overlap_px   = max(6, int(window_half_size * 0.6))
    min_keep_len = max(8.0, 0.6 * window_half_size)

    if use_shapely:
        for S in stitched:
            g = LineString(S)
            if dom_buffer is None:
                kept_segs.append(S)
                dom_buffer = g.buffer(overlap_px, cap_style=2, join_style=2)
            else:
                rem = g.difference(dom_buffer)
                if rem.is_empty: continue
                for piece in _split_lines(rem):
                    if piece.length >= min_keep_len:
                        kept_segs.append(np.asarray(piece.coords, float))
                dom_buffer = unary_union([dom_buffer, g.buffer(overlap_px, cap_style=2, join_style=2)])
        segs = kept_segs if kept_segs else stitched
    else:
        segs = stitched

    edge1_segs, edge2_segs = [], []
    norm1_segs, norm2_segs = [], []
    union_mask = np.zeros((H, W), np.uint8)
    all_widths = []

    for S in segs:
        if S is None or len(S) < 2: continue
        x0 = max(0, int(np.floor(S[:,0].min()) - pad))
        x1 = min(W, int(np.ceil(S[:,0].max()) + pad))
        y0 = max(0, int(np.floor(S[:,1].min()) - pad))
        y1 = min(H, int(np.ceil(S[:,1].max()) + pad))
        if x1-x0 < 2 or y1-y0 < 2: continue

        crop = gray_full[y0:y1, x0:x1]
        track_local_yx = np.vstack([S[:,1]-y0, S[:,0]-x0])
        pts_crop = [S[0]-[x0,y0], S[-1]-[x0,y0]]

        em1, em2 = edge_masks(crop.astype(np.uint8), track_local_yx, window_half_size=window_half_size)

        midline_xy_crop = np.column_stack([track_local_yx[1], track_local_yx[0]])
        res = edges_tracking(
            image_crop=crop,
            pts_cropp=pts_crop,
            edge_mask1_cropp=em1, edge_mask2_cropp=em2,
            midline=midline_xy_crop,
            mu=int(mu), l=int(l), p=int(p),
            return_normal_edges=True,
            prefer_gpu=prefer_gpu
        )
        if not isinstance(res, dict):
            continue

        ge = res.get("geodesic_edges", [None, None])
        if ge is None or len(ge) != 2: continue
        e1, e2 = ge
        if e1 is None or e2 is None or len(e1)<2 or len(e2)<2:
            continue

        e1 = np.asarray(e1, float); e2 = np.asarray(e2, float)
        e1_full = _finite_xy(np.column_stack([e1[:,0]+x0, e1[:,1]+y0]))
        e2_full = _finite_xy(np.column_stack([e2[:,0]+x0, e2[:,1]+y0]))
        if len(e1_full)<2 or len(e2_full)<2:
            continue

        e1_full = _align_edge_to_midline(S, e1_full)
        e2_full = _align_edge_to_midline(S, e2_full)

        normals = res.get("normal_edge_points")
        if normals is not None:
            (e1x, e1y), (e2x, e2y) = normals
            n1_full = _finite_xy(np.column_stack([np.asarray(e1x)+x0, np.asarray(e1y)+y0]))
            n2_full = _finite_xy(np.column_stack([np.asarray(e2x)+x0, np.asarray(e2y)+y0]))
            m = min(len(n1_full), len(n2_full))
            if m>=2:
                d = np.sqrt(np.sum((n1_full[:m] - n2_full[:m])**2, axis=1))
                if d.size:
                    all_widths.append(d[np.isfinite(d)])
        else:
            n1_full = np.empty((0,2)); n2_full = np.empty((0,2))

        edge1_segs.append(e1_full)
        edge2_segs.append(e2_full)
        norm1_segs.append(n1_full)
        norm2_segs.append(n2_full)

        ex = np.concatenate((e1_full[:,0][::-1], e2_full[:,0]))
        ey = np.concatenate((e1_full[:,1][::-1], e2_full[:,1]))
        exc, eyc = np.clip(ex, 0, W-1), np.clip(ey, 0, H-1)
        area = _shoelace_area(exc, eyc)
        if area > 0.5:
            poly = np.stack([exc, eyc], axis=1).astype(np.int32).reshape(-1,1,2)
            mask_seg = np.zeros((H,W), np.uint8)
            cv2.fillPoly(mask_seg, [poly], 255)
        else:
            mask_seg = _ribbon_mask_from_midline(H, W, S, thickness_px=max(3, window_half_size//3))

        union_mask |= (mask_seg > 0).astype(np.uint8)

    if np.any(union_mask):
        x,y,w,h = bbox_from_mask(union_mask) or [0,0,W,H]
        crop = union_mask[y:y+h, x:x+w].astype(np.uint8)
    else:
        x=y=0; w=h=1; crop = np.zeros((1,1), np.uint8)

    def _flatten(seg_list):
        out=[]
        for i, arr in enumerate(seg_list):
            out.extend([[float(xx), float(yy)] for xx,yy in arr])
            if i < len(seg_list)-1:
                out.append([None,None])
        return out

    combined_length = float(sum(_linestring_length(s) for s in segs))
    mean_width = float(np.nanmean(np.concatenate(all_widths))) if len(all_widths) else None

    # ---- DEBUG CALLBACK ----
    if callable(debug_callback):
        try:
            # Build an RGB image for plotting
            if img.ndim == 3:
                rgb = img[:, :, ::-1]  # BGR→RGB
            else:
                rgb = np.stack([img]*3, axis=-1)

            # Compute mask bbox
            if np.any(union_mask):
                x,y,w,h = bbox_from_mask(union_mask)
            else:
                x=y=0; w=h=1

            debug_callback(
                image_rgb=rgb,
                segs=segs,
                edge1_segs=edge1_segs,
                edge2_segs=edge2_segs,
                norm1_segs=norm1_segs,
                norm2_segs=norm2_segs,
                mask_bbox=[x,y,w,h],
                member_ids=member_ids,
                union_mask=union_mask,   # ★★★★★ REQUIRED ★★★★★
            )
        except Exception as e:
            print("[STATLESS_DEBUG] error:", e)
            print("[STATLESS_DEBUG] traceback follows:")
            import traceback
            traceback.print_exc()


    return {
        "source": "combined",
        "members": [str(m) for m in member_ids],
        "midline_segments": [ [[float(xx), float(yy)] for (xx,yy) in s] for s in segs ],
        "midline": _flatten(segs),
        "geodesic_edges": {"edge1": _flatten(edge1_segs), "edge2": _flatten(edge2_segs)},
        "normal_edge_points": {"edge1": _flatten(norm1_segs), "edge2": _flatten(norm2_segs)},
        "mask_crop": crop.tolist(),
        "mask_bbox": [int(x), int(y), int(w), int(h)],
        "combined_length": combined_length,
        "mean_width": mean_width,
    }
