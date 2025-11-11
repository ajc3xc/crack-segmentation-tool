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

# put these near the top of your module (same file as compute_mask_and_width_metrics_for_image)

def read_authoring_combined(ann_json: dict) -> dict:
    """Accept both new and legacy locations of combined annotations."""
    ann = (ann_json.get("annotations", {}) or {})
    cmb = ann.get("combined_cracks", None)
    if not cmb:
        cmb = ann.get("combined", None)
    return cmb or {}

def auto_groups_from_atomic(authoring_atomic: dict, px_thresh: float = 12.0) -> dict:
    """
    Extremely simple fallback: group atomics whose endpoints are close.
    Returns a dict like {"0": {"members":[...]} , "1": {"members":[...]}, ...}
    If nothing groups, returns one group with all atomics.
    """
    import numpy as np
    # extract endpoints from each atomic midline
    def _endpoints(mid):
        m = np.asarray(mid, float)
        if m.ndim != 2 or m.shape[0] < 2: return None
        return m[0], m[-1]

    ids = [str(k) for k in authoring_atomic.keys()]
    eps = {}
    for k in ids:
        mid = authoring_atomic[k].get("midline", [])
        ep = _endpoints(mid)
        if ep is not None:
            eps[k] = ep

    used = set()
    groups = []
    for i in ids:
        if i in used or i not in eps:
            continue
        g = [i]; used.add(i)
        a0, a1 = eps[i]
        for j in ids:
            if j in used or j not in eps: 
                continue
            b0, b1 = eps[j]
            d = min(
                np.linalg.norm(a0 - b0), np.linalg.norm(a0 - b1),
                np.linalg.norm(a1 - b0), np.linalg.norm(a1 - b1),
            )
            if d <= px_thresh:
                g.append(j); used.add(j)
        groups.append(g)

    # if everything was skipped or all singletons → make one big group
    if not groups or all(len(g) == 1 for g in groups):
        groups = [ids] if ids else []

    return {str(idx): {"members": g} for idx, g in enumerate(groups)}

def build_combined_crack_stateless(
    original_image: np.ndarray,
    authoring_atomic: dict,
    member_ids: list[str],
    *,
    window_half_size: int = 45,
    mu: float = 0.0,
    l: int = 5,
    p: int = 14,
    color_channel: int = 0,     # 0/1/2 = R/B/G as in your GUI mapping
    pad: int = 10,
    prefer_gpu: bool = True,
    save_folder: str = None,
    image_base: str = None
) -> dict:
    """
    Stateless “metrics-safe” combiner:
      - stitches member midlines using user endpoints
      - for each segment, calls edge_param_worker (GPU/CPU) to get geodesic edges + mask
      - unions per-seg masks and returns combined record
    """
    import numpy as np, cv2, math, os, traceback
    from shapely.geometry import LineString
    from edge_workers import edge_param_worker

    H, W = original_image.shape[:2]

    # --- helpers ---
    def _finite_xy(arr):
        if arr is None or len(arr) == 0:
            return np.empty((0, 2), float)
        a = np.asarray(arr, float)
        if a.ndim != 2 or a.shape[1] != 2:
            return np.empty((0, 2), float)
        ok = np.isfinite(a).all(axis=1)
        a = a[ok]
        if len(a) <= 1:
            return a
        keep = [0]
        for i in range(1, len(a)):
            if not (abs(a[i, 0] - a[i-1, 0]) < 1e-9 and abs(a[i, 1] - a[i-1, 1]) < 1e-9):
                keep.append(i)
        return a[keep]

    def _shoelace_area(xs, ys):
        return 0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))

    # --- gather member midlines ---
    stitched = []
    for mid in member_ids:
        crack = authoring_atomic.get(mid)
        if not crack:
            continue
        ml = crack.get("midline", []) or []
        if len(ml) < 2:
            continue
        stitched.append(np.array(ml, float))
    if not stitched:
        return {}

    edge1_segs, edge2_segs = [], []
    norm1_segs, norm2_segs = [], []
    union_mask = np.zeros((H, W), np.uint8)
    all_widths = []

    # --- color channel mapping (BGR safe) ---
    bgr_idx = {0: 2, 1: 0, 2: 1}.get(color_channel, 2)

    for idx, S in enumerate(stitched):
        if S is None or len(S) < 2:
            continue
        try:
            x0 = max(0, int(np.floor(S[:, 0].min()) - pad))
            x1 = min(W, int(np.ceil(S[:, 0].max()) + pad))
            y0 = max(0, int(np.floor(S[:, 1].min()) - pad))
            y1 = min(H, int(np.ceil(S[:, 1].max()) + pad))
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue

            crop = original_image[y0:y1, x0:x1]
            crop_gray = crop[:, :, bgr_idx] if crop.ndim == 3 else crop

            # --- corrected payload fields (JSON safe + atomic compatible) ---
            pts_crop = [
                [float(S[0, 0] - x0), float(S[0, 1] - y0)],
                [float(S[-1, 0] - x0), float(S[-1, 1] - y0)]
            ]

            track_local_yx = np.vstack([S[:, 1] - y0, S[:, 0] - x0]).astype(float)
            bbox = [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]

            payload = dict(
                image_crop_gray=crop_gray.astype(np.uint8),
                pts_crop=pts_crop,
                adjusted_track=track_local_yx,
                manual_midline_global=S.astype(float),
                bbox=bbox,
                params={
                    "window_half_size": int(window_half_size),
                    "mu": float(mu),
                    "l": int(l),
                    "p": int(p)
                },
                save_folder=save_folder or ".",
                image_base=os.path.basename(image_base or "unknown"),
                crack_id=f"cmbseg{idx}"
            )

            result = edge_param_worker(payload)
            if not isinstance(result, dict):
                continue

            ge = result.get("geodesic_edges", [None, None])
            if not ge or not isinstance(ge, (list, tuple)) or len(ge) != 2:
                continue

            e1, e2 = ge
            if e1 is None or e2 is None or len(e1) < 2 or len(e2) < 2:
                continue

            e1 = np.asarray(e1, float)
            e2 = np.asarray(e2, float)
            e1_full = _finite_xy(np.column_stack([e1[:, 0] + x0, e1[:, 1] + y0]))
            e2_full = _finite_xy(np.column_stack([e2[:, 0] + x0, e2[:, 1] + y0]))
            if len(e1_full) < 2 or len(e2_full) < 2:
                continue

            edge1_segs.append(e1_full)
            edge2_segs.append(e2_full)

            normals = result.get("normal_edge_points")
            if normals is not None:
                (e1x, e1y), (e2x, e2y) = normals
                n1_full = _finite_xy(np.column_stack([np.asarray(e1x) + x0, np.asarray(e1y) + y0]))
                n2_full = _finite_xy(np.column_stack([np.asarray(e2x) + x0, np.asarray(e2y) + y0]))
                norm1_segs.append(n1_full)
                norm2_segs.append(n2_full)
                m = min(len(n1_full), len(n2_full))
                if m >= 2:
                    d = np.sqrt(np.sum((n1_full[:m] - n2_full[:m]) ** 2, axis=1))
                    if d.size:
                        all_widths.append(d[np.isfinite(d)])
            else:
                norm1_segs.append(np.empty((0, 2)))
                norm2_segs.append(np.empty((0, 2)))

            # --- mask assembly ---
            mask_crop = result.get("mask_crop")
            if mask_crop is not None:
                mask_seg = np.array(mask_crop, np.uint8)
                x, y, w, h = bbox
                union_mask[y:y + h, x:x + w] |= (mask_seg > 0).astype(np.uint8)
            else:
                ex = np.concatenate((e1_full[:, 0][::-1], e2_full[:, 0]))
                ey = np.concatenate((e1_full[:, 1][::-1], e2_full[:, 1]))
                exc, eyc = np.clip(ex, 0, W - 1), np.clip(ey, 0, H - 1)
                area = _shoelace_area(exc, eyc)
                if area > 0.5:
                    poly = np.stack([exc, eyc], axis=1).astype(np.int32).reshape(-1, 1, 2)
                    cv2.fillPoly(union_mask, [poly], 255, lineType=cv2.LINE_AA)

        except Exception as e:
            print(f"[COMBINE_SEG] seg idx={idx} failed: {e}")
            traceback.print_exc()
            continue

    # --- final crop/bbox ---
    if np.any(union_mask):
        ys, xs = np.where(union_mask > 0)
        Y0, Y1 = int(ys.min()), int(ys.max() + 1)
        X0, X1 = int(xs.min()), int(xs.max() + 1)
        crop = union_mask[Y0:Y1, X0:X1].astype(np.uint8)
        mask_bbox = [int(X0), int(Y0), int(X1 - X0), int(Y1 - Y0)]
    else:
        crop = np.zeros((1, 1), np.uint8)
        mask_bbox = [0, 0, 1, 1]

    def _flatten(seg_list):
        out = []
        for i, arr in enumerate(seg_list):
            out.extend([[float(xx), float(yy)] for xx, yy in arr])
            if i < len(seg_list) - 1:
                out.append([None, None])
        return out

    mean_width = float(np.nanmean(np.concatenate(all_widths))) if len(all_widths) else None
    combined_length = float(sum(LineString(s).length for s in edge1_segs if len(s) >= 2))

    return {
        "source": "combined",
        "members": [str(m) for m in member_ids],
        "midline_segments": [[[float(xx), float(yy)] for (xx, yy) in s] for s in stitched],
        "midline": _flatten(stitched),
        "geodesic_edges": {"edge1": _flatten(edge1_segs), "edge2": _flatten(edge2_segs)},
        "normal_edge_points": {"edge1": _flatten(norm1_segs), "edge2": _flatten(norm2_segs)},
        "mask_crop": crop.tolist(),
        "mask_bbox": mask_bbox,
        "combined_length": combined_length,
        "mean_width": mean_width,
    }


