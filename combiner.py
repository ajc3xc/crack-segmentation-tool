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
        ml = np.asarray(cr.get("midline", []), float)

        if ml.ndim != 2 or len(ml) < 2:
            continue

        ml = _finite_xy(ml)
        if len(ml) < 2:
            continue

        atomics.append((str(m), ml))
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

def _atomic_mask_global(cr, H, W):
    """
    Expand per-atomic mask_crop into global image space.
    """
    mc = cr.get("mask_crop")
    bb = cr.get("mask_bbox")

    if mc is None or bb is None:
        return None

    x, y, w, h = map(int, bb)
    m = np.zeros((H, W), np.uint8)
    mc = np.asarray(mc, np.uint8)
    m[y:y+h, x:x+w] = mc[:h, :w]
    return m
    
def dominant_segments_from_group(
    *,
    members,
    atomic,
    crack_mask_u8=None,
    window_half_size,
    debug_dir=None,
    debug_tag="group",
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

    def _union_member_bbox_xyxy(member_ids, W, H):
        boxes = []
        for m in member_ids:
            bb = (atomic.get(str(m), {}) or {}).get("mask_bbox")
            if bb and len(bb) == 4:
                x, y, w, h = map(int, bb)
                if w > 0 and h > 0:
                    boxes.append((x, y, x + w, y + h))
        if not boxes:
            return None
        bx0 = max(0, min(b[0] for b in boxes))
        by0 = max(0, min(b[1] for b in boxes))
        bx1 = min(W, max(b[2] for b in boxes))
        by1 = min(H, max(b[3] for b in boxes))
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

    # -----------------------------
    # 1) collect atomics + endpoints
    # -----------------------------
    atomics = []     # [(cid_str, S_user)]
    endpoints = []   # [set((x,y), ...)]

    for m in members:
        cr = atomic.get(str(m), {}) or {}
        ml = np.asarray(cr.get("midline", []), float)
        if ml.ndim == 2 and len(ml) >= 2:
            atomics.append((str(m), _finite_xy(ml)))
            endpoints.append(get_user_endpoints(cr))

    if not atomics:
        return [], {"branches": [], "segments_meta": [], "bite": None}

    # -----------------------------
    # 1.5) mask / canvas init (NOW atomics exists)
    # -----------------------------
    use_dt_territory = (crack_mask_u8 is not None)   # DT territory needs real crack mask
    use_atomic_mask_territory = (crack_mask_u8 is None)

    if crack_mask_u8 is not None:
        H, W = crack_mask_u8.shape[:2]
        crack_mask = (crack_mask_u8 > 0).astype(np.uint8)
    else:
        # fallback canvas from USER geometry
        xs = []
        ys = []
        for _, S in atomics:
            if S is None or len(S) == 0:
                continue
            fx = S[:, 0]
            fy = S[:, 1]
            fx = fx[np.isfinite(fx)]
            fy = fy[np.isfinite(fy)]
            if len(fx):
                xs.append(float(np.nanmax(fx)))
            if len(fy):
                ys.append(float(np.nanmax(fy)))

        W = int(max(xs) + 2) if xs else 2
        H = int(max(ys) + 2) if ys else 2
        crack_mask = None

    # bbox for cropped bite storage
    bbox_xyxy = _union_member_bbox_xyxy(members, W, H)
    if bbox_xyxy is None:
        bx0, by0, bx1, by1 = 0, 0, W, H
    else:
        bx0, by0, bx1, by1 = bbox_xyxy

    # -----------------------------
    # 2) build branches in atomic space
    # -----------------------------
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

    # -----------------------------
    # 3) per-branch user length + seg lists
    # -----------------------------
    branch_user_len = []
    branch_user_segs = []        # list[list[(atomic_id, USER_polyline)]]
    branch_terr_segs = []        # list[list[polyline_piece]] (ONLY used for DT territory mode)

    for atom_ids in branches:
        total_len = 0.0
        user_segs = []
        terr_segs = []

        for ai in atom_ids:
            atomic_id, S_user = atomics[ai]
            if S_user is None or len(S_user) < 2:
                continue

            total_len += _linestring_length(S_user)
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

    ''' # -----------------------------
    # 4) dominance between branches (ordered by USER length)
    # -----------------------------
    if use_dt_territory:
        dt = cv2.distanceTransform(crack_mask, cv2.DIST_L2, 5)

        def seg_radius(S):
            ys = np.clip(np.round(S[:, 1]).astype(int), 0, H - 1)
            xs = np.clip(np.round(S[:, 0]).astype(int), 0, W - 1)
            d = dt[ys, xs]
            d = d[np.isfinite(d)]
            if len(d) == 0:
                return 0.3 * window_half_size
            return max(3.0, min(float(np.median(d)), window_half_size))
    else:
        dt = None

        def seg_radius(S):
            return max(3.0, 0.5 * float(window_half_size))

    order = sorted(
        range(len(branches)),
        key=lambda i: branch_user_len[i],
        reverse=True,
    )

    claimed = np.zeros((H, W), np.uint8)
    bite_total = np.zeros((H, W), np.uint8)

    kept_meta = []
    branch_stats = []
    primary_branch = order[0] if order else None

    branch_terr_masks = {}  # debug only

    for rank, bi in enumerate(order):

        # -------------------------------------------------
        # Build territory mask for this branch
        # -------------------------------------------------
        branch_terr = np.zeros((H, W), np.uint8)

        if use_atomic_mask_territory:
            # === ATOMIC-MASK TERRITORY (authoritative) ===
            for atomic_id, _ in branch_user_segs[bi]:
                cr = atomic.get(str(atomic_id), {}) or {}
                am = _atomic_mask_global(cr, H, W)  # expects global H,W
                if am is not None:
                    branch_terr |= (am > 0).astype(np.uint8)

        else:
            # === DT TERRITORY (unchanged) ===
            for S_clip in branch_terr_segs[bi]:
                if S_clip is None or len(S_clip) < 2:
                    continue

                r = seg_radius(S_clip)
                rad = int(max(4, 1.3 * r))
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1)
                )

                line = _polyline_mask(S_clip, H, W)
                terr = cv2.dilate(line, kernel, iterations=1) & crack_mask
                branch_terr |= terr

        branch_terr_masks[bi] = branch_terr.copy()

        # -------------------------------------------------
        # Suppression gate
        # -------------------------------------------------
        if use_atomic_mask_territory:
            terr_len = float(branch_user_len[bi])
        else:
            terr_len = (
                _linestring_length(np.vstack(branch_terr_segs[bi]))
                if branch_terr_segs[bi]
                else 0.0
            )

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

        else:

            # -------------------------------------------------
            # PRIMARY branch: keep FULL USER geometry
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

            # -------------------------------------------------
            # SUBORDINATE branches: clip OUT claimed territory
            # -------------------------------------------------
            else:
                bite = branch_terr & claimed

                # --- AUTHORITATIVE bite expansion (atomic-mask mode only) ---
                if use_atomic_mask_territory and np.any(bite):
                    bite = cv2.dilate(
                        bite,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                        iterations=1,
                    )

                if np.any(bite):
                    bite_total |= bite
                    claimed |= bite   # <-- makes dilation affect cutting

                kept_any = False
                kept_len = 0.0

                if crack_mask is not None:
                    allowed = ((crack_mask > 0) & (claimed == 0)).astype(np.uint8)
                else:
                    allowed = (claimed == 0).astype(np.uint8)

                for atomic_id, S_user in branch_user_segs[bi]:
                    if S_user is None or len(S_user) < 2:
                        continue

                    pieces = _clip_polyline_to_mask(S_user, allowed)
                    for p in pieces:
                        if p is not None and len(p) >= 2:
                            kept_meta.append((bi, atomic_id, p, False))
                            kept_any = True
                            kept_len += _linestring_length(p)

                branch_stats.append({
                    "branch_id": int(bi),
                    "rank": int(rank),
                    "atomic_ids": [aid for (aid, _) in branch_user_segs[bi]],
                    "user_len": float(branch_user_len[bi]),
                    "kept_len": float(kept_len),
                    "suppressed": not kept_any,
                })

        # -------------------------------------------------
        # CRITICAL INVARIANT:
        # Territory is always claimed, even if geometry is fully deleted
        # -------------------------------------------------
        claimed |= branch_terr'''
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
            am = _atomic_mask_global(cr, H, W)
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
    bite_total = np.zeros((H, W), np.uint8)

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

    for rank, bi in enumerate(order):

        # -------------------------------------------------
        # Build DT-based territory
        # -------------------------------------------------
        branch_terr = np.zeros((H, W), np.uint8)

        for atomic_id, S_user in branch_user_segs[bi]:
            if S_user is None or len(S_user) < 2:
                continue

            r = seg_radius(S_user)
            rad = int(max(3, r))

            line = _polyline_mask(S_user, H, W)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1)
            )

            terr = cv2.dilate(line, kernel, iterations=1)
            terr &= domain_mask
            branch_terr |= terr

        branch_terr_masks[bi] = branch_terr.copy()

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
        # hard exclusion: DT territory + dominant atomic masks
        forbidden = claimed.copy()
        for obi in order[:rank]:
            forbidden |= branch_atomic_masks[obi]

        bite = branch_terr & forbidden
        if np.any(bite):
            bite_total |= bite

        allowed = (forbidden == 0).astype(np.uint8)

        kept_any = False
        kept_len = 0.0

        for atomic_id, S_user in branch_user_segs[bi]:
            if S_user is None or len(S_user) < 2:
                continue

            pieces = _clip_polyline_to_mask(S_user, allowed.astype(np.uint8))
            for p in pieces:
                if p is not None and len(p) >= 2:
                    kept_meta.append((bi, atomic_id, p, False))
                    kept_any = True
                    kept_len += _linestring_length(p)

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
    kept = [S for (_, _, S, _) in kept_meta]

    segments_meta = []
    for (bi, atomic_id, S, is_primary) in kept_meta:
        segments_meta.append({
            "branch_id": int(bi),
            "atomic_id": str(atomic_id),
            "is_primary": bool(is_primary),
            "branch_rank": int(order.index(bi)) if bi in order else -1,
            "length": float(_linestring_length(S)),
        })

    bite_crop = bite_total[by0:by1, bx0:bx1].astype(np.uint8)
    bite_blob = _pack_mask_b64(bite_crop)

    meta = {
        "members": [str(m) for m in members],
        "primary_branch_id": int(primary_branch) if primary_branch is not None else None,
        "order": [int(b) for b in order],
        "branches": branch_stats,
        "segments_meta": segments_meta,
        "bite": {
            "bbox": [int(bx0), int(by0), int(bx1 - bx0), int(by1 - by0)],
            "shape": bite_blob["shape"],
            "packbits_b64": bite_blob["packbits_b64"],
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
    # DEBUG: PRE (territory + bite + user)
    # -----------------------------
    if debug_dir:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.lines import Line2D
        import numpy as np
        import os

        os.makedirs(debug_dir, exist_ok=True)

        pad = 30
        x0 = max(0, bx0 - pad)
        y0 = max(0, by0 - pad)
        x1 = min(W, bx1 + pad)
        y1 = min(H, by1 + pad)

        Hc, Wc = y1 - y0, x1 - x0

        fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
        ax.imshow(plot_mask[y0:y1, x0:x1], cmap="gray", zorder=0)

        # 5 branch colors (identity only)
        branch_colors = [
            np.array([0.18, 0.80, 0.32]),  # green
            np.array([0.93, 0.54, 0.16]),  # orange
            np.array([0.20, 0.50, 0.90]),  # blue
            np.array([0.62, 0.35, 0.75]),  # purple
            np.array([0.10, 0.75, 0.70]),  # teal
        ]

        # -------------------------------------------------
        # TERRITORY (below user)
        # -------------------------------------------------
        for bi, terr in branch_terr_masks.items():
            bm = terr[y0:y1, x0:x1]
            if not np.any(bm):
                continue
            rgba = np.zeros((Hc, Wc, 4), float)
            rgba[..., :3] = branch_colors[bi % len(branch_colors)]
            rgba[..., 3] = 0.25 * (bm > 0)
            ax.imshow(rgba, zorder=1)

        # -------------------------------------------------
        # USER GEOMETRY — draw short -> long so long is on top
        # -------------------------------------------------
        order_by_len = sorted(range(len(branch_user_segs)), key=lambda i: branch_user_len[i])

        for rank, bi in enumerate(order_by_len):
            col = branch_colors[bi % len(branch_colors)]
            z = 3 + rank

            # user geometry
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

            # label at midpoint of longest user segment (white box)
            if branch_user_segs[bi]:
                longest = max(branch_user_segs[bi], key=lambda t: _linestring_length(t[1]))[1]
                mid = longest[len(longest) // 2]
                if x0 <= mid[0] <= x1 and y0 <= mid[1] <= y1:
                    mx, my = mid - np.array([x0, y0])
                    ax.text(
                        mx, my,
                        f"{int(branch_user_len[bi])} px",
                        fontsize=8,
                        color="black",
                        ha="center",
                        va="center",
                        zorder=z + 0.2,
                        bbox=dict(
                            boxstyle="round,pad=0.25",
                            facecolor="white",
                            edgecolor="none",
                            alpha=0.85,
                        ),
                    )

        # -------------------------------------------------
        # BITE (between territory and user)
        # -------------------------------------------------
        bite_crop = bite_total[y0:y1, x0:x1]
        if np.any(bite_crop):
            bite_rgba = np.zeros((Hc, Wc, 4), float)
            bite_rgba[..., 0] = 1.0
            bite_rgba[..., 3] = 0.45 * (bite_crop > 0)
            ax.imshow(bite_rgba, zorder=2)

        # -------------------------------------------------
        # BBOX
        # -------------------------------------------------
        ax.add_patch(
            Rectangle(
                (bx0 - x0, by0 - y0),
                bx1 - bx0,
                by1 - by0,
                fill=False,
                edgecolor="#0033cc",
                linewidth=1.5,
                zorder=10,
            )
        )

        # -------------------------------------------------
        # LEGEND (colors = branch identity)
        # -------------------------------------------------
        legend_items = [
            Line2D([0], [0], color="white", lw=3, label="User midlines (colored per branch)"),
            Line2D([0], [0], color="black", lw=0, label="Branch colors = identity only"),
            Line2D([0], [0], color="red", lw=6, alpha=0.5, label="Bite (dominance overlap)"),
            Line2D([0], [0], color="#0033cc", lw=1.5, label="BBox"),
        ]
        leg = ax.legend(handles=legend_items, loc="lower right", fontsize=8, framealpha=0.9)
        leg.set_zorder(100)

        ax.set_title(f"{debug_tag} — PRE", fontsize=10)
        ax.axis("off")

        out = os.path.join(debug_dir, f"{debug_tag}_pre.png")
        fig.savefig(out, bbox_inches="tight", dpi=200)
        plt.close(fig)

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
            "#2ecc71",  # green
            "#e67e22",  # orange
            "#3498db",  # blue
            "#9b59b6",  # purple
            "#1abc9c",  # teal
        ]

        # --- bite (make it visible) ---
        bite_crop = bite_total[y0:y1, x0:x1]
        if np.any(bite_crop):
            bite_rgba = np.zeros((Hc, Wc, 4), float)
            bite_rgba[..., 0] = 1.0
            bite_rgba[..., 3] = 0.55 * (bite_crop > 0)
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
                color="black",
                ha="center",
                va="center",
                zorder=20,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.9,
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
                label="User midlines (only if clipped)"),
            Line2D([0], [0], color="black", lw=3,
                label="Kept midlines (branch-colored; primary drawn on top)"),
            Line2D([0], [0], color="red", lw=6, alpha=0.55,
                label="Bite (dominance overlap)"),
            Line2D([0], [0], color="#0033cc", lw=1.5, label="BBox"),
        ]
        leg = ax.legend(handles=legend_items, loc="lower right", fontsize=8, framealpha=0.9)
        leg.set_zorder(100)

        ax.set_title(f"{debug_tag} — FINAL", fontsize=10)
        ax.axis("off")

        out = os.path.join(debug_dir, f"{debug_tag}_final.png")
        fig.savefig(out, bbox_inches="tight", dpi=200)
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
    
    return kept, meta




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
    debug_dir=None,
    debug_callback=None,
    crack_mask_full: np.ndarray = None,   # ← NEW (H,W) uint8/bool
    mode = "new"
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
        
    H, W = original_image.shape[:2]

    if crack_mask_full is None:
        raise ValueError("build_combined_crack_stateless: crack_mask_full is required")

    if crack_mask_full.shape[:2] != (H, W):
        raise ValueError(
            "build_combined_crack_stateless: crack_mask_full shape mismatch"
        )

    crack_mask_full = (crack_mask_full > 0).astype(np.uint8)

    segs, dom_meta = dominant_segments_from_group(
        members=member_ids,
        atomic=authoring_atomic,
        #crack_mask_u8=crack_mask_full,
        window_half_size=window_half_size,
        debug_dir=debug_dir,
    )

    # attach metadata so width eval can do opsec
    midline_segments_meta = dom_meta.get("segments_meta", [])
    dominance_meta = {"bite": dom_meta.get("bite")}

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
        
        # -----------------------------
        # INSTRUMENTATION: pre-edge audit
        # -----------------------------
        n_pts = track_local_yx.shape[1]
        if n_pts < 3:
            print(
                f"[COMBINER AUDIT] ⚠ short segment skipped: "
                f"n_pts={n_pts}, "
                f"bbox=({x0},{y0})-({x1},{y1}), "
                f"pad={pad}"
            )
            continue

        
        em1, em2 = edge_masks(
            crop.astype(np.uint8),
            track_local_yx,
            window_half_size=window_half_size,
            mode=mode
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
            prefer_gpu=prefer_gpu,
            mode=mode
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

        # ---- lift edges into full image coordinates (NO MODIFICATION) ----
        e1 = _finite_xy(np.asarray(e1, float))
        e2 = _finite_xy(np.asarray(e2, float))

        if len(e1) < 2 or len(e2) < 2:
            raise ValueError(
                "postprocess: edge1 / edge2 have fewer than 2 valid points"
            )

        e1_full = _finite_xy(np.column_stack([e1[:, 0] + x0, e1[:, 1] + y0]))
        e2_full = _finite_xy(np.column_stack([e2[:, 0] + x0, e2[:, 1] + y0]))

        if len(e1_full) < 2 or len(e2_full) < 2:
            raise ValueError(
                "postprocess: lifted edges invalid after bbox offset"
            )

        # ---- normals (metrics only; NOT used for mask) ----
        normals = res.get("normal_edge_points")
        if normals is not None:
            (e1x, e1y), (e2x, e2y) = normals

            n1_full = _finite_xy(
                np.column_stack([np.asarray(e1x) + x0, np.asarray(e1y) + y0])
            )
            n2_full = _finite_xy(
                np.column_stack([np.asarray(e2x) + x0, np.asarray(e2y) + y0])
            )

            m = min(len(n1_full), len(n2_full))
            if m >= 2:
                d = np.sqrt(np.sum((n1_full[:m] - n2_full[:m]) ** 2, axis=1))
                if d.size:
                    all_widths.append(d[np.isfinite(d)])
        else:
            n1_full = np.empty((0, 2))
            n2_full = np.empty((0, 2))

        # ---- store geometry (authoritative) ----
        edge1_segs.append(e1_full)
        edge2_segs.append(e2_full)
        norm1_segs.append(n1_full)
        norm2_segs.append(n2_full)

        from cracktools.segmentation import generate_mask_from_edges
        # ---- MASK GENERATION (AUTHORITATIVE) ----
        mask_seg = generate_mask_from_edges(
            img_gray=gray_full,          # FULL image, not crop
            edge1_xy=e1_full,
            edge2_xy=e2_full,
            midline_xy=S,                # DEBUG ONLY
            out_dir=None,                # no disk writes here
            tag=None,
            do_morph=False,
        )

        if mask_seg is None or not mask_seg.any():
            raise ValueError(
                "postprocess: generated mask is empty — invalid edge geometry"
            )

        union_mask |= (mask_seg > 0).astype(np.uint8)

        t_post1 = time.perf_counter()
        t_post_total += (t_post1 - t_post0)
        t_loop_total += (time.perf_counter() - t_loop0)


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
    
    semantic_id = "_".join(str(m) for m in member_ids)

    return {
        "source": "combined",
        "semantic_id": semantic_id,
        "mode": mode,
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
        "midline_segments_meta": midline_segments_meta,
        "dominance_meta": dominance_meta,
    }
