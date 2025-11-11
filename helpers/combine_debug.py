# helpers/combine_debug.py
import os, numpy as np, pandas as pd
from scipy.spatial.distance import cdist

def _mask_from_crack(crack, H, W):
    mc, bb = crack.get("mask_crop"), crack.get("mask_bbox")
    if mc is not None and bb is not None:
        crop = np.array(mc, dtype=np.uint8)
        x,y,w,h = map(int, bb)
        x2,y2 = min(x+w, W), min(y+h, H)
        m = np.zeros((H,W), np.uint8)
        m[y:y+(y2-y), x:x+(x2-x)] = (crop>0).astype(np.uint8)[:(y2-y),:(x2-x)]
        return m
    full = np.array(crack.get("mask", []), dtype=np.uint8)
    return (full>0).astype(np.uint8) if full.size == H*W else np.zeros((H,W), np.uint8)

def diag_combine_table(annotation_dict: dict, image_hw: tuple,
                       out_csv: str, px_thresh: float = 10.0):
    """Pure function: no self. Writes a pairwise table explaining combine reasons."""
    H,W = image_hw
    atomic = (annotation_dict.get("annotations", {}) or {}).get("atomic_cracks", {}) or {}
    ids = sorted([int(k) for k in atomic.keys() if str(k).isdigit()])
    rows=[]
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            a, b = str(ids[i]), str(ids[j])
            ca, cb = atomic[a], atomic[b]
            mA = _mask_from_crack(ca,H,W); mB = _mask_from_crack(cb,H,W)
            overlap = bool(np.any(mA & mB))
            upA = set(map(tuple, ca.get("user_points",[]) or []))
            upB = set(map(tuple, cb.get("user_points",[]) or []))
            shared = bool(upA & upB)
            A = np.asarray(ca.get("midline",[]), float); B = np.asarray(cb.get("midline",[]), float)
            dmin = float(np.min(cdist(A,B))) if (A.size and B.size) else float("inf")
            prox = (dmin < px_thresh)
            why = ";".join(k for k,ok in [("overlap",overlap),("shared",shared),("prox",prox)] if ok) or "none"
            rows.append({"aid":a,"bid":b,"overlap":overlap,"shared":shared,"dmin":dmin,"prox<th":prox,"why":why})
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[COMBINE_DBG] wrote pairwise table → {out_csv}")

def auto_groups_from_atomic(annotation_dict_or_atomic, image_hw=None, px_thresh: float = 10.0) -> dict:
    """
    Automatically group atomic cracks into combined sets based on mask overlap,
    shared user points, or endpoint proximity (same logic as diag_combine_table).

    Returns a 'combined_cracks'-style dict:
        {"0": {"members": ["1","2"]}, "1": {"members": ["3","4"]}, ...}
    """
    import numpy as np
    from scipy.spatial.distance import cdist

    # Accept either raw atomic dict or full annotation JSON
    if "annotations" in (annotation_dict_or_atomic or {}):
        atomic = (annotation_dict_or_atomic.get("annotations", {}) or {}).get("atomic_cracks", {}) or {}
    else:
        atomic = dict(annotation_dict_or_atomic or {})

    ids = sorted([str(k) for k in atomic.keys()])
    if len(ids) < 2:
        return {}

    # Precompute endpoints
    endpoints = {}
    for cid in ids:
        mid = np.asarray(atomic[cid].get("midline", []), float)
        if mid.ndim == 2 and mid.shape[0] >= 2:
            endpoints[cid] = (mid[0], mid[-1])

    # Build adjacency by mask overlap or endpoint proximity
    adj = {cid: set() for cid in ids}
    H, W = image_hw if image_hw is not None else (None, None)

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            ca, cb = atomic[a], atomic[b]

            # (1) shared user points
            upA = set(map(tuple, ca.get("user_points", []) or []))
            upB = set(map(tuple, cb.get("user_points", []) or []))
            shared = bool(upA & upB)

            # (2) endpoint proximity
            prox = False
            if a in endpoints and b in endpoints:
                eA, eB = endpoints[a], endpoints[b]
                dmin = float(
                    min(
                        np.linalg.norm(np.asarray(eA[0]) - np.asarray(eB[0])),
                        np.linalg.norm(np.asarray(eA[0]) - np.asarray(eB[1])),
                        np.linalg.norm(np.asarray(eA[1]) - np.asarray(eB[0])),
                        np.linalg.norm(np.asarray(eA[1]) - np.asarray(eB[1])),
                    )
                )
                prox = dmin < px_thresh

            # (3) mask overlap
            overlap = False
            if image_hw is not None:
                from helpers.combine_debug import _mask_from_crack
                mA = _mask_from_crack(ca, H, W)
                mB = _mask_from_crack(cb, H, W)
                overlap = bool(np.any(mA & mB))

            # Combine criteria
            if shared or prox or overlap:
                adj[a].add(b)
                adj[b].add(a)

    # Connected components → groups
    seen, groups = set(), []
    for start in ids:
        if start in seen:
            continue
        comp = []
        stack = [start]
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            comp.append(v)
            stack.extend(list(adj[v]))
        groups.append(sorted(comp, key=lambda x: int(x) if x.isdigit() else x))

    return {str(i): {"members": g} for i, g in enumerate(groups) if len(g) >= 1}
