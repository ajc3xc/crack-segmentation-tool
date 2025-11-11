# helpers/combine_debug.py
import os, numpy as np, pandas as pd
from scipy.spatial.distance import cdist

def _mask_from_crack(crack, H, W):
    """
    Safely reconstruct (H,W) mask from mask_crop+mask_bbox.
    Accepts [x,y,w,h] or [x0,y0,x1,y1].
    """
    import numpy as np

    mc, bb = crack.get("mask_crop"), crack.get("mask_bbox")
    if mc is None or bb is None:
        print("There is no mask for this crack")
        return np.zeros((H, W), np.uint8)

    crop = np.asarray(mc, dtype=np.uint8)
    bb = [int(v) for v in bb]
    if len(bb) != 4:
        return np.zeros((H, W), np.uint8)

    x0, y0 = bb[0], bb[1]
    # heuristic: detect [x0,y0,x1,y1] vs [x,y,w,h]
    if bb[2] > x0 and bb[3] > y0 and (bb[2]-x0) < W and (bb[3]-y0) < H:
        x1, y1 = bb[2], bb[3]
    else:
        x1, y1 = x0 + bb[2], y0 + bb[3]

    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((H, W), np.uint8)

    mask = np.zeros((H, W), np.uint8)
    crop = (crop > 0).astype(np.uint8)
    h_t, w_t = y1 - y0, x1 - x0
    mask[y0:y0 + min(crop.shape[0], h_t), x0:x0 + min(crop.shape[1], w_t)] = crop[:h_t, :w_t]
    return mask

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
    Connected-component grouping with full diagnostics:
    - prints per-pair shared/prox/overlap + mask sizes
    - saves CSV and optional debug overlays
    """
    import numpy as np, os, pandas as pd, cv2
    from scipy.spatial.distance import cdist

    if "annotations" in (annotation_dict_or_atomic or {}):
        atomic = (annotation_dict_or_atomic.get("annotations", {}) or {}).get("atomic_cracks", {}) or {}
    else:
        atomic = dict(annotation_dict_or_atomic or {})

    ids = sorted([str(k) for k in atomic.keys()])
    if len(ids) < 2:
        print("[COMBINE_DBG] only one or zero atomics — skipping grouping.")
        return {}

    H, W = image_hw if image_hw is not None else (None, None)
    print(f"[COMBINE_DBG] auto_groups_from_atomic starting for {len(ids)} atomics (th={px_thresh})")

    endpoints = {}
    for cid in ids:
        mid = np.asarray(atomic[cid].get("midline", []), float)
        if mid.ndim == 2 and len(mid) >= 2:
            endpoints[cid] = (mid[0], mid[-1])

    adj = {cid: set() for cid in ids}
    debug_rows = []
    overlay_dir = os.path.join("combine_debug", "mask_overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    from helpers.combine_debug import _mask_from_crack
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            ca, cb = atomic[a], atomic[b]

            # shared endpoints
            upA = set(map(tuple, ca.get("user_points", []) or []))
            upB = set(map(tuple, cb.get("user_points", []) or []))
            shared = bool(upA & upB)

            # endpoint proximity
            prox = False
            dmin = np.inf
            if a in endpoints and b in endpoints:
                eA, eB = endpoints[a], endpoints[b]
                dmin = min(
                    np.linalg.norm(np.asarray(eA[0]) - np.asarray(eB[0])),
                    np.linalg.norm(np.asarray(eA[0]) - np.asarray(eB[1])),
                    np.linalg.norm(np.asarray(eA[1]) - np.asarray(eB[0])),
                    np.linalg.norm(np.asarray(eA[1]) - np.asarray(eB[1])),
                )
                prox = np.isfinite(dmin) and (dmin < px_thresh)

            # mask overlap
            overlap = False
            overlap_area = 0
            maskA_px = maskB_px = 0
            bbox_fmt = "?"
            if image_hw is not None:
                mA = _mask_from_crack(ca, H, W)
                mB = _mask_from_crack(cb, H, W)
                maskA_px = int(mA.sum())
                maskB_px = int(mB.sum())
                inter = np.logical_and(mA, mB)
                overlap_area = int(inter.sum())
                overlap = overlap_area > 0
                # guess bbox format
                bb = ca.get("mask_bbox")
                if bb is not None and len(bb) == 4:
                    x0, y0, w, h = map(int, bb)
                    if w > x0 and h > y0 and (w - x0 < W) and (h - y0 < H):
                        bbox_fmt = "xyXY"
                    else:
                        bbox_fmt = "xywh"
                # dump overlay if both masks exist but no overlap
                if maskA_px > 0 and maskB_px > 0 and not overlap:
                    vis = np.zeros((H, W, 3), np.uint8)
                    vis[..., 1] = np.clip(vis[..., 1] + (mA * 255), 0, 255)
                    vis[..., 2] = np.clip(vis[..., 2] + (mB * 255), 0, 255)
                    out = os.path.join(overlay_dir, f"pair_{a}_{b}_nooverlap.png")
                    cv2.imwrite(out, vis)

            reason = []
            if shared: reason.append("shared")
            if prox: reason.append(f"prox<{px_thresh}")
            if overlap: reason.append(f"overlap(px={overlap_area})")
            why = ";".join(reason) if reason else "none"

            debug_rows.append({
                "aid": a, "bid": b,
                "shared": shared,
                "prox<th": prox,
                "dmin": round(float(dmin), 2) if np.isfinite(dmin) else None,
                "maskA_px": maskA_px,
                "maskB_px": maskB_px,
                "bbox_fmt": bbox_fmt,
                "overlap": overlap,
                "overlap_px": overlap_area,
                "why": why
            })

            if shared or prox or overlap:
                adj[a].add(b)
                adj[b].add(a)

    df_dbg = pd.DataFrame(debug_rows)
    print(df_dbg.to_string(index=False))
    out_csv = os.path.join("combine_debug", "pairwise_debug.csv")
    df_dbg.to_csv(out_csv, index=False)
    print(f"[COMBINE_DBG] detailed pairwise debug table → {out_csv}")

    # Connected components
    seen, groups = set(), []
    for start in ids:
        if start in seen:
            continue
        stack = [start]
        comp = []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            comp.append(node)
            stack.extend(adj[node])
        groups.append(sorted(comp, key=lambda x: int(x) if x.isdigit() else x))

    print(f"[COMBINE_DBG] found {len(groups)} groups: {groups}")
    return {str(i): {"members": g} for i, g in enumerate(groups) if len(g) >= 2}
