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
