import json
import numpy as np
ROUNDING_DIGITS=6


def _to_py(obj, ndigits=6):
    """
    Recursively converts NumPy / CuPy arrays, pandas, etc. to plain
    Python lists, rounding floats to `ndigits` decimals for compact JSON.
    """
    import numpy as np

    if obj is None:
        return None
    if isinstance(obj, (int, bool, str)):
        return obj
    if isinstance(obj, float):
        # round floats directly
        return round(obj, ndigits)
    if isinstance(obj, (list, tuple)):
        return [_to_py(x, ndigits) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_py(v, ndigits) for k, v in obj.items()}
    if hasattr(obj, "tolist"):  # numpy / cupy array
        return _to_py(obj.tolist(), ndigits)
    return obj


def safe_json_dump(data, path, compact=True):
    """Atomic JSON writer — supports compact (semi-human) or fully minified mode."""
    import os, tempfile, json
    d = _to_py(data)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path)+".", suffix=".tmp",
                            dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            if compact:
                # minimal spacing, arrays inline
                json.dump(d, f, ensure_ascii=False, indent=None, separators=(',', ':'))
            else:
                # readable multi-line, smaller indent
                json.dump(d, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try: os.remove(tmp)
        except Exception: pass
        raise


def _normals_to_json(normals, xmin, ymin, ndigits=ROUNDING_DIGITS):
    import numpy as np

    def to_xy2(arr):
        a = np.asarray(arr, float)
        if a.ndim == 2 and a.shape[1] == 2:       # already Nx2
            x, y = a[:, 0], a[:, 1]
        elif a.ndim == 2 and a.shape[0] == 2:     # 2xN ( [xlist, ylist] )
            x, y = a[0], a[1]
        elif a.ndim == 1:                         # degenerate 1-D
            x, y = a, np.full_like(a, np.nan, dtype=float)
        else:
            x = y = np.array([], float)

        x = x + float(xmin)
        y = y + float(ymin)

        out = np.stack([x, y], axis=1)
        # round
        if np.isfinite(out).any():
            out = np.round(out, ndigits=ndigits, where=np.isfinite(out))
        # JSON-safe NaNs
        out[~np.isfinite(out)] = None
        return out.tolist()

    # dict form: {"edge1":[xlist,ylist], "edge2":[xlist,ylist]} or {"edge1":Nx2,...}
    if isinstance(normals, dict):
        e1 = normals.get("edge1", [])
        e2 = normals.get("edge2", [])
        # accept either [xlist,ylist] or Nx2
        e1 = e1 if isinstance(e1, (list, tuple)) else []
        e2 = e2 if isinstance(e2, (list, tuple)) else []
        e1 = to_xy2(e1)
        e2 = to_xy2(e2)
        return {"edge1": e1, "edge2": e2}

    # tuple/list form: ((e1x,e1y), (e2x,e2y))
    try:
        (e1x, e1y), (e2x, e2y) = normals
        return {"edge1": to_xy2([e1x, e1y]), "edge2": to_xy2([e2x, e2y])}
    except Exception:
        return {"edge1": [], "edge2": []}

def _json_has_manual_midlines(ann_path: str) -> bool:
    """Lightweight on-disk check: does JSON have any manual midline with ≥2 points?
       Also purges stale annotations that contain no valid atomic cracks.
    """
    import json, os
    try:
        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except UnicodeDecodeError:
        with open(ann_path, "r", encoding="cp1252", errors="strict") as f:
            data = json.load(f)
    except Exception:
        return False

    ann = (data or {}).get("annotations", {}) or {}
    atomic = ann.get("atomic_cracks", {}) or {}

    # --- purge safety: no atomic cracks or all midlines empty ---
    if not atomic or all(len(cr.get("midline", [])) < 2 for cr in atomic.values()):
        print(f"[global-metrics] 🧹 purging stale JSON with no manual midlines: {ann_path}")
        try:
            os.remove(ann_path)
        except Exception as e:
            print(f"[global-metrics] ⚠ failed to remove {ann_path}: {e}")
        return False

    # --- otherwise: check for ≥2-pt manual midlines ---
    for crack in atomic.values():
        src = (crack.get("source") or "").lower()
        if src.startswith("auto") or src == "combined":
            continue
        mid = crack.get("midline", [])
        if isinstance(mid, list) and len(mid) >= 2:
            return True
    return False

def safe_json_dump(data, path, compact=True):
    """Atomic JSON writer — supports compact (semi-human) or fully minified mode."""
    import os, tempfile, json
    d = _to_py(data)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path)+".", suffix=".tmp",
                            dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            if compact:
                # minimal spacing, arrays inline
                json.dump(d, f, ensure_ascii=False, indent=None, separators=(',', ':'))
            else:
                # readable multi-line, smaller indent
                json.dump(d, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try: os.remove(tmp)
        except Exception: pass
        raise
    
# ======================== metrics.py (snapshot helpers) ========================
import os, json, hashlib, math
import numpy as np

# --- small filesystem helpers -------------------------------------------------
def _ensure_dir(p): os.makedirs(p, exist_ok=True); return p

def metric_snapshot_root(save_folder, image_base):
    """metrics/<image>/snapshot/"""
    return _ensure_dir(os.path.join(save_folder, "metrics", image_base, "snapshot"))

def metric_atomic_dir(save_folder, image_base):
    """metrics/<image>/snapshot/atomic/"""
    return _ensure_dir(os.path.join(metric_snapshot_root(save_folder, image_base), "atomic"))

'''def metric_atomic_path_for(save_folder, image_base, crack_id):
    return os.path.join(metric_atomic_dir(save_folder, image_base), f"cid{crack_id}_metrics.json")'''
    
# -------------------- SNAPSHOT PATHS --------------------
def metric_image_dir(save_folder, base_name):
    return os.path.join(save_folder, "metrics", base_name)

'''def metric_atomic_path_for(save_folder, base_name, crack_id):
    return os.path.join(save_folder, "metrics", base_name, f"cid{crack_id}.json")'''
    
def metric_atomic_path_for(save_folder, base_name, crack_id):
    """
    Canonical atomic snapshot path:
        metrics/<image_base>/cid{cid}/cid{cid}.json
    """
    import os
    cid_str = str(crack_id)
    dir_path = os.path.join(save_folder, "metrics", base_name, f"cid{cid_str}")
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, f"cid{cid_str}.json")

def metric_combined_path(save_folder, image_base):
    return os.path.join(metric_snapshot_root(save_folder, image_base), "combined.json")

def safe_read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

# in helpers/metrics.py
#from helpers.metrics import _to_py   # if not already in the same file

def safe_write_json(path, data):
    """Drop-in replacement that auto-converts NumPy arrays."""
    import json, os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_to_py(data), f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

# --- snapshot assembly / persistence ------------------------------------------
def snapshot_pick_crack_fields(cr):
    """Keep only fields metrics care about (read-only in snapshot)."""
    keep = {}
    for k in ("source","midline","mask_bbox","geodesic_edges",
              "normal_edge_points_full","normal_edge_points","mask_crop"):
        if k in cr: keep[k] = cr[k]
    # do NOT keep bulky 'variants' tree in authoring snapshot; we will store per-crack auto artifacts separately
    return keep

def snapshot_from_authoring(authoring_ann, cache_key=None):
    """
    Build an in-memory snapshot dict *without touching disk*.
    You will later persist per-crack files with split_snapshot_to_files().
    authoring_ann: {'atomic_cracks':{...}, 'combined_cracks':{...}}  (the .get('annotations', {}) block)
    """
    atomic_src   = (authoring_ann or {}).get("atomic_cracks", {}) or {}
    combined_src = (authoring_ann or {}).get("combined_cracks", {}) or {}

    atomic = { cid: snapshot_pick_crack_fields(cr) for cid, cr in atomic_src.items() }
    combined = {}
    for k, cmb in combined_src.items():
        cc = {}
        for fld in ("members","midline","mask_bbox","normal_edge_points_full","normal_edge_points"):
            if fld in cmb: cc[fld] = cmb[fld]
        # store optional 'auto' minimal fields, if present on authoring (not required)
        if "auto" in cmb and isinstance(cmb["auto"], dict):
            a = {}
            for fld in ("midline","mask_bbox","normal_edge_points_full","normal_edge_points"):
                if fld in cmb["auto"]: a[fld] = cmb["auto"][fld]
            cc["auto"] = a
        combined[k] = cc

    # auto_best is not taken from authoring variants here; it’s populated later from per-crack files
    return {"atomic_cracks": atomic, "combined_cracks": combined, "auto_best_atomic_cracks": {}}

def split_snapshot_to_files(snapshot, save_folder, image_base, merge_if_exists=True):
    """
    Persist per-crack snapshot: one JSON per crack + one combined.json.
    If merge_if_exists=True, keep previously computed fields (e.g., tracked edges, auto_best) already on disk.
    """
    atomic = snapshot.get("atomic_cracks", {}) or {}
    for cid, cr in atomic.items():
        p = metric_atomic_path_for(save_folder, image_base, cid)
        if merge_if_exists:
            old = safe_read_json(p, {})
            # merge old computed stuff (e.g., autotrack, auto_best) into new minimal authoring view
            for k in ("geodesic_edges","normal_edge_points_full","normal_edge_points",
                      "mask_crop","auto_best"):
                if k in old and k not in cr:
                    cr[k] = old[k]
        safe_write_json(p, cr)
        #safe_json_dump(cnew, cpath)

    # combined
    cpath = metric_combined_path(save_folder, image_base)
    cold  = safe_read_json(cpath, {}) if merge_if_exists else {}
    cnew  = snapshot.get("combined_cracks", {}) or {}
    # merge optional 'auto' sub if it already lived on disk
    for k, v in (cold or {}).items():
        if k in cnew and isinstance(v, dict) and "auto" in v and "auto" not in cnew[k]:
            cnew[k]["auto"] = v["auto"]
    safe_write_json(cpath, cnew)
    #safe_json_dump(cnew, cpath)
    
'''def load_snapshot_from_files(save_folder, base_name):
    """Load all cid*.json under metrics/<base> into a dict."""
    root = metric_image_dir(save_folder, base_name)
    out = {"atomic_cracks": {}, "combined_cracks": {}, "auto_best": {}}
    if not os.path.isdir(root):
        return out
    for fn in os.listdir(root):
        if not fn.startswith("cid") or not fn.endswith(".json"): 
            continue
        p = os.path.join(root, fn)
        rec = safe_read_json(p, {})
        if not rec: 
            continue
        cid = rec.get("crack_id")
        if cid is None:
            # fallback from filename
            try: cid = int(fn[3:-5])
            except: continue
        out["atomic_cracks"][cid] = rec
        if "auto_best" in rec and rec["auto_best"]:
            out["auto_best"][cid] = rec["auto_best"]
    return out'''
    
# metrics.py  --- add this inside load_snapshot_from_files(...)
'''def load_snapshot_from_files(save_folder, base_name):
    """
    Load snapshot for an image from disk.

    Supports:
      - NEW: metrics/<base>/cid{cid}/cid{cid}.json
      - LEGACY: metrics/<base>/cid{cid}.json

    Also loads combined snapshot from:
      metrics/<base>/snapshot/combined.json
    """
    root = metric_image_dir(save_folder, base_name)
    out = {
        "atomic_cracks": {},
        "combined_cracks": {},
        "auto_best_atomic_cracks": {},
    }

    if not os.path.isdir(root):
        return out

    # --- load combined snapshot if it exists ---
    try:
        cpath = metric_combined_path(save_folder, base_name)
        combined = safe_read_json(cpath, {}) or {}
        if isinstance(combined, dict):
            out["combined_cracks"] = combined
    except Exception as e:
        print(f"[snapshot] ⚠ failed loading combined snapshot: {e}")

    # --- scan atomic snapshots ---
    for entry in os.listdir(root):
        full = os.path.join(root, entry)

        # 1) NEW nested layout: metrics/<base>/cid0/cid0.json
        if os.path.isdir(full) and entry.startswith("cid"):
            cid_str = entry[3:]  # e.g. "0" from "cid0"
            # first try cid0/cid0.json
            cand1 = os.path.join(full, f"{entry}.json")
            # fallback: cid0/0.json (in case you ever used that)
            cand2 = os.path.join(full, f"{cid_str}.json")

            for jp in (cand1, cand2):
                rec = safe_read_json(jp, None)
                if not (isinstance(rec, dict) and rec):
                    continue

                cid = rec.get("crack_id", cid_str)
                cid_key = str(cid)

                out["atomic_cracks"][cid_key] = rec
                if "auto_best" in rec and rec["auto_best"]:
                    out["auto_best_atomic_cracks"][cid_key] = rec["auto_best"]
                break

            continue  # don’t treat the directory as a flat file

        # 2) LEGACY flat layout: metrics/<base>/cid0.json
        if os.path.isfile(full) and entry.startswith("cid") and entry.endswith(".json"):
            rec = safe_read_json(full, None)
            if not (isinstance(rec, dict) and rec):
                continue

            cid = rec.get("crack_id")
            if cid is None:
                try:
                    # strip "cid" and ".json"
                    cid = entry[3:-5]  # keep as string; we'll normalize below
                except Exception:
                    continue

            cid_key = str(cid)
            out["atomic_cracks"][cid_key] = rec
            if "auto_best" in rec and rec["auto_best"]:
                out["auto_best_atomic_cracks"][cid_key] = rec["auto_best"]

    return out'''
    
def load_snapshot_from_files(save_folder, base_name):
    """
    Load snapshot for an image from disk.

    Supports:
      - NEW: metrics/<base>/cid{cid}/cid{cid}.json
      - LEGACY: metrics/<base>/cid{cid}.json
    Falls back to authoring <base>.json if no cid files exist yet.
    """
    root = metric_image_dir(save_folder, base_name)
    out = {
        "atomic_cracks": {},
        "combined_cracks": {},
        "auto_best_atomic_cracks": {},
    }

    if not os.path.isdir(root):
        return out

    # === load nested + legacy per-cid files ===
    for entry in os.listdir(root):
        full = os.path.join(root, entry)

        # NEW nested layout: metrics/base/cid0/cid0.json
        if os.path.isdir(full) and entry.startswith("cid"):
            cid_str = entry[3:]
            cand1 = os.path.join(full, f"{entry}.json")
            cand2 = os.path.join(full, f"{cid_str}.json")

            for jp in (cand1, cand2):
                rec = safe_read_json(jp, None)
                if isinstance(rec, dict) and rec:
                    cid_key = str(rec.get("crack_id", cid_str))
                    out["atomic_cracks"][cid_key] = rec
                    if "auto_best" in rec:
                        out["auto_best_atomic_cracks"][cid_key] = rec["auto_best"]
                    break
            continue

        # LEGACY flat layout
        if os.path.isfile(full) and entry.startswith("cid") and entry.endswith(".json"):
            rec = safe_read_json(full, None)
            if isinstance(rec, dict) and rec:
                cid = rec.get("crack_id", entry[3:-5])
                cid_key = str(cid)
                out["atomic_cracks"][cid_key] = rec
                if "auto_best" in rec:
                    out["auto_best_atomic_cracks"][cid_key] = rec["auto_best"]

    # === fallback: authoring JSON ===
    if not out["atomic_cracks"]:
        authoring_path = os.path.join(save_folder, f"{base_name}.json")
        ann = safe_read_json(authoring_path, {}) or {}
        anns = ann.get("annotations", {})

        atomics = anns.get("atomic_cracks", {})
        for cid, entry in (atomics or {}).items():
            out["atomic_cracks"][str(cid)] = entry

        combined = anns.get("combined_cracks", {})
        if combined:
            out["combined_cracks"] = combined

    return out

def snapshot_fingerprint(snapshot):
    j = json.dumps(snapshot or {}, sort_keys=True, separators=(",",":"))
    return hashlib.sha1(j.encode("utf-8")).hexdigest()

def _flatten_variant_record(vid: int, vrec: dict, params: dict, scores: dict = None):
    """
    Keep bottom-level clean and single-line-ish:
    {
      "variant_id": 2, "midline": [[...],[...]], 
      "params": {"g11":1,"g22":35,"g33":25,"win":45,"mu":0,"ell":5,"p":14},
      "scores": {"chamfer_mean":..., "hausdorff":..., "coverage":...},
      "normal_edge_points": {"edge1":[[x,y],...], "edge2":[[x,y],...]}
    }
    """
    out = {
        "variant_id": int(vid),
        "midline": vrec.get("midline", []),
        "params": params or {},
        "scores": scores or {},
    }
    for k in ("normal_edge_points_full", "normal_edge_points"):
        if k in vrec and vrec[k]:
            out["normal_edge_points"] = vrec[k]
            break
    return out
    
def set_auto_variant_for_crack(save_folder, base_name, crack_id, vrec, params=None, is_best=False, scores=None):
    """
    Persist a variant under atomic crack snapshot (flat).
    Ensures keys: auto_variants (dict of vN -> {...}), auto_best (single flat dict).
    """
    p = metric_atomic_path_for(save_folder, base_name, crack_id)
    rec = safe_read_json(p, {})
    if not rec: 
        rec = {"crack_id": crack_id}

    # Normalize container
    av = rec.get("auto_variants")
    if not isinstance(av, dict):
        av = {}
    vid = None
    if isinstance(vrec, dict):
        # best effort: try to extract known id from params/desc
        vid = vrec.get("params", {}).get("variant_id", None)
    # fallback from caller
    if vid is None:
        # caller usually passes 'v<id>' externally; we won't rely on that here
        vid = len(av)

    flat = _flatten_variant_record(vid, vrec, params or vrec.get("params", {}), scores=scores)
    av[f"v{vid}"] = flat
    rec["auto_variants"] = av

    if is_best:
        rec["auto_best"] = dict(flat)  # copy

    safe_write_json(p, rec)
    return rec

'''def set_tracked_edges_for_crack(save_folder: str, base: str, cid, payload: dict, mask_crop=None):
    p = metric_atomic_path_for(save_folder, base, cid)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    data = dict(payload or {})
    if mask_crop is not None:
        data["mask_crop"] = mask_crop
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return p'''
def set_tracked_edges_for_crack(save_folder: str, base: str, cid, payload: dict, mask_crop=None):
    """
    Safely merge edge-tracking outputs into the existing per-cid snapshot.

    CRITICAL FIX:
      - DOES NOT OVERWRITE the entire cidX.json.
      - PRESERVES: auto_best, auto_variants, source, manual midline, RS3 midline, etc.
      - UPDATES ONLY: geodesic edges, mask_crop, timing, metrics, and worker params.

    This prevents auto variants from being destroyed during manual-edge generation.
    """

    import os
    from helpers.metrics import safe_read_json

    p = metric_atomic_path_for(save_folder, base, cid)
    os.makedirs(os.path.dirname(p), exist_ok=True)

    # --- Load old record FIRST ---------------
    old = safe_read_json(p, {})

    # Normalize to dict
    if not isinstance(old, dict):
        old = {}

    # --- Begin with previous data -----------
    merged = dict(old)

    # --- Merge IN new worker payload fields --
    # (payload contains: status, bbox, mask_bbox, mask_crop, geodesic_edges, timing,
    #  window_half_size, mu, l, p, IoU metrics, boundary metrics, ASSD, HD95 …)
    for k, v in (payload or {}).items():
        merged[k] = v

    # --- Explicit mask_crop overwrite -------
    if mask_crop is not None:
        merged["mask_crop"] = mask_crop

    # --- Ensure crack_id preserved ----------
    if "crack_id" not in merged:
        merged["crack_id"] = cid

    # --- Write MERGED JSON ------------------
    with open(p, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    return p
    
def set_geodesic_edges_for_crack(save_folder, base_name, crack_id, ge_dict):
    """Store geodesic edges (edge1/edge2 lists) under atomic crack snapshot."""
    p = metric_atomic_path_for(save_folder, base_name, crack_id)
    rec = safe_read_json(p, {})
    if not rec:
        rec = {"crack_id": crack_id}
    rec["geodesic_edges"] = ge_dict or {}
    safe_write_json(p, rec)
    return rec