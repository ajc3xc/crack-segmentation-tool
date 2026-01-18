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
    for k in (
        "source",
        "midline",
        "mask_bbox",
        "geodesic_edges",
        "normal_edge_points_full",
        "normal_edge_points",
        "mask_crop",
        "user_points",
        "user_connections",
    ):

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

def _flatten_variant_record(vid: int, vrec: dict, params: dict, scores: dict = None):
    """
    Keep bottom-level clean and single-line-ish:
    {
      "variant_id": 2, "midline": [[...],[...]],
      "params": {"g11":1,"g22":35,"g33":25,"win":45,"mu":0,"ell":5,"p":14},
      "scores": {"nn_mean_bidirectional":..., "hausdorff_max":..., "coverage_min":...},
      "normal_edge_points": {"edge1":[[x,y],...], "edge2":[[x,y],...]}
    }
    """
    import numpy as np

    if not isinstance(vrec, dict):
        print(f"[_flatten_variant_record] ⚠ vrec is not dict (vid={vid}): {type(vrec)}")
        vrec = {}

    midline = vrec.get("midline", [])
    n_mid = len(midline) if isinstance(midline, (list, tuple)) else 0

    out = {
        "variant_id": int(vid),
        "midline": midline,
        "params": params or {},
        "scores": scores or {},
    }
    
    # --------------------------------------------------
    # ✅ TOPOLOGY (CRITICAL)
    # --------------------------------------------------
    if "user_points" in vrec:
        out["user_points"] = vrec.get("user_points", [])
    if "user_connections" in vrec:
        out["user_connections"] = vrec.get("user_connections", [])

    # Optional normals
    for k in ("normal_edge_points_full", "normal_edge_points"):
        if k in vrec and vrec[k]:
            out["normal_edge_points"] = vrec[k]
            break

    # Light debug to understand what's being stored
    try:
        g11 = out["params"].get("g11", None)
        g22 = out["params"].get("g22", None)
        g33 = out["params"].get("g33", None)
        ch  = out["scores"].get("nn_mean_bidirectional", None)
        cov = out["scores"].get("coverage_min", None)
        print(
            f"[_flatten_variant_record] vid={vid} n_mid={n_mid} "
            f"g=({g11},{g22},{g33}) nn_mean_bidirectional={ch} cov={cov}"
        )
    except Exception as e:
        print(f"[_flatten_variant_record] ⚠ debug print failed for vid={vid}: {e}")

    return out
    
def set_auto_variant_for_crack(
    save_folder, base_name, crack_id,
    vrec, params=None, is_best=False, scores=None
):
    """
    Persist an auto-variant under atomic crack snapshot (flat).

    UPDATED:
      - Always keeps auto_variants[vN]
      - If is_best=True:
          * writes auto_best (flattened)
          * publishes its midline into top-level `midline`
          * ensures `mask_bbox` is present (if provided)
          * marks `source = "auto_best"` so Phase 2 can see it
      - Adds debug prints so we can see exactly what happens.
    """
    import numpy as np

    p = metric_atomic_path_for(save_folder, base_name, crack_id)
    rec = safe_read_json(p, {})
    if not isinstance(rec, dict):
        rec = {"crack_id": crack_id}

    # Normalize container
    av = rec.get("auto_variants")
    if not isinstance(av, dict):
        av = {}

    # Determine variant ID
    vid = None
    if isinstance(vrec, dict):
        vid = (vrec.get("params") or {}).get("variant_id", None)
    if vid is None:
        vid = len(av)

    # Flatten
    flat = _flatten_variant_record(
        vid,
        vrec,
        params or vrec.get("params", {}),
        scores=scores,
    )
    av[f"v{vid}"] = flat
    rec["auto_variants"] = av

    # Debug: basic info about this variant
    midline = flat.get("midline", [])
    n_mid = len(midline) if isinstance(midline, (list, tuple)) else 0
    print(
        f"[set_auto_variant_for_crack] cid={crack_id} vid={vid} "
        f"is_best={is_best} n_mid={n_mid}"
    )

    # If this is the best one, publish it
    if is_best:
        rec["auto_best"] = dict(flat)  # copy

        # Publish midline
        if n_mid > 0:
            rec["midline"] = midline
        else:
            print(
                f"[set_auto_variant_for_crack] ⚠ best variant vid={vid} "
                f"has empty midline for cid={crack_id}"
            )

        # Publish bbox if present in vrec
        if isinstance(vrec, dict) and "mask_bbox" in vrec:
            rec["mask_bbox"] = vrec["mask_bbox"]

        # Mark source so Phase 2 picks it up
        old_src = rec.get("source", None)
        rec["source"] = "auto_best"
        print(
            f"[set_auto_variant_for_crack] cid={crack_id} "
            f"source: {old_src!r} → 'auto_best'"
        )

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

def set_tracked_edges_for_crack(save_folder, base, cid, payload, mask_crop=None):
    """
    Merge new edge results into existing per-crack JSON
    instead of overwriting the entire file.
    """
    from helpers.metrics import metric_atomic_path_for, safe_read_json, safe_write_json

    p = metric_atomic_path_for(save_folder, base, cid)

    # Load existing (manual OR auto) data
    old = safe_read_json(p, {}) or {}

    # Start from the old snapshot, not from payload
    out = dict(old)

    # Merge edge-tracking-specific results
    for k, v in (payload or {}).items():
        out[k] = v

    # Optional crop
    if mask_crop is not None:
        out["mask_crop"] = mask_crop

    # DO NOT remove:
    # - source ("manual" / "auto_best")
    # - auto_best
    # - auto_variants
    # - midline (auto or manual)
    # - anything else that belongs to auto mode

    safe_write_json(p, out)
    return p