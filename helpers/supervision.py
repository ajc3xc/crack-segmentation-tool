import os, json
import numpy as np
import cv2
from helpers.plot_metrics import plot_edges_and_normals


# --------------------------------------------
# UTIL
# --------------------------------------------
def _ensure(p):
    os.makedirs(p, exist_ok=True)
    return p


def _rebuild_segs(flat):
    """
    Convert a flattened [ [x,y], [x,y], [None,None], ... ]
    into a list of contiguous Nx2 numpy arrays.
    """
    segs = []
    curr = []
    for x, y in flat:
        if x is None or y is None:
            if curr:
                segs.append(np.array(curr))
                curr = []
        else:
            curr.append([x, y])
    if curr:
        segs.append(np.array(curr))
    return segs


# --------------------------------------------
# MAIN ENTRYPOINT
# --------------------------------------------
def export_all_supervision(*, atomic, combined, metrics_dir, original_image):
    """
    Writes:
        metrics/<image>/supervision/supervision.json
        metrics/<image>/supervision/preview/*.png
    """

    sup_root = _ensure(os.path.join(metrics_dir, "supervision"))
    sup_prev = _ensure(os.path.join(sup_root, "preview"))

    H, W = original_image.shape[:2]

    sup_list = []

    # -------------------------------------------------------
    # FLATTEN combined membership so we avoid double-exporting atomics
    # -------------------------------------------------------
    combined_members_flat = {
        m for g in (combined or {}).values()
        for m in (g.get("members") or [])
    }

    # =======================================================
    # ATOMIC (standalone only)
    # =======================================================
    for cid, cr in atomic.items():
        if str(cid) in combined_members_flat:
            continue

        entry = {
            "type": "atomic",
            "id": str(cid),
            "source": cr.get("source"),
            "members": [],
            "midline": cr.get("midline", []),
            "normals": cr.get("normal_edge_points", {}),
            "mask_bbox": cr.get("mask_bbox"),
            "mask_crop": cr.get("mask_crop"),
        }
        sup_list.append(entry)

        _supervision_preview(entry, original_image, os.path.join(sup_prev, f"cid{cid}.png"))

    # =======================================================
    # COMBINED
    # =======================================================
    for ccid, cmb in (combined or {}).items():
        entry = {
            "type": "combined",
            "id": str(ccid),
            "source": "combined",
            "members": cmb.get("members", []),
            "midline": cmb.get("midline", []),
            "normals": cmb.get("normal_edge_points", {}),
            "mask_bbox": cmb.get("mask_bbox"),
            "mask_crop": cmb.get("mask_crop"),
        }
        sup_list.append(entry)

        tag = f"combined{ccid}_{'_'.join(entry['members'])}"
        _supervision_preview(entry, original_image, os.path.join(sup_prev, f"{tag}.png"))

    # =======================================================
    # WRITE ONE UNIFIED JSON
    # =======================================================
    out_json = os.path.join(sup_root, "supervision.json")
    with open(out_json, "w") as f:
        json.dump(sup_list, f)

    print(f"[SUPERVISION] ✓ wrote → {out_json}")


# --------------------------------------------
# PREVIEW GENERATOR
# --------------------------------------------
def _supervision_preview(entry, original_image, out_path):
    """
    Preview:
        - Overlay GT mask (from mask_bbox + mask_crop) blended 50/50 with raw
        - Plot midline + normals (no geodesic edges)
        - Uses unified plot_edges_and_normals
    """
    H, W = original_image.shape[:2]

    # ---- rebuild segments from flattened format ----
    mid_segs = _rebuild_segs(entry.get("midline") or [])
    normals = entry.get("normals") or {}
    n1 = _rebuild_segs(normals.get("edge1") or [])
    n2 = _rebuild_segs(normals.get("edge2") or [])

    # ---- reconstruct ground truth mask (full image size) ----
    full_mask = np.zeros((H, W), np.uint8)
    bbox = entry.get("mask_bbox")

    if bbox and entry.get("mask_crop") is not None:
        x, y, w, h = map(int, bbox)
        crop_arr = np.asarray(entry["mask_crop"], np.uint8)
        full_mask[y:y+h, x:x+w] = crop_arr > 0

    # ---- build overlay (50/50 raw image + GT mask) ----
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mask_f = full_mask.astype(np.float32)

    # Bright highlights for GT mask
    gt_vis = gray * 0.25 + mask_f * 0.75
    gt_vis = np.clip(gt_vis, 0, 1.0)

    gt_rgb = np.stack([gt_vis]*3, axis=-1)

    # ---- call unified plot function ----
    title = f"Ground truth crack normals — {entry['type']} {entry['id']}"

    plot_edges_and_normals(
        base_image=(gt_rgb * 255).astype(np.uint8),
        midline_segs=mid_segs,
        edge1_segs=[],     # GT has no geodesic edges
        edge2_segs=[],
        norm1_segs=n1,
        norm2_segs=n2,
        bbox=bbox,
        out_png=out_path,
        title=title,
    )

    print(f"[SUPERVISION] preview → {out_path}")
