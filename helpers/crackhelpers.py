#!/usr/bin/env python3
# crack_helpers.py
import json, os, shutil, tempfile
import numpy as np
import cv2
from skimage.segmentation import mark_boundaries

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox
def error(e):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText(f"Error: {e}")
    msg.setWindowTitle("Error")
    msg.exec_()

# ---------- Mask reconstruction / compaction ----------

def reconstruct_full_mask_from_crack(crack: dict, H: int, W: int) -> np.ndarray:
    mc = crack.get("mask_crop", None)
    bb = crack.get("mask_bbox", None)

    if mc is not None and bb is not None:
        crop = np.array(mc, dtype=np.uint8)
        x, y, w, h = [int(v) for v in bb]
        mask = np.zeros((H, W), dtype=np.uint8)

        # auto-fix legacy swapped bbox order [x,y,h,w]
        if crop.shape == (w, h) and (h, w) != crop.shape:
            print(f"[DEBUG] fixing transposed legacy mask for crack (expected (h={h},w={w}), got {crop.shape})")
            crop = crop.T
            h, w = crop.shape  # update after transpose

        # if mismatch, clip
        h_eff, w_eff = crop.shape
        h_eff, w_eff = min(h_eff, h), min(w_eff, w)

        if h_eff > 0 and w_eff > 0:
            mask[y:y+h_eff, x:x+w_eff] = crop[:h_eff, :w_eff]

        return (mask > 0).astype(np.uint8)

    # legacy full-size
    m = np.array(crack.get("mask", []), dtype=np.uint8)
    if m.size > 0 and m.shape == (H, W):
        return (m > 0).astype(np.uint8)

    return np.zeros((H, W), dtype=np.uint8)

def compact_full_masks_in_ann(ann: dict, H: int, W: int) -> None:
    """
    In-place: convert any legacy full masks to (mask_crop, mask_bbox) then delete 'mask'.
    """
    for _, crack in list(ann.get("atomic_cracks", {}).items()):
        m = np.array(crack.get("mask", []), dtype=np.uint8)
        if m.size > 0 and m.shape == (H, W) and np.any(m):
            ys, xs = np.where(m > 0)
            y0, y1 = int(ys.min()), int(ys.max() + 1)
            x0, x1 = int(xs.min()), int(xs.max() + 1)
            crop = m[y0:y1, x0:x1].astype(np.uint8)
            crack["mask_crop"] = crop.tolist()
            crack["mask_bbox"] = [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]
            if "mask" in crack:
                del crack["mask"]


def build_combined_mask(atomic_cracks: dict, H: int, W: int) -> np.ndarray:
    """
    Combine all cracks' masks (whatever storage) into a single (H,W) uint8 binary mask (0/1).
    """
    out = np.zeros((H, W), dtype=np.uint8)
    if not atomic_cracks:
        return out
    for crack in atomic_cracks.values():
        out |= reconstruct_full_mask_from_crack(crack, H, W)
    out[out > 0] = 1
    return out


def filter_valid_cracks(cracks, H, W):
    """
    Keep cracks that have:
      - a compact mask, OR
      - at least one geodesic/normal edge set, OR
      - a manual midline with >=2 points, OR
      - two user_points + connection (manual endpoint pair).
    """
    valid = {}
    kept_manual = kept_masked = kept_edges = 0

    for cid, crack in (cracks or {}).items():
        if not isinstance(crack, dict):
            continue

        # 1) compact mask
        mask = crack.get("mask_compact")
        if isinstance(mask, list) and len(mask) > 0:
            valid[cid] = crack
            kept_masked += 1
            continue

        # 2) geodesic edges
        ge = crack.get("geodesic_edges")
        if isinstance(ge, dict) and any(len(v) >= 2 for v in ge.values()):
            valid[cid] = crack
            kept_edges += 1
            continue
        if isinstance(ge, (list, tuple)) and any(isinstance(e, (list, tuple)) and len(e) >= 2 for e in ge):
            valid[cid] = crack
            kept_edges += 1
            continue

        # 3) manual midline with >=2 pts
        ml = crack.get("midline") or []
        if isinstance(ml, (list, tuple)) and len(ml) >= 2:
            valid[cid] = crack
            kept_manual += 1
            continue

        # 4) fallback: 2 user_points + 1 connection
        ups = crack.get("user_points") or []
        conns = crack.get("user_connections") or []
        if len(ups) == 2 and len(conns) >= 1:
            valid[cid] = crack
            kept_manual += 1
            continue

    print(f"[DEBUG filter_valid_cracks] total_in={len(cracks)} "
          f"→ kept={len(valid)} (manual={kept_manual}, mask={kept_masked}, edges={kept_edges})")
    return valid

def filter_valid_cracks(cracks, H=None, W=None):
    """
    A crack is valid if it represents intentional geometry.

    Valid if:
      - it has a midline with >= 2 points, OR
      - it has exactly two user_points with at least one connection

    Masks, edges, and compact forms are NOT validation criteria.
    """
    valid = {}
    kept_midline = kept_endpoint = 0
    rejects = []

    for cid, crack in (cracks or {}).items():
        if not isinstance(crack, dict):
            rejects.append((str(cid), "non-dict crack entry"))
            continue

        # 1) midline-based crack (manual or pipeline)
        ml = crack.get("midline")
        if isinstance(ml, (list, tuple)) and len(ml) >= 2:
            valid[cid] = crack
            kept_midline += 1
            continue

        # 2) endpoint-defined crack (future materialization)
        ups = crack.get("user_points") or []
        conns = crack.get("user_connections") or []
        if len(ups) == 2 and len(conns) >= 1:
            valid[cid] = crack
            kept_endpoint += 1
            continue

        src = crack.get("source", crack.get("src", "?"))
        ml_len = len(ml) if isinstance(ml, (list, tuple)) else 0
        rejects.append(
            (
                str(cid),
                f"src={src}, midline_len={ml_len}, user_points={len(ups)}, user_connections={len(conns)}",
            )
        )

    print(
        f"[DEBUG filter_valid_cracks] total_in={len(cracks)} -> kept={len(valid)} "
        f"(midline={kept_midline}, endpoint_only={kept_endpoint})"
    )
    if rejects:
        print(f"[DEBUG filter_valid_cracks] rejected={len(rejects)}")
        for cid, reason in rejects:
            print(f"[DEBUG filter_valid_cracks] reject cid={cid}: {reason}")
    return valid

# ---------- Rendering helpers (numpy in / numpy out) ----------

def overlay_mask_boundaries(image_rgb_uint8: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    """
    Return a new RGB image with blue mask boundaries overlaid.
    image_rgb_uint8: (H,W,3) uint8
    mask01:          (H,W)   uint8 {0,1}
    """
    mask01 = (mask01 > 0).astype(np.uint8)
    out = (mark_boundaries(image_rgb_uint8 / 255.0, mask01, color=(0, 0, 1), background_label=0) * 255).astype(np.uint8)
    return out


def numpy_to_qimage_and_scaled_pixmap(img_uint8: np.ndarray, target_w: int, target_h: int, is_gray: bool):
    """
    Returns (QImage, QPixmap) already scaled for a given widget size.
    """
    from PyQt5.QtGui import QImage, QPixmap
    from PyQt5.QtCore import Qt

    if is_gray:
        qimg = QImage(img_uint8, img_uint8.shape[1], img_uint8.shape[0],
                      img_uint8.strides[0], QImage.Format_Grayscale8)
    else:
        qimg = QImage(img_uint8, img_uint8.shape[1], img_uint8.shape[0],
                      img_uint8.strides[0], QImage.Format_RGB888)

    pm = QPixmap.fromImage(qimg)
    spm = pm.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.FastTransformation)
    return qimg, spm


# ---------- JSON write helper ----------

'''def safe_json_dump(obj: dict, path: str) -> None:
    """
    Atomic write: dump JSON to tmp then move.
    """
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f)
    shutil.move(tmp, path)'''

# base_app.py
import os, cv2, numpy as np
from typing import Any, Dict, List
# uses your existing helpers from this repo

