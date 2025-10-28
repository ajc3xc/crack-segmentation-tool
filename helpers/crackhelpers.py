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


def filter_valid_cracks(atomic_cracks: dict, H: int, W: int) -> dict:
    """
    Keep only cracks that have a mask (crop or legacy) OR geodesic edges OR a real midline.
    Ensures no bloated legacy 'mask' if compact info already exists.
    """
    kept = {}
    for cid, crack in (atomic_cracks or {}).items():
        has_crop = (crack.get("mask_crop") is not None) and (crack.get("mask_bbox") is not None) \
                   and np.any(np.array(crack.get("mask_crop"), dtype=np.uint8))
        full_m = np.array(crack.get("mask", []), dtype=np.uint8)
        has_full = full_m.size > 0 and full_m.shape == (H, W) and np.any(full_m)
        has_mask = has_crop or has_full

        edges = crack.get("geodesic_edges", {}) or {}
        has_edges = bool(edges.get("edge1")) or bool(edges.get("edge2"))
        has_midline = len(crack.get("midline", [])) > 1

        if has_mask or has_edges or has_midline:
            if has_crop and "mask" in crack:
                del crack["mask"]
            kept[cid] = crack
    return kept


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

def safe_json_dump(obj: dict, path: str) -> None:
    """
    Atomic write: dump JSON to tmp then move.
    """
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f)
    shutil.move(tmp, path)

# base_app.py
import os, cv2, numpy as np
from typing import Any, Dict, List
# uses your existing helpers from this repo
