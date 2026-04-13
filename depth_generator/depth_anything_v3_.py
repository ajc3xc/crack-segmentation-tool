import warnings
print("importing torch")
import torch
import cv2
import numpy as np
import os
import time
import csv
import json
import re
from pathlib import Path

# Suppress moviepy warnings
warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    module=r"moviepy(\.|$)",
)

print("importing depth anything3")
from depth_anything_3.api import DepthAnything3


# -------------------------------------------------
# Hardcoded paths / settings
# -------------------------------------------------
INPUT_DIR = "/blue/cli2/a.camerer/crack_segmentation/SUT_Compressed/Original_Image/"
ANN_DIR = "/home/a.camerer/Masters/SUT_outputs"   # <base_name>.json lives here
OUTPUT_DIR = "depth"

# The crack tool / supervision currently work in half-res coordinates.
# Atomic mask_bbox values in the JSON are assumed to be in HALF-RES coords.
BBOX_SCALE_TO_FULL = 2.0

# Padding is specified in HALF-RES pixels, then scaled to full-res for inference.
PAD_HALF_PX = 16
MIN_CONTEXT_FULL_PX = 256
BATCH_SIZE = 6  # 4-8 is usually a good VRAM/speed tradeoff

# Optional qualitative full-image depth preview (not used by pipeline).
SAVE_GLOBAL_FULL_PREVIEW = True
SAVE_GLOBAL_FULL_NPY = False

# Depth Anything input-size divisor
DIVISOR = 14

TIMING_PER_IMAGE = os.path.join(OUTPUT_DIR, "timing_per_image.csv")
TIMING_SUMMARY_GLOBAL = os.path.join(OUTPUT_DIR, "timing_summary_global.csv")
TIMING_SUMMARY_CROP = os.path.join(OUTPUT_DIR, "timing_summary_crop.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

GLOBAL_NPY_DIR = os.path.join(OUTPUT_DIR, "global_npy")
GLOBAL_PNG_DIR = os.path.join(OUTPUT_DIR, "global_png")
ATOMIC_NPY_DIR = os.path.join(OUTPUT_DIR, "atomic_npy")
ATOMIC_PNG_DIR = os.path.join(OUTPUT_DIR, "atomic_png")
META_DIR = os.path.join(OUTPUT_DIR, "metadata")

for d in [GLOBAL_NPY_DIR, GLOBAL_PNG_DIR, ATOMIC_NPY_DIR, ATOMIC_PNG_DIR, META_DIR]:
    os.makedirs(d, exist_ok=True)


# -------------------------------------------------
# Device
# -------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("CUDA available:", torch.cuda.is_available())
print("Using device:", device)
#import sys; sys.exit(0)


# -------------------------------------------------
# Load model
# -------------------------------------------------
print("Loading Depth Anything v3...")

model = DepthAnything3.from_pretrained("depth-anything/da3-large")
model = model.to(device)
model.eval()

print("Model running on:", next(model.parameters()).device)


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def compute_target_size(h, w, divisor=DIVISOR):
    """
    Keep aspect ratio, round dimensions down to a valid multiple of divisor.
    """
    h = int(h)
    w = int(w)
    if h <= 0 or w <= 0:
        return divisor, divisor

    short_side = min(h, w)
    short_side = max(divisor, (short_side // divisor) * divisor)
    scale = short_side / float(min(h, w))

    new_h = max(divisor, int(h * scale))
    new_w = max(divisor, int(w * scale))

    new_h = max(divisor, (new_h // divisor) * divisor)
    new_w = max(divisor, (new_w // divisor) * divisor)

    return int(new_h), int(new_w)


def normalize_for_vis(arr, mask=None):
    """
    Percentile-normalize to uint8 for visual debugging.
    """
    x = np.asarray(arr, np.float32)
    if x.ndim != 2:
        raise ValueError("normalize_for_vis expects 2D array")

    if mask is not None:
        m = np.asarray(mask).astype(bool)
        vals = x[m & np.isfinite(x)]
    else:
        vals = x[np.isfinite(x)]

    out = np.zeros_like(x, dtype=np.uint8)
    if vals.size == 0:
        return out

    lo = float(np.percentile(vals, 2))
    hi = float(np.percentile(vals, 98))

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-9:
        return out

    y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return (y * 255.0).astype(np.uint8)


def infer_depth_raw(model_obj, rgb_u8):
    """
    Run Depth Anything on one RGB uint8 image and return raw depth resized
    back to the original input shape.
    """
    if rgb_u8 is None or rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError("infer_depth_raw expects HxWx3 RGB image")

    h, w = rgb_u8.shape[:2]
    new_h, new_w = compute_target_size(h, w)

    img_resized = cv2.resize(rgb_u8, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    with torch.inference_mode():
        pred = model_obj.inference([img_resized])

    depth = pred.depth[0]
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()

    depth = np.asarray(depth, np.float32)
    if depth.shape[:2] != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)

    return depth.astype(np.float32, copy=False)


def infer_depth_dynamic(model_obj, rgb_u8, max_attempts=3, scale_decay=0.8):
    """
    Native bbox resolution inference (no upscaling).
    Minimal fallback if OOM.
    """
    if rgb_u8 is None or rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError("infer_depth_dynamic expects HxWx3 RGB image")

    orig_h, orig_w = rgb_u8.shape[:2]
    scale = 1.0
    attempt = 0

    while attempt < int(max_attempts):
        try:
            # Scaled attempt only when fallback is triggered.
            if abs(scale - 1.0) < 1e-9:
                img_attempt = rgb_u8
            else:
                ah = max(1, int(round(orig_h * scale)))
                aw = max(1, int(round(orig_w * scale)))
                img_attempt = cv2.resize(rgb_u8, (aw, ah), interpolation=cv2.INTER_AREA)

            ah, aw = img_attempt.shape[:2]

            # Key behavior: run at native crop resolution.
            process_res = int(max(ah, aw))

            print(f"[DEPTH] native bbox inference ({ah},{aw}) process_res={process_res}")

            # FP16 inference on CUDA.
            if device.type == "cuda":
                with torch.inference_mode():
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        pred = model_obj.inference(
                            image=[img_attempt],
                            process_res=process_res,
                            process_res_method="upper_bound_resize",
                        )
            else:
                with torch.inference_mode():
                    pred = model_obj.inference(
                        image=[img_attempt],
                        process_res=process_res,
                        process_res_method="upper_bound_resize",
                    )

            depth = pred.depth[0]

            if isinstance(depth, torch.Tensor):
                depth = depth.detach().cpu().numpy()

            depth = np.asarray(depth, np.float32)

            # Resize back if fallback scaling was used.
            if depth.shape[:2] != (ah, aw):
                depth = cv2.resize(depth, (aw, ah), interpolation=cv2.INTER_CUBIC)

            if (ah, aw) != (orig_h, orig_w):
                depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
                print(f"[DEPTH] recovered scale={scale:.3f}")

            return depth.astype(np.float32, copy=False)

        except RuntimeError as e:
            msg = str(e).lower()

            if "out of memory" in msg or "cuda" in msg:
                print("[DEPTH][OOM] retrying smaller scale...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                scale *= float(scale_decay)
                attempt += 1
                continue

            raise

    raise RuntimeError("Depth inference failed after retries")


def normalize_ann_ids(ann_root):
    """
    Normalize atomic/combined IDs so int/string mismatches never matter.
    """
    if not isinstance(ann_root, dict):
        return
    atomic = ann_root.setdefault("atomic_cracks", {})
    combined = ann_root.setdefault("combined_cracks", {})

    ann_root["atomic_cracks"] = {str(k): v for k, v in atomic.items()}

    new_combined = {}
    for k, cmb in combined.items():
        sk = str(k)
        if isinstance(cmb, dict) and "members" in cmb:
            cmb["members"] = [str(m) for m in cmb["members"]]
        new_combined[sk] = cmb
    ann_root["combined_cracks"] = new_combined


def pad_xyxy(x0, y0, x1, y1, pad, H, W):
    return (
        max(0, int(x0) - int(pad)),
        max(0, int(y0) - int(pad)),
        min(int(W), int(x1) + int(pad)),
        min(int(H), int(y1) + int(pad)),
    )


def expand_bbox_min_side(x0, y0, x1, y1, H, W, min_side=128):
    """
    Expand bbox so that both width and height are at least min_side.
    No resizing, only context expansion.
    """
    bw = x1 - x0
    bh = y1 - y0

    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2

    new_w = max(bw, min_side)
    new_h = max(bh, min_side)

    new_x0 = max(0, cx - new_w // 2)
    new_y0 = max(0, cy - new_h // 2)
    new_x1 = min(W, cx + new_w // 2)
    new_y1 = min(H, cy + new_h // 2)

    return new_x0, new_y0, new_x1, new_y1


def load_atomic_bboxes_half(json_path):
    """
    Returns a sorted list of dicts:
      {
        "cid": "atomic_id",
        "bbox_half_xywh": [x, y, w, h],
      }

    Primary source: annotations.atomic_cracks[*].mask_bbox
    Fallback source: annotations.box[*].bounding_box (converted to xywh)
    """
    out = []

    if not os.path.isfile(json_path):
        return out

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            root = json.load(f)
    except Exception as e:
        print(f"[DEPTH] failed to read json {json_path}: {e}")
        return out

    ann = root.get("annotations", {})
    if not isinstance(ann, dict):
        return out

    normalize_ann_ids(ann)

    atomic = ann.get("atomic_cracks", {}) or {}
    if isinstance(atomic, dict) and atomic:
        for cid, cr in atomic.items():
            bb = cr.get("mask_bbox", None)
            if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                continue
            try:
                x, y, w, h = [int(round(float(v))) for v in bb]
            except Exception:
                continue
            if w <= 0 or h <= 0:
                continue
            out.append({
                "cid": str(cid),
                "bbox_half_xywh": [int(x), int(y), int(w), int(h)],
            })

    if out:
        out.sort(key=lambda d: int(d["cid"]) if str(d["cid"]).isdigit() else str(d["cid"]))
        return out

    # Fallback: generic annotation boxes
    box = ann.get("box", {}) or {}
    if isinstance(box, dict):
        for k, v in box.items():
            bb = v.get("bounding_box", None)
            if not (isinstance(bb, (list, tuple)) and len(bb) == 2):
                continue
            try:
                (x0, y0), (x1, y1) = bb
                x0 = int(round(float(x0)))
                y0 = int(round(float(y0)))
                x1 = int(round(float(x1)))
                y1 = int(round(float(y1)))
            except Exception:
                continue
            x = min(x0, x1)
            y = min(y0, y1)
            w = abs(x1 - x0)
            h = abs(y1 - y0)
            if w <= 0 or h <= 0:
                continue
            out.append({
                "cid": str(k),
                "bbox_half_xywh": [int(x), int(y), int(w), int(h)],
            })

    out.sort(key=lambda d: int(d["cid"]) if str(d["cid"]).isdigit() else str(d["cid"]))
    return out


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def natural_sort_key(path_obj):
    """
    Natural filename sort: 1,2,10 instead of 1,10,2.
    Works for numeric and mixed alphanumeric stems.
    """
    stem = path_obj.stem
    parts = re.split(r"(\d+)", stem)
    key = []
    for p in parts:
        if p.isdigit():
            key.append((0, int(p)))
        else:
            key.append((1, p.lower()))
    # Tiebreak with full lowercase name to keep ordering stable.
    key.append((2, path_obj.name.lower()))
    return tuple(key)


# -------------------------------------------------
# Warmup
# -------------------------------------------------
print("Running warmup inference...")

dummy = np.zeros((256, 256, 3), dtype=np.uint8)

with torch.inference_mode():
    _ = model.inference([dummy])

print("Warmup complete.\n")


# -------------------------------------------------
# Collect images
# -------------------------------------------------
image_paths = sorted([
    p for p in Path(INPUT_DIR).glob("*")
    if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
], key=natural_sort_key)

print(f"Found {len(image_paths)} images.\n")

# Temporary preflight check: image/json pairing
image_stems = {p.stem for p in image_paths}
json_paths = sorted(Path(ANN_DIR).glob("*.json"))
json_stems = {p.stem for p in json_paths}

missing_json_stems = sorted(image_stems - json_stems)
orphan_json_stems = sorted(json_stems - image_stems)

print(
    f"[DEPTH][PRECHECK] images={len(image_stems)} "
    f"jsons={len(json_stems)} missing_json={len(missing_json_stems)} "
    f"orphan_json={len(orphan_json_stems)}"
)
for stem in missing_json_stems:
    print(f"[DEPTH][PRECHECK][MISSING_JSON] image={stem}")
for stem in orphan_json_stems:
    print(f"[DEPTH][PRECHECK][ORPHAN_JSON] json={stem}.json")
print("")
#import sys; sys.exit()


timings = []


# -------------------------------------------------
# Process images
# -------------------------------------------------
i=0
for img_path in image_paths:
    name = img_path.stem
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

    if img is None:
        print("Skipping unreadable image:", img_path)
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H_full, W_full = img_rgb.shape[:2]
    H_out, W_out = H_full, W_full

    json_path = os.path.join(ANN_DIR, name + ".json")
    atomic_boxes = load_atomic_bboxes_half(json_path)

    stitched_full = np.zeros((H_out, W_out), dtype=np.float32)
    stitched_weight = np.zeros((H_out, W_out), dtype=np.float32)

    bbox_records = []
    bbox_total_seconds = 0.0
    bbox_failed = 0

    base_pad_full = int(round(float(PAD_HALF_PX) * float(BBOX_SCALE_TO_FULL)))

    print(f"[DEPTH] image={name} full_hw=({H_full},{W_full}) output_hw=({H_out},{W_out}) atomic_boxes={len(atomic_boxes)}")

    # -------------------------------------------------
    # Optional full-image qualitative depth preview
    # -------------------------------------------------
    global_full_seconds = 0.0
    global_png_file = None
    if SAVE_GLOBAL_FULL_PREVIEW or SAVE_GLOBAL_FULL_NPY:
        try:
            t0g = time.perf_counter()
            global_depth_full = infer_depth_dynamic(model, img_rgb)
            global_full_seconds = float(time.perf_counter() - t0g)

            if SAVE_GLOBAL_FULL_NPY:
                global_npy_file = f"{name}.npy"
                np.save(os.path.join(GLOBAL_NPY_DIR, global_npy_file), global_depth_full)

            if SAVE_GLOBAL_FULL_PREVIEW:
                global_vis = normalize_for_vis(global_depth_full)
                global_png_file = f"{name}.png"
                cv2.imwrite(os.path.join(GLOBAL_PNG_DIR, global_png_file), global_vis)

            print(f"[DEPTH] image={name} global_full_s={global_full_seconds:.4f}")
        except Exception as e:
            print(f"[DEPTH] image={name} global preview failed: {type(e).__name__}: {e}")
            global_full_seconds = 0.0

    # -------------------------------------------------
    # Batched per-bbox depth inference
    # -------------------------------------------------
    t_bbox_start = time.perf_counter()
    batch_imgs = []
    batch_meta = []
    bbox_total_seconds_acc = [float(bbox_total_seconds)]

    def flush_batch():
        if not batch_imgs:
            return

        depth_preds = []
        per_item_times = []
        for i, bimg in enumerate(batch_imgs):
            t0i = time.perf_counter()
            try:
                depth_i = infer_depth_dynamic(model, bimg)
            except Exception as e:
                ch, cw = bimg.shape[:2]
                print(f"[DEPTH][FAILSAFE] image={name} batch_idx={i} -> zeros ({type(e).__name__}: {e})")
                depth_i = np.zeros((ch, cw), dtype=np.float32)
            per_item_times.append(float(time.perf_counter() - t0i))
            depth_preds.append(np.asarray(depth_i, np.float32))
        batch_time = float(np.sum(per_item_times)) if per_item_times else 0.0
        bbox_total_seconds_acc[0] += float(batch_time)

        for i, meta in enumerate(batch_meta):
            bbox_idx = meta["bbox_idx"]
            cid = meta["cid"]
            depth = np.asarray(depth_preds[i], np.float32)
            bbox_s = float(per_item_times[i]) if i < len(per_item_times) else 0.0

            ph, pw = meta["crop_shape"]
            if depth.shape[:2] != (ph, pw):
                depth = cv2.resize(depth, (pw, ph), interpolation=cv2.INTER_CUBIC)

            x0f, y0f, x1f, y1f = meta["bbox_full"]
            px0f, py0f = meta["pad_origin"]

            inner_x0 = max(0, x0f - px0f)
            inner_y0 = max(0, y0f - py0f)
            inner_x1 = inner_x0 + (x1f - x0f)
            inner_y1 = inner_y0 + (y1f - y0f)
            depth_unpadded_full = depth[inner_y0:inner_y1, inner_x0:inner_x1]

            if depth_unpadded_full.size == 0:
                bw, bh = meta["bbox_size"]
                depth_unpadded_full = np.zeros((bh, bw), dtype=np.float32)

            depth_full = depth_unpadded_full.astype(np.float32)

            atomic_npy_file = f"{name}_{bbox_idx}.npy"
            atomic_png_file = f"{name}_{bbox_idx}.png"
            np.save(os.path.join(ATOMIC_NPY_DIR, atomic_npy_file), depth_full)
            cv2.imwrite(os.path.join(ATOMIC_PNG_DIR, atomic_png_file), normalize_for_vis(depth_full))

            h, w = depth_full.shape
            y1f = min(y1f, y0f + h)
            x1f = min(x1f, x0f + w)

            stitched_full[y0f:y1f, x0f:x1f] += depth_full[:(y1f - y0f), :(x1f - x0f)]
            stitched_weight[y0f:y1f, x0f:x1f] += 1.0

            bbox_records.append({
                "bbox_index": int(bbox_idx),
                "cid": cid,
                "bbox_half_xywh": [int(meta["bbox_half_xywh"][0]), int(meta["bbox_half_xywh"][1]), int(meta["bbox_half_xywh"][2]), int(meta["bbox_half_xywh"][3])],
                "bbox_full_xyxy": [int(x0f), int(y0f), int(x1f), int(y1f)],
                "padded_full_xyxy": [int(meta["pad_bounds"][0]), int(meta["pad_bounds"][1]), int(meta["pad_bounds"][2]), int(meta["pad_bounds"][3])],
                "multi_cue_npy_file": atomic_npy_file,
                "multi_cue_png_file": atomic_png_file,
                "inference_s": float(bbox_s),
            })

            print(f"[DEPTH] image={name} bbox_idx={bbox_idx} bbox_s={bbox_s:.4f}")

        batch_imgs.clear()
        batch_meta.clear()

    for bbox_idx, rec in enumerate(atomic_boxes):
        cid = str(rec["cid"])
        xh, yh, wh, hh = [int(v) for v in rec["bbox_half_xywh"]]

        x0f = int(np.floor(xh * BBOX_SCALE_TO_FULL))
        y0f = int(np.floor(yh * BBOX_SCALE_TO_FULL))
        x1f = int(np.ceil((xh + wh) * BBOX_SCALE_TO_FULL))
        y1f = int(np.ceil((yh + hh) * BBOX_SCALE_TO_FULL))

        x0f = max(0, min(W_full - 1, x0f))
        y0f = max(0, min(H_full - 1, y0f))
        x1f = max(x0f + 1, min(W_full, x1f))
        y1f = max(y0f + 1, min(H_full, y1f))

        bbox_w_full = max(1, x1f - x0f)
        bbox_h_full = max(1, y1f - y0f)

        if bbox_w_full < 8 or bbox_h_full < 8:
            cx = (x0f + x1f) // 2
            cy = (y0f + y1f) // 2
            half_size = 8
            x0f = max(0, cx - half_size)
            y0f = max(0, cy - half_size)
            x1f = min(W_full, cx + half_size)
            y1f = min(H_full, cy + half_size)
            bbox_w_full = max(1, x1f - x0f)
            bbox_h_full = max(1, y1f - y0f)

        px0f, py0f, px1f, py1f = expand_bbox_min_side(
            x0f, y0f, x1f, y1f,
            H_full, W_full,
            min_side=128,
        )
        crop = img_rgb[py0f:py1f, px0f:px1f]

        if crop.size == 0:
            print(f"[DEPTH][FORCED] image={name} bbox_idx={bbox_idx} empty crop -> using full image")
            crop = img_rgb.copy()
            px0f, py0f, px1f, py1f = 0, 0, W_full, H_full

        batch_imgs.append(crop)
        batch_meta.append({
            "bbox_idx": int(bbox_idx),
            "cid": cid,
            "bbox_full": (int(x0f), int(y0f), int(x1f), int(y1f)),
            "bbox_half_xywh": (int(xh), int(yh), int(wh), int(hh)),
            "pad_origin": (int(px0f), int(py0f)),
            "pad_bounds": (int(px0f), int(py0f), int(px1f), int(py1f)),
            "bbox_size": (int(bbox_w_full), int(bbox_h_full)),
            "crop_shape": crop.shape[:2],
        })

        if len(batch_imgs) >= int(BATCH_SIZE):
            flush_batch()

    flush_batch()
    bbox_total_seconds = float(bbox_total_seconds_acc[0])
    t_bbox_wall = float(time.perf_counter() - t_bbox_start)

    valid = stitched_weight > 0
    if np.any(valid):
        stitched_full[valid] /= stitched_weight[valid]

    stitched_vis_u8 = normalize_for_vis(stitched_full, mask=valid)
    stitched_full_png = os.path.join(OUTPUT_DIR, name + ".png")
    cv2.imwrite(stitched_full_png, stitched_vis_u8)

    # Debug overlay: stitched depth + bbox placement/context visualization.
    overlay = cv2.cvtColor(stitched_vis_u8, cv2.COLOR_GRAY2BGR)

    # Green: original bbox (scaled from half-res annotations).
    for rec in atomic_boxes:
        xh, yh, wh, hh = [int(v) for v in rec["bbox_half_xywh"]]
        x0 = int(np.floor(xh * BBOX_SCALE_TO_FULL))
        y0 = int(np.floor(yh * BBOX_SCALE_TO_FULL))
        x1 = int(np.ceil((xh + wh) * BBOX_SCALE_TO_FULL))
        y1 = int(np.ceil((yh + hh) * BBOX_SCALE_TO_FULL))

        x0 = max(0, min(W_out - 1, x0))
        y0 = max(0, min(H_out - 1, y0))
        x1 = max(x0 + 1, min(W_out - 1, x1))
        y1 = max(y0 + 1, min(H_out - 1, y1))

        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)

    # Blue: padded context region actually used for inference.
    for rec in bbox_records:
        px0, py0, px1, py1 = [int(v) for v in rec["padded_full_xyxy"]]
        px0 = max(0, min(W_out - 1, px0))
        py0 = max(0, min(H_out - 1, py0))
        px1 = max(px0 + 1, min(W_out - 1, px1))
        py1 = max(py0 + 1, min(H_out - 1, py1))
        cv2.rectangle(overlay, (px0, py0), (px1, py1), (255, 0, 0), 1)

    stitched_overlay_png = os.path.join(OUTPUT_DIR, f"{name}_stitched_bbox_overlay.png")
    cv2.imwrite(stitched_overlay_png, overlay)

    # Save per-image metadata
    metadata = {
        "image": img_path.name,
        "image_stem": name,
        "input_image_path": str(img_path),
        "annotation_json_path": json_path,
        "full_hw": [int(H_full), int(W_full)],
        "output_hw": [int(H_out), int(W_out)],
        "bbox_scale_to_full": float(BBOX_SCALE_TO_FULL),
        "pad_half_px": int(PAD_HALF_PX),
        "pad_full_px": int(base_pad_full),
        "min_context_full_px": int(MIN_CONTEXT_FULL_PX),
        "stitched_full_png": os.path.basename(stitched_full_png),
        "stitched_overlay_png": os.path.basename(stitched_overlay_png),
        "global_full_png": global_png_file,
        "n_atomic_boxes": int(len(atomic_boxes)),
        "n_saved_boxes": int(len(bbox_records)),
        "n_failed_boxes": int(bbox_failed),
        "bbox_total_seconds": float(bbox_total_seconds),
        "bbox_wall_seconds": float(t_bbox_wall),
        "global_full_seconds": float(global_full_seconds),
        "total_seconds": float(t_bbox_wall + global_full_seconds),
        "atomic_boxes": bbox_records,
    }
    save_json(os.path.join(META_DIR, f"{name}.json"), metadata)

    timings.append({
        "image": img_path.name,
        "n_atomic_boxes": int(len(atomic_boxes)),
        "n_saved_boxes": int(len(bbox_records)),
        "n_failed_boxes": int(bbox_failed),
        "bbox_seconds": float(bbox_total_seconds),
        "bbox_wall_seconds": float(t_bbox_wall),
        "global_full_seconds": float(global_full_seconds),
        "seconds": float(t_bbox_wall + global_full_seconds),
    })

    print(
        f"[DEPTH] done image={name} "
        f"saved_boxes={len(bbox_records)} failed_boxes={bbox_failed} "
        f"bbox_gpu_seconds={bbox_total_seconds:.4f} bbox_wall_seconds={t_bbox_wall:.4f} "
        f"total_seconds={(t_bbox_wall + global_full_seconds):.4f}\n"
    )
    #if i==5: break
    #i+=1


# -------------------------------------------------
# Save per-image timing
# -------------------------------------------------
with open(TIMING_PER_IMAGE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "image",
        "n_atomic_boxes",
        "n_saved_boxes",
        "n_failed_boxes",
        "bbox_seconds",
        "bbox_wall_seconds",
        "global_full_seconds",
        "seconds",
    ])
    for row in timings:
        writer.writerow([
            row["image"],
            row["n_atomic_boxes"],
            row["n_saved_boxes"],
            row["n_failed_boxes"],
            f"{row['bbox_seconds']:.8f}",
            f"{row['bbox_wall_seconds']:.8f}",
            f"{row['global_full_seconds']:.8f}",
            f"{row['seconds']:.8f}",
        ])


# -------------------------------------------------
# Summary stats
# -------------------------------------------------
global_times = [float(t["global_full_seconds"]) for t in timings if t["global_full_seconds"] > 0]

global_summary = {
    "num_images": int(len(global_times)),
    "mean_global_time": float(np.mean(global_times)) if global_times else 0.0,
    "median_global_time": float(np.median(global_times)) if global_times else 0.0,
    "min_global_time": float(np.min(global_times)) if global_times else 0.0,
    "max_global_time": float(np.max(global_times)) if global_times else 0.0,
    "total_global_time": float(np.sum(global_times)) if global_times else 0.0,
}

with open(TIMING_SUMMARY_GLOBAL, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])
    for k, v in global_summary.items():
        writer.writerow([k, v])

# Crop summary uses GPU inference time (bbox_seconds).
# Swap to bbox_wall_seconds if you want end-to-end crop wall time instead.
crop_times = [float(t["bbox_seconds"]) for t in timings if t["bbox_seconds"] > 0]

crop_summary = {
    "num_images": int(len(crop_times)),
    "mean_crop_time": float(np.mean(crop_times)) if crop_times else 0.0,
    "median_crop_time": float(np.median(crop_times)) if crop_times else 0.0,
    "min_crop_time": float(np.min(crop_times)) if crop_times else 0.0,
    "max_crop_time": float(np.max(crop_times)) if crop_times else 0.0,
    "total_crop_time": float(np.sum(crop_times)) if crop_times else 0.0,
}

with open(TIMING_SUMMARY_CROP, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])
    for k, v in crop_summary.items():
        writer.writerow([k, v])


print("\nFinished processing.")
print("Per-image timing:", TIMING_PER_IMAGE)
print("Global summary:", TIMING_SUMMARY_GLOBAL)
print("Crop summary:", TIMING_SUMMARY_CROP)
print("Output dir:", OUTPUT_DIR)
