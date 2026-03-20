import warnings
print("importing torch")
import torch
import cv2
import numpy as np
import os
import time
import csv
import json
import inspect
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
INPUT_DIR = r"C:\Users\13144\Documents\Masters_Thesis\datasets\SUT_1-Segmentation\Original_Image"
ANN_DIR = r"C:\Users\13144\Documents\Masters_Thesis\datasets\SUT_Compressed\Outputs"   # <base_name>.json lives here
OUTPUT_DIR = r"C:\Users\13144\Documents\Masters_Thesis\datasets\SUT_Compressed\depth"

# The crack tool / supervision currently work in half-res coordinates.
# Atomic mask_bbox values in the JSON are assumed to be in HALF-RES coords.
BBOX_SCALE_TO_FULL = 2.0

# Padding is specified in HALF-RES pixels, then scaled to full-res for inference.
PAD_HALF_PX = 16
MIN_CONTEXT_FULL_PX = 256
BATCH_SIZE = 6  # 4-8 is usually a good VRAM/speed tradeoff

# Optional qualitative full-image depth preview (not used by pipeline).
SAVE_GLOBAL_FULL_PREVIEW = True
SAVE_GLOBAL_FULL_NPY = True   # set True only if you explicitly want raw full-image depth saved

# Depth Anything input-size divisor
DIVISOR = 14

TIMING_PER_IMAGE = os.path.join(OUTPUT_DIR, "timing_per_image.csv")
TIMING_SUMMARY = os.path.join(OUTPUT_DIR, "timing_summary.csv")

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


def infer_depth_dynamic(model_obj, rgb_u8, max_attempts=5, scale_decay=0.75):
    """
    Native-resolution first; on CUDA OOM, retry with reduced resolution.
    Output is always resized back to the original input shape.
    """
    if rgb_u8 is None or rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError("infer_depth_dynamic expects HxWx3 RGB image")

    orig_h, orig_w = rgb_u8.shape[:2]
    scale = 1.0
    attempt = 0

    while attempt < int(max_attempts):
        try:
            if abs(scale - 1.0) < 1e-9:
                img_attempt = rgb_u8
            else:
                ah = max(1, int(round(float(orig_h) * float(scale))))
                aw = max(1, int(round(float(orig_w) * float(scale))))
                img_attempt = cv2.resize(rgb_u8, (aw, ah), interpolation=cv2.INTER_AREA)

            ah, aw = img_attempt.shape[:2]
            process_res = int(max(ah, aw))
            print(
                f"[DEPTH] preprocess input=({ah},{aw}) "
                f"process_res={process_res} scale={scale:.3f}"
            )

            depth = None

            # Preferred path: explicit process_res override.
            if hasattr(model_obj, "infer_image"):
                try:
                    sig = inspect.signature(model_obj.infer_image)
                    kwargs = {}
                    if "process_res" in sig.parameters:
                        kwargs["process_res"] = int(process_res)
                    if "process_res_method" in sig.parameters:
                        kwargs["process_res_method"] = "upper_bound_resize"
                    with torch.inference_mode():
                        depth = model_obj.infer_image(img_attempt, **kwargs)
                except TypeError:
                    with torch.inference_mode():
                        depth = model_obj.infer_image(img_attempt)
                except Exception:
                    depth = None

            # Fallback path: legacy batch inference with pre-resized image.
            if depth is None:
                with torch.inference_mode():
                    pred = model_obj.inference([img_attempt])
                depth = pred.depth[0]

            if isinstance(depth, torch.Tensor):
                depth = depth.detach().cpu().numpy()
            depth = np.asarray(depth, np.float32)

            if depth.shape[:2] != (ah, aw):
                depth = cv2.resize(depth, (aw, ah), interpolation=cv2.INTER_CUBIC)
            if (ah, aw) != (orig_h, orig_w):
                depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
                print(
                    f"[DEPTH] recovered with scale={scale:.3f} "
                    f"res=({aw},{ah}) for orig=({orig_w},{orig_h})"
                )

            return depth.astype(np.float32, copy=False)

        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg and torch.cuda.is_available():
                torch.cuda.empty_cache()
                scale *= float(scale_decay)
                attempt += 1
                print(f"[DEPTH][OOM] retrying with scale={scale:.3f}")
                continue
            raise

    raise RuntimeError("Depth inference failed after multiple OOM retries")


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


def half_shape_from_full(full_h, full_w):
    """
    Match the common 'half-res working image' convention.
    """
    return int(full_h // 2), int(full_w // 2)


def pad_xyxy(x0, y0, x1, y1, pad, H, W):
    return (
        max(0, int(x0) - int(pad)),
        max(0, int(y0) - int(pad)),
        min(int(W), int(x1) + int(pad)),
        min(int(H), int(y1) + int(pad)),
    )


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
])

print(f"Found {len(image_paths)} images.\n")


timings = []


# -------------------------------------------------
# Process images
# -------------------------------------------------
for img_path in image_paths:
    name = img_path.stem
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

    if img is None:
        print("Skipping unreadable image:", img_path)
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H_full, W_full = img_rgb.shape[:2]
    H_half, W_half = half_shape_from_full(H_full, W_full)

    json_path = os.path.join(ANN_DIR, name + ".json")
    atomic_boxes = load_atomic_bboxes_half(json_path)

    stitched_half = np.zeros((H_half, W_half), dtype=np.float32)
    stitched_weight = np.zeros((H_half, W_half), dtype=np.float32)

    bbox_records = []
    bbox_total_seconds = 0.0
    bbox_failed = 0

    base_pad_full = int(round(float(PAD_HALF_PX) * float(BBOX_SCALE_TO_FULL)))

    print(f"[DEPTH] image={name} full_hw=({H_full},{W_full}) half_hw=({H_half},{W_half}) atomic_boxes={len(atomic_boxes)}")

    # -------------------------------------------------
    # Optional full-image qualitative depth preview
    # -------------------------------------------------
    global_full_seconds = 0.0
    global_npy_file = None
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
                Hf, Wf = global_vis.shape[:2]
                Hh = max(1, int(Hf // 2))
                Wh = max(1, int(Wf // 2))
                global_vis_half = cv2.resize(
                    global_vis,
                    (Wh, Hh),
                    interpolation=cv2.INTER_AREA,
                )
                global_png_file = f"{name}.png"
                cv2.imwrite(os.path.join(GLOBAL_PNG_DIR, global_png_file), global_vis_half)
                print(
                    f"[DEPTH] image={name} global_png_downscaled "
                    f"full=({Hf},{Wf}) -> half=({Hh},{Wh})"
                )

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

            wh, hh = meta["half_size"]
            depth_half = cv2.resize(
                depth_unpadded_full,
                (max(1, wh), max(1, hh)),
                interpolation=cv2.INTER_AREA,
            ).astype(np.float32)

            atomic_npy_file = f"{name}_{bbox_idx}.npy"
            atomic_png_file = f"{name}_{bbox_idx}.png"
            np.save(os.path.join(ATOMIC_NPY_DIR, atomic_npy_file), depth_half)
            cv2.imwrite(os.path.join(ATOMIC_PNG_DIR, atomic_png_file), normalize_for_vis(depth_half))

            xh, yh = meta["half_origin"]
            x0h, y0h = int(xh), int(yh)
            x1h, y1h = x0h + wh, y0h + hh

            x0h_clip = max(0, min(W_half, x0h))
            y0h_clip = max(0, min(H_half, y0h))
            x1h_clip = max(0, min(W_half, x1h))
            y1h_clip = max(0, min(H_half, y1h))

            if x1h_clip > x0h_clip and y1h_clip > y0h_clip:
                dx0 = x0h_clip - x0h
                dy0 = y0h_clip - y0h
                dx1 = dx0 + (x1h_clip - x0h_clip)
                dy1 = dy0 + (y1h_clip - y0h_clip)
                stitched_half[y0h_clip:y1h_clip, x0h_clip:x1h_clip] += depth_half[dy0:dy1, dx0:dx1]
                stitched_weight[y0h_clip:y1h_clip, x0h_clip:x1h_clip] += 1.0

            bbox_records.append({
                "bbox_index": int(bbox_idx),
                "cid": cid,
                "bbox_half_xywh": [int(meta["half_origin"][0]), int(meta["half_origin"][1]), int(wh), int(hh)],
                "bbox_full_xyxy": [int(x0f), int(y0f), int(x1f), int(y1f)],
                "padded_full_xyxy": [int(meta["pad_bounds"][0]), int(meta["pad_bounds"][1]), int(meta["pad_bounds"][2]), int(meta["pad_bounds"][3])],
                "depth_npy_file": atomic_npy_file,
                "depth_png_file": atomic_png_file,
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

        pad_x = max(base_pad_full, int((MIN_CONTEXT_FULL_PX - bbox_w_full) // 2))
        pad_y = max(base_pad_full, int((MIN_CONTEXT_FULL_PX - bbox_h_full) // 2))

        px0f = max(0, x0f - pad_x)
        py0f = max(0, y0f - pad_y)
        px1f = min(W_full, x1f + pad_x)
        py1f = min(H_full, y1f + pad_y)
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
            "pad_origin": (int(px0f), int(py0f)),
            "pad_bounds": (int(px0f), int(py0f), int(px1f), int(py1f)),
            "bbox_size": (int(bbox_w_full), int(bbox_h_full)),
            "half_size": (int(wh), int(hh)),
            "half_origin": (int(xh), int(yh)),
            "crop_shape": crop.shape[:2],
        })

        if len(batch_imgs) >= int(BATCH_SIZE):
            flush_batch()

    flush_batch()
    bbox_total_seconds = float(bbox_total_seconds_acc[0])
    t_bbox_wall = float(time.perf_counter() - t_bbox_start)

    # Average overlaps for stitched half canvas
    valid_stitched = stitched_weight > 0
    if np.any(valid_stitched):
        stitched_half[valid_stitched] /= stitched_weight[valid_stitched]

    # Save stitched half-res raw depth canvas for preview / compatibility
    stitched_half_npy = os.path.join(OUTPUT_DIR, name + ".npy")
    np.save(stitched_half_npy, stitched_half.astype(np.float32, copy=False))

    # Save stitched half-res PNG preview
    stitched_vis_u8 = normalize_for_vis(stitched_half, mask=valid_stitched)
    stitched_half_png = os.path.join(OUTPUT_DIR, name + ".png")
    cv2.imwrite(stitched_half_png, stitched_vis_u8)

    # Save per-image metadata
    metadata = {
        "image": img_path.name,
        "image_stem": name,
        "input_image_path": str(img_path),
        "annotation_json_path": json_path,
        "full_hw": [int(H_full), int(W_full)],
        "half_hw": [int(H_half), int(W_half)],
        "bbox_scale_to_full": float(BBOX_SCALE_TO_FULL),
        "pad_half_px": int(PAD_HALF_PX),
        "pad_full_px": int(base_pad_full),
        "min_context_full_px": int(MIN_CONTEXT_FULL_PX),
        "stitched_half_npy": os.path.basename(stitched_half_npy),
        "stitched_half_png": os.path.basename(stitched_half_png),
        "global_full_npy": global_npy_file,
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
times = [float(t["seconds"]) for t in timings] if timings else []

summary = {
    "num_images": int(len(times)),
    "mean_time": float(np.mean(times)) if times else 0.0,
    "median_time": float(np.median(times)) if times else 0.0,
    "min_time": float(np.min(times)) if times else 0.0,
    "max_time": float(np.max(times)) if times else 0.0,
    "total_time": float(np.sum(times)) if times else 0.0,
    "total_atomic_boxes": int(sum(int(t["n_atomic_boxes"]) for t in timings)),
    "total_saved_boxes": int(sum(int(t["n_saved_boxes"]) for t in timings)),
    "total_failed_boxes": int(sum(int(t["n_failed_boxes"]) for t in timings)),
}

with open(TIMING_SUMMARY, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])
    for k, v in summary.items():
        writer.writerow([k, v])


print("\nFinished processing.")
print("Per-image timing:", TIMING_PER_IMAGE)
print("Summary:", TIMING_SUMMARY)
print("Output dir:", OUTPUT_DIR)
