#!/usr/bin/env python3
import csv
from pathlib import Path
import re
import time

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

# ===================== WINDOWS PATHS (EDIT THESE) =====================
WIN_INPUT_DIR = r"C:\Users\13144\Documents\Masters_Thesis\datasets\SUT_Compressed\Original_Image"
WIN_OUTPUT_DIR = r"C:\Users\13144\Documents\Masters_Thesis\datasets\SUT_Compressed\seg_baselines\sam3"
PROMPT = "crack"
CONF_THRESH = 0.45
# ======================================================================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ---------------- Windows -> WSL path conversion ----------------
def win_to_wsl_path(win_path: str) -> str:
    """
    Convert Windows path to WSL path with pure Python logic.
    Example:
        C:\\Users\\Name\\dir  ->  /mnt/c/Users/Name/dir
    """
    p = win_path.strip()
    if not p:
        return p

    if p.startswith("/"):
        return p

    p = p.replace("\\", "/")

    m = re.match(r"^([A-Za-z]):(.*)$", p)
    if m:
        drive = m.group(1).lower()
        rest = m.group(2)
        if rest.startswith("/"):
            return f"/mnt/{drive}{rest}"
        return f"/mnt/{drive}/{rest}"

    if p.startswith("//"):
        return f"/mnt/unc/{p.lstrip('/')}"

    return p


INPUT_DIR = win_to_wsl_path(WIN_INPUT_DIR)
OUTPUT_DIR = win_to_wsl_path(WIN_OUTPUT_DIR)

# ---------------- Load model once ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_REPO = "jetjodh/sam3"

print(f"Using device: {device}")
print(f"Input dir : {INPUT_DIR}")
print(f"Output dir: {OUTPUT_DIR}")
print("Loading SAM3...")

model = Sam3Model.from_pretrained(MODEL_REPO).to(device)
processor = Sam3Processor.from_pretrained(MODEL_REPO)

print("Model loaded.\n")


def cuda_sync():
    if device == "cuda":
        torch.cuda.synchronize()


# ---------------- Mask utils ----------------
def union_instance_masks(results):
    masks = results.get("masks", None)
    if masks is None:
        return None

    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()

    masks = np.asarray(masks)

    if masks.ndim == 4:  # (B,N,H,W)
        masks = masks[0]
    if masks.ndim == 3:  # (N,H,W)
        sem = np.any(masks > 0, axis=0)
    elif masks.ndim == 2:  # (H,W)
        sem = masks > 0
    else:
        return None

    return sem.astype(np.uint8)


# ---------------- Inference ----------------
def segment_image(pil_img):
    inputs = processor(images=pil_img, text=PROMPT, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=CONF_THRESH,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    sem = union_instance_masks(results)
    if sem is None:
        h, w = pil_img.size[1], pil_img.size[0]
        sem = np.zeros((h, w), np.uint8)

    return sem


# ---------------- Batch loop ----------------
def main():
    in_root = Path(INPUT_DIR)
    out_root = Path(OUTPUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    paths = sorted([p for p in in_root.iterdir() if p.suffix.lower() in IMG_EXTS])
    if not paths:
        print("No images found.")
        return

    print(f"Found {len(paths)} images\n")

    # ---------------- Warm-up (untimed) ----------------
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        _ = segment_image(img)
        cuda_sync()
        break

    t0_wall = time.perf_counter()

    timing_rows = []
    num_processed = 0
    num_skipped_exists = 0
    num_read_failed = 0
    total_inference_seconds = 0.0

    for i, p in enumerate(paths, 1):
        out_path = out_root / f"{p.stem}.png"

        if out_path.exists():
            num_skipped_exists += 1

        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            num_read_failed += 1
            timing_rows.append(
                {
                    "image_name": p.name,
                    "status": "read_failed",
                    "inference_seconds": "",
                }
            )
            print(f"[{i}] FAIL open {p.name}: {e}")
            continue

        cuda_sync()
        t0 = time.perf_counter()
        mask = segment_image(img)
        cuda_sync()
        infer_s = time.perf_counter() - t0

        total_inference_seconds += infer_s
        num_processed += 1

        mask_255 = (mask > 0).astype(np.uint8) * 255
        Image.fromarray(mask_255).save(out_path)

        timing_rows.append(
            {
                "image_name": p.name,
                "status": "processed",
                "inference_seconds": f"{infer_s:.6f}",
            }
        )

        print(f"[{i}] OK  {p.name} -> {out_path.name} ({infer_s:.6f}s)")

    total_wall_seconds = time.perf_counter() - t0_wall
    mean_inference_seconds = (
        total_inference_seconds / num_processed if num_processed > 0 else 0.0
    )

    timing_per_image_path = out_root / "timing_per_image.csv"
    with timing_per_image_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_name", "status", "inference_seconds"]
        )
        writer.writeheader()
        writer.writerows(timing_rows)

    timing_summary_path = out_root / "timing_summary.csv"
    with timing_summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "num_total_images",
                "num_processed",
                "num_skipped_exists",
                "num_read_failed",
                "total_inference_seconds",
                "mean_inference_seconds",
                "total_wall_seconds",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "num_total_images": len(paths),
                "num_processed": num_processed,
                "num_skipped_exists": num_skipped_exists,
                "num_read_failed": num_read_failed,
                "total_inference_seconds": f"{total_inference_seconds:.6f}",
                "mean_inference_seconds": f"{mean_inference_seconds:.6f}",
                "total_wall_seconds": f"{total_wall_seconds:.6f}",
            }
        )

    print(f"\nDone.\n- {timing_per_image_path}\n- {timing_summary_path}")


if __name__ == "__main__":
    main()
