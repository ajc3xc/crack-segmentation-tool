#!/usr/bin/env python3
import glob
import csv
import time
import shutil
from pathlib import Path

import cv2
import numpy as np
import paddle

from models.crackscopenet import CrackScopeNet

# Hardcoded config (simple export-only script)
INPUT_DIR = Path(r"C:\Users\13144\Documents\Masters_Thesis\datasets\SUT_Compressed\Original_Image")
#CrackScopeNet
#OUTPUT_DIR = Path(r"C:\Users\13144\Documents\Masters_Thesis\datasets\SUT_Compressed\seg_baselines\crackscopenet")
#WEIGHTS_PATH = Path(r"C:\Users\13144\Documents\Masters_Thesis\CURRENT_PROJECT_CODE\mask_baseline_generator\crackscopenet\crackscopenet_9k\best_model\model.pdparams")
#CrackScopeNet9k_Large
OUTPUT_DIR = Path(r"C:\Users\13144\Documents\Masters_Thesis\datasets\SUT_Compressed\seg_baselines\crackscopenet_large")
WEIGHTS_PATH = Path(r"C:\Users\13144\Documents\Masters_Thesis\CURRENT_PROJECT_CODE\mask_baseline_generator\crackscopenet\crackscopenet_9k_large\best_model\model.pdparams")
DEVICE = "gpu"  # "cpu" or "gpu"
MAX_IMAGES = None  # e.g., 100 for quick test
OUTPUT_MODE = "rerun"  # "skip", "rerun", "recreate"

IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")


def choose_device() -> str:
    if DEVICE == "cpu":
        return "cpu"
    if DEVICE == "gpu":
        return "gpu"
    return "gpu" if paddle.is_compiled_with_cuda() else "cpu"


def collect_images(input_dir: Path):
    images = []
    for pattern in IMAGE_GLOBS:
        images.extend(Path(p) for p in glob.glob(str(input_dir / pattern)))
    images = sorted(set(images))
    if MAX_IMAGES is not None:
        images = images[:MAX_IMAGES]
    return images


def infer_mask(model: paddle.nn.Layer, rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    inp = rgb.astype("float32") / 255.0
    inp = np.transpose(inp, (2, 0, 1))[None, ...]
    tensor = paddle.to_tensor(inp)

    with paddle.no_grad():
        logits = model(tensor)[0]
        pred = paddle.argmax(logits, axis=1).numpy().squeeze(0).astype("uint8")

    mask = (pred > 0).astype("uint8") * 255
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError("Input directory not found: %s" % INPUT_DIR)
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError("Weights file not found: %s" % WEIGHTS_PATH)
    if OUTPUT_MODE not in {"skip", "rerun", "recreate"}:
        raise ValueError("Invalid OUTPUT_MODE: %s" % OUTPUT_MODE)

    images = collect_images(INPUT_DIR)
    if not images:
        raise RuntimeError("No supported images found in: %s" % INPUT_DIR)

    paddle_device = choose_device()
    if paddle_device == "gpu" and not paddle.is_compiled_with_cuda():
        print("[CrackScopeNet] GPU requested but CUDA is unavailable. Falling back to CPU.")
        paddle_device = "cpu"
    paddle.set_device(paddle_device)

    if OUTPUT_MODE == "recreate" and OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[CrackScopeNet] Device: %s" % paddle_device)
    print("[CrackScopeNet] Output mode: %s" % OUTPUT_MODE)
    print("[CrackScopeNet] Variant: b32")
    print("[CrackScopeNet] Weights: %s" % WEIGHTS_PATH)
    print("[CrackScopeNet] Images: %d" % len(images))
    print("[CrackScopeNet] Output: %s" % OUTPUT_DIR)

    model = CrackScopeNet(num_classes=2)
    model.set_state_dict(paddle.load(str(WEIGHTS_PATH)))
    model.eval()

    timing_rows = []
    processed = 0
    skipped = 0
    failed = 0
    total_infer_seconds = 0.0
    wall_start = None

    # Warm-up once using the first valid image only (untimed).
    for img_path in images:
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        _ = infer_mask(model, rgb)
        break

    for idx, img_path in enumerate(images, start=1):
        out_path = OUTPUT_DIR / (img_path.stem + ".png")
        if out_path.exists() and OUTPUT_MODE == "skip":
            print("[%d/%d] skip %s (exists)" % (idx, len(images), img_path.name))
            skipped += 1
            timing_rows.append([img_path.name, "skipped_exists", ""])
            continue

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print("[%d/%d] skip %s (read failed)" % (idx, len(images), img_path.name))
            failed += 1
            timing_rows.append([img_path.name, "read_failed", ""])
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if paddle_device == "gpu" and paddle.is_compiled_with_cuda():
            paddle.device.cuda.synchronize()
        if wall_start is None:
            wall_start = time.perf_counter()
        t0 = time.perf_counter()
        mask = infer_mask(model, rgb)
        if paddle_device == "gpu" and paddle.is_compiled_with_cuda():
            paddle.device.cuda.synchronize()
        infer_seconds = time.perf_counter() - t0

        ok = cv2.imwrite(str(out_path), mask)
        if not ok:
            raise RuntimeError("Failed writing mask: %s" % out_path)
        print("[%d/%d] wrote %s" % (idx, len(images), out_path.name))
        processed += 1
        total_infer_seconds += infer_seconds
        timing_rows.append([img_path.name, "processed", f"{infer_seconds:.6f}"])

    wall_seconds = (time.perf_counter() - wall_start) if wall_start is not None else 0.0
    mean_infer_seconds = (total_infer_seconds / processed) if processed else 0.0

    per_image_csv = OUTPUT_DIR / "timing_per_image.csv"
    with per_image_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_name", "status", "inference_seconds"])
        w.writerows(timing_rows)

    summary_csv = OUTPUT_DIR / "timing_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "num_total_images",
            "num_processed",
            "num_skipped_exists",
            "num_read_failed",
            "total_inference_seconds",
            "mean_inference_seconds",
            "total_wall_seconds",
        ])
        w.writerow([
            len(images),
            processed,
            skipped,
            failed,
            f"{total_infer_seconds:.6f}",
            f"{mean_infer_seconds:.6f}",
            f"{wall_seconds:.6f}",
        ])

    print("[CrackScopeNet] Wrote timing CSV: %s" % per_image_csv)
    print("[CrackScopeNet] Wrote summary CSV: %s" % summary_csv)
    print("[CrackScopeNet] Export complete.")


if __name__ == "__main__":
    main()
