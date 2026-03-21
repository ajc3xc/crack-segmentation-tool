import csv
import os
import sys

import numpy as np


if len(sys.argv) < 2:
    print("Usage: python split_timing_csv.py timing_per_image.csv")
    raise SystemExit(1)

input_csv = sys.argv[1]
out_dir = os.path.dirname(input_csv) or "."

global_out = os.path.join(out_dir, "timing_summary_global.csv")
crop_out = os.path.join(out_dir, "timing_summary_crop.csv")

timings = []
with open(input_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        timings.append(
            {
                "image": row["image"],
                "bbox_seconds": float(row["bbox_seconds"]),
                "bbox_wall_seconds": float(row["bbox_wall_seconds"]),
                "global_full_seconds": float(row["global_full_seconds"]),
            }
        )

# ---------------------------
# GLOBAL
# ---------------------------
global_times = [t["global_full_seconds"] for t in timings if t["global_full_seconds"] > 0]

global_summary = {
    "num_images": int(len(global_times)),
    "mean_global_time": float(np.mean(global_times)) if global_times else 0.0,
    "median_global_time": float(np.median(global_times)) if global_times else 0.0,
    "min_global_time": float(np.min(global_times)) if global_times else 0.0,
    "max_global_time": float(np.max(global_times)) if global_times else 0.0,
    "total_global_time": float(np.sum(global_times)) if global_times else 0.0,
}

with open(global_out, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])
    for k, v in global_summary.items():
        writer.writerow([k, v])

# ---------------------------
# CROP (GPU inference)
# ---------------------------
crop_times = [t["bbox_seconds"] for t in timings if t["bbox_seconds"] > 0]

crop_summary = {
    "num_images": int(len(crop_times)),
    "mean_crop_time": float(np.mean(crop_times)) if crop_times else 0.0,
    "median_crop_time": float(np.median(crop_times)) if crop_times else 0.0,
    "min_crop_time": float(np.min(crop_times)) if crop_times else 0.0,
    "max_crop_time": float(np.max(crop_times)) if crop_times else 0.0,
    "total_crop_time": float(np.sum(crop_times)) if crop_times else 0.0,
}

with open(crop_out, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])
    for k, v in crop_summary.items():
        writer.writerow([k, v])

print("Wrote:")
print("  ", global_out)
print("  ", crop_out)
