import warnings
print("importing torch")
import torch
import cv2
import numpy as np
import os
import time
import csv
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
# Hardcoded paths
# -------------------------------------------------
INPUT_DIR = r"C:\Users\13144\Documents\Masters_Thesis\datasets\SUT_Compressed\Original_Image"
OUTPUT_DIR = r"C:\Users\13144\Documents\Masters_Thesis\datasets\SUT_Compressed\old_depth"

TIMING_PER_IMAGE = os.path.join(OUTPUT_DIR, "timing_per_image.csv")
TIMING_SUMMARY = os.path.join(OUTPUT_DIR, "timing_summary.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


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
# Helper: compute valid inference size
# -------------------------------------------------
def compute_target_size(h, w, divisor=14):
    short_side = min(h, w)
    short_side = (short_side // divisor) * divisor
    scale = short_side / min(h, w)

    new_h = int(h * scale)
    new_w = int(w * scale)

    new_h = (new_h // divisor) * divisor
    new_w = (new_w // divisor) * divisor

    return new_h, new_w


# -------------------------------------------------
# Warmup
# -------------------------------------------------
print("Running warmup inference...")

dummy = np.zeros((256,256,3), dtype=np.uint8)

with torch.inference_mode():
    _ = model.inference([dummy])

print("Warmup complete.\n")


# -------------------------------------------------
# Collect images
# -------------------------------------------------
image_paths = sorted([
    p for p in Path(INPUT_DIR).glob("*")
    if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".tif",".tiff"]
])

print(f"Found {len(image_paths)} images.\n")


timings = []


# -------------------------------------------------
# Process images
# -------------------------------------------------
for img_path in image_paths:

    name = img_path.stem
    output_path = os.path.join(OUTPUT_DIR, name + ".png")

    img = cv2.imread(str(img_path))

    if img is None:
        print("Skipping unreadable image:", img_path)
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img_rgb.shape[:2]

    # compute model input size
    new_h, new_w = compute_target_size(h, w)

    print(f"Running model at {new_h} x {new_w}")

    img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    start = time.perf_counter()

    with torch.inference_mode():
        pred = model.inference([img_resized])

    depth = pred.depth[0]

    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()

    # upscale depth back to original resolution
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)

    end = time.perf_counter()

    elapsed = end - start


    # Normalize for visualization
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_vis = (depth_norm * 255).astype(np.uint8)

    cv2.imwrite(output_path, depth_vis)

    timings.append((img_path.name, elapsed))

    print(f"{img_path.name:30s}  {elapsed:.4f} sec")


# -------------------------------------------------
# Save per-image timing
# -------------------------------------------------
with open(TIMING_PER_IMAGE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "seconds"])
    writer.writerows(timings)


# -------------------------------------------------
# Summary stats
# -------------------------------------------------
times = [t[1] for t in timings]

summary = {
    "num_images": len(times),
    "mean_time": float(np.mean(times)),
    "median_time": float(np.median(times)),
    "min_time": float(np.min(times)),
    "max_time": float(np.max(times)),
    "total_time": float(np.sum(times)),
}

with open(TIMING_SUMMARY, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["metric","value"])
    for k,v in summary.items():
        writer.writerow([k,v])


print("\nFinished processing.")
print("Per-image timing:", TIMING_PER_IMAGE)
print("Summary:", TIMING_SUMMARY)