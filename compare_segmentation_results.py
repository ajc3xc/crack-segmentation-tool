import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.draw import line
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from matplotlib.lines import Line2D

img_path = r"D:\camerer_ml\evaluate_width_methods\jpg_exports\CMd_0.23_2mths\CMd_0.23_2mths_Image1.jpg"
csv_path = r"D:\camerer_ml\evaluate_width_methods\hand_labeled_widths.csv"
mask_paths = [
    (r"D:\camerer_ml\evaluate_width_methods\seg_maps\CMd_0.23_2mths\CMd_0.23_2mths_Image1_mask255.png", "TOS_ET_Mask"),
    (r"D:\camerer_ml\evaluate_width_methods\mask.png", "HrSegNet32_TL_Mask")
]
save_path = "overlay_with_mask_regions.png"
microns_per_pixel = 3.96875

mask_overlay_colors = [(0, 1, 0), (1, 0, 1)]  # green for mask1, magenta for mask2
mask_overlay_alpha = 0.35  # semi-transparent

mask_colors = [
    lambda diff: 'lime' if abs(diff) < 10 else ('red' if diff > 0 else 'deepskyblue'),
    lambda diff: 'gold' if abs(diff) < 10 else ('magenta' if diff > 0 else 'blue')
]

img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Could not load image: {img_path}")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (img.shape[1]*4, img.shape[0]*4), interpolation=cv2.INTER_CUBIC)
df = pd.read_csv(csv_path)

# --- Prepare mask overlays ---
# --- Prepare mask overlays ---
overlay = img.astype(np.float32) / 255.0
for mask_idx, (mask_path, label) in enumerate(mask_paths):
    mask = cv2.imread(mask_path, 0)
    if mask is None:
        print(f"WARNING: Could not load mask image: {mask_path}.")
        continue
    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8)
    # Always force mask to match overlay/image
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    color = np.array(mask_overlay_colors[mask_idx])[None, None, :]
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = (
        (1 - mask_overlay_alpha) * overlay[mask_bool]
        + mask_overlay_alpha * color
    )

# --- Collect per-method errors for scoring ---
errors_by_mask = [[] for _ in mask_paths]
roi_ids = []

# --- Also collect for classic Medial Axis/Distance Transform method ---
classic_label = "Classic_MAT"
classic_errors = []

# --- Plot overlay image ---
fig, ax = plt.subplots(figsize=(overlay.shape[1]/100, overlay.shape[0]/100), dpi=100)
ax.imshow(np.clip(overlay, 0, 1))
ax.axis('off')
ax.set_xlim(0, overlay.shape[1])
ax.set_ylim(overlay.shape[0], 0)

for mask_idx, (mask_path, label) in enumerate(mask_paths):
    mask = cv2.imread(mask_path, 0)
    if mask is None:
        continue
    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8)
    if "CMd_0.23_2mths_Image1_mask255.png" in mask_path:
        mask = cv2.resize(mask, (mask.shape[1]*4, mask.shape[0]*4), interpolation=cv2.INTER_NEAREST)
    color_func = mask_colors[mask_idx]
    
    # Prepare classic width estimator ONCE per mask
    mask_bin = mask > 0
    skeleton = skeletonize(mask_bin)
    dist_map = distance_transform_edt(mask_bin)
    
    for ridx, row in df.iterrows():
        x1, y1, x2, y2 = row['edge1_x'], row['edge1_y'], row['edge2_x'], row['edge2_y']
        gt_width = row['width_um']
        roi_ids.append(row['gridline_id'] if 'gridline_id' in row else ridx)

        rr, cc = line(int(round(y1)), int(round(x1)), int(round(y2)), int(round(x2)))
        rr = np.clip(rr, 0, mask.shape[0]-1)
        cc = np.clip(cc, 0, mask.shape[1]-1)
        seg_values = mask[rr, cc]
        crack_idxs = np.where(seg_values > 0)[0]
        if crack_idxs.size > 0:
            width_px = crack_idxs[-1] - crack_idxs[0] + 1
            width_um = width_px * microns_per_pixel
        else:
            width_um = 0.0
        diff = width_um - gt_width
        errors_by_mask[mask_idx].append(diff)
        color = color_func(diff)
        alpha = 0.8 if mask_idx == 0 else 0.5
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=3, alpha=alpha)

        # -- Classic width at this ROI --
        # For each point along ROI line, if it's on the skeleton, get local width via distance map
        skel_on_line = [i for i in range(len(rr)) if skeleton[rr[i], cc[i]]]
        if skel_on_line:
            # Use the largest width found on this intersection (or mean, or median)
            # Typically, just take the max width at skeleton pixel along the ROI
            skel_idx = skel_on_line[len(skel_on_line)//2]  # central intersection, robust to branches
            radius = dist_map[rr[skel_idx], cc[skel_idx]]
            classic_width_um = 2 * radius * microns_per_pixel
            classic_errors.append(classic_width_um - gt_width)
        else:
            classic_errors.append(-gt_width)  # Penalize for no width found

# --- Classic method (Medial Axis/DST) scoring and plotting ---
print("\n--- Classic Medial Axis/Distance Transform ---")
classic_errors = np.array(classic_errors)
print(f"MAE (um): {np.mean(np.abs(classic_errors)):.2f}")
print(f"Median AE (um): {np.median(np.abs(classic_errors)):.2f}")
print(f"Mean error (um): {np.mean(classic_errors):.2f}")
print(f"Std (um): {np.std(classic_errors):.2f}")

# --- Score each segmentation mask method ---
for idx, (mask_path, label) in enumerate(mask_paths):
    errors = np.array(errors_by_mask[idx])
    print(f"\n--- {label} ---")
    print(f"MAE (um): {np.mean(np.abs(errors)):.2f}")
    print(f"Median AE (um): {np.median(np.abs(errors)):.2f}")
    print(f"Mean error (um): {np.mean(errors):.2f}")
    print(f"Std (um): {np.std(errors):.2f}")

# --- Optional: Export to summary CSV ---
summary = []
for idx, (mask_path, label) in enumerate(mask_paths):
    errors = np.array(errors_by_mask[idx])
    summary.append({
        "Method": label,
        "MAE_um": np.mean(np.abs(errors)),
        "MedianAE_um": np.median(np.abs(errors)),
        "MeanError_um": np.mean(errors),
        "Std_um": np.std(errors)
    })
summary.append({
    "Method": classic_label,
    "MAE_um": np.mean(np.abs(classic_errors)),
    "MedianAE_um": np.median(np.abs(classic_errors)),
    "MeanError_um": np.mean(classic_errors),
    "Std_um": np.std(classic_errors)
})
pd.DataFrame(summary).to_csv("segmentation_method_scores.csv", index=False)

# --- Legend ---
legend_elements = [
    Line2D([0], [0], color='lime', lw=3, label='Mask1: good (Δ<10)'),
    Line2D([0], [0], color='red', lw=3, label='Mask1: over'),
    Line2D([0], [0], color='deepskyblue', lw=3, label='Mask1: under'),
    Line2D([0], [0], color='gold', lw=3, label='Mask2: good (Δ<10)'),
    Line2D([0], [0], color='magenta', lw=3, label='Mask2: over'),
    Line2D([0], [0], color='blue', lw=3, label='Mask2: under'),
]
fig.legend(handles=legend_elements, loc='lower left', fontsize=14)
fig.suptitle("Mask Regions + Comparison of Segmentations vs Hand-labeled & Classic Widths")
plt.tight_layout()
plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
plt.show()
print(f"Saved with overlays: {save_path}")
