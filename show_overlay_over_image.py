import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.draw import line

# Paths (modify as needed)
img_path = r"overlay.png"                  # Your overlay image
csv_path = r"hand_labeled_widths.csv"      # Your CSV with edges
mask_path = r"mask.png"          # Your segmentation mask (0/255 or 0/1)
save_path = "overlay_with_widths.png"
microns_per_pixel = 3.96875                # Update as needed

# Load image (cv2 reads as BGR)
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Load mask (add this block; comment/remove if you don't want this feature)
mask = cv2.imread(mask_path, 0)
if mask is None:
    print(f"WARNING: Could not load mask image: {mask_path}. Will only show hand-labeled lines.")
else:
    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8)

# Load hand-labeled widths
df = pd.read_csv(csv_path)

plt.figure(figsize=(18, 12))
plt.imshow(img)

for _, row in df.iterrows():
    x1, y1, x2, y2 = row['edge1_x'], row['edge1_y'], row['edge2_x'], row['edge2_y']
    gt_width = row['width_um']
    color = 'yellow'
    
    if mask is not None:
        rr, cc = line(int(round(y1)), int(round(x1)), int(round(y2)), int(round(x2)))
        seg_values = mask[rr, cc]
        crack_idxs = np.where(seg_values > 0)[0]
        if crack_idxs.size > 0:
            width_px = crack_idxs[-1] - crack_idxs[0] + 1
            width_um = width_px * microns_per_pixel
        else:
            width_um = 0.0
        
        # Color-code:
        diff = width_um - gt_width
        abs_diff = abs(diff)
        if abs_diff < 10:         # widths are similar
            color = 'lime'        # green (good)
        elif diff > 0:
            color = 'red'         # seg > gt (overestimation)
        else:
            color = 'deepskyblue' # seg < gt (underestimation)

    # Draw the width line, now with color based on agreement
    plt.plot([x1, x2], [y1, y2], color=color, linewidth=3, alpha=0.8)

plt.title("Overlay + Hand-labeled Crack Widths (yellow lines)\n(cyan text: GT | Segmentation μm)")
plt.axis('off')
plt.tight_layout()
plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=500)
plt.show()

print(f"Saved with overlays: {save_path}")
