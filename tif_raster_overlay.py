import paddle
import numpy as np
from tifffile import imread
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from paddleseg.models.hrsegnet_b32 import HrSegNetB32  # Or B16/B48

DEVICE = 'gpu'
WEIGHTS = r"E:\camerer_ml\finished_models\hrsegnetb32_tl_cnc3k\model.pdparams"
IMG_PATH = r"E:\camerer_ml\datasets\krkCMd\krkCMd_images\CMd_0.23_2mths\CMd_0.23_2mths_Image1.tif"  # Or .png/.jpg
OUT_PATH = r"overlay.png"
MASK_OUT_PATH = r"mask.png"   # <--- ADD THIS

paddle.set_device('gpu' if (DEVICE == 'gpu' and paddle.is_compiled_with_cuda()) else 'cpu')

# 1. Load model
model = HrSegNetB32(num_classes=2)
state_dict = paddle.load(str(WEIGHTS))
model.set_state_dict(state_dict)
model.eval()

# 2. Load image
img = imread(IMG_PATH)
if img.ndim == 4:
    img = img[0]
if img.ndim == 3:
    if img.shape[0] <= 4:  # (C, H, W)
        img = np.transpose(img, (1, 2, 0))
    elif img.shape[2] > 4:
        img = img[:, :, :3]
elif img.ndim == 2:
    img = np.stack([img]*3, axis=-1)
img_rgb = (img / img.max() * 255).astype(np.uint8)

# 3. Model inference (auto-scale to 0-1 float32)
img_input = img.astype(np.float32)
img_input -= img_input.min()
img_input /= (img_input.max() + 1e-6)
img_input = np.transpose(img_input, (2, 0, 1))
img_input = paddle.to_tensor(img_input[np.newaxis, :])

with paddle.no_grad():
    logits = model(img_input)
    if isinstance(logits, (list, tuple)):
        mask_pred = logits[0][0].numpy()
    else:
        mask_pred = logits[0].numpy()
    if mask_pred.shape[0] > 1:
        mask_pred = mask_pred[1]
mask_bin = (mask_pred > 0.0).astype(np.uint8)  # 0 or 1

# 4. Overlay: white cracks (alpha=0.5) on original
overlay = img_rgb.copy()
mask_alpha = np.zeros_like(img_rgb, dtype=np.float32)
mask_alpha[mask_bin == 1] = [255, 255, 255]
alpha = 0.5

# Use cv2.addWeighted for direct overlay
out_img = cv2.addWeighted(overlay, 1 - alpha, mask_alpha.astype(np.uint8), alpha, 0)

# 5. Show and save
plt.figure(figsize=(8, 8))
plt.imshow(out_img)
plt.title("Model Prediction Overlay (white=crack, alpha=0.5)")
plt.axis('off')
plt.show()

cv2.imwrite(str(OUT_PATH), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
cv2.imwrite(str(MASK_OUT_PATH), (mask_bin * 255).astype(np.uint8))  # <--- THIS LINE
print(f"Saved overlay to: {OUT_PATH}")
print(f"Saved mask to: {MASK_OUT_PATH}")
