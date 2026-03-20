import torch
import cv2
import numpy as np
import os

print("torch loaded")

from depth_anything_3.api import DepthAnything3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DepthAnything3.from_pretrained("depth-anything/da3-large")
model = model.to(device).eval()

print("Device:", device)

img_path = r"C:\Users\13144\Documents\Masters_Thesis\datasets\SUT_1-Segmentation\Original_Image\1.jpg"

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

H, W = img.shape[:2]
print(f"[INPUT] shape = ({H}, {W})")

process_res = max(H, W) // 2
print(f"[INFO] process_res = {process_res}")

try:
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred = model.inference(
                image=[img],
                process_res=process_res,
                process_res_method="upper_bound_resize"
            )

    depth = pred.depth[0]

    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()

    print(f"[OUTPUT] depth shape = {depth.shape}")

    # ---- save ----
    out_dir = "./depth_outputs"
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "depth.npy"), depth)

    d_min, d_max = depth.min(), depth.max()
    depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)
    depth_img = (depth_norm * 255).astype(np.uint8)

    cv2.imwrite(os.path.join(out_dir, "depth.png"), depth_img)

except RuntimeError as e:
    print("❌ RuntimeError:")
    print(e)

finally:
    if torch.cuda.is_available():
        print(f"[VRAM] allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"[VRAM] reserved : {torch.cuda.memory_reserved()/1e9:.2f} GB")
        torch.cuda.empty_cache()