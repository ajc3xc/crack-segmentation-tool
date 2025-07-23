import numpy as np
import torch
import time
from PIL import Image
import FastGeodis

# --- Create synthetic cost image ---
H, W = 256, 256
cost = np.ones((H, W), dtype=np.float32)
cost[80:180, 120:140] = 10.0  # a vertical high-cost barrier

# --- Define seed mask (distance origin) ---
start_y, start_x = 20, 20
mask = np.ones((1, 1, H, W), dtype=np.float32)
mask[0, 0, start_y, start_x] = 0

# --- Run FastGeodis (geodesic distance) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
img_pt = torch.from_numpy(cost).unsqueeze(0).unsqueeze(0).to(device)
mask_pt = torch.from_numpy(mask).to(device)

t0 = time.time()
dist = FastGeodis.generalised_geodesic2d(img_pt, mask_pt, 1e10, 1.0, iter=2)
dist = dist.squeeze().cpu().numpy()
print(f"FastGeodis time: {time.time()-t0:.4f}s")

# --- Run FastGeodis Fast-Marching version ---
t1 = time.time()
dist_fm = FastGeodis.geodesic2d_fastmarch(img_pt, mask_pt, lamb=1.0)
dist_fm = dist_fm.squeeze().cpu().numpy()
print(f"Fast-Marching time: {time.time()-t1:.4f}s")

# --- Save distance maps for manual inspection ---
import matplotlib.pyplot as plt
plt.imsave("dist_fastgeodis.png", dist, cmap="hot")
plt.imsave("dist_fastmarch.png", dist_fm, cmap="hot")
print("Distance maps saved as dist_fastgeodis.png and dist_fastmarch.png")
