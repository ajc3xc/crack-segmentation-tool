import tifffile
import cv2
import numpy as np
from pathlib import Path
import concurrent.futures

IMAGE_FOLDER = Path(r"C:\Users\ajc3xc\Downloads\krkCMd_images")
OUTPUT_ROOT = Path("jpg_exports")
OUTPUT_ROOT.mkdir(exist_ok=True)

tif_paths = list(IMAGE_FOLDER.rglob("*.tif"))

def convert_save(tif_path):
    rel_path = tif_path.relative_to(IMAGE_FOLDER).with_suffix('.jpg')
    out_path = OUTPUT_ROOT / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        img = tifffile.imread(str(tif_path))
        if img.ndim == 4:
            img = img[0]
        if img.ndim == 3:
            if img.shape[0] <= 4:
                img = img.transpose(1, 2, 0)
            elif img.shape[2] > 4:
                img = img[..., :3]
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        img = img.astype(float)
        img -= img.min()
        img /= (img.max() + 1e-8)
        img = (img * 255).astype('uint8')

        # --- Downscale by 4x ---
        h, w = img.shape[:2]
        new_size = (w // 4, h // 4)
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        # -----------------------

        cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return f"OK: {out_path}"
    except Exception as e:
        return f"FAILED: {tif_path} ({e})"

with concurrent.futures.ThreadPoolExecutor() as executor:
    for res in executor.map(convert_save, tif_paths):
        print(res)

print("All conversions done!")
