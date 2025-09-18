import os
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# ---- Hardcoded paths ----
in_path = Path(r"C:\Users\13144\Documents\Masters_Thesis\datasets\SUT_1-Segmentation")
images_name = "Original_Image"
gt_name = "Ground Truth"
out_path = Path(r"C:\Users\13144\Documents\Masters_Thesis\datasets\SUT_Compressed")
images_in = in_path / images_name
masks_in = in_path / gt_name
images_out = out_path / images_name
masks_out = out_path / gt_name

# Make sure output folders exist
images_out.mkdir(parents=True, exist_ok=True)
masks_out.mkdir(parents=True, exist_ok=True)

import cv2
import numpy as np
from pathlib import Path

def align_mask_to_image(mask, image, name="mask"):
    H, W = image.shape[:2]
    print(f"[DEBUG] {name} BEFORE: mask.shape={mask.shape}, image.shape={image.shape}")

    # Force check for rotated case
    if mask.shape[::-1] == (H, W):
        print(f"[DEBUG] {name} appears rotated, fixing by 90° clockwise")
        return cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)

    # already aligned
    if mask.shape[:2] == (H, W):
        print(f"[DEBUG] {name} already aligned — no rotation applied.")
        return mask

    # Try rotations
    rotated_cw  = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    rotated_ccw = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if rotated_cw.shape[:2] == (H, W):
        print(f"[DEBUG] {name} rotated 90° clockwise to match.")
        return rotated_cw
    if rotated_ccw.shape[:2] == (H, W):
        print(f"[DEBUG] {name} rotated 90° counterclockwise to match.")
        return rotated_ccw

    print(f"[DEBUG] {name} could not be aligned automatically. Leaving as-is.")
    return mask

def process_file(in_path: Path, out_path: Path, image_for_alignment=None):
    try:
        img = cv2.imread(str(in_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"⚠️ Skipping {in_path}, not a valid image")
            return

        h, w = img.shape[:2]
        new_size = (w // 4, h // 4)

        if image_for_alignment is not None:  # mask
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = align_mask_to_image(img, image_for_alignment, name=in_path.name)
            resized = cv2.resize(img, new_size, interpolation=cv2.INTER_NEAREST)
        else:  # image
            resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

        # ---- Force consistent orientation (height, width) ----
        if resized.shape[0] < resized.shape[1]:
            resized = cv2.rotate(resized, cv2.ROTATE_90_CLOCKWISE)

        cv2.imwrite(str(out_path), resized)
        print(f"✅ Saved {out_path} (h={resized.shape[0]}, w={resized.shape[1]})")
    except Exception as e:
        print(f"❌ Error processing {in_path}: {e}")

def main():
    img_files  = sorted([f for f in images_in.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    mask_files = sorted([f for f in masks_in.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    # Pair by filename stem (without extension)
    img_map  = {f.stem: f for f in img_files}
    mask_map = {f.stem: f for f in mask_files}

    tasks = []
    with ThreadPoolExecutor() as ex:
        # Process images
        for stem, img_f in img_map.items():
            out_f = images_out / img_f.name
            tasks.append(ex.submit(process_file, img_f, out_f))

            # If mask exists for this image, align it
            if stem in mask_map:
                mask_f = mask_map[stem]
                out_m = masks_out / mask_f.name
                tasks.append(ex.submit(process_file, mask_f, out_m, cv2.imread(str(img_f))))

        # Wait for all to finish
        for t in tasks:
            t.result()

if __name__ == "__main__":
    main()
