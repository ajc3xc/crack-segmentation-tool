import torch
import cv2
import numpy as np
import os
import argparse
from pathlib import Path
import json

print("torch loaded")

from depth_anything_3.api import DepthAnything3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DepthAnything3.from_pretrained("depth-anything/da3-large")
model = model.to(device).eval()

print("Device:", device)

def resolve_image_path(explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser()

    # Try repo folder config first.
    config_path = Path(__file__).resolve().parents[1] / "folder_config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
            cfg_img_folder = cfg.get("img_folder")
            if cfg_img_folder:
                folder = Path(cfg_img_folder)
                if folder.exists():
                    jpgs = sorted(folder.glob("*.jpg"))
                    if jpgs:
                        return jpgs[0]
        except Exception:
            pass

    # Fallback: first jpg under common local image directories.
    repo_root = Path(__file__).resolve().parents[1]
    for candidate_dir in [repo_root / "images", repo_root]:
        jpgs = sorted(candidate_dir.glob("*.jpg"))
        if jpgs:
            return jpgs[0]

    raise FileNotFoundError(
        "No input image found. Pass one with --img /path/to/image.jpg "
        "or set img_folder in folder_config.json."
    )


parser = argparse.ArgumentParser(description="Depth Anything v3 quick test")
parser.add_argument(
    "--img",
    type=str,
    default="/blue/cli2/a.camerer/crack_segmentation/SUT_Compressed/Original_Image/1.jpg",
    help="Path to input image",
)
args = parser.parse_args()

img_path = resolve_image_path(args.img)
print(f"[INPUT] image path = {img_path}")

img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(
        f"OpenCV could not read image: {img_path}. "
        "Check that the path exists and the file is a valid image."
    )
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

H, W = img.shape[:2]
print(f"[INPUT] shape = ({H}, {W})")

process_res = max(H, W)
print(f"[INFO] process_res = {process_res}")

try:
    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model.inference(
                    image=[img],
                    process_res=process_res,
                    process_res_method="upper_bound_resize"
                )
        else:
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
