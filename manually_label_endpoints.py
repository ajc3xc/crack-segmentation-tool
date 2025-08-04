import tifffile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from read_roi import read_roi_zip
from pathlib import Path
from scipy.ndimage import map_coordinates

IMAGE_FOLDER = Path(r"C:\Users\ajc3xc\Downloads\krkCMd_images")
OUTPUT_CSV = Path("hand_labeled_widths.csv")
UM_PER_PIXEL = 3.96875
PROFILE_LEN = 501

if OUTPUT_CSV.exists():
    df = pd.read_csv(OUTPUT_CSV)
else:
    df = pd.DataFrame(
        columns=[
            "image_name", "gridline_id", "x1_roi", "y1_roi", "x2_roi", "y2_roi",
            "edge1_x", "edge1_y", "edge2_x", "edge2_y", "width_px", "width_um"
        ]
    )

def remove_row(df, img_name, grid_id):
    return df[~((df["image_name"] == img_name) & (df["gridline_id"] == grid_id))]

def find_next_gridline(df):
    used = set(df["image_name"] + "_" + df["gridline_id"].astype(str))
    return used

image_paths = sorted(IMAGE_FOLDER.rglob("*.tif"))

print("Instructions:")
print(" - Click TWO points (edges) on the colored ROI line (snaps to line).")
print(" - Confirm/Skip/Delete Last/Previous ROI buttons for control.")
print(" - Previous ROI lets you relabel the last ROI if needed.")
print(" - Always starts from the first image/gridline each run.")
print()

for img_path in image_paths:
    if '20mths' in str(img_path) or '0.28' in str(img_path):
        continue  # skip
    img_name = img_path.stem
    roi_zip = img_path.parent / f"ROI{img_name[3:]}.zip"
    if not roi_zip.exists():
        print(f"WARNING: No ROI zip for {img_name}")
        continue

    rois = read_roi_zip(str(roi_zip))
    xs_list, ys_list = [], []
    for roi in rois.values():
        xs = np.linspace(roi['x1'], roi['x2'], PROFILE_LEN)
        ys = np.linspace(roi['y1'], roi['y2'], PROFILE_LEN)
        xs_list.append(xs)
        ys_list.append(ys)

    img = tifffile.imread(str(img_path))
    if img.ndim > 2:
        if img.ndim == 4:
            img = img[0]
        if img.shape[0] <= 4:
            img = np.transpose(img, (1, 2, 0))
    if img.ndim == 3:
        img = img[..., 0]
    img = img.astype(np.float32)
    img -= img.min()
    img /= max(img.max(), 1e-6)

    roi_keys = list(rois.items())
    roi_idx = 0
    while roi_idx < len(roi_keys):
        grid_id, (name, roi) = roi_idx, roi_keys[roi_idx]
        x1, y1 = roi['x1'], roi['y1']
        x2, y2 = roi['x2'], roi['y2']

        xs, ys = xs_list[roi_idx], ys_list[roi_idx]
        profile = map_coordinates(img, [ys, xs], order=1)
        prof_norm = (profile - profile.min()) / (profile.ptp() + 1e-6)
        cmap = plt.get_cmap('jet_r')
        colors = cmap(prof_norm)

        # --- Crop
        mid_x = int((x1 + x2) / 2)
        mid_y = int((y1 + y2) / 2)
        R = 250
        y0, y1r = max(mid_y - R, 0), min(mid_y + R, img.shape[0] - 1)
        x0, x1r = max(mid_x - R, 0), min(mid_x + R, img.shape[1] - 1)
        sub = img[y0:y1r, x0:x1r]

        fig, ax = plt.subplots(figsize=(16, 8))
        plt.subplots_adjust(bottom=0.25)
        ax.imshow(sub, cmap="gray")
        ax.set_title(f"{img_name} - ROI {grid_id}: Click 2 edge points on colored line")

        # Draw colored ROI line (with alpha!)
        for i in range(PROFILE_LEN - 1):
            ax.plot(
                [xs[i] - x0, xs[i + 1] - x0],
                [ys[i] - y0, ys[i + 1] - y0],
                color=colors[i], linewidth=3, solid_capstyle='round', alpha=0.3
            )
        ax.axis('off')

        pts = []

        def onclick(event):
            if event.inaxes != ax:
                return
            # Snap to nearest line pixel
            click = np.array([event.xdata + x0, event.ydata + y0])
            dists = (xs - click[0]) ** 2 + (ys - click[1]) ** 2
            idx = np.argmin(dists)
            px, py = xs[idx] - x0, ys[idx] - y0
            if len(pts) < 2:
                pts.append(idx)
                ax.plot(px, py, 'rx', markersize=10)
                fig.canvas.draw_idle()

        def delete_last(event):
            if pts:
                pts.pop()
                ax.cla()
                ax.imshow(sub, cmap="gray")
                for i in range(PROFILE_LEN - 1):
                    ax.plot(
                        [xs[i] - x0, xs[i + 1] - x0],
                        [ys[i] - y0, ys[i + 1] - y0],
                        color=colors[i], linewidth=3, solid_capstyle='round', alpha=0.3
                    )
                for idx in pts:
                    ax.plot(xs[idx]-x0, ys[idx]-y0, 'rx', markersize=10)
                ax.axis('off')
                fig.canvas.draw_idle()

        confirmed = []
        go_back = []

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        delete_ax = plt.axes([0.44, 0.13, 0.16, 0.07])
        delete_btn = Button(delete_ax, 'Delete Last', color='orange')
        delete_btn.on_clicked(delete_last)

        confirm_ax = plt.axes([0.29, 0.13, 0.15, 0.07])
        skip_ax = plt.axes([0.6, 0.13, 0.20, 0.07])
        prev_ax = plt.axes([0.15, 0.13, 0.15, 0.07])

        confirm_btn = Button(confirm_ax, 'Confirm', color='lightgreen')
        skip_btn = Button(skip_ax, 'Skip', color='tomato')
        prev_btn = Button(prev_ax, 'Previous ROI', color='skyblue')

        def confirm(event):
            if len(pts) == 2:
                confirmed.append(True)
                plt.close(fig)

        def skip(event):
            confirmed.append(False)
            plt.close(fig)

        def prev(event):
            go_back.append(True)
            plt.close(fig)

        confirm_btn.on_clicked(confirm)
        skip_btn.on_clicked(skip)
        prev_btn.on_clicked(prev)

        plt.show()
        fig.canvas.mpl_disconnect(cid)

        if go_back:
            if roi_idx > 0:
                roi_idx -= 1
            continue
        elif not confirmed or len(pts) != 2:
            print(f"Skipped {img_name}, gridline {grid_id}.")
            roi_idx += 1
            continue

        idx1, idx2 = sorted(pts)
        ex1, ey1 = xs[idx1], ys[idx1]
        ex2, ey2 = xs[idx2], ys[idx2]
        w_px = abs(idx2 - idx1)
        w_um = w_px * (np.hypot(x2 - x1, y2 - y1) / (PROFILE_LEN - 1)) * UM_PER_PIXEL

        # Remove old label for this ROI (if any), then add new
        df = remove_row(df, img_name, grid_id)
        record = {
            "image_name": img_name,
            "gridline_id": grid_id,
            "x1_roi": x1, "y1_roi": y1,
            "x2_roi": x2, "y2_roi": y2,
            "edge1_x": ex1, "edge1_y": ey1,
            "edge2_x": ex2, "edge2_y": ey2,
            "width_px": w_px,
            "width_um": w_um,
        }
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved: {img_name}, gridline {grid_id}")
        roi_idx += 1

print("Labeling session complete!")
