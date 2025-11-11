# ==== helpers/export_gt_normals.py ============================================
import os, numpy as np, pandas as pd, cv2
from helpers import plot_metrics
from helpers.metrics import normals_from_mask_for_midline

def export_gt_normals_for_image(
    gt_mask_u8,
    atomic_cracks,
    image_hw,
    out_dir,
    step=2,
    max_radius=50,
):
    """
    Exports ground-truth normals sampled along each manual crack midline.
    Creates:
      - gt_normals.csv   : numeric dump of normals per crack (x,y,e1x,e1y,e2x,e2y)
      - gt_normals_plot.png : quick visualization overlay
    """
    import os, csv
    import numpy as np
    import matplotlib.pyplot as plt
    from helpers.metrics import normals_from_mask_for_midline

    H, W = image_hw
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    plt.figure(figsize=(6,6), dpi=160)
    plt.imshow(gt_mask_u8, cmap="gray")

    for cid, crack in atomic_cracks.items():
        ml_data = crack.get("midline", [])
        # --- Robust shapely conversion ---
        if hasattr(ml_data, "coords"):  # Shapely LineString or Polygon
            ml = np.array(ml_data.coords, dtype=float)
        else:
            ml = np.asarray(ml_data, float)

        if ml.ndim != 2 or ml.shape[1] != 2 or len(ml) < 2:
            continue  # skip invalid midlines

        try:
            # compute outward normals (from GT mask)
            (e1x, e1y, e2x, e2y, _), _ = normals_from_mask_for_midline(
                ml, gt_mask_u8 > 0, step=step, max_radius=max_radius
            )

            for j in range(len(e1x)):
                rows.append([
                    cid,
                    float(ml[j, 0]),
                    float(ml[j, 1]),
                    float(e1x[j]),
                    float(e1y[j]),
                    float(e2x[j]),
                    float(e2y[j]),
                ])

            # plotting
            plt.plot(ml[:,0], ml[:,1], "r-", lw=1)
            plt.scatter(e1x, e1y, s=3, c="lime", label=f"cid{cid} e1" if cid==0 else "")
            plt.scatter(e2x, e2y, s=3, c="cyan", label=f"cid{cid} e2" if cid==0 else "")

        except Exception as e:
            print(f"[GT-NORMALS] cid{cid} failed: {e}")
            continue

    # --- save CSV ---
    csv_path = os.path.join(out_dir, "gt_normals.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cid","x","y","e1x","e1y","e2x","e2y"])
        writer.writerows(rows)
    print(f"[GT-NORMALS] ✅ wrote {len(rows)} normals → {csv_path}")

    # --- save visualization ---
    plt.legend(loc="lower right", fontsize=6)
    plt.axis("equal"); plt.tight_layout()
    plot_path = os.path.join(out_dir, "gt_normals_plot.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[GT-NORMALS] ✅ wrote plot → {plot_path}")
