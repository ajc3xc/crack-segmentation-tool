import matplotlib.pyplot as plt

import numpy as np
from math import hypot, atan2, pi
from skimage.morphology import skeletonize
import hashlib
import time
import os

# -------------------- GT NORMALS + OVERLAY HELPERS --------------------

def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def plot_gt_normals_on_gtbw(gt_mask_u8: np.ndarray,
                             midline_xy: np.ndarray,
                             e1: np.ndarray, e2: np.ndarray,
                             out_png: str,
                             crop_bbox=None):
    """Draw GT BW image with cyan normals and white manual midline."""
    import matplotlib.pyplot as plt
    H, W = gt_mask_u8.shape[:2]
    fig, ax = plt.subplots(figsize=(7, 7), dpi=320)
    ax.imshow(gt_mask_u8, cmap='gray', interpolation='nearest')

    # midline (white)
    if midline_xy is not None and midline_xy.ndim == 2 and len(midline_xy) >= 2:
        ax.plot(midline_xy[:,0], midline_xy[:,1], '-', lw=1.2, color='red', alpha=0.95, zorder=3)

    # normals (cyan)
    if e1 is not None and e2 is not None and len(e1) > 1 and len(e2) > 1:
        step = max(1, len(e1)//60)
        for i in range(0, len(e1), step):
            ax.plot([e1[i,0], e2[i,0]], [e1[i,1], e2[i,1]],
                    '-', lw=0.9, color='cyan', alpha=0.9, zorder=2)

    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')
    if crop_bbox is not None:
        x, y, w, h = map(int, crop_bbox)
        ax.set_xlim(x, x+w); ax.set_ylim(y+h, y)
    plt.tight_layout(pad=0)
    fig.savefig(out_png, dpi=320, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def plot_gt_normals_for_crack(crack: dict, gt_full_u8: np.ndarray,
                               out_dir: str, fname: str, bbox=None):
    """
    Compute GT normals along THIS crack's manual midline
    and draw them on the GT BW mask.
    """
    import os, numpy as np
    from helpers.metrics import normals_from_mask_for_midline
    os.makedirs(out_dir, exist_ok=True)

    mid = np.asarray(crack.get("midline", []), float)
    if mid.ndim != 2 or mid.shape[1] != 2 or len(mid) < 3:
        return

    mask_bin = (gt_full_u8 > 0).astype(np.uint8)
    (e1x, e1y, e2x, e2y, _w), _ = normals_from_mask_for_midline(mid, mask_bin, max_radius=50)
    e1 = np.column_stack([e1x, e1y]).astype(float)
    e2 = np.column_stack([e2x, e2y]).astype(float)
    plot_gt_normals_on_gtbw(gt_full_u8, mid, e1, e2,
                             os.path.join(out_dir, fname), crop_bbox=bbox)


'''def save_gt_vs_manual_overlay(H, W, gt_full, man_full, out_png, bbox=None, original_image=None):
    """
    Overlay GT vs manual mask (white=overlap, yellow=manual-only, red=GT-only)
    on top of the original image background if provided.
    """
    import numpy as np, cv2, os

    # Background: prefer original image, else black
    if original_image is not None:
        if original_image.ndim == 2:
            overlay = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            overlay = original_image.copy()
    else:
        overlay = np.zeros((H, W, 3), np.uint8)

    # Create boolean masks
    gt_bin = (gt_full > 0)
    man_bin = (man_full > 0)
    intersect = np.logical_and(gt_bin, man_bin)
    pred_only = np.logical_and(man_bin, np.logical_not(gt_bin))
    gt_only   = np.logical_and(gt_bin, np.logical_not(man_bin))

    # Apply colors (OpenCV = BGR)
    overlay[gt_only]   = (0,   0, 255)   # red = GT only
    overlay[pred_only] = (0, 255, 255)   # yellow = manual only
    overlay[intersect] = (255, 255, 255) # white = overlap

    # Optional bbox
    if bbox is not None:
        x, y, w, h = map(int, bbox)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 1)

    # Save
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    cv2.imwrite(out_png, overlay)
    print(f"[DEBUG OVERLAY] wrote → {out_png}")'''
    
def save_gt_vs_manual_overlay(H, W, gt_full, man_full, out_png, bbox=None, original_image=None):
    """
    Overlay GT vs manual mask on a background.
      white  = overlap
      yellow = manual-only
      red    = GT-only
    If original_image (HxWx{1,3}) is passed, blend on it; else render on black.
    If bbox is given, crop to bbox (global view still uses full-image coordinates).
    """
    import numpy as np, cv2

    # --- base background
    if original_image is not None:
        base = original_image.copy()
        if base.ndim == 2:
            base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        base = base.astype(np.float32) / 255.0
    else:
        base = np.zeros((H, W, 3), np.float32)

    # --- color classes
    overlay = base.copy()
    intersect = (gt_full.astype(bool) & man_full.astype(bool))
    pred_only = (man_full.astype(bool) & ~gt_full.astype(bool))
    gt_only   = (gt_full.astype(bool) & ~man_full.astype(bool))

    # RGB (OpenCV is BGR in memory but we’re writing via cv2.imwrite so pass BGR):
    # We’ll set in BGR directly to avoid confusion
    # white (255,255,255), yellow (0,255,255), red (0,0,255)
    mask = np.zeros_like(overlay, np.float32)
    mask[intersect] = (1.0, 1.0, 1.0)
    mask[pred_only] = (0.0, 1.0, 1.0)
    mask[gt_only]   = (0.0, 0.0, 1.0)

    # blend to keep scene structure
    blended = cv2.addWeighted(mask, 0.85, base, 0.15, 0)

    # optional bbox crop
    if bbox is not None:
        x, y, w, h = map(int, bbox)
        x0 = max(0, x); y0 = max(0, y)
        x1 = min(W, x + max(1, w)); y1 = min(H, y + max(1, h))
        blended = blended[y0:y1, x0:x1]

    out_u8 = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(out_png, out_u8)



# --- minimal geometry utils (existing from your metrics) ----------------------
# Expect these to already exist in your codebase:
#   - _reconstruct_full_mask(obj, H, W)
#   - mask_iou(m1, m2)
#   - normals_from_mask_for_midline(midline_xy, mask, max_radius=50)
#   - compute_midline_metrics(auto_xy, man_xy, tau)
#   - compare_widths_for_cracks(ann_like, crack_mask, base_name, metrics_dir, display=False, tag=None)

# For clarity: ann_like for compare_widths_for_cracks is {"atomic_cracks": {cid: crackdict, ...}}

# -------------------- PLOTTING (GT QUICK LOOK) --------------------
def debug_plot_gt_preview(mask_bin, mid_xy, e1_xy=None, e2_xy=None, out_png=None, title="GT preview"):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    H, W = mask_bin.shape
    import numpy as np, cv2
    img = np.zeros((H, W, 3), np.uint8); img[mask_bin > 0] = (255,255,255)

    plt.figure(figsize=(6,6))
    plt.imshow(img, origin="upper")
    if isinstance(mid_xy, np.ndarray) and len(mid_xy) >= 2:
        plt.plot(mid_xy[:,0], mid_xy[:,1], 'k-', lw=3)
        plt.plot(mid_xy[:,0], mid_xy[:,1], 'w-', lw=1.5)
    if (e1_xy is not None) and (e2_xy is not None):
        n = min(len(e1_xy), len(e2_xy))
        stride = max(1, n // max(1, n // 8))  # ~8x sparser just for display
        segs = np.stack([e1_xy[:n:stride], e2_xy[:n:stride]], axis=1)
        lc = LineCollection(segs, colors='C0', linewidths=1.5, alpha=0.85)
        plt.gca().add_collection(lc)
    plt.title(title); plt.axis("equal"); plt.tight_layout()
    if out_png:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=180)
        plt.close()
    else:
        plt.show()
        
def normals_from_mask_for_midline(midline_xy, mask, max_radius=50):
    """
    Pixel-accurate version:
    - Polygonizes the mask into exact pixel-boundary polygons using rasterio.
    - Shifts coords by -0.5 so edges align with imshow pixel grid.
    - Intersects midline normals with those polygons so endpoints lie exactly on the mask edge.
    """
    import numpy as np
    import shapely
    from shapely.geometry import shape, LineString, Point, MultiPoint
    import rasterio.features

    H, W = mask.shape
    midline_xy = np.asarray(midline_xy, float)
    if midline_xy.ndim != 2 or midline_xy.shape[1] != 2 or len(midline_xy) < 2:
        n = len(midline_xy) if midline_xy.ndim > 0 else 0
        return (np.full(n, np.nan),) * 5, []

    # tangent + normals
    try:
        from cracktools.segmentation import compute_smooth_tangent_normals
        _, nor = compute_smooth_tangent_normals(midline_xy[:, 0], midline_xy[:, 1])
    except Exception:
        dx, dy = np.gradient(midline_xy[:, 0]), np.gradient(midline_xy[:, 1])
        nrm = np.hypot(dx, dy) + 1e-12
        tan = np.stack([dx/nrm, dy/nrm], axis=1)
        nor = np.stack([-tan[:, 1], tan[:, 0]], axis=1)

    # polygonize mask -> shapely polygons
    mask_bin = (mask > 0).astype(np.uint8)
    polygons = []
    for geom, val in rasterio.features.shapes(mask_bin, mask=mask_bin):
        if val == 1:
            poly = shape(geom)
            # shift by -0.5 in both x and y
            poly = shapely.affinity.translate(poly, xoff=-0.5, yoff=-0.5)
            polygons.append(poly)
    edges = [poly.boundary for poly in polygons]

    N = len(midline_xy)
    e1x = np.full(N, np.nan); e1y = np.full(N, np.nan)
    e2x = np.full(N, np.nan); e2y = np.full(N, np.nan)
    widths_mask = np.full(N, np.nan)

    for i, (p, nvec) in enumerate(zip(midline_xy, nor)):
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(nvec)):
            continue

        # build long ray
        A = (p[0] - max_radius * nvec[0], p[1] - max_radius * nvec[1])
        B = (p[0] + max_radius * nvec[0], p[1] + max_radius * nvec[1])
        ray = LineString([A, B])

        hits = []
        for edge in edges:
            inter = edge.intersection(ray)
            if inter.is_empty:
                continue
            if isinstance(inter, Point):
                hits.append((inter.x, inter.y))
            elif isinstance(inter, MultiPoint):
                for g in inter.geoms:
                    hits.append((g.x, g.y))
            elif inter.geom_type == "LineString":
                coords = np.asarray(inter.coords, float)
                hits.append(tuple(coords[0])); hits.append(tuple(coords[-1]))

        if len(hits) >= 2:
            dists = [np.dot([hx - p[0], hy - p[1]], nvec) for (hx, hy) in hits]
            left_pts = [(hx, hy) for (hx, hy), d in zip(hits, dists) if d < 0]
            right_pts = [(hx, hy) for (hx, hy), d in zip(hits, dists) if d > 0]
            if left_pts and right_pts:
                lp = max(left_pts, key=lambda q: np.dot([q[0]-p[0], q[1]-p[1]], nvec))
                rp = min(right_pts, key=lambda q: np.dot([q[0]-p[0], q[1]-p[1]], nvec))
                e1x[i], e1y[i] = lp
                e2x[i], e2y[i] = rp
                widths_mask[i] = np.hypot(rp[0]-lp[0], rp[1]-lp[1])

    return (e1x, e1y, e2x, e2y, widths_mask), polygons

def plot_mask_normals(midline, e1x, e1y, e2x, e2y, mask, contours=None,
                    spacing_px=20, show=True, out_path=None, crack_label=""):
    """
    Plot normals + crack contours (polygons) for visualization.
    - contours: list of Shapely Polygons (from rasterio.features.shapes)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    H, W = mask.shape
    plt.figure(figsize=(8, 8))

    # Force 0 = black, 255 = white
    mask_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    mask_rgb[mask > 0] = [255, 255, 255]

    plt.imshow(mask_rgb, alpha=1.0)  # alpha=1 for full opaque b/w

    # plot polygon contours
    if contours:
        for poly in contours:
            if poly.is_empty:
                continue
            # in plot_mask_normals when drawing contours
            if poly.geom_type == "Polygon":
                x, y = poly.exterior.xy
                plt.plot(np.array(x), np.array(y),
                        color="orange", lw=1.5, alpha=0.8)
                for interior in poly.interiors:
                    xi, yi = interior.xy
                    plt.plot(np.array(xi), np.array(yi),
                            color="orange", lw=1.5, alpha=0.5)
            elif poly.geom_type == "MultiPolygon":
                for sub in poly.geoms:
                    x, y = sub.exterior.xy
                    plt.plot(x, y, color="orange", lw=1.0, alpha=0.8)

    # plot midline
    if midline is not None and len(midline) > 1:
        plt.plot(midline[:,0], midline[:,1], 'g-', lw=1.0, label="midline")

    # plot normals
    N = len(midline)
    for i in range(0, N, spacing_px):
        if np.isfinite(e1x[i]) and np.isfinite(e2x[i]):
            plt.plot([e1x[i], e2x[i]], [e1y[i], e2y[i]],
                    color="cyan", lw=0.5, alpha=0.8)
            plt.scatter([e1x[i], e2x[i]], [e1y[i], e2y[i]],
                        c=["red","blue"], s=8, marker="o", alpha=0.7)

    plt.title(f"Mask normals — {crack_label}")
    plt.axis("equal"); plt.legend(); plt.tight_layout()

    if show:
        plt.show()
    elif out_path:
        plt.savefig(out_path, dpi=200); plt.close()
        
def plot_width_differences(midline, w_mask, w_edge, mask, contours=None,
                        spacing_px=20, show=True, out_path=None, crack_label=""):
    """
    Visualize width differences along the midline:
    - Background mask (0=black, 255=white)
    - Midline (green)
    - Points colored by relative error (red=mask wider, blue=edge wider)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    H, W = mask.shape
    # force black/white background
    mask_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    mask_rgb[mask > 0] = [255, 255, 255]

    plt.figure(figsize=(8, 8))
    plt.imshow(mask_rgb, origin="upper")

    # plot contours if available
    if contours:
        for poly in contours:
            if poly.geom_type == "Polygon":
                x, y = poly.exterior.xy
                plt.plot(x, y, color="orange", lw=0.8, alpha=0.7)
                for interior in poly.interiors:
                    xi, yi = interior.xy
                    plt.plot(xi, yi, color="orange", lw=0.5, alpha=0.5)

    if midline is not None and len(midline) > 1:
        plt.plot(midline[:, 0], midline[:, 1], 'g-', lw=1.0, label="midline")

    # compute diffs
    valid = np.isfinite(w_mask) & np.isfinite(w_edge)
    diffs = w_edge - w_mask
    diffs = np.where(valid, diffs, np.nan)

    # color map: red (mask larger), blue (edge larger)
    colors = []
    for d in diffs:
        if np.isnan(d):
            colors.append("gray")
        elif d > 0:
            colors.append("blue")   # edge wider
        else:
            colors.append("red")    # mask wider

    # sample points along midline
    N = len(midline)
    for i in range(0, N, spacing_px):
        if np.isfinite(diffs[i]):
            plt.scatter(midline[i, 0], midline[i, 1],
                        c=colors[i], s=20, marker="o", alpha=0.8)

    plt.title(f"Width comparison — {crack_label}")
    plt.axis("equal"); plt.legend(); plt.tight_layout()

    if show:
        plt.show()
    elif out_path:
        plt.savefig(out_path, dpi=200)
        plt.close()

    return diffs