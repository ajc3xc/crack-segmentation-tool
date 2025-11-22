import matplotlib.pyplot as plt

import numpy as np
from math import hypot, atan2, pi
from skimage.morphology import skeletonize
import hashlib
import time
import os


def plot_edges_and_normals(
    *,
    base_image,           # full-res image (BGR or gray)
    midline_segs,         # list of midline segments (list of Nx2 arrays)
    edge1_segs,           # list of Nx2 arrays
    edge2_segs,           # list of Nx2 arrays
    norm1_segs,           # list-of-lists of normals (Nx2 arrays)
    norm2_segs,
    bbox=None,            # optional [x0, y0, w, h]
    out_png,
    title="",
):
    """
    Unified visualization for both:
      - atomic 'pretty' plot
      - combined debug plot

    Handles:
      - optional bbox shift
      - gray or BGR base image
      - edges / midline / normals
      - publication-grade legend (blue title, bold)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    H, W = base_image.shape[:2]

    # --- resolve crop ---
    if bbox is not None:
        x0, y0, w, h = map(int, bbox)
        x1, y1 = x0 + w, y0 + h
        pad = 40
        x0p = max(0, x0 - pad)
        y0p = max(0, y0 - pad)
        x1p = min(W, x1 + pad)
        y1p = min(H, y1 + pad)
        crop = base_image[y0p:y1p, x0p:x1p]
        shift_x, shift_y = x0p, y0p
    else:
        crop = base_image.copy()
        shift_x, shift_y = 0, 0

    # convert to RGB for plotting
    if crop.ndim == 2:
        crop_rgb = np.stack([crop]*3, axis=-1)
    else:
        crop_rgb = crop[:, :, ::-1]

    fig, ax = plt.subplots(figsize=(9, 9), dpi=320)
    ax.imshow(crop_rgb)

    # ------------------------------
    # Helper for splitting long lines
    # ------------------------------
    def split(arr, max_step=50):
        arr = np.asarray(arr)
        if len(arr) < 2:
            return []
        d = np.sqrt(np.sum(np.diff(arr, axis=0)**2, axis=1))
        breaks = np.where(d > max_step)[0]
        out = []; s = 0
        for b in breaks:
            if b + 1 - s >= 2:
                out.append(arr[s:b+1])
            s = b + 1
        if len(arr) - s >= 2:
            out.append(arr[s:])
        return out or [arr]

    # ------------------------------
    # Draw midline
    # ------------------------------
    for seg in midline_segs:
        seg = np.asarray(seg)
        for s in split(seg):
            ax.plot(s[:,0]-shift_x, s[:,1]-shift_y, "w-", lw=1.2)

    # ------------------------------
    # Draw edges
    # ------------------------------
    for seg in edge1_segs:
        seg = np.asarray(seg)
        for s in split(seg):
            ax.plot(s[:,0]-shift_x, s[:,1]-shift_y, "r-", lw=1.2)

    for seg in edge2_segs:
        seg = np.asarray(seg)
        for s in split(seg):
            ax.plot(s[:,0]-shift_x, s[:,1]-shift_y, "g-", lw=1.2)

    # ------------------------------
    # Draw normals (sparse)
    # ------------------------------
    for n1, n2 in zip(norm1_segs, norm2_segs):
        n = min(len(n1), len(n2))
        for i in range(0, n, 10):
            p1 = n1[i] - [shift_x, shift_y]
            p2 = n2[i] - [shift_x, shift_y]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color="cyan", lw=1.0)

    # ------------------------------
    # Legend — blue title + bold
    # ------------------------------
    handles = [
        Line2D([], [], color='white', lw=1.4, label='Midline'),
        Line2D([], [], color='red', lw=1.4, label='Edge 1 (Left)'),
        Line2D([], [], color='green', lw=1.4, label='Edge 2 (Right)'),
        Line2D([], [], color='cyan', lw=1.4, label='Normals'),
    ]

    leg = ax.legend(
        handles=handles,
        fontsize=11,
        loc="lower right",
        title="Legend",
        title_fontsize=13,
        framealpha=0.85,
    )
    plt.setp(leg.get_title(), color="blue", fontweight="bold")
    for t in leg.get_texts():
        t.set_fontweight("bold")

    if title:
        ax.set_title(title, fontsize=14)

    ax.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(out_png, dpi=350, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

# -------------------- GT NORMALS + OVERLAY HELPERS --------------------

def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        print("Empty mask for bbox_from_mask call")
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def plot_gt_normals_on_gtbw(gt_mask_u8, midline_xy, e1, e2, out_png, crop_bbox=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    H, W = gt_mask_u8.shape[:2]
    fig, ax = plt.subplots(figsize=(7, 7), dpi=320)
    ax.imshow(gt_mask_u8, cmap='gray', interpolation='nearest')

    handles = []

    if midline_xy is not None and midline_xy.ndim == 2 and len(midline_xy) >= 2:
        ax.plot(midline_xy[:,0], midline_xy[:,1], '-', lw=1.4, color='red', alpha=0.95)
        handles.append(Line2D([], [], color='red', lw=1.8, label='Midline'))

    if e1 is not None and e2 is not None and len(e1) > 1 and len(e2) > 1:
        step = max(1, len(e1)//60)
        for i in range(0, len(e1), step):
            ax.plot([e1[i,0], e2[i,0]], [e1[i,1], e2[i,1]],
                    '-', lw=1.0, color='cyan', alpha=0.9)
        handles.append(Line2D([], [], color='cyan', lw=1.8, label='Normals'))

    if handles:
        leg = ax.legend(
            handles=handles,
            fontsize=10,
            title="Legend",
            title_fontsize=12,
            loc="lower right",
            framealpha=0.85
        )
        plt.setp(leg.get_title(), color="blue", fontweight="bold")
        for t in leg.get_texts():
            t.set_fontweight("bold")

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    if crop_bbox is not None:
        x, y, w, h = map(int, crop_bbox)
        ax.set_xlim(x, x+w)
        ax.set_ylim(y+h, y)

    plt.tight_layout(pad=0)
    fig.savefig(out_png, dpi=320, bbox_inches="tight", pad_inches=0)
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

    
def save_gt_vs_manual_overlay(H, W, gt_full, man_full, out_png, bbox=None, original_image=None):
    """
    Overlay GT vs manual mask, rendered on top of the full original image.
      - white   = overlap (GT ∩ MANUAL)
      - yellow  = manual-only
      - red     = GT-only
    If `original_image` (HxWx{1,3}) is provided, it is used as a dimmed background
    so structure is still visible. Otherwise, a black background is used.
    """
    import numpy as np, cv2

    # --- base background ---
    if original_image is not None:
        base = original_image.copy()
        if base.ndim == 2:
            base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        base = base.astype(np.float32) / 255.0
        base = (base * 0.35)  # slightly darken to make overlays pop
    else:
        base = np.zeros((H, W, 3), dtype=np.float32)

    # --- classes ---
    gt_bin   = (np.asarray(gt_full,  dtype=np.uint8) > 0)
    man_bin  = (np.asarray(man_full, dtype=np.uint8) > 0)
    inter    = np.logical_and(gt_bin, man_bin)
    pred_only= np.logical_and(man_bin, np.logical_not(gt_bin))
    gt_only  = np.logical_and(gt_bin, np.logical_not(man_bin))

    # NOTE: colors expressed in RGB here then converted to BGR for cv2
    overlay = base.copy()
    overlay[gt_only]    = (.2, 0.20, 1)  # red
    overlay[pred_only]  = (.2, 1.00, 1)  # yellow
    overlay[inter]      = (0.97, 0.97, 0.97) # white

    out = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)

    # optional bbox rectangle
    if bbox is not None:
        x, y, w, h = map(int, bbox)
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 128, 0), 5, cv2.LINE_AA)

    cv2.imwrite(out_png, out)


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
        
'''def normals_from_mask_for_midline(midline_xy, mask, max_radius=50):
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
            poly = shapely.affinity.translate(poly, xoff=-0.5, yoff=-0.5)
            polygons.append(poly)
    edges = [poly.boundary for poly in polygons]

    # --- helper: clamp midline point to nearest polygon if outside ---
    def clamp_to_polygon(p):
        """Ensure p lies on/in the polygon (returns closest point on boundary if outside)."""
        for poly in polygons:
            if poly.contains(Point(p[0], p[1])):
                return p  # already valid
        # otherwise clamp to nearest boundary
        dmin = float("inf")
        best = p
        for edge in edges:
            proj = edge.interpolate(edge.project(Point(p[0], p[1])))
            d = proj.distance(Point(p[0], p[1]))
            if d < dmin:
                dmin = d
                best = (proj.x, proj.y)
        return np.asarray(best, float)

    N = len(midline_xy)
    e1x = np.full(N, np.nan); e1y = np.full(N, np.nan)
    e2x = np.full(N, np.nan); e2y = np.full(N, np.nan)
    widths_mask = np.full(N, np.nan)

    for i, (p_raw, nvec) in enumerate(zip(midline_xy, nor)):
        if not np.all(np.isfinite(p_raw)) or not np.all(np.isfinite(nvec)):
            continue

        # ----- KEY FIX HERE -----
        p = clamp_to_polygon(p_raw)   # ensures normals ALWAYS intersect mask
        # -------------------------

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

    return (e1x, e1y, e2x, e2y, widths_mask), polygons'''
    
def normals_from_mask_for_midline(midline_xy, mask, max_radius=50):
    """
    Pixel-accurate version:
    - Polygonizes the mask into exact pixel-boundary polygons using rasterio.
    - Shifts coords by -0.5 so edges align with imshow pixel grid.
    - Intersects midline normals with those polygons so endpoints lie exactly on the mask edge.
    Robustified to avoid NaNs / zero-length edges.
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

    # ---- tangent + normals ----
    try:
        from cracktools.segmentation import compute_smooth_tangent_normals
        _, nor = compute_smooth_tangent_normals(midline_xy[:, 0], midline_xy[:, 1])
    except Exception:
        dx, dy = np.gradient(midline_xy[:, 0]), np.gradient(midline_xy[:, 1])
        nrm = np.hypot(dx, dy) + 1e-12
        tan = np.stack([dx / nrm, dy / nrm], axis=1)
        nor = np.stack([-tan[:, 1], tan[:, 0]], axis=1)

    # ---- polygonize mask -> shapely polygons ----
    mask_bin = (mask > 0).astype(np.uint8)
    polygons = []
    for geom, val in rasterio.features.shapes(mask_bin, mask=mask_bin):
        if val == 1:
            poly = shape(geom)
            # shift by -0.5 so edges align with imshow pixel grid
            poly = shapely.affinity.translate(poly, xoff=-0.5, yoff=-0.5)
            polygons.append(poly)

    if not polygons:
        # empty mask: nothing we can do safely
        N = len(midline_xy)
        return (np.full(N, np.nan),) * 5, []

    edges = [poly.boundary for poly in polygons]

    # ---- helper: clamp midline point to nearest polygon boundary if outside ----
    from math import inf

    def clamp_to_polygon(p):
        """Ensure p lies on/in the mask: returns closest point on any polygon boundary."""
        P = Point(p[0], p[1])
        # already inside some polygon
        for poly in polygons:
            if poly.contains(P) or poly.touches(P):
                return np.asarray(p, float)

        best = None
        best_d = inf
        for edge in edges:
            # nearest point on this boundary
            proj = edge.interpolate(edge.project(P))
            d = proj.distance(P)
            if d < best_d:
                best_d = d
                best = (proj.x, proj.y)

        if best is None:
            # fall back to original point (will be skipped later if invalid)
            return np.asarray(p, float)
        return np.asarray(best, float)

    N = len(midline_xy)
    e1x = np.full(N, np.nan); e1y = np.full(N, np.nan)
    e2x = np.full(N, np.nan); e2y = np.full(N, np.nan)
    widths_mask = np.full(N, np.nan)

    eps = 1e-6

    for i, (p_raw, nvec) in enumerate(zip(midline_xy, nor)):
        # basic sanity checks
        if not np.all(np.isfinite(p_raw)) or not np.all(np.isfinite(nvec)):
            continue

        nlen = float(np.hypot(nvec[0], nvec[1]))
        if nlen < eps:
            # direction is undefined here; skip
            continue
        nvec = nvec / nlen

        # clamp midline point to polygon if it's outside
        p = clamp_to_polygon(p_raw)
        if not np.all(np.isfinite(p)):
            continue

        # build long ray
        A = (p[0] - max_radius * nvec[0], p[1] - max_radius * nvec[1])
        B = (p[0] + max_radius * nvec[0], p[1] + max_radius * nvec[1])
        if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))):
            continue

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
                if len(coords) >= 1:
                    hits.append(tuple(coords[0]))
                if len(coords) >= 2:
                    hits.append(tuple(coords[-1]))

        if len(hits) < 2:
            # cannot define a width here
            continue

        # remove exact duplicates among hits
        hits_arr = np.asarray(hits, float)
        hits_arr = np.unique(hits_arr, axis=0)
        if len(hits_arr) < 2:
            continue
        hits = [tuple(h) for h in hits_arr]

        # project hits along the normal direction
        dists = [np.dot([hx - p[0], hy - p[1]], nvec) for (hx, hy) in hits]

        # classify as "left" (negative side) / "right" (positive side)
        left_pts  = [(hx, hy, d) for (hx, hy), d in zip(hits, dists) if d < -eps]
        right_pts = [(hx, hy, d) for (hx, hy), d in zip(hits, dists) if d > eps]

        if not left_pts or not right_pts:
            # ray didn't properly cross the crack
            continue

        # choose the closest in magnitude on each side
        # choose closest point on the negative side
        lp, _ = min(
            (((hx, hy), abs(d)) for (hx, hy, d) in left_pts),
            key=lambda t: t[1]
        )

        # choose closest point on the positive side
        rp, _ = min(
            (((hx, hy), abs(d)) for (hx, hy, d) in right_pts),
            key=lambda t: t[1]
        )


        # avoid zero-length normals (which later cause "Edge direction cannot be determined")
        w = np.hypot(rp[0] - lp[0], rp[1] - lp[1])
        if not np.isfinite(w) or w < eps:
            continue

        e1x[i], e1y[i] = lp
        e2x[i], e2y[i] = rp
        widths_mask[i] = w

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

def plot_core_timing_bars(metrics_dir, base_name, out_png):
    """
    Simple bar chart of core routine runtimes for one image.

    Reads <base_name>_timings_core.csv and sums:
      - edge_masks_sec
      - edges_tracking_sec
      - build_combined_sec
    """
    import os, numpy as np, pandas as pd
    import matplotlib.pyplot as plt

    csv_path = os.path.join(metrics_dir, f"{base_name}_timings_core.csv")
    if not os.path.exists(csv_path):
        print("[TIMING_PLOT] no timing CSV:", csv_path)
        return

    df = pd.read_csv(csv_path)

    stage_map = [
        ("edge_masks_sec",    "Edge masks"),
        ("edges_tracking_sec","Edge tracking"),
        ("build_combined_sec","Combined crack"),
    ]

    cols, labels, vals = [], [], []
    for col, label in stage_map:
        if col in df.columns:
            cols.append(col)
            labels.append(label)
            vals.append(np.nansum(df[col].astype(float).values))

    if not cols:
        print("[TIMING_PLOT] no known timing columns found")
        return

    xs = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    ax.bar(xs, vals)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9, fontweight="bold")
    ax.set_ylabel("Time (s)")
    ax.set_title("Core runtime per image", fontsize=13, fontweight="bold")

    for i, v in enumerate(vals):
        ax.text(
            xs[i],
            v * 1.03 if v > 0 else 0.02,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)