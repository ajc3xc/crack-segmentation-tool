import matplotlib
matplotlib.use("Agg")   # fastest non-GUI backend for PNG/PDF/SVG export
matplotlib.rcParams.update({
    "figure.max_open_warning": 0,
    "text.kerning_factor": 0,
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "path.simplify": True,
    "path.simplify_threshold": 1.0,
                "savefig.dpi": 180,
})

import matplotlib.pyplot as plt
plt.ioff()   # no interactive figure updates

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
    derived_midline_segs=None,  # optional list of derived midline segments
    edge1_segs,           # list of Nx2 arrays
    edge2_segs,           # list of Nx2 arrays
    norm1_segs,           # list-of-lists of normals (Nx2 arrays)
    norm2_segs,
    bbox=None,            # optional [x0, y0, w, h]
    out_png,
    sparsity:int = 10,
    gt_plot:bool = False,
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
    _plot_fix_warned = {"xy_swap": False}
    _diag_printed = {"done": False}

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
        crop_rgb = np.stack([crop] * 3, axis=-1)
    else:
        crop_rgb = crop[:, :, ::-1]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(crop_rgb)

    # ------------------------------
    # Draw bbox (visual reference)
    # ------------------------------
    if bbox is not None:
        x0, y0, w, h = map(int, bbox)
        x1, y1 = x0 + w, y0 + h

        # bbox coordinates in plot space
        bx = [x0 - shift_x, x1 - shift_x, x1 - shift_x, x0 - shift_x, x0 - shift_x]
        by = [y0 - shift_y, y0 - shift_y, y1 - shift_y, y1 - shift_y, y0 - shift_y]

        ax.plot(
            bx, by,
            color="dodgerblue",
            lw=2.0,
            alpha=0.9,
        )

    # ------------------------------
    # Helper for splitting long lines
    # ------------------------------
    def split(arr, max_step=50):
        arr = np.asarray(arr)
        if len(arr) < 2:
            return []
        d = np.sqrt(np.sum(np.diff(arr, axis=0) ** 2, axis=1))
        breaks = np.where(d > max_step)[0]
        out = []
        s = 0
        for b in breaks:
            if b + 1 - s >= 2:
                out.append(arr[s:b + 1])
            s = b + 1
        if len(arr) - s >= 2:
            out.append(arr[s:])
        return out or [arr]

    # ------------------------------
    # Cosmetic x/y swap guard (plot only)
    # ------------------------------
    def maybe_fix_xy(seg, H, W):
        seg = np.asarray(seg, float)
        if seg.ndim != 2 or seg.shape[1] != 2 or len(seg) == 0:
            return seg

        x_ok = 0 <= seg[:, 0].min() and seg[:, 0].max() <= W
        y_ok = 0 <= seg[:, 1].min() and seg[:, 1].max() <= H
        swapped_x_ok = 0 <= seg[:, 1].min() and seg[:, 1].max() <= W
        swapped_y_ok = 0 <= seg[:, 0].min() and seg[:, 0].max() <= H

        if not (x_ok and y_ok) and (swapped_x_ok and swapped_y_ok):
            if not _plot_fix_warned["xy_swap"]:
                print("[PLOT FIX] Detected x/y swap - correcting for visualization only.")
                _plot_fix_warned["xy_swap"] = True
            return seg[:, ::-1]
        return seg

    def _seg_diag(name, seg):
        seg = np.asarray(seg, float)
        if seg.ndim != 2 or seg.shape[1] != 2 or len(seg) == 0:
            print(f"[PLOT DIAG] {name}: empty/invalid")
            return None
        p0 = seg[0]
        p1 = seg[-1]
        dx, dy = (p1 - p0)
        ang = np.degrees(np.arctan2(dy, dx))
        '''print(
            f"[PLOT DIAG] {name}: n={len(seg)} "
            f"start=({p0[0]:.2f},{p0[1]:.2f}) end=({p1[0]:.2f},{p1[1]:.2f}) "
            f"x=[{seg[:,0].min():.2f},{seg[:,0].max():.2f}] "
            f"y=[{seg[:,1].min():.2f},{seg[:,1].max():.2f}] "
            f"d=({dx:.2f},{dy:.2f}) angle={ang:.1f}deg"
        )'''
        return seg

    def _endpoint_dist_diag(name_a, seg_a, name_b, seg_b):
        if seg_a is None or seg_b is None or len(seg_a) < 1 or len(seg_b) < 1:
            return
        d_ss = np.linalg.norm(seg_a[0] - seg_b[0])
        d_se = np.linalg.norm(seg_a[0] - seg_b[-1])
        d_es = np.linalg.norm(seg_a[-1] - seg_b[0])
        d_ee = np.linalg.norm(seg_a[-1] - seg_b[-1])
        '''print(
            f"[PLOT DIAG] {name_a}<->{name_b} endpoint dists: "
            f"ss={d_ss:.2f} se={d_se:.2f} es={d_es:.2f} ee={d_ee:.2f}"
        )'''

    # Pre-normalize/correct lists once so plotting and diagnostics use the same data.
    midline_plot = [maybe_fix_xy(seg, H, W) for seg in (midline_segs or [])]
    derived_plot = [maybe_fix_xy(seg, H, W) for seg in (derived_midline_segs or [])]
    edge1_plot = [maybe_fix_xy(seg, H, W) for seg in (edge1_segs or [])]
    edge2_plot = [maybe_fix_xy(seg, H, W) for seg in (edge2_segs or [])]
    norm1_plot = [maybe_fix_xy(seg, H, W) for seg in (norm1_segs or [])]
    norm2_plot = [maybe_fix_xy(seg, H, W) for seg in (norm2_segs or [])]

    # One-shot geometric diagnostics (helps identify swapped/reversed edge tracks).
    if not _diag_printed["done"]:
        dmid0 = _seg_diag("derived_midline[0]", derived_plot[0]) if derived_plot else None
        mid0 = _seg_diag("midline[0]", midline_plot[0]) if midline_plot else None
        e10 = _seg_diag("edge1[0]", edge1_plot[0]) if edge1_plot else None
        e20 = _seg_diag("edge2[0]", edge2_plot[0]) if edge2_plot else None
        _endpoint_dist_diag("edge1[0]", e10, "derived_midline[0]", dmid0)
        _endpoint_dist_diag("edge2[0]", e20, "derived_midline[0]", dmid0)
        _endpoint_dist_diag("edge1[0]", e10, "edge2[0]", e20)
        _endpoint_dist_diag("midline[0]", mid0, "derived_midline[0]", dmid0)
        _diag_printed["done"] = True

    # ------------------------------
    # Draw midline
    # ------------------------------
    for seg in midline_plot:
        if seg.ndim != 2 or len(seg) < 2:
            continue

        if gt_plot:
            # GT mode -> solid white only
            ax.plot(
                seg[:, 0] - shift_x,
                seg[:, 1] - shift_y,
                color="white",
                lw=2.2,
                linestyle="-",
            )
            continue

        # Debug mode -> manual dashed yellow
        for s in split(seg):
            ax.plot(
                s[:, 0] - shift_x,
                s[:, 1] - shift_y,
                color="darkorange",
                lw=2.5,
                linestyle="--",
                alpha=0.75,
                zorder=2,
            )

    # ------------------------------
    # Draw derived midline
    # ------------------------------
    if not gt_plot:
        for seg in derived_plot:
            if seg.ndim != 2 or len(seg) < 2:
                continue
            for s in split(seg):
                ax.plot(
                    s[:, 0] - shift_x,
                    s[:, 1] - shift_y,
                    color="white",
                    lw=2.2,
                    alpha=0.95,
                    zorder=5,
                )

    # ------------------------------
    # Draw edges
    # ------------------------------
    if not gt_plot:
        for seg in edge1_plot:
            for s in split(seg):
                ax.plot(s[:, 0] - shift_x, s[:, 1] - shift_y, "r-", lw=1.2)

        for seg in edge2_plot:
            for s in split(seg):
                ax.plot(s[:, 0] - shift_x, s[:, 1] - shift_y, "g-", lw=1.2)

    # ------------------------------
    # Draw normals (sparse)
    # ------------------------------
    for n1, n2 in zip(norm1_plot, norm2_plot):
        n = min(len(n1), len(n2))
        for i in range(0, n, sparsity):
            p1 = n1[i] - [shift_x, shift_y]
            p2 = n2[i] - [shift_x, shift_y]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="cyan", lw=1.0)

    # ------------------------------
    # Start/end markers (debug only)
    # ------------------------------
    if not gt_plot:
        marker_specs = [
            ("derived", derived_plot, "cyan"),
            ("edge1", edge1_plot, "red"),
            ("edge2", edge2_plot, "lime"),
        ]
        for _name, segs, color in marker_specs:
            if not segs:
                continue
            seg = np.asarray(segs[0], float)
            if seg.ndim != 2 or len(seg) < 1:
                continue
            p_start = seg[0] - [shift_x, shift_y]
            p_end = seg[-1] - [shift_x, shift_y]
            ax.scatter([p_start[0]], [p_start[1]], s=22, c=color, marker="o", edgecolors="black", linewidths=0.4, zorder=10)
            ax.scatter([p_end[0]], [p_end[1]], s=26, c=color, marker="x", linewidths=1.0, zorder=10)

    # ------------------------------
    # Legend
    # ------------------------------
    if gt_plot:
        handles = [
            Line2D([], [], color='white', lw=2.0, linestyle='-', label='Midline'),
            Line2D([], [], color='cyan', lw=1.4, label='Normals'),
        ]
    else:
        handles = [
            Line2D([], [], color='darkorange', lw=2.0, linestyle='--', label='Midline (Manual)'),
            Line2D([], [], color='white', lw=2.0, label='Midline (Centered)'),
            Line2D([], [], color='red', lw=1.4, label='Edge 1 (Left)'),
            Line2D([], [], color='green', lw=1.4, label='Edge 2 (Right)'),
            Line2D([], [], color='cyan', lw=1.4, label='Normals'),
        ]

    if bbox is not None:
        handles.append(Line2D([], [], color='dodgerblue', lw=2.0, label='BBox'))

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
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0)
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


def plot_gt_normals_on_gtbw(gt_mask_u8, derived_midline_xy, midline_xy, e1, e2, out_png, crop_bbox=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    t0 = time.perf_counter()
    def _pdbg(msg: str) -> None:
        print(f"[NORMALS_VIS_DBG] +{(time.perf_counter()-t0):.3f}s {msg}", flush=True)

    H, W = gt_mask_u8.shape[:2]
    #_pdbg(f"start shape=({H},{W}) out={out_png}")
    #_pdbg("figure:create:start")
    fig, ax = plt.subplots(figsize=(7, 7))
    #_pdbg("figure:create:done")
    #_pdbg("imshow:start")
    ax.imshow(gt_mask_u8, cmap='gray', interpolation='nearest')
    #_pdbg("imshow:done")

    handles = []

    if derived_midline_xy is not None and derived_midline_xy.ndim == 2 and len(derived_midline_xy) >= 2:
        ax.plot(derived_midline_xy[:,0], derived_midline_xy[:,1], '-', lw=1.4, color='darkorange', alpha=0.95)
        handles.append(Line2D([], [], color='darkorange', lw=1.8, label='Derived Midline'))
    
    if midline_xy is not None and midline_xy.ndim == 2 and len(midline_xy) >= 2:
        ax.plot(midline_xy[:,0], midline_xy[:,1], '--', lw=1, color='red', alpha=0.95)
        handles.append(Line2D([], [], color='red', lw=1.8, label='Midline'))

    if e1 is not None and e2 is not None and len(e1) > 1 and len(e2) > 1:
        step = max(1, len(e1)//60)
        idx = np.arange(0, len(e1), step, dtype=int)
        idx = idx[(idx >= 0) & (idx < len(e1)) & (idx < len(e2))]
        #_pdbg(f"normals_segments:prep count={len(idx)} step={step}")
        if len(idx) > 0:
            segs = np.stack([e1[idx, :2], e2[idx, :2]], axis=1)
            lc = LineCollection(segs, colors="cyan", linewidths=1.0, alpha=0.9)
            ax.add_collection(lc)
        #_pdbg("normals_segments:done")
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

    #_pdbg("tight_layout:start")
    plt.tight_layout(pad=0)
    #_pdbg("tight_layout:done")
    #_pdbg("savefig:start")
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0)
    #_pdbg("savefig:done")
    #_pdbg("close_fig:start")
    plt.close(fig)
    #_pdbg("done")

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
    (e1x, e1y, e2x, e2y, _w), _ = normals_from_mask_for_midline(
        mid,
        mask_bin,
        max_radius=50,
        image_hw=mask_bin.shape[:2],
    )
    e1 = np.column_stack([e1x, e1y]).astype(float)
    e2 = np.column_stack([e2x, e2y]).astype(float)
    plot_gt_normals_on_gtbw(gt_full_u8, mid, None, e1, e2,
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
#   - compare_widths_for_aligned_cracks(ann_like, crack_mask, base_name, metrics_dir, display=False, tag=None)

# For clarity: ann_like for compare_widths_for_aligned_cracks is {"atomic_cracks": {cid: crackdict, ...}}

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
        plt.savefig(out_png)
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
        plt.savefig(out_path); plt.close()
        
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
        plt.savefig(out_path)
        plt.close()

    return diffs

def plot_core_timing_bars(metrics_dir):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    csv_path = os.path.join(metrics_dir, "timings_core.csv")
    if not os.path.exists(csv_path):
        print("[TIMING_PLOT] ❌ no timing CSV:", csv_path)
        return

    df = pd.read_csv(csv_path)
    print("[TIMING_PLOT] loaded CSV rows =", len(df))
    print(df[["crack_type", "supervision", "algo_variant", "crack_id"]])

    def _to_num(v):
        try:
            return float(pd.to_numeric(v, errors="coerce"))
        except Exception:
            return 0.0

    def _fmt_id(v):
        try:
            fv = float(v)
            if abs(fv - round(fv)) < 1e-9:
                return str(int(round(fv)))
        except Exception:
            pass
        return str(v)

    # ------------------------------------------------------------
    # LOOP BY SUPERVISION (NO COLLAPSING, NO MAGIC)
    # ------------------------------------------------------------
    for supervision in sorted(df["supervision"].dropna().unique()):
        print(f"\n[TIMING_PLOT] === SUPERVISION: {supervision} ===")
        
        #dumb but works
        supervision_type_dir = os.path.join(metrics_dir, supervision)
        os.makedirs(supervision_type_dir, exist_ok=True)

        dfm = df[df["supervision"] == supervision].copy()

        atomic_df = dfm[dfm["crack_type"] == "atomic"].copy()
        combined_df = dfm[dfm["crack_type"] == "combined"].copy()

        print("[TIMING_PLOT] atomic rows   =", len(atomic_df))
        print("[TIMING_PLOT] combined rows =", len(combined_df))

        # ==========================================================
        # (A) CORE RUNTIME SUMMARY
        # ==========================================================
        labels, vals = [], []

        if not atomic_df.empty:
            labels.append("Edge masks / atomic")
            vals.append(np.nansum(atomic_df["edge_masks_sec"].apply(_to_num)))

            labels.append("Edge tracking / atomic")
            vals.append(np.nansum(atomic_df["edges_tracking_sec"].apply(_to_num)))

        if not combined_df.empty:
            labels.append("Combined crack")
            vals.append(np.nansum(combined_df["build_combined_sec"].apply(_to_num)))

        if labels:
            fig, ax = plt.subplots(figsize=(8, 4))
            xs = np.arange(len(labels))
            ax.bar(xs, vals)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=15, ha="right", fontweight="bold")
            ax.set_ylabel("Time (s)")
            ax.set_title(f"Core runtime per image — {supervision}")

            for i, v in enumerate(vals):
                ax.text(xs[i], max(v * 1.03, 0.01), f"{v:.2f}", ha="center")

            out_core = os.path.join(metrics_dir, supervision, f"timings_core_{supervision}.png")
            plt.tight_layout()
            plt.savefig(out_core, bbox_inches="tight")
            plt.close()
            print("[TIMING_PLOT] wrote:", out_core)

        # ==========================================================
        # (B) ATOMIC EDGE-TRACKING SUBTIMINGS (RESTORED)
        # ==========================================================
        if not atomic_df.empty:
            print("[TIMING_PLOT] building atomic edge-tracking breakdown")

            atomic_df["_cid_num"] = atomic_df["crack_id"].apply(_to_num)
            atomic_df = atomic_df.sort_values("_cid_num")

            core_cols = [
                "edges_gradients_sec",
                "edges_tensor_sec",
                "edges_mask_norm_sec",
                "edges_metric_build_sec",
                "edges_geodesic1_sec",
                "edges_geodesic2_sec",
                "edges_pair_normals_sec",
            ]

            crack_ids = [_fmt_id(x) for x in atomic_df["crack_id"]]
            fig2, ax2 = plt.subplots(
                figsize=(6 + len(crack_ids) * 1.5, 4)
            )
            xs = np.arange(len(crack_ids))

            for idx, (_, row) in enumerate(atomic_df.iterrows()):
                total = _to_num(row.get("edges_tracking_sec", 0.0))
                internal = sum(_to_num(row.get(c, 0.0)) for c in core_cols)
                remainder = max(total - internal, 0.0)

                bottom = 0.0
                for c in core_cols:
                    v = _to_num(row.get(c, 0.0))
                    ax2.bar(idx, v, bottom=bottom,
                            label=(c if idx == 0 else ""))
                    bottom += v

                ax2.bar(idx, remainder, bottom=bottom,
                        label=("edges_remainder_sec" if idx == 0 else ""))

            ax2.set_xticks(xs)
            ax2.set_xticklabels([f"cid {c}" for c in crack_ids],
                                fontsize=10, fontweight="bold")
            ax2.set_ylabel("Time (s)")
            ax2.set_title(f"Atomic edges tracking breakdown — {supervision}",
                          fontsize=13, fontweight="bold")
            ax2.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")

            out_atomic = os.path.join(
                metrics_dir, supervision, f"timings_edges_tracking_{supervision}.png"
            )
            plt.tight_layout()
            plt.savefig(out_atomic, bbox_inches="tight")
            plt.close()
            print("[TIMING_PLOT] wrote:", out_atomic)

        # ==========================================================
        # (C) COMBINED CRACK TIMING STACK
        # ==========================================================
        #print(combined_df)
        if not combined_df.empty:
            print("[TIMING_PLOT] building combined timing breakdown")

            # Preserve insertion / CSV order for semantic IDs
            combined_df = combined_df.reset_index(drop=True)
            combined_df["_sort_key"] = combined_df["crack_id"].apply(
                lambda s: [int(x) for x in s.split("_")]
            )
            combined_df = combined_df.sort_values("_sort_key").drop(columns="_sort_key")


            fig3, ax3 = plt.subplots(
                figsize=(6 + len(combined_df) * 1.5, 4)
            )
            xs = np.arange(len(combined_df))

            for idx, (_, r) in enumerate(combined_df.iterrows()):
                t_stitch = _to_num(r.get("stitching_sec", 0))
                t_masks  = _to_num(r.get("combine_edge_masks_sec", 0))
                t_track  = _to_num(r.get("combine_edge_tracking_sec", 0))
                t_post   = _to_num(r.get("combine_postprocess_sec", 0))
                total    = _to_num(r.get("build_combined_sec", 0))

                rem = max(total - (t_stitch + t_masks + t_track + t_post), 0.0)

                parts = [
                    ("stitching_sec", t_stitch),
                    ("combine_edge_masks_sec", t_masks),
                    ("combine_edge_tracking_sec", t_track),
                    ("combine_postprocess_sec", t_post),
                    ("overhead_remainder_sec", rem),
                ]

                bottom = 0.0
                for name, v in parts:
                    ax3.bar(idx, v, bottom=bottom,
                            label=(name if idx == 0 else ""))
                    bottom += v

            labels = combined_df["crack_id"].astype(str)

            ax3.set_xticks(xs)
            ax3.set_xticklabels([f"comb_{l}" for l in labels],
                                fontsize=10, fontweight="bold")
            ax3.set_ylabel("Time (s)")
            ax3.set_title(f"Combined crack subtiming breakdown — {supervision}",
                          fontsize=13, fontweight="bold")
            ax3.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")

            out_comb = os.path.join(
                metrics_dir, supervision, f"timings_combined_{supervision}.png"
            )
            plt.tight_layout()
            plt.savefig(out_comb, bbox_inches="tight")
            plt.close()
            print("[TIMING_PLOT] wrote:", out_comb)



