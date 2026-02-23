#!/usr/bin/env python3

# --- CPU AFFINITY PINNING (Windows-safe) --- for core ultra
import os
'''try:
    import psutil
    _p = psutil.Process(os.getpid())
    # Pin to P-cores only
    _p.cpu_affinity([0, 1, 2, 3])
    print(f"[AFFINITY] Worker PID={os.getpid()} pinned to {_p.cpu_affinity()}")
except Exception as e:
    print(f"[AFFINITY] Could not set affinity: {e}")'''

from typing import Dict, Any
import numpy as np
import cv2
import os
import matplotlib
matplotlib.use("Agg")   # fastest non-GUI backend for PNG/PDF/SVG export

import matplotlib.pyplot as plt
plt.ioff()   # no interactive figure updates

from cracktools.segmentation import edge_masks, edges_tracking
#from helpers import metrics
from helpers import *
from helpers import plot_metrics
from helpers.plot_metrics import *
from helpers.geometry_canonical import orient_segment_to_reference

# ---------------------------------------------------------------------
# Helper: mini diagnostic plot for failed or weird edge cases
# ---------------------------------------------------------------------
def _debug_plot_edge_worker(img, em1, em2, midline, e1, e2, params, tag="debug"):
    """Save visual diagnostics for a given crop under ./debug_failures/"""
    try:
        os.makedirs("debug_failures", exist_ok=True)
        fig, axs = plt.subplots(1, 3, figsize=(10, 3))
        axs[0].imshow(img, cmap='gray');  axs[0].set_title("Crop gray")
        axs[1].imshow(em1, cmap='magma'); axs[1].set_title("edge_mask1")
        axs[2].imshow(em2, cmap='magma'); axs[2].set_title("edge_mask2")

        for ax in axs:
            if midline is not None and len(midline) >= 2:
                ax.plot(midline[:,0], midline[:,1], 'y-', lw=1)
            if e1 is not None and np.ndim(e1) == 2 and len(e1) >= 2:
                ax.plot(e1[:,0], e1[:,1], 'r-', lw=0.8)
            if e2 is not None and np.ndim(e2) == 2 and len(e2) >= 2:
                ax.plot(e2[:,0], e2[:,1], 'b-', lw=0.8)
            ax.set_xlim(0, img.shape[1])
            ax.set_ylim(img.shape[0], 0)
            ax.axis("off")

        fig.suptitle(f"edge_worker {tag}\nparams={params}")
        fname = f"debug_failures/edge_worker_{tag}_mu{params['mu']}_l{params['l']}_p{params['p']}_w{params['window_half_size']}.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close(fig)
        print(f"[edge_worker] saved debug plot → {fname}")
    except Exception as e:
        print(f"[edge_worker] plot fail: {e}")


# ---------------------------------------------------------------------
# Helper: compare payload vs GUI reference geometry
# ---------------------------------------------------------------------
def _compare_reference_debug(payload, track_local_yx):
    """Prints geometry differences between payload (worker) and GUI crop."""
    import numpy as np
    print("\n=== [COMPARE DEBUG: edge_param_worker()] ===")
    bbox = payload.get("bbox")
    gray = payload.get("image_crop_gray")
    man_g = np.asarray(payload.get("midline_global"), float)
    pts_crop = np.asarray(payload.get("pts_crop"))
    print(f"bbox={bbox}")
    print(f"crop_shape={gray.shape if gray is not None else None}")
    print(f"midline_global first={man_g[0]} last={man_g[-1]} len={len(man_g)}")
    print(f"pts_crop[0]={pts_crop[0]} pts_crop[1]={pts_crop[1]}")
    print(f"track_local_yx start(yx)={track_local_yx[:,0]} end(yx)={track_local_yx[:,-1]}")
    # flip to [x,y] for intuitive view
    txy0 = track_local_yx[:,0][::-1]
    txy1 = track_local_yx[:,-1][::-1]
    print(f"track_local_yx start(xy)={txy0} end(xy)={txy1}")
    d0 = np.linalg.norm(txy0 - pts_crop[0])
    d1 = np.linalg.norm(txy1 - pts_crop[1])
    print(f"Δ_start_to_p0={d0:.2f}px, Δ_end_to_p1={d1:.2f}px (should both ≈0)")
    print("===========================================\n")
    
def _crop_mask_from_edges(hc, wc, e1, e2, midline_xy=None, min_area=0.5, ribbon_px=4, debug_save=False, debug_dir="./debug_compare", tag=""):
    """
    Build a filled polygon mask from edge1/edge2 or fall back to midline ribbon.
    Works both for crop-local and global coords (auto-clipped).
    Optionally saves a debug PNG of the mask.
    """
    import numpy as np, cv2, os

    mask = np.zeros((hc, wc), np.uint8)

    def _finite_xy(A):
        if A is None:
            return np.empty((0, 2))
        A = np.asarray(A, float)
        return A[np.isfinite(A).all(1)]

    e1 = _finite_xy(e1)
    e2 = _finite_xy(e2)

    # fallback to ribbon if missing
    if len(e1) < 2 or len(e2) < 2:
        if midline_xy is not None and len(midline_xy) >= 2:
            pts = np.round(midline_xy).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(mask, [pts], False, 1,
                          thickness=max(3, ribbon_px), lineType=cv2.LINE_AA)
        if debug_save:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, f"mask_{tag}_fallback.png"), mask * 255)
        return mask

    # combine edges into polygon
    ex = np.concatenate([e1[:, 0][::-1], e2[:, 0]])
    ey = np.concatenate([e1[:, 1][::-1], e2[:, 1]])

    # clip to valid region
    ex = np.clip(ex, 0, wc - 1)
    ey = np.clip(ey, 0, hc - 1)

    area = 0.5 * abs(np.dot(ex, np.roll(ey, -1)) - np.dot(ey, np.roll(ex, -1)))
    if area > min_area:
        poly = np.round(np.column_stack([ex, ey])).astype(np.int32)
        cv2.fillPoly(mask, [poly], 1, lineType=cv2.LINE_AA)
    elif midline_xy is not None and len(midline_xy) >= 2:
        pts = np.round(midline_xy).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(mask, [pts], False, 1,
                      thickness=max(3, ribbon_px), lineType=cv2.LINE_AA)

    if debug_save:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"mask_{tag}.png"), mask * 255)

    return mask

# ======== NORMALS / VIZ HELPERS ========

import numpy as np
import matplotlib.pyplot as plt
import cv2, os
from typing import Dict, Any

def _nx2(a) -> np.ndarray:
    """Coerce to (N,2) float array (robust to 1D, (2,N), weird shapes)."""
    a = np.asarray(a, float)
    if a.ndim == 1:
        return a.reshape(1, -1) if a.size == 2 else np.zeros((0, 2), float)
    if a.ndim == 2:
        if a.shape[0] == 2 and a.shape[1] != 2:
            a = a.T
        if a.shape[1] != 2:
            a = a.reshape(-1, 2)
        return a.astype(float)
    return a.reshape(-1, 2).astype(float)

def extract_normals_from_res(res):
    """Return (e1,e2) in crop coords, each (N,2)."""
    nf = (res.get("normal_edge_points_full")
          or res.get("normal_edge_points_clipped")
          or res.get("normal_edge_points"))
    if nf is None:
        return np.zeros((0, 2)), np.zeros((0, 2))
    if isinstance(nf, dict):
        e1 = _nx2(nf.get("edge1", []))
        e2 = _nx2(nf.get("edge2", []))
    elif isinstance(nf, (list, tuple)) and len(nf) == 2:
        e1 = _nx2(nf[0]); e2 = _nx2(nf[1])
    else:
        e1 = e2 = np.zeros((0, 2))
    m = min(len(e1), len(e2))
    return e1[:m], e2[:m]
    
def plot_normals_pretty(
    image_gray,
    track_e1,
    track_e2,
    midline_xy,
    e1,
    e2,
    out_png,
    crack_id,
    derived_midline_xy=None,
):
    plot_edges_and_normals(
        base_image=image_gray,
        midline_segs=[midline_xy],
        derived_midline_segs=([derived_midline_xy] if derived_midline_xy is not None else []),
        edge1_segs=[track_e1],
        edge2_segs=[track_e2],
        norm1_segs=[e1],
        norm2_segs=[e2],
        bbox=None,
        out_png=out_png,
        title=f"Atomic Crack {crack_id}",
    )

def plot_widths_colormap_on_crop(
    gt_vs_manual_rgb,
    e1, e2,
    midline_xy,
    derived_midline_xy=None,
    track_e1=None,
    track_e2=None,
    out_png=None
):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from scipy.ndimage import gaussian_filter1d
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # ---- convert arrays ----
    e1  = np.asarray(e1, float)
    e2  = np.asarray(e2, float)
    mid_src = derived_midline_xy if derived_midline_xy is not None else midline_xy
    mid = np.asarray(mid_src, float)

    if e1.ndim != 2 or e2.ndim != 2 or mid.ndim != 2:
        return
    if e1.shape[1] != 2 or e2.shape[1] != 2 or mid.shape[1] != 2:
        return

    n = min(len(e1), len(e2), len(mid))
    if n < 2:
        return

    e1  = e1[:n]
    e2  = e2[:n]
    mid = mid[:n]

    # ------------------------------------------------------------
    # Split into NaN-separated runs
    # ------------------------------------------------------------
    finite = (
        np.isfinite(e1[:, 0]) & np.isfinite(e1[:, 1]) &
        np.isfinite(e2[:, 0]) & np.isfinite(e2[:, 1]) &
        np.isfinite(mid[:, 0]) & np.isfinite(mid[:, 1])
    )

    runs = []
    i = 0
    while i < n:
        if not finite[i]:
            i += 1
            continue
        j = i + 1
        while j < n and finite[j]:
            j += 1
        if j - i >= 2:
            runs.append((i, j))
        i = j

    if not runs:
        return

    # ------------------------------------------------------------
    # GLOBAL width scale (0 → max width)
    # ------------------------------------------------------------
    all_widths = []
    for i0, i1 in runs:
        w = np.linalg.norm(e1[i0:i1] - e2[i0:i1], axis=1)
        w = w[np.isfinite(w)]
        if w.size:
            all_widths.append(w)

    if not all_widths:
        return

    all_widths = np.concatenate(all_widths)
    max_w = float(np.nanmax(all_widths))
    if not np.isfinite(max_w) or max_w <= 0:
        max_w = 1.0

    cmap = plt.get_cmap("inferno")
    norm = mpl.colors.Normalize(vmin=0.0, vmax=max_w)

    H, W = gt_vs_manual_rgb.shape[:2]

    fig, ax = plt.subplots(figsize=(7, 7), dpi=320)

    # ---- background ----
    ax.imshow(gt_vs_manual_rgb[..., ::-1], interpolation="bilinear")

    # ------------------------------------------------------------
    # Plot each segment independently
    # ------------------------------------------------------------
    for i0, i1 in runs:
        coords = mid[i0:i1].copy()
        widths = np.linalg.norm(e1[i0:i1] - e2[i0:i1], axis=1)

        ok = np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1]) & np.isfinite(widths)
        coords = coords[ok]
        widths = widths[ok]

        if len(coords) < 2:
            continue

        widths_smooth = gaussian_filter1d(
            widths.astype(float), sigma=1.2, mode="nearest"
        )

        # Remove only consecutive duplicates
        dxy = np.diff(coords, axis=0)
        keep = np.ones(len(coords), dtype=bool)
        keep[1:] = np.any(np.abs(dxy) > 1e-6, axis=1)

        coords = coords[keep]
        widths_smooth = widths_smooth[keep]

        if len(coords) < 2:
            continue

        colors = cmap(norm(widths_smooth))

        for k in range(len(coords) - 1):
            ax.plot(
                [coords[k, 0], coords[k + 1, 0]],
                [coords[k, 1], coords[k + 1, 1]],
                color=colors[k],
                linewidth=2.4,
                alpha=0.97,
                solid_capstyle="round",
            )

    # ---- optional geodesic edges ----
    if track_e1 is not None and len(track_e1) > 1:
        te1 = np.asarray(track_e1, float)
        ax.plot(te1[:, 0], te1[:, 1], "-", lw=1.4,
                color="magenta", alpha=0.9, label="Edge 1 (Left)")
    if track_e2 is not None and len(track_e2) > 1:
        te2 = np.asarray(track_e2, float)
        ax.plot(te2[:, 0], te2[:, 1], "-", lw=1.4,
                color="lime", alpha=0.9, label="Edge 2 (Right)")

    # ---- colorbar with explicit ticks ----
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Estimated width (px)", fontsize=10, fontweight="bold")

    # Keep automatic ticks, but ensure endpoints are present & labeled
    '''ticks = list(cb.get_ticks())

    if ticks:
        ticks[0] = 0.0
        ticks[-1] = max_w
        cb.set_ticks(ticks)

    # Override only the endpoint labels
    ticklabels = [f"{t:g}" for t in ticks]
    ticklabels[0] = "0"
    ticklabels[-1] = f"{max_w:.1f}"

    cb.set_ticklabels(ticklabels)'''
    
    ticks = list(cb.get_ticks())

    if len(ticks) >= 2:
        tol = 0.3

        # Replace endpoints
        vmin, vmax = 0.0, max_w
        ticks[0]  = vmin
        ticks[-1] = vmax

        # Remove ticks too close to endpoints (except endpoints themselves)
        cleaned = []
        for i, t in enumerate(ticks):
            if i == 0 or i == len(ticks) - 1:
                cleaned.append(t)
            else:
                if abs(t - vmin) > tol and abs(t - vmax) > tol:
                    cleaned.append(t)

        cb.set_ticks(cleaned)
        cb.set_ticklabels([f"{t:.1f}" for t in cleaned])

    # ---- legend ----
    mid_label = "Derived midline (width color map)" if derived_midline_xy is not None else "Midline (width color map)"
    handles = [
        Line2D([], [], color="gray", lw=2.4,
               label=mid_label),
    ]

    if track_e1 is not None and len(track_e1) > 1:
        handles.append(Line2D([], [], color="magenta", lw=1.8,
                              label="Edge 1 (Left)"))
    if track_e2 is not None and len(track_e2) > 1:
        handles.append(Line2D([], [], color="lime", lw=1.8,
                              label="Edge 2 (Right)"))

    handles.extend([
        Patch(facecolor=(1, 1, 1), edgecolor="gray", label="Overlap (IoU)"),
        Patch(facecolor=(1, 1, 0), edgecolor="gray", label="Manual only"),
        Patch(facecolor=(1, 0, 0), edgecolor="gray", label="GT only"),
    ])

    leg = ax.legend(
        handles=handles,
        loc="lower right",
        fontsize=8,
        frameon=True,
        framealpha=0.80,
        handlelength=2.8,
        handletextpad=0.7,
        title="Legend",
        title_fontsize=11,
    )

    plt.setp(leg.get_title(), color="blue", fontweight="bold")
    for t in leg.get_texts():
        t.set_fontweight("bold")

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    plt.tight_layout(pad=0)

    if out_png:
        fig.savefig(out_png, dpi=320, bbox_inches="tight", pad_inches=0)

    plt.close(fig)

def save_cropped_overlay(img_full_bgr, bbox, mask_or_rgb, out_png, margin=0):
    x, y, w, h = map(int, bbox)
    H, W = img_full_bgr.shape[:2]
    x0 = max(0, x - margin); y0 = max(0, y - margin)
    x1 = min(W, x + w + margin); y1 = min(H, y + h + margin)
    if mask_or_rgb.ndim == 2:
        overlay = cv2.cvtColor(mask_or_rgb, cv2.COLOR_GRAY2BGR)
    else:
        overlay = mask_or_rgb
    crop = overlay[y0:y1, x0:x1]
    cv2.imwrite(out_png, crop)

# ---------------------------------------------------------------------
# Worker: edge mask → edge tracking → mask creation → midline metrics
# ---------------------------------------------------------------------     
from typing import Dict, Any

def edge_param_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Core edge worker for atomic cracks.

    Supports two modes:

    1) Normal compute mode (default):
        - runs edge_masks + edges_tracking + mask generation
        - writes plots (if not calibration_only)
        - (optionally) writes geom_cache.npz (if calibration_only)

    2) Plot-only mode (AUTO):
        - if payload["geom_cache_path"] is provided and exists
        - loads geometry from geom_cache.npz and SKIPS ALL heavy compute
        - runs plotting + metrics using cached mask/edges/normals

    Expects payload to contain (at minimum):
      - image_crop_gray      : (Hc, Wc) float32 or uint8
      - pts_crop             : list of 2 points in crop coords
      - adjusted_track       : (2, N) local [y, x] midline
      - manual_midline_global: (N, 2) global [x, y] midline
      - bbox                 : (x, y, w, h) in full-image coords
      - params               : dict with window_half_size, mu, l, p, seg_mode, ...
      - save_folder          : base folder for metrics/debug
      - image_base           : image basename (e.g., "1")
      - crack_id             : string crack id
      - gt_crop              : optional (Hc, Wc) uint8 binary GT mask (0/1 or 0/255)
      - source               : optional "manual" / "auto..." for directory tagging

    Optional:
      - geom_cache_path      : if provided and file exists -> plot-only mode
      - calibration_only     : if True -> writes geom_cache.npz under debug/param_tag
      - image_shape, gt_full, original_image : for global overlay plot

    Returns:
      result dict with:
        - status
        - bbox, mask_bbox
        - mask_crop (list)
        - geodesic_edges (global coords)
        - normal_edge_points_full
        - timing
        - and any region/boundary/surface metrics if GT provided
    """
    import numpy as np
    import cv2
    import os
    import time

    from helpers.metrics import (
        compute_mask_metrics,
        boundary_fscore,
        assd_hd95,
        normals_from_mask_for_midline,
    )
    from helpers.plot_metrics import (
        plot_gt_normals_on_gtbw,
    )
    # edge_masks / edges_tracking assumed imported at module level
    # plot_normals_pretty, plot_widths_colormap_on_crop, save_gt_vs_manual_overlay assumed available
    # extract_normals_from_res assumed available

    img        = payload["image_crop_gray"]
    pts_crop   = payload["pts_crop"]
    track_yx   = payload["adjusted_track"]  # (2, N) [y, x]
    mid_xy_g   = np.asarray(payload["midline_global"], float)
    x, y, w, h = map(int, payload["bbox"])
    P          = payload["params"]

    crack_id   = str(payload.get("crack_id", "?"))
    base_name  = str(payload.get("image_base", "unknown"))

    # global info for overlay (may be absent in some modes)
    image_shape    = payload.get("image_shape", None)
    gt_full        = payload.get("gt_full", None)
    original_image = payload.get("original_image", None)

    calib_only = bool(payload.get("calibration_only", False))

    seg_mode = str(P.get("seg_mode", "new")).lower()
    if seg_mode not in ("old", "new"):
        seg_mode = "new"

    geom_cache_path = payload.get("geom_cache_path", None)
    plot_only = bool(geom_cache_path) and os.path.isfile(str(geom_cache_path))

    print(f"[SUPER DEBUG] Payload type {str(payload.get('midline_type', '')).lower()} | plot_only={plot_only}")

    try:
        ORIENT_DEBUG = True

        # -------------------------------------------------------
        # Normalize crop to 8-bit (still needed for plotting)
        # -------------------------------------------------------
        img_norm = cv2.normalize(
            img.astype(np.float32),
            None, 0, 255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        midline_xy_crop = np.column_stack([track_yx[1], track_yx[0]])

        # -------------------------------------------------------
        # manual vs auto tag
        # -------------------------------------------------------
        src = str(payload.get("midline_type", "")).lower()
        midline_tag = "auto" if src.startswith("auto") else "manual"

        param_tag = (
            f"_wsize{P['window_half_size']}"
            f"_mu{P['mu']}"
            f"_l{P['l']}"
            f"_p{P['p']}"
            f"_{seg_mode}"
        )

        cid       = crack_id
        save_root = payload["save_folder"]

        # ---- root per crack
        cid_root = os.path.join(save_root, "metrics", base_name, f"cid{cid}")
        cid_root = os.path.join(cid_root, midline_tag)  # .../cidX/manual or .../cidX/auto

        # ---- calibration artifacts go under debug/
        if calib_only:
            dbg_dir = os.path.join(cid_root, "debug", param_tag)
        else:
            dbg_dir = os.path.join(cid_root, "best" + param_tag)

        os.makedirs(dbg_dir, exist_ok=True)

        # =======================================================
        # PLOT-ONLY MODE: load geometry from geom_cache.npz
        # =======================================================
        if plot_only:
            t_load0 = time.perf_counter()
            data = np.load(str(geom_cache_path), allow_pickle=True)

            # Required keys
            for k in ["mask_crop", "track_e1", "track_e2", "normals_e1", "normals_e2", "midline_xy_crop", "bbox"]:
                if k not in data:
                    return {
                        "status": "fail_bad_geom_cache",
                        "error": f"geom_cache missing key: {k}",
                        "geom_cache_path": str(geom_cache_path),
                        **P,
                    }

            mask_crop       = np.asarray(data["mask_crop"])
            track_e1        = np.asarray(data["track_e1"], float)
            track_e2        = np.asarray(data["track_e2"], float)
            normals_e1      = np.asarray(data["normals_e1"], float)
            normals_e2      = np.asarray(data["normals_e2"], float)
            midline_xy_crop = np.asarray(data["midline_xy_crop"], float)
            derived_midline_crop = np.asarray(
                data["derived_midline_crop"], float
            ) if "derived_midline_crop" in data else np.asarray(midline_xy_crop, float)

            bbox_arr = np.asarray(data["bbox"]).astype(int).ravel()
            if bbox_arr.size != 4:
                return {
                    "status": "fail_bad_geom_cache",
                    "error": f"geom_cache bbox invalid shape: {bbox_arr.shape}",
                    "geom_cache_path": str(geom_cache_path),
                    **P,
                }

            x, y, w, h = map(int, bbox_arr.tolist())

            t_edge_masks = 0.0
            t_edges_tracking = 0.0
            subtiming = {}
            t_load = float(time.perf_counter() - t_load0)

        # =======================================================
        # NORMAL MODE: compute geometry (edge_masks + edges_tracking + mask)
        # =======================================================
        else:
            # -------------------------------------------------------
            # edge_masks timing
            # -------------------------------------------------------
            t0 = time.perf_counter()
            em1, em2 = edge_masks(
                img_norm,
                track_yx,
                window_half_size=int(P["window_half_size"]),
            )
            t_edge_masks = float(time.perf_counter() - t0)

            # -------------------------------------------------------
            # edges_tracking timing (geodesics + normals)
            # -------------------------------------------------------
            t1 = time.perf_counter()
            res = edges_tracking(
                image_crop=img_norm,
                pts_cropp=pts_crop,
                edge_mask1_cropp=em1, edge_mask2_cropp=em2,
                midline=midline_xy_crop,
                mu=int(P["mu"]), l=int(P["l"]), p=int(P["p"]),
                return_normal_edges=True,
                prefer_gpu=True,
                mode=seg_mode,
                debug_dir=dbg_dir
            )
            t_edges_tracking = float(time.perf_counter() - t1)

            if not isinstance(res, dict):
                print(f"[edge_worker] ⚠ edges_tracking returned {type(res)} — expected dict")
                return {"status": "fail_invalid_return", **P}

            track_e1, track_e2 = res.get("geodesic_edges", (None, None))
            if track_e1 is None or track_e2 is None:
                print(f"[edge_worker] ❌ no geodesic edges returned for params={P}")
                return {"status": "fail_no_edges", **P}

            track_e1 = np.asarray(track_e1, float)
            track_e2 = np.asarray(track_e2, float)
            derived_midline_crop = np.asarray(res.get("derived_midline", []), float)
            if derived_midline_crop.ndim != 2 or derived_midline_crop.shape[1] != 2 or len(derived_midline_crop) < 2:
                return {"status": "fail_no_derived_midline", **P}

            normals_e1, normals_e2 = extract_normals_from_res(res)
            subtiming = res.get("subtiming", {}) or {}

            t_load = 0.0

        try:
            p0 = np.asarray(pts_crop[0], float)
            p1 = np.asarray(pts_crop[1], float)
            d0 = np.asarray(derived_midline_crop[0], float)
            d1 = np.asarray(derived_midline_crop[-1], float)
            ddf = float(np.linalg.norm(d0 - p0) + np.linalg.norm(d1 - p1))
            ddr = float(np.linalg.norm(d0 - p1) + np.linalg.norm(d1 - p0))
            dflag = "reversed_candidate" if ddr < ddf else "forward_candidate"

            (
                derived_midline_crop,
                _n_unused,
                _w_unused,
                track_e1,
                track_e2,
                orient_info,
            ) = orient_segment_to_reference(
                derived_midline_crop,
                ref_start=p0,
                ref_end=p1,
                normals=None,
                widths=None,
                edge1=track_e1,
                edge2=track_e2,
                normals_are_vectors=False,
            )
            if orient_info.get("flipped", False):
                normals_e1 = np.asarray(normals_e1, float)[::-1].copy()
                normals_e2 = np.asarray(normals_e2, float)[::-1].copy()

            if ORIENT_DEBUG:
                m0 = np.asarray(midline_xy_crop[0], float)
                m1 = np.asarray(midline_xy_crop[-1], float)
                dmf = float(np.linalg.norm(m0 - p0) + np.linalg.norm(m1 - p1))
                dmr = float(np.linalg.norm(m0 - p1) + np.linalg.norm(m1 - p0))
                mflag = "reversed_candidate" if dmr < dmf else "forward_candidate"
                d0c = np.asarray(derived_midline_crop[0], float)
                d1c = np.asarray(derived_midline_crop[-1], float)
                dcf = float(np.linalg.norm(d0c - p0) + np.linalg.norm(d1c - p1))
                dcr = float(np.linalg.norm(d0c - p1) + np.linalg.norm(d1c - p0))
                dcflag = "reversed_candidate" if dcr < dcf else "forward_candidate"
                '''print(
                    f"[ORIENT DBG][edge_worker] cid={cid} mode={seg_mode} plot_only={plot_only} "
                    f"manual_vs_pts fwd={dmf:.4f} rev={dmr:.4f} flag={mflag} "
                    f"derived_pre fwd={ddf:.4f} rev={ddr:.4f} flag={dflag} "
                    f"derived_post fwd={dcf:.4f} rev={dcr:.4f} flag={dcflag} "
                    f"flipped={bool(orient_info.get('flipped', False))}"
                )'''
        except Exception as e:
            print(f"[ORIENT DBG][edge_worker] cid={cid} orientation check failed: {e}")

        if not plot_only:
            from cracktools.segmentation import generate_mask_from_edges  # adjust import path
            mask_crop = generate_mask_from_edges(
                img_gray=img_norm,
                edge1_xy=track_e1,
                edge2_xy=track_e2,
                midline_xy=derived_midline_crop,
                normals_xy=(normals_e1, normals_e2),
                out_dir=dbg_dir,
                tag=f"cid{cid}",
                do_morph=True,
            )

        # Global coords (after orientation canonicalization).
        track_e1_global = np.column_stack([track_e1[:, 0] + x, track_e1[:, 1] + y])
        track_e2_global = np.column_stack([track_e2[:, 0] + x, track_e2[:, 1] + y])

        # =======================================================
        # 1) Pretty edges + normals (crop-level)
        # =======================================================
        if not calib_only:
            pretty_path = os.path.join(dbg_dir, "edges_midlines_normals_pretty.png")
            try:
                plot_normals_pretty(
                    img_norm,
                    track_e1, track_e2,
                    midline_xy_crop,
                    normals_e1, normals_e2,
                    pretty_path,
                    cid,
                    derived_midline_xy=derived_midline_crop,
                )
            except Exception as e:
                print(f"[DEBUG VIS] ⚠ plot_normals_pretty failed for cid{cid}: {e}")

        # =======================================================
        # 2) Crop-level GT + metrics
        # =======================================================
        gt_crop = payload.get("gt_crop", None)
        metrics_all: Dict[str, float] = {}
        gt_vs_manual_overlay = None

        if gt_crop is not None:
            gt_bin   = (np.asarray(gt_crop) > 0).astype(np.uint8)
            pred_bin = (np.asarray(mask_crop) > 0).astype(np.uint8)

            base = compute_mask_metrics(gt_bin, pred_bin)
            bnd  = boundary_fscore(gt_bin, pred_bin, tau=2.0)
            surf = assd_hd95(gt_bin, pred_bin)
            metrics_all = {**base, **bnd, **surf}

            # IoU-style overlay (crop-level, 3x upscaled)
            if not calib_only:
                try:
                    vis_gray = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
                    dark_base = np.clip(vis_gray * 0.35, 0.0, 1.0)
                    overlay = dark_base.copy()

                    intersect = np.logical_and(gt_bin, pred_bin)
                    pred_only = np.logical_and(pred_bin, np.logical_not(gt_bin))
                    gt_only   = np.logical_and(gt_bin, np.logical_not(pred_bin))

                    overlay[gt_only == 1]   = (0.2, 0.2, 1.0)
                    overlay[pred_only == 1] = (0.2, 1.0, 1.0)
                    overlay[intersect == 1] = (0.95, 0.95, 0.95)

                    blended = cv2.addWeighted(overlay, 0.85, dark_base, 0.15, 0.0)
                    vis_large = cv2.resize(
                        np.clip(blended * 255.0, 0, 255).astype(np.uint8),
                        None, fx=3.0, fy=3.0,
                        interpolation=cv2.INTER_NEAREST,
                    )
                    gt_vs_manual_overlay = vis_large
                    out_iou = os.path.join(dbg_dir, "gt_vs_manual_mask.png")
                    cv2.imwrite(out_iou, vis_large)
                    print(f"[DEBUG VIS] wrote → {out_iou}")
                except Exception as e:
                    print(f"[DEBUG VIS] ⚠ crop IoU overlay failed cid{cid}: {e}")

        # =======================================================
        # 3) Widths colormap (crop-level)
        # =======================================================
        if not calib_only:
            widths_path = os.path.join(dbg_dir, "widths_colormap_on_crop.png")
            try:
                if gt_vs_manual_overlay is not None:
                    S = 3.0
                    plot_widths_colormap_on_crop(
                        gt_vs_manual_rgb = gt_vs_manual_overlay,
                        e1               = normals_e1 * S,
                        e2               = normals_e2 * S,
                        midline_xy       = derived_midline_crop * S,
                        track_e1         = track_e1 * S,
                        track_e2         = track_e2 * S,
                        out_png          = widths_path,
                    )
                else:
                    gray_rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)
                    plot_widths_colormap_on_crop(
                        gt_vs_manual_rgb = gray_rgb,
                        e1               = normals_e1,
                        e2               = normals_e2,
                        midline_xy       = derived_midline_crop,
                        track_e1         = track_e1,
                        track_e2         = track_e2,
                        out_png          = widths_path,
                    )
                print(f"[DEBUG VIS] wrote → {widths_path}")
            except Exception as e:
                print(f"[DEBUG VIS] ⚠ widths colormap failed cid{cid}: {e}")

            # =======================================================
            # 4) Normals-on-mask plots (crop-level)
            #   - always: prediction-mask normals
            #   - manual only: also GT-mask normals for side-by-side diagnosis
            # =======================================================
            try:
                pred_mask_u8 = (np.asarray(mask_crop) > 0).astype(np.uint8) * 255
                (pe1x, pe1y, pe2x, pe2y, _), _ = normals_from_mask_for_midline(
                    derived_midline_crop,
                    pred_mask_u8 > 0,
                    max_radius=50,
                )
                pe1 = np.column_stack([pe1x, pe1y])
                pe2 = np.column_stack([pe2x, pe2y])

                # Keep the historical filename as the prediction-mask plot.
                pred_normals_path = os.path.join(dbg_dir, f"{midline_tag}_derived_normals.png")
                plot_gt_normals_on_gtbw(
                    pred_mask_u8,
                    derived_midline_crop,
                    None,
                    pe1,
                    pe2,
                    pred_normals_path,
                )
                print(f"[DEBUG VIS] wrote → {pred_normals_path}")

                if midline_tag == "manual" and gt_crop is not None:
                    gt_mask_u8 = (np.asarray(gt_crop) > 0).astype(np.uint8) * 255
                    # For GT-mask diagnostics, use the original/manual midline reference.
                    # Prediction-mask diagnostics above intentionally use the derived midline.
                    (ge1x, ge1y, ge2x, ge2y, _), _ = normals_from_mask_for_midline(
                        midline_xy_crop,
                        gt_mask_u8 > 0,
                        max_radius=50,
                    )
                    ge1 = np.column_stack([ge1x, ge1y])
                    ge2 = np.column_stack([ge2x, ge2y])
                    gt_normals_path = os.path.join(dbg_dir, "manual_normals_on_gt.png")
                    plot_gt_normals_on_gtbw(
                        gt_mask_u8,
                        midline_xy_crop,
                        None,
                        ge1,
                        ge2,
                        gt_normals_path,
                    )
                    print(f"[DEBUG VIS] wrote → {gt_normals_path}")
            except Exception as e:
                print(f"[DEBUG VIS] ⚠ normals plotting failed cid{cid}: {e}")

            # =======================================================
            # 5) GLOBAL overlay via save_gt_vs_manual_overlay
            # =======================================================
            try:
                if (
                    image_shape is not None and
                    gt_full is not None and
                    original_image is not None
                ):
                    H, W = image_shape
                    pred_full = np.zeros((H, W), np.uint8)
                    pred_full[y:y + h, x:x + w] = (np.asarray(mask_crop) > 0).astype(np.uint8)

                    ys, xs = np.where(pred_full > 0)
                    if xs.size > 0:
                        x0 = max(xs.min() - 5, 0)
                        y0 = max(ys.min() - 5, 0)
                        x1 = min(xs.max() + 5, W)
                        y1 = min(ys.max() + 5, H)
                        bbox = [x0, y0, x1 - x0, y1 - y0]
                    else:
                        bbox = [0, 0, W, H]

                    global_overlay_path = os.path.join(dbg_dir, "gt_vs_manual_mask_global.png")
                    save_gt_vs_manual_overlay(
                        H,
                        W,
                        gt_full,
                        pred_full,
                        global_overlay_path,
                        bbox=bbox,
                        original_image=original_image,
                    )
                    print(f"[edge_worker] wrote global mask overlay → {global_overlay_path}")
                else:
                    print(
                        f"[edge_worker] global overlay skipped: "
                        f"image_shape_missing={image_shape is None}, "
                        f"gt_full_missing={gt_full is None}, "
                        f"original_image_missing={original_image is None}"
                    )
            except Exception as e:
                print(f"[edge_worker] ⚠ save_gt_vs_manual_overlay failed cid{cid}: {e}")

        # -------------------------------------------------------
        # Cache geometry for later plotting (calibration runs)
        # -------------------------------------------------------
        if calib_only and (not plot_only):
            try:
                geom_cache = {
                    "mask_crop": np.asarray(mask_crop).astype(np.uint8),
                    "track_e1": np.asarray(track_e1).astype(np.float32),
                    "track_e2": np.asarray(track_e2).astype(np.float32),
                    "normals_e1": np.asarray(normals_e1).astype(np.float32),
                    "normals_e2": np.asarray(normals_e2).astype(np.float32),
                    "midline_xy_crop": np.asarray(midline_xy_crop).astype(np.float32),
                    "derived_midline_crop": np.asarray(derived_midline_crop).astype(np.float32),
                    "bbox": np.asarray([x, y, w, h], np.int32),
                }
                np.savez_compressed(
                    os.path.join(dbg_dir, "geom_cache.npz"),
                    **geom_cache
                )
            except Exception as e:
                print(f"[edge_worker] ⚠ geom cache write failed: {e}")

        # =======================================================
        # 6) Pack result for caller (no snapshot write here)
        # =======================================================
        timing_dict = {
            "edge_masks_sec":     float(t_edge_masks),
            "edges_tracking_sec": float(t_edges_tracking),
        }
        if plot_only:
            timing_dict["geom_cache_load_sec"] = float(t_load)

        # include subtiming if present
        if isinstance(subtiming, dict):
            timing_dict.update(subtiming)

        normals_e1_arr = np.asarray(normals_e1, float)
        normals_e2_arr = np.asarray(normals_e2, float)
        if normals_e1_arr.ndim == 2 and normals_e1_arr.shape[1] == 2:
            normals_e1_global = np.column_stack([
                normals_e1_arr[:, 0] + x,
                normals_e1_arr[:, 1] + y,
            ])
        else:
            normals_e1_global = np.zeros((0, 2), float)

        if normals_e2_arr.ndim == 2 and normals_e2_arr.shape[1] == 2:
            normals_e2_global = np.column_stack([
                normals_e2_arr[:, 0] + x,
                normals_e2_arr[:, 1] + y,
            ])
        else:
            normals_e2_global = np.zeros((0, 2), float)

        n_w = min(len(track_e1_global), len(track_e2_global))
        pred_widths = (
            np.linalg.norm(
                np.asarray(track_e1_global[:n_w], float) - np.asarray(track_e2_global[:n_w], float),
                axis=1,
            ).tolist()
            if n_w >= 2 else []
        )

        result: Dict[str, Any] = {
            "status": "ok",
            "bbox": [x, y, w, h],
            "mask_bbox": [x, y, w, h],
            "mask_crop": np.asarray(mask_crop).tolist(),
            "midline": np.asarray(mid_xy_g, float).tolist(),
            "midline_global": np.asarray(mid_xy_g, float).tolist(),
            "derived_midline_crop": np.asarray(derived_midline_crop).tolist(),
            "derived_midline": np.column_stack([
                np.asarray(derived_midline_crop)[:, 0] + x,
                np.asarray(derived_midline_crop)[:, 1] + y,
            ]).tolist(),
            "geodesic_edges": {
                "edge1": track_e1_global.tolist(),
                "edge2": track_e2_global.tolist(),
            },
            "normal_edge_points_full": {
                "edge1": normals_e1_global.tolist(),
                "edge2": normals_e2_global.tolist(),
            },
            # Backward-compat aliases used by older callers.
            "normal_edge_points": {
                "edge1": normals_e1_global.tolist(),
                "edge2": normals_e2_global.tolist(),
            },
            "normal_edge_points_crop": {
                "edge1": normals_e1_arr.tolist(),
                "edge2": normals_e2_arr.tolist(),
            },
            "pred_widths": pred_widths,
            "timing": timing_dict,
        }

        result.update(P)
        result.update(metrics_all)
        return result

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[edge_worker] ❌ unexpected failure for params={P}")
        print(tb)

        out: Dict[str, Any] = {
            "status": "fail_exception",
            "error": str(e),
            "traceback": tb,
        }
        out.update(P)
        return out
