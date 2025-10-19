from typing import Dict, Any
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from cracktools.segmentation import edge_masks, edges_tracking
from crackutils import CrackUtils


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
    man_g = np.asarray(payload.get("manual_midline_global"), float)
    pts_crop = np.asarray(payload.get("pts_crop"))
    print(f"bbox={bbox}")
    print(f"crop_shape={gray.shape if gray is not None else None}")
    print(f"manual_midline_global first={man_g[0]} last={man_g[-1]} len={len(man_g)}")
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


# ---------------------------------------------------------------------
# Helper: build binary mask from two edges or fallback ribbon
# ---------------------------------------------------------------------
def _crop_mask_from_edges(hc, wc, e1, e2, midline_xy=None, min_area=0.5, ribbon_px=4):
    mask = np.zeros((hc, wc), np.uint8)
    if e1 is None or e2 is None or len(e1) < 2 or len(e2) < 2:
        # fallback to ribbon if missing
        if midline_xy is not None and len(midline_xy) >= 2:
            pts = np.round(midline_xy).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(mask, [pts], False, 1,
                          thickness=max(3, ribbon_px), lineType=cv2.LINE_AA)
        return mask

    ex = np.concatenate([e1[:, 0][::-1], e2[:, 0]])
    ey = np.concatenate([e1[:, 1][::-1], e2[:, 1]])
    area = 0.5 * abs(np.dot(ex, np.roll(ey, -1)) - np.dot(ey, np.roll(ex, -1)))

    if area > min_area:
        poly = np.round(np.column_stack([ex, ey])).astype(np.int32)
        cv2.fillPoly(mask, [poly], 1, lineType=cv2.LINE_AA)
    elif midline_xy is not None and len(midline_xy) >= 2:
        pts = np.round(midline_xy).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(mask, [pts], False, 1,
                      thickness=max(3, ribbon_px), lineType=cv2.LINE_AA)
    return mask


# ---------------------------------------------------------------------
# Worker: edge mask → edge tracking → mask creation → midline metrics
# ---------------------------------------------------------------------
def edge_param_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np, cv2, os, matplotlib.pyplot as plt

    img = payload["image_crop_gray"]
    pts_crop = payload["pts_crop"]
    track_local_yx = payload["adjusted_track"]  # (2,N) [y,x]
    man_xy_g  = np.asarray(payload["manual_midline_global"], float)
    manual_normals_crop = payload.get("manual_normals_crop", None)
    x, y, w, h = payload["bbox"]
    P = payload["params"]

    # --- Deep comparison debug for orientation and alignment ---
    _compare_reference_debug(payload, track_local_yx)

    try:
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        em1, em2 = edge_masks(img_norm, track_local_yx, window_half_size=int(P["window_half_size"]))
        print(f"[edge_worker] params={P} em1.max={em1.max():.2f}, em2.max={em2.max():.2f}")
        print(f"[edge_worker] bbox=({x},{y},{w},{h}) crop_shape={img.shape}")

        # ---- debug midline / track consistency ----
        print(f"[edge_worker] manual_midline_global[0]={man_xy_g[0]}, [-1]={man_xy_g[-1]}")
        print(f"[edge_worker] pts_crop[0]={pts_crop[0]}, pts_crop[1]={pts_crop[1]}")
        print(f"[edge_worker] track_local_yx start={track_local_yx[:,0]}, end={track_local_yx[:,-1]}")

        midline_xy_crop = np.column_stack([track_local_yx[1], track_local_yx[0]])
        res = edges_tracking(
            image_crop=img_norm,
            pts_cropp=pts_crop,
            edge_mask1_cropp=em1, edge_mask2_cropp=em2,
            midline=midline_xy_crop,
            mu=int(P["mu"]), l=int(P["l"]), p=int(P["p"]),
            return_normal_edges=True,
        )

        track_e1, track_e2 = res.get("geodesic_edges", (None, None))
        if track_e1 is None or track_e2 is None:
            print(f"[edge_worker] ❌ no geodesic edges returned for {P}")
            _debug_plot_edge_worker(img_norm, em1, em2, midline_xy_crop, None, None, P, tag="no_edges")
            return {"status": "fail_no_edges", **P}

        # ---- debug geodesic outputs ----
        track_e1 = np.asarray(track_e1, float)
        track_e2 = np.asarray(track_e2, float)
        print(f"[edge_worker] e1[0]={track_e1[0]}, e1[-1]={track_e1[-1]}")
        print(f"[edge_worker] e2[0]={track_e2[0]}, e2[-1]={track_e2[-1]}")
        print(f"[edge_worker] e1.shape={track_e1.shape}, e2.shape={track_e2.shape}")

        hc, wc = img.shape[:2]
        mask_crop = _crop_mask_from_edges(hc, wc, track_e1, track_e2, midline_xy=midline_xy_crop)

        # ---- quick overlay image for visual debugging ----
        dbg_dir = "debug_compare"
        os.makedirs(dbg_dir, exist_ok=True)
        overlay = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
        for pt in midline_xy_crop.astype(int):
            cv2.circle(overlay, tuple(pt), 1, (255, 255, 0), -1)
        if len(track_e1) > 1:
            cv2.polylines(overlay, [track_e1.astype(np.int32)], False, (0, 0, 255), 1)
        if len(track_e2) > 1:
            cv2.polylines(overlay, [track_e2.astype(np.int32)], False, (0, 255, 0), 1)
        fsave = os.path.join(dbg_dir,
            f"overlay_mu{P['mu']}_l{P['l']}_p{P['p']}_w{P['window_half_size']}.png")
        cv2.imwrite(fsave, overlay)
        print(f"[edge_worker] overlay saved → {fsave}")

        # ======================================================
        # 🔍 Compute midline metrics vs manual midline (edge-aware)
        # ======================================================
        try:
            from crackutils import CrackUtils
            import numpy as np

            # --- Build auto midline from manual normals if available ---
            if manual_normals_crop is not None:
                e1 = np.column_stack(manual_normals_crop[0])
                e2 = np.column_stack(manual_normals_crop[1])
                if len(e1) == len(e2) and len(e1) > 2:
                    auto_midline = 0.5 * (e1 + e2)
                else:
                    n = min(len(e1), len(e2))
                    auto_midline = 0.5 * (e1[:n] + e2[:n])
            else:
                # fallback if no manual normals available — use crude midpoint of tracked edges
                n = min(len(track_e1), len(track_e2))
                auto_midline = 0.5 * (track_e1[:n] + track_e2[:n])

            # --- Normalize coordinate frames before metric computation ---
            # Convert manual midline (global) into local crop coordinates
            man_midline = man_xy_g - np.array([x, y], float)

            # Build auto_midline as before
            if manual_normals_crop is not None:
                e1 = np.column_stack(manual_normals_crop[0])
                e2 = np.column_stack(manual_normals_crop[1])
                n = min(len(e1), len(e2))
                auto_midline = 0.5 * (e1[:n] + e2[:n])
            else:
                n = min(len(track_e1), len(track_e2))
                auto_midline = 0.5 * (track_e1[:n] + track_e2[:n])

            # --- Compute metrics ---
            metrics = CrackUtils.compute_midline_metrics(auto_midline, man_midline, tau=3.0)
            print(f"[edge_worker] metrics computed: chamfer_mean={metrics.get('chamfer_mean', np.nan):.3f}")

            metrics = CrackUtils.compute_midline_metrics(auto_midline, man_midline, tau=3.0)
            print(f"[edge_worker] metrics computed: chamfer_mean={metrics.get('chamfer_mean', np.nan):.3f}")
        except Exception as e:
            print(f"[edge_worker] ⚠️ midline metrics failed: {e}")
            metrics = {k: np.nan for k in [
                "chamfer_mean","hausdorff","angle_err_deg","coverage",
                "directional_bias","curvature_rms_ratio","local_thickness_corr"
            ]}

        return {
            "status": "ok",
            "mask_crop_sum": int(mask_crop.sum()),
            **P,
            **metrics
        }

    except Exception as e:
        print(f"[edge_worker] ❌ unexpected failure for params={P}: {e}")
        _debug_plot_edge_worker(img, np.zeros_like(img), np.zeros_like(img),
                                None, None, None, P, tag="crash")
        return {"status": "fail_exception", "error": str(e), **P}
