#!/usr/bin/env python3

from typing import Dict, Any
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from cracktools.segmentation import edge_masks, edges_tracking
from helpers import metrics
from helpers.metrics import *

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
'''def _crop_mask_from_edges(hc, wc, e1, e2, midline_xy=None, min_area=0.5, ribbon_px=4):
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
    return mask'''
    
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

# ---------------------------------------------------------------------
# Worker: edge mask → edge tracking → mask creation → midline metrics
# ---------------------------------------------------------------------
'''def edge_param_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np, cv2, os, matplotlib.pyplot as plt

    img = payload["image_crop_gray"]
    pts_crop = payload["pts_crop"]
    track_local_yx = payload["adjusted_track"]  # (2,N) [y,x]
    man_xy_g  = np.asarray(payload["manual_midline_global"], float)
    manual_normals_crop = payload.get("manual_normals_crop", None)
    x, y, w, h = payload["bbox"]
    P = payload["params"]

    # --- Deep comparison debug for orientation and alignment ---
    #_compare_reference_debug(payload, track_local_yx)

    try:
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        em1, em2 = edge_masks(img_norm, track_local_yx, window_half_size=int(P["window_half_size"]))
        #print(f"[edge_worker] params={P} em1.max={em1.max():.2f}, em2.max={em2.max():.2f}")
        #print(f"[edge_worker] bbox=({x},{y},{w},{h}) crop_shape={img.shape}")

        # ---- debug midline / track consistency ----
        #print(f"[edge_worker] manual_midline_global[0]={man_xy_g[0]}, [-1]={man_xy_g[-1]}")
        #print(f"[edge_worker] pts_crop[0]={pts_crop[0]}, pts_crop[1]={pts_crop[1]}")
        #print(f"[edge_worker] track_local_yx start={track_local_yx[:,0]}, end={track_local_yx[:,-1]}")

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
        #print(f"[edge_worker] e1[0]={track_e1[0]}, e1[-1]={track_e1[-1]}")
        #print(f"[edge_worker] e2[0]={track_e2[0]}, e2[-1]={track_e2[-1]}")
        #print(f"[edge_worker] e1.shape={track_e1.shape}, e2.shape={track_e2.shape}")

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
            metrics = compute_midline_metrics(auto_midline, man_midline, tau=3.0)
            #print(f"[edge_worker] metrics computed: chamfer_mean={metrics.get('chamfer_mean', np.nan):.3f}")

            #metrics = compute_midline_metrics(auto_midline, man_midline, tau=3.0)
            #print(f"[edge_worker] metrics computed: chamfer_mean={metrics.get('chamfer_mean', np.nan):.3f}")
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
'''

# ---------------------------------------------------------------------
# Worker: edge mask → edge tracking → mask creation → midline metrics
# ---------------------------------------------------------------------
def edge_param_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np, cv2, os, json
    from helpers.metrics import compute_midline_metrics, set_tracked_edges_for_crack

    img = payload["image_crop_gray"]
    pts_crop = payload["pts_crop"]
    track_local_yx = payload["adjusted_track"]  # (2,N) [y,x]
    man_xy_g = np.asarray(payload["manual_midline_global"], float)
    x, y, w, h = map(int, payload["bbox"])  # bbox in GLOBAL coords
    P = payload["params"]
    crack_id = payload.get("crack_id", "?")
    base_name = payload.get("image_base", "unknown")

    try:
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        em1, em2 = edge_masks(img_norm, track_local_yx, window_half_size=int(P["window_half_size"]))

        # --- derive crop-local midline (x,y) ---
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

        # convert to numpy
        track_e1 = np.asarray(track_e1, float)
        track_e2 = np.asarray(track_e2, float)

        # --- build local crop mask (for debug) ---
        hc, wc = img.shape[:2]
        mask_crop = _crop_mask_from_edges(hc, wc, track_e1, track_e2, midline_xy=midline_xy_crop)

        # --- NORMALS ---
        normals = res.get("normal_edge_points")
        if normals is not None:
            (e1x, e1y), (e2x, e2y) = normals
            e1x = np.asarray(e1x, float); e1y = np.asarray(e1y, float)
            e2x = np.asarray(e2x, float); e2y = np.asarray(e2y, float)

            # legacy/local format
            normal_edge_points = [
                [e1x.tolist(), e1y.tolist()],
                [e2x.tolist(), e2y.tolist()],
            ]

            # GLOBAL coords (for exports)
            n1_full = np.column_stack([e1x + x, e1y + y])
            n2_full = np.column_stack([e2x + x, e2y + y])
            normal_edge_points_full = {
                "edge1": n1_full.tolist(),
                "edge2": n2_full.tolist(),
            }
        else:
            normal_edge_points = None
            normal_edge_points_full = None

        # --- convert edges to GLOBAL coords ---
        track_e1_global = np.column_stack([track_e1[:, 0] + x, track_e1[:, 1] + y])
        track_e2_global = np.column_stack([track_e2[:, 0] + x, track_e2[:, 1] + y])


        # --- unified visualization: 2 clean overlays ---
        DEBUG_SAVE = True
        if DEBUG_SAVE:
            dbg_dir = os.path.join(payload["save_folder"], "metrics", base_name, f"cid{crack_id}")
            os.makedirs(dbg_dir, exist_ok=True)

            # ========== (1) Geometry overlay ==========
            vis_geom = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)

            # draw edges
            cv2.polylines(vis_geom, [track_e1.astype(np.int32)], False, (0, 255, 0), 1, lineType=8)
            cv2.polylines(vis_geom, [track_e2.astype(np.int32)], False, (0, 255, 0), 1, lineType=8)

            # midline (white)
            try:
                n = min(len(track_e1), len(track_e2))
                auto_midline = 0.5 * (track_e1[:n] + track_e2[:n])
                cv2.polylines(vis_geom, [auto_midline.astype(np.int32)], False, (255, 255, 255), 1, lineType=8)
            except Exception:
                pass

            # --- draw true computed normals from edge_tracking (sampled & filtered) ---
            normals_full = res.get("normal_edge_points_full") or res.get("normal_edge_points")
            if isinstance(normals_full, dict):
                e1 = np.asarray(normals_full.get("edge1", []), float)
                e2 = np.asarray(normals_full.get("edge2", []), float)

                if e1.ndim == 2 and e2.ndim == 2 and len(e1) > 1 and len(e2) > 1:
                    m = min(len(e1), len(e2))
                    e1, e2 = e1[:m], e2[:m]
                    print(f'm = {len(m)}')

                    # subsample so you see only every ~Nth vector
                    step = max(1, m // 50)
                    # optional length sanity clamp
                    lengths = np.linalg.norm(e1 - e2, axis=1)
                    median_len = np.median(lengths[np.isfinite(lengths)])
                    good = (lengths < 3 * median_len) & np.isfinite(lengths)

                    for i in range(0, m, step):
                        if not good[i]:
                            continue
                        p1 = tuple(np.round(e1[i]).astype(int))
                        p2 = tuple(np.round(e2[i]).astype(int))
                        cv2.line(vis_geom, p1, p2, (255, 255, 0), 1, lineType=cv2.LINE_AA)
            else:
                print("There are literally no normals man!")

            import matplotlib.pyplot as plt

            # --- render via matplotlib for high-quality anti-aliased output ---
            out_geom = os.path.join(dbg_dir, "edges_midlines_normals.png")

            plt.figure(figsize=(6, 6), dpi=600)
            plt.imshow(cv2.cvtColor(vis_geom, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.tight_layout(pad=0)
            plt.savefig(out_geom, dpi=600, bbox_inches="tight", pad_inches=0)
            plt.close()

            print(f"[DEBUG VIS] saved high-quality matplotlib render → {out_geom}")

            # ========== (2) GT vs manual mask overlay ==========
            gt_crop = payload.get("gt_crop", None)
            vis_iou = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
            pred_mask = (mask_crop > 0).astype(np.uint8)
            if gt_crop is not None:
                gt_bin = (np.asarray(gt_crop, dtype=np.uint8) > 0).astype(np.uint8)
                intersect = np.logical_and(gt_bin, pred_mask)
                pred_only = np.logical_and(pred_mask, np.logical_not(gt_bin))
                gt_only   = np.logical_and(gt_bin, np.logical_not(pred_mask))

                vis_iou[gt_only]   = (0, 0, 255)      # red GT only
                vis_iou[pred_only] = (0, 255, 255)    # yellow pred only
                vis_iou[intersect] = (255, 255, 255)  # white intersection

            vis_iou_large = cv2.resize(vis_iou, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
            out_iou = os.path.join(dbg_dir, "gt_vs_manual_mask.png")
            cv2.imwrite(out_iou, vis_iou_large)

            print(f"[DEBUG VIS] wrote → {out_geom} and {out_iou}")

        # --- midline comparison metrics (crop-local) ---
        try:
            n = min(len(track_e1), len(track_e2))
            auto_midline = 0.5 * (track_e1[:n] + track_e2[:n])
            man_midline = man_xy_g - np.array([x, y], float)  # shift GT to crop coords
            metrics = compute_midline_metrics(auto_midline, man_midline, tau=3.0)
        except Exception as e:
            print(f"[edge_worker] ⚠️ midline metrics failed: {e}")
            metrics = {k: np.nan for k in [
                "chamfer_mean","hausdorff","angle_err_deg","coverage",
                "directional_bias","curvature_rms_ratio","local_thickness_corr"
            ]}

        # --- compose final payload ---
        result = {
            "status": "ok",
            "bbox": [x, y, w, h],
            "mask_bbox": [x, y, w, h],
            "mask_crop": mask_crop.tolist(),
            "geodesic_edges": {
                "edge1": track_e1_global.tolist(),
                "edge2": track_e2_global.tolist(),
            },
            "normal_edge_points": normal_edge_points,
            "normal_edge_points_full": normal_edge_points_full,
            **P,
            **metrics
        }

        # --- SAVE SNAPSHOT (flat filename) ---
        set_tracked_edges_for_crack(payload["save_folder"], base_name, crack_id, result)

        return result

    except Exception as e:
        print(f"[edge_worker] ❌ unexpected failure for params={P}: {e}")
        _debug_plot_edge_worker(img, np.zeros_like(img), np.zeros_like(img),
                                None, None, None, P, tag="crash")
        out = {"status": "fail_exception", "error": str(e)}
        out.update(P)
        return out

