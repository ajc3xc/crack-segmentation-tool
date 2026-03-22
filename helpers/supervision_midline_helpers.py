import time
import os
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from helpers.metrics import normals_from_mask_for_midline, resample_by_arclength


def _arr_to_list(a):
    if a is None:
        return []
    return np.asarray(a).tolist()


def build_centering_domain_mask(*, crack_mask_u8, territory_u8=None, mode="soft"):
    """
    Build allowed domain for center snapping.
    mode:
      - soft: crack mask only
      - terr_or_mask: crack | territory
      - terr_and_mask: crack & territory
    """
    m = (np.asarray(crack_mask_u8) > 0).astype(np.uint8)
    if territory_u8 is None:
        return m
    t = (np.asarray(territory_u8) > 0).astype(np.uint8)

    if mode == "terr_and_mask":
        return (m & t).astype(np.uint8)
    if mode == "terr_or_mask":
        return (m | t).astype(np.uint8)
    return m


def gaussian(img, sigma):
    return cv2.GaussianBlur(np.asarray(img), (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))


def bilinear_sample(img, x, y):
    arr = np.asarray(img)
    h, w = arr.shape

    x0 = np.clip(np.floor(x).astype(int), 0, w - 1)
    y0 = np.clip(np.floor(y).astype(int), 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    dx = x - x0
    dy = y - y0

    v00 = arr[y0, x0]
    v10 = arr[y0, x1]
    v01 = arr[y1, x0]
    v11 = arr[y1, x1]

    return (
        v00 * (1 - dx) * (1 - dy)
        + v10 * dx * (1 - dy)
        + v01 * (1 - dx) * dy
        + v11 * dx * dy
    )


def compute_dt_gradient(dt, ksize=3):
    k = int(ksize)
    gx = cv2.Sobel(np.asarray(dt, np.float32), cv2.CV_32F, 1, 0, ksize=k)
    gy = cv2.Sobel(np.asarray(dt, np.float32), cv2.CV_32F, 0, 1, ksize=k)
    return gx, gy


def _compute_dt_fixed(domain_u8, *, pad=5, mask_size=3, allow_pad=True):
    dom = (np.asarray(domain_u8) > 0).astype(np.uint8)
    if dom.size == 0:
        return np.zeros((0, 0), np.float32)
    if not np.any(dom):
        return np.zeros(dom.shape[:2], np.float32)
    p = int(max(0, pad))
    use_pad = bool(allow_pad) and p > 0
    if use_pad:
        padded = np.pad(dom, pad_width=p, mode="constant", constant_values=0)
        dt_full = cv2.distanceTransform(
            np.asarray(padded, np.uint8),
            cv2.DIST_L2,
            int(mask_size),
        ).astype(np.float32)
        dt = dt_full[p:-p, p:-p]
    else:
        dt = cv2.distanceTransform(
            np.asarray(dom, np.uint8),
            cv2.DIST_L2,
            int(mask_size),
        ).astype(np.float32)
    return np.asarray(dt, np.float32)


def _apply_edge_unbias(dt_norm, domain_u8, margin=10):
    dt = np.asarray(dt_norm, np.float32)
    dom = (np.asarray(domain_u8) > 0)
    if dt.ndim != 2 or dom.ndim != 2 or dt.shape != dom.shape or not np.any(dom):
        return np.asarray(dt_norm, np.float32)

    H, W = dt.shape[:2]
    y = np.minimum(np.arange(H), np.arange(H)[::-1])
    x = np.minimum(np.arange(W), np.arange(W)[::-1])
    Y, X = np.meshgrid(y, x, indexing="ij")
    edge_dist = np.minimum(Y, X).astype(np.float32)

    m = float(max(1.0, margin))
    edge_proximity = 1.0 - np.clip(edge_dist / m, 0.0, 1.0)

    corrected = dt.copy()
    corrected[dom] = np.maximum(
        dt[dom],
        edge_proximity[dom],
    )
    return np.clip(corrected, 0.0, 1.0).astype(np.float32, copy=False)


def smooth_polyline(
    S,
    *,
    keep_endpoints=True,
    freeze_k=3,
    ds_target=2,
    window=7,
    poly=2,
):
    out = np.asarray(S, float).copy()
    if out.ndim != 2 or out.shape[1] != 2 or len(out) < 2:
        return np.asarray(S, float)

    rs = resample_by_arclength(
        out,
        ds_target=float(ds_target),
        min_pts=2,
        preserve_endpoints=bool(keep_endpoints),
    )
    if isinstance(rs, tuple) and len(rs) >= 1 and rs[0] is not None:
        out = np.asarray(rs[0], float)

    try:
        from scipy.signal import savgol_filter

        n = len(out)
        if n >= 5:
            win = int(window)
            if win > n:
                win = n if (n % 2 == 1) else (n - 1)
            if win < 5:
                win = 5 if n >= 5 else win
            if win % 2 == 0:
                win = max(3, win - 1)
            if win > int(poly):
                xs = savgol_filter(out[:, 0], win, int(poly))
                ys = savgol_filter(out[:, 1], win, int(poly))
                out2 = out.copy()
                out2[:, 0] = xs
                out2[:, 1] = ys
                if keep_endpoints and len(out2) >= 2:
                    fk = int(max(1, freeze_k))
                    fk = min(fk, len(out2) // 2 if len(out2) > 2 else 1)
                    out2[:fk] = out[:fk]
                    out2[-fk:] = out[-fk:]
                out = out2
    except Exception:
        pass

    return np.asarray(out, float)


def _ridge_ascent_polyline(
    S,
    dt,
    gx,
    gy,
    domain,
    *,
    n_iters,
    step_px,
    keep_endpoints,
    freeze_k,
):
    out = np.asarray(S, float).copy()
    H, W = np.asarray(domain).shape[:2]
    dom = (np.asarray(domain) > 0)

    fk = int(max(0, freeze_k))
    idx_lo = fk if keep_endpoints else 0
    idx_hi = len(out) - 1 - fk if keep_endpoints else len(out) - 1

    if idx_hi <= idx_lo:
        return out

    for _ in range(int(max(1, n_iters))):
        pts = out[idx_lo:idx_hi + 1]

        x = pts[:, 0]
        y = pts[:, 1]

        xi = np.round(x).astype(int)
        yi = np.round(y).astype(int)

        valid = (
            (xi >= 0) & (xi < W) &
            (yi >= 0) & (yi < H)
        )
        valid &= dom[np.clip(yi, 0, H - 1), np.clip(xi, 0, W - 1)]
        if not np.any(valid):
            break

        prev = out[idx_lo - 1:idx_hi]
        nxt = out[idx_lo + 1:idx_hi + 2]
        t = nxt - prev
        tn = np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
        t = t / tn
        n = np.stack([-t[:, 1], t[:, 0]], axis=1)

        gxi = bilinear_sample(gx, x, y)
        gyi = bilinear_sample(gy, x, y)
        g_proj = gxi * n[:, 0] + gyi * n[:, 1]

        nonzero = np.abs(g_proj) > 1e-12
        active = valid & nonzero
        if not np.any(active):
            break

        sgn = np.sign(g_proj)
        u = n * sgn[:, None]
        xn = x + step_px * u[:, 0]
        yn = y + step_px * u[:, 1]

        xi2 = np.round(xn).astype(int)
        yi2 = np.round(yn).astype(int)
        valid2 = (
            (xi2 >= 0) & (xi2 < W) &
            (yi2 >= 0) & (yi2 < H)
        )
        valid2 &= dom[np.clip(yi2, 0, H - 1), np.clip(xi2, 0, W - 1)]
        active &= valid2
        if not np.any(active):
            break

        dt0 = bilinear_sample(dt, x, y)
        dt1 = bilinear_sample(dt, xn, yn)
        accept = active & (dt1 >= dt0 - 1e-6)
        if not np.any(accept):
            break

        pts[accept, 0] = xn[accept]
        pts[accept, 1] = yn[accept]
        out[idx_lo:idx_hi + 1] = pts

    return out


def snap_polyline_to_dt_ridge(
    mid_xy,
    domain_mask_u8,
    *,
    n_iters=25,
    step_px=0.35,
    grad_ksize=3,
    keep_endpoints=True,
    freeze_k=3,
    debug=False,
    dt_float=None,
):
    """
    Nudge polyline points toward DT ridge by gradient ascent.
    """
    t0_snap = time.perf_counter()

    S = np.asarray(mid_xy, float).copy()
    if S.ndim != 2 or S.shape[1] != 2 or len(S) < 2:
        print(f"[SNAP DT] elapsed_sec={time.perf_counter() - t0_snap:.6f} (early_return=bad_input)")
        return S

    domain = (np.asarray(domain_mask_u8) > 0).astype(np.uint8)
    if not np.any(domain):
        print(f"[SNAP DT] elapsed_sec={time.perf_counter() - t0_snap:.6f} (early_return=empty_domain)")
        return S

    if dt_float is not None:
        dt = np.asarray(dt_float, np.float32)
        if dt.shape[:2] != domain.shape[:2]:
            dt = _compute_dt_fixed(
                domain,
                pad=DT_BACKGROUND_RING_PX,
                mask_size=3,
                allow_pad=bool(ENABLE_DT_BBOX_PAD),
            ).astype(np.float32)
    else:
        dt = _compute_dt_fixed(
            domain,
            pad=DT_BACKGROUND_RING_PX,
            mask_size=3,
            allow_pad=bool(ENABLE_DT_BBOX_PAD),
        ).astype(np.float32)

    gx, gy = compute_dt_gradient(dt, ksize=int(grad_ksize))

    S = _ridge_ascent_polyline(
        S,
        dt,
        gx,
        gy,
        domain,
        n_iters=n_iters,
        step_px=step_px,
        keep_endpoints=keep_endpoints,
        freeze_k=freeze_k,
    )

    S = smooth_polyline(
        S,
        keep_endpoints=bool(keep_endpoints),
        freeze_k=int(max(0, freeze_k)),
        ds_target=2,
        window=7,
        poly=2,
    )

    print(f"[SNAP DT] elapsed_sec={time.perf_counter() - t0_snap:.6f}")
    return S


def _compute_dt_for_domain(
    domain_u8,
    *,
    full_image_hw=None,
):
    t0 = time.perf_counter()
    domain = (np.asarray(domain_u8) > 0).astype(np.uint8)
    if domain.size == 0:
        z = np.zeros((0, 0), np.float32)
        return z, z, {"compute_s": 0.0, "mode": "empty"}
    if not np.any(domain):
        z = np.zeros(domain.shape[:2], np.float32)
        return z, z, {"compute_s": float(time.perf_counter() - t0), "mode": "empty_domain"}

    dt = cv2.distanceTransform(domain, cv2.DIST_L2, 3).astype(np.float32)

    mx = float(np.max(dt))
    if mx > 1e-12:
        dt_norm = dt / mx
    else:
        dt_norm = np.zeros_like(dt, dtype=np.float32)

    return dt, dt_norm.astype(np.float32, copy=False), {
        "compute_s": float(time.perf_counter() - t0),
        "mode": "direct_local",
    }


def _compute_dt_ridge_midline(mid_xy, domain_u8, dt_float, snap_kwargs):
    t0 = time.perf_counter()
    centered = snap_polyline_to_dt_ridge(
        np.asarray(mid_xy, float),
        (np.asarray(domain_u8) > 0).astype(np.uint8),
        dt_float=dt_float,
        **(snap_kwargs or {}),
    )
    return np.asarray(centered, float), {"snap_s": float(time.perf_counter() - t0)}

def _closest_valid_xy_in_mask(xy, mask_bool):
    if xy is None:
        return None
    m = np.asarray(mask_bool).astype(bool)
    if m.ndim != 2 or not np.any(m):
        return None

    x = int(np.round(float(xy[0])))
    y = int(np.round(float(xy[1])))
    H, W = m.shape[:2]
    if 0 <= x < W and 0 <= y < H and m[y, x]:
        return (x, y)

    ys, xs = np.where(m)
    if xs.size == 0:
        return None
    dx = xs.astype(np.float32) - float(x)
    dy = ys.astype(np.float32) - float(y)
    idx = int(np.argmin(dx * dx + dy * dy))
    return (int(xs[idx]), int(ys[idx]))


def _compute_dijkstra_midline(mid_xy, costmap_f32, domain_u8):
    t0 = time.perf_counter()
    result_meta = {
        "dijkstra_s": 0.0,
        "backend": None,
        "reason": None,
        "path_cost": None,
    }

    S = np.asarray(mid_xy, float)
    if S.ndim != 2 or S.shape[1] != 2 or len(S) < 2:
        result_meta["reason"] = "bad_midline_input"
        return None, result_meta

    dom = (np.asarray(domain_u8) > 0)
    if not np.any(dom):
        result_meta["reason"] = "empty_domain"
        return None, result_meta

    start_xy = _closest_valid_xy_in_mask(S[0], dom)
    end_xy = _closest_valid_xy_in_mask(S[-1], dom)
    if start_xy is None or end_xy is None:
        result_meta["reason"] = "no_valid_endpoints"
        return None, result_meta

    sx, sy = start_xy
    ex, ey = end_xy
    path_xy = None

    try:
        from skimage.graph import route_through_array

        path_rc, total_cost = route_through_array(
            np.asarray(costmap_f32, np.float32),
            start=(int(sy), int(sx)),
            end=(int(ey), int(ex)),
            fully_connected=True,
            geometric=True,
        )
        if path_rc:
            path_xy = np.asarray([[float(c), float(r)] for r, c in path_rc], float)
            result_meta["path_cost"] = float(total_cost)
            result_meta["backend"] = "skimage.route_through_array"
    except Exception as e:
        result_meta["reason"] = f"dijkstra_failed:{type(e).__name__}"

    if path_xy is None or len(path_xy) < 2:
        result_meta["reason"] = result_meta["reason"] or "empty_path"
        result_meta["dijkstra_s"] = float(time.perf_counter() - t0)
        return None, result_meta

    result_meta["dijkstra_s"] = float(time.perf_counter() - t0)
    return path_xy, result_meta


def _postprocess_midline_polyline(mid_xy, *, keep_endpoints=True):
    return smooth_polyline(
        mid_xy,
        keep_endpoints=bool(keep_endpoints),
        freeze_k=1,
        ds_target=2,
        window=7,
        poly=2,
    )


def _refine_path_sobel(path_xy, score_map, *, iterations=4, step=0.35):
    """
    Subpixel ridge refinement for a path using score-map Sobel gradients.
    Endpoints are kept fixed.
    """
    path = np.asarray(path_xy, np.float32).copy()
    score = np.asarray(score_map, np.float32)

    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 3:
        return np.asarray(path_xy, float)
    if score.ndim != 2:
        return np.asarray(path_xy, float)

    gx, gy = compute_dt_gradient(score, ksize=3)
    h, w = score.shape

    for _ in range(max(0, int(iterations))):
        for i in range(1, len(path) - 1):
            x, y = float(path[i, 0]), float(path[i, 1])
            gxi = float(bilinear_sample(gx, x, y))
            gyi = float(bilinear_sample(gy, x, y))
            g = np.array([gxi, gyi], dtype=np.float32)
            n = float(np.linalg.norm(g)) + 1e-6
            path[i] += float(step) * (g / n)
            path[i, 0] = np.clip(path[i, 0], 0.0, float(w - 1))
            path[i, 1] = np.clip(path[i, 1], 0.0, float(h - 1))

    return np.asarray(path, float)


def _compute_normals_for_midline(
    *,
    mid_xy,
    crack_mask_u8,
    max_radius,
    diag_out=None,
    endpoint_mode="atomic",
):
    t0 = time.perf_counter()
    diag = diag_out if isinstance(diag_out, dict) else {}
    (e1x, e1y, e2x, e2y, widths), _ = normals_from_mask_for_midline(
        np.asarray(mid_xy, float),
        (np.asarray(crack_mask_u8) > 0),
        int(max_radius),
        diagnostics=diag,
        image_hw=np.asarray(crack_mask_u8).shape[:2],
        endpoint_mode=endpoint_mode,
    )
    normals = {
        "edge1_x": _arr_to_list(e1x),
        "edge1_y": _arr_to_list(e1y),
        "edge2_x": _arr_to_list(e2x),
        "edge2_y": _arr_to_list(e2y),
        "width_px": _arr_to_list(widths),
    }
    return normals, diag, {"compute_s": float(time.perf_counter() - t0)}


def _log_method_failure(method_key, reason, extra=None):
    msg = f"[MIDLINE FAIL] method={method_key} reason={reason}"
    if isinstance(extra, dict):
        parts = []
        for k, v in extra.items():
            try:
                parts.append(f"{k}={v}")
            except Exception:
                continue
        if parts:
            msg += " | " + " ".join(parts)
    print(msg)


def _extract_depth_crop_for_bbox_or_domain(
    *,
    domain_u8,
    depth_full=None,
    depth_crop=None,
    depth_bbox_xywh=None,
    full_image_hw=None,
):
    dom = (np.asarray(domain_u8) > 0)
    target_h, target_w = dom.shape[:2]
    if target_h <= 0 or target_w <= 0:
        return None, {"reason": "empty_domain"}

    src = None
    source_name = "none"
    if depth_crop is not None:
        src = np.asarray(depth_crop)
        source_name = "depth_crop"
    elif depth_full is not None:
        src = np.asarray(depth_full)
        source_name = "depth_full"
    else:
        return None, {"reason": "missing_depth"}

    if src.ndim == 3:
        if src.shape[2] == 1:
            src = src[:, :, 0]
        else:
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    src = np.squeeze(src)
    if src.ndim != 2:
        return None, {"reason": f"bad_depth_ndim:{src.ndim}"}

    src = src.astype(np.float32, copy=False)
    raw_shape = tuple(int(v) for v in src.shape[:2])
    used_bbox_crop = False

    if depth_bbox_xywh is not None and len(depth_bbox_xywh) == 4:
        x, y, w, h = [int(v) for v in depth_bbox_xywh]
        Hs, Ws = src.shape[:2]
        if (
            isinstance(full_image_hw, (list, tuple))
            and len(full_image_hw) == 2
            and int(full_image_hw[0]) > 0
            and int(full_image_hw[1]) > 0
        ):
            Hf = int(full_image_hw[0])
            Wf = int(full_image_hw[1])
            sy = float(Hs) / float(Hf)
            sx = float(Ws) / float(Wf)
        else:
            sy = 1.0
            sx = 1.0

        x0 = int(np.floor(float(x) * sx))
        y0 = int(np.floor(float(y) * sy))
        x1 = int(np.ceil(float(x + max(0, w)) * sx))
        y1 = int(np.ceil(float(y + max(0, h)) * sy))

        x0 = max(0, min(Ws, x0))
        y0 = max(0, min(Hs, y0))
        x1 = max(0, min(Ws, x1))
        y1 = max(0, min(Hs, y1))

        if x1 > x0 and y1 > y0:
            crop = src[y0:y1, x0:x1]
            if crop.size > 0:
                src = crop
                used_bbox_crop = True

    resized = False
    if src.shape[:2] != (target_h, target_w):
        src = cv2.resize(src, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        resized = True

    return src.astype(np.float32, copy=False), {
        "source": source_name,
        "raw_shape": list(raw_shape),
        "used_bbox_crop": bool(used_bbox_crop),
        "aligned_shape": [int(target_h), int(target_w)],
        "resized": bool(resized),
    }

def _compute_depth_recess_signal(depth_local_f32, domain_u8, dt_norm=None):
    t0 = time.perf_counter()

    dom = (np.asarray(domain_u8) > 0)
    z = np.asarray(depth_local_f32, np.float32)

    if z.ndim != 2 or not np.any(dom):
        z0 = np.zeros_like(z, np.float32)
        return z0, z0, {
            "compute_s": float(time.perf_counter() - t0),
            "reason": "invalid_depth_or_domain",
        }

    # 1) Fill NaNs.
    finite = np.isfinite(z)
    if not np.any(finite):
        z[:] = 0.0
    else:
        fill = float(np.nanmedian(z[finite]))
        z[~finite] = fill

    # 2) Robust normalization (NOT naive).
    lo = np.percentile(z[dom], 5)
    hi = np.percentile(z[dom], 95)
    if hi <= lo + 1e-6:
        z0 = np.zeros_like(z, np.float32)
        return z0, z0, {
            "compute_s": float(time.perf_counter() - t0),
            "reason": "depth_flat",
        }
    z = np.clip((z - lo) / (hi - lo), 0.0, 1.0)

    # 3) Aggressive smoothing to suppress speckle before valley extraction.
    z_s = cv2.GaussianBlur(z, (0, 0), sigmaX=3.0, sigmaY=3.0).astype(np.float32, copy=False)

    # 4) Directional valley likelihood from gradient magnitude.
    gx = cv2.Sobel(z_s, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(z_s, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy).astype(np.float32, copy=False)
    grad_n = _normalize_01_masked(grad_mag, dom)
    recess = (1.0 - grad_n).astype(np.float32, copy=False)

    # 5) Continuity boost and center-ribbon sharpening.
    recess = cv2.GaussianBlur(recess, (0, 0), sigmaX=3.0, sigmaY=3.0).astype(np.float32, copy=False)
    recess = np.clip(recess, 0.0, 1.0) ** 1.5

    # 6) Optional DT gating (if explicitly provided).
    if dt_norm is not None:
        dt = np.asarray(dt_norm, np.float32)
        dt = np.clip(dt, 0.0, 1.0)
        interior = dt ** 1.5
        recess = recess * interior

    # 7) Mask outside.
    recess[~dom] = 0.0
    z_s[~dom] = 0.0

    return z_s.astype(np.float32), recess.astype(np.float32), {
        "compute_s": float(time.perf_counter() - t0),
        "mode": "gradient_valley_dt_gated",
    }
    

def _normalize_01(x):
    x = np.asarray(x, np.float32)
    v = x[np.isfinite(x)]
    if v.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo = float(np.percentile(v, 2))
    hi = float(np.percentile(v, 98))
    if hi <= lo + 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return y.astype(np.float32, copy=False)


def _normalize_01_masked(x, mask_bool):
    x = np.asarray(x, np.float32)
    m = np.asarray(mask_bool).astype(bool)
    out = np.zeros_like(x, dtype=np.float32)
    if x.ndim != 2 or m.ndim != 2 or x.shape != m.shape or not np.any(m):
        return out
    vals = x[m & np.isfinite(x)]
    if vals.size == 0:
        return out
    lo = float(np.percentile(vals, 2))
    hi = float(np.percentile(vals, 98))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-9:
        return out
    out[m] = np.clip((x[m] - lo) / (hi - lo), 0.0, 1.0)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32, copy=False)


def _prepare_gray_image(image_rgb_or_gray, target_hw):
    if image_rgb_or_gray is None:
        return None
    arr = np.asarray(image_rgb_or_gray)
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        else:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        return None
    h, w = target_hw
    if arr.shape[:2] != (h, w):
        arr = cv2.resize(arr, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
    arr = arr.astype(np.float32, copy=False)
    mx = float(np.nanmax(arr)) if arr.size else 0.0
    if mx > 1.5:
        arr = arr / 255.0
    arr[~np.isfinite(arr)] = 0.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)


def _compute_rgb_ridge_valley_cues(image_gray_f32, domain_u8):
    t0 = time.perf_counter()
    dom = (np.asarray(domain_u8) > 0)
    g = np.asarray(image_gray_f32, np.float32)
    z = np.zeros_like(g, dtype=np.float32)
    meta = {
        "compute_s": 0.0,
        "ridge_backend": None,
        "reason": None,
    }
    if g.ndim != 2 or dom.ndim != 2 or g.shape != dom.shape or not np.any(dom):
        meta["reason"] = "bad_gray_or_domain"
        meta["compute_s"] = float(time.perf_counter() - t0)
        return z, z, meta

    gs = cv2.GaussianBlur(g, (0, 0), sigmaX=1.2, sigmaY=1.2).astype(np.float32)

    ridge = None
    try:
        from skimage.filters import frangi
        ridge = frangi(gs, sigmas=(1.0, 2.0, 3.0), black_ridges=True).astype(np.float32, copy=False)
        meta["ridge_backend"] = "skimage.filters.frangi"
    except Exception:
        try:
            from skimage.feature import hessian_matrix, hessian_matrix_eigvals
            h_elems = hessian_matrix(gs, sigma=1.6, order="rc", use_gaussian_derivatives=True)
            eigs = hessian_matrix_eigvals(h_elems)
            l1 = np.asarray(eigs[0], np.float32)
            l2 = np.asarray(eigs[1], np.float32)
            # Dark-line cue: stronger negative principal curvature.
            ridge = np.maximum(0.0, -np.minimum(l1, l2)).astype(np.float32)
            meta["ridge_backend"] = "skimage.feature.hessian_eigvals"
        except Exception:
            lap = cv2.Laplacian(gs, cv2.CV_32F, ksize=3)
            ridge = np.maximum(0.0, -lap).astype(np.float32)
            meta["ridge_backend"] = "cv2.laplacian_fallback"

    gx = cv2.Sobel(gs, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gs, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy).astype(np.float32, copy=False)
    log = cv2.Laplacian(cv2.GaussianBlur(gs, (0, 0), sigmaX=1.0, sigmaY=1.0), cv2.CV_32F, ksize=3)
    valley_core = np.maximum(0.0, -log).astype(np.float32, copy=False)
    valley = (0.6 * _normalize_01_masked(valley_core, dom) + 0.4 * _normalize_01_masked(grad_mag, dom)).astype(np.float32)

    ridge_n = _normalize_01_masked(ridge, dom).astype(np.float32, copy=False)
    valley_n = _normalize_01_masked(valley, dom).astype(np.float32, copy=False)
    ridge_n[~dom] = 0.0
    valley_n[~dom] = 0.0
    meta["compute_s"] = float(time.perf_counter() - t0)
    return ridge_n, valley_n, meta

def _compute_rgb_trench_signal(image_gray_f32, domain_u8):
    t0 = time.perf_counter()

    dom = (np.asarray(domain_u8) > 0)
    g = np.asarray(image_gray_f32, np.float32)

    z = np.zeros_like(g, dtype=np.float32)
    meta = {
        "compute_s": 0.0,
        "reason": None,
        "mode": "valley_structural_fused",
    }

    if g.ndim != 2 or g.shape != dom.shape or not np.any(dom):
        meta["reason"] = "bad_gray_or_domain"
        meta["compute_s"] = float(time.perf_counter() - t0)
        return z, z, z, z, meta

    # --------------------------------------------------
    # 1) STRUCTURAL SMOOTHING (not texture blur)
    # --------------------------------------------------
    g_s = cv2.GaussianBlur(g, (0, 0), sigmaX=2.0, sigmaY=2.0)

    # --------------------------------------------------
    # 2) MULTI-SCALE VALLEY (key fix)
    # --------------------------------------------------
    g_small = cv2.GaussianBlur(g_s, (0, 0), sigmaX=2.0)
    g_large = cv2.GaussianBlur(g_s, (0, 0), sigmaX=8.0)

    valley = np.maximum(0.0, g_large - g_small)
    valley_good = _normalize_01_masked(valley, dom)

    # --------------------------------------------------
    # 3) RIDGE PRIOR (SOFT, not multiplicative)
    # --------------------------------------------------
    ridge_good, _, _ = _compute_rgb_ridge_valley_cues(g_s, dom.astype(np.uint8))
    ridge_good = np.clip(ridge_good, 0.0, 1.0)

    # soften ridge → avoid leopard effect
    ridge_soft = cv2.GaussianBlur(ridge_good, (0, 0), sigmaX=2.0)

    # --------------------------------------------------
    # 4) EDGE AWARE SUPPRESSION (weak)
    # --------------------------------------------------
    gx = cv2.Sobel(g_s, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g_s, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)

    edge_n = _normalize_01_masked(grad, dom)
    edge_suppress = 1.0 - edge_n

    # --------------------------------------------------
    # 5) FUSION (ADDITIVE, not multiplicative)
    # --------------------------------------------------
    rgb_trench = (
        0.6 * valley_good +      # dominant
        0.3 * ridge_soft +      # structure prior
        0.1 * edge_suppress     # weak helper
    )

    rgb_trench = np.clip(rgb_trench, 0.0, 1.0)

    # --------------------------------------------------
    # 6) LIGHT STRUCTURE SMOOTHING (NOT HEAVY BLUR)
    # --------------------------------------------------
    rgb_trench = cv2.GaussianBlur(rgb_trench, (0, 0), sigmaX=1.5)

    rgb_trench = _normalize_01_masked(rgb_trench, dom)

    # mask out
    valley_good[~dom] = 0.0
    ridge_soft[~dom] = 0.0
    edge_suppress[~dom] = 0.0
    rgb_trench[~dom] = 0.0

    meta["compute_s"] = float(time.perf_counter() - t0)
    return rgb_trench, valley_good, ridge_soft, edge_suppress, meta


METHOD_SPECS = {
    "dt": {
        "label": "DT",
        "use_rgb": False,
        "use_depth": False,
    },
    "dt_depth": {
        "label": "DT + Depth",
        "use_rgb": False,
        "use_depth": True,
    },
    "dt_ridge_valley": {
        "label": "DT + Ridge/Valley",
        "use_rgb": True,
        "use_depth": False,
    },
    "dt_ridge_valley_depth": {
        "label": "DT + Ridge/Valley + Depth",
        "use_rgb": True,
        "use_depth": True,
    },
}

ENABLE_DEPTH_PRIOR = True
ENABLE_PATH_REFINE = False
ENABLE_PATH_POSTPROCESS = False
PRINT_DT_PATH_DIAGNOSTICS = True
# Stage-1 parallelism: run method-family variants concurrently per call.
ENABLE_METHOD_PARALLEL = True
METHOD_PARALLEL_MAX_WORKERS = 8
FLAGSHIP_PATH_SMOOTH_SIGMA = 1.5
ENABLE_COSTMAP_SMOOTH = False
COSTMAP_SMOOTH_SIGMA = 2.5
RGB_TRENCH_PRE_SMOOTH_SIGMA = 3.0
RGB_TRENCH_POST_SMOOTH_SIGMA = 2.0
RGB_TRENCH_POWER = 1.5
RGB_COST_WEIGHT = 0.4
ENABLE_DT_BBOX_PAD = True
DT_BACKGROUND_RING_PX = 5
# Keep disabled by default; preferred behavior is pad->DT->crop in _compute_dt_fixed.
ENABLE_DT_EDGE_UNBIAS = False
DT_EDGE_UNBIAS_MARGIN = 10


def _smooth_midline_gaussian(mid_xy, sigma=1.5):
    arr = np.asarray(mid_xy, np.float32).copy()
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 5:
        return np.asarray(mid_xy, float)
    arr[:, 0] = gaussian_filter1d(arr[:, 0], float(sigma))
    arr[:, 1] = gaussian_filter1d(arr[:, 1], float(sigma))
    return np.asarray(arr, float)


def _smooth_costmap_in_domain(costmap_f32, domain_u8, sigma=2.5):
    c = np.asarray(costmap_f32, np.float32)
    dom = (np.asarray(domain_u8) > 0)
    if c.ndim != 2 or dom.ndim != 2 or c.shape != dom.shape or not np.any(dom):
        return np.asarray(costmap_f32, np.float32)
    inf = np.float32(1e9)
    valid = dom & np.isfinite(c) & (c < inf)
    if not np.any(valid):
        return np.asarray(costmap_f32, np.float32)
    v = np.zeros_like(c, np.float32)
    w = np.zeros_like(c, np.float32)
    v[valid] = c[valid]
    w[valid] = 1.0
    vb = cv2.GaussianBlur(v, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma)).astype(np.float32, copy=False)
    wb = cv2.GaussianBlur(w, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma)).astype(np.float32, copy=False)
    out = np.full_like(c, inf, np.float32)
    ok = dom & (wb > 1e-6)
    out[ok] = (vb[ok] / wb[ok]).astype(np.float32)
    return out.astype(np.float32, copy=False)

def _build_ridge_valley_method_costmaps(
    domain_u8,
    dt_norm,
    *,
    method_key,
    rgb_trench_norm=None,
    ridge_norm=None,
    valley_norm=None,
    recess_norm=None,
):
    t0 = time.perf_counter()
    dom = (np.asarray(domain_u8) > 0)
    dtn = np.clip(np.asarray(dt_norm, np.float32), 0.0, 1.0)
    inf = np.float32(1e9)

    dt_bad = np.ones_like(dtn, dtype=np.float32)
    dt_bad[dom] = (1.0 - dtn[dom]).astype(np.float32)
    vals = dt_bad[dom]
    if vals.size:
        print(
            f"[DT COST DBG] method={method_key} "
            f"dt_norm_min={float(dtn[dom].min()):.6f} "
            f"dt_norm_max={float(dtn[dom].max()):.6f} "
            f"dt_bad_min={float(vals.min()):.6f} "
            f"dt_bad_max={float(vals.max()):.6f} "
            f"dt_bad_unique_1e4={int(np.unique(np.round(vals, 4)).size)}"
        )

    costmaps = {
        "dt": np.full_like(dtn, inf, dtype=np.float32),
    }
    costmaps["dt"][dom] = dt_bad[dom]

    debug = {
        "dt_term": dtn.copy(),
        "ridge_term": None,
        "valley_term": None,
        "rgb_cue_term": None,
        "depth_term": None,
    }

    depth_term = None
    if recess_norm is not None:
        dep = np.clip(np.asarray(recess_norm, np.float32), 0.0, 1.0)
        # Weak depth influence to avoid cost collapse on noisy/flat depth.
        depth_term = np.ones_like(dtn, dtype=np.float32)
        depth_term[dom] = (1.0 - 0.25 * dep[dom]).astype(np.float32)
        debug["depth_term"] = dep.copy()
        costmaps["dt_depth"] = np.full_like(dtn, inf, dtype=np.float32)
        costmaps["dt_depth"][dom] = (dt_bad[dom] * depth_term[dom]).astype(np.float32)

    rgb_bad = None
    if rgb_trench_norm is not None:
        trench = np.clip(np.asarray(rgb_trench_norm, np.float32), 0.0, 1.0)
        rgb_bad = np.ones_like(dtn, dtype=np.float32)
        rgb_bad[dom] = (1.0 - float(RGB_COST_WEIGHT) * trench[dom]).astype(np.float32)
        debug["rgb_cue_term"] = trench.copy()

        costmaps["rgb_cue"] = np.full_like(dtn, inf, dtype=np.float32)
        costmaps["rgb_cue"][dom] = rgb_bad[dom]

        costmaps["dt_ridge_valley"] = np.full_like(dtn, inf, dtype=np.float32)
        costmaps["dt_ridge_valley"][dom] = (dt_bad[dom] * rgb_bad[dom]).astype(np.float32)

        if depth_term is not None:
            costmaps["dt_ridge_valley_depth"] = np.full_like(dtn, inf, dtype=np.float32)
            costmaps["dt_ridge_valley_depth"][dom] = (
                dt_bad[dom] * rgb_bad[dom] * depth_term[dom]
            ).astype(np.float32)
    elif ridge_norm is not None and valley_norm is not None:
        ridge = np.clip(np.asarray(ridge_norm, np.float32), 0.0, 1.0)
        valley = np.clip(np.asarray(valley_norm, np.float32), 0.0, 1.0)
        ridge_bad = np.ones_like(dtn, dtype=np.float32)
        valley_bad = np.ones_like(dtn, dtype=np.float32)
        ridge_bad[dom] = (1.0 - ridge[dom]).astype(np.float32)
        valley_bad[dom] = (1.0 - valley[dom]).astype(np.float32)
        rgb_bad = np.sqrt(np.maximum(ridge_bad * valley_bad, np.float32(1e-9))).astype(np.float32)
        debug["ridge_term"] = ridge.copy()
        debug["valley_term"] = valley.copy()
        rgb_good = (1.0 - np.clip(rgb_bad, 0.0, 1.0)).astype(np.float32)
        rgb_good[~dom] = 0.0
        debug["rgb_cue_term"] = rgb_good

        costmaps["rgb_cue"] = np.full_like(dtn, inf, dtype=np.float32)
        costmaps["rgb_cue"][dom] = rgb_bad[dom]

        costmaps["dt_ridge_valley"] = np.full_like(dtn, inf, dtype=np.float32)
        costmaps["dt_ridge_valley"][dom] = (dt_bad[dom] * rgb_bad[dom]).astype(np.float32)

        if depth_term is not None:
            costmaps["dt_ridge_valley_depth"] = np.full_like(dtn, inf, dtype=np.float32)
            costmaps["dt_ridge_valley_depth"][dom] = (
                dt_bad[dom] * rgb_bad[dom] * depth_term[dom]
            ).astype(np.float32)

    selected_key = str(method_key)
    if selected_key not in costmaps:
        selected_key = "dt"
    costmaps["selected"] = np.asarray(costmaps[selected_key], np.float32)
    costmaps["selected_key"] = str(selected_key)
    return costmaps, debug, {
        "compute_s": float(time.perf_counter() - t0),
        "selected_cost_key": str(selected_key),
        "used_depth": bool(depth_term is not None),
        "used_rgb": bool(rgb_bad is not None),
        "depth_scale": 0.25,
        "rgb_weight": float(RGB_COST_WEIGHT),
    }


def _run_single_midline_method(
    *,
    method_key,
    method_spec,
    mid_xy,
    mask_u8,
    domain_u8,
    dt_norm,
    dt_float,
    image_rgb=None,
    depth_full=None,
    depth_crop=None,
    depth_bbox_xywh=None,
    full_image_hw=None,
    max_radius=60,
    endpoint_mode="atomic",
    dt_compute_s=0.0,
    w_dt=1.0,
    w_geo=1.0,
    w_depth=1.0,
    eps=1e-3,
):
    t0_method = time.perf_counter()
    dom = (np.asarray(domain_u8) > 0).astype(np.uint8)
    dtn = np.asarray(dt_norm, np.float32)
    mid = np.asarray(mid_xy, float)

    timing = {
        "dt_compute_s": float(dt_compute_s),
        "depth_align_s": 0.0,
        "depth_recess_s": 0.0,
        "costmap_s": 0.0,
        "dijkstra_s": 0.0,
        "refine_s": 0.0,
        "postprocess_s": 0.0,
        "normals_s": 0.0,
        "total_s": 0.0,
    }
    out = {
        "midline": None,
        "normals": None,
        "normals_diag": {},
        "timing": timing,
        "debug": {
            "domain_u8": (np.asarray(dom) > 0).astype(np.uint8),
            "dt_norm": np.asarray(dtn, np.float32),
            "depth_norm": None,
            "recess_norm": None,
            "ridge_norm": None,
            "valley_norm": None,
            "rgb_cue_norm": None,
            "edge_suppress_norm": None,
            "dt_term": None,
            "depth_term": None,
            "costmap": None,
            "selected_cost_key": None,
            "score_for_refine": None,
        },
        "meta": {
            "method_key": str(method_key),
            "label": str(method_spec.get("label", method_key)),
            "use_rgb": bool(method_spec.get("use_rgb", False)),
            "use_depth": bool(method_spec.get("use_depth", False)),
            "reason": None,
        },
    }

    if dom.ndim != 2 or not np.any(dom):
        out["meta"]["reason"] = "bad_domain"
        _log_method_failure(method_key, "bad_domain", {
            "domain_shape": tuple(np.asarray(dom).shape),
            "nonzero": int(np.count_nonzero(dom)),
        })
        timing["total_s"] = float(time.perf_counter() - t0_method)
        return out
    if mid.ndim != 2 or mid.shape[1] != 2 or len(mid) < 2:
        out["meta"]["reason"] = "bad_midline_input"
        _log_method_failure(method_key, "bad_midline_input", {
            "mid_shape": tuple(np.asarray(mid).shape),
        })
        timing["total_s"] = float(time.perf_counter() - t0_method)
        return out

    dom_b = (np.asarray(dom) > 0)
    use_depth = bool(method_spec.get("use_depth", False))
    if use_depth and not bool(ENABLE_DEPTH_PRIOR):
        out["meta"]["depth_disabled"] = True
        use_depth = False

    use_rgb = bool(method_spec.get("use_rgb", False))
    depth_local = None
    depth_align_meta = {}
    if use_depth:
        t_align0 = time.perf_counter()
        depth_local, depth_align_meta = _extract_depth_crop_for_bbox_or_domain(
            domain_u8=dom,
            depth_full=depth_full,
            depth_crop=depth_crop,
            depth_bbox_xywh=depth_bbox_xywh,
            full_image_hw=full_image_hw,
        )
        timing["depth_align_s"] = float(time.perf_counter() - t_align0)
        if depth_local is None:
            out["meta"]["reason"] = (depth_align_meta or {}).get("reason", "depth_align_failed")
            _log_method_failure(method_key, "depth_align_failed", depth_align_meta if isinstance(depth_align_meta, dict) else {})
            if str(method_key) in ("dt_depth", "dt_ridge_valley_depth"):
                raise RuntimeError(f"[CRITICAL] Depth required but failed for {method_key}")
            timing["total_s"] = float(time.perf_counter() - t0_method)
            return out
        print(
            f"[DEPTH OK] method={method_key} "
            f"min={float(np.nanmin(depth_local)):.4f} "
            f"max={float(np.nanmax(depth_local)):.4f} "
            f"mean={float(np.nanmean(depth_local)):.4f}"
        )

    recess_norm = None
    depth_norm = None
    depth_sig_meta = {}
    if use_depth:
        depth_norm, recess_norm, depth_sig_meta = _compute_depth_recess_signal(
            depth_local,
            dom,
            dt_norm=None,
        )
        timing["depth_recess_s"] = float(depth_sig_meta.get("compute_s", 0.0))
        out["debug"]["depth_norm"] = np.asarray(depth_norm, np.float32)
        out["debug"]["recess_norm"] = np.asarray(recess_norm, np.float32)

    ridge_norm = None
    valley_norm = None
    if use_rgb:
        gray = _prepare_gray_image(image_rgb, dom.shape[:2])
        if gray is None:
            out["meta"]["reason"] = "missing_rgb_image"
            _log_method_failure(method_key, "missing_rgb_image")
            timing["total_s"] = float(time.perf_counter() - t0_method)
            return out
        rgb_trench_norm, valley_norm, ridge_norm, edge_suppress_norm, _rgb_meta = _compute_rgb_trench_signal(gray, dom)
        out["debug"]["ridge_norm"] = np.asarray(ridge_norm, np.float32)
        out["debug"]["valley_norm"] = np.asarray(valley_norm, np.float32)
        out["debug"]["rgb_cue_norm"] = np.asarray(rgb_trench_norm, np.float32)
        out["debug"]["edge_suppress_norm"] = np.asarray(edge_suppress_norm, np.float32)

    costmaps, score_debug, cost_meta = _build_ridge_valley_method_costmaps(
        dom,
        dtn,
        method_key=str(method_key),
        rgb_trench_norm=(out["debug"]["rgb_cue_norm"] if use_rgb else None),
        ridge_norm=ridge_norm,
        valley_norm=valley_norm,
        recess_norm=recess_norm,
    )
    timing["costmap_s"] = float(cost_meta.get("compute_s", 0.0))
    selected_key = str(cost_meta.get("selected_cost_key", "dt"))
    costmap = np.asarray(costmaps.get(selected_key), np.float32)
    if costmap is None or costmap.size == 0:
        out["meta"]["reason"] = "missing_selected_costmap"
        _log_method_failure(method_key, "missing_selected_costmap", {"selected_key": str(selected_key)})
        timing["total_s"] = float(time.perf_counter() - t0_method)
        return out
    if bool(ENABLE_COSTMAP_SMOOTH):
        costmap = _smooth_costmap_in_domain(
            costmap,
            dom,
            sigma=float(COSTMAP_SMOOTH_SIGMA),
        )
        if isinstance(costmaps, dict):
            costmaps["selected"] = np.asarray(costmap, np.float32)

    route_domain = np.asarray(dom, np.uint8).copy()

    route_costmap = np.asarray(costmap, np.float32)
    path_raw, dijkstra_meta = _compute_dijkstra_midline(mid, route_costmap, route_domain)
    timing["dijkstra_s"] = float(dijkstra_meta.get("dijkstra_s", 0.0))
    if path_raw is None or len(path_raw) < 2:
        out["meta"]["reason"] = "empty_path"
        out["meta"]["dijkstra"] = dijkstra_meta
        out["debug"]["dt_term"] = score_debug.get("dt_term")
        out["debug"]["depth_term"] = score_debug.get("depth_term")
        out["debug"]["costmap"] = np.asarray(route_costmap, np.float32)
        _log_method_failure(method_key, "empty_path", {"dijkstra_meta": dijkstra_meta})
        timing["total_s"] = float(time.perf_counter() - t0_method)
        return out

    if bool(PRINT_DT_PATH_DIAGNOSTICS):
        raw_np = np.asarray(path_raw, np.float32)
        uniq_last10 = int(np.unique(np.round(raw_np[-10:], 3), axis=0).shape[0]) if len(raw_np) >= 10 else int(len(raw_np))
        print(
            f"[DT RAW] method={method_key} npts={len(raw_np)} "
            f"first={raw_np[:1].tolist()} last={raw_np[-1:].tolist()} "
            f"unique_last10={uniq_last10}"
        )

    path_refined = np.asarray(path_raw, float)
    score_for_refine = None
    if bool(ENABLE_PATH_REFINE):
        t_ref0 = time.perf_counter()
        score_for_refine = np.asarray(dtn, np.float32)
        if recess_norm is not None:
            score_for_refine = (
                0.5 * np.asarray(dtn, np.float32)
                + 0.5 * np.asarray(recess_norm, np.float32)
            ).astype(np.float32)
        score_for_refine[dom_b] = np.clip(score_for_refine[dom_b], 0.0, 1.0)
        score_for_refine[~dom_b] = 0.0
        score_ref_smooth = np.asarray(score_for_refine, np.float32)  # no smoothing
        path_refined = _refine_path_sobel(
            path_raw,
            score_ref_smooth,
            iterations=2,
            step=0.15,
        )
        timing["refine_s"] = float(time.perf_counter() - t_ref0)

    path_post = np.asarray(path_refined, float)
    if bool(ENABLE_PATH_POSTPROCESS):
        t_post0 = time.perf_counter()
        path_post = _postprocess_midline_polyline(path_refined, keep_endpoints=True)
        timing["postprocess_s"] = float(time.perf_counter() - t_post0)
    if use_rgb and path_post is not None and len(path_post) >= 5:
        path_post = _smooth_midline_gaussian(path_post, sigma=float(FLAGSHIP_PATH_SMOOTH_SIGMA))
    if path_post is None:
        out["meta"]["reason"] = "empty_path"
        _log_method_failure(method_key, "empty_path_postprocess")
        timing["total_s"] = float(time.perf_counter() - t0_method)
        return out

    normals_diag = {}
    normals, normals_diag, t_normals = _compute_normals_for_midline(
        mid_xy=path_post,
        crack_mask_u8=mask_u8,
        max_radius=max_radius,
        diag_out=normals_diag,
        endpoint_mode=endpoint_mode,
    )
    timing["normals_s"] = float(t_normals.get("compute_s", 0.0))

    out["midline"] = np.asarray(path_post, float)
    out["normals"] = normals
    out["normals_diag"] = normals_diag
    out["debug"]["dt_term"] = score_debug.get("dt_term")
    out["debug"]["depth_term"] = score_debug.get("depth_term")
    out["debug"]["ridge_norm"] = score_debug.get("ridge_term")
    out["debug"]["valley_norm"] = score_debug.get("valley_term")
    out["debug"]["rgb_cue_norm"] = score_debug.get("rgb_cue_term")
    out["debug"]["costmap"] = np.asarray(route_costmap, np.float32)
    out["debug"]["selected_cost_key"] = str(selected_key)
    out["debug"]["costmaps"] = {}
    if isinstance(costmaps, dict):
        for k, v in costmaps.items():
            if isinstance(v, (str, bytes)):
                continue
            try:
                arr = np.asarray(v, np.float32)
            except Exception:
                continue
            if arr.ndim >= 2:
                out["debug"]["costmaps"][str(k)] = arr
    if isinstance(costmaps, dict) and "selected_key" in costmaps:
        out["debug"]["costmaps"]["selected_key"] = str(costmaps.get("selected_key"))
    out["debug"]["score_for_refine"] = np.asarray(score_for_refine, np.float32) if score_for_refine is not None else None
    out["meta"]["reason"] = None
    out["meta"]["depth_align"] = depth_align_meta if isinstance(depth_align_meta, dict) else {}
    out["meta"]["depth_recess"] = depth_sig_meta if isinstance(depth_sig_meta, dict) else {}
    out["meta"]["costmap"] = cost_meta if isinstance(cost_meta, dict) else {}
    out["meta"]["dijkstra"] = dijkstra_meta if isinstance(dijkstra_meta, dict) else {}
    out["meta"]["route_domain"] = {
        "domain_nonzero": int(np.count_nonzero(dom)),
        "route_domain_nonzero": int(np.count_nonzero(route_domain)),
    }
    timing["total_s"] = float(time.perf_counter() - t0_method)
    print(
        f"[MIDLINE OK] method={method_key} "
        f"points={len(path_post)} "
        f"cost={selected_key}"
    )
    return out


def compute_midline_method_variants_and_normals(
    *,
    mid_xy,
    crack_mask_u8,
    domain_u8=None,
    image_rgb=None,
    depth_full=None,
    depth_crop=None,
    depth_bbox_xywh=None,
    full_image_hw=None,
    max_radius=60,
    snap_kwargs=None,
    depth_alpha=0.5,
    depth_beta=0.5,
    depth_eps=1e-3,
    w_dt=1.0,
    w_geo=None,
    w_depth=None,
    diag_out=None,
    endpoint_mode="atomic",
):
    """
    Compute the 4-method thesis family in one pass:
      dt
      dt_depth
      dt_ridge_valley
      dt_ridge_valley_depth
    """
    if snap_kwargs is None:
        snap_kwargs = {}

    mask_u8 = (np.asarray(crack_mask_u8) > 0).astype(np.uint8)
    # DE NOVO DOMAIN: no territory prior / no manual-midline influence.
    if domain_u8 is None:
        domain = mask_u8.copy()
    else:
        domain = (np.asarray(domain_u8) > 0).astype(np.uint8)
    if not np.any(domain):
        domain = mask_u8.copy()

    dt_float, dt_norm, t_dt = _compute_dt_for_domain(
        domain,
        full_image_hw=full_image_hw,
    )
    methods = {}
    method_items = list(METHOD_SPECS.items())

    def _run_one_method(mkey, mspec):
        d_crop = depth_crop if bool(mspec.get("use_depth", False)) else None
        d_full = depth_full if bool(mspec.get("use_depth", False)) else None

        return _run_single_midline_method(
            method_key=mkey,
            method_spec=mspec,
            mid_xy=np.asarray(mid_xy, float),
            mask_u8=mask_u8,
            domain_u8=domain,
            dt_norm=dt_norm,
            dt_float=dt_float,
            image_rgb=image_rgb,
            depth_full=d_full,
            depth_crop=d_crop,
            depth_bbox_xywh=depth_bbox_xywh,
            full_image_hw=full_image_hw,
            max_radius=max_radius,
            endpoint_mode=endpoint_mode,
            dt_compute_s=float(t_dt.get("compute_s", 0.0)),
            w_dt=float(w_dt),
            w_geo=float(depth_alpha if w_geo is None else w_geo),
            w_depth=float(depth_beta if w_depth is None else w_depth),
            eps=float(depth_eps),
        )

    use_parallel = bool(ENABLE_METHOD_PARALLEL) and len(method_items) > 1
    if use_parallel:
        max_workers = int(max(1, min(
            len(method_items),
            int(METHOD_PARALLEL_MAX_WORKERS),
            int(os.cpu_count() or 1),
        )))
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut_map = {
                ex.submit(_run_one_method, mkey, mspec): str(mkey)
                for mkey, mspec in method_items
            }
            for fut in as_completed(fut_map):
                mkey = fut_map[fut]
                results[mkey] = fut.result()
        methods = {str(k): results.get(str(k), {}) for k in METHOD_SPECS.keys()}
    else:
        for mkey, mspec in method_items:
            methods[str(mkey)] = _run_one_method(mkey, mspec)

    for mkey in METHOD_SPECS.keys():
        res = methods.get(str(mkey), {}) if isinstance(methods.get(str(mkey), {}), dict) else {}
        if res.get("midline") is None:
            print(f"[METHOD DROPPED] {mkey} -> {(res.get('meta', {}) or {}).get('reason')}")
        #else:
            #print(f"[METHOD ADDED] {mkey} OK")

    result = {
        "methods": methods,
        "shared": {
            "domain_u8": (np.asarray(domain) > 0).astype(np.uint8),
            "dt_float": np.asarray(dt_float, np.float32),
            "dt_norm": np.asarray(dt_norm, np.float32),
        },
    }
    return result


def compute_midline_methods_and_normals(
    *,
    mid_xy,
    crack_mask_u8,
    domain_u8=None,
    image_rgb=None,
    depth_full=None,
    depth_crop=None,
    depth_bbox_xywh=None,
    full_image_hw=None,
    max_radius=60,
    snap_kwargs=None,
    depth_alpha=0.5,
    depth_beta=0.5,
    depth_eps=1e-3,
    diag_out=None,
    endpoint_mode="atomic",
):
    """
    Wrapper that returns centered/depth picks plus full method family.
    Centered output is DT-ridge snapped midline (manual-annotation centering path).
    """
    res = compute_midline_method_variants_and_normals(
        mid_xy=mid_xy,
        crack_mask_u8=crack_mask_u8,
        domain_u8=domain_u8,
        image_rgb=image_rgb,
        depth_full=depth_full,
        depth_crop=depth_crop,
        depth_bbox_xywh=depth_bbox_xywh,
        full_image_hw=full_image_hw,
        max_radius=max_radius,
        snap_kwargs=snap_kwargs,
        depth_alpha=depth_alpha,
        depth_beta=depth_beta,
        depth_eps=depth_eps,
        diag_out=diag_out,
        endpoint_mode=endpoint_mode,
    )

    methods = res.get("methods", {}) if isinstance(res, dict) else {}
    m_dt = methods.get("dt", {}) if isinstance(methods.get("dt", {}), dict) else {}
    m_depth = methods.get("dt_ridge_valley_depth", {}) if isinstance(methods.get("dt_ridge_valley_depth", {}), dict) else {}
    if m_depth.get("midline") is None:
        m_depth = methods.get("dt_depth", {}) if isinstance(methods.get("dt_depth", {}), dict) else {}
    shared = res.get("shared", {}) if isinstance(res.get("shared", {}), dict) else {}

    # -------------------------------------------------
    # True "centered" path: DT ridge snapping
    # -------------------------------------------------
    centered_midline = None
    centered_normals = None
    centered_normals_diag = {}
    centered_snap_s = 0.0
    centered_normals_s = 0.0
    try:
        domain_u8_used = shared.get("domain_u8", None)
        dt_float_used = shared.get("dt_float", None)
        if domain_u8_used is not None and dt_float_used is not None:
            centered_midline, t_centered = _compute_dt_ridge_midline(
                np.asarray(mid_xy, float),
                np.asarray(domain_u8_used, np.uint8),
                np.asarray(dt_float_used, np.float32),
                snap_kwargs or {},
            )
            centered_snap_s = float((t_centered or {}).get("snap_s", 0.0))
            centered_normals, centered_normals_diag, t_center_normals = _compute_normals_for_midline(
                mid_xy=np.asarray(centered_midline, float),
                crack_mask_u8=(np.asarray(crack_mask_u8) > 0).astype(np.uint8),
                max_radius=max_radius,
                diag_out=centered_normals_diag,
                endpoint_mode=endpoint_mode,
            )
            centered_normals_s = float((t_center_normals or {}).get("compute_s", 0.0))
    except Exception:
        centered_midline = None
        centered_normals = None
        centered_normals_diag = {}
        centered_snap_s = 0.0
        centered_normals_s = 0.0

    # Fallback to method 1 if DT-ridge centering fails
    if centered_midline is None:
        centered_midline = m_dt.get("midline")
    if centered_normals is None:
        centered_normals = m_dt.get("normals")
    if not centered_normals_diag:
        centered_normals_diag = m_dt.get("normals_diag", {}) or {}
    if centered_snap_s <= 0.0:
        centered_snap_s = float((m_dt.get("timing", {}) or {}).get("dijkstra_s", 0.0))
    if centered_normals_s <= 0.0:
        centered_normals_s = float((m_dt.get("timing", {}) or {}).get("normals_s", 0.0))

    out = {
        "centered_midline": centered_midline,
        "centered_normals": centered_normals,
        "depth_midline": m_depth.get("midline"),
        "depth_normals": m_depth.get("normals"),
        "centered_normals_diag": centered_normals_diag,
        "depth_normals_diag": m_depth.get("normals_diag", {}) or {},
        "depth_cost_meta": m_depth.get("meta", {}) or {},
        "debug": {
            "domain_u8": shared.get("domain_u8"),
            "dt_norm": shared.get("dt_norm"),
            "depth_norm": m_depth.get("debug", {}).get("depth_norm"),
            "recess_norm": m_depth.get("debug", {}).get("recess_norm"),
            "depth_score": m_depth.get("debug", {}).get("score_for_refine"),
            "depth_costmap": m_depth.get("debug", {}).get("costmap"),
        },
        "timing": {
            "dt": {
                "compute_s": float((m_dt.get("timing", {}) or {}).get("dt_compute_s", 0.0)),
            },
            "centered": {
                "snap_s": float(centered_snap_s),
            },
            "depth": {
                "depth_align_s": float((m_depth.get("timing", {}) or {}).get("depth_align_s", 0.0)),
                "recess_s": float((m_depth.get("timing", {}) or {}).get("depth_recess_s", 0.0)),
                "costmap_s": float((m_depth.get("timing", {}) or {}).get("costmap_s", 0.0)),
                "dijkstra_s": float((m_depth.get("timing", {}) or {}).get("dijkstra_s", 0.0)),
                "refine_s": float((m_depth.get("timing", {}) or {}).get("refine_s", 0.0)),
                "postprocess_s": float((m_depth.get("timing", {}) or {}).get("postprocess_s", 0.0)),
            },
            "normals": {
                "centered_s": float(centered_normals_s),
                "depth_s": float((m_depth.get("timing", {}) or {}).get("normals_s", 0.0)),
            },
        },
        "methods": methods,
        "shared": shared,
    }
    return out


def compute_centered_midline_and_normals(
    *,
    mid_xy,
    crack_mask_u8,
    territory_u8=None,
    domain_u8=None,
    image_rgb=None,
    depth_full=None,
    depth_crop=None,
    depth_bbox_xywh=None,
    full_image_hw=None,
    max_radius=60,
    domain_mode="terr_and_mask",
    snap_kwargs=None,
    depth_alpha=0.5,
    depth_beta=0.5,
    depth_eps=1e-3,
    diag_out=None,
    endpoint_mode="atomic",
):
    """
    Backward-compatible alias.
    """
    return compute_midline_methods_and_normals(
        mid_xy=mid_xy,
        crack_mask_u8=crack_mask_u8,
        domain_u8=domain_u8,
        image_rgb=image_rgb,
        depth_full=depth_full,
        depth_crop=depth_crop,
        depth_bbox_xywh=depth_bbox_xywh,
        full_image_hw=full_image_hw,
        max_radius=max_radius,
        snap_kwargs=snap_kwargs,
        depth_alpha=depth_alpha,
        depth_beta=depth_beta,
        depth_eps=depth_eps,
        diag_out=diag_out,
        endpoint_mode=endpoint_mode,
    )

