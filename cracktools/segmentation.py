import scipy
import numpy as np
import cracktools.tracking
import cv2
from skimage import measure
import matplotlib
matplotlib.use("Agg")   # fastest non-GUI backend for PNG/PDF/SVG export
matplotlib.rcParams.update({
    "figure.max_open_warning": 0,
    "text.kerning_factor": 0,
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
})

import matplotlib.pyplot as plt
plt.ioff()   # no interactive figure updates

from agd import Eikonal
from agd.Metrics import Riemann
from agd.Plotting import savefig, quiver; #savefig.dirName = 'Figures/Riemannian'
from agd import LinearParallel as lp
from agd import AutomaticDifferentiation as ad
norm_infinity = ad.Optimization.norm_infinity
import scipy.ndimage

import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered", category=RuntimeWarning)
  
def edge_masks(
    image_gray,
    track,
    window_half_size=45,
    mode="old",   # "new" (GPU hybrid) or "old" (akomp22)
):
    """
    Unified edge mask extractor.

    mode="new" → your CURRENT GPU/CPU hybrid implementation
                 (fully working, used for actual experiments)

    mode="old" → faithful port of the original akomp22 version
                 (keeps the x/y quirks, mainly for ablation)
    """
    import numpy as np
    import scipy.ndimage as ndi
    mode = "new"
    track = np.asarray(track, dtype=float)
    if track.shape[0] != 2:
        raise ValueError(f"Track must be (2, N), got {track.shape}")
    H, W = image_gray.shape
    track[0] = np.clip(track[0], 0, H - 1)
    track[1] = np.clip(track[1], 0, W - 1)

    # ------------------------------------------------------------------
    # =============== MODE: NEW (YOUR GPU HYBRID) ======================
    # ------------------------------------------------------------------
    if str(mode).lower() == "new":
        """
        This branch is IDENTICAL to your working GPU hybrid version.
        """
        # Try to import CuPy and test for an actual GPU
        try:
            import cupy as cp, cupyx.scipy.ndimage as cndi
            try:
                n_devices = cp.cuda.runtime.getDeviceCount()
                use_gpu = n_devices > 0
            except Exception:
                use_gpu = False
        except Exception:
            cp, cndi, use_gpu = np, ndi, False
            
        use_gpu = False

        if not use_gpu:
            print("[edge_mask] ⚙️ running in CPU mode (no CUDA device detected)")

        img_h, img_w = image_gray.shape
        edge_mask = np.zeros_like(image_gray, dtype=float)
        center_line_length = 3
        n_skipped = 0

        for i in range(track.shape[1] - 1):
            start_y = float(track[0, i])
            start_x = float(track[1, i])
            if i < track.shape[1] - center_line_length:
                end_y = float(track[0, i + center_line_length])
                end_x = float(track[1, i + center_line_length])
                flip = False
            else:
                end_y = float(track[0, i - center_line_length])
                end_x = float(track[1, i - center_line_length])
                flip = True

            if start_y == end_y and start_x == end_x:
                n_skipped += 1
                continue

            try:
                from cracktools.tracking import tang_len
                ddx, ddy, _ = tang_len(start_x, start_y, end_x, end_y)
            except Exception:
                ddy = end_y - start_y
                ddx = end_x - start_x
            if flip:
                ddx, ddy = -ddx, -ddy

            angle = np.degrees(np.arctan2(ddx, ddy))
            y0 = max(0, int(start_y - window_half_size))
            y1 = min(img_h, int(start_y + window_half_size))
            x0 = max(0, int(start_x - window_half_size))
            x1 = min(img_w, int(start_x + window_half_size))
            if (y1 - y0) < 5 or (x1 - x0) < 5:
                continue
            window = image_gray[y0:y1, x0:x1]

            try:
                patch = window.astype(float) / 255.0
                if use_gpu:
                    patch = cp.asarray(patch)
                    grad_y = cndi.gaussian_filter(patch, sigma=1, order=(1, 0), mode='reflect')
                    grad_x = cndi.gaussian_filter(patch, sigma=1, order=(0, 1), mode='reflect')
                    grad_y, grad_x = cp.asnumpy(grad_y), cp.asnumpy(grad_x)
                else:
                    grad_y = ndi.gaussian_filter(patch, sigma=1, order=(1, 0), mode='reflect')
                    grad_x = ndi.gaussian_filter(patch, sigma=1, order=(0, 1), mode='reflect')

                projected = grad_x * (-ddy) + grad_y * ddx
                m = max(1, int(min(window.shape[0], window.shape[1]) / 5))
                projected[:m, :] = projected[-m:, :] = 0
                projected[:, :m] = projected[:, -m:] = 0
                edge_window = edge_mask[y0:y1, x0:x1]
                edge_mask[y0:y1, x0:x1] = edge_window + projected
            except Exception as e:
                print(f"[edge_mask-new] Failed at i={i}: {e}")
                continue

        print(f"[edge_mask] (mode=new) skipped {n_skipped} zero-length segments")
        edge_mask1 = edge_mask - np.min(edge_mask)
        edge_mask2 = -edge_mask1 - np.min(-edge_mask1)
        print(f"[edge_mask] (mode=new) done (GPU per-patch={'yes' if use_gpu else 'no'})")
        return edge_mask1, edge_mask2

    # ------------------------------------------------------------------
    # =============== MODE: OLD (AKOMP22 ORIGINAL) =====================
    # ------------------------------------------------------------------
    # NOTE: this intentionally keeps the old x/y usage and window logic.
    mode = "old"
    edge_mask = np.zeros_like(image_gray, dtype=float)
    center_line_length = 3
    n_skipped = 0
    H, W = image_gray.shape

    for i in range(track.shape[1] - 1):
        start_y = track[0, i]
        start_x = track[1, i]
        a = False

        if i < track.shape[1] - center_line_length:
            end_x = track[1, i + center_line_length]
            end_y = track[0, i + center_line_length]
        else:
            a = True
            end_x = track[1, i - center_line_length]
            end_y = track[0, i - center_line_length]

        if start_x == end_x and start_y == end_y:
            n_skipped += 1
            continue

        try:
            from cracktools.tracking import tang_len
            ddx, ddy, _ = tang_len(start_x, start_y, end_x, end_y)
        except Exception:
            ddy = end_y - start_y
            ddx = end_x - start_x

        if a:
            ddx = -ddx
            ddy = -ddy

        angle_deg = np.arctan2(ddx, ddy) * 57.3

        y0 = max(0, int(start_y - window_half_size))
        y1 = min(H, int(start_y + window_half_size))
        x0 = max(0, int(start_x - window_half_size))
        x1 = min(W, int(start_x + window_half_size))
        if (y1 - y0) < 5 or (x1 - x0) < 5:
            continue
        window = image_gray[y0:y1, x0:x1]

        win_rot = ndi.rotate(window, angle_deg, reshape=False)
        sobel = ndi.gaussian_filter(
            win_rot / 255.0,
            1,
            order=(1, 0),
            mode='reflect',
            cval=0.0,
            truncate=4.0,
        )
        sobel_rot = ndi.rotate(sobel, -angle_deg, reshape=False)

        m = max(1, int(min(window.shape[0], window.shape[1]) / 5))
        sobel_rot[:m, :] = 0
        sobel_rot[-m:, :] = 0
        sobel_rot[:, :m] = 0
        sobel_rot[:, -m:] = 0

        edge_window = edge_mask[y0:y1, x0:x1]
        edge_mask[y0:y1, x0:x1] = edge_window + sobel_rot

    print(f"[edge_mask] (mode=old) skipped {n_skipped} zero-length segments")
    edge_mask1 = edge_mask - np.min(edge_mask)
    edge_mask2 = -edge_mask1 - np.min(-edge_mask1)
    print(f"[edge_mask] (mode=old) done")
    return edge_mask1, edge_mask2

import numpy as np
from shapely.geometry import LineString, Point

###################################################################################
# Normal Projection Edge Correspondence, by Adam Camerer
'''def compute_tangent_normals(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    norm = np.sqrt(dx**2 + dy**2) + 1e-10
    tangent = np.stack([dx / norm, dy / norm], axis=1)
    normal = np.stack([-dy / norm, dx / norm], axis=1)
    return tangent, normal
'''

###################################################################################
from shapely.geometry import LineString, Point, MultiPoint
from shapely.ops import nearest_points
import numpy as np
from scipy.signal import savgol_filter

def _smooth_tangent_angles(tangent, window=9, poly=2):
    """
    Smooth tangent directions in angle space.

    This stabilizes the derivative field (normals) without
    altering crack geometry.

    Rationale:
    - Derivatives amplify pixel noise
    - Angle-domain smoothing preserves curvature better than XY smoothing
    """
    import numpy as np
    from scipy.signal import savgol_filter

    tangent = np.asarray(tangent, float)
    if tangent.ndim != 2 or tangent.shape[1] != 2:
        return tangent

    n = len(tangent)
    if n < 5:
        return tangent

    # Ensure unit vectors
    mag = np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-12
    t = tangent / mag

    # Convert to angle representation
    theta = np.arctan2(t[:, 1], t[:, 0])
    theta = np.unwrap(theta)  # avoid discontinuities

    # Window must be odd
    k = int(window)
    if k % 2 == 0:
        k += 1
    k = max(5, min(k, n - (1 - n % 2)))

    if 5 <= k <= n:
        try:
            theta = savgol_filter(theta, k, poly, mode="interp")
        except Exception:
            pass

    # Rebuild unit tangent vectors
    t_s = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return t_s


def compute_smooth_tangent_normals(x, y, window=7, poly=2, angle_smooth=True):
    """
    Compute C^0-smoothed tangents/normals along a polyline:
      - Optional Savitzky–Golay smoothing of coordinates (auto-odd window).
      - Derivatives w.r.t. arc-length (not index) for stable directions.
      - Enforce normal direction continuity (no frame-to-frame flips).
    Returns:
      tangent: (N,2) unit vectors (dx/ds, dy/ds)
      normal : (N,2) unit vectors, 90° CCW from tangent with sign continuity
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)

    if n < 3:
        # trivial fallback
        dx = np.gradient(x)
        dy = np.gradient(y)
        norm = np.hypot(dx, dy) + 1e-12
        tan = np.stack([dx / norm, dy / norm], axis=1)
        nor = np.stack([-tan[:,1], tan[:,0]], axis=1)
        return tan, nor

    # --- optional smoothing (auto-odd window clipped to [3, n- (1 - n%2)])
    if window is not None and window > 2:
        win = int(window)
        if win % 2 == 0:
            win += 1
        win = max(3, min(win, n - (1 - n % 2)))
        if win >= 3 and win <= n:
            try:
                x = savgol_filter(x, win, poly, mode="interp")
                y = savgol_filter(y, win, poly, mode="interp")
            except Exception:
                pass  # keep raw if SG fails

    # --- arc-length parameter
    ds = np.hypot(np.diff(x), np.diff(y))
    s = np.empty(n, float)
    s[0] = 0.0
    s[1:] = np.cumsum(ds)

    # numerical gradient w.r.t. arc-length (stable even for clustered samples)
    dx_ds = np.gradient(x, s, edge_order=2)
    dy_ds = np.gradient(y, s, edge_order=2)

    mag = np.hypot(dx_ds, dy_ds) + 1e-12
    tangent = np.stack([dx_ds / mag, dy_ds / mag], axis=1)

    # NOTE:
    # We apply light Savitzky-Golay smoothing to tangent angles
    # (not coordinates) to stabilize normal orientation.
    #
    # This reduces derivative noise without altering crack geometry.
    # Window size (~9 px at 1 px sampling) is below feature scale and
    # preserves genuine curvature.
    #
    # TODO: Consider curvature-adaptive smoothing for highly kinked
    #       centered-midlines (human annotation bias correction).
    if angle_smooth:
        tangent = _smooth_tangent_angles(tangent, window=9, poly=2)

    normal = np.stack([-tangent[:,1], tangent[:,0]], axis=1)

    # --- enforce sign continuity on normals
    # if dot(n_i, n_{i-1}) < 0, flip current normal
    for i in range(1, n):
        if np.dot(normal[i], normal[i-1]) < 0:
            normal[i] = -normal[i]

    return tangent, normal


def resolve_normal_pair_with_fallback(
    p,
    nvec,
    cand_a,
    cand_b,
    *,
    fallback_a=None,
    fallback_b=None,
    score_a=None,
    score_b=None,
    max_dist=12.0,
    max_dist_mult=3.0,
    scale_ref=None,
    fallback_min_px=3.0,
    fallback_max_px=5.0,
    fallback_scale=0.003,
    normal_align_min=0.30,
    span_max_mult=3.0,
    eps=1e-9,
):
    """
    Shared robust resolver for normal-edge point pairs.
    """
    p = np.asarray(p, float).reshape(2)
    nvec = np.asarray(nvec, float).reshape(2)
    nlen = float(np.hypot(nvec[0], nvec[1]))
    if not np.isfinite(nlen) or nlen <= eps:
        return None
    nvec = nvec / nlen

    fallback_cap = float(fallback_max_px)
    if scale_ref is not None and np.isfinite(scale_ref):
        fallback_cap = float(
            np.clip(float(fallback_scale) * float(scale_ref), fallback_min_px, fallback_max_px)
        )

    def _pick(cands, score_fn):
        if not cands:
            return None, np.inf
        best_pt, best_score = None, np.inf
        for q in cands:
            q = np.asarray(q, float).reshape(2)
            if not np.all(np.isfinite(q)):
                continue
            if score_fn is None:
                s = float(np.hypot(q[0] - p[0], q[1] - p[1]))
            else:
                s = float(score_fn((float(q[0]), float(q[1]))))
            if np.isfinite(s) and s < best_score:
                best_score = s
                best_pt = (float(q[0]), float(q[1]))
        return best_pt, best_score

    pa, _ = _pick(cand_a, score_a)
    pb, _ = _pick(cand_b, score_b)

    if pa is None and callable(fallback_a):
        q = fallback_a()
        if q is not None:
            q = np.asarray(q, float).reshape(2)
            if np.all(np.isfinite(q)):
                d = float(np.hypot(q[0] - p[0], q[1] - p[1]))
                if d <= fallback_cap:
                    pa = (float(q[0]), float(q[1]))

    if pb is None and callable(fallback_b):
        q = fallback_b()
        if q is not None:
            q = np.asarray(q, float).reshape(2)
            if np.all(np.isfinite(q)):
                d = float(np.hypot(q[0] - p[0], q[1] - p[1]))
                if d <= fallback_cap:
                    pb = (float(q[0]), float(q[1]))

    if pa is None or pb is None:
        return None

    da = float(np.hypot(pa[0] - p[0], pa[1] - p[1]))
    db = float(np.hypot(pb[0] - p[0], pb[1] - p[1]))
    if da > max_dist_mult * max_dist or db > max_dist_mult * max_dist:
        return None

    v = np.asarray([pb[0] - pa[0], pb[1] - pa[1]], float)
    w = float(np.hypot(v[0], v[1]))
    if not np.isfinite(w) or w <= eps:
        return None

    # Reject implausible long bridges.
    if w > span_max_mult * max_dist:
        return None

    # Reject pair directions that are nearly tangent (should align with normal).
    align = float(abs(np.dot(v / w, nvec)))
    if align < float(normal_align_min):
        return None

    return pa, pb, da, db, w

def find_normal_pair(
    mid_x, mid_y, edge1, edge2,
    max_dist_ratio=0.18,
    min_max_dist=12.0,
    length_scale=1.5,
    allow_projection_fallback=True,
):
    """
    For each midline point, cast normals and intersect edges. 
    If intersections are missing, fall back to direct projection 
    onto edge1/edge2 (nearest_points). This ensures normals exist 
    even if the midline is slightly outside the edge corridor.
    """
    mid_x = np.asarray(mid_x, float)
    mid_y = np.asarray(mid_y, float)
    e1 = np.asarray(edge1, float)
    e2 = np.asarray(edge2, float)

    if len(mid_x) == 0 or e1.ndim != 2 or e2.ndim != 2 or len(e1) < 2 or len(e2) < 2:
        n = len(mid_x)
        return (np.full(n, np.nan),) * 4

    _, normals = compute_smooth_tangent_normals(mid_x, mid_y)

    # bbox diagonal for scale
    xmin = min(e1[:,0].min(), e2[:,0].min(), mid_x.min())
    xmax = max(e1[:,0].max(), e2[:,0].max(), mid_x.max())
    ymin = min(e1[:,1].min(), e2[:,1].min(), mid_y.min())
    ymax = max(e1[:,1].max(), e2[:,1].max(), mid_y.max())
    diag = np.hypot(xmax - xmin, ymax - ymin)
    ray_len = max(32.0, float(length_scale) * float(diag))
    max_dist = max(min_max_dist, float(max_dist_ratio) * float(diag))

    line1 = LineString(e1)
    line2 = LineString(e2)

    N = len(mid_x)
    e1x = np.full(N, np.nan)
    e1y = np.full(N, np.nan)
    e2x = np.full(N, np.nan)
    e2y = np.full(N, np.nan)

    def _collect_points(geom):
        if geom.is_empty:
            return []
        if isinstance(geom, Point):
            return [(geom.x, geom.y)]
        if isinstance(geom, MultiPoint):
            return [(g.x, g.y) for g in geom.geoms]
        if isinstance(geom, LineString):
            coords = np.asarray(geom.coords, float)
            return [tuple(coords[0]), tuple(coords[-1])]
        return []

    for i, (mx, my) in enumerate(zip(mid_x, mid_y)):
        nx, ny = normals[i]
        if not np.isfinite(nx) or not np.isfinite(ny):
            continue

        A = (mx - ray_len * nx, my - ray_len * ny)
        B = (mx + ray_len * nx, my + ray_len * ny)
        ray_line = LineString([A, B])

        inter1 = _collect_points(line1.intersection(ray_line))
        inter2 = _collect_points(line2.intersection(ray_line))

        def _fallback1():
            if not allow_projection_fallback:
                return None
            p1 = nearest_points(line1, Point(mx, my))[0]
            return (p1.x, p1.y)

        def _fallback2():
            if not allow_projection_fallback:
                return None
            p2 = nearest_points(line2, Point(mx, my))[0]
            return (p2.x, p2.y)

        pair = resolve_normal_pair_with_fallback(
            p=(mx, my),
            nvec=(nx, ny),
            cand_a=inter1,
            cand_b=inter2,
            fallback_a=_fallback1,
            fallback_b=_fallback2,
            max_dist=max_dist,
            max_dist_mult=3.0,
            scale_ref=diag,
            fallback_min_px=3.0,
            fallback_max_px=5.0,
            fallback_scale=0.003,
            normal_align_min=0.30,
            span_max_mult=3.0,
        )
        if pair is None:
            continue

        (p1, p2, _, _, _) = pair
        e1x[i], e1y[i] = p1
        e2x[i], e2y[i] = p2

    return e1x, e1y, e2x, e2y

###################################################################################
import numpy as np
import scipy.ndimage
from agd import Eikonal
from agd.Metrics import Riemann
from shapely.ops import nearest_points
from shapely.geometry import LineString, Point, MultiPoint
   
# --- add near the top of segmentation.py (after numpy imports) ---
# --- Safe CuPy detection ---
try:
    import cupy as cp
    n_devices = cp.cuda.runtime.getDeviceCount()
    _CUPY = n_devices > 0
    if not _CUPY:
        print("[geodesic] ⚙️ CuPy found but no CUDA devices; using CPU fallback.")
except Exception:
    import numpy as cp
    _CUPY = False


def _to_xp(a, xp, dtype=None, order='C'):
    """Send array to NumPy/CuPy with dtype/order; no copy if already correct."""
    if xp is np:
        return np.asarray(a, dtype=dtype or getattr(a, 'dtype', np.float64), order=order)
    return xp.asarray(a, dtype=dtype or getattr(a, 'dtype', xp.float64), order=order)


def _run_geodesic(metric_array, seeds, tips, sides, dims, strict=True, prefer_gpu=True):
    """
    Run a Riemann2 geodesic; if prefer_gpu and CuPy available, do it on GPU and
    return NumPy (y,x) track. Soft-retry with flattened metric if tip miss > tol.
    """
    # Force CPU mode globally for stability in this environment.
    # (CUDA path can fail with mixed NumPy/CuPy metric internals.)
    use_gpu = False
    xp = cp if use_gpu else np

    # Keep FP64 to preserve results 1:1 with CPU
    metric_xp = _to_xp(metric_array, xp, dtype=(xp.float64 if use_gpu else np.float64))

    # seeds/tips/sides/dims can stay on CPU (binding handles mixed host/device)
    hfm_args = {
        'model'        : 'Riemann2',
        'seeds'        : np.expand_dims(seeds, axis=0),
        'arrayOrdering': 'RowMajor',
        'tips'         : np.expand_dims(tips, axis=0),
        'metric'       : Riemann(metric_xp),
        'verbosity'    : 0,
    }
    if use_gpu:
        hfm_args['mode'] = 'gpu'

    hfmIn = Eikonal.dictIn(hfm_args)
    hfmIn.SetRect(sides=sides, dims=dims)

    try:
        hfmOut = hfmIn.Run()
    except Exception as e:
        # Hard fallback to CPU if GPU build is missing / OOM / etc.
        if use_gpu:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass
        hfmIn = Eikonal.dictIn({
            'model'        : 'Riemann2',
            'seeds'        : np.expand_dims(seeds, axis=0),
            'arrayOrdering': 'RowMajor',
            'tips'         : np.expand_dims(tips, axis=0),
            'metric'       : Riemann(np.asarray(metric_array, dtype=np.float64)),
            'verbosity'    : 0,
        })
        hfmIn.SetRect(sides=sides, dims=dims)
        hfmOut = hfmIn.Run()

    track_yx = [g.T for g in hfmOut['geodesics']][0]  # (N,2), (y,x)
    # Ensure NumPy for downstream code
    if _CUPY and isinstance(track_yx, cp.ndarray):
        track_yx = cp.asnumpy(track_yx)

    if strict:
        d_tip = np.linalg.norm(track_yx[-1] - tips[::-1])  # compare (y,x)
        if d_tip > 10:  # px tolerance
            softened = np.power(np.asarray(metric_array, dtype=np.float64), 0.5)
            return _run_geodesic(softened, seeds, tips, sides, dims, strict=False, prefer_gpu=prefer_gpu)
    return track_yx

def edges_tracking(
    image_crop, pts_cropp,
    edge_mask1_cropp, edge_mask2_cropp,
    midline=None, mu=5, l=2, p=6,
    return_normal_edges=True,
    prefer_gpu=True,
    mode="new",   # "new" or "old"
    domain_mask=None,   # uint8/bool crop-space mask; metric set to 1e30 outside

    # ----------------------------
    # Conservative post-processing
    # ----------------------------
    postprocess_edges="auto",   # auto=True for new, False for old
    resample_ratio=1.15,        # VERY conservative upsample
    resample_min_points=80,
    smooth_k=5,                 # 1 = OFF (recommended default)
    debug_dir=None,             # None → no plots
):
    """
    Edge tracking with conservative anti-quantization.

    HARD RULE (per your request):
      - ALWAYS derive a midline from the extracted edges.
      - If derived midline cannot be computed, this function RAISES.
      - Normals are computed ONLY against the derived midline (no fallback).

    Guarantees:
      - mode="old" edge geometry is untouched (same metric & geodesics)
      - mode="new" keeps your postprocess only; edge masks are NEVER modified
    """
    import time
    import numpy as np
    import scipy.ndimage as ndi
    import scipy.ndimage
    import cv2

    t0_all = time.perf_counter()

    # --------------------------------------------------
    # Setup
    # --------------------------------------------------
    seeds_yx = np.array([pts_cropp[0][1], pts_cropp[0][0]], dtype=float)
    tips_yx  = np.array([pts_cropp[1][1], pts_cropp[1][0]], dtype=float)

    H, W = image_crop.shape[:2]
    sides = np.array([[0, H], [0, W]])
    dims  = np.array([H, W])

    # --------------------------------------------------
    # Small helpers
    # --------------------------------------------------
    def finite_xy(xy):
        xy = np.asarray(xy, float)
        if xy.ndim != 2 or xy.shape[1] != 2:
            return np.empty((0, 2), float)
        return xy[np.isfinite(xy).all(1)]

    def arclen_resample(xy, N):
        xy = np.asarray(xy, float)
        if len(xy) < 2:
            return xy
        d = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
        s = np.concatenate([[0.0], np.cumsum(d)])
        if not np.isfinite(s[-1]) or s[-1] <= 1e-9:
            return xy
        t = np.linspace(0.0, s[-1], int(N))
        x = np.interp(t, s, xy[:, 0])
        y = np.interp(t, s, xy[:, 1])
        return np.column_stack([x, y])

    def _snap_xy_to_mask(xy, mask_u8):
        """
        Snap an (x,y) point to the nearest nonzero pixel in mask_u8.
        Uses EDT with indices (fast, deterministic).
        """
        x, y = float(xy[0]), float(xy[1])
        xi = int(np.clip(np.round(x), 0, W - 1))
        yi = int(np.clip(np.round(y), 0, H - 1))

        if mask_u8[yi, xi] > 0:
            return np.array([xi, yi], float)

        inv = (mask_u8 == 0)
        dist, inds = ndi.distance_transform_edt(inv, return_indices=True)
        ny = int(inds[0, yi, xi])
        nx = int(inds[1, yi, xi])
        return np.array([nx, ny], float)

    def _save_stage5c_debug_plot(
        e1,
        e2,
        orig_mid_xy,
        derived_mid_xy,
        debug_dir,
        mode,
        mu,
        l,
        p
    ):
        """
        PURE visualization.
        Black background.
        Shows:
            - e1 (cyan)
            - e2 (lime)
            - original midline (white dashed)
            - derived midpoint midline (red)
        """

        import os
        import numpy as np
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))

        # --- Black background ---
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        # Plot Edge 1 (cyan)
        if e1 is not None and len(e1) > 1:
            e1 = np.asarray(e1)
            ax.plot(e1[:,0], e1[:,1], color='cyan', linewidth=3, alpha=0.3)
            ax.plot(e1[:,0], e1[:,1], color='cyan', linewidth=1.5, label='Edge 1')

        # Plot Edge 2 (lime)
        if e2 is not None and len(e2) > 1:
            e2 = np.asarray(e2)
            ax.plot(e2[:,0], e2[:,1], color='lime', linewidth=3, alpha=0.3)
            ax.plot(e2[:,0], e2[:,1], color='lime', linewidth=1.5, label='Edge 2')

        # Original midline (white dashed)
        if orig_mid_xy is not None and len(orig_mid_xy) > 1:
            o = np.asarray(orig_mid_xy)
            ax.plot(o[:,0], o[:,1], 'w--', linewidth=2, label='Original midline')

        # Derived midpoint centerline (red)
        if derived_mid_xy is not None and len(derived_mid_xy) > 1:
            d = np.asarray(derived_mid_xy)
            ax.plot(d[:,0], d[:,1], color='red', linewidth=4, alpha=0.25)
            ax.plot(d[:,0], d[:,1], color='red', linewidth=2, label='Midpoint centerline')

        ax.set_aspect('equal', adjustable='box')
        ax.invert_yaxis()
        ax.set_axis_off()

        # --- Clean legend styling ---
        legend = ax.legend(
            loc="lower right",
            frameon=True,
            facecolor='black',
            edgecolor='white',
            fontsize=9
        )
        for text in legend.get_texts():
            text.set_color("white")

        plt.tight_layout()

        os.makedirs(debug_dir, exist_ok=True)
        out_png = os.path.join(
            debug_dir,
            f"edges_tracking_5c_midpoint_mode-{mode}_mu{int(mu)}_l{int(l)}_p{int(p)}.png"
        )

        plt.savefig(out_png, dpi=200, facecolor=fig.get_facecolor())
        plt.close(fig)

    # --------------------------------------------------
    # 1. Gradients
    # --------------------------------------------------
    t0 = time.perf_counter()
    Dx, Dy = np.gradient(image_crop.astype(np.float64))
    t_grad = time.perf_counter() - t0

    # --------------------------------------------------
    # 2. Structure tensor (KEEP OLD IDENTICAL; FIX new to numeric float64)
    # --------------------------------------------------
    t0 = time.perf_counter()
    if mode == "old":
        a11 = scipy.ndimage.gaussian_filter(mu * Dx * Dx, 1, order=(0, 0))
        a12 = scipy.ndimage.gaussian_filter(mu * Dx * Dy, 1, order=(0, 0))
        a21 = scipy.ndimage.gaussian_filter(mu * Dx * Dy, 1, order=(0, 0))
        a22 = scipy.ndimage.gaussian_filter(mu * Dy * Dy, 1, order=(0, 0))
        df = np.array([[1 + a11, a12],
                       [a21,     1 + a22]])
    else:
        a11 = scipy.ndimage.gaussian_filter(mu * Dx * Dx, 1)
        a22 = scipy.ndimage.gaussian_filter(mu * Dy * Dy, 1)
        a12 = scipy.ndimage.gaussian_filter(mu * Dx * Dy, 1)
        a21 = a12
        df = np.array([[1.0 + a11, a12],
                       [a21, 1.0 + a22]], dtype=np.float64)
        df = np.abs(df)
    t_tensor = time.perf_counter() - t0

    # --------------------------------------------------
    # 3. Mask normalization (KEEP OLD IDENTICAL)
    # --------------------------------------------------
    t0 = time.perf_counter()
    if mode == "old":
        em1 = np.squeeze(edge_mask1_cropp)
        em2 = np.squeeze(edge_mask2_cropp)
    else:
        def _norm01(m):
            m = m.astype(np.float64)
            m -= float(m.min())
            mx = float(m.max())
            return m / (mx + 1e-12) if mx > 0 else m
        em1 = _norm01(np.squeeze(edge_mask1_cropp))
        em2 = _norm01(np.squeeze(edge_mask2_cropp))
    t_mask_norm = time.perf_counter() - t0

    # --------------------------------------------------
    # 4. Metric build (KEEP OLD IDENTICAL; keep NEW conservative exponent cap)
    # --------------------------------------------------
    t0 = time.perf_counter()
    metric1 = (1 + em1 * l) ** p * df
    metric2 = (1 + em2 * l) ** p * df
    t_metric = time.perf_counter() - t0

    # --------------------------------------------------
    # 5. Edge geodesics (UNCHANGED behavior)
    # --------------------------------------------------
    # Apply corridor domain mask if provided.
    # We can't use inf or extremely large values (violates HFM positive-definiteness).
    # Instead, scale the barrier to 1000x the max metric inside the corridor —
    # large enough to make shortcuts prohibitively expensive, small enough to be stable.
    if domain_mask is not None:
        _dm = np.asarray(domain_mask, bool)
        _forbidden = ~_dm  # (H, W) 2D mask

        # Build a barrier tensor: barrier * Identity at forbidden pixels.
        # CRITICAL: must be positive definite (det > 0) so HFM solver stays valid.
        # Setting all 4 components to barrier gives det=0 (singular) → solver breaks.
        # Correct form: [[barrier,0],[0,barrier]] = barrier*I, det = barrier^2 > 0.
        _max_inside = float(np.max(metric1[:, :, _dm]))
        _barrier    = _max_inside * 1000.0
        _max_inside2 = float(np.max(metric2[:, :, _dm]))
        _barrier2    = _max_inside2 * 1000.0

        for _met, _bar in [(metric1, _barrier), (metric2, _barrier2)]:
            _met[0, 0][_forbidden] = _bar   # diagonal: barrier
            _met[1, 1][_forbidden] = _bar   # diagonal: barrier
            _met[0, 1][_forbidden] = 0.0    # off-diagonal: zero
            _met[1, 0][_forbidden] = 0.0    # off-diagonal: zero

        print(f"[DOMAIN_MASK] applied: {int(_forbidden.sum())} px forbidden "
              f"({100*_forbidden.mean():.1f}% of crop) "
              f"barrier={_barrier:.2e}", flush=True)

    t0 = time.perf_counter()
    g1_yx = _run_geodesic(metric1, seeds_yx, tips_yx, sides, dims, prefer_gpu=prefer_gpu)
    t_geo1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    g2_yx = _run_geodesic(metric2, seeds_yx, tips_yx, sides, dims, prefer_gpu=prefer_gpu)
    t_geo2 = time.perf_counter() - t0

    # _run_geodesic returns grid indices as (row, col) = (y, x).
    # Convert explicitly once here to geometry coordinates (x, y).
    g1_yx = np.asarray(g1_yx, dtype=float)
    g2_yx = np.asarray(g2_yx, dtype=float)
    e1_raw = np.column_stack([g1_yx[:, 1], g1_yx[:, 0]])  # (x,y)
    e2_raw = np.column_stack([g2_yx[:, 1], g2_yx[:, 0]])  # (x,y)

    # --------------------------------------------------
    # 5b. Conservative anti-quantization (NEW only)
    # --------------------------------------------------
    e1 = finite_xy(e1_raw)
    e2 = finite_xy(e2_raw)

    do_pp = (mode == "new") if postprocess_edges == "auto" else bool(postprocess_edges)

    if do_pp and len(e1) >= 2 and len(e2) >= 2:
        baseN = max(len(e1), len(e2))
        targetN = max(int(resample_min_points), int(baseN * float(resample_ratio)))
        targetN = min(targetN, int(1.2 * baseN))

        e1 = arclen_resample(e1, targetN)
        e2 = arclen_resample(e2, targetN)

        if smooth_k and int(smooth_k) > 1:
            from scipy.signal import savgol_filter

            def _savgol_smooth(xy, k):
                xy = np.asarray(xy, float)
                k = int(k)
                if k <= 1 or len(xy) < k:
                    return xy
                if k % 2 == 0:
                    k += 1
                x = savgol_filter(xy[:, 0], window_length=k, polyorder=2, mode="interp")
                y = savgol_filter(xy[:, 1], window_length=k, polyorder=2, mode="interp")
                return np.column_stack([x, y])

            e1 = _savgol_smooth(e1, smooth_k)
            e2 = _savgol_smooth(e2, smooth_k)

        e1[:, 0] = np.clip(e1[:, 0], 0, W - 1)
        e1[:, 1] = np.clip(e1[:, 1], 0, H - 1)
        e2[:, 0] = np.clip(e2[:, 0], 0, W - 1)
        e2[:, 1] = np.clip(e2[:, 1], 0, H - 1)

    if len(e1) < 2 or len(e2) < 2:
        raise ValueError("[edges_tracking] extracted edges are degenerate (<2 pts); cannot derive midline")

    # --------------------------------------------------
    # 5c. DERIVED MIDLINE from edges (MIDPOINT METHOD)
    # --------------------------------------------------
    t0 = time.perf_counter()

    import numpy as np

    # --- Ensure same direction ---
    def _ensure_same_direction(e1, e2):
        d_same = np.linalg.norm(e1[0] - e2[0]) + np.linalg.norm(e1[-1] - e2[-1])
        d_flip = np.linalg.norm(e1[0] - e2[-1]) + np.linalg.norm(e1[-1] - e2[0])
        return e2 if d_same <= d_flip else e2[::-1]

    # --- Arclength parameterization ---
    def _arclen_param(xy):
        xy = np.asarray(xy, float)
        d = np.sqrt(((xy[1:] - xy[:-1])**2).sum(1))
        s = np.concatenate([[0.0], np.cumsum(d)])
        L = float(s[-1])
        if L <= 1e-9:
            return s, 0.0
        return s / L, L

    def _sample_at_s(xy, s_norm, s_query):
        x = np.interp(s_query, s_norm, xy[:,0])
        y = np.interp(s_query, s_norm, xy[:,1])
        return np.column_stack([x, y])

    # --- Align edge directions ---
    e2 = _ensure_same_direction(e1, e2)

    # --- Parameterize both edges ---
    s1, _ = _arclen_param(e1)
    s2, _ = _arclen_param(e2)

    # --- Sample uniformly in normalized arclength ---
    M = int(max(120, min(len(e1), len(e2))))
    s = np.linspace(0.0, 1.0, M)

    p1 = _sample_at_s(e1, s1, s)
    p2 = _sample_at_s(e2, s2, s)

    # --- Midpoint centerline ---
    derived_midline = 0.5 * (p1 + p2)

    derived_midline = finite_xy(derived_midline)

    if len(derived_midline) < 2:
        raise ValueError("[edges_tracking] derived midline degenerate")

    derived_midline[:, 0] = np.clip(derived_midline[:, 0], 0, W - 1)
    derived_midline[:, 1] = np.clip(derived_midline[:, 1], 0, H - 1)

    # --- Debug plot ---
    if debug_dir:
        _save_stage5c_debug_plot(
            e1=e1,
            e2=e2,
            orig_mid_xy=midline,
            derived_mid_xy=derived_midline,
            debug_dir=debug_dir,
            mode=mode,
            mu=mu,
            l=l,
            p=p,
        )

    t_midline = time.perf_counter() - t0

    # --------------------------------------------------
    # 6. Normals (ONLY against derived midline)  [NO FALLBACK]
    # --------------------------------------------------
    t0 = time.perf_counter()
    normal_edges = None
    normal_edges_clipped = None

    if return_normal_edges:
        try:
            from .segmentation import find_normal_pair
        except Exception:
            from segmentation import find_normal_pair

        mid_x, mid_y = derived_midline[:, 0], derived_midline[:, 1]
        e1x, e1y, e2x, e2y = find_normal_pair(mid_x, mid_y, e1, e2)
        normal_edges = [[e1x.copy(), e1y.copy()], [e2x.copy(), e2y.copy()]]
        normal_edges_clipped = [
            [np.clip(e1x, 0, W - 1), np.clip(e1y, 0, H - 1)],
            [np.clip(e2x, 0, W - 1), np.clip(e2y, 0, H - 1)],
        ]
    t_normals = time.perf_counter() - t0

    # --------------------------------------------------
    # Timing
    # --------------------------------------------------
    t_all = time.perf_counter() - t0_all

    subtiming = {
        "mode": mode,
        "edges_tracking_sec": float(t_all),

        "edges_gradients_sec": float(t_grad),
        "edges_tensor_sec": float(t_tensor),
        "edges_mask_norm_sec": float(t_mask_norm),
        "edges_metric_build_sec": float(t_metric),
        "edges_geodesic1_sec": float(t_geo1),
        "edges_geodesic2_sec": float(t_geo2),

        "derived_midline_sec": float(t_midline),
        "edges_pair_normals_sec": float(t_normals),

        "edges_postprocess": bool(do_pp),
        "edges_resample_ratio": float(resample_ratio),
        "edges_smooth_k": int(smooth_k),
    }

    return {
        "geodesic_edges": [e1, e2],
        "geodesic_edges_raw": [e1_raw, e2_raw],
        "geodesic_edges_proc": [e1, e2],

        # REQUIRED output now
        "derived_midline": derived_midline,

        # normals computed against derived_midline only
        "normal_edge_points": normal_edges,
        "normal_edge_points_clipped": normal_edges_clipped,

        "subtiming": subtiming,
    }

import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes
from skimage.morphology import binary_opening, disk

'''def create_mask(image, x, y, mode="new"):
    """
    Ablation-ready masking function.
    
    Parameters
    ----------
    image : np.ndarray (H,W or H,W,3)
    x, y  : crack boundary coordinates, same shape
    mode  : "new" (default) or "old"
    
    Returns
    -------
    mask : float32 array (H,W), values 0 or 1
    """

    import numpy as np
    import cv2

    mode = mode.lower().strip()
    if mode not in ("old", "new"):
        raise ValueError("mode must be 'old' or 'new'")

    # ============================================================
    #  NEW MODE  (polygon fill + hole-fill + morphological clean)
    # ============================================================
    if mode == "new":
        from scipy.ndimage import binary_fill_holes
        from skimage.morphology import binary_opening, disk

        flat_x = np.array(x, dtype=np.int32)
        flat_y = np.array(y, dtype=np.int32)
        pts = np.vstack([flat_x, flat_y]).T.reshape((-1, 1, 2))

        # 1. Direct polygon rasterization
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)

        # 2. Ensure interior is filled
        mask_filled = binary_fill_holes(mask > 0)

        # 3. Small smoothing (removes jagged artifacts)
        mask_clean = binary_opening(mask_filled, disk(1))

        return mask_clean.astype(np.float32)

    # ============================================================
    #  OLD MODE  (line-drawn contour + connected-component fill)
    # ============================================================
    else:
        from skimage import measure

        def int2(v):
            return int(np.round(v))

        # Local inline reproduction of drow_mask_lines
        def drow_mask_lines(img, contours_x, contours_y, color, t=1, close_contur=False):
            img2 = img.copy()
            for i in range(len(contours_x) - 1):
                x1 = int2(contours_x[i])
                x2 = int2(contours_x[i+1])
                y1 = int2(contours_y[i])
                y2 = int2(contours_y[i+1])
                img2 = cv2.line(img2, (x1, y1), (x2, y2),
                                color=color, thickness=int2(np.ceil(t)))

            # Close contour if required
            if close_contur:
                x1 = int2(contours_x[0]);   y1 = int2(contours_y[0])
                x2 = int2(contours_x[-1]);  y2 = int2(contours_y[-1])
                img2 = cv2.line(img2, (x1, y1), (x2, y2),
                                color=color, thickness=int2(np.ceil(t)))
            return img2

        # --- start of old algorithm ------------------------------------
        flat_x = np.array(x)
        flat_y = np.array(y)

        zeros = np.ones_like(image) * 255
        mask_contour = drow_mask_lines(
            zeros,
            flat_x,
            flat_y,
            color=(0, 0, 0),
            t=1,
            close_contur=True
        )

        # connected component labeling on single channel
        labels = measure.label(mask_contour[:, :, 0], connectivity=1)
        labels[labels != 1] = 0

        # convert to float mask 0/1
        mask = labels.astype(float)
        mask = mask * -1 + 1  # invert

        # morphological erosion (matching original behavior)
        kernel = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype=np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

        return mask.astype(np.float32)'''

def generate_mask_from_edges(
    *,
    img_gray,                  # (H,W) uint8 crop
    edge1_xy, edge2_xy,         # (N,2) float arrays (x,y) — ALREADY FINAL
    midline_xy=None,            # required (N,2) for robust masking
    normals_xy=None,            # required tuple/list: (norm1_xy, norm2_xy)
    out_dir=None,
    tag="cidX",
    mode="new",
    do_morph=False,             # DEFAULT OFF (important)
):
    """
    Minimal debug + mask builder.

    Assumptions:
      - edge1_xy / edge2_xy are ALREADY postprocessed
        (anti-quantization handled in edges_tracking)
      - midline_xy must be provided and aligned to the same frame
      - normals_xy must be provided and aligned to the same frame
      - this function must NOT modify geometry

    Outputs:
      - <tag>_mask.png
      - <tag>_mask_overlay.png (GT-style overlay)
      - <tag>_mask_quads.png (quad rasterization debug)

    Returns
    -------
    mask : uint8 (H,W)
    """

    import os
    import numpy as np
    import cv2
    from helpers.plot_metrics import plot_edges_and_normals

    H, W = img_gray.shape[:2]

    # --------------------------------------------------
    # sanitize (finite only, no resampling)
    # --------------------------------------------------
    def finite_xy(A):
        A = np.asarray(A, float)
        if A.ndim != 2 or A.shape[1] != 2:
            return np.empty((0, 2))
        return A[np.isfinite(A).all(1)]

    def _resample_polyline_xy(P, n_out):
        """
        Arc-length resample for (N,2) polylines.
        """
        P = finite_xy(P)
        n_out = int(max(2, n_out))
        if len(P) < 2:
            return P
        if len(P) == n_out:
            return P
        d = np.sqrt(np.sum(np.diff(P, axis=0) ** 2, axis=1))
        s = np.concatenate([[0.0], np.cumsum(d)])
        total = float(s[-1])
        if total <= 1e-9:
            return np.repeat(P[:1], n_out, axis=0)
        t = np.linspace(0.0, total, n_out)
        x = np.interp(t, s, P[:, 0])
        y = np.interp(t, s, P[:, 1])
        return np.column_stack([x, y])

    e1 = finite_xy(edge1_xy)
    e2 = finite_xy(edge2_xy)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        if len(e1) < 2 or len(e2) < 2:
            mask0 = np.zeros((H, W), np.uint8)
            cv2.imwrite(os.path.join(out_dir, f"{tag}_mask.png"), mask0 * 255)
            return mask0

    # --------------------------------------------------
    # MASK: midline-normal span rasterization (robust)
    # --------------------------------------------------
    if midline_xy is None:
        raise ValueError("generate_mask_from_edges requires midline_xy for robust masking")
    if normals_xy is None or not isinstance(normals_xy, (tuple, list)) or len(normals_xy) != 2:
        raise ValueError("generate_mask_from_edges requires normals_xy=(norm1_xy, norm2_xy)")

    mid = finite_xy(midline_xy)
    if len(mid) < 2:
        raise ValueError("midline_xy invalid (<2 pts)")

    n1 = np.asarray(normals_xy[0], float)
    n2 = np.asarray(normals_xy[1], float)
    if n1.ndim != 2 or n2.ndim != 2 or n1.shape[1] != 2 or n2.shape[1] != 2:
        raise ValueError("normals_xy arrays must have shape (N,2)")

    m = min(len(n1), len(n2))
    if m < 2:
        raise ValueError("normals_xy has insufficient points")
    n1 = n1[:m]
    n2 = n2[:m]

    good = np.isfinite(n1).all(1) & np.isfinite(n2).all(1)
    good_frac = float(np.mean(good)) if len(good) else 0.0
    if good_frac < 0.70:
        raise ValueError(f"insufficient usable normals ({good_frac*100.0:.1f}%)")

    n1_good = n1[good]
    n2_good = n2[good]
    if len(n1_good) < 2 or len(n2_good) < 2:
        raise ValueError("normals_xy has insufficient valid paired points")

    # Defensively equalize both sides to a shared parameterization to avoid
    # long, mismatched quad bridges when one side is sparse/degraded.
    n_target = int(max(16, min(4 * max(2, len(mid)), max(len(n1_good), len(n2_good)))))
    n1r = _resample_polyline_xy(n1_good, n_target)
    n2r = _resample_polyline_xy(n2_good, n_target)
    if len(n1r) < 2 or len(n2r) < 2:
        raise ValueError("resampled normals are insufficient")

    mask = np.zeros((H, W), np.uint8)
    raster_quads = []

    # Fill a continuous strip by rasterizing quads between consecutive normal pairs.
    widths = np.sqrt(np.sum((n1r - n2r) ** 2, axis=1))
    mid_from_resampled_normals = 0.5 * (n1r + n2r)
    steps = np.sqrt(np.sum(np.diff(mid_from_resampled_normals, axis=0) ** 2, axis=1)) if len(mid_from_resampled_normals) >= 2 else np.array([], float)
    med_w = float(np.median(widths[np.isfinite(widths)])) if np.any(np.isfinite(widths)) else 1.0
    med_step = float(np.median(steps[np.isfinite(steps)])) if len(steps) and np.any(np.isfinite(steps)) else 1.0
    max_quad_area = float(max(256.0, 12.0 * med_w * med_step))

    kept_quads = 0
    skipped_large = 0
    skipped_long_bridge = 0
    for i0 in range(len(n1r) - 1):
        i1 = i0 + 1
        quad = np.array([
            n1r[i0],  # side 1 at i
            n1r[i1],  # side 1 at i+1
            n2r[i1],  # side 2 at i+1
            n2r[i0],  # side 2 at i
        ], dtype=np.float32)

        # Guard against catastrophic wedge quads.
        q_area = float(abs(cv2.contourArea(np.round(quad).astype(np.float32))))
        side_step = max(
            float(np.linalg.norm(quad[1] - quad[0])),
            float(np.linalg.norm(quad[2] - quad[3])),
        )
        if q_area > max_quad_area:
            skipped_large += 1
            continue
        if side_step > float(max(10.0, 6.0 * med_step)):
            skipped_long_bridge += 1
            continue

        quad[:, 0] = np.clip(quad[:, 0], 0, W - 1)
        quad[:, 1] = np.clip(quad[:, 1], 0, H - 1)
        raster_quads.append(quad.copy())
        quad_i = np.round(quad).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [quad_i], 1)
        kept_quads += 1

    if not np.any(mask):
        raise ValueError("empty mask after midline-normal span rasterization")

    # --------------------------------------------------
    # DEBUG: the rasterized quad uses normals only (n1/n2), not e1/e2.
    # Compare plotted inputs against the normals-driven geometry.
    # --------------------------------------------------
    def _diag_xy(name, A):
        A = np.asarray(A, float)
        if A.ndim != 2 or A.shape[1] != 2 or len(A) == 0:
            print(f"[MASK DIAG] {name}: empty/invalid")
            return None
        p0, p1 = A[0], A[-1]
        d = p1 - p0
        ang = np.degrees(np.arctan2(d[1], d[0]))
        '''print(
            f"[MASK DIAG] {name}: n={len(A)} "
            f"start=({p0[0]:.2f},{p0[1]:.2f}) end=({p1[0]:.2f},{p1[1]:.2f}) "
            f"x=[{A[:,0].min():.2f},{A[:,0].max():.2f}] "
            f"y=[{A[:,1].min():.2f},{A[:,1].max():.2f}] angle={ang:.1f}deg"
        )'''
        return A

    def _end_dists(name_a, A, name_b, B):
        if A is None or B is None or len(A) == 0 or len(B) == 0:
            return
        d_ss = np.linalg.norm(A[0] - B[0])
        d_se = np.linalg.norm(A[0] - B[-1])
        d_es = np.linalg.norm(A[-1] - B[0])
        d_ee = np.linalg.norm(A[-1] - B[-1])
        #print(f"[MASK DIAG] {name_a}<->{name_b}: ss={d_ss:.2f} se={d_se:.2f} es={d_es:.2f} ee={d_ee:.2f}")

    n1_good = n1r
    n2_good = n2r
    mid_from_normals = mid_from_resampled_normals

    _mid_diag = _diag_xy("midline_xy (plotted input)", mid)
    _e1_diag = _diag_xy("edge1_xy (plotted input)", e1)
    _e2_diag = _diag_xy("edge2_xy (plotted input)", e2)
    _n1_diag = _diag_xy("norm1_xy (quad side)", n1_good)
    _n2_diag = _diag_xy("norm2_xy (quad side)", n2_good)
    _mf_diag = _diag_xy("mid_from_normals (quad implied)", mid_from_normals)
    _end_dists("midline_xy", _mid_diag, "mid_from_normals", _mf_diag)
    _end_dists("edge1_xy", _e1_diag, "norm1_xy", _n1_diag)
    _end_dists("edge2_xy", _e2_diag, "norm2_xy", _n2_diag)
    print(
        f"[MASK QUAD GUARD] quads_kept={kept_quads} "
        f"skip_area={skipped_large} skip_bridge={skipped_long_bridge} "
        f"max_quad_area={max_quad_area:.1f}"
    )

    # Plot-only edge correction: if edges are supplied as (y,x) while normals/midline are (x,y),
    # fix only the overlay inputs. Mask rasterization above already used normals and is correct.
    e1_plot = e1
    e2_plot = e2
    if len(e1) >= 2 and len(e2) >= 2 and len(n1_good) >= 2 and len(n2_good) >= 2:
        m1 = min(len(e1), len(n1_good))
        m2 = min(len(e2), len(n2_good))
        e1m, n1m = e1[:m1], n1_good[:m1]
        e2m, n2m = e2[:m2], n2_good[:m2]

        direct_err = float(np.mean(np.linalg.norm(e1m - n1m, axis=1)) + np.mean(np.linalg.norm(e2m - n2m, axis=1)))
        swapped_err = float(
            np.mean(np.linalg.norm(e1m[:, ::-1] - n1m, axis=1)) +
            np.mean(np.linalg.norm(e2m[:, ::-1] - n2m, axis=1))
        )
        if swapped_err + 1e-6 < direct_err:
            print(
                f"[MASK PLOT FIX] edge inputs look like (y,x); swapping for overlay only "
                f"(direct_err={direct_err:.2f}, swapped_err={swapped_err:.2f})"
            )
            e1_plot = e1[:, ::-1]
            e2_plot = e2[:, ::-1]

    if do_morph:
        # ONLY hole filling — no opening / erosion
        from scipy.ndimage import binary_fill_holes
        mask = binary_fill_holes(mask > 0).astype(np.uint8)

    if out_dir:
        cv2.imwrite(os.path.join(out_dir, f"{tag}_mask.png"), mask * 255)
        try:
            save_mask_quadrilateral_debug_plot(
                img_gray=img_gray,
                quads_xy=raster_quads,
                n1_xy=n1[good],
                n2_xy=n2[good],
                out_png=os.path.join(out_dir, f"{tag}_mask_quads.png"),
            )
        except Exception as e:
            print(f"[MASK DEBUG] failed to save quad plot: {e}")

    # --------------------------------------------------
    # DEBUG: GT-style overlay (KEEP THIS)
    # --------------------------------------------------
    if out_dir:
        src = np.asarray(img_gray)
        if src.ndim == 2:
            vis = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        elif src.ndim == 3 and src.shape[2] == 3:
            vis = src.copy()
        else:
            raise ValueError(f"unexpected img_gray shape: {src.shape}")

        vis_gray = vis.astype(np.float32) / 255.0
        dark_base = np.clip(vis_gray * 0.35, 0.0, 1.0)
        overlay = dark_base.copy()

        overlay[mask == 1] = (0.95, 0.95, 0.95)
        blended = cv2.addWeighted(overlay, 0.8, dark_base, 0.2, 0.0)

        plot_edges_and_normals(
            base_image=(blended * 255).astype(np.uint8),
            midline_segs=[],
            derived_midline_segs=[mid] if len(mid) >= 2 else [],
            edge1_segs=[e1_plot],
            edge2_segs=[e2_plot],
            norm1_segs=[],
            norm2_segs=[],
            out_png=os.path.join(out_dir, f"{tag}_mask_overlay.png"),
            title=f"{tag} — mask overlay",
        )

    return mask


def save_mask_quadrilateral_debug_plot(
    *,
    img_gray,
    quads_xy,
    n1_xy=None,
    n2_xy=None,
    out_png,
):
    """
    Save a side-by-side debug image:
      - Left: cv2.polylines view of n1/n2 normal-side traces.
      - Right: quadrilateral rasterization view used for mask construction.

    Parameters
    ----------
    img_gray : (H,W) or (H,W,3)
        Base crop image.
    quads_xy : list of (4,2) arrays
        Quad vertices in (x,y) order.
    n1_xy, n2_xy : optional (N,2)
        Normal-side points used to build quads.
    out_png : str
        Output path.
    """
    import os
    import numpy as np
    import cv2

    src = np.asarray(img_gray)
    if src.ndim == 2:
        base = cv2.cvtColor(src.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif src.ndim == 3 and src.shape[2] == 3:
        base = src.astype(np.uint8).copy()
    else:
        raise ValueError(f"unexpected img_gray shape: {src.shape}")

    H, W = base.shape[:2]
    sample_stride = 10

    # ----------------------------
    # Left panel: polylines only
    # ----------------------------
    left = base.copy()
    left_fill = left.copy()
    for qi, q in enumerate(quads_xy or []):
        if (qi % sample_stride) != 0:
            continue
        q = np.asarray(q, np.float32).reshape(-1, 2)
        if q.shape != (4, 2):
            continue
        q[:, 0] = np.clip(q[:, 0], 0, W - 1)
        q[:, 1] = np.clip(q[:, 1], 0, H - 1)
        q_i = np.round(q).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(left_fill, [q_i], color=(120, 70, 255))
    left = cv2.addWeighted(left_fill, 0.28, left, 0.72, 0.0)

    for arr, col in ((n1_xy, (0, 255, 255)), (n2_xy, (255, 255, 0))):
        if arr is None:
            continue
        A = np.asarray(arr, np.float32)
        if A.ndim != 2 or A.shape[1] != 2:
            continue
        A = A[np.isfinite(A).all(axis=1)]
        if len(A) >= 2:
            A[:, 0] = np.clip(A[:, 0], 0, W - 1)
            A[:, 1] = np.clip(A[:, 1], 0, H - 1)
            pts = np.round(A).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(left, [pts], isClosed=False, color=col, thickness=1, lineType=cv2.LINE_AA)

    # ----------------------------
    # Right panel: quad rasterization
    # ----------------------------
    overlay = base.copy()
    line = base.copy()
    for q in (quads_xy or []):
        q = np.asarray(q, np.float32).reshape(-1, 2)
        if q.shape != (4, 2):
            continue
        q[:, 0] = np.clip(q[:, 0], 0, W - 1)
        q[:, 1] = np.clip(q[:, 1], 0, H - 1)
        qi = np.round(q).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [qi], color=(40, 180, 255))
        cv2.polylines(line, [qi], isClosed=True, color=(255, 90, 20), thickness=1, lineType=cv2.LINE_AA)
    right = cv2.addWeighted(overlay, 0.33, line, 0.67, 0.0)

    def _legend_row(dst, x, y, color, text):
        cv2.rectangle(dst, (x, y - 7), (x + 12, y + 5), color, thickness=-1)
        cv2.putText(dst, text, (x + 18, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (235, 235, 235), 1, cv2.LINE_AA)

    # Legends per panel so colors are self-explanatory.
    _legend_row(left, 8, 30, (0, 255, 255), "n1 polyline")
    _legend_row(left, 8, 48, (255, 255, 0), "n2 polyline")
    _legend_row(left, 8, 66, (120, 70, 255), f"sampled quad fill (every {sample_stride})")
    _legend_row(right, 8, 30, (40, 180, 255), "quad fill")
    _legend_row(right, 8, 48, (255, 90, 20), "quad boundary")

    # Side-by-side canvas
    gap = np.full((H, 8, 3), 40, dtype=np.uint8)
    vis = np.hstack([left, gap, right])
    cv2.putText(vis, "Polylines (n1/n2)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(vis, "Quads", (W + 16, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    out_dir = os.path.dirname(out_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(out_png, vis)


def int2(a):
    return (int(np.round(a)))

