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
    window_half_size=40,
    mode="new",   # "new" (GPU hybrid) or "old" (akomp22)
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

        if not use_gpu:
            print("[edge_mask] ⚙️ running in CPU mode (no CUDA device detected)")

        img_h, img_w = image_gray.shape
        edge_mask = np.zeros_like(image_gray, dtype=float)
        center_line_length = 3
        n_skipped = 0

        for i in range(track.shape[1] - 1):
            y0 = float(track[0, i])
            x0 = float(track[1, i])
            if i < track.shape[1] - center_line_length:
                y1 = float(track[0, i + center_line_length])
                x1 = float(track[1, i + center_line_length])
                flip = False
            else:
                y1 = float(track[0, i - center_line_length])
                x1 = float(track[1, i - center_line_length])
                flip = True

            if y0 == y1 and x0 == x1:
                n_skipped += 1
                continue

            try:
                from cracktools.tracking import tang_len
                ddx, ddy, _ = tang_len(x0, y0, x1, y1)
            except Exception:
                ddy = y1 - y0
                ddx = x1 - x0
            if flip:
                ddx, ddy = -ddx, -ddy

            angle = np.degrees(np.arctan2(ddx, ddy))
            half_r = int(min(window_half_size, y0, img_h - y0 - 1))
            half_c = int(min(window_half_size, x0, img_w - x0 - 1))
            r1, r2 = int(round(y0 - half_r)), int(round(y0 + half_r))
            c1, c2 = int(round(x0 - half_c)), int(round(x0 + half_c))
            if r1 < 0 or r2 > img_h or c1 < 0 or c2 > img_w:
                continue
            window = image_gray[r1:r2, c1:c2]
            if window.shape[0] < 3 or window.shape[1] < 3:
                continue

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
                m = max(1, int(min(half_r, half_c) / 5))
                projected[:m, :] = projected[-m:, :] = 0
                projected[:, :m] = projected[:, -m:] = 0
                edge_mask[r1:r2, c1:c2] += projected
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
        # ORIGINAL code: track[1] -> "x", track[0] -> "y"
        start_point_x = track[1, i]
        start_point_y = track[0, i]
        a = False

        if i < track.shape[1] - center_line_length:
            end_point_x = track[1, i + center_line_length]
            end_point_y = track[0, i + center_line_length]
        else:
            a = True
            end_point_x = track[1, i - center_line_length]
            end_point_y = track[0, i - center_line_length]

        if start_point_x == end_point_x and start_point_y == end_point_y:
            n_skipped += 1
            continue

        # ORIGINAL tangent computation
        try:
            from cracktools.tracking import tang_len
            ddx, ddy, _ = tang_len(start_point_x, start_point_y,
                                   end_point_x, end_point_y)
        except Exception:
            ddy = end_point_y - start_point_y
            ddx = end_point_x - start_point_x

        if a:
            ddx = -ddx
            ddy = -ddy

        # ORIGINAL angle
        angle_deg = np.arctan2(ddx, ddy) * 57.3

        # ORIGINAL window (note: index order kept as in repo)
        r1 = int(start_point_x - window_half_size)
        r2 = int(start_point_x + window_half_size)
        c1 = int(start_point_y - window_half_size)
        c2 = int(start_point_y + window_half_size)

        # In the original this could crash; here we safely skip
        if r1 < 0 or c1 < 0 or r2 > H or c2 > W:
            continue

        window = image_gray[r1:r2, c1:c2]
        if window.shape[0] < 3 or window.shape[1] < 3:
            continue

        # ORIGINAL rotate → sobel → unrotate
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

        # ORIGINAL border suppression
        m = max(1, int(window_half_size / 5))
        sobel_rot[:m, :] = 0
        sobel_rot[-m:, :] = 0
        sobel_rot[:, :m] = 0
        sobel_rot[:, -m:] = 0

        edge_window = edge_mask[r1:r2, c1:c2]
        edge_mask[r1:r2, c1:c2] = edge_window + sobel_rot

    print(f"[edge_mask] (mode=old) skipped {n_skipped} zero-length segments")
    edge_mask1 = edge_mask - np.min(edge_mask)
    edge_mask2 = -edge_mask1 - np.min(-edge_mask1)
    print(f"[edge_mask] (mode=old) done")
    return edge_mask1, edge_mask2

import numpy as np
from shapely.geometry import LineString, Point

###################################################################################
# Normal Projection Edge Correspondence, by Adam Camerer
def compute_tangent_normals(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    norm = np.sqrt(dx**2 + dy**2) + 1e-10
    tangent = np.stack([dx / norm, dy / norm], axis=1)
    normal = np.stack([-dy / norm, dx / norm], axis=1)
    return tangent, normal

###################################################################################
from shapely.geometry import LineString, Point, MultiPoint
from shapely.ops import nearest_points
import numpy as np
from scipy.signal import savgol_filter

def compute_smooth_tangent_normals(x, y, window=7, poly=2):
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
    normal = np.stack([-tangent[:,1], tangent[:,0]], axis=1)

    # --- enforce sign continuity on normals
    # if dot(n_i, n_{i-1}) < 0, flip current normal
    for i in range(1, n):
        if np.dot(normal[i], normal[i-1]) < 0:
            normal[i] = -normal[i]

    return tangent, normal

def find_normal_pair(
    mid_x, mid_y, edge1, edge2,
    max_dist_ratio=0.18,
    min_max_dist=12.0,
    length_scale=1.5
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

        # fall back to projection if intersections empty
        if not inter1:
            np1 = nearest_points(line1, Point(mx, my))[0]
            inter1 = [(np1.x, np1.y)]
        if not inter2:
            np2 = nearest_points(line2, Point(mx, my))[0]
            inter2 = [(np2.x, np2.y)]

        # pick nearest from each edge
        def _nearest(pt_list):
            dists = [np.hypot(px - mx, py - my) for (px, py) in pt_list]
            j = int(np.argmin(dists))
            return pt_list[j], dists[j]

        (px1, py1), d1 = _nearest(inter1)
        (px2, py2), d2 = _nearest(inter2)

        # accept if within relaxed max_dist (here ×3)
        if d1 <= 3*max_dist and d2 <= 3*max_dist:
            e1x[i], e1y[i] = px1, py1
            e2x[i], e2y[i] = px2, py2

    return e1x, e1y, e2x, e2y

###################################################################################
import numpy as np
import scipy.ndimage
from agd import Eikonal
from agd.Metrics import Riemann
from shapely.ops import nearest_points
from shapely.geometry import LineString, Point, MultiPoint

'''def _run_geodesic(metric_array, seeds, tips, sides, dims, strict=True):
    """Run geodesic; retry with softened metric if path deviates."""
    metric = Riemann(metric_array)
    hfmIn = Eikonal.dictIn({
        'model': 'Riemann2',
        'seeds': np.expand_dims(seeds, axis=0),
        'arrayOrdering': 'RowMajor',
        'tips': np.expand_dims(tips, axis=0),
        'metric': metric,
        'verbosity': 0,
    })
    hfmIn.SetRect(sides=sides, dims=dims)
    hfmOut = hfmIn.Run()
    track = [g.T for g in hfmOut['geodesics']][0]  # (N,2) (y,x)

    if strict:
        d_tip = np.linalg.norm(track[-1] - tips[::-1])  # compare (y,x)
        if d_tip > 10:   # pixels tolerance
            print(f"[WARN] Geodesic missed tip by {d_tip:.1f}px → retry with softened metric")
            softened = np.power(metric_array, 0.5)  # flatten penalties
            return _run_geodesic(softened, seeds, tips, sides, dims, strict=False)
    return track'''
    
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
    use_gpu = bool(prefer_gpu and _CUPY)
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
    
'''def edges_tracking(
    image_crop, pts_cropp,
    edge_mask1_cropp, edge_mask2_cropp,
    midline=None, mu=5, l=2, p=6,
    return_normal_edges=True
):
    """
    Robust edge tracking: uses normalized masks, safer parameters,
    and geodesic retry if path deviates.
    """
    seeds = np.array([*pts_cropp[0][::-1]])  # (y,x)
    tips  = np.array([*pts_cropp[1][::-1]])  # (y,x)
    b = np.array([0, image_crop.shape[0]])
    c = np.array([0, image_crop.shape[1]])
    sides = np.array([b, c])
    dims = np.array([image_crop.shape[0], image_crop.shape[1]])

    DxZ, DyZ = np.gradient(image_crop)
    a11 = scipy.ndimage.gaussian_filter(mu * DxZ**2, 1)
    a12 = scipy.ndimage.gaussian_filter(mu * DxZ * DyZ, 1)
    a21 = scipy.ndimage.gaussian_filter(mu * DxZ * DyZ, 1)
    a22 = scipy.ndimage.gaussian_filter(mu * DyZ**2, 1)

    # --- enforce symmetry & positivity ---
    a12 = a21 = 0.5 * (a12 + a21)
    df = np.abs(np.array([[1 + a11, a12],
                          [a21, 1 + a22]]))

    # --- normalize edge masks safely ---
    def norm_mask(m):
        m = m.astype(float)
        m = m - m.min()
        m = m / (m.max() + 1e-9)
        return np.clip(m, 0, 1)

    em1 = norm_mask(edge_mask1_cropp.squeeze())
    em2 = norm_mask(edge_mask2_cropp.squeeze())

    # --- safer exponentiation and debug checks ---
    metric1 = (1 + em1 * l) ** min(p, 4) * df
    metric2 = (1 + em2 * l) ** min(p, 4) * df

    # --- NaN / negative debug diagnostics ---
    for name, arr in [("metric1", metric1), ("metric2", metric2)]:
        if np.any(np.isnan(arr)):
            nan_ratio = np.mean(np.isnan(arr))
            print(f"[DEBUG seg] NaNs detected in {name} ({nan_ratio*100:.2f}% NaN)")
        if np.any(arr < 0):
            neg_ratio = np.mean(arr < 0)
            print(f"[DEBUG seg] Negative values detected in {name} ({neg_ratio*100:.2f}% of pixels)")
        if np.any(np.isinf(arr)):
            inf_ratio = np.mean(np.isinf(arr))
            print(f"[DEBUG seg] Infs detected in {name} ({inf_ratio*100:.2f}% of pixels)")
        print(f"[DEBUG seg] {name} stats: min={np.nanmin(arr):.3e}, max={np.nanmax(arr):.3e}")

    # --- geodesics ---
    track_e1 = _run_geodesic(metric1, seeds, tips, sides, dims)
    track_e2 = _run_geodesic(metric2, seeds, tips, sides, dims)

    # convert to (x,y)
    track_e1 = np.stack([track_e1[:, 1], track_e1[:, 0]], axis=1)
    track_e2 = np.stack([track_e2[:, 1], track_e2[:, 0]], axis=1)

    normal_edges = None
    normal_edges_clipped = None

    if return_normal_edges and midline is not None:
        from .segmentation import find_normal_pair  # import your existing function
        m = np.asarray(midline)
        if m.ndim == 2 and m.shape[1] == 2:
            mid_x, mid_y = m[:, 0], m[:, 1]
        elif m.ndim == 2 and m.shape[0] == 2:
            mid_x, mid_y = m[0], m[1]
        else:
            raise ValueError("midline must be (N,2) or (2,N)")

        e1x, e1y, e2x, e2y = find_normal_pair(mid_x, mid_y, track_e1, track_e2)
        normal_edges = [[e1x.copy(), e1y.copy()], [e2x.copy(), e2y.copy()]]
        e1x_clip = np.clip(e1x, 0, image_crop.shape[1]-1)
        e1y_clip = np.clip(e1y, 0, image_crop.shape[0]-1)
        e2x_clip = np.clip(e2x, 0, image_crop.shape[1]-1)
        e2y_clip = np.clip(e2y, 0, image_crop.shape[0]-1)
        normal_edges_clipped = [[e1x_clip, e1y_clip], [e2x_clip, e2y_clip]]

    return {
        "geodesic_edges": [track_e1, track_e2],
        "normal_edge_points": normal_edges,
        "normal_edge_points_clipped": normal_edges_clipped,
    }'''
    
def edges_tracking(
    image_crop, pts_cropp,
    edge_mask1_cropp, edge_mask2_cropp,
    midline=None, mu=5, l=2, p=6,
    return_normal_edges=True,
    prefer_gpu=True,
    mode="new"   # "new" or "old"
):
    """
    Edge-tracking with complete ablation support and full subtiming.
    mode="old" → EXACT legacy math (repository version)
    mode="new" → stabilized metric (normalized masks, symmetric df, exponent clamp)

    Both modes:
        • Use the same _run_geodesic (GPU-optional)
        • Use same normal pairing
        • Return identical shapes
        • Full per-stage timing identical to your pre-refactor version
    """
    import time
    import numpy as np
    import scipy.ndimage

    from agd.Metrics import Riemann

    # GLOBAL timing
    t0_all = time.perf_counter()

    # seeds/tips (x,y) → (y,x)
    seeds_yx = np.array([pts_cropp[0][1], pts_cropp[0][0]], dtype=float)
    tips_yx  = np.array([pts_cropp[1][1], pts_cropp[1][0]], dtype=float)

    H, W = image_crop.shape[:2]
    sides = np.array([[0, H], [0, W]])
    dims  = np.array([H, W])

    # ----------------------------------------------------------
    # 1. GRADIENTS  (same timing block for both modes)
    # ----------------------------------------------------------
    t0 = time.perf_counter()
    Dx, Dy = np.gradient(image_crop.astype(np.float64))
    t1 = time.perf_counter()
    t_grad = t1 - t0

    # ----------------------------------------------------------
    # 2. STRUCTURE TENSOR + METRIC BASE (mode-dependent)
    # ----------------------------------------------------------
    if mode == "old":
        # Original non-symmetric structure tensor (order=(0,0))
        a11 = scipy.ndimage.gaussian_filter(mu * Dx*Dx, 1, order=(0,0))
        a12 = scipy.ndimage.gaussian_filter(mu * Dx*Dy, 1, order=(0,0))
        a21 = scipy.ndimage.gaussian_filter(mu * Dx*Dy, 1, order=(0,0))
        a22 = scipy.ndimage.gaussian_filter(mu * Dy*Dy, 1, order=(0,0))

        df = np.array([[1 + a11, a12],
                       [a21,     1 + a22]])

    else:  # NEW
        # Symmetric tensor with abs for numeric safety
        a11 = scipy.ndimage.gaussian_filter(mu * Dx*Dx, 1)
        a22 = scipy.ndimage.gaussian_filter(mu * Dy*Dy, 1)
        a12 = scipy.ndimage.gaussian_filter(mu * Dx*Dy, 1)
        a21 = a12
        a12 = a21 = 0.5 * (a12 + a21)

        df = np.abs(np.array([
            [1 + a11, a12],
            [a21,     1 + a22]
        ], dtype=object))

    t2 = time.perf_counter()
    t_tensor = t2 - t1

    # ----------------------------------------------------------
    # 3. MASK NORMALIZATION (real difference between old/new)
    # ----------------------------------------------------------
    if mode == "old":
        em1 = np.squeeze(edge_mask1_cropp)
        em2 = np.squeeze(edge_mask2_cropp)

    else:  # NEW
        def _norm01(m):
            m = m.astype(np.float64)
            m -= m.min()
            mx = m.max()
            return m / (mx + 1e-12) if mx > 0 else m

        em1 = _norm01(np.squeeze(edge_mask1_cropp))
        em2 = _norm01(np.squeeze(edge_mask2_cropp))

    t3 = time.perf_counter()
    t_normmasks = t3 - t2

    # ----------------------------------------------------------
    # 4. METRIC BUILD  (old exponent vs new exponent clamp)
    # ----------------------------------------------------------
    if mode == "old":
        metric1 = (1 + em1 * l)**p * df
        metric2 = (1 + em2 * l)**p * df
    else:
        pp = min(int(p), 4)
        metric1 = (1 + em1 * l)**pp * df
        metric2 = (1 + em2 * l)**pp * df

    t4 = time.perf_counter()
    t_metric = t4 - t3

    # ----------------------------------------------------------
    # 5. GEODESIC SOLVES (combined timing)
    # ----------------------------------------------------------
    t_geo0 = time.perf_counter()
    track1_yx = _run_geodesic(metric1, seeds_yx, tips_yx, sides, dims, prefer_gpu=prefer_gpu)
    track2_yx = _run_geodesic(metric2, seeds_yx, tips_yx, sides, dims, prefer_gpu=prefer_gpu)
    t_geo1 = time.perf_counter()
    t_geodesics = t_geo1 - t_geo0

    # convert to (x,y)
    track_e1 = np.stack([track1_yx[:, 1], track1_yx[:, 0]], axis=1)
    track_e2 = np.stack([track2_yx[:, 1], track2_yx[:, 0]], axis=1)

    # ----------------------------------------------------------
    # 6. NORMALS (shared)
    # ----------------------------------------------------------
    t_normals_start = time.perf_counter()
    normal_edges = None
    normal_edges_clipped = None

    if return_normal_edges and midline is not None:
        try: 
            from .segmentation import find_normal_pair
        except Exception:
            from segmentation import find_normal_pair

        m = np.asarray(midline)
        if m.ndim == 2 and m.shape[1] == 2:
            mid_x, mid_y = m[:, 0], m[:, 1]
        else:
            mid_x, mid_y = m[0], m[1]

        e1x, e1y, e2x, e2y = find_normal_pair(mid_x, mid_y, track_e1, track_e2)
        normal_edges = [[e1x.copy(), e1y.copy()], [e2x.copy(), e2y.copy()]]

        e1x_c = np.clip(e1x, 0, W-1); e1y_c = np.clip(e1y, 0, H-1)
        e2x_c = np.clip(e2x, 0, W-1); e2y_c = np.clip(e2y, 0, H-1)
        normal_edges_clipped = [[e1x_c, e1y_c], [e2x_c, e2y_c]]

    t_normals_end = time.perf_counter()
    t_normals = t_normals_end - t_normals_start

    # ----------------------------------------------------------
    # 7. FINAL TIMING CONSISTENCY
    # ----------------------------------------------------------
    measured = (
        t_grad +
        t_tensor +
        t_normmasks +
        t_metric +
        t_geodesics +
        t_normals
    )

    t_all = time.perf_counter() - t0_all
    t_remainder = max(t_all - measured, 0)

    subtiming = {
        "mode": mode,
        "edges_gradients_sec":      float(t_grad),
        "edges_tensor_sec":         float(t_tensor),
        "edges_mask_norm_sec":      float(t_normmasks),
        "edges_metric_build_sec":   float(t_metric),
        "edges_geodesic_both_sec":  float(t_geodesics),
        "edges_pair_normals_sec":   float(t_normals),
        "edges_remainder_sec":      float(t_remainder),
        "edges_total_internal_sec": float(measured + t_remainder),
    }

    return {
        "geodesic_edges": [track_e1, track_e2],
        "normal_edge_points": normal_edges,
        "normal_edge_points_clipped": normal_edges_clipped,
        "subtiming": subtiming,
    }


import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes
from skimage.morphology import binary_opening, disk

def create_mask(image, x, y, mode="new"):
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

        return mask.astype(np.float32)

def redrow_lines(img,contours_x,contours_y,t,scale):
    flat_x = [item for sublist in contours_x for item in sublist]
    flat_y = [item for sublist in contours_y for item in sublist]
    img2 = img.copy()
    for i in range(len(flat_x)-1):
        x1 = int2(flat_x[i]-0.5)
        x2 = int2(flat_x[i+1]-0.5)
        y1 = int2(flat_y[i]-0.5)
        y2 = int2(flat_y[i+1]-0.5)
        img2 = cv2.line(img2,(x1,y1),(x2,y2),color=(0,255,0),thickness=int2(np.ceil(t*scale)))
    return (img2)

def drow_mask_lines(img,contours_x,contours_y,color,t=1,close_contur = False):
#     flat_x = [item for sublist in contours_x for item in sublist]
#     flat_y = [item for sublist in contours_y for item in sublist]
    img2 = img.copy()
    for i in range(len(contours_x)-1):
        x1 = int2(np.round(contours_x[i]))
        x2 = int2(np.round(contours_x[i+1]))
        y1 = int2(np.round(contours_y[i]))
        y2 = int2(np.round(contours_y[i+1]))
        img2 = cv2.line(img2,(x1,y1),(x2,y2),color=color,thickness=int2(np.ceil(t)))
        
    x1 = int2(np.round(contours_x[0]))
    x2 = int2(np.round(contours_x[-1]))
    y1 = int2(np.round(contours_y[0]))
    y2 = int2(np.round(contours_y[-1]))
    if close_contur == True:
        img2 = cv2.line(img2,(x1,y1),(x2,y2),color=color,thickness=int2(np.ceil(t)))
    return (img2)

def int2(a):
    return (int(np.round(a)))