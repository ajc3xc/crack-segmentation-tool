import scipy
import numpy as np
import cracktools.tracking
import cv2
from skimage import measure
import matplotlib.pyplot as plt

from agd import Eikonal
from agd.Metrics import Riemann
from agd.Plotting import savefig, quiver; #savefig.dirName = 'Figures/Riemannian'
from agd import LinearParallel as lp
from agd import AutomaticDifferentiation as ad
norm_infinity = ad.Optimization.norm_infinity
import scipy.ndimage
    
def edge_masks(image_gray, track, window_half_size=40):
    import numpy as np
    import scipy.ndimage

    edge_mask = np.zeros_like(image_gray, dtype=float)
    center_line_length = 3
    img_h, img_w = image_gray.shape
    n_skipped = 0

    for i in range(track.shape[1] - 1):
        start_row = float(track[0, i])  # y
        start_col = float(track[1, i])  # x

        if i < track.shape[1] - center_line_length:
            end_row = float(track[0, i + center_line_length])
            end_col = float(track[1, i + center_line_length])
            a = False
        else:
            end_row = float(track[0, i - center_line_length])
            end_col = float(track[1, i - center_line_length])
            a = True

        if start_row == end_row and start_col == end_col:
            n_skipped += 1
            continue

        ddx, ddy, _ = cracktools.tracking.tang_len(start_col, start_row, end_col, end_row)
        if a:
            ddx = -ddx
            ddy = -ddy

        angle_deg = np.arctan2(ddx, ddy) * 180.0 / np.pi

        # Extract safe window
        half_win_r = int(min(window_half_size, start_row, img_h - start_row - 1))
        half_win_c = int(min(window_half_size, start_col, img_w - start_col - 1))
        half_win_r = max(1, half_win_r)
        half_win_c = max(1, half_win_c)

        r1 = int(round(start_row - half_win_r))
        r2 = int(round(start_row + half_win_r))
        c1 = int(round(start_col - half_win_c))
        c2 = int(round(start_col + half_win_c))

        if r1 < 0 or r2 > img_h or c1 < 0 or c2 > img_w:
            continue

        window = image_gray[r1:r2, c1:c2]
        if window.shape[0] < 3 or window.shape[1] < 3:
            continue

        try:
            # Convert to float and normalize
            patch = window.astype(float) / 255.0

            # Apply Gaussian smoothed Sobel
            grad_y = scipy.ndimage.gaussian_filter(patch, sigma=1, order=(1, 0), mode='reflect')
            grad_x = scipy.ndimage.gaussian_filter(patch, sigma=1, order=(0, 1), mode='reflect')

            # Project gradient along normal direction
            # normal = [-ddy, ddx]
            projected = grad_x * (-ddy) + grad_y * ddx

            # Center crop to avoid edge effects
            m = max(1, int(min(half_win_r, half_win_c) / 5))
            projected[:m, :] = 0
            projected[-m:, :] = 0
            projected[:, :m] = 0
            projected[:, -m:] = 0

            # Add to edge_mask
            edge_mask[r1:r2, c1:c2] += projected

        except Exception as e:
            print(f"Failed at i={i}: {e}")
            continue

    print(f"Skipped {n_skipped} zero-length segments.")

    edge_mask1 = edge_mask - np.min(edge_mask)
    edge_mask2 = -edge_mask1 - np.min(-edge_mask1)
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

def _run_geodesic(metric_array, seeds, tips, sides, dims, strict=True):
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
    return track

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
    df = np.array([[1 + a11, a12], [a21, 1 + a22]])

    # --- normalize edge masks
    def norm_mask(m):
        m = m.astype(float)
        m = m - m.min()
        return m / (m.max() + 1e-9)

    em1 = norm_mask(edge_mask1_cropp.squeeze())
    em2 = norm_mask(edge_mask2_cropp.squeeze())

    metric1 = (1 + em1 * l) ** p * df
    metric2 = (1 + em2 * l) ** p * df

    # --- geodesics
    track_e1 = _run_geodesic(metric1, seeds, tips, sides, dims)
    track_e2 = _run_geodesic(metric2, seeds, tips, sides, dims)

    # convert to (x,y)
    track_e1 = np.stack([track_e1[:,1], track_e1[:,0]], axis=1)
    track_e2 = np.stack([track_e2[:,1], track_e2[:,0]], axis=1)

    normal_edges = None
    normal_edges_clipped = None

    if return_normal_edges and midline is not None:
        from .segmentation import find_normal_pair  # import your existing function
        m = np.asarray(midline)
        if m.ndim == 2 and m.shape[1] == 2:
            mid_x, mid_y = m[:,0], m[:,1]
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
    }

import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes
from skimage.morphology import binary_opening, disk

def create_mask(image, x, y):
    """
    Create a filled binary mask for a crack defined by (x, y) edge coordinates.
    - Fills the area inside the crack polyline.
    - Optionally smooths edges with a small morphological opening.
    - Returns a float mask (1.0 = crack, 0.0 = background).
    """
    # Convert x/y to int and format as OpenCV expects
    flat_x = np.array(x, dtype=np.int32)
    flat_y = np.array(y, dtype=np.int32)
    pts = np.vstack([flat_x, flat_y]).T.reshape((-1, 1, 2))

    # 1. Draw and fill the polygon
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)

    # 2. Fill any holes (robust for possible open paths)
    mask_filled = binary_fill_holes(mask > 0)

    # 3. (Optional) Clean rough edges with morphological opening
    mask_clean = binary_opening(mask_filled, disk(1))

    # 4. Return as float (if you want to match previous behavior)
    return mask_clean.astype(float)

def redrow_lines(img,counturs_x,counturs_y,t,scale):
    flat_x = [item for sublist in counturs_x for item in sublist]
    flat_y = [item for sublist in counturs_y for item in sublist]
    img2 = img.copy()
    for i in range(len(flat_x)-1):
        x1 = int2(flat_x[i]-0.5)
        x2 = int2(flat_x[i+1]-0.5)
        y1 = int2(flat_y[i]-0.5)
        y2 = int2(flat_y[i+1]-0.5)
        img2 = cv2.line(img2,(x1,y1),(x2,y2),color=(0,255,0),thickness=int2(np.ceil(t*scale)))
    return (img2)

def drow_mask_lines(img,counturs_x,counturs_y,color,t=1,close_contur = False):
#     flat_x = [item for sublist in counturs_x for item in sublist]
#     flat_y = [item for sublist in counturs_y for item in sublist]
    img2 = img.copy()
    for i in range(len(counturs_x)-1):
        x1 = int2(np.round(counturs_x[i]))
        x2 = int2(np.round(counturs_x[i+1]))
        y1 = int2(np.round(counturs_y[i]))
        y2 = int2(np.round(counturs_y[i+1]))
        img2 = cv2.line(img2,(x1,y1),(x2,y2),color=color,thickness=int2(np.ceil(t)))
        
    x1 = int2(np.round(counturs_x[0]))
    x2 = int2(np.round(counturs_x[-1]))
    y1 = int2(np.round(counturs_y[0]))
    y2 = int2(np.round(counturs_y[-1]))
    if close_contur == True:
        img2 = cv2.line(img2,(x1,y1),(x2,y2),color=color,thickness=int2(np.ceil(t)))
    return (img2)

def int2(a):
    return (int(np.round(a)))