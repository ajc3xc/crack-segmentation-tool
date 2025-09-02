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

from shapely.geometry import LineString, Point, MultiPoint, GeometryCollection
'''
def normal_intersections_bruteforce(mid_x, mid_y, edge_x, edge_y, normal_length):
    _, normal = compute_tangent_normals(mid_x, mid_y)
    edge_line = LineString(np.column_stack([edge_x, edge_y]))
    n = len(mid_x)
    rx = np.full(n, np.nan, float)
    ry = np.full(n, np.nan, float)

    for i in range(n):
        mx, my = float(mid_x[i]), float(mid_y[i])
        if not np.isfinite(mx) or not np.isfinite(my): 
            continue
        nx, ny = normal[i]
        a = (mx - normal_length*nx, my - normal_length*ny)
        b = (mx + normal_length*nx, my + normal_length*ny)
        inter = edge_line.intersection(LineString([a, b]))

        def nearest_on_edge(px, py):
            t = edge_line.project(Point(px, py))
            p = edge_line.interpolate(t)
            return p.x, p.y

        if inter.is_empty:
            rx[i], ry[i] = nearest_on_edge(mx, my)
        elif isinstance(inter, Point):
            rx[i], ry[i] = inter.x, inter.y
        elif isinstance(inter, (MultiPoint, GeometryCollection)):
            # pick the hit closest to the mid point
            pts = [g for g in getattr(inter, 'geoms', []) if isinstance(g, Point)]
            if pts:
                j = np.argmin([np.hypot(p.x - mx, p.y - my) for p in pts])
                rx[i], ry[i] = pts[j].x, pts[j].y
            else:
                rx[i], ry[i] = nearest_on_edge(mx, my)
        else:
            rx[i], ry[i] = nearest_on_edge(mx, my)
    return rx, ry'''

'''def normal_intersections_bruteforce(mid_x, mid_y, edge_x, edge_y, normal_length):
    """
    Intersect each midline normal with the edge polyline.
    Safer version:
      - only accept intersections within max_dist
      - fallback projects locally (nearest edge *segment*), not whole line
      - avoids "teleporting" across sparse edges
    """
    _, normal = compute_tangent_normals(mid_x, mid_y)
    edge_coords = np.column_stack([edge_x, edge_y])
    edge_line = LineString(edge_coords)

    n = len(mid_x)
    rx = np.full(n, np.nan, float)
    ry = np.full(n, np.nan, float)

    max_dist = max(10.0, 0.12 * float(normal_length))

    for i in range(n):
        mx, my = float(mid_x[i]), float(mid_y[i])
        if not np.isfinite(mx) or not np.isfinite(my):
            continue
        nx, ny = normal[i]

        # Build normal ray
        a = (mx - normal_length*nx, my - normal_length*ny)
        b = (mx + normal_length*nx, my + normal_length*ny)
        ray = LineString([a, b])
        inter = edge_line.intersection(ray)

        def accept(px, py):
            d = np.hypot(px - mx, py - my)
            return (px, py) if d <= max_dist else (np.nan, np.nan)

        if isinstance(inter, Point):
            rx[i], ry[i] = accept(inter.x, inter.y)
        elif isinstance(inter, (MultiPoint, GeometryCollection)):
            pts = [g for g in getattr(inter, "geoms", []) if isinstance(g, Point)]
            if pts:
                j = np.argmin([np.hypot(p.x - mx, p.y - my) for p in pts])
                rx[i], ry[i] = accept(pts[j].x, pts[j].y)
        else:
            # fallback: find nearest *segment* instead of whole polyline
            dists = np.hypot(edge_coords[:,0] - mx, edge_coords[:,1] - my)
            j = np.argmin(dists)
            if 0 < j < len(edge_coords)-1:
                seg = LineString([edge_coords[j-1], edge_coords[j+1]])
            elif j > 0:
                seg = LineString([edge_coords[j-1], edge_coords[j]])
            else:
                seg = LineString([edge_coords[j], edge_coords[j+1]])
            proj = seg.interpolate(seg.project(Point(mx, my)))
            rx[i], ry[i] = accept(proj.x, proj.y)

    return rx, ry'''
    
###################################################################################
from shapely.geometry import LineString, Point, MultiPoint
import numpy as np
from scipy.signal import savgol_filter

def compute_smooth_tangent_normals(x, y, window=7, poly=2):
    """Smoothed tangent + normal from midline coords."""
    if len(x) > window:
        x = savgol_filter(x, window, poly)
        y = savgol_filter(y, window, poly)
    dx = np.gradient(x)
    dy = np.gradient(y)
    norm = np.sqrt(dx**2 + dy**2) + 1e-10
    tangent = np.stack([dx / norm, dy / norm], axis=1)
    normal = np.stack([-dy / norm, dx / norm], axis=1)
    return tangent, normal

def find_normal_pair(mid_x, mid_y, edge1, edge2, max_dist_ratio=0.12):
    """
    Cast a perpendicular at each midline point, intersect with edge1+edge2.
    Ensures both sides are found (nearest intersection each side).
    """
    _, normals = compute_smooth_tangent_normals(mid_x, mid_y)
    line1 = LineString(edge1)
    line2 = LineString(edge2)

    diag = np.hypot(
        max(edge1[:,0].max(), edge2[:,0].max()) - min(edge1[:,0].min(), edge2[:,0].min()),
        max(edge1[:,1].max(), edge2[:,1].max()) - min(edge1[:,1].min(), edge2[:,1].min())
    )
    normal_length = int(np.ceil(diag))
    max_dist = max(10.0, max_dist_ratio * normal_length)

    e1x = np.full(len(mid_x), np.nan)
    e1y = np.full(len(mid_x), np.nan)
    e2x = np.full(len(mid_x), np.nan)
    e2y = np.full(len(mid_x), np.nan)

    for i, (mx, my) in enumerate(zip(mid_x, mid_y)):
        nx, ny = normals[i]
        a = (mx - normal_length*nx, my - normal_length*ny)
        b = (mx + normal_length*nx, my + normal_length*ny)
        ray = LineString([a, b])

        def nearest_intersection(line):
            inter = line.intersection(ray)
            pts = []
            if isinstance(inter, Point):
                pts = [inter]
            elif isinstance(inter, MultiPoint):
                pts = list(inter.geoms)
            if not pts:
                return None
            # closest to (mx,my)
            dists = [np.hypot(p.x - mx, p.y - my) for p in pts]
            j = int(np.argmin(dists))
            if dists[j] > max_dist:
                return None
            return pts[j].x, pts[j].y

        p1 = nearest_intersection(line1)
        p2 = nearest_intersection(line2)

        if p1 and p2:
            e1x[i], e1y[i] = p1
            e2x[i], e2y[i] = p2
        # else: leave NaNs → avoids bent escape lines

    return e1x, e1y, e2x, e2y

###################################################################################
def edges_tracking(
    image_crop, pts_cropp,
    edge_mask1_cropp, edge_mask2_cropp,
    midline=None, mu=5, l=1, p=12,
    return_normal_edges=True
):
    """
    Returns:
      {
        "geodesic_edges": [track_e1, track_e2],          # (N,2) arrays (x, y)
        "normal_edge_points": [ [edge1_x, edge1_y], [edge2_x, edge2_y] ] or None,
        "normal_edge_points_clipped": same but clipped to image bounds
      }
    """
    import time, os, random

    # seeds/tips come in crop coords as (x,y) in your code; hfmm wants (y,x)
    seeds = np.array([*pts_cropp[0][::-1]])
    tips  = np.array([*pts_cropp[1][::-1]])
    b = np.array([0, image_crop.shape[0]])
    c = np.array([0, image_crop.shape[1]])
    sides = np.array([b, c])
    dims = np.array([image_crop.shape[0], image_crop.shape[1]])

    DxZ, DyZ = np.gradient(image_crop)
    a11 = scipy.ndimage.gaussian_filter(mu * DxZ**2, 1, order=(0,0))
    a12 = scipy.ndimage.gaussian_filter(mu * DxZ * DyZ, 1, order=(0,0))
    a21 = scipy.ndimage.gaussian_filter(mu * DxZ * DyZ, 1, order=(0,0))
    a22 = scipy.ndimage.gaussian_filter(mu * DyZ**2, 1, order=(0,0))
    df = np.array([[1 + a11, a12], [a21, 1 + a22]])

    metric1 = (1 + edge_mask1_cropp.squeeze() * l) ** p * df
    metric2 = (1 + edge_mask2_cropp.squeeze() * l) ** p * df

    # ---- geodesic 1
    metric = Riemann(metric1)
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
    track_e1 = [g.T for g in hfmOut['geodesics']][0]     # (N,2) (y,x)

    # ---- geodesic 2
    metric = Riemann(metric2)
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
    track_e2 = [g.T for g in hfmOut['geodesics']][0]     # (N,2) (y,x)

    # → convert to (x,y) for all downstream code
    track_e1 = np.stack([track_e1[:,1], track_e1[:,0]], axis=1)  # (N,2) (x,y)
    track_e2 = np.stack([track_e2[:,1], track_e2[:,0]], axis=1)

    normal_edges = None
    normal_edges_clipped = None

    if return_normal_edges and midline is not None:
        m = np.asarray(midline)
        if m.ndim != 2:
            raise ValueError("midline must be 2D array")
        if m.shape[1] == 2:
            mid_x, mid_y = m[:,0], m[:,1]
        elif m.shape[0] == 2:
            mid_x, mid_y = m[0], m[1]
        else:
            raise ValueError("midline must be (N,2) or (2,N)")

        # paired normals
        e1x, e1y, e2x, e2y = find_normal_pair(mid_x, mid_y, track_e1, track_e2)

        normal_edges = [[e1x.copy(), e1y.copy()],
                        [e2x.copy(), e2y.copy()]]

        e1x_clip = np.clip(e1x, 0, image_crop.shape[1]-1)
        e1y_clip = np.clip(e1y, 0, image_crop.shape[0]-1)
        e2x_clip = np.clip(e2x, 0, image_crop.shape[1]-1)
        e2y_clip = np.clip(e2y, 0, image_crop.shape[0]-1)

        normal_edges_clipped = [[e1x_clip, e1y_clip], [e2x_clip, e2y_clip]]

        # ---------------- DEBUG PLOT ----------------
        '''fig, ax = plt.subplots(figsize=(10,6))
        ax.imshow(image_crop, cmap='gray')
        ax.plot(mid_x, mid_y, 'g-', lw=1, label="midline")
        ax.plot(track_e1[:,0], track_e1[:,1], 'r-', lw=1, label="edge1")
        ax.plot(track_e2[:,0], track_e2[:,1], 'b-', lw=1, label="edge2")

        step = max(1, len(mid_x)//40)
        for i in range(0, len(mid_x), step):
            if np.isfinite(e1x[i]) and np.isfinite(e2x[i]):
                ax.plot([mid_x[i], e1x[i]], [mid_y[i], e1y[i]],
                        color='cyan', lw=0.8, alpha=0.7)
                ax.plot([mid_x[i], e2x[i]], [mid_y[i], e2y[i]],
                        color='magenta', lw=0.8, alpha=0.7)
                ax.plot(e1x[i], e1y[i], 'co', ms=2)
                ax.plot(e2x[i], e2y[i], 'mo', ms=2)

        ax.set_title("Debug Normals vs Geodesic Edges")
        ax.legend()
        plt.tight_layout()

        # make filename unique with timestamp + random suffix
        ts = int(time.time()*1000)
        fname = f"debug_normals_{ts}.png"

        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"[DEBUG] normals plotted to {os.path.abspath(fname)}")'''

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