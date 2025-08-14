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

'''def edge_masks(image_gray, track, window_half_size=40):

    edge_mask = np.zeros_like(image_gray, dtype=float)
    center_line_length = 3
    img_h, img_w = image_gray.shape
    n_skipped = 0

    for i in range(track.shape[1] - 1):
        start_row = float(track[0, i])  # row (y)
        start_col = float(track[1, i])  # col (x)

        if i < track.shape[1] - center_line_length:
            end_row = float(track[0, i + center_line_length])
            end_col = float(track[1, i + center_line_length])
            a = False
        else:
            end_row = float(track[0, i - center_line_length])
            end_col = float(track[1, i - center_line_length])
            a = True

        if start_col == end_col and start_row == end_row:
            n_skipped += 1
            continue

        ddx, ddy, _ = cracktools.tracking.tang_len(start_col, start_row, end_col, end_row)
        if a:
            ddx = -ddx
            ddy = -ddy

        # Correct window computation: rows (Y), cols (X)
        half_win_r = int(min(window_half_size, start_row, img_h - start_row - 1))
        half_win_c = int(min(window_half_size, start_col, img_w - start_col - 1))
        half_win_r = max(1, half_win_r)
        half_win_c = max(1, half_win_c)

        r1 = int(round(start_row - half_win_r))
        r2 = int(round(start_row + half_win_r))
        c1 = int(round(start_col - half_win_c))
        c2 = int(round(start_col + half_win_c))

        if r1 < 0 or r2 > img_h or c1 < 0 or c2 > img_w:
            print(f"Skipping i={i}: window out of bounds ({r1}:{r2},{c1}:{c2}) in image of shape {image_gray.shape}")
            continue

        window = image_gray[r1:r2, c1:c2]
        if window.shape[0] < 3 or window.shape[1] < 3:
            print(f"Skipping i={i}: window too small ({window.shape})")
            continue

        try:
            angle = np.arctan2(ddx, ddy) * 57.3
            window_rotate = scipy.ndimage.rotate(window, angle, reshape=False)
            sobel = scipy.ndimage.gaussian_filter(window_rotate / 255.0, 1, order=(1, 0), mode='reflect')
            sobel_rotate = scipy.ndimage.rotate(sobel, -angle, reshape=False)
        except Exception as e:
            print(f"Skipping i={i}: sobel rotation failed — {e}")
            continue

        m = max(1, int(min(half_win_r, half_win_c) / 5))
        sobel_rotate[:m, :] = 0
        sobel_rotate[-m:, :] = 0
        sobel_rotate[:, :m] = 0
        sobel_rotate[:, -m:] = 0

        # Insert into the correct position in edge_mask
        mask_r1 = r1
        mask_r2 = r1 + sobel_rotate.shape[0]
        mask_c1 = c1
        mask_c2 = c1 + sobel_rotate.shape[1]

        if mask_r1 < 0 or mask_r2 > img_h or mask_c1 < 0 or mask_c2 > img_w:
            print(f"Skip insertion i={i}: mask indices out of bounds ({mask_r1}:{mask_r2},{mask_c1}:{mask_c2})")
            continue

        edge_mask[mask_r1:mask_r2, mask_c1:mask_c2] += sobel_rotate

    print(f"Skipped {n_skipped} zero-length segments.")
    edge_mask1 = edge_mask - np.min(edge_mask)
    edge_mask2 = -edge_mask1 - np.min(-edge_mask1)
    return edge_mask1, edge_mask2'''
    
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

'''def normal_intersections_bruteforce(mid_x, mid_y, edge_x, edge_y, normal_length):
    tangent, normal = compute_tangent_normals(mid_x, mid_y)
    n = len(mid_x)
    result_x = np.full(n, np.nan)
    result_y = np.full(n, np.nan)
    edge_x = np.asarray(edge_x, dtype=float)
    edge_y = np.asarray(edge_y, dtype=float)
    edge_segments = [
        LineString([(edge_x[i], edge_y[i]), (edge_x[i+1], edge_y[i+1])])
        for i in range(len(edge_x) - 1)
        if np.all(np.isfinite([edge_x[i], edge_y[i], edge_x[i+1], edge_y[i+1]]))
    ]
    # Edge as array for fallback
    edge_points = np.column_stack([edge_x, edge_y])

    for i in range(n):
        mx, my = float(mid_x[i]), float(mid_y[i])
        if not np.isfinite(mx) or not np.isfinite(my):
            continue
        nvec = normal[i]
        norm_a = (mx - normal_length * nvec[0], my - normal_length * nvec[1])
        norm_b = (mx + normal_length * nvec[0], my + normal_length * nvec[1])
        norm_line = LineString([norm_a, norm_b])
        best_pt = None
        best_dist = np.inf
        for seg in edge_segments:
            inter = norm_line.intersection(seg)
            if isinstance(inter, Point):
                dist = np.hypot(inter.x - mx, inter.y - my)
                if dist < best_dist:
                    best_dist = dist
                    best_pt = inter
        if best_pt:
            result_x[i], result_y[i] = best_pt.x, best_pt.y
        else:
            # fallback: nearest edge point
            dists = np.hypot(edge_x - mx, edge_y - my)
            j = np.argmin(dists)
            result_x[i], result_y[i] = edge_x[j], edge_y[j]

    return result_x, result_y'''

from shapely.geometry import LineString, Point, MultiPoint, GeometryCollection

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
    return rx, ry

def edges_tracking(image_crop, pts_cropp, edge_mask1_cropp, edge_mask2_cropp, midline, mu=5, l=1, p=12):
    # --- Geodesic edge extraction (unchanged from your code) ---
    seeds = np.array([*pts_cropp[0][::-1]])
    tips = np.array([*pts_cropp[1][::-1]])
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

    metric = Riemann(metric1)
    hfmIn = Eikonal.dictIn({
        'model': 'Riemann2',
        'seeds': np.expand_dims(seeds, axis=0),
        'arrayOrdering': 'RowMajor',
        'tips': np.expand_dims(tips, axis=0),
        'metric': metric
    })
    hfmIn.SetRect(sides=sides, dims=dims)
    hfmOut = hfmIn.Run()
    track_e1 = [g.T for g in hfmOut['geodesics']][0]

    metric = Riemann(metric2)
    hfmIn = Eikonal.dictIn({
        'model': 'Riemann2',
        'seeds': np.expand_dims(seeds, axis=0),
        'arrayOrdering': 'RowMajor',
        'tips': np.expand_dims(tips, axis=0),
        'metric': metric
    })
    hfmIn.SetRect(sides=sides, dims=dims)
    hfmOut = hfmIn.Run()
    track_e2 = [g.T for g in hfmOut['geodesics']][0]

    # --- Robust midline handling ---
    if isinstance(midline, np.ndarray):
        if midline.shape[0] == 2:  # shape (2, N)
            mid_y, mid_x = midline[0], midline[1]
        elif midline.shape[1] == 2:  # shape (N, 2)
            mid_y, mid_x = midline[:, 0], midline[:, 1]
        else:
            raise ValueError("midline array must be shape (2, N) or (N, 2)")
    elif isinstance(midline, list) and len(midline) == 2:
        mid_y, mid_x = midline[0], midline[1]
    else:
        raise ValueError("midline must be a (2, N) or (N, 2) array, or a list [y_array, x_array]")

    # --- Normal length & bounds ---
    height, width = image_crop.shape[:2]
    normal_length = int(np.ceil(np.hypot(height, width)))

    # --- Normal intersections ---
    edge1_x, edge1_y = normal_intersections_bruteforce(mid_x, mid_y, track_e1[:,1], track_e1[:,0], normal_length)
    edge2_x, edge2_y = normal_intersections_bruteforce(mid_x, mid_y, track_e2[:,1], track_e2[:,0], normal_length)

    # --- Output clipping ---
    edge1_x = np.clip(edge1_x, 0, width-1)
    edge1_y = np.clip(edge1_y, 0, height-1)
    edge2_x = np.clip(edge2_x, 0, width-1)
    edge2_y = np.clip(edge2_y, 0, height-1)

    return [edge1_x, edge1_y], [edge2_x, edge2_y]

def edges_tracking(
    image_crop, pts_cropp,
    edge_mask1_cropp, edge_mask2_cropp,
    midline=None, mu=5, l=1, p=12,
    return_normal_edges=True
):
    """
    Returns:
      {
        "geodesic_edges": [track_e1, track_e2],  # as (N,2) arrays (x, y)
        "normal_edge_points": [ [edge1_x, edge1_y], [edge2_x, edge2_y] ] or None,
      }
    """
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

    # Geodesic extraction
    metric = Riemann(metric1)
    hfmIn = Eikonal.dictIn({
        'model': 'Riemann2',
        'seeds': np.expand_dims(seeds, axis=0),
        'arrayOrdering': 'RowMajor',
        'tips': np.expand_dims(tips, axis=0),
        'metric': metric
    })
    hfmIn.SetRect(sides=sides, dims=dims)
    hfmOut = hfmIn.Run()
    track_e1 = [g.T for g in hfmOut['geodesics']][0]  # shape (N,2) as (y, x)

    metric = Riemann(metric2)
    hfmIn = Eikonal.dictIn({
        'model': 'Riemann2',
        'seeds': np.expand_dims(seeds, axis=0),
        'arrayOrdering': 'RowMajor',
        'tips': np.expand_dims(tips, axis=0),
        'metric': metric
    })
    hfmIn.SetRect(sides=sides, dims=dims)
    hfmOut = hfmIn.Run()
    track_e2 = [g.T for g in hfmOut['geodesics']][0]  # shape (N,2) as (y, x)

    # --- Convert tracks to (x, y) convention for everything ---
    # (track_e1/2 are (N,2) as (y, x); convert to (x, y))
    track_e1 = np.stack([track_e1[:,1], track_e1[:,0]], axis=1)  # (N,2), (x, y)
    track_e2 = np.stack([track_e2[:,1], track_e2[:,0]], axis=1)

    # --- Normal edge points for widths ---
    normal_edges = None
    if return_normal_edges and midline is not None:
        # Always extract mid_x, mid_y (x then y) in list/array form
        if isinstance(midline, np.ndarray):
            if midline.shape[0] == 2:
                mid_x, mid_y = midline[0], midline[1]
            elif midline.shape[1] == 2:
                mid_x, mid_y = midline[:,0], midline[:,1]
            else:
                raise ValueError("midline array must be shape (2, N) or (N, 2)")
        elif isinstance(midline, list) and len(midline) == 2:
            mid_x, mid_y = midline[0], midline[1]
        else:
            raise ValueError("midline must be a (2, N) or (N, 2) array, or a list [x_array, y_array]")

        height, width = image_crop.shape[:2]
        normal_length = int(np.ceil(np.hypot(height, width)))

        '''# Inputs: mid_x, mid_y, edge_x, edge_y (all (N,))
        edge1_x, edge1_y = normal_intersections_bruteforce(mid_x, mid_y, track_e1[:,0], track_e1[:,1], normal_length)
        edge2_x, edge2_y = normal_intersections_bruteforce(mid_x, mid_y, track_e2[:,0], track_e2[:,1], normal_length)
        edge1_x = np.clip(edge1_x, 0, width-1)
        edge1_y = np.clip(edge1_y, 0, height-1)
        edge2_x = np.clip(edge2_x, 0, width-1)
        edge2_y = np.clip(edge2_y, 0, height-1)
        normal_edges = [[edge1_x, edge1_y], [edge2_x, edge2_y]]

    return {
        "geodesic_edges": [track_e1, track_e2],  # (N,2) as (x, y)
        "normal_edge_points": normal_edges
    }'''

        normal_edges = [[edge1_x.copy(), edge1_y.copy()],
                    [edge2_x.copy(), edge2_y.copy()]]

        # For mask use only:
        edge1_x = np.clip(edge1_x, 0, width-1)
        edge1_y = np.clip(edge1_y, 0, height-1)
        edge2_x = np.clip(edge2_x, 0, width-1)
        edge2_y = np.clip(edge2_y, 0, height-1)

        normal_edges_clipped = [[edge1_x, edge1_y], [edge2_x, edge2_y]]

    return {
        "geodesic_edges": [track_e1, track_e2],
        "normal_edge_points_clipped": normal_edges_clipped,
        "normal_edge_points": normal_edges  # <-- for exact geometry
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