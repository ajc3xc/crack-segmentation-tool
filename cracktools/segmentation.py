import scipy
import numpy as np
import cracktools.tracking
import cv2
from skimage import measure

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

'''def edges_tracking(image_crop, pts_cropp, edge_mask1_cropp, edge_mask2_cropp,mu = 5,l = 1, p = 12):
    
    seeds = np.array([*pts_cropp[0][::-1]])
    tips = np.array([*pts_cropp[1][::-1]])
    b = np.array([0,image_crop.shape[0]])
    c = np.array([0,image_crop.shape[1]])
    sides = np.array([b,c])
    dims = np.array([image_crop.shape[0],image_crop.shape[1]])

    DxZ,DyZ = np.gradient(image_crop) 

    a11 = scipy.ndimage.gaussian_filter(mu*DxZ**2, 1, order=(0,0))
    a12 = scipy.ndimage.gaussian_filter(mu*DxZ*DyZ, 1, order=(0,0))
    a21 = scipy.ndimage.gaussian_filter(mu*DxZ*DyZ, 1, order=(0,0))
    a22 = scipy.ndimage.gaussian_filter(mu*DyZ**2, 1, order=(0,0))
    df = np.array([[1+a11,a12],[a21,1+a22]])
    metric1 = (1+edge_mask1_cropp.squeeze()*l)**p*df
    metric2 = (1+edge_mask2_cropp.squeeze()*l)**p*df 
    
    metric = Riemann(metric1)
    hfmIn = Eikonal.dictIn({
        'model' : 'Riemann2',
        'seeds' : np.expand_dims(seeds,axis = 0),
        'arrayOrdering' : 'RowMajor',
        'tips' : np.expand_dims(tips,axis = 0),
        'metric' : metric})
    hfmIn.SetRect(sides = sides, dims = dims)
    hfmOut = hfmIn.Run()
    geos1 = [g.T for g in hfmOut['geodesics']]
    # geos1[0][:,0] = geos1[0][:,0]+y1
    # geos1[0][:,1] = geos1[0][:,1]+x1
    # track_e1 = ct.tools.track_crop_to_full(geos1[0].T,pts[0],pts[1],y_margin,x_margin)
    track_e1 = geos1[0]
    
    metric = Riemann(metric2)
    hfmIn = Eikonal.dictIn({
        'model' : 'Riemann2',
        'seeds' : np.expand_dims(seeds,axis = 0),
        'arrayOrdering' : 'RowMajor',
        'tips' : np.expand_dims(tips,axis = 0),
        'metric' : metric})
    hfmIn.SetRect(sides = sides, dims = dims)
    hfmOut = hfmIn.Run()
    geos2 = [g.T for g in hfmOut['geodesics']]
    # geos1[0][:,0] = geos1[0][:,0]+y1
    # geos1[0][:,1] = geos1[0][:,1]+x1
    # track_e2 = ct.tools.track_crop_to_full(geos2[0].T,pts[0],pts[1],y_margin,x_margin)
    track_e2 = geos2[0]
    
    return [track_e1[:,0],track_e1[:,1]], [track_e2[:,0],track_e2[:,1]]'''

import numpy as np
import scipy.ndimage
from scipy.spatial import cKDTree

def compute_tangent_normals(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    norm = np.sqrt(dx**2 + dy**2) + 1e-8
    tangent = np.stack([dx / norm, dy / norm], axis=1)
    normal = np.stack([-dy / norm, dx / norm], axis=1)
    return tangent, normal

def nearest_edge_points_along_normal(mid_x, mid_y, edge_x, edge_y, search_radius=5):
    tangent, normal = compute_tangent_normals(mid_x, mid_y)
    edge_points = np.column_stack([edge_x, edge_y])
    tree = cKDTree(edge_points)
    matched_x = np.empty_like(mid_x)
    matched_y = np.empty_like(mid_y)
    for i, (x0, y0, nvec) in enumerate(zip(mid_x, mid_y, normal)):
        rel = edge_points - np.array([x0, y0])
        dist_along_normal = np.abs(rel @ nvec)
        mask = dist_along_normal < search_radius
        candidates = edge_points[mask]
        if len(candidates) == 0:
            dist, idx = tree.query([x0, y0], k=1)
            matched_x[i], matched_y[i] = edge_points[idx]
        else:
            dists = np.linalg.norm(candidates - np.array([x0, y0]), axis=1)
            j = np.argmin(dists)
            matched_x[i], matched_y[i] = candidates[j]
    return matched_x, matched_y

'''def edges_tracking(image_crop, pts_cropp, edge_mask1_cropp, edge_mask2_cropp, mu=5, l=1, p=12, search_radius=5):
    """
    Compute edge tracks using Riemann2 geodesics and pair each midline point
    to the nearest edge point along its normal (no interpolation, just nearest).
    Returns: ([edge1_x, edge1_y], [edge2_x, edge2_y])
    """
    # --- Riemann metric setup ---
    DxZ, DyZ = np.gradient(image_crop)
    a11 = scipy.ndimage.gaussian_filter(mu * DxZ ** 2, 1)
    a12 = scipy.ndimage.gaussian_filter(mu * DxZ * DyZ, 1)
    a21 = a12
    a22 = scipy.ndimage.gaussian_filter(mu * DyZ ** 2, 1)
    df = np.stack([[1 + a11, a12], [a21, 1 + a22]], axis=0)

    seeds = np.array([*pts_cropp[0][::-1]])
    tips  = np.array([*pts_cropp[1][::-1]])
    sides = np.array([[0, image_crop.shape[0]], [0, image_crop.shape[1]]])
    dims  = np.array(image_crop.shape[:2])

    # --- Compute midline geodesic (Riemann2) ---
    metric_mid = Riemann(df)
    hfmIn_mid = Eikonal.dictIn({
        'model': 'Riemann2',
        'seeds': np.expand_dims(seeds, axis=0),
        'tips': np.expand_dims(tips, axis=0),
        'metric': metric_mid,
        'arrayOrdering': 'RowMajor'
    })
    hfmIn_mid.SetRect(sides=sides, dims=dims)
    out_mid = hfmIn_mid.Run()
    geos_mid = out_mid['geodesics'][0].T  # shape (N_mid, 2)
    mid_y, mid_x = geos_mid[:,0], geos_mid[:,1]

    # --- Compute edge tracks (Riemann2 geodesic) ---
    m1 = edge_mask1_cropp.squeeze()
    m2 = edge_mask2_cropp.squeeze()
    metric1 = Riemann((1 + m1 * l)**p * df)
    metric2 = Riemann((1 + m2 * l)**p * df)

    def compute_edge_track(metric):
        hfm = Eikonal.dictIn({
            'model': 'Riemann2',
            'seeds': np.expand_dims(seeds, axis=0),
            'tips': np.expand_dims(tips, axis=0),
            'metric': metric,
            'arrayOrdering': 'RowMajor'
        })
        hfm.SetRect(sides=sides, dims=dims)
        out = hfm.Run()
        return out['geodesics'][0].T

    geos1 = compute_edge_track(metric1)
    geos2 = compute_edge_track(metric2)
    e1_y, e1_x = geos1[:,0], geos1[:,1]
    e2_y, e2_x = geos2[:,0], geos2[:,1]

    # --- Pair midline to nearest edge points along normal (fast) ---
    e1_x_matched, e1_y_matched = nearest_edge_points_along_normal(mid_x, mid_y, e1_x, e1_y, search_radius)
    e2_x_matched, e2_y_matched = nearest_edge_points_along_normal(mid_x, mid_y, e2_x, e2_y, search_radius)

    return [e1_x_matched, e1_y_matched], [e2_x_matched, e2_y_matched]'''
import numpy as np
from scipy.spatial import cKDTree
import scipy.ndimage

def edges_tracking(image_crop, pts_cropp, edge_mask1_cropp, edge_mask2_cropp, mu=5, l=1, p=12):
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

    # Geodesic edge tracks
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

    print(f"track_e1: shape={track_e1.shape}, x=[{track_e1[:,1].min():.1f}, {track_e1[:,1].max():.1f}], y=[{track_e1[:,0].min():.1f}, {track_e1[:,0].max():.1f}]")
    print(f"  sample start: ({track_e1[0,1]:.1f},{track_e1[0,0]:.1f}), middle: ({track_e1[len(track_e1)//2,1]:.1f},{track_e1[len(track_e1)//2,0]:.1f}), end: ({track_e1[-1,1]:.1f},{track_e1[-1,0]:.1f})")
    print(f"track_e2: shape={track_e2.shape}, x=[{track_e2[:,1].min():.1f}, {track_e2[:,1].max():.1f}], y=[{track_e2[:,0].min():.1f}, {track_e2[:,0].max():.1f}]")
    print(f"  sample start: ({track_e2[0,1]:.1f},{track_e2[0,0]:.1f}), middle: ({track_e2[len(track_e2)//2,1]:.1f},{track_e2[len(track_e2)//2,0]:.1f}), end: ({track_e2[-1,1]:.1f},{track_e2[-1,0]:.1f})")

    # Compute midline as mean
    min_len = min(len(track_e1), len(track_e2))
    mid_x = (track_e1[:min_len,1] + track_e2[:min_len,1]) / 2
    mid_y = (track_e1[:min_len,0] + track_e2[:min_len,0]) / 2

    print(f"mid_x range: [{mid_x.min():.1f}, {mid_x.max():.1f}]")
    print(f"mid_y range: [{mid_y.min():.1f}, {mid_y.max():.1f}]")
    print(f"  sample start: ({mid_x[0]:.1f},{mid_y[0]:.1f}), middle: ({mid_x[len(mid_x)//2]:.1f},{mid_y[len(mid_y)//2]:.1f}), end: ({mid_x[-1]:.1f},{mid_y[-1]:.1f})")

    # KDTree NN
    edge1_tree = cKDTree(np.column_stack([track_e1[:,1], track_e1[:,0]]))
    edge2_tree = cKDTree(np.column_stack([track_e2[:,1], track_e2[:,0]]))

    dist1, idx1 = edge1_tree.query(np.column_stack([mid_x, mid_y]))
    dist2, idx2 = edge2_tree.query(np.column_stack([mid_x, mid_y]))
    nearest_e1_x, nearest_e1_y = track_e1[idx1,1], track_e1[idx1,0]
    nearest_e2_x, nearest_e2_y = track_e2[idx2,1], track_e2[idx2,0]

    print(f"Nearest E1: x [{nearest_e1_x.min():.1f}, {nearest_e1_x.max():.1f}], y [{nearest_e1_y.min():.1f}, {nearest_e1_y.max():.1f}]")
    print(f"Nearest E2: x [{nearest_e2_x.min():.1f}, {nearest_e2_x.max():.1f}], y [{nearest_e2_y.min():.1f}, {nearest_e2_y.max():.1f}]")
    print(f"First 5 midline pts: {list(zip(mid_x[:5], mid_y[:5]))}")
    print(f"First 5 NN e1 pts: {list(zip(nearest_e1_x[:5], nearest_e1_y[:5]))}")
    print(f"First 5 NN e2 pts: {list(zip(nearest_e2_x[:5], nearest_e2_y[:5]))}")

    print("track_e1 start/end:", track_e1[0], track_e1[-1])
    print("track_e2 start/end:", track_e2[0], track_e2[-1])
    print("midline start/end:", mid_x[0], mid_y[0], mid_x[-1], mid_y[-1])

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(track_e1[:,1], track_e1[:,0], label='Edge 1', color='red')
    plt.plot(track_e2[:,1], track_e2[:,0], label='Edge 2', color='blue')
    plt.plot(mid_x, mid_y, label='Midline', color='green')
    plt.scatter(nearest_e1_x, nearest_e1_y, label='NN Edge1', color='orange', s=2)
    plt.scatter(nearest_e2_x, nearest_e2_y, label='NN Edge2', color='purple', s=2)
    plt.legend()
    plt.gca().invert_yaxis()
    plt.show()

    return [nearest_e1_x, nearest_e1_y], [nearest_e2_x, nearest_e2_y]

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