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

'''def edge_masks(image_gray,track,window_half_size= 40):

    edge1 = []
    edge2 = []
    step = 2
    n = 265
    center_line_length = 3
    edge_mask = np.zeros_like((image_gray),dtype = float)
    for i in range(track.shape[1]-1):
    # for i in range(n,n+1):
        start_point_x = track[1,i]
        start_point_y = track[0,i]
        a= False
        if i<track.shape[1]-center_line_length:
            end_point_x = track[1,i+center_line_length]
            end_point_y = track[0,i+center_line_length]
        else:
            a = True
            end_point_x = track[1,i-center_line_length]
            end_point_y = track[0,i-center_line_length]
        if start_point_x==end_point_x and start_point_y==end_point_y:
            continue

        ddx,ddy,l = cracktools.tracking.tang_len(start_point_x,start_point_y,end_point_x,end_point_y)
        if a == True:
            ddx = -ddx
            ddy = -ddy
        window = np.zeros((window_half_size*2,window_half_size*2))
        window = image_gray[int(start_point_x-window_half_size):int(start_point_x+window_half_size),
                                  int(start_point_y-window_half_size):int(start_point_y+window_half_size)]

        angle = np.arctan2(ddx,ddy)*57.3

        window_rotate = scipy.ndimage.rotate(window,angle,reshape=False)

        sobel2 = scipy.ndimage.sobel(window_rotate/255,axis=0)
        sobel = scipy.ndimage.gaussian_filter(window_rotate/255, 1, order=(1,0), output=None, mode='reflect', cval=0.0, truncate=4.0)
        sobel_rotate = scipy.ndimage.rotate(sobel,-angle,reshape=False)
    #     plt.imshow(window)
    #     plt.show()
    #     plt.imshow(sobel2)
    #     plt.show()
#         m = int(window_half_size)/5
        m = np.max([1,int(window_half_size/5)])
        sobel_rotate[:m,:] = 0
        sobel_rotate[-m:,:] = 0
        sobel_rotate[:,:m] = 0
        sobel_rotate[:,-m:] = 0
        
        edge_window = edge_mask[int(start_point_x-window_half_size):int(start_point_x+window_half_size),
                                  int(start_point_y-window_half_size):int(start_point_y+window_half_size)]

        edge_mask[int(start_point_x-window_half_size):
                  int(start_point_x+window_half_size),
                  int(start_point_y-window_half_size):
                  int(start_point_y+window_half_size)] = edge_window + sobel_rotate
        
    edge_mask1 = edge_mask - np.min(edge_mask)
    edge_mask2 = edge_mask1*-1-np.min(edge_mask1*-1)
    
    return edge_mask1,edge_mask2'''
    
import numpy as np
import scipy.ndimage

def edge_masks(image_gray, track, window_half_size=40):
    import numpy as np
    import scipy.ndimage

    edge_mask = np.zeros_like(image_gray, dtype=float)
    center_line_length = 3
    img_h, img_w = image_gray.shape
    n_skipped = 0

    for i in range(track.shape[1] - 1):
        start_x = float(track[1, i])
        start_y = float(track[0, i])
        if i < track.shape[1] - center_line_length:
            end_x = float(track[1, i + center_line_length])
            end_y = float(track[0, i + center_line_length])
            a = False
        else:
            end_x = float(track[1, i - center_line_length])
            end_y = float(track[0, i - center_line_length])
            a = True
        if start_x == end_x and start_y == end_y:
            n_skipped += 1
            continue

        ddx, ddy, _ = cracktools.tracking.tang_len(start_x, start_y, end_x, end_y)
        if a:
            ddx = -ddx
            ddy = -ddy

        # Shrink window so it's always in bounds
        half_win_x = int(min(window_half_size, start_x, img_h - start_x - 1))
        half_win_y = int(min(window_half_size, start_y, img_w - start_y - 1))
        half_win_x = max(1, half_win_x)
        half_win_y = max(1, half_win_y)

        x1 = int(round(start_x - half_win_x))
        x2 = int(round(start_x + half_win_x))
        y1 = int(round(start_y - half_win_y))
        y2 = int(round(start_y + half_win_y))

        # Check for valid window
        if x1 < 0 or x2 > img_h or y1 < 0 or y2 > img_w:
            print(f"Skipping i={i}: window out of bounds ({x1}:{x2},{y1}:{y2}) in image of shape {image_gray.shape}")
            continue

        window = image_gray[x1:x2, y1:y2]
        if window.shape[0] < 3 or window.shape[1] < 3:
            print(f"Skipping i={i}: window too small ({window.shape})")
            continue

        #print(f"Processing i={i}: window shape {window.shape} at ({x1}:{x2},{y1}:{y2})")

        # Continue as before
        angle = np.arctan2(ddx, ddy) * 57.3
        window_rotate = scipy.ndimage.rotate(window, angle, reshape=False)
        sobel = scipy.ndimage.gaussian_filter(window_rotate / 255.0, 1, order=(1, 0), mode='reflect')
        sobel_rotate = scipy.ndimage.rotate(sobel, -angle, reshape=False)
        m = max(1, int(min(half_win_x, half_win_y) / 5))
        sobel_rotate[:m, :] = 0
        sobel_rotate[-m:, :] = 0
        sobel_rotate[:, :m] = 0
        sobel_rotate[:, -m:] = 0

        mask_x1 = x1
        mask_x2 = x1 + sobel_rotate.shape[0]
        mask_y1 = y1
        mask_y2 = y1 + sobel_rotate.shape[1]
        # Defensive: Only insert into legal area
        if (mask_x1 < 0 or mask_y1 < 0 or mask_x2 > img_h or mask_y2 > img_w):
            print(f"Skip insertion i={i}: mask indices out of bounds ({mask_x1}:{mask_x2},{mask_y1}:{mask_y2})")
            continue

        edge_mask[mask_x1:mask_x2, mask_y1:mask_y2] += sobel_rotate

    print(f"Skipped {n_skipped} zero-length segments.")
    edge_mask1 = edge_mask - np.min(edge_mask)
    edge_mask2 = -edge_mask1 - np.min(-edge_mask1)
    return edge_mask1, edge_mask2

import scipy
from agd import Eikonal
from agd.Metrics import Riemann
from agd.Plotting import savefig, quiver; #savefig.dirName = 'Figures/Riemannian'
from agd import LinearParallel as lp
from agd import AutomaticDifferentiation as ad
norm_infinity = ad.Optimization.norm_infinity

def edges_tracking(image_crop, pts_cropp, edge_mask1_cropp, edge_mask2_cropp,mu = 5,l = 1, p = 12):
    
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
    
    return [track_e1[:,0],track_e1[:,1]], [track_e2[:,0],track_e2[:,1]]

def create_mask(image, x, y):
    # x and y: concatenated as [top_edge (reversed), bottom_edge]
    flat_x = np.array(x, dtype=np.int32)
    flat_y = np.array(y, dtype=np.int32)
    
    # Make a (N, 2) array for points in (col, row) format
    pts = np.vstack([flat_x, flat_y]).T.reshape((-1, 1, 2))
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask

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