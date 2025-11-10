import numpy as np

def _finite_xy(arr):
    if arr is None:
        return np.empty((0,2), float)
    a = np.asarray(arr, float)
    if a.ndim != 2 or a.shape[1] != 2:
        return np.empty((0,2), float)
    m = np.all(np.isfinite(a), axis=1)
    a = a[m]
    # drop exact duplicates in sequence
    if len(a) > 1:
        keep = [0]
        for i in range(1, len(a)):
            if not (abs(a[i,0]-a[i-1,0]) < 1e-12 and abs(a[i,1]-a[i-1,1]) < 1e-12):
                keep.append(i)
        a = a[keep]
    return a
