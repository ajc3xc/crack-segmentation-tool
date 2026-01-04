import numpy as np

def _resample_xy_by_arclen(xy, N=400):
    xy = np.asarray(xy, float)
    if len(xy) < 2: return xy
    d = np.hypot(np.diff(xy[:,0]), np.diff(xy[:,1]))
    s = np.concatenate([[0], np.cumsum(d)])
    if s[-1] <= 0: return xy
    t = np.linspace(0, s[-1], min(N, len(xy)))
    x = np.interp(t, s, xy[:,0]); y = np.interp(t, s, xy[:,1])
    return np.column_stack([x, y])


def _rms_curvature(xy):
    import numpy as np
    xy = np.asarray(xy, float)
    n = len(xy)
    if n < 3:
        return float('nan')

    # arc-length parameterization
    ds = np.hypot(np.diff(xy[:,0]), np.diff(xy[:,1]))
    s = np.concatenate([[0], np.cumsum(ds)])   # len = n
    if s[-1] <= 0:
        return float('nan')

    dx = np.gradient(xy[:,0], s, edge_order=2)
    dy = np.gradient(xy[:,1], s, edge_order=2)
    ddx = np.gradient(dx, s, edge_order=2)
    ddy = np.gradient(dy, s, edge_order=2)

    num = np.abs(dx*ddy - dy*ddx)
    den = (dx*dx + dy*dy)**1.5 + 1e-12
    kappa = num / den
    return float(np.sqrt(np.nanmean(kappa**2)))


def _orth_stats(orth_dev_arr):
    """
    Accepts either a NumPy array of orthogonal deviations or a dict
    returned by orthogonal_deviation(). Extracts numeric values
    robustly and returns summary stats including signed_bias_z.
    """
    import numpy as np

    # unwrap dict form (e.g. {"orth_dev": [...]} or {"array": [...]} etc.)
    if isinstance(orth_dev_arr, dict):
        # try common keys
        for k in ("orth_dev", "values", "array", "data"):
            if k in orth_dev_arr:
                orth_dev_arr = orth_dev_arr[k]
                break
        # if dict values are numeric scalars, flatten them
        if isinstance(orth_dev_arr, dict):
            orth_dev_arr = list(orth_dev_arr.values())

    # convert to array
    a = np.asarray(orth_dev_arr, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return dict(
            orth_mean=np.nan,
            orth_mean_abs=np.nan,
            orth_std=np.nan,
            signed_bias_z=np.nan
        )

    mu = float(np.mean(a))
    sd = float(np.std(a) + 1e-12)
    return {
        "orth_mean": mu,
        "orth_mean_abs": float(np.mean(np.abs(a))),
        "orth_std": sd,
        # signed normalized bias
        "signed_bias_z": float(np.sign(mu) * (abs(mu)/sd))
    }


def _split_bins_by_arclen(xy, n_bins=5):
    xy = np.asarray(xy, float)
    if len(xy) < 2: return [xy]
    d = np.hypot(np.diff(xy[:,0]), np.diff(xy[:,1]))
    s = np.concatenate([[0], np.cumsum(d)])
    edges = np.linspace(0, s[-1], n_bins+1)
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        idx = np.where((s >= lo) & (s <= hi))[0]
        if len(idx) >= 2:
            bins.append(xy[idx])
        else:
            j = np.searchsorted(s, (lo+hi)/2)
            j0 = max(0, j-1); j1 = min(len(xy)-1, j+1)
            if j1 > j0:
                bins.append(xy[j0:j1+1])
    return [b for b in bins if len(b) >= 2]


def _widths_from_normal_pairs(normals):
    """
    normals (crop or full coords): [[e1x,e1y],[e2x,e2y]]
    returns width array (NaN where missing)
    """
    if normals is None: return np.array([])
    (e1x, e1y), (e2x, e2y) = normals
    e1 = np.column_stack([np.asarray(e1x,float), np.asarray(e1y,float)])
    e2 = np.column_stack([np.asarray(e2x,float), np.asarray(e2y,float)])
    ok = np.isfinite(e1).all(axis=1) & np.isfinite(e2).all(axis=1)
    w = np.full(len(e1), np.nan)
    if ok.any():
        w[ok] = np.hypot(e1[ok,0]-e2[ok,0], e1[ok,1]-e2[ok,1])
    return w


def _pearson_nan(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = min(a.size, b.size)
    if n == 0: return float('nan')
    a = a[:n]; b = b[:n]
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3: return float('nan')
    a = (a[ok] - a[ok].mean())/(a[ok].std()+1e-12)
    b = (b[ok] - b[ok].mean())/(b[ok].std()+1e-12)
    return float(np.mean(a*b))