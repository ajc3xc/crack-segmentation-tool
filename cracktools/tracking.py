import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import os
import psutil
# import sys

# Prevent accidental giant ndarray dumps in debug prints.
np.set_printoptions(threshold=32, edgeitems=2, linewidth=120)

'''try:
    import cupy as cp
    try:
        _ = cp.cuda.runtime.getDeviceCount()
        if _ > 0:
            CUPY_AVAILABLE = True
        else:
            raise RuntimeError("No CUDA device found")
    except Exception:
        import numpy as np
        cp = np
        CUPY_AVAILABLE = False
except ImportError:
    import numpy as np
    cp = np
    CUPY_AVAILABLE = False
   
def asnumpy(x):
    if CUPY_AVAILABLE:
        return cp.asnumpy(x)
    return x'''
    
try:
    import cupy as cp
    try:
        n_devices = cp.cuda.runtime.getDeviceCount()
        if n_devices > 0:
            force_cupy = os.environ.get("CRACKTOOLS_FORCE_CUPY", "0") == "1"
            rt_ver = int(cp.cuda.runtime.runtimeGetVersion())
            if (rt_ver >= 12090) and (not force_cupy):
                raise RuntimeError(
                    "CUDA runtime >= 12.9 detected; forcing CPU fallback "
                    "(set CRACKTOOLS_FORCE_CUPY=1 to override)."
                )
            CUPY_AVAILABLE = True

            # -------------------------------
            # 💾 GPU + Pinned Memory Pools
            # -------------------------------
            from cupy.cuda import pinned_memory

            # Device (VRAM) pool
            gpu_pool = cp.cuda.MemoryPool()
            cp.cuda.set_allocator(gpu_pool.malloc)

            # Host (CPU) pinned-memory pool
            pin_pool = pinned_memory.PinnedMemoryPool()
            pinned_memory.set_pinned_memory_allocator(pin_pool.malloc)

            # 👇 New universal safe limit setter
            if hasattr(pin_pool, "set_limit"):
                pin_pool.set_limit(size=1024 * 1024 * 1024)  # 1 GB
                pinned_limit = pin_pool.get_limit() / (1024**2)
            else:
                pinned_limit = "auto"

            #print(f"[cupy] ✅ CUDA detected ({n_devices} device(s))")
            #print(f"[cupy] Memory pools active (pinned_limit={pinned_limit} MiB)")

        else:
            raise RuntimeError("No CUDA device found")

    except Exception as e:
        print(f"[cupy] [WARN] GPU unavailable or failed ({e}) - falling back to NumPy")
        import numpy as np
        cp = np
        CUPY_AVAILABLE = False

except ImportError:
    print("[cupy] [WARN] CuPy not installed - using NumPy fallback")
    import numpy as np
    cp = np
    CUPY_AVAILABLE = False


def asnumpy(x):
    """Safe conversion to NumPy array whether x is CuPy or NumPy."""
    if CUPY_AVAILABLE and hasattr(cp, "asnumpy"):
        return cp.asnumpy(x)
    return x


def free_cupy_caches():
    """Manual cleanup — free all cached GPU + pinned blocks between images."""
    if not CUPY_AVAILABLE:
        return
    try:
        cp.get_default_memory_pool().free_all_blocks()
        if hasattr(cp.cuda.pinned_memory, "get_default_pinned_memory_pool"):
            cp.cuda.pinned_memory.get_default_pinned_memory_pool().free_all_blocks()
    except Exception as e:
        print(f"[cupy] cleanup warning: {e}")

def _rss_gib():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)

def _mem_point(tag, arr=None):
    rss = _rss_gib()
    msg = f"[MEM] {tag} | RSS={rss:.3f} GiB"
    if arr is not None:
        shape = getattr(arr, "shape", None)
        dtype = getattr(arr, "dtype", None)
        nbytes = getattr(arr, "nbytes", None)
        if shape is not None and dtype is not None and nbytes is not None:
            msg += f" | shape={shape} dtype={dtype} size={nbytes/(1024**3):.3f} GiB"
        else:
            msg += f" | type={type(arr).__module__}.{type(arr).__name__}"
    print(msg)

def mem_snapshot(tag, arrays=None, cupy_mod=None, do_gc=False):
    """
    Lightweight memory snapshot for debugging OOM retries.
    arrays: optional iterable of (name, arr)
    """
    import gc

    if do_gc:
        gc.collect()

    print(f"[MEMSNAP] {tag}")
    _mem_point("rss")

    if arrays:
        for name, arr in arrays:
            _mem_point(name, arr)

    cm = cupy_mod if cupy_mod is not None else (cp if CUPY_AVAILABLE else None)
    if cm is not None and hasattr(cm, "cuda"):
        try:
            free, total = cm.cuda.runtime.memGetInfo()
            print(f"[MEMSNAP][GPU] free={free/(1024**3):.2f} GiB total={total/(1024**3):.2f} GiB")
            try:
                print(f"[MEMSNAP][GPU] pool_used={cm.get_default_memory_pool().used_bytes()/(1024**3):.2f} GiB")
            except Exception:
                pass
            try:
                pinned_pool = cm.get_default_pinned_memory_pool()
                if hasattr(pinned_pool, "used_bytes"):
                    print(f"[MEMSNAP][GPU] pinned_used={pinned_pool.used_bytes()/(1024**3):.2f} GiB")
                elif hasattr(pinned_pool, "n_free_blocks"):
                    print(f"[MEMSNAP][GPU] pinned_free_blocks={pinned_pool.n_free_blocks()}")
            except Exception:
                pass
        except Exception as e:
            print(f"[MEMSNAP][GPU] unavailable: {e}")

def _format_exception_brief(e, max_chars=600):
    msg = f"{type(e).__name__}: {e}"
    msg = " ".join(str(msg).split())
    if len(msg) > max_chars:
        msg = msg[:max_chars] + " ... [truncated]"
    return msg


def asnumpy(x):
    """Safe conversion to NumPy array whether x is CuPy or NumPy."""
    if CUPY_AVAILABLE and hasattr(cp, "asnumpy"):
        return cp.asnumpy(x)
    return x


def free_cupy_caches():
    """Manual cleanup — free all cached GPU + pinned blocks between images."""
    if not CUPY_AVAILABLE:
        return
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.pinned_memory.get_default_pinned_memory_pool().free_all_blocks()
        cp._default_memory_pool = None
    except Exception as e:
        print(f"[cupy] cleanup warning: {e}")

def tang_len(start_point_x,start_point_y,end_point_x,end_point_y):
    """Function defines oriantation and direction of line that connects two points"""
    dx = end_point_x - start_point_x
    dy = end_point_y - start_point_y
    l = np.sqrt(dx**2+dy**2)
    ddx = dx/l
    ddy = dy/l
    return ddx,ddy,l

from agd import Eikonal
from agd.Metrics import AsymQuad, Riemann  # Riemannian metric and Asymmetric Quadratic Models
from agd import AutomaticDifferentiation as ad
from agd import LinearParallel as lp
from agd import FiniteDifferences as fd
from agd import Eikonal

from agd.LinearParallel import outer_self as Outer  # outer product v \v^T of a vector with itself
norm = ad.Optimization.norm
import numpy as np; xp = np

# ---------------------------------------------------------------------
# Safety guard: CUPY_AVAILABLE may or may not exist in this module
# ---------------------------------------------------------------------
try:
    CUPY_AVAILABLE
except NameError:
    CUPY_AVAILABLE = False

# ---------------------------------------------------------------------
# Shared helpers (unchanged math)
# ---------------------------------------------------------------------
def GGF(g11, g22, g33, GFtoLIFinv, LIFtoEuclideaninv):
    GF = np.diag([g11, g22, g33])
    transformMatrix = np.dot(LIFtoEuclideaninv, GFtoLIFinv)
    G = np.dot(transformMatrix, np.dot(GF, transformMatrix.T))
    return G

def GLIFtoEuclideanOld(theta):
    return np.array([
        [ np.cos(theta),  np.sin(theta), 0],
        [-np.sin(theta),  np.cos(theta), 0],
        [ 0            ,  0            , 1]
    ])

def GLIFtoEuclideanOld_vec(nt):
    """
    Vectorized version of GLIFtoEuclideanOld for all t.
    Returns [nt, 3, 3].
    """
    t = np.arange(nt) * 2 * np.pi / nt
    LIF = np.zeros((nt, 3, 3))
    LIF[:, 0, 0] = np.cos(t)
    LIF[:, 0, 1] = np.sin(t)
    LIF[:, 1, 0] = -np.sin(t)
    LIF[:, 1, 1] = np.cos(t)
    LIF[:, 2, 2] = 1.0
    return LIF

# ---------------------------------------------------------------------
# Original (unoptimized) metric builder – for "old_unoptimized"
# ---------------------------------------------------------------------
def ReedsSheppMetricGFOld_naive(GF, dims, g11, g22, g33):
    """
    Original nested-loop implementation.
    Returns metric[t, x, y, i, j] of shape (nt, nx, ny, 3, 3).
    """
    nx = dims[1]
    ny = dims[2]
    nt = dims[0]

    GFinv = GF  # inverse of identity matrix. much faster this way
    LIFtoEuclidean = np.zeros((dims[0], 3, 3))
    for t in range(0, nt):
        LIFtoEuclidean[t, :, :] = GLIFtoEuclideanOld(t * 2 * np.pi / nt)

    LIFtoEuclideaninv = np.array(
        [np.linalg.inv(LIFtoEuclidean[i]) for i in range(LIFtoEuclidean.shape[0])]
    )
    metric = np.zeros((dims[0], dims[1], dims[2], 3, 3))
    for t in range(nt):
        for x in range(nx):
            for y in range(ny):
                metric[t, x, y, :, :] = GGF(
                    g11, g22, g33, GFinv[t, x, y], LIFtoEuclideaninv[t, :, :]
                )
    return metric

# ---------------------------------------------------------------------
# Optimized metric builder – same math, less memory – for "old_optimized"/"new_optimized"
# ---------------------------------------------------------------------
def ReedsSheppMetricGFOld_vec(GF, dims, g11, g22, g33):
    """
    Memory-efficient version of ReedsSheppMetricGFOld.
    Assumes GF is unused (typically identity) and omits full spatial tiling.

    Returns:
        (nt, 3, 3) metric tensor; later broadcast over (x,y).
    """
    nt, nx, ny = dims[0], dims[1], dims[2]
    LIFtoEuclidean = GLIFtoEuclideanOld_vec(nt)   # (nt, 3, 3)
    LIFtoEuclideaninv = np.linalg.inv(LIFtoEuclidean)

    # Diagonal GF metric
    GFmat = np.diag([g11, g22, g33])

    # Compose per-theta metric: M_t = LIFinv @ GF @ LIFinv^T
    M = LIFtoEuclideaninv @ GFmat @ np.transpose(LIFtoEuclideaninv, (0, 2, 1))  # (nt, 3, 3)

    return M

# ---------------------------------------------------------------------
# Original IncludeCost – expansion implementation (old_unoptimized)
# ---------------------------------------------------------------------
def IncludeCost_naive(cost, metric):
    """
    Original cost inclusion:
      - cost is (nt, nx, ny)
      - internally squared again → overall cost^4 if upstream passes os_cost**2
    """
    cost = cost**2
    cost = np.expand_dims(cost, axis=3)
    cost = np.concatenate([cost, cost, cost], axis=3)
    cost = np.expand_dims(cost, axis=4)
    cost = np.concatenate([cost, cost, cost], axis=4)
    metric = metric * cost
    return metric  # shape (nt, nx, ny, 3, 3)

# ---------------------------------------------------------------------
# Optimized IncludeCost – broadcast implementation (old/new optimized)
# float32 + cp-aware
# ---------------------------------------------------------------------
def IncludeCost(cost, metric):
    """
    Broadcast-based IncludeCost (float32, cp-aware)

      - cost is (nt, nx, ny)
      - metric is:
          * (nt, 3, 3)  OR
          * (nt, nx, ny, 3, 3)
      - returns (nt, nx, ny, 3, 3) as float32
    """

    # Force float32 on active backend (CuPy or NumPy fallback)
    cost = cp.asarray(cost, dtype=cp.float32)
    metric = cp.asarray(metric, dtype=cp.float32)

    # cost^2 (still float32)
    cost_sq = cost * cost

    # Expand to (nt, nx, ny, 1, 1)
    cost_exp = cost_sq[..., None, None]

    if metric.ndim == 3:
        # (nt, 3, 3) → broadcast over x,y
        metric_exp = metric[:, None, None, :, :]
    else:
        # already (nt, nx, ny, 3, 3)
        metric_exp = metric

    out = cost_exp * metric_exp

    # Ensure float32 output (avoid accidental upcast)
    return out.astype(cp.float32, copy=False)

'''# ---------------------------------------------------------------------
# Riemann3_Periodic solver wrapper (unchanged except verbosity)
# ---------------------------------------------------------------------
def runReedsSheppGF(sides, dims, seeds, tips, metric, solver_dtype="float32"):

    # right before metric.get()
    import gc
    gc.collect()
    if CUPY_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    
    # If metric is CuPy → move to CPU explicitly
    if CUPY_AVAILABLE and isinstance(metric, cp.ndarray):
        metric = metric.get()  # GPU → CPU copy

    # AGD/HFMIO bindings expect a CPU NumPy float64 metric array.
    metric = np.ascontiguousarray(metric, dtype=np.float64)
    metric = Riemann(metric)

    hfmIn = Eikonal.dictIn({
        'model'        : 'Riemann3_Periodic',
        'seeds'        : seeds,
        'arrayOrdering': 'RowMajor',
        'tips'         : tips,
        'metric'       : metric,
        'verbosity'    : 0,
    })

    hfmIn.SetRect(sides=sides, dims=dims)
    hfmOut = hfmIn.Run()

    geos = [g.T for g in hfmOut['geodesics']]
    print('Done.')
    return geos

# ---------------------------------------------------------------------
# Ablation-ready fast_marching
# ---------------------------------------------------------------------
from time import time

def fast_marching(
    os_cost,
    start_point,
    end_point,
    g11=1,
    g22=25,
    g33=25,
    *,
    mode="new_optimized",       # "old_unoptimized" | "old_optimized" | "new_optimized"
    return_subtiming=False
):
    """
    Reeds–Shepp fast marching for midline tracking (Riemann3_Periodic).

    Modes
    -----
    - "old_unoptimized":
        * EXACT original behavior:
          - ReedsSheppMetricGFOld_naive (nested loops)
          - IncludeCost_naive (concat-based)
          - default anisotropy if g22/g33 not overridden: 100
    - "old_optimized":
        * Same math as original, but:
          - ReedsSheppMetricGFOld_vec (vectorized over theta)
          - IncludeCost (broadcast)
          - default g22=g33 forced to 100 if left at 25
    - "new_optimized" (default):
        * Same math as optimized, but:
          - default g22=g33 = 25 (your "new" behavior)

    Returns
    -------
    - If return_subtiming is False:
        [x_coords, y_coords]
    - If return_subtiming is True:
        ([x_coords, y_coords], subtiming_dict)
    """
    # -----------------------------------
    # Basic setup (same for all modes)
    # -----------------------------------
    t_all0 = time()
    _mem_point("FM entry", os_cost)

    NxCost = os_cost.shape[1]
    NyCost = os_cost.shape[2]
    NoCost = os_cost.shape[0]
    s_theta = 2 * np.pi / NoCost

    gfLIF = np.zeros((NoCost, NxCost, NyCost, 3, 3), dtype=np.float32)
    gfLIF[:, :, :, 0, 0] = 1.0
    gfLIF[:, :, :, 1, 1] = 1.0
    gfLIF[:, :, :, 2, 2] = 1.0
    _mem_point("After gfLIF alloc", gfLIF)
    
    dims = np.array([NoCost, NxCost, NyCost])
    sidesLIFmetric = np.array([[0, NxCost], [0, NyCost], [0, 2 * np.pi - s_theta]])
    print("Initial constructions done")

    # match old defaults for "old_*" modes when user leaves g22,g33 at new defaults
    if mode in ("old_unoptimized", "old_optimized") and g22 == 25 and g33 == 25:
        g22_local = 100
        g33_local = 100
    else:
        g22_local = g22
        g33_local = g33

    # we always pass cost**2 into IncludeCost, as in the original code
    cost_sq_input = os_cost**2
    _mem_point("After cost_sq_input", cost_sq_input)

    # timing placeholders
    t_metric = 0.0
    t_include = 0.0
    t_transpose = 0.0
    t_solver = 0.0

    # -----------------------------------
    # Metric build (per mode)
    # -----------------------------------
    if mode == "old_unoptimized":
        # 1) metric build – original nested implementation
        t0 = time()
        metricLIFOld = ReedsSheppMetricGFOld_naive(gfLIF, dims, g11, g22_local, g33_local)
        t_metric = time() - t0
        print(f"[fast_marching:{mode}] ReedsSheppMetricGFOld_naive = {t_metric:.4f} s")
        _mem_point("After metric build", metricLIFOld)

        # GC / GPU pool cleanup (optional, non-algorithmic)
        import gc
        gc.collect()
        if CUPY_AVAILABLE:
            import cupy as cp
            mempool = cp.get_default_memory_pool()
            pinned = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned.free_all_blocks()

        # 2) Include cost – original expansion
        t0 = time()
        metricLIFinclCostOld = IncludeCost_naive(cost_sq_input, metricLIFOld)
        t_include = time() - t0
        print(f"[fast_marching:{mode}] IncludeCost_naive = {t_include:.4f} s")
        _mem_point("After include cost", metricLIFinclCostOld)

    elif mode == "old_optimized":
        # 1) metric build – vectorized over theta
        t0 = time()
        metric_theta = ReedsSheppMetricGFOld_vec(gfLIF, dims, g11, g22_local, g33_local)  # (nt,3,3)
        t_metric = time() - t0
        print(f"[fast_marching:{mode}] ReedsSheppMetricGFOld_vec = {t_metric:.4f} s")
        _mem_point("After metric build", metric_theta)

        import gc
        gc.collect()
        if CUPY_AVAILABLE:
            import cupy as cp
            mempool = cp.get_default_memory_pool()
            pinned = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned.free_all_blocks()

        # 2) Include cost – broadcast version
        t0 = time()
        metricLIFinclCostOld = IncludeCost(cost_sq_input, metric_theta)  # (nt,nx,ny,3,3)
        t_include = time() - t0
        print(f"[fast_marching:{mode}] IncludeCost(broadcast) = {t_include:.4f} s")
        _mem_point("After include cost", metricLIFinclCostOld)

    elif mode == "new_optimized":
        # Same as old_optimized, but with "new" defaults (g22=g33=25 unless overridden)
        t0 = time()
        metric_theta = ReedsSheppMetricGFOld_vec(gfLIF, dims, g11, g22_local, g33_local)
        t_metric = time() - t0
        print(f"[fast_marching:{mode}] ReedsSheppMetricGFOld_vec = {t_metric:.4f} s")
        _mem_point("After metric build", metric_theta)

        import gc
        gc.collect()
        if CUPY_AVAILABLE:
            import cupy as cp
            mempool = cp.get_default_memory_pool()
            pinned = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned.free_all_blocks()

        t0 = time()
        metricLIFinclCostOld = IncludeCost(cost_sq_input, metric_theta)  # (nt,nx,ny,3,3)
        t_include = time() - t0
        print(f"[fast_marching:{mode}] IncludeCost(broadcast) = {t_include:.4f} s")
        _mem_point("After include cost", metricLIFinclCostOld)

    else:
        raise ValueError(f"Unknown fast_marching mode: {mode}")

    # -----------------------------------
    # Transpose / reshape for AGD API
    # -----------------------------------
    t0 = time()
    metricLIFinclCostOld1 = metricLIFinclCostOld.transpose((3, 4, 1, 2, 0))  # (3,3,Nx,Ny,No)
    t_transpose = time() - t0
    print(f"[fast_marching:{mode}] transpose = {t_transpose:.4f} s")
    _mem_point("After transpose", metricLIFinclCostOld1)

    a = np.array([0, 2 * np.pi]) - s_theta / 2
    b = np.array([0, NxCost])
    c = np.array([0, NyCost])
    sides = np.array([b, c, a])

    seeds = np.array([*start_point[::-1], np.pi / 2])
    tips = np.array([*end_point[::-1], np.pi / 2])

    # The AGD Riemann3_Periodic expects dims in (nx, ny, nt) order
    dims_agd = [dims[1], dims[2], dims[0]]

    # -----------------------------------
    # Run solver
    # -----------------------------------
    _mem_point("Before solver", metricLIFinclCostOld1)
    t0 = time()
    geos1 = runReedsSheppGF(sides, dims_agd, [seeds], [tips], metricLIFinclCostOld1)
    t_solver = time() - t0
    print(f"[fast_marching:{mode}] runReedsSheppGF = {t_solver:.4f} s")
    _mem_point("After solver")

    # -----------------------------------
    # Prepare outputs
    # -----------------------------------
    path = [geos1[0][:, 1], geos1[0][:, 0]]  # [x, y]
    t_total = time() - t_all0

    subtiming = {
        "fm_mode": mode,
        "fm_metric_build_sec": float(t_metric),
        "fm_include_cost_sec": float(t_include),
        "fm_transpose_sec": float(t_transpose),
        "fm_solver_sec": float(t_solver),
        "fm_total_sec": float(t_total),
    }

    if return_subtiming:
        return path, subtiming
    else:
        return path'''
# ---------------------------------------------------------------------
# Riemann3_Periodic solver wrapper (memory-stable)
# ---------------------------------------------------------------------
def runReedsSheppGF(sides, dims, seeds, tips, metric, solver_dtype):
    """
    Expects:
        metric: NumPy/CuPy array, shape (3,3,Nx,Ny,No)
        solver_dtype: "float32" or "float64" (CPU dtype passed to AGD)
    """

    import gc
    gc.collect()

    if solver_dtype not in ("float32", "float64"):
        raise ValueError(f"solver_dtype must be 'float32' or 'float64', got {solver_dtype!r}")

    # If metric is CuPy, transfer once in current dtype; cast on CPU as needed.
    if CUPY_AVAILABLE and hasattr(cp, "ndarray") and isinstance(metric, cp.ndarray):
        metric_np = metric.get(order="C")
    else:
        metric_np = metric

    target_dtype = np.float32 if solver_dtype == "float32" else np.float64
    metric_np = np.ascontiguousarray(metric_np, dtype=target_dtype)
    gc.collect()

    try:
        metric = Riemann(metric_np)

        hfmIn = Eikonal.dictIn({
            'model'        : 'Riemann3_Periodic',
            'seeds'        : seeds,
            'arrayOrdering': 'RowMajor',
            'tips'         : tips,
            'metric'       : metric,
            'verbosity'    : 0,
        })

        hfmIn.SetRect(sides=sides, dims=dims)
        hfmOut = hfmIn.Run()
    except TypeError as e:
        msg = str(e)
        if "set_array()" in msg and "'metric'" in msg:
            shape = getattr(metric_np, "shape", None)
            dtype = getattr(metric_np, "dtype", None)
            cflag = bool(metric_np.flags["C_CONTIGUOUS"]) if hasattr(metric_np, "flags") else None
            raise TypeError(
                f"HFM metric type mismatch for set_array('metric'): "
                f"shape={shape}, dtype={dtype}, C_CONTIGUOUS={cflag}, solver_dtype={solver_dtype}. "
                f"The binding expects numpy.ndarray[numpy.float64]."
            ) from None
        raise

    geos = [g.T for g in hfmOut['geodesics']]
    print("Done.")
    return geos

from time import time
import numpy as np
import gc

def fast_marching(
    os_cost,
    start_point,
    end_point,
    g11=1,
    g22=25,
    g33=25,
    *,
    mode="new_optimized",
    return_subtiming=False,
    solver_dtype="float32",
):
    t_all0 = time()
    _mem_point("FM entry", os_cost)

    # Ensure float32 early (preserve backend if already CuPy)
    if CUPY_AVAILABLE and hasattr(cp, "ndarray") and isinstance(os_cost, cp.ndarray):
        os_cost = os_cost.astype(cp.float32, copy=False)
    else:
        os_cost = np.asarray(os_cost, dtype=np.float32, order="C")
    _mem_point("FM os_cost float32", os_cost)

    NxCost = os_cost.shape[1]
    NyCost = os_cost.shape[2]
    NoCost = os_cost.shape[0]
    s_theta = 2 * np.pi / NoCost

    dims = np.array([NoCost, NxCost, NyCost], dtype=np.int32)
    print("Initial constructions done")

    if mode in ("old_unoptimized", "old_optimized") and g22 == 25 and g33 == 25:
        g22_local = 100
        g33_local = 100
    else:
        g22_local = g22
        g33_local = g33

    t_metric = t_include = t_transpose = t_solver = 0.0

    # -----------------------------------
    # Metric build
    # -----------------------------------

    if mode == "old_unoptimized":
        # Only allocate gfLIF for this mode
        gfLIF = np.zeros((NoCost, NxCost, NyCost, 3, 3), dtype=np.float32)
        gfLIF[:, :, :, 0, 0] = 1.0
        gfLIF[:, :, :, 1, 1] = 1.0
        gfLIF[:, :, :, 2, 2] = 1.0
        _mem_point("After gfLIF alloc", gfLIF)

        cost_sq_input = os_cost * os_cost
        _mem_point("After cost_sq_input", cost_sq_input)

        t0 = time()
        metricLIFOld = ReedsSheppMetricGFOld_naive(gfLIF, dims, g11, g22_local, g33_local)
        t_metric = time() - t0
        _mem_point("After metric build", metricLIFOld)

        del gfLIF
        gc.collect()

        t0 = time()
        metricLIFinclCostOld = IncludeCost_naive(cost_sq_input, metricLIFOld)
        t_include = time() - t0
        _mem_point("After include cost", metricLIFinclCostOld)

        del metricLIFOld, cost_sq_input
        gc.collect()

    else:
        # Optimized modes: NO gfLIF allocation
        cost_sq_input = os_cost * os_cost
        _mem_point("After cost_sq_input", cost_sq_input)

        t0 = time()
        metric_theta = ReedsSheppMetricGFOld_vec(
            None, dims, g11, g22_local, g33_local
        )
        t_metric = time() - t0
        _mem_point("After metric build", metric_theta)

        # Build the large metric tensor on CPU (NumPy) because the solver is CPU-only.
        # This avoids a later fragile CuPy -> NumPy transfer of the full 5D tensor.
        t0 = time()
        if CUPY_AVAILABLE and hasattr(cp, "ndarray"):
            cost_sq_np = cost_sq_input.get() if isinstance(cost_sq_input, cp.ndarray) else np.asarray(cost_sq_input, dtype=np.float32)
            metric_theta_np = metric_theta.get() if isinstance(metric_theta, cp.ndarray) else np.asarray(metric_theta, dtype=np.float32)
        else:
            cost_sq_np = np.asarray(cost_sq_input, dtype=np.float32)
            metric_theta_np = np.asarray(metric_theta, dtype=np.float32)

        cost_sq_np = np.ascontiguousarray(cost_sq_np, dtype=np.float32)
        metric_theta_np = np.ascontiguousarray(metric_theta_np, dtype=np.float32)
        metricLIFinclCostOld = (cost_sq_np[..., None, None] * metric_theta_np[:, None, None, :, :]).astype(np.float32, copy=False)
        t_include = time() - t0
        _mem_point("After include cost", metricLIFinclCostOld)

        del metric_theta, cost_sq_input, cost_sq_np, metric_theta_np
        gc.collect()
        if CUPY_AVAILABLE:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass

    # -----------------------------------
    # Transpose for solver
    # -----------------------------------
    t0 = time()
    metric_view = metricLIFinclCostOld.transpose((3, 4, 1, 2, 0))
    t_transpose = time() - t0
    _mem_point("After transpose (view)", metric_view)

    # Free float32 base tensor immediately
    del metricLIFinclCostOld
    gc.collect()

    # -----------------------------------
    # Build solver args
    # -----------------------------------
    solver_np_dtype = np.float32 if solver_dtype == "float32" else np.float64
    a = np.array([0, 2 * np.pi], dtype=solver_np_dtype) - s_theta / 2
    b = np.array([0, NxCost], dtype=solver_np_dtype)
    c = np.array([0, NyCost], dtype=solver_np_dtype)
    sides = np.array([b, c, a], dtype=solver_np_dtype)

    seeds = np.array([*start_point[::-1], np.pi / 2], dtype=solver_np_dtype)
    tips  = np.array([*end_point[::-1],   np.pi / 2], dtype=solver_np_dtype)
    dims_agd = [int(dims[1]), int(dims[2]), int(dims[0])]

    # -----------------------------------
    # Solver-boundary metric prep
    # - float32: keep metric_view on backend and let runReedsSheppGF transfer once
    # - float64: move to CPU float32 first, then chunk-cast on CPU (avoid VRAM spike)
    # -----------------------------------
    def _is_cupy_array(x):
        return CUPY_AVAILABLE and hasattr(cp, "ndarray") and isinstance(x, cp.ndarray)

    def _free_gpu_pools():
        if CUPY_AVAILABLE:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass

    metric_for_solver = metric_view
    if solver_dtype == "float64":
        _mem_point("Before metric CPU move", metric_view)

        if _is_cupy_array(metric_view):
            metric32_cpu = metric_view.get(order="C")
            del metric_view
            _free_gpu_pools()
        else:
            metric32_cpu = np.ascontiguousarray(metric_view, dtype=np.float32)
            del metric_view

        gc.collect()
        _mem_point("After metric CPU move (float32)", metric32_cpu)

        _mem_point("Before float64 cast", metric32_cpu)
        metric64 = np.empty(metric32_cpu.shape, dtype=np.float64, order="C")
        n_theta = metric32_cpu.shape[-1]
        for t in range(n_theta):
            metric64[..., t] = metric32_cpu[..., t].astype(np.float64, copy=False)

        del metric32_cpu
        gc.collect()

        _mem_point("After float64 cast (CPU)", metric64)
        metric_for_solver = metric64

    # -----------------------------------
    # Run solver
    # -----------------------------------
    _mem_point("Before solver (metric_view)", metric_for_solver)
    t0 = time()
    geos1 = runReedsSheppGF(
        sides, dims_agd, [seeds], [tips], metric_for_solver, solver_dtype=solver_dtype
    )
    t_solver = time() - t0
    print(f"[fast_marching:{mode}] runReedsSheppGF = {t_solver:.4f} s")

    # Free solver metric buffers immediately
    del metric_for_solver
    if 'metric_view' in locals():
        del metric_view
    if 'metric64' in locals():
        del metric64
    gc.collect()

    path = [geos1[0][:, 1], geos1[0][:, 0]]
    t_total = time() - t_all0

    subtiming = {
        "fm_mode": mode,
        "fm_metric_build_sec": float(t_metric),
        "fm_include_cost_sec": float(t_include),
        "fm_transpose_sec": float(t_transpose),
        "fm_solver_sec": float(t_solver),
        "fm_total_sec": float(t_total),
    }

    return (path, subtiming) if return_subtiming else path

def fast_marching_with_fallback(
    os_cost,
    start_point,
    end_point,
    *,
    g11,
    g22,
    g33,
    mode="new_optimized",
    solver_dtype="float32",
    max_downsample_attempts=(1, 2, 4),
):
    """
    Try fast marching at full resolution and retry with spatial downsampling on OOM.
    """
    import gc

    last_exception = None
    np_oom_type = getattr(getattr(np, "core", None), "_exceptions", None)
    np_oom_type = getattr(np_oom_type, "_ArrayMemoryError", ())

    for ds in max_downsample_attempts:
        try:
            print(f"\n=== FAST MARCHING ATTEMPT (downsample={ds}) ===")

            if ds == 1:
                cost_try = os_cost
                sp_try = start_point
                ep_try = end_point
            else:
                cost_try = os_cost[:, ::ds, ::ds]
                sp_try = (start_point[0] / ds, start_point[1] / ds)
                ep_try = (end_point[0] / ds, end_point[1] / ds)

            mem_snapshot(
                f"Before FM attempt ds={ds}",
                arrays=[("cost_try", cost_try)],
                cupy_mod=(cp if CUPY_AVAILABLE else None),
                do_gc=True,
            )

            path = fast_marching(
                cost_try,
                sp_try,
                ep_try,
                g11=g11,
                g22=g22,
                g33=g33,
                mode=mode,
                solver_dtype=solver_dtype,
            )

            if ds != 1:
                path = [np.asarray(path[0]) * ds, np.asarray(path[1]) * ds]

            print(f"Fast marching succeeded at downsample={ds}")
            return path

        except Exception as e:
            last_exception = e
            print(f"\nFast marching failed at downsample={ds}")
            print(f"Exception: {_format_exception_brief(e)}")

            mem_snapshot(
                f"After failure ds={ds}",
                cupy_mod=(cp if CUPY_AVAILABLE else None),
                do_gc=True,
            )

            if CUPY_AVAILABLE:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                except Exception:
                    pass

            gc.collect()

            msg = str(e)
            is_oom = (
                (np_oom_type and isinstance(e, np_oom_type))
                or ("OutOfMemory" in msg)
                or ("Unable to allocate" in msg)
            )
            if not is_oom:
                print("Non-memory exception, not retrying.")
                raise

            print("Retrying with next downsample level...")

    print("All fast marching attempts failed.")
    raise last_exception

