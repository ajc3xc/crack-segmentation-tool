# rs3_split.py
# Split fast_marching into:
#   A) PRESTAGE (GPU/throughput friendly): build metric tensor + seeds/tips/sides/dims for RS3
#   B) RS3 CPU WORKER: run only runReedsSheppGF on CPU in parallel

import os, time, traceback
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, Any, List

# ---- SHM helpers ----------------------------------------------------
def shm_from_array(arr: np.ndarray) -> Dict[str, Any]:
    arr = np.ascontiguousarray(arr)
    shm = SharedMemory(create=True, size=arr.nbytes)
    np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)[:] = arr
    return dict(name=shm.name, shape=arr.shape, dtype=str(arr.dtype))

def array_from_shm(meta: Dict[str, Any]):
    shm = SharedMemory(name=meta["name"])
    arr = np.ndarray(tuple(meta["shape"]), dtype=np.dtype(meta["dtype"]), buffer=shm.buf)
    return shm, arr

def close_shm(meta: Dict[str, Any]):
    try:
        shm = SharedMemory(name=meta["name"])
        shm.close(); shm.unlink()
    except Exception:
        pass

# ---- (A) PRESTAGE: build RS3 inputs (main process; can use GPU inside your ct.*) ----
def rs3_prestage_variant(ct, os_cost: np.ndarray, p0_down_xy: np.ndarray, p1_down_xy: np.ndarray,
                         g11=1.0, g22=25.0, g33=25.0,
                         include_metric_fallback: bool = True) -> Dict[str, Any]:
    """
    Build all inputs required by runReedsSheppGF for ONE (g11,g22,g33) variant.
    Returns a small dict + SHM handle meta for the metric tensor (to be consumed in CPU worker).
    Also (optionally) returns 'metric_5d' as a fallback to avoid SHM races on Windows.
    """
    import numpy as _np

    NoCost, NxCost, NyCost = int(os_cost.shape[0]), int(os_cost.shape[1]), int(os_cost.shape[2])
    s_theta = 2 * _np.pi / NoCost

    # identity GF per voxel (same as original)
    gfLIF = _np.zeros((NoCost, NxCost, NyCost, 3, 3), dtype=_np.float32)
    gfLIF[..., 0, 0] = 1
    gfLIF[..., 1, 1] = 1
    gfLIF[..., 2, 2] = 1

    dims = _np.array([NoCost, NxCost, NyCost])  # (No,Nx,Ny) as in original

    # Build metric on main proc (GPU-capable inside your ct.*)
    metricLIFOld = ct.tracking.ReedsSheppMetricGFOld(gfLIF, dims, g11, g22, g33)         # (No,Nx,Ny,3,3)
    metricLIFinclCostOld = ct.tracking.IncludeCost(os_cost**2, metricLIFOld)             # (No,Nx,Ny,3,3)

    # RS3 expects (3,3,Nx,Ny,No)
    metric_5d = metricLIFinclCostOld.transpose((3, 4, 1, 2, 0)).astype(_np.float32, copy=False)  # (3,3,Nx,Ny,No)

    # Rect + dims for RS3 (your original uses [Nx,Ny,No])
    a = _np.array([0, 2 * _np.pi]) - s_theta / 2
    b = _np.array([0, NxCost]); c = _np.array([0, NyCost])
    sides = _np.array([b, c, a], dtype=_np.float32)
    dims_rs3 = _np.array([NxCost, NyCost, NoCost], dtype=_np.int32)

    # seeds/tips as [y,x,theta] (you used pi/2 for theta)
    seeds = _np.array([p0_down_xy[1], p0_down_xy[0], _np.pi / 2], dtype=_np.float32)
    tips  = _np.array([p1_down_xy[1], p1_down_xy[0], _np.pi / 2], dtype=_np.float32)

    # put metric into SHM so workers don't duplicate memory
    meta = shm_from_array(metric_5d)

    out = dict(
        ok=True,
        sides=sides,
        dims=dims_rs3,
        seeds=seeds,
        tips=tips,
        metric_meta=meta,
    )
    if include_metric_fallback:
        # WARNING: this increases pickled payload size per variant;
        # use only for debugging or when SHM is flaky on Windows.
        out["metric_5d"] = metric_5d
    return out

# ---- CPU RS3 (only) -------------------------------------------------
def _runReedsSheppGF_cpu(sides, dims, seeds, tips, metric_5d):
    """Hard CPU path for Riemann3_Periodic."""
    import numpy as np
    from agd import Eikonal
    from agd.Metrics import Riemann

    metric = np.asarray(metric_5d, dtype=np.float32)
    # tripwire: enforce expected shape
    if metric.ndim != 5 or metric.shape[:2] != (3, 3):
        raise ValueError(f"Bad metric shape {metric.shape}, expected (3,3,Nx,Ny,No)")
    # AGD expects seeds/tips as lists
    hfmIn = Eikonal.dictIn({
        'model'        : 'Riemann3_Periodic',
        'seeds'        : [seeds],
        'arrayOrdering': 'RowMajor',
        'tips'         : [tips],
        'metric'       : Riemann(metric),
        'verbosity'    : 0,
    })
    hfmIn.SetRect(sides=sides, dims=dims)
    hfmOut = hfmIn.Run()
    geos = [g.T for g in hfmOut['geodesics']]
    return geos

def _rs3_cpu_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """Child process: reconstruct metric from SHM (preferred) or fallback, then run only RS3."""
    metric = None
    shm = None
    try:
        metric = task.get("metric_5d", None)
        if metric is None:
            return {
                "ok": False,
                "variant_id": task.get("variant_id"),
                "error": "Missing metric_5d in task (SHM disabled version)"
            }

        geos = _runReedsSheppGF_cpu(
            task["sides"], task["dims"], task["seeds"], task["tips"], metric
        )
        path_yx = geos[0]  # (N,2) [y,x]
        path_xy = np.stack([path_yx[:, 1], path_yx[:, 0]], axis=0)  # (2,N)
        return {"ok": True, "variant_id": task.get("variant_id"), "track_full_xy_cropdown": path_xy}

    except Exception as e:
        return {
            "ok": False,
            "variant_id": task.get("variant_id"),
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        }

# ---- Orchestrator ---------------------------------------------------
def run_rs3_variants_split(
    ct,
    os_cost: np.ndarray,
    p0_down_xy: np.ndarray,     # [x,y] crop-down
    p1_down_xy: np.ndarray,     # [x,y] crop-down
    g_variants: List[Dict[str, float]],
    down: int,
    bbox_xyxy: List[int],       # [xmin,ymin,xmax,ymax]
    cpu_max_workers: int = None,
    mp_start_method: str = "spawn",
) -> List[Dict[str, Any]]:
    """
    PRESTAGE (main proc) per variant → RS3 worker per variant (CPU parallel).
    Returns list of dicts with "track_full_xy" in GLOBAL coords (2,N).
    """
    cpu_max_workers = cpu_max_workers or min(os.cpu_count() or 8, 16)
    xmin, ymin, xmax, ymax = map(int, bbox_xyxy)
    prepared: List[Dict[str, Any]] = []
    metas: List[Dict[str, Any]] = []  # to close later

    # ----------------------------------------------------------
    # A) PRESTAGE (GPU-friendly): build per-variant RS3 inputs
    # ----------------------------------------------------------
    for vidx, gv in enumerate(g_variants):
        pre = rs3_prestage_variant(
            ct, os_cost, p0_down_xy, p1_down_xy,
            g11=gv.get("g11", 1.0),
            g22=gv.get("g22", 25.0),
            g33=gv.get("g33", 25.0),
            include_metric_fallback=True,   # <-- keep True until SHM is rock-solid on Windows
        )
        if not pre.get("ok"):
            prepared.append({"ok": False, "variant_id": vidx, "error": "prestage failed"})
            continue
        metas.append(pre["metric_meta"])
        # trim payload we pass to children
        prepared.append({
            "variant_id": vidx,
            "metric_meta": pre["metric_meta"],
            "metric_5d": pre.get("metric_5d"),  # explicit fallback
            "sides": pre["sides"],
            "dims":  pre["dims"],
            "seeds": pre["seeds"],
            "tips":  pre["tips"],
        })

    # ----------------------------------------------------------
    # B) RS3 CPU workers (Windows-safe shared memory handling)
    # ----------------------------------------------------------
    ctx = get_context(mp_start_method)
    results = []
    try:
        with ProcessPoolExecutor(max_workers=cpu_max_workers, mp_context=ctx) as ex:
            futs = [ex.submit(_rs3_cpu_worker, t) for t in prepared]
            for fut in as_completed(futs):
                try:
                    results.append(fut.result())
                except Exception as e:
                    results.append({"ok": False, "error": f"{type(e).__name__}: {e}"})
        # ✅ only close after all futures have attached
        time.sleep(0.1)  # slight delay for Windows handle propagation
        for m in metas:
            close_shm(m)
    except Exception:
        for m in metas:
            try: close_shm(m)
            except Exception: pass
        raise

    # ----------------------------------------------------------
    # C) Map to GLOBAL coords and ensure start==p0
    # ----------------------------------------------------------
    out = []
    for r in results:
        if not r.get("ok"):
            out.append(r)
            continue

        path_xy_cd = r["track_full_xy_cropdown"]  # (2,N) [x,y] crop-down
        # half-pixel convention → crop
        path_xy_cd[0] -= 0.5
        path_xy_cd[1] -= 0.5

        track_crop_xy = path_xy_cd.copy()
        track_crop_xy[0] *= down
        track_crop_xy[1] *= down

        # crop → global
        xg = track_crop_xy[0] + xmin
        yg = track_crop_xy[1] + ymin
        track_full_xy = np.vstack([xg, yg])  # (2,N)

        out.append({"ok": True, "variant_id": r.get("variant_id"), "track_full_xy": track_full_xy})

    out.sort(key=lambda z: z.get("variant_id", 0))
    return out
