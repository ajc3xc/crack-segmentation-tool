# --- CPU AFFINITY PINNING (Windows-safe) --- for core ultra
import os
'''try:
    import psutil
    _p = psutil.Process(os.getpid())
    # Pin to P-cores only
    _p.cpu_affinity([0, 1, 2, 3])
    print(f"[AFFINITY] Worker PID={os.getpid()} pinned to {_p.cpu_affinity()}")
except Exception as e:
    print(f"[AFFINITY] Could not set affinity: {e}")'''


# rs3_split.py (refactored for fast-marching ablation)
# ---------------------------------------------------
# Split Reeds–Shepp fast marching into:
#   A) PRESTAGE (main process): build metric tensor + sides/dims/seeds/tips per (g11,g22,g33)
#   B) CPU WORKER: run only runReedsSheppGF on CPU in parallel
#
# This version:
#   - Uses the *same* metric builder and IncludeCost as tracking.fast_marching's
#     optimized path (ReedsSheppMetricGFOld_vec + IncludeCost).
#   - Adds simple timing info (metric build, include cost, transpose, solver).
#   - Keeps the existing SHM helpers, but currently still uses the metric_5d fallback.

import os, time, traceback, gc
import numpy as np
import psutil
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, Any, List, Optional, Tuple
from helpers.cpu_affinity import process_pool_affinity_config

try:
    import cupy as cp  # type: ignore
    CUPY_AVAILABLE = True
except Exception:
    cp = None
    CUPY_AVAILABLE = False


def _gib(n):
    return float(n) / (1024**3)


def _rss_gib():
    return _gib(psutil.Process(os.getpid()).memory_info().rss)


def _avail_gib():
    return _gib(psutil.virtual_memory().available)


def estimate_metric5d_gib(No, Nx, Ny, dtype=np.float32):
    item = np.dtype(dtype).itemsize
    return _gib(3 * 3 * Nx * Ny * No * item)


def is_oom_exc(e: Exception) -> bool:
    msg = str(e)
    try:
        np_oom = np.core._exceptions._ArrayMemoryError
        if isinstance(e, np_oom):
            return True
    except Exception:
        pass
    if ("Unable to allocate" in msg) or ("out of memory" in msg.lower()):
        return True
    if ("cuda" in msg.lower() and "memory" in msg.lower()):
        return True
    return isinstance(e, MemoryError)


def free_gpu_pools(ct=None):
    try:
        if ct is not None and getattr(ct.tracking, "CUPY_AVAILABLE", False) and hasattr(ct.tracking, "cp"):
            cp = ct.tracking.cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass

# ---- SHM helpers ----------------------------------------------------
def shm_from_array(arr: np.ndarray) -> Dict[str, Any]:
    """
    Create SHM and copy arr bytes into it.
    Returns meta needed to reattach elsewhere.
    """
    arr_c = np.ascontiguousarray(arr)
    shm = SharedMemory(create=True, size=arr_c.nbytes)
    view = np.ndarray(arr_c.shape, dtype=arr_c.dtype, buffer=shm.buf)
    view[...] = arr_c
    return dict(name=shm.name, shape=arr_c.shape, dtype=str(arr_c.dtype), nbytes=int(arr_c.nbytes))


def shm_from_array_keepalive(arr: np.ndarray):
    """
    Windows-safe: return (meta, shm_obj). Parent must keep shm_obj alive until
    workers have attached and completed.
    """
    arr_c = np.ascontiguousarray(arr)
    shm = SharedMemory(create=True, size=arr_c.nbytes)
    view = np.ndarray(arr_c.shape, dtype=arr_c.dtype, buffer=shm.buf)
    view[...] = arr_c
    meta = dict(name=shm.name, shape=arr_c.shape, dtype=str(arr_c.dtype), nbytes=int(arr_c.nbytes))
    return meta, shm

def array_from_shm(meta: Dict[str, Any]):
    """
    Attach to existing SHM and return (shm_handle, np_view).
    Returned ndarray is a view into SHM (no copy).
    """
    shm = SharedMemory(name=meta["name"])
    arr = np.ndarray(tuple(meta["shape"]), dtype=np.dtype(meta["dtype"]), buffer=shm.buf)
    return shm, arr

def close_shm_meta(meta: Dict[str, Any]) -> None:
    """Parent-side cleanup: close + unlink after workers finish."""
    try:
        shm = SharedMemory(name=meta["name"])
        shm.close()
        shm.unlink()
    except Exception:
        pass


def _free_cupy_pools():
    if not CUPY_AVAILABLE:
        return
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def _is_oom_exception(e: BaseException) -> bool:
    return is_oom_exc(e)


def close_shm_keepalive(meta: Dict[str, Any], shm_obj=None) -> None:
    """Close/unlink by name and also close parent's keepalive handle if provided."""
    try:
        shm = SharedMemory(name=meta["name"])
        shm.close()
        shm.unlink()
    except Exception:
        pass
    if shm_obj is not None:
        try:
            shm_obj.close()
        except Exception:
            pass

# Backward-compatible alias
def close_shm(meta: Dict[str, Any]):
    close_shm_meta(meta)


def _metric5d_bytes(No, Nx, Ny, dtype=np.float32):
    return 3 * 3 * int(Nx) * int(Ny) * int(No) * np.dtype(dtype).itemsize


def should_use_shm(No, Nx, Ny, dtype=np.float32, max_gib=0.75):
    return _gib(_metric5d_bytes(No, Nx, Ny, dtype)) <= float(max_gib)


@dataclass
class CorridorParams:
    radius_px: int = 80
    window_len_px: int = 700
    overlap_px: int = 200
    resample_step_px: float = 3.0
    drop_overlap_pts: int = 15

# ---- (A) PRESTAGE: build RS3 inputs (main process; can use ct.tracking helpers) ----
def rs3_prestage_variant_stream(
    ct,
    os_cost: np.ndarray,
    p0_down_xy: np.ndarray,
    p1_down_xy: np.ndarray,
    g11: float = 1.0,
    g22: float = 25.0,
    g33: float = 25.0,
    solver_dtype: str = "float64",
    *,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Memory-stable prestage for one RS3 variant.

    Builds metric_5d directly as (3,3,Nx,Ny,No) float32 and fills it one
    orientation slice at a time so we avoid allocating the giant broadcast
    intermediate (No,Nx,Ny,3,3).
    """
    import numpy as _np
    from time import perf_counter

    timing = {
        "fm_metric_build_sec": 0.0,
        "fm_include_cost_sec": 0.0,
        "fm_transpose_sec": 0.0,
    }

    try:
        if solver_dtype not in ("float32", "float64"):
            raise ValueError(f"solver_dtype must be 'float32' or 'float64', got {solver_dtype!r}")

        # Solver is CPU-side; bring os_cost to CPU float32 once.
        if getattr(ct.tracking, "CUPY_AVAILABLE", False) and hasattr(ct.tracking, "cp"):
            _cp = ct.tracking.cp
            if hasattr(_cp, "ndarray") and isinstance(os_cost, _cp.ndarray):
                os_cost = os_cost.get()
                free_gpu_pools(ct)
            else:
                os_cost = _np.asarray(os_cost, dtype=_np.float32, order="C")
        else:
            os_cost = _np.asarray(os_cost, dtype=_np.float32, order="C")

        os_cost = _np.asarray(os_cost, dtype=_np.float32, order="C")
        NoCost, NxCost, NyCost = map(int, os_cost.shape)
        s_theta = 2 * _np.pi / NoCost
        dims = _np.array([NoCost, NxCost, NyCost], dtype=_np.int32)

        # 1) Metric build (optimized path: no gfLIF allocation), then move to CPU float32.
        t0 = perf_counter()
        metric_theta = ct.tracking.ReedsSheppMetricGFOld_vec(None, dims, g11, g22, g33)  # (No,3,3)
        if getattr(ct.tracking, "CUPY_AVAILABLE", False) and hasattr(ct.tracking, "cp"):
            _cp = ct.tracking.cp
            if hasattr(_cp, "ndarray") and isinstance(metric_theta, _cp.ndarray):
                metric_theta = metric_theta.get()
                free_gpu_pools(ct)
        metric_theta = _np.asarray(metric_theta, dtype=_np.float32, order="C")
        timing["fm_metric_build_sec"] = float(perf_counter() - t0)

        # 2) Build cost^2 and fill final metric_5d slice-by-slice (streamed).
        t0 = perf_counter()
        cost_sq = _np.empty_like(os_cost, dtype=_np.float32, order="C")
        _np.multiply(os_cost, os_cost, out=cost_sq)
        del os_cost
        gc.collect()

        # Final solver metric shape (3,3,Nx,Ny,No). This is the one unavoidable large array.
        metric_5d = _np.empty((3, 3, NxCost, NyCost, NoCost), dtype=_np.float32, order="C")
        for t in range(NoCost):
            metric_5d[:, :, :, :, t] = (
                cost_sq[t][None, None, :, :] * metric_theta[t][:, :, None, None]
            )

        timing["fm_include_cost_sec"] = float(perf_counter() - t0)
        del cost_sq, metric_theta
        gc.collect()
        timing["fm_transpose_sec"] = 0.0  # already written in solver layout

        # Rect + dims for RS3 (dims in [Nx,Ny,No] order for AGD)
        solver_np_dtype = _np.float32 if solver_dtype == "float32" else _np.float64
        a = _np.array([0, 2 * _np.pi], dtype=solver_np_dtype) - s_theta / 2
        b = _np.array([0, NxCost], dtype=solver_np_dtype); c = _np.array([0, NyCost], dtype=solver_np_dtype)
        sides = _np.array([b, c, a], dtype=solver_np_dtype)
        dims_rs3 = _np.array([NxCost, NyCost, NoCost], dtype=_np.int32)

        # seeds/tips as [y,x,theta] (π/2 orientation like your original code)
        seeds = _np.array([p0_down_xy[1], p0_down_xy[0], _np.pi / 2], dtype=solver_np_dtype)
        tips  = _np.array([p1_down_xy[1], p1_down_xy[0], _np.pi / 2], dtype=solver_np_dtype)

        out = dict(
            ok=True,
            sides=sides,
            dims=dims_rs3,
            seeds=seeds,
            tips=tips,
            metric5d=metric_5d,
            timing=timing,
            solver_dtype=solver_dtype,
        )
        if verbose:
            print(
                f"[RS3 prestage] cost={NoCost}x{NxCost}x{NyCost} "
                f"metric5d~{estimate_metric5d_gib(NoCost, NxCost, NyCost):.2f} GiB "
                f"RSS={_rss_gib():.2f} GiB avail={_avail_gib():.2f} GiB"
            )
        return out

    except Exception as e:
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            "timing": timing,
            "solver_dtype": solver_dtype,
        }


def rs3_prestage_variant(
    ct,
    os_cost: np.ndarray,
    p0_down_xy: np.ndarray,
    p1_down_xy: np.ndarray,
    g11: float = 1.0,
    g22: float = 25.0,
    g33: float = 25.0,
    include_metric_fallback: bool = True,
    solver_dtype: str = "float64",
) -> Dict[str, Any]:
    """
    Compatibility wrapper around the streamed prestage builder.

    This path intentionally avoids SharedMemory for metric_5d (Windows-safe).
    The caller decides whether to run locally or submit to a worker process.
    """
    _ = include_metric_fallback  # compatibility only; ignored
    return rs3_prestage_variant_stream(
        ct=ct,
        os_cost=os_cost,
        p0_down_xy=p0_down_xy,
        p1_down_xy=p1_down_xy,
        g11=g11,
        g22=g22,
        g33=g33,
        solver_dtype=solver_dtype,
        verbose=False,
    )

# ---- CPU RS3 (only) -------------------------------------------------
def _runReedsSheppGF_cpu(sides, dims, seeds, tips, metric_5d, solver_dtype="float64"):
    """Hard CPU path for Riemann3_Periodic."""
    import numpy as np
    from agd import Eikonal
    from agd.Metrics import Riemann

    if solver_dtype not in ("float32", "float64"):
        raise ValueError(f"solver_dtype must be 'float32' or 'float64', got {solver_dtype!r}")
    #target_dtype = np.float32 if solver_dtype == "float32" else np.float64
    metric = np.ascontiguousarray(metric_5d, dtype=np.float64)
    # tripwire: enforce expected shape
    if metric.ndim != 5 or metric.shape[:2] != (3, 3):
        raise ValueError(f"Bad metric shape {metric.shape}, expected (3,3,Nx,Ny,No)")

    try:
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
    except TypeError as e:
        msg = str(e)
        if "set_array()" in msg and "'metric'" in msg:
            raise TypeError(
                f"HFM metric type mismatch for set_array('metric'): "
                f"shape={metric.shape}, dtype={metric.dtype}, "
                f"C_CONTIGUOUS={bool(metric.flags['C_CONTIGUOUS'])}, solver_dtype={solver_dtype}. "
                f"The binding expects numpy.ndarray[numpy.float64]."
            ) from None
        raise
    geos = [g.T for g in hfmOut['geodesics']]
    return geos

def _rs3_cpu_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Child process:
      - attach to SHM via metric_meta
      - run RS3
      - close SHM handle
      - return path in crop-down coords
    """
    import numpy as np
    from time import perf_counter

    try:
        if "error" in task and task.get("error"):
            return {
                "ok": False,
                "variant_id": task.get("variant_id"),
                "error": task.get("error"),
                "timing": dict(task.get("timing", {})),
            }

        metric_local = task.get("metric5d", None)
        meta = task.get("metric_meta", None)
        if metric_local is None and meta is None:
            return {
                "ok": False,
                "variant_id": task.get("variant_id"),
                "error": "Missing metric5d/metric_meta in task"
            }

        if metric_local is not None:
            t0 = perf_counter()
            geos = _runReedsSheppGF_cpu(
                task["sides"], task["dims"], task["seeds"], task["tips"], metric_local,
                solver_dtype=task.get("solver_dtype", "float64")
            )
            solver_sec = float(perf_counter() - t0)
        else:
            shm, metric_view = array_from_shm(meta)  # zero-copy view into SHM
            try:
                t0 = perf_counter()
                geos = _runReedsSheppGF_cpu(
                    task["sides"], task["dims"], task["seeds"], task["tips"], metric_view,
                    solver_dtype=task.get("solver_dtype", "float64")
                )
                solver_sec = float(perf_counter() - t0)
            finally:
                shm.close()  # close child handle only; parent unlinks later
                del metric_view
                gc.collect()

        path_yx = geos[0]  # (N,2) [y,x]
        path_xy = np.stack([path_yx[:, 1], path_yx[:, 0]], axis=0)  # (2,N)

        # merge timing: metric build/include/transpose from main proc + solver from worker
        timing = dict(task.get("timing", {}))
        timing["fm_solver_sec"] = solver_sec
        total = timing.get("fm_metric_build_sec", 0.0) \
                + timing.get("fm_include_cost_sec", 0.0) \
                + timing.get("fm_transpose_sec", 0.0) \
                + solver_sec
        timing["fm_total_sec"] = float(total)

        return {
            "ok": True,
            "variant_id": task.get("variant_id"),
            "track_full_xy_cropdown": path_xy,
            "timing": timing,
        }

    except Exception as e:
        return {
            "ok": False,
            "variant_id": task.get("variant_id"),
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }

# ---- Orchestrator ---------------------------------------------------
def _map_rs3_results_to_global(results, *, ds, down, bbox_xyxy):
    xmin, ymin, xmax, ymax = map(int, bbox_xyxy)
    _ = (xmax, ymax)

    out: List[Dict[str, Any]] = []
    for r in results:
        if not r.get("ok"):
            out.append(r)
            continue

        path_xy_cd = np.asarray(r["track_full_xy_cropdown"], dtype=np.float64).copy()
        path_xy_cd[0] -= 0.5
        path_xy_cd[1] -= 0.5

        # ds-space -> original crop-down space
        path_xy_cd *= float(ds)

        # crop-down -> crop pixels
        path_xy_cd[0] *= float(down)
        path_xy_cd[1] *= float(down)

        xg = path_xy_cd[0] + xmin
        yg = path_xy_cd[1] + ymin
        out.append({
            "ok": True,
            "variant_id": r.get("variant_id"),
            "track_full_xy": np.vstack([xg, yg]),
            "timing": r.get("timing", {}),
            "ds_used": int(ds),
        })

    out.sort(key=lambda z: z.get("variant_id", 0))
    return out


def run_rs3_variants_split_adaptive(
    ct,
    os_cost: np.ndarray,
    p0_down_xy: np.ndarray,
    p1_down_xy: np.ndarray,
    g_variants: List[Dict[str, float]],
    down: int,
    bbox_xyxy: List[int],
    *,
    cpu_max_workers: int = None,
    mp_start_method: str = "spawn",
    solver_dtype: str = "float64",
    small_metric_gib_parallel: float = 0.80,
    max_workers_cap: int = 8,
    ds_attempts=(1, 2, 4),
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Adaptive RS3 runner with OOM fallback and streamed prestage.
    """

    def _ds_view(cost, p0_xy, p1_xy, ds_val):
        ds_val = int(ds_val)
        if ds_val == 1:
            return cost, p0_xy, p1_xy
        return (
            cost[:, ::ds_val, ::ds_val],
            (float(p0_xy[0]) / ds_val, float(p0_xy[1]) / ds_val),
            (float(p1_xy[0]) / ds_val, float(p1_xy[1]) / ds_val),
        )

    last_err = None

    for ds in tuple(ds_attempts):
        ds = int(ds)
        try:
            if verbose:
                print(f"[RS3] === attempt ds={ds} === RSS={_rss_gib():.2f} GiB avail={_avail_gib():.2f} GiB")

            cost_try, p0_try, p1_try = _ds_view(os_cost, p0_down_xy, p1_down_xy, ds)
            NoCost, NxCost, NyCost = map(int, cost_try.shape)
            metric_gib = estimate_metric5d_gib(NoCost, NxCost, NyCost, np.float32)

            req_workers = cpu_max_workers or (os.cpu_count() or 8)
            workers = max(1, min(int(req_workers), int(max_workers_cap)))
            if metric_gib >= float(small_metric_gib_parallel):
                workers = 1

            if verbose:
                print(f"[RS3] ds={ds} cost={tuple(cost_try.shape)} metric5d~{metric_gib:.2f} GiB workers={workers}")

            results: List[Dict[str, Any]] = []

            if workers == 1:
                # Sequential variant loop: prestage -> solve locally, no SHM.
                for vidx, gv in enumerate(g_variants):
                    pre = rs3_prestage_variant_stream(
                        ct=ct,
                        os_cost=cost_try,
                        p0_down_xy=np.asarray(p0_try, float),
                        p1_down_xy=np.asarray(p1_try, float),
                        g11=float(gv.get("g11", 1.0)),
                        g22=float(gv.get("g22", 25.0)),
                        g33=float(gv.get("g33", 25.0)),
                        solver_dtype=solver_dtype,
                        verbose=verbose,
                    )
                    if not pre.get("ok"):
                        results.append({
                            "ok": False,
                            "variant_id": vidx,
                            "error": pre.get("error", "prestage failed"),
                            "timing": pre.get("timing", {}),
                        })
                        continue

                    task = dict(pre, variant_id=vidx)
                    try:
                        results.append(_rs3_cpu_worker(task))
                    finally:
                        if "metric5d" in task:
                            del task["metric5d"]
                        if "metric5d" in pre:
                            del pre["metric5d"]
                        gc.collect()
            else:
                prepared: List[Dict[str, Any]] = []
                for vidx, gv in enumerate(g_variants):
                    pre = rs3_prestage_variant(
                        ct=ct,
                        os_cost=cost_try,
                        p0_down_xy=np.asarray(p0_try, float),
                        p1_down_xy=np.asarray(p1_try, float),
                        g11=float(gv.get("g11", 1.0)),
                        g22=float(gv.get("g22", 25.0)),
                        g33=float(gv.get("g33", 25.0)),
                        include_metric_fallback=False,
                        solver_dtype=solver_dtype,
                    )
                    if not pre.get("ok"):
                        prepared.append({
                            "ok": False,
                            "variant_id": vidx,
                            "error": pre.get("error", "prestage failed"),
                            "timing": pre.get("timing", {}),
                        })
                        continue
                    if "metric5d" not in pre:
                        prepared.append({
                            "ok": False,
                            "variant_id": vidx,
                            "error": "prestage returned no metric5d",
                            "timing": pre.get("timing", {}),
                        })
                        continue
                    prepared.append(dict(pre, variant_id=vidx))
                    gc.collect()

                ctx = get_context(mp_start_method)
                pool_workers, pool_init, pool_initargs, pool_cpus = process_pool_affinity_config(
                    workers, label="rs3-prestaged"
                )
                if verbose:
                    print(f"[AFFINITY] rs3-prestaged workers={pool_workers} cpus={pool_cpus}")
                with ProcessPoolExecutor(
                    max_workers=pool_workers,
                    mp_context=ctx,
                    initializer=pool_init,
                    initargs=pool_initargs,
                ) as ex:
                    futs = [ex.submit(_rs3_cpu_worker, t) for t in prepared]
                    for fut in as_completed(futs):
                        try:
                            results.append(fut.result())
                        except Exception as e:
                            results.append({"ok": False, "error": f"{type(e).__name__}: {e}"})

                # Free parent copies after worker processes receive tasks.
                for t in prepared:
                    if "metric5d" in t:
                        del t["metric5d"]
                gc.collect()

            # If every variant failed due to OOM-like error, try next ds.
            if results and all((not r.get("ok")) and is_oom_exc(Exception(str(r.get("error", "")))) for r in results):
                raise MemoryError("All variants failed with OOM-like errors")

            return _map_rs3_results_to_global(results, ds=ds, down=down, bbox_xyxy=bbox_xyxy)

        except Exception as e:
            last_err = e
            free_gpu_pools(ct)
            gc.collect()

            if verbose:
                print(f"[RS3] attempt ds={ds} failed: {type(e).__name__}: {e}")

            if not is_oom_exc(e):
                raise
            continue

    raise last_err


def _path_to_xy(path_yx: np.ndarray) -> np.ndarray:
    return np.stack([path_yx[:, 1], path_yx[:, 0]], axis=0)


def _arc_length_xy(path_xy_2n: np.ndarray) -> np.ndarray:
    if path_xy_2n.shape[1] <= 1:
        return np.zeros(path_xy_2n.shape[1], dtype=np.float64)
    dx = np.diff(path_xy_2n[0])
    dy = np.diff(path_xy_2n[1])
    ds = np.sqrt(dx * dx + dy * dy)
    s = np.zeros(path_xy_2n.shape[1], dtype=np.float64)
    s[1:] = np.cumsum(ds)
    return s


def _resample_path_by_arclen(path_xy_2n: np.ndarray, step: float) -> np.ndarray:
    path_xy_2n = np.asarray(path_xy_2n, dtype=np.float32)
    if path_xy_2n.ndim != 2 or path_xy_2n.shape[0] != 2 or path_xy_2n.shape[1] < 2:
        return path_xy_2n
    s = _arc_length_xy(path_xy_2n)
    total = float(s[-1]) if len(s) else 0.0
    if total <= 1e-9:
        return path_xy_2n
    m = max(2, int(np.ceil(total / float(step))) + 1)
    s_new = np.linspace(0.0, total, m, dtype=np.float64)
    x_new = np.interp(s_new, s, path_xy_2n[0])
    y_new = np.interp(s_new, s, path_xy_2n[1])
    return np.vstack([x_new, y_new]).astype(np.float32, copy=False)


def _clamp_bbox(x0: int, y0: int, x1: int, y1: int, Nx: int, Ny: int) -> Tuple[int, int, int, int]:
    x0 = max(0, min(int(x0), max(0, Nx - 1)))
    y0 = max(0, min(int(y0), max(0, Ny - 1)))
    x1 = max(0, min(int(x1), Nx))
    y1 = max(0, min(int(y1), Ny))
    if x1 <= x0:
        x1 = min(Nx, x0 + 1)
    if y1 <= y0:
        y1 = min(Ny, y0 + 1)
    return x0, y0, x1, y1


def _bbox_from_points(points_xy_2n: np.ndarray, radius: int, Nx: int, Ny: int) -> Tuple[int, int, int, int]:
    xs = np.asarray(points_xy_2n[0], dtype=np.float64)
    ys = np.asarray(points_xy_2n[1], dtype=np.float64)
    x0 = int(np.floor(xs.min() - radius))
    y0 = int(np.floor(ys.min() - radius))
    x1 = int(np.ceil(xs.max() + radius))
    y1 = int(np.ceil(ys.max() + radius))
    return _clamp_bbox(x0, y0, x1, y1, Nx, Ny)


def _split_path_into_windows(path_xy_2n: np.ndarray, window_len_px: float, overlap_px: float) -> List[Tuple[int, int]]:
    if path_xy_2n.shape[1] < 2:
        return [(0, max(0, path_xy_2n.shape[1] - 1))]
    s = _arc_length_xy(path_xy_2n)
    total = float(s[-1])
    if total <= 1e-9:
        return [(0, path_xy_2n.shape[1] - 1)]

    win = float(window_len_px)
    ov = float(overlap_px)
    step = max(1.0, win - ov)

    out: List[Tuple[int, int]] = []
    t = 0.0
    n = path_xy_2n.shape[1]
    while t < total:
        t0 = t
        t1 = min(total, t + win)
        i0 = int(np.searchsorted(s, t0, side="left"))
        i1 = int(np.searchsorted(s, t1, side="right")) - 1
        i0 = max(0, min(i0, n - 2))
        i1 = max(i0 + 1, min(i1, n - 1))
        out.append((i0, i1))
        if t1 >= total:
            break
        t += step

    if out and out[-1][1] != n - 1:
        out[-1] = (out[-1][0], n - 1)
    return out


def _stitch_paths(paths_xy_2n: List[np.ndarray], drop_overlap_points: int = 10) -> np.ndarray:
    if not paths_xy_2n:
        return np.zeros((2, 0), dtype=np.float32)
    merged = [np.asarray(paths_xy_2n[0], dtype=np.float32)]
    for seg in paths_xy_2n[1:]:
        seg = np.asarray(seg, dtype=np.float32)
        if seg.ndim != 2 or seg.shape[0] != 2:
            continue
        if seg.shape[1] <= int(drop_overlap_points):
            continue
        merged.append(seg[:, int(drop_overlap_points):])
    return np.concatenate(merged, axis=1) if merged else np.zeros((2, 0), dtype=np.float32)


def _downsample_cost(os_cost: np.ndarray, ds: int) -> np.ndarray:
    ds = int(ds)
    if ds == 1:
        return os_cost
    return os_cost[:, ::ds, ::ds]


def _scale_point_xy(pt_xy: np.ndarray, scale: float) -> np.ndarray:
    pt_xy = np.asarray(pt_xy, dtype=np.float32)
    return np.asarray([pt_xy[0] * scale, pt_xy[1] * scale], dtype=np.float32)


def _scale_path_xy(path_xy_2n: np.ndarray, scale: float) -> np.ndarray:
    return (np.asarray(path_xy_2n, dtype=np.float32) * float(scale)).astype(np.float32, copy=False)


def _run_rs3_single_local(
    ct,
    os_cost: np.ndarray,
    p0_xy: np.ndarray,
    p1_xy: np.ndarray,
    *,
    g11: float,
    g22: float,
    g33: float,
    solver_dtype: str = "float64",
    mode_tag: str = "new",
) -> Dict[str, Any]:
    """
    SHM-free single RS3 solve using streamed prestage + existing AGD wrapper.
    Returns local path in the same coordinate system as os_cost.
    """
    from time import perf_counter

    t_all0 = perf_counter()
    pre = rs3_prestage_variant_stream(
        ct=ct,
        os_cost=os_cost,
        p0_down_xy=np.asarray(p0_xy, dtype=np.float32),
        p1_down_xy=np.asarray(p1_xy, dtype=np.float32),
        g11=float(g11),
        g22=float(g22),
        g33=float(g33),
        solver_dtype=solver_dtype,
        verbose=False,
    )
    if not pre.get("ok"):
        return {
            "ok": False,
            "mode": mode_tag,
            "error": pre.get("error", "prestage failed"),
            "timing": dict(pre.get("timing", {})),
        }

    metric5d = pre.get("metric5d")
    if metric5d is None:
        return {
            "ok": False,
            "mode": mode_tag,
            "error": "prestage returned no metric5d",
            "timing": dict(pre.get("timing", {})),
        }

    t0 = perf_counter()
    geos = _runReedsSheppGF_cpu(
        pre["sides"], pre["dims"], pre["seeds"], pre["tips"], metric5d, solver_dtype=solver_dtype
    )
    solver_sec = float(perf_counter() - t0)

    path_yx = geos[0]
    path_xy = _path_to_xy(path_yx).astype(np.float32, copy=False)

    timing = dict(pre.get("timing", {}))
    timing["fm_solver_sec"] = solver_sec
    timing["fm_total_sec"] = float(
        timing.get("fm_metric_build_sec", 0.0)
        + timing.get("fm_include_cost_sec", 0.0)
        + timing.get("fm_transpose_sec", 0.0)
        + solver_sec
    )
    timing["fm_wall_total_sec"] = float(perf_counter() - t_all0)

    del metric5d, geos
    gc.collect()
    _free_cupy_pools()

    return {"ok": True, "mode": mode_tag, "track_local_xy": path_xy, "timing": timing}


def _rs3_worker_small(task: Dict[str, Any]) -> Dict[str, Any]:
    try:
        ct = task["ct"]
        return _run_rs3_single_local(
            ct,
            task["os_cost"],
            task["p0_xy"],
            task["p1_xy"],
            g11=task["g11"],
            g22=task["g22"],
            g33=task["g33"],
            solver_dtype=task.get("solver_dtype", "float64"),
            mode_tag=task.get("mode_tag", "new"),
        )
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}


def refine_with_corridor_windows(
    ct,
    os_cost_full: np.ndarray,
    p0_xy_full: np.ndarray,
    p1_xy_full: np.ndarray,
    coarse_path_xy_full: np.ndarray,
    *,
    g11: float,
    g22: float,
    g33: float,
    solver_dtype: str = "float64",
    params: CorridorParams = CorridorParams(),
    mode_tag: str = "new",
) -> Dict[str, Any]:
    """
    Solve multiple smaller full-res RS3 windows guided by a coarse path.
    Returns a stitched path in the same coordinates as os_cost_full.
    """
    _ = (p0_xy_full, p1_xy_full)
    timing: Dict[str, float] = {}
    try:
        No, Nx, Ny = map(int, np.asarray(os_cost_full).shape)
        _ = No
        coarse_rs = _resample_path_by_arclen(np.asarray(coarse_path_xy_full, dtype=np.float32), float(params.resample_step_px))
        windows = _split_path_into_windows(coarse_rs, float(params.window_len_px), float(params.overlap_px))

        seg_paths: List[np.ndarray] = []
        t_sum = 0.0
        for i0, i1 in windows:
            pts = coarse_rs[:, i0:i1 + 1]
            x0, y0, x1, y1 = _bbox_from_points(pts, int(params.radius_px), Nx, Ny)
            os_cost_crop = os_cost_full[:, x0:x1, y0:y1]
            p0_local = np.asarray([pts[0, 0] - x0, pts[1, 0] - y0], dtype=np.float32)
            p1_local = np.asarray([pts[0, -1] - x0, pts[1, -1] - y0], dtype=np.float32)

            r = _run_rs3_single_local(
                ct,
                os_cost_crop,
                p0_local,
                p1_local,
                g11=g11,
                g22=g22,
                g33=g33,
                solver_dtype=solver_dtype,
                mode_tag=mode_tag,
            )
            if not r.get("ok"):
                return r

            path_local = np.asarray(r["track_local_xy"], dtype=np.float32)
            path_full = path_local.copy()
            path_full[0] += float(x0)
            path_full[1] += float(y0)
            seg_paths.append(path_full)
            t_sum += float(r.get("timing", {}).get("fm_total_sec", 0.0))

            del os_cost_crop, path_local, path_full, r
            gc.collect()
            _free_cupy_pools()

        stitched = _stitch_paths(seg_paths, drop_overlap_points=int(params.drop_overlap_pts))
        timing["corridor_total_sec"] = float(t_sum)
        timing["corridor_windows"] = float(len(windows))
        return {"ok": True, "track_local_xy": stitched, "timing": timing}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}", "timing": timing}


def run_rs3_variants_adaptive(
    ct,
    os_cost_full: np.ndarray,
    p0_xy_full: np.ndarray,
    p1_xy_full: np.ndarray,
    g_variants: List[Dict[str, float]],
    *,
    mode_tag: str = "",
    solver_dtype: str = "float64",
    parallel_metric_gib_threshold: float = 0.0,
    enable_corridor_refine: bool = False,
    corridor_params: CorridorParams = None,
    down: int = 1,
    bbox_xyxy: List[int] = None,
    cpu_max_workers: int = None,
    mp_start_method: str = "spawn",
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    SHM-free adaptive RS3 variants runner.

    Returns the same downstream-facing shape as the legacy rs3_split path:
      result[i]["track_full_xy"] is (2,N) in GLOBAL coordinates (after bbox/down mapping).
    """
    if corridor_params is None:
        corridor_params = CorridorParams()

    if bbox_xyxy is None:
        bbox_xyxy = [0, 0, 0, 0]
    xmin, ymin, xmax, ymax = map(int, bbox_xyxy)
    _ = (xmax, ymax)

    os_cost_full = np.asarray(os_cost_full) if not (CUPY_AVAILABLE and hasattr(cp, "ndarray") and isinstance(os_cost_full, cp.ndarray)) else os_cost_full
    No0, Nx0, Ny0 = map(int, os_cost_full.shape)
    metric_gib_1x = estimate_metric5d_gib(No0, Nx0, Ny0, np.float32)
    can_parallel = (metric_gib_1x <= float(parallel_metric_gib_threshold)) and (len(g_variants) > 1)
    cpu_max_workers = cpu_max_workers or min(os.cpu_count() or 8, 8)

    def _map_local_to_global(path_local_xy_2n: np.ndarray) -> np.ndarray:
        path = np.asarray(path_local_xy_2n, dtype=np.float64).copy()
        path[0] -= 0.5
        path[1] -= 0.5
        path[0] *= float(down)
        path[1] *= float(down)
        path[0] += float(xmin)
        path[1] += float(ymin)
        return path

    def _solve_one_variant(vid: int, gv: Dict[str, float]) -> Dict[str, Any]:
        g11 = float(gv.get("g11", 1.0))
        g22 = float(gv.get("g22", 25.0))
        g33 = float(gv.get("g33", 25.0))

        # 1x first
        try:
            r1 = _run_rs3_single_local(
                ct,
                os_cost_full,
                np.asarray(p0_xy_full, dtype=np.float32),
                np.asarray(p1_xy_full, dtype=np.float32),
                g11=g11, g22=g22, g33=g33,
                solver_dtype=solver_dtype,
                mode_tag=str(mode_tag or "rs3"),
            )
            if r1.get("ok"):
                return {
                    "ok": True,
                    "variant_id": vid,
                    "g11": g11, "g22": g22, "g33": g33,
                    "track_full_xy": _map_local_to_global(r1["track_local_xy"]),
                    "timing": r1.get("timing", {}),
                    "used_downsample": 1,
                    "used_corridor_refine": False,
                }
            raise RuntimeError(r1.get("error", "1x RS3 failed"))
        except Exception as e1:
            if not _is_oom_exception(e1):
                return {
                    "ok": False,
                    "variant_id": vid,
                    "g11": g11, "g22": g22, "g33": g33,
                    "error": f"[1x non-OOM] {type(e1).__name__}: {e1}\n{traceback.format_exc()}",
                }

            last_e: BaseException = e1
            gc.collect()
            _free_cupy_pools()

            # Coarse fallback(s)
            for ds in (2, 4):
                try:
                    os_cost_ds = _downsample_cost(os_cost_full, ds)
                    p0_ds = _scale_point_xy(np.asarray(p0_xy_full, dtype=np.float32), 1.0 / ds)
                    p1_ds = _scale_point_xy(np.asarray(p1_xy_full, dtype=np.float32), 1.0 / ds)
                    r_ds = _run_rs3_single_local(
                        ct,
                        os_cost_ds,
                        p0_ds,
                        p1_ds,
                        g11=g11, g22=g22, g33=g33,
                        solver_dtype=solver_dtype,
                        mode_tag=f"{mode_tag}_ds{ds}",
                    )
                    if not r_ds.get("ok"):
                        raise RuntimeError(r_ds.get("error", "coarse RS3 failed"))

                    coarse_path_local_full = _scale_path_xy(r_ds["track_local_xy"], float(ds))

                    if enable_corridor_refine:
                        rr = refine_with_corridor_windows(
                            ct,
                            os_cost_full,
                            np.asarray(p0_xy_full, dtype=np.float32),
                            np.asarray(p1_xy_full, dtype=np.float32),
                            coarse_path_local_full,
                            g11=g11, g22=g22, g33=g33,
                            solver_dtype=solver_dtype,
                            params=corridor_params,
                            mode_tag=str(mode_tag or "rs3"),
                        )
                        if rr.get("ok"):
                            timing = dict(r_ds.get("timing", {}))
                            timing.update(rr.get("timing", {}))
                            return {
                                "ok": True,
                                "variant_id": vid,
                                "g11": g11, "g22": g22, "g33": g33,
                                "track_full_xy": _map_local_to_global(rr["track_local_xy"]),
                                "timing": timing,
                                "used_downsample": ds,
                                "used_corridor_refine": True,
                            }
                        # Refine failed: if it was OOM-like, fall back to coarse result.
                        if verbose:
                            print(f"[RS3] v{vid} corridor refine failed at ds={ds}: {rr.get('error', 'unknown')}")

                    return {
                        "ok": True,
                        "variant_id": vid,
                        "g11": g11, "g22": g22, "g33": g33,
                        "track_full_xy": _map_local_to_global(coarse_path_local_full),
                        "timing": r_ds.get("timing", {}),
                        "used_downsample": ds,
                        "used_corridor_refine": False,
                    }

                except Exception as e2:
                    last_e = e2
                    gc.collect()
                    _free_cupy_pools()
                    if not _is_oom_exception(e2):
                        break

            return {
                "ok": False,
                "variant_id": vid,
                "g11": g11, "g22": g22, "g33": g33,
                "error": f"[OOM after fallbacks] {type(last_e).__name__}: {last_e}",
            }

    results: List[Dict[str, Any]] = []

    if can_parallel:
        if verbose:
            print(f"[RS3] parallel small-metric mode enabled (metric~{metric_gib_1x:.2f} GiB)")
        ctx = get_context(mp_start_method)
        tasks: List[Dict[str, Any]] = []
        for vid, gv in enumerate(g_variants):
            tasks.append({
                "ct": ct,
                "os_cost": np.asarray(os_cost_full, dtype=np.float32, order="C")
                if not (CUPY_AVAILABLE and hasattr(cp, "ndarray") and isinstance(os_cost_full, cp.ndarray))
                else os_cost_full,
                "p0_xy": np.asarray(p0_xy_full, dtype=np.float32),
                "p1_xy": np.asarray(p1_xy_full, dtype=np.float32),
                "g11": float(gv.get("g11", 1.0)),
                "g22": float(gv.get("g22", 25.0)),
                "g33": float(gv.get("g33", 25.0)),
                "solver_dtype": solver_dtype,
                "mode_tag": mode_tag,
            })
        try:
            req_workers = min(cpu_max_workers, len(tasks))
            pool_workers, pool_init, pool_initargs, pool_cpus = process_pool_affinity_config(
                req_workers, label="rs3-small"
            )
            if verbose:
                print(f"[AFFINITY] rs3-small workers={pool_workers} cpus={pool_cpus}")
            with ProcessPoolExecutor(
                max_workers=pool_workers,
                mp_context=ctx,
                initializer=pool_init,
                initargs=pool_initargs,
            ) as ex:
                futs = {ex.submit(_rs3_worker_small, t): i for i, t in enumerate(tasks)}
                for fut in as_completed(futs):
                    i = futs[fut]
                    try:
                        r = fut.result()
                        if r.get("ok"):
                            gv = g_variants[i]
                            results.append({
                                "ok": True,
                                "variant_id": i,
                                "g11": float(gv.get("g11", 1.0)),
                                "g22": float(gv.get("g22", 25.0)),
                                "g33": float(gv.get("g33", 25.0)),
                                "track_full_xy": _map_local_to_global(r["track_local_xy"]),
                                "timing": r.get("timing", {}),
                                "used_downsample": 1,
                                "used_corridor_refine": False,
                            })
                        else:
                            results.append(_solve_one_variant(i, g_variants[i]))
                    except Exception:
                        results.append(_solve_one_variant(i, g_variants[i]))
        except Exception:
            # Fallback to sequential if ct or other payload is not picklable.
            results.clear()
            for vid, gv in enumerate(g_variants):
                results.append(_solve_one_variant(vid, gv))
                gc.collect()
                _free_cupy_pools()
    else:
        if verbose:
            print(f"[RS3] sequential mode (metric~{metric_gib_1x:.2f} GiB, threshold={parallel_metric_gib_threshold:.2f})")
        for vid, gv in enumerate(g_variants):
            results.append(_solve_one_variant(vid, gv))
            gc.collect()
            _free_cupy_pools()

    results.sort(key=lambda z: int(z.get("variant_id", 0)))
    return results


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
    solver_dtype: str = "float64",
) -> List[Dict[str, Any]]:
    """
    Backward-compatible entrypoint using the adaptive streamed-prestage runner.
    """
    return run_rs3_variants_split_adaptive(
        ct=ct,
        os_cost=os_cost,
        p0_down_xy=p0_down_xy,
        p1_down_xy=p1_down_xy,
        g_variants=g_variants,
        down=down,
        bbox_xyxy=bbox_xyxy,
        cpu_max_workers=cpu_max_workers,
        mp_start_method=mp_start_method,
        solver_dtype=solver_dtype,
        verbose=False,
    )
def _variant_desc(vid, g, edge):
    return dict(
        variant_id=vid,
        label=f"v{vid} — g11={g.get('g11',1):g}, g22={g.get('g22',25):g}, g33={g.get('g33',25):g}, "
              f"w={edge.get('window_half_size')}, μ={edge.get('mu')}, l={edge.get('l')}, p={edge.get('p')}",
        g11=float(g.get("g11", 1.0)),
        g22=float(g.get("g22", 25.0)),
        g33=float(g.get("g33", 25.0)),
        win=int(edge.get("window_half_size")),
        mu=float(edge.get("mu")),
        ell=int(edge.get("l")),
        p=int(edge.get("p")),
    )

def plot_midlines_overlay_all(image_crop_rgb, man_local_xy, var_local_xy_by_id, variant_labels_by_id,
                              save_overlay_png, save_legend_png):
    import matplotlib.pyplot as plt
    import numpy as np
    import re

    # safe defaults
    im = image_crop_rgb
    if im.ndim == 2:
        import cv2
        im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def _fmt_intish(v):
        try:
            vf = float(v)
            if abs(vf - round(vf)) < 1e-9:
                return str(int(round(vf)))
            return f"{vf:g}"
        except Exception:
            return str(v)

    def _parse_label(lbl):
        s = str(lbl or "")
        m = re.search(
            r"^\s*([A-Za-z0-9_]+)\s*:\s*g11=([^\s]+)\s+g22=([^\s]+)\s+g33=([^\s]+)",
            s,
        )
        if not m:
            return None
        mode = str(m.group(1)).lower()
        try:
            g11 = float(m.group(2))
            g22 = float(m.group(3))
            g33 = float(m.group(4))
        except Exception:
            return None
        return {
            "mode": mode,
            "g11": g11,
            "g22": g22,
            "g33": g33,
            "short": f"{mode},{_fmt_intish(g11)},{_fmt_intish(g22)},{_fmt_intish(g33)}",
        }

    records = []
    for vid, xy in sorted((var_local_xy_by_id or {}).items()):
        if xy is None:
            continue
        arr = np.asarray(xy, float)
        if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 2:
            continue
        info = _parse_label((variant_labels_by_id or {}).get(vid, f"v{vid}"))
        if info is None:
            continue
        info["vid"] = int(vid)
        info["xy"] = arr
        records.append(info)

    def _match(mode, g22, g33=None, g11=1.0):
        if g33 is None:
            g33 = g22
        for r in records:
            if r["mode"] != str(mode).lower():
                continue
            if abs(float(r["g11"]) - float(g11)) > 1e-9:
                continue
            if abs(float(r["g22"]) - float(g22)) > 1e-9:
                continue
            if abs(float(r["g33"]) - float(g33)) > 1e-9:
                continue
            return r
        return None

    old_100 = _match("old", 100.0)
    new_100 = _match("new", 100.0)

    flex_new = []
    for r in records:
        if r["mode"] != "new":
            continue
        if abs(float(r["g11"]) - 1.0) > 1e-9:
            continue
        if abs(float(r["g22"]) - float(r["g33"])) > 1e-9:
            continue
        if abs(float(r["g22"]) - 100.0) <= 1e-9:
            continue
        flex_new.append(r)
    flex_new = sorted(flex_new, key=lambda r: (float(r["g22"]), float(r["g33"]), int(r["vid"])))

    panels = []
    panels.append({
        "title": "new,1,100,100",
        "curves": [("manual", np.asarray(man_local_xy, float))],
    })
    if old_100 is not None:
        panels[0]["curves"].append((old_100["short"], old_100["xy"]))
    if new_100 is not None:
        panels[0]["curves"].append((new_100["short"], new_100["xy"]))

    for r in flex_new:
        curves = [("manual", np.asarray(man_local_xy, float))]
        if old_100 is not None:
            curves.append((old_100["short"], old_100["xy"]))
        curves.append((r["short"], r["xy"]))
        panels.append({
            "title": r["short"],
            "curves": curves,
        })

    if not panels:
        panels = [{"title": "Manual", "curves": [("manual", np.asarray(man_local_xy, float))]}]

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(max(6, 4.5 * n), 5.0), dpi=180)
    if n == 1:
        axes = [axes]

    # Consistent colors/styles across panels
    color_map = {
        "manual": ("black", "-", 2.4),
    }
    if old_100 is not None:
        color_map[old_100["short"]] = ("red", "--", 1.8)
    if new_100 is not None:
        color_map[new_100["short"]] = ("dodgerblue", "-", 1.8)
    flex_palette = ["limegreen", "orange", "magenta", "cyan", "gold", "deepskyblue", "violet"]
    for i, r in enumerate(flex_new):
        color_map.setdefault(r["short"], (flex_palette[i % len(flex_palette)], "-", 1.6))

    for ax, panel in zip(axes, panels):
        ax.imshow(im, origin="upper")

        for lbl, xy in panel["curves"]:
            arr = np.asarray(xy, float)
            if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 2:
                continue
            key = str(lbl).lower()
            if key == "manual":
                # white halo for readability
                ax.plot(arr[:, 0], arr[:, 1], "-", lw=4.2, color="white", alpha=0.9, zorder=2)
                c, ls, lw = color_map["manual"]
                ax.plot(arr[:, 0], arr[:, 1], ls, lw=lw, color=c, label="manual", zorder=3)
            else:
                c, ls, lw = color_map.get(lbl, ("tab:gray", "-", 1.6))
                ax.plot(arr[:, 0], arr[:, 1], ls, lw=lw, color=c, label=lbl, zorder=3)

        ax.set_title(panel["title"], fontsize=9)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="lower right", fontsize=6, framealpha=0.9)

    fig.suptitle("manual vs baseline vs variants (os_ablation, g11, g22, g33)", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_overlay_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Legend is embedded in each subplot; skip separate legend tile.
    if save_legend_png:
        try:
            # Keep downstream tooling from failing on missing file by writing a tiny note.
            fig_leg = plt.figure(figsize=(3.5, 1.0), dpi=160)
            fig_leg.text(0.5, 0.5, "Legend embedded in overlay", ha="center", va="center", fontsize=8)
            fig_leg.savefig(save_legend_png, dpi=160, bbox_inches="tight")
            plt.close(fig_leg)
        except Exception:
            pass
