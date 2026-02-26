from __future__ import annotations

import gc
import os
import tempfile
import traceback
import ctypes
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cupy as cp  # type: ignore
    CUPY_AVAILABLE = True
except Exception:
    cp = None
    CUPY_AVAILABLE = False

# ============================================================
# RS3 PARALLELIZATION CONTROLS
# ============================================================

RS3_ENABLE_PARALLEL = True          # <-- MASTER SWITCH
RS3_MAX_WORKERS = 4              # None = auto
RS3_MEMORY_FRACTION = 0.75           # max fraction of total RAM to use
RS3_ASSUME_PER_WORKER_GIB = 11.0      # empirical ds=1 peak


def _rs3_compute_safe_worker_count() -> int:
    """
    Estimate safe worker count based on system RAM and empirical
    per-worker RS3 memory footprint.
    """
    try:
        import psutil
        total_gib = psutil.virtual_memory().total / (1024**3)
    except Exception:
        total_gib = 32.0

    usable = total_gib * float(RS3_MEMORY_FRACTION)
    per = float(RS3_ASSUME_PER_WORKER_GIB)
    max_by_mem = max(1, int(usable // per))

    if RS3_MAX_WORKERS is not None:
        return max(1, min(int(RS3_MAX_WORKERS), max_by_mem))

    return max_by_mem


def _rs3_result_is_oom(res: Dict[str, Any]) -> bool:
    if not isinstance(res, dict):
        return False
    if res.get("ok"):
        return False
    err = str(res.get("error", "")).lower()
    return (
        "oom" in err
        or "out of memory" in err
        or "unable to allocate" in err
        or "paging file is too small" in err
    )


def hard_gpu_cleanup(cp_mod: Optional[Any]) -> None:
    if cp_mod is not None:
        try:
            cp_mod.cuda.Device().synchronize()
        except Exception:
            pass
        try:
            cp_mod.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        try:
            cp_mod.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
    gc.collect()


def hard_cpu_cleanup(*objs: Any) -> None:
    _ = objs
    # Windows-only best-effort heap compaction after large frees.
    try:
        ctypes.CDLL("msvcrt")._heapmin()
    except Exception:
        pass
    gc.collect()


def _close_memmap_file(arr: Any) -> None:
    """Best-effort close of numpy.memmap backing file (Windows keeps handle open)."""
    try:
        if isinstance(arr, np.memmap):
            try:
                arr.flush()
            except Exception:
                pass
            mm = getattr(arr, "_mmap", None)
            if mm is not None:
                try:
                    mm.close()
                except Exception:
                    pass
    except Exception:
        pass


def _cleanup_rs3_tmp_metric(metric_obj: Any, tmp_path: Optional[str]) -> None:
    """Close memmap handles first, then unlink temp file."""
    try:
        _close_memmap_file(metric_obj)
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _purge_stale_rs3_memmaps(*, older_than_hours: float = 12.0, max_delete: int = 256) -> int:
    """
    Best-effort cleanup of stale rs3 memmap temp files from prior failed runs.
    Conservative: only removes old files matching our prefix in the system temp dir.
    """
    deleted = 0
    try:
        tmp_dir = tempfile.gettempdir()
        cutoff = time.time() - float(older_than_hours) * 3600.0
        for name in os.listdir(tmp_dir):
            if deleted >= int(max_delete):
                break
            if not (name.startswith("rs3_metric_") and name.endswith(".dat")):
                continue
            path = os.path.join(tmp_dir, name)
            try:
                st = os.stat(path)
                if st.st_mtime > cutoff:
                    continue
                os.remove(path)
                deleted += 1
            except Exception:
                continue
    except Exception:
        pass
    return deleted


def metric_bytes(Nx: int, Ny: int, No: int, dtype=np.float64) -> int:
    return int(3 * 3 * Nx * Ny * No * np.dtype(dtype).itemsize)


def human_bytes(n: int) -> str:
    x = float(n)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if x < 1024.0:
            return f"{x:.1f} {unit}"
        x /= 1024.0
    return f"{x:.1f} PiB"


@dataclass
class RS3SolveConfig:
    solver_dtype: str = "float64"
    downsample_attempts: Tuple[int, ...] = (1, 2, 4)
    theta_block: int = 4
    use_memmap_if_large: bool = True
    memmap_threshold_gib: float = 3.5
    verbose: bool = True


def _is_oom_exception(e: BaseException) -> bool:
    msg = str(e).lower()
    np_oom_type = getattr(getattr(np, "core", None), "_exceptions", None)
    np_oom_type = getattr(np_oom_type, "_ArrayMemoryError", ())
    if np_oom_type and isinstance(e, np_oom_type):
        return True
    if isinstance(e, MemoryError):
        return True
    if isinstance(e, OSError) and getattr(e, "winerror", None) == 1455:
        return True
    if "unable to allocate" in msg or "out of memory" in msg or "paging file is too small" in msg:
        return True
    return False


def MultiScaleVesselness_streaming(
    U: np.ndarray,
    ksi: float,
    zeta: float,
    sigmas_s,
    method,
    *,
    CostFunctionVesselnessFiltering_fn,
    OS_MODE: str,
    CUPY_AVAILABLE: bool,
    cp: Optional[Any],
) -> List[np.ndarray]:
    """Optional helper: free temporaries aggressively between sigma passes."""
    import scipy.ndimage

    out: List[np.ndarray] = []
    for sigma in list(sigmas_s)[:2]:
        vesselness = CostFunctionVesselnessFiltering_fn(U, ksi, zeta, sigma, method)
        pad = 5
        vesselness_pad = np.pad(vesselness, pad, mode="wrap")
        vesselness_erosion = scipy.ndimage.grey_erosion(vesselness_pad, size=(3, 0, 0))
        vesselness_erosion = vesselness_erosion[pad:-pad, pad:-pad, pad:-pad]
        if OS_MODE == "old":
            vesselness_erosion[:, :5, :] = 0
        else:
            vesselness_erosion[:, :1, :] = 0
        out.append(vesselness_erosion.astype(np.float32, copy=False))
        del vesselness, vesselness_pad, vesselness_erosion
        gc.collect()
        if CUPY_AVAILABLE and cp is not None:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass
    return out


def build_metric_5d_float64_blocked(
    ct,
    os_cost_np_f32: np.ndarray,
    g11: float,
    g22: float,
    g33: float,
    *,
    theta_block: int = 4,
    use_memmap: bool = False,
    memmap_path: Optional[str] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Build solver metric directly in final layout (3,3,Nx,Ny,No), float64, C-order.
    Avoids allocating (No,Nx,Ny,3,3).
    """
    if os_cost_np_f32.ndim != 3:
        raise ValueError(f"os_cost must be (No,Nx,Ny), got {os_cost_np_f32.shape}")

    No, Nx, Ny = map(int, os_cost_np_f32.shape)
    dims = np.array([No, Nx, Ny], dtype=np.int32)

    # Keep the temporary to 3D (No,Nx,Ny), not 5D.
    cost_sq = np.square(np.asarray(os_cost_np_f32, dtype=np.float64, order="C"), dtype=np.float64)

    metric_theta = ct.tracking.ReedsSheppMetricGFOld_vec(None, dims, g11, g22, g33)
    if hasattr(metric_theta, "get"):
        metric_theta = metric_theta.get()
    metric_theta = np.asarray(metric_theta, dtype=np.float64, order="C")  # (No,3,3)

    if use_memmap:
        if not memmap_path:
            raise ValueError("memmap_path required when use_memmap=True")
        if verbose:
            print(f"[RS3] metric_5d memmap -> {memmap_path}")
        metric_5d = np.memmap(
            memmap_path,
            mode="w+",
            dtype=np.float64,
            shape=(3, 3, Nx, Ny, No),
            order="C",
        )
    else:
        metric_5d = np.empty((3, 3, Nx, Ny, No), dtype=np.float64, order="C")

    block = max(1, int(theta_block))
    for t0 in range(0, No, block):
        t1 = min(No, t0 + block)
        if verbose:
            print(f"[RS3] build block theta[{t0}:{t1}]")
        cs = cost_sq[t0:t1]                      # (b,Nx,Ny)
        cs_xyb = np.transpose(cs, (1, 2, 0))     # (Nx,Ny,b)
        th = metric_theta[t0:t1]                 # (b,3,3)
        for i in range(3):
            for j in range(3):
                metric_5d[i, j, :, :, t0:t1] = cs_xyb * th[:, i, j].reshape(1, 1, -1)
        del cs, cs_xyb, th
        gc.collect()

    del cost_sq, metric_theta
    gc.collect()
    return metric_5d


def runReedsSheppGF_cpu(
    sides: np.ndarray,
    dims: Sequence[int],
    seeds: np.ndarray,
    tips: np.ndarray,
    metric_in: np.ndarray,
) -> List[np.ndarray]:
    from agd import Eikonal
    import numpy as np

    def _as_f64_c(a):
        # Always return a plain ndarray float64 in C order
        if type(a) is np.ndarray and a.dtype == np.float64 and a.flags["C_CONTIGUOUS"]:
            return a
        return np.array(a, dtype=np.float64, order="C", copy=True)

    def _pack_metric_riemann3(metric_any: np.ndarray) -> np.ndarray:
        """
        HFM/AGD expects lower-triangular flatten order for a 3x3 symmetric metric:
          [g_xx, g_xy, g_yy, g_xt, g_yt, g_tt]
        (equivalently [m00, m10, m11, m20, m21, m22]).
        For HFM row-major NumPy input, pass channel-last packed layout:
          shape (Nx, Ny, No, 6)
        (the backend handles its own internal ordering conversion).
        """
        m = metric_any

        # Case A: already packed channel-last (Nx,Ny,No,6)
        if getattr(m, "ndim", None) == 4 and m.shape[-1] == 6:
            return _as_f64_c(m)

        # Case B: packed channel-first (6,Nx,Ny,No) -> move channels last
        if getattr(m, "ndim", None) == 4 and m.shape[0] == 6:
            m = _as_f64_c(m)
            return _as_f64_c(np.moveaxis(m, 0, -1))

        # Case C: full symmetric (3,3,Nx,Ny,No)
        if getattr(m, "ndim", None) == 5 and m.shape[:2] == (3, 3):
            m = _as_f64_c(m)  # ensure ndarray before slicing
            Nx, Ny, No = int(m.shape[2]), int(m.shape[3]), int(m.shape[4])
            packed = np.empty((Nx, Ny, No, 6), dtype=np.float64, order="C")

            packed[..., 0] = m[0, 0, ...]  # g_xx (m00)
            packed[..., 1] = m[1, 0, ...]  # g_xy (m10)
            packed[..., 2] = m[1, 1, ...]  # g_yy (m11)
            packed[..., 3] = m[2, 0, ...]  # g_xt (m20)
            packed[..., 4] = m[2, 1, ...]  # g_yt (m21)
            packed[..., 5] = m[2, 2, ...]  # g_tt (m22)

            return packed

        raise ValueError(
            f"Bad metric shape {getattr(m, 'shape', None)}; expected "
            f"(3,3,Nx,Ny,No) or packed (Nx,Ny,No,6)/(6,Nx,Ny,No)"
        )

    metric = _pack_metric_riemann3(metric_in)

    # sanity vs dims
    NxCost, NyCost, NoCost = map(int, dims)
    if metric.shape[0] != NxCost or metric.shape[1] != NyCost or metric.shape[2] != NoCost or metric.shape[3] != 6:
        raise ValueError(
            f"Metric dims mismatch: metric is ({metric.shape[0]},{metric.shape[1]},{metric.shape[2]},{metric.shape[3]}) "
            f"but dims={dims}"
        )

    print(f"[RS3 HFM] packed metric shape={metric.shape} dtype={metric.dtype} C={bool(metric.flags['C_CONTIGUOUS'])}")

    hfmIn = Eikonal.dictIn({
        "model": "Riemann3_Periodic",
        "seeds": [seeds],
        "tips": [tips],
        "arrayOrdering": "RowMajor",
        "metric": metric,   # <-- IMPORTANT: plain packed ndarray, not Riemann(...)
        "verbosity": 0,
    })
    hfmIn.SetRect(sides=sides, dims=[NxCost, NyCost, NoCost])
    hfmOut = hfmIn.Run()
    return [g.T for g in hfmOut["geodesics"]]


def solve_rs3_one_variant_with_fallback(
    ct,
    os_cost: np.ndarray,
    p0_xy: np.ndarray,
    p1_xy: np.ndarray,
    g11: float,
    g22: float,
    g33: float,
    *,
    cfg: RS3SolveConfig,
) -> Dict[str, Any]:
    import time

    os_cost = np.asarray(os_cost, dtype=np.float32, order="C")
    No, Nx, Ny = map(int, os_cost.shape)

    def make_solver_args(NxCost: int, NyCost: int, NoCost: int, p0_try: np.ndarray, p1_try: np.ndarray):
        s_theta = 2 * np.pi / NoCost
        dims_rs3 = np.array([NxCost, NyCost, NoCost], dtype=np.int32)
        a = np.array([0.0, 2 * np.pi], dtype=np.float64) - (s_theta / 2.0)
        b = np.array([0.0, float(NxCost)], dtype=np.float64)
        c = np.array([0.0, float(NyCost)], dtype=np.float64)
        sides = np.array([b, c, a], dtype=np.float64)
        seeds = np.array([float(p0_try[1]), float(p0_try[0]), np.pi / 2], dtype=np.float64)
        tips = np.array([float(p1_try[1]), float(p1_try[0]), np.pi / 2], dtype=np.float64)
        return sides, dims_rs3, seeds, tips

    last_exc: Optional[BaseException] = None
    for ds in tuple(cfg.downsample_attempts):
        metric_5d = None
        tmp_path = None
        try:
            ds = int(ds)
            if cfg.verbose:
                print(f"[RS3] attempt ds={ds} Nx={Nx//ds} Ny={Ny//ds} No={No}")

            if ds == 1:
                os_try = os_cost
                p0_try = np.asarray(p0_xy, dtype=np.float64)
                p1_try = np.asarray(p1_xy, dtype=np.float64)
            else:
                os_try = os_cost[:, ::ds, ::ds]
                p0_try = np.asarray(p0_xy, dtype=np.float64) / ds
                p1_try = np.asarray(p1_xy, dtype=np.float64) / ds

            NoCost, NxCost, NyCost = map(int, os_try.shape)
            sides, dims_rs3, seeds, tips = make_solver_args(NxCost, NyCost, NoCost, p0_try, p1_try)

            need_bytes = metric_bytes(NxCost, NyCost, NoCost, np.float64)
            use_memmap = bool(cfg.use_memmap_if_large and need_bytes >= int(cfg.memmap_threshold_gib * (1024**3)))
            if cfg.verbose:
                print(f"[RS3] metric bytes ~ {human_bytes(need_bytes)} use_memmap={use_memmap}")
            if use_memmap:
                # Best-effort cleanup from prior crashed runs before allocating another 5GB file.
                _purge_stale_rs3_memmaps(older_than_hours=1.0)
                fd, tmp_path = tempfile.mkstemp(prefix="rs3_metric_", suffix=".dat")
                os.close(fd)

            t0 = time.perf_counter()
            metric_5d = build_metric_5d_float64_blocked(
                ct,
                os_try,
                g11,
                g22,
                g33,
                theta_block=cfg.theta_block,
                use_memmap=use_memmap,
                memmap_path=tmp_path,
                verbose=cfg.verbose,
            )
            t_build = time.perf_counter() - t0

            t0 = time.perf_counter()
            geos = runReedsSheppGF_cpu(sides, dims_rs3, seeds, tips, metric_5d)
            t_solve = time.perf_counter() - t0

            path_yx = geos[0]
            path_xy = np.stack([path_yx[:, 1], path_yx[:, 0]], axis=0).astype(np.float64, copy=False)
            if ds != 1:
                path_xy *= ds

            del geos
            _cleanup_rs3_tmp_metric(metric_5d, tmp_path)
            del metric_5d
            metric_5d = None
            tmp_path = None
            hard_cpu_cleanup()

            return {
                "ok": True,
                "ds_used": ds,
                "track_full_xy_cropdown": path_xy,
                "timing": {
                    "fm_metric_build_sec": float(t_build),
                    "fm_solver_sec": float(t_solve),
                    "fm_total_sec": float(t_build + t_solve),
                    "rs3_build_metric_sec": float(t_build),
                    "rs3_solver_sec": float(t_solve),
                    "rs3_total_sec": float(t_build + t_solve),
                },
            }
        except Exception as e:
            last_exc = e
            if cfg.verbose:
                print(f"[RS3] failed ds={ds}: {type(e).__name__}: {e}")
            if metric_5d is not None:
                try:
                    _cleanup_rs3_tmp_metric(metric_5d, tmp_path)
                    del metric_5d
                except Exception:
                    pass
                metric_5d = None
                tmp_path = None
            hard_cpu_cleanup()
            _cleanup_rs3_tmp_metric(None, tmp_path)
            tmp_path = None
            if not _is_oom_exception(e):
                return {"ok": False, "ds_used": ds, "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}
            continue

    return {"ok": False, "ds_used": None, "error": f"[OOM after fallbacks] {type(last_exc).__name__}: {last_exc}"}


def _rs3_worker_variant(args):
    """
    Worker wrapper for single variant.
    Keeps solver logic untouched.
    """
    import cracktools as ct

    (
        os_cost,
        p0_down_xy,
        p1_down_xy,
        gv,
        down,
        bbox_xyxy,
        cfg,
        vidx,
    ) = args

    xmin, ymin, xmax, ymax = map(int, bbox_xyxy)
    _ = (xmax, ymax)

    g11 = float(gv.get("g11", 1.0))
    g22 = float(gv.get("g22", 25.0))
    g33 = float(gv.get("g33", 25.0))

    r = solve_rs3_one_variant_with_fallback(
        ct,
        os_cost=os_cost,
        p0_xy=np.asarray(p0_down_xy, dtype=np.float64),
        p1_xy=np.asarray(p1_down_xy, dtype=np.float64),
        g11=g11,
        g22=g22,
        g33=g33,
        cfg=cfg,
    )

    if not r.get("ok"):
        return {
            "ok": False,
            "variant_id": vidx,
            "ds_used": r.get("ds_used"),
            "error": r.get("error", "RS3 failed"),
            "timing": r.get("timing", {}),
        }

    path_xy_cd = np.asarray(r["track_full_xy_cropdown"], dtype=np.float64).copy()
    path_xy_cd[0] -= 0.5
    path_xy_cd[1] -= 0.5
    path_xy_cd[0] *= float(down)
    path_xy_cd[1] *= float(down)

    xg = path_xy_cd[0] + float(xmin)
    yg = path_xy_cd[1] + float(ymin)
    track_full_xy = np.vstack([xg, yg])

    return {
        "ok": True,
        "variant_id": vidx,
        "ds_used": r.get("ds_used"),
        "track_full_xy": track_full_xy,
        "timing": r.get("timing", {}),
    }


def run_rs3_variants_memory_stable(
    ct,
    os_cost: np.ndarray,
    p0_down_xy: np.ndarray,
    p1_down_xy: np.ndarray,
    g_variants: List[Dict[str, float]],
    down: int,
    bbox_xyxy: Sequence[int],
    *,
    cfg: RS3SolveConfig,
) -> List[Dict[str, Any]]:
    """
    Memory-stable RS3 variants runner. Returns same core fields as rs3_split:
      ok, variant_id, track_full_xy, timing (+ ds_used on success).
    """
    os_cost = np.asarray(os_cost, dtype=np.float32, order="C")

    # ============================================================
    # PARALLEL MODE (adaptive, OOM-safe)
    # ============================================================
    if RS3_ENABLE_PARALLEL and len(g_variants) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from concurrent.futures.process import BrokenProcessPool

        print("\n========== RS3 PARALLEL DEBUG ==========")
        print("RS3_ENABLE_PARALLEL =", RS3_ENABLE_PARALLEL)
        print("len(g_variants) =", len(g_variants))

        initial_workers = _rs3_compute_safe_worker_count()
        workers = max(1, min(int(initial_workers), len(g_variants)))
        print(f"[RS3 PARALLEL] initial_workers={initial_workers}")
        print(f"[RS3 PARALLEL] starting_workers={workers}")

        args_list = []
        for vidx, gv in enumerate(g_variants):
            args_list.append((
                os_cost,
                np.asarray(p0_down_xy, dtype=np.float64),
                np.asarray(p1_down_xy, dtype=np.float64),
                gv,
                down,
                bbox_xyxy,
                cfg,
                vidx,
            ))

        while workers >= 1:
            print(f"[RS3 PARALLEL] attempting run with workers={workers}")
            pool_workers = workers
            pool_init = None
            pool_initargs = ()
            pool_cpus = None
            try:
                from helpers.cpu_affinity import process_pool_affinity_config

                pool_workers, pool_init, pool_initargs, pool_cpus = process_pool_affinity_config(
                    workers, label="rs3-memory-stable"
                )
            except Exception:
                pass

            if cfg.verbose:
                print(f"[RS3 PARALLEL] attempting workers={pool_workers} (requested={workers})")
                if pool_cpus is not None:
                    print(f"[AFFINITY] rs3-memory-stable workers={pool_workers} cpus={pool_cpus}")

            out: List[Dict[str, Any]] = []
            had_oom = False

            try:
                with ProcessPoolExecutor(
                    max_workers=pool_workers,
                    initializer=pool_init,
                    initargs=pool_initargs,
                ) as ex:
                    futures = [ex.submit(_rs3_worker_variant, a) for a in args_list]

                    for f in as_completed(futures):
                        try:
                            res = f.result()
                        except Exception as e:
                            # Abrupt worker termination often surfaces as BrokenProcessPool,
                            # which is typically an OOM/process-kill in this workload.
                            print(f"[RS3 PARALLEL] worker crashed: {e}")
                            had_oom = True
                            raise BrokenProcessPool(str(e)) from e

                        if _rs3_result_is_oom(res):
                            had_oom = True

                        out.append(res)

                if had_oom:
                    raise MemoryError("RS3 parallel OOM detected")

                out.sort(key=lambda d: d.get("variant_id", -1))
                print(f"[RS3 PARALLEL] success with workers={pool_workers}")
                return out

            except (MemoryError, BrokenProcessPool) as e:
                print(f"[RS3 PARALLEL] ⚠ failure at workers={workers}: {type(e).__name__}")

                workers = max(1, (int(pool_workers) + 1) // 2)
                print(f"[RS3 PARALLEL] reducing workers to {workers}")
                hard_cpu_cleanup()
                try:
                    gc.collect()
                except Exception:
                    pass

                if workers < 1:
                    break

        print("[RS3 PARALLEL] ⛔ falling back to SEQUENTIAL mode")
        print("========================================\n")

    # ============================================================
    # SEQUENTIAL FALLBACK (UNCHANGED LOGIC)
    # ============================================================
    out: List[Dict[str, Any]] = []

    xmin, ymin, xmax, ymax = map(int, bbox_xyxy)
    _ = (xmax, ymax)

    for vidx, gv in enumerate(g_variants):
        g11 = float(gv.get("g11", 1.0))
        g22 = float(gv.get("g22", 25.0))
        g33 = float(gv.get("g33", 25.0))

        if cfg.verbose:
            print(f"\n[RS3] variant v{vidx} g11={g11} g22={g22} g33={g33}")

        r = solve_rs3_one_variant_with_fallback(
            ct,
            os_cost=os_cost,
            p0_xy=np.asarray(p0_down_xy, dtype=np.float64),
            p1_xy=np.asarray(p1_down_xy, dtype=np.float64),
            g11=g11,
            g22=g22,
            g33=g33,
            cfg=cfg,
        )

        if not r.get("ok"):
            out.append({
                "ok": False,
                "variant_id": vidx,
                "ds_used": r.get("ds_used"),
                "error": r.get("error", "RS3 failed"),
                "timing": r.get("timing", {}),
            })
            hard_cpu_cleanup()
            continue

        path_xy_cd = np.asarray(r["track_full_xy_cropdown"], dtype=np.float64).copy()
        path_xy_cd[0] -= 0.5
        path_xy_cd[1] -= 0.5
        path_xy_cd[0] *= float(down)
        path_xy_cd[1] *= float(down)

        xg = path_xy_cd[0] + float(xmin)
        yg = path_xy_cd[1] + float(ymin)
        track_full_xy = np.vstack([xg, yg])

        out.append({
            "ok": True,
            "variant_id": vidx,
            "ds_used": r.get("ds_used"),
            "track_full_xy": track_full_xy,
            "timing": r.get("timing", {}),
        })

        del r, path_xy_cd, track_full_xy
        hard_cpu_cleanup()

    return out
