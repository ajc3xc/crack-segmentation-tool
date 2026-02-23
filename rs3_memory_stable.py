from __future__ import annotations

import gc
import os
import tempfile
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cupy as cp  # type: ignore
    CUPY_AVAILABLE = True
except Exception:
    cp = None
    CUPY_AVAILABLE = False


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
    gc.collect()


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
    metric_5d_f64: np.ndarray,
) -> List[np.ndarray]:
    from agd import Eikonal
    from agd.Metrics import Riemann

    # HFM binding is strict: it wants a plain numpy.ndarray[float64] in C order.
    # np.memmap (subclass), non-contiguous views, or odd strides can be rejected.
    flags = getattr(metric_5d_f64, "flags", None)
    is_c_contig = bool(flags["C_CONTIGUOUS"]) if flags is not None else False
    if (type(metric_5d_f64) is not np.ndarray) or (getattr(metric_5d_f64, "dtype", None) != np.float64) or (not is_c_contig):
        metric_5d_f64 = np.array(metric_5d_f64, dtype=np.float64, order="C", copy=True)
    if metric_5d_f64.ndim != 5 or metric_5d_f64.shape[:2] != (3, 3):
        raise ValueError(f"Bad metric shape {metric_5d_f64.shape}, expected (3,3,Nx,Ny,No)")
    try:
        metric_arg = Riemann(metric_5d_f64)

        hfmIn = Eikonal.dictIn({
            "model": "Riemann3_Periodic",
            "seeds": [seeds],
            "tips": [tips],
            "arrayOrdering": "RowMajor",
            "metric": metric_arg,
            "verbosity": 0,
        })
        hfmIn.SetRect(sides=sides, dims=list(map(int, dims)))
        hfmOut = hfmIn.Run()
        return [g.T for g in hfmOut["geodesics"]]
    except TypeError as e:
        msg = str(e)
        if "set_array()" in msg and "'metric'" in msg:
            # Avoid huge ndarray dumps in the traceback by rethrowing a short diagnostic.
            raw_flags = getattr(metric_5d_f64, "flags", None)
            raw_c = bool(raw_flags["C_CONTIGUOUS"]) if raw_flags is not None else None
            shape = getattr(metric_5d_f64, "shape", None)
            dtype = getattr(metric_5d_f64, "dtype", None)
            exact_type = type(metric_5d_f64)
            metric_arg_obj = locals().get("metric_arg", None)
            arg_type = type(metric_arg_obj) if metric_arg_obj is not None else None
            arg_shape = getattr(metric_arg_obj, "shape", None)
            arg_dtype = getattr(metric_arg_obj, "dtype", None)
            arg_flags = getattr(metric_arg_obj, "flags", None)
            arg_c = bool(arg_flags["C_CONTIGUOUS"]) if arg_flags is not None else None
            raise TypeError(
                "HFM metric type mismatch for set_array('metric'): "
                f"raw_type={exact_type.__name__}, raw_shape={shape}, raw_dtype={dtype}, raw_C={raw_c}; "
                f"arg_type={getattr(arg_type, '__name__', arg_type)}, arg_shape={arg_shape}, "
                f"arg_dtype={arg_dtype}, arg_C={arg_c}. "
                "Expected plain numpy.ndarray[numpy.float64] in C order (after AGD unwrap)."
            ) from None
        raise


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
            del metric_5d
            metric_5d = None
            gc.collect()
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

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
                    del metric_5d
                except Exception:
                    pass
            gc.collect()
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            if not _is_oom_exception(e):
                return {"ok": False, "ds_used": ds, "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}
            continue

    return {"ok": False, "ds_used": None, "error": f"[OOM after fallbacks] {type(last_exc).__name__}: {last_exc}"}


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
    Sequential, memory-stable RS3 variants runner. Returns same core fields as rs3_split:
      ok, variant_id, track_full_xy, timing (+ ds_used on success).
    """
    xmin, ymin, xmax, ymax = map(int, bbox_xyxy)
    _ = (xmax, ymax)

    os_cost = np.asarray(os_cost, dtype=np.float32, order="C")
    out: List[Dict[str, Any]] = []

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
            gc.collect()
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
        gc.collect()

    return out
