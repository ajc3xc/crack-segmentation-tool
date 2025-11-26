# --- CPU AFFINITY PINNING (Windows-safe) --- for core ultra
import os
try:
    import psutil
    _p = psutil.Process(os.getpid())
    # Pin to P-cores only
    _p.cpu_affinity([0, 1, 2, 3])
    print(f"[AFFINITY] Worker PID={os.getpid()} pinned to {_p.cpu_affinity()}")
except Exception as e:
    print(f"[AFFINITY] Could not set affinity: {e}")


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

# ---- (A) PRESTAGE: build RS3 inputs (main process; can use ct.tracking helpers) ----
def rs3_prestage_variant(
    ct,
    os_cost: np.ndarray,
    p0_down_xy: np.ndarray,
    p1_down_xy: np.ndarray,
    g11: float = 1.0,
    g22: float = 25.0,
    g33: float = 25.0,
    include_metric_fallback: bool = True,
) -> Dict[str, Any]:
    """
    Build all inputs required by runReedsSheppGF for ONE (g11,g22,g33) variant.

    This refactor:
      - Uses ct.tracking.ReedsSheppMetricGFOld_vec + ct.tracking.IncludeCost
        so that metric semantics match tracking.fast_marching's optimized path.
      - Attaches simple timing information for metric build / include / transpose.

    Returns a dict with:
      ok, sides, dims, seeds, tips, metric_meta, timing, (optional) metric_5d.
    """
    import numpy as _np
    from time import perf_counter

    NoCost, NxCost, NyCost = int(os_cost.shape[0]), int(os_cost.shape[1]), int(os_cost.shape[2])
    s_theta = 2 * _np.pi / NoCost

    # Identity GF per voxel (same setup as in tracking.fast_marching)
    gfLIF = _np.zeros((NoCost, NxCost, NyCost, 3, 3), dtype=_np.float32)
    gfLIF[..., 0, 0] = 1
    gfLIF[..., 1, 1] = 1
    gfLIF[..., 2, 2] = 1

    dims = _np.array([NoCost, NxCost, NyCost], dtype=_np.int32)

    timing = {
        "fm_metric_build_sec": 0.0,
        "fm_include_cost_sec": 0.0,
        "fm_transpose_sec": 0.0,
    }

    try:
        # 1) Metric build (vectorized over theta, broadcast over x,y inside IncludeCost)
        t0 = perf_counter()
        metric_theta = ct.tracking.ReedsSheppMetricGFOld_vec(gfLIF, dims, g11, g22, g33)  # (No,3,3)
        timing["fm_metric_build_sec"] = float(perf_counter() - t0)

        # 2) Include cost: same pattern as tracking.fast_marching
        t0 = perf_counter()
        cost_sq_input = os_cost**2
        metricLIFinclCostOld = ct.tracking.IncludeCost(cost_sq_input, metric_theta)  # (No,Nx,Ny,3,3)
        timing["fm_include_cost_sec"] = float(perf_counter() - t0)

        # 3) Transpose to (3,3,Nx,Ny,No) for Riemann3_Periodic
        t0 = perf_counter()
        metric_5d = metricLIFinclCostOld.transpose((3, 4, 1, 2, 0)).astype(_np.float32, copy=False)
        timing["fm_transpose_sec"] = float(perf_counter() - t0)

        # Rect + dims for RS3 (dims in [Nx,Ny,No] order for AGD)
        a = _np.array([0, 2 * _np.pi]) - s_theta / 2
        b = _np.array([0, NxCost]); c = _np.array([0, NyCost])
        sides = _np.array([b, c, a], dtype=_np.float32)
        dims_rs3 = _np.array([NxCost, NyCost, NoCost], dtype=_np.int32)

        # seeds/tips as [y,x,theta] (π/2 orientation like your original code)
        seeds = _np.array([p0_down_xy[1], p0_down_xy[0], _np.pi / 2], dtype=_np.float32)
        tips  = _np.array([p1_down_xy[1], p1_down_xy[0], _np.pi / 2], dtype=_np.float32)

        # put metric into SHM so workers *could* attach without copying
        meta = shm_from_array(metric_5d)

        out = dict(
            ok=True,
            sides=sides,
            dims=dims_rs3,
            seeds=seeds,
            tips=tips,
            metric_meta=meta,
            timing=timing,
        )
        if include_metric_fallback:
            # Fallback: embed metric_5d directly in the payload (what workers actually use now)
            out["metric_5d"] = metric_5d
        return out

    except Exception as e:
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            "timing": timing,
        }

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
    """
    Child process: consume metric_5d (currently passed directly),
    run RS3, and attach solver timing. Metric build timings come from the task.
    """
    import numpy as np
    from time import perf_counter

    try:
        metric = task.get("metric_5d", None)
        if metric is None:
            return {
                "ok": False,
                "variant_id": task.get("variant_id"),
                "error": "Missing metric_5d in task (SHM-disabled version)"
            }

        t0 = perf_counter()
        geos = _runReedsSheppGF_cpu(
            task["sides"], task["dims"], task["seeds"], task["tips"], metric
        )
        solver_sec = float(perf_counter() - t0)

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

    This version:
      - Uses the same metric builder path as tracking.fast_marching's optimized modes.
      - Returns per-variant timing dicts under result["timing"] (fm_* keys).
      - Keeps the external API (os_cost, p0_down_xy, p1_down_xy, g_variants, down, bbox_xyxy, ...).

    Returns list of dicts with:
      {
        "ok": bool,
        "variant_id": int,
        "track_full_xy": (2,N) [x,y] in GLOBAL coords,
        "timing": {...},          # when ok=True
        "error": "...",           # when ok=False
      }
    """
    cpu_max_workers = cpu_max_workers or min(os.cpu_count() or 8, 16)
    xmin, ymin, xmax, ymax = map(int, bbox_xyxy)
    prepared: List[Dict[str, Any]] = []
    metas: List[Dict[str, Any]] = []  # to close later

    # ----------------------------------------------------------
    # A) PRESTAGE (main process): build per-variant RS3 inputs
    # ----------------------------------------------------------
    for vidx, gv in enumerate(g_variants):
        pre = rs3_prestage_variant(
            ct,
            os_cost,
            p0_down_xy,
            p1_down_xy,
            g11=gv.get("g11", 1.0),
            g22=gv.get("g22", 25.0),
            g33=gv.get("g33", 25.0),
            include_metric_fallback=True,   # keep True until SHM is fully trusted
        )
        if not pre.get("ok"):
            prepared.append({
                "variant_id": vidx,
                "metric_5d": None,
                "sides": None,
                "dims": None,
                "seeds": None,
                "tips": None,
                "timing": pre.get("timing", {}),
                "error": pre.get("error", "prestage failed"),
            })
            continue

        metas.append(pre["metric_meta"])  # so we can unlink SHM later

        # trim payload passed to workers; we use metric_5d fallback, plus timing
        prepared.append({
            "variant_id": vidx,
            "metric_meta": pre["metric_meta"],
            "metric_5d": pre.get("metric_5d"),  # explicit fallback
            "sides": pre["sides"],
            "dims":  pre["dims"],
            "seeds": pre["seeds"],
            "tips":  pre["tips"],
            "timing": pre.get("timing", {}),
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
            try:
                close_shm(m)
            except Exception:
                pass
        raise

    # ----------------------------------------------------------
    # C) Map to GLOBAL coords and ensure start==p0
    # ----------------------------------------------------------
    out: List[Dict[str, Any]] = []
    for r in results:
        if not r.get("ok"):
            out.append(r)
            continue

        path_xy_cd = r["track_full_xy_cropdown"]  # (2,N) [x,y] crop-down
        # half-pixel convention → crop
        path_xy_cd = path_xy_cd.copy()
        path_xy_cd[0] -= 0.5
        path_xy_cd[1] -= 0.5

        track_crop_xy = path_xy_cd
        track_crop_xy[0] *= down
        track_crop_xy[1] *= down

        # crop → global
        xg = track_crop_xy[0] + xmin
        yg = track_crop_xy[1] + ymin
        track_full_xy = np.vstack([xg, yg])  # (2,N)

        out.append({
            "ok": True,
            "variant_id": r.get("variant_id"),
            "track_full_xy": track_full_xy,
            "timing": r.get("timing", {}),
        })

    out.sort(key=lambda z: z.get("variant_id", 0))
    return out

# helper for human label + flat params to store alongside metrics
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
    from itertools import cycle

    # safe defaults
    im = image_crop_rgb
    if im.ndim == 2:
        import cv2
        im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # line styles & colors (repeat as needed)
    styles = cycle(["-", "--", "-.", ":"])
    # 10 distinct-ish colors
    palette = [
        (31/255,119/255,180/255),
        (255/255,127/255,14/255),
        (44/255,160/255,44/255),
        (214/255,39/255,40/255),
        (148/255,103/255,189/255),
        (140/255,86/255,75/255),
        (227/255,119/255,194/255),
        (127/255,127/255,127/255),
        (188/255,189/255,34/255),
        (23/255,190/255,207/255),
    ]
    colors = cycle(palette)

    # -------- overlay ----------
    plt.figure(figsize=(8, 8), dpi=150)
    plt.imshow(im, origin="upper")
    # manual with white outline + black core
    if len(man_local_xy) > 1:
        x, y = man_local_xy[:, 0], man_local_xy[:, 1]
        plt.plot(x, y, '-', lw=4, color='white', alpha=0.9)
        plt.plot(x, y, '-', lw=2, color='black', label='manual')

    # autos
    handles = []
    for vid, xy in sorted(var_local_xy_by_id.items()):
        if xy is None or len(xy) < 2:
            continue
        col = next(colors); ls = next(styles)
        h, = plt.plot(
            xy[:, 0], xy[:, 1],
            ls, lw=2, color=col,
            label=variant_labels_by_id.get(vid, f"v{vid}")
        )
        handles.append(h)

    plt.title("auto vs manual (crop)")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(save_overlay_png, dpi=200)
    plt.close()

    # -------- legend-only tile ----------
    if handles:
        fig_leg = plt.figure(figsize=(6, 2 + 0.2 * len(handles)), dpi=200)
        fig_leg.legend(
            handles=handles,
            labels=[h.get_label() for h in handles],
            loc="center",
            frameon=False,
            ncol=1,
        )
        fig_leg.canvas.draw()
        fig_leg.savefig(save_legend_png, dpi=200, bbox_inches="tight")
        plt.close(fig_leg)
