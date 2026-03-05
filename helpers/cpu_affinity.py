#!/usr/bin/env python3
"""Best-effort CPU affinity helpers (prefer P-cores on Linux hybrid CPUs)."""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

try:
    import psutil  # type: ignore
except Exception:
    psutil = None


def _current_affinity() -> List[int]:
    if hasattr(os, "sched_getaffinity"):
        try:
            return sorted(os.sched_getaffinity(0))
        except Exception:
            pass

    if psutil is not None:
        try:
            return sorted(psutil.Process().cpu_affinity())
        except Exception:
            pass

    try:
        n = psutil.cpu_count(logical=True) if psutil is not None else os.cpu_count()
    except Exception:
        n = os.cpu_count()
    return list(range(n or 1))


def _linux_core_type(cpu_id: int) -> Optional[int]:
    path = f"/sys/devices/system/cpu/cpu{cpu_id}/topology/core_type"
    try:
        with open(path, "r", encoding="ascii") as f:
            return int(f.read().strip())
    except Exception:
        return None


def detect_preferred_pcore_cpus() -> List[int]:
    """
    Return logical CPU ids for the highest Linux core_type present (usually P-cores).
    Falls back to current affinity if topology data is unavailable.
    """
    if os.name != "posix" or not hasattr(os, "sched_setaffinity"):
        return _current_affinity()

    allowed = _current_affinity()
    typed: List[Tuple[int, int]] = []
    for cpu in allowed:
        ct = _linux_core_type(cpu)
        if ct is not None:
            typed.append((cpu, ct))

    if not typed:
        return allowed

    best_type = max(ct for _, ct in typed)
    picked = [cpu for cpu, ct in typed if ct == best_type]
    return picked or allowed


def apply_process_affinity(cpu_ids: Sequence[int], label: str = "worker") -> None:
    """Best-effort process affinity pinning for current process."""
    if not cpu_ids:
        return
    pinset = sorted(set(int(c) for c in cpu_ids))
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(pinset))
            return
        if psutil is not None:
            psutil.Process().cpu_affinity(pinset)
            return
        print(
            f"[AFFINITY] {label} pid={os.getpid()} setaffinity unsupported on this platform",
            flush=True,
        )
    except Exception as e:
        print(f"[AFFINITY] {label} pid={os.getpid()} setaffinity failed: {e}", flush=True)


def process_pool_affinity_config(
    requested_workers: Optional[int],
    *,
    label: str = "pool",
    enable_worker_pinning: bool = True,
):
    """
    Return (max_workers, initializer, initargs, cpu_ids) for ProcessPoolExecutor.
    """
    cpu_ids = detect_preferred_pcore_cpus()
    if requested_workers is None or int(requested_workers) <= 0:
        max_workers = max(1, len(cpu_ids))
    else:
        max_workers = max(1, min(int(requested_workers), len(cpu_ids)))

    initializer = apply_process_affinity if enable_worker_pinning else None
    initargs = (list(cpu_ids), f"{label}-worker") if enable_worker_pinning else ()
    return max_workers, initializer, initargs, list(cpu_ids)
