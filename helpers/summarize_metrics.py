import os
import glob
import json
import shutil
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import re

plt.ioff()


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg)


def _clean_plot_label(s: str) -> str:
    """Normalize plot titles and axis labels to consistent terminology."""
    replacements = [
        (r"\bscore_mid_wmean\b", "Mid Score"),
        (r"\bscore_mid\b", "Mid Score"),
        (r"\blwmean_score_mid\b", "Mid Score"),
        (r"\(combined cracks,?\s*lower\s*=\s*better\)", "(combined)"),
        (r"\(combined cracks\)", "(combined)"),
        (r"\(combined,?\s*lower\s*is\s*better\)", "(combined)"),
        (r",?\s*lower\s*(is|=)\s*better", ""),
        (r"\s+by method\b", ""),
        (r"\s+per method\b", ""),
        (r"\s*\(box\+whisker\)", ""),
        (r"\s*\(mean \+ IQR \+ outliers\)", ""),
        (r"^Dataset width MAE\b", "Mean Width MAE"),
        (r"^Dataset width RMSE\b", "Mean Width RMSE"),
        (r"^Dataset width \|bias\|\b", "Mean Width |Bias|"),
        (r"^Dataset width bias\b", "Mean Width Bias"),
        (r"^Dataset width metrics\b", "Mean Width MAE"),
        (r"RS3-Style Score", "Mid Score"),
        (r"score_mid_wmean \(lower is better\)", "Mid Score"),
    ]
    result = str(s or "")
    for pattern, repl in replacements:
        result = re.sub(pattern, repl, result, flags=re.IGNORECASE)
    return result.strip()


def _safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        if not os.path.isfile(path):
            return None
        df = pd.read_csv(path)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def _numeric_cols(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    exclude = set(exclude or [])
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _aggregate_numeric(
    df: pd.DataFrame,
    *,
    group_cols: List[str],
    numeric_cols: List[str],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    missing_group = [c for c in group_cols if c not in df.columns]
    if missing_group:
        return pd.DataFrame()

    if "image" not in df.columns:
        df = df.copy()
        df["image"] = "unknown"

    agg_spec = {
        "n_rows": ("image", "size"),
        "n_images": ("image", "nunique"),
    }
    for c in numeric_cols:
        agg_spec[f"{c}_mean"] = (c, "mean")
        agg_spec[f"{c}_median"] = (c, "median")
        agg_spec[f"{c}_std"] = (c, "std")
    return df.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()


def _save_bar(
    labels: List[str],
    values: List[float],
    *,
    colors: Optional[List[str]] = None,
    color_legend: Optional[List[Tuple[str, str]]] = None,
    out_png: str,
    title: str,
    ylabel: str,
    rotate: int = 30,
) -> None:
    if not labels or not values:
        return
    arr = np.asarray(values, float)
    keep = np.isfinite(arr)
    if not np.any(keep):
        return
    labs = [labels[i] for i in range(len(labels)) if keep[i]]
    vals = arr[keep]
    if len(labs) == 0:
        return
    fig_w = max(7.0, 0.45 * len(labs))
    plt.figure(figsize=(fig_w, 4.2), dpi=180)
    xs = np.arange(len(labs))
    bar_kwargs = {}
    if colors is not None:
        try:
            color_arr = [colors[i] for i in range(len(colors)) if i < len(keep) and keep[i]]
            if len(color_arr) == len(labs):
                bar_kwargs["color"] = color_arr
        except Exception:
            pass
    plt.bar(xs, vals, **bar_kwargs)
    plt.xticks(xs, labs, rotation=rotate, ha="right")
    plt.ylabel(_clean_plot_label(ylabel))
    plt.title(_clean_plot_label(title))
    if color_legend:
        handles = [Patch(facecolor=c, edgecolor="none", label=str(lbl)) for lbl, c in color_legend]
        plt.legend(handles=handles, loc="best", framealpha=0.9, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def _display_timing_component_name(name: str) -> str:
    s = str(name)
    low = s.lower()
    if low.startswith("shared_mat_gpu"):
        return "MAT"
    if low.startswith("skeleton_dse"):
        return "DSE pruning"
    s = re.sub(r"(_sec|_s)$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"_width", "", s, flags=re.IGNORECASE)
    # Only strip _dse/_raw for non-MAT methods.
    first = s.split("_")[0].lower()
    if first in ("eob", "esd", "pca"):
        s = re.sub(r"_dse$", "", s, flags=re.IGNORECASE)
        s = re.sub(r"_raw$", "", s, flags=re.IGNORECASE)
        s = {"eob": "EOB", "esd": "ESD", "pca": "PCA"}[first]
    elif first == "mat":
        suffix = ""
        if "_dse" in s.lower():
            suffix = " (DSE)"
        elif "_raw" in s.lower():
            suffix = " (raw)"
        s = "MAT" + suffix
    return s


def _display_width_method_label(name: str) -> str:
    """
    Clean up width baseline method display names.
    Preserve MAT raw vs DSE distinction.
    """
    s0 = str(name or "").strip()
    key = s0.lower()
    width_method_display = {
        "mat_width_raw": "MAT (raw)",
        "mat_width_dse": "MAT (DSE)",
        "pca_width_dse": "PCA",
        "esd_width_dse": "ESD",
        "eob_width_dse": "EOB",
        "skel_mat_raw": "MAT (raw)",
        "skel_mat_dse": "MAT (DSE)",
    }
    if key in width_method_display:
        return width_method_display[key]
    s = s0
    s = re.sub(r"(_sec|_s)$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"_width", "", s, flags=re.IGNORECASE)
    first = s.split("_")[0].lower()
    if first in ("eob", "esd", "pca"):
        s = {"eob": "EOB", "esd": "ESD", "pca": "PCA"}[first]
    elif first == "mat":
        suffix = " (DSE)" if "_dse" in s.lower() else (" (raw)" if "_raw" in s.lower() else "")
        s = "MAT" + suffix
    return s if s else str(name or "")


_METHOD_DISPLAY_NAMES = {
    "dt_ridge_valley": "dt_ridge",
    "dt_ridge_valley_depth": "dt_ridge_depth",
    "dt_ridge_color_depth": "dt_ridge_color_depth",
    "best_dt_depth": "dt_depth",
    "best_dt": "dt",
    "skel_mat_dse": "MAT (DSE)",
    "skel_mat_raw": "MAT (raw)",
}


def _display_method_name(name: str) -> str:
    return _METHOD_DISPLAY_NAMES.get(str(name), str(name))


def _build_width_method_name(df: pd.DataFrame) -> pd.Series:
    """Converts method_family / baseline_method / midline_type cols to display label."""
    if not {"method_family", "baseline_method", "midline_type"}.issubset(df.columns):
        return df.get("midline_type", pd.Series(["unknown"] * len(df), index=df.index)).astype(str)
    mf = df["method_family"].astype(str)
    bm = df["baseline_method"].astype(str)
    mt = df["midline_type"].astype(str).map(_display_method_name)
    return pd.Series(
        np.where(
            mf.str.lower().eq("baseline"),
            bm.map(_display_width_method_label).where(bm.str.len() > 0, "baseline"),
            mt.where(mt.str.len() > 0, "model"),
        ),
        index=df.index,
    )


def _classify_width_source(row) -> str:
    """Returns 'baseline', 'ET', or 'model'."""
    mf = str(row.get("method_family", "") or "").lower()
    if mf == "baseline":
        return "baseline"
    bm = str(row.get("baseline_method", "") or "").strip().lower()
    mn = str(row.get("method_name", "") or "").strip().lower()
    baseline_keys = {"mat_width_raw", "mat_width_dse", "pca_width_dse", "esd_width_dse", "eob_width_dse", "skel_mat_raw", "skel_mat_dse"}
    if bm in baseline_keys:
        return "baseline"
    if any(tok in mn for tok in ("mat", "pca", "esd", "eob")):
        return "baseline"
    if _is_et_like(str(row.get("midline_type", "") or "")) or _is_et_like(str(row.get("method_name", "") or "")):
        return "ET"
    return "model"


def _plot_width_metric(
    *,
    df: pd.DataFrame,
    metric_col: str,
    out_png: str,
    title: str,
    ylabel: str,
    plot_type: str = "bar",
    max_methods: int = 20,
    color_map: Optional[Dict] = None,
) -> bool:
    if df is None or df.empty or metric_col not in df.columns:
        return False
    d = df.copy()
    d["method_name"] = _build_width_method_name(d)
    d["source_class"] = d.apply(_classify_width_source, axis=1)
    d[metric_col] = pd.to_numeric(d[metric_col], errors="coerce")
    d = d[np.isfinite(d[metric_col].to_numpy(float))].copy()
    if d.empty:
        return False

    cmap = color_map or {"baseline": "#2ca02c", "model": "#1f77b4", "ET": "#d62728"}

    if str(plot_type).lower() == "box":
        order = (
            d.groupby("method_name", dropna=False)[metric_col]
            .size()
            .sort_values(ascending=False)
            .head(int(max_methods))
            .index.astype(str)
            .tolist()
        )
        d = d[d["method_name"].astype(str).isin(order)].copy()
        if d.empty:
            return False
        med_order = (
            d.groupby("method_name", dropna=False)[metric_col]
            .median()
            .sort_values(ascending=True)
            .index.astype(str)
            .tolist()
        )
        d["method_name"] = pd.Categorical(d["method_name"].astype(str), categories=med_order, ordered=True)
        fig_w = max(9.0, 0.55 * len(med_order))
        fig, ax = plt.subplots(figsize=(fig_w, 5.0), dpi=180)
        parts = ax.boxplot(
            [d.loc[d["method_name"] == m, metric_col].to_numpy(float) for m in med_order],
            labels=med_order,
            patch_artist=True,
            showfliers=True,
            medianprops=dict(color="black", linewidth=1.2),
            whiskerprops=dict(color="#555555"),
            capprops=dict(color="#555555"),
            flierprops=dict(marker="o", markersize=3, alpha=0.5, markerfacecolor="#888888", markeredgecolor="#888888"),
        )
        cls_map = {}
        for m in med_order:
            g = d.loc[d["method_name"] == m]
            if g.empty:
                cls_map[m] = "model"
                continue
            cls_map[m] = str(g["source_class"].iloc[0])
        for box, m in zip(parts["boxes"], med_order):
            box.set_facecolor(cmap.get(cls_map.get(m, "model"), "#4c78a8"))
            box.set_alpha(0.65)
        ax.set_ylabel(_clean_plot_label(ylabel))
        ax.set_title(_clean_plot_label(title))
        ax.set_xticklabels(med_order, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(
            handles=[
                Patch(facecolor=cmap.get("ET", "#d62728"), edgecolor="none", label="ET"),
                Patch(facecolor=cmap.get("model", "#1f77b4"), edgecolor="none", label="model (dt/dt_depth)"),
                Patch(facecolor=cmap.get("baseline", "#2ca02c"), edgecolor="none", label="baseline"),
            ],
            loc="best",
            fontsize=8,
            framealpha=0.9,
        )
        plt.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        return True

    rows = []
    for m, g in d.groupby("method_name", dropna=False):
        v = pd.to_numeric(g[metric_col], errors="coerce").to_numpy(float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        q1 = float(np.percentile(v, 25))
        q3 = float(np.percentile(v, 75))
        iqr_val = float(q3 - q1)
        lo_fence = float(q1 - 1.5 * iqr_val)
        hi_fence = float(q3 + 1.5 * iqr_val)
        outliers = v[(v < lo_fence) | (v > hi_fence)]
        rows.append(
            {
                "method_name": str(m),
                "mean": float(np.mean(v)),
                "q1": q1,
                "q3": q3,
                "outliers": outliers,
                "source_class": str(g["source_class"].iloc[0]) if "source_class" in g.columns else "model",
                "n": int(v.size),
            }
        )
    if not rows:
        return False

    p = pd.DataFrame(rows).sort_values("mean", ascending=True).head(int(max_methods))
    x = np.arange(len(p), dtype=float)
    y = p["mean"].to_numpy(float)
    lo = np.clip(y - p["q1"].to_numpy(float), 0.0, None)
    hi = np.clip(p["q3"].to_numpy(float) - y, 0.0, None)
    cols = [cmap.get(str(c), "#4c78a8") for c in p["source_class"].astype(str).tolist()]
    fig, ax = plt.subplots(figsize=(max(8.0, 0.55 * len(p)), 4.8), dpi=180)
    ax.bar(x, y, color=cols, alpha=0.85)
    ax.errorbar(x, y, yerr=np.vstack([lo, hi]), fmt="none", ecolor="#f58518", elinewidth=1.3, capsize=3)
    for i, arr in enumerate(p["outliers"].tolist()):
        if arr is None or len(arr) == 0:
            continue
        yy = np.asarray(arr, float)
        xx = np.full(len(yy), float(x[i])) + np.linspace(-0.08, 0.08, len(yy))
        ax.scatter(xx, yy, s=10, color="#d62728", alpha=0.8, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(p["method_name"].astype(str).tolist(), rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(_clean_plot_label(ylabel))
    ax.set_title(_clean_plot_label(title))
    ax.grid(axis="y", alpha=0.2)
    ax.legend(
        handles=[
            Patch(facecolor=cmap.get("ET", "#d62728"), edgecolor="none", label="ET"),
            Patch(facecolor=cmap.get("model", "#1f77b4"), edgecolor="none", label="model (dt/dt_depth)"),
            Patch(facecolor=cmap.get("baseline", "#2ca02c"), edgecolor="none", label="baseline"),
            Patch(facecolor="#f58518", edgecolor="none", label="IQR"),
        ],
        loc="best",
        fontsize=8,
        framealpha=0.9,
    )
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return True


def _list_image_metric_dirs(metrics_root: str) -> List[str]:
    if not os.path.isdir(metrics_root):
        return []
    out = []
    # HARD-CODED TEMP FILTER:
    # Set to False (or comment this line) to include folders containing "_".
    SKIP_DIRS_WITH_UNDERSCORE = True
    for name in sorted(os.listdir(metrics_root)):
        if name.startswith("_"):
            continue
        if SKIP_DIRS_WITH_UNDERSCORE and ("_" in name):
            continue
        p = os.path.join(metrics_root, name)
        if os.path.isdir(p):
            out.append(p)
    return out


def _parse_width_summary_context(rel_path: str) -> Tuple[str, str, str]:
    parts = rel_path.replace("\\", "/").split("/")
    baseline_method = ""
    method_family = "model"
    midline_type = "unknown"

    if len(parts) >= 2 and parts[0] in ("manual", "auto", "et", "ET"):
        midline_type = parts[0]
    elif len(parts) >= 3 and parts[1] in ("manual", "auto", "et", "ET"):
        baseline_method = str(parts[0])
        method_family = "baseline"
        midline_type = parts[1]
    elif parts:
        midline_type = parts[0]

    return midline_type, method_family, baseline_method


def _parse_midline_context(rel_path: str) -> Tuple[str, str, str]:
    parts = rel_path.replace("\\", "/").split("/")
    _KNOWN_BASELINE_METHODS = {
        "eob_width_dse", "esd_width_dse", "mat_width_dse", "mat_width_raw",
        "pca_width_dse", "skel_mat_dse", "skel_mat_raw",
    }
    method_family = "baseline" if (len(parts) > 0 and parts[0].lower() in _KNOWN_BASELINE_METHODS) else "model"
    baseline_method = parts[0] if method_family == "baseline" else ""
    midline_type = "unknown"

    # Normalize: find where "midline_metrics" appears in path
    try:
        mm_idx = next(i for i, p in enumerate(parts) if "midline_metrics" in p.lower())
    except StopIteration:
        mm_idx = -1

    if mm_idx == 1:
        # model: {midline_type}/midline_metrics/...
        midline_type = parts[0]
        method_family = "model"
    elif mm_idx == 2:
        # baseline: {method}/{midline_type}/midline_metrics/...
        baseline_method = str(parts[0])
        method_family = "baseline"
        midline_type = parts[1]
    elif mm_idx == 3:
        # baseline with crack_type: {method}/{midline_type}/{crack_type}/midline_metrics_*.csv
        baseline_method = str(parts[0])
        method_family = "baseline"
        midline_type = parts[1]
    elif parts:
        # fallback: use filename to detect baseline
        fname = parts[-1].lower()
        if "baseline" in fname:
            # try to infer from path structure
            if len(parts) >= 2:
                baseline_method = str(parts[0])
                method_family = "baseline"
                midline_type = parts[1] if len(parts) > 2 else "ET"
            else:
                midline_type = parts[0]
        else:
            midline_type = parts[0]

    return midline_type, method_family, baseline_method


def _is_et_like(midline_type: str) -> bool:
    s = str(midline_type or "").strip().lower()
    if s in {"manual", "et"}:
        return True
    # Catch compound labels like "manual:geodesic", "ET:geodesic", "et:anything"
    if s.startswith("manual:") or s.startswith("et:"):
        return True
    return False


def _display_midline_label(midline_type: str) -> str:
    if _is_et_like(midline_type):
        return "ET"
    s = str(midline_type or "")
    if s.startswith("best_"):
        s = s[len("best_"):]
    return _display_method_name(s)


def _aggregate_mask_metrics(
    image_dirs: List[str],
    out_dir: str,
    *,
    verbose: bool,
) -> Dict[str, str]:
    def _find_col_ci(df: pd.DataFrame, name: str) -> Optional[str]:
        low = str(name).lower()
        for c in df.columns:
            if str(c).lower() == low:
                return c
        return None

    def _weighted_mean(series: pd.Series, weights: np.ndarray) -> float:
        x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        w = np.asarray(weights, dtype=float)
        ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
        if not np.any(ok):
            return float("nan")
        return float(np.sum(x[ok] * w[ok]) / np.sum(w[ok]))

    def _safe_file_tag(s: str) -> str:
        return "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in str(s))

    def _plot_mask_triplet_variant(
        *,
        region: Dict[str, float],
        boundary: Dict[str, float],
        cm: np.ndarray,
        out_png: str,
        title: str,
    ) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=160)

        def _bar(ax, data, ttl):
            labels = list(data.keys())
            vals = np.asarray(list(data.values()), float)
            xs = np.arange(len(vals))
            vals_plot = np.where(np.isfinite(vals), vals, 0.0)
            ax.bar(xs, vals_plot)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, fontsize=8, fontweight="bold")
            ax.set_ylim(0.0, 1.0)
            ax.set_title(ttl, fontsize=12, fontweight="bold")
            for i, v in enumerate(vals):
                txt = "nan" if not np.isfinite(v) else f"{v:.3f}"
                y = 0.02 if not np.isfinite(v) else min(float(v) + 0.03, 0.98)
                ax.text(xs[i], y, txt, ha="center", fontsize=7)

        _bar(axes[0], region, "Region metrics (weighted mean)")
        _bar(axes[1], boundary, "Boundary metrics (weighted mean)")

        im = axes[2].imshow(cm, cmap="Blues")
        axes[2].set_xticks([0, 1])
        axes[2].set_yticks([0, 1])
        axes[2].set_xticklabels(["Pred +", "Pred -"])
        axes[2].set_yticklabels(["GT +", "GT -"])
        axes[2].set_title("Confusion matrix (summed counts)", fontsize=12, fontweight="bold")
        for (rr, cc), val in np.ndenumerate(cm):
            axes[2].text(cc, rr, f"{int(round(float(val)))}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.02)

        fig.suptitle(title, fontsize=12, fontweight="bold")
        plt.tight_layout()
        fig.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close(fig)

    outputs = {}
    mask_dir = os.path.join(out_dir, "mask")
    os.makedirs(mask_dir, exist_ok=True)
    frames = []
    for img_dir in image_dirs:
        image = os.path.basename(img_dir)
        p = os.path.join(img_dir, "mask_metrics.csv")
        df = _safe_read_csv(p)
        if df is None:
            continue
        df = df.copy()
        if "image" not in df.columns:
            df["image"] = image
        if "supervision" not in df.columns:
            df["supervision"] = "manual"
        if "method" not in df.columns:
            df["method"] = "geodesic"
        if "crack_type" not in df.columns:
            df["crack_type"] = "unknown"
        df["variant"] = (
            df["supervision"].astype(str).str.strip()
            + ":"
            + df["method"].astype(str).str.strip()
        )
        frames.append(df)

    if not frames:
        return outputs

    all_df = pd.concat(frames, ignore_index=True)
    all_csv = os.path.join(mask_dir, "dataset_mask_metrics_all.csv")
    all_df.to_csv(all_csv, index=False)
    outputs["mask_all_csv"] = all_csv

    # Dataset-level mask triplet summaries by variant:
    # weighted means for region/boundary + summed confusion counts.
    triplet_src = all_df.copy()
    ct_col = _find_col_ci(triplet_src, "crack_type")
    if ct_col is not None:
        m_total = triplet_src[ct_col].astype(str).str.upper() == "TOTAL"
        if m_total.any():
            triplet_src = triplet_src[m_total].copy()
            _log(verbose, f"[summarize] mask triplet: using TOTAL rows only ({int(len(triplet_src))} rows)")
        else:
            _log(verbose, "[summarize] mask triplet: TOTAL rows not found; falling back to all mask rows")
    else:
        _log(verbose, "[summarize] mask triplet: crack_type column missing; falling back to all mask rows")

    if not triplet_src.empty and "variant" in triplet_src.columns:
        tp_col = _find_col_ci(triplet_src, "tp")
        fp_col = _find_col_ci(triplet_src, "fp")
        fn_col = _find_col_ci(triplet_src, "fn")
        tn_col = _find_col_ci(triplet_src, "tn")
        area_col = _find_col_ci(triplet_src, "gt_area_px") or _find_col_ci(triplet_src, "union_area_px")

        triplet_rows = []
        triplet_dir = os.path.join(mask_dir, "mask_triplets")
        os.makedirs(triplet_dir, exist_ok=True)

        for variant, g in triplet_src.groupby("variant", dropna=False):
            if g.empty:
                continue

            if all(c is not None for c in (tp_col, fp_col, fn_col, tn_col)):
                weights = (
                    pd.to_numeric(g[tp_col], errors="coerce").fillna(0.0).to_numpy(float)
                    + pd.to_numeric(g[fp_col], errors="coerce").fillna(0.0).to_numpy(float)
                    + pd.to_numeric(g[fn_col], errors="coerce").fillna(0.0).to_numpy(float)
                    + pd.to_numeric(g[tn_col], errors="coerce").fillna(0.0).to_numpy(float)
                )
            elif area_col is not None:
                weights = pd.to_numeric(g[area_col], errors="coerce").fillna(0.0).to_numpy(float)
            else:
                weights = np.ones(len(g), dtype=float)

            def _wm(colname: str) -> float:
                c = _find_col_ci(g, colname)
                if c is None:
                    return float("nan")
                return _weighted_mean(g[c], weights)

            region = {
                "Precision": _wm("precision"),
                "Recall": _wm("recall"),
                "F1": _wm("f1"),
                "IoU": _wm("iou"),
            }
            boundary = {
                "Boundary Precision": _wm("boundary_precision"),
                "Boundary Recall": _wm("boundary_recall"),
                "Boundary F1": _wm("boundary_f1"),
            }

            tp_sum = float(pd.to_numeric(g[tp_col], errors="coerce").fillna(0.0).sum()) if tp_col else 0.0
            fp_sum = float(pd.to_numeric(g[fp_col], errors="coerce").fillna(0.0).sum()) if fp_col else 0.0
            fn_sum = float(pd.to_numeric(g[fn_col], errors="coerce").fillna(0.0).sum()) if fn_col else 0.0
            tn_sum = float(pd.to_numeric(g[tn_col], errors="coerce").fillna(0.0).sum()) if tn_col else 0.0
            cm = np.array([[tp_sum, fn_sum], [fp_sum, tn_sum]], float)

            row = {
                "variant": str(variant),
                "n_rows": int(len(g)),
                "weight_sum": float(np.nansum(weights)),
                "tp_sum": tp_sum,
                "fp_sum": fp_sum,
                "fn_sum": fn_sum,
                "tn_sum": tn_sum,
                "precision_wmean": float(region["Precision"]) if np.isfinite(region["Precision"]) else np.nan,
                "recall_wmean": float(region["Recall"]) if np.isfinite(region["Recall"]) else np.nan,
                "f1_wmean": float(region["F1"]) if np.isfinite(region["F1"]) else np.nan,
                "iou_wmean": float(region["IoU"]) if np.isfinite(region["IoU"]) else np.nan,
                "boundary_precision_wmean": float(boundary["Boundary Precision"]) if np.isfinite(boundary["Boundary Precision"]) else np.nan,
                "boundary_recall_wmean": float(boundary["Boundary Recall"]) if np.isfinite(boundary["Boundary Recall"]) else np.nan,
                "boundary_f1_wmean": float(boundary["Boundary F1"]) if np.isfinite(boundary["Boundary F1"]) else np.nan,
            }
            triplet_rows.append(row)

            out_png = os.path.join(triplet_dir, f"dataset_mask_triplet_{_safe_file_tag(variant)}.png")
            _plot_mask_triplet_variant(
                region=region,
                boundary=boundary,
                cm=cm,
                out_png=out_png,
                title=f"Dataset Mask Metrics Triplet - {variant}",
            )

        if triplet_rows:
            triplet_df = pd.DataFrame(triplet_rows).sort_values("variant")
            triplet_csv = os.path.join(mask_dir, "dataset_mask_triplet_weighted_summary.csv")
            triplet_df.to_csv(triplet_csv, index=False)
            outputs["mask_triplet_weighted_summary_csv"] = triplet_csv
            outputs["mask_triplet_png_dir"] = triplet_dir

    num_cols = _numeric_cols(
        all_df,
        exclude=["image", "variant", "supervision", "method", "crack_type", "crack_id", "members", "source_path"],
    )
    grouped = _aggregate_numeric(
        all_df,
        group_cols=["variant", "supervision", "method", "crack_type"],
        numeric_cols=num_cols,
    )
    if not grouped.empty:
        grp_csv = os.path.join(mask_dir, "dataset_mask_metrics_grouped.csv")
        grouped.to_csv(grp_csv, index=False)
        outputs["mask_grouped_csv"] = grp_csv

        total = grouped[grouped["crack_type"].astype(str).str.upper() == "TOTAL"].copy()
        if not total.empty:
            iou_col = next((c for c in ["iou_mean", "iou_manual_vs_gt_mean", "iou_auto_vs_gt_mean"] if c in total.columns), None)
            bf1_col = "boundary_f1_mean" if "boundary_f1_mean" in total.columns else None
            def _pretty_variant(v: str) -> str:
                s = str(v or "").strip()
                sl = s.lower()
                prefix = sl.split(":")[0]
                if _is_et_like(prefix):
                    return "ET"
                if sl.startswith("auto:"):
                    return "auto"
                if sl.startswith("baseline:"):
                    # Show baseline method name only (e.g., sam3, hrsegnet).
                    p = s.split(":", 1)
                    return p[1] if len(p) == 2 and p[1] else "baseline"
                return s

            def _variant_class(v: str) -> str:
                sl = str(v or "").strip().lower()
                prefix = sl.split(":")[0]
                if _is_et_like(prefix):
                    return "ET"
                if sl.startswith("auto:"):
                    return "auto"
                if sl.startswith("baseline:"):
                    return "baseline"
                return "other"

            var_labels = [_pretty_variant(v) for v in total["variant"].astype(str).tolist()]
            cls_vals = [_variant_class(v) for v in total["variant"].astype(str).tolist()]
            cls_color = {
                "ET": "#1f77b4",
                "auto": "#ff7f0e",
                "baseline": "#2ca02c",
                "other": "#7f7f7f",
            }
            var_colors = [cls_color.get(c, "#7f7f7f") for c in cls_vals]
            legend_items = [
                ("ET", cls_color["ET"]),
                ("auto", cls_color["auto"]),
                ("baseline", cls_color["baseline"]),
            ]
            legend_items = [x for x in legend_items if x[0] in set(cls_vals)]
            def _find_col_ci_local(df_in: pd.DataFrame, target: str) -> Optional[str]:
                t = str(target).strip().lower()
                for cc in df_in.columns:
                    if str(cc).strip().lower() == t:
                        return cc
                return None
            if iou_col:
                out_png = os.path.join(mask_dir, "dataset_mask_total_iou_by_variant.png")
                _save_bar(
                    var_labels,
                    total[iou_col].astype(float).tolist(),
                    colors=var_colors,
                    color_legend=legend_items,
                    out_png=out_png,
                    title="Dataset TOTAL IoU by variant",
                    ylabel="IoU",
                )
                outputs["mask_total_iou_png"] = out_png
            if bf1_col:
                out_png = os.path.join(mask_dir, "dataset_mask_total_boundary_f1_by_variant.png")
                _save_bar(
                    var_labels,
                    total[bf1_col].astype(float).tolist(),
                    colors=var_colors,
                    color_legend=legend_items,
                    out_png=out_png,
                    title="Dataset TOTAL boundary F1 by variant",
                    ylabel="Boundary F1",
                )
                outputs["mask_total_bf1_png"] = out_png

            # ASSD/HD95 comparison for manual vs sam3 vs hrsegnet.
            assd_col = _find_col_ci_local(total, "ASSD_mean") or _find_col_ci_local(total, "assd_mean")
            hd95_col = _find_col_ci_local(total, "HD95_mean") or _find_col_ci_local(total, "hd95_mean")
            if assd_col is not None and hd95_col is not None and "variant" in total.columns:
                t0 = total.copy()
                t0["variant_l"] = t0["variant"].astype(str).str.lower()

                def _pick_row(token: str):
                    tok = str(token).strip().lower()
                    m = t0["variant_l"].str.contains(tok, na=False)
                    if not m.any():
                        return None
                    s = t0.loc[m].copy()
                    # prefer TOTAL row with largest support if duplicates exist.
                    if "n_rows" in s.columns:
                        s["n_rows_num"] = pd.to_numeric(s["n_rows"], errors="coerce").fillna(0.0)
                        s = s.sort_values("n_rows_num", ascending=False)
                    return s.iloc[0]

                def _pick_et_row():
                    r = _pick_row("manual:")
                    if r is not None:
                        return r
                    return _pick_row("et:")

                # Build picks dynamically: ET first, then every unique baseline:* variant.
                picks = [("ET", _pick_et_row())]
                seen_baseline_tokens = set()
                for v in t0["variant"].astype(str).tolist():
                    vl = v.strip().lower()
                    if vl.startswith("baseline:"):
                        token = vl
                        if token in seen_baseline_tokens:
                            continue
                        seen_baseline_tokens.add(token)
                        display = v.split(":", 1)[1] if ":" in v else v
                        picks.append((display, _pick_row(token)))
                labels = []
                assd_vals = []
                hd95_vals = []
                for lab, row in picks:
                    if row is None:
                        continue
                    a = float(pd.to_numeric(row.get(assd_col, np.nan), errors="coerce"))
                    h = float(pd.to_numeric(row.get(hd95_col, np.nan), errors="coerce"))
                    if not (np.isfinite(a) and np.isfinite(h)):
                        continue
                    labels.append(lab)
                    assd_vals.append(a)
                    hd95_vals.append(h)

                if labels:
                    palette = ["#f28e2b", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
                    base_idx = 0
                    bar_cols = []
                    for lab in labels:
                        if lab == "ET":
                            bar_cols.append("#4c78a8")
                        else:
                            bar_cols.append(palette[base_idx % len(palette)])
                            base_idx += 1
                    fig_w = max(6.0, 0.7 * len(labels))
                    x = np.arange(len(labels), dtype=float)
                    fig, axes = plt.subplots(1, 2, figsize=(fig_w * 2, 4.2), dpi=180, sharex=True)
                    axes[0].bar(x, assd_vals, color=bar_cols, alpha=0.88)
                    axes[0].set_title("ASSD")
                    axes[0].set_ylabel("ASSD")
                    axes[0].grid(axis="y", alpha=0.2)
                    axes[1].bar(x, hd95_vals, color=bar_cols, alpha=0.88)
                    axes[1].set_title("HD95")
                    axes[1].set_ylabel("HD95")
                    axes[1].grid(axis="y", alpha=0.2)
                    for ax in axes:
                        ax.set_xticks(x)
                        ax.set_xticklabels(labels, rotation=20, ha="right")
                    fig.suptitle("Dataset Mask ASSD / HD95 by method", fontsize=11, fontweight="bold")
                    plt.tight_layout()
                    out_png = os.path.join(mask_dir, "dataset_mask_assd_hd95.png")
                    fig.savefig(out_png, bbox_inches="tight")
                    plt.close(fig)
                    outputs["mask_assd_hd95_compare_png"] = out_png

    _log(verbose, f"[summarize] mask metrics rows={len(all_df)}")
    return outputs


def _load_diffs_frames(image_dirs: List[str]) -> pd.DataFrame:
    """
    Load per-sample width diffs rows from *_width_diffs_combined.csv.
    Each row is approximately 1px arc-length, so row-wise aggregation is
    naturally length-weighted.
    """
    frames = []
    for img_dir in image_dirs:
        image = os.path.basename(img_dir)
        found = glob.glob(
            os.path.join(img_dir, "**", "*_width_diffs_combined.csv"),
            recursive=True,
        )
        for p in found:
            df = _safe_read_csv(p)
            if df is None or df.empty:
                continue

            rel = os.path.relpath(p, img_dir)
            mt, mf, _ = _parse_width_summary_context(rel)
            cols_lower = {str(c).lower(): c for c in df.columns}

            gc = next((cols_lower[k] for k in ("gt_width_px", "gt_width", "gt") if k in cols_lower), None)
            dc = next((cols_lower[k] for k in ("width_diff_px", "diff_px", "diff") if k in cols_lower), None)
            mc = next((cols_lower[k] for k in ("method",) if k in cols_lower), None)

            # ET positional schema fallback: x,y,gt,pred,diff,...
            if gc is None and dc is None and len(df.columns) >= 5:
                if pd.to_numeric(df.iloc[:, 2], errors="coerce").notna().sum() > 0:
                    df = df.rename(columns={df.columns[2]: "gt_width_px", df.columns[4]: "diff_px"})
                    cols_lower = {str(c).lower(): c for c in df.columns}
                    gc = cols_lower.get("gt_width_px")
                    dc = cols_lower.get("diff_px")

            if gc is None or dc is None:
                continue

            d = pd.DataFrame(
                {
                    "gt_width_px": pd.to_numeric(df[gc], errors="coerce"),
                    "diff_px": pd.to_numeric(df[dc], errors="coerce"),
                }
            )

            if mc is not None and str(mf).lower() == "baseline":
                d["_method"] = df[mc].astype(str)
            else:
                d["_method"] = str(mt)

            d["image"] = image
            d["source_relpath"] = rel.replace("\\", "/")
            frames.append(d)

    if not frames:
        return pd.DataFrame(columns=["gt_width_px", "diff_px", "_method", "image", "source_relpath"])

    out = pd.concat(frames, ignore_index=True)
    out["gt_width_px"] = pd.to_numeric(out["gt_width_px"], errors="coerce")
    out["diff_px"] = pd.to_numeric(out["diff_px"], errors="coerce")
    out = out[
        np.isfinite(out["gt_width_px"])
        & np.isfinite(out["diff_px"])
        & (out["gt_width_px"] > 0)
    ].copy()
    return out


def _aggregate_width_metrics(
    image_dirs: List[str],
    out_dir: str,
    *,
    verbose: bool,
) -> Dict[str, str]:
    # Hard-coded switch:
    # True  -> exclude baseline methods whose name starts with "skel_"
    # False -> keep all baseline methods
    EXCLUDE_SKEL_BASELINE_METHODS = True

    outputs = {}
    width_dir = os.path.join(out_dir, "width")
    overview_dir = os.path.join(width_dir, "overview")
    no_et_dir = os.path.join(width_dir, "no_et")
    bias_dir = os.path.join(width_dir, "bias")
    length_weighted_dir = os.path.join(width_dir, "length_weighted")
    for _d in (width_dir, overview_dir, no_et_dir, bias_dir, length_weighted_dir):
        os.makedirs(_d, exist_ok=True)
    frames = []
    COLOR_MAP = {
        "baseline": "#2ca02c",  # green
        "auto": "#1f77b4",      # blue
        "model": "#1f77b4",     # blue (same as auto)
        "ET": "#d62728",        # red
    }

    for img_dir in image_dirs:
        image = os.path.basename(img_dir)
        for p in glob.glob(os.path.join(img_dir, "**", "*_width_summary_*.csv"), recursive=True):
            df = _safe_read_csv(p)
            if df is None:
                continue

            rel = os.path.relpath(p, img_dir)
            midline_type, method_family, baseline_method = _parse_width_summary_context(rel)

            crack_type = "unknown"
            stem = os.path.basename(p)
            token = "_width_summary_"
            if token in stem:
                crack_type = stem.split(token, 1)[1].rsplit(".csv", 1)[0]

            d = df.copy()
            d["image"] = image
            d["midline_type"] = midline_type
            d["method_family"] = method_family
            d["baseline_method"] = baseline_method
            d["crack_type"] = crack_type
            d["source_relpath"] = rel.replace("\\", "/")
            frames.append(d)

    if not frames:
        return outputs

    all_df = pd.concat(frames, ignore_index=True)

    if EXCLUDE_SKEL_BASELINE_METHODS and not all_df.empty:
        m_skel_all = (
            all_df["method_family"].astype(str).str.lower().eq("baseline")
            & (
                all_df["baseline_method"].astype(str).str.lower().str.startswith("skel_")
                | all_df["baseline_method"].astype(str).str.lower().str.contains("skeleton_graph", na=False)
            )
        )
        n_drop_all = int(np.count_nonzero(m_skel_all.to_numpy(dtype=bool)))
        if n_drop_all > 0:
            all_df = all_df.loc[~m_skel_all].copy()
            _log(verbose, f"[summarize] width summary(all): excluded skel_ baseline rows={n_drop_all}")

    if all_df.empty:
        _log(verbose, "[summarize] width summary: empty after skel_ filtering at all_df stage")
        return outputs

    all_csv = os.path.join(overview_dir, "dataset_width_summary_all.csv")
    all_df.to_csv(all_csv, index=False)
    outputs["width_summary_all_csv"] = all_csv

    metric_cols = [c for c in ["n_samples", "mae_px", "rmse_px", "bias_px", "corr"] if c in all_df.columns]

    grouped = _aggregate_numeric(
        all_df,
        group_cols=["midline_type", "method_family", "baseline_method", "crack_type"],
        numeric_cols=metric_cols,
    )

    if grouped.empty:
        return outputs

    # --------------------------------------------------------
    # Clean method labeling (x-axis)
    # --------------------------------------------------------
    grouped["method_name"] = _build_width_method_name(grouped)

    # --------------------------------------------------------
    # Color classification
    # --------------------------------------------------------
    grouped["source_class"] = grouped.apply(_classify_width_source, axis=1)
    grouped["color"] = grouped["source_class"].map(COLOR_MAP)

    if EXCLUDE_SKEL_BASELINE_METHODS and not grouped.empty:
        m_skel = (
            grouped["method_family"].astype(str).str.lower().eq("baseline")
            & (
                grouped["baseline_method"].astype(str).str.lower().str.startswith("skel_")
                | grouped["baseline_method"].astype(str).str.lower().str.contains("skeleton_graph", na=False)
                | grouped["method_name"].astype(str).str.lower().str.contains("skeleton_graph", na=False)
            )
        )
        n_drop = int(np.count_nonzero(m_skel.to_numpy(dtype=bool)))
        if n_drop > 0:
            grouped = grouped.loc[~m_skel].copy()
            _log(verbose, f"[summarize] width summary: excluded skel_ baseline methods rows={n_drop}")

    if grouped.empty:
        _log(verbose, "[summarize] width summary: empty after skel_ baseline filtering")
        return outputs

    grp_csv = os.path.join(overview_dir, "dataset_width_summary_grouped.csv")
    grouped.to_csv(grp_csv, index=False)
    outputs["width_summary_grouped_csv"] = grp_csv

    # Split width error view: atomic vs combined+noncombined_atomic in two columns.
    def _normalize_width_crack_group(v: str) -> str:
        s = str(v or "").strip().lower()
        if s == "atomic":
            return "atomic"
        if ("combined" in s) and ("noncombined" in s) and ("atomic" in s):
            return "combined_plus_noncombined_atomic"
        return ""

    split_df = grouped.copy()
    split_df["crack_group"] = split_df["crack_type"].map(_normalize_width_crack_group)
    split_df = split_df[split_df["crack_group"].astype(str).str.len() > 0].copy()
    if not split_df.empty:
        idx_cols = ["method_name", "midline_type", "method_family", "baseline_method", "source_class"]
        idx_cols = [c for c in idx_cols if c in split_df.columns]
        out_parts = []
        if "mae_px_mean" in split_df.columns:
            piv_mae = (
                split_df.pivot_table(
                    index=idx_cols,
                    columns="crack_group",
                    values="mae_px_mean",
                    aggfunc="mean",
                )
                .reset_index()
                .rename(
                    columns={
                        "atomic": "mae_atomic_error",
                        "combined_plus_noncombined_atomic": "mae_combined_plus_noncombined_atomic_error",
                    }
                )
            )
            out_parts.append(piv_mae)
        if "rmse_px_mean" in split_df.columns:
            piv_rmse = (
                split_df.pivot_table(
                    index=idx_cols,
                    columns="crack_group",
                    values="rmse_px_mean",
                    aggfunc="mean",
                )
                .reset_index()
                .rename(
                    columns={
                        "atomic": "rmse_atomic_error",
                        "combined_plus_noncombined_atomic": "rmse_combined_plus_noncombined_atomic_error",
                    }
                )
            )
            out_parts.append(piv_rmse)
        if out_parts:
            split_out = out_parts[0]
            for nxt in out_parts[1:]:
                split_out = split_out.merge(nxt, on=idx_cols, how="outer")
            split_csv = os.path.join(overview_dir, "dataset_width_error_split.csv")
            split_out.to_csv(split_csv, index=False)
            outputs["width_error_split_csv"] = split_csv

    mae_col = "mae_px_mean" if "mae_px_mean" in grouped.columns else None
    rmse_col = "rmse_px_mean" if "rmse_px_mean" in grouped.columns else None
    total_grouped = grouped[grouped["crack_type"].astype(str).str.upper() == "TOTAL"].copy()
    plot_grouped = total_grouped if not total_grouped.empty else grouped
    legend_items = [(k, v) for k, v in [("ET", COLOR_MAP["ET"]), ("model", COLOR_MAP["model"]), ("baseline", COLOR_MAP["baseline"])]
                    if k in set(plot_grouped["source_class"].astype(str).tolist())]

    _df_total = all_df if total_grouped.empty else all_df[all_df["crack_type"].astype(str).str.upper() == "TOTAL"].copy()
    _plot_jobs = []
    if mae_col:
        _plot_jobs.extend(
            [
                (
                    "width_mae_png",
                    dict(
                        df=_df_total,
                        metric_col="mae_px",
                        out_png=os.path.join(overview_dir, "dataset_width_mae_by_method.png"),
                        title="Dataset width MAE by method (mean + IQR + outliers)",
                        ylabel="MAE (px)",
                        plot_type="bar",
                        color_map=COLOR_MAP,
                    ),
                ),
                (
                    "width_mae_box_png",
                    dict(
                        df=_df_total,
                        metric_col="mae_px",
                        out_png=os.path.join(overview_dir, "dataset_width_mae_box_by_method.png"),
                        title="Dataset width MAE by method (box+whisker)",
                        ylabel="MAE (px)",
                        plot_type="box",
                        color_map=COLOR_MAP,
                    ),
                ),
            ]
        )
    if rmse_col:
        _plot_jobs.extend(
            [
                (
                    "width_rmse_png",
                    dict(
                        df=_df_total,
                        metric_col="rmse_px",
                        out_png=os.path.join(overview_dir, "dataset_width_rmse_by_method.png"),
                        title="Dataset width RMSE by method (mean + IQR + outliers)",
                        ylabel="RMSE (px)",
                        plot_type="bar",
                        color_map=COLOR_MAP,
                    ),
                ),
                (
                    "width_rmse_box_png",
                    dict(
                        df=_df_total,
                        metric_col="rmse_px",
                        out_png=os.path.join(overview_dir, "dataset_width_rmse_box_by_method.png"),
                        title="Dataset width RMSE by method (box+whisker)",
                        ylabel="RMSE (px)",
                        plot_type="box",
                        color_map=COLOR_MAP,
                    ),
                ),
            ]
        )
    if _plot_jobs:
        with ThreadPoolExecutor(max_workers=4) as _pool:
            _futs = {_pool.submit(_plot_width_metric, **kwargs): (out_key, kwargs["out_png"]) for out_key, kwargs in _plot_jobs}
            for _f in as_completed(_futs):
                _key, _png = _futs[_f]
                try:
                    if bool(_f.result()):
                        outputs[_key] = _png
                except Exception as _e_plot:
                    _log(verbose, f"[summarize] plot {_key} failed: {_e_plot}")

    def _plot_length_weighted_summary() -> Dict[str, str]:
        lw_outputs: Dict[str, str] = {}
        diffs = _load_diffs_frames(image_dirs)
        if diffs is None or diffs.empty:
            return lw_outputs

        d = diffs.copy()
        d["_abs_diff"] = np.abs(pd.to_numeric(d["diff_px"], errors="coerce"))
        d["_sq_diff"] = np.square(pd.to_numeric(d["diff_px"], errors="coerce"))
        d = d[np.isfinite(d["diff_px"].to_numpy(float))].copy()
        if d.empty:
            return lw_outputs

        _baseline_raw_keys = {
            "mat_width_raw",
            "mat_width_dse",
            "pca_width_dse",
            "esd_width_dse",
            "eob_width_dse",
            "skel_mat_raw",
            "skel_mat_dse",
        }
        rows = []
        for raw_method, g in d.groupby("_method", dropna=False):
            raw_key = str(raw_method).strip().lower()
            if raw_key in _baseline_raw_keys:
                src_class = "baseline"
            elif _is_et_like(raw_key):
                src_class = "ET"
            else:
                src_class = "model"
            if raw_key in _baseline_raw_keys:
                display_label = _display_width_method_label(raw_key)
            else:
                display_label = _display_method_name(str(raw_method))
            diff = pd.to_numeric(g["diff_px"], errors="coerce").to_numpy(float)
            diff = diff[np.isfinite(diff)]
            if diff.size == 0:
                continue
            ad = np.abs(diff)
            sd = np.square(diff)
            rows.append(
                {
                    "method": str(display_label),
                    "source_class": src_class,
                    "n_samples": int(diff.size),
                    "mae_px": float(np.mean(ad)),
                    "mae_q1": float(np.percentile(ad, 25)),
                    "mae_q3": float(np.percentile(ad, 75)),
                    "rmse_px": float(np.sqrt(np.mean(sd))),
                    "rmse_q1": float(np.sqrt(np.percentile(sd, 25))),
                    "rmse_q3": float(np.sqrt(np.percentile(sd, 75))),
                    "bias_px": float(np.mean(diff)),
                    "bias_q1": float(np.percentile(diff, 25)),
                    "bias_q3": float(np.percentile(diff, 75)),
                }
            )
        if not rows:
            return lw_outputs

        lw_df = pd.DataFrame(rows).sort_values("mae_px", ascending=True)
        lw_df["abs_bias_px"] = lw_df["bias_px"].abs()
        lw_df["abs_bias_q1"] = lw_df[["bias_q1", "bias_q3"]].abs().min(axis=1)
        lw_df["abs_bias_q3"] = lw_df[["bias_q1", "bias_q3"]].abs().max(axis=1)
        lw_csv = os.path.join(length_weighted_dir, "dataset_width_lw_summary.csv")
        lw_df.to_csv(lw_csv, index=False)
        lw_outputs["width_lw_csv"] = lw_csv

        def _plot_lw(df_plot: pd.DataFrame, metric: str, q1_col: str, q3_col: str, out_png: str, title: str, ylabel: str):
            if df_plot is None or df_plot.empty:
                return
            p = df_plot.sort_values(metric, ascending=True).copy()
            x = np.arange(len(p), dtype=float)
            y = pd.to_numeric(p[metric], errors="coerce").to_numpy(float)
            q1 = pd.to_numeric(p[q1_col], errors="coerce").to_numpy(float)
            q3 = pd.to_numeric(p[q3_col], errors="coerce").to_numpy(float)
            lo = np.clip(y - q1, 0.0, None)
            hi = np.clip(q3 - y, 0.0, None)
            cols = [COLOR_MAP.get(str(c), "#4c78a8") for c in p["source_class"].astype(str).tolist()]
            fig, ax = plt.subplots(figsize=(max(8.0, 0.55 * len(p)), 4.8), dpi=180)
            ax.bar(x, y, color=cols, alpha=0.86)
            ax.errorbar(x, y, yerr=np.vstack([lo, hi]), fmt="none", ecolor="#f58518", elinewidth=1.3, capsize=3)
            ax.set_xticks(x)
            ax.set_xticklabels(p["method"].astype(str).tolist(), rotation=35, ha="right", fontsize=8)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.2)
            ax.legend(
                handles=[
                    Patch(facecolor=COLOR_MAP["ET"], edgecolor="none", label="ET"),
                    Patch(facecolor=COLOR_MAP["model"], edgecolor="none", label="model (dt/dt_depth)"),
                    Patch(facecolor=COLOR_MAP["baseline"], edgecolor="none", label="baseline"),
                    Patch(facecolor="#f58518", edgecolor="none", label="IQR"),
                ],
                loc="best",
                fontsize=8,
                framealpha=0.9,
            )
            plt.tight_layout()
            fig.savefig(out_png, bbox_inches="tight")
            plt.close(fig)

        _plot_lw(lw_df, "mae_px", "mae_q1", "mae_q3", os.path.join(length_weighted_dir, "dataset_width_lw_mae.png"), "Length-weighted width MAE by method", "MAE (px)")
        lw_outputs["width_lw_mae_png"] = os.path.join(length_weighted_dir, "dataset_width_lw_mae.png")
        _plot_lw(lw_df, "rmse_px", "rmse_q1", "rmse_q3", os.path.join(length_weighted_dir, "dataset_width_lw_rmse.png"), "Length-weighted width RMSE by method", "RMSE (px)")
        lw_outputs["width_lw_rmse_png"] = os.path.join(length_weighted_dir, "dataset_width_lw_rmse.png")
        _plot_lw(lw_df, "abs_bias_px", "abs_bias_q1", "abs_bias_q3", os.path.join(length_weighted_dir, "dataset_width_lw_bias.png"), "Length-weighted width |bias| by method", "|bias| (px)")
        lw_outputs["width_lw_bias_png"] = os.path.join(length_weighted_dir, "dataset_width_lw_bias.png")

        lw_no_et = lw_df[~lw_df["method"].astype(str).map(_is_et_like)].copy()
        _plot_lw(lw_no_et, "mae_px", "mae_q1", "mae_q3", os.path.join(length_weighted_dir, "dataset_width_lw_mae_no_et.png"), "Length-weighted width MAE by method (no ET)", "MAE (px)")
        lw_outputs["width_lw_mae_no_et_png"] = os.path.join(length_weighted_dir, "dataset_width_lw_mae_no_et.png")
        _plot_lw(lw_no_et, "rmse_px", "rmse_q1", "rmse_q3", os.path.join(length_weighted_dir, "dataset_width_lw_rmse_no_et.png"), "Length-weighted width RMSE by method (no ET)", "RMSE (px)")
        lw_outputs["width_lw_rmse_no_et_png"] = os.path.join(length_weighted_dir, "dataset_width_lw_rmse_no_et.png")
        _plot_lw(lw_no_et, "abs_bias_px", "abs_bias_q1", "abs_bias_q3", os.path.join(length_weighted_dir, "dataset_width_lw_bias_no_et.png"), "Length-weighted width |bias| by method (no ET)", "|bias| (px)")
        lw_outputs["width_lw_bias_no_et_png"] = os.path.join(length_weighted_dir, "dataset_width_lw_bias_no_et.png")
        return lw_outputs

    outputs.update(_plot_width_metrics_with_without_et(all_df, no_et_dir, bias_dir, verbose=verbose))
    outputs.update(_plot_length_weighted_summary())

    # Additional distribution-aware plots: mean bar + IQR + outlier dots.
    # separate distribution plot files removed; canonical by_method plots include IQR/outliers

    _log(verbose, f"[summarize] width summary rows={len(all_df)}")
    return outputs


def _plot_width_metrics_with_without_et(
    df_width: pd.DataFrame,
    out_dir: str,
    bias_dir: Optional[str] = None,
    *,
    verbose: bool = False,
) -> Dict[str, str]:
    """
    Produces:
      - dataset_width_mae_no_et.png (MAE, without ET)
      - dataset_width_bias_by_method.png (|bias|, with ET)
      - dataset_width_bias_no_et.png (|bias|, without ET)
    """
    outputs: Dict[str, str] = {}
    if df_width is None or df_width.empty:
        return outputs
    if bias_dir is None:
        bias_dir = out_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(bias_dir, exist_ok=True)

    color_map = {"baseline": "#2ca02c", "model": "#1f77b4", "ET": "#d62728"}

    def _drop_et(df: pd.DataFrame) -> pd.DataFrame:
        if "method_family" in df.columns:
            is_baseline = df["method_family"].astype(str).str.lower().eq("baseline")
        else:
            is_baseline = pd.Series([False] * len(df), index=df.index)
        mt = df.get("midline_type", pd.Series([""] * len(df), index=df.index)).astype(str)
        return df[is_baseline | ~mt.map(_is_et_like)].copy()

    df_no_et = _drop_et(df_width)
    _jobs = []
    if "mae_px" in df_width.columns and not df_no_et.empty:
        _jobs.append(
            (
                "width_mae_no_et_png",
                dict(
                    df=df_no_et,
                    metric_col="mae_px",
                    out_png=os.path.join(out_dir, "dataset_width_mae_no_et.png"),
                    title="Dataset width MAE by method (mean + IQR + outliers)",
                    ylabel="MAE (px)",
                    plot_type="bar",
                    color_map=color_map,
                ),
            )
        )
    if "bias_px" in df_width.columns:
        df_bias_all = df_width.copy()
        df_bias_all["abs_bias_px"] = pd.to_numeric(df_bias_all["bias_px"], errors="coerce").abs()
        _jobs.append(
            (
                "width_bias_all_png",
                dict(
                    df=df_bias_all,
                    metric_col="abs_bias_px",
                    out_png=os.path.join(bias_dir, "dataset_width_bias_by_method.png"),
                    title="Dataset width |bias| by method (mean + IQR + outliers)",
                    ylabel="|bias| (px)",
                    plot_type="bar",
                    color_map=color_map,
                ),
            )
        )
        if not df_no_et.empty:
            df_bias_no_et = df_no_et.copy()
            df_bias_no_et["abs_bias_px"] = pd.to_numeric(df_bias_no_et["bias_px"], errors="coerce").abs()
            _jobs.append(
                (
                    "width_bias_no_et_png",
                    dict(
                        df=df_bias_no_et,
                        metric_col="abs_bias_px",
                        out_png=os.path.join(out_dir, "dataset_width_bias_no_et.png"),
                        title="Dataset width |bias| by method (mean + IQR + outliers, no ET)",
                        ylabel="|bias| (px)",
                        plot_type="bar",
                        color_map=color_map,
                    ),
                )
            )
    if "rmse_px" in df_width.columns and not df_no_et.empty:
        _jobs.append(
            (
                "width_rmse_no_et_png",
                dict(
                    df=df_no_et,
                    metric_col="rmse_px",
                    out_png=os.path.join(out_dir, "dataset_width_rmse_no_et.png"),
                    title="Dataset width RMSE by method (mean + IQR + outliers, no ET)",
                    ylabel="RMSE (px)",
                    plot_type="bar",
                    color_map=color_map,
                ),
            )
        )
    if _jobs:
        with ThreadPoolExecutor(max_workers=4) as _pool:
            _futs = {_pool.submit(_plot_width_metric, **kwargs): (out_key, kwargs["out_png"]) for out_key, kwargs in _jobs}
            for _f in as_completed(_futs):
                _key, _png = _futs[_f]
                try:
                    if bool(_f.result()):
                        outputs[_key] = _png
                except Exception as _e_plot:
                    _log(verbose, f"[summarize] plot {_key} failed: {_e_plot}")

    _log(verbose, f"[summarize] width metrics plots generated: {list(outputs.keys())}")
    return outputs

def _aggregate_midline_metrics(
    image_dirs: List[str],
    out_dir: str,
    *,
    verbose: bool,
) -> Dict[str, str]:
    outputs = {}
    midline_dir = os.path.join(out_dir, "midline")
    os.makedirs(midline_dir, exist_ok=True)
    EXCLUDE_SKEL_BASELINE_METHODS = False  # skel baselines are valid midline comparisons
    MIDLINE_DEBUG = True
    frames = []
    for img_dir in image_dirs:
        image = os.path.basename(img_dir)
        found = [
            f for f in glob.glob(os.path.join(img_dir, "**", "*_midline_metrics_*.csv"), recursive=True)
            if re.search(r"\d+_midline_metrics_combined_(derived|baseline)\.csv$", os.path.basename(f))
        ]
        found += [
            p for p in glob.glob(os.path.join(img_dir, "**", "midline_metrics*.csv"), recursive=True)
            if p not in found
        ]
        for p in found:
            df = _safe_read_csv(p)
            if df is None:
                continue
            rel = os.path.relpath(p, img_dir)
            midline_type, method_family, baseline_method = _parse_midline_context(rel)
            d = df.copy()
            d["image"] = image
            d["midline_type_path"] = midline_type
            d["method_family"] = method_family
            d["baseline_method"] = baseline_method
            d["source_relpath"] = rel.replace("\\", "/")
            frames.append(d)

        # Also pull midline scores from GT supervision ablation CSV
        _abl_path = os.path.join(
            os.path.dirname(os.path.dirname(img_dir)),  # up from metrics/ to save_folder
            "supervision", image, "analysis", "gt_ablation_midline_metrics.csv"
        )
        if not os.path.isfile(_abl_path):
            # try relative to save_folder sibling
            _abl_path = os.path.join(img_dir, "..", "..", "..", "supervision", image, "analysis", "gt_ablation_midline_metrics.csv")
            _abl_path = os.path.normpath(_abl_path)
        if os.path.isfile(_abl_path):
            _abl_df = _safe_read_csv(_abl_path)
            if _abl_df is not None and not _abl_df.empty and "variant_id" in _abl_df.columns:
                #print(f"[MIDLINE ABL] {image} ablation variant_ids: {_abl_df['variant_id'].unique().tolist()}")
                # Map supervision variant_id -> display midline_type, keep only ET/dt/dt_depth
                _vid_map = {
                    "manual": "ET",
                    "dt": "dt",
                    "dt_depth": "best_dt_depth",
                    "dt_ridge_color_depth": "best_dt_depth",
                    "dt_ridge_valley_depth": "best_dt_depth",
                }
                _abl_df = _abl_df[_abl_df["variant_id"].astype(str).isin(_vid_map)].copy()
                if _abl_df.empty:
                    pass
                else:
                    _abl_df["midline_type"] = _abl_df["variant_id"].astype(str).map(_vid_map)
                    # Keep best score if multiple variants map to same midline_type
                    if "score_mid" in _abl_df.columns:
                        _abl_df = _abl_df.sort_values("score_mid").groupby("midline_type", as_index=False).first()
                    _abl_df["image"] = image
                    _abl_df["midline_type_path"] = _abl_df["midline_type"]
                    _abl_df["method_family"] = "model"
                    _abl_df["baseline_method"] = ""
                    # Use a path that _parse_midline_context will handle correctly
                    # Format: {midline_type}/midline_metrics/... -> parsed as model with midline_type=midline_type
                    _abl_df["source_relpath"] = _abl_df["midline_type"].astype(str).apply(
                        lambda mt: f"{mt}/midline_metrics/combined/gt_ablation_midline_metrics.csv"
                    )
                    if "crack_type" not in _abl_df.columns:
                        _abl_df["crack_type"] = "combined"
                    if "geometry_type" not in _abl_df.columns:
                        _abl_df["geometry_type"] = "derived"
                    frames.append(_abl_df)

    if not frames:
        return outputs

    all_df = pd.concat(frames, ignore_index=True)
    if verbose:
        print(
            "[midline-debug] pre-filter all rows by midline_type:\n"
            + (
                all_df["midline_type"].value_counts(dropna=False).to_string()
                if "midline_type" in all_df.columns
                else "NO midline_type"
            )
        )
        print(
            f"[midline-debug] pre-filter score_mid finite: "
            f"{int(np.isfinite(pd.to_numeric(all_df.get('score_mid', pd.Series()), errors='coerce').to_numpy(float)).sum())} / {len(all_df)}"
        )
    if "crack_type" in all_df.columns:
        _log(verbose, f"[midline-debug] crack_type values:\n{all_df[['midline_type','crack_type','method_family','source_relpath']].head().to_string()}")
    if MIDLINE_DEBUG:
        try:
            dbg_cols = [c for c in ["image", "crack_type", "geometry_type", "method_family", "baseline_method", "midline_type"] if c in all_df.columns]
            dbg_csv = os.path.join(midline_dir, "dataset_midline_debug_all_rows_raw.csv")
            all_df.loc[:, dbg_cols + [c for c in ["score_mid", "length_px", "source_relpath"] if c in all_df.columns]].to_csv(dbg_csv, index=False)
            _log(verbose, f"[midline-debug] wrote raw rows -> {dbg_csv}")
            if "method_family" in all_df.columns:
                _log(verbose, f"[midline-debug] raw method_family counts:\n{all_df['method_family'].astype(str).value_counts(dropna=False).to_string()}")
        except Exception as e:
            _log(verbose, f"[midline-debug] raw dump failed: {e}")
    if "midline_type" not in all_df.columns:
        all_df["midline_type"] = all_df["midline_type_path"]

    # Fallback safety net: ensure score_mid exists and is finite enough for ranking.
    if ("score_mid" not in all_df.columns) or all_df["score_mid"].isna().any():
        nn = pd.to_numeric(all_df.get("nn_mean_bidirectional"), errors="coerce")
        hd = pd.to_numeric(all_df.get("hausdorff_max"), errors="coerce")
        cov = pd.to_numeric(all_df.get("coverage_min"), errors="coerce")
        nn = nn.fillna(0.0)
        hd = hd.fillna(0.0)
        cov = cov.fillna(0.0)
        all_df["score_mid"] = (
            np.log1p(np.maximum(nn.to_numpy(float), 0.0))
            + 0.5 * np.log1p(np.maximum(hd.to_numpy(float), 0.0))
            + (1.0 - np.clip(cov.to_numpy(float), 0.0, 1.0))
        )

    if EXCLUDE_SKEL_BASELINE_METHODS and {"method_family", "baseline_method"}.issubset(all_df.columns):
        m_skel = (
            all_df["method_family"].astype(str).str.lower().eq("baseline")
            & (
                all_df["baseline_method"].astype(str).str.lower().str.startswith("skel_")
                | all_df["baseline_method"].astype(str).str.lower().str.contains("skeleton_graph", na=False)
                | all_df["midline_type"].astype(str).str.lower().str.contains("skeleton_graph", na=False)
            )
        )
        if m_skel.any():
            all_df = all_df.loc[~m_skel].copy()
            if MIDLINE_DEBUG:
                _log(verbose, f"[midline-debug] dropped skel_ baseline rows: {int(np.count_nonzero(m_skel.to_numpy(dtype=bool)))}")

    if all_df.empty:
        return outputs

    all_csv = os.path.join(midline_dir, "dataset_midline_metrics_all.csv")
    all_df.to_csv(all_csv, index=False)
    outputs["midline_all_csv"] = all_csv

    group_cols = ["midline_type", "crack_type", "geometry_type", "method_family", "baseline_method"]
    group_cols = [c for c in group_cols if c in all_df.columns]
    metric_candidates = [
        "score_mid",
        "nn_mean_bidirectional",
        "hausdorff_max",
        "coverage_min",
        "mean_tan_angle_error_deg",
        "relative_length_error",
        "orth_mean",
        "orth_std",
        "frechet_discrete_ds",
    ]
    metric_cols = [c for c in metric_candidates if c in all_df.columns]
    grouped = _aggregate_numeric(all_df, group_cols=group_cols, numeric_cols=metric_cols)
    if not grouped.empty:
        grp_csv = os.path.join(midline_dir, "dataset_midline_metrics_grouped.csv")
        grouped.to_csv(grp_csv, index=False)
        outputs["midline_grouped_csv"] = grp_csv

        if "score_mid_mean" in grouped.columns:
            d = grouped.copy()
            if "crack_type" in d.columns:
                m_comb = d["crack_type"].astype(str).str.lower().str.contains("combined", na=False)
                if m_comb.any():
                    d = d[m_comb].copy()
            # Group by method+crack_type, collapsing geometry_type by taking min score.
            # Include baseline_method so baselines are not merged into ET buckets.
            gcols = [c for c in ["midline_type", "method_family", "baseline_method", "crack_type"] if c in d.columns]
            d = (
                d.groupby(
                    gcols,
                    dropna=False,
                )["score_mid_mean"]
                .min()
                .reset_index()
            )

            def _grp_label(row):
                mf = str(row.get("method_family", "") or "").lower()
                bm = str(row.get("baseline_method", "") or "").strip()
                mt = str(row.get("midline_type", "") or "").strip()
                method = bm if (mf == "baseline" and bm) else (_display_midline_label(mt) if mt else "model")
                return method

            d["group_label"] = d.apply(_grp_label, axis=1)
            d = d.sort_values("score_mid_mean", ascending=True).head(20)

            grp_color_map = {"ET": "#d62728", "auto": "#1f77b4", "model": "#1f77b4", "baseline": "#2ca02c"}

            def _grp_color(row):
                mf = str(row.get("method_family", "") or "").lower()
                mt = str(row.get("midline_type", "") or "")
                if mf == "baseline":
                    return grp_color_map["baseline"]
                if _is_et_like(mt):
                    return grp_color_map["ET"]
                sc = row.get("source_class", "")
                return grp_color_map.get(sc, grp_color_map["auto"])

            grp_colors = d.apply(_grp_color, axis=1).tolist()
            legend_items = [
                ("ET", grp_color_map["ET"]),
                ("model (dt/dt_depth)", grp_color_map.get("model", grp_color_map["auto"])),
                ("baseline", grp_color_map["baseline"]),
            ]
            out_png = os.path.join(midline_dir, "dataset_midline_score_by_method_crack.png")
            _save_bar(
                d["group_label"].astype(str).tolist(),
                d["score_mid_mean"].astype(float).tolist(),
                colors=grp_colors,
                color_legend=legend_items,
                out_png=out_png,
                title="Dataset Midline Score (min over geometry)",
                ylabel="Mid Score",
            )
            outputs["midline_score_by_method_crack_png"] = out_png

    # RS3-style dataset midline score ranking (lower is better),
    # focused on combined crack metrics when available.
    if "score_mid" in all_df.columns:
        d0 = all_df.copy()
        if MIDLINE_DEBUG:
            try:
                if {"method_family", "baseline_method"}.issubset(d0.columns):
                    m_base = d0["method_family"].astype(str).str.lower().eq("baseline")
                    _log(verbose, f"[midline-debug] pre-score filter baseline rows={int(np.count_nonzero(m_base.to_numpy(dtype=bool)))}")
            except Exception:
                pass
        d0["score_mid"] = pd.to_numeric(d0["score_mid"], errors="coerce")
        finite_mask = np.isfinite(d0["score_mid"].to_numpy(float))
        if MIDLINE_DEBUG:
            try:
                if "method_family" in d0.columns:
                    m_base = d0["method_family"].astype(str).str.lower().eq("baseline")
                    n_base = int(np.count_nonzero(m_base.to_numpy(dtype=bool)))
                    n_base_finite = int(np.count_nonzero((m_base & finite_mask).to_numpy(dtype=bool)))
                    _log(verbose, f"[midline-debug] baseline finite score_mid rows={n_base_finite}/{n_base}")
            except Exception:
                pass
        d0 = d0[finite_mask]
        if not d0.empty:
            if "length_px" in d0.columns:
                ww = pd.to_numeric(d0["length_px"], errors="coerce").fillna(0.0).to_numpy(float)
                d0["_w"] = np.where(np.isfinite(ww) & (ww > 0), ww, 1.0)
            else:
                d0["_w"] = 1.0

            if {"method_family", "baseline_method", "midline_type"}.issubset(d0.columns):
                mf = d0["method_family"].astype(str)
                bm = d0["baseline_method"].astype(str)
                mt = d0["midline_type"].astype(str)
                d0["method_name"] = np.where(
                    mf.str.lower().eq("baseline"),
                    bm.where(bm.str.len() > 0, "baseline"),
                    mt.where(mt.str.len() > 0, "model"),
                )
            else:
                d0["method_name"] = "unknown"

            def _src_class(row):
                if str(row.get("method_family", "")).lower() == "baseline":
                    return "baseline"
                if _is_et_like(row.get("midline_type", "")):
                    return "ET"
                mt = str(row.get("midline_type", "")).lower()
                if mt in ("dt", "best_dt_depth", "dt_depth", "dt_ridge_color_depth", "dt_ridge_valley_depth"):
                    return "model"
                return "auto"

            d0["source_class"] = d0.apply(_src_class, axis=1)

            # Prefer combined for this ranking view; fallback to all crack types.
            if "crack_type" in d0.columns:
                # Include all combined-like labels (e.g., combined, combined_plus_noncombined_atomic).
                m_comb = d0["crack_type"].astype(str).str.lower().str.contains("combined", na=False)
                d_rank_src = d0.loc[m_comb].copy() if m_comb.any() else d0.copy()
                if MIDLINE_DEBUG:
                    try:
                        if "method_family" in d0.columns:
                            m_base = d0["method_family"].astype(str).str.lower().eq("baseline")
                            n_base_all = int(np.count_nonzero(m_base.to_numpy(dtype=bool)))
                            n_base_comb = int(np.count_nonzero((m_base & m_comb).to_numpy(dtype=bool)))
                            _log(verbose, f"[midline-debug] baseline rows in combined-like set={n_base_comb}/{n_base_all}")
                    except Exception:
                        pass
            else:
                d_rank_src = d0.copy()

            if MIDLINE_DEBUG:
                try:
                    dbg_rank_csv = os.path.join(midline_dir, "dataset_midline_debug_rank_input.csv")
                    dbg_cols = [c for c in ["image", "method_family", "baseline_method", "midline_type", "crack_type", "geometry_type", "score_mid", "length_px", "_w", "source_relpath"] if c in d_rank_src.columns]
                    d_rank_src.loc[:, dbg_cols].to_csv(dbg_rank_csv, index=False)
                    _log(verbose, f"[midline-debug] wrote rank-input rows -> {dbg_rank_csv}")
                except Exception as e:
                    _log(verbose, f"[midline-debug] rank-input dump failed: {e}")

            key_cols = ["method_name", "source_class"]
            key_cols = [c for c in key_cols if c in d_rank_src.columns]
            rank_rows = []
            for key, g in d_rank_src.groupby(key_cols, dropna=False):
                s = pd.to_numeric(g["score_mid"], errors="coerce").to_numpy(float)
                w = pd.to_numeric(g["_w"], errors="coerce").to_numpy(float)
                ok = np.isfinite(s) & np.isfinite(w) & (w > 0)
                if not np.any(ok):
                    continue
                key_vals = key if isinstance(key, tuple) else (key,)
                row = {k: v for k, v in zip(key_cols, key_vals)}
                row.update(
                    {
                        "score_mid_wmean": float(np.average(s[ok], weights=w[ok])),
                        "score_mid_mean": float(np.mean(s[ok])),
                        "n_rows": int(np.count_nonzero(ok)),
                        "weight_sum": float(np.sum(w[ok])),
                    }
                )
                rank_rows.append(row)

            if rank_rows:
                rank_df = pd.DataFrame(rank_rows)
                if "method_name" in rank_df.columns:
                    agg = {"score_mid_wmean": "min"}
                    if "source_class" in rank_df.columns:
                        agg["source_class"] = "first"
                    if "score_mid_mean" in rank_df.columns:
                        agg["score_mid_mean"] = "min"
                    if "n_rows" in rank_df.columns:
                        agg["n_rows"] = "sum"
                    if "weight_sum" in rank_df.columns:
                        agg["weight_sum"] = "sum"
                    rank_df = rank_df.groupby("method_name", as_index=False).agg(agg)
                rank_df = rank_df.sort_values("score_mid_wmean", ascending=True)
                rank_csv = os.path.join(midline_dir, "dataset_midline_score_ranked.csv")
                rank_df.to_csv(rank_csv, index=False)
                outputs["midline_score_ranked_csv"] = rank_csv
                if MIDLINE_DEBUG:
                    try:
                        if "source_class" in rank_df.columns:
                            _log(verbose, f"[midline-debug] ranked source_class counts:\n{rank_df['source_class'].astype(str).value_counts(dropna=False).to_string()}")
                    except Exception:
                        pass

                top = rank_df.head(20).copy()
                top["label"] = top.get("method_name", pd.Series(["unknown"] * len(top))).astype(str).map(_display_midline_label)
                color_map = {"ET": "#d62728", "model": "#1f77b4", "auto": "#1f77b4", "baseline": "#2ca02c"}
                colors = [color_map.get(str(c), "#4c78a8") for c in top["source_class"].astype(str).tolist()]
                fig_w = max(10.0, 0.55 * len(top))
                fig, ax = plt.subplots(figsize=(fig_w, 5.0), dpi=180)
                x = np.arange(len(top), dtype=float)
                vals = top["score_mid_wmean"].astype(float).to_numpy(float)
                ax.bar(x, vals, color=colors, alpha=0.88)
                ax.set_xticks(x)
                ax.set_xticklabels(top["label"].astype(str).tolist(), rotation=35, ha="right", fontsize=8)
                ax.set_ylabel("Mid Score")
                ax.set_title("Dataset Midline Mid Score (combined)")
                ax.grid(axis="y", alpha=0.2)
                present_cls = set(top["source_class"].astype(str).tolist())
                legend_handles = [
                    Patch(facecolor=color_map[cls], edgecolor="none", label=cls)
                    for cls in ["ET", "model", "auto", "baseline"]
                    if cls in present_cls and cls in color_map
                ]
                ax.legend(handles=legend_handles, loc="best", framealpha=0.9, fontsize=8)
                plt.tight_layout()
                out_png = os.path.join(midline_dir, "dataset_midline_score_ranked.png")
                fig.savefig(out_png, bbox_inches="tight")
                plt.close(fig)
                outputs["midline_score_ranked_png"] = out_png

                rank_no_et = rank_df[~rank_df["source_class"].astype(str).str.upper().isin(["ET", "MANUAL"])].copy()
                if not rank_no_et.empty:
                    rank_no_et = rank_no_et.sort_values("score_mid_wmean", ascending=True)
                    rank_no_et_csv = os.path.join(midline_dir, "dataset_midline_score_ranked_no_et.csv")
                    rank_no_et.to_csv(rank_no_et_csv, index=False)
                    outputs["midline_score_ranked_no_et_csv"] = rank_no_et_csv

                    top_no_et = rank_no_et.head(20).copy()
                    top_no_et["label"] = top_no_et.get("method_name", pd.Series(["unknown"] * len(top_no_et))).astype(str).map(_display_midline_label)
                    colors_no_et = [color_map.get(str(c), "#4c78a8") for c in top_no_et["source_class"].astype(str).tolist()]
                    fig_w2 = max(10.0, 0.55 * len(top_no_et))
                    fig2, ax2 = plt.subplots(figsize=(fig_w2, 5.0), dpi=180)
                    x2 = np.arange(len(top_no_et), dtype=float)
                    vals2 = top_no_et["score_mid_wmean"].astype(float).to_numpy(float)
                    ax2.bar(x2, vals2, color=colors_no_et, alpha=0.88)
                    ax2.set_xticks(x2)
                    ax2.set_xticklabels(top_no_et["label"].astype(str).tolist(), rotation=35, ha="right", fontsize=8)
                    ax2.set_ylabel("Mid Score")
                    ax2.set_title("Dataset Midline Mid Score (combined, no ET)")
                    ax2.grid(axis="y", alpha=0.2)
                    present_no_et = set(top_no_et["source_class"].astype(str).tolist())
                    legend_handles_no_et = [
                        Patch(facecolor=color_map[cls], edgecolor="none", label=cls)
                        for cls in ["ET", "auto", "baseline"]
                        if cls in present_no_et and cls in color_map
                    ]
                    ax2.legend(handles=legend_handles_no_et, loc="best", framealpha=0.9, fontsize=8)
                    plt.tight_layout()
                    out_png_no_et = os.path.join(midline_dir, "dataset_midline_score_ranked_no_et.png")
                    fig2.savefig(out_png_no_et, bbox_inches="tight")
                    plt.close(fig2)
                    outputs["dataset_midline_score_ranked_no_et_png"] = out_png_no_et

    _log(verbose, f"[summarize] midline rows={len(all_df)}")
    return outputs


def _aggregate_timing_metrics(
    image_dirs: List[str],
    out_dir: str,
    *,
    verbose: bool,
) -> Dict[str, str]:
    outputs = {}

    def _drop_skeleton_graph_cols(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if not isinstance(df, pd.DataFrame):
            return df
        skel_cols = [c for c in df.columns if "skeleton_graph" in str(c).lower()]
        if skel_cols:
            return df.drop(columns=skel_cols)
        return df

    # timings_core.csv (per-crack timing rows for edge-tracking stage only)
    timing_frames = []
    for img_dir in image_dirs:
        image = os.path.basename(img_dir)
        p = os.path.join(img_dir, "timings_core.csv")
        df = _safe_read_csv(p)
        if df is None:
            continue
        df = _drop_skeleton_graph_cols(df)
        d = df.copy()
        d["image"] = image
        timing_frames.append(d)
    # Also read timings_driver.csv for actual component seconds.
    driver_frames = []
    for img_dir in image_dirs:
        p = os.path.join(img_dir, "timings_driver.csv")
        df = _safe_read_csv(p)
        if df is None or df.empty:
            continue
        d = df.copy()
        d["image"] = os.path.basename(img_dir)
        driver_frames.append(d)

    if timing_frames:
        # Hard-coded naming switch:
        # True  -> also write legacy "timings_core" filenames/keys for compatibility
        # False -> only write the clearer edge-tracking-stage names
        WRITE_LEGACY_TIMINGS_CORE_ALIASES = True
        stage_tag = "edge_tracking_stage"

        all_df = pd.concat(timing_frames, ignore_index=True)
        all_csv = os.path.join(out_dir, f"dataset_{stage_tag}_all.csv")
        all_df.to_csv(all_csv, index=False)
        outputs[f"{stage_tag}_all_csv"] = all_csv
        if WRITE_LEGACY_TIMINGS_CORE_ALIASES:
            legacy_all_csv = os.path.join(out_dir, "dataset_timings_core_all.csv")
            all_df.to_csv(legacy_all_csv, index=False)
            outputs["timings_core_all_csv"] = legacy_all_csv

        group_cols = [c for c in ["supervision", "crack_type", "algo_variant"] if c in all_df.columns]
        num_cols = [
            c for c in all_df.columns
            if (
                pd.api.types.is_numeric_dtype(all_df[c])
                and (c.endswith("_sec") or c.endswith("_s"))
            )
        ]
        grouped = _aggregate_numeric(all_df, group_cols=group_cols, numeric_cols=num_cols)
        if not grouped.empty:
            grp_csv = os.path.join(out_dir, f"dataset_{stage_tag}_grouped.csv")
            grouped.to_csv(grp_csv, index=False)
            outputs[f"{stage_tag}_grouped_csv"] = grp_csv
            if WRITE_LEGACY_TIMINGS_CORE_ALIASES:
                legacy_grp_csv = os.path.join(out_dir, "dataset_timings_core_grouped.csv")
                grouped.to_csv(legacy_grp_csv, index=False)
                outputs["timings_core_grouped_csv"] = legacy_grp_csv

        # Aggregate mean timing components (prefer timings_driver with real per-stage seconds).
        _driver_df = pd.concat(driver_frames, ignore_index=True) if driver_frames else None
        all_df_for_components = _driver_df if (_driver_df is not None and not _driver_df.empty) else all_df
        _driver_num_cols = [
            c for c in all_df_for_components.columns
            if pd.api.types.is_numeric_dtype(all_df_for_components[c])
            and (c.endswith("_sec") or c.endswith("_s"))
            and c not in ("crack_id",)
        ]
        key_cols = [c for c in ["edge_masks_sec", "edges_tracking_sec", "build_combined_sec"] if c in _driver_num_cols]
        if not key_cols:
            key_cols = [c for c in _driver_num_cols if any(k in c for k in ("edge", "mask", "build"))][:6]
        if not key_cols:
            key_cols = _driver_num_cols[:6]
        if key_cols:
            vals = [float(pd.to_numeric(all_df_for_components[c], errors="coerce").mean()) for c in key_cols]
            out_png = os.path.join(out_dir, f"dataset_{stage_tag}_components.png")
            _save_bar(
                key_cols,
                vals,
                out_png=out_png,
                title="Edge-Tracking Stage Timing Components (mean)",
                ylabel="seconds",
            )
            outputs[f"{stage_tag}_components_png"] = out_png
            # Sum version (total seconds across dataset).
            vals_sum = [float(pd.to_numeric(all_df_for_components[c], errors="coerce").sum()) for c in key_cols]
            out_png_sum = os.path.join(out_dir, f"sum_dataset_{stage_tag}_components.png")
            _save_bar(
                key_cols,
                vals_sum,
                out_png=out_png_sum,
                title="Edge-Tracking Stage Timing Components (sum)",
                ylabel="total seconds",
            )
            outputs[f"sum_{stage_tag}_components_png"] = out_png_sum
            if WRITE_LEGACY_TIMINGS_CORE_ALIASES:
                legacy_out_png = os.path.join(out_dir, "dataset_timings_core_components.png")
                _save_bar(
                    key_cols,
                    vals,
                    out_png=legacy_out_png,
                    title="Edge-Tracking Stage Timing Components (mean)",
                    ylabel="seconds",
                )
                outputs["timings_core_components_png"] = legacy_out_png
                legacy_out_png_sum = os.path.join(out_dir, "sum_dataset_timings_core_components.png")
                _save_bar(
                    key_cols,
                    vals_sum,
                    out_png=legacy_out_png_sum,
                    title="Edge-Tracking Stage Timing Components (sum)",
                    ylabel="total seconds",
                )
                outputs["sum_timings_core_components_png"] = legacy_out_png_sum

        _log(verbose, f"[summarize] {stage_tag} rows={len(all_df)}")

        # Stage rollup (mean / weighted mean / total) per timing component.
        roll_rows = []
        weight_col = next((c for c in ["length_px", "finite_len_px", "crack_px"] if c in all_df.columns), None)
        if weight_col is not None:
            w_all = pd.to_numeric(all_df[weight_col], errors="coerce").fillna(0.0).to_numpy(float)
        else:
            w_all = np.ones(len(all_df), dtype=float)

        for col in num_cols:
            vals = pd.to_numeric(all_df[col], errors="coerce").to_numpy(float)
            ok = np.isfinite(vals)
            if not np.any(ok):
                continue
            vv = vals[ok]
            ww = w_all[ok]
            mean_t = float(np.mean(vv))
            total_t = float(np.sum(vv))
            if np.any(np.isfinite(ww) & (ww > 0)):
                w_ok = np.isfinite(ww) & (ww > 0)
                weighted_mean_t = float(np.sum(vv[w_ok] * ww[w_ok]) / np.sum(ww[w_ok]))
            else:
                weighted_mean_t = mean_t
            roll_rows.append(
                {
                    "category": "edge_tracking_stage",
                    "component": str(col),
                    "mean_sec": mean_t,
                    "weighted_mean_sec": weighted_mean_t,
                    "total_sec": total_t,
                    "count": int(np.count_nonzero(ok)),
                    "weight_col": str(weight_col) if weight_col else "",
                }
            )
        if roll_rows:
            roll_csv = os.path.join(out_dir, "dataset_edge_tracking_stage_rollup.csv")
            pd.DataFrame(roll_rows).to_csv(roll_csv, index=False)
            outputs["edge_tracking_stage_rollup_csv"] = roll_csv

    # runtime_log.csv (per-image rollup from quick runs)
    runtime_frames = []
    for img_dir in image_dirs:
        image = os.path.basename(img_dir)
        p = os.path.join(img_dir, "runtime_log.csv")
        df = _safe_read_csv(p)
        if df is None:
            continue
        df = _drop_skeleton_graph_cols(df)
        d = df.copy()
        d["image"] = image
        runtime_frames.append(d)
    if runtime_frames:
        rdf = pd.concat(runtime_frames, ignore_index=True)
        out_csv = os.path.join(out_dir, "dataset_runtime_log_all.csv")
        rdf.to_csv(out_csv, index=False)
        outputs["runtime_log_all_csv"] = out_csv
        key = "total_s" if "total_s" in rdf.columns else None
        if key:
            out_png = os.path.join(out_dir, "dataset_runtime_total_by_image.png")
            # keep latest row per image for cleaner bar chart
            dd = rdf.groupby("image", as_index=False)[key].mean().sort_values(key, ascending=True)
            _save_bar(
                dd["image"].astype(str).tolist(),
                dd[key].astype(float).tolist(),
                out_png=out_png,
                title="Dataset runtime total (mean per image)",
                ylabel="seconds",
                rotate=45,
            )
            outputs["runtime_total_png"] = out_png

    # batch_timing.csv (per-image batch driver timing rows)
    batch_frames = []
    for img_dir in image_dirs:
        image = os.path.basename(img_dir)
        p = os.path.join(img_dir, "batch_timing.csv")
        df = _safe_read_csv(p)
        if df is None:
            continue
        df = _drop_skeleton_graph_cols(df)
        d = df.copy()
        if "image" not in d.columns:
            d["image"] = image
        batch_frames.append(d)
    if batch_frames:
        bdf = pd.concat(batch_frames, ignore_index=True)
        out_csv = os.path.join(out_dir, "dataset_batch_timing_all.csv")
        bdf.to_csv(out_csv, index=False)
        outputs["batch_timing_all_csv"] = out_csv

        # dataset_batch_runtime_by_image plot disabled (noisy for single-image runs)
        # if "total_image_s" in bdf.columns: ...

    return outputs


def _aggregate_width_distribution(
    image_dirs: List[str],
    out_dir: str,
    *,
    verbose: bool,
) -> Dict[str, str]:
    outputs = {}
    width_dir = os.path.join(out_dir, "width")
    dist_dir = os.path.join(width_dir, "distribution")
    os.makedirs(dist_dir, exist_ok=True)
    frames = []
    for img_dir in image_dirs:
        image = os.path.basename(img_dir)
        p = os.path.join(img_dir, "width_distribution_summary.csv")
        df = _safe_read_csv(p)
        if df is None:
            continue
        d = df.copy()
        if "image" not in d.columns:
            d["image"] = image
        frames.append(d)

    if not frames:
        return outputs

    all_df = pd.concat(frames, ignore_index=True)
    # Normalize method-like labels for downstream sorting/plots.
    # Purge "best_" prefix (e.g. best_dt_depth -> dt_depth).
    for _c in ("method", "midline_type", "variant_id", "algo_variant"):
        if _c in all_df.columns:
            all_df[_c] = all_df[_c].astype(str).apply(
                lambda s: s[len("best_") :] if s.startswith("best_") else s
            )
    csv_path = os.path.join(dist_dir, "dataset_width_distribution_all.csv")
    all_df.to_csv(csv_path, index=False)
    outputs["width_dist_all_csv"] = csv_path

    try:
        from helpers.present_plots import plot_width_distribution_report

        rep_dir = os.path.join(dist_dir, "width_distribution_report_dataset")
        result = plot_width_distribution_report(
            csv_path=csv_path,
            out_dir=rep_dir,
            title_suffix="(dataset aggregate)",
        )
        if isinstance(result, dict):
            outputs["width_dist_agg_csv"] = result.get("agg_csv", "")
    except Exception as e:
        _log(verbose, f"[summarize] width distribution report failed: {e}")

    _log(verbose, f"[summarize] width distribution rows={len(all_df)}")
    return outputs


def _aggregate_edge_rs3_selection(
    image_dirs: List[str],
    out_dir: str,
    *,
    verbose: bool,
) -> Dict[str, str]:
    outputs = {}

    # Edge family aggregation across images
    edge_frames = []
    edge_all_frames = []
    for img_dir in image_dirs:
        image = os.path.basename(img_dir)
        p = os.path.join(img_dir, "edge_sweep_family_agg.csv")
        df = _safe_read_csv(p)
        if df is None:
            continue
        d = df.copy()
        d["image"] = image
        edge_frames.append(d)
        p_all = os.path.join(img_dir, "edge_sweep_all.csv")
        df_all = _safe_read_csv(p_all)
        if df_all is not None:
            da = df_all.copy()
            da["image"] = image
            edge_all_frames.append(da)

    if edge_frames:
        edge_df = pd.concat(edge_frames, ignore_index=True)
        all_csv = os.path.join(out_dir, "dataset_edge_family_agg_all.csv")
        edge_df.to_csv(all_csv, index=False)
        outputs["edge_family_all_csv"] = all_csv

        fam_cols = [
            "param_window_half_size",
            "param_mu",
            "param_l",
            "param_p",
            "param_seg_mode",
        ]
        if all(c in edge_df.columns for c in fam_cols) and "edge_score_wmean" in edge_df.columns:
            rank = (
                edge_df.groupby(fam_cols, dropna=False)
                .agg(
                    n_images=("image", "nunique"),
                    n_rows=("image", "size"),
                    edge_score_wmean_mean=("edge_score_wmean", "mean"),
                    edge_score_wmean_median=("edge_score_wmean", "median"),
                )
                .reset_index()
                .sort_values("edge_score_wmean_mean", ascending=True)
            )
            rank_csv = os.path.join(out_dir, "dataset_edge_family_ranked.csv")
            rank.to_csv(rank_csv, index=False)
            outputs["edge_family_ranked_csv"] = rank_csv

            top = rank.head(15).copy()
            top["label"] = top.apply(
                lambda r: (
                    f"mu,l,p={r['param_mu']},{int(r['param_l'])},{int(r['param_p'])}"
                ),
                axis=1,
            )
            out_png = os.path.join(out_dir, "dataset_edge_family_scores.png")

            # Optional decomposition from edge_sweep_all.csv:
            # edge_score ~= (1-boundary_f1) + 0.50*(ASSD/med) + 0.25*(HD95/med)
            # medians are computed per image to mirror image-level sweep summaries.
            decomp_df = pd.DataFrame()
            if edge_all_frames:
                try:
                    raw_all = pd.concat(edge_all_frames, ignore_index=True)
                    need_cols = fam_cols + ["image", "boundary_f1", "ASSD", "HD95", "global_weight"]
                    if all(c in raw_all.columns for c in need_cols):
                        decomp_rows = []
                        for image, gi in raw_all.groupby("image", dropna=False):
                            g = gi.copy()
                            g["boundary_f1"] = pd.to_numeric(g["boundary_f1"], errors="coerce")
                            g["ASSD"] = pd.to_numeric(g["ASSD"], errors="coerce")
                            g["HD95"] = pd.to_numeric(g["HD95"], errors="coerce")
                            g["global_weight"] = pd.to_numeric(g["global_weight"], errors="coerce").fillna(1.0)
                            assd_med = float(np.nanmedian(g["ASSD"].to_numpy(float))) + 1e-9
                            hd95_med = float(np.nanmedian(g["HD95"].to_numpy(float))) + 1e-9
                            g["comp_boundary"] = 1.0 - g["boundary_f1"]
                            g["comp_assd"] = 0.50 * (g["ASSD"] / assd_med)
                            g["comp_hd95"] = 0.25 * (g["HD95"] / hd95_med)
                            for key, gf in g.groupby(fam_cols, dropna=False):
                                w = pd.to_numeric(gf["global_weight"], errors="coerce").to_numpy(float)
                                w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
                                if not np.any(w > 0):
                                    continue
                                b = pd.to_numeric(gf["comp_boundary"], errors="coerce").to_numpy(float)
                                a = pd.to_numeric(gf["comp_assd"], errors="coerce").to_numpy(float)
                                h = pd.to_numeric(gf["comp_hd95"], errors="coerce").to_numpy(float)
                                ok_b = np.isfinite(b) & np.isfinite(w) & (w > 0)
                                ok_a = np.isfinite(a) & np.isfinite(w) & (w > 0)
                                ok_h = np.isfinite(h) & np.isfinite(w) & (w > 0)
                                if not (np.any(ok_b) and np.any(ok_a) and np.any(ok_h)):
                                    continue
                                decomp_rows.append(
                                    {
                                        "image": image,
                                        **dict(zip(fam_cols, key)),
                                        "comp_boundary_wmean": float(np.average(b[ok_b], weights=w[ok_b])),
                                        "comp_assd_wmean": float(np.average(a[ok_a], weights=w[ok_a])),
                                        "comp_hd95_wmean": float(np.average(h[ok_h], weights=w[ok_h])),
                                    }
                                )
                        if decomp_rows:
                            decomp_raw = pd.DataFrame(decomp_rows)
                            decomp_df = (
                                decomp_raw.groupby(fam_cols, dropna=False)
                                .agg(
                                    comp_boundary_wmean_mean=("comp_boundary_wmean", "mean"),
                                    comp_assd_wmean_mean=("comp_assd_wmean", "mean"),
                                    comp_hd95_wmean_mean=("comp_hd95_wmean", "mean"),
                                )
                                .reset_index()
                            )
                except Exception as e:
                    _log(verbose, f"[summarize] edge score decomposition failed: {e}")

            top_plot = top.merge(decomp_df, on=fam_cols, how="left") if not decomp_df.empty else top.copy()
            fig_w = max(10.0, 0.60 * len(top_plot))
            fig, ax = plt.subplots(figsize=(fig_w, 5.2), dpi=180)
            x = np.arange(len(top_plot), dtype=float)
            total_vals = pd.to_numeric(top_plot["edge_score_wmean_mean"], errors="coerce").to_numpy(float)
            modes = top_plot["param_seg_mode"].astype(str).str.lower().tolist()
            mode_edge = ["#39b54a" if m == "old" else "#1f77b4" for m in modes]  # old=brighter green, new=blue

            has_decomp = all(
                c in top_plot.columns
                for c in ["comp_boundary_wmean_mean", "comp_assd_wmean_mean", "comp_hd95_wmean_mean"]
            )
            if has_decomp:
                cb = pd.to_numeric(top_plot["comp_boundary_wmean_mean"], errors="coerce").to_numpy(float)
                ca = pd.to_numeric(top_plot["comp_assd_wmean_mean"], errors="coerce").to_numpy(float)
                ch = pd.to_numeric(top_plot["comp_hd95_wmean_mean"], errors="coerce").to_numpy(float)
                cb = np.where(np.isfinite(cb), cb, 0.0)
                ca = np.where(np.isfinite(ca), ca, 0.0)
                ch = np.where(np.isfinite(ch), ch, 0.0)
                bars1 = ax.bar(x, cb, color="#e15759", alpha=0.85, label="boundary term")
                bars2 = ax.bar(x, ca, bottom=cb, color="#b07aa1", alpha=0.85, label="ASSD term")
                bars3 = ax.bar(x, ch, bottom=(cb + ca), color="#f28e2b", alpha=0.85, label="HD95 term")
                # Mode overlay: colored outlines on stacked bars.
                for i in range(len(x)):
                    for bar in (bars1[i], bars2[i], bars3[i]):
                        bar.set_edgecolor(mode_edge[i])
                        bar.set_linewidth(2.6)
                ax.plot(x, total_vals, "o", color="black", markersize=4, label="edge_score_wmean_mean")
            else:
                ax.bar(x, total_vals, color="#4c78a8", alpha=0.85)
                for i, p in enumerate(ax.patches):
                    p.set_edgecolor(mode_edge[i])
                    p.set_linewidth(2.6)
                ax.plot(x, total_vals, "o", color="black", markersize=4, label="edge_score_wmean_mean")

            ax.set_xticks(x)
            ax.set_xticklabels(top_plot["label"].tolist(), rotation=65, ha="right", fontsize=8)
            ax.set_ylabel("edge score / components")
            ax.set_title("Dataset edge family score decomposition (lower is better)")
            ax.grid(axis="y", alpha=0.2)
            legend_handles = [
                Patch(facecolor="#1f77b4", edgecolor="none", label="mode=new (outline)"),
                Patch(facecolor="#39b54a", edgecolor="none", label="mode=old (outline)"),
            ]
            if has_decomp:
                legend_handles = [
                    Patch(facecolor="#e15759", edgecolor="none", label="boundary term"),
                    Patch(facecolor="#b07aa1", edgecolor="none", label="ASSD term"),
                    Patch(facecolor="#f28e2b", edgecolor="none", label="HD95 term"),
                    Patch(facecolor="#1f77b4", edgecolor="none", label="mode=new (outline)"),
                    Patch(facecolor="#39b54a", edgecolor="none", label="mode=old (outline)"),
                ]
            ax.legend(handles=legend_handles, loc="lower right", framealpha=0.9, fontsize=8)
            plt.tight_layout()
            fig.savefig(out_png, bbox_inches="tight")
            plt.close(fig)
            outputs["edge_family_scores_png"] = out_png

    # RS3 family aggregation across images
    rs3_frames = []
    for img_dir in image_dirs:
        image = os.path.basename(img_dir)
        p = os.path.join(img_dir, "auto", "rs3_family_agg.csv")
        df = _safe_read_csv(p)
        if df is None:
            continue
        d = df.copy()
        d["image"] = image
        rs3_frames.append(d)

    if rs3_frames:
        rs3_df = pd.concat(rs3_frames, ignore_index=True)
        all_csv = os.path.join(out_dir, "dataset_rs3_family_agg_all.csv")
        rs3_df.to_csv(all_csv, index=False)
        outputs["rs3_family_all_csv"] = all_csv

        fam_cols = ["os_mode", "g11", "g22", "g33"]
        if all(c in rs3_df.columns for c in fam_cols) and "score_mid_wmean" in rs3_df.columns:
            rank = (
                rs3_df.groupby(fam_cols, dropna=False)
                .agg(
                    n_images=("image", "nunique"),
                    n_rows=("image", "size"),
                    score_mid_wmean_mean=("score_mid_wmean", "mean"),
                    score_mid_wmean_median=("score_mid_wmean", "median"),
                )
                .reset_index()
                .sort_values("score_mid_wmean_mean", ascending=True)
            )
            rank_csv = os.path.join(out_dir, "dataset_rs3_family_ranked.csv")
            rank.to_csv(rank_csv, index=False)
            outputs["rs3_family_ranked_csv"] = rank_csv

            top = rank.head(15).copy()
            top["label"] = top.apply(
                lambda r: f"{r['os_mode']}:({r['g11']},{r['g22']},{r['g33']})",
                axis=1,
            )
            out_png = os.path.join(out_dir, "dataset_rs3_family_scores.png")
            _save_bar(
                top["label"].tolist(),
                top["score_mid_wmean_mean"].astype(float).tolist(),
                out_png=out_png,
                title="Dataset RS3 family score (lower is better)",
                ylabel="score_mid_wmean",
                rotate=55,
            )
            outputs["rs3_family_scores_png"] = out_png

    return outputs


def _aggregate_baseline_timings(
    out_dir: str,
    *,
    baseline_roots: Optional[List[str]],
    evaluated_images: Optional[set] = None,
    verbose: bool,
) -> Dict[str, str]:
    outputs = {}
    if not baseline_roots:
        return outputs

    roots = [r for r in baseline_roots if r and os.path.isdir(r)]
    if not roots:
        return outputs

    timing_files = []
    summary_files = []
    seg_infer_files = []
    for root in roots:
        p1 = os.path.join(root, "width_baseline_timings.csv")
        p2 = os.path.join(root, "width_baseline_timings_summary.csv")
        if os.path.isfile(p1):
            timing_files.append(p1)
        if os.path.isfile(p2):
            summary_files.append(p2)
        # Per-method segmentation inference timing CSVs.
        seg_infer_files.extend(glob.glob(os.path.join(root, "**", "timing_per_image.csv"), recursive=True))

    # Also search metrics/_baseline_timings/ for B1 projection timing
    metrics_root = os.path.join(os.path.dirname(out_dir), "metrics") if out_dir else ""
    if metrics_root and os.path.isdir(metrics_root):
        _b1_timing = os.path.join(metrics_root, "_baseline_timings", "timing_per_image.csv")
        if os.path.isfile(_b1_timing) and _b1_timing not in timing_files:
            timing_files.append(_b1_timing)

    # Also pick other timing-like files from explicit roots only.
    for root in roots:
        for p in glob.glob(os.path.join(root, "*baseline*tim*.csv")):
            if p not in timing_files and p not in summary_files:
                timing_files.append(p)

    frames = []
    for p in timing_files:
        df = _safe_read_csv(p)
        if df is None:
            continue
        d = df.copy()
        d["source_file"] = p
        frames.append(d)
    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        print(f"[BASELINE TIMING] all_df columns: {all_df.columns.tolist()}")
        print(f"[BASELINE TIMING] num rows: {len(all_df)}")
        if "b1_projection_s" in all_df.columns:
            print(f"[BASELINE TIMING] b1_projection_s values: {all_df['b1_projection_s'].tolist()}")
        if "image" not in all_df.columns and "stem" in all_df.columns:
            all_df["image"] = all_df["stem"].astype(str)
        if evaluated_images:
            _eval_set_norm = {os.path.splitext(os.path.basename(str(x)))[0].replace("_modified", "") for x in evaluated_images}
            for _id_col in ("image", "stem", "path"):
                if _id_col in all_df.columns:
                    _norm = all_df[_id_col].astype(str).apply(
                        lambda s: os.path.splitext(os.path.basename(s))[0].replace("_modified", "")
                    )
                    all_df = all_df[_norm.isin(_eval_set_norm)].copy()
                    break

        out_csv = os.path.join(out_dir, "dataset_baseline_timings_all.csv")
        all_df.to_csv(out_csv, index=False)
        outputs["baseline_timings_all_csv"] = out_csv

        _ordered_cols = [
            "shared_mat_gpu_s",
            "skeleton_dse_s",
            "pca_width_dse_s",
            "esd_width_dse_s",
            "eob_width_dse_s",
        ]
        num_cols = [
            c for c in _ordered_cols
            if c in all_df.columns and pd.api.types.is_numeric_dtype(all_df[c])
        ]
        # mat_width_raw and mat_width_dse have no separate timing (DT*2, negligible)
        # but they share the shared_mat_gpu_s precompute cost — add it to each
        _col_display = {
            "shared_mat_gpu_s": "MAT",
            "skeleton_dse_s": "DSE pruning",
            "pca_width_dse_s": "PCA",
            "esd_width_dse_s": "ESD",
            "eob_width_dse_s": "EOB",
            "mat_width_dse_s": "MAT (DSE)",
            "mat_width_raw_s": "MAT (raw)",
        }
        if num_cols:
            disp_cols = [_col_display.get(c, _display_timing_component_name(c)) for c in num_cols]
            vals = [float(pd.to_numeric(all_df[c], errors="coerce").mean()) for c in num_cols]
            out_png = os.path.join(out_dir, "dataset_baseline_timings_components.png")
            _save_bar(
                disp_cols,
                vals,
                out_png=out_png,
                title="Width Baseline Timing Components (mean)",
                ylabel="seconds",
                rotate=40,
            )
            outputs["baseline_timings_components_png"] = out_png
            vals_sum = [float(pd.to_numeric(all_df[c], errors="coerce").sum()) for c in num_cols]
            out_png_sum = os.path.join(out_dir, "sum_dataset_baseline_timings_components.png")
            _save_bar(
                disp_cols,
                vals_sum,
                out_png=out_png_sum,
                title="Width Baseline Timing Components (sum)",
                ylabel="total seconds",
                rotate=40,
            )
            outputs["sum_baseline_timings_components_png"] = out_png_sum
            out_csv_sum = os.path.join(out_dir, "sum_dataset_baseline_timings_components.csv")
            pd.DataFrame(
                {
                    "component": [str(c) for c in disp_cols],
                    "total_sec": [float(v) for v in vals_sum],
                }
            ).to_csv(out_csv_sum, index=False)
            outputs["sum_baseline_timings_components_csv"] = out_csv_sum

            weight_col = "crack_px" if "crack_px" in all_df.columns else None
            w_all = (
                pd.to_numeric(all_df[weight_col], errors="coerce").fillna(0.0).to_numpy(float)
                if weight_col
                else np.ones(len(all_df), dtype=float)
            )
            roll_rows = []
            for col in num_cols:
                x = pd.to_numeric(all_df[col], errors="coerce").to_numpy(float)
                ok = np.isfinite(x)
                if not np.any(ok):
                    continue
                xx = x[ok]
                ww = w_all[ok]
                mean_t = float(np.mean(xx))
                total_t = float(np.sum(xx))
                if np.any(np.isfinite(ww) & (ww > 0)):
                    w_ok = np.isfinite(ww) & (ww > 0)
                    wmean_t = float(np.sum(xx[w_ok] * ww[w_ok]) / np.sum(ww[w_ok]))
                else:
                    wmean_t = mean_t
                roll_rows.append(
                    {
                        "category": "width_method",
                        "component": str(col),
                        "mean_sec": mean_t,
                        "weighted_mean_sec": wmean_t,
                        "total_sec": total_t,
                        "count": int(np.count_nonzero(ok)),
                        "weight_col": str(weight_col) if weight_col else "",
                    }
                )
            if roll_rows:
                roll_csv = os.path.join(out_dir, "dataset_baseline_timings_rollup.csv")
                pd.DataFrame(roll_rows).to_csv(roll_csv, index=False)
                outputs["baseline_timings_rollup_csv"] = roll_csv

    sframes = []
    for p in summary_files:
        df = _safe_read_csv(p)
        if df is None:
            continue
        d = df.copy()
        d["source_file"] = p
        sframes.append(d)
    if sframes:
        sdf = pd.concat(sframes, ignore_index=True)
        out_csv = os.path.join(out_dir, "dataset_baseline_timings_summary_all.csv")
        sdf.to_csv(out_csv, index=False)
        outputs["baseline_timings_summary_all_csv"] = out_csv

    # Additional baseline segmentation inference timings:
    # expected columns include image_name, status, inference_seconds
    seg_rows = []
    eval_set = {str(x) for x in (evaluated_images or set())}
    for p in seg_infer_files:
        df = _safe_read_csv(p)
        if df is None:
            continue
        method = os.path.basename(os.path.dirname(p))
        d = df.copy()
        if "image_name" in d.columns:
            image_col = d["image_name"].astype(str)
        elif "image" in d.columns:
            image_col = d["image"].astype(str)
        else:
            image_col = pd.Series(["unknown"] * len(d))
        d["image"] = image_col.apply(lambda s: os.path.splitext(os.path.basename(str(s)))[0])
        if eval_set:
            d = d[d["image"].isin(eval_set)].copy()
        if d.empty:
            continue
        sec_col = None
        for c in ("inference_seconds", "inference_sec", "seconds", "time_sec"):
            if c in d.columns:
                sec_col = c
                break
        if sec_col is None:
            continue
        sec = pd.to_numeric(d[sec_col], errors="coerce").to_numpy(float)
        sec = sec[np.isfinite(sec)]
        if sec.size == 0:
            continue
        seg_rows.append(
            {
                "category": "seg",
                "component": f"inference_seconds:{method}",
                "mean_sec": float(np.mean(sec)),
                "weighted_mean_sec": float(np.mean(sec)),
                "total_sec": float(np.sum(sec)),
                "count": int(sec.size),
                "weight_col": "",
                "source_file": p,
            }
        )
    if seg_rows:
        seg_df = pd.DataFrame(seg_rows).sort_values("component")
        seg_all_csv = os.path.join(out_dir, "dataset_baseline_segmentation_inference_rollup.csv")
        seg_df.to_csv(seg_all_csv, index=False)
        outputs["baseline_segmentation_inference_rollup_csv"] = seg_all_csv

        # Merge into main baseline rollup for downstream full timing overview.
        main_roll_csv = os.path.join(out_dir, "dataset_baseline_timings_rollup.csv")
        main_df = _safe_read_csv(main_roll_csv)
        if main_df is not None and not main_df.empty:
            merged = pd.concat([main_df, seg_df.drop(columns=["source_file"], errors="ignore")], ignore_index=True)
            merged.to_csv(main_roll_csv, index=False)
        else:
            seg_df.drop(columns=["source_file"], errors="ignore").to_csv(main_roll_csv, index=False)
        outputs["baseline_timings_rollup_csv"] = main_roll_csv

    _log(verbose, f"[summarize] baseline timing files={len(timing_files)} summaries={len(summary_files)}")
    return outputs


def _aggregate_supervision_timings(
    supervision_root: str,
    out_dir: str,
    *,
    evaluated_images: Optional[set] = None,
    verbose: bool = True,
) -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    if not supervision_root or not os.path.isdir(supervision_root):
        return outputs

    files = []
    files.extend(glob.glob(os.path.join(supervision_root, "**", "gt_compute_timing.csv"), recursive=True))
    files.extend(glob.glob(os.path.join(supervision_root, "**", "gt_centering_timing.csv"), recursive=True))
    if not files:
        return outputs

    _GT_SUP_EXCLUDE_PATTERNS = (
        "multi_cue_", "multi_cue_", "dijkstra", "costmap", "multi_cue_generation",
        "normals_depth", "multi_cue_align", "depth_recess",
    )
    rows = []
    for p in files:
        df = _safe_read_csv(p)
        if df is None:
            continue

        parts = os.path.normpath(p).split(os.sep)
        image = "unknown"
        if "supervision" in parts:
            i = parts.index("supervision")
            if i + 1 < len(parts):
                image = str(parts[i + 1])
        elif len(parts) >= 3:
            image = str(parts[-3])

        if evaluated_images and image not in {str(x) for x in evaluated_images}:
            continue

        stage_from_file = "normals" if "gt_compute_timing.csv" in os.path.basename(p) else "centering"
        d = df.copy()
        d["image"] = image
        d["source_file"] = p

        num_cols = [
            c for c in d.columns
            if (
                pd.api.types.is_numeric_dtype(d[c])
                and (c.endswith("_sec") or c.endswith("_s"))
                and not any(pat in c.lower() for pat in _GT_SUP_EXCLUDE_PATTERNS)
            )
        ]
        for _, rr in d.iterrows():
            for col in num_cols:
                sec = pd.to_numeric(rr.get(col, np.nan), errors="coerce")
                if not np.isfinite(sec):
                    continue
                c = str(col).lower()
                if "combined_plus_noncombined" in c:
                    mode = "combined_plus_noncombined_atomic"
                elif c.startswith("noncombined_atomic"):
                    mode = "noncombined_atomic"
                elif c.startswith("atomic"):
                    mode = "atomic"
                elif c.startswith("combined"):
                    mode = "combined"
                elif "total" in c:
                    mode = "total"
                else:
                    mode = "unknown"
                stage = "centering" if ("center" in c or stage_from_file == "centering") else "normals"
                rows.append(
                    {
                        "image": image,
                        "category": "gt_supervision",
                        "stage": stage,
                        "mode": mode,
                        "component": str(col),
                        "sec": float(sec),
                        "source_file": p,
                    }
                )

    if not rows:
        return outputs

    all_df = pd.DataFrame(rows)
    all_csv = os.path.join(out_dir, "dataset_gt_supervision_timings_all.csv")
    all_df.to_csv(all_csv, index=False)
    outputs["gt_supervision_timings_all_csv"] = all_csv

    # Optional per-image weights from edge-stage timing output.
    # We prefer crack/length-like columns when available.
    img_weight = {}
    edge_csv = os.path.join(out_dir, "dataset_edge_tracking_stage_all.csv")
    edge_df = _safe_read_csv(edge_csv)
    if edge_df is not None and not edge_df.empty and "image" in edge_df.columns:
        wcol = next((c for c in ["crack_px", "length_px", "finite_len_px"] if c in edge_df.columns), None)
        if wcol is not None:
            ew = edge_df[["image", wcol]].copy()
            ew["image"] = ew["image"].astype(str)
            ew[wcol] = pd.to_numeric(ew[wcol], errors="coerce")
            ew = ew[np.isfinite(ew[wcol]) & (ew[wcol] > 0)]
            if not ew.empty:
                img_weight = ew.groupby("image", dropna=False)[wcol].sum().to_dict()

    roll_rows = []
    for (stage, mode, component), g in all_df.groupby(["stage", "mode", "component"], dropna=False):
        vals_all = pd.to_numeric(g["sec"], errors="coerce").to_numpy(float)
        imgs_all = g["image"].astype(str).to_numpy()
        ok = np.isfinite(vals_all)
        if not np.any(ok):
            continue
        vals = vals_all[ok]
        imgs = imgs_all[ok]
        mean_t = float(np.mean(vals))
        total_t = float(np.sum(vals))

        # Length-weighted mean if weights are available; otherwise fallback to mean.
        if img_weight:
            ww = np.asarray([float(img_weight.get(im, np.nan)) for im in imgs], dtype=float)
            okw = np.isfinite(ww) & (ww > 0)
            if np.any(okw):
                weighted_mean_t = float(np.sum(vals[okw] * ww[okw]) / np.sum(ww[okw]))
            else:
                weighted_mean_t = mean_t
        else:
            weighted_mean_t = mean_t

        roll_rows.append(
            {
                "category": "gt_supervision",
                "stage": str(stage),
                "mode": str(mode),
                "component": str(component),
                "mean_sec": mean_t,
                "weighted_mean_sec": weighted_mean_t,
                "total_sec": total_t,
                "count": int(vals.size),
            }
        )
    if roll_rows:
        roll_df = pd.DataFrame(roll_rows).sort_values(["stage", "mode", "component"])
        out_csv = os.path.join(out_dir, "dataset_gt_supervision_timings.csv")
        roll_df.to_csv(out_csv, index=False)
        outputs["gt_supervision_timings_csv"] = out_csv

        # Core rows used in algorithm overview: atomic + final combined rows.
        core_specs = [
            ("gt_compute_atomic", "normals", "atomic_compute_sec"),
            ("gt_compute_combined", "normals", "combined_plus_noncombined_atomics_sec"),
            ("gt_centering_atomic", "centering", "atomic_centering_sec"),
            ("gt_centering_combined", "centering", "combined_plus_noncombined_atomics_centering_sec"),
        ]
        core_rows = []
        for method, stage, comp in core_specs:
            sub = roll_df[
                (roll_df["stage"].astype(str) == stage)
                & (roll_df["component"].astype(str) == comp)
            ]
            if sub.empty:
                continue
            r0 = sub.iloc[0]
            core_rows.append(
                {
                    "method": method,
                    "stage": stage,
                    "component": comp,
                    "weighted_mean_sec": float(pd.to_numeric(r0.get("weighted_mean_sec"), errors="coerce")),
                    "total_sec": float(pd.to_numeric(r0.get("total_sec"), errors="coerce")),
                    "count": int(pd.to_numeric(r0.get("count"), errors="coerce")),
                }
            )
        if core_rows:
            core_csv = os.path.join(out_dir, "dataset_gt_supervision_core_timings.csv")
            pd.DataFrame(core_rows).to_csv(core_csv, index=False)
            outputs["gt_supervision_core_timings_csv"] = core_csv

        _log(verbose, f"[summarize] gt supervision timing rows={len(all_df)}")

    return outputs


def _aggregate_gt_centering_weighted_summaries(
    save_folder: str,
    out_dir: str,
    *,
    evaluated_images: Optional[set] = None,
    verbose: bool = True,
) -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    sup_root = os.path.join(save_folder, "supervision")
    if not os.path.isdir(sup_root):
        return outputs

    files = glob.glob(os.path.join(sup_root, "*", "analysis", "gt_centering_metrics_weighted_summary.csv"))
    if not files:
        return outputs

    frames = []
    eval_set = {str(x) for x in (evaluated_images or set())}
    for p in files:
        df = _safe_read_csv(p)
        if df is None:
            continue
        image = os.path.basename(os.path.dirname(os.path.dirname(p)))
        if eval_set and image not in eval_set:
            continue
        d = df.copy()
        if "image" not in d.columns:
            d["image"] = image
        d["source_file"] = p
        frames.append(d)

    if not frames:
        return outputs

    all_df = pd.concat(frames, ignore_index=True)
    all_csv = os.path.join(out_dir, "dataset_gt_centering_weighted_summary_all.csv")
    all_df.to_csv(all_csv, index=False)
    outputs["gt_centering_weighted_summary_all_csv"] = all_csv

    metric_cols = [c for c in all_df.columns if str(c).startswith("lwmean_")]
    if not metric_cols or "group" not in all_df.columns:
        return outputs

    rows = []
    for (grp, metric), g in (
        all_df.melt(
            id_vars=[c for c in ["image", "group"] if c in all_df.columns],
            value_vars=metric_cols,
            var_name="metric",
            value_name="value",
        )
        .groupby(["group", "metric"], dropna=False)
    ):
        vals = pd.to_numeric(g["value"], errors="coerce").to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        rows.append(
            {
                "group": str(grp),
                "metric": str(metric),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "std": float(np.std(vals)),
                "q25": float(np.percentile(vals, 25)),
                "q75": float(np.percentile(vals, 75)),
                "count": int(vals.size),
            }
        )
    if rows:
        sum_df = pd.DataFrame(rows).sort_values(["metric", "group"])
        out_csv = os.path.join(out_dir, "dataset_gt_centering_weighted_summary_grouped.csv")
        sum_df.to_csv(out_csv, index=False)
        outputs["gt_centering_weighted_summary_grouped_csv"] = out_csv


    # Two consolidated plots: one for atomic, one for combined(+noncombined atomic)
    # using mean bars with IQR error bars, and highlighting critical metrics.
    plot_dir = os.path.join(out_dir, "gt_centering_weighted_plots")
    os.makedirs(plot_dir, exist_ok=True)
    metric_order = [
        "lwmean_score_mid",
        "lwmean_nn_mean_bidirectional",
        "lwmean_hausdorff_max",
        "lwmean_coverage_min",
        "lwmean_hausdorff_p95",
        "lwmean_frechet_discrete_ds",
        "lwmean_mean_tan_angle_error_deg",
    ]
    metric_order = [m for m in metric_order if m in metric_cols]
    if not metric_order:
        metric_order = sorted(metric_cols)

    critical_metrics = {
        "lwmean_score_mid",
        "lwmean_nn_mean_bidirectional",
        "lwmean_hausdorff_max",
        "lwmean_coverage_min",
    }

    group_map = {
        "atomic": "atomic",
        "combined_plus_noncombined_atomic": "combined",
    }

    plot_df = sum_df.copy() if rows else pd.DataFrame()
    if not plot_df.empty:
        for raw_group, disp_group in group_map.items():
            sub = plot_df[plot_df["group"].astype(str) == raw_group].copy()
            if sub.empty:
                continue
            sub = sub.set_index("metric")
            labels = []
            means = []
            lo_err = []
            hi_err = []
            colors = []
            for m in metric_order:
                if m not in sub.index:
                    continue
                mean_v = float(pd.to_numeric(sub.loc[m, "mean"], errors="coerce"))
                q25_v = float(pd.to_numeric(sub.loc[m, "q25"], errors="coerce"))
                q75_v = float(pd.to_numeric(sub.loc[m, "q75"], errors="coerce"))
                if not np.isfinite(mean_v):
                    continue
                labels.append(m.replace("lwmean_", ""))
                means.append(mean_v)
                lo_err.append(max(0.0, mean_v - q25_v) if np.isfinite(q25_v) else 0.0)
                hi_err.append(max(0.0, q75_v - mean_v) if np.isfinite(q75_v) else 0.0)
                colors.append("#d62728" if m in critical_metrics else "#4c78a8")

            if not labels:
                continue

            x = np.arange(len(labels), dtype=float)
            fig_w = max(10.0, 0.65 * len(labels))
            fig, ax = plt.subplots(figsize=(fig_w, 5.0), dpi=180)
            ax.bar(x, means, color=colors, alpha=0.86)
            ax.errorbar(
                x,
                means,
                yerr=np.vstack([np.asarray(lo_err, float), np.asarray(hi_err, float)]),
                fmt="none",
                ecolor="black",
                elinewidth=1.2,
                capsize=3,
                zorder=3,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel("value")
            ax.set_title(f"GT Centering Weighted Summary ({disp_group})")
            ax.grid(axis="y", alpha=0.2)
            legend_handles = [
                Patch(facecolor="#d62728", edgecolor="none", label="critical"),
                Patch(facecolor="#4c78a8", edgecolor="none", label="non-critical"),
            ]
            ax.legend(handles=legend_handles, loc="best", framealpha=0.9, fontsize=8)
            plt.tight_layout()
            out_png = os.path.join(plot_dir, f"gt_centering_weighted_{disp_group}.png")
            fig.savefig(out_png, bbox_inches="tight")
            plt.close(fig)

        # Single RS3-like score comparison plot (atomic vs combined).
        score_metric = "lwmean_score_mid"
        if score_metric in plot_df["metric"].astype(str).values:
            labels = []
            means = []
            lo_err = []
            hi_err = []
            for raw_group, disp_group in group_map.items():
                row = plot_df[
                    (plot_df["group"].astype(str) == raw_group)
                    & (plot_df["metric"].astype(str) == score_metric)
                ]
                if row.empty:
                    continue
                r0 = row.iloc[0]
                mean_v = float(pd.to_numeric(r0.get("mean", np.nan), errors="coerce"))
                q25_v = float(pd.to_numeric(r0.get("q25", np.nan), errors="coerce"))
                q75_v = float(pd.to_numeric(r0.get("q75", np.nan), errors="coerce"))
                if not np.isfinite(mean_v):
                    continue
                labels.append("atomic" if disp_group == "atomic" else "combined")
                means.append(mean_v)
                lo_err.append(max(0.0, mean_v - q25_v) if np.isfinite(q25_v) else 0.0)
                hi_err.append(max(0.0, q75_v - mean_v) if np.isfinite(q75_v) else 0.0)

            if labels:
                x = np.arange(len(labels), dtype=float)
                fig, ax = plt.subplots(figsize=(5.4, 4.0), dpi=180)
                ax.bar(x, means, color=["#d62728", "#4c78a8"][: len(labels)], alpha=0.9)
                ax.errorbar(
                    x,
                    means,
                    yerr=np.vstack([np.asarray(lo_err, float), np.asarray(hi_err, float)]),
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.2,
                    capsize=3,
                    zorder=3,
                )
                ax.set_xticks(x)
                ax.set_xticklabels(labels)
                ax.set_ylabel("lwmean_score_mid")
                ax.set_title("GT Centering RS3 Score (lower is better): Atomic vs Combined")
                ax.grid(axis="y", alpha=0.2)
                plt.tight_layout()
                out_png = os.path.join(plot_dir, "gt_centering_weighted_score_mid_atomic_vs_combined.png")
                fig.savefig(out_png, bbox_inches="tight")
                plt.close(fig)

    outputs["gt_centering_weighted_plots_dir"] = plot_dir
    _log(verbose, f"[summarize] gt centering weighted summaries rows={len(all_df)}")
    return outputs


def _aggregate_gt_component_timings(
    supervision_root: str,
    out_dir: str,
    *,
    evaluated_images: Optional[set] = None,
    verbose: bool = True,
) -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    if not supervision_root or not os.path.isdir(supervision_root):
        return outputs

    eval_set = {str(x) for x in (evaluated_images or set())}
    center_files = glob.glob(os.path.join(supervision_root, "**", "analysis", "gt_centering_timing.csv"), recursive=True)
    center_files += glob.glob(os.path.join(supervision_root, "**", "analysis", "gt_compute_timing.csv"), recursive=True)
    center_files += glob.glob(os.path.join(supervision_root, "**", "gt_centering", "timing.csv"), recursive=True)
    dt_files = glob.glob(os.path.join(supervision_root, "**", "dt", "timing.csv"), recursive=True)
    depth_files = glob.glob(os.path.join(supervision_root, "**", "multi_cue_track", "timing.csv"), recursive=True)
    depth_files += glob.glob(os.path.join(supervision_root, "**", "multi_cue", "timing.csv"), recursive=True)
    depth_files += glob.glob(os.path.join(supervision_root, "**", "dt_ridge_color_depth", "timing.csv"), recursive=True)
    depth_files = list(dict.fromkeys(depth_files))

    def _collect(files: List[str]) -> pd.DataFrame:
        frames = []
        for p in files:
            df = _safe_read_csv(p)
            if df is None or df.empty:
                continue
            parts = os.path.normpath(p).split(os.sep)
            image = "unknown"
            if "supervision" in parts:
                i = parts.index("supervision")
                if i + 1 < len(parts):
                    image = str(parts[i + 1])
            if eval_set and image not in eval_set:
                continue
            d = df.copy()
            if "image" not in d.columns:
                d["image"] = image
            d["source_file"] = p
            frames.append(d)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    center_df = _collect(center_files)
    dt_df = _collect(dt_files)
    depth_df = _collect(depth_files)

    if not center_df.empty:
        p = os.path.join(out_dir, "dataset_gt_centering_timing_all.csv")
        center_df.to_csv(p, index=False)
        outputs["dataset_gt_centering_timing_all_csv"] = p
    if not dt_df.empty:
        p = os.path.join(out_dir, "dataset_dt_timing_all.csv")
        dt_df.to_csv(p, index=False)
        outputs["dataset_dt_timing_all_csv"] = p
    if not depth_df.empty:
        p = os.path.join(out_dir, "dataset_multi_cue_timing_all.csv")
        depth_df.to_csv(p, index=False)
        outputs["dataset_multi_cue_timing_all_csv"] = p

    if not center_df.empty or not dt_df.empty or not depth_df.empty:
        _log(
            verbose,
            f"[summarize] gt component timings: gt_centering_rows={len(center_df)} dt_rows={len(dt_df)} depth_rows={len(depth_df)}",
        )
    return outputs


def _aggregate_depth_generation_timings(
    *,
    out_dir: str,
    depth_timing_csv: Optional[str],
    evaluated_images: Optional[set] = None,
    verbose: bool = True,
) -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    if not depth_timing_csv or not os.path.isfile(depth_timing_csv):
        return outputs

    df = _safe_read_csv(depth_timing_csv)
    if df is None or df.empty:
        return outputs

    # Accept common schemas from depth export.
    img_col = "image" if "image" in df.columns else None
    sec_col = "seconds" if "seconds" in df.columns else None
    if img_col is None:
        for c in df.columns:
            if str(c).lower() in {"image_name", "filename", "file", "name"}:
                img_col = c
                break
    if sec_col is None:
        for c in df.columns:
            if str(c).lower() in {"sec", "time_sec", "inference_seconds", "seconds"}:
                sec_col = c
                break
    if img_col is None or sec_col is None:
        _log(verbose, f"[summarize] depth generation timing csv missing required columns: {depth_timing_csv}")
        return outputs

    eval_set = {str(x) for x in (evaluated_images or set())}
    d = df[[img_col, sec_col]].copy()
    d["image"] = d[img_col].astype(str).apply(lambda s: os.path.splitext(os.path.basename(s))[0])
    d["multi_cue_generation_s"] = pd.to_numeric(d[sec_col], errors="coerce")
    d = d[np.isfinite(d["multi_cue_generation_s"])].copy()
    if eval_set:
        d = d[d["image"].isin(eval_set)].copy()
    if d.empty:
        return outputs

    # Collapse duplicates per image if present.
    d = d.groupby("image", dropna=False, as_index=False)["multi_cue_generation_s"].mean()
    d["source_file"] = depth_timing_csv

    all_csv = os.path.join(out_dir, "dataset_depth_generation_timing_all.csv")
    d.to_csv(all_csv, index=False)
    outputs["dataset_depth_generation_timing_all_csv"] = all_csv

    # Optional weights by image from edge-stage timing output.
    weight_map = {}
    edge_csv = os.path.join(out_dir, "dataset_edge_tracking_stage_all.csv")
    edge_df = _safe_read_csv(edge_csv)
    if edge_df is not None and not edge_df.empty and "image" in edge_df.columns:
        wcol = next((c for c in ["crack_px", "length_px", "finite_len_px"] if c in edge_df.columns), None)
        if wcol is not None:
            ew = edge_df[["image", wcol]].copy()
            ew["image"] = ew["image"].astype(str)
            ew[wcol] = pd.to_numeric(ew[wcol], errors="coerce")
            ew = ew[np.isfinite(ew[wcol]) & (ew[wcol] > 0)]
            if not ew.empty:
                weight_map = ew.groupby("image", dropna=False)[wcol].sum().to_dict()

    vals = pd.to_numeric(d["multi_cue_generation_s"], errors="coerce").to_numpy(float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return outputs
    mean_t = float(np.mean(vals))
    total_t = float(np.sum(vals))

    if weight_map:
        ww = np.asarray([float(weight_map.get(im, np.nan)) for im in d["image"].astype(str).tolist()], dtype=float)
        vv = pd.to_numeric(d["multi_cue_generation_s"], errors="coerce").to_numpy(float)
        ok = np.isfinite(vv) & np.isfinite(ww) & (ww > 0)
        weighted_mean_t = float(np.sum(vv[ok] * ww[ok]) / np.sum(ww[ok])) if np.any(ok) else mean_t
    else:
        weighted_mean_t = mean_t

    missing_count = 0
    if eval_set:
        missing_count = int(len(eval_set.difference(set(d["image"].astype(str).tolist()))))

    roll_df = pd.DataFrame(
        [
            {
                "category": "multi_cue_generation",
                "component": "multi_cue_generation_s",
                "mean_sec": mean_t,
                "weighted_mean_sec": weighted_mean_t,
                "total_sec": total_t,
                "count": int(len(d)),
                "missing_count": int(missing_count),
            }
        ]
    )
    roll_csv = os.path.join(out_dir, "dataset_depth_generation_timing_rollup.csv")
    roll_df.to_csv(roll_csv, index=False)
    outputs["dataset_depth_generation_timing_rollup_csv"] = roll_csv

    _log(verbose, f"[summarize] depth generation timings matched={len(d)} missing={missing_count}")
    return outputs


def _plot_dataset_full_timing_overview(
    out_dir: str,
    *,
    verbose: bool = True,
):
    outputs = {}

    def _log(v, msg):
        if v:
            print(msg)

    def _safe_read(path):
        try:
            if os.path.isfile(path):
                return pd.read_csv(path)
        except Exception:
            pass
        return None

    def _safe_num(v):
        try:
            n = float(pd.to_numeric(v, errors="coerce"))
        except Exception:
            n = np.nan
        return n if np.isfinite(n) else np.nan

    def _add(rows, method, sec, category, err=None):
        v = _safe_num(sec)
        if np.isfinite(v) and (v > 0):
            e = _safe_num(err)
            if not np.isfinite(e) or e < 0:
                e = 0.0
            rows.append({"method": str(method), "sec": float(v), "category": str(category), "err": float(e)})

    def _plot_algorithm_overview(method_rows, out_png, title):
        if not method_rows:
            return None
        method_rows = sorted(method_rows, key=lambda x: x["sec"])
        labels = [r["method"] for r in method_rows]
        vals = [r["sec"] for r in method_rows]
        errs = [max(0.0, float(pd.to_numeric(r.get("err", 0.0), errors="coerce") or 0.0)) for r in method_rows]
        colors = {
            "baseline_seg": "#2ca02c",
            "baseline_width": "#ff7f0e",
            "ET": "#d62728",
            "manual": "#d62728",
            "auto": "#9467bd",
            "multi_cue": "#17becf",
            "gt_centering": "#1f77b4",
            "gt_supervision": "#8c564b",
            "gt_subtiming": "#7f7f7f",
        }
        bar_colors = [colors.get(r["category"], "#777777") for r in method_rows]
        fig_w = max(9.0, 0.46 * len(labels))
        fig, ax = plt.subplots(figsize=(fig_w, 4.2), dpi=180)
        x = np.arange(len(labels))
        ax.bar(x, vals, color=bar_colors)
        ax.errorbar(
            x,
            vals,
            yerr=np.asarray(errs, float),
            fmt="none",
            ecolor="black",
            elinewidth=1.1,
            capsize=3,
            zorder=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(_clean_plot_label("seconds"))
        ax.set_title(_clean_plot_label(title))
        ax.grid(axis="y", alpha=0.25)
        present = []
        for r in method_rows:
            c = str(r.get("category", ""))
            if c and c not in present:
                present.append(c)
        if present:
            handles = [Patch(facecolor=colors.get(c, "#777777"), edgecolor="none", label=c) for c in present]
            ax.legend(handles=handles, loc="best", framealpha=0.9, fontsize=8)
        plt.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        return out_png

    def _plot_components(df, cols, title, out_png):
        if df is None or df.empty:
            return None
        labels = []
        vals = []
        errs = []
        for c in cols:
            if c not in df.columns:
                continue
            arr = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            v = _safe_num(np.mean(arr))
            e = _safe_num(np.std(arr))
            if np.isfinite(v):
                labels.append(re.sub(r"(_s|_sec)$", "", str(c)))
                vals.append(float(v))
                errs.append(float(e) if np.isfinite(e) and e > 0 else 0.0)
        if not vals:
            return None
        fig_w = max(6.0, 0.65 * len(labels))
        fig, ax = plt.subplots(figsize=(fig_w, 4.0), dpi=180)
        x = np.arange(len(labels))
        ax.bar(x, vals, color="#4c78a8")
        ax.errorbar(
            x,
            vals,
            yerr=np.asarray(errs, float),
            fmt="none",
            ecolor="black",
            elinewidth=1.1,
            capsize=3,
            zorder=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(_clean_plot_label("seconds"))
        ax.set_title(_clean_plot_label(title))
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        return out_png

    def _plot_components_with_total(df, cols, title, out_png):
        """Like _plot_components but appends a TOTAL bar summing all components."""
        if df is None or df.empty:
            return None
        labels = []
        vals = []
        errs = []
        per_row_sum = None
        for c in cols:
            if c not in df.columns:
                continue
            arr = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
            arr_clean = np.where(np.isfinite(arr), arr, 0.0)
            if np.all(arr_clean == 0.0):
                continue
            pos = arr_clean[arr_clean > 0]
            v = _safe_num(np.mean(pos) if pos.size else 0.0)
            e = _safe_num(np.std(pos) if pos.size else 0.0)
            if not np.isfinite(v):
                continue
            disp = re.sub(r"(_s|_sec)$", "", c)
            labels.append(disp)
            vals.append(float(v))
            errs.append(float(e) if np.isfinite(e) and e > 0 else 0.0)
            if per_row_sum is None:
                per_row_sum = arr_clean.copy()
            else:
                per_row_sum = per_row_sum + arr_clean

        if not vals:
            return None

        if per_row_sum is not None and np.any(per_row_sum > 0):
            pos_total = per_row_sum[per_row_sum > 0]
            total_mean = _safe_num(float(np.mean(pos_total))) if pos_total.size else np.nan
            total_err = _safe_num(float(np.std(pos_total))) if pos_total.size else 0.0
            if np.isfinite(total_mean):
                labels.append("TOTAL")
                vals.append(float(total_mean))
                errs.append(float(total_err) if np.isfinite(total_err) and total_err > 0 else 0.0)

        bar_colors = ["#4c78a8"] * (len(labels) - 1) + ["#d62728"]
        fig_w = max(7.0, 0.75 * len(labels))
        fig, ax = plt.subplots(figsize=(fig_w, 4.2), dpi=180)
        x = np.arange(len(labels))
        ax.bar(x, vals, color=bar_colors, alpha=0.88)
        ax.errorbar(
            x,
            vals,
            yerr=np.asarray(errs, float),
            fmt="none",
            ecolor="black",
            elinewidth=1.1,
            capsize=3,
            zorder=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(_clean_plot_label("seconds (mean per image)"))
        ax.set_title(_clean_plot_label(title))
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        return out_png

    def _plot_atomic_vs_combined(df, title, out_png):
        if df is None or df.empty:
            return None
        # Support both old and renamed column variants
        def _get_col(df, *candidates):
            for c in candidates:
                if c in df.columns:
                    v = pd.to_numeric(df[c], errors="coerce")
                    if v.notna().any():
                        return v.to_numpy(float)
            return np.zeros(len(df), dtype=float)
        a_arr = _get_col(df, "atomic_centering_sec", "multi_cue_atomic_centering_sec")
        c_arr = _get_col(df, "combined_centering_sec", "multi_cue_combined_centering_sec", "combined_atomics_centering_sec")
        a_arr = a_arr[np.isfinite(a_arr)]
        c_arr = c_arr[np.isfinite(c_arr)]
        atomic = _safe_num(np.mean(a_arr) if a_arr.size else np.nan)
        combined = _safe_num(np.mean(c_arr) if c_arr.size else np.nan)
        atomic_err = _safe_num(np.std(a_arr) if a_arr.size else 0.0)
        combined_err = _safe_num(np.std(c_arr) if c_arr.size else 0.0)
        if not (np.isfinite(atomic) or np.isfinite(combined)):
            return None
        vals = [float(atomic) if np.isfinite(atomic) else 0.0, float(combined) if np.isfinite(combined) else 0.0]
        errs = [
            float(atomic_err) if np.isfinite(atomic_err) and atomic_err > 0 else 0.0,
            float(combined_err) if np.isfinite(combined_err) and combined_err > 0 else 0.0,
        ]
        fig, ax = plt.subplots(figsize=(5.0, 4.0), dpi=180)
        ax.bar(["atomic", "combined"], vals, color=["#f28e2b", "#4e79a7"])
        ax.errorbar(
            np.arange(2, dtype=float),
            vals,
            yerr=np.asarray(errs, float),
            fmt="none",
            ecolor="black",
            elinewidth=1.1,
            capsize=3,
            zorder=3,
        )
        ax.set_ylabel(_clean_plot_label("seconds"))
        ax.set_title(_clean_plot_label(title))
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        return out_png

    mean_rows = []
    sum_rows = []

    # Baseline segmentation + width algorithms from rollup.
    df_base = _safe_read(os.path.join(out_dir, "dataset_baseline_timings_rollup.csv"))
    if df_base is not None and not df_base.empty:
        df_base_all = _safe_read(os.path.join(out_dir, "dataset_baseline_timings_all.csv"))
        base_wmean_shared = np.nan
        base_wmean_skeleton = np.nan
        base_sum_shared = np.nan
        base_sum_skeleton = np.nan
        for _, r in df_base.iterrows():
            c0 = str(r.get("component", "")).lower()
            if c0.startswith("shared_mat_gpu"):
                base_wmean_shared = _safe_num(r.get("weighted_mean_sec"))
                base_sum_shared = _safe_num(r.get("total_sec"))
            elif c0.startswith("skeleton_dse"):
                base_wmean_skeleton = _safe_num(r.get("weighted_mean_sec"))
                base_sum_skeleton = _safe_num(r.get("total_sec"))

        comp_err_mean = {}
        comp_err_sum = {}
        if df_base_all is not None and not df_base_all.empty:
            for c in df_base_all.columns:
                if c in df_base.columns or (str(c).endswith("_s") or str(c).endswith("_sec")):
                    arr = pd.to_numeric(df_base_all[c], errors="coerce").to_numpy(float)
                    arr = arr[np.isfinite(arr)]
                    if arr.size:
                        s = float(np.std(arr))
                        comp_err_mean[str(c)] = s
                        comp_err_sum[str(c)] = float(s * np.sqrt(arr.size))

        def _err_by_prefix(err_map: dict, prefix: str) -> float:
            p = str(prefix).lower()
            for k, v in err_map.items():
                if str(k).lower().startswith(p):
                    try:
                        return float(v)
                    except Exception:
                        return 0.0
            return 0.0

        for _, r in df_base.iterrows():
            comp = str(r.get("component", ""))
            v_mean = _safe_num(r.get("weighted_mean_sec"))
            v_sum = _safe_num(r.get("total_sec"))
            err_mean = float(comp_err_mean.get(comp, 0.0))
            err_sum = float(comp_err_sum.get(comp, 0.0))
            if comp.startswith("inference_seconds:"):
                _add(mean_rows, comp.split(":", 1)[1], v_mean, "baseline_seg", err_mean)
                _add(sum_rows, comp.split(":", 1)[1], v_sum, "baseline_seg", err_sum)
            elif any(k in comp for k in ("mat_", "pca_", "esd_", "eob_", "width", "skeleton_dse")) and "skeleton_graph" not in comp.lower():
                comp_low = comp.lower()
                label = _display_timing_component_name(comp)
                if any(k in comp_low for k in ("pca", "esd", "eob")):
                    if np.isfinite(base_wmean_shared):
                        v_mean = _safe_num(v_mean + base_wmean_shared)
                        err_mean = float(np.sqrt((err_mean ** 2) + (_err_by_prefix(comp_err_mean, "shared_mat_gpu") ** 2)))
                    if np.isfinite(base_wmean_skeleton):
                        v_mean = _safe_num(v_mean + base_wmean_skeleton)
                        err_mean = float(np.sqrt((err_mean ** 2) + (_err_by_prefix(comp_err_mean, "skeleton_dse") ** 2)))
                    if np.isfinite(base_sum_shared):
                        v_sum = _safe_num(v_sum + base_sum_shared)
                        err_sum = float(np.sqrt((err_sum ** 2) + (_err_by_prefix(comp_err_sum, "shared_mat_gpu") ** 2)))
                    if np.isfinite(base_sum_skeleton):
                        v_sum = _safe_num(v_sum + base_sum_skeleton)
                        err_sum = float(np.sqrt((err_sum ** 2) + (_err_by_prefix(comp_err_sum, "skeleton_dse") ** 2)))
                _add(mean_rows, label, v_mean, "baseline_width", err_mean)
                _add(sum_rows, label, v_sum, "baseline_width", err_sum)

    # Manual / auto pipeline from stage timing rows.
    df_edge = _safe_read(os.path.join(out_dir, "dataset_edge_tracking_stage_all.csv"))
    if df_edge is not None and not df_edge.empty and "supervision" in df_edge.columns:
        weight_col = next((c for c in ["length_px", "finite_len_px", "crack_px"] if c in df_edge.columns), None)
        for sup in ("manual", "auto"):
            g = df_edge[df_edge["supervision"].astype(str) == sup].copy()
            if g.empty:
                continue
            comp_cols = [c for c in ["edge_masks_sec", "edges_tracking_sec", "build_combined_sec"] if c in g.columns]
            if sup == "auto":
                comp_cols += [c for c in ["os_cost_sec", "midline_tracking_sec"] if c in g.columns]
            if not comp_cols:
                continue
            per_row = np.zeros(len(g), float)
            for c in comp_cols:
                per_row += pd.to_numeric(g[c], errors="coerce").fillna(0.0).to_numpy(float)
            total_sec = float(np.nansum(per_row))
            s = float(np.nanstd(per_row))
            n = int(np.count_nonzero(np.isfinite(per_row)))
            if weight_col:
                w = pd.to_numeric(g[weight_col], errors="coerce").fillna(0.0).to_numpy(float)
                ok = np.isfinite(per_row) & np.isfinite(w) & (w > 0)
                mean_sec = float(np.sum(per_row[ok] * w[ok]) / np.sum(w[ok])) if np.any(ok) else float(np.nanmean(per_row))
            else:
                mean_sec = float(np.nanmean(per_row))
            method_label = "ET" if sup == "manual" else sup
            category_label = "ET" if sup == "manual" else sup
            _add(mean_rows, method_label, mean_sec, category_label, s)
            _add(sum_rows, method_label, total_sec, category_label, float(s * np.sqrt(max(1, n))))

    # GT supervision timing rollup (dataset-level): explicit atomic/combined rows.
    df_sup = _safe_read(os.path.join(out_dir, "dataset_gt_supervision_timings.csv"))
    df_sup_core = _safe_read(os.path.join(out_dir, "dataset_gt_supervision_core_timings.csv"))
    gt_overview_specs = [
        ("gt_compute", "normals", "combined_plus_noncombined_atomics_sec"),
        ("gt_centering", "centering", "combined_plus_noncombined_atomics_centering_sec"),
    ]
    if df_sup_core is not None and not df_sup_core.empty:
        for _, r in df_sup_core.iterrows():
            method = str(r.get("method", ""))
            if method in ("gt_compute_combined", "gt_compute_atomic"):
                label = "gt_compute"
            elif method in ("gt_centering_combined", "gt_centering_atomic"):
                label = "gt_centering"
            else:
                continue
            if "combined" in method and "atomic" not in method:
                _add(mean_rows, label, r.get("weighted_mean_sec"), "gt_supervision", 0.0)
                _add(sum_rows, label, r.get("total_sec"), "gt_supervision", 0.0)
    elif df_sup is not None and not df_sup.empty:
        for method_name, stage_name, comp_name in gt_overview_specs:
            g = df_sup[
                (df_sup["stage"].astype(str) == stage_name)
                & (df_sup["component"].astype(str) == comp_name)
            ]
            if g.empty:
                continue
            r = g.iloc[0]
            _add(mean_rows, method_name, r.get("weighted_mean_sec"), "gt_supervision", 0.0)
            _add(sum_rows, method_name, r.get("total_sec"), "gt_supervision", 0.0)

    # dt timing from dt track CSV summary.
    df_dt = _safe_read(os.path.join(out_dir, "dataset_dt_timing_all.csv"))
    if df_dt is not None and not df_dt.empty:
        # Use pure DT compute timing, not full centering loop totals.
        dt_col = "dt_compute_s" if "dt_compute_s" in df_dt.columns else None
        if dt_col:
            dt_arr = pd.to_numeric(df_dt[dt_col], errors="coerce").to_numpy(float)
            dt_arr = dt_arr[np.isfinite(dt_arr)]
            if dt_arr.size:
                _add(mean_rows, "dt", float(np.mean(dt_arr)), "gt_centering", float(np.std(dt_arr)))
                _add(sum_rows, "dt", float(np.sum(dt_arr)), "gt_centering", float(np.std(dt_arr) * np.sqrt(dt_arr.size)))

    # dt / dt_best from calibration ablation timing summary.
    abl_tim_csv = os.path.join(out_dir, "ablation", "ablation_timings_summary.csv")
    df_abl_tim = _safe_read(abl_tim_csv)
    if df_abl_tim is not None and not df_abl_tim.empty:
        for _, r in df_abl_tim.iterrows():
            method = str(r.get("method", ""))
            mean_s = _safe_num(r.get("mean_sec"))
            std_s = _safe_num(r.get("std_sec", 0.0))
            n = int(pd.to_numeric(r.get("n", 1), errors="coerce") or 1)
            total_s = mean_s * n if np.isfinite(mean_s) and n > 0 else np.nan
            err_sum = float(std_s * np.sqrt(n)) if np.isfinite(std_s) else 0.0
            category = "auto"
            if "et" in method.lower():
                category = "ET"
            _add(mean_rows, method, mean_s, category, float(std_s) if np.isfinite(std_s) else 0.0)
            _add(sum_rows, method, total_s, category, err_sum)

    # GT centering + depth distridge (dataset-level aggregations).
    df_gt = _safe_read(os.path.join(out_dir, "dataset_gt_centering_timing_all.csv"))
    if (df_gt is None) or df_gt.empty:
        df_gt = _safe_read(os.path.join(out_dir, "gt_centering", "timing.csv"))
    gt_centering_already_added = any(str(r.get("method", "")) == "gt_centering" for r in mean_rows)
    if (not gt_centering_already_added) and df_gt is not None and not df_gt.empty:
        gt_mean_arr = pd.to_numeric(df_gt.get("combined_centering_sec"), errors="coerce").to_numpy(float)
        gt_mean_arr = gt_mean_arr[np.isfinite(gt_mean_arr)]
        gt_sum_arr = pd.to_numeric(
            df_gt["centering_total_sec"] if "centering_total_sec" in df_gt.columns else df_gt.get("combined_centering_sec"),
            errors="coerce",
        ).to_numpy(float)
        gt_sum_arr = gt_sum_arr[np.isfinite(gt_sum_arr)]
        _add(
            mean_rows,
            "gt_centering",
            pd.to_numeric(df_gt.get("combined_centering_sec"), errors="coerce").mean(),
            "gt_centering",
            float(np.std(gt_mean_arr)) if gt_mean_arr.size else 0.0,
        )
        _add(
            sum_rows,
            "gt_centering",
            pd.to_numeric(
                df_gt["centering_total_sec"] if "centering_total_sec" in df_gt.columns else df_gt.get("combined_centering_sec"),
                errors="coerce",
            ).sum(),
            "gt_centering",
            float(np.std(gt_sum_arr) * np.sqrt(max(1, gt_sum_arr.size))) if gt_sum_arr.size else 0.0,
        )

    df_depth = _safe_read(os.path.join(out_dir, "dataset_multi_cue_timing_all.csv"))
    if (df_depth is None) or df_depth.empty:
        df_depth = _safe_read(os.path.join(out_dir, "multi_cue", "timing.csv"))
    df_depth_gen = _safe_read(os.path.join(out_dir, "dataset_depth_generation_timing_all.csv"))
    df_depth_gen_roll = _safe_read(os.path.join(out_dir, "dataset_depth_generation_timing_rollup.csv"))
    if df_depth is not None and not df_depth.empty:
        dd = df_depth.copy()
        if df_depth_gen is not None and not df_depth_gen.empty:
            gmap = (
                df_depth_gen[["image", "multi_cue_generation_s"]]
                .dropna()
                .assign(image=lambda x: x["image"].astype(str))
                .drop_duplicates(subset=["image"], keep="last")
                .set_index("image")["multi_cue_generation_s"]
                .to_dict()
            )
            if "image" in dd.columns and gmap:
                dd["multi_cue_generation_s"] = dd["image"].astype(str).map(gmap)
            else:
                dd["multi_cue_generation_s"] = np.nan
        elif df_depth_gen_roll is not None and not df_depth_gen_roll.empty:
            dd["multi_cue_generation_s"] = pd.to_numeric(
                df_depth_gen_roll.iloc[0].get("weighted_mean_sec", np.nan),
                errors="coerce",
            )
        else:
            dd["multi_cue_generation_s"] = np.nan

        mc_centering_col = next(
            (
                c for c in (
                    "combined_plus_noncombined_atomics_centering_sec",
                    "combined_centering_sec",
                    "centering_total_sec",
                )
                if c in dd.columns
            ),
            None,
        )
        # Prefer explicit multi-cue components.
        _mc_component_cols = [
            c for c in ("dt_compute_s", "multi_cue_costmap_s", "multi_cue_dijkstra_s", "normals_depth_s")
            if c in dd.columns
        ]
        if _mc_component_cols:
            dd["multi_cue_total_with_generation_sec"] = sum(
                pd.to_numeric(dd[c], errors="coerce").fillna(0.0) for c in _mc_component_cols
            )
        elif mc_centering_col:
            dd["multi_cue_total_with_generation_sec"] = (
                pd.to_numeric(dd[mc_centering_col], errors="coerce")
                + pd.to_numeric(dd["multi_cue_generation_s"], errors="coerce").fillna(0.0)
            )
        else:
            dd["multi_cue_total_with_generation_sec"] = pd.to_numeric(
                dd["multi_cue_generation_s"], errors="coerce"
            ).fillna(0.0)

        print(f"[TIMING DBG] multi_cue timing CSV columns: {list(dd.columns[:20])}")
        print(
            f"[TIMING DBG] multi_cue_total_with_generation_sec sample: "
            f"{pd.to_numeric(dd.get('multi_cue_total_with_generation_sec'), errors='coerce').describe()}"
        )
        _add(
            mean_rows,
            "multi_cue",
            pd.to_numeric(dd.get("multi_cue_total_with_generation_sec"), errors="coerce").mean(),
            "multi_cue",
            float(np.nanstd(pd.to_numeric(dd.get("multi_cue_total_with_generation_sec"), errors="coerce").to_numpy(float))),
        )
        _add(
            sum_rows,
            "multi_cue",
            pd.to_numeric(
                dd["multi_cue_total_with_generation_sec"],
                errors="coerce",
            ).sum(),
            "multi_cue",
            float(
                np.nanstd(pd.to_numeric(dd.get("multi_cue_total_with_generation_sec"), errors="coerce").to_numpy(float))
                * np.sqrt(max(1, int(np.isfinite(pd.to_numeric(dd.get("multi_cue_total_with_generation_sec"), errors="coerce")).sum())))
            ),
        )
        dg_arr = pd.to_numeric(dd.get("multi_cue_generation_s"), errors="coerce").to_numpy(float)
        dg_arr = dg_arr[np.isfinite(dg_arr)]
        if dg_arr.size:
            _add(mean_rows, "multi_cue_generation", float(np.mean(dg_arr)), "multi_cue", float(np.std(dg_arr)))
            _add(sum_rows, "multi_cue_generation", float(np.sum(dg_arr)), "multi_cue", float(np.std(dg_arr) * np.sqrt(dg_arr.size)))
        df_depth = dd

    if mean_rows:
        out_mean_png = os.path.join(out_dir, "dataset_algorithm_timing_overview.png")
        outputs["dataset_algorithm_timing_overview_png"] = _plot_algorithm_overview(
            mean_rows,
            out_mean_png,
            "Dataset Algorithm Timing Overview",
        )
        out_mean_csv = os.path.join(out_dir, "dataset_algorithm_timing_overview.csv")
        pd.DataFrame(mean_rows).sort_values("sec").to_csv(out_mean_csv, index=False)
        outputs["dataset_algorithm_timing_overview_csv"] = out_mean_csv

    if sum_rows:
        out_sum_png = os.path.join(out_dir, "dataset_algorithm_timing_sum.png")
        outputs["dataset_algorithm_timing_sum_png"] = _plot_algorithm_overview(
            sum_rows,
            out_sum_png,
            "Dataset Algorithm Timing Sum",
        )
        out_sum_csv = os.path.join(out_dir, "dataset_algorithm_timing_sum.csv")
        pd.DataFrame(sum_rows).sort_values("sec").to_csv(out_sum_csv, index=False)
        outputs["dataset_algorithm_timing_sum_csv"] = out_sum_csv

    # Segmentation-only and width/midline subset charts (mean metric).
    if mean_rows:
        df_mean = pd.DataFrame(mean_rows)
        seg_subset = df_mean[
            df_mean["category"].isin(["baseline_seg", "ET"])
            | df_mean["method"].astype(str).str.upper().eq("ET")
        ].to_dict("records")
        width_subset = df_mean[
            df_mean["category"].isin(["baseline_width", "ET", "auto", "multi_cue", "gt_centering"])
        ].to_dict("records")
        outputs["seg_chart"] = _plot_algorithm_overview(
            seg_subset,
            os.path.join(out_dir, "dataset_segmentation_timing.png"),
            "Segmentation Timing",
        )
        outputs["width_chart"] = _plot_algorithm_overview(
            width_subset,
            os.path.join(out_dir, "dataset_width_methods_timing.png"),
            "Width / Midline Methods Timing",
        )
        outputs["overview_chart"] = outputs.get("dataset_algorithm_timing_overview_png")

    # Component charts
    outputs["gt_centering_components_chart"] = _plot_components(
        df_gt,
        ["dt_compute_s", "centered_snap_s", "normals_centered_s"],
        "Dataset GT Centering Components",
        os.path.join(out_dir, "dataset_gt_centering_components.png"),
    )
    multicue_comp_pairs = [
        ("dt_compute_s", None),
        ("multi_cue_generation_s", None),
        ("multi_cue_align_s", "multi_cue_align_s"),
        ("depth_recess_s", "depth_recess_s"),
        ("multi_cue_costmap_s", "multi_cue_costmap_s"),
        ("multi_cue_dijkstra_s", "multi_cue_dijkstra_s"),
        ("multi_cue_postprocess_s", "multi_cue_postprocess_s"),
        ("normals_multi_cue_s", "normals_depth_s"),
    ]
    multicue_comp_cols: List[str] = []
    if df_depth is not None and not df_depth.empty:
        for pref, legacy in multicue_comp_pairs:
            if pref in df_depth.columns:
                multicue_comp_cols.append(pref)
            elif legacy and legacy in df_depth.columns:
                multicue_comp_cols.append(legacy)
    outputs["multi_cue_components_chart"] = _plot_components_with_total(
        df_depth,
        multicue_comp_cols,
        "Multi-Cue Dataset Components",
        os.path.join(out_dir, "dataset_multi_cue_components.png"),
    )

    # Atomic vs combined charts
    outputs["gt_centering_atomic_vs_combined_chart"] = _plot_atomic_vs_combined(
        df_gt,
        "GT Centering Atomic vs Combined",
        os.path.join(out_dir, "dataset_gt_centering_atomic_vs_combined.png"),
    )
    outputs["multi_cue_atomic_vs_combined_chart"] = _plot_atomic_vs_combined(
        df_depth,
        "Multi-Cue Atomic vs Combined",
        os.path.join(out_dir, "dataset_multi_cue_atomic_vs_combined.png"),
    )

    _log(verbose, f"[timing] mean_rows={len(mean_rows)} sum_rows={len(sum_rows)}")
    return outputs


def _aggregate_invalid_matches(
    image_dirs: List[str],
    out_dir: str,
    *,
    verbose: bool,
) -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    rows: List[pd.DataFrame] = []

    def _parse_context_from_rel(rel_dir: str) -> Tuple[str, str, str]:
        parts = [p for p in rel_dir.replace("\\", "/").split("/") if p and p != "."]
        midline_type = "unknown"
        crack_type = "unknown"
        method = "unknown"
        if len(parts) >= 1 and parts[0] in ("manual", "auto", "et", "ET"):
            midline_type = _display_midline_label(parts[0])
            crack_type = parts[1] if len(parts) >= 2 else "unknown"
            method = str(midline_type)
        elif len(parts) >= 2 and parts[1] in ("manual", "auto", "et", "ET"):
            baseline = str(parts[0])
            midline_type = _display_midline_label(str(parts[1]))
            crack_type = parts[2] if len(parts) >= 3 else "unknown"
            method = f"{baseline}:{midline_type}"
        elif len(parts) >= 1:
            midline_type = str(parts[0])
            crack_type = parts[1] if len(parts) >= 2 else "unknown"
            method = str(midline_type)
        return midline_type, crack_type, method

    for img_dir in image_dirs:
        image_name = os.path.basename(img_dir)
        found = glob.glob(os.path.join(img_dir, "**", "invalid_matches.csv"), recursive=True)
        for csv_path in found:
            df = _safe_read_csv(csv_path)
            if df is None or df.empty:
                continue
            rel_dir = os.path.relpath(os.path.dirname(csv_path), img_dir)
            midline_type, crack_type, method = _parse_context_from_rel(rel_dir)

            d = df.copy()
            if "image" not in d.columns:
                d["image"] = image_name
            d["image"] = d["image"].astype(str)
            d["method"] = method
            d["midline_type"] = midline_type
            d["crack_type"] = crack_type
            d["source_csv"] = csv_path

            for c in ("cid", "level", "entity_id", "reason", "branch_id", "extra_info"):
                if c not in d.columns:
                    d[c] = ""
            for c in ("length", "n_segments", "overlap"):
                if c not in d.columns:
                    d[c] = np.nan

            rows.append(
                d[
                    [
                        "image",
                        "method",
                        "midline_type",
                        "crack_type",
                        "cid",
                        "level",
                        "entity_id",
                        "reason",
                        "length",
                        "n_segments",
                        "overlap",
                        "branch_id",
                        "extra_info",
                        "source_csv",
                    ]
                ].copy()
            )

    if not rows:
        _log(verbose, "[summarize] no invalid/exclusion rows found")
        return outputs

    all_df = pd.concat(rows, ignore_index=True)
    all_df["reason"] = all_df["reason"].astype(str)
    all_df["method"] = all_df["method"].astype(str)
    all_df["level"] = all_df["level"].astype(str)

    out_csv = os.path.join(out_dir, "dataset_invalid_matches_all.csv")
    all_df.to_csv(out_csv, index=False)
    outputs["dataset_invalid_matches_all_csv"] = out_csv

    # Total failure reasons (counts).
    reason_counts = (
        all_df.groupby("reason", dropna=False).size().rename("count").reset_index()
        .sort_values("count", ascending=False)
    )
    if not reason_counts.empty:
        out_png = os.path.join(out_dir, "dataset_invalid_reason_counts_total.png")
        _save_bar(
            labels=reason_counts["reason"].astype(str).tolist(),
            values=reason_counts["count"].astype(float).tolist(),
            out_png=out_png,
            title="Invalid / Exclusion Reasons (Total Count)",
            ylabel="count",
            rotate=40,
        )
        outputs["dataset_invalid_reason_counts_total_png"] = out_png

    # Total failure reasons by clipped length.
    tmp_len = all_df.copy()
    tmp_len["length"] = pd.to_numeric(tmp_len["length"], errors="coerce")
    reason_len = (
        tmp_len[np.isfinite(tmp_len["length"]) & (tmp_len["length"] > 0)]
        .groupby("reason", dropna=False)["length"]
        .sum()
        .reset_index()
        .sort_values("length", ascending=False)
    )
    if not reason_len.empty:
        out_png = os.path.join(out_dir, "dataset_invalid_reason_length_total.png")
        _save_bar(
            labels=reason_len["reason"].astype(str).tolist(),
            values=reason_len["length"].astype(float).tolist(),
            out_png=out_png,
            title="Invalid / Exclusion Reasons (Total Length)",
            ylabel="length (px)",
            rotate=40,
        )
        outputs["dataset_invalid_reason_length_total_png"] = out_png

    # Stacked bar: reasons by method.
    pivot = (
        all_df.groupby(["method", "reason"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .pivot(index="method", columns="reason", values="count")
        .fillna(0.0)
    )
    if not pivot.empty:
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
        methods = pivot.index.astype(str).tolist()
        reasons = pivot.columns.astype(str).tolist()
        x = np.arange(len(methods), dtype=float)
        fig_w = max(8.0, 0.65 * len(methods))
        fig, ax = plt.subplots(figsize=(fig_w, 4.8), dpi=180)
        bottom = np.zeros(len(methods), dtype=float)
        cmap = plt.get_cmap("tab20")
        handles = []
        for j, reason in enumerate(reasons):
            vals = pivot[reason].to_numpy(float)
            color = cmap(j % 20)
            bars = ax.bar(x, vals, bottom=bottom, color=color, alpha=0.9, label=reason)
            bottom += vals
            if len(bars) > 0:
                handles.append(Patch(facecolor=color, edgecolor="none", label=str(reason)))
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("count")
        ax.set_title("Invalid / Exclusion Reasons by Method (Stacked)")
        ax.grid(axis="y", alpha=0.25)
        if handles:
            ax.legend(handles=handles, loc="best", fontsize=8, framealpha=0.9)
        plt.tight_layout()
        out_png = os.path.join(out_dir, "dataset_invalid_reason_counts_stacked_by_method.png")
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        outputs["dataset_invalid_reason_counts_stacked_by_method_png"] = out_png

    _log(verbose, f"[summarize] invalid/exclusion rows aggregated: {len(all_df)}")
    return outputs


def _plot_multicue_ablation(
    df_width: Optional[pd.DataFrame],
    df_midline: Optional[pd.DataFrame],
    out_dir: str,
    *,
    verbose: bool = False,
) -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    if df_width is None or df_width.empty:
        return outputs

    multicue_methods = ["ET", "dt", "dt_depth", "dt_ridge_valley", "dt_ridge_valley_depth", "dt_ridge_color_depth"]

    def _norm_method(v: str) -> str:
        s = str(v or "").strip()
        if s.startswith("best_"):
            s = s[len("best_") :].strip()
        if _is_et_like(s):
            return "ET"
        return s

    w = df_width.copy()
    if "midline_type" not in w.columns:
        return outputs
    w["method"] = w["midline_type"].map(_norm_method)
    w["crack_type_norm"] = w.get("crack_type", pd.Series([""] * len(w))).astype(str).str.lower()
    w = w[w["crack_type_norm"].str.contains("combined", na=False)].copy()
    w = w[w["method"].isin(multicue_methods)].copy()

    m = (df_midline.copy() if isinstance(df_midline, pd.DataFrame) else pd.DataFrame())
    if not m.empty:
        if "midline_type" in m.columns and "midline_type_path" in m.columns:
            mt_col = m["midline_type"].astype(str)
            mtp_col = m["midline_type_path"].astype(str)
            m["midline_type"] = np.where(mt_col.isin(["unknown", "", "nan"]), mtp_col, mt_col)
        if "midline_type" in m.columns:
            m["method"] = m["midline_type"].map(_norm_method)
            m = m[m["method"].isin(multicue_methods)].copy()
            if "crack_type" in m.columns:
                m = m[m["crack_type"].astype(str).str.lower().str.contains("combined", na=False)].copy()

    # Single no-ET view.
    ww = w[w["method"] != "ET"].copy() if not w.empty else w
    mm = m[m["method"] != "ET"].copy() if isinstance(m, pd.DataFrame) and not m.empty else m

    if ww.empty and (not isinstance(mm, pd.DataFrame) or mm.empty):
        return outputs

    present_in_width = set(ww["method"].astype(str)) if not ww.empty else set()
    present_in_midline = set(mm["method"].astype(str)) if isinstance(mm, pd.DataFrame) and not mm.empty else set()
    present = present_in_width | present_in_midline
    methods = [x for x in multicue_methods if x != "ET" and x in present]
    if not methods:
        return outputs
    if ww.empty:
        ww = pd.DataFrame(columns=w.columns)

    fig, axes = plt.subplots(2, 1, figsize=(max(9.0, 1.2 * len(methods)), 8.0), dpi=170, height_ratios=[1.2, 1.0])
    ax0, ax1 = axes

    # score_mid decomposition terms.
    score_terms = [
        ("nn_mean_bidirectional", "#e15759", "nn_mean", lambda v: np.log1p(np.maximum(v, 0.0))),
        ("hausdorff_max", "#b07aa1", "hausdorff_max", lambda v: 0.5 * np.log1p(np.maximum(v, 0.0))),
        ("coverage_min", "#f28e2b", "coverage_min", lambda v: 1.0 - np.clip(v, 0.0, 1.0)),
    ]
    comp_data = {}
    if isinstance(mm, pd.DataFrame) and not mm.empty:
        mm_c = mm.copy()
        for col, color, lbl, fn in score_terms:
            per_method = []
            for mth in methods:
                raw = pd.to_numeric(mm_c.loc[mm_c["method"] == mth, col], errors="coerce").to_numpy(float) if col in mm_c.columns else np.array([], dtype=float)
                raw = raw[np.isfinite(raw)]
                per_method.append(float(np.mean(fn(raw))) if raw.size else 0.0)
            comp_data[col] = np.asarray(per_method, float)

    bottoms = np.zeros(len(methods), dtype=float)
    for col, color, lbl, _ in score_terms:
        vals = comp_data.get(col, np.zeros(len(methods)))
        vals = np.where(np.isfinite(vals), vals, 0.0)
        ax0.bar(np.arange(len(methods)), vals, bottom=bottoms, color=color, alpha=0.85, label=lbl)
        bottoms += vals

    method_display = [_display_method_name(m) for m in methods]
    ax0.set_xticks(np.arange(len(methods)))
    ax0.set_xticklabels(method_display, rotation=25, ha="right")
    ax0.set_title("Multi-Cue Ablation - Mid Score decomposition\n(combined cracks, per-crack mean)")
    ax0.set_ylabel("Mid Score")
    ax0.legend(loc="best", fontsize=8, framealpha=0.9)
    ax0.grid(axis="y", alpha=0.2)

    metric_sources = []
    for mcol in ("nn_mean_bidirectional", "hausdorff_max", "coverage_min", "precision_tau", "recall_tau", "f1_tau", "mean_tan_angle_error_deg"):
        if isinstance(mm, pd.DataFrame) and not mm.empty and mcol in mm.columns:
            metric_sources.append((mcol, mm, mcol))
    metric_sources = metric_sources[:8]

    if metric_sources:
        x = np.arange(len(metric_sources), dtype=float)
        bar_w = 0.8 / max(1, len(methods))
        tab_colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948"]
        for i, method in enumerate(methods):
            means = []
            lo_errs = []
            hi_errs = []
            for _, df_src, col in metric_sources:
                sub = df_src[df_src["method"] == method]
                v = pd.to_numeric(sub[col], errors="coerce").to_numpy(float)
                v = v[np.isfinite(v)]
                if v.size == 0:
                    means.append(np.nan)
                    lo_errs.append(0.0)
                    hi_errs.append(0.0)
                else:
                    q1 = float(np.percentile(v, 25))
                    q3 = float(np.percentile(v, 75))
                    mn = float(np.mean(v))
                    means.append(mn)
                    lo_errs.append(max(0.0, mn - q1))
                    hi_errs.append(max(0.0, q3 - mn))
            means_arr = np.asarray(means, float)
            lo_arr = np.asarray(lo_errs, float)
            hi_arr = np.asarray(hi_errs, float)
            xpos = x - 0.4 + (i + 0.5) * bar_w
            color = tab_colors[i % len(tab_colors)]
            valid = np.isfinite(means_arr)
            ax1.bar(
                xpos[valid],
                means_arr[valid],
                width=bar_w * 0.9,
                color=color,
                alpha=0.85,
                label=_display_method_name(method),
            )
            ax1.errorbar(xpos[valid], means_arr[valid], yerr=np.vstack([lo_arr[valid], hi_arr[valid]]), fmt="none", ecolor="black", elinewidth=0.8, capsize=2)

        ax1.set_xticks(x)
        ax1.set_xticklabels([ms[0] for ms in metric_sources], rotation=25, ha="right", fontsize=8)
        ax1.set_ylabel("mean +/- IQR")
        ax1.grid(axis="y", alpha=0.2)
        ax1.legend(loc="best", fontsize=7, ncol=2)

    plt.tight_layout()
    out_png = os.path.join(out_dir, "dataset_multicue_ablation.png")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    outputs["dataset_multicue_ablation_png"] = out_png

    _log(verbose, f"[summarize] multicue ablation outputs: {list(outputs.keys())}")
    return outputs

def _plot_gt_supervision_timing_detail(out_dir: str, *, verbose: bool = False) -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    df_sup = _safe_read_csv(os.path.join(out_dir, "dataset_gt_supervision_timings.csv"))
    df_center = _safe_read_csv(os.path.join(out_dir, "dataset_gt_centering_timing_all.csv"))
    df_multi = _safe_read_csv(os.path.join(out_dir, "dataset_multi_cue_timing_all.csv"))
    if df_sup is None and df_center is None and df_multi is None:
        return outputs

    if df_center is not None and not df_center.empty:
        if "image" in df_center.columns and "atomic_centering_sec" in df_center.columns and "combined_centering_sec" in df_center.columns:
            d = df_center.copy()
            d["image"] = d["image"].astype(str)
            a = pd.to_numeric(d["atomic_centering_sec"], errors="coerce").fillna(0.0)
            c = pd.to_numeric(d["combined_centering_sec"], errors="coerce").fillna(0.0)
            x = np.arange(len(d), dtype=float)
            fig_w = max(9.0, 0.35 * len(d))
            fig, ax = plt.subplots(figsize=(fig_w, 4.2), dpi=170)
            ax.bar(x, a.to_numpy(float), color="#4c78a8", label="atomic_centering_sec")
            ax.bar(x, c.to_numpy(float), bottom=a.to_numpy(float), color="#f28e2b", label="combined_centering_sec")
            ax.set_xticks(x)
            ax.set_xticklabels(d["image"].tolist(), rotation=35, ha="right", fontsize=8)
            ax.set_ylabel("seconds")
            ax.set_title("GT centering time per image")
            ax.grid(axis="y", alpha=0.2)
            ax.legend(loc="best", fontsize=8)
            plt.tight_layout()
            out_png = os.path.join(out_dir, "dataset_timing_gt_centering.png")
            fig.savefig(out_png, bbox_inches="tight")
            plt.close(fig)
            outputs["dataset_timing_gt_centering_png"] = out_png

    if df_multi is not None and not df_multi.empty:
        _mc_wanted = (
            "dt_compute_s",
            "normals_depth_s",
            "multi_cue_align_s",
            "depth_recess_s",
            "multi_cue_costmap_s",
            "multi_cue_dijkstra_s",
            "multi_cue_postprocess_s",
        )
        method_cols = [
            c
            for c in df_multi.columns
            if any(
                w in str(c).lower()
                for w in (
                    "costmap",
                    "dijkstra",
                    "dt_compute",
                    "normals_depth",
                    "multi_cue_align",
                    "depth_recess",
                    "postprocess",
                )
            )
        ]
        method_cols = [c for c in _mc_wanted if c in df_multi.columns] + [c for c in method_cols if c not in _mc_wanted]
        if method_cols:
            sums = []
            labels = []
            for c in method_cols:
                vals = pd.to_numeric(df_multi[c], errors="coerce").to_numpy(float)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    _mc_label_map = {
                        "dt_compute_s": "DT",
                        "normals_depth_s": "normals",
                        "multi_cue_align_s": "align",
                        "depth_recess_s": "recess",
                        "multi_cue_costmap_s": "costmap",
                        "multi_cue_dijkstra_s": "dijkstra",
                        "multi_cue_postprocess_s": "postprocess",
                    }
                    labels.append(_mc_label_map.get(c, re.sub(r"(_s|_sec)$", "", str(c))))
                    sums.append(float(np.sum(vals)))
            if sums:
                fig, ax = plt.subplots(figsize=(max(8.0, 0.45 * len(labels)), 4.2), dpi=170)
                ax.bar(np.arange(len(labels)), sums, color="#59a14f", alpha=0.88)
                ax.set_xticks(np.arange(len(labels)))
                ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
                ax.set_ylabel("total seconds")
                ax.set_title("Per multi-cue method total time")
                ax.grid(axis="y", alpha=0.2)
                plt.tight_layout()
                out_png = os.path.join(out_dir, "dataset_timing_multicue_per_method.png")
                fig.savefig(out_png, bbox_inches="tight")
                plt.close(fig)
                outputs["dataset_timing_multicue_per_method_png"] = out_png

    if df_center is not None and not df_center.empty:
        gt_center_cols = [
            "dt_compute_s",
            "centered_snap_s",
            "normals_centered_s",
            "atomic_centering_sec",
            "combined_centering_sec",
        ]
        labels = []
        vals = []
        for c in gt_center_cols:
            if c not in df_center.columns:
                continue
            arr = pd.to_numeric(df_center[c], errors="coerce").to_numpy(float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            labels.append(re.sub(r"(_s|_sec)$", "", c))
            vals.append(float(np.mean(arr)))
        if labels:
            fig, ax = plt.subplots(figsize=(max(7.0, 0.7 * len(labels)), 4.2), dpi=170)
            x = np.arange(len(labels))
            ax.bar(x, vals, color="#1f77b4", alpha=0.88)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel("seconds (mean per image)")
            ax.set_title("GT Centering Components")
            ax.grid(axis="y", alpha=0.2)
            plt.tight_layout()
            out_png = os.path.join(out_dir, "dataset_timing_gt_centering_components.png")
            fig.savefig(out_png, bbox_inches="tight")
            plt.close(fig)
            outputs["dataset_timing_gt_centering_components_png"] = out_png

    _log(verbose, f"[summarize] gt timing detail outputs: {list(outputs.keys())}")
    return outputs


def _aggregate_calibration_ablation(
    save_folder: str,
    out_dir: str,
    *,
    evaluated_images: Optional[set] = None,
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Aggregates calibration ablation results from metrics/_calibration/.

    Reads:
      - ablation_rs3_weighted_summary.csv   (score_mid + component columns per method)
      - dt_best_et_*.csv                    (timing per image for dt_best and ET)

    Writes to out_dir/ablation/:
      - ablation_rs3_score_decomposition.png   (edge-family-style stacked bar)
      - ablation_timings.png                   (mean timing bar per method)
      - ablation_rs3_weighted_summary_all.csv
    """
    outputs: Dict[str, str] = {}
    calib_dir = os.path.join(save_folder, "metrics", "_calibration")
    if not os.path.isdir(calib_dir):
        _log(verbose, f"[ablation] calibration dir not found: {calib_dir}")
        return outputs

    abl_dir = os.path.join(out_dir, "ablation")
    os.makedirs(abl_dir, exist_ok=True)

    # 1) Load ablation score summary.
    abl_csv = os.path.join(calib_dir, "ablation_rs3_weighted_summary.csv")
    abl_df = _safe_read_csv(abl_csv)
    if abl_df is None or abl_df.empty:
        _log(verbose, "[ablation] ablation_rs3_weighted_summary.csv not found or empty")
        return outputs

    import shutil
    abl_copy_csv = os.path.join(abl_dir, "ablation_rs3_weighted_summary_all.csv")
    shutil.copy2(abl_csv, abl_copy_csv)
    outputs["ablation_summary_csv"] = abl_copy_csv

    method_col = None
    for c in ("method", "name", "variant", "label"):
        if c in abl_df.columns:
            method_col = c
            break
    if method_col is None:
        for c in abl_df.columns:
            if not pd.api.types.is_numeric_dtype(abl_df[c]):
                method_col = c
                break
    if method_col is None:
        _log(verbose, "[ablation] cannot identify method column in ablation CSV")
        return outputs

    total_col = next(
        (c for c in abl_df.columns if "score_mid" in c.lower() and "wmean" in c.lower()),
        next((c for c in abl_df.columns if "score_mid" in c.lower()), None),
    )
    score_terms = [
        ("comp_nn_wmean", "#e15759", "nn_mean"),
        ("comp_hausdorff_wmean", "#b07aa1", "hausdorff_max"),
        ("comp_coverage_wmean", "#f28e2b", "coverage_min"),
    ]
    has_decomp = all(col in abl_df.columns for col, _, _ in score_terms)
    if total_col is None and not has_decomp:
        _log(verbose, "[ablation] no score columns found in ablation CSV")
        return outputs

    if total_col and total_col in abl_df.columns:
        abl_df = abl_df.copy()
        abl_df["_sort_score"] = pd.to_numeric(abl_df[total_col], errors="coerce")
        abl_df = abl_df.sort_values("_sort_score", ascending=True).drop(columns=["_sort_score"]).reset_index(drop=True)

    methods = abl_df[method_col].astype(str).tolist()
    total_vals = (
        pd.to_numeric(abl_df[total_col], errors="coerce").to_numpy(float)
        if total_col else np.zeros(len(methods), dtype=float)
    )

    fig_w = max(10.0, 0.65 * len(methods))
    fig, ax = plt.subplots(figsize=(fig_w, 5.2), dpi=180)
    x = np.arange(len(methods), dtype=float)
    if has_decomp:
        bottoms = np.zeros(len(methods), dtype=float)
        legend_handles = []
        for col, color, lbl in score_terms:
            raw = pd.to_numeric(abl_df[col], errors="coerce").to_numpy(float)
            vals = np.where(np.isfinite(raw), raw, 0.0)
            vals = np.where(np.isfinite(vals), vals, 0.0)
            ax.bar(x, vals, bottom=bottoms, color=color, alpha=0.85, label=lbl)
            legend_handles.append(Patch(facecolor=color, edgecolor="none", label=lbl))
            bottoms += vals
        if total_col:
            tv = np.where(np.isfinite(total_vals), total_vals, np.nan)
            ax.plot(x, tv, "o", color="black", markersize=5, zorder=5, label="RS3 score (wmean)")
            legend_handles.append(Patch(facecolor="black", edgecolor="none", label="RS3 score (wmean)"))
        ax.legend(handles=legend_handles, loc="best", framealpha=0.9, fontsize=8)
    else:
        tv = np.where(np.isfinite(total_vals), total_vals, 0.0)
        ax.bar(x, tv, color="#4c78a8", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([_display_method_name(m) for m in methods], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("RS3 Score")
    ax.set_title("Calibration Ablation - RS3 Score Decomposition\n(combined + orphan atomics, length-weighted)")
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    out_png = os.path.join(abl_dir, "ablation_rs3_score_decomposition.png")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    outputs["ablation_score_decomposition_png"] = out_png

    # 2) Load dt_best + ET timing CSVs.
    eval_set = {str(x) for x in (evaluated_images or set())}
    timing_rows = []
    timing_csvs = glob.glob(os.path.join(calib_dir, "dt_best_et*.csv"))
    timing_csvs += glob.glob(os.path.join(calib_dir, "dt_best*.csv"))
    timing_csvs = list(dict.fromkeys(timing_csvs))

    timing_display = {
        "dt_best_et_pass2b_edge_tracking": "ET",
    }
    timing_skip = {"dt_best_et_pass1", "dt_best_et_pass2c_width_eval"}
    for p in timing_csvs:
        df = _safe_read_csv(p)
        if df is None or df.empty:
            continue
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem in timing_skip:
            continue
        method_label = timing_display.get(stem, stem)
        img_col = next((c for c in ("image", "image_name", "stem") if c in df.columns), None)
        sec_col = next(
            (c for c in ("total_s", "total_sec", "seconds", "time_sec", "inference_seconds") if c in df.columns),
            None,
        )
        if sec_col is None:
            sec_col = next(
                (
                    c for c in df.columns
                    if pd.api.types.is_numeric_dtype(df[c]) and (c.endswith("_s") or c.endswith("_sec"))
                ),
                None,
            )
        if sec_col is None:
            continue
        d = df.copy()
        if img_col:
            d["image"] = d[img_col].astype(str).apply(lambda s: os.path.splitext(os.path.basename(s))[0])
            if eval_set:
                d = d[d["image"].isin(eval_set)].copy()
        arr = pd.to_numeric(d[sec_col], errors="coerce").to_numpy(float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        timing_rows.append(
            {
                "method": method_label,
                "mean_sec": float(np.mean(arr)),
                "std_sec": float(np.std(arr)),
                "n": int(arr.size),
            }
        )

    if timing_rows:
        tim_df = pd.DataFrame(timing_rows).sort_values("mean_sec")
        fig_w2 = max(7.0, 0.65 * len(tim_df))
        fig2, ax2 = plt.subplots(figsize=(fig_w2, 4.2), dpi=180)
        x2 = np.arange(len(tim_df), dtype=float)
        y2 = tim_df["mean_sec"].to_numpy(float)
        e2 = tim_df["std_sec"].to_numpy(float)
        ax2.bar(x2, y2, color="#4c78a8", alpha=0.85)
        ax2.errorbar(x2, y2, yerr=e2, fmt="none", ecolor="black", elinewidth=1.2, capsize=3)
        ax2.set_xticks(x2)
        ax2.set_xticklabels(tim_df["method"].astype(str).tolist(), rotation=35, ha="right", fontsize=8)
        ax2.set_ylabel("mean seconds")
        ax2.set_title("Calibration Ablation - Timing (mean +/- std per image)")
        ax2.grid(axis="y", alpha=0.2)
        plt.tight_layout()
        out_timing_png = os.path.join(abl_dir, "ablation_timings.png")
        fig2.savefig(out_timing_png, bbox_inches="tight")
        plt.close(fig2)
        outputs["ablation_timings_png"] = out_timing_png

        tim_csv = os.path.join(abl_dir, "ablation_timings_summary.csv")
        tim_df.to_csv(tim_csv, index=False)
        outputs["ablation_timings_csv"] = tim_csv

    _log(verbose, f"[ablation] outputs: {list(outputs.keys())}")
    return outputs


def summarize_dataset_metrics(
    save_folder: str,
    *,
    out_dir: Optional[str] = None,
    baseline_roots: Optional[List[str]] = None,
    depth_timing_csv: Optional[str] = None,
    image_filter: Optional[List[str]] = None,
    index: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Dataset-level metrics aggregation entrypoint.

    Aggregates and summarizes:
    - mask metrics
    - width metrics
    - midline metrics
    - timing metrics
    - width distribution summaries
    - edge-family and RS3-family calibration summaries
    - optional external baseline timing CSVs
    """
    # If `index` is supplied it overrides / merges with image_filter.
    if index is not None:
        index_str = str(index)
        if image_filter is None:
            image_filter = [index_str]
        elif index_str not in image_filter:
            image_filter = list(image_filter) + [index_str]

    metrics_root = os.path.join(save_folder, "metrics")
    if out_dir is None:
        out_dir = os.path.join(metrics_root, "_summary")
    # Hard reset summary dir each run.
    try:
        import shutil
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        _log(verbose, f"[summarize] reset summary dir: {out_dir}")
    except Exception as e:
        _log(verbose, f"[summarize] failed to reset summary dir: {e}")
        os.makedirs(out_dir, exist_ok=True)

    # Clean up stale debug artifacts
    import glob as _glob
    for _stale in _glob.glob(os.path.join(metrics_root, "**", "et_ref.csv"), recursive=True):
        try:
            os.remove(_stale)
            _log(verbose, f"[summarize] removed stale et_ref.csv: {_stale}")
        except Exception:
            pass

    image_dirs = _list_image_metric_dirs(metrics_root)
    if image_filter is not None:
        filter_set = {str(f) for f in image_filter}
        image_dirs = [d for d in image_dirs if os.path.basename(d) in filter_set]
        _log(verbose, f"[summarize] image_filter active: {len(image_dirs)} image(s) kept from filter={filter_set}")
    evaluated_images_set = {os.path.basename(p) for p in image_dirs}
    report: Dict[str, object] = {
        "metrics_root": metrics_root,
        "summary_dir": out_dir,
        "n_image_dirs": int(len(image_dirs)),
        "outputs": {},
    }

    if not image_dirs:
        _log(verbose, "[summarize] no metrics image directories found")
        return report

    outputs: Dict[str, str] = {}

    supervision_root = os.path.join(save_folder, "supervision")
    parallel_tasks = {
        "mask": lambda: _aggregate_mask_metrics(image_dirs, out_dir, verbose=verbose),
        "width": lambda: _aggregate_width_metrics(image_dirs, out_dir, verbose=verbose),
        "midline": lambda: _aggregate_midline_metrics(image_dirs, out_dir, verbose=verbose),
        "invalid": lambda: _aggregate_invalid_matches(image_dirs, out_dir, verbose=verbose),
        "timing": lambda: _aggregate_timing_metrics(image_dirs, out_dir, verbose=verbose),
        "width_dist": lambda: _aggregate_width_distribution(image_dirs, out_dir, verbose=verbose),
        "edge_rs3": lambda: _aggregate_edge_rs3_selection(image_dirs, out_dir, verbose=verbose),
        "sup_timing": lambda: _aggregate_supervision_timings(
            supervision_root=supervision_root,
            out_dir=out_dir,
            evaluated_images=evaluated_images_set,
            verbose=verbose,
        ),
        "gt_component": lambda: _aggregate_gt_component_timings(
            supervision_root=supervision_root,
            out_dir=out_dir,
            evaluated_images=evaluated_images_set,
            verbose=verbose,
        ),
        "depth_timing": lambda: _aggregate_depth_generation_timings(
            out_dir=out_dir,
            depth_timing_csv=depth_timing_csv,
            evaluated_images=evaluated_images_set,
            verbose=verbose,
        ),
        "baseline_timing": lambda: _aggregate_baseline_timings(
            out_dir,
            baseline_roots=baseline_roots,
            evaluated_images=evaluated_images_set,
            verbose=verbose,
        ),
    }
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(fn): name for name, fn in parallel_tasks.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                res = fut.result()
                if isinstance(res, dict):
                    outputs.update(res)
            except Exception as e:
                _log(verbose, f"[summarize] warning: {name} aggregation failed: {e}")

    outputs.update(
        _aggregate_gt_centering_weighted_summaries(
            save_folder=save_folder,
            out_dir=out_dir,
            evaluated_images=evaluated_images_set,
            verbose=verbose,
        )
    )
    outputs.update(
        _aggregate_calibration_ablation(
            save_folder=save_folder,
            out_dir=out_dir,
            evaluated_images=evaluated_images_set,
            verbose=verbose,
        )
    )
    # Edge-tracking variant timing breakdown from timings_core (if timing columns exist).
    _et_variant_rows = []
    for _img_dir in image_dirs:
        _tc = _safe_read_csv(os.path.join(_img_dir, "timings_core.csv"))
        if _tc is None or _tc.empty:
            continue
        _tc = _tc.copy()
        _tc["image"] = os.path.basename(_img_dir)
        _et_variant_rows.append(_tc)
    if _et_variant_rows:
        _tc_all = pd.concat(_et_variant_rows, ignore_index=True)
        _tc_all_csv = os.path.join(out_dir, "dataset_edge_variant_timings_all.csv")
        _tc_all.to_csv(_tc_all_csv, index=False)
        outputs["edge_variant_timings_all_csv"] = _tc_all_csv
        _grp_cols = [c for c in ("supervision", "algo_variant", "crack_type") if c in _tc_all.columns]
        _time_col = next((c for c in ("elapsed_s", "total_s", "edge_total_s") if c in _tc_all.columns), None)
        if _grp_cols and _time_col:
            _tc_grp = (
                _tc_all.groupby(_grp_cols)[_time_col]
                .agg(mean_s="mean", std_s="std", n="count")
                .reset_index()
            )
            _tc_grp["label"] = _tc_grp.apply(
                lambda r: f"{r.get('supervision', '')}/{r.get('algo_variant', '')}".strip("/"),
                axis=1,
            )
            _tc_grp = _tc_grp.sort_values("mean_s", ascending=False)
            if not _tc_grp.empty and _tc_grp["mean_s"].notna().any():
                _fig, _ax = plt.subplots(figsize=(max(7.0, 0.6 * len(_tc_grp)), 4.2), dpi=180)
                _x = np.arange(len(_tc_grp))
                _y = _tc_grp["mean_s"].to_numpy(float)
                _e = _tc_grp["std_s"].fillna(0.0).to_numpy(float)
                _ax.bar(_x, _y, color="#4c78a8", alpha=0.85)
                _ax.errorbar(_x, _y, yerr=_e, fmt="none", ecolor="black", elinewidth=1.2, capsize=3)
                _ax.set_xticks(_x)
                _ax.set_xticklabels(_tc_grp["label"].tolist(), rotation=35, ha="right", fontsize=8)
                _ax.set_ylabel("mean seconds")
                _ax.set_title("Edge-tracking variant timings (mean +/- std per image)")
                _ax.grid(axis="y", alpha=0.2)
                plt.tight_layout()
                _vt_png = os.path.join(out_dir, "dataset_edge_variant_timings.png")
                _fig.savefig(_vt_png, bbox_inches="tight")
                plt.close(_fig)
                outputs["edge_variant_timings_png"] = _vt_png
    outputs.update(
        _plot_dataset_full_timing_overview(
            out_dir=out_dir,
            verbose=verbose,
        )
    )
    df_width_all = _safe_read_csv(os.path.join(out_dir, "width", "overview", "dataset_width_summary_all.csv"))
    df_midline_all = _safe_read_csv(os.path.join(out_dir, "midline", "dataset_midline_metrics_all.csv"))

    # Supplement midline with ablation per-image rows so dt_depth/etc. appear in multicue plot.
    _abl_all = _safe_read_csv(os.path.join(save_folder, "metrics", "_calibration", "ablation_rs3_all_rows.csv"))
    if _abl_all is not None and not _abl_all.empty and "variant_id" in _abl_all.columns:
        _abl_mid = _abl_all.rename(
            columns={
                "variant_id": "midline_type",
                "lwmean_score_mid": "score_mid",
                "lwmean_nn_mean_bidirectional": "nn_mean_bidirectional",
                "lwmean_hausdorff_max": "hausdorff_max",
                "lwmean_coverage_min": "coverage_min",
                "lwmean_mean_tan_angle_error_deg": "mean_tan_angle_error_deg",
                "lwmean_precision_tau": "precision_tau",
                "lwmean_recall_tau": "recall_tau",
                "lwmean_f1_tau": "f1_tau",
            }
        )
        _abl_mid["crack_type"] = "combined_plus_noncombined_atomic"
        _abl_mid["method_family"] = "model"
        _abl_mid["baseline_method"] = ""
        if df_midline_all is not None and not df_midline_all.empty:
            df_midline_all = pd.concat([df_midline_all, _abl_mid], ignore_index=True)
        else:
            df_midline_all = _abl_mid

    outputs.update(_plot_multicue_ablation(df_width_all, df_midline_all, out_dir, verbose=verbose))

    # Width stratified by GT width bin - sample-level length-weighted
    # Each row in diffs CSV = ~1px arc length, so raw aggregation is length-weighted.
    try:
        _strat_all = _load_diffs_frames(image_dirs)
        if _strat_all is not None and not _strat_all.empty:
            # display labels
            _strat_all["_method_label"] = _strat_all["_method"].astype(str).apply(
                lambda s: _display_width_method_label(s) if s not in ("dt", "best_dt_depth", "ET")
                else _display_method_name(s)
            )

            _bins       = [0.0, 6.0, 12.0, 20.0, np.inf]
            _bin_labels = ["<6px", "6-12px", "12-20px", ">20px"]
            _strat_all["gt_width_bin"] = pd.cut(
                _strat_all["gt_width_px"],
                bins=_bins,
                labels=_bin_labels,
                include_lowest=True,
                right=False,
            )

            _strat_rows = []
            for (method, wbin), g in _strat_all.groupby(
                ["_method_label", "gt_width_bin"], observed=True
            ):
                d_arr = g["diff_px"].to_numpy(float)
                d_arr = d_arr[np.isfinite(d_arr)]
                if d_arr.size == 0:
                    continue
                _strat_rows.append({
                    "method":       str(method),
                    "gt_width_bin": str(wbin),
                    "n_samples":    int(d_arr.size),
                    "mae_px":       round(float(np.mean(np.abs(d_arr))),  3),
                    "rmse_px":      round(float(np.sqrt(np.mean(d_arr**2))), 3),
                    "bias_px":      round(float(np.mean(d_arr)),           3),
                })

            if _strat_rows:
                _strat_df = pd.DataFrame(_strat_rows)
                _width_out_dir = os.path.join(out_dir, "width", "stratified")
                os.makedirs(_width_out_dir, exist_ok=True)
                _strat_csv = os.path.join(_width_out_dir, "dataset_width_stratified_by_gt_width.csv")
                _strat_df.to_csv(_strat_csv, index=False)
                outputs["width_stratified_csv"] = _strat_csv
                _log(verbose, f"[stratified] wrote {len(_strat_rows)} rows -> {_strat_csv}")

                _methods_order = ["dt", "dt_depth", "ET", "MAT (DSE)", "MAT (raw)", "EOB", "ESD", "PCA"]
                _methods_present = [m for m in _methods_order if m in _strat_df["method"].values]
                _methods_present += [m for m in _strat_df["method"].unique() if m not in _methods_present]
                _bins_order = [b for b in _bin_labels if b in _strat_df["gt_width_bin"].values]
                _x = np.arange(len(_bins_order), dtype=float)
                _bar_w = 0.8 / max(1, len(_methods_present))
                _strat_method_colors = {
                    "dt": "#1f77b4",
                    "dt_depth": "#1f77b4",
                    "ET": "#d62728",
                    "MAT (DSE)": "#2ca02c",
                    "MAT (raw)": "#2ca02c",
                    "PCA": "#2ca02c",
                    "ESD": "#2ca02c",
                    "EOB": "#2ca02c",
                }

                for _metric, _col, _ylabel in [
                    ("MAE",    "mae_px",      "MAE (px)"),
                    ("RMSE",   "rmse_px",     "RMSE (px)"),
                    ("bias",   "bias_px",     "bias (px) - signed"),
                ]:
                    try:
                        fig, ax = plt.subplots(
                            figsize=(max(8.0, 1.5 * len(_bins_order)), 4.8), dpi=170
                        )
                        for mi, method in enumerate(_methods_present):
                            sub = _strat_df[_strat_df["method"] == method]
                            vals = [
                                float(sub.loc[sub["gt_width_bin"] == b, _col].iloc[0])
                                if len(sub[sub["gt_width_bin"] == b]) > 0 else np.nan
                                for b in _bins_order
                            ]
                            xpos = _x - 0.4 + (mi + 0.5) * _bar_w
                            _col_c = _strat_method_colors.get(str(method), "#888888")
                            _alpha = max(0.4, 1.0 - mi * 0.08)
                            ax.bar(xpos, vals, width=_bar_w, color=_col_c,
                                   alpha=_alpha, label=method)
                        ax.set_xticks(_x)
                        ax.set_xticklabels(_bins_order)
                        ax.set_xlabel("GT crack width bin")
                        ax.set_ylabel(_ylabel)
                        ax.set_title(
                            f"Width {_metric} by GT Width Bin\n"
                            f"(length-weighted, n_samples per bin)"
                        )
                        ax.legend(fontsize=7, framealpha=0.9, ncol=2)
                        ax.grid(axis="y", alpha=0.2)
                        for bi, b in enumerate(_bins_order):
                            _ns = int(_strat_df[_strat_df["gt_width_bin"] == b]["n_samples"].sum())
                            ax.text(bi, ax.get_ylim()[1] * 0.97,
                                    f"n={_ns:,}", ha="center", va="top",
                                    fontsize=6, color="#555555")
                        plt.tight_layout()
                        _tag = _metric.lower().replace("|", "").strip()
                        _png = os.path.join(
                            _width_out_dir,
                            f"dataset_width_stratified_{_tag}.png",
                        )
                        fig.savefig(_png, bbox_inches="tight")
                        plt.close(fig)
                        outputs[f"width_stratified_{_tag}_png"] = _png
                    except Exception as _ep:
                        _log(verbose, f"[stratified plot {_metric}] failed: {_ep}")
            else:
                _log(verbose, "[stratified] no rows produced after groupby")
        else:
            _log(verbose, "[stratified] no diffs frames loaded")
    except Exception as _e_strat:
        import traceback
        _log(verbose, f"[summarize] width stratified-by-gt failed: {_e_strat}")
        traceback.print_exc()
    outputs.update(_plot_gt_supervision_timing_detail(out_dir, verbose=verbose))

    # Organize timing-related summary artifacts under a dedicated subfolder.
    timing_dir = os.path.join(out_dir, "timing")
    os.makedirs(timing_dir, exist_ok=True)

    def _is_timing_output(k: str, p: str) -> bool:
        kk = str(k).lower()
        pp = str(p).lower()
        base = os.path.basename(pp)
        if "timing" in kk or "timings" in kk:
            return True
        if "timing" in base or "timings" in base:
            return True
        if kk in {
            "seg_chart",
            "width_chart",
            "overview_chart",
            "gt_centering_components_chart",
            "multi_cue_components_chart",
            "gt_centering_atomic_vs_combined_chart",
            "multi_cue_atomic_vs_combined_chart",
        }:
            return True
        return False

    moved_outputs: Dict[str, str] = {}
    for k, v in outputs.items():
        if not isinstance(v, str):
            continue
        if not os.path.exists(v):
            continue
        try:
            src_abs_pre = os.path.abspath(v)
            if src_abs_pre.startswith(os.path.abspath(os.path.join(out_dir, "ablation")) + os.sep):
                continue
        except Exception:
            pass
        if not _is_timing_output(k, v):
            continue
        try:
            src_abs = os.path.abspath(v)
            out_abs = os.path.abspath(out_dir)
            if not src_abs.startswith(out_abs + os.sep):
                continue
            dst = os.path.join(timing_dir, os.path.basename(src_abs))
            if os.path.abspath(dst) == src_abs:
                continue
            if os.path.exists(dst):
                os.remove(dst)
            import shutil
            shutil.move(src_abs, dst)
            moved_outputs[k] = dst
        except Exception as e:
            _log(verbose, f"[summarize] timing move failed for {k}: {e}")

    outputs.update(moved_outputs)
    report["outputs"] = outputs

    report_json = os.path.join(out_dir, "dataset_summary_report.json")
    try:
        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        outputs["dataset_summary_report_json"] = report_json
    except Exception as e:
        _log(verbose, f"[summarize] failed to write dataset_summary_report.json: {e}")

    _log(verbose, f"[summarize] wrote dataset summary artifacts to {out_dir}")
    return report
