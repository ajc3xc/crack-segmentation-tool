import os
import glob
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.ioff()


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg)


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
    plt.ylabel(ylabel)
    plt.title(title)
    if color_legend:
        handles = [Patch(facecolor=c, edgecolor="none", label=str(lbl)) for lbl, c in color_legend]
        plt.legend(handles=handles, loc="best", framealpha=0.9, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


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

    if len(parts) >= 2 and parts[0] in ("manual", "auto"):
        midline_type = parts[0]
    elif len(parts) >= 3 and parts[1] in ("manual", "auto"):
        baseline_method = str(parts[0])
        method_family = "baseline"
        midline_type = parts[1]
    elif parts:
        midline_type = parts[0]

    return midline_type, method_family, baseline_method


def _parse_midline_context(rel_path: str) -> Tuple[str, str, str]:
    parts = rel_path.replace("\\", "/").split("/")
    baseline_method = ""
    method_family = "model"
    midline_type = "unknown"

    if len(parts) >= 4 and parts[1] == "midline_metrics":
        midline_type = parts[0]
    elif len(parts) >= 5 and parts[2] == "midline_metrics":
        baseline_method = str(parts[0])
        method_family = "baseline"
        midline_type = parts[1]
    elif parts:
        midline_type = parts[0]

    return midline_type, method_family, baseline_method


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
    all_csv = os.path.join(out_dir, "dataset_mask_metrics_all.csv")
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
        triplet_dir = os.path.join(out_dir, "mask_triplets")
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
            triplet_csv = os.path.join(out_dir, "dataset_mask_triplet_weighted_summary.csv")
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
        grp_csv = os.path.join(out_dir, "dataset_mask_metrics_grouped.csv")
        grouped.to_csv(grp_csv, index=False)
        outputs["mask_grouped_csv"] = grp_csv

        total = grouped[grouped["crack_type"].astype(str).str.upper() == "TOTAL"].copy()
        if not total.empty:
            iou_col = next((c for c in ["iou_mean", "iou_manual_vs_gt_mean", "iou_auto_vs_gt_mean"] if c in total.columns), None)
            bf1_col = "boundary_f1_mean" if "boundary_f1_mean" in total.columns else None
            def _pretty_variant(v: str) -> str:
                s = str(v or "").strip()
                sl = s.lower()
                if sl.startswith("manual:"):
                    return "manual"
                if sl.startswith("auto:"):
                    return "auto"
                if sl.startswith("baseline:"):
                    # Show baseline method name only (e.g., sam3, hrsegnet).
                    p = s.split(":", 1)
                    return p[1] if len(p) == 2 and p[1] else "baseline"
                return s

            def _variant_class(v: str) -> str:
                sl = str(v or "").strip().lower()
                if sl.startswith("manual:"):
                    return "manual"
                if sl.startswith("auto:"):
                    return "auto"
                if sl.startswith("baseline:"):
                    return "baseline"
                return "other"

            var_labels = [_pretty_variant(v) for v in total["variant"].astype(str).tolist()]
            cls_vals = [_variant_class(v) for v in total["variant"].astype(str).tolist()]
            cls_color = {
                "manual": "#1f77b4",
                "auto": "#ff7f0e",
                "baseline": "#2ca02c",
                "other": "#7f7f7f",
            }
            var_colors = [cls_color.get(c, "#7f7f7f") for c in cls_vals]
            legend_items = [
                ("manual", cls_color["manual"]),
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
                out_png = os.path.join(out_dir, "dataset_mask_total_iou_by_variant.png")
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
                out_png = os.path.join(out_dir, "dataset_mask_total_boundary_f1_by_variant.png")
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

                picks = [
                    ("manual", _pick_row("manual:")),
                    ("sam3", _pick_row("baseline:sam3")),
                    ("hrsegnet", _pick_row("baseline:hrsegnet")),
                ]
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
                    cmap = {"manual": "#4c78a8", "sam3": "#f28e2b", "hrsegnet": "#2ca02c"}
                    bar_cols = [cmap.get(x, "#777777") for x in labels]
                    x = np.arange(len(labels), dtype=float)
                    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2), dpi=180, sharex=True)
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
                    fig.suptitle("Dataset Mask Error Comparison: manual vs sam3 vs hrsegnet", fontsize=11, fontweight="bold")
                    plt.tight_layout()
                    out_png = os.path.join(out_dir, "dataset_mask_assd_hd95_manual_sam3_hrsegnet.png")
                    fig.savefig(out_png, bbox_inches="tight")
                    plt.close(fig)
                    outputs["mask_assd_hd95_compare_png"] = out_png

    _log(verbose, f"[summarize] mask metrics rows={len(all_df)}")
    return outputs


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
    frames = []
    COLOR_MAP = {
        "baseline": "#2ca02c",  # green
        "auto": "#1f77b4",      # blue
        "manual": "#d62728",    # red
    }

    def _plot_metric_bar_iqr_outliers(
        *,
        df_plot: pd.DataFrame,
        metric_col: str,
        out_png: str,
        title: str,
        max_methods: int = 20,
    ) -> None:
        if metric_col not in df_plot.columns:
            return
        local_df = df_plot.copy()
        if "method_name" not in local_df.columns:
            if {"method_family", "baseline_method", "midline_type"}.issubset(local_df.columns):
                mf = local_df["method_family"].astype(str)
                bm = local_df["baseline_method"].astype(str)
                mt = local_df["midline_type"].astype(str)
                local_df["method_name"] = np.where(
                    mf.str.lower().eq("baseline"),
                    bm.where(bm.str.len() > 0, "baseline"),
                    mt.where(mt.str.len() > 0, "model"),
                )
            else:
                return
        rows = []
        for m, g in local_df.groupby("method_name", dropna=False):
            v = pd.to_numeric(g[metric_col], errors="coerce").to_numpy(float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            mf0 = str(g.get("method_family", pd.Series([""])).astype(str).iloc[0]).lower() if "method_family" in g.columns else ""
            mt0 = str(g.get("midline_type", pd.Series([""])).astype(str).iloc[0]).lower() if "midline_type" in g.columns else ""
            if mf0 == "baseline":
                source_class = "baseline"
            elif mt0 == "manual":
                source_class = "manual"
            else:
                source_class = "auto"
            q1 = float(np.percentile(v, 25))
            q3 = float(np.percentile(v, 75))
            iqr = float(q3 - q1)
            lo = float(q1 - 1.5 * iqr)
            hi = float(q3 + 1.5 * iqr)
            out = v[(v < lo) | (v > hi)]
            rows.append(
                {
                    "method_name": str(m),
                    "mean": float(np.mean(v)),
                    "q1": q1,
                    "q3": q3,
                    "outliers": out,
                    "n": int(v.size),
                    "source_class": source_class,
                }
            )
        if not rows:
            return
        d = pd.DataFrame(rows).sort_values("mean", ascending=True).head(int(max_methods))
        fig_w = max(8.0, 0.55 * len(d))
        fig, ax = plt.subplots(figsize=(fig_w, 4.8), dpi=180)
        xs = np.arange(len(d))
        means = d["mean"].to_numpy(float)
        bar_colors = [COLOR_MAP.get(str(c), "#4c78a8") for c in d["source_class"].astype(str).tolist()]
        ax.bar(xs, means, color=bar_colors, alpha=0.85, label="mean")
        lo_err = np.clip(means - d["q1"].to_numpy(float), 0.0, None)
        hi_err = np.clip(d["q3"].to_numpy(float) - means, 0.0, None)
        ax.errorbar(xs, means, yerr=np.vstack([lo_err, hi_err]), fmt="none", ecolor="#f58518", elinewidth=1.3, capsize=3, label="IQR")
        for i, arr in enumerate(d["outliers"].tolist()):
            if arr is None or len(arr) == 0:
                continue
            yy = np.asarray(arr, float)
            xx = np.full(len(yy), float(xs[i])) + np.linspace(-0.08, 0.08, len(yy))
            ax.scatter(xx, yy, s=10, color="#d62728", alpha=0.8, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels(d["method_name"].astype(str).tolist(), rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(metric_col)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        legend_handles = [
            Patch(facecolor=COLOR_MAP["manual"], edgecolor="none", label="manual"),
            Patch(facecolor=COLOR_MAP["auto"], edgecolor="none", label="auto"),
            Patch(facecolor=COLOR_MAP["baseline"], edgecolor="none", label="baseline"),
            Patch(facecolor="#f58518", edgecolor="none", label="IQR"),
        ]
        ax.legend(handles=legend_handles, loc="best", fontsize=8, framealpha=0.9)
        plt.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)

    def _plot_metric_box_by_method(
        *,
        df_plot: pd.DataFrame,
        metric_col: str,
        out_png: str,
        title: str,
        max_methods: int = 20,
    ) -> None:
        if metric_col not in df_plot.columns:
            return
        local_df = df_plot.copy()
        if "method_name" not in local_df.columns:
            if {"method_family", "baseline_method", "midline_type"}.issubset(local_df.columns):
                mf = local_df["method_family"].astype(str)
                bm = local_df["baseline_method"].astype(str)
                mt = local_df["midline_type"].astype(str)
                local_df["method_name"] = np.where(
                    mf.str.lower().eq("baseline"),
                    bm.where(bm.str.len() > 0, "baseline"),
                    mt.where(mt.str.len() > 0, "model"),
                )
            else:
                return
        local_df[metric_col] = pd.to_numeric(local_df[metric_col], errors="coerce")
        local_df = local_df[np.isfinite(local_df[metric_col].to_numpy(float))]
        if local_df.empty:
            return

        # Keep top methods by count for readability.
        order = (
            local_df.groupby("method_name", dropna=False)[metric_col]
            .size()
            .sort_values(ascending=False)
            .head(int(max_methods))
            .index.astype(str)
            .tolist()
        )
        local_df = local_df[local_df["method_name"].astype(str).isin(order)].copy()
        if local_df.empty:
            return
        # Order by median error (lower is better).
        med_order = (
            local_df.groupby("method_name", dropna=False)[metric_col]
            .median()
            .sort_values(ascending=True)
            .index.astype(str)
            .tolist()
        )
        local_df["method_name"] = pd.Categorical(local_df["method_name"].astype(str), categories=med_order, ordered=True)

        fig_w = max(9.0, 0.55 * len(med_order))
        fig, ax = plt.subplots(figsize=(fig_w, 5.0), dpi=180)
        parts = ax.boxplot(
            [local_df.loc[local_df["method_name"] == m, metric_col].to_numpy(float) for m in med_order],
            labels=med_order,
            patch_artist=True,
            showfliers=True,
            medianprops=dict(color="black", linewidth=1.2),
            whiskerprops=dict(color="#555555"),
            capprops=dict(color="#555555"),
            flierprops=dict(marker="o", markersize=3, alpha=0.5, markerfacecolor="#888888", markeredgecolor="#888888"),
        )
        # Color by source class.
        cls_map = {}
        for m in med_order:
            g = local_df.loc[local_df["method_name"] == m]
            if g.empty:
                cls_map[m] = "auto"
                continue
            mf0 = str(g.get("method_family", pd.Series([""])).astype(str).iloc[0]).lower() if "method_family" in g.columns else ""
            mt0 = str(g.get("midline_type", pd.Series([""])).astype(str).iloc[0]).lower() if "midline_type" in g.columns else ""
            cls_map[m] = "baseline" if mf0 == "baseline" else ("manual" if mt0 == "manual" else "auto")
        for box, m in zip(parts["boxes"], med_order):
            box.set_facecolor(COLOR_MAP.get(cls_map.get(m, "auto"), "#4c78a8"))
            box.set_alpha(0.65)

        ax.set_ylabel(metric_col)
        ax.set_title(title)
        ax.set_xticklabels(med_order, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        legend_handles = [
            Patch(facecolor=COLOR_MAP["manual"], edgecolor="none", label="manual"),
            Patch(facecolor=COLOR_MAP["auto"], edgecolor="none", label="auto"),
            Patch(facecolor=COLOR_MAP["baseline"], edgecolor="none", label="baseline"),
        ]
        ax.legend(handles=legend_handles, loc="best", fontsize=8, framealpha=0.9)
        plt.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)

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
            & all_df["baseline_method"].astype(str).str.lower().str.startswith("skel_")
        )
        n_drop_all = int(np.count_nonzero(m_skel_all.to_numpy(dtype=bool)))
        if n_drop_all > 0:
            all_df = all_df.loc[~m_skel_all].copy()
            _log(verbose, f"[summarize] width summary(all): excluded skel_ baseline rows={n_drop_all}")

    if all_df.empty:
        _log(verbose, "[summarize] width summary: empty after skel_ filtering at all_df stage")
        return outputs

    all_csv = os.path.join(out_dir, "dataset_width_summary_all.csv")
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
    def _width_method_label(row):
        mf = str(row.get("method_family", "") or "")
        mt = str(row.get("midline_type", "") or "")
        bm = str(row.get("baseline_method", "") or "")
        if mf == "baseline":
            return bm or "baseline"
        if mf in ("", "model", "nan", "None"):
            # "model" is just an internal bucket; use the actual supervision source.
            return mt or "model"
        return f"{mt}:{mf}" if mt and mt != "unknown" else mf

    grouped["method_name"] = grouped.apply(_width_method_label, axis=1)

    # --------------------------------------------------------
    # Color classification
    # --------------------------------------------------------
    def _classify(row):
        if str(row["method_family"]) == "baseline":
            return "baseline"
        if str(row["midline_type"]) == "manual":
            return "manual"
        return "auto"

    grouped["source_class"] = grouped.apply(_classify, axis=1)
    grouped["color"] = grouped["source_class"].map(COLOR_MAP)

    if EXCLUDE_SKEL_BASELINE_METHODS and not grouped.empty:
        m_skel = (
            grouped["method_family"].astype(str).str.lower().eq("baseline")
            & grouped["baseline_method"].astype(str).str.lower().str.startswith("skel_")
        )
        n_drop = int(np.count_nonzero(m_skel.to_numpy(dtype=bool)))
        if n_drop > 0:
            grouped = grouped.loc[~m_skel].copy()
            _log(verbose, f"[summarize] width summary: excluded skel_ baseline methods rows={n_drop}")

    if grouped.empty:
        _log(verbose, "[summarize] width summary: empty after skel_ baseline filtering")
        return outputs

    grp_csv = os.path.join(out_dir, "dataset_width_summary_grouped.csv")
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
            split_csv = os.path.join(out_dir, "dataset_width_error_split.csv")
            split_out.to_csv(split_csv, index=False)
            outputs["width_error_split_csv"] = split_csv

    mae_col = "mae_px_mean" if "mae_px_mean" in grouped.columns else None
    rmse_col = "rmse_px_mean" if "rmse_px_mean" in grouped.columns else None
    total_grouped = grouped[grouped["crack_type"].astype(str).str.upper() == "TOTAL"].copy()
    plot_grouped = total_grouped if not total_grouped.empty else grouped
    legend_items = [(k, v) for k, v in [("manual", COLOR_MAP["manual"]), ("auto", COLOR_MAP["auto"]), ("baseline", COLOR_MAP["baseline"])]
                    if k in set(plot_grouped["source_class"].astype(str).tolist())]

    if mae_col:
        # Keep canonical filename, but with richer uncertainty display.
        out_png = os.path.join(out_dir, "dataset_width_mae_by_method.png")
        _plot_metric_bar_iqr_outliers(
            df_plot=all_df if total_grouped.empty else all_df[all_df["crack_type"].astype(str).str.upper() == "TOTAL"].copy(),
            metric_col="mae_px",
            out_png=out_png,
            title="Dataset width MAE by method (mean + IQR + outliers)"
                  if not total_grouped.empty
                  else "Dataset width MAE by method (mean + IQR + outliers)",
        )
        outputs["width_mae_png"] = out_png
        out_png_box = os.path.join(out_dir, "dataset_width_mae_box_by_method.png")
        _plot_metric_box_by_method(
            df_plot=all_df if total_grouped.empty else all_df[all_df["crack_type"].astype(str).str.upper() == "TOTAL"].copy(),
            metric_col="mae_px",
            out_png=out_png_box,
            title="Dataset width MAE by method (box+whisker)",
        )
        outputs["width_mae_box_png"] = out_png_box

    if rmse_col:
        # Keep canonical filename, but with richer uncertainty display.
        out_png = os.path.join(out_dir, "dataset_width_rmse_by_method.png")
        _plot_metric_bar_iqr_outliers(
            df_plot=all_df if total_grouped.empty else all_df[all_df["crack_type"].astype(str).str.upper() == "TOTAL"].copy(),
            metric_col="rmse_px",
            out_png=out_png,
            title="Dataset width RMSE by method (mean + IQR + outliers)"
                  if not total_grouped.empty
                  else "Dataset width RMSE by method (mean + IQR + outliers)",
        )
        outputs["width_rmse_png"] = out_png
        out_png_box = os.path.join(out_dir, "dataset_width_rmse_box_by_method.png")
        _plot_metric_box_by_method(
            df_plot=all_df if total_grouped.empty else all_df[all_df["crack_type"].astype(str).str.upper() == "TOTAL"].copy(),
            metric_col="rmse_px",
            out_png=out_png_box,
            title="Dataset width RMSE by method (box+whisker)",
        )
        outputs["width_rmse_box_png"] = out_png_box

    # Additional distribution-aware plots: mean bar + IQR + outlier dots.
    # separate distribution plot files removed; canonical by_method plots include IQR/outliers

    _log(verbose, f"[summarize] width summary rows={len(all_df)}")
    return outputs


def _aggregate_midline_metrics(
    image_dirs: List[str],
    out_dir: str,
    *,
    verbose: bool,
) -> Dict[str, str]:
    outputs = {}
    EXCLUDE_SKEL_BASELINE_METHODS = True
    MIDLINE_DEBUG = True
    frames = []
    for img_dir in image_dirs:
        image = os.path.basename(img_dir)
        for p in glob.glob(os.path.join(img_dir, "**", "*_midline_metrics_*.csv"), recursive=True):
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

    if not frames:
        return outputs

    all_df = pd.concat(frames, ignore_index=True)
    if MIDLINE_DEBUG:
        try:
            dbg_cols = [c for c in ["image", "crack_type", "geometry_type", "method_family", "baseline_method", "midline_type"] if c in all_df.columns]
            dbg_csv = os.path.join(out_dir, "dataset_midline_debug_all_rows_raw.csv")
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
            & all_df["baseline_method"].astype(str).str.lower().str.startswith("skel_")
        )
        if m_skel.any():
            all_df = all_df.loc[~m_skel].copy()
            if MIDLINE_DEBUG:
                _log(verbose, f"[midline-debug] dropped skel_ baseline rows: {int(np.count_nonzero(m_skel.to_numpy(dtype=bool)))}")

    if all_df.empty:
        return outputs

    all_csv = os.path.join(out_dir, "dataset_midline_metrics_all.csv")
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
        grp_csv = os.path.join(out_dir, "dataset_midline_metrics_grouped.csv")
        grouped.to_csv(grp_csv, index=False)
        outputs["midline_grouped_csv"] = grp_csv

        if "score_mid_mean" in grouped.columns:
            d = grouped.copy()
            label_cols = [c for c in ["midline_type", "crack_type", "geometry_type"] if c in d.columns]
            d["group_label"] = d[label_cols].astype(str).agg(":".join, axis=1)
            d = d.sort_values("score_mid_mean", ascending=True).head(20)
            out_png = os.path.join(out_dir, "dataset_midline_score_by_group.png")
            _save_bar(
                d["group_label"].astype(str).tolist(),
                d["score_mid_mean"].astype(float).tolist(),
                out_png=out_png,
                title="Dataset midline score by group",
                ylabel="score_mid",
            )
            outputs["midline_score_png"] = out_png

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
                if str(row.get("midline_type", "")).lower() == "manual":
                    return "manual"
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
                    dbg_rank_csv = os.path.join(out_dir, "dataset_midline_debug_rank_input.csv")
                    dbg_cols = [c for c in ["image", "method_family", "baseline_method", "midline_type", "crack_type", "geometry_type", "score_mid", "length_px", "_w", "source_relpath"] if c in d_rank_src.columns]
                    d_rank_src.loc[:, dbg_cols].to_csv(dbg_rank_csv, index=False)
                    _log(verbose, f"[midline-debug] wrote rank-input rows -> {dbg_rank_csv}")
                except Exception as e:
                    _log(verbose, f"[midline-debug] rank-input dump failed: {e}")

            key_cols = ["method_name", "geometry_type", "source_class"]
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
                rank_df = pd.DataFrame(rank_rows).sort_values("score_mid_wmean", ascending=True)
                rank_csv = os.path.join(out_dir, "dataset_midline_score_ranked.csv")
                rank_df.to_csv(rank_csv, index=False)
                outputs["midline_score_ranked_csv"] = rank_csv
                if MIDLINE_DEBUG:
                    try:
                        if "source_class" in rank_df.columns:
                            _log(verbose, f"[midline-debug] ranked source_class counts:\n{rank_df['source_class'].astype(str).value_counts(dropna=False).to_string()}")
                    except Exception:
                        pass

                top = rank_df.head(20).copy()
                top["label"] = top.apply(
                    lambda r: f"{r.get('method_name', 'unknown')}|{r.get('geometry_type', 'unknown')}",
                    axis=1,
                )
                color_map = {"manual": "#d62728", "auto": "#1f77b4", "baseline": "#2ca02c"}
                colors = [color_map.get(str(c), "#4c78a8") for c in top["source_class"].astype(str).tolist()]
                fig_w = max(10.0, 0.55 * len(top))
                fig, ax = plt.subplots(figsize=(fig_w, 5.0), dpi=180)
                x = np.arange(len(top), dtype=float)
                vals = top["score_mid_wmean"].astype(float).to_numpy(float)
                ax.bar(x, vals, color=colors, alpha=0.88)
                ax.set_xticks(x)
                ax.set_xticklabels(top["label"].astype(str).tolist(), rotation=35, ha="right", fontsize=8)
                ax.set_ylabel("score_mid_wmean")
                ax.set_title("Dataset Midline RS3-Style Score (combined, lower is better)")
                ax.grid(axis="y", alpha=0.2)
                legend_handles = [
                    Patch(facecolor=color_map["manual"], edgecolor="none", label="manual"),
                    Patch(facecolor=color_map["auto"], edgecolor="none", label="auto"),
                    Patch(facecolor=color_map["baseline"], edgecolor="none", label="baseline"),
                ]
                ax.legend(handles=legend_handles, loc="best", framealpha=0.9, fontsize=8)
                plt.tight_layout()
                out_png = os.path.join(out_dir, "dataset_midline_score_ranked.png")
                fig.savefig(out_png, bbox_inches="tight")
                plt.close(fig)
                outputs["midline_score_ranked_png"] = out_png

    _log(verbose, f"[summarize] midline rows={len(all_df)}")
    return outputs


def _aggregate_timing_metrics(
    image_dirs: List[str],
    out_dir: str,
    *,
    verbose: bool,
) -> Dict[str, str]:
    outputs = {}

    # timings_core.csv (per-crack timing rows for edge-tracking stage only)
    timing_frames = []
    for img_dir in image_dirs:
        image = os.path.basename(img_dir)
        p = os.path.join(img_dir, "timings_core.csv")
        df = _safe_read_csv(p)
        if df is None:
            continue
        d = df.copy()
        d["image"] = image
        timing_frames.append(d)

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

        # aggregate mean timing components
        key_cols = [c for c in ["edge_masks_sec", "edges_tracking_sec", "build_combined_sec"] if c in all_df.columns]
        if not key_cols:
            key_cols = num_cols[:6]
        if key_cols:
            vals = [float(pd.to_numeric(all_df[c], errors="coerce").mean()) for c in key_cols]
            out_png = os.path.join(out_dir, f"dataset_{stage_tag}_components.png")
            _save_bar(
                key_cols,
                vals,
                out_png=out_png,
                title="Edge-Tracking Stage Timing Components (mean)",
                ylabel="seconds",
            )
            outputs[f"{stage_tag}_components_png"] = out_png
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

    return outputs


def _aggregate_width_distribution(
    image_dirs: List[str],
    out_dir: str,
    *,
    verbose: bool,
) -> Dict[str, str]:
    outputs = {}
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
    csv_path = os.path.join(out_dir, "dataset_width_distribution_all.csv")
    all_df.to_csv(csv_path, index=False)
    outputs["width_dist_all_csv"] = csv_path

    try:
        from helpers.present_plots import plot_width_distribution_report

        rep_dir = os.path.join(out_dir, "width_distribution_report_dataset")
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
        if "image" not in all_df.columns and "stem" in all_df.columns:
            all_df["image"] = all_df["stem"].astype(str)
        if evaluated_images and "image" in all_df.columns:
            all_df = all_df[all_df["image"].astype(str).isin({str(x) for x in evaluated_images})].copy()

        out_csv = os.path.join(out_dir, "dataset_baseline_timings_all.csv")
        all_df.to_csv(out_csv, index=False)
        outputs["baseline_timings_all_csv"] = out_csv

        num_cols = [
            c
            for c in all_df.columns
            if pd.api.types.is_numeric_dtype(all_df[c]) and (c.endswith("_s") or c.endswith("_sec"))
        ]
        if num_cols:
            vals = [float(pd.to_numeric(all_df[c], errors="coerce").mean()) for c in num_cols]
            out_png = os.path.join(out_dir, "dataset_baseline_timings_components.png")
            _save_bar(
                num_cols,
                vals,
                out_png=out_png,
                title="Width Baseline Timing Components (mean)",
                ylabel="seconds",
                rotate=40,
            )
            outputs["baseline_timings_components_png"] = out_png

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
                        "category": "baseline_segmentation",
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
                "category": "baseline_segmentation",
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
            if pd.api.types.is_numeric_dtype(d[c]) and (c.endswith("_sec") or c.endswith("_s"))
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

    roll_rows = []
    for (stage, mode, component), g in all_df.groupby(["stage", "mode", "component"], dropna=False):
        vals = pd.to_numeric(g["sec"], errors="coerce").to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        mean_t = float(np.mean(vals))
        total_t = float(np.sum(vals))
        roll_rows.append(
            {
                "category": "gt_supervision",
                "stage": str(stage),
                "mode": str(mode),
                "component": str(component),
                "mean_sec": mean_t,
                "weighted_mean_sec": mean_t,
                "total_sec": total_t,
                "count": int(vals.size),
            }
        )
    if roll_rows:
        roll_df = pd.DataFrame(roll_rows).sort_values(["stage", "mode", "component"])
        out_csv = os.path.join(out_dir, "dataset_gt_supervision_timings.csv")
        roll_df.to_csv(out_csv, index=False)
        outputs["gt_supervision_timings_csv"] = out_csv
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


def _plot_dataset_full_timing_overview(
    out_dir: str,
    *,
    verbose: bool = True,
) -> Dict[str, str]:
    outputs: Dict[str, str] = {}

    rows = []

    # Baseline segmentation timing components.
    p_base = os.path.join(out_dir, "dataset_baseline_timings_rollup.csv")
    df_base = _safe_read_csv(p_base)
    if df_base is not None and not df_base.empty:
        for _, r in df_base.iterrows():
            comp = str(r.get("component", ""))
            # Prefer baseline method-like components.
            if not any(k in comp for k in ("mat_", "pca_", "esd_", "eob_", "width", "inference_seconds:")):
                continue
            if comp.startswith("inference_seconds:"):
                b_label = comp.split(":", 1)[1]
                b_cat = "baseline_segmentation"
            else:
                b_label = comp
                b_cat = "width_baseline"
            rows.append(
                {
                    "category": b_cat,
                    "label": b_label,
                    "value_sec": float(pd.to_numeric(r.get("weighted_mean_sec", np.nan), errors="coerce")),
                }
            )

    # GT supervision timings (atomic/combined normals + centering).
    p_sup = os.path.join(out_dir, "dataset_gt_supervision_timings.csv")
    df_sup = _safe_read_csv(p_sup)
    if df_sup is not None and not df_sup.empty:
        for _, r in df_sup.iterrows():
            stage = str(r.get("stage", ""))
            mode = str(r.get("mode", ""))
            if mode not in {"atomic", "combined"}:
                continue
            if stage not in {"normals", "centering"}:
                continue
            rows.append(
                {
                    "category": "gt_supervision",
                    "label": f"{stage}:{mode}",
                    "value_sec": float(pd.to_numeric(r.get("weighted_mean_sec", np.nan), errors="coerce")),
                }
            )

    # Edge/manual-auto total timing from per-crack edge-tracking stage table.
    p_edge = os.path.join(out_dir, "dataset_edge_tracking_stage_all.csv")
    df_edge = _safe_read_csv(p_edge)
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
            if weight_col:
                w = pd.to_numeric(g[weight_col], errors="coerce").fillna(0.0).to_numpy(float)
                ok = np.isfinite(per_row) & np.isfinite(w) & (w > 0)
                val = float(np.sum(per_row[ok] * w[ok]) / np.sum(w[ok])) if np.any(ok) else float(np.nanmean(per_row))
            else:
                val = float(np.nanmean(per_row))
            rows.append({"category": "edge_pipeline_total", "label": f"{sup}:total_pipeline", "value_sec": val})

    if not rows:
        return outputs

    df = pd.DataFrame(rows)
    df = df[np.isfinite(pd.to_numeric(df["value_sec"], errors="coerce"))].copy()
    if df.empty:
        return outputs
    df["value_sec"] = pd.to_numeric(df["value_sec"], errors="coerce")
    df = df.sort_values("value_sec", ascending=True).reset_index(drop=True)
    out_csv = os.path.join(out_dir, "dataset_timing_full_overview.csv")
    df.to_csv(out_csv, index=False)
    outputs["timing_full_overview_csv"] = out_csv

    color_map = {
        "baseline_segmentation": "#2ca02c",  # green (hrsegnet/sam3 inference)
        "width_baseline": "#ff7f0e",         # orange (width baseline stages)
        "gt_supervision": "#1f77b4",
        "edge_pipeline_total": "#d62728",
    }
    colors = [color_map.get(str(c), "#4c78a8") for c in df["category"].astype(str)]
    fig_w = max(9.5, 0.42 * len(df))
    fig, ax = plt.subplots(figsize=(fig_w, 4.8), dpi=180)
    xs = np.arange(len(df))
    ax.bar(xs, df["value_sec"].to_numpy(float), color=colors, alpha=0.86)
    ax.set_xticks(xs)
    ax.set_xticklabels(df["label"].astype(str).tolist(), rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("weighted mean sec")
    ax.set_title("Dataset Full Timing Overview")
    ax.grid(axis="y", alpha=0.2)
    handles = [Patch(facecolor=v, edgecolor="none", label=k) for k, v in color_map.items() if k in set(df["category"].astype(str))]
    if handles:
        ax.legend(handles=handles, loc="best", framealpha=0.9, fontsize=8)
    plt.tight_layout()
    out_png = os.path.join(out_dir, "dataset_timing_full_overview.png")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    outputs["timing_full_overview_png"] = out_png
    _log(verbose, f"[summarize] wrote full timing overview rows={len(df)}")
    return outputs


def summarize_dataset_metrics(
    save_folder: str,
    *,
    out_dir: Optional[str] = None,
    baseline_roots: Optional[List[str]] = None,
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

    image_dirs = _list_image_metric_dirs(metrics_root)
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
    outputs.update(_aggregate_mask_metrics(image_dirs, out_dir, verbose=verbose))
    outputs.update(_aggregate_width_metrics(image_dirs, out_dir, verbose=verbose))
    outputs.update(_aggregate_midline_metrics(image_dirs, out_dir, verbose=verbose))
    outputs.update(_aggregate_timing_metrics(image_dirs, out_dir, verbose=verbose))
    outputs.update(_aggregate_width_distribution(image_dirs, out_dir, verbose=verbose))
    outputs.update(_aggregate_edge_rs3_selection(image_dirs, out_dir, verbose=verbose))
    outputs.update(
        _aggregate_supervision_timings(
            supervision_root=os.path.join(save_folder, "supervision"),
            out_dir=out_dir,
            evaluated_images=evaluated_images_set,
            verbose=verbose,
        )
    )
    outputs.update(
        _aggregate_gt_centering_weighted_summaries(
            save_folder=save_folder,
            out_dir=out_dir,
            evaluated_images=evaluated_images_set,
            verbose=verbose,
        )
    )
    outputs.update(
        _aggregate_baseline_timings(
            out_dir,
            baseline_roots=baseline_roots,
            evaluated_images=evaluated_images_set,
            verbose=verbose,
        )
    )
    outputs.update(
        _plot_dataset_full_timing_overview(
            out_dir=out_dir,
            verbose=verbose,
        )
    )

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
