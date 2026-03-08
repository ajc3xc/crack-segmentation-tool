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
            if iou_col:
                out_png = os.path.join(out_dir, "dataset_mask_total_iou_by_variant.png")
                _save_bar(
                    total["variant"].astype(str).tolist(),
                    total[iou_col].astype(float).tolist(),
                    out_png=out_png,
                    title="Dataset TOTAL IoU by variant",
                    ylabel="IoU",
                )
                outputs["mask_total_iou_png"] = out_png
            if bf1_col:
                out_png = os.path.join(out_dir, "dataset_mask_total_boundary_f1_by_variant.png")
                _save_bar(
                    total["variant"].astype(str).tolist(),
                    total[bf1_col].astype(float).tolist(),
                    out_png=out_png,
                    title="Dataset TOTAL boundary F1 by variant",
                    ylabel="Boundary F1",
                )
                outputs["mask_total_bf1_png"] = out_png

    _log(verbose, f"[summarize] mask metrics rows={len(all_df)}")
    return outputs


def _aggregate_width_metrics(
    image_dirs: List[str],
    out_dir: str,
    *,
    verbose: bool,
) -> Dict[str, str]:
    outputs = {}
    frames = []

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

    COLOR_MAP = {
        "baseline": "#6c757d",  # gray
        "auto": "#0077cc",      # blue
        "manual": "#d62728",    # red
    }

    grouped["color"] = grouped["source_class"].map(COLOR_MAP)

    grp_csv = os.path.join(out_dir, "dataset_width_summary_grouped.csv")
    grouped.to_csv(grp_csv, index=False)
    outputs["width_summary_grouped_csv"] = grp_csv

    mae_col = "mae_px_mean" if "mae_px_mean" in grouped.columns else None
    rmse_col = "rmse_px_mean" if "rmse_px_mean" in grouped.columns else None
    total_grouped = grouped[grouped["crack_type"].astype(str).str.upper() == "TOTAL"].copy()
    plot_grouped = total_grouped if not total_grouped.empty else grouped
    legend_items = [(k, v) for k, v in [("manual", COLOR_MAP["manual"]), ("auto", COLOR_MAP["auto"]), ("baseline", COLOR_MAP["baseline"])]
                    if k in set(plot_grouped["source_class"].astype(str).tolist())]

    if mae_col:
        top = plot_grouped.sort_values(mae_col, ascending=True).head(20)

        out_png = os.path.join(out_dir, "dataset_width_mae_by_method.png")
        _save_bar(
            labels=top["method_name"].astype(str).tolist(),
            values=top[mae_col].astype(float).tolist(),
            colors=top["color"].tolist(),
            color_legend=legend_items,
            out_png=out_png,
            title="Dataset width MAE by method (TOTAL cracks)" if not total_grouped.empty else "Dataset width MAE by method",
            ylabel="MAE (px)",
        )
        outputs["width_mae_png"] = out_png

    if rmse_col:
        top = plot_grouped.sort_values(rmse_col, ascending=True).head(20)

        out_png = os.path.join(out_dir, "dataset_width_rmse_by_method.png")
        _save_bar(
            labels=top["method_name"].astype(str).tolist(),
            values=top[rmse_col].astype(float).tolist(),
            colors=top["color"].tolist(),
            color_legend=legend_items,
            out_png=out_png,
            title="Dataset width RMSE by method (TOTAL cracks)" if not total_grouped.empty else "Dataset width RMSE by method",
            ylabel="RMSE (px)",
        )
        outputs["width_rmse_png"] = out_png

    _log(verbose, f"[summarize] width summary rows={len(all_df)}")
    return outputs


def _aggregate_midline_metrics(
    image_dirs: List[str],
    out_dir: str,
    *,
    verbose: bool,
) -> Dict[str, str]:
    outputs = {}
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
    if "midline_type" not in all_df.columns:
        all_df["midline_type"] = all_df["midline_type_path"]
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

    _log(verbose, f"[summarize] midline rows={len(all_df)}")
    return outputs


def _aggregate_timing_metrics(
    image_dirs: List[str],
    out_dir: str,
    *,
    verbose: bool,
) -> Dict[str, str]:
    outputs = {}

    # timings_core.csv (per-crack timing rows)
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
        all_df = pd.concat(timing_frames, ignore_index=True)
        all_csv = os.path.join(out_dir, "dataset_timings_core_all.csv")
        all_df.to_csv(all_csv, index=False)
        outputs["timings_core_all_csv"] = all_csv

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
            grp_csv = os.path.join(out_dir, "dataset_timings_core_grouped.csv")
            grouped.to_csv(grp_csv, index=False)
            outputs["timings_core_grouped_csv"] = grp_csv

        # aggregate mean timing components
        key_cols = [c for c in ["edge_masks_sec", "edges_tracking_sec", "build_combined_sec"] if c in all_df.columns]
        if not key_cols:
            key_cols = num_cols[:6]
        if key_cols:
            vals = [float(pd.to_numeric(all_df[c], errors="coerce").mean()) for c in key_cols]
            out_png = os.path.join(out_dir, "dataset_timings_core_components.png")
            _save_bar(
                key_cols,
                vals,
                out_png=out_png,
                title="Dataset timing components (mean)",
                ylabel="seconds",
            )
            outputs["timings_core_components_png"] = out_png

        _log(verbose, f"[summarize] timings_core rows={len(all_df)}")

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
    for img_dir in image_dirs:
        image = os.path.basename(img_dir)
        p = os.path.join(img_dir, "edge_sweep_family_agg.csv")
        df = _safe_read_csv(p)
        if df is None:
            continue
        d = df.copy()
        d["image"] = image
        edge_frames.append(d)

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
                    f"w={int(r['param_window_half_size'])},mu={r['param_mu']},"
                    f"l={int(r['param_l'])},p={int(r['param_p'])},m={r['param_seg_mode']}"
                ),
                axis=1,
            )
            out_png = os.path.join(out_dir, "dataset_edge_family_scores.png")
            _save_bar(
                top["label"].tolist(),
                top["edge_score_wmean_mean"].astype(float).tolist(),
                out_png=out_png,
                title="Dataset edge family score (lower is better)",
                ylabel="edge_score_wmean",
                rotate=65,
            )
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
    for root in roots:
        p1 = os.path.join(root, "width_baseline_timings.csv")
        p2 = os.path.join(root, "width_baseline_timings_summary.csv")
        if os.path.isfile(p1):
            timing_files.append(p1)
        if os.path.isfile(p2):
            summary_files.append(p2)

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
        out_csv = os.path.join(out_dir, "dataset_baseline_timings_all.csv")
        all_df.to_csv(out_csv, index=False)
        outputs["baseline_timings_all_csv"] = out_csv

        num_cols = [c for c in all_df.columns if pd.api.types.is_numeric_dtype(all_df[c]) and c.endswith("_s")]
        if num_cols:
            vals = [float(pd.to_numeric(all_df[c], errors="coerce").mean()) for c in num_cols]
            out_png = os.path.join(out_dir, "dataset_baseline_timings_components.png")
            _save_bar(
                num_cols,
                vals,
                out_png=out_png,
                title="Baseline timing components (mean)",
                ylabel="seconds",
                rotate=40,
            )
            outputs["baseline_timings_components_png"] = out_png

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

    _log(verbose, f"[summarize] baseline timing files={len(timing_files)} summaries={len(summary_files)}")
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
    os.makedirs(out_dir, exist_ok=True)

    image_dirs = _list_image_metric_dirs(metrics_root)
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
        _aggregate_baseline_timings(
            out_dir,
            baseline_roots=baseline_roots,
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
