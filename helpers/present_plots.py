# helpers/present_plots.py
import os, numpy as np, pandas as pd, cv2
import matplotlib
matplotlib.use("Agg")   # fastest non-GUI backend for PNG/PDF/SVG export
matplotlib.rcParams.update({
    "figure.max_open_warning": 0,
    "text.kerning_factor": 0,
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "path.simplify": True,
    "path.simplify_threshold": 1.0,
    "savefig.dpi": 180,
})

import matplotlib.pyplot as plt
plt.ioff()   # no interactive figure updates

import seaborn as sns

# ------------------------------ #
# A) DECK-READY SUMMARY FIGURES  #
# ------------------------------ #

def plot_iou_vs_bf1_scatter(metrics_csv, out_png, *, supervision=None):
    df = pd.read_csv(metrics_csv)

    if supervision is not None and "supervision" in df.columns:
        df = df[df["supervision"].astype(str) == str(supervision)].copy()

    # keep only per-crack rows, not TOTAL
    df = df[df["crack_type"].isin(["atomic", "combined"])].copy()
    if df.empty:
        return

    # robust cols
    iou = df["iou"].astype(float).values if "iou" in df else np.array([])
    bf1 = df["boundary_f1"].astype(float).values if "boundary_f1" in df else np.array([])
    assd_col = "ASSD" if "ASSD" in df.columns else ("assd" if "assd" in df.columns else None)
    assd = df[assd_col].astype(float).values if assd_col else None

    if len(iou) == 0 or len(bf1) == 0:
        return

    # marker by crack_type (optional but useful)
    is_atomic = (df["crack_type"].astype(str) == "atomic").values
    markers = np.where(is_atomic, "o", "s")

    plt.figure(figsize=(6, 5))

    if assd is not None and len(assd) == len(iou):
        # two scatters so marker changes still work with a single colorbar
        sc0 = plt.scatter(iou[is_atomic], bf1[is_atomic], c=assd[is_atomic], s=55, alpha=0.9, marker="o")
        sc1 = plt.scatter(iou[~is_atomic], bf1[~is_atomic], c=assd[~is_atomic], s=55, alpha=0.9, marker="s")
        # Use a scatter that actually has points for colorbar mappable.
        _cb_mappable = sc1 if int(np.sum(~is_atomic)) > 0 else sc0
        if int(np.sum(is_atomic)) > 0 or int(np.sum(~is_atomic)) > 0:
            cb = plt.colorbar(_cb_mappable)
            cb.set_label("ASSD (px)")
    else:
        plt.scatter(iou[is_atomic], bf1[is_atomic], s=55, alpha=0.9, marker="o", label="atomic")
        plt.scatter(iou[~is_atomic], bf1[~is_atomic], s=55, alpha=0.9, marker="s", label="combined")
        plt.legend()

    plt.xlabel("IoU")
    plt.ylabel("Boundary F1")
    title = "IoU vs Boundary F1"
    if supervision is not None:
        title += f" ({supervision})"
    plt.title(title)
    plt.grid(True, alpha=.3)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def plot_assd_hd95_box(metrics_csv, out_png):
    df = pd.read_csv(metrics_csv)
    dfi = df[df["crack_type"].isin(["atomic","combined"])].copy()
    if dfi.empty or ("ASSD" not in dfi or "HD95" not in dfi): return
    data = [dfi["ASSD"].astype(float).values, dfi["HD95"].astype(float).values]
    plt.figure(figsize=(5,4))
    plt.boxplot(data, labels=["ASSD","HD95"])
    plt.ylabel("pixels")
    plt.title("Surface distance distribution")
    plt.tight_layout(); plt.savefig(out_png); plt.close()
    
def plot_width_summary_triplet(metrics_dir, base_name, out_png):
    import os, numpy as np, pandas as pd
    import matplotlib.pyplot as plt

    paths = {
        "manual":   os.path.join(metrics_dir, f"{base_name}_width_summary_manual.csv"),
        "auto":     os.path.join(metrics_dir, f"{base_name}_width_summary_auto.csv"),
        "combined": os.path.join(metrics_dir, f"{base_name}_width_summary_combined.csv"),
    }

    rows = []
    for tag, p in paths.items():
        if not os.path.exists(p):
            continue

        df = pd.read_csv(p)
        df.columns = [c.lower() for c in df.columns]

        rows.append({
            "method": tag,
            "MAE":  df["mae_px"].mean()  if "mae_px" in df else np.nan,
            "RMSE": df["rmse_px"].mean() if "rmse_px" in df else np.nan,
            "Bias": df["bias_px"].mean() if "bias_px" in df else np.nan,
            "Corr": df["corr"].mean()    if "corr"    in df else np.nan
        })

    if not rows:
        print("[TRIPLET] no usable width-summary CSVs")
        return

    d = pd.DataFrame(rows)
    methods = d["method"].tolist()
    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))

    # LEFT – grouped MAE/RMSE/Bias
    ax0 = ax[0]
    ax0.bar(x - width, d["MAE"],  width, label="MAE")
    ax0.bar(x,         d["RMSE"], width, label="RMSE")
    ax0.bar(x + width, d["Bias"], width, label="Bias")

    ax0.set_xticks(x)
    ax0.set_xticklabels(methods, fontsize=10, fontweight="normal")  # <— NOT bold
    ax0.set_title("Width errors (px)", fontsize=14, fontweight="bold")
    ax0.set_ylabel("px")
    ax0.legend(fontsize=11)

    # annotate
    for bars in ax0.containers:
        for b in bars:
            h = b.get_height()
            if np.isfinite(h):
                ax0.text(b.get_x() + b.get_width()/2, h + 0.05,
                         f"{h:.2f}", ha="center",
                         fontsize=8, fontweight="bold")  # <— bold values

    # RIGHT – correlation
    ax1 = ax[1]
    ax1.bar(methods, d["Corr"])
    ax1.set_ylim(0, 1)
    ax1.set_title("Width correlation", fontsize=14, fontweight="bold")
    ax1.set_ylabel("r")
    for lbl in ax1.get_xticklabels():
        lbl.set_fontweight("normal")  # <— NOT bold
        lbl.set_ha("center")

    for i, v in enumerate(d["Corr"]):
        if np.isfinite(v):
            ax1.text(i, v + 0.02, f"{v:.2f}", ha="center",
                     fontsize=9, fontweight="bold")  # <— bold value

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

def plot_mask_metrics_triplet(metrics_dir, base_name, supervision, out_png):
    """
    Single-row, 3-column summary plot:
        [0] Region metrics
        [1] Boundary metrics
        [2] Confusion matrix heatmap

    Uses mask_metrics.csv inside metrics_dir.
    """
    import os, numpy as np, pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    csv_path = os.path.join(metrics_dir, f"{supervision}_mask_metrics.csv")
    if not os.path.exists(csv_path):
        print(f"[MASK_TRIPLET] {supervision}_mask_metrics.csv not found")
        return

    df = pd.read_csv(csv_path)

    # Prefer TOTAL row, fallback to mean
    total = None
    if "crack_type" in df.columns:
        m = df["crack_type"].astype(str).str.upper() == "TOTAL"
        if m.any():
            total = df[m].iloc[0]
    if total is None:
        total = df.mean(numeric_only=True)

    def get(col, default=np.nan):
        for c in total.index:
            if c.lower() == col.lower():
                try:
                    return float(total[c])
                except Exception:
                    return default
        return default

    region = {
        "Precision": get("precision"),
        "Recall":    get("recall"),
        "F1":        get("f1"),
        "IoU":       get("iou"),
    }

    boundary = {
        "Boundary Precision": get("boundary_precision"),
        "Boundary Recall":    get("boundary_recall"),
        "Boundary F1":        get("boundary_f1"),
    }

    tp, fp, fn, tn = (
        get("tp", 0),
        get("fp", 0),
        get("fn", 0),
        get("tn", 0),
    )
    cm = np.array([[tp, fn],
                   [fp, tn]], float)

    # --- PLOT: 1 row × 3 columns ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # ------------------------------
    # Column 0: Region metrics
    # ------------------------------
    def bar(ax, data, title, ylim=(0,1)):
        labels = list(data.keys())
        vals = list(data.values())
        xs = np.arange(len(vals))

        ax.bar(xs, vals)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8, fontweight="bold")
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=13, fontweight="bold")

        maxv = np.nanmax(vals) if np.isfinite(np.nanmax(vals)) else 1.0
        off = 0.03 * maxv
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(xs[i], v + off, f"{v:.2f}",
                        ha="center", fontsize=7)

    bar(axes[0], region, "Region metrics")
    bar(axes[1], boundary, "Boundary metrics")

    # ------------------------------
    # Column 2: Confusion matrix
    # ------------------------------
    sns.heatmap(
        cm,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        cbar=True,
        xticklabels=["Pred +","Pred -"],
        yticklabels=["GT +","GT -"],
        ax=axes[2]
    )
    axes[2].set_title("Confusion matrix", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def plot_midline_edge_metrics_bars(midline_csv, out_png):
    if not os.path.exists(midline_csv): return
    df = pd.read_csv(midline_csv)
    picks = [c for c in ["nn_mean_bidirectional","hausdorff_max","coverage_min","mean_tan_angle_error_deg"]
             if c in df.columns]
    if not picks: return
    means = df[picks].astype(float).mean()
    stds  = df[picks].astype(float).std()
    plt.figure(figsize=(6,4))
    x = np.arange(len(picks))
    plt.bar(x, means.values, yerr=stds.values)
    plt.xticks(x, picks, rotation=12)
    plt.title("Midline/edge metrics (mean ± sd)")
    plt.tight_layout(); plt.savefig(out_png); plt.close()

##################################################################
# New Metrics Plots
#################################################################

def plot_surface_distance_histogram(metrics_csv, out_png, *, supervision=None):
    if not os.path.exists(metrics_csv):
        return

    df = pd.read_csv(metrics_csv)

    if supervision is not None and "supervision" in df.columns:
        df = df[df["supervision"].astype(str) == str(supervision)].copy()

    # Prefer per-crack rows. If unavailable (e.g., baseline TOTAL-only),
    # fall back to TOTAL rows so the plot is still informative.
    if "crack_type" in df.columns:
        per = df[df["crack_type"].isin(["atomic", "combined"])].copy()
        if not per.empty:
            df = per
        else:
            df = df[df["crack_type"].astype(str).str.upper() == "TOTAL"].copy()

    assd_col = "ASSD" if "ASSD" in df.columns else ("assd" if "assd" in df.columns else None)
    hd95_col = "HD95" if "HD95" in df.columns else ("hd95" if "hd95" in df.columns else None)
    if assd_col is None or hd95_col is None:
        return

    assd = df[assd_col].astype(float).values
    hd95 = df[hd95_col].astype(float).values
    assd = assd[np.isfinite(assd)]
    hd95 = hd95[np.isfinite(hd95)]
    if assd.size == 0 and hd95.size == 0:
        return

    # TOTAL-only variants often have 1 sample; use bars instead of histogram.
    if assd.size <= 1 and hd95.size <= 1:
        plt.figure(figsize=(5, 4))
        vals = [
            float(assd[0]) if assd.size else np.nan,
            float(hd95[0]) if hd95.size else np.nan,
        ]
        plt.bar(["ASSD", "HD95"], vals)
        plt.ylabel("pixels")
        plt.title("Surface distance (TOTAL)")
        for i, v in enumerate(vals):
            if np.isfinite(v):
                plt.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        return

    plt.figure(figsize=(8, 4))
    plt.hist(assd, bins=30, alpha=0.6, label="ASSD")
    plt.hist(hd95, bins=30, alpha=0.6, label="HD95")
    plt.title("Surface distance histogram")
    plt.xlabel("pixels")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_total_mask_metrics_card(metrics_csv, out_png):
    """
    TOTAL-row summary card, useful for baseline variants that have no
    per-crack atomic/combined rows.
    """
    if not os.path.exists(metrics_csv):
        return

    df = pd.read_csv(metrics_csv)
    if df.empty:
        return

    if "crack_type" in df.columns:
        dft = df[df["crack_type"].astype(str).str.upper() == "TOTAL"].copy()
    else:
        dft = df.copy()
    if dft.empty:
        return

    row = dft.iloc[0]

    def _get(*names):
        for n in names:
            if n in row.index:
                try:
                    return float(row[n])
                except Exception:
                    return np.nan
        return np.nan

    region = {
        "IoU": _get("iou"),
        "F1": _get("f1"),
        "bF1": _get("boundary_f1"),
    }
    surface = {
        "ASSD": _get("ASSD", "assd"),
        "HD95": _get("HD95", "hd95"),
    }

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].bar(list(region.keys()), list(region.values()))
    ax[0].set_ylim(0, 1)
    ax[0].set_title("TOTAL region/boundary")
    for i, v in enumerate(region.values()):
        if np.isfinite(v):
            ax[0].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax[1].bar(list(surface.keys()), list(surface.values()))
    ax[1].set_title("TOTAL surface distance")
    for i, v in enumerate(surface.values()):
        if np.isfinite(v):
            ax[1].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)
    
def _add_mask_derived_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds derived size/fill columns if missing, using tp/fp/fn if present.
    Works on your per-crack rows and TOTAL rows (but plots usually exclude TOTAL).
    """
    out = df.copy()
    eps = 1e-9

    # try to infer sizes from confusion counts
    for c in ("tp", "fp", "fn", "tn"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if all(c in out.columns for c in ("tp", "fp", "fn")):
        if "gt_area_px" not in out.columns:
            out["gt_area_px"] = (out["tp"] + out["fn"]).astype("float")
        if "pred_area_px" not in out.columns:
            out["pred_area_px"] = (out["tp"] + out["fp"]).astype("float")
        if "union_area_px" not in out.columns:
            out["union_area_px"] = (out["tp"] + out["fp"] + out["fn"]).astype("float")

        if "underfill_rate" not in out.columns:
            out["underfill_rate"] = out["fn"] / (out["gt_area_px"] + eps)
        if "overfill_rate" not in out.columns:
            out["overfill_rate"] = out["fp"] / (out["pred_area_px"] + eps)
        if "fill_ratio" not in out.columns:
            out["fill_ratio"] = out["pred_area_px"] / (out["gt_area_px"] + eps)

    return out


def plot_size_vs_iou_scatter(metrics_csv, out_png, *, supervision=None, x_mode="gt_area_px"):
    """
    Scatter: size proxy vs IoU, colored by crack_type.
    x_mode:
      - "bbox_area"  (if you later add bbox_area)
      - "gt_area_px" (default, always available if tp/fp/fn exist)
      - "pred_area_px"
    """
    if not os.path.exists(metrics_csv):
        return

    df = pd.read_csv(metrics_csv)

    if supervision is not None and "supervision" in df.columns:
        df = df[df["supervision"].astype(str) == str(supervision)].copy()

    if "crack_type" in df.columns:
        df = df[df["crack_type"].isin(["atomic", "combined"])].copy()

    if df.empty or "iou" not in df.columns:
        return

    df = _add_mask_derived_cols(df)

    # choose x
    if x_mode in df.columns:
        x = pd.to_numeric(df[x_mode], errors="coerce").values.astype(float)
        xlab = x_mode
    else:
        # fallback to gt area if requested col not present
        if "gt_area_px" in df.columns:
            x = pd.to_numeric(df["gt_area_px"], errors="coerce").values.astype(float)
            xlab = "gt_area_px"
        else:
            return

    y = pd.to_numeric(df["iou"], errors="coerce").values.astype(float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) == 0:
        return

    # color by crack_type
    ctype = df.loc[keep, "crack_type"].astype(str).values
    is_atomic = (ctype == "atomic")

    plt.figure(figsize=(6, 5))
    plt.scatter(x[is_atomic], y[is_atomic], s=55, alpha=0.9, marker="o", label="atomic")
    plt.scatter(x[~is_atomic], y[~is_atomic], s=55, alpha=0.9, marker="s", label="combined")

    plt.xlabel(xlab)
    plt.ylabel("IoU")
    title = f"Size vs IoU ({xlab})"
    if supervision is not None:
        title += f" ({supervision})"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_error_contribution_bars(metrics_csv, out_png, *, supervision=None, which="fn", topk=15):
    """
    "What weighted the TOTAL?" plot.
    Shows top contributors to FN or FP across per-crack rows.
    """
    if not os.path.exists(metrics_csv):
        return

    df = pd.read_csv(metrics_csv)

    if supervision is not None and "supervision" in df.columns:
        df = df[df["supervision"].astype(str) == str(supervision)].copy()

    # only per-crack rows
    if "crack_type" in df.columns:
        df = df[df["crack_type"].isin(["atomic", "combined"])].copy()

    if df.empty or which not in df.columns:
        return

    df[which] = pd.to_numeric(df[which], errors="coerce")
    df = df[np.isfinite(df[which].values)]
    if df.empty:
        return

    total = float(df[which].sum())
    if total <= 0:
        return

    # label = type + id
    if "crack_id" in df.columns:
        labels = (df["crack_type"].astype(str) + ":" + df["crack_id"].astype(str)).values
    else:
        labels = df["crack_type"].astype(str).values

    df2 = df.copy()
    df2["label"] = labels
    df2["share"] = df2[which] / total
    df2 = df2.sort_values("share", ascending=False).head(int(topk))

    plt.figure(figsize=(8, 4.8))
    plt.barh(df2["label"].values[::-1], (100.0 * df2["share"].values[::-1]))
    plt.xlabel(f"% of total {which.upper()}")
    title = f"Top contributors to {which.upper()}"
    if supervision is not None:
        title += f" ({supervision})"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_under_overfill_scatter(metrics_csv, out_png, *, supervision=None):
    """
    Scatter: underfill_rate vs overfill_rate, point size ~ gt_area.
    Great for seeing "missed thin cracks" vs "oversegmentation".
    """
    if not os.path.exists(metrics_csv):
        return

    df = pd.read_csv(metrics_csv)

    if supervision is not None and "supervision" in df.columns:
        df = df[df["supervision"].astype(str) == str(supervision)].copy()

    if "crack_type" in df.columns:
        df = df[df["crack_type"].isin(["atomic", "combined"])].copy()

    if df.empty:
        return

    df = _add_mask_derived_cols(df)
    req = ["underfill_rate", "overfill_rate", "gt_area_px"]
    if not all(c in df.columns for c in req):
        return

    x = pd.to_numeric(df["underfill_rate"], errors="coerce").values.astype(float)
    y = pd.to_numeric(df["overfill_rate"], errors="coerce").values.astype(float)
    s = pd.to_numeric(df["gt_area_px"], errors="coerce").values.astype(float)

    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(s)
    x, y, s = x[keep], y[keep], s[keep]
    if len(x) == 0:
        return

    # normalize marker sizes
    s = np.clip(s, 1.0, np.percentile(s, 95))
    s = 25.0 + 175.0 * (s - s.min()) / (s.max() - s.min() + 1e-9)

    ctype = df.loc[keep, "crack_type"].astype(str).values
    is_atomic = (ctype == "atomic")

    plt.figure(figsize=(6.2, 5.2))
    plt.scatter(x[is_atomic], y[is_atomic], s=s[is_atomic], alpha=0.8, marker="o", label="atomic")
    plt.scatter(x[~is_atomic], y[~is_atomic], s=s[~is_atomic], alpha=0.8, marker="s", label="combined")

    plt.xlabel("Underfill rate (FN / GT area)")
    plt.ylabel("Overfill rate (FP / Pred area)")
    title = "Underfill vs Overfill"
    if supervision is not None:
        title += f" ({supervision})"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def plot_boundary_pr_curve(metrics_csv, out_png):
    import numpy as np, pandas as pd, matplotlib.pyplot as plt
    if not os.path.exists(metrics_csv): return

    df = pd.read_csv(metrics_csv)
    if "boundary_precision" not in df or "boundary_recall" not in df:
        return

    prec = df["boundary_precision"].astype(float)
    rec  = df["boundary_recall"].astype(float)

    plt.figure(figsize=(5,5))
    plt.plot(rec, prec, "o-", alpha=0.8)
    plt.xlabel("Boundary Recall")
    plt.ylabel("Boundary Precision")
    plt.title("Boundary Precision–Recall Curve")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def plot_midline_angle_distribution(midline_csv, out_png):
    import numpy as np, pandas as pd, matplotlib.pyplot as plt
    if not os.path.exists(midline_csv): return
    df = pd.read_csv(midline_csv)
    if "mean_tan_angle_error_deg" not in df: return

    ang = df["mean_tan_angle_error_deg"].astype(float)

    plt.figure(figsize=(6,4))
    plt.hist(ang, bins=40, color="royalblue", alpha=0.8)
    plt.xlabel("Angle error (deg)")
    plt.ylabel("count")
    plt.title("Midline angle error distribution")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def plot_width_error_distribution(width_diffs_csv, out_png):
    import numpy as np, pandas as pd, matplotlib.pyplot as plt
    if not os.path.exists(width_diffs_csv): return
    df = pd.read_csv(width_diffs_csv)
    if "width_diff_px" not in df: return

    diffs = df["width_diff_px"].astype(float)

    plt.figure(figsize=(6,4))
    plt.hist(diffs, bins=50, alpha=0.8, color="salmon")
    plt.xlabel("Geodesic width − GT width (px)")
    plt.title("Width error distribution")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    
def plot_confusion_matrix(df, out_png):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not {"tp","fp","fn","tn"}.issubset(df.columns):
        print("[CONFMAT] Missing confusion columns")
        return

    tp = df["tp"].sum()
    fp = df["fp"].sum()
    fn = df["fn"].sum()
    tn = df["tn"].sum()

    cm = np.array([[tp, fn],
                   [fp, tn]], float)

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues",
                xticklabels=["Pred +","Pred -"],
                yticklabels=["GT +","GT -"],
                ax=ax)
    ax.set_title("Confusion matrix (aggregated)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def plot_crack_statistics_overview(metrics_dir, base_name, out_png):
    import os, pandas as pd, numpy as np
    import matplotlib.pyplot as plt

    csv = os.path.join(metrics_dir, f"{base_name}_midline_edge_metrics.csv")
    if not os.path.exists(csv):
        print("[CRACK_STATS] no midline metrics CSV")
        return

    df = pd.read_csv(csv)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    # Length histogram
    if "mid_length" in df:
        ax[0].hist(df["mid_length"], bins=20, color="steelblue")
        ax[0].set_title("Midline length distribution", fontweight="bold")

    # Curvature histogram
    if "curvature_mean" in df:
        ax[1].hist(df["curvature_mean"], bins=20, color="darkorange")
        ax[1].set_title("Midline curvature distribution", fontweight="bold")

    # Count atomic vs combined
    if "src" in df:
        counts = df["src"].value_counts()
        ax[2].bar(counts.index, counts.values)
        ax[2].set_title("Atomic vs Combined count", fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

##################################################################
# New Metrics Plots
#################################################################

def build_deck_plots_for_image(metrics_dir: str, base_name: str):
    import os, pandas as pd

    metrics_csv = os.path.join(metrics_dir, "mask_metrics.csv")
    midline_csv = os.path.join(metrics_dir, f"{base_name}_midline_edge_metrics.csv")

    if not os.path.exists(metrics_csv):
        print("[DEBUG PLOT] no mask_metrics.csv")
        return

    df_all = pd.read_csv(metrics_csv)

    if "supervision" not in df_all.columns:
        print("[DEBUG PLOT] mask_metrics.csv missing supervision column")
        return

    # ---------------------------
    # variant-specific plots (supervision:method)
    # ---------------------------
    if "method" not in df_all.columns:
        df_all["method"] = "geodesic"

    variants = (
        df_all[["supervision", "method"]]
        .dropna(how="all")
        .drop_duplicates()
        .to_dict("records")
    )

    for vm in variants:
        supervision = str(vm.get("supervision", "unknown"))
        method = str(vm.get("method", "geodesic"))
        tag = f"{supervision}_{method}".replace(" ", "_").replace("/", "_").replace("\\", "_")

        df = df_all[
            (df_all["supervision"].astype(str) == supervision)
            & (df_all["method"].astype(str) == method)
        ].copy()
        if df.empty:
            continue

        subdir = os.path.join(metrics_dir, tag)
        os.makedirs(subdir, exist_ok=True)

        csv_sub = os.path.join(subdir, f"{tag}_mask_metrics.csv")
        df.to_csv(csv_sub, index=False)

        print(f"[DEBUG PLOT] building plots for {tag}")

        plot_iou_vs_bf1_scatter(
            csv_sub,
            os.path.join(subdir, f"{tag}_iou_vs_bf1_scatter.png"),
            supervision=None,  # CSV is already filtered to one variant.
        )

        plot_assd_hd95_box(
            csv_sub,
            os.path.join(subdir, f"{tag}_assd_hd95_box.png"),
        )

        plot_mask_metrics_triplet(
            subdir, base_name,
            tag,
            os.path.join(subdir, f"{tag}_mask_metrics_triplet.png"),
        )
        plot_total_mask_metrics_card(
            csv_sub,
            os.path.join(subdir, f"{tag}_total_mask_metrics_card.png"),
        )

        plot_surface_distance_histogram(
            csv_sub,
            os.path.join(subdir, f"{tag}_surface_distance_histogram.png"),
            supervision=None,
        )

        # Diagnostic per-crack plots only make sense if atomic/combined rows exist.
        has_per_crack = False
        if "crack_type" in df.columns:
            has_per_crack = df["crack_type"].isin(["atomic", "combined"]).any()

        if has_per_crack:
            plot_size_vs_iou_scatter(
                csv_sub,
                os.path.join(subdir, f"{tag}_size_vs_iou_gt_area.png"),
                supervision=None,
                x_mode="gt_area_px",   # works immediately
            )

            plot_under_overfill_scatter(
                csv_sub,
                os.path.join(subdir, f"{tag}_underfill_vs_overfill.png"),
                supervision=None,
            )

            plot_error_contribution_bars(
                csv_sub,
                os.path.join(subdir, f"{tag}_fn_contribution.png"),
                supervision=None,
                which="fn",
                topk=15,
            )

            plot_error_contribution_bars(
                csv_sub,
                os.path.join(subdir, f"{tag}_fp_contribution.png"),
                supervision=None,
                which="fp",
                topk=15,
            )

    # ---------------------------
    # non-supervision plots
    # ---------------------------
    if os.path.exists(midline_csv):
        plot_midline_edge_metrics_bars(
            midline_csv,
            os.path.join(metrics_dir, f"{base_name}_midline_edge_metrics_bars.png"),
        )

        plot_midline_angle_distribution(
            midline_csv,
            os.path.join(metrics_dir, f"{base_name}_midline_angle_hist.png"),
        )

        plot_crack_statistics_overview(
            metrics_dir, base_name,
            os.path.join(metrics_dir, f"{base_name}_crack_statistics.png"),
        )


########################################################
# Width differences plots
########################################################
def plot_width_diff_histogram(width_diffs_csv, out_png, title=None, bins=60, vlim=None):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    if not os.path.exists(width_diffs_csv):
        print("[WIDTH HIST] missing:", width_diffs_csv)
        return

    df = pd.read_csv(width_diffs_csv)
    if df.empty:
        print("[WIDTH HIST] empty:", width_diffs_csv)
        return

    # permissive column detection
    cols = {c.lower(): c for c in df.columns}
    diff_col = None
    for k in cols:
        if "diff" in k:
            diff_col = cols[k]
            break

    gt_col = next((cols[k] for k in cols if k in ("gt", "gt_width", "gt_width_px")), None)
    pred_col = next((cols[k] for k in cols if k in ("geodesic", "pred", "pred_width", "pred_width_px")), None)

    if diff_col is not None:
        diffs = df[diff_col].astype(float).values
    elif gt_col is not None and pred_col is not None:
        gt = df[gt_col].astype(float).values
        pred = df[pred_col].astype(float).values
        diffs = pred - gt
    else:
        print("[WIDTH HIST] could not find diff or gt/pred columns in:", width_diffs_csv)
        return

    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        print("[WIDTH HIST] no finite diffs:", width_diffs_csv)
        return

    if vlim is not None:
        diffs = np.clip(diffs, -float(vlim), float(vlim))

    plt.figure(figsize=(6, 4))
    plt.hist(diffs, bins=bins, alpha=0.85)
    plt.axvline(0.0, linewidth=1)
    plt.xlabel("Pred width − GT width (px)")
    plt.ylabel("count")
    plt.title(title or "Width diff histogram")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    print("[WIDTH HIST] wrote:", out_png)

'''def plot_width_summary_bars(metrics_dir, base_name, out_png):
    """
    Bar chart comparing TOTAL width metrics across methods
    (e.g. manual vs auto).

    Expects:
      metrics_dir/manual/<base>_width_summary_TOTAL.csv
      metrics_dir/auto/<base>_width_summary_TOTAL.csv
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    rows = []

    for method in ("manual", "auto"):
        p = os.path.join(
            metrics_dir,
            method,
            f"{base_name}_width_summary_total.csv"
        )
        if not os.path.exists(p):
            continue

        df = pd.read_csv(p)
        if df.empty:
            continue

        r = df.iloc[0].to_dict()
        r["method"] = method
        rows.append(r)

    if not rows:
        print("[WIDTH BAR] no total summaries found")
        return

    d = pd.DataFrame(rows)

    metrics = ["mae_px", "rmse_px", "bias_px", "corr"]
    labels  = ["MAE", "RMSE", "Bias", "Corr"]

    x = np.arange(len(metrics))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))

    for i, row in d.iterrows():
        vals = [row[m] for m in metrics]
        ax.bar(x + i * w, vals, w, label=row["method"])

    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("px / corr")
    ax.set_title(f"Crack {base_name} width error summary")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

    print("[WIDTH BAR] wrote:", out_png)'''

def plot_gt_width_vs_delta_w_scatter(
    diffs_csv,
    out_png,
    *,
    title=None,
    alpha=0.25,
    s=8,
    max_points=200_000,
):
    """
    Scatter: GT width (x) vs delta width (pred - GT) (y)

    Intended use:
      - diagnostic scaling behavior
      - thin cracks vs thick cracks stability

    diffs_csv must contain:
      - gt_width_px
      - width_diff_px
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    if not os.path.exists(diffs_csv):
        print("[WIDTH SCATTER] missing:", diffs_csv)
        return

    df = pd.read_csv(diffs_csv)
    if df.empty:
        print("[WIDTH SCATTER] empty CSV")
        return

    # permissive column detection
    cols = {c.lower(): c for c in df.columns}

    gt_col = next((cols[k] for k in cols if k in ("gt_width_px", "gt_width", "gt")), None)
    diff_col = next((cols[k] for k in cols if "diff" in k), None)

    if gt_col is None or diff_col is None:
        print("[WIDTH SCATTER] required columns missing")
        return

    gt = df[gt_col].astype(float).values
    dw = df[diff_col].astype(float).values

    keep = np.isfinite(gt) & np.isfinite(dw)
    gt, dw = gt[keep], dw[keep]

    if gt.size == 0:
        print("[WIDTH SCATTER] no valid samples")
        return

    # subsample for sanity if massive
    if gt.size > max_points:
        idx = np.random.choice(gt.size, max_points, replace=False)
        gt = gt[idx]
        dw = dw[idx]

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(gt, dw, s=s, alpha=alpha)

    ax.axhline(0.0, color="k", lw=1, ls="--", alpha=0.6)

    ax.set_xlabel("GT width (px)")
    ax.set_ylabel("Δ width = pred − GT (px)")

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

    print("[WIDTH SCATTER] wrote:", out_png)
    
def plot_relative_width_error_kde(
    diffs_csv,
    out_png,
    label=None,
    eps=1e-3,
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv(diffs_csv)

    gt = df["gt_width_px"].astype(float).values
    diff = df["width_diff_px"].astype(float).values

    keep = np.isfinite(gt) & np.isfinite(diff) & (gt > 0)
    gt, diff = gt[keep], diff[keep]

    rel = np.abs(diff) / np.maximum(gt, eps)

    plt.figure(figsize=(6, 4))
    sns.kdeplot(
        rel,
        fill=True,
        bw_adjust=1.2,
        label=label,
    )

    plt.xlabel("Relative width error  |Δw| / GT width")
    plt.ylabel("Density")
    plt.title("Relative width error distribution")
    if label:
        plt.legend()

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

def plot_width_error_hexbin(
    diffs_csv,
    out_png,
    gridsize=40,
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(diffs_csv)

    gt = df["gt_width_px"].astype(float).values
    diff = np.abs(df["width_diff_px"].astype(float).values)

    keep = np.isfinite(gt) & np.isfinite(diff)
    gt, diff = gt[keep], diff[keep]

    plt.figure(figsize=(6, 5))

    hb = plt.hexbin(
        gt,
        diff,
        gridsize=gridsize,
        bins="log",
        cmap="viridis",
        mincnt=1,
    )

    cb = plt.colorbar(hb)
    cb.set_label("log10(N)")

    plt.xlabel("GT width (px)")
    plt.ylabel("|Pred − GT| width (px)")
    plt.title("Width error vs crack thickness")

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()







#########################################################
# Edge parameter sweep
#########################################################

def plot_edge_sweep_summary(
    df,
    out_png,
    *,
    weight_col="global_weight",
    selected_family=None,
    hd95_guardrail=None,
):
    """
    Thesis-grade edge calibration diagnostic (FULLY EXPLAINABLE).

    Panel A: Pareto space (RAW sweep points)
      - each dot = one (crack, parameter family) evaluation
      - color = parameter family (explicitly labeled by parameters)
      - small text = crack id
      - black X = weighted mean of each parameter family

    Panel B: Edge score decomposition (parameter families)
      - stacked bars = WEIGHTED mean penalty terms
      - x labels explicitly show (w, μ, l, p, mode)

    This plot is ONLY valid when df is RAW sweep output (pre-selection).
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # ------------------------------------------------------------
    # Required columns
    # ------------------------------------------------------------
    required = {
        "crack_id",
        "boundary_f1",
        "ASSD",
        "HD95",
        "edge_score",
        "param_window_half_size",
        "param_mu",
        "param_l",
        "param_p",
        "param_seg_mode",
    }
    if not required.issubset(df.columns):
        print("[plot_edge_sweep_summary] ❌ missing required columns")
        print("Have:", sorted(df.columns))
        return

    D = df.copy()

    # ------------------------------------------------------------
    # Numeric coercion
    # ------------------------------------------------------------
    for c in ["boundary_f1", "ASSD", "HD95", "edge_score"]:
        D[c] = pd.to_numeric(D[c], errors="coerce")

    if weight_col in D.columns:
        D[weight_col] = pd.to_numeric(D[weight_col], errors="coerce").fillna(1.0)
        D[weight_col] = np.maximum(D[weight_col].values.astype(float), 1e-9)
    else:
        D[weight_col] = 1.0

    if hd95_guardrail is not None:
        D = D[D["HD95"] <= float(hd95_guardrail)]

    D = D.dropna(subset=["boundary_f1", "ASSD", "HD95", "edge_score"])
    if D.empty:
        print("[plot_edge_sweep_summary] ❌ empty dataframe after filtering")
        return

    fam_cols = [
        "param_window_half_size",
        "param_mu",
        "param_l",
        "param_p",
        "param_seg_mode",
    ]

    # ------------------------------------------------------------
    # Penalties (MATCH edge_score exactly)
    # ------------------------------------------------------------
    assd_med = float(D["ASSD"].median()) + 1e-9
    hd95_med = float(D["HD95"].median()) + 1e-9

    D["penalty_bf1"]  = (1.0 - D["boundary_f1"])
    D["penalty_assd"] = 0.50 * (D["ASSD"] / assd_med)
    D["penalty_hd95"] = 0.25 * (D["HD95"] / hd95_med)

    # ------------------------------------------------------------
    # Weighted mean helper
    # ------------------------------------------------------------
    def _wmean(x, w):
        x = np.asarray(x, float)
        w = np.asarray(w, float)
        ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
        if not np.any(ok):
            return np.nan
        return float(np.average(x[ok], weights=w[ok]))

    # ------------------------------------------------------------
    # Aggregate per parameter family (WEIGHTED)
    # ------------------------------------------------------------
    rows = []
    for key, g in D.groupby(fam_cols):
        w = g[weight_col].values
        rows.append({
            **dict(zip(fam_cols, key)),
            "bf1_wmean":  _wmean(g["boundary_f1"], w),
            "ASSD_wmean": _wmean(g["ASSD"], w),
            "penalty_bf1":  _wmean(g["penalty_bf1"], w),
            "penalty_assd": _wmean(g["penalty_assd"], w),
            "penalty_hd95": _wmean(g["penalty_hd95"], w),
            "edge_score_wmean": _wmean(g["edge_score"], w),
            "n_rows": len(g),
            "n_cracks": g["crack_id"].nunique(),
        })

    P = pd.DataFrame(rows).sort_values("edge_score_wmean").reset_index(drop=True)

    # ------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(14.0, 5.2),
        gridspec_kw=dict(width_ratios=[1.15, 1.45]),
    )

    # ============================================================
    # PANEL A — RAW Pareto space (NO aggregation hiding)
    # ============================================================
    fam_keys = list(P[fam_cols].itertuples(index=False, name=None))
    colors = plt.cm.tab10(np.linspace(0, 1, len(fam_keys)))
    fam_to_color = dict(zip(fam_keys, colors))

    legend_items = []

    for key in fam_keys:
        g = D.loc[(D[fam_cols] == pd.Series(key, index=fam_cols)).all(axis=1)]
        c = fam_to_color[key]

        ax1.scatter(
            g["boundary_f1"],
            g["ASSD"],
            s=55,
            alpha=0.6,
            color=c,
            edgecolor="none",
        )

        # crack id annotations
        for _, r in g.iterrows():
            ax1.annotate(
                str(int(r["crack_id"])),
                (r["boundary_f1"], r["ASSD"]),
                xytext=(3, 2),
                textcoords="offset points",
                fontsize=7,
                alpha=0.6,
            )

        # weighted family mean (THIS IS WHAT IS SELECTED)
        w = g[weight_col].values
        bx = _wmean(g["boundary_f1"], w)
        ay = _wmean(g["ASSD"], w)
        ax1.scatter(bx, ay, marker="x", s=150, linewidths=2.6, color="black")

        label = (
            f"w={key[0]}, μ={key[1]}, l={key[2]}, "
            f"p={key[3]}, mode={key[4]}"
        )
        legend_items.append(
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=c, markersize=8,
                   label=label)
        )

    legend_items.append(
        Line2D([0], [0], marker="x", color="black",
               markersize=9, linewidth=0,
               label="Weighted family mean")
    )

    ax1.legend(handles=legend_items, fontsize=11, loc="best", frameon=True)

    ax1.set_xlabel("Boundary F1 ↑")
    ax1.set_ylabel("ASSD (px) ↓")
    ax1.set_title("Pareto space (raw sweep points)")
    ax1.grid(True, alpha=0.25)

    # ============================================================
    # PANEL B — Edge score decomposition (parameter families)
    # ============================================================
    x = np.arange(len(P))

    ax2.bar(x, P["penalty_bf1"], label="Boundary error", color="#d62728")
    ax2.bar(
        x,
        P["penalty_assd"],
        bottom=P["penalty_bf1"],
        label="ASSD penalty",
        color="#ff7f0e",
    )
    ax2.bar(
        x,
        P["penalty_hd95"],
        bottom=P["penalty_bf1"] + P["penalty_assd"],
        label="HD95 penalty",
        color="#9467bd",
    )

    labels = [
        f"w={int(r.param_window_half_size)} μ={r.param_mu}\n"
        f"l={int(r.param_l)}, p={int(r.param_p)}\n"
        f"mode={r.param_seg_mode} (cracks={r.n_cracks})"
        for r in P.itertuples()
    ]

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=90, fontsize=8)
    ax2.set_ylabel("Weighted mean normalized penalty")
    ax2.set_title("Edge score decomposition (parameter families)")
    ax2.legend(fontsize=11)
    ax2.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "Edge calibration: multi-objective tradeoff and score justification",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(out_png)
    plt.close(fig)
   
def plot_from_cached_geometry(
    geom_npz,
    img_crop_gray,
    gt_crop=None,
    original_image=None,
    out_dir=".",
):
    """
    Plot overlays from cached geometry without recomputation.
    """
    import os
    import numpy as np
    import cv2

    from helpers.plot_metrics import (
        plot_gt_normals_on_gtbw,
    )
    from edge_workers import (
        plot_normals_pretty,
        plot_widths_colormap_on_crop,
    )
    from helpers.metrics import normals_from_mask_for_midline
    from helpers.plot_metrics import save_gt_vs_manual_overlay

    os.makedirs(out_dir, exist_ok=True)

    data = np.load(geom_npz)

    mask_crop        = data["mask_crop"]
    track_e1         = data["track_e1"]
    track_e2         = data["track_e2"]
    normals_e1       = data["normals_e1"]
    normals_e2       = data["normals_e2"]
    midline_xy_crop  = data["midline_xy_crop"]
    x, y, w, h       = data["bbox"].tolist()

    img_norm = cv2.normalize(
        img_crop_gray.astype(np.float32),
        None, 0, 255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # ---- pretty edges ----
    plot_normals_pretty(
        img_norm,
        track_e1,
        track_e2,
        midline_xy_crop,
        normals_e1,
        normals_e2,
        os.path.join(out_dir, "edges_midlines_normals_pretty.png"),
        cid="cached",
    )

    # ---- widths ----
    plot_widths_colormap_on_crop(
        gt_vs_manual_rgb=cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB),
        e1=normals_e1,
        e2=normals_e2,
        midline_xy=midline_xy_crop,
        track_e1=track_e1,
        track_e2=track_e2,
        out_png=os.path.join(out_dir, "widths_colormap_on_crop.png"),
    )

    # ---- GT overlays ----
    if gt_crop is not None:
        gt_bin = (gt_crop > 0).astype(np.uint8)

        (e1x, e1y, e2x, e2y, _), _ = normals_from_mask_for_midline(
            midline_xy_crop,
            gt_bin > 0,
            max_radius=50,
            image_hw=gt_bin.shape[:2],
        )

        plot_gt_normals_on_gtbw(
            gt_bin * 255,
            midline_xy_crop,
            None,
            np.column_stack([e1x, e1y]),
            np.column_stack([e2x, e2y]),
            os.path.join(out_dir, "gt_normals.png"),
        )

        if original_image is not None:
            H, W = original_image.shape[:2]
            pred_full = np.zeros((H, W), np.uint8)
            pred_full[y:y+h, x:x+w] = mask_crop > 0

            save_gt_vs_manual_overlay(
                H, W,
                gt_bin,
                pred_full,
                os.path.join(out_dir, "gt_vs_manual_mask_global.png"),
                bbox=[x, y, w, h],
                original_image=original_image,
            )







##################################################################
# Auto variants plots
##################################################################
def _generate_rs3_geometry_panels(
    D,
    save_folder,
    image_base,
    out_dir,
):
    """
    Generates:
        rs3_geometry_representative.png
        rs3_geometry_decisive.png
    """
    import pandas as pd

    if D is None or getattr(D, "empty", True):
        return

    if "crack_id" not in D.columns:
        return

    if "length_px" in D.columns:
        cid_rep = (
            D.groupby("crack_id")["length_px"]
            .mean()
            .sort_values(ascending=False)
            .index[0]
        )
    else:
        cid_rep = sorted(D["crack_id"].unique())[0]

    best_per_crack = (
        D.groupby(["crack_id", "family"])["score_mid"]
        .min()
        .reset_index()
    )

    margins = []
    for cid, g in best_per_crack.groupby("crack_id"):
        g = g.sort_values("score_mid")
        if len(g) >= 2:
            margin = float(g.iloc[1]["score_mid"] - g.iloc[0]["score_mid"])
            margins.append((cid, margin))

    if margins:
        cid_dec = sorted(margins, key=lambda x: -x[1])[0][0]
    else:
        cid_dec = cid_rep

    try:
        _plot_single_crack_geometry(
            cid_rep,
            D,
            save_folder,
            image_base,
            out_dir,
            filename="rs3_geometry_representative.png",
        )
    except Exception as e:
        print(f"[plot_rs3] representative geometry skipped: {e}")

    try:
        _plot_single_crack_geometry(
            cid_dec,
            D,
            save_folder,
            image_base,
            out_dir,
            filename="rs3_geometry_decisive.png",
        )
    except Exception as e:
        print(f"[plot_rs3] decisive geometry skipped: {e}")


def _plot_single_crack_geometry(
    cid,
    D,
    save_folder,
    image_base,
    out_dir,
    filename,
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from helpers.metrics import metric_atomic_path_for, safe_read_json

    crack_rows = D[D["crack_id"] == cid].copy()
    if crack_rows.empty:
        return

    # ---------------------------------------------------------
    # Load manual midline (required)
    # ---------------------------------------------------------
    p_cr = metric_atomic_path_for(save_folder, image_base, cid)
    crack_json = safe_read_json(p_cr, None)
    if not crack_json:
        raise RuntimeError(f"Missing crack json for crack {cid}: {p_cr}")

    manual_xy = np.asarray(crack_json.get("midline", []), float)
    if manual_xy.ndim != 2 or manual_xy.shape[1] != 2 or len(manual_xy) < 2:
        raise RuntimeError(f"Invalid manual midline for crack {cid}")

    metrics_root = os.path.join(save_folder, "metrics", image_base)
    auto_dir = os.path.join(metrics_root, f"cid{cid}", "auto")

    # ---------------------------------------------------------
    # Variant loader
    # ---------------------------------------------------------
    def load_variant(mode, g22):
        row = crack_rows[
            (crack_rows["os_mode"] == mode) &
            (crack_rows["g11"] == 1.0) &
            (crack_rows["g22"] == g22) &
            (crack_rows["g33"] == g22)
        ]
        if row.empty:
            return None

        row0 = row.sort_values("score_mid", ascending=True).iloc[0]
        vid = int(row0["variant_global_id"])
        vfile = os.path.join(auto_dir, f"v{vid}.json")
        if not os.path.exists(vfile):
            return None

        vjson = safe_read_json(vfile, {})
        if "midline" in vjson:
            xy = np.asarray(vjson["midline"], float)
        elif isinstance(vjson.get("auto_best"), dict) and "midline" in vjson["auto_best"]:
            xy = np.asarray(vjson["auto_best"]["midline"], float)
        else:
            return None

        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2:
            return None

        return xy

    # ---------------------------------------------------------
    # Load required baseline: old,1,100,100
    # ---------------------------------------------------------
    xy_old_100 = load_variant("old", 100.0)
    if xy_old_100 is None:
        raise RuntimeError(
            f"Missing required baseline old,1,100,100 for crack {cid}"
        )

    # ---------------------------------------------------------
    # Collect all NEW variants (one subplot per new)
    # ---------------------------------------------------------
    new_variants = []

    # new,100
    xy_new_100 = load_variant("new", 100.0)
    if xy_new_100 is not None:
        new_variants.append(("new,1,100,100", xy_new_100))

    # flexible new variants
    flex_rows = crack_rows[
        (crack_rows["os_mode"] == "new") &
        (crack_rows["g22"] != 100.0)
    ].sort_values("g22")

    for _, row in flex_rows.iterrows():
        g22 = float(row["g22"])
        xy = load_variant("new", g22)
        if xy is None:
            continue
        label = f"new,1,{int(g22)},{int(g22)}"
        new_variants.append((label, xy))

    if len(new_variants) == 0:
        raise RuntimeError(f"No new variants available for crack {cid}")

    # ---------------------------------------------------------
    # Subplot grid
    # ---------------------------------------------------------
    n = len(new_variants)

    # Flexible layout
    if n <= 4:
        ncols = n
        nrows = 1
    else:
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))
    
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4 * ncols, 4 * nrows),
    )

    if nrows == 1:
        axes = np.atleast_1d(axes)
    axes = axes.flatten()

    # ---------------------------------------------------------
    # Fix global axis limits (fair comparison)
    # ---------------------------------------------------------
    all_x = [manual_xy[:, 0], xy_old_100[:, 0]]
    all_y = [manual_xy[:, 1], xy_old_100[:, 1]]

    for _, xy in new_variants:
        all_x.append(xy[:, 0])
        all_y.append(xy[:, 1])

    xmin = min(np.min(x) for x in all_x)
    xmax = max(np.max(x) for x in all_x)
    ymin = min(np.min(y) for y in all_y)
    ymax = max(np.max(y) for y in all_y)

    # ---------------------------------------------------------
    # Plot each new variant against baseline
    # ---------------------------------------------------------
    for ax, (label, xy_new) in zip(axes, new_variants):

        # Manual
        ax.plot(
            manual_xy[:, 0],
            manual_xy[:, 1],
            color="black",
            linewidth=2.8,
            label="Manual",
            alpha=.85
        )

        # Old baseline
        ax.plot(
            xy_old_100[:, 0],
            xy_old_100[:, 1],
            linestyle="--",
            linewidth=1.8,
            label="old,1,100,100",
            alpha=.85
        )

        # Current new variant
        ax.plot(
            xy_new[:, 0],
            xy_new[:, 1],
            linewidth=2.0,
            label=label,
            alpha=.85
        )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)  # inverted
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=9)
        ax.legend(fontsize=11, framealpha=0.9)

    # Hide unused axes
    for ax in axes[len(new_variants):]:
        ax.axis("off")

    fig.suptitle(f"RS3 sweep — crack {cid}", fontsize=12)
    fig.tight_layout()

    fig.savefig(
        os.path.join(out_dir, filename),
        bbox_inches="tight",
    )
    plt.close(fig)

def plot_rs3_sweep_summary(
    df_all,
    out_dir,
    *,
    weight_col="global_weight",
    selected_family=None,
    save_folder=None,
    image_base=None,
):
    """
    Thesis-grade RS3 auto-variant calibration summary.

    Produces:
      1) rs3_pareto.png
      2) rs3_decomposition.png
      3) rs3_family_crack_<id>.png
      4) rs3_family_win_count.png
      5) rs3_os_compare.png
    """

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch

    os.makedirs(out_dir, exist_ok=True)

    # --------------------------------------------------
    # Required columns
    # --------------------------------------------------
    req = {
        "crack_id",
        "os_mode",
        "g11", "g22", "g33",
        "score_mid",
        "nn_mean_bidirectional",
        "hausdorff_max",
        "coverage_min",
        weight_col,
    }
    if not req.issubset(df_all.columns):
        print("[plot_rs3] ❌ missing columns:", sorted(req - set(df_all.columns)))
        return

    D = df_all.copy()

    # --------------------------------------------------
    # Numeric coercion
    # --------------------------------------------------
    for c in ["score_mid", "nn_mean_bidirectional", "hausdorff_max", "coverage_min", weight_col]:
        D[c] = pd.to_numeric(D[c], errors="coerce")

    D = D.dropna(subset=["score_mid", "nn_mean_bidirectional", "hausdorff_max"])
    if D.empty:
        print("[plot_rs3] ❌ empty after coercion")
        return

    # --------------------------------------------------
    # Human-readable family label
    # --------------------------------------------------
    def _fmt_intish(x):
        try:
            xf = float(x)
            if abs(xf - round(xf)) < 1e-9:
                return str(int(round(xf)))
            return f"{xf:g}"
        except Exception:
            return str(x)

    def format_family_raw(row):
        return (
            f"{str(row['os_mode']).lower()},"
            f"{_fmt_intish(row['g11'])},"
            f"{_fmt_intish(row['g22'])},"
            f"{_fmt_intish(row['g33'])}"
        )

    def _pretty_xtick(fam):
        parts = str(fam).split(",", 1)
        return parts[0] + "\n" + parts[1] if len(parts) == 2 else str(fam)

    D["family"] = D.apply(format_family_raw, axis=1)

    # --------------------------------------------------
    # Stable family IDs + infinite color palette
    # --------------------------------------------------
    families = sorted(D["family"].unique())
    family_id = {fam: f"F{i+1}" for i, fam in enumerate(families)}

    cmap = plt.cm.get_cmap("turbo", len(families))
    fam_color = {fam: cmap(i) for i, fam in enumerate(families)}

    D["family_id"] = D["family"].map(family_id)

    # --------------------------------------------------
    # Proper legend handles (CRITICAL FIX)
    # --------------------------------------------------
    legend_handles = [Patch(color=fam_color[f], label=f"{f}") for f in families]

    # ==================================================
    # 1) RAW PARETO SCATTER
    # ==================================================
    plt.figure(figsize=(7.0, 5.5))

    for fam, g in D.groupby("family"):
        plt.scatter(
            g["nn_mean_bidirectional"],
            g["hausdorff_max"],
            s=55,
            alpha=0.6,
            color=fam_color[fam],
        )

        for _, r in g.iterrows():
            plt.annotate(
                str(int(r["crack_id"])),
                (r["nn_mean_bidirectional"], r["hausdorff_max"]),
                xytext=(3, 2),
                textcoords="offset points",
                fontsize=7,
                alpha=0.6,
            )

        w = g[weight_col].values
        ok = np.isfinite(w) & (w > 0)
        if ok.any():
            bx = np.average(g["nn_mean_bidirectional"].values[ok], weights=w[ok])
            by = np.average(g["hausdorff_max"].values[ok], weights=w[ok])
            plt.scatter(bx, by, marker="x", s=160, linewidths=2.5, color="black")

    plt.xlabel("nn_mean_bidirectional mean ↓")
    plt.ylabel("Hausdorff ↓")
    plt.title("RS3 Pareto space (raw variants)")
    plt.grid(True, alpha=0.25)
    plt.legend(
        handles=legend_handles,
        fontsize=11,
        frameon=True,
        title="os_mode,g11,g22,g33",
        title_fontsize=8,
        loc="lower right",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rs3_pareto.png"), bbox_inches="tight")
    plt.close()

    # ==================================================
    # 2) SCORE DECOMPOSITION
    # ==================================================
    rows = []
    for fam, g in D.groupby("family"):
        w = g[weight_col].values
        ok = np.isfinite(w) & (w > 0)
        if not ok.any():
            continue

        rows.append({
            "family": fam,
            "log_nn_mean_bidirectional": np.average(np.log1p(g["nn_mean_bidirectional"].values[ok]), weights=w[ok]),
            "log_hausdorff_max": 0.5 * np.average(np.log1p(g["hausdorff_max"].values[ok]), weights=w[ok]),
            "one_minus_coverage_min": np.average((1.0 - g["coverage_min"].values[ok]), weights=w[ok]),
            "score": np.average(g["score_mid"].values[ok], weights=w[ok]),
        })

    P = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
    x = np.arange(len(P))

    plt.figure(figsize=(7.5, 4.5))
    plt.bar(x, P["log_nn_mean_bidirectional"], label="log(nn_mean_bidirectional)")
    plt.bar(x, P["log_hausdorff_max"], bottom=P["log_nn_mean_bidirectional"], label="0.5·log(Hausdorff)")
    plt.bar(
        x,
        P["one_minus_coverage_min"],
        bottom=P["log_nn_mean_bidirectional"] + P["log_hausdorff_max"],
        label="1 − coverage_min",
    )

    plt.xticks(x, [_pretty_xtick(f) for f in P["family"]], fontsize=8)
    plt.ylabel("Weighted mean score components")
    plt.title("RS3 score decomposition by family")
    plt.legend(fontsize=11)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rs3_decomposition.png"), bbox_inches="tight")
    plt.close()

    # ==================================================
    # 3) GEOMETRY PANELS (REPRESENTATIVE + DECISIVE)
    # ==================================================
    if save_folder is None or image_base is None:
        print("[plot_rs3] geometry skipped — missing explicit save_folder/image_base")
    else:
        _generate_rs3_geometry_panels(
            D,
            save_folder,
            image_base,
            out_dir,
        )

    best_per_crack = (
        D.groupby(["crack_id", "family"])["score_mid"]
        .min()
        .reset_index()
    )
    best_per_crack["family_id"] = best_per_crack["family"].map(family_id)

    # ==================================================
    # 4) FAMILY SELECTION FREQUENCY
    # ==================================================
    wins = (
        best_per_crack
        .loc[best_per_crack.groupby("crack_id")["score_mid"].idxmin()]
        .groupby("family")
        .size()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(3.8, 2.8))
    plt.bar(
        [_pretty_xtick(f) for f in wins.index],
        wins.values,
        color=[fam_color[f] for f in wins.index],
    )

    plt.ylabel("number of cracks won")
    plt.xlabel("RS3 family")
    plt.title("RS3 family selection frequency")
    plt.grid(axis="y", alpha=0.25)
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    if any(lbl and not str(lbl).startswith("_") for lbl in labels):
        plt.legend(
            fontsize=11,
            frameon=True,
            title="os_mode,g11,g22,g33",
            title_fontsize=8,
            loc="lower right",
        )
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "rs3_family_win_count.png"),
        bbox_inches="tight",
    )
    plt.close()

    # ==================================================
    # 5) OS-MODE ABLATION
    # ==================================================
    plt.figure(figsize=(5.0, 4.5))
    sns.boxplot(data=D, x="os_mode", y="score_mid", showfliers=False)
    sns.stripplot(
        data=D,
        x="os_mode",
        y="score_mid",
        color="black",
        alpha=0.5,
        size=4,
        jitter=True,
    )

    plt.ylabel("score_mid ↓")
    plt.title("RS3 OS-mode comparison")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rs3_os_compare.png"), bbox_inches="tight")
    plt.close()

    print(f"[plot_rs3] ✓ wrote RS3 summary plots to {out_dir}")

def plot_rs3_timing_summary(
    df_timing,
    out_dir,
    *,
    weight_col=None,
    selected_family=None,
    time_unit="sec",
):
    """
    RS3 TIMING BREAKDOWN (OS / COST / FAST MARCHING + SUBTIMINGS)

    Produces CLEAN, thesis-ready timing plots.

    Figures generated:
      1) Weighted-mean top-level timing
      2) Total top-level timing
      3) Cost subtiming (weighted mean)
      4) Cost subtiming (total)
      5) Fast-marching subtiming (weighted mean)
      6) Fast-marching subtiming (total)

    Timing NEVER affects selection.
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    if df_timing is None or df_timing.empty:
        print("[plot_rs3_timing_summary] ⚠ no timing data")
        return

    os.makedirs(out_dir, exist_ok=True)
    D = df_timing.copy()

    # ------------------------------------------------------------
    # Required columns
    # ------------------------------------------------------------
    required = {"crack_id", "os_sec", "cost_sec", "fm_total_sec"}
    missing = required - set(D.columns)
    if missing:
        print(f"[plot_rs3_timing_summary] ⚠ missing columns: {missing}")
        return

    # ------------------------------------------------------------
    # Filter to SELECTED FAMILY ONLY
    # ------------------------------------------------------------
    if selected_family is not None:
        os_mode, g11, g22, g33 = selected_family
        D = D[
            (D["os_mode"] == os_mode) &
            (D["g11"] == g11) &
            (D["g22"] == g22) &
            (D["g33"] == g33)
        ]

    if D.empty:
        print("[plot_rs3_timing_summary] ⚠ no rows after family filter")
        return

    # ------------------------------------------------------------
    # Aggregate per subcrack
    # ------------------------------------------------------------
    agg = []
    for cid, g in D.groupby("crack_id"):
        rec = {
            "crack_id": cid,
            "os_sec": float(np.nanmean(g["os_sec"])),
            "cost_sec": float(np.nanmean(g["cost_sec"])),
            "fm_total_sec": float(np.nansum(g["fm_total_sec"])),
        }
        if weight_col and weight_col in g.columns:
            rec[weight_col] = float(np.nanmean(g[weight_col]))
        agg.append(rec)

    A = pd.DataFrame(agg)
    if A.empty:
        return

    # ------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------
    if weight_col and weight_col in A.columns:
        w = np.maximum(A[weight_col].values.astype(float), 1e-9)
    else:
        w = np.ones(len(A), float)

    # ------------------------------------------------------------
    # Top-level stats
    # ------------------------------------------------------------
    os_mean   = float(np.average(A["os_sec"], weights=w))
    cost_mean = float(np.average(A["cost_sec"], weights=w))
    fm_mean   = float(np.average(A["fm_total_sec"], weights=w))

    os_sum   = float(A["os_sec"].sum())
    cost_sum = float(A["cost_sec"].sum())
    fm_sum   = float(A["fm_total_sec"].sum())

    # ------------------------------------------------------------
    # Helper: stacked bar with clean legend
    # ------------------------------------------------------------
    def _stacked_bar(outfile, title, parts):
        fig, ax = plt.subplots(figsize=(5.4, 4.8))
        bottom = 0.0
        for name, val, color in parts:
            ax.bar([0], [val], bottom=[bottom],
                   label=f"{name}={val:.3f}s", color=color)
            bottom += val

        ax.set_xticks([0])
        ax.set_xticklabels([title])
        ax.set_ylabel(f"Runtime ({time_unit})")
        ax.legend(fontsize=11)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(outfile)
        plt.close(fig)

    # ------------------------------------------------------------
    # 1) WEIGHTED MEAN (TOP LEVEL)
    # ------------------------------------------------------------
    _stacked_bar(
        os.path.join(out_dir, "rs3_timing_weighted_mean.png"),
        "Weighted Mean",
        [
            ("OS", os_mean, "#4C72B0"),
            ("Cost", cost_mean, "#55A868"),
            ("Fast marching", fm_mean, "#C44E52"),
        ],
    )

    # ------------------------------------------------------------
    # 2) TOTAL (TOP LEVEL)
    # ------------------------------------------------------------
    _stacked_bar(
        os.path.join(out_dir, "rs3_timing_total.png"),
        "Total",
        [
            ("OS", os_sum, "#4C72B0"),
            ("Cost", cost_sum, "#55A868"),
            ("Fast marching", fm_sum, "#C44E52"),
        ],
    )

    # ------------------------------------------------------------
    # COST SUBTIMINGS
    # ------------------------------------------------------------
    cost_cols = ["t_ms_vessel", "t_ms_filter", "t_cost_fun"]
    if all(c in D.columns for c in cost_cols):
        C = D.groupby("crack_id")[cost_cols].mean()
        wC = (
            D.groupby("crack_id")[weight_col].mean().values
            if weight_col and weight_col in D.columns
            else np.ones(len(C))
        )

        cost_mean_vals = {
            c: float(np.average(C[c].values, weights=wC))
            for c in cost_cols
        }
        cost_sum_vals = {
            c: float(D[c].sum())
            for c in cost_cols
        }

        _stacked_bar(
            os.path.join(out_dir, "rs3_cost_subtiming_weighted_mean.png"),
            "Weighted Mean",
            [
                ("t_ms_vessel", cost_mean_vals["t_ms_vessel"], "#4C72B0"),
                ("t_ms_filter", cost_mean_vals["t_ms_filter"], "#55A868"),
                ("t_cost_fun",  cost_mean_vals["t_cost_fun"],  "#C44E52"),
            ],
        )

        _stacked_bar(
            os.path.join(out_dir, "rs3_cost_subtiming_total.png"),
            "Sum",
            [
                ("t_ms_vessel", cost_sum_vals["t_ms_vessel"], "#4C72B0"),
                ("t_ms_filter", cost_sum_vals["t_ms_filter"], "#55A868"),
                ("t_cost_fun",  cost_sum_vals["t_cost_fun"],  "#C44E52"),
            ],
        )

    # ------------------------------------------------------------
    # FAST MARCHING SUBTIMINGS
    # ------------------------------------------------------------
    fm_cols = [
        "fm_metric_build_sec",
        "fm_include_cost_sec",
        "fm_transpose_sec",
        "fm_solver_sec",
    ]
    if all(c in D.columns for c in fm_cols):
        F = D.groupby("crack_id")[fm_cols].sum()
        wF = (
            D.groupby("crack_id")[weight_col].mean().values
            if weight_col and weight_col in D.columns
            else np.ones(len(F))
        )

        fm_mean_vals = {
            c: float(np.average(F[c].values, weights=wF))
            for c in fm_cols
        }
        fm_sum_vals = {
            c: float(D[c].sum())
            for c in fm_cols
        }

        _stacked_bar(
            os.path.join(out_dir, "rs3_fm_subtiming_weighted_mean.png"),
            "Weighted Mean",
            [
                ("fm_metric_build", fm_mean_vals["fm_metric_build_sec"], "#4C72B0"),
                ("fm_include_cost", fm_mean_vals["fm_include_cost_sec"], "#55A868"),
                ("fm_transpose",    fm_mean_vals["fm_transpose_sec"], "#C44E52"),
                ("fm_solver",       fm_mean_vals["fm_solver_sec"], "#8172B3"),
            ],
        )

        _stacked_bar(
            os.path.join(out_dir, "rs3_fm_subtiming_total.png"),
            "Sum",
            [
                ("fm_metric_build", fm_sum_vals["fm_metric_build_sec"], "#4C72B0"),
                ("fm_include_cost", fm_sum_vals["fm_include_cost_sec"], "#55A868"),
                ("fm_transpose",    fm_sum_vals["fm_transpose_sec"], "#C44E52"),
                ("fm_solver",       fm_sum_vals["fm_solver_sec"], "#8172B3"),
            ],
        )

    print(f"[plot_rs3_timing_summary] ✓ clean timing plots written to {out_dir}")


def plot_rs3_midline_diagnostics(
    df_all,
    out_dir,
    selected_family=None,
    title_suffix=None,
    compare_mode="all",
    group_cols=("midline_type", "geometry_type"),
    max_groups=10,
    include_diagnostic_metrics=True,
):
    """
    Midline diagnostic plotting.

    Modes:
      - compare_mode="all": legacy single-distribution plots across rows
      - compare_mode="grouped": grouped bars (mean±std) split by group_cols
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    df = df_all.copy()

    # ------------------------------------------------------------
    # Optional RS3 family filtering
    # ------------------------------------------------------------
    if selected_family is not None:
        fam_mask = (
            (df["os_mode"] == selected_family[0]) &
            (df["g11"] == selected_family[1]) &
            (df["g22"] == selected_family[2]) &
            (df["g33"] == selected_family[3])
        )
        df = df[fam_mask].copy()

    if df.empty:
        return

    primary_metrics = ["coverage_min", "nn_mean_bidirectional", "hausdorff_max", "score_mid"]
    primary_metrics = [m for m in primary_metrics if m in df.columns]

    diagnostic_metrics = [
        "relative_length_error",
        "orth_mean",
        "orth_std",
        "curvature_rms_ratio",
        "mean_tan_angle_error_deg",
        "frechet_discrete_ds",
    ]
    diagnostic_metrics = [m for m in diagnostic_metrics if m in df.columns]

    def _make_group_key(frame: pd.DataFrame) -> pd.Series:
        parts = []
        for c in group_cols:
            if c in frame.columns:
                parts.append(frame[c].astype(str).fillna(""))
        if not parts:
            return pd.Series(["all"] * len(frame), index=frame.index)
        key = parts[0]
        for p in parts[1:]:
            key = key + "|" + p
        return key

    def _plot_grouped_bars(metrics, filename, plot_title):
        if not metrics:
            return

        dfx = df.copy()
        dfx["_group"] = _make_group_key(dfx)
        grp_counts = dfx["_group"].value_counts()
        keep_groups = list(grp_counts.index[:max_groups])
        dfx = dfx[dfx["_group"].isin(keep_groups)].copy()
        if dfx.empty:
            return

        groups = sorted(list(dfx["_group"].unique()))
        means = np.zeros((len(groups), len(metrics)), float)
        stds = np.zeros((len(groups), len(metrics)), float)

        for gi, gname in enumerate(groups):
            dfg = dfx[dfx["_group"] == gname]
            for mi, m in enumerate(metrics):
                vals = pd.to_numeric(dfg[m], errors="coerce").dropna().values
                means[gi, mi] = float(np.mean(vals)) if len(vals) else np.nan
                stds[gi, mi] = float(np.std(vals)) if len(vals) else np.nan

        x = np.arange(len(metrics), dtype=float)
        n_groups = len(groups)
        width = 0.8 / max(1, n_groups)

        plt.figure(figsize=(max(8, 1.7 * len(metrics)), 4.8))
        for gi, gname in enumerate(groups):
            xpos = x - 0.4 + (gi + 0.5) * width
            plt.bar(xpos, means[gi], width=width, yerr=stds[gi], capsize=3, label=str(gname))

        plt.xticks(x, metrics, rotation=25, ha="right")
        plt.ylabel("value")

        title = plot_title
        if selected_family is not None:
            title = "RS3 " + title
        if title_suffix:
            title += f" — {title_suffix}"
        plt.title(title)
        plt.legend(fontsize=11, framealpha=0.9)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename))
        plt.close()

    def _plot_boxplots(metrics, filename, plot_title):
        if not metrics:
            return
        data, labels = [], []
        for m in metrics:
            vals = pd.to_numeric(df[m], errors="coerce").dropna().values
            if len(vals):
                data.append(vals)
                labels.append(m)
        if not data:
            return

        plt.figure(figsize=(1.8 * len(data), 4.4))
        # For tiny-N, boxplots collapse into short line glyphs; use scatter+mean instead.
        if max(len(v) for v in data) < 4:
            means = [float(np.mean(v)) for v in data]
            x = np.arange(1, len(labels) + 1, dtype=float)
            plt.bar(x, means, width=0.52, color="#4c78a8", alpha=0.75, edgecolor="black", linewidth=0.8)
            for i, vals in enumerate(data, start=1):
                xx = np.full(len(vals), float(i))
                plt.scatter(xx, vals, s=22, color="black", alpha=0.85, zorder=3)
            plt.xticks(x, labels, rotation=15, ha="right")
        else:
            plt.boxplot(data, labels=labels, showfliers=False)
        plt.ylabel("value")

        title = plot_title
        if selected_family is not None:
            title = "RS3 " + title
        if title_suffix:
            title += f" — {title_suffix}"

        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename))
        plt.close()

    if compare_mode == "grouped":
        _plot_grouped_bars(primary_metrics, "primary_midline_metrics_grouped.png", "Midline Primary Metrics (grouped)")
    else:
        _plot_boxplots(primary_metrics, "primary_midline_metrics.png", "Midline Primary Metrics")

    if include_diagnostic_metrics:
        if compare_mode == "grouped":
            _plot_grouped_bars(diagnostic_metrics, "diagnostic_bars_grouped.png", "Midline Diagnostic Metrics (grouped)")
        else:
            rows = []
            for m in diagnostic_metrics:
                vals = pd.to_numeric(df[m], errors="coerce").dropna().values
                if not len(vals):
                    continue
                rows.append({"metric": m, "mean": float(np.mean(vals)), "std": float(np.std(vals))})

            if rows:
                dfd = pd.DataFrame(rows)
                plt.figure(figsize=(1.6 * len(dfd), 4))
                plt.bar(dfd["metric"], dfd["mean"], yerr=dfd["std"], capsize=5)
                plt.ylabel("value")

                title = "Midline Diagnostic Metrics (distribution summary)"
                if selected_family is not None:
                    title = "RS3 " + title
                if title_suffix:
                    title += f" - {title_suffix}"

                plt.title(title)
                plt.xticks(rotation=30, ha="right")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "diagnostic_bars.png"))
                plt.close()


def plot_rs3_midline_decomposition(
    *,
    df_all,
    out_path,
    group_col="variant_id",
    title="Midline Metrics - Full Decomposition (All Metrics)",
):
    """
    Edge-decomposition style stacked chart for all midline metrics/group.
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    df = df_all.copy() if isinstance(df_all, pd.DataFrame) else pd.DataFrame()
    if df.empty or group_col not in df.columns:
        return

    primary = [
        "score_mid",
        "nn_mean_bidirectional",
        "hausdorff_max",
        "coverage_min",
    ]
    secondary = [
        "relative_length_error",
        "orth_mean",
        "orth_std",
        "curvature_rms_ratio",
        "mean_tan_angle_error_deg",
        "frechet_discrete_ds",
    ]
    metrics = [m for m in (primary + secondary) if m in df.columns]
    if not metrics:
        return

    agg = (
        df.groupby(group_col, dropna=False)[metrics]
        .mean(numeric_only=True)
        .reset_index()
    )
    if agg.empty:
        return

    labels = agg[group_col].astype(str).tolist()
    x = np.arange(len(labels), dtype=float)
    bottom = np.zeros(len(labels), float)

    fig_w = max(10.0, 1.2 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 6.0))
    cmap = plt.get_cmap("tab20")

    for i, m in enumerate(metrics):
        vals = pd.to_numeric(agg[m], errors="coerce").to_numpy(float)
        vals = np.where(np.isfinite(vals), vals, 0.0)
        ax.bar(
            x,
            vals,
            width=0.62,
            bottom=bottom,
            label=m,
            color=cmap(i % 20),
            alpha=0.88,
        )
        bottom += vals

    ax.plot(x, bottom, "ko", markersize=4, label="total")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("metric contribution (stacked)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=11, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_midline_length_score_relationship(
    df_all,
    out_dir,
    *,
    prefix="midline_length_score",
    length_col="length_px",
    score_col="score_mid",
    group_cols=("midline_type", "geometry_type", "os_mode", "variant_id"),
    bins=12,
    max_groups=10,
):
    """
    Export + plot the relationship between segment length and a midline score.

    Outputs:
      - <prefix>_points.csv
      - <prefix>_bins.csv
      - <prefix>_high_vs_low.csv
      - <prefix>_scatter.png
      - <prefix>_binned.png
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    if not isinstance(df_all, pd.DataFrame):
        try:
            df = pd.DataFrame(df_all)
        except Exception:
            return
    else:
        df = df_all.copy()

    if length_col not in df.columns or score_col not in df.columns:
        return

    df[length_col] = pd.to_numeric(df[length_col], errors="coerce")
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df[np.isfinite(df[length_col]) & np.isfinite(df[score_col])].copy()
    df = df[df[length_col] > 0].copy()
    if df.empty:
        return

    def _group_key(frame):
        keys = []
        for c in group_cols:
            if c in frame.columns:
                s = frame[c].astype(str).fillna("")
                s = s.replace("nan", "")
                keys.append(s)

        if not keys:
            return pd.Series(["all"] * len(frame), index=frame.index)

        out = keys[0].copy()
        for s in keys[1:]:
            out = np.where(
                (out != "") & (s != ""),
                out + "|" + s,
                np.where(out != "", out, s),
            )
        out = pd.Series(out, index=frame.index)
        out = out.replace("", "all")
        return out

    df["_group"] = _group_key(df)
    grp_counts = df["_group"].value_counts()
    keep_groups = list(grp_counts.index[: max(1, int(max_groups))])
    if not keep_groups:
        keep_groups = ["all"]
        df["_group"] = "all"
    else:
        df["_group"] = np.where(df["_group"].isin(keep_groups), df["_group"], "other")
        if "other" in set(df["_group"]):
            keep_groups = keep_groups + ["other"]

    points_csv = os.path.join(out_dir, f"{prefix}_points.csv")
    df.to_csv(points_csv, index=False)

    # Continuous view: scatter
    try:
        plt.figure(figsize=(8.0, 5.2))
        for g in keep_groups:
            sub = df[df["_group"] == g]
            if sub.empty:
                continue
            if len(sub) > 5000:
                sub = sub.sample(n=5000, random_state=0)
            plt.scatter(
                sub[length_col].values,
                sub[score_col].values,
                s=12,
                alpha=0.28,
                label=str(g),
            )

        plt.xlabel(length_col)
        plt.ylabel(score_col)
        plt.title("Segment Length vs Midline Score")
        if len(keep_groups) <= 12:
            plt.legend(fontsize=11, framealpha=0.9)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_scatter.png"))
        plt.close()
    except Exception:
        pass

    # Discrete view: quantile-bin stats
    bins_rows = []
    for g in keep_groups:
        sub = df[df["_group"] == g].copy()
        if sub.empty:
            continue

        n_unique = int(sub[length_col].nunique(dropna=True))
        q = min(max(2, int(bins)), max(2, n_unique))
        if q <= 1:
            continue

        try:
            sub["_len_bin"] = pd.qcut(sub[length_col], q=q, duplicates="drop")
        except Exception:
            lo = float(np.min(sub[length_col].values))
            hi = float(np.max(sub[length_col].values))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                continue
            edges = np.linspace(lo, hi, num=min(q, 20) + 1)
            edges = np.unique(edges)
            if len(edges) < 2:
                continue
            sub["_len_bin"] = pd.cut(sub[length_col], bins=edges, include_lowest=True)

        for bin_key, gb in sub.groupby("_len_bin", dropna=False, observed=False):
            vals_len = pd.to_numeric(gb[length_col], errors="coerce").values
            vals_sc = pd.to_numeric(gb[score_col], errors="coerce").values
            mask = np.isfinite(vals_len) & np.isfinite(vals_sc)
            vals_len = vals_len[mask]
            vals_sc = vals_sc[mask]
            if vals_len.size == 0:
                continue

            left = np.nan
            right = np.nan
            if hasattr(bin_key, "left") and hasattr(bin_key, "right"):
                try:
                    left = float(bin_key.left)
                    right = float(bin_key.right)
                except Exception:
                    left = np.nan
                    right = np.nan

            bins_rows.append(
                {
                    "group": str(g),
                    "bin_label": str(bin_key),
                    "n": int(vals_len.size),
                    "length_bin_left": left,
                    "length_bin_right": right,
                    "length_bin_center": float(np.nanmedian(vals_len)),
                    "length_mean": float(np.nanmean(vals_len)),
                    "length_median": float(np.nanmedian(vals_len)),
                    "score_mean": float(np.nanmean(vals_sc)),
                    "score_median": float(np.nanmedian(vals_sc)),
                    "score_p10": float(np.nanpercentile(vals_sc, 10)),
                    "score_p90": float(np.nanpercentile(vals_sc, 90)),
                }
            )

    df_bins = pd.DataFrame(bins_rows)
    bins_csv = os.path.join(out_dir, f"{prefix}_bins.csv")
    if not df_bins.empty:
        df_bins = df_bins.sort_values(["group", "length_bin_center"]).reset_index(drop=True)
        df_bins.to_csv(bins_csv, index=False)

        try:
            plt.figure(figsize=(8.4, 5.4))
            for g in keep_groups:
                sub = df_bins[df_bins["group"] == str(g)].copy()
                if sub.empty:
                    continue
                x = sub["length_bin_center"].values.astype(float)
                y = sub["score_median"].values.astype(float)
                ylo = sub["score_p10"].values.astype(float)
                yhi = sub["score_p90"].values.astype(float)
                plt.plot(x, y, marker="o", linewidth=1.6, label=str(g))
                plt.fill_between(x, ylo, yhi, alpha=0.15)

            plt.xlabel("Segment length (px)")
            plt.ylabel(score_col)
            plt.title("Midline Score by Length Bin")
            if len(keep_groups) <= 12:
                plt.legend(fontsize=11, framealpha=0.9)
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{prefix}_binned.png"))
            plt.close()
        except Exception:
            pass
    else:
        pd.DataFrame(columns=[
            "group", "bin_label", "n",
            "length_bin_left", "length_bin_right", "length_bin_center",
            "length_mean", "length_median",
            "score_mean", "score_median", "score_p10", "score_p90",
        ]).to_csv(bins_csv, index=False)

    # "Where it gets bad at high lengths" summary by quartiles.
    summary_rows = []
    for g in keep_groups:
        sub = df[df["_group"] == g].copy()
        if len(sub) < 8:
            continue
        try:
            q = pd.qcut(sub[length_col], q=4, labels=False, duplicates="drop")
        except Exception:
            continue
        if q is None:
            continue
        sub["_q"] = q
        if sub["_q"].dropna().empty:
            continue
        qmin = int(sub["_q"].min())
        qmax = int(sub["_q"].max())
        low = pd.to_numeric(sub.loc[sub["_q"] == qmin, score_col], errors="coerce").dropna()
        high = pd.to_numeric(sub.loc[sub["_q"] == qmax, score_col], errors="coerce").dropna()
        if low.empty or high.empty:
            continue

        pear = float(pd.to_numeric(sub[length_col], errors="coerce").corr(
            pd.to_numeric(sub[score_col], errors="coerce"),
            method="pearson",
        ))
        spear = float(pd.to_numeric(sub[length_col], errors="coerce").corr(
            pd.to_numeric(sub[score_col], errors="coerce"),
            method="spearman",
        ))

        summary_rows.append(
            {
                "group": str(g),
                "n": int(len(sub)),
                "score_low_len_mean": float(low.mean()),
                "score_high_len_mean": float(high.mean()),
                "delta_high_minus_low": float(high.mean() - low.mean()),
                "pearson_length_score": pear,
                "spearman_length_score": spear,
            }
        )

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(out_dir, f"{prefix}_high_vs_low.csv"),
        index=False,
    )


# ======================================================================
# helpers/present_plots.py
#   Regime-A aggregation + committee plotting (single entry function)
#   - Reads width_distribution_summary.csv
#   - Aggregates across images
#   - Produces 3 plots (median diff, spread ratio, wasserstein)
# ======================================================================

def plot_width_distribution_report(
    *,
    csv_path,
    out_dir,
    group_keys=("variant", "gt_tier", "midline_type", "filtered"),
    x_key="variant",
    hue_key="gt_tier",
    filter_midline_type=None,   # e.g. "manual" or "auto" or None
    filter_gt_tier=None,        # e.g. "combined_unfiltered" or None
    title_suffix="",            # optional string appended to figure titles
):
    """
    Committee-friendly Regime-A report.

    Inputs:
      csv_path: path to width_distribution_summary.csv (appended row-per-call)
      out_dir: directory to write plots + aggregated table
      group_keys: how to aggregate
      x_key, hue_key: plot grouping
      filter_midline_type: optional filter (baseline/manual/auto)
      filter_gt_tier: optional filter ("atomic"/"combined_unfiltered"/"combined_filtered")
      title_suffix: optional string for plot titles

    Outputs (written into out_dir):
      - width_distribution_agg.csv
      - width_dist_median_diff.png
      - width_dist_spread_iqr_ratio.png
      - width_dist_wasserstein.png
    """

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Prefer SciPy for stats if available; otherwise degrade gracefully.
    try:
        from scipy.stats import wasserstein_distance, ks_2samp  # noqa: F401
        _has_scipy = True
    except Exception:
        _has_scipy = False

    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------
    # (0) Load
    # ------------------------------------------------------------
    if not os.path.exists(csv_path):
        print(f"[DIST-PLOT] missing csv: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("[DIST-PLOT] empty csv")
        return

    # ------------------------------------------------------------
    # (1) Optional filtering
    # ------------------------------------------------------------
    if filter_midline_type is not None:
        df = df[df["midline_type"].astype(str) == str(filter_midline_type)]
    if filter_gt_tier is not None:
        df = df[df["gt_tier"].astype(str) == str(filter_gt_tier)]

    if df.empty:
        print("[DIST-PLOT] nothing left after filters")
        return

    # ------------------------------------------------------------
    # (2) Aggregate helper
    # ------------------------------------------------------------
    def _iqr(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return np.nan
        return float(np.percentile(x, 75) - np.percentile(x, 25))

    gk = list(group_keys)
    agg = (
        df.groupby(gk)
          .agg(
              n_images=("image", "nunique"),
              n_samples=("n_samples", "sum"),

              median_diff_med=("median_diff", "median"),
              median_diff_iqr=("median_diff", _iqr),

              iqr_ratio_med=("iqr_ratio", "median"),
              std_ratio_med=("std_ratio", "median"),

              wasserstein_med=("wasserstein_dist", "median"),
              ks_stat_med=("ks_stat", "median"),
          )
          .reset_index()
    )

    out_agg_csv = os.path.join(out_dir, "width_distribution_agg.csv")
    agg.to_csv(out_agg_csv, index=False)
    print(f"[DIST-PLOT] wrote agg table: {out_agg_csv}")

    # ------------------------------------------------------------
    # (3) Plot helper (simple grouped bars, no seaborn dependency)
    # ------------------------------------------------------------
    def _barplot_grouped(df_in, *, y_col, out_path, ylabel, title):
        """
        Produces a simple grouped bar plot:
          x categories = x_key
          hue categories = hue_key (optional; if absent uses single bars)
        """
        # Ensure columns exist
        if x_key not in df_in.columns:
            print(f"[DIST-PLOT] missing x_key={x_key} in aggregated df")
            return
        if y_col not in df_in.columns:
            print(f"[DIST-PLOT] missing y_col={y_col} in aggregated df")
            return

        # Build categories
        xs = [str(v) for v in sorted(df_in[x_key].dropna().unique().tolist())]
        hues = None
        if hue_key is not None and hue_key in df_in.columns:
            hues = [str(v) for v in sorted(df_in[hue_key].dropna().unique().tolist())]
        else:
            hues = []

        # Prepare plot
        plt.figure(figsize=(8, 4))

        if not hues:
            y = []
            for x in xs:
                sub = df_in[df_in[x_key].astype(str) == x]
                y.append(float(np.nanmedian(sub[y_col].values)) if len(sub) else np.nan)

            x_pos = np.arange(len(xs))
            plt.bar(x_pos, y)
            plt.xticks(x_pos, xs, rotation=20, ha="right")

        else:
            # grouped bars
            x_pos = np.arange(len(xs))
            width = 0.8 / max(len(hues), 1)

            for j, h in enumerate(hues):
                y = []
                for x in xs:
                    sub = df_in[
                        (df_in[x_key].astype(str) == x) &
                        (df_in[hue_key].astype(str) == h)
                    ]
                    y.append(float(np.nanmedian(sub[y_col].values)) if len(sub) else np.nan)

                plt.bar(x_pos + (j - (len(hues) - 1) / 2) * width, y, width=width, label=h)

            plt.xticks(x_pos, xs, rotation=20, ha="right")
            leg = plt.legend(
                loc="lower right",
                frameon=True,
                framealpha=0.5,
                facecolor="white",
                edgecolor="none",
            )
            if leg is not None:
                for txt in leg.get_texts():
                    txt.set_alpha(1.0)

        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"[DIST-PLOT] wrote: {out_path}")

    # ------------------------------------------------------------
    # (4) Emit plots
    # ------------------------------------------------------------
    t_suf = f" {title_suffix}".rstrip()

    # 4.1 median difference
    out1 = os.path.join(out_dir, "width_dist_median_diff.png")
    _barplot_grouped(
        agg,
        y_col="median_diff_med",
        out_path=out1,
        ylabel="Median width difference (px)",
        title=f"Median width distribution difference vs GT{t_suf}",
    )

    # 4.2 spread ratio (IQR)
    out2 = os.path.join(out_dir, "width_dist_spread_iqr_ratio.png")
    _barplot_grouped(
        agg,
        y_col="iqr_ratio_med",
        out_path=out2,
        ylabel="IQR ratio (pred / GT)",
        title=f"Relative width distribution spread (IQR ratio){t_suf}",
    )

    # 4.3 wasserstein distance
    out3 = os.path.join(out_dir, "width_dist_wasserstein.png")
    _barplot_grouped(
        agg,
        y_col="wasserstein_med",
        out_path=out3,
        ylabel="Wasserstein distance (px)",
        title=f"Distributional divergence from GT (Wasserstein){t_suf}",
    )

    return {
        "agg_csv": out_agg_csv,
        "plots": [out1, out2, out3],
        "n_rows_in": int(len(df)),
        "n_groups": int(len(agg)),
        "used_scipy": bool(_has_scipy),
    }



