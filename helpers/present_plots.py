# helpers/present_plots.py
import os, numpy as np, pandas as pd, cv2
import matplotlib
matplotlib.use("Agg")   # fastest non-GUI backend for PNG/PDF/SVG export
matplotlib.rcParams.update({
    "figure.max_open_warning": 0,
    "text.kerning_factor": 0,
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
})

import matplotlib.pyplot as plt
plt.ioff()   # no interactive figure updates

import seaborn as sns

# ------------------------------ #
# A) DECK-READY SUMMARY FIGURES  #
# ------------------------------ #

def _safe_cols(df, cols):
    out = {}
    for c in cols:
        out[c] = df[c].astype(float).values if c in df.columns else np.array([])
    return out

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

    plt.figure(figsize=(6, 5), dpi=160)

    if assd is not None and len(assd) == len(iou):
        # two scatters so marker changes still work with a single colorbar
        sc0 = plt.scatter(iou[is_atomic], bf1[is_atomic], c=assd[is_atomic], s=55, alpha=0.9, marker="o")
        sc1 = plt.scatter(iou[~is_atomic], bf1[~is_atomic], c=assd[~is_atomic], s=55, alpha=0.9, marker="s")
        cb = plt.colorbar(sc1)
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
    plt.figure(figsize=(5,4), dpi=160)
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

    fig, ax = plt.subplots(1, 2, figsize=(9, 4), dpi=160)

    # LEFT – grouped MAE/RMSE/Bias
    ax0 = ax[0]
    ax0.bar(x - width, d["MAE"],  width, label="MAE")
    ax0.bar(x,         d["RMSE"], width, label="RMSE")
    ax0.bar(x + width, d["Bias"], width, label="Bias")

    ax0.set_xticks(x)
    ax0.set_xticklabels(methods, fontsize=10, fontweight="normal")  # <— NOT bold
    ax0.set_title("Width errors (px)", fontsize=14, fontweight="bold")
    ax0.set_ylabel("px")
    ax0.legend(fontsize=9)

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
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
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
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=160)

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
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)

def plot_midline_edge_metrics_bars(midline_csv, out_png):
    if not os.path.exists(midline_csv): return
    df = pd.read_csv(midline_csv)
    picks = [c for c in ["chamfer_mean","hausdorff","coverage","angle_err_deg"]
             if c in df.columns]
    if not picks: return
    means = df[picks].astype(float).mean()
    stds  = df[picks].astype(float).std()
    plt.figure(figsize=(6,4), dpi=160)
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

    # exclude TOTAL for per-crack distribution plots
    if "crack_type" in df.columns:
        df = df[df["crack_type"].isin(["atomic", "combined"])].copy()

    assd_col = "ASSD" if "ASSD" in df.columns else ("assd" if "assd" in df.columns else None)
    hd95_col = "HD95" if "HD95" in df.columns else ("hd95" if "hd95" in df.columns else None)
    if assd_col is None or hd95_col is None:
        return

    assd = df[assd_col].astype(float).values
    hd95 = df[hd95_col].astype(float).values

    plt.figure(figsize=(8, 4), dpi=160)
    plt.hist(assd[np.isfinite(assd)], bins=30, alpha=0.6, label="ASSD")
    plt.hist(hd95[np.isfinite(hd95)], bins=30, alpha=0.6, label="HD95")
    plt.title("Surface distance histogram")
    plt.xlabel("pixels")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    
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

    plt.figure(figsize=(6, 5), dpi=160)
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

    plt.figure(figsize=(8, 4.8), dpi=160)
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

    plt.figure(figsize=(6.2, 5.2), dpi=160)
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

    plt.figure(figsize=(5,5), dpi=160)
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
    if "angle_err_deg" not in df: return

    ang = df["angle_err_deg"].astype(float)

    plt.figure(figsize=(6,4), dpi=160)
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

    plt.figure(figsize=(6,4), dpi=160)
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

    fig, ax = plt.subplots(figsize=(4, 4), dpi=160)
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues",
                xticklabels=["Pred +","Pred -"],
                yticklabels=["GT +","GT -"],
                ax=ax)
    ax.set_title("Confusion matrix (aggregated)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def plot_crack_statistics_overview(metrics_dir, base_name, out_png):
    import os, pandas as pd, numpy as np
    import matplotlib.pyplot as plt

    csv = os.path.join(metrics_dir, f"{base_name}_midline_edge_metrics.csv")
    if not os.path.exists(csv):
        print("[CRACK_STATS] no midline metrics CSV")
        return

    df = pd.read_csv(csv)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4), dpi=160)

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
    plt.savefig(out_png, dpi=160)
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
    # supervision-specific plots
    # ---------------------------
    for supervision in ("manual", "auto"):
        df = df_all[df_all["supervision"].astype(str) == supervision].copy()
        if df.empty:
            continue

        subdir = os.path.join(metrics_dir, supervision)
        os.makedirs(subdir, exist_ok=True)

        csv_sub = os.path.join(subdir, f"{supervision}_mask_metrics.csv")
        df.to_csv(csv_sub, index=False)

        print(f"[DEBUG PLOT] building plots for {supervision}")

        plot_iou_vs_bf1_scatter(
            csv_sub,
            os.path.join(subdir, f"{supervision}_iou_vs_bf1_scatter.png"),
            supervision=supervision,
        )

        plot_assd_hd95_box(
            csv_sub,
            os.path.join(subdir, f"{supervision}_assd_hd95_box.png"),
        )

        plot_mask_metrics_triplet(
            subdir, base_name,
            supervision,
            os.path.join(subdir, f"{supervision}_mask_metrics_triplet.png"),
        )

        plot_surface_distance_histogram(
            csv_sub,
            os.path.join(subdir, f"{supervision}_surface_distance_histogram.png"),
            supervision=supervision,
        )

        # diagnostic: "why did TOTAL end up like this?"
        plot_size_vs_iou_scatter(
            csv_sub,
            os.path.join(subdir, f"{supervision}_size_vs_iou_gt_area.png"),
            supervision=supervision,
            x_mode="gt_area_px",   # works immediately
        )

        plot_under_overfill_scatter(
            csv_sub,
            os.path.join(subdir, f"{supervision}_underfill_vs_overfill.png"),
            supervision=supervision,
        )

        plot_error_contribution_bars(
            csv_sub,
            os.path.join(subdir, f"{supervision}_fn_contribution.png"),
            supervision=supervision,
            which="fn",
            topk=15,
        )

        plot_error_contribution_bars(
            csv_sub,
            os.path.join(subdir, f"{supervision}_fp_contribution.png"),
            supervision=supervision,
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

    plt.figure(figsize=(6, 4), dpi=160)
    plt.hist(diffs, bins=bins, alpha=0.85)
    plt.axvline(0.0, linewidth=1)
    plt.xlabel("Pred width − GT width (px)")
    plt.ylabel("count")
    plt.title(title or "Width diff histogram")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()
    print("[WIDTH HIST] wrote:", out_png)

def plot_width_summary_bars(metrics_dir, base_name, out_png):
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

    fig, ax = plt.subplots(figsize=(8, 4), dpi=160)

    for i, row in d.iterrows():
        vals = [row[m] for m in metrics]
        ax.bar(x + i * w, vals, w, label=row["method"])

    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("px / corr")
    ax.set_title(f"Crack {base_name} width error summary")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()

    print("[WIDTH BAR] wrote:", out_png)

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

    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)

    ax.scatter(gt, dw, s=s, alpha=alpha)

    ax.axhline(0.0, color="k", lw=1, ls="--", alpha=0.6)

    ax.set_xlabel("GT width (px)")
    ax.set_ylabel("Δ width = pred − GT (px)")

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
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

    plt.figure(figsize=(6, 4), dpi=160)
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
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
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

    plt.figure(figsize=(6, 5), dpi=160)

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
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()

