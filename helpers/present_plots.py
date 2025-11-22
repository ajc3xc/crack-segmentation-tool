# helpers/present_plots.py
import os, numpy as np, pandas as pd, cv2
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------ #
# A) DECK-READY SUMMARY FIGURES  #
# ------------------------------ #

def _safe_cols(df, cols):
    out = {}
    for c in cols:
        out[c] = df[c].astype(float).values if c in df.columns else np.array([])
    return out

def plot_iou_vs_bf1_scatter(metrics_csv, out_png):
    df = pd.read_csv(metrics_csv)
    dfi = df[df["crack_type"].isin(["atomic","combined"])].copy()
    if dfi.empty: return
    vals = _safe_cols(dfi, ["iou","boundary_f1","ASSD"])
    plt.figure(figsize=(6,5), dpi=160)
    sc = plt.scatter(vals["iou"], vals["boundary_f1"],
                     c=vals["ASSD"] if len(vals["ASSD"]) else None,
                     s=50, alpha=0.9)
    plt.xlabel("IoU"); plt.ylabel("Boundary F1")
    plt.title("IoU vs Boundary F1 (color = ASSD px)")
    if len(vals["ASSD"]):
        cb = plt.colorbar(sc); cb.set_label("ASSD (px)")
    plt.grid(True, alpha=.3)
    plt.tight_layout(); plt.savefig(out_png); plt.close()

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
    
'''def plot_width_summary_triplet(metrics_dir, base_name, out_png):
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

        # normalize column names (old + new)
        df.columns = [c.lower() for c in df.columns]

        mae  = df["mae_px"].mean()  if "mae_px"  in df else np.nan
        rmse = df["rmse_px"].mean() if "rmse_px" in df else np.nan
        bias = df["bias_px"].mean() if "bias_px" in df else np.nan
        corr = df["corr"].mean()    if "corr"    in df else np.nan

        rows.append({
            "method": tag,
            "MAE":  mae,
            "RMSE": rmse,
            "Bias": bias,
            "Corr": corr
        })

    if not rows:
        print("[TRIPLET] no usable width-summary CSVs")
        return

    d = pd.DataFrame(rows)

    fig, ax = plt.subplots(1, 2, figsize=(9, 4), dpi=160)

    # LEFT PANEL — Errors
    d_err = d[["method", "MAE", "RMSE", "Bias"]].set_index("method")
    d_err.plot(kind="bar", ax=ax[0])
    ax[0].set_title("Width errors (px)")
    ax[0].set_ylabel("px")
    ax[0].legend()

    # RIGHT PANEL — Correlation
    ax[1].bar(d["method"], d["Corr"])
    ax[1].set_ylim(0, 1)
    ax[1].set_title("Width correlation")
    ax[1].set_ylabel("r")

    for i, v in enumerate(d["Corr"]):
        if np.isfinite(v):
            ax[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()'''
    
def plot_width_summary_triplet(metrics_dir, base_name, out_png):
    import os, numpy as np, pandas as pd
    import matplotlib.pyplot as plt

    paths = {
        "manual":   os.path.join(metrics_dir, f"{base_name}_width_summary_manual.csv"),
        "auto":     os.path.join(metrics_dir, f"{base_name}_width_summary_auto.csv"),
        "combined": os.path.join(metrics_dir, f"{base_name}_width_summary_combined.csv"),
    }

    rows=[]
    for tag,p in paths.items():
        if not os.path.exists(p):
            continue
        df=pd.read_csv(p)
        df.columns=[c.lower() for c in df.columns]
        rows.append({
            "method":tag,
            "MAE":  df["mae_px"].mean()  if "mae_px"  in df else np.nan,
            "RMSE": df["rmse_px"].mean() if "rmse_px" in df else np.nan,
            "Bias": df["bias_px"].mean() if "bias_px" in df else np.nan,
            "Corr": df["corr"].mean()    if "corr"    in df else np.nan,
        })

    if not rows:
        print("[TRIPLET] no usable width summary")
        return

    d=pd.DataFrame(rows)
    fig,ax=plt.subplots(1,2,figsize=(10,4),dpi=160)

    # LEFT Errors
    d_err=d[["method","MAE","RMSE","Bias"]].set_index("method")
    d_err.plot(kind="bar",ax=ax[0])
    ax[0].set_title("Width errors (px)",fontsize=14,fontweight="bold")
    ax[0].set_ylabel("px")
    ax[0].legend()

    for lbl in ax[0].get_xticklabels():
        lbl.set_fontsize(11)
        lbl.set_fontweight("bold")
        lbl.set_ha("center")

    # RIGHT Corr
    ax[1].bar(d["method"], d["Corr"])
    ax[1].set_ylim(0,1)
    ax[1].set_title("Width correlation",fontsize=14,fontweight="bold")
    ax[1].set_ylabel("r")

    for lbl in ax[1].get_xticklabels():
        lbl.set_fontsize(11)
        lbl.set_fontweight("bold")
        lbl.set_ha("center")

    for i,v in enumerate(d["Corr"]):
        if np.isfinite(v):
            ax[1].text(i, v+0.02, f"{v:.2f}",ha="center",fontsize=9)

    plt.tight_layout()
    plt.savefig(out_png,bbox_inches="tight",dpi=160)
    plt.close()

def plot_mask_metrics_triplet(metrics_dir, base_name, out_png):
    """
    Region metrics, boundary metrics, and confusion counts summary plot.
    Looks for mask_metrics.csv inside metrics_dir.
    """
    import os, numpy as np, pandas as pd
    import matplotlib.pyplot as plt

    csv_path = os.path.join(metrics_dir, "mask_metrics.csv")
    if not os.path.exists(csv_path):
        print("[MASK_TRIPLET] mask_metrics.csv not found")
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
                return float(total[c])
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

    confusion = {
        "TP": get("tp", 0),
        "FP": get("fp", 0),
        "FN": get("fn", 0),
        "TN": get("tn", 0),
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), dpi=160)

    def bar(ax, data, title, ylabel=None, ylim=None, fmt="{:.2f}"):
        labels = list(data.keys())
        vals = np.array(list(data.values()), float)
        xs = np.arange(len(vals))
        ax.bar(xs, vals)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=0, fontsize=8, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        if ylabel:
            ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        for i, v in enumerate(vals):
            if not np.isfinite(v):
                continue
            ax.text(xs[i], v + (0.03 if ylim else 0.03*np.max(vals)), fmt.format(v),
                    ha="center", va="bottom", fontsize=7)

    bar(axes[0], region,   "Region metrics",   "score", ylim=(0,1))
    bar(axes[1], boundary, "Boundary metrics", "score", ylim=(0,1))
    bar(axes[2], confusion,"Confusion counts", "count", fmt="{:.0f}")

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=160)
    plt.close()

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

def plot_surface_distance_histogram(metrics_csv, out_png):
    import numpy as np, pandas as pd, matplotlib.pyplot as plt
    if not os.path.exists(metrics_csv): return

    df = pd.read_csv(metrics_csv)
    if "assd" not in df or "hd95" not in df:
        return

    assd = df["assd"].astype(float)
    hd95 = df["hd95"].astype(float)

    plt.figure(figsize=(8,4), dpi=160)
    plt.hist(assd, bins=30, alpha=0.6, label="ASSD")
    plt.hist(hd95, bins=30, alpha=0.6, label="HD95")
    plt.title("Surface distance histogram")
    plt.xlabel("pixels")
    plt.ylabel("count")
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
    
def plot_confusion_matrix(df_mask, out_png):
    """
    Paper-style confusion matrix (TP/FP/FN/TN).
    Expects mask_metrics TOTAL row OR full df_mask (will reduce automatically).
    """

    if df_mask is None or df_mask.empty:
        print("[CONFUSION] empty df")
        return

    # Prefer TOTAL row
    row = None
    if "crack_type" in df_mask.columns:
        m = df_mask["crack_type"].astype(str).str.upper() == "TOTAL"
        if m.any():
            row = df_mask[m].iloc[0]

    if row is None:
        row = df_mask.mean(numeric_only=True)

    def get(col):
        for c in row.index:
            if c.lower() == col.lower():
                return float(row[c])
        return np.nan

    cm = np.array([
        [get("TP"), get("FP")],
        [get("FN"), get("TN")]
    ])

    fig, ax = plt.subplots(figsize=(4.5, 4.2), dpi=160)
    sns.heatmap(
        cm,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        xticklabels=["Pred+","Pred−"],
        yticklabels=["GT+","GT−"],
        cbar=False,
        linewidths=.5,
        square=True,
        ax=ax
    )

    ax.set_title("Confusion matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()

##################################################################
# New Metrics Plots
#################################################################
        
def build_deck_plots_for_image(metrics_dir: str, base_name: str):
    os.makedirs(metrics_dir, exist_ok=True)

    metrics_csv = os.path.join(metrics_dir, "mask_metrics.csv")
    midline_csv = os.path.join(metrics_dir, f"{base_name}_midline_edge_metrics.csv")

    if os.path.exists(metrics_csv):
        df = pd.read_csv(metrics_csv)

        # Traditional plots
        plot_iou_vs_bf1_scatter(metrics_csv, os.path.join(metrics_dir, "iou_vs_bf1_scatter.png"))
        plot_assd_hd95_box(metrics_csv,   os.path.join(metrics_dir, "assd_hd95_box.png"))
        plot_width_summary_triplet(metrics_dir, base_name,
                                   os.path.join(metrics_dir, "width_summary_triplet.png"))
        plot_mask_metrics_triplet(metrics_dir, base_name,
                                   os.path.join(metrics_dir, "mask_metrics_triplet.png"))

        # NEW → confusion matrix
        plot_confusion_matrix(df, os.path.join(metrics_dir,"confusion_matrix.png"))

    # midline/edge plots
    plot_midline_edge_metrics_bars(midline_csv,
                                   os.path.join(metrics_dir, "midline_edge_metrics_bars.png"))

    # Extra plots
    plot_surface_distance_histogram(metrics_csv,
        os.path.join(metrics_dir,"surface_distance_histogram.png"))

    plot_boundary_pr_curve(metrics_csv,
        os.path.join(metrics_dir,"boundary_pr_curve.png"))

    plot_width_error_distribution(
        os.path.join(metrics_dir,f"{base_name}_width_diffs_manual.csv"),
        os.path.join(metrics_dir,"width_error_hist_manual.png"))

    plot_width_error_distribution(
        os.path.join(metrics_dir,f"{base_name}_width_diffs_combined.csv"),
        os.path.join(metrics_dir,"width_error_hist_combined.png"))

    plot_midline_angle_distribution(
        midline_csv,
        os.path.join(metrics_dir,"midline_angle_hist.png"))


# --------------------------------------- #
# C) EXPORT TRUE GT NORMALS (CSV + plot)  #
# --------------------------------------- #

def export_gt_normals_for_image(gt_mask_u8: np.ndarray,
                                atomic_cracks: dict,
                                image_hw: tuple,
                                out_dir: str,
                                step: int = 2,
                                max_radius: int = 50):
    """
    Args
    ----
    gt_mask_u8: HxW uint8 mask (1 for crack, 0 elsewhere)
    atomic_cracks: dict like metric_annotations['atomic_cracks'] (manual only)
                   each crack should have 'midline' (Nx2), optional 'source'
    image_hw: (H, W)
    out_dir: where to write CSV & overlays
    """
    from helpers.metrics import normals_from_mask_for_midline

    H, W = image_hw
    gt_bin = (gt_mask_u8 > 0).astype(np.uint8)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for cid, cr in (atomic_cracks or {}).items():
        src = (cr.get("source") or "").lower()
        if src.startswith("auto") or src == "combined":  # only manual atomics
            continue

        ml = np.asarray(cr.get("midline", []), float)
        if ml.ndim != 2 or ml.shape[1] != 2 or len(ml) < 2:
            continue

        ml_s = ml[::max(step,1)]
        (e1x,e1y,e2x,e2y,_), widths = normals_from_mask_for_midline(ml_s, gt_bin > 0,
                                                                     max_radius=max_radius)
        e1 = np.column_stack([e1x,e1y]); e2 = np.column_stack([e2x,e2y])
        w  = np.asarray(widths, float)

        for k,(m,a,b) in enumerate(zip(ml_s, e1, e2)):
            rows.append({
                "cid": str(cid), "idx": int(k),
                "mid_x": float(m[0]), "mid_y": float(m[1]),
                "e1x": float(a[0]), "e1y": float(a[1]),
                "e2x": float(b[0]), "e2y": float(b[1]),
                "width_px": float(w[k] if k < len(w) else np.nan)
            })

        # per-crack overlay
        try:
            canvas = cv2.cvtColor(gt_bin*255, cv2.COLOR_GRAY2BGR)
            for A,B in zip(e1.astype(int), e2.astype(int)):
                cv2.line(canvas, tuple(A), tuple(B), (0,255,255), 1, cv2.LINE_AA)
            cv2.polylines(canvas, [ml.astype(int)], False, (0,0,255), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(out_dir, f"cid{cid}_gt_normals_overlay.png"), canvas)
        except Exception as e:
            print(f"[GT-NORMALS] overlay failed for cid{cid}: {e}")

    if rows:
        csv_path = os.path.join(out_dir, f"gt_normals.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"[GT-NORMALS] wrote {len(rows)} rows → {csv_path}")
