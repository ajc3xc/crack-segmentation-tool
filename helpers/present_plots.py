# helpers/present_plots.py
import os, numpy as np, pandas as pd, cv2
import matplotlib.pyplot as plt

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
    paths = {
        "manual":   os.path.join(metrics_dir, f"{base_name}_width_summary_manual.csv"),
        "auto":     os.path.join(metrics_dir, f"{base_name}_width_summary_auto.csv"),
        "combined": os.path.join(metrics_dir, f"{base_name}_width_summary_combined.csv"),
    }
    rows = []
    for tag, p in paths.items():
        if not os.path.exists(p): continue
        df = pd.read_csv(p)
        for k in ("mae_px","rmse_px","bias_px","corr"):
            if k not in df.columns: df[k] = np.nan
        rows.append({"method": tag,
                     "MAE":  df["mae_px"].mean(),
                     "RMSE": df["rmse_px"].mean(),
                     "Bias": df["bias_px"].mean(),
                     "Corr": df["corr"].mean()})
    if not rows: return
    d = pd.DataFrame(rows)

    fig, ax = plt.subplots(1,2, figsize=(9,4), dpi=160)
    d_err = d[["method","MAE","RMSE","Bias"]].set_index("method")
    d_err.plot(kind="bar", ax=ax[0])
    ax[0].set_title("Width errors (px)"); ax[0].set_ylabel("px")
    ax[0].legend()

    ax[1].bar(d["method"], d["Corr"])
    ax[1].set_ylim(0,1); ax[1].set_title("Width correlation"); ax[1].set_ylabel("r")
    for i, v in enumerate(d["Corr"]):
        if np.isfinite(v): ax[1].text(i, v+0.02, f"{v:.2f}", ha="center", fontsize=8)

    plt.tight_layout(); plt.savefig(out_png); plt.close()'''
    
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

def build_deck_plots_for_image(metrics_dir: str, base_name: str):
    """
    Creates 4 figures in metrics_dir:
      - iou_vs_bf1_scatter.png
      - assd_hd95_box.png
      - width_summary_triplet.png
      - midline_edge_metrics_bars.png
    """
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_csv = os.path.join(metrics_dir, "mask_metrics.csv")
    midline_csv = os.path.join(metrics_dir, f"{base_name}_midline_edge_metrics.csv")
    if os.path.exists(metrics_csv):
        plot_iou_vs_bf1_scatter(metrics_csv, os.path.join(metrics_dir, "iou_vs_bf1_scatter.png"))
        plot_assd_hd95_box(metrics_csv,       os.path.join(metrics_dir, "assd_hd95_box.png"))
        plot_width_summary_triplet(metrics_dir, base_name,
                                   os.path.join(metrics_dir, "width_summary_triplet.png"))
    plot_midline_edge_metrics_bars(midline_csv,
                                   os.path.join(metrics_dir, "midline_edge_metrics_bars.png"))

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
