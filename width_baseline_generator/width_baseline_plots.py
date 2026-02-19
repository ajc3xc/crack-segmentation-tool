# ============================================================
# THESIS-COMMITTEE SUMMARY PLOTS
# ============================================================

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless / batch safe
import matplotlib.pyplot as plt

def _safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def plot_width_baseline_summary(res_df, summary_df, out_dir):
    """
    Master entrypoint called from main().
    Generates all committee-facing plots.
    """
    _safe_mkdir(out_dir)

    plot_weighted_runtime(summary_df, out_dir)
    plot_runtime_distributions(res_df, out_dir)
    plot_scaling_behavior(res_df, out_dir)
    plot_runtime_breakdown(res_df, out_dir)


# ------------------------------------------------------------
# 1) Weighted mean runtime (primary figure)
# ------------------------------------------------------------
def plot_weighted_runtime(summary_df, out_dir):
    df = summary_df[
        summary_df["metric"].str.endswith("_weighted_mean_by_crack_px")
    ].copy()

    if df.empty:
        return

    df["method"] = (
        df["metric"]
        .str.replace("_s_weighted_mean_by_crack_px", "", regex=False)
        .str.replace("_weighted_mean_by_crack_px", "", regex=False)
    )

    df = df.sort_values("value")

    plt.figure(figsize=(7, 4))
    plt.barh(df["method"], df["value"])
    plt.xlabel("Weighted Mean Runtime (seconds)")
    plt.title("Runtime by Method (Weighted by Crack Area)")
    plt.tight_layout()

    out = os.path.join(out_dir, "fig_runtime_weighted_mean.png")
    plt.savefig(out, dpi=200)
    plt.close()


# ------------------------------------------------------------
# 2) Per-image runtime distribution
# ------------------------------------------------------------
def plot_runtime_distributions(res_df, out_dir):
    method_cols = [c for c in res_df.columns if c.endswith("_s") and c != "total_s"]
    if not method_cols:
        return

    data = []
    for c in method_cols:
        for v in res_df[c]:
            if pd.notna(v):
                data.append({
                    "method": c.replace("_s", ""),
                    "time_s": float(v),
                })

    if not data:
        return

    df = pd.DataFrame(data)

    plt.figure(figsize=(8, 4))
    df.boxplot(
        column="time_s",
        by="method",
        grid=False,
        rot=30,
    )

    plt.suptitle("")
    plt.title("Per-image Runtime Distribution")
    plt.ylabel("Runtime (seconds)")
    plt.tight_layout()

    out = os.path.join(out_dir, "fig_runtime_distributions.png")
    plt.savefig(out, dpi=200)
    plt.close()


# ------------------------------------------------------------
# 3) Scaling behavior: runtime vs crack size (strongest figure)
# ------------------------------------------------------------
def plot_scaling_behavior(res_df, out_dir):
    if "crack_px" not in res_df.columns:
        return

    method_cols = [c for c in res_df.columns if c.endswith("_s") and c != "total_s"]
    if not method_cols:
        return

    plt.figure(figsize=(7, 5))

    for c in method_cols:
        mask = res_df["crack_px"] > 0
        if mask.sum() == 0:
            continue

        plt.scatter(
            res_df.loc[mask, "crack_px"],
            res_df.loc[mask, c],
            alpha=0.6,
            label=c.replace("_s", ""),
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Crack Area (pixels)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime Scaling vs Crack Size")
    plt.legend()
    plt.tight_layout()

    out = os.path.join(out_dir, "fig_runtime_scaling.png")
    plt.savefig(out, dpi=200)
    plt.close()


# ------------------------------------------------------------
# 4) Runtime breakdown (mean per method)
# ------------------------------------------------------------
def plot_runtime_breakdown(res_df, out_dir):
    method_cols = [c for c in res_df.columns if c.endswith("_s") and c != "total_s"]
    if not method_cols:
        return

    means = res_df[method_cols].mean()

    plt.figure(figsize=(7, 4))
    plt.bar(means.index.str.replace("_s", ""), means.values)
    plt.ylabel("Mean Runtime (seconds)")
    plt.title("Average Runtime per Method")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    out = os.path.join(out_dir, "fig_runtime_breakdown.png")
    plt.savefig(out, dpi=200)
    plt.close()
