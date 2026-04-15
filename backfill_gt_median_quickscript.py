"""
backfill_gt_median.py

Reads every *_width_diffs_combined.csv already on disk,
computes gt_mean_px / gt_median_px / gt_std_px from the gt_width_px column,
and rewrites the matching *_width_summary_combined.csv with those columns added.

Run once, then re-run summarize_dataset_metrics() only.
"""

import os
import glob
import numpy as np
import pandas as pd

METRICS_ROOT = r"C:\Users\13144\Documents\Masters_Thesis\datasets\SUT_Compressed\Outputs\metrics"

updated = 0
skipped_already = 0
skipped_no_gt = 0
skipped_no_summary = 0
failed = 0

diffs_files = glob.glob(
    os.path.join(METRICS_ROOT, "**", "*_width_diffs_combined.csv"),
    recursive=True,
)
print(f"Found {len(diffs_files)} width_diffs_combined.csv files")

for p in diffs_files:
    try:
        df = pd.read_csv(p)

        # detect gt col
        cols = {c.lower(): c for c in df.columns}
        gt_col = next(
            (cols[k] for k in ("gt_width_px", "gt_width", "gt") if k in cols),
            None,
        )
        diff_col = next(
            (cols[k] for k in ("width_diff_px", "diff_px", "diff") if k in cols),
            None,
        )
        pred_col = next(
            (cols[k] for k in ("pred_width_px", "pred_width", "pred", "geodesic") if k in cols),
            None,
        )

        if gt_col is None or diff_col is None or pred_col is None:
            skipped_no_gt += 1
            continue

        gt   = pd.to_numeric(df[gt_col],   errors="coerce").values
        pred = pd.to_numeric(df[pred_col], errors="coerce").values
        diff = pd.to_numeric(df[diff_col], errors="coerce").values
        keep = np.isfinite(gt) & np.isfinite(pred) & np.isfinite(diff)
        gt, pred, diff = gt[keep], pred[keep], diff[keep]

        if len(diff) == 0:
            skipped_no_gt += 1
            continue

        # find matching summary CSV
        stem = os.path.basename(p)  # e.g. "8_width_diffs_combined.csv"
        base = stem.replace("_width_diffs_combined.csv", "")
        summary_csv = os.path.join(
            os.path.dirname(p),
            f"{base}_width_summary_combined.csv",
        )

        if not os.path.isfile(summary_csv):
            skipped_no_summary += 1
            continue

        sdf = pd.read_csv(summary_csv)

        if "gt_median_px" in sdf.columns:
            skipped_already += 1
            continue

        sdf["gt_mean_px"]   = float(np.mean(gt))
        sdf["gt_median_px"] = float(np.median(gt))
        sdf["gt_std_px"]    = float(np.std(gt))
        sdf.to_csv(summary_csv, index=False)
        updated += 1

    except Exception as e:
        print(f"FAILED {p}: {e}")
        failed += 1

print(f"\nDone.")
print(f"  updated:          {updated}")
print(f"  already had col:  {skipped_already}")
print(f"  no gt/diff col:   {skipped_no_gt}")
print(f"  no summary found: {skipped_no_summary}")
print(f"  failed:           {failed}")
print(f"\nNow re-run summarize_dataset_metrics() only.")