"""
Dataset-level metrics summary across all 130 images.
Reads gt_ablation_midline_weighted_summary.csv for each image and aggregates.
Output: printed table + saves summary CSV.
"""
import os, glob, csv
import numpy as np

SUP_ROOT  = r'C:\Users\13144\Documents\Masters_Thesis\datasets\SUT_Compressed\Outputs\supervision'
OUT_CSV   = os.path.join(SUP_ROOT, 'dataset_metrics_summary.csv')

METHOD_KEY = 'dt'   # which variant to pull for headline numbers

rows = []
missing = []

for sup_dir in sorted(glob.glob(os.path.join(SUP_ROOT, '*')), key=lambda p: int(os.path.basename(p)) if os.path.basename(p).isdigit() else 9999):
    img_id = os.path.basename(sup_dir)
    if not img_id.isdigit():
        continue
    wsv = os.path.join(sup_dir, 'analysis', 'gt_ablation_midline_weighted_summary.csv')
    if not os.path.exists(wsv):
        missing.append(img_id)
        continue
    try:
        with open(wsv) as f:
            reader = csv.DictReader(f)
            data = {r['variant_id']: r for r in reader}
        r = data.get(METHOD_KEY)
        if not r:
            missing.append(img_id); continue
        rows.append({
            'image':           img_id,
            'nn_mean_bid':     float(r.get('nn_mean_bidirectional', 'nan')),
            'hausdorff_p95':   float(r.get('hausdorff_p95', 'nan')),
            'coverage_min':    float(r.get('coverage_min', 'nan')),
            'orth_mean':       float(r.get('orth_mean', 'nan')),
            'orth_std':        float(r.get('orth_std', 'nan')),
            'rel_len_err':     float(r.get('relative_length_error', 'nan')),
            'score_mid':       float(r.get('score_mid', 'nan')),
            'length_px':       float(r.get('length_px', 'nan')),
        })
    except Exception as e:
        print(f"  WARN {img_id}: {e}")
        missing.append(img_id)

if not rows:
    print("No data found — have you run the batch yet?")
    exit()

vals = {k: np.array([r[k] for r in rows if not np.isnan(r[k])]) for k in rows[0] if k != 'image'}

print(f"\n{'='*65}")
print(f"  DATASET METRICS SUMMARY  ({len(rows)} images, method={METHOD_KEY})")
print(f"{'='*65}")
print(f"  {'Metric':<28} {'Mean':>8} {'Median':>8} {'Std':>7} {'Min':>8} {'Max':>8}")
print(f"  {'-'*63}")

labels = {
    'nn_mean_bid':   'NN Mean Bidirectional (px)',
    'hausdorff_p95': 'Hausdorff p95 (px)',
    'coverage_min':  'Coverage Min (fraction)',
    'orth_mean':     'Orth. Mean Bias (px)',
    'orth_std':      'Orth. Std (px)',
    'rel_len_err':   'Relative Length Error',
    'score_mid':     'Score Mid',
}

for k, lbl in labels.items():
    v = vals[k]
    if len(v) == 0: continue
    print(f"  {lbl:<28} {np.mean(v):>8.3f} {np.median(v):>8.3f} {np.std(v):>7.3f} {np.min(v):>8.3f} {np.max(v):>8.3f}")

print(f"{'='*65}")
print(f"  Images included: {len(rows)}  |  Missing/incomplete: {len(missing)}")
if missing:
    print(f"  Missing: {sorted(missing, key=int)}")

# Worst performers by coverage
sorted_cov = sorted(rows, key=lambda r: r['coverage_min'])
print(f"\n  Bottom 10 by coverage_min:")
for r in sorted_cov[:10]:
    print(f"    img {r['image']:>4}  cov={r['coverage_min']:.3f}  nn={r['nn_mean_bid']:.2f}px  orth_std={r['orth_std']:.2f}px")

# Save CSV
with open(OUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"\n  Saved: {OUT_CSV}")
