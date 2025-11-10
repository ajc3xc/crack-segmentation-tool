# ==== helpers/export_gt_normals.py ============================================
import os, numpy as np, pandas as pd, cv2
from helpers import plot_metrics
from helpers.metrics import normals_from_mask_for_midline

def export_gt_normals_for_image(self, step=2, max_radius=50):
    """
    For every *manual atomic* crack:
      - samples normals along the midline from the GT mask
      - writes one CSV with (mid_x,mid_y,e1x,e1y,e2x,e2y,width_px, idx, cid)
      - saves a quick overlay figure
    """
    H,W = self.original_image.shape[:2]
    gt = (self.current_mask > 0).astype(np.uint8)
    base = self._image_base()
    out_dir = self._metrics_dir()
    os.makedirs(out_dir, exist_ok=True)

    atomic = self._metric_atomic() or {}
    rows=[]
    for cid, cr in atomic.items():
        src = (cr.get("source") or "").lower()
        if src.startswith("auto") or src=="combined": continue

        ml = np.asarray(cr.get("midline",[]), float)
        if ml.ndim!=2 or ml.shape[1]!=2 or len(ml)<2: continue

        # sample every 'step' point to keep files light
        ml_s = ml[::max(step,1)]
        (e1x,e1y,e2x,e2y,_), widths = normals_from_mask_for_midline(ml_s, gt>0, max_radius=max_radius)
        e1 = np.column_stack([e1x,e1y]); e2 = np.column_stack([e2x,e2y])
        w  = np.asarray(widths, float)

        for k,(m,eA,eB) in enumerate(zip(ml_s, e1, e2)):
            rows.append({
                "image": base, "cid": str(cid), "idx": int(k),
                "mid_x": float(m[0]), "mid_y": float(m[1]),
                "e1x": float(eA[0]), "e1y": float(eA[1]),
                "e2x": float(eB[0]), "e2y": float(eB[1]),
                "width_px": float(w[k] if k < len(w) else np.nan)
            })

        # quick overlay
        try:
            canvas = (gt*255).astype(np.uint8)
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
            for a,b in zip(e1.astype(int), e2.astype(int)):
                cv2.line(canvas, tuple(a), tuple(b), (0,255,255), 1, cv2.LINE_AA)
            cv2.polylines(canvas, [ml.astype(int)], False, (0,0,255), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(out_dir, f"cid{cid}_gt_normals_overlay.png"), canvas)
        except Exception as e:
            print(f"[GT-NORMALS] overlay failed for cid{cid}: {e}")

    if rows:
        csv_path = os.path.join(out_dir, f"{base}_gt_normals.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"[GT-NORMALS] wrote {len(rows)} rows → {csv_path}")
# ==============================================================================
