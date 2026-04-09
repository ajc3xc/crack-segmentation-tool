import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cv2


def inside_ratio(mid: np.ndarray, bbox_xyxy) -> float:
    (x0, y0), (x1, y1) = bbox_xyxy
    xs = mid[:, 0]
    ys = mid[:, 1]
    inside = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
    return float(np.mean(inside)) if mid.size else 0.0


def area(bbox_xyxy) -> float:
    (x0, y0), (x1, y1) = bbox_xyxy
    return float(max(0.0, x1 - x0) * max(0.0, y1 - y0))


def plot_atomic_debug(image, annotation, out_png: Path, show_old_bbox: bool = False, title: str = "ATOMIC DEBUG"):
    ann = (annotation or {}).get("annotations", {}) or {}
    atomic = ann.get("atomic_cracks", {}) or {}
    if image is None or not atomic:
        return

    fig, ax = plt.subplots(figsize=(10, 12), dpi=200)
    ax.imshow(image)

    for cid_str, crack in atomic.items():
        try:
            cid_i = int(cid_str)
        except Exception:
            cid_i = 0

        mid = np.asarray((crack or {}).get("midline", []), float)
        bbox = (crack or {}).get("mask_bbox", None)
        old_bbox = (crack or {}).get("mask_bbox_old", None)
        color = plt.cm.tab10(cid_i % 10)

        if mid.ndim == 2 and mid.shape[1] == 2 and len(mid) > 0:
            ax.plot(mid[:, 0], mid[:, 1], color=color, linewidth=2)
            ax.scatter(mid[0, 0], mid[0, 1], c="red", s=30)
            ax.scatter(mid[-1, 0], mid[-1, 1], c="red", s=30)

        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x, y, w, h = [int(v) for v in bbox]
            rect = plt.Rectangle((x, y), w, h, edgecolor=color, facecolor="none", linewidth=2)
            ax.add_patch(rect)
            ax.text(x, y - 5, f"ID {cid_str}", color=color, fontsize=9, weight="bold")

        if show_old_bbox and isinstance(old_bbox, (list, tuple)) and len(old_bbox) == 4:
            x, y, w, h = [int(v) for v in old_bbox]
            rect_old = plt.Rectangle(
                (x, y), w, h, edgecolor="red", facecolor="none", linestyle="--", linewidth=1
            )
            ax.add_patch(rect_old)

    H, W = image.shape[:2]
    ax.set_title(title)
    ax.set_xlim([0, W])
    ax.set_ylim([H, 0])
    ax.axis("off")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] wrote {out_png}")


def _load_boxes(ann: dict) -> dict:
    boxes_blob = ann.get("box", {}) or ann.get("boxes", {}) or {}
    out = {}
    for bid, box in boxes_blob.items():
        if not isinstance(box, dict):
            continue
        bb = box.get("bounding_box", None)
        if not (isinstance(bb, (list, tuple)) and len(bb) == 2):
            continue
        p0, p1 = bb
        if (
            not isinstance(p0, (list, tuple))
            or not isinstance(p1, (list, tuple))
            or len(p0) != 2
            or len(p1) != 2
        ):
            continue
        try:
            x0, y0 = float(p0[0]), float(p0[1])
            x1, y1 = float(p1[0]), float(p1[1])
        except Exception:
            continue
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        out[str(bid)] = ((x0, y0), (x1, y1))
    return out


def repair_bboxes(
    json_path: Path,
    contain_thresh: float = 0.8,
    weak_thresh: float = 0.3,
    image_path: Path | None = None,
    before_png: Path | None = None,
    after_png: Path | None = None,
    return_data: bool = False,
):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    ann = data.get("annotations", {}) or {}
    atomic = ann.get("atomic_cracks", {}) or {}
    box_dict = _load_boxes(ann)

    print(f"[BBOX] json={json_path} atomics={len(atomic)} boxes={len(box_dict)}")

    if not atomic:
        print("[INFO] no atomic cracks found")
        return 0
    if not box_dict:
        print("[INFO] no boxes found under annotations.box/annotations.boxes")
        return 0

    image = None
    if image_path is not None and image_path.exists():
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is not None:
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if image is not None and before_png is not None:
        plot_atomic_debug(image, data, before_png, show_old_bbox=False, title="BEFORE FIX")

    fixed = 0
    skipped = 0
    invalid = 0

    for cid, crack in atomic.items():
        mid = np.asarray((crack or {}).get("midline", []), dtype=float)
        if mid.ndim != 2 or mid.shape[1] != 2 or len(mid) == 0:
            print(f"[SKIP] atomic {cid} no valid midline")
            continue

        candidates = []
        for bid, bbox in box_dict.items():
            score = inside_ratio(mid, bbox)
            if score >= contain_thresh:
                candidates.append((bid, bbox, score))

        if candidates:
            best_bid, best_bbox, best_score = min(candidates, key=lambda x: area(x[1]))
            source = "contain"
        else:
            best_bid = None
            best_bbox = None
            best_score = -1.0
            for bid, bbox in box_dict.items():
                score = inside_ratio(mid, bbox)
                if score > best_score:
                    best_score = score
                    best_bbox = bbox
                    best_bid = bid
            source = "fallback_max"
            if best_score < weak_thresh:
                crack["invalid_bbox"] = True
                crack["bbox_source"] = "invalid"
                invalid += 1
                print(f"[INVALID] atomic {cid} (score={best_score:.3f})")
                continue

        (x0, y0), (x1, y1) = best_bbox
        new_bbox = [int(round(x0)), int(round(y0)), int(round(x1 - x0)), int(round(y1 - y0))]
        old_bbox = crack.get("mask_bbox")
        crack["bbox_source"] = f"box_{best_bid}:{source}"
        crack.pop("invalid_bbox", None)

        if list(old_bbox) == list(new_bbox) if isinstance(old_bbox, (list, tuple)) else False:
            skipped += 1
            print(f"[OK] atomic {cid} unchanged -> box {best_bid} (score={best_score:.3f}, mode={source})")
        else:
            crack["mask_bbox_old"] = old_bbox
            crack["mask_bbox"] = new_bbox
            fixed += 1
            print(f"[FIX] atomic {cid} -> box {best_bid} (score={best_score:.3f}, mode={source})")
            print(f"  old={old_bbox}")
            print(f"  new={new_bbox}")

    total = int(len(atomic))
    if fixed == 0:
        print(f"[BBOX] no changes ({total} checked)")
    else:
        print(f"[BBOX] fixed={fixed}/{total}")
    print(f"[SUMMARY] fixed={fixed}, skipped={skipped}, invalid={invalid}, total={total}")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[WRITE] {json_path}")
    if image is not None and after_png is not None:
        plot_atomic_debug(image, data, after_png, show_old_bbox=True, title="AFTER FIX")
    if return_data:
        return fixed, data
    return fixed


def main():
    parser = argparse.ArgumentParser(description="Repair atomic mask_bbox from midline-to-box containment.")
    parser.add_argument("json_path", type=Path, help="Path to annotation JSON")
    parser.add_argument("--contain-thresh", type=float, default=0.8, help="Containment threshold")
    parser.add_argument("--weak-thresh", type=float, default=0.3, help="Fallback weak threshold")
    parser.add_argument("--image", type=Path, default=None, help="Optional image path for before/after debug plots")
    parser.add_argument("--before-png", type=Path, default=None, help="Optional output path for BEFORE debug plot")
    parser.add_argument("--after-png", type=Path, default=None, help="Optional output path for AFTER debug plot")
    args = parser.parse_args()

    repair_bboxes(
        json_path=args.json_path,
        contain_thresh=float(args.contain_thresh),
        weak_thresh=float(args.weak_thresh),
        image_path=args.image,
        before_png=args.before_png,
        after_png=args.after_png,
    )


if __name__ == "__main__":
    main()
