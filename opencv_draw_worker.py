#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import traceback

import numpy as np


def _write_output(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: opencv_draw_worker.py <input.json> <output.json>", file=sys.stderr)
        return 2

    in_json = sys.argv[1]
    out_json = sys.argv[2]

    try:
        with open(in_json, "r", encoding="utf-8") as f:
            payload = json.load(f) or {}

        mode = str(payload.get("mode") or "").strip().lower()
        image_npy = payload.get("image_npy")
        if not image_npy:
            raise ValueError("missing image_npy")

        image = np.load(str(image_npy))
        image_npy_alt = payload.get("image_npy_alt")
        image_alt = np.load(str(image_npy_alt)) if image_npy_alt else None
        initial_pts = payload.get("initial_pts") or []
        image_size = float(payload.get("image_size", 1.0))

        import cracktools as ct

        if mode == "bounding_box":
            pts, classes = ct.tools.Draw().bounding_box(
                image,
                image_size,
                image_alt=image_alt,
                initial_pts=initial_pts,
            )

            pts_json = []
            if pts is not None:
                for p in list(pts):
                    arr = np.asarray(p, dtype=float).ravel()
                    if arr.size >= 2:
                        pts_json.append([float(arr[0]), float(arr[1])])

            classes_json = []
            if classes is not None:
                for c in list(classes):
                    try:
                        classes_json.append(int(c))
                    except Exception:
                        classes_json.append(c)

            _write_output(
                out_json,
                {
                    "ok": True,
                    "pts": pts_json,
                    "classes": classes_json,
                },
            )
        elif mode == "contours":
            annotations = payload.get("annotations", {}) or {}
            contour_mode = str(payload.get("contour_mode", "add"))
            strokes = ct.tools.Draw().contours(
                image,
                image_size,
                annotations=annotations,
                mode=contour_mode,
                return_strokes=True,
            )
            if strokes is None:
                strokes = []
            _write_output(
                out_json,
                {
                    "ok": True,
                    "strokes": strokes,
                },
            )
        else:
            raise ValueError(f"unsupported mode: {mode!r}")
        return 0

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        try:
            _write_output(out_json, {"ok": False, "error": err, "traceback": tb})
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
