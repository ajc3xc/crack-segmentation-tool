# helpers/variant_optimizer.py

import os
import json
import numpy as np
import pandas as pd
from helpers.metrics import set_auto_variant_for_crack


# -------------------------------------------------------------
# WEIGHT FUNCTION
# -------------------------------------------------------------
def _compute_weight(row):
    """Weight image-level contribution of a subcrack.
       You can redefine this anytime.
    """
    len_px = max(float(row.get("length_px", 1.0)), 1.0)
    area   = max(float(row.get("bbox_area", 1.0)), 1.0)

    # Default:
    # LONG subcracks matter more
    # LARGE bboxes matter more (shape complexity)
    return len_px * np.sqrt(area)


# -------------------------------------------------------------
# PER-IMAGE OPTIMIZER (LEVEL 2)
# -------------------------------------------------------------
def optimize_per_image(packs_for_image, save_json_path=None):
    """
    Given:
      packs_for_image = {cid -> pack_from_generate_auto_variants}

    Returns:
      dict with global RS3 family for this image.
    """

    frames = []
    for cid, pack in packs_for_image.items():
        df = pack.get("metrics_df")
        if df is None or df.empty:
            continue

        df2 = df.copy()
        df2["crack_id"] = cid
        frames.append(df2)

    if not frames:
        print("[OPT IMG] no metrics")
        return None

    df_all = pd.concat(frames, ignore_index=True)

    # Weighting
    df_all["weight"] = df_all.apply(_compute_weight, axis=1)

    # Group by param family
    families = []
    for (os_mode, g11, g22, g33), grp in df_all.groupby(
        ["os_mode", "g11", "g22", "g33"]
    ):
        w = grp["weight"].values
        s = grp["score_mid"].values
        score = float(np.average(s, weights=w))
        families.append({
            "os_mode": os_mode,
            "g11": float(g11), "g22": float(g22), "g33": float(g33),
            "score_mid_wmean": score,
            "n": int(len(grp)),
        })

    fam_df = pd.DataFrame(families)
    fam_df = fam_df.sort_values("score_mid_wmean", ascending=True)

    if fam_df.empty:
        return None

    best = fam_df.iloc[0].to_dict()

    if save_json_path:
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2)

    return best


# -------------------------------------------------------------
# APPLY WINNING FAMILY TO IMAGE (LEVEL 2)
# -------------------------------------------------------------
def apply_family_to_image(best_family, packs_for_image, base_name, save_folder):
    """
    Re-mark auto_best for each subcrack so that all use the selected family.
    """

    for cid, pack in packs_for_image.items():
        df = pack.get("metrics_df")
        if df is None or df.empty:
            continue

        df_loc = df[
            (df["os_mode"] == best_family["os_mode"]) &
            (df["g11"] == best_family["g11"]) &
            (df["g22"] == best_family["g22"]) &
            (df["g33"] == best_family["g33"])
        ]
        if df_loc.empty:
            continue

        vid = int(df_loc.iloc[0]["variant_global_id"])
        vkey = f"v{vid}"

        vdict = pack["variants"].get(vkey)
        if not vdict:
            continue

        params = (vdict.get("params") or {}).copy()
        params.update(best_family)

        set_auto_variant_for_crack(
            save_folder,
            base_name,
            cid,
            vdict,
            params=params,
            is_best=True,
        )


# -------------------------------------------------------------
# DATASET-LEVEL OPTIMIZER (LEVEL 3)
# -------------------------------------------------------------
def optimize_across_dataset(packs_by_image, save_global_json=None):
    """
    packs_by_image = {image_base_name : {cid -> pack}}

    Returns the ONE best family across the whole dataset.
    """

    frames = []

    for image, packs_for_image in packs_by_image.items():
        for cid, pack in packs_for_image.items():
            df = pack.get("metrics_df")
            if df is None or df.empty:
                continue
            df2 = df.copy()
            df2["image"] = image
            df2["cid"] = cid
            frames.append(df2)

    if not frames:
        print("[OPT DS] no data across dataset")
        return None

    df_all = pd.concat(frames, ignore_index=True)
    df_all["weight"] = df_all.apply(_compute_weight, axis=1)

    # Group by param family
    families = []
    for (os_mode, g11, g22, g33), grp in df_all.groupby(
        ["os_mode", "g11", "g22", "g33"]
    ):
        w = grp["weight"].values
        s = grp["score_mid"].values
        score = float(np.average(s, weights=w))
        families.append({
            "os_mode": os_mode,
            "g11": float(g11), "g22": float(g22), "g33": float(g33),
            "score_mid_wmean": score,
            "subcracks_total": int(len(grp)),
            "images_covered": int(len(grp["image"].unique())),
        })

    fam_df = pd.DataFrame(families)
    fam_df = fam_df.sort_values("score_mid_wmean", ascending=True)

    if fam_df.empty:
        return None

    best = fam_df.iloc[0].to_dict()

    if save_global_json:
        with open(save_global_json, "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2)

    return best


# -------------------------------------------------------------
# APPLY WINNING FAMILY TO ENTIRE DATASET
# -------------------------------------------------------------
def apply_family_to_dataset(best_family, packs_by_image, save_folder):
    for image, packs_for_image in packs_by_image.items():
        apply_family_to_image(
            best_family,
            packs_for_image,
            base_name=image,
            save_folder=save_folder
        )
