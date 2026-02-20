import numpy as np


def _as_xy(pts):
    arr = np.asarray(pts, float)
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 2:
        return np.empty((0, 2), float)
    return arr


def _endpoint_key(pt, ndigits=3):
    p = np.asarray(pt, float).reshape(2)
    return (round(float(p[0]), ndigits), round(float(p[1]), ndigits))


def orientation_cost(ref_pts, cand_pts):
    ref = _as_xy(ref_pts)
    cand = _as_xy(cand_pts)
    if len(ref) < 2 or len(cand) < 2:
        return np.nan, np.nan

    r0, r1 = ref[0], ref[-1]
    c0, c1 = cand[0], cand[-1]
    d_forward = float(np.linalg.norm(r0 - c0) + np.linalg.norm(r1 - c1))
    d_reverse = float(np.linalg.norm(r0 - c1) + np.linalg.norm(r1 - c0))
    return d_forward, d_reverse


def maybe_flip_segment(
    pts,
    normals=None,
    widths=None,
    edge1=None,
    edge2=None,
    *,
    force=False,
    normals_are_vectors=False,
):
    pts_arr = _as_xy(pts).copy()
    if len(pts_arr) < 2:
        return pts_arr, normals, widths, edge1, edge2

    if not force:
        return pts_arr, normals, widths, edge1, edge2

    pts_arr = pts_arr[::-1].copy()

    normals_arr = normals
    if normals is not None:
        normals_arr = np.asarray(normals, float)
        if normals_arr.ndim >= 1:
            normals_arr = normals_arr[::-1].copy()
            if normals_are_vectors:
                normals_arr = -normals_arr

    widths_arr = widths
    if widths is not None:
        widths_arr = np.asarray(widths, float).reshape(-1)[::-1].copy()

    edge1_arr = edge1
    if edge1 is not None:
        edge1_arr = np.asarray(edge1, float)
        if edge1_arr.ndim >= 1:
            edge1_arr = edge1_arr[::-1].copy()

    edge2_arr = edge2
    if edge2 is not None:
        edge2_arr = np.asarray(edge2, float)
        if edge2_arr.ndim >= 1:
            edge2_arr = edge2_arr[::-1].copy()

    return pts_arr, normals_arr, widths_arr, edge1_arr, edge2_arr


def orient_segment_to_reference(
    pts,
    *,
    ref_start,
    ref_end,
    normals=None,
    widths=None,
    edge1=None,
    edge2=None,
    normals_are_vectors=False,
):
    pts_arr = _as_xy(pts).copy()
    if len(pts_arr) < 2:
        return pts_arr, normals, widths, edge1, edge2, {
            "flipped": False,
            "d_forward": np.nan,
            "d_reverse": np.nan,
            "flag": "invalid",
        }

    ref = np.vstack([np.asarray(ref_start, float).reshape(2), np.asarray(ref_end, float).reshape(2)])
    d_forward, d_reverse = orientation_cost(ref, pts_arr)
    flip = bool(np.isfinite(d_forward) and np.isfinite(d_reverse) and (d_reverse < d_forward))

    pts_out, normals_out, widths_out, edge1_out, edge2_out = maybe_flip_segment(
        pts_arr,
        normals=normals,
        widths=widths,
        edge1=edge1,
        edge2=edge2,
        force=flip,
        normals_are_vectors=normals_are_vectors,
    )
    return pts_out, normals_out, widths_out, edge1_out, edge2_out, {
        "flipped": flip,
        "d_forward": d_forward,
        "d_reverse": d_reverse,
        "flag": "reversed_candidate" if flip else "forward_candidate",
    }


def canonicalize_segment_direction(
    pts,
    *,
    normals=None,
    widths=None,
    edge1=None,
    edge2=None,
    normals_are_vectors=False,
):
    pts_arr = _as_xy(pts).copy()
    if len(pts_arr) < 2:
        return pts_arr, normals, widths, edge1, edge2, {"flipped": False, "flag": "invalid"}

    k0 = _endpoint_key(pts_arr[0])
    k1 = _endpoint_key(pts_arr[-1])
    flip = bool(k1 < k0)
    pts_out, normals_out, widths_out, edge1_out, edge2_out = maybe_flip_segment(
        pts_arr,
        normals=normals,
        widths=widths,
        edge1=edge1,
        edge2=edge2,
        force=flip,
        normals_are_vectors=normals_are_vectors,
    )
    return pts_out, normals_out, widths_out, edge1_out, edge2_out, {
        "flipped": flip,
        "flag": "reversed_candidate" if flip else "forward_candidate",
    }


def enforce_branch_continuity(segments, associated_data=None):
    segs = [_as_xy(s).copy() for s in (segments or []) if _as_xy(s).shape[0] >= 2]
    if not segs:
        return [], ([] if associated_data is not None else None)

    if associated_data is None:
        assoc = [None] * len(segs)
    else:
        assoc = list(associated_data)
        if len(assoc) != len(segs):
            raise ValueError("associated_data length must match segments length")

    out_segs = [segs[0]]
    out_assoc = [assoc[0]]
    prev_end = out_segs[0][-1]

    for i in range(1, len(segs)):
        S = segs[i]
        d_fwd = float(np.linalg.norm(prev_end - S[0]))
        d_rev = float(np.linalg.norm(prev_end - S[-1]))
        flip = d_rev < d_fwd

        item = assoc[i]
        if isinstance(item, dict):
            n = item.get("normals", None)
            w = item.get("widths", None)
            e1 = item.get("edge1", None)
            e2 = item.get("edge2", None)
            nav = bool(item.get("normals_are_vectors", False))
            S2, n2, w2, e12, e22 = maybe_flip_segment(
                S, normals=n, widths=w, edge1=e1, edge2=e2, force=flip, normals_are_vectors=nav
            )
            item2 = dict(item)
            item2["normals"] = n2
            item2["widths"] = w2
            item2["edge1"] = e12
            item2["edge2"] = e22
        else:
            S2, _, _, _, _ = maybe_flip_segment(S, force=flip)
            item2 = item

        out_segs.append(S2)
        out_assoc.append(item2)
        prev_end = S2[-1]

    return out_segs, (out_assoc if associated_data is not None else None)


def canonicalize_branch_direction(segments, associated_data=None):
    segs = [_as_xy(s).copy() for s in (segments or []) if _as_xy(s).shape[0] >= 2]
    if not segs:
        return [], ([] if associated_data is not None else None), False

    if associated_data is None:
        assoc = [None] * len(segs)
    else:
        assoc = list(associated_data)
        if len(assoc) != len(segs):
            raise ValueError("associated_data length must match segments length")

    start_key = _endpoint_key(segs[0][0])
    end_key = _endpoint_key(segs[-1][-1])
    if end_key >= start_key:
        return segs, (assoc if associated_data is not None else None), False

    segs_rev = [s[::-1].copy() for s in segs[::-1]]
    if associated_data is None:
        return segs_rev, None, True

    assoc_rev = assoc[::-1]
    return segs_rev, assoc_rev, True


def assert_direction_consistency(segments, *, eps=1e-6):
    segs = [_as_xy(s) for s in (segments or []) if _as_xy(s).shape[0] >= 2]
    if len(segs) < 2:
        return True

    for i in range(len(segs) - 1):
        a = segs[i]
        b = segs[i + 1]
        d0 = float(np.linalg.norm(a[-1] - b[0]))
        d1 = float(np.linalg.norm(a[-1] - b[-1]))
        if d0 > d1 + float(eps):
            raise AssertionError(
                f"direction inconsistency at pair {i}->{i+1}: d_end_start={d0:.6f} > d_end_end={d1:.6f}"
            )
    return True
