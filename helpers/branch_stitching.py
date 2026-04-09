import numpy as np


def stitch_branch_segments(seg_list, max_jump=10.0, allow_teleport=False):
    """
    Order + orient segments into a single chain.

    Returns:
        S_chain: (N,2) array or None
        ok: bool
        reason: str
    """

    segs = [np.asarray(s, float) for s in seg_list if s is not None and len(s) >= 2]
    if not segs:
        return None, False, "no valid segs"

    used = set()
    chain = [segs[0]]
    used.add(0)

    while len(used) < len(segs):
        cur_end = chain[-1][-1]
        cur_start = chain[0][0]

        best_j = None
        best_dist = float("inf")
        best_oriented = None
        best_action = None  # "append" or "prepend"

        for j, sj in enumerate(segs):
            if j in used:
                continue

            a = sj[0]
            b = sj[-1]

            # append cases
            d1 = np.linalg.norm(cur_end - a)
            d2 = np.linalg.norm(cur_end - b)

            if d1 < best_dist:
                best_dist = d1
                best_j = j
                best_oriented = sj
                best_action = "append"

            if d2 < best_dist:
                best_dist = d2
                best_j = j
                best_oriented = sj[::-1]
                best_action = "append"

            # prepend cases
            d3 = np.linalg.norm(b - cur_start)
            d4 = np.linalg.norm(a - cur_start)

            if d3 < best_dist:
                best_dist = d3
                best_j = j
                best_oriented = sj
                best_action = "prepend"

            if d4 < best_dist:
                best_dist = d4
                best_j = j
                best_oriented = sj[::-1]
                best_action = "prepend"

        if best_j is None:
            break

        if best_action == "prepend":
            chain.insert(0, best_oriented)
        else:
            chain.append(best_oriented)

        used.add(best_j)

    if len(used) != len(segs):
        return None, False, "incomplete ordering"

    S_chain = np.vstack(chain)

    # detect teleport
    if len(S_chain) >= 2:
        d = np.linalg.norm(np.diff(S_chain, axis=0), axis=1)
        if np.any(d > max_jump):
            if not allow_teleport:
                return None, False, "teleport detected"

    return S_chain, True, "ok"
