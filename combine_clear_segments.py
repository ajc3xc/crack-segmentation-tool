#!/usr/bin/env python3
from shapely import geometry, ops
import numpy as np, cv2, os, matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

# ---------------------------------------------------------------------------
# Module-level helpers shared between combine and delete dialogs
# ---------------------------------------------------------------------------
def _segment_has_real_edges(crack):
    ge = crack.get("geodesic_edges")
    if not isinstance(ge, dict):
        return False
    e1 = ge.get("edge1") or []
    e2 = ge.get("edge2") or []
    def _n_real(lst):
        return sum(
            1 for p in lst
            if isinstance(p, (list, tuple)) and len(p) == 2
            and isinstance(p[0], (int, float)) and isinstance(p[1], (int, float))
        )
    return _n_real(e1) >= 2 and _n_real(e2) >= 2

def _segment_has_real_mask(crack):
    mc = crack.get("mask_crop")
    if mc is None:
        return False
    try:
        arr = np.asarray(mc, dtype=np.uint8)
        return arr.ndim == 2 and bool(np.any(arr))
    except Exception:
        return False

def _draw_midline_safe(img, crack, color, thickness=2):
    """Draw midline handling [None,None] segment separators correctly."""
    raw = crack.get("midline", [])
    if not raw:
        return
    segment, segments = [], []
    for pt in raw:
        if pt is None or (isinstance(pt, (list, tuple)) and len(pt) == 2
                          and (pt[0] is None or pt[1] is None)):
            if len(segment) >= 2:
                segments.append(segment)
            segment = []
        else:
            try:
                segment.append([float(pt[0]), float(pt[1])])
            except (TypeError, ValueError, IndexError):
                pass
    if len(segment) >= 2:
        segments.append(segment)
    for seg in segments:
        pts = np.round(np.array(seg, dtype=float)[:, :2]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, color, thickness, lineType=cv2.LINE_AA)

def _remaining_members_connected(cracks, threshold=60.0):
    """Return True if all cracks in the list form a single connected chain
    based on endpoint proximity. Used to detect when deleting a middle member
    breaks the combined crack into disconnected fragments."""
    if len(cracks) < 2:
        return True

    def _endpoints(crack):
        ml = crack.get("midline") or []
        # Filter out None separators
        pts = [p for p in ml if p is not None
               and isinstance(p, (list, tuple)) and len(p) == 2
               and p[0] is not None and p[1] is not None]
        if len(pts) < 2:
            return None, None
        return np.array(pts[0], float), np.array(pts[-1], float)

    endpoints = [_endpoints(c) for c in cracks]

    # Build adjacency: two cracks are adjacent if any endpoint pair is within threshold
    n = len(cracks)
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s_i, e_i = endpoints[i]
            s_j, e_j = endpoints[j]
            if s_i is None or s_j is None:
                continue
            dists = [
                np.linalg.norm(s_i - s_j),
                np.linalg.norm(s_i - e_j),
                np.linalg.norm(e_i - s_j),
                np.linalg.norm(e_i - e_j),
            ]
            if min(dists) <= threshold:
                adj[i].add(j)
                adj[j].add(i)

    # BFS from node 0 — if all nodes reachable, chain is connected
    visited = {0}
    queue = [0]
    while queue:
        node = queue.pop()
        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return len(visited) == n

from crackutils import *
from helpers.crackhelpers import *
import cracktools as ct
#from helpers.layout import Ui_MainWindow
from time import time

from combiner import gt_groups_from_midlines_and_gtmask

class CombineClearSegments(CrackUtils):        
    def combine_segments(self):
        """
        Combine multiple cracks (atomic or already-combined) into a new combined crack.
        - UI is the same as before.
        - Connectivity check is now consistent with auto_combine_segments:

            1) If a GT mask is available and valid:
                use gt_groups_from_midlines_and_gtmask to ensure the selection
                lies inside a single GT-group.
            2) Otherwise:
                use auto_groups_from_atomic to ensure the selection lies inside
                a single auto-group.
            3) Only if auto_groups_from_atomic returns no groups do we fall back
            to the old cracks_all_connected (overlap/endpoints) test.
        """
        from PyQt5.QtWidgets import (
            QDialog, QVBoxLayout, QListWidget, QPushButton, QHBoxLayout,
            QRadioButton, QButtonGroup, QLabel
        )
        from PyQt5.QtGui import QImage, QPixmap
        from PyQt5.QtCore import Qt
        import numpy as np, cv2

        from combiner import gt_groups_from_midlines_and_gtmask
        from helpers.combine_debug import auto_groups_from_atomic

        if not hasattr(self, "annotation") or not isinstance(self.annotation, dict):
            error("No annotation data loaded.")
            return

        ann_root = self.annotation.setdefault("annotations", {})
        atomic_cracks = ann_root.setdefault("atomic_cracks", {})
        combined_cracks = ann_root.setdefault("combined_cracks", {})

        H, W = self.original_image.shape[:2]

        # ------------------------------------------------------------------
        # Helpers (mask + legacy connectivity used only as final fallback)
        # ------------------------------------------------------------------
        def mask_from_crack(crack):
            mc = crack.get("mask_crop")
            bb = crack.get("mask_bbox")
            if mc is not None and bb is not None:
                crop = np.array(mc, dtype=np.uint8)
                x, y, w, h = [int(v) for v in bb]
                x2, y2 = min(x + w, W), min(y + h, H)
                w_eff, h_eff = max(0, x2 - x), max(0, y2 - y)
                if h_eff > 0 and w_eff > 0:
                    crop = (crop > 0).astype(np.uint8)[:h_eff, :w_eff]
                    m = np.zeros((H, W), np.uint8)
                    m[y:y + h_eff, x:x + w_eff] = crop
                    return m
            full = np.array(crack.get("mask", []), dtype=np.uint8)
            if full.size == H * W and full.shape == (H, W):
                return (full > 0).astype(np.uint8)
            return np.zeros((H, W), np.uint8)

        def cracks_overlap_or_connect(crackA, crackB):
            # Overlap check
            mA = mask_from_crack(crackA)
            mB = mask_from_crack(crackB)
            if np.any(mA & mB):
                return True
            # User endpoint check
            upA = [tuple(pt) for pt in crackA.get("user_points", [])]
            upB = [tuple(pt) for pt in crackB.get("user_points", [])]
            if set(upA) & set(upB):
                return True
            return False

        def cracks_all_connected(cracks):
            """Legacy: ensure all cracks form one connected component by overlap/endpoints."""
            n = len(cracks)
            if n < 2:
                return False
            adj = {i: set() for i in range(n)}
            for i in range(n):
                for j in range(i + 1, n):
                    if cracks_overlap_or_connect(cracks[i], cracks[j]):
                        adj[i].add(j)
                        adj[j].add(i)
            visited = set()
            stack = [0]
            while stack:
                u = stack.pop()
                if u in visited:
                    continue
                visited.add(u)
                stack.extend(adj[u] - visited)
            return len(visited) == n

        # ------------------------------------------------------------------
        # Build a unique display list where each atomic belongs to ≤1 combined
        # ------------------------------------------------------------------
        display_items = []  # list of (type, id)
        seen_atomic = set()

        # Add combined cracks first as single entries
        for cmb_id, cmb in sorted(combined_cracks.items(), key=lambda kv: int(kv[0])):
            members = cmb.get("members", [])
            if any(m in atomic_cracks for m in members):
                display_items.append(("combined", cmb_id))
                seen_atomic.update(members)

        # Then add remaining atomic cracks
        for atom_id in sorted(atomic_cracks.keys(), key=lambda s: int(s)):
            if atom_id not in seen_atomic:
                display_items.append(("atomic", atom_id))

        if len(display_items) < 2:
            error("Need at least two segments (atomic or combined) to combine.")
            return

        # ------------------------------------------------------------------
        # Dialog UI
        # ------------------------------------------------------------------
        dlg = QDialog(self.MainWindow)
        dlg.setWindowTitle("Combine Segments")
        layout = QVBoxLayout(dlg)
        layout.setSpacing(6)

        listwidget = QListWidget()
        listwidget.setSelectionMode(QListWidget.MultiSelection)
        for tpe, cid in display_items:
            if tpe == "atomic":
                lbl = f"Atomic {cid}"
            else:
                members = combined_cracks[cid].get("members", [])
                lbl = f"Combined {cid} (members: {','.join(members)})"
            listwidget.addItem(lbl)
        layout.addWidget(listwidget)

        # --- MODE SELECTOR ---
        rb_gt = QRadioButton("GT combining")
        rb_pred = QRadioButton("Pred combining")
        mode_group = QButtonGroup(dlg)
        mode_group.addButton(rb_gt)
        mode_group.addButton(rb_pred)
        rb_gt.setChecked(True)

        lbl_mode_info = QLabel("")
        lbl_mode_info.setStyleSheet("color: gray; font-size: 9pt;")
        lbl_mode_info.setVisible(False)

        layout.addWidget(rb_gt)
        layout.addWidget(rb_pred)
        layout.addWidget(lbl_mode_info)

        # Buttons directly under mode controls (tight spacing)
        btns = QHBoxLayout()
        btns.setContentsMargins(0, 0, 0, 0)
        btns.setSpacing(6)
        btn_ok = QPushButton("Combine Selected")
        btn_cancel = QPushButton("Cancel")
        btns.addWidget(btn_ok)
        btns.addWidget(btn_cancel)
        layout.addLayout(btns)

        btn_ok.clicked.connect(dlg.accept)
        btn_cancel.clicked.connect(dlg.reject)

        # ------------------------------------------------------------------
        # Highlight selection on the image
        # ------------------------------------------------------------------
        def highlight():
            display = self.original_image.copy()

            # Track which combined crack IDs are in display_items
            # so we don't double-draw them in the associated overlay below
            combined_in_list = {cid for tpe, cid in display_items if tpe == "combined"}

            for i, (tpe, cid) in enumerate(display_items):
                crack = atomic_cracks[cid] if tpe == "atomic" else combined_cracks[cid]
                m_full = mask_from_crack(crack)
                is_selected = listwidget.item(i).isSelected()
                is_combined = (tpe == "combined")
                if is_selected:
                    seg_color     = (255, 255, 0)   # yellow
                    midline_color = (0, 140, 0)     # dark green
                    alpha         = 0.55
                elif is_combined:
                    seg_color     = (255, 255, 255) # white
                    midline_color = (255, 0, 255)   # pink
                    alpha         = 0.35
                else:
                    seg_color     = (255, 0, 0)     # red
                    midline_color = (0, 0, 255)     # blue
                    alpha         = 0.25
                if np.any(m_full):
                    overlay = np.zeros_like(display)
                    overlay[m_full.astype(bool)] = seg_color
                    display = cv2.addWeighted(display, 1, overlay, alpha, 0)
                _draw_midline_safe(display, crack, midline_color, thickness=2)

            # --- Overlay existing combined cracks that share members with selection,
            #     but ONLY if they're not already shown as a display_item above ---
            selected_atomic_ids = set()
            for i, (tpe, cid) in enumerate(display_items):
                if listwidget.item(i).isSelected():
                    if tpe == "atomic":
                        selected_atomic_ids.add(cid)
                    else:
                        selected_atomic_ids.update(combined_cracks.get(cid, {}).get("members", []))

            if selected_atomic_ids:
                H_img, W_img = display.shape[:2]
                for cid, combo in combined_cracks.items():
                    if cid in combined_in_list:
                        continue  # already drawn above, don't double-draw
                    members = combo.get("members", [])
                    if not any(m in selected_atomic_ids for m in members):
                        continue
                    cm = reconstruct_full_mask_from_crack(combo, H_img, W_img)
                    if np.any(cm):
                        ov = np.zeros_like(display)
                        ov[cm.astype(bool)] = (255, 0, 255)
                        display = cv2.addWeighted(display, 1, ov, 0.30, 0)
                    _draw_midline_safe(display, combo, (180, 0, 180), thickness=4)
                    _draw_midline_safe(display, combo, (255, 0, 255), thickness=2)
                    ml_raw = [p for p in (combo.get("midline") or [])
                              if p is not None and isinstance(p, (list, tuple))
                              and len(p) == 2 and p[0] is not None]
                    if ml_raw:
                        lx, ly = int(round(ml_raw[0][0])), int(round(ml_raw[0][1]))
                        cv2.putText(display, f"C{cid}", (lx + 4, ly - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(display, f"C{cid}", (lx + 4, ly - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 1, cv2.LINE_AA)

            im = display.astype(np.uint8)
            qimage = QImage(im, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            scaled = pixmap.scaled(self.ImageScreen.width(), self.ImageScreen.height(),
                                Qt.KeepAspectRatio, Qt.FastTransformation)
            self.ImageScreen.setPixmap(scaled)

        def is_auto_segment(crack):
            return bool(crack.get("auto_midline")) or str(crack.get("source", "")).lower() == "auto"

        def _segment_usable_for_pred(crack):
            return _segment_has_real_edges(crack) or _segment_has_real_mask(crack)

        def _selected_atomic_ids():
            rows = [i.row() for i in listwidget.selectedIndexes()]
            out = set()
            for idx in rows:
                tpe, cid = display_items[idx]
                if tpe == "atomic":
                    out.add(cid)
                else:
                    out.update(combined_cracks.get(cid, {}).get("members", []))
            return out

        def update_mode_buttons():
            selected_ids = _selected_atomic_ids()

            # No selection yet: keep both available and neutral message.
            if not selected_ids:
                rb_gt.setEnabled(True)
                rb_pred.setEnabled(True)
                lbl_mode_info.setText("")
                lbl_mode_info.setVisible(False)
                return

            has_auto_sel = any(
                is_auto_segment(atomic_cracks.get(cid, {}))
                for cid in selected_ids
            )
            missing_pred = sorted(
                [
                    cid for cid in selected_ids
                    if (not is_auto_segment(atomic_cracks.get(cid, {})))
                    and (not _segment_usable_for_pred(atomic_cracks.get(cid, {})))
                ],
                key=int
            )

            if has_auto_sel:
                rb_gt.setEnabled(False)
                rb_pred.setEnabled(True)
                rb_pred.setChecked(True)
                lbl_mode_info.setText("GT unavailable: selection contains auto segments.")
                lbl_mode_info.setVisible(True)
            elif missing_pred:
                rb_gt.setEnabled(True)
                rb_pred.setEnabled(False)
                rb_gt.setChecked(True)
                ids = ", ".join(sorted(missing_pred, key=int))
                lbl_mode_info.setText(
                    f"Pred unavailable: Atomics {ids} have no edges or mask - GT only."
                )
                lbl_mode_info.setVisible(True)
            else:
                rb_gt.setEnabled(True)
                rb_pred.setEnabled(True)
                lbl_mode_info.setText("")
                lbl_mode_info.setVisible(False)
        listwidget.itemSelectionChanged.connect(update_mode_buttons)
        listwidget.itemSelectionChanged.connect(highlight)
        update_mode_buttons()
        highlight()

        if dlg.exec_() != QDialog.Accepted:
            self.change_image()
            return

        selected_rows = [i.row() for i in listwidget.selectedIndexes()]
        if len(selected_rows) < 2:
            error("Select at least two segments to combine.")
            self.change_image()
            return

        # ------------------------------------------------------------------
        # Gather all atomic members from selection
        # ------------------------------------------------------------------
        selected_atomic_ids = set()
        for idx in selected_rows:
            tpe, cid = display_items[idx]
            if tpe == "atomic":
                selected_atomic_ids.add(cid)
            else:
                selected_atomic_ids.update(combined_cracks[cid].get("members", []))

        # Final guard (authoritative) so execution matches UI gating.
        has_auto_sel = any(
            is_auto_segment(atomic_cracks.get(cid, {}))
            for cid in selected_atomic_ids
        )
        missing_pred = sorted(
            [
                cid for cid in selected_atomic_ids
                if (not is_auto_segment(atomic_cracks.get(cid, {})))
                and (not _segment_usable_for_pred(atomic_cracks.get(cid, {})))
            ],
            key=int
        )

        # ==========================================================
        # CONNECTIVITY CHECK (STRICT MODE: GT or PRED, no fallback)
        # ==========================================================
        candidate = set(selected_atomic_ids)
        if has_auto_sel:
            use_gt_mode = False
        elif missing_pred:
            use_gt_mode = True
        else:
            use_gt_mode = rb_gt.isChecked()
        connectivity_mode = "gt" if use_gt_mode else "pred"

        if use_gt_mode:
            gt_mask = None
            if hasattr(self, "current_mask") and self.current_mask is not None:
                gt_mask = np.asarray(self.current_mask)

            if gt_mask is None:
                error("GT mask selected but none available.")
                self.change_image()
                return

            if gt_mask.ndim == 3:
                gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
            if gt_mask.shape[:2] != (H, W) or not np.any(gt_mask):
                error("GT mask selected but unavailable/invalid for this image.")
                self.change_image()
                return

            groups_gt = gt_groups_from_midlines_and_gtmask(atomic_cracks, gt_mask, H, W)
            ok_gt = any(candidate.issubset(set(g["members"])) for g in groups_gt.values())
            if not ok_gt:
                error("Selected cracks not in a single GT group.")
                self.change_image()
                return
        else:
            groups_auto = auto_groups_from_atomic(
                atomic_cracks,
                image_hw=(H, W),
                px_thresh=10.0,
                debug_root=None,
            )
            if not groups_auto:
                error("No grouping found in auto/pred mode.")
                self.change_image()
                return

            ok_auto = any(candidate.issubset(set(g["members"])) for g in groups_auto.values())
            if not ok_auto:
                error("Selected cracks not in a single auto/pred group.")
                self.change_image()
                return

        # ------------------------------------------------------------------
        # Remove any existing combined fully contained by this new merge
        # ------------------------------------------------------------------
        to_delete = []
        for cmb_id, cmb in list(combined_cracks.items()):
            members = set(cmb.get("members", []))
            if members.issubset(selected_atomic_ids):
                to_delete.append(cmb_id)
        for cmb_id in to_delete:
            combined_cracks.pop(cmb_id, None)

        # ------------------------------------------------------------------
        # Allocate a new combined id
        # ------------------------------------------------------------------
        cmb_ids = []
        for k in combined_cracks.keys():
            try:
                cmb_ids.append(int(k))
            except Exception:
                pass
        new_cmb_id = str(max(cmb_ids) + 1 if cmb_ids else 0)

        # ------------------------------------------------------------------
        # Build the combined crack (midline + edges + widths + crop)
        # ------------------------------------------------------------------
        try:
            combined_entry = self._build_combined_crack(
                sorted(selected_atomic_ids, key=lambda s: int(s)),
                connectivity_mode=connectivity_mode,
            )
        except Exception as _comb_exc:
            import traceback
            print(f"[COMBINE CRASH] _build_combined_crack raised: {_comb_exc}", flush=True)
            traceback.print_exc()
            error(f"Combine crashed: {_comb_exc}")
            self.change_image()
            return
        if combined_entry is None:
            error("Failed to build combined crack (no valid midlines or masks).")
            self.change_image()
            return

        combined_cracks[new_cmb_id] = combined_entry
        self.annotation["annotations"]["combined_cracks"] = combined_cracks
        self.combined_cracks = combined_cracks  # keep self.combined_cracks in sync

        self.save_annotation()
        self.change_image()

    # crack_tool_v2.py  --- add to class CrackToolsApplication
    def clear_combined_cracks(self):
        """
        Delete (some or all) combined cracks; atomic cracks remain untouched.
        In the dialog:
        - All combined cracks are shown in yellow by default
        - Selected ones are shown in red
        """
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QListWidget, QPushButton, QHBoxLayout
        from PyQt5.QtGui import QImage, QPixmap
        import numpy as np, cv2

        if not hasattr(self, "annotation") or not isinstance(self.annotation, dict):
            error("No annotation data loaded.")
            return

        ann = self.annotation.setdefault("annotations", {})
        combined = ann.setdefault("combined_cracks", {})
        atomic = ann.setdefault("atomic_cracks", {})
        if not combined:
            error("No combined cracks to delete.")
            return

        keys_sorted = sorted(combined.keys(), key=lambda s: int(s))
        combined_member_ids = set()
        for cmb in combined.values():
            combined_member_ids.update(cmb.get("members", []))
        labels = []
        H, W = self.original_image.shape[:2]
        for cid in keys_sorted:
            members = combined[cid].get("members", [])
            labels.append(f"Combined {cid}  (members: {','.join(map(str, members))})")

        dlg = QDialog(self.MainWindow)
        dlg.setWindowTitle("Clear (Delete) Combined Cracks")
        layout = QVBoxLayout(dlg)
        lw = QListWidget()
        lw.setSelectionMode(QListWidget.MultiSelection)
        for lbl in labels:
            lw.addItem(lbl)
        layout.addWidget(lw)

        btns = QHBoxLayout()
        ok = QPushButton("Delete Selected")
        cancel = QPushButton("Cancel")
        btns.addWidget(ok)
        btns.addWidget(cancel)
        layout.addLayout(btns)
        ok.clicked.connect(dlg.accept)
        cancel.clicked.connect(dlg.reject)

        # Highlight function
        def highlight():
            def draw_midline(img, crack, color, thickness=2):
                """Draw midline handling [None,None] segment separators correctly."""
                raw = crack.get("midline", [])
                if not raw:
                    return
                segment, segments = [], []
                for pt in raw:
                    if pt is None or (isinstance(pt, (list, tuple)) and len(pt) == 2
                                      and (pt[0] is None or pt[1] is None)):
                        if len(segment) >= 2:
                            segments.append(segment)
                        segment = []
                    else:
                        try:
                            segment.append([float(pt[0]), float(pt[1])])
                        except (TypeError, ValueError, IndexError):
                            pass
                if len(segment) >= 2:
                    segments.append(segment)
                for seg in segments:
                    pts = np.round(np.array(seg, dtype=float)[:, :2]).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(img, [pts], False, color, thickness, lineType=cv2.LINE_AA)

            disp = self.original_image.copy()
            for i, cid in enumerate(keys_sorted):
                mask = reconstruct_full_mask_from_crack(combined[cid], H, W)
                if not np.any(mask):
                    continue
                overlay = np.zeros_like(disp)
                if lw.item(i).isSelected():
                    overlay[mask.astype(bool)] = (255, 255, 0)  # yellow
                    alpha = 0.6
                else:
                    overlay[mask.astype(bool)] = (255, 255, 255)  # white
                    alpha = 0.35
                disp = cv2.addWeighted(disp, 1, overlay, alpha, 0)

            # Show combined midlines (selected in dark green, unselected in magenta).
            for i, cid in enumerate(keys_sorted):
                cmb = combined[cid]
                midline_color = (0, 140, 0) if lw.item(i).isSelected() else (255, 0, 255)
                draw_midline(disp, cmb, midline_color, thickness=2)

            # Show non-combined atomic midlines in blue.
            for atom_id, crack in atomic.items():
                if atom_id in combined_member_ids:
                    continue
                draw_midline(disp, crack, (0, 0, 255), thickness=2)

            # For selected combined cracks, highlight member atomic midlines in green only.
            for i, cid in enumerate(keys_sorted):
                if not lw.item(i).isSelected():
                    continue
                members = combined[cid].get("members", [])
                for m_id in members:
                    crack = atomic.get(m_id)
                    if crack is None:
                        continue
                    draw_midline(disp, crack, (0, 180, 0), thickness=2)
            qimage = QImage(disp, disp.shape[1], disp.shape[0], disp.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            self.ImageScreen.setPixmap(
                pixmap.scaled(self.ImageScreen.width(), self.ImageScreen.height(),
                            Qt.KeepAspectRatio, Qt.FastTransformation)
            )

        lw.itemSelectionChanged.connect(highlight)
        highlight()

        if dlg.exec_() == QDialog.Accepted:
            sel = [r.row() for r in lw.selectedIndexes()]
            for idx in sorted(sel, reverse=True):
                combined.pop(keys_sorted[idx], None)
            self.annotation["annotations"]["combined_cracks"] = combined
            self.combined_cracks = combined  # sync so save_annotation persists correctly

            self.save_annotation()
            self.change_image()
        else:
            self.change_image()

    def clear_segmentation(self):
        from PyQt5.QtWidgets import (
            QDialog, QVBoxLayout, QListWidget, QPushButton, QHBoxLayout,
            QRadioButton, QButtonGroup, QLabel
        )
        import numpy as np, cv2

        if not hasattr(self, "annotation") or not isinstance(self.annotation, dict):
            error("No annotation data loaded.")
            return

        ann = self.annotation.get("annotations", {})
        atomic_cracks = ann.setdefault("atomic_cracks", {})
        combined_cracks = ann.setdefault("combined_cracks", {})

        masks = []
        labels = []
        items = []

        H, W = self.original_image.shape[:2]

        # --- Atomic cracks only ---
        for crack_id, crack in atomic_cracks.items():
            m = reconstruct_full_mask_from_crack(crack, H, W)
            masks.append(m)
            labels.append(f"Atomic {crack_id}" + ("" if np.any(m) else " (empty)"))
            items.append(("atomic", crack_id))

        if not masks:
            error("No atomic cracks to delete.")
            return

        # --- Selection dialog ---
        dlg = QDialog(self.MainWindow)
        dlg.setWindowTitle("Select Atomic Segments to Delete")
        layout = QVBoxLayout(dlg)
        layout.setSpacing(6)
        listwidget = QListWidget()
        listwidget.setSelectionMode(QListWidget.MultiSelection)
        for lbl in labels:
            listwidget.addItem(lbl)
        layout.addWidget(listwidget)

        # --- Mode selector (same semantics as combine_segments) ---
        rb_gt = QRadioButton("GT combining")
        rb_pred = QRadioButton("Pred combining")
        mode_group = QButtonGroup(dlg)
        mode_group.addButton(rb_gt)
        mode_group.addButton(rb_pred)
        rb_gt.setChecked(True)

        lbl_mode_info = QLabel("")
        lbl_mode_info.setStyleSheet("color: gray; font-size: 9pt;")
        lbl_mode_info.setVisible(False)
        layout.addWidget(rb_gt)
        layout.addWidget(rb_pred)
        layout.addWidget(lbl_mode_info)

        btns = QHBoxLayout()
        btns.setContentsMargins(0, 0, 0, 0)
        btns.setSpacing(6)
        btn_ok = QPushButton("Delete Selected")
        btn_cancel = QPushButton("Cancel")
        btns.addWidget(btn_ok)
        btns.addWidget(btn_cancel)
        layout.addLayout(btns)

        btn_ok.clicked.connect(dlg.accept)
        btn_cancel.clicked.connect(dlg.reject)

        def highlight_selected_segments():
            display = self.original_image.copy()

            in_combined = {m for combo in combined_cracks.values()
                           for m in combo.get("members", [])}

            selected_ids = {items[i][1] for i in range(len(items))
                            if listwidget.item(i).isSelected()}

            co_members = set()
            for combo in combined_cracks.values():
                members = combo.get("members", [])
                if any(m in selected_ids for m in members):
                    co_members.update(m for m in members if m not in selected_ids)

            # ---- Pass 1: masks only ----
            # Combined masks first (background), then uncombined atomic masks on top
            for cid, combo in combined_cracks.items():
                members  = combo.get("members", [])
                is_affected = any(m in selected_ids for m in members)
                cm = reconstruct_full_mask_from_crack(combo, H, W)
                if np.any(cm):
                    ov = np.zeros_like(display)
                    ov[cm.astype(bool)] = (255, 255, 0) if is_affected else (255, 255, 255)
                    display = cv2.addWeighted(display, 1, ov, 0.50 if is_affected else 0.30, 0)

            for i, m in enumerate(masks):
                crack_id = items[i][1]
                if crack_id in in_combined:
                    continue   # mask shown via combined overlay above
                is_selected = crack_id in selected_ids
                seg_color = (255, 255, 0) if is_selected else (255, 0, 0)
                if np.any(m):
                    ov = np.zeros_like(display)
                    ov[m.astype(bool)] = seg_color
                    display = cv2.addWeighted(display, 1, ov, 0.40, 0)

            # ---- Pass 2: combined midlines ----
            for cid, combo in combined_cracks.items():
                members = combo.get("members", [])
                is_affected = any(m in selected_ids for m in members)
                line_color = (0, 140, 0) if is_affected else (255, 0, 255)
                _draw_midline_safe(display, combo, (0, 0, 0),  thickness=4)
                _draw_midline_safe(display, combo, line_color, thickness=2)
                ml_raw = [p for p in (combo.get("midline") or [])
                          if p is not None and isinstance(p, (list, tuple))
                          and len(p) == 2 and p[0] is not None]
                if ml_raw:
                    lx, ly = int(round(ml_raw[0][0])), int(round(ml_raw[0][1]))
                    cv2.putText(display, f"C{cid}", (lx + 4, ly - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0),       3, cv2.LINE_AA)
                    cv2.putText(display, f"C{cid}", (lx + 4, ly - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255),   1, cv2.LINE_AA)

            # ---- Pass 3: atomic midlines on top so they're always visible ----
            for i, _ in enumerate(masks):
                crack_id     = items[i][1]
                is_selected  = crack_id in selected_ids
                is_co_member = crack_id in co_members
                if is_selected:
                    midline_color = (0, 140, 0)    # dark green
                elif is_co_member:
                    midline_color = (0, 220, 255)  # cyan
                else:
                    midline_color = (0, 0, 255)    # blue
                crack = atomic_cracks.get(crack_id, {})
                _draw_midline_safe(display, crack, midline_color, thickness=2)

            from PyQt5.QtGui import QImage, QPixmap
            qimage = QImage(display, display.shape[1], display.shape[0],
                            display.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            self.ImageScreen.setPixmap(
                pixmap.scaled(self.ImageScreen.width(), self.ImageScreen.height(),
                            Qt.KeepAspectRatio, Qt.FastTransformation)
            )

        listwidget.itemSelectionChanged.connect(highlight_selected_segments)
        highlight_selected_segments()

        def is_auto_segment(crack):
            return bool(crack.get("auto_midline")) or str(crack.get("source", "")).lower() == "auto"

        _gt_mask_raw   = getattr(self, "current_mask",          None)
        _pred_mask_raw = getattr(self, "full_prediction_mask", None)
        gt_mask_ok   = _gt_mask_raw   is not None and np.any(np.asarray(_gt_mask_raw))
        pred_mask_ok = _pred_mask_raw is not None and np.any(np.asarray(_pred_mask_raw))

        def update_mode_for_delete():
            selected_indices = [i.row() for i in listwidget.selectedIndexes()]
            if not selected_indices:
                rb_gt.setEnabled(False)
                rb_pred.setEnabled(False)
                lbl_mode_info.setText("")
                lbl_mode_info.setVisible(False)
                return

            selected_ids = {items[i][1] for i in selected_indices}

            # Separate affected combined cracks into those that will be rebuilt
            # vs those that will be fully deleted (< 2 remaining members)
            will_rebuild = {}   # cid -> list of remaining atomic crack objects
            will_delete  = []   # cids that will be fully dropped

            for cid, combo in combined_cracks.items():
                members = combo.get("members", [])
                if any(m in selected_ids for m in members):
                    remaining = [atomic_cracks[m] for m in members
                                 if m not in selected_ids and m in atomic_cracks]
                    if len(remaining) >= 2 and _remaining_members_connected(remaining):
                        will_rebuild[cid] = remaining
                    else:
                        will_delete.append(cid)

            # No combined cracks touched at all — mode irrelevant, gray out
            if not will_rebuild and not will_delete:
                rb_gt.setEnabled(False)
                rb_pred.setEnabled(False)
                lbl_mode_info.setText("No combined cracks affected — mode irrelevant.")
                lbl_mode_info.setVisible(True)
                return

            # All affected combined cracks will be fully deleted — mode irrelevant
            if not will_rebuild:
                rb_gt.setEnabled(False)
                rb_pred.setEnabled(False)
                lbl_mode_info.setText(
                    f"Combined crack(s) {', '.join(will_delete)} will be fully removed "
                    f"(too few members remaining or deletion breaks chain) — mode irrelevant."
                )
                lbl_mode_info.setVisible(True)
                return

            # Some combined cracks need rebuilding — determine mode per crack
            any_forced_pred  = False
            any_forced_gt    = False
            force_pred_reasons = []
            force_gt_reasons   = []

            for cid, remaining in will_rebuild.items():
                has_auto_in_remaining = any(is_auto_segment(c) for c in remaining)
                if has_auto_in_remaining:
                    any_forced_pred = True
                    force_pred_reasons.append(f"Combined {cid} has auto member")
                else:
                    all_have_pred = all(
                        _segment_has_real_edges(c) or _segment_has_real_mask(c)
                        for c in remaining
                    )
                    if not all_have_pred:
                        any_forced_gt = True
                        force_gt_reasons.append(f"Combined {cid} members lack pred data")

            delete_suffix = (f"  (Combined {', '.join(will_delete)} fully removed.)"
                             if will_delete else "")

            if any_forced_pred and not any_forced_gt:
                rb_gt.setEnabled(False)
                rb_pred.setEnabled(True)
                rb_pred.setChecked(True)
                lbl_mode_info.setText(
                    "Pred mode forced: " + "; ".join(force_pred_reasons) + delete_suffix
                )
                lbl_mode_info.setVisible(True)
            elif any_forced_gt and not any_forced_pred:
                rb_gt.setEnabled(True)
                rb_pred.setEnabled(False)
                rb_gt.setChecked(True)
                lbl_mode_info.setText(
                    "GT mode forced: " + "; ".join(force_gt_reasons) + delete_suffix
                )
                lbl_mode_info.setVisible(True)
            elif any_forced_pred and any_forced_gt:
                rb_gt.setEnabled(False)
                rb_pred.setEnabled(True)
                rb_pred.setChecked(True)
                lbl_mode_info.setText(
                    "Pred mode forced (mixed constraints — pred is safest)." + delete_suffix
                )
                lbl_mode_info.setVisible(True)
            else:
                # All rebuilds are all-manual with pred data — user can freely choose
                rb_gt.setEnabled(gt_mask_ok)
                rb_pred.setEnabled(pred_mask_ok)
                mode_str = "GT" if rb_gt.isChecked() else "Pred"
                lbl_mode_info.setText(
                    f"All affected combined cracks are manual — {mode_str} mode active. "
                    f"({'GT mask OK' if gt_mask_ok else 'GT mask missing'} / "
                    f"{'Pred mask OK' if pred_mask_ok else 'Pred mask missing'})"
                    + delete_suffix
                )
                lbl_mode_info.setVisible(True)

        rb_gt.toggled.connect(update_mode_for_delete)
        listwidget.itemSelectionChanged.connect(update_mode_for_delete)
        update_mode_for_delete()

        if dlg.exec_() == QDialog.Accepted:
            selected_indices = [i.row() for i in listwidget.selectedIndexes()]
            if not selected_indices:
                self.change_image()
                return

            use_gt_mode = rb_gt.isChecked()
            connectivity_mode = "gt" if use_gt_mode else "pred"

            # Strict mode validation: no silent fallback.
            if connectivity_mode == "gt":
                m = getattr(self, "current_mask", None)
                if m is None or not np.any(np.asarray(m)):
                    error("GT mask selected but unavailable/invalid.")
                    self.change_image()
                    return
            else:
                m = getattr(self, "full_prediction_mask", None)
                if m is None or not np.any(np.asarray(m)):
                    error("Pred mask selected but unavailable/invalid.")
                    self.change_image()
                    return

            # --- Delete selected ---
            for idx in sorted(selected_indices, reverse=True):
                tpe, crack_id = items[idx]
                if tpe == "atomic":
                    atomic_cracks.pop(crack_id, None)
                    # Remove from combined_cracks members if present
                    for cid, combo in list(combined_cracks.items()):
                        combo["members"] = [m for m in combo.get("members", []) if m in atomic_cracks]

                        # Delete if fewer than 2 members remain
                        if len(combo["members"]) < 2:
                            combined_cracks.pop(cid, None)
                            continue

                        # Get full image size
                        H, W = self.original_image.shape[:2]

                        # --- Rebuild union mask ---
                        union_mask = np.zeros((H, W), dtype=np.uint8)
                        for m_id in combo["members"]:
                            member_crack = atomic_cracks.get(m_id)
                            if member_crack is not None:
                                union_mask |= reconstruct_full_mask_from_crack(member_crack, H, W)

                        if np.any(union_mask):
                            ys, xs = np.where(union_mask > 0)
                            y0, y1 = int(ys.min()), int(ys.max() + 1)
                            x0, x1 = int(xs.min()), int(xs.max() + 1)
                            crop = union_mask[y0:y1, x0:x1].astype(np.uint8)
                            combo["mask_crop"] = crop.tolist()
                            combo["mask_bbox"] = [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]
                        else:
                            # Members have empty masks (normal for manual_poly segments before ET).
                            # Do NOT delete — _build_combined_crack in the remap section
                            # will recompute geometry properly. Just clear stale mask fields.
                            combo.pop("mask_crop", None)
                            combo.pop("mask_bbox", None)
                            print(f"[CLEAR_DEL_LOOP] combined {cid}: union mask empty "
                                  f"(members have no mask yet) — keeping for ET rebuild", flush=True)

            # --- Reindex atomic cracks only ---
            # --- Reindex atomic cracks (build old->new map first) ---
            if atomic_cracks:
                old_ids_sorted = sorted(atomic_cracks.keys(), key=lambda x: int(x))
                old_to_new = {old_id: str(i) for i, old_id in enumerate(old_ids_sorted)}
                new_atomic = {old_to_new[old_id]: atomic_cracks[old_id] for old_id in old_ids_sorted}
                atomic_cracks.clear()
                atomic_cracks.update(new_atomic)

            # --- Remap combined crack members after atomic reindex, drop small ones, and rebuild full geometry ---
            if combined_cracks:
                to_delete = []
                for cid, combo in list(combined_cracks.items()):
                    members_old = combo.get("members", [])
                    # remap to new ids; keep only those still present
                    members_new = [old_to_new[m] for m in members_old if m in old_to_new]
                    members_new = sorted(members_new, key=lambda s: int(s))

                    if len(members_new) < 2:
                        to_delete.append(cid)
                        continue

                    # Per-crack mode: if any remaining member is auto → pred, else GT
                    _remaining_cracks = [atomic_cracks[m] for m in members_new if m in atomic_cracks]
                    _any_auto = any(is_auto_segment(c) for c in _remaining_cracks)
                    _crack_mode = "pred" if _any_auto else "gt"

                    # Validate mask availability for chosen mode
                    if _crack_mode == "gt":
                        _gt_mask = getattr(self, "current_mask", None)
                        if _gt_mask is None or not np.any(np.asarray(_gt_mask)):
                            print(f"[CLEAR_SEG] combined {cid}: GT mask unavailable, falling back to pred", flush=True)
                            _crack_mode = "pred"

                    print(f"[CLEAR_SEG] rebuilding combined {cid} members={members_new} mode={_crack_mode}", flush=True)

                    # Recompute full combined geometry
                    try:
                        combined_entry = self._build_combined_crack(
                            members_new,
                            connectivity_mode=_crack_mode,
                        )
                    except Exception as _rebuild_exc:
                        import traceback
                        print(f"[CLEAR_SEG][WARN] _build_combined_crack raised: {_rebuild_exc}", flush=True)
                        traceback.print_exc()
                        combined_entry = None

                    if combined_entry is None:
                        print(f"[CLEAR_SEG][WARN] rebuild returned None for combined {cid} — keeping with union mask only", flush=True)
                        # Graceful fallback: keep the combined crack with the union mask
                        # that was already rebuilt in the deletion loop rather than deleting it
                        combo = combined_cracks.get(cid)
                        if combo is not None:
                            combo["members"] = members_new
                            # Clear stale ET geometry so it doesn't mislead
                            combo.pop("geodesic_edges", None)
                            combo.pop("midline", None)
                            combo.pop("normal_edge_points", None)
                        continue

                    combined_cracks[cid] = combined_entry

                for cid in to_delete:
                    combined_cracks.pop(cid, None)

            self.annotation["annotations"]["atomic_cracks"] = atomic_cracks
            self.annotation["annotations"]["combined_cracks"] = combined_cracks
            self.combined_cracks = combined_cracks  # keep self.combined_cracks in sync

            # --- Save + refresh ---
            self.save_annotation()
            self.change_image()
        else:
            self.change_image()
 
    '''def auto_combine_segments(self):
        """
        Automatically combine atomic cracks that overlap, connect, or are close in midline space.
        Reuses _build_combined_crack for consistency.
        If an atomic crack already belongs to a combined crack, it will extend that
        combined crack when new overlaps/branches are detected.
        """
        import numpy as np
        from scipy.spatial.distance import cdist

        if not hasattr(self, "original_image") or self.original_image is None:
            print("⚠️ No image loaded — skipping auto_combine_segments.")
            return
        if not hasattr(self, "annotation") or not self.annotation:
            print("⚠️ No annotation loaded — skipping auto_combine_segments.")
            return

        ann = self.annotation.setdefault("annotations", {})
        atomic = ann.setdefault("atomic_cracks", {})
        combined = ann.setdefault("combined_cracks", {})

        H, W = self.original_image.shape[:2]

        # --- Helper: reconstruct full mask if present ---
        def mask_from_crack(crack):
            mc, bb = crack.get("mask_crop"), crack.get("mask_bbox")
            if mc is not None and bb is not None:
                crop = np.array(mc, dtype=np.uint8)
                x, y, w, h = [int(v) for v in bb]
                x2, y2 = min(x+w, W), min(y+h, H)
                w_eff, h_eff = max(0, x2-x), max(0, y2-y)
                if h_eff > 0 and w_eff > 0:
                    crop = (crop > 0).astype(np.uint8)[:h_eff, :w_eff]
                    m = np.zeros((H, W), dtype=np.uint8)
                    m[y:y+h_eff, x:x+w_eff] = crop
                    return m
            full = np.array(crack.get("mask", []), dtype=np.uint8)
            if full.size == H*W and full.shape == (H, W):
                return (full > 0).astype(np.uint8)
            return np.zeros((H, W), dtype=np.uint8)

        # --- Overlap / connection / proximity test ---
        def cracks_overlap_or_connect(crackA, crackB, px_thresh=10.0):
            # mask overlap
            mA = mask_from_crack(crackA)
            mB = mask_from_crack(crackB)
            if np.any(mA & mB):
                print(f"[COMBINE_DBG] mask overlap between {crackA.get('id','?')} and {crackB.get('id','?')}")
                return True

            # shared user endpoints
            upA = [tuple(pt) for pt in crackA.get("user_points", [])]
            upB = [tuple(pt) for pt in crackB.get("user_points", [])]
            if set(upA) & set(upB):
                print(f"[COMBINE_DBG] shared endpoints between {crackA.get('id','?')} and {crackB.get('id','?')}")
                return True

            # --- NEW: midline proximity fallback ---
            a = np.asarray(crackA.get("midline", []), float)
            b = np.asarray(crackB.get("midline", []), float)
            if a.size and b.size:
                try:
                    dmin = np.min(cdist(a, b))
                    if np.isfinite(dmin) and dmin < px_thresh:
                        print(f"[COMBINE_DBG] midline proximity {dmin:.2f}px between {crackA.get('id','?')} and {crackB.get('id','?')}")
                        return True
                except Exception:
                    pass
            return False

        # --- Build list of entries ---
        seen_atomic = set(m for cmb in combined.values() for m in cmb.get("members", []))
        entries = [("combined", cid) for cid in combined.keys()]
        entries.extend(("atomic", aid) for aid in atomic.keys() if aid not in seen_atomic)

        print(f"[COMBINE_DBG] starting auto_combine_segments with {len(atomic)} atomics")

        # --- Check for attachments / new combines ---
        for tpe, cid in list(entries):
            if tpe != "atomic":
                continue
            crack = atomic[cid]
            attached_to = None
            for cmb_id, cmb in combined.items():
                for m in cmb.get("members", []):
                    if cracks_overlap_or_connect(crack, atomic.get(m, {})):
                        attached_to = cmb_id
                        break
                if attached_to:
                    break
            if attached_to:
                members = set(combined[attached_to]["members"])
                members_clean = [m for m in members if str(m).isdigit()]
                if not members_clean:
                    continue
                combined[attached_to] = self._build_combined_crack(sorted(members_clean, key=lambda s: int(s)))
                print(f"[COMBINE_DBG] Extended combined {attached_to} with atomic {cid}")
            else:
                overlaps = [cid]
                for tpe2, cid2 in entries:
                    if tpe2 == "atomic" and cid2 != cid:
                        if cracks_overlap_or_connect(crack, atomic[cid2]):
                            overlaps.append(cid2)
                if len(overlaps) > 1:
                    new_id = str(max([int(k) for k in combined.keys() if k.isdigit()] or [-1]) + 1)
                    combined[new_id] = self._build_combined_crack(sorted(overlaps, key=lambda s: int(s)))
                    print(f"[COMBINE_DBG] Auto-created combined {new_id} from atomics {overlaps}")

        self.save_annotation()
        self.change_image()'''
        
    def auto_combine_segments(self):
        """
        Automatically combine atomic cracks in strict mode:
        - If any auto-derived atomic exists: force Pred grouping
        - Else: use GT grouping only if a valid GT mask exists
        No silent fallback between modes.

        Then rebuild self.annotation['annotations']['combined_cracks']
        by calling self._build_combined_crack() for each group.
        """

        import numpy as np, cv2
        from combiner import gt_groups_from_midlines_and_gtmask
        from helpers.combine_debug import auto_groups_from_atomic

        if not hasattr(self, "original_image") or self.original_image is None:
            print("⚠️ No image loaded — skipping auto_combine_segments.")
            return
        if not hasattr(self, "annotation") or not self.annotation:
            print("⚠️ No annotation loaded — skipping auto_combine_segments.")
            return

        ann = self.annotation.setdefault("annotations", {})
        atomic = ann.setdefault("atomic_cracks", {})
        combined = ann.setdefault("combined_cracks", {})

        H, W = self.original_image.shape[:2]

        # -----------------------------------------
        # Locate GT mask (if used)
        # -----------------------------------------
        gt_mask = None

        if hasattr(self, "current_mask") and self.current_mask is not None:
            gt_mask = np.asarray(self.current_mask)

        if gt_mask is not None:
            # convert to gray if needed
            if gt_mask.ndim == 3:
                gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
            if gt_mask.shape[:2] != (H, W):
                print(f"[COMBINE_DBG] gt_mask shape {gt_mask.shape[:2]} != image {(H, W)} — ignoring GT.")
                gt_mask = None

        def is_auto_segment(crack):
            return bool(crack.get("auto_midline")) or str(crack.get("source", "")).lower() == "auto"

        has_auto = any(is_auto_segment(cr) for cr in atomic.values())
        use_gt = not has_auto

        if use_gt:
            if gt_mask is not None and np.any(gt_mask):
                print("[COMBINE_DBG] using GT-based grouping via gt_groups_from_midlines_and_gtmask()")
                groups = gt_groups_from_midlines_and_gtmask(atomic, gt_mask, H, W)
            else:
                print("[COMBINE_DBG] GT mode selected but no valid GT mask; skipping auto-combine.")
                return
        else:
            print("[COMBINE_DBG] using Pred/auto grouping via auto_groups_from_atomic()")
            groups = auto_groups_from_atomic(
                atomic,
                image_hw=(H, W),
                px_thresh=10.0,
                debug_root=None
            )

        print(f"[COMBINE_DBG] grouping produced {len(groups)} combined groups: {groups}")

        # -----------------------------------------
        # Rebuild combined_cracks from groups
        # (GT / auto grouping is authoritative)
        # -----------------------------------------
        combined.clear()

        for gid, ginfo in groups.items():
            members = ginfo.get("members") or []
            if len(members) < 2:
                continue

            try:
                sorted_members = sorted(members, key=lambda s: int(s))
            except Exception:
                sorted_members = sorted(members)

            entry = self._build_combined_crack(
                sorted_members,
                connectivity_mode=("gt" if use_gt else "pred"),
            )
            if entry is None:
                print(f"[COMBINE_DBG] _build_combined_crack failed for group {gid} ({members})")
                continue

            combined[gid] = entry
            print(f"[COMBINE_DBG] combined[{gid}] ← members={sorted_members}")

        self.save_annotation()
        self.change_image()
