#!/usr/bin/env python3
from shapely import geometry, ops
import numpy as np, cv2, os, matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

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

        def highlight():
            display = self.original_image.copy()
            for i, (tpe, cid) in enumerate(display_items):
                crack = atomic_cracks[cid] if tpe == "atomic" else combined_cracks[cid]
                m_full = mask_from_crack(crack)
                is_selected = listwidget.item(i).isSelected()
                seg_color = (255, 255, 0) if is_selected else (255, 0, 0)  # yellow / red
                midline_color = (0, 140, 0) if is_selected else (0, 0, 255)  # dark green / blue
                alpha = 0.6 if is_selected else 0.25
                if np.any(m_full):
                    overlay = np.zeros_like(display)
                    overlay[m_full.astype(bool)] = seg_color
                    display = cv2.addWeighted(display, 1, overlay, alpha, 0)
                _draw_midline_safe(display, crack, midline_color, thickness=2)
            im = display.astype(np.uint8)
            qimage = QImage(im, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            scaled = pixmap.scaled(self.ImageScreen.width(), self.ImageScreen.height(),
                                Qt.KeepAspectRatio, Qt.FastTransformation)
            self.ImageScreen.setPixmap(scaled)

        def is_auto_segment(crack):
            return bool(crack.get("auto_midline")) or str(crack.get("source", "")).lower() == "auto"

        def _segment_has_real_edges(crack):
            ge = crack.get("geodesic_edges")
            if not isinstance(ge, dict):
                return False
            e1 = ge.get("edge1") or []
            e2 = ge.get("edge2") or []

            def _n_real(lst):
                return sum(
                    1 for p in lst
                    if isinstance(p, (list, tuple))
                    and len(p) == 2
                    and isinstance(p[0], (int, float))
                    and isinstance(p[1], (int, float))
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
                    overlay[mask.astype(bool)] = (255, 0, 0)  # red if selected
                    alpha = 0.6
                else:
                    overlay[mask.astype(bool)] = (255, 255, 0)  # yellow if not selected
                    alpha = 0.4
                disp = cv2.addWeighted(disp, 1, overlay, alpha, 0)

            # Show combined midlines (selected in dark green, unselected in blue).
            for i, cid in enumerate(keys_sorted):
                cmb = combined[cid]
                midline_color = (0, 140, 0) if lw.item(i).isSelected() else (0, 0, 255)
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
            for i, m in enumerate(masks):
                is_selected = listwidget.item(i).isSelected()
                seg_color = (255, 255, 0) if is_selected else (255, 0, 0)  # yellow / red
                midline_color = (0, 140, 0) if is_selected else (0, 0, 255)  # dark green / blue
                alpha = 0.6 if is_selected else 0.25
                if np.any(m):
                    overlay = np.zeros_like(display)
                    overlay[m.astype(bool)] = seg_color
                    display = cv2.addWeighted(display, 1, overlay, alpha, 0)
                crack_id = items[i][1]
                crack = atomic_cracks.get(crack_id, {})
                midline = np.asarray(crack.get("midline", []), dtype=float)
                if midline.ndim == 2 and midline.shape[0] >= 2 and midline.shape[1] >= 2:
                    pts = np.round(midline[:, :2]).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(display, [pts], False, midline_color, 2, lineType=cv2.LINE_AA)
                elif midline.ndim == 2 and midline.shape[0] == 1 and midline.shape[1] >= 2:
                    p = tuple(np.round(midline[0, :2]).astype(np.int32))
                    cv2.circle(display, p, 2, midline_color, -1, lineType=cv2.LINE_AA)
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

        has_auto = any(is_auto_segment(c) for c in atomic_cracks.values())
        if has_auto:
            rb_gt.setEnabled(False)
            rb_pred.setChecked(True)
            lbl_mode_info.setText("GT unavailable: auto segments present.")
            lbl_mode_info.setVisible(True)
        else:
            rb_gt.setEnabled(True)
            lbl_mode_info.setText("")
            lbl_mode_info.setVisible(False)

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

            print(f"[DEBUG] clear_segmentation START")
            print(f"  Atomic cracks before = {list(atomic_cracks.keys())}")

            # --- Delete selected ---
            for idx in sorted(selected_indices, reverse=True):
                tpe, crack_id = items[idx]
                if tpe == "atomic":
                    print(f"[DEBUG] Deleting atomic crack_id={crack_id}")
                    atomic_cracks.pop(crack_id, None)
                    # Remove from combined_cracks members if present
                    for cid, combo in list(combined_cracks.items()):
                        # Remove deleted atomic cracks from members
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
                            combined_cracks.pop(cid, None)

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

                    # Recompute full combined geometry
                    combined_entry = self._build_combined_crack(
                        members_new,
                        connectivity_mode=connectivity_mode,
                    )
                    if combined_entry is None:
                        to_delete.append(cid)
                        continue

                    combined_cracks[cid] = combined_entry

                for cid in to_delete:
                    combined_cracks.pop(cid, None)

            self.annotation["annotations"]["atomic_cracks"] = atomic_cracks
            self.annotation["annotations"]["combined_cracks"] = combined_cracks

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
