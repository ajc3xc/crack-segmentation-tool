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

#Functions for manual segmentation of preexisting cracks

class ManualDrawing(CrackUtils):
    '''def _draw_manual_mode_overlay(self, canvas):
        """
        Draws a lite version of the main visualization:
        - midlines for every crack
        - endpoints
        - crack ID labels
        """
        import cv2
        import numpy as np

        H, W = self.original_image.shape[:2]
        shown = canvas.copy()

        atomic = (self.annotation.get("annotations", {}) or {}).get("atomic_cracks", {}) or {}

        for cid, crack in atomic.items():
            scid = str(cid)
            
            # ---- midline ----
            ml = crack.get("midline")
            if ml and len(ml) >= 2:
                pts = np.asarray([[p[0], p[1]] for p in ml if p is not None], np.int32)
                if len(pts) >= 2:
                    cv2.polylines(shown, [pts], False, (255, 255, 255), 2)

            # ---- endpoints ----
            endpoints = crack.get("endpoints") or crack.get("pts") or []
            for p in endpoints:
                if p is None:
                    continue
                x, y = int(p[0]), int(p[1])
                cv2.circle(shown, (x, y), 4, (0, 255, 255), -1)

            # ---- crack ID ----
            if ml and len(ml) >= 1:
                x0, y0 = int(ml[0][0]), int(ml[0][1])
                cv2.putText(shown, f"id={cid}", (x0+5, y0-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 200, 255), 1, cv2.LINE_AA)

        return shown
    
    def save_manual_segment(self):
        """
        Save or erase the drawn manual polyline.

        ADD:
        - Requires overlap with an existing atomic (so we already have a midline).
        - Union mask; rebuild geodesic_edges from the unioned mask boundary; recompute normals.

        ERASE:
        - Subtract from mask; if anything remains, rebuild geodesic_edges from the new mask boundary; recompute normals.
        - If nothing remains, delete crack (and remove it from any combined members).

        Always calls self.change_image() to refresh the main UI.
        """
        # ==================  DEBUG INSTRUMENTATION  =====================
        print("\n" + "="*80)
        print("[MANUAL_SAVE] ENTER save_manual_segment")
        print(f"[MANUAL_SAVE] pending_mode = {getattr(self, 'pending_mode', None)}")

        try:
            ann_dbg = self.annotation.get("annotations", {})
            atomic_dbg = ann_dbg.get("atomic_cracks", {})
            combined_dbg = ann_dbg.get("combined_cracks", {})

            print("[MANUAL_SAVE] AT ENTRY → atomic keys:", list(atomic_dbg.keys()))
            print("[MANUAL_SAVE] AT ENTRY → combined keys:", list(combined_dbg.keys()))
            for k, v in atomic_dbg.items():
                print(
                    f"[MANUAL_SAVE]   cid={k} → has mask_crop={('mask_crop' in v)}, "
                    f"mask_bbox={v.get('mask_bbox')}"
                )
        except Exception as e:
            print("[MANUAL_SAVE] ERROR printing initial debug:", e)
        # ================================================================

        try:
            # sanity: we must have a drawn loop
            if not hasattr(self, "manuall_x") or not hasattr(self, "manuall_y"):
                error("No manual polyline to save/erase.")
                return
            if len(self.manuall_x) < 2 or len(self.manuall_y) < 2:
                error("Manual polyline too short.")
                return

            poly = np.column_stack([self.manuall_x, self.manuall_y]).astype(float)

            ann = self.annotation.setdefault("annotations", {})
            atomic = ann.setdefault("atomic_cracks", {})
            combined = ann.setdefault("combined_cracks", {})

            H, W = self.original_image.shape[:2]
            mode = getattr(self, "pending_mode", "add")

            # ---------- helpers ----------
            def rebuild_edges_from_mask(full_mask, crack):
                cnts, _ = cv2.findContours(
                    (full_mask > 0).astype(np.uint8),
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                if not cnts:
                    return None, None, None

                cnt = max(cnts, key=cv2.contourArea)
                ring = cnt[:, 0, :].astype(float)
                if ring.shape[0] < 4:
                    return None, None, None
                if not (ring[0] == ring[-1]).all():
                    ring = np.vstack([ring, ring[0]])

                m = np.asarray(crack.get("midline", []), float)
                if m.ndim != 2 or m.shape[0] < 2 or m.shape[1] != 2:
                    return None, None, None

                mid_x, mid_y = m[:, 0], m[:, 1]
                start, end = m[0], m[-1]

                d0 = np.sum((ring - start) ** 2, axis=1)
                d1 = np.sum((ring - end) ** 2, axis=1)
                i0, i1 = int(np.argmin(d0)), int(np.argmin(d1))

                if i0 <= i1:
                    path1 = ring[i0:i1 + 1]
                    path2 = np.vstack([ring[i1:], ring[:i0 + 1]])
                else:
                    path1 = np.vstack([ring[i0:], ring[:i1 + 1]])
                    path2 = ring[i1:i0 + 1]

                e1, e2 = np.array(path1, float), np.array(path2, float)
                e1x, e1y, e2x, e2y = ct.segmentation.find_normal_pair(
                    mid_x, mid_y, e1, e2
                )
                normals = {
                    "edge1": [e1x.tolist(), e1y.tolist()],
                    "edge2": [e2x.tolist(), e2y.tolist()],
                }
                return e1, e2, normals

            def save_debug_plot(crack_id, crack_type, crack, e1, e2, normals):
                import matplotlib.pyplot as plt

                m = np.asarray(crack.get("midline", []), float)
                if m.ndim != 2 or m.shape[0] < 2:
                    print("[MANUAL_SAVE] EARLY RETURN in save_debug_plot — invalid midline")
                    return

                mid_x, mid_y = m[:, 0], m[:, 1]
                e1x, e1y = np.array(normals["edge1"][0]), np.array(normals["edge1"][1])
                e2x, e2y = np.array(normals["edge2"][0]), np.array(normals["edge2"][1])

                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(self.original_image)
                ax.plot(mid_x, mid_y, 'g-', lw=1, label="midline")
                ax.plot(e1[:, 0], e1[:, 1], 'r-', lw=1, label="edge1")
                ax.plot(e2[:, 0], e2[:, 1], 'b-', lw=1, label="edge2")

                step = max(1, len(mid_x) // 40)
                for i in range(0, len(mid_x), step):
                    if np.isfinite(e1x[i]) and np.isfinite(e2x[i]):
                        ax.plot([mid_x[i], e1x[i]], [mid_y[i], e1y[i]], 'c-', lw=0.8)
                        ax.plot([mid_x[i], e2x[i]], [mid_y[i], e2y[i]], 'm-', lw=0.8)

                ax.set_title(f"{crack_type.capitalize()} crack {crack_id}")
                ax.legend()

                all_x = np.concatenate([mid_x, e1[:, 0], e2[:, 0]])
                all_y = np.concatenate([mid_y, e1[:, 1], e2[:, 1]])
                margin = 20
                ax.set_xlim(max(0, all_x.min() - margin), min(W, all_x.max() + margin))
                ax.set_ylim(min(H, all_y.max() + margin), max(0, all_y.min() - margin))

                save_dir = os.path.join(self.save_folder, "debug_outputs")
                os.makedirs(save_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(self.name))[0]
                fname = os.path.join(
                    save_dir,
                    f"{base_name}_{crack_type}_{crack_id}_manual.png"
                )
                plt.savefig(fname, dpi=250)
                plt.close(fig)
                print(f"[DEBUG] Saved debug plot → {fname}")

            # quick mask state dump
            print("[DBG ADD] atomic masks:")
            for cid, crack in atomic.items():
                mc = crack.get("mask_crop")
                bb = crack.get("mask_bbox")
                if mc is None or bb is None:
                    print(f"  cid={cid}: NO mask_crop/bbox")
                else:
                    arr = np.array(mc, dtype=np.uint8)
                    print(
                        f"  cid={cid}: mask_crop nonzero={np.count_nonzero(arr)} "
                        f"bbox={bb}"
                    )

            # ====================== ERASE MODE ==========================
            if mode == "erase":
                erase_mask = np.zeros((H, W), np.uint8)
                poly_pts = poly.astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(erase_mask, [poly_pts], 255)

                to_delete, changed = [], []

                # 1) apply erase to each atomic
                for cid, crack in list(atomic.items()):
                    mc, bb = crack.get("mask_crop"), crack.get("mask_bbox")
                    if mc is None or bb is None or not len(mc):
                        full_old = np.zeros((H, W), np.uint8)
                    else:
                        crop = np.array(mc, dtype=np.uint8)
                        x0, y0, w, h = [int(v) for v in bb]
                        x1, y1 = min(x0 + w, W), min(y0 + h, H)
                        full_old = np.zeros((H, W), np.uint8)
                        full_old[y0:y1, x0:x1] = crop[:y1 - y0, :x1 - x0]

                    full_new = cv2.bitwise_and(
                        full_old,
                        cv2.bitwise_not(erase_mask)
                    )

                    if np.any(full_new):
                        ys, xs = np.where(full_new > 0)
                        x0, x1 = xs.min(), xs.max() + 1
                        y0, y1 = ys.min(), ys.max() + 1
                        crop = full_new[y0:y1, x0:x1]
                        crack["mask_crop"] = crop.tolist()
                        crack["mask_bbox"] = [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]

                        e1, e2, normals = rebuild_edges_from_mask(full_new, crack)
                        if e1 is not None:
                            crack["geodesic_edges"] = {
                                "edge1": e1.tolist(),
                                "edge2": e2.tolist(),
                            }
                            crack["normal_edge_points"] = normals
                            save_debug_plot(cid, "atomic", crack, e1, e2, normals)
                            changed.append(cid)
                    else:
                        to_delete.append(cid)

                # 2) delete fully erased atomics and purge from combined
                for cid in to_delete:
                    if cid in atomic:
                        del atomic[cid]
                        print(f"[FIX] Deleted atomic id={cid} (fully erased)")

                    for cmb_id, cmb in list(combined.items()):
                        members = cmb.get("members", [])
                        if cid in members:
                            members = [m for m in members if m != cid]
                            cmb["members"] = members
                            print(
                                f"[FIX] Purged deleted atomic id={cid} from combined {cmb_id}"
                            )
                            # if combined has no members left, drop it
                            if not members:
                                del combined[cmb_id]
                                print(
                                    f"[FIX] Removed empty combined crack {cmb_id} "
                                    f"(no members left)"
                                )

                # 3) if exactly one atomic changed, rebuild the combined it belongs to
                if len(changed) == 1:
                    changed_id = changed[0]
                    for cmb_id, cmb in list(combined.items()):
                        members = cmb.get("members", [])

                        # prune any stale members that no longer exist
                        valid_members = [m for m in members if m in atomic]
                        if len(valid_members) != len(members):
                            print(
                                f"[FIX] Pruned stale members from combined {cmb_id}: "
                                f"before={members}, after={valid_members}"
                            )
                            cmb["members"] = valid_members
                            members = valid_members

                        if not members:
                            # nothing left to combine
                            del combined[cmb_id]
                            print(
                                f"[FIX] Removed combined {cmb_id} during rebuild "
                                f"(no valid members after prune)"
                            )
                            continue

                        if changed_id not in members:
                            continue

                        # rebuild combined geometry
                        combined[cmb_id] = self._build_combined_crack(members)

                        # rebuild combined mask robustly
                        full_mask = np.zeros((H, W), np.uint8)
                        for member_id in members:
                            crack_m = atomic.get(member_id)
                            if not crack_m:
                                continue
                            mc = crack_m.get("mask_crop")
                            bb = crack_m.get("mask_bbox")
                            if mc is None or bb is None or not len(mc):
                                continue
                            crop = np.array(mc, dtype=np.uint8)
                            x0, y0, w, h = [int(v) for v in bb]
                            x1, y1 = min(x0 + w, W), min(y0 + h, H)
                            full_mask[y0:y1, x0:x1] |= crop[:y1 - y0, :x1 - x0]

                        if changed_id in atomic:
                            e1, e2, normals = rebuild_edges_from_mask(
                                full_mask,
                                atomic[changed_id]
                            )
                            if e1 is not None:
                                save_debug_plot(
                                    cmb_id, "combined", atomic[changed_id],
                                    e1, e2, normals
                                )
                        break  # only one combined expected

                self.save_annotation()

            # ====================== ADD MODE ===========================
            else:
                target_id, target_crack = None, None
                poly_mask = np.zeros((H, W), np.uint8)
                cv2.fillPoly(
                    poly_mask,
                    [poly.astype(np.int32).reshape((-1, 1, 2))],
                    255
                )

                # 1) find first atomic that overlaps the drawn region
                for cid, crack in atomic.items():
                    mc, bb = crack.get("mask_crop"), crack.get("mask_bbox")
                    if mc is None or bb is None or not len(mc):
                        continue
                    crop = np.array(mc, dtype=np.uint8)
                    x0, y0, w, h = [int(v) for v in bb]
                    x1, y1 = min(x0 + w, W), min(y0 + h, H)
                    full_old = np.zeros((H, W), np.uint8)
                    full_old[y0:y1, x0:x1] = crop[:y1 - y0, :x1 - x0]

                    if np.any(cv2.bitwise_and(full_old, poly_mask)):
                        target_id, target_crack = cid, crack
                        break

                if target_crack is None:
                    print("[MANUAL_SAVE] EARLY RETURN — no overlapping atomic found")
                    print(
                        "[MANUAL_SAVE] atomic keys at abort:",
                        list(self.annotation.get("annotations", {})
                            .get("atomic_cracks", {}).keys())
                    )
                    return

                # 2) union new region into that atomic's mask
                mc, bb = target_crack.get("mask_crop"), target_crack.get("mask_bbox")
                if mc is None or bb is None or not len(mc):
                    full_old = np.zeros((H, W), np.uint8)
                else:
                    crop = np.array(mc, dtype=np.uint8)
                    x0, y0, w, h = [int(v) for v in bb]
                    x1, y1 = min(x0 + w, W), min(y0 + h, H)
                    full_old = np.zeros((H, W), np.uint8)
                    full_old[y0:y1, x0:x1] = crop[:y1 - y0, :x1 - x0]

                full_new = cv2.bitwise_or(full_old, poly_mask)
                ys, xs = np.where(full_new > 0)
                if len(xs) and len(ys):
                    x0, x1 = xs.min(), xs.max() + 1
                    y0, y1 = ys.min(), ys.max() + 1
                    crop = full_new[y0:y1, x0:x1]
                    target_crack["mask_crop"] = crop.tolist()
                    target_crack["mask_bbox"] = [
                        int(x0), int(y0), int(x1 - x0), int(y1 - y0)
                    ]

                e1, e2, normals = rebuild_edges_from_mask(full_new, target_crack)
                if e1 is not None:
                    target_crack["geodesic_edges"] = {
                        "edge1": e1.tolist(),
                        "edge2": e2.tolist(),
                    }
                    target_crack["normal_edge_points"] = normals
                    save_debug_plot(target_id, "atomic", target_crack, e1, e2, normals)

                # 3) update the single combined (if any) that contains this atomic
                for cmb_id, cmb in list(combined.items()):
                    members = cmb.get("members", [])

                    # prune stale members just in case
                    valid_members = [m for m in members if m in atomic]
                    if len(valid_members) != len(members):
                        print(
                            f"[FIX] Pruned stale members from combined {cmb_id}: "
                            f"before={members}, after={valid_members}"
                        )
                        cmb["members"] = valid_members
                        members = valid_members

                    if not members:
                        del combined[cmb_id]
                        print(
                            f"[FIX] Removed combined {cmb_id} during add "
                            f"(no valid members left)"
                        )
                        continue

                    if target_id not in members:
                        continue

                    combined[cmb_id] = self._build_combined_crack(members)

                    full_mask = np.zeros((H, W), np.uint8)
                    for member_id in members:
                        crack_m = atomic.get(member_id)
                        if not crack_m:
                            continue
                        mc = crack_m.get("mask_crop")
                        bb = crack_m.get("mask_bbox")
                        if mc is None or bb is None or not len(mc):
                            continue
                        crop = np.array(mc, dtype=np.uint8)
                        x0, y0, w, h = [int(v) for v in bb]
                        x1, y1 = min(x0 + w, W), min(y0 + h, H)
                        full_mask[y0:y1, x0:x1] |= crop[:y1 - y0, :x1 - x0]

                    e1, e2, normals = rebuild_edges_from_mask(full_mask, target_crack)
                    if e1 is not None:
                        save_debug_plot(
                            cmb_id, "combined", target_crack, e1, e2, normals
                        )
                    break  # only one combined expected

                self.save_annotation()

            # ==================== REFRESH UI ============================
            im = self.image.astype(np.uint8).copy()
            im = self.draw_existing_cracks(im)
            qimage = QImage(
                im, im.shape[1], im.shape[0],
                im.strides[0], QImage.Format_RGB888
            )
            pixmap = QPixmap.fromImage(qimage)
            scaled = pixmap.scaled(
                self.manual_segment_screen.width(),
                self.manual_segment_screen.height(),
                Qt.KeepAspectRatio,
                Qt.FastTransformation
            )
            self.manual_segment_screen.setPixmap(scaled)

            self.change_image()

            if hasattr(self, "manuall_x"):
                del self.manuall_x
            if hasattr(self, "manuall_y"):
                del self.manuall_y

        except Exception as e:
            import traceback
            traceback.print_exc()
            error(e)

    def clear_pending_segment(self):
            """
            Clear any unsaved manual segment (used when overwriting).
            """
            if hasattr(self, "manuall_x"):
                del self.manuall_x
            if hasattr(self, "manuall_y"):
                del self.manuall_y
            if hasattr(self, "pending_mode"):
                del self.pending_mode

            # Clear the preview in the Qt widget too
            if hasattr(self, "manual_segment_screen"):
                from PyQt5.QtGui import QPixmap
                self.manual_segment_screen.setPixmap(QPixmap())

    def draw_segment(self, mode):
        from shapely.geometry import Polygon, MultiPolygon
        from shapely.ops import unary_union
        import matplotlib.pyplot as plt
        print(mode)
        try:
            # --- Handle unsaved previous strokes ---
            if hasattr(self, "manuall_x") and len(getattr(self, "manuall_x", [])) > 0:
                from PyQt5.QtWidgets import QMessageBox
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("You already have an unsaved manual segment.\n"
                            "Continuing will overwrite it. Proceed?")
                msg.setWindowTitle("Overwrite Segment")
                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                ret = msg.exec_()
                if ret == QMessageBox.No:
                    return
                else:
                    self.clear_pending_segment()

            image_size = self.select_image_size.value()
            x, y = ct.tools.Draw().contours(
                self.image[:, :, ::-1],
                image_size,
                annotations=self.annotation.get("annotations", {}),
                mode=mode
            )
            if len(x) < 3:
                return  # not enough points

            coords = np.column_stack([x, y]).astype(np.int32)

            tol = 15       # closure distance in px
            min_gap = 10   # min separation in indices
            polys = []

            # --- Find ALL closures ---
            for i in range(len(coords) - min_gap):
                for j in range(i + min_gap, len(coords)):
                    d = np.linalg.norm(coords[i] - coords[j])
                    if d < tol:
                        loop_coords = coords[i:j+1]
                        if len(loop_coords) >= 3:
                            poly = Polygon(loop_coords)
                            if not poly.is_valid:
                                poly = poly.buffer(0)
                            if not poly.is_empty and poly.area > 30:
                                polys.append(poly)

            print(f"[DEBUG] Loops found: {len(polys)}")

            if not polys:
                print("[INFO] No closed loop detected in stroke.")
                return

            # --- Merge overlapping polygons ---
            merged = unary_union(polys)

            # always get a list of polygons back
            if isinstance(merged, Polygon):
                merged = [merged]
            elif isinstance(merged, MultiPolygon):
                merged = list(merged.geoms)

            print(f"[DEBUG] Independent loops after merge: {len(merged)}")

            # --- Convert back to OpenCV contours ---
            valid_loops = []
            for poly in merged:
                coords = np.array(poly.exterior.coords, dtype=np.int32).reshape((-1, 1, 2))
                valid_loops.append(coords)

            # save first loop coords for later use
            biggest = max(valid_loops, key=cv2.contourArea)
            self.manuall_x = biggest[:, 0, 0]
            self.manuall_y = biggest[:, 0, 1]
            self.pending_mode = mode
            
            # --- Preview ---
            H, W = self.image.shape[:2]
            im = self.image.astype(np.uint8).copy()
            
            im = self.draw_existing_cracks(im)

            preview_mask = np.zeros((H, W), np.uint8)
            cv2.fillPoly(preview_mask, valid_loops, 255)

            fill_color = (0, 255, 0) if mode == 'add' else (255, 50, 0)
            overlay = np.zeros_like(im)
            overlay[preview_mask > 0] = fill_color
            im = cv2.addWeighted(im, 1, overlay, 0.7, 0)

            # --- Show in Qt ---
            qimage = QImage(im, im.shape[1], im.shape[0],
                            im.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            scaled = pixmap.scaled(
                self.manual_segment_screen.width(),
                self.manual_segment_screen.height(),
                Qt.KeepAspectRatio,
                Qt.FastTransformation
            )
            self.manual_segment_screen.setPixmap(scaled)

        except Exception as e:
            import traceback; traceback.print_exc()
            error(e)'''

    def erase_segment(self):
        """Convenience wrapper for erase mode."""
        return self.draw_segment('erase')
    
    '''def reset_canvas(self):
        """Clear pending manual segments and reset the preview canvas."""
        try:
            self.manuall_x = []
            self.manuall_y = []
            self.pending_mode = None

            im = self.image.astype(np.uint8).copy()
            im = self.draw_existing_cracks(im)  # <--- cracks back in

            qimage = QImage(im, im.shape[1], im.shape[0],
                            im.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            scaled = pixmap.scaled(
                self.manual_segment_screen.width(),
                self.manual_segment_screen.height(),
                Qt.KeepAspectRatio,
                Qt.FastTransformation
            )
            self.manual_segment_screen.setPixmap(scaled)

            print("[INFO] Canvas reset complete.")
        except Exception as e:
            import traceback; traceback.print_exc()
            error(e)'''
    
    def clear_pending_segment(self):
        """
        Clear any unsaved manual segment (used when overwriting).
        """
        if hasattr(self, "manuall_x"):
            del self.manuall_x
        if hasattr(self, "manuall_y"):
            del self.manuall_y
        if hasattr(self, "pending_mode"):
            del self.pending_mode

        # Clear the preview in the Qt widget too
        if hasattr(self, "manual_segment_screen"):
            from PyQt5.QtGui import QPixmap
            self.manual_segment_screen.setPixmap(QPixmap())
            
    def draw_segment(self, mode):
        from shapely.geometry import Polygon, MultiPolygon
        from shapely.ops import unary_union
        import matplotlib.pyplot as plt
        import numpy as np
        import cv2

        print(f"[DRAW_SEGMENT] mode={mode}")
        try:
            # --- Handle unsaved previous strokes ---
            if hasattr(self, "manuall_x") and len(getattr(self, "manuall_x", [])) > 0:
                from PyQt5.QtWidgets import QMessageBox
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("You already have an unsaved manual segment.\n"
                            "Continuing will overwrite it. Proceed?")
                msg.setWindowTitle("Overwrite Segment")
                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                ret = msg.exec_()
                if ret == QMessageBox.No:
                    return
                else:
                    self.clear_pending_segment()

            image_size = self.select_image_size.value()
            x, y = ct.tools.Draw().contours(
                self.image[:, :, ::-1],
                image_size,
                annotations=self.annotation.get("annotations", {}),
                mode=mode
            )
            if len(x) < 3:
                print("[DRAW_SEGMENT] too few points; abort.")
                return

            coords = np.column_stack([x, y]).astype(np.int32)

            tol = 15       # closure distance in px
            min_gap = 10   # min separation in indices
            polys = []

            # --- Find ALL closures along stroke ---
            for i in range(len(coords) - min_gap):
                for j in range(i + min_gap, len(coords)):
                    d = np.linalg.norm(coords[i] - coords[j])
                    if d < tol:
                        loop_coords = coords[i:j+1]
                        if len(loop_coords) >= 3:
                            poly = Polygon(loop_coords)
                            if not poly.is_valid:
                                poly = poly.buffer(0)
                            if not poly.is_empty and poly.area > 30:
                                polys.append(poly)

            print(f"[DRAW_SEGMENT] raw loops found: {len(polys)}")

            if not polys:
                print("[DRAW_SEGMENT] No closed loop detected in stroke.")
                return

            # --- Merge overlapping polygons ---
            merged = unary_union(polys)

            if isinstance(merged, Polygon):
                merged = [merged]
            elif isinstance(merged, MultiPolygon):
                merged = list(merged.geoms)

            print(f"[DRAW_SEGMENT] independent loops after merge: {len(merged)}")

            # --- Convert to OpenCV contours ---
            valid_loops = []
            for poly in merged:
                c = np.array(poly.exterior.coords, dtype=np.int32).reshape((-1, 1, 2))
                if c.shape[0] >= 3:
                    valid_loops.append(c)

            if not valid_loops:
                print("[DRAW_SEGMENT] no valid loops after merge; abort.")
                return

            # --- Store biggest loop as the working poly ---
            biggest = max(valid_loops, key=cv2.contourArea)
            self.manuall_x = biggest[:, 0, 0]
            self.manuall_y = biggest[:, 0, 1]
            self.pending_mode = mode

            # --- Preview ---
            H, W = self.image.shape[:2]
            im = self.image.astype(np.uint8).copy()
            im = self.draw_existing_cracks(im)

            preview_mask = np.zeros((H, W), np.uint8)
            cv2.fillPoly(preview_mask, valid_loops, 255)

            fill_color = (0, 255, 0) if mode == "add" else (255, 50, 0)
            overlay = np.zeros_like(im)
            overlay[preview_mask > 0] = fill_color
            im = cv2.addWeighted(im, 1.0, overlay, 0.7, 0)

            # --- Add midline + endpoint overlay (lite view) ---
            im = self._draw_manual_mode_overlay(im)

            # --- Show in Qt ---
            qimage = QImage(im, im.shape[1], im.shape[0],
                            im.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            scaled = pixmap.scaled(
                self.manual_segment_screen.width(),
                self.manual_segment_screen.height(),
                Qt.KeepAspectRatio,
                Qt.FastTransformation
            )
            self.manual_segment_screen.setPixmap(scaled)

        except Exception as e:
            import traceback
            traceback.print_exc()
            error(e)
            
    def save_manual_segment(self):
        """
        Save or erase the drawn manual polyline.

        ADD:
        - Requires overlap with an existing *final* crack
        (non-member atomic or combined).
        - Union mask; rebuild geodesic_edges from the unioned mask boundary;
        recompute normals.

        ERASE:
        - Subtract from mask; if anything remains, rebuild geodesic_edges from
        the new mask boundary; recompute normals.
        - If nothing remains, KEEP the crack (midline/endpoints intact) but
        clear mask + edges.

        Final cracks = all atomic cracks that are NOT members of any combined
        + all combined cracks themselves.

        Always calls self.change_image() to refresh the main UI.
        """
        import numpy as np
        import cv2

        print("\n" + "=" * 80)
        print("[MANUAL_SAVE] ENTER save_manual_segment")
        print(f"[MANUAL_SAVE] pending_mode = {getattr(self, 'pending_mode', None)}")

        try:
            ann_dbg = self.annotation.get("annotations", {}) or {}
            atomic_dbg = ann_dbg.get("atomic_cracks", {}) or {}
            combined_dbg = ann_dbg.get("combined_cracks", {}) or {}

            print("[MANUAL_SAVE] AT ENTRY → atomic keys:", list(atomic_dbg.keys()))
            print("[MANUAL_SAVE] AT ENTRY → combined keys:", list(combined_dbg.keys()))
        except Exception as e:
            print("[MANUAL_SAVE] ERROR printing initial debug:", e)

        try:
            # sanity: we must have a drawn loop
            if not hasattr(self, "manuall_x") or not hasattr(self, "manuall_y"):
                error("No manual polyline to save/erase.")
                return
            if len(self.manuall_x) < 2 or len(self.manuall_y) < 2:
                error("Manual polyline too short.")
                return

            poly = np.column_stack([self.manuall_x, self.manuall_y]).astype(float)

            ann = self.annotation.setdefault("annotations", {})
            atomic = ann.setdefault("atomic_cracks", {})
            combined = ann.setdefault("combined_cracks", {})

            H, W = self.original_image.shape[:2]
            mode = getattr(self, "pending_mode", "add")

            # ---------- helper: which cracks are "final"? ----------
            def iter_final_cracks():
                """
                Yield (kind, cid, crack_dict) where:
                kind ∈ {'atomic', 'combined'}
                Final = non-member atomics + all combined.
                """
                # collect atomic ids that are in any combined.members
                member_ids = set()
                for cmb_id, cmb in combined.items():
                    for m in cmb.get("members", []) or []:
                        member_ids.add(str(m))

                # non-member atomics
                for cid, crack in atomic.items():
                    if str(cid) not in member_ids:
                        yield "atomic", cid, crack

                # all combined cracks
                for cmb_id, crack in combined.items():
                    yield "combined", cmb_id, crack

            # ---------- helper: full-image mask from crop+bbox ----------
            def reconstruct_full_mask(crack):
                mc, bb = crack.get("mask_crop"), crack.get("mask_bbox")
                if mc is None or bb is None or not len(mc):
                    return np.zeros((H, W), np.uint8)
                crop = np.array(mc, dtype=np.uint8)
                x0, y0, w, h = [int(v) for v in bb]
                x1, y1 = min(x0 + w, W), min(y0 + h, H)
                full = np.zeros((H, W), np.uint8)
                full[y0:y1, x0:x1] = crop[:y1 - y0, :x1 - x0]
                return full

            # ---------- helper: rebuild edges + normals from full mask ----------
            def rebuild_edges_from_mask(full_mask, crack):
                cnts, _ = cv2.findContours(
                    (full_mask > 0).astype(np.uint8),
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                if not cnts:
                    return None, None, None

                cnt = max(cnts, key=cv2.contourArea)
                ring = cnt[:, 0, :].astype(float)
                if ring.shape[0] < 4:
                    return None, None, None
                if not (ring[0] == ring[-1]).all():
                    ring = np.vstack([ring, ring[0]])

                m = np.asarray(crack.get("midline", []), float)
                if m.ndim != 2 or m.shape[0] < 2 or m.shape[1] != 2:
                    return None, None, None

                mid_x, mid_y = m[:, 0], m[:, 1]
                start, end = m[0], m[-1]

                d0 = np.sum((ring - start) ** 2, axis=1)
                d1 = np.sum((ring - end) ** 2, axis=1)
                i0, i1 = int(np.argmin(d0)), int(np.argmin(d1))

                if i0 <= i1:
                    path1 = ring[i0:i1 + 1]
                    path2 = np.vstack([ring[i1:], ring[:i0 + 1]])
                else:
                    path1 = np.vstack([ring[i0:], ring[:i1 + 1]])
                    path2 = ring[i1:i0 + 1]

                e1, e2 = np.array(path1, float), np.array(path2, float)

                # normals along the midline
                e1x, e1y, e2x, e2y = ct.segmentation.find_normal_pair(
                    mid_x, mid_y, e1, e2
                )
                normals = {
                    "edge1": [e1x.tolist(), e1y.tolist()],
                    "edge2": [e2x.tolist(), e2y.tolist()],
                }
                return e1, e2, normals

            # ---------- debug: print mask state for final cracks ----------
            print("[MANUAL_SAVE] final cracks (pre-op):")
            for kind, cid, crack in iter_final_cracks():
                mc = crack.get("mask_crop")
                bb = crack.get("mask_bbox")
                if mc is None or bb is None:
                    print(f"  {kind} id={cid}: NO mask_crop/bbox")
                else:
                    arr = np.array(mc, dtype=np.uint8)
                    print(
                        f"  {kind} id={cid}: mask_crop nonzero={np.count_nonzero(arr)} "
                        f"bbox={bb}"
                    )

            # ====================== ERASE MODE ==========================
            if mode == "erase":
                erase_mask = np.zeros((H, W), np.uint8)
                poly_pts = poly.astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(erase_mask, [poly_pts], 255)

                for kind, cid, crack in iter_final_cracks():
                    full_old = reconstruct_full_mask(crack)

                    full_new = cv2.bitwise_and(
                        full_old,
                        cv2.bitwise_not(erase_mask)
                    )

                    if np.any(full_new):
                        # still some mask left → update crop + edges
                        ys, xs = np.where(full_new > 0)
                        x0, x1 = xs.min(), xs.max() + 1
                        y0, y1 = ys.min(), ys.max() + 1
                        crop = full_new[y0:y1, x0:x1]

                        crack["mask_crop"] = crop.tolist()
                        crack["mask_bbox"] = [int(x0), int(y0),
                                            int(x1 - x0), int(y1 - y0)]

                        e1, e2, normals = rebuild_edges_from_mask(full_new, crack)
                        if e1 is not None:
                            crack["geodesic_edges"] = {
                                "edge1": e1.tolist(),
                                "edge2": e2.tolist(),
                            }
                            crack["normal_edge_points"] = normals
                    else:
                        # fully erased: keep midline/user_points, clear mask + edges
                        for key in ("mask_crop", "geodesic_edges", "normal_edge_points"):
                            crack.pop(key, None)
                        print(f"[MANUAL_SAVE] Cleared mask/edges for {kind} id={cid}")

                self.save_annotation()

            # ====================== ADD MODE ===========================
            else:
                target_kind, target_id, target_crack = None, None, None
                poly_mask = np.zeros((H, W), np.uint8)
                cv2.fillPoly(
                    poly_mask,
                    [poly.astype(np.int32).reshape((-1, 1, 2))],
                    255
                )

                # 1) find first final crack that overlaps the drawn region
                for kind, cid, crack in iter_final_cracks():
                    full_old = reconstruct_full_mask(crack)
                    if np.any(cv2.bitwise_and(full_old, poly_mask)):
                        target_kind, target_id, target_crack = kind, cid, crack
                        break

                if target_crack is None:
                    print("[MANUAL_SAVE] EARLY RETURN — no overlapping final crack found")
                    return

                # 2) union new region into that crack's mask
                full_old = reconstruct_full_mask(target_crack)
                full_new = cv2.bitwise_or(full_old, poly_mask)

                ys, xs = np.where(full_new > 0)
                if len(xs) and len(ys):
                    x0, x1 = xs.min(), xs.max() + 1
                    y0, y1 = ys.min(), ys.max() + 1
                    crop = full_new[y0:y1, x0:x1]
                    target_crack["mask_crop"] = crop.tolist()
                    target_crack["mask_bbox"] = [
                        int(x0), int(y0), int(x1 - x0), int(y1 - y0)
                    ]

                e1, e2, normals = rebuild_edges_from_mask(full_new, target_crack)
                if e1 is not None:
                    target_crack["geodesic_edges"] = {
                        "edge1": e1.tolist(),
                        "edge2": e2.tolist(),
                    }
                    target_crack["normal_edge_points"] = normals

                self.save_annotation()

            # ==================== REFRESH UI ============================
            im = self.image.astype(np.uint8).copy()
            im = self.draw_existing_cracks(im)
            im = self._draw_manual_mode_overlay(im)

            qimage = QImage(
                im, im.shape[1], im.shape[0],
                im.strides[0], QImage.Format_RGB888
            )
            pixmap = QPixmap.fromImage(qimage)
            scaled = pixmap.scaled(
                self.manual_segment_screen.width(),
                self.manual_segment_screen.height(),
                Qt.KeepAspectRatio,
                Qt.FastTransformation
            )
            self.manual_segment_screen.setPixmap(scaled)

            self.change_image()

            if hasattr(self, "manuall_x"):
                del self.manuall_x
            if hasattr(self, "manuall_y"):
                del self.manuall_y
            if hasattr(self, "pending_mode"):
                del self.pending_mode

        except Exception as e:
            import traceback
            traceback.print_exc()
            error(e)
                 
    def commit_manual_midlines_full(self):
        """
        Update existing manual_poly cracks with a valid mask_bbox only.
        Do NOT store mask_crop here (edge rasterization is produced later by the worker).
        """
        import numpy as np, json

        ann = self.annotation.setdefault("annotations", {})
        atomic = ann.setdefault("atomic_cracks", {})

        def _norm_pair(pA, pB, r=6):
            return (round(float(pA[0]), r), round(float(pA[1]), r),
                    round(float(pB[0]), r), round(float(pB[1]), r))

        # reverse lookup: user_points -> crack id
        pair_to_id = {}
        for cid, cr in atomic.items():
            up = cr.get("user_points") or []
            if len(up) == 2:
                pair_to_id[_norm_pair(up[0], up[1])] = cid
                pair_to_id[_norm_pair(up[1], up[0])] = cid
        print(f"[DEBUG FULL] built pair_to_id for {len(pair_to_id)//2} cracks")

        points = getattr(self, "all_selected_points", None) or getattr(self, "user_points", []) or []
        mm = dict(getattr(self, "manual_midlines_tmp", {}) or {})
        print(f"[DEBUG FULL] processing {len(mm)} manual midlines")

        updated = 0
        for k, _poly in mm.items():
            if isinstance(k, tuple) and len(k) == 2:
                i1, i2 = map(int, k)
            elif isinstance(k, str) and "_" in k:
                try: i1, i2 = map(int, k.split("_"))
                except ValueError: continue
            else:
                continue

            if not (0 <= i1 < len(points) and 0 <= i2 < len(points)) or i1 == i2:
                continue
            p1, p2 = tuple(points[i1]), tuple(points[i2])
            norm_key = _norm_pair(p1, p2)

            if norm_key not in pair_to_id:
                print(f"[ERROR FULL] ❌ No existing crack found for pair {p1}->{p2} (key={k})")
                continue

            cid = pair_to_id[norm_key]
            crack = atomic[cid]

            # choose bbox that contains both endpoints (use your existing get_all_bounding_boxes)
            bbox = None
            for xmin, ymin, xmax, ymax in self.get_all_bounding_boxes():
                if (xmin <= p1[0] <= xmax and ymin <= p1[1] <= ymax and
                    xmin <= p2[0] <= xmax and ymin <= p2[1] <= ymax):
                    bbox = [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)]
                    break
            if bbox is None:
                print(f"[WARN FULL] no bbox found for {p1}->{p2}; skipping")
                continue

            crack["mask_bbox"] = bbox
            crack.pop("mask_crop", None)  # ensure we don't leave stale crops
            updated += 1
            print(f"[MEM FULL] ✅ id={cid} mask_bbox={bbox} (no mask_crop)")

        print(f"[SUMMARY FULL] updated {updated}/{len(mm)} manual midlines")
 
    # in select_save_end_points
    def select_save_end_points(self):
        """
        Select endpoints + manual midlines and immediately save annotation.
        By default, runs in full pipeline mode (creates usable mask_crop entries).
        Set pipeline=False for lightweight endpoint-only testing.
        """
        self.select_end_points_manmidlines(metrics=True)
        self.commit_manual_midlines_full()
        self.save_annotation()
        self.change_image()

        
    def manual_segment_full_screen(self):
        try:
            pts = np.array([self.manuall_x,self.manuall_y]).transpose(1,0).reshape((-1,1,2)).astype(np.int32)
            im = self.image.astype(np.uint8).copy()
            plt.imshow(im)
            plt.plot(self.manuall_x,self.manuall_y,'r',linewidth = 1)
            plt.show()
        except Exception as e:
            error(e)