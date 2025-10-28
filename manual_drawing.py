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

class ManualDrawing(CrackUtils):
    def save_manual_segment(self):
        """
        Save or erase the drawn manual polyline.

        ADD:
        - Requires overlap with an existing atomic (so we already have a midline).
        - Union mask; rebuild geodesic_edges from the unioned mask boundary; recompute normals.

        ERASE:
        - Subtract from mask; if anything remains, rebuild geodesic_edges from the new mask boundary; recompute normals.
        - If nothing remains, delete crack.

        Always calls self.change_image() to refresh the main UI.
        """
        try:
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

            def rebuild_edges_from_mask(full_mask, crack):
                cnts, _ = cv2.findContours((full_mask > 0).astype(np.uint8),
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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
                e1x, e1y, e2x, e2y = ct.segmentation.find_normal_pair(mid_x, mid_y, e1, e2)
                normals = {
                    "edge1": [e1x.tolist(), e1y.tolist()],
                    "edge2": [e2x.tolist(), e2y.tolist()]
                }
                return e1, e2, normals

            def save_debug_plot(crack_id, crack_type, crack, e1, e2, normals):
                import matplotlib.pyplot as plt

                m = np.asarray(crack.get("midline", []), float)
                if m.ndim != 2 or m.shape[0] < 2:
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
                ts = int(time() * 1000)
                fname = os.path.join(save_dir, f"{base_name}_{crack_type}_{crack_id}_manual.png")
                plt.savefig(fname, dpi=250)
                plt.close(fig)
                print(f"[DEBUG] Saved debug plot → {fname}")

            # ERASE MODE
            if mode == "erase":
                erase_mask = np.zeros((H, W), np.uint8)
                poly_pts = poly.astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(erase_mask, [poly_pts], 255)

                to_delete, changed = [], []
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

                    full_new = cv2.bitwise_and(full_old, cv2.bitwise_not(erase_mask))

                    if np.any(full_new):
                        ys, xs = np.where(full_new > 0)
                        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
                        crop = full_new[y0:y1, x0:x1]
                        crack["mask_crop"] = crop.tolist()
                        crack["mask_bbox"] = [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]

                        e1, e2, normals = rebuild_edges_from_mask(full_new, crack)
                        if e1 is not None:
                            crack["geodesic_edges"] = {"edge1": e1.tolist(), "edge2": e2.tolist()}
                            crack["normal_edge_points"] = normals
                            save_debug_plot(cid, "atomic", crack, e1, e2, normals)
                            changed.append(cid)
                    else:
                        to_delete.append(cid)

                for cid in to_delete:
                    del atomic[cid]

                # if exactly one atomic was changed, check its combined membership
                if len(changed) == 1:
                    changed_id = changed[0]
                    for cmb_id, cmb in combined.items():
                        if changed_id in cmb.get("members", []):
                            combined[cmb_id] = self._build_combined_crack(cmb["members"])
                            # build mask for that combined
                            full_mask = np.zeros((H, W), np.uint8)
                            for member_id in cmb["members"]:
                                mc, bb = atomic[member_id]["mask_crop"], atomic[member_id]["mask_bbox"]
                                crop = np.array(mc, dtype=np.uint8)
                                x0, y0, w, h = [int(v) for v in bb]
                                x1, y1 = min(x0 + w, W), min(y0 + h, H)
                                full_mask[y0:y1, x0:x1] |= crop[:y1-y0, :x1-x0]
                            e1, e2, normals = rebuild_edges_from_mask(full_mask, atomic[changed_id])
                            if e1 is not None:
                                save_debug_plot(cmb_id, "combined", atomic[changed_id], e1, e2, normals)
                            break

                self.save_annotation()

            # ADD MODE
            else:
                target_id, target_crack = None, None
                poly_mask = np.zeros((H, W), np.uint8)
                cv2.fillPoly(poly_mask, [poly.astype(np.int32).reshape((-1, 1, 2))], 255)

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
                    return

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
                    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
                    crop = full_new[y0:y1, x0:x1]
                    target_crack["mask_crop"] = crop.tolist()
                    target_crack["mask_bbox"] = [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]

                e1, e2, normals = rebuild_edges_from_mask(full_new, target_crack)
                if e1 is not None:
                    target_crack["geodesic_edges"] = {"edge1": e1.tolist(), "edge2": e2.tolist()}
                    target_crack["normal_edge_points"] = normals
                    save_debug_plot(target_id, "atomic", target_crack, e1, e2, normals)

                # only one combined if any
                for cmb_id, cmb in combined.items():
                    if target_id in cmb.get("members", []):
                        combined[cmb_id] = self._build_combined_crack(cmb["members"])
                        full_mask = np.zeros((H, W), np.uint8)
                        for member_id in cmb["members"]:
                            mc, bb = atomic[member_id]["mask_crop"], atomic[member_id]["mask_bbox"]
                            crop = np.array(mc, dtype=np.uint8)
                            x0, y0, w, h = [int(v) for v in bb]
                            x1, y1 = min(x0 + w, W), min(y0 + h, H)
                            full_mask[y0:y1, x0:x1] |= crop[:y1-y0, :x1-x0]
                        e1, e2, normals = rebuild_edges_from_mask(full_mask, target_crack)
                        if e1 is not None:
                            save_debug_plot(cmb_id, "combined", target_crack, e1, e2, normals)
                        break

                self.save_annotation()

            # refresh manual preview screen
            im = self.image.astype(np.uint8).copy()
            im = self.draw_existing_cracks(im)
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

            self.change_image()

            if hasattr(self, "manuall_x"): del self.manuall_x
            if hasattr(self, "manuall_y"): del self.manuall_y

        except Exception as e:
            import traceback; traceback.print_exc()
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
            x, y = ct.tools.Draw().counturs(
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
            error(e)

    def erase_segment(self):
        """Convenience wrapper for erase mode."""
        return self.draw_segment('erase')
    
    def reset_canvas(self):
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
            error(e)
                 
    # in select_save_end_points
    def select_save_end_points(self):
        self.select_end_points_manmidlines()
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