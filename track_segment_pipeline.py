#!/usr/bin/env python3
from shapely import geometry, ops
import numpy as np, cv2, os, matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from crackutils import *
import cracktools as ct
from cracktools.os import set_os_mode, OS_MODE
from helpers.layout import Ui_MainWindow
from time import time
from helpers.crackhelpers import *
from helpers import metrics, save_load_files

#This class is for the main tracking / segmentation pipeline for generating the automatic midline and edge tracking submodules
#This is effectively the 'core' of the entire program's outer function wrappers

class TrackSegmentPipeline(CrackUtils, Ui_MainWindow):
    def update_os(self):
        try:
            self.os_progress_bar.setValue(0)
            color_channel = [0 if self.color_chenel_box.currentText()=='R' else 1 if self.color_chenel_box.currentText()=='B' else 2][0]
            black_crack = -1 if self.crack_color_box.currentText() =='Bright crack' else 1
            size = self.wavelet_size_box.value()
            nOrientations = self.wavelet_norientations_box.value()
            design = "N"
            inflectionPoint = self.wavelet_inflection_point_box.value()
            mnOrder = self.wavelet_mnorder_box.value()
            splineOrder = 3
            overlapFactor = self.wavelet_overlap_factor_box.value()
            dcStdDev = self.wavelet_STD_box.value()
            directional = False
            #from time import time
            start_time = time()

            # --- Use the full downsampled image for OS ---
            img_full = self.image_crop_down[:,:,color_channel].copy()  # shape (H_full, W_full)

            # --- Compute the orientation score on the full downsampled image ---
            os_results = ct.os.OrientationScoreTransform(
                img_full / 255 * black_crack, size=size, nOrientations=nOrientations,
                design=design, inflectionPoint=inflectionPoint, mnOrder=mnOrder,
                splineOrder=splineOrder, overlapFactor=overlapFactor,
                dcStdDev=dcStdDev, directional=directional
            )
            '''mask_bin = (self.current_mask > 0)
            xmin, ymin, xmax, ymax = [int(round(v)) for v in self.active_bbox]
            mask_cropped = mask_bin[ymin:ymax, xmin:xmax]
            mask3d = np.broadcast_to(mask_cropped, os_results.shape)
            os_results = np.where(mask3d, os_results, 0)'''
            self.osGFCost = os_results

            print(f"OrientationScoreTransform time: {time() - start_time}")
            self.os_progress_bar.setValue(100)
            self.update_cost_button.setStyleSheet("background-color : lightblue")
            self.show_os_button.setStyleSheet("background-color : lightblue")
        except Exception as e:
            error(e)
            self.update_cost_button.setStyleSheet("background-color : red")
            self.show_os_button.setStyleSheet("background-color : red")
    
    def update_cost(self, return_timing=False):
        import time
        timing = {}

        try:
            t_all0 = time.time()
            self.update_cost_bar.setValue(0)

            lambdaa = self.lambda_box.value()
            p       = self.power_box.value()
            ksi     = 1
            zeta    = 1
            sigmas      = [float(i) for i in self.sigmas_line_edit.text().split(',')]
            sigmas_ext  = 1

            # ---- MultiScaleVesselness ----
            t0 = time.time()
            self.multiscalecostLIFExtReg = ct.os.MultiScaleVesselness(
                self.osGFCost.real, ksi, 1, sigmas, "LIF",
                sigmas_ext=sigmas_ext
            )
            timing["t_ms_vessel"] = time.time() - t0
            print(f"[update_cost] MultiScaleVesselness={timing['t_ms_vessel']:.3f}s")

            # ---- MultiScaleVesselnessFilter ----
            t0 = time.time()
            costmultiscale = ct.os.MultiScaleVesselnessFilter(self.multiscalecostLIFExtReg)
            timing["t_ms_filter"] = time.time() - t0
            print(f"[update_cost] MultiScaleVesselnessFilter={timing['t_ms_filter']:.3f}s")

            # ---- CostFunction ----
            t0 = time.time()
            self.costFunction = ct.os.CostFunction(costmultiscale,
                                                lambdaa=lambdaa, p=p)
            timing["t_cost_fun"] = time.time() - t0
            print(f"[update_cost] CostFunction={timing['t_cost_fun']:.3f}s")

            # ---- Mask sharpening (new mode only) ----
            if ct.os.OS_MODE == "new" and getattr(self, 'current_mask', None) is not None:
                mask_bin = (self.current_mask > 0)
                xmin, ymin, xmax, ymax = [int(round(v)) for v in self.active_bbox]
                mask_cropped = mask_bin[ymin:ymax, xmin:xmax]
                mask3d = np.broadcast_to(mask_cropped, self.costFunction.shape)
                improved = self.costFunction ** 2.0
                PENALTY = 5.0
                self.costFunction = np.where(mask3d, improved,
                                            np.clip(improved * PENALTY, 0, 1))

            # ---- Visualization (c00) ----
            c00 = np.min(ct.os.Rescale(self.costFunction), axis=0)
            c00 = c00 - np.min(c00)
            if c00.max() > 0:
                c00 = (c00 * 255.0 / c00.max()).astype(np.uint8)
            else:
                c00 = np.zeros_like(c00, dtype=np.uint8)

            self.update_cost_bar.setValue(100)

            # display
            qimage = QImage(c00, c00.shape[1], c00.shape[0],
                            c00.strides[0], QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimage)
            scaled = pixmap.scaled(self.cost_display.width(),
                                self.cost_display.height(),
                                Qt.KeepAspectRatio, Qt.FastTransformation)
            self.cost_display.setPixmap(scaled)

            timing["t_cost_total"] = time.time() - t_all0
            print(f"[update_cost] total={timing['t_cost_total']:.3f}s")

            self.midline_track_button.setStyleSheet("background-color : lightblue")

            return (c00, timing) if return_timing else None

        except Exception as e:
            error(e)
            self.midline_track_button.setStyleSheet("background-color : red")
            return None
           
    def midline_tracking(self):
        try:
            import numpy as np
            self.tracking_bar.setValue(0)
            w = self.track_width_box.value()
            color_name = self.track_color_box.currentText()
            color = {'R': (255,0,0), 'G': (0,255,0), 'B': (0,0,255), 'W': (255,255,255)}.get(color_name, (255,0,0))

            y_margin = self.y_margin_box.value()
            x_margin = self.x_margin_box.value()
            g11 = self.g11_box.value()
            g22 = self.g22_box.value()
            g33 = self.g33_box.value()
            downsample_factor = self.downsample_factor_box.value()

            # 1) Inputs
            print("\n[TRACK_DBG] === midline_tracking() ENTRY ===")
            print(f"[TRACK_DBG] downsample_factor={downsample_factor}, g11={g11}, g22={g22}, g33={g33}")
            print(f"[TRACK_DBG] margins: x_margin={x_margin}, y_margin={y_margin}")
            print(f"[TRACK_DBG] self.pts global[0]={self.pts[0]}, global[1]={self.pts[1]}")
            print(f"[TRACK_DBG] self.pts_crop_down[0]={self.pts_crop_down[0]}, self.pts_crop_down[1]={self.pts_crop_down[1]}")

            # 2) Fast marching in crop-down space
            fm_out = ct.tracking.fast_marching(
                self.costFunction,
                self.pts_crop_down[0],
                self.pts_crop_down[1],
                g11=g11, g22=g22, g33=g33
            )
            track_crop_down = np.array(fm_out, dtype=float)  # handle list/tuple
            print(f"[TRACK_DBG] fast_marching output shape={track_crop_down.shape}, "
                f"start={track_crop_down[:,0]}, end={track_crop_down[:,-1]}")

            # 2a) Ensure the path STARTS at p0 (not p1) in crop-down coords
            d0 = np.linalg.norm(track_crop_down[:,0] - self.pts_crop_down[0])
            d1 = np.linalg.norm(track_crop_down[:,0] - self.pts_crop_down[1])
            need_reverse = (d1 < d0)
            if need_reverse:
                track_crop_down = track_crop_down[:, ::-1]
                print(f"[TRACK_DBG] ⚠ path reversed so that first sample anchors at p0 "
                    f"(d0={d0:.2f}, d1={d1:.2f})")
            else:
                print(f"[TRACK_DBG] path already starts near p0 (d0={d0:.2f}, d1={d1:.2f})")

            # 3) Rescale back to crop space
            track_crop_down[0] -= 0.5
            track_crop_down[1] -= 0.5
            track_crop = track_crop_down.copy()
            track_crop[0] *= downsample_factor
            track_crop[1] *= downsample_factor
            self.track_crop = track_crop
            print(f"[TRACK_DBG] track_crop start={track_crop[:,0]}, end={track_crop[:,-1]}")
            # Tripwire: check start vs p0 in crop space
            p0_local = np.asarray(self.pts[0]) - np.asarray(self.active_bbox[:2][::-1])[::-1]  # safer below:
            xmin, ymin, xmax, ymax = [int(round(v)) for v in self.active_bbox]
            p0_local = np.array([self.pts[0][0]-xmin, self.pts[0][1]-ymin], float)
            print(f"[TRACK_DBG] Δ_start(crop→p0_local)={np.linalg.norm(track_crop[:,0]-p0_local):.2f}px")

            # 4) Map to full image space
            track_full_out = ct.tools.track_crop_to_full(track_crop, self.pts[0], self.pts[1],
                                                        y_margin, x_margin)
            track_full = np.array(track_full_out, dtype=float)
            self.track = track_full
            start_full = track_full[:, 0]
            end_full   = track_full[:, -1]
            print(f"[TRACK_DBG] track_full start={start_full}, end={end_full}")
            print(f"[TRACK_DBG] pts[0]={self.pts[0]}, pts[1]={self.pts[1]}")
            print(f"[TRACK_DBG] Δ_start(full→manual)={np.linalg.norm(start_full - self.pts[0]):.2f}px, "
                f"Δ_end(full→manual)={np.linalg.norm(end_full - self.pts[1]):.2f}px")

            # 5) Visualize (unchanged)
            pts = np.array(track_crop).transpose(1,0).reshape((-1,1,2)).astype(np.int32)
            im = self.image_crop.astype(np.uint8)
            im = cv2.polylines(im, [pts], False, color, w)
            qimage = QImage(im, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            scaled_pixmap = pixmap.scaled(self.track_display.width(), self.track_display.height(),
                                        Qt.KeepAspectRatio, Qt.FastTransformation)
            self.track_display.setPixmap(scaled_pixmap)
            self.tracking_bar.setValue(100)
            self.update_track_display_button.setStyleSheet("background-color : lightblue")
            self.track_full_screen_button.setStyleSheet("background-color : lightblue")
            self.edge_mask_button.setStyleSheet("background-color : lightblue")
            print("[TRACK_DBG] === midline_tracking() EXIT ===\n")

        except Exception as e:
            import traceback; traceback.print_exc()
            error(e)
            self.update_track_display_button.setStyleSheet("background-color : red")
            self.track_full_screen_button.setStyleSheet("background-color : red")
            self.edge_mask_button.setStyleSheet("background-color : red")
                        
    def edge_mask(self):
        try:
            window_half_size = int(self.edge_filter_size_box.value() / 2)
            black_crack = -1 if self.crack_color_box.currentText() == 'Bright crack' else 1
            color_channel = 0 if self.color_chenel_box.currentText() == 'R' else (1 if self.color_chenel_box.currentText() == 'B' else 2)

            print(f"[EDGE_MASK] window_half_size={window_half_size}, black_crack={black_crack}, color_channel={color_channel}")
            print(f"[EDGE_MASK] self.track shape={np.array(self.track).shape}, sample={np.array(self.track)[:, :5]}")
            print(f"[EDGE_MASK] self.pts={self.pts}, active_bbox={self.active_bbox}, current_source={getattr(self, 'current_source', None)}")

            img_gray = self.original_image[:, :, color_channel] * black_crack
            track_local_xy = np.array(self.track)
            xmin, ymin, xmax, ymax = [int(round(v)) for v in self.active_bbox]

            # --- coordinate system detection / selection ---
            src = getattr(self, "current_source", None)
            # Heuristic: if within bbox limits, it's already local
            is_local_like = (
                np.all(track_local_xy[0] >= 0) and np.all(track_local_xy[0] < (xmax - xmin)) and
                np.all(track_local_xy[1] >= 0) and np.all(track_local_xy[1] < (ymax - ymin))
            )
            print(f"[EDGE_MASK] coord detection → src={src}, is_local_like={is_local_like}")

            if src in ("manual", "manual_poly"):
                # Manual track was stored in crop coords → convert to full image coords
                track_full_y = track_local_xy[0] + ymin
                track_full_x = track_local_xy[1] + xmin
                track_full_yx = np.vstack([track_full_y, track_full_x])
                print("[EDGE_MASK] Manual mode - local→full via +bbox")

            elif is_local_like:
                # Modern AUTO (new midline_tracking output) or eval variant
                track_full_y = track_local_xy[1] + ymin
                track_full_x = track_local_xy[0] + xmin
                track_full_yx = np.vstack([track_full_y, track_full_x])
                print("[EDGE_MASK] Auto eval/new pipeline mode - local→full via +bbox (no shift)")

            else:
                # Legacy AUTO (old GUI version, needs swap + shift)
                t = np.vstack([track_local_xy[1], track_local_xy[0]])  # swap to [y,x]
                target_point = np.array([self.pts[1][1], self.pts[1][0]])
                shift_vector = target_point - t[:, 0]
                track_full_yx = t + shift_vector[:, np.newaxis]
                print(f"[EDGE_MASK] Auto legacy GUI mode - applied legacy shift: {shift_vector}")

            # --- tripwire check ---
            auto_start_full = np.array([track_full_yx[1, 0], track_full_yx[0, 0]], float)
            man_start_full = np.asarray(self.pts[0], float)
            print(f"[EDGE_MASK] tripwire C (full coords): auto_start_full={auto_start_full} "
                f"manual_start_full={man_start_full} Δ={np.linalg.norm(auto_start_full - man_start_full):.2f}px")

            # --- mask generation ---
            self.edge_mask1, self.edge_mask2 = ct.segmentation.edge_masks(img_gray, track_full_yx)
            print(f"[EDGE_MASK] edge_mask1 stats: min={self.edge_mask1.min()}, max={self.edge_mask1.max()}, shape={self.edge_mask1.shape}")
            print(f"[EDGE_MASK] edge_mask2 stats: min={self.edge_mask2.min()}, max={self.edge_mask2.max()}, shape={self.edge_mask2.shape}")

            # Crop masks
            self.edge_mask1_crop = self.edge_mask1[ymin:ymax, xmin:xmax]
            self.edge_mask2_crop = self.edge_mask2[ymin:ymax, xmin:xmax]
            print(f"[EDGE_MASK] Cropped masks: shape1={self.edge_mask1_crop.shape}, shape2={self.edge_mask2_crop.shape}")

            # Adjust track to crop coordinates (local [y,x])
            shifted_track = np.zeros_like(track_full_yx)
            shifted_track[0] = track_full_yx[0] - ymin
            shifted_track[1] = track_full_yx[1] - xmin
            self.adjusted_track = shifted_track
            print(f"[EDGE_MASK] adjusted_track start (local [y,x])={self.adjusted_track[:, 0]}")

            # Normalize for display
            edge_mask1_crop = self.edge_mask1_crop - np.min(self.edge_mask1_crop)
            if np.max(edge_mask1_crop) != 0:
                edge_mask1_crop = (edge_mask1_crop * 255 / np.max(edge_mask1_crop)).astype(np.uint8)
            else:
                edge_mask1_crop = (edge_mask1_crop * 255).astype(np.uint8)

            qimage = QImage(edge_mask1_crop, edge_mask1_crop.shape[1], edge_mask1_crop.shape[0],
                            edge_mask1_crop.strides[0], QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimage)
            scaled_pixmap = pixmap.scaled(self.edge_map_display.width(), self.edge_map_display.height(),
                                        Qt.KeepAspectRatio, Qt.FastTransformation)
            self.edge_map_display.setPixmap(scaled_pixmap)
            self.edge_tracks_button.setStyleSheet("background-color : lightblue")

        except Exception as e:
            import traceback
            traceback.print_exc()
            error(e)
            self.edge_tracks_button.setStyleSheet("background-color : red")
        
    def edge_tracks_full_screen(self):
        try:
            w = self.edge_track_width_box.value()
            if self.edge_track_color_box.currentText() == "R":
                c = 'r'
            elif self.edge_track_color_box.currentText() == "G":
                c = 'g'
            elif self.edge_track_color_box.currentText() == "B":
                c = 'b'
            elif self.edge_track_color_box.currentText() == "W":
                c = 'w'

            im = self.image.astype(np.uint8)
            plt.imshow(im)
            plt.plot(self.track_e1[0],self.track_e1[1],color = c,linewidth=w)
            plt.plot(self.track_e2[0],self.track_e2[1],color = c,linewidth=w)
            plt.show()
        except Exception as e:
            error(e)
                    
    def edge_tracking(self):
        try:
            color_channel = [0 if self.edge_track_color_box.currentText() == 'R'
                            else 1 if self.edge_track_color_box.currentText() == 'B' else 2][0]
            w = self.edge_track_width_box.value()
            mu = self.mu_box.value()
            l = self.l_box.value()
            p = self.p_box.value()

            res = ct.segmentation.edges_tracking(
                self.image_crop[:, :, color_channel],
                self.pts_crop,
                self.edge_mask1_crop, self.edge_mask2_crop, self.adjusted_track, mu=mu, l=l, p=p,
                return_normal_edges=True
            )

            track_e1, track_e2 = res["geodesic_edges"]
            derived_midline = res.get("derived_midline", None)
            normal_edges = res["normal_edge_points"]
            normal_edges_clipped = res["normal_edge_points_clipped"]

            # ---- Respect pre-assigned crack ID (set in run_pipeline) ----
            if not hasattr(self, "crack_tracks"):
                self.crack_tracks = {}

            if hasattr(self, "current_crack_id") and self.current_crack_id is not None:
                crack_id = int(self.current_crack_id)
            else:
                # Fallback allocator if someone calls edge_tracking() ad-hoc
                ann = self.annotation.get("annotations", {})
                ac = ann.get("atomic_cracks", {})
                used = {int(k) for k in ac.keys() if str(k).isdigit()}
                used |= set(int(k) for k in self.crack_tracks.keys())
                crack_id = (max(used) + 1) if used else 0
                self.current_crack_id = crack_id

            print(f"Using crack id {self.current_crack_id} in edge_tracking")

            self.crack_tracks[crack_id] = [track_e1[:, 0], track_e1[:, 1]]
            self.track_e1 = [track_e1[:, 0], track_e1[:, 1]]
            self.track_e2 = [track_e2[:, 0], track_e2[:, 1]]

            # --- Store derived midline in crop + full-image coordinates ---
            if not hasattr(self, "derived_midline_points"):
                self.derived_midline_points = {}
            if not hasattr(self, "derived_midline_points_full"):
                self.derived_midline_points_full = {}

            # --- Store both crop and full-image normal edges ---
            if not hasattr(self, "normal_edge_points"):
                self.normal_edge_points = {}
            if not hasattr(self, "normal_edge_points_full"):
                self.normal_edge_points_full = {}

            self.normal_edge_points[crack_id] = normal_edges  # crop coords

            xmin, ymin, xmax, ymax = [int(round(v)) for v in self.active_bbox]
            if derived_midline is not None:
                dm = np.asarray(derived_midline, float)
                if dm.ndim == 2 and dm.shape[1] == 2 and len(dm) >= 2:
                    self.derived_midline_points[crack_id] = dm.tolist()
                    dm_global = np.stack([dm[:, 0] + xmin, dm[:, 1] + ymin], axis=1)
                    self.derived_midline_points_full[crack_id] = dm_global.tolist()
                else:
                    print(f"[WARN] derived_midline malformed for crack {crack_id}; skipping derived-midline save")
            (e1x, e1y), (e2x, e2y) = normal_edges
            e1_global = np.stack([e1x + xmin, e1y + ymin], axis=1)
            e2_global = np.stack([e2x + xmin, e2y + ymin], axis=1)
            self.normal_edge_points_full[crack_id] = {
                "edge1": e1_global.tolist(),
                "edge2": e2_global.tolist()
            }

            if not hasattr(self, "normal_edge_points_clipped"):
                self.normal_edge_points_clipped = {}
            self.normal_edge_points_clipped[crack_id] = normal_edges_clipped

            # --- Draw crop tracks in lower-left panel ---
            pts1 = np.array([track_e1[:, 0], track_e1[:, 1]]).T.reshape((-1, 1, 2)).astype(np.int32)
            pts2 = np.array([track_e2[:, 0], track_e2[:, 1]]).T.reshape((-1, 1, 2)).astype(np.int32)
            im = self.image_crop.astype(np.uint8)
            im = cv2.polylines(im, [pts1], False, (0, 255, 0), w)
            im = cv2.polylines(im, [pts2], False, (0, 255, 0), w)
            qimage = QImage(im, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            scaled_pixmap = pixmap.scaled(self.edge_tracks_display.width(), self.edge_tracks_display.height(),
                                        Qt.KeepAspectRatio, Qt.FastTransformation)
            self.edge_tracks_display.setPixmap(scaled_pixmap)

            # --- NEW: Show all cracks' masks in the right-hand Segmentation panel ---
            full_mask_display = np.zeros(self.image.shape[:2], dtype=np.uint8)
            ann = self.annotation.get("annotations", {})
            atomic_cracks = ann.get("atomic_cracks", {})
            for crack in atomic_cracks.values():
                mc = crack.get("mask_crop")
                bb = crack.get("mask_bbox")
                if mc is not None and bb is not None:
                    crop = np.array(mc, dtype=np.uint8)
                    x, y, w_box, h_box = [int(v) for v in bb]
                    x2, y2 = min(x + w_box, self.image.shape[1]), min(y + h_box, self.image.shape[0])
                    w_eff, h_eff = max(0, x2 - x), max(0, y2 - y)
                    if h_eff > 0 and w_eff > 0:
                        crop = (crop > 0).astype(np.uint8)[:h_eff, :w_eff]
                        full_mask_display[y:y + h_eff, x:x + w_eff] |= crop

            # Overlay mask boundaries
            from skimage.segmentation import mark_boundaries
            display_im = (mark_boundaries(self.original_image / 255.0, full_mask_display,
                                        color=(0, 0, 1), background_label=0) * 255).astype(np.uint8)

            qimage = QImage(display_im, display_im.shape[1], display_im.shape[0],
                            display_im.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            scaled_pixmap = pixmap.scaled(self.ImageScreen.width(), self.ImageScreen.height(),
                                        Qt.KeepAspectRatio, Qt.FastTransformation)
            self.ImageScreen.setPixmap(scaled_pixmap)

            self.edge_tracks_full_screen_button.setStyleSheet("background-color : lightblue")
            self.save_current_segment_button.setStyleSheet("background-color : lightblue")

        except Exception as e:
            error(e)
            self.edge_tracks_full_screen_button.setStyleSheet("background-color : red")
            self.save_current_segment_button.setStyleSheet("background-color : red")

    def save_current_segment(self):
        try:
            xmin, ymin, xmax, ymax = [int(round(v)) for v in self.active_bbox]
            print(f"[DEBUG] save_current_segment bbox: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")

            # --------------------------------------------------------
            # 🧩 1. Track sanity — BEFORE any mask building
            if hasattr(self, "track"):
                t = np.asarray(self.track)
                print(f"[DEBUG TRACK] shape={t.shape}, min(x)={t[1].min():.2f}, max(x)={t[1].max():.2f}, "
                    f"min(y)={t[0].min():.2f}, max(y)={t[0].max():.2f}")
            else:
                print("[DEBUG TRACK] MISSING self.track!")

            if hasattr(self, "image_crop"):
                h_c, w_c = self.image_crop.shape[:2]
                print(f"[DEBUG CROP] image_crop shape={self.image_crop.shape}, "
                    f"global_bbox size={(xmax-xmin)}x{(ymax-ymin)}")
            else:
                print("[DEBUG CROP] MISSING image_crop!")

            # --------------------------------------------------------
            # 🧩 2. Edge sanity
            for nm, arr in [("track_e1", getattr(self, "track_e1", None)),
                            ("track_e2", getattr(self, "track_e2", None))]:
                if arr is None:
                    print(f"[DEBUG {nm}] MISSING")
                    continue
                print(f"[DEBUG {nm}] len={len(arr[0]) if isinstance(arr, (list,tuple)) else 'n/a'}, "
                    f"minx={np.min(arr[1]):.2f} maxx={np.max(arr[1]):.2f} "
                    f"miny={np.min(arr[0]):.2f} maxy={np.max(arr[0]):.2f}")

            '''# --------------------------------------------------------
            # 3️⃣ Build crop mask
            edge_x_crop = np.concatenate((self.track_e1[1][::-1], self.track_e2[1]))
            edge_y_crop = np.concatenate((self.track_e1[0][::-1], self.track_e2[0]))
            mask_crop = ct.segmentation.generate_mask_from_edges(self.image_crop, edge_y_crop, edge_x_crop).astype(np.uint8)
            h, w = mask_crop.shape[:2]
            nz = int((mask_crop > 0).sum())
            print(f"[DEBUG MASK_CROP] shape={mask_crop.shape}, nonzero={nz}")

            # Sanity check for out-of-bounds coordinates
            if (edge_x_crop < 0).any() or (edge_y_crop < 0).any() or \
            (edge_x_crop >= w).any() or (edge_y_crop >= h).any():
                print(f"[WARN] edge coords out of crop bounds! "
                    f"x range=({edge_x_crop.min():.1f},{edge_x_crop.max():.1f}), "
                    f"y range=({edge_y_crop.min():.1f},{edge_y_crop.max():.1f}), "
                    f"crop wh=({w},{h})")'''
                    
            # --------------------------------------------------------
            src = getattr(self, "current_source", "auto")
            track_arr = np.array(self.adjusted_track, dtype=float)
            midline_coords = [[int(track_arr[1][i] + xmin), int(track_arr[0][i] + ymin)]
                            for i in range(track_arr.shape[1])]
            print(f"[DEBUG MIDLINE] len={len(midline_coords)}, "
                f"first={midline_coords[0]}, last={midline_coords[-1]}")

            # Derived midline is produced by edges_tracking and stored in both crop/full coords.
            derived_midline_coords = getattr(self, "derived_midline_points_full", {}).get(self.current_crack_id)
            dm_crop = getattr(self, "derived_midline_points", {}).get(self.current_crack_id)
            if dm_crop is not None:
                dm_crop = np.asarray(dm_crop, float)
                if dm_crop.ndim != 2 or dm_crop.shape[1] != 2 or len(dm_crop) < 2:
                    dm_crop = None
            if derived_midline_coords is None and dm_crop is not None:
                dm_global = np.stack([dm_crop[:, 0] + xmin, dm_crop[:, 1] + ymin], axis=1)
                derived_midline_coords = dm_global.tolist()

            # If derived midline is unavailable, use adjusted track as legacy fallback in crop coords.
            if dm_crop is None:
                dm_crop = np.column_stack([track_arr[1], track_arr[0]])

            # --------------------------------------------------------
            # 3️⃣ Build crop mask (CORRECT API + GEOMETRY)
            # --------------------------------------------------------
            e1 = np.column_stack([
                np.asarray(self.track_e1[1], float),  # x
                np.asarray(self.track_e1[0], float),  # y
            ])
            e2 = np.column_stack([
                np.asarray(self.track_e2[1], float),  # x
                np.asarray(self.track_e2[0], float),  # y
            ])

            # Authoritative normals from edges_tracking (crop coords)
            normal_edges = getattr(self, "normal_edge_points", {}).get(self.current_crack_id)
            if isinstance(normal_edges, dict) and "edge1" in normal_edges and "edge2" in normal_edges:
                n1 = np.asarray(normal_edges["edge1"], float)
                n2 = np.asarray(normal_edges["edge2"], float)
            elif isinstance(normal_edges, (list, tuple)) and len(normal_edges) == 2:
                n1 = np.column_stack([np.asarray(normal_edges[0][0], float), np.asarray(normal_edges[0][1], float)])
                n2 = np.column_stack([np.asarray(normal_edges[1][0], float), np.asarray(normal_edges[1][1], float)])
            else:
                raise RuntimeError("save_current_segment requires normal_edge_points from edge_tracking")

            dbg_dir = os.path.join(self.save_folder, "debug_masks", f"cid{self.current_crack_id}")
            os.makedirs(dbg_dir, exist_ok=True)

            mask_crop = ct.segmentation.generate_mask_from_edges(
                img_gray=self.image_crop,
                edge1_xy=e1,
                edge2_xy=e2,
                midline_xy=dm_crop,
                normals_xy=(n1, n2),
                out_dir=dbg_dir,
                tag=f"cid{self.current_crack_id}",
                do_morph=False,
            ).astype(np.uint8)

            h, w = mask_crop.shape[:2]
            nz = int((mask_crop > 0).sum())
            print(f"[DEBUG MASK_CROP] shape={mask_crop.shape}, nonzero={nz}")

            if (
                (e1[:, 0] < 0).any() or (e1[:, 1] < 0).any() or
                (e2[:, 0] < 0).any() or (e2[:, 1] < 0).any() or
                (e1[:, 0] >= w).any() or (e1[:, 1] >= h).any() or
                (e2[:, 0] >= w).any() or (e2[:, 1] >= h).any()
            ):
                print(
                    f"[WARN] edge coords out of crop bounds! "
                    f"e1 x=({e1[:,0].min():.1f},{e1[:,0].max():.1f}) "
                    f"y=({e1[:,1].min():.1f},{e1[:,1].max():.1f}), "
                    f"e2 x=({e2[:,0].min():.1f},{e2[:,0].max():.1f}) "
                    f"y=({e2[:,1].min():.1f},{e2[:,1].max():.1f}), "
                    f"crop wh=({w},{h})"
                )

            ann = self.annotation.setdefault("annotations", {})
            atomic_cracks = ann.setdefault("atomic_cracks", {})

            # --------------------------------------------------------
            # 🧩 4. Normal edges sanity
            normal_edges = getattr(self, "normal_edge_points", {}).get(self.current_crack_id)
            print(f"[DEBUG] normal_edge_points type={type(normal_edges)}, "
                f"keys={list(getattr(self,'normal_edge_points',{}).keys())}")

            e1x = e1y = e2x = e2y = None
            if isinstance(normal_edges, dict):
                if "edge1" in normal_edges and "edge2" in normal_edges:
                    e1 = np.array(normal_edges["edge1"], float)
                    e2 = np.array(normal_edges["edge2"], float)
                    print(f"[DEBUG NORMAL_EDGE] edge1 shape={e1.shape}, edge2 shape={e2.shape}")
            elif isinstance(normal_edges, (list, tuple)) and len(normal_edges) == 2:
                e1x, e1y = np.array(normal_edges[0][0]), np.array(normal_edges[0][1])
                e2x, e2y = np.array(normal_edges[1][0]), np.array(normal_edges[1][1])
            else:
                print("[WARN] normal_edge_points missing or malformed → fallback using track_e1/e2")

            # --------------------------------------------------------
            # 🧩 5. Mask write summary
            print(f"[DEBUG SAVE_SUMMARY] src={src}, bbox=({xmin},{ymin},{xmax},{ymax}), "
                f"mask_nonzero={nz}, track_pts={track_arr.shape[1]}")

            # original remainder
            e1x, e1y = self.track_e1[1], self.track_e1[0]
            e2x, e2y = self.track_e2[1], self.track_e2[0]
            e1_global = np.stack([e1x + xmin, e1y + ymin], axis=1)
            e2_global = np.stack([e2x + xmin, e2y + ymin], axis=1)
            normal_edges_full = {
                "edge1": np.round(e1_global, 2).tolist(),
                "edge2": np.round(e2_global, 2).tolist(),
            }

            crack_entry = {
                "source": src,
                "midline": midline_coords,
                "derived_midline": derived_midline_coords,
                "geodesic_edges": {
                    "edge1": np.stack([self.track_e1[0] + xmin, self.track_e1[1] + ymin], axis=1),
                    "edge2": np.stack([self.track_e2[0] + xmin, self.track_e2[1] + ymin], axis=1),
                },
                "normal_edge_points": normal_edges_full,
                "mask_crop": mask_crop,
                "mask_bbox": [int(xmin), int(ymin), int(w), int(h)],
                "user_points": getattr(self, "user_points", []),
                "user_connections": getattr(self, "user_connections", []),
            }

            crack_entry = save_load_files._to_py(crack_entry)
            atomic_cracks[str(self.current_crack_id)] = crack_entry
            print(f"[DEBUG] atomic_cracks updated for id={self.current_crack_id}")
            
            # ✅ only insert if it’s valid and non-empty
            if crack_entry.get("midline") and len(crack_entry["midline"]) > 1:
                atomic_cracks[str(self.current_crack_id)] = crack_entry
                print(f"[DEBUG] atomic_cracks updated for id={self.current_crack_id}")
            else:
                print(f"[SKIP] empty or invalid crack for id={self.current_crack_id} — not saving.")

            self.use_masks = True
            self.save_annotation()
        except Exception as e:
            print(f"[FUBAR] save_current_segment crashed: {e}")

            
    def show_os(self):
        import plotly.graph_objects as go
        import numpy as np
        color_channel = [0 if self.color_chenel_box.currentText()=='R' else 1 if self.color_chenel_box.currentText()=='B' else 2]
        shift = -30
        osGFCost_shift = np.roll(self.osGFCost, shift=shift,axis = 0)
        osGFCost_shift.shape
        downsample_factor = self.downsample_factor_box.value()
        downsample_rate_spatial = downsample_factor
        downsample_rate_angular = 1
        # angular_shift = int(1*osGFCost.shape[0])
        # osGFCosts = 
        values = osGFCost_shift[:int(self.osGFCost.shape[0]/2),:,:].real
        X, Y, Z = np.mgrid[:values.shape[0],:values.shape[1],:values.shape[2]]
   
        yim = np.linspace(0,self.image_crop_down.shape[0],self.image_crop_down.shape[0])
        zim = np.linspace(0,self.image_crop_down.shape[1],self.image_crop_down.shape[1])


        yim,zim = np.meshgrid(yim,zim)
        xim = np.ones(yim.shape)*0

        a = self.osGFCost.shape[0]/2-1
        b = self.osGFCost.shape[1]
        c = self.osGFCost.shape[2]

        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=values.flatten(),
            isomin=np.min(values),
            isomax=np.max(values),
            opacity=1, # needs to be small to see through all surfaces
            opacityscale='min',
            colorscale='Hot',
            caps= dict(x_show=False, y_show=False, z_show=False, x_fill=1),

        ))

        fig.update_traces(lighting=dict(ambient = 0.4,diffuse = 0.9,fresnel = 0.8,roughness = 0.5,specular = 0.05),
                        selector=dict(type='volume'))


        fig.update_traces(surface=dict(count=5,fill = 1,pattern='all',show=True), selector=dict(type='volume'))


        r = self.image_crop.shape[0]/self.image_crop_down.shape[1]
        fig.update_layout(scene_aspectmode='manual',
                        scene_aspectratio=dict(x=0.5, y=r, z=1))

        fig.update_layout(scene_xaxis_showticklabels=False,
                        scene_yaxis_showticklabels=False,
                        scene_zaxis_showticklabels=False),
        fig.add_surface(x=xim, y=yim, z=zim, 
                        surfacecolor=self.image_crop_down[:,:,color_channel][:,:,0].T, 
                        colorscale='gray', 
                        showscale=False)

        fig.add_scatter3d(
                x=[0, 0, a, a, 0, 0, a, a, a, 0, 0, 0, 0, a, a, a],
                y=[0, b, b, 0, 0, 0, 0, b, b, b, b, 0, b, b, 0, 0],
                z=[0, 0, 0, 0, 0, c, c, c, 0, 0, c, c, c, c, c, 0],
            mode = 'lines',
            line=dict(
                color='black',
                width=2
            )
        )

        fig.update_layout(scene = dict(
                            xaxis = dict(showbackground=False),
                            yaxis = dict(showbackground=False),
                            zaxis = dict(showbackground=False)))
        fig.update_layout(scene = dict(
                            xaxis_title="θ",
                            yaxis_title='x1',
                            zaxis_title='x2'))



        fig.update_traces(showscale=True, selector=dict(type='volume'))
        fig.update_layout(margin_autoexpand=True)
        fig.update_layout(font=dict(size = 60))
        k = 4.5
        camera = dict(
            up=dict(x=1, y=0, z=0),
            center=dict(x=-0.15, y=0, z=0),
            eye=dict(x=0.085*k, y=0.23*k, z=0.2*k)
        )
        fig.update_layout(scene_camera=camera,title='default')

        fig.show()
    
    def update_track_display(self):
        try :
            w = self.track_width_box.value()
            if self.track_color_box.currentText() == "R":
                color = (255,0,0)
            elif self.track_color_box.currentText() == "G":
                color = (0,255,0)
            elif self.track_color_box.currentText() == "B":
                color = (0,0,255)
            elif self.track_color_box.currentText() == "W":
                color = (255,255,255)
            pts = np.array(self.track_crop).transpose(1,0).reshape((-1,1,2)).astype(np.int32)
            im = self.image_crop.astype(np.uint8)
            im = cv2.polylines(im, [pts], False, color, w)
            qimage = QImage(im, im.shape[1], im.shape[0], 
                            im.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            scaled_pixmap = pixmap.scaled(self.track_display.width(), self.track_display.height(), Qt.KeepAspectRatio, Qt.FastTransformation)
            self.track_display.setPixmap(scaled_pixmap)
        except Exception as e:
            error(e)

    def track_full_screen(self):
        try :
            w = self.track_width_box.value()
            if self.track_color_box.currentText() == "R":
                color = (255,0,0)
                c = 'r'
            elif self.track_color_box.currentText() == "G":
                color = (0,255,0)
                c = 'g'
            elif self.track_color_box.currentText() == "B":
                color = (0,0,255)
                c = 'b'
            elif self.track_color_box.currentText() == "W":
                color = (255,255,255)
                c = 'w'

            im = self.image.astype(np.uint8)
        except Exception as e:
            error(e)

    def check_wavelet(self):
        try:
            size = self.wavelet_size_box.value()
            nOrientations = self.wavelet_norientations_box.value()
            design = "N"
            inflectionPoint = self.wavelet_inflection_point_box.value()
            mnOrder = self.wavelet_mnorder_box.value()
            splineOrder = 3
            overlapFactor = self.wavelet_overlap_factor_box.value()
            dcStdDev = self.wavelet_STD_box.value()
            directional = False
            window_size = self.wavelet_window_size_box.value()

            wavelet = ct.os.CheckWavelet(window_size = window_size, size = size, nOrientations = nOrientations, design = design, 
                            inflectionPoint = inflectionPoint, mnOrder = mnOrder, splineOrder = splineOrder,
                            overlapFactor = overlapFactor, dcStdDev = dcStdDev, directional = directional,
                            display_orientations=[0])[0,:,:]
            wavelet = wavelet - np.min(wavelet)
            wavelet = (wavelet*254/np.max(wavelet)).astype(dtype=np.int8)

            qimage = QImage(wavelet.astype(dtype=np.uint8), wavelet.shape[1], wavelet.shape[0], 
                            wavelet.strides[0], QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimage)
            scaled_pixmap = pixmap.scaled(self.wavelet_check_display.width(), self.wavelet_check_display.height(), Qt.KeepAspectRatio, Qt.FastTransformation)
            self.wavelet_check_display.setPixmap(scaled_pixmap)
        except Exception as e:
            error(e)
            
    def update_os_cost(self, mode="new", save_dir=None, return_timing=False):
        """
        Unified OS + COST generator.
        Returns:
            (cost_volume, timing_dict)  if return_timing=True
            cost_volume                 if return_timing=False

        cost_volume = self.costFunction (3-D), suitable for RS3.
        """
        import time, cv2
        import numpy as np
        import cracktools as ct

        # Remember old mode so global state is restored later
        prev_mode = getattr(ct.os, "OS_MODE", "new")

        try:
            # ---- 1) Set OS MODE ----
            if hasattr(ct.os, "set_os_mode"):
                ct.os.set_os_mode(mode)
            else:
                ct.os.OS_MODE = mode

            # ---- 2) Ensure crop is up to date ----
            if hasattr(self, "update_image_crop"):
                self.update_image_crop()

            # ---- 3) OS stage ----
            t0 = time.time()
            if hasattr(self, "update_os"):
                self.update_os()
            t_os = time.time() - t0
            print(f"[update_os_cost] OS ({mode}) = {t_os:.3f}s")

            # ---- 4) COST stage ----
            out = self.update_cost(return_timing=True)
            if out is None:
                print(f"[update_os_cost] ❌ update_cost failed (mode={mode})")
                return None if not return_timing else (None, {})
            c00, timing_cost = out
            t_cost = timing_cost.get("t_cost_total", 0.0)

            # ---- 5) Extract full 3-D cost volume for RS3 ----
            cost_volume = getattr(self, "costFunction", None)
            if cost_volume is None:
                print(f"[update_os_cost] ❌ self.costFunction is None (mode={mode})")
                return None if not return_timing else (None, {})

            cost_volume = np.asarray(cost_volume, dtype=float)

            # ---- 6) Optionally save 2-D preview ----
            png_path = None
            if save_dir is not None:
                try:
                    os.makedirs(save_dir, exist_ok=True)
                    png_path = f"{save_dir}/os_cost_{mode}.png"
                    cv2.imwrite(png_path, c00)
                    print(f"[update_os_cost] saved preview → {png_path}")
                except Exception as e:
                    print(f"[update_os_cost] ⚠ preview save failed: {e}")

            # ---- 7) Package timing ----
            timing = {
                "mode": mode,
                "os_sec": t_os,
                "cost_sec": t_cost,
                **timing_cost,
                "png_path": png_path,
            }

            # ---- 8) Return ----
            if return_timing:
                return cost_volume, timing
            else:
                return cost_volume

        except Exception as e:
            print(f"[update_os_cost] ❌ exception: {e}")
            return None if not return_timing else (None, {})

        finally:
            # Restore previous OS_MODE so GUI stays consistent
            ct.os.OS_MODE = prev_mode
