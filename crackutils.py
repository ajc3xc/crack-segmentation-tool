#!/usr/bin/env python3

import cracktools as ct
from helpers.crackhelpers import *

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import matplotlib
matplotlib.use("Agg")   # fastest non-GUI backend for PNG/PDF/SVG export
matplotlib.rcParams.update({
    "figure.max_open_warning": 0,
    "text.kerning_factor": 0,
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
})
import matplotlib.pyplot as plt
plt.ioff()   # no interactive figure updates

import numpy as np
from math import hypot, atan2, pi
from skimage.morphology import skeletonize
import skimage
import hashlib
import time

from helpers.endpoint_annotator import CrackAnnotator


min_crop_size = 16
ROUNDING_DIGITS=6

from helpers.metrics import *

#This class is basically is all of the utility / save and load or unimportant functions that aren't directly accessible via a ui button or aren't important
#
class CrackUtils:
    """
    UI-agnostic helpers you can call from your main app.
    These methods are copied verbatim from your current file.
    """

    # --- moved verbatim ---
    def save_annotation(self):
        self._debug_print_atomic_cracks("save_annotation START")
        import numpy as np
        import os, cv2

        try:
            ann = self.annotation.setdefault("annotations", {})
            H, W = self.image.shape[:2]

            base_name = os.path.splitext(os.path.basename(self.name))[0]
            json_path = os.path.join(self.save_folder, base_name + '.json')
            mask_bin_path = os.path.join(self.save_folder, base_name + '_mask.png')
            mask_255_path = os.path.join(self.save_folder, base_name + '_mask255.png')

            # NO MERGE: trust in-memory
            atomic_cracks = ann.setdefault("atomic_cracks", {})

            # 1) Compact any legacy full masks
            compact_full_masks_in_ann(ann, H, W)

            # 2) Build combined mask from atomic cracks
            mask_combined = build_combined_mask(ann.get("atomic_cracks", {}), H, W)
            print(f"[DEBUG] mask_combined nonzero: {int(mask_combined.sum())}")

            # 3) Filter to valid cracks (and drop legacy 'mask' if we have compact)
            ann["atomic_cracks"] = filter_valid_cracks(ann.get("atomic_cracks", {}), H, W)
            print(f"[DEBUG] save_annotation END: {len(ann['atomic_cracks'])} cracks kept.")

            # Optional: keep combined_cracks if you manage them elsewhere
            if hasattr(self, "combined_cracks"):
                ann["combined_cracks"] = self.combined_cracks

            # 4) Safe JSON write
            safe_json_dump(self.annotation, json_path)

            # 5) Write PNGs (compressed)
            comp = int(getattr(self, "png_compression_level", 9))
            comp = max(0, min(comp, 9))
            cv2.imwrite(mask_bin_path, (mask_combined * 1).astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, comp])
            cv2.imwrite(mask_255_path, (mask_combined * 255).astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, comp])
            print(f"Saved: {json_path}, {mask_bin_path}, {mask_255_path}  (png_compression_level={comp})")

        except Exception as e:
            # keep your existing global error() function in main file
            error(e)

    # --- moved verbatim ---
    def get_all_bounding_boxes(self):
        """
        Return a list of all saved bounding boxes as [ [xmin, ymin, xmax, ymax], ... ]
        """
        boxes = []
        try:
            if 'box' in self.annotation['annotations']:
                for box_k in self.annotation['annotations']['box']:
                    bb = self.annotation['annotations']['box'][box_k]['bounding_box']
                    # bb: [[x0, y0], [x1, y1]]
                    xs = [bb[0][0], bb[1][0]]
                    ys = [bb[0][1], bb[1][1]]
                    xmin, xmax = min(xs), max(xs)
                    ymin, ymax = min(ys), max(ys)
                    boxes.append([xmin, ymin, xmax, ymax])
        except Exception as e:
            print(f"Could not parse bounding boxes: {e}")
        return boxes

    # --- moved verbatim ---
    def _debug_print_atomic_cracks(self, label):
        if not hasattr(self, "annotation") or self.annotation is None:
            error(f"No annotation object when called from {label}")
            return
        ann = self.annotation.get("annotations", {})
        cracks = ann.get("atomic_cracks", {})
        print(f"\n[DEBUG] {label}: {len(cracks)} cracks currently stored.")
        for cid, crack in cracks.items():
            src = crack.get("source", crack.get("src", "?"))
            midline_len = len(crack.get("midline", []))
            mask_nonzero = np.count_nonzero(np.array(crack.get("mask", []), dtype=np.uint8))
            print(f"    ID={cid} src={src} midline_len={midline_len} mask_pixels={mask_nonzero}")
    
    def select_folder(self):
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox, QCheckBox

        def load_last_folders():
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'folder_config.json')
            if os.path.isfile(config_path):
                try:
                    with open(config_path, 'r') as f:
                        d = json.load(f)
                        return d.get("img_folder", ""), d.get("save_folder", ""), d.get("mask_folder", ""), d.get("use_masks", False)
                except Exception:
                    return "", "", "", False
            return "", "", "", False

        def save_last_folders(img_folder, save_folder, mask_folder, use_masks):
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'folder_config.json')
            try:
                with open(config_path, 'w') as f:
                    json.dump({
                        "img_folder": img_folder,
                        "save_folder": save_folder,
                        "mask_folder": mask_folder,
                        "use_masks": use_masks
                    }, f)
            except Exception:
                pass

        # --- Load previous defaults ---
        default_img_folder, default_save_folder, default_mask_folder, default_use_masks = load_last_folders()
        img_folder_init = getattr(self, "current_folder", default_img_folder)
        save_folder_init = getattr(self, "save_folder", default_save_folder)
        mask_folder_init = getattr(self, "mask_folder", default_mask_folder)
        use_masks_init = getattr(self, "use_masks", default_use_masks)

        dlg = QDialog(self.MainWindow)
        dlg.setWindowTitle("Select Image & Save Folders")
        layout = QVBoxLayout(dlg)

        # Image folder row
        img_row = QHBoxLayout()
        img_label = QLabel("Image folder:")
        image_folder_edit = QLineEdit()
        image_folder_edit.setText(img_folder_init)
        img_browse_btn = QPushButton("Browse...")
        img_row.addWidget(img_label)
        img_row.addWidget(image_folder_edit)
        img_row.addWidget(img_browse_btn)
        layout.addLayout(img_row)

        # Save folder row
        save_row = QHBoxLayout()
        save_label = QLabel("Save folder:")
        save_folder_edit = QLineEdit()
        save_folder_edit.setText(save_folder_init)
        save_browse_btn = QPushButton("Browse...")
        save_row.addWidget(save_label)
        save_row.addWidget(save_folder_edit)
        save_row.addWidget(save_browse_btn)
        layout.addLayout(save_row)

        # Use Masks checkbox (controls mask row visibility)
        use_mask_checkbox = QCheckBox("Use Masks")
        use_mask_checkbox.setChecked(use_masks_init)
        layout.addWidget(use_mask_checkbox)

        # Mask folder row
        mask_row = QHBoxLayout()
        mask_label = QLabel("Mask folder:")
        mask_folder_edit = QLineEdit()
        mask_folder_edit.setText(mask_folder_init)
        mask_browse_btn = QPushButton("Browse...")
        mask_row.addWidget(mask_label)
        mask_row.addWidget(mask_folder_edit)
        mask_row.addWidget(mask_browse_btn)
        layout.addLayout(mask_row)

        # Show/hide mask folder row based on checkbox
        def update_mask_row():
            visible = use_mask_checkbox.isChecked()
            for i in range(mask_row.count()):
                w = mask_row.itemAt(i).widget()
                if w: w.setVisible(visible)
        use_mask_checkbox.toggled.connect(update_mask_row)
        update_mask_row()  # set initial state

        # Button row
        btn_row = QHBoxLayout()
        ok_btn = QPushButton("Select")
        cancel_btn = QPushButton("Cancel")
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        # Browse logic
        def img_browse():
            folder = QFileDialog.getExistingDirectory(dlg, "Select Image Folder")
            if folder:
                image_folder_edit.setText(folder)
        def save_browse():
            folder = QFileDialog.getExistingDirectory(dlg, "Select Save Folder")
            if folder:
                save_folder_edit.setText(folder)
        def mask_browse():
            folder = QFileDialog.getExistingDirectory(dlg, "Select Mask Folder")
            if folder:
                mask_folder_edit.setText(folder)
        img_browse_btn.clicked.connect(img_browse)
        save_browse_btn.clicked.connect(save_browse)
        mask_browse_btn.clicked.connect(mask_browse)

        ok_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)

        def strip_quotes(path):
            # Remove ONLY a single " from start/end if present
            if path.startswith('"') and path.endswith('"'):
                return path[1:-1]
            elif path.startswith('"'):
                return path[1:]
            elif path.endswith('"'):
                return path[:-1]
            else:
                return path

        # --- Show dialog & check folders ---
        while True:
            if dlg.exec_() == QDialog.Accepted:
                img_folder = strip_quotes(image_folder_edit.text().strip())
                save_folder = strip_quotes(save_folder_edit.text().strip())
                mask_folder = strip_quotes(mask_folder_edit.text().strip())
                use_masks = use_mask_checkbox.isChecked()
                if not os.path.isdir(img_folder):
                    QMessageBox.critical(dlg, "Error", "Please select a valid image folder.")
                    continue
                if not os.path.isdir(save_folder):
                    QMessageBox.critical(dlg, "Error", "Please select a valid save folder.")
                    continue
                if use_masks and (not os.path.isdir(mask_folder)):
                    QMessageBox.critical(dlg, "Error", "Please select a valid mask folder or uncheck 'Use Masks'.")
                    continue
                break
            else:
                print("Folder selection cancelled.")
                return  # Don't continue if user cancels

        # Save for use elsewhere and persist for next run!
        self.current_folder = img_folder
        self.save_folder = save_folder
        self.mask_folder = mask_folder if use_masks else ""
        self.use_masks = use_masks
        save_last_folders(img_folder, save_folder, mask_folder, use_masks)   # Write to config file

        # ---- Wipe all memory/state/arrays for previous folder ----
        self.files_list.clear()
        self.image_names = []
        self.n = 0

        for attr in [
            'mask', 'crack_tracks', 'cracks_stored_endpoints',
            'track_crop', 'track',
            'image', 'original_image', 'image_crop', 'image_crop_down', 'image_down',
            'osGFCost', 'multiscalecostLIFExtReg', 'costFunction',
            'bb_pts_list', 'mid_pt', 'end_points', 'points_pairs_list', 'annotation'
        ]:
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except Exception as e:
                    print(f"Error deleting attribute {attr}: {e}")
                    pass

        import gc; gc.collect()

        # ---- Now load new image list ----
        self.image_names = ct.tools.get_files(folder=img_folder, formats=['jpeg','jpg','png'], basename=False)
        if self.use_masks and self.mask_folder:
            self.mask_names = ct.tools.get_files(folder=self.mask_folder, formats=['png','npy'], basename=False)
            self.mask_map = {os.path.splitext(os.path.basename(f))[0]: f for f in self.mask_names}
            print(f"{len(self.mask_names)} gt masks loaded in with image")
        else:
            self.mask_names = []
            self.mask_map = {}
        #print("....................................................")
        for filename in self.image_names:
            self.files_list.addItem(os.path.basename(filename))
        if self.image_names:
            self.n = 0
            self.change_image()
        else:
            self.ImageScreen.clear()
            self.filename_label_2.setText("No images found in folder.")
        #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        # ---- Optionally load mask file mapping ----

    def name_selected(self):
        self.n = self.files_list.currentRow()
        self.mid_pt = []
        self.end_points = []
        self.mask = []
        self.cracks_stored_endpoints = {}
        self.crack_tracks = {}
        self.change_image()

    def update_selected_item(self,name):
        items = self.files_list.findItems(name, QtCore.Qt.MatchExactly)
        self.files_list.setCurrentItem(items[0])
    
    def next_image(self):
        try:
            self.mid_pt = []
            self.end_points = []
            self.mask = []
            self.cracks_stored_endpoints = {}
            self.crack_tracks = {}
            self.n = self.n + 1
            self.change_image()
        except Exception as e:
            error(e)

    def previous_image(self):
        try:
            self.mid_pt = []
            self.end_points = []
            self.mask = []
            self.cracks_stored_endpoints = {}
            self.crack_tracks = {}
            self.n = self.n - 1
            self.change_image()
        except Exception as e:
            error(e)
        
    # In CrackToolsApplication           
    def draw_box(self):
        # --- Check image loaded ---
        if not hasattr(self, 'original_image') or self.original_image is None:
            error("No image loaded. Please select a folder and an image first.")
            return

        # --- Snap toggle and margin ---
        # Option 1: Use a checkbox if you have it:
        snap_to_edge = getattr(self, 'snap_box_edges_checkbox', None)
        if snap_to_edge is not None:
            snap = snap_to_edge.isChecked()
        else:
            snap = True  # Set default snap behavior here
        
        def snap_box_points(bb_pts, img_shape, margin):
            # bb_pts: 2x2 array [[x1,y1],[x2,y2]], float or int
            h, w = img_shape[:2]
            snapped = []
            for pt in bb_pts:
                x, y = pt
                # Snap each coordinate
                if abs(x) < margin:
                    x = 0
                elif abs(x - (w-1)) < margin:
                    x = w-1
                if abs(y) < margin:
                    y = 0
                elif abs(y - (h-1)) < margin:
                    y = h-1
                snapped.append([int(x), int(y)])
            snapped = np.array(snapped, dtype=int)

            # --- Now check for "sucked" box (identical on any axis) ---
            # For x (column 0)
            if snapped[0,0] == snapped[1,0]:
                # Move furthest point back in by one pixel
                if bb_pts[0,0] < bb_pts[1,0]:
                    snapped[1,0] = min(w-2, max(0, snapped[1,0]-1))
                else:
                    snapped[0,0] = min(w-2, max(0, snapped[0,0]-1))
            # For y (column 1)
            if snapped[0,1] == snapped[1,1]:
                if bb_pts[0,1] < bb_pts[1,1]:
                    snapped[1,1] = min(h-2, max(0, snapped[1,1]-1))
                else:
                    snapped[0,1] = min(h-2, max(0, snapped[0,1]-1))
            return snapped

        self.image_size = self.select_image_size_2.value()
        self.bb_pts_list = getattr(self, 'bb_pts_list', [])
        display_image = self.original_image.copy()

        # Draw saved (blue) and pending (green) boxes
        for box_dict in (self.annotation.get('annotations', {}).get('box') or {}).values():
            bb = np.array(box_dict['bounding_box'], dtype=np.int32)
            cv2.rectangle(display_image, tuple(bb[0]), tuple(bb[1]), (0,128,255), 3)
        for bb in self.bb_pts_list:
            if bb.shape == (2,2):
                cv2.rectangle(display_image, tuple(bb[0]), tuple(bb[1]), (0,255,0), 3)

        # --- Smart upscaling for small images ---
        screen_rect = QtWidgets.QApplication.desktop().screenGeometry()
        print(display_image.shape, screen_rect.width(), screen_rect.height())
        target_w = max(display_image.shape[1], int(screen_rect.width() * 2.5))
        target_h = max(display_image.shape[0], int(screen_rect.height() * 2.5))

        scale_factor = 1.0
        h, w = display_image.shape[:2]
        if h < target_h or w < target_w:
            scale_factor = min(target_w / w, target_h / h)
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            display_for_box = cv2.resize(display_image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            display_for_box = display_image

        # --- Qt display for user feedback ---
        qimage = QImage(display_for_box.astype(np.uint8), display_for_box.shape[1], display_for_box.shape[0], display_for_box.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(self.ImageScreen.width(), self.ImageScreen.height(), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.ImageScreen.setPixmap(scaled_pixmap)

        # --- Draw box using your custom tool ---
        bb_pts, _ = ct.tools.Draw().bounding_box(display_for_box[:, :, ::-1], self.image_size)

        if bb_pts is None or len(bb_pts) < 2 or len(bb_pts) % 2 != 0:
            print("No complete boxes drawn, nothing to add.")
            return

        # Group every two points into a box
        boxes = [np.array([bb_pts[i], bb_pts[i + 1]], dtype=np.float32) for i in range(0, len(bb_pts), 2)]
        h, w = self.original_image.shape[:2]
        snap_margin = max(2, int(0.01 * min(h, w)))

        for box in boxes:
            # Scale down if image was upscaled for display
            box = box / scale_factor
            if snap:
                box = snap_box_points(box, self.original_image.shape, margin=snap_margin)
            box = box.astype(np.int32)
            if box.shape == (2, 2):
                # Compute crop size
                xmin, ymin = box[0]
                xmax, ymax = box[1]
                width = abs(xmax - xmin)
                height = abs(ymax - ymin)
                if width < min_crop_size or height < min_crop_size:
                    from PyQt5.QtWidgets import QMessageBox
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText(
                        f"Box too small!\nWidth: {width}, Height: {height}\n"
                        f"Minimum size is {min_crop_size}."
                    )
                    msg.setWindowTitle("Box Too Small")
                    msg.exec_()
                    continue  # Skip adding this box or previewing it
                self.bb_pts_list.append(box)

        print("DRAW BOX: List after session:", self.bb_pts_list)
        self.update_green_preview()
            
    def update_green_preview(self):
        display_image = self.original_image.copy()
        # Draw saved boxes (blue)
        if 'annotations' in self.annotation and 'box' in self.annotation['annotations']:
            for box_k, box_data in self.annotation['annotations']['box'].items():
                bb = np.array(box_data['bounding_box'], dtype=np.int32)
                if len(bb) == 2:
                    cv2.rectangle(display_image, tuple(bb[0]), tuple(bb[1]), (0, 128, 255), 3)
        # Draw pending (green)
        for bb in self.bb_pts_list:
            if len(bb) == 2:
                cv2.rectangle(display_image, tuple(bb[0]), tuple(bb[1]), (0, 255, 0), 3)

        im = display_image.copy()
        qimage = QImage(im.astype(np.uint8), im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(self.ImageScreen.width(), self.ImageScreen.height(), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.ImageScreen.setPixmap(scaled_pixmap)

    def save_box(self):
        class_ = self.ClassSpinBox.value()
        if not hasattr(self, 'bb_pts_list') or len(self.bb_pts_list) == 0:
            print('No new boxes to save.')
            return

        # Ensure annotation structure
        if 'annotations' not in self.annotation:
            self.annotation['annotations'] = {}
        if 'box' not in self.annotation['annotations']:
            self.annotation['annotations']['box'] = {}

        box_dict = self.annotation['annotations']['box']
        existing_keys = [int(k) for k in box_dict.keys()] if box_dict else []
        next_idx = max(existing_keys) + 1 if existing_keys else 1

        # Save each pending box with a unique key
        for bb_pts in self.bb_pts_list:
            box_dict[str(next_idx)] = {
                'bounding_box': bb_pts.tolist(),
                'class': class_
            }
            next_idx += 1

        # Write to file
        from helpers.save_load_files import safe_write_json

        safe_write_json(self.ann_name, self.annotation)

        print(f'Saved {len(self.bb_pts_list)} box(es).')

        # Clear pending after save!
        self.bb_pts_list = []

        self.change_image()
        #self.update_green_preview()
            
    def clear_boxes(self):
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QListWidget, QPushButton, QHBoxLayout

        if not hasattr(self, "annotation") or not isinstance(self.annotation, dict):
            error("No annotation data loaded.")
            return
        
        # Load all boxes from your annotation data structure
        box_dict = self.annotation.get('annotations', {}).get('box', {})
        if not box_dict:
            error("No saved boxes to delete.")
            return

        keys = list(box_dict.keys())
        dlg = QDialog(self.MainWindow)
        dlg.setWindowTitle("Select Bounding Boxes to Delete")
        layout = QVBoxLayout(dlg)
        listwidget = QListWidget()
        listwidget.setSelectionMode(QListWidget.MultiSelection)

        for key in keys:
            bbox = box_dict[key]['bounding_box']
            # Nice display string: show the coordinates
            xs = [bbox[0][0], bbox[1][0]]
            ys = [bbox[0][1], bbox[1][1]]
            box_str = f"Box {key}"
            listwidget.addItem(box_str)

        layout.addWidget(listwidget)
        btns = QHBoxLayout()
        btn_ok = QPushButton("Delete Selected")
        btn_cancel = QPushButton("Cancel")
        btns.addWidget(btn_ok)
        btns.addWidget(btn_cancel)
        layout.addLayout(btns)

        btn_ok.clicked.connect(dlg.accept)
        btn_cancel.clicked.connect(dlg.reject)

        def highlight_selected_boxes():
            # Start with base image
            display = self.original_image.copy()
            for i, key in enumerate(keys):
                bbox = box_dict[key]['bounding_box']
                xmin, ymin = bbox[0]
                xmax, ymax = bbox[1]
                color = (0, 128, 255)  # blue for normal
                thickness = 3
                if listwidget.item(i).isSelected():
                    color = (255, 140, 0)  # orange for selected
                    thickness = 6
                cv2.rectangle(display, (xmin, ymin), (xmax, ymax), color, thickness)
            # Show it
            im = display.astype(np.uint8)
            qimage = QImage(im, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            scaled_pixmap = pixmap.scaled(self.ImageScreen.width(), self.ImageScreen.height(), Qt.KeepAspectRatio, Qt.FastTransformation)
            self.ImageScreen.setPixmap(scaled_pixmap)

        # Connect highlight update
        listwidget.itemSelectionChanged.connect(highlight_selected_boxes)
        highlight_selected_boxes()  # initial call

        # Dialog execution
        if dlg.exec_() == QDialog.Accepted:
            selected_indices = [i.row() for i in listwidget.selectedIndexes()]
            if not selected_indices:
                self.change_image()
                return

            for idx in sorted(selected_indices, reverse=True):
                del box_dict[keys[idx]]
            self.annotation['annotations']['box'] = box_dict

            try:
                self.save_annotation()
                print(f"Deleted {len(selected_indices)} bounding box(es).")
            except Exception as e:
                error(f"[clear_boxes] Failed to save annotation: {e}")
            self.change_image()
        else:
            self.change_image()

    def annotation_full_screen(self):
        try:
            self.image_size = self.select_image_size.value()
            _,_ = ct.tools.Draw().bounding_box(self.image[:,:,::-1],self.image_size)
        except Exception as e:
            error(e)
    
    def select_middle_point(self):
        try :
            self.image_size = self.select_image_size.value()
            downsample_factor = self.downsample_factor_box.value()
            mid_pt = ct.tools.Draw().points(self.image[:,:,::-1],self.image_size,move_x = 0,move_y = 0)
            self.mid_pt = (int(mid_pt[0][0]/downsample_factor),int(mid_pt[0][1]/downsample_factor))
            self.middpoint_update_button.setStyleSheet("background-color : lightblue")
        except Exception as e:
            error(e)
            self.middpoint_update_button.setStyleSheet("background-color : red")

    def update_midpoint_image(self):
        try:
            self.image_size = self.select_image_size.value()
            downsample_factor = self.downsample_factor_box.value()
            color_channel = [0 if self.color_chenel_box.currentText()=='R' else 1 if self.color_chenel_box.currentText()=='B' else 2]
            d = int(self.wavelet_window_size_box.value()/2)
            if self.mid_pt == []:
                self.select_middle_point()
            try :
                mid_image = self.image_down[int(self.mid_pt[1])-d:int(self.mid_pt[1])+d,
                                int(self.mid_pt[0])-d:int(self.mid_pt[0])+d]
            except Exception as e:
                black_crack = [-1 if self.crack_color_box.currentText() =='Bright crack' else 1 ][0]
                if black_crack==1:
                    func = np.min
                elif  black_crack==-1:
                    func = np.max
                self.image_down = skimage.measure.block_reduce(self.original_image, block_size=(downsample_factor, downsample_factor, 1),
                                                    func=func, cval=0, func_kwargs=None)
                mid_image = self.image_down[int(self.mid_pt[1])-d:int(self.mid_pt[1])+d,
                                int(self.mid_pt[0])-d:int(self.mid_pt[0])+d]
            mid_image = mid_image[:,:,color_channel]
            qimage = QImage(mid_image.astype(dtype=np.uint8), mid_image.shape[1], mid_image.shape[0], 
                            mid_image.strides[0], QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimage)
            scaled_pixmap = pixmap.scaled(self.middlepoint_display.width(), self.middlepoint_display.height(), Qt.KeepAspectRatio, Qt.FastTransformation)
            self.middlepoint_display.setPixmap(scaled_pixmap)
        except Exception as e:
            error(e)
      
    def _draw_crack(self, im, crack,
                color_mask=(0,1,1),
                color_midline=(0,0,255),
                color_edges=(255,255,0),
                color_points=(255,0,0)):

        import numpy as np, cv2
        from skimage.segmentation import mark_boundaries

        def _draw_polyline(im, pts, color, thickness=2):
            """Draw a polyline where NaN rows mark gaps."""
            pts = np.asarray(pts, dtype=float)
            if pts.ndim != 2 or pts.shape[1] != 2:
                return im
            valid = ~np.isnan(pts).any(axis=1)
            if not valid.any():
                return im
            idx = np.where(~valid)[0]
            splits = np.split(np.arange(len(pts)), idx)
            for seg in splits:
                seg = [i for i in seg if valid[i]]
                if len(seg) > 1:
                    for i in range(1, len(seg)):
                        x1, y1 = pts[seg[i-1]]
                        x2, y2 = pts[seg[i]]
                        cv2.line(im,
                                (int(round(x1)), int(round(y1))),
                                (int(round(x2)), int(round(y2))),
                                color, thickness)
            return im

        """
        Draw mask, midline, edges, and endpoints for a single crack dict.
        Returns the updated image and mask array (if any).
        """
        H, W = im.shape[:2]
        mask_out = None

        # --- Mask ---
        mask_full = reconstruct_full_mask_from_crack(crack, H, W)
        if np.any(mask_full):
            im = (mark_boundaries(im/255.0, (mask_full>0).astype(np.uint8),
                                color=color_mask, background_label=0)*255).astype(np.uint8)
            mask_out = mask_full

        # --- Geodesic edges ---
        edges = crack.get("geodesic_edges", {}) or {}
        edge_list = []

        # Normalize to a flat list of arrays
        if isinstance(edges, dict):
            edge_list = list(edges.values())
        elif isinstance(edges, (list, tuple)):
            for e in edges:
                if isinstance(e, dict):
                    edge_list.extend(list(e.values()))
                elif isinstance(e, (list, np.ndarray)):
                    edge_list.append(e)

        print(f"[DRAW_DBG] crack {crack.get('id', '?')} has {len(edge_list)} geodesic edges")
        for e in edge_list:
            e = np.asarray(e, dtype=float)
            if e.ndim == 2 and e.shape[1] == 2 and len(e) > 1:
                im = _draw_polyline(im, e, color_edges, 2)

        # --- Midline ---
        midline = np.array(crack.get("midline", []), dtype=float)
        if len(midline) > 1:
            im = _draw_polyline(im, midline, color_midline, 2)

        # --- Endpoints (user_points or all_user_points) ---
        up = crack.get("user_points") or crack.get("all_user_points") or []
        for p in up:
            if p is None or len(p) < 2:
                continue
            if not np.isfinite(p[0]) or not np.isfinite(p[1]):
                continue
            x, y = int(round(p[0])), int(round(p[1]))
            if 0 <= x < W and 0 <= y < H:
                endpoint_radius = max(3, int(min(H, W) * 0.0035))
                cv2.circle(im, (x, y), endpoint_radius, color_points, -1)

        return im, mask_out
    
    def _reset_edit_state(self):
        """Hard reset of all per-image editable UI state."""
        # editable endpoints/links
        self.points = []
        self.connections = []

        # read-only overlays (drawn from JSON)
        self.readonly_connections = []
        self.readonly_midlines = {}
        self.midlines = {}
        self.endpoint_pairs = None
        # live drawing modes
        self.polyline = []
        self.polyline_mode = False
        self.connection_mode = False
        self.connecting_index = None

        # hover/selection indices
        self.hover_index = None
        self.hover_line_index = None
        self._hover_midline_key = None
        
        self.user_points = None
        self.user_connections = None

        # selection / bbox helpers
        self.current_crack_id = None
        self.bb_pts_list = []
        # If you cache rectangles separately:
        if hasattr(self, "boxes"):
            self.boxes = []

        # if any tools keep their own state, clear them too
        if hasattr(self, "tool_state") and hasattr(self.tool_state, "clear"):
            try:
                self.tool_state.clear()
            except Exception:
                pass

    def change_image(self):
        import os, json, cv2
        import numpy as np

        # ------------------------------------------------------------
        # A) ID NORMALIZER (LOCAL, SAFE, ONLY TOUCHES LOADED DATA)
        # ------------------------------------------------------------
        def _normalize_ann_ids(ann_root):
            """
            Normalize atomic + combined IDs and member lists so that:
            - all atomic_cracks keys are strings
            - all combined_cracks keys are strings
            - all combined members are strings
            This prevents KeyError('1') vs KeyError(1).
            """
            if not isinstance(ann_root, dict):
                return

            atomic = ann_root.setdefault("atomic_cracks", {})
            combined = ann_root.setdefault("combined_cracks", {})

            # normalize atomic keys
            new_atomic = {}
            for k, v in atomic.items():
                sk = str(k)
                new_atomic[sk] = v
            ann_root["atomic_cracks"] = new_atomic

            # normalize combined keys + members
            new_combined = {}
            for k, cmb in combined.items():
                sk = str(k)
                if isinstance(cmb, dict) and "members" in cmb:
                    cmb["members"] = [str(m) for m in cmb["members"]]
                new_combined[sk] = cmb
            ann_root["combined_cracks"] = new_combined

        # ------------------------------------------------------------

        if not hasattr(self, "image_names") or not self.image_names:
            error("No images loaded. Please load images before using change_image().")
            return

        # --- HARD RESET of transient interactive state (fixes ghost points/boxes) ---
        self._reset_edit_state()

        # ---------------------------------------------------------------------------
        self.update_selected_item(os.path.basename(self.image_names[self.n]))
        self.name = self.image_names[self.n]
        self.image = cv2.imread(self.name)[:, :, ::-1].astype(np.uint8)
        self.original_image = self.image.copy()
        self.filename_label_2.setText(os.path.basename(self.name))
        base_name = os.path.splitext(os.path.basename(self.name))[0]

        # ---- MASK LOADING (optional external masks) ----
        self.current_mask = None
        if getattr(self, "use_masks", False) and hasattr(self, "mask_map"):
            mask_path = self.mask_map.get(base_name)
            print(f"[DEBUG change_image] mask_path for {base_name}: {mask_path}")
            if mask_path:
                if mask_path.endswith('.npy'):
                    mask = np.load(mask_path)
                    mask = (mask > 0).astype(np.uint8) if mask.max() > 1 else mask.astype(np.uint8)
                else:
                    mask = cv2.imread(mask_path, 0)
                    mask = (mask > 0).astype(np.uint8) if mask is not None else None
                self.current_mask = mask
            else:
                print(f"[DEBUG change_image] No mask found for {base_name}")

        im = self.original_image.copy()
        H, W = im.shape[:2]

        # -------- Load annotation data --------
        self.ann_name = os.path.join(self.save_folder, base_name + '.json')
        self.mask_name_bin = os.path.join(self.save_folder, base_name + '_mask.png')
        self.mask_name_255 = os.path.join(self.save_folder, base_name + '_mask255.png')
        self.mask = []
        self.annotation = {}

        if os.path.exists(self.ann_name):
            with open(self.ann_name, encoding="utf-8") as f:
                self.annotation = json.load(f)

            ann = self.annotation.setdefault('annotations', {})

            # ------------------------------------------------------------
            # B) *** APPLY ID NORMALIZATION RIGHT AFTER LOADING ***
            # ------------------------------------------------------------
            _normalize_ann_ids(ann)
            # ------------------------------------------------------------

            atomic = ann.get("atomic_cracks", {}) or {}
            combined = ann.get("combined_cracks", {}) or {}

            # ==== Build read-only midline overlays with tags/colors ====
            self.readonly_midlines = {}
            for cid, crack in atomic.items():
                src = str(crack.get("source") or crack.get("src") or "").lower()
                mid = crack.get("midline", [])
                if isinstance(mid, list) and len(mid) >= 2:
                    has_auto = bool(crack.get("variants", {}).get("auto", {}))
                    tag = "unprocessed" if src.startswith("manual") and not has_auto else "manual"
                    color = (255, 165, 0) if tag == "unprocessed" else (0, 200, 255)
                    self.readonly_midlines[f"manual_{cid}"] = {"poly": mid, "color": color, "tag": tag}

                # auto variants
                vroot = crack.get("variants", {}).get("auto", {})
                for ck, pack in vroot.items():
                    bid = pack.get("best_variant_id")
                    if bid is None:
                        continue
                    best = pack["variants"].get(f"v{bid}", {})
                    if "midline" in best and len(best["midline"]) >= 2:
                        self.readonly_midlines[f"auto_{cid}"] = {
                            "poly": best["midline"], "color": (0, 255, 0), "tag": "auto"
                        }
                        break

            # ---- Bounding boxes (optional) ----
            if 'box' in ann:
                for key, box_data in ann['box'].items():
                    bb_pts = np.array(box_data['bounding_box'])
                    if box_data['class'] == 0: box_color = (0,0,255)
                    elif box_data['class'] == 1: box_color = (0,255,0)
                    else: box_color = (255,0,0)
                    cv2.rectangle(im, tuple(bb_pts[0]), tuple(bb_pts[1]), box_color, 3)

            drawn_atomic = set()

            # ---- Combined cracks ----
            for crack_id, crack in combined.items():
                for m in crack.get("members", []):
                    drawn_atomic.add(m)

                geodesic_edges = crack.get("geodesic_edges", [])
                if isinstance(geodesic_edges, dict):
                    geodesic_edges = list(geodesic_edges.values())

                flat = []
                for e in geodesic_edges:
                    if isinstance(e, dict): flat.extend(list(e.values()))
                    elif isinstance(e, (list, tuple)) and len(e)>0: flat.append(e)
                crack["geodesic_edges"] = [np.array(x, dtype=float) for x in flat if len(x)>=2]

                normal_edges = crack.get("normal_edge_points", [])
                if isinstance(normal_edges, dict):
                    normal_edges = list(normal_edges.values())
                elif (isinstance(normal_edges, list) and len(normal_edges)==2 and
                    isinstance(normal_edges[0], (list,tuple))):
                    normal_edges = [np.array(n, float) for n in normal_edges]
                crack["normal_edge_points"] = normal_edges

                im, _ = self._draw_crack(
                    im, crack,
                    color_mask=(0,0,0),
                    color_midline=(0,0,255),
                    color_edges=(255,255,0),
                    color_points=(255,0,0)
                )

            # ---- Atomic cracks ----
            for crack_id, crack in atomic.items():
                if any(crack_id in c.get("members", []) for c in combined.values()):
                    continue
                im, mask_full = self._draw_crack(im, crack)
                if mask_full is not None:
                    self.mask.append(mask_full)

        # ---- Render to main display ----
        _, pixmap = numpy_to_qimage_and_scaled_pixmap(
            im.astype(np.uint8),
            self.ImageScreen.width(),
            self.ImageScreen.height(),
            is_gray=False
        )
        self.ImageScreen.setPixmap(pixmap)

        # ---- Update all-segments preview ----
        ann = self.annotation.get("annotations", {})
        atomic = ann.get("atomic_cracks", {}) or {}
        combined = ann.get("combined_cracks", {}) or {}

        full_mask_display = build_combined_mask(atomic, H, W)
        for crack in combined.values():
            full_mask_display |= reconstruct_full_mask_from_crack(crack, H, W)
        full_mask_display[full_mask_display > 0] = 1

        _, pixmap_mask = numpy_to_qimage_and_scaled_pixmap(
            (full_mask_display*255).astype(np.uint8),
            self.all_segments_display.width(),
            self.all_segments_display.height(),
            is_gray=True
        )
        self.all_segments_display.setPixmap(pixmap_mask)

    '''def _build_combined_crack(self, member_ids, pad=10):
        """
        Combined crack builder (user-endpoint chaining + dominant trimming):
        - chain polylines by explicit user endpoints (user_points/user_connections)
        - pick longest chain as dominant, carve others by its buffer
        - keep only true outside-branch remnants (endpoint+midpoint outside, len guard)
        - compute geodesic edges + normals per kept piece
        - build mask (x,y order), save debug plots, and return combined summary
        """
        import numpy as np, cv2, os
        import matplotlib.pyplot as plt
        from shapely.geometry import LineString, MultiLineString
        from shapely.ops import unary_union

        ann = self.annotation.setdefault("annotations", {})
        atomic = ann.setdefault("atomic_cracks", {})

        H, W = self.original_image.shape[:2]

        # ---------------- helpers ----------------
        def full_mask_from_atomic(crack):
            mc = crack.get("mask_crop"); bb = crack.get("mask_bbox")
            if mc is not None and bb is not None:
                crop = np.array(mc, dtype=np.uint8)
                x, y, w, h = map(int, bb)
                x2, y2 = min(x+w, W), min(y+h, H)
                w_eff, h_eff = max(0, x2-x), max(0, y2-y)
                if h_eff > 0 and w_eff > 0:
                    crop = (crop > 0).astype(np.uint8)[:h_eff, :w_eff]
                    m = np.zeros((H, W), dtype=np.uint8)
                    m[y:y+h_eff, x:x+w_eff] = crop
                    return m
            return np.zeros((H, W), dtype=np.uint8)

        def ls_coords(ls: LineString):
            return np.asarray(ls.coords, dtype=float)

        def split_lines(geom):
            if geom.is_empty: return []
            if isinstance(geom, LineString): return [geom]
            if isinstance(geom, MultiLineString): return list(geom.geoms)
            return []

        def shoelace_area(xs, ys):
            return 0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))

        def ribbon_mask_from_midline(S_xy, thickness_px=4):
            mask = np.zeros((H, W), dtype=np.uint8)
            pts = np.round(S_xy).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(mask, [pts], isClosed=False, color=255,
                        thickness=thickness_px, lineType=cv2.LINE_AA)
            return mask

        def align_edge_to_midline_direction(S_xy, E_xy):
            d_f = np.linalg.norm(E_xy[0]-S_xy[0]) + np.linalg.norm(E_xy[-1]-S_xy[-1])
            d_r = np.linalg.norm(E_xy[0]-S_xy[-1]) + np.linalg.norm(E_xy[-1]-S_xy[0])
            return (E_xy[::-1] if d_r < d_f else E_xy)

        def linestring_length(arr):
            try:
                return float(LineString(arr).length)
            except Exception:
                return 0.0

        def finite_xy(arr):
            if arr is None or len(arr) == 0: return np.empty((0,2), float)
            a = np.asarray(arr, float)
            ok = np.isfinite(a).all(axis=1)
            a = a[ok]
            if len(a) <= 1: return a
            keep = [0]
            for i in range(1, len(a)):
                if not (abs(a[i,0]-a[i-1,0]) < 1e-9 and abs(a[i,1]-a[i-1,1]) < 1e-9):
                    keep.append(i)
            return a[keep]

        def split_on_teleports(arr, max_step=50.0):
            arr = np.asarray(arr, float)
            if len(arr) < 2: return []
            d = np.sqrt(np.sum(np.diff(arr, axis=0)**2, axis=1))
            breaks = np.where(d > max_step)[0]
            segs, start = [], 0
            for b in breaks:
                if b+1 - start >= 2:
                    segs.append(arr[start:b+1])
                start = b+1
            if len(arr) - start >= 2:
                segs.append(arr[start:])
            return segs if segs else [arr]

        # --- endpoint-aware stitching ---
        def _endpoints_from_crack(crack):
            ups = crack.get("user_points", []) or []
            ucs = crack.get("user_connections", []) or []
            ends = set()
            for conn in ucs:
                for idx in conn:
                    if 0 <= idx < len(ups):
                        pt = ups[idx]
                        ends.add((float(pt[0]), float(pt[1])))
            return ends

        def stitch_lines_by_user(member_ids, atomic):
            mid2arr, mid2ends = {}, {}
            for mid in member_ids:
                crack = atomic.get(mid)
                if not crack: continue
                ml = crack.get("midline", []) or []
                if len(ml) < 2: continue
                arr = np.array([[float(x), float(y)] for (x,y) in ml], dtype=float)
                mid2arr[mid] = arr
                mid2ends[mid] = _endpoints_from_crack(crack)
            if not mid2arr: return []

            end_to_mids = {}
            for mid, ends in mid2ends.items():
                for e in ends:
                    end_to_mids.setdefault(e, set()).add(mid)

            adj = {mid: set() for mid in mid2arr}
            for mids in end_to_mids.values():
                mids = list(mids)
                for i in range(len(mids)):
                    for j in range(i+1, len(mids)):
                        adj[mids[i]].add(mids[j])
                        adj[mids[j]].add(mids[i])

            comps, seen = [], set()
            for mid in adj:
                if mid in seen: continue
                stack, comp = [mid], []
                seen.add(mid)
                while stack:
                    u = stack.pop()
                    comp.append(u)
                    for v in adj[u]:
                        if v not in seen:
                            seen.add(v); stack.append(v)
                comps.append(comp)

            stitched = []
            for comp in comps:
                comp_sorted = sorted(comp, key=lambda m: linestring_length(mid2arr[m]), reverse=True)
                used = set()
                if comp_sorted:
                    cur = mid2arr[comp_sorted[0]].copy()
                    used.add(comp_sorted[0])
                    extended = True
                    while extended:
                        extended = False
                        end_pt = tuple(cur[-1])
                        for m in comp_sorted:
                            if m in used: continue
                            arr2 = mid2arr[m]
                            if tuple(arr2[0]) == end_pt:
                                cur = np.vstack([cur, arr2[1:]])
                                used.add(m); extended = True; break
                            elif tuple(arr2[-1]) == end_pt:
                                cur = np.vstack([cur, arr2[-2::-1]])
                                used.add(m); extended = True; break
                    stitched.append(cur)
                for m in comp_sorted:
                    if m not in used:
                        stitched.append(mid2arr[m])
            return [finite_xy(s) for s in stitched if len(s) >= 2]

        # ---------------- collect ----------------
        union_mask_existing = np.zeros((H, W), dtype=np.uint8)
        for mid in member_ids:
            crack = atomic.get(mid)
            if not crack: continue
            union_mask_existing |= full_mask_from_atomic(crack)

        try:
            w_half = int(self.window_half_size_box.value())
        except Exception:
            w_half = 15
        prune_radius = max(3, int(w_half * 0.5))
        overlap_px   = max(6, int(w_half * 0.6))
        min_keep_len = max(8.0, 0.6 * w_half)
        max_plot_jump = max(25.0, 1.2 * w_half)

        stitched = stitch_lines_by_user(member_ids, atomic)
        stitched.sort(key=linestring_length, reverse=True)

        kept_segs, dom_buffer = [], None
        for S in stitched:
            g = LineString(S)
            if dom_buffer is None:
                kept_segs.append(S)
                dom_buffer = g.buffer(overlap_px, cap_style=2, join_style=2)
            else:
                remainder = g.difference(dom_buffer)
                if remainder.is_empty: continue
                for piece in split_lines(remainder):
                    if piece.length >= min_keep_len:
                        kept_segs.append(ls_coords(piece))
                dom_buffer = unary_union([dom_buffer, g.buffer(overlap_px, cap_style=2, join_style=2)])
        segs = kept_segs if kept_segs else stitched

        # ---------------- tracking params ----------------
        color_idx = 0 if self.edge_track_color_box.currentText() == 'R' else \
                    1 if self.edge_track_color_box.currentText() == 'B' else 2
        mu = self.mu_box.value(); l = self.l_box.value(); p = self.p_box.value()

        edge1_segs, edge2_segs = [], []
        norm1_segs, norm2_segs = [], []
        union_mask = np.zeros((H, W), dtype=np.uint8)
        widths_all = []

        # ---------------- per segment ----------------
        for S in segs:
            if S is None or len(S) < 2: continue
            x0 = max(0, int(np.floor(S[:,0].min()) - pad))
            x1 = min(W, int(np.ceil(S[:,0].max()) + pad))
            y0 = max(0, int(np.floor(S[:,1].min()) - pad))
            y1 = min(H, int(np.ceil(S[:,1].max()) + pad))
            if x1-x0 < 2 or y1-y0 < 2: continue

            self.active_bbox = [x0, y0, x1, y1]
            self.pts = [np.array([S[0,0], S[0,1]]), np.array([S[-1,0], S[-1,1]])]
            self.end_points = self.pts
            self.update_image_crop()
            if getattr(self, "skip_current_segment", False): continue

            cx, cy = S[:,0] - x0, S[:,1] - y0
            self.track = np.vstack([cy, cx])
            self.current_source = "manual_poly"
            self.pts_crop = [np.array(self.pts[0]) - np.array([x0, y0]),
                            np.array(self.pts[1]) - np.array([x0, y0])]
            down = self.downsample_factor_box.value()
            self.pts_crop_down = [p / down for p in self.pts_crop]
            self.edge_mask()

            midline_xy_crop = np.column_stack([self.adjusted_track[1], self.adjusted_track[0]])

            res = ct.segmentation.edges_tracking(
                self.image_crop[:, :, color_idx],
                self.pts_crop,
                self.edge_mask1_crop, self.edge_mask2_crop,
                midline=midline_xy_crop, mu=mu, l=l, p=p,
                return_normal_edges=True
            )

            track_e1, track_e2 = res["geodesic_edges"]
            if track_e1 is None or track_e2 is None or len(track_e1)<2 or len(track_e2)<2: continue

            e1_full = finite_xy(np.column_stack([track_e1[:,0]+x0, track_e1[:,1]+y0]))
            e2_full = finite_xy(np.column_stack([track_e2[:,0]+x0, track_e2[:,1]+y0]))
            if len(e1_full)<2 or len(e2_full)<2: continue

            e1_full = align_edge_to_midline_direction(S, e1_full)
            e2_full = align_edge_to_midline_direction(S, e2_full)

            normals = res.get("normal_edge_points")
            if normals is not None:
                (e1x, e1y), (e2x, e2y) = normals
                n1_full = finite_xy(np.column_stack([e1x + x0, e1y + y0]))
                n2_full = finite_xy(np.column_stack([e2x + x0, e2y + y0]))
                m = min(len(n1_full), len(n2_full))
                if m >= 2:
                    d = np.sqrt(np.sum((n1_full[:m] - n2_full[:m])**2, axis=1))
                    if d.size: widths_all.append(d[np.isfinite(d)])
            else:
                n1_full = np.empty((0,2)); n2_full = np.empty((0,2))

            edge1_segs.append(e1_full)
            edge2_segs.append(e2_full)
            norm1_segs.append(n1_full)
            norm2_segs.append(n2_full)

            ex = np.concatenate((e1_full[:,0][::-1], e2_full[:,0]))
            ey = np.concatenate((e1_full[:,1][::-1], e2_full[:,1]))
            exc, eyc = np.clip(ex, 0, W-1), np.clip(ey, 0, H-1)
            area = shoelace_area(exc, eyc)
            if area > 0.5:
                mask_seg = ct.segmentation.create_mask(self.original_image, exc, eyc).astype(np.uint8)
            else:
                mask_seg = ribbon_mask_from_midline(S, thickness_px=max(3, prune_radius//2))
            union_mask |= (mask_seg > 0).astype(np.uint8)

        # ---------------- final crop ----------------
        if np.any(union_mask):
            ys, xs = np.where(union_mask>0)
            Y0, Y1 = int(ys.min()), int(ys.max()+1)
            X0, X1 = int(xs.min()), int(xs.max()+1)
            crop = union_mask[Y0:Y1, X0:X1].astype(np.uint8)
            h, w = crop.shape
        else:
            X0=Y0=0; w=h=1
            crop = np.zeros((h,w), np.uint8)
              
                # ---------------- DEBUG PLOT ----------------        
        save_dir = os.path.join(self.save_folder, "debug_outputs")
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(self.name))[0]
        member_str = "_".join(sorted(member_ids, key=lambda s: int(s)))
        fname = os.path.join(save_dir, f"{base_name}_combined_debug.png")

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(self.original_image)
        ax.set_title("All cracks (atomic + combined) with current merge highlighted")

        ann = self.annotation.get("annotations", {})
        atomic = ann.get("atomic_cracks", {})
        combined = ann.get("combined_cracks", {})

        combined_members = {m for cmb in combined.values() for m in cmb.get("members", [])}

        # --- fallback normal generator for cracks without stored normals ---
        def _fallback_normals(ml, step=50, length=6):
            from numpy.linalg import norm
            pts = np.array(ml, float)
            if len(pts) < 3:
                return []
            tang = np.gradient(pts, axis=0)
            tang /= np.maximum(1e-9, np.linalg.norm(tang, axis=1)[:, None])
            # rotate 90 deg CCW
            normvec = np.column_stack([-tang[:, 1], tang[:, 0]])
            lines = []
            for i in range(0, len(pts), step):
                p = pts[i]
                n = normvec[i]
                lines.append(([p[0]-length*n[0], p[0]+length*n[0]],
                              [p[1]-length*n[1], p[1]+length*n[1]]))
            return lines

        def plot_crack(crack, color_idx=0):
            ml = crack.get("midline", []) or crack.get("midline_segments", [])
            if not ml:
                return
            if isinstance(ml[0][0], list):  # segments
                segs_to_plot = [np.array(seg, float) for seg in ml if seg]
            else:  # flat midline
                segs_to_plot = [np.array(ml, float)]
            for S in segs_to_plot:
                if len(S) < 2: continue
                for segp in split_on_teleports(S, max_step=max_plot_jump):
                    ax.plot(segp[:,0], segp[:,1], 'g-', lw=0.6)

            edges = crack.get("geodesic_edges", {})
            for key, arr in edges.items():
                arr = np.array(arr, float)
                if arr.ndim == 2 and len(arr) >= 2:
                    for segp in split_on_teleports(arr, max_step=max_plot_jump):
                        ax.plot(segp[:,0], segp[:,1], 'r-' if "edge1" in key else 'b-', lw=0.4)

            normals = crack.get("normal_edge_points", {})
            if normals:
                n1 = normals.get("edge1", [])
                n2 = normals.get("edge2", [])
                # Handle JSON-style [[xlist],[ylist]] format
                if isinstance(n1, list) and len(n1) == 2 and isinstance(n1[0], (list, tuple)):
                    n1 = np.column_stack([n1[0], n1[1]])
                else:
                    n1 = np.array(n1, float)
                if isinstance(n2, list) and len(n2) == 2 and isinstance(n2[0], (list, tuple)):
                    n2 = np.column_stack([n2[0], n2[1]])
                else:
                    n2 = np.array(n2, float)

                if n1.ndim == 2 and n2.ndim == 2 and len(n1) and len(n2):
                    step = max(1, min(len(n1), len(n2)) // 70)
                    for i in range(0, min(len(n1), len(n2)), step):
                        if np.isfinite(n1[i]).all() and np.isfinite(n2[i]).all():
                            ax.plot([n1[i,0], n2[i,0]],
                                    [n1[i,1], n2[i,1]],
                                    color='cyan', lw=0.3, alpha=0.5)

        # --- plot all existing cracks for context ---
        # Other combined cracks (not this one)
        for cid, cmb in combined.items():
            if set(cmb.get("members", [])) == set(member_ids):
                continue
            plot_crack(cmb)

        # Atomic cracks that are NOT in *any* combined (skip ones in member_ids or already absorbed)
        for aid, crack in atomic.items():
            if aid in member_ids or aid in combined_members:
                continue
            plot_crack(crack)

        # --- plot the new combined crack being built ---
        for S in segs:
            for segp in split_on_teleports(S, max_step=max_plot_jump):
                ax.plot(segp[:,0], segp[:,1], 'g-', lw=.8)
        for e in edge1_segs:
            for segp in split_on_teleports(e, max_step=max_plot_jump):
                ax.plot(segp[:,0], segp[:,1], 'r-', lw=.6)
        for e in edge2_segs:
            for segp in split_on_teleports(e, max_step=max_plot_jump):
                ax.plot(segp[:,0], segp[:,1], 'b-', lw=.6)

        # Plot normals for the *current combined*
        for n1, n2 in zip(norm1_segs, norm2_segs):
            if len(n1) == 0 or len(n2) == 0:
                continue
            step = 50
            for i in range(0, min(len(n1), len(n2)), step):
                if np.isfinite(n1[i]).all() and np.isfinite(n2[i]).all():
                    ax.plot([n1[i,0], n2[i,0]], [n1[i,1], n2[i,1]],
                            color='cyan', lw=0.4, alpha=0.8)

        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis('equal')
        plt.tight_layout()
        plt.savefig(fname, dpi=250)
        plt.close()

        def _flatten(seg_list):
            out=[]
            for i, arr in enumerate(seg_list):
                out.extend([[float(x), float(y)] for x,y in arr])
                if i<len(seg_list)-1: out.append([None,None])
            return out

        combined_length = float(sum(linestring_length(s) for s in segs))
        mean_width = float(np.nanmean(np.concatenate(widths_all))) if widths_all else None
        
        # ---- collect derived endpoints/connections from members ----
        derived_points = []
        derived_conns = []
        for mid in member_ids:
            crack = atomic.get(mid)
            if not crack:
                continue
            ups = crack.get("user_points", []) or []
            ucs = crack.get("user_connections", []) or []
            # offset conn indices so they stay unique across members
            base = len(derived_points)
            derived_points.extend(ups)
            derived_conns.extend([[base+idx for idx in conn] for conn in ucs])
        print(derived_points, derived_conns)

        return {
            "source": "combined",
            "members": sorted(member_ids, key=lambda s: int(s)),
            "midline_segments": [ [[float(x), float(y)] for (x,y) in s] for s in segs ],
            "midline": _flatten(segs),
            "geodesic_edges": {"edge1": _flatten(edge1_segs), "edge2": _flatten(edge2_segs)},
            "normal_edge_points": {"edge1": _flatten(norm1_segs), "edge2": _flatten(norm2_segs)},
            "mask_crop": crop.tolist(),
            "mask_bbox": [int(X0), int(Y0), int(w), int(h)],
            "combined_length": combined_length,
            "mean_width": mean_width,
            # --- new derived fields ---
            "all_user_points": derived_points,
            "all_user_connections": derived_conns,
        }'''
        
    def _build_combined_crack(self, member_ids, pad=10):
        """
        GUI-safe wrapper that delegates computation to the stateless combiner,
        and delegates DEBUG plotting to the pure helper plot function.
        """

        from combiner import build_combined_crack_stateless, plot_combined_debug

        atomic = self.annotation["annotations"]["atomic_cracks"]

        # ---- Safe parameter getters ----
        def safe_val(name, default):
            box = getattr(self, name, None)
            if box is None:
                return default
            try:
                return box.value()
            except Exception:
                return default

        window_half = safe_val("window_half_size_box", 15)
        mu          = safe_val("mu_box", 0.0)
        l           = safe_val("l_box", 5)
        p           = safe_val("p_box", 14)

        # color channel
        try:
            mode = self.edge_track_color_box.currentText()
            color_idx = 0 if mode == 'R' else (1 if mode == 'B' else 2)
        except Exception:
            color_idx = 0

        # ----------------------------
        # DEBUG CALLBACK (new, correct signature)
        # ----------------------------
        def _debug_cb(
            *,
            image_rgb,
            segs,
            edge1_segs,
            edge2_segs,
            norm1_segs,
            norm2_segs,
            mask_bbox,
            member_ids,
            union_mask   # REQUIRED
        ):
            """
            Forward the stateless combined-crack output to the pure plotting helper.
            Saves into GUI's debug_outputs folder.
            """
            import os

            out_dir = os.path.join(self.save_folder, "debug_outputs")
            os.makedirs(out_dir, exist_ok=True)

            plot_combined_debug(
                original_image=image_rgb,
                segs=segs,
                edge1_segs=edge1_segs,
                edge2_segs=edge2_segs,
                norm1_segs=norm1_segs,
                norm2_segs=norm2_segs,
                mask_bbox=mask_bbox,
                member_ids=member_ids,
                out_dir=out_dir
                # NOTE: union_mask is accepted here (required by signature)
                # but not used by plot_combined_debug(), which is fine.
            )

        # ===== CALL STATELESS BUILDER =====
        result = build_combined_crack_stateless(
            original_image=self.original_image,
            authoring_atomic=atomic,
            member_ids=member_ids,
            window_half_size=window_half,
            mu=mu, l=l, p=p,
            color_channel=color_idx,
            pad=pad,
            prefer_gpu=True,
            debug_callback=_debug_cb,
        )

        return result
    
    def _combined_debug_plot(
            self,
            image_rgb,
            segs,
            edge1_segs,
            edge2_segs,
            norm1_segs,
            norm2_segs,
            mask_bbox,
            member_ids,
        ):
        """
        High-res combined-crack debug plot.
        Zooms to mask_bbox, draws midline (green), edges (red/blue), normals (cyan).
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        # -------------------------------
        # 1. Extract bounding box (from stateless builder)
        # -------------------------------
        x0, y0, w, h = mask_bbox
        x1 = x0 + w
        y1 = y0 + h

        # pad around box
        pad = 40
        H, W = self.original_image.shape[:2]

        x0p = max(0, x0 - pad)
        x1p = min(W, x1 + pad)
        y0p = max(0, y0 - pad)
        y1p = min(H, y1 + pad)

        # -------------------------------
        # 2. Crop from high-res original, not zoomed version
        # -------------------------------
        crop_rgb = self.original_image[y0p:y1p, x0p:x1p, :]
        crop_rgb = crop_rgb[:, :, ::-1]  # BGR→RGB

        # -------------------------------
        # 3. Prepare output path
        # -------------------------------
        save_dir = os.path.join(self.save_folder, "debug_outputs")
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(self.name))[0]
        fname = os.path.join(save_dir, f"{base}_combined_{'_'.join(member_ids)}.png")

        # -------------------------------
        # 4. Plot
        # -------------------------------
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(crop_rgb)

        # helper: split long polylines by jumps
        def split(arr, max_step=50):
            arr = np.asarray(arr)
            if len(arr) < 2:
                return []
            d = np.sqrt(np.sum(np.diff(arr, axis=0)**2, axis=1))
            breaks = np.where(d > max_step)[0]
            out = []
            s = 0
            for b in breaks:
                if b + 1 - s >= 2:
                    out.append(arr[s:b+1])
                s = b + 1
            if len(arr) - s >= 2:
                out.append(arr[s:])
            return out or [arr]

        # -----------------------------------------
        # 5. Midline (green)
        # -----------------------------------------
        for S in segs:
            for segp in split(S):
                ax.plot(segp[:,0]-x0p, segp[:,1]-y0p, "w-", lw=1)

        # -----------------------------------------
        # 6. Edges (red = edge1, blue = edge2)
        # -----------------------------------------
        for E in edge1_segs:
            for segp in split(E):
                ax.plot(segp[:,0]-x0p, segp[:,1]-y0p, "r-", lw=1)

        for E in edge2_segs:
            for segp in split(E):
                ax.plot(segp[:,0]-x0p, segp[:,1]-y0p, "g-", lw=1)

        # -----------------------------------------
        # 7. Normals (cyan) — constant stride of 100 px
        # -----------------------------------------
        NORMAL_STRIDE = 10

        for n1, n2 in zip(norm1_segs, norm2_segs):
            print(f"Total left and right segments: {len(n1), len(n2)}")
            m = min(len(n1), len(n2))
            if m == 0:
                continue
            for i in range(0, m, NORMAL_STRIDE):
                p1 = n1[i]
                p2 = n2[i]
                ax.plot(
                    [p1[0]-x0p, p2[0]-x0p],
                    [p1[1]-y0p, p2[1]-y0p],
                    color="cyan", lw=1
                )

        # -----------------------------------------
        # final touches
        # -----------------------------------------
        title = f"Combined Crack (members: {', '.join(member_ids)})"
        ax.set_title(title, fontsize=14)
        ax.axis("off")
        fig.savefig(fname, dpi=350, bbox_inches="tight")
        plt.close(fig)

    def draw_existing_cracks(self, im):
        """Overlay existing cracks (atomic + combined) in red onto a copy of the image."""
        H, W = im.shape[:2]
        ann = self.annotation.setdefault("annotations", {})
        atomic = ann.setdefault("atomic_cracks", {})
        combined = ann.setdefault("combined_cracks", {})

        def reconstruct_full_mask(crack):
            mc, bb = crack.get("mask_crop"), crack.get("mask_bbox")
            if mc is None or bb is None or not len(mc):
                return np.zeros((H, W), np.uint8)
            crop = np.array(mc, dtype=np.uint8)
            x0, y0, w, h = [int(v) for v in bb]
            x1, y1 = min(x0 + w, W), min(y0 + h, H)
            mask = np.zeros((H, W), np.uint8)
            mask[y0:y1, x0:x1] = crop[:y1 - y0, :x1 - x0]
            return (mask > 0).astype(np.uint8)

        red = np.zeros_like(im)
        for crack in list(atomic.values()) + list(combined.values()):
            m = reconstruct_full_mask(crack)
            if np.any(m):
                red[m.astype(bool)] = (255, 0, 0)

        return cv2.addWeighted(im, 1, red, 0.35, 0)    

    def select_end_points_manmidlines(self, metrics: bool = False):
        print(metrics)
        if not hasattr(self, "original_image") or self.original_image is None:
            error("No original image found.")
            return
        
        if not hasattr(self, "annotation") or self.annotation is None:
            error("No annotation data found.")
            return

        from PyQt5.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
            QSizePolicy, QApplication, QMessageBox, QScrollArea, QLabel
        )

        boxes = self.get_all_bounding_boxes()

        readonly_midlines = {}
        readonly_connections = []
        existing_midlines = {}

        initial_points = list(getattr(self, "user_points", []) or [])
        initial_conns = list(getattr(self, "user_connections", []) or [])
        point_index_map = {tuple(pt): idx for idx, pt in enumerate(initial_points)}

        def ensure_point(pt):
            if tuple(pt) not in point_index_map:
                point_index_map[tuple(pt)] = len(initial_points)
                initial_points.append(pt)
            return point_index_map[tuple(pt)]

        if "annotations" in self.annotation:
            ann = self.annotation["annotations"]

            ann_midlines = ann.get("midlines", {})
            for k_str, pts in ann_midlines.items():
                try:
                    i1, i2 = map(int, k_str.split("_"))
                    if i1 != i2:  # Prevent self-midlines here
                        poly = [tuple(map(float, xy)) for xy in pts]
                        existing_midlines[(min(i1, i2), max(i1, i2))] = poly
                except Exception as e:
                    print(f"[WARN] Failed to parse manual midline {k_str}: {e}")

            for cid, crack in ann.get("atomic_cracks", {}).items():
                if hasattr(self, "current_crack_id") and str(cid) == str(self.current_crack_id):
                    continue

                src = crack.get("source", "auto")
                up = crack.get("user_points", [])
                if len(up) == 2:
                    p1, p2 = tuple(up[0]), tuple(up[1])
                else:
                    ml = crack.get("midline", [])
                    if len(ml) >= 2:
                        p1, p2 = tuple(ml[0]), tuple(ml[-1])
                    else:
                        continue

                idx1 = ensure_point(p1)
                idx2 = ensure_point(p2)

                if idx1 == idx2:
                    continue  # Prevent self-midlines here too

                if src in ("manual", "manual_poly"):
                    poly = crack.get("midline", [])
                    if poly:
                        readonly_midlines[(idx1, idx2)] = [tuple(map(float, xy)) for xy in poly]
                else:
                    readonly_connections.append((p1, p2))

        for (i1, i2), poly in existing_midlines.items():
            start_idx = ensure_point(poly[0])
            end_idx = ensure_point(poly[-1])
            if start_idx != end_idx:
                readonly_midlines[(start_idx, end_idx)] = poly

        readonly_conn_idx = []
        for p1, p2 in readonly_connections:
            idx1 = ensure_point(p1)
            idx2 = ensure_point(p2)
            if idx1 != idx2:
                readonly_conn_idx.append((min(idx1, idx2), max(idx1, idx2)))

        annot = CrackAnnotator(
            image=self.original_image,
            boxes=boxes,
            initial_points=initial_points,
            initial_connections=initial_conns,
            initial_midlines={},
        )

        #annot.readonly_midlines = readonly_midlines
        #annot.readonly_connections = readonly_conn_idx
        
        ro_conns = []
        for i1, i2 in readonly_conn_idx:
            a, b = (i1, i2) if i1 <= i2 else (i2, i1)
            ro_conns.append((a, b))

        # --- wrap read-only midlines into records the painter expects ---
        ro_mid = {}
        for (i1, i2), poly in readonly_midlines.items():
            a, b = (i1, i2) if i1 <= i2 else (i2, i1)
            ro_mid[(a, b)] = {
                "poly": [[float(x), float(y)] for (x, y) in poly],  # <- the key paintEvent reads
                "color": (160, 160, 160),                           # optional; safe default
                "readonly": True,                                   # optional; used by tools
                "tag": "existing"                                   # optional; for legend/debug
            }

        annot.readonly_midlines = ro_mid
        annot.readonly_connections = ro_conns

        # Also enforce at the widget level:
        # Wrap the original mousePressEvent to prevent bad start/finish overlaps
        # Also enforce at the widget level:
        orig_mousePressEvent = annot.mousePressEvent

        def guarded_mousePressEvent(ev):
            p = annot._to_image_coords(ev.pos())
            point_i = annot._find_point_at(p)

            # Gather extra debug state
            last_pt = annot.polyline[-1] if annot.polyline else None
            last_two_pts = annot.polyline[-2:] if len(annot.polyline) >= 2 else []
            conn_key = None
            if annot._is_drawing and point_i is not None and point_i != annot._start_idx:
                conn_key = annot._sorted(annot._start_idx, point_i)
            exists_midline = conn_key in annot.midlines if conn_key else None
            exists_conn = conn_key in annot.connections if conn_key else None
            exists_readonly_conn = conn_key in annot.readonly_connections if conn_key else None
            exists_readonly_mid = conn_key in annot.readonly_midlines if conn_key else None

            start_pt = annot.points[annot._start_idx] if annot._start_idx is not None and annot._start_idx < len(annot.points) else None

            print(
                f"[GUARDED_PRESS] btn={ev.button()} at {p} | "
                f"point_i={point_i}, start_idx={annot._start_idx}, start_pt={start_pt} | "
                f"_is_drawing={annot._is_drawing}, polyline_mode={annot.polyline_mode} | "
                f"polyline_len={len(annot.polyline)}, last_pt={last_pt}, last_two_pts={last_two_pts} | "
                f"_just_committed_midline={getattr(annot, '_just_committed_midline', False)}, "
                f"_last_polyline_end_idx={getattr(annot, '_last_polyline_end_idx', None)} | "
                f"conn_key={conn_key}, exists_midline={exists_midline}, exists_conn={exists_conn}, "
                f"exists_readonly_conn={exists_readonly_conn}, exists_readonly_mid={exists_readonly_mid}"
            )

            # If not in polyline mode — just pass through immediately
            if not annot.polyline_mode:
                print("[GUARDED_PRESS] Not in polyline_mode — passing through")
                orig_mousePressEvent(ev)
                return

            # --- Prevent immediate restart after a commit on either endpoint of last line ---
            if getattr(annot, "_just_committed_midline", False):
                last_end = getattr(annot, "_last_polyline_end_idx", None)
                last_start = getattr(annot, "_last_polyline_start_idx", None)
                if point_i is not None and point_i in (last_end, last_start):
                    print(
                        f"[GUARDED_PRESS] blocked: click is on endpoint {point_i} of last committed line "
                        f"({last_start}, {last_end}) — skipping to avoid rubberband"
                    )
                    annot._just_committed_midline = False
                    return
                # Reset flag if click is elsewhere
                annot._just_committed_midline = False

            # Polyline mode but NOT drawing yet
            if not annot._is_drawing:
                if point_i is not None and annot._start_idx == point_i:
                    print("[GUARDED_PRESS] blocked: start on same point as last start")
                    return
                print("[GUARDED_PRESS] passing to orig to START polyline")
                orig_mousePressEvent(ev)
                return

            # Polyline mode AND currently drawing
            if point_i is not None and point_i != annot._start_idx:
                print(f"[GUARDED_PRESS] finishing midline via click on endpoint {point_i}, key={conn_key}")
            elif point_i is None:
                print("[GUARDED_PRESS] adding freehand point via click")
            else:
                print("[GUARDED_PRESS] ignored click (same as start idx)")

            orig_mousePressEvent(ev)

        annot.mousePressEvent = guarded_mousePressEvent

        annot.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)

        dlg = QDialog(self.MainWindow)
        dlg.setWindowTitle("Endpoints, Connections & Manual Midlines")
        dlg.setWindowModality(Qt.ApplicationModal)
        dlg.setWindowFlags(dlg.windowFlags() | Qt.WindowMaximizeButtonHint)
        layout = QVBoxLayout(dlg)

        mode_btn = QPushButton("Switch to Connection Mode")
        mode_btn.setCheckable(True)
        layout.addWidget(mode_btn)

        manual_btn = QPushButton("Manual Midlines: OFF")
        manual_btn.setCheckable(True)
        manual_btn.setVisible(False)
        layout.addWidget(manual_btn)

        hint = QLabel(
            "Editable: Auto/Manual in black/light-blue.     Read-only: Auto/Manual cracks in gray/beige — cant delete in editor, must delete segment in Delete Segmentations.\n"
            "Click unconnected point in Connection Mode to create new point.        Hover and click in respective mode to delete.        Mousewheel/2-finger swipe = zoom in/out.\n"
            "Manual: Left-hold on starting point → draw → finish on a different endpoint.       Backspace/Z or Right-hold for fast/slow deletion.       Shimmy: zoom in toward direction, zoom out slightly to move; image may distort until 1.0 scale."
        )
        layout.addWidget(hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(annot)
        layout.addWidget(scroll, 1)

        btn_row = QHBoxLayout()
        btn_done, btn_cancel = QPushButton("Done"), QPushButton("Cancel")
        btn_row.addWidget(btn_done)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

        def all_points_in_boxes():
            def in_any(pt):
                x, y = pt
                for xmin, ymin, xmax, ymax in boxes:
                    if xmin <= x <= xmax and ymin <= y <= ymax:
                        return True
                return False

            readonly_point_idxs = set()
            for (i1, i2) in annot.readonly_connections:
                readonly_point_idxs.update([i1, i2])
            for (i1, i2) in annot.readonly_midlines.keys():
                readonly_point_idxs.update([i1, i2])

            bad = [
                pt for idx, pt in enumerate(annot.points)
                if idx not in readonly_point_idxs and not in_any(pt)
            ]
            return (len(bad) == 0, bad)

        def confirm_discard():
            mb = QMessageBox(dlg)
            mb.setIcon(QMessageBox.Warning)
            mb.setWindowTitle("Discard current midline?")
            mb.setText("You're in the middle of drawing a midline. Discard it?")
            mb.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            return mb.exec_() == QMessageBox.Yes

        def update_controls_visibility():
            if annot.polyline_mode:
                mode_btn.setVisible(False)
                manual_btn.setVisible(True)
                manual_btn.setText("Manual Midlines: ON")
                manual_btn.setStyleSheet("background:#79d2e6;")
            else:
                mode_btn.setVisible(True)
                if annot.connection_mode:
                    mode_btn.setText("Switch to Point Mode")
                    mode_btn.setStyleSheet("background:#97e297;")
                else:
                    mode_btn.setText("Switch to Connection Mode")
                    mode_btn.setStyleSheet("background:#e2c297;")
                manual_btn.setVisible(annot.connection_mode and (not annot.all_pairs_saturated()))
                manual_btn.setText("Manual Midlines: OFF")
                manual_btn.setStyleSheet("")

        mode_btn.clicked.connect(lambda: (annot.toggle_mode(), update_controls_visibility()))
        manual_btn.clicked.connect(lambda checked: (
            annot.set_mode_polyline(checked, confirm_cb=confirm_discard),
            update_controls_visibility()
        ))
        update_controls_visibility()
            
        def on_done():
            # --- gracefully handle unfinished polylines ---
            if annot.polyline_mode and annot._is_drawing:
                if not confirm_discard():
                    return
                annot.set_mode_polyline(False)

            # --- check all points are inside boxes ---
            ok, bad = all_points_in_boxes()
            if not ok:
                bad_str = "\n".join([f"({x:.1f}, {y:.1f})" for (x, y) in bad])
                QMessageBox.warning(
                    dlg,
                    "Points outside boxes",
                    f"The following points are outside all bounding boxes:\n{bad_str}"
                )
                return

            # --- validate each midline lies fully within one valid region ---
            for (i1, i2), poly in annot.midlines.items():
                if i1 == i2:
                    continue
                try:
                    sx, sy = annot.points[i1]
                    ex, ey = annot.points[i2]
                except Exception:
                    continue

                # --- Enhanced shared-edge / overlap rule (fixed) ---
                def boxes_containing(x, y, tol=0.5):
                    hits = []
                    for i, (xmin, ymin, xmax, ymax) in enumerate(boxes):
                        if (xmin - tol) <= x <= (xmax + tol) and (ymin - tol) <= y <= (ymax + tol):
                            hits.append(i)
                    return hits

                sx, sy = float(sx), float(sy)
                ex, ey = float(ex), float(ey)
                S = set(boxes_containing(sx, sy))
                E = set(boxes_containing(ex, ey))

                if not S or not E:
                    QMessageBox.warning(dlg, "Invalid midline", "One or both endpoints are outside all boxes.")
                    return

                # 1️⃣ use shared box if possible
                effective_region = None
                shared = S & E
                if shared:
                    bidx = next(iter(shared))
                    effective_region = boxes[bidx]
                else:
                    # 2️⃣ otherwise use overlap of their boxes
                    for i in S:
                        for j in E:
                            xmin1, ymin1, xmax1, ymax1 = boxes[i]
                            xmin2, ymin2, xmax2, ymax2 = boxes[j]
                            oxmin, oymin = max(xmin1, xmin2), max(ymin1, ymin2)
                            oxmax, oymax = min(xmax1, xmax2), min(ymax1, ymax2)
                            if oxmin <= oxmax and oymin <= oymax:
                                effective_region = (oxmin, oymin, oxmax, oymax)
                                break
                        if effective_region:
                            break

                if effective_region is None:
                    QMessageBox.warning(
                        dlg,
                        "Invalid midline",
                        "A manual midline spans boxes that don't share a region. Please fix before continuing."
                    )
                    return

                xmin, ymin, xmax, ymax = effective_region
                for (x, y) in poly:
                    x, y = float(x), float(y)
                    if not (xmin <= x <= xmax and ymin <= y <= ymax):
                        QMessageBox.warning(
                            dlg,
                            "Invalid midline",
                            "A manual midline has points outside its valid box/overlap region. Please fix before continuing."
                        )
                        return

            # --- collect points and connections ---
            self.user_points = annot.points
            self.user_connections = [c for c in annot.connections if c not in annot.readonly_connections]
            self.endpoint_pairs = [
                (self.user_points[i1], self.user_points[i2]) for (i1, i2) in self.user_connections
            ]

            # --- finalize and pull actual midlines before the dialog closes ---
            if hasattr(annot, "finalize_drawing_if_needed"):
                annot.finalize_drawing_if_needed()
            self.manual_midlines_tmp = dict(getattr(annot, "midlines", {}))

            # --- build manual endpoint list for later reference ---
            self.manual_endpoint_pairs = []
            for (i1, i2), poly in annot.midlines.items():
                if i1 != i2:
                    self.manual_endpoint_pairs.append((self.user_points[i1], self.user_points[i2]))

            # --- debug print to confirm ---
            print(f"[DEBUG] manual_midlines_tmp contents: {len(self.manual_midlines_tmp)} midlines")
            for k, v in self.manual_midlines_tmp.items():
                print(f"   key={k}, len={len(v)})")

            # --- persist manual midlines into annotation for saving ---
            if not hasattr(self, "annotation") or not isinstance(self.annotation, dict):
                self.annotation = {"annotations": {"atomic_cracks": {}}}

            ann = self.annotation.setdefault("annotations", {})
            if "atomic_cracks" not in ann:
                ann["atomic_cracks"] = {}
            ac = ann["atomic_cracks"]

            # one entry per manual midline
            '''if metrics:
                for k, poly in self.manual_midlines_tmp.items():
                    try:
                        if isinstance(k, tuple):
                            i1, i2 = k
                        elif isinstance(k, str) and "_" in k:
                            i1, i2 = map(int, k.split("_"))
                        else:
                            continue

                        cid = str(len(ac))

                        # --- compute synthetic bounding box of midline ---
                        if poly:
                            xs = [p[0] for p in poly]
                            ys = [p[1] for p in poly]
                            x, y = min(xs), min(ys)
                            w, h = max(xs) - x, max(ys) - y
                            mask_bbox = [float(x), float(y), float(w), float(h)]
                        else:
                            mask_bbox = [0.0, 0.0, 1.0, 1.0]

                        ac[cid] = {
                            "src": "manual_poly",
                            "midline": [[float(x), float(y)] for (x, y) in poly],
                            "user_points": [list(self.user_points[i1]), list(self.user_points[i2])],
                            "user_connections": [[0, 1]],
                            "mask_compact": [],
                            "mask_bbox": mask_bbox,
                        }

                    except Exception as e:
                        print(f"[DEBUG persist_manual_midline] failed for key={k}: {e}")'''

            print(f"[SAVE] Manual selections committed to in-memory annotations.")
            dlg.accept()


        btn_done.clicked.connect(on_done)
        btn_cancel.clicked.connect(lambda: dlg.reject())

        dlg.showMaximized()
        QApplication.processEvents()
        if dlg.exec_() != QDialog.Accepted:
            return

        print(f"Points: {self.user_points}")
        print(f"Connections: {self.user_connections}")
        print(f"Endpoint pairs: {self.endpoint_pairs}")
        print(f"Manual endpoint pairs: {getattr(self, 'manual_endpoint_pairs', [])}")
        print(f"Midlines saved: {len(self.manual_midlines_tmp)}")
        self.update_image_crop_button.setStyleSheet("background-color: lightblue")
        self._debug_print_atomic_cracks("select_end_points_manmidlines AFTER ACCEPT")
        self.all_selected_points = list(self.user_points)
        
        # === NEW: persist manual selections into annotations (runs before select_end_points_manmidlines returns) ===
        try:
            ann = self.annotation.setdefault("annotations", {})
            atomic = ann.setdefault("atomic_cracks", {})

            # helper: next gap-free numeric id as string (0,1,2,...)
            def _next_id_str():
                used = sorted(int(k) for k in atomic.keys() if str(k).isdigit())
                nxt = 0
                for k in used:
                    if k == nxt:
                        nxt += 1
                    elif k > nxt:
                        break
                return str(nxt)

            # helper: normalize a pair (for simple dup checks)
            def _norm_pair(pA, pB, r=6):
                return (round(float(pA[0]), r), round(float(pA[1]), r),
                        round(float(pB[0]), r), round(float(pB[1]), r))

            # build an existing-pairs set
            existing_pairs = set()
            for crack in atomic.values():
                up = crack.get("user_points") or []
                if len(up) == 2:
                    existing_pairs.add(_norm_pair(up[0], up[1]))
                    existing_pairs.add(_norm_pair(up[1], up[0]))

            # 1) commit each drawn manual polyline as its own atomic crack
            mm = dict(getattr(self, "manual_midlines_tmp", {}) or {})
            print(f"manual_midlines_tmp len (pre-commit): {len(mm)}")
            if not mm:
                print("[WARN] No manual midlines to commit!")
            print(f"metrics value: {metrics}")

            if metrics:
                print("committing new manual_polys to in-memory annotations")
                for k, poly in mm.items():
                    print(f"[DEBUG] keys in mm: {list(mm.keys())}")
                    # --- normalize key format ---
                    if isinstance(k, tuple):
                        i1, i2 = k
                    elif isinstance(k, str) and "_" in k:
                        try:
                            i1, i2 = map(int, k.split("_"))
                        except ValueError:
                            print(f"[WARN] bad key format: {k}")
                            continue
                    else:
                        print(f"[WARN] skipping unrecognized key type {type(k)}: {k}")
                        continue

                    if not (0 <= i1 < len(self.user_points) and 0 <= i2 < len(self.user_points)):
                        print(f"[WARN] invalid indices for key {k}")
                        continue

                    p1 = self.user_points[i1]
                    p2 = self.user_points[i2]
                    if _norm_pair(p1, p2) in existing_pairs:
                        continue

                    cid = _next_id_str()
                    atomic[cid] = {
                        "source": "manual_poly",
                        "user_points": [[float(p1[0]), float(p1[1])],
                                        [float(p2[0]), float(p2[1])]],
                        "user_connections": [[0, 1]],
                        "midline": [[float(x), float(y)] for (x, y) in poly],
                    }
                    existing_pairs.add(_norm_pair(p1, p2))
                    print(f"[MEM] added manual_poly id={cid}, len={len(poly)} pts")

            # 🔹 ensure all nested dicts exist and rebind them
            self.annotation.setdefault("annotations", {})["atomic_cracks"] = atomic
            print(f"[MEM] committed manual_polys to self.annotation (len={len(atomic)})")

        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self.MainWindow, "Persist error", f"Failed to commit manual selections:\n{e}")
            print(f"[SAVE] Persist error: {e}")
            
    '''def select_end_points_manmidlines(self, metrics: bool = False):
        """
        Endpoint/connection + manual-midline picker.

        modes:
        - metrics=False (default): legacy/pipeline behavior ("manual_poly" entries, no crops)
        - metrics=True: store extra fields for metrics (mask_crop, mask_bbox, mask_pixels, status)
        """
        if not hasattr(self, "original_image") or self.original_image is None:
            error("No original image found.")
            return

        if not hasattr(self, "annotation") or self.annotation is None:
            error("No annotation data found.")
            return

        # ---- Qt imports & setup ----
        from PyQt5 import QtWidgets
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
            QApplication, QMessageBox, QScrollArea, QLabel
        )

        import numpy as np
        import cv2

        boxes = self.get_all_bounding_boxes()

        readonly_midlines = {}
        readonly_connections = []
        existing_midlines = {}

        initial_points = list(getattr(self, "user_points", []) or [])
        initial_conns  = list(getattr(self, "user_connections", []) or [])
        point_index_map = {tuple(pt): idx for idx, pt in enumerate(initial_points)}

        def ensure_point(pt):
            if tuple(pt) not in point_index_map:
                point_index_map[tuple(pt)] = len(initial_points)
                initial_points.append(pt)
            return point_index_map[tuple(pt)]

        # ---- preload existing / readonly from annotations ----
        if "annotations" in self.annotation:
            ann = self.annotation["annotations"]

            # previous freehand midlines (keyed by "i_j")
            ann_midlines = ann.get("midlines", {})
            for k_str, pts in ann_midlines.items():
                try:
                    i1, i2 = map(int, k_str.split("_"))
                    if i1 != i2:
                        poly = [tuple(map(float, xy)) for xy in pts]
                        existing_midlines[(min(i1, i2), max(i1, i2))] = poly
                except Exception as e:
                    print(f"[WARN] Failed to parse manual midline {k_str}: {e}")

            # atomic cracks → read-only overlays (manuals are read-only here)
            for cid, crack in ann.get("atomic_cracks", {}).items():
                if hasattr(self, "current_crack_id") and str(cid) == str(self.current_crack_id):
                    continue

                src = crack.get("source", crack.get("src", "auto"))
                up  = crack.get("user_points", [])
                if len(up) == 2:
                    p1, p2 = tuple(up[0]), tuple(up[1])
                else:
                    ml = crack.get("midline", [])
                    if len(ml) >= 2:
                        p1, p2 = tuple(ml[0]), tuple(ml[-1])
                    else:
                        continue

                idx1 = ensure_point(p1)
                idx2 = ensure_point(p2)
                if idx1 == idx2:
                    continue

                if src in ("manual", "manual_poly"):
                    poly = crack.get("midline", [])
                    if poly:
                        readonly_midlines[(idx1, idx2)] = [tuple(map(float, xy)) for xy in poly]
                else:
                    readonly_connections.append((p1, p2))

        # hoist prior readonly midlines keyed by point indices
        for (i1, i2), poly in existing_midlines.items():
            s = ensure_point(poly[0])
            e = ensure_point(poly[-1])
            if s != e:
                readonly_midlines[(s, e)] = poly

        readonly_conn_idx = []
        for p1, p2 in readonly_connections:
            i1 = ensure_point(p1)
            i2 = ensure_point(p2)
            if i1 != i2:
                readonly_conn_idx.append((min(i1, i2), max(i1, i2)))

        # ---- build annotator ----
        annot = CrackAnnotator(
            image=self.original_image,
            boxes=boxes,
            initial_points=initial_points,
            initial_connections=initial_conns,
            initial_midlines={},
        )
        annot.readonly_midlines = readonly_midlines
        annot.readonly_connections = readonly_conn_idx

        # guard mouse presses to avoid degenerate restarts
        orig_mousePressEvent = annot.mousePressEvent

        def guarded_mousePressEvent(ev):
            p = annot._to_image_coords(ev.pos())
            point_i = annot._find_point_at(p)

            last_pt = annot.polyline[-1] if annot.polyline else None
            last_two_pts = annot.polyline[-2:] if len(annot.polyline) >= 2 else []
            conn_key = None
            if annot._is_drawing and point_i is not None and point_i != annot._start_idx:
                conn_key = annot._sorted(annot._start_idx, point_i)

            print(
                f"[GUARDED_PRESS] btn={ev.button()} at {p} | "
                f"point_i={point_i}, start_idx={annot._start_idx} | "
                f"_is_drawing={annot._is_drawing}, polyline_mode={annot.polyline_mode} | "
                f"polyline_len={len(annot.polyline)}, last_pt={last_pt}, last_two_pts={last_two_pts} | "
                f"conn_key={conn_key}"
            )

            if not annot.polyline_mode:
                return orig_mousePressEvent(ev)

            if getattr(annot, "_just_committed_midline", False):
                last_end   = getattr(annot, "_last_polyline_end_idx", None)
                last_start = getattr(annot, "_last_polyline_start_idx", None)
                if point_i is not None and point_i in (last_end, last_start):
                    print("[GUARDED_PRESS] blocked immediate restart on last line endpoint")
                    annot._just_committed_midline = False
                    return
                annot._just_committed_midline = False

            if not annot._is_drawing:
                if point_i is not None and annot._start_idx == point_i:
                    print("[GUARDED_PRESS] blocked: same start point")
                    return
                return orig_mousePressEvent(ev)

            # currently drawing
            return orig_mousePressEvent(ev)

        annot.mousePressEvent = guarded_mousePressEvent
        annot.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)

        # ---- dialog ----
        dlg = QDialog(self.MainWindow)
        dlg.setWindowTitle("Endpoints, Connections & Manual Midlines")
        dlg.setWindowModality(Qt.ApplicationModal)
        dlg.setWindowFlags(dlg.windowFlags() | Qt.WindowMaximizeButtonHint)
        layout = QVBoxLayout(dlg)

        mode_btn = QPushButton("Switch to Connection Mode")
        mode_btn.setCheckable(True)
        layout.addWidget(mode_btn)

        manual_btn = QPushButton("Manual Midlines: OFF")
        manual_btn.setCheckable(True)
        manual_btn.setVisible(False)
        layout.addWidget(manual_btn)

        hint = QLabel(
            "Editable: Auto/Manual in black/light-blue.   Read-only shown in gray/beige.\n"
            "Point/Connection modes add endpoints or connect them.  Mousewheel = zoom.\n"
            "Manual midline: Left-hold on start endpoint → draw → release on different endpoint."
        )
        layout.addWidget(hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(annot)
        layout.addWidget(scroll, 1)

        btn_row = QHBoxLayout()
        btn_done, btn_cancel = QPushButton("Done"), QPushButton("Cancel")
        btn_row.addWidget(btn_done)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

        # ---- helpers ----
        def all_points_in_boxes():
            def in_any(pt):
                x, y = pt
                for xmin, ymin, xmax, ymax in boxes:
                    if xmin <= x <= xmax and ymin <= y <= ymax:
                        return True
                return False

            readonly_point_idxs = set()
            for (i1, i2) in annot.readonly_connections:
                readonly_point_idxs.update([i1, i2])
            for (i1, i2) in annot.readonly_midlines.keys():
                readonly_point_idxs.update([i1, i2])

            bad = [
                pt for idx, pt in enumerate(annot.points)
                if idx not in readonly_point_idxs and not in_any(pt)
            ]
            return (len(bad) == 0, bad)

        def confirm_discard():
            mb = QMessageBox(dlg)
            mb.setIcon(QMessageBox.Warning)
            mb.setWindowTitle("Discard current midline?")
            mb.setText("You're in the middle of drawing a midline. Discard it?")
            mb.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            return mb.exec_() == QMessageBox.Yes

        def update_controls_visibility():
            if annot.polyline_mode:
                mode_btn.setVisible(False)
                manual_btn.setVisible(True)
                manual_btn.setText("Manual Midlines: ON")
                manual_btn.setStyleSheet("background:#79d2e6;")
            else:
                mode_btn.setVisible(True)
                if annot.connection_mode:
                    mode_btn.setText("Switch to Point Mode")
                    mode_btn.setStyleSheet("background:#97e297;")
                else:
                    mode_btn.setText("Switch to Connection Mode")
                    mode_btn.setStyleSheet("background:#e2c297;")
                manual_btn.setVisible(annot.connection_mode and (not annot.all_pairs_saturated()))
                manual_btn.setText("Manual Midlines: OFF")
                manual_btn.setStyleSheet("")

        mode_btn.clicked.connect(lambda: (annot.toggle_mode(), update_controls_visibility()))
        manual_btn.clicked.connect(lambda checked: (
            annot.set_mode_polyline(checked, confirm_cb=confirm_discard),
            update_controls_visibility()
        ))
        update_controls_visibility()

        # ---- Done handler ----
        def on_done():
            # close any in-progress polyline safely
            if annot.polyline_mode and annot._is_drawing:
                if not confirm_discard():
                    return
                annot.set_mode_polyline(False)

            # validate points in boxes
            ok, bad = all_points_in_boxes()
            if not ok:
                bad_str = "\n".join([f"({x:.1f}, {y:.1f})" for (x, y) in bad])
                QMessageBox.warning(dlg, "Points outside boxes",
                                    f"The following points are outside all bounding boxes:\n{bad_str}")
                return

            # validate each drawn midline fully lies in a shared/overlap region
            for (i1, i2), poly in annot.midlines.items():
                if i1 == i2:
                    continue
                try:
                    sx, sy = annot.points[i1]
                    ex, ey = annot.points[i2]
                except Exception:
                    continue

                def boxes_containing(x, y, tol=0.5):
                    hits = []
                    for i, (xmin, ymin, xmax, ymax) in enumerate(boxes):
                        if (xmin - tol) <= x <= (xmax + tol) and (ymin - tol) <= y <= (ymax + tol):
                            hits.append(i)
                    return hits

                sx, sy = float(sx), float(sy)
                ex, ey = float(ex), float(ey)
                S = set(boxes_containing(sx, sy))
                E = set(boxes_containing(ex, ey))

                if not S or not E:
                    QMessageBox.warning(dlg, "Invalid midline",
                                        "One or both endpoints are outside all boxes.")
                    return

                effective_region = None
                shared = S & E
                if shared:
                    bidx = next(iter(shared))
                    effective_region = boxes[bidx]
                else:
                    for i in S:
                        for j in E:
                            xmin1, ymin1, xmax1, ymax1 = boxes[i]
                            xmin2, ymin2, xmax2, ymax2 = boxes[j]
                            oxmin, oymin = max(xmin1, xmin2), max(ymin1, ymin2)
                            oxmax, oymax = min(xmax1, xmax2), min(ymax1, ymax2)
                            if oxmin <= oxmax and oymin <= oymax:
                                effective_region = (oxmin, oymin, oxmax, oymax)
                                break
                        if effective_region:
                            break

                if effective_region is None:
                    QMessageBox.warning(
                        dlg, "Invalid midline",
                        "A manual midline spans boxes that don't share a region. Please fix before continuing."
                    )
                    return

                xmin, ymin, xmax, ymax = effective_region
                for (x, y) in poly:
                    x, y = float(x), float(y)
                    if not (xmin <= x <= xmax and ymin <= y <= ymax):
                        QMessageBox.warning(
                            dlg, "Invalid midline",
                            "A manual midline has points outside its valid region. Please fix."
                        )
                        return

            # ---- collect ui results ----
            self.user_points = annot.points
            self.user_connections = [c for c in annot.connections if c not in annot.readonly_connections]
            self.endpoint_pairs = [
                (self.user_points[i1], self.user_points[i2]) for (i1, i2) in self.user_connections
            ]

            # finalize drawing & capture midlines
            if hasattr(annot, "finalize_drawing_if_needed"):
                annot.finalize_drawing_if_needed()
            self.manual_midlines_tmp = dict(getattr(annot, "midlines", {}))

            # build manual endpoint pairs (needed by pipeline later)
            self.manual_endpoint_pairs = []
            for (i1, i2), poly in (self.manual_midlines_tmp or {}).items():
                if i1 != i2 and 0 <= i1 < len(self.user_points) and 0 <= i2 < len(self.user_points):
                    self.manual_endpoint_pairs.append((
                        tuple(map(float, self.user_points[i1])),
                        tuple(map(float, self.user_points[i2]))
                    ))

            print(f"[DEBUG] manual_midlines_tmp contents: {len(self.manual_midlines_tmp)} midlines")

            # ---- persist selections into annotations ----
            if not hasattr(self, "annotation") or not isinstance(self.annotation, dict):
                self.annotation = {"annotations": {"atomic_cracks": {}}}

            ann = self.annotation.setdefault("annotations", {})
            ac  = ann.setdefault("atomic_cracks", {})

            # append each drawn manual polyline as one atomic crack
            for k, poly in (self.manual_midlines_tmp or {}).items():
                try:
                    if isinstance(k, tuple):
                        i1, i2 = k
                    elif isinstance(k, str) and "_" in k:
                        i1, i2 = map(int, k.split("_"))
                    else:
                        continue

                    cid = str(len(ac))

                    # bbox around the polyline (for metrics=True we also create a tiny ribbon mask)
                    if poly:
                        xs = [p[0] for p in poly]
                        ys = [p[1] for p in poly]
                        x0, y0 = min(xs), min(ys)
                        x1, y1 = max(xs), max(ys)
                        w, h  = max(1, int(np.ceil(x1 - x0) + 3)), max(1, int(np.ceil(y1 - y0) + 3))
                    else:
                        x0 = y0 = 0.0
                        w = h = 1

                    entry = {
                        "user_points": [list(self.user_points[i1]), list(self.user_points[i2])],
                        "user_connections": [[0, 1]],
                        "midline": [[float(x), float(y)] for (x, y) in poly],
                        "mask_compact": [],
                        "mask_bbox": [float(x0), float(y0), float(w), float(h)],
                    }

                    if metrics:
                        # richer entry used by metrics/export (gives pipeline a head-start)
                        mask_crop = np.zeros((int(h), int(w)), np.uint8)
                        if poly:
                            shifted = np.array([[px - x0, py - y0] for (px, py) in poly], np.int32)
                            for i in range(len(shifted) - 1):
                                cv2.line(mask_crop, tuple(shifted[i]), tuple(shifted[i + 1]), 1, 1)
                        entry.update({
                            "src": "manual",
                            "mask_crop": mask_crop.tolist(),
                            "mask_pixels": int(mask_crop.sum()),
                            "status": "pending_edge_tracking",
                        })
                    else:
                        # legacy lightweight
                        entry.update({"src": "manual_poly"})

                    ac[cid] = entry

                except Exception as e:
                    print(f"[DEBUG persist_manual_midline] failed for key={k}: {e}")

            print(f"[SAVE] Manual selections committed ({'metrics' if metrics else 'legacy'}) mode.")
            dlg.accept()

        btn_done.clicked.connect(on_done)
        btn_cancel.clicked.connect(lambda: dlg.reject())

        dlg.showMaximized()
        QApplication.processEvents()
        if dlg.exec_() != QDialog.Accepted:
            return

        print(f"Points: {self.user_points}")
        print(f"Connections: {self.user_connections}")
        print(f"Endpoint pairs: {self.endpoint_pairs}")
        print(f"Manual endpoint pairs: {getattr(self, 'manual_endpoint_pairs', [])}")
        print(f"Midlines saved: {len(self.manual_midlines_tmp)}")
        self.update_image_crop_button.setStyleSheet("background-color: lightblue")
        self._debug_print_atomic_cracks("select_end_points_manmidlines AFTER ACCEPT")
        self.all_selected_points = list(self.user_points)

        # --- also persist raw connections (without polylines) as "manual" for completeness ---
        try:
            ann = self.annotation.setdefault("annotations", {})
            atomic = ann.setdefault("atomic_cracks", {})

            def _next_id_str():
                used = sorted(int(k) for k in atomic.keys() if str(k).isdigit())
                nxt = 0
                for k in used:
                    if k == nxt:
                        nxt += 1
                    elif k > nxt:
                        break
                return str(nxt)

            def _norm_pair(pA, pB, r=6):
                return (round(float(pA[0]), r), round(float(pA[1]), r),
                        round(float(pB[0]), r), round(float(pB[1]), r))

            existing_pairs = set()
            for crack in atomic.values():
                up = crack.get("user_points") or []
                if len(up) == 2:
                    existing_pairs.add(_norm_pair(up[0], up[1]))
                    existing_pairs.add(_norm_pair(up[1], up[0]))

            for (i1, i2) in (getattr(self, "user_connections", []) or []):
                if not (0 <= i1 < len(self.user_points) and 0 <= i2 < len(self.user_points)):
                    continue
                p1 = self.user_points[i1]
                p2 = self.user_points[i2]
                if _norm_pair(p1, p2) in existing_pairs:
                    continue

                cid = _next_id_str()
                atomic[cid] = {
                    "source": "manual",
                    "user_points": [[float(p1[0]), float(p1[1])],
                                    [float(p2[0]), float(p2[1])]],
                    "user_connections": [[0, 1]],
                    "midline": [],  # connection only
                }
                existing_pairs.add(_norm_pair(p1, p2))

            print("[SAVE] Manual connections committed.")
        except Exception as e:
            QMessageBox.critical(self.MainWindow, "Persist error", f"Failed to commit manual selections:\n{e}")
            print(f"[SAVE] Persist error: {e}")'''

    def update_image_crop(self):
        try:
            import numpy as np, cv2, skimage.measure
            from PyQt5.QtGui import QImage, QPixmap
            from PyQt5.QtCore import Qt

            downsample_factor = self.downsample_factor_box.value()
            color_channel = [0 if self.color_chenel_box.currentText()=='R'
                            else 1 if self.color_chenel_box.currentText()=='B'
                            else 2][0]

            if not (hasattr(self, "active_bbox") and self.active_bbox is not None):
                raise ValueError("No active bounding box for cropping.")

            H, W = self.original_image.shape[:2]
            x0, y0, x1, y1 = self.active_bbox

            # --- make crop inclusive (keep border pixels) ---
            xmin = max(0, int(np.floor(x0)))
            ymin = max(0, int(np.floor(y0)))
            xmax = min(W, int(np.ceil(x1)))     # +1 to be inclusive
            ymax = min(H, int(np.ceil(y1)))

            if xmax <= xmin or ymax <= ymin:
                print(f"Invalid bounding box cropping window: ({xmin},{ymin})→({xmax},{ymax}), skipping.")
                return

            self.image_crop = self.original_image[ymin:ymax, xmin:xmax]
            h, w = self.image_crop.shape[:2]

            min_crop_size = 8  # keep what you used before if different
            if h < min_crop_size or w < min_crop_size:
                print(f"Skipping segment: Crop too small for processing ({w}x{h})")
                self.skip_current_segment = True
                return
            else:
                self.skip_current_segment = False

            # shift endpoints into crop frame
            self.pts_crop = [np.array([pt[0]-xmin, pt[1]-ymin], dtype=float) for pt in self.pts]

            # relaxed endpoint check (+ clamp)
            tol = 1e-2
            for i, pt in enumerate(self.pts_crop):
                x, y = float(pt[0]), float(pt[1])
                if not (-tol <= x <= (w - 1) + tol) or not (-tol <= y <= (h - 1) + tol):
                    print(f"[WARN] Endpoint {pt} slightly outside crop ({w},{h}) → clamping instead of skipping.")
                    x = np.clip(x, 0, w - 1 - 1e-6)   # ensure floor stays < w
                    y = np.clip(y, 0, h - 1 - 1e-6)
                    self.pts_crop[i] = np.array([x, y], dtype=float)

            # Downsample
            black_crack = [0 if self.crack_color_box.currentText() =='Bright crack' else 1 ][0]
            func = np.min if black_crack==1 else np.max

            self.image_crop_down = skimage.measure.block_reduce(
                self.image_crop, block_size=(downsample_factor, downsample_factor, 1),
                func=func, cval=0, func_kwargs=None)

            self.pts_crop_down = [np.array(x)/downsample_factor for x in self.pts_crop]

            self.image_down = skimage.measure.block_reduce(
                self.original_image, block_size=(downsample_factor, downsample_factor, 1),
                func=func, cval=0, func_kwargs=None)

            self.pts_down = [np.array(x)/downsample_factor for x in self.pts]

            gs_image = self.image_crop_down[:, :, color_channel].astype(np.uint8)
            hh, ww = gs_image.shape[:2]

            # --- draw with FLOOR + clamp so (w-1e-3) can't round to w ---
            for pt in self.pts_crop_down:
                x = int(np.floor(pt[0]))
                y = int(np.floor(pt[1]))
                if 0 <= x < ww and 0 <= y < hh:
                    gs_image = cv2.circle(gs_image, (x, y), 2, (255,), thickness=2)
                else:
                    print(f"Warning: Skipping drawing point ({x},{y}) out of image bounds ({ww},{hh})")

            qimage = QImage(gs_image, gs_image.shape[1], gs_image.shape[0],
                            gs_image.strides[0], QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimage)
            scaled = pixmap.scaled(self.image_crop_down_display.width(),
                                self.image_crop_down_display.height(),
                                Qt.KeepAspectRatio, Qt.FastTransformation)

            self.image_crop_down_display.setPixmap(scaled)
            self.x_size_show.display(self.image_crop_down.shape[1])
            self.y_size_show.display(self.image_crop_down.shape[0])
            self.update_os_button.setStyleSheet("background-color : lightblue")

        except Exception as e:
            import traceback
            traceback.print_exc()
            error(f"update_image_crop: {e}")
            self.update_os_button.setStyleSheet("background-color : red")

