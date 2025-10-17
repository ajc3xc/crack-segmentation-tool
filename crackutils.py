import cracktools as ct
from helpers.crackhelpers import *

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

import matplotlib.pyplot as plt

import numpy as np
from math import hypot, atan2, pi
from skimage.morphology import skeletonize
import hashlib
import time


min_crop_size = 16
ROUNDING_DIGITS=6

#This class is basically is all of the utility / save and load or unimportant functions that aren't directly accessible via a ui button or aren't important
#that way the 'main' / work in progress code can be more easily accessed and modified
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
            src = crack.get("source", "?")
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
        with open(self.ann_name, 'w') as f:
            json.dump(self.annotation, f)
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
            # Save file and reload
            with open(self.ann_name, 'w') as fp:
                json.dump(self.annotation, fp)
            print(f"Deleted {len(selected_indices)} bounding box(es).")
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

    def manual_segment_full_screen(self):
        try:
            pts = np.array([self.manuall_x,self.manuall_y]).transpose(1,0).reshape((-1,1,2)).astype(np.int32)
            im = self.image.astype(np.uint8).copy()
            plt.imshow(im)
            plt.plot(self.manuall_x,self.manuall_y,'r',linewidth = 1)
            plt.show()
        except Exception as e:
            error(e)
    
        
    def _draw_crack(self, im, crack, color_mask=(0,1,1), color_midline=(0,0,255),
        color_edges=(255,255,0), color_points=(255,0,0)):
        def _draw_polyline(im, pts, color, thickness=2):
            """Draw a polyline where NaN rows mark gaps."""
            pts = np.asarray(pts, dtype=float)
            if pts.ndim != 2 or pts.shape[1] != 2:
                return im
            valid = ~np.isnan(pts).any(axis=1)
            # split into contiguous chunks without NaNs
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
        for _, edge_pts in edges.items():
            edge_pts = np.array(edge_pts, dtype=float)
            if len(edge_pts) > 1:
                im = _draw_polyline(im, edge_pts, color_edges, 2)
                
        # --- Midline ---
        midline = np.array(crack.get("midline", []), dtype=float)
        if len(midline) > 1:
            im = _draw_polyline(im, midline, color_midline, 2)

        # --- Endpoints ---
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
    
    # crackutils.py  --- CrackUtils.change_image (DEBUG PATCHED)
    def change_image(self):
        import os, json, cv2
        import numpy as np
        import matplotlib.pyplot as plt
        from skimage.segmentation import mark_boundaries  # for colored boundaries
        
        if not hasattr(self, "image_names") or not self.image_names:
            error("No images loaded. Please load images before using change_image().")
            return

        self.current_crack_id = None
        self.bb_pts_list = []
        w = self.segment_width_box_2.value()

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

            #print(f"[DEBUG change_image] base_name={base_name}")
            #print(f"[DEBUG change_image] available mask_map keys (first 20): {list(self.mask_map.keys())[:20]}")
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
            with open(self.ann_name) as f:
                self.annotation = json.load(f)
            ann = self.annotation.get('annotations', {}) or {}
            atomic = ann.get("atomic_cracks", {}) or {}
            combined = ann.get("combined_cracks", {}) or {}

            # ---- Bounding boxes ----
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

                # draw combined with reddish mask, same style for midline/edges/points
                im, _ = self._draw_crack(im, crack,
                                        color_mask=(.6, 0.6, 0.6),  # blue
                                        color_midline=(0,0,255),   # green
                                        color_edges=(255,255,0),     # yellow edges
                                        color_points=(255,0,0))      # red-pink endpoints

            # ---- Atomic cracks ----
            for crack_id, crack in atomic.items():
                if crack_id in drawn_atomic:
                    continue  # skip, already represented by combined
                im, mask_full = self._draw_crack(im, crack)
                if mask_full is not None:
                    self.mask.append(mask_full)

                    # --- Debug figure ---
                    '''fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                    axes[0].imshow(self.original_image); axes[0].set_title("Original Image")

                    axes[1].imshow(union_members, cmap="gray")
                    axes[1].set_title("Union of atomic members")

                    axes[2].imshow(mask_crop, cmap="gray")
                    axes[2].set_title(f"Combined {crack_id} crop (shape={mask_crop.shape})")

                    axes[3].imshow(union_crop, cmap="gray")
                    axes[3].set_title("Union crop in bbox")

                    for ax in axes: ax.axis("off")
                    plt.tight_layout()
                    out_path = f"debug_combined_{crack_id}.png"
                    plt.savefig(out_path); plt.close()
                    print(f"[DEBUG change_image] wrote {out_path}")'''

            '''for crack_id, crack in list(combined.items()):
                members = [m for m in crack.get("members", []) if m in atomic]
                crack["members"] = members
                if not members:
                    continue

                mask_full = reconstruct_full_mask_from_crack(crack, H, W)
                if np.any(mask_full):
                    im = (mark_boundaries(im/255.0, (mask_full>0).astype(np.uint8),
                                        color=(1.0, 0.6, 0.6), background_label=0)*255).astype(np.uint8)

                mc = crack.get("mask_crop"); bb = crack.get("mask_bbox")
                if mc is not None and bb is not None:
                    x, y, w, h = map(int, bb)
                    mask_crop = np.array(mc, dtype=np.uint8)

                    print(f"[DEBUG change_image] placing combined {crack_id}: "
                        f"bbox(w={w},h={h}), mask_crop.shape={mask_crop.shape}")

                    if mask_crop.shape != (h, w):
                        print(f"  ⚠️ Shape mismatch: expected ({h},{w}), got {mask_crop.shape}")
                        # auto-fix if clearly just transposed
                        if mask_crop.shape == (w, h):
                            print(f"  → fixing by transpose for {crack_id}")
                            mask_crop = mask_crop.T

                    mask_canvas = np.zeros((H, W), dtype=np.uint8)
                    try:
                        mask_canvas[y:y+h, x:x+w] = mask_crop[:h, :w]
                    except Exception as e:
                        print(f"[DEBUG change_image] ERROR placing mask_crop for combined {crack_id}: {e}")

                    # --- Debug figure ---
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].imshow(self.original_image); axes[0].set_title("Original Image")
                    axes[1].imshow(self.original_image)
                    axes[1].imshow(mask_canvas, alpha=0.5, cmap="Reds")
                    axes[1].set_title(f"Combined {crack_id} in full coords")
                    axes[2].imshow(mask_crop, cmap="gray")
                    axes[2].set_title(f"mask_crop raw (shape={mask_crop.shape})")
                    for ax in axes: ax.axis("off")
                    plt.tight_layout()
                    out_path = f"debug_combined_{crack_id}.png"
                    plt.savefig(out_path); plt.close()
                    print(f"[DEBUG change_image] wrote {out_path}")'''

        # ---- Render main image to screen ----
        _, pixmap = numpy_to_qimage_and_scaled_pixmap(
            im.astype(np.uint8), self.ImageScreen.width(), self.ImageScreen.height(), is_gray=False
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
            (full_mask_display * 255).astype(np.uint8),
            self.all_segments_display.width(),
            self.all_segments_display.height(),
            is_gray=True
        )
        self.all_segments_display.setPixmap(pixmap_mask)

    def _build_combined_crack(self, member_ids, pad=10):
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
        }
        
    def auto_combine_segments(self):
        """
        Automatically combine atomic cracks that overlap or share endpoints.
        Reuses _build_combined_crack for consistency.
        If an atomic crack already belongs to a combined crack, it will extend that
        combined crack when new overlaps/branches are detected.
        """
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

        # --- Helper: same as in combine_segments ---
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
            if full.size == H*W and full.shape == (H,W):
                return (full > 0).astype(np.uint8)
            return np.zeros((H,W), dtype=np.uint8)

        def cracks_overlap_or_connect(crackA, crackB):
            if np.any(mask_from_crack(crackA) & mask_from_crack(crackB)):
                return True
            upA = [tuple(pt) for pt in crackA.get("user_points", [])]
            upB = [tuple(pt) for pt in crackB.get("user_points", [])]
            return bool(set(upA) & set(upB))

        # --- Build list of "entries": atomic or combined ---
        # Each atomic should appear only once (if in combined, skip here)
        seen_atomic = set(m for cmb in combined.values() for m in cmb.get("members", []))
        entries = [("combined", cid) for cid in combined.keys()]
        entries.extend(("atomic", aid) for aid in atomic.keys() if aid not in seen_atomic)

        # --- For each atomic not yet combined, check if it connects to an existing combined ---
        for tpe, cid in list(entries):
            if tpe != "atomic":
                continue
            crack = atomic[cid]
            # see if it overlaps/attaches to any combined
            attached_to = None
            for cmb_id, cmb in combined.items():
                for m in cmb.get("members", []):
                    if cracks_overlap_or_connect(crack, atomic.get(m, {})):
                        attached_to = cmb_id
                        break
                if attached_to:
                    break
            if attached_to:
                # extend existing combined by rebuilding with old members + this new one
                members = set(combined[attached_to]["members"])
                '''members.add(cid)
                combined[attached_to] = self._build_combined_crack(sorted(members, key=lambda s: int(s)))'''
                members_clean = [m for m in members if isinstance(m, (str, int)) and str(m).isdigit()]
                if not members_clean:
                    continue
                combined[attached_to] = self._build_combined_crack(sorted(members_clean, key=lambda s: int(s)))
                print(f"Extended combined {attached_to} with atomic {cid}")
            else:
                # check if it overlaps with other "free" atomics → make a new combined
                overlaps = [cid]
                for tpe2, cid2 in entries:
                    if tpe2 == "atomic" and cid2 != cid:
                        if cracks_overlap_or_connect(crack, atomic[cid2]):
                            overlaps.append(cid2)
                if len(overlaps) > 1:
                    new_id = str(max([int(k) for k in combined.keys() if k.isdigit()] or [-1]) + 1)
                    combined[new_id] = self._build_combined_crack(sorted(overlaps, key=lambda s: int(s)))
                    print(f"Auto-created combined {new_id} from atomics {overlaps}")

        self.save_annotation()
        self.change_image()
        
    
    #################################################################################
    # Metrics calculations functions
    #################################################################################
    
    # ---- local helpers
    @staticmethod
    def compute_mask_metrics(gt_mask, pred_mask):
        gt = gt_mask.astype(bool); pr = pred_mask.astype(bool)
        tp = np.logical_and(gt, pr).sum()
        fp = np.logical_and(~gt, pr).sum()
        fn = np.logical_and(gt, ~pr).sum()
        tn = np.logical_and(~gt, ~pr).sum()
        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)
        iou       = tp / (tp + fp + fn + 1e-9)
        return {"precision": precision, "recall": recall, "f1": f1, "iou": iou,
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}

    @staticmethod
    def save_mask_comparison_plot(gt_mask, pred_mask, out_path, show=False):
        gt = gt_mask.astype(bool)
        pr = pred_mask.astype(bool)
        iou = np.logical_and(gt, pr)   # intersection
        oou = np.logical_and(gt, ~pr)  # missed crack
        cou = np.logical_and(~gt, pr)  # false positive
        vis = np.zeros((*gt.shape, 3), dtype=np.uint8)
        vis[iou] = [255, 255, 255]
        vis[oou] = [255,   0,   0]
        vis[cou] = [  0,   0, 255]
        if show:
            plt.figure(figsize=(8, 6))
            plt.imshow(vis); plt.title("Mask Comparison Overlay"); plt.axis("off"); plt.show()
        else:
            plt.imsave(out_path, vis)
            
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

    @staticmethod
    def _reconstruct_full_mask(crack, H, W):
        try:
            return reconstruct_full_mask_from_crack(crack, H, W)
        except Exception:
            mc = crack.get("mask_crop"); bb = crack.get("mask_bbox")
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
        
    @staticmethod
    def normals_from_mask_for_midline(midline_xy, mask, max_radius=50):
        """
        Pixel-accurate version:
        - Polygonizes the mask into exact pixel-boundary polygons using rasterio.
        - Shifts coords by -0.5 so edges align with imshow pixel grid.
        - Intersects midline normals with those polygons so endpoints lie exactly on the mask edge.
        """
        import numpy as np
        import shapely
        from shapely.geometry import shape, LineString, Point, MultiPoint
        import rasterio.features

        H, W = mask.shape
        midline_xy = np.asarray(midline_xy, float)
        if midline_xy.ndim != 2 or midline_xy.shape[1] != 2 or len(midline_xy) < 2:
            n = len(midline_xy) if midline_xy.ndim > 0 else 0
            return (np.full(n, np.nan),) * 5, []

        # tangent + normals
        try:
            from cracktools.segmentation import compute_smooth_tangent_normals
            _, nor = compute_smooth_tangent_normals(midline_xy[:, 0], midline_xy[:, 1])
        except Exception:
            dx, dy = np.gradient(midline_xy[:, 0]), np.gradient(midline_xy[:, 1])
            nrm = np.hypot(dx, dy) + 1e-12
            tan = np.stack([dx/nrm, dy/nrm], axis=1)
            nor = np.stack([-tan[:, 1], tan[:, 0]], axis=1)

        # polygonize mask -> shapely polygons
        mask_bin = (mask > 0).astype(np.uint8)
        polygons = []
        for geom, val in rasterio.features.shapes(mask_bin, mask=mask_bin):
            if val == 1:
                poly = shape(geom)
                # shift by -0.5 in both x and y
                poly = shapely.affinity.translate(poly, xoff=-0.5, yoff=-0.5)
                polygons.append(poly)
        edges = [poly.boundary for poly in polygons]

        N = len(midline_xy)
        e1x = np.full(N, np.nan); e1y = np.full(N, np.nan)
        e2x = np.full(N, np.nan); e2y = np.full(N, np.nan)
        widths_mask = np.full(N, np.nan)

        for i, (p, nvec) in enumerate(zip(midline_xy, nor)):
            if not np.all(np.isfinite(p)) or not np.all(np.isfinite(nvec)):
                continue

            # build long ray
            A = (p[0] - max_radius * nvec[0], p[1] - max_radius * nvec[1])
            B = (p[0] + max_radius * nvec[0], p[1] + max_radius * nvec[1])
            ray = LineString([A, B])

            hits = []
            for edge in edges:
                inter = edge.intersection(ray)
                if inter.is_empty:
                    continue
                if isinstance(inter, Point):
                    hits.append((inter.x, inter.y))
                elif isinstance(inter, MultiPoint):
                    for g in inter.geoms:
                        hits.append((g.x, g.y))
                elif inter.geom_type == "LineString":
                    coords = np.asarray(inter.coords, float)
                    hits.append(tuple(coords[0])); hits.append(tuple(coords[-1]))

            if len(hits) >= 2:
                dists = [np.dot([hx - p[0], hy - p[1]], nvec) for (hx, hy) in hits]
                left_pts = [(hx, hy) for (hx, hy), d in zip(hits, dists) if d < 0]
                right_pts = [(hx, hy) for (hx, hy), d in zip(hits, dists) if d > 0]
                if left_pts and right_pts:
                    lp = max(left_pts, key=lambda q: np.dot([q[0]-p[0], q[1]-p[1]], nvec))
                    rp = min(right_pts, key=lambda q: np.dot([q[0]-p[0], q[1]-p[1]], nvec))
                    e1x[i], e1y[i] = lp
                    e2x[i], e2y[i] = rp
                    widths_mask[i] = np.hypot(rp[0]-lp[0], rp[1]-lp[1])

        return (e1x, e1y, e2x, e2y, widths_mask), polygons

    @staticmethod
    def plot_mask_normals(midline, e1x, e1y, e2x, e2y, mask, contours=None,
                        spacing_px=20, show=True, out_path=None, crack_label=""):
        """
        Plot normals + crack contours (polygons) for visualization.
        - contours: list of Shapely Polygons (from rasterio.features.shapes)
        """
        import matplotlib.pyplot as plt
        import numpy as np

        H, W = mask.shape
        plt.figure(figsize=(8, 8))

        # Force 0 = black, 255 = white
        mask_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        mask_rgb[mask > 0] = [255, 255, 255]

        plt.imshow(mask_rgb, alpha=1.0)  # alpha=1 for full opaque b/w

        # plot polygon contours
        if contours:
            for poly in contours:
                if poly.is_empty:
                    continue
                # in plot_mask_normals when drawing contours
                if poly.geom_type == "Polygon":
                    x, y = poly.exterior.xy
                    plt.plot(np.array(x), np.array(y),
                            color="orange", lw=1.5, alpha=0.8)
                    for interior in poly.interiors:
                        xi, yi = interior.xy
                        plt.plot(np.array(xi), np.array(yi),
                                color="orange", lw=1.5, alpha=0.5)
                elif poly.geom_type == "MultiPolygon":
                    for sub in poly.geoms:
                        x, y = sub.exterior.xy
                        plt.plot(x, y, color="orange", lw=1.0, alpha=0.8)

        # plot midline
        if midline is not None and len(midline) > 1:
            plt.plot(midline[:,0], midline[:,1], 'g-', lw=1.0, label="midline")

        # plot normals
        N = len(midline)
        for i in range(0, N, spacing_px):
            if np.isfinite(e1x[i]) and np.isfinite(e2x[i]):
                plt.plot([e1x[i], e2x[i]], [e1y[i], e2y[i]],
                        color="cyan", lw=0.5, alpha=0.8)
                plt.scatter([e1x[i], e2x[i]], [e1y[i], e2y[i]],
                            c=["red","blue"], s=8, marker="o", alpha=0.7)

        plt.title(f"Mask normals — {crack_label}")
        plt.axis("equal"); plt.legend(); plt.tight_layout()

        if show:
            plt.show()
        elif out_path:
            plt.savefig(out_path, dpi=200); plt.close()
            
    @staticmethod
    def plot_width_differences(midline, w_mask, w_edge, mask, contours=None,
                            spacing_px=20, show=True, out_path=None, crack_label=""):
        """
        Visualize width differences along the midline:
        - Background mask (0=black, 255=white)
        - Midline (green)
        - Points colored by relative error (red=mask wider, blue=edge wider)
        """
        import matplotlib.pyplot as plt
        import numpy as np

        H, W = mask.shape
        # force black/white background
        mask_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        mask_rgb[mask > 0] = [255, 255, 255]

        plt.figure(figsize=(8, 8))
        plt.imshow(mask_rgb, origin="upper")

        # plot contours if available
        if contours:
            for poly in contours:
                if poly.geom_type == "Polygon":
                    x, y = poly.exterior.xy
                    plt.plot(x, y, color="orange", lw=0.8, alpha=0.7)
                    for interior in poly.interiors:
                        xi, yi = interior.xy
                        plt.plot(xi, yi, color="orange", lw=0.5, alpha=0.5)

        if midline is not None and len(midline) > 1:
            plt.plot(midline[:, 0], midline[:, 1], 'g-', lw=1.0, label="midline")

        # compute diffs
        valid = np.isfinite(w_mask) & np.isfinite(w_edge)
        diffs = w_edge - w_mask
        diffs = np.where(valid, diffs, np.nan)

        # color map: red (mask larger), blue (edge larger)
        colors = []
        for d in diffs:
            if np.isnan(d):
                colors.append("gray")
            elif d > 0:
                colors.append("blue")   # edge wider
            else:
                colors.append("red")    # mask wider

        # sample points along midline
        N = len(midline)
        for i in range(0, N, spacing_px):
            if np.isfinite(diffs[i]):
                plt.scatter(midline[i, 0], midline[i, 1],
                            c=colors[i], s=20, marker="o", alpha=0.8)

        plt.title(f"Width comparison — {crack_label}")
        plt.axis("equal"); plt.legend(); plt.tight_layout()

        if show:
            plt.show()
        elif out_path:
            plt.savefig(out_path, dpi=200)
            plt.close()

        return diffs
    
    @staticmethod
    def compare_widths_for_cracks(ann, crack_mask, base_name, metrics_dir, display=True):
        """
        Compare mask-derived vs edge-tracking widths for all cracks.
        - Plots midlines color-coded by signed width difference (edge - mask).
        - Saves summary stats + per-point diffs.
        """
        import numpy as np, matplotlib.pyplot as plt, os, pandas as pd
        from matplotlib.collections import LineCollection

        H, W = crack_mask.shape
        width_rows = []
        diffs_rows = []

        atomic = ann.get("atomic_cracks", {}) or {}
        combined = ann.get("combined_cracks", {}) or {}

        # skip atomics already absorbed in combined
        atomics_in_combined = {m for cmb in combined.values() for m in cmb.get("members", [])}
        all_cracks = [("atomic", cid, crack) for cid, crack in atomic.items() if cid not in atomics_in_combined]
        all_cracks += [("combined", cid, crack) for cid, crack in combined.items()]

        for ctype, cid, crack in all_cracks:
            midline = np.asarray(crack.get("midline", []), float)
            if midline.ndim != 2 or midline.shape[1] != 2 or len(midline) < 3:
                continue

            # mask-based widths
            (_, _, _, _, w_mask), _ = CrackUtils.normals_from_mask_for_midline(
                midline, crack_mask, max_radius=50
            )

            # edge-tracking widths
            ne = crack.get("normal_edge_points")
            w_edge = None
            if ne and isinstance(ne, dict):
                def _to_array(v):
                    if isinstance(v, list) and len(v) == 2 and isinstance(v[0], (list, tuple)):
                        return np.column_stack([v[0], v[1]]).astype(float)
                    return np.array(v, float)
                e1 = _to_array(ne.get("edge1", []))
                e2 = _to_array(ne.get("edge2", []))
                if e1.ndim == 2 and e2.ndim == 2 and len(e1) and len(e2):
                    m = min(len(e1), len(e2), len(w_mask), len(midline))
                    w_edge = np.full(m, np.nan)
                    for i in range(m):
                        if np.all(np.isfinite(e1[i])) and np.all(np.isfinite(e2[i])):
                            w_edge[i] = np.hypot(e1[i,0] - e2[i,0], e1[i,1] - e2[i,1])
                    # trim everything consistently
                    w_mask = w_mask[:m]
                    midline = midline[:m]

            if w_edge is None:
                continue

            valid = np.isfinite(w_mask) & np.isfinite(w_edge)
            n_valid = int(valid.sum())
            if n_valid < 3:
                continue

            diff = w_edge[valid] - w_mask[valid]
            coords = midline[valid]

            # --- add stats row
            width_rows.append({
                "image": base_name, "crack_type": ctype, "crack_id": cid,
                "n_valid": n_valid,
                "mask_width_mean": float(np.mean(w_mask[valid])),
                "edge_width_mean": float(np.mean(w_edge[valid])),
                "width_diff_mae": float(np.mean(np.abs(diff))),
                "width_diff_rmse": float(np.sqrt(np.mean(diff**2))),
                "width_diff_mean": float(np.mean(diff)),
                "width_diff_std": float(np.std(diff)),
                "width_diff_min": float(np.min(diff)),
                "width_diff_max": float(np.max(diff))
            })

            # --- save raw diffs
            diffs_out = os.path.join(metrics_dir, f"{base_name}_{ctype}{cid}_width_diffs.csv")
            pd.DataFrame({
                "mid_x": coords[:,0], "mid_y": coords[:,1],
                "mask_width": w_mask[valid],
                "edge_width": w_edge[valid],
                "width_diff": diff
            }).to_csv(diffs_out, index=False)

            from matplotlib.colors import TwoSlopeNorm

            # --- plot crack with midline color-coded
            if len(coords) > 1:
                segments = np.stack([coords[:-1], coords[1:]], axis=1)

                # get actual min/max
                vmin = np.min(diff)
                vmax = np.max(diff)

                # symmetric scale around 0 so colors are proportional
                max_abs = max(abs(vmin), abs(vmax))
                norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=max_abs)

                mask_rgb = np.zeros((H, W, 3), dtype=np.uint8)
                mask_rgb[crack_mask > 0] = [255, 255, 255]

                plt.figure(figsize=(8, 8))
                plt.imshow(mask_rgb, origin="upper")

                lc = LineCollection(
                    segments, cmap="coolwarm", norm=norm,
                    linewidth=3.0, alpha=0.9
                )
                lc.set_array(diff[:-1])  # color from diffs
                plt.gca().add_collection(lc)

                # colorbar with explicit min/max ticks
                cbar = plt.colorbar(lc, ax=plt.gca(), shrink=0.7)
                cbar.set_label("Width difference (edge - mask) [px]")
                cbar.set_ticks([vmin, 0, vmax])
                cbar.ax.set_yticklabels([f"{vmin:.2f}", "0", f"{vmax:.2f}"])

                plt.title(f"Width diffs — {ctype} {cid}")
                plt.axis("equal"); plt.tight_layout()

                out_plot = os.path.join(metrics_dir, f"{base_name}_{ctype}{cid}_width_diffs.png")
                if display: 
                    plt.show()
                else: 
                    plt.savefig(out_plot, dpi=200); plt.close()

        return width_rows, diffs_rows
    
    
    
    ###############################################################################################
    # Midline Metrics
    ###############################################################################################
    # ---------- small utils ----------
    @staticmethod
    def _finite_xy(arr):
        if arr is None: return np.empty((0,2), float)
        a = np.asarray(arr, float)
        if a.ndim != 2 or a.shape[1] != 2: return np.empty((0,2), float)
        m = np.all(np.isfinite(a), axis=1)
        a = a[m]
        # drop exact duplicates in sequence
        if len(a) > 1:
            keep = [0]
            for i in range(1, len(a)):
                if not (abs(a[i,0]-a[i-1,0]) < 1e-12 and abs(a[i,1]-a[i-1,1]) < 1e-12):
                    keep.append(i)
            a = a[keep]
        return a

    @staticmethod
    def _split_nan_none(arr):
        """Split polyline on [None,None] or NaNs into contiguous segments."""
        a = np.asarray(arr, float)
        if a.ndim != 2 or a.shape[1] != 2: return []
        bad = ~np.isfinite(a).all(axis=1)
        idx = np.where(bad)[0]
        pieces, start = [], 0
        for k in idx:
            if k - start >= 2:
                pieces.append(a[start:k])
            start = k+1
        if len(a) - start >= 2:
            pieces.append(a[start:])
        return pieces

    @staticmethod
    def _resample_by_arclen(xy, N=200):
        """Uniform arclength resample (handles multiple segments)."""
        segs = CrackUtils._split_nan_none(xy) if np.any(~np.isfinite(xy)) else [xy]
        out = []
        for s in segs:
            s = CrackUtils._finite_xy(s)
            if len(s) < 2: continue
            d = np.sqrt(((s[1:]-s[:-1])**2).sum(1))
            L = np.concatenate([[0], np.cumsum(d)])
            if L[-1] <= 1e-9:
                continue
            t = np.linspace(0, L[-1], max(2, int(N * (L[-1]/sum(max(1e-9, CrackUtils._len_seg(CrackUtils._finite_xy(u))) for u in segs)))))
            xi = np.interp(t, L, s[:,0])
            yi = np.interp(t, L, s[:,1])
            out.append(np.column_stack([xi, yi]))
        if not out:
            return np.empty((0,2), float)
        return np.vstack(out)

    @staticmethod
    def _len_seg(xy):
        if xy is None or len(xy) < 2: return 0.0
        return float(np.sqrt(((xy[1:]-xy[:-1])**2).sum(1)).sum())

    # --- DROP-IN REPLACEMENT in crackutils.py ---
    @staticmethod
    def _nn_dists(A, B):
        """
        Compute nearest-neighbor distances from each point in A to the closest point in B.
        Automatically uses GPU (CuPy + cupyx.scipy.spatial.cKDTree) if available,
        otherwise falls back to SciPy's CPU cKDTree.

        Returns
        -------
        dists : np.ndarray of shape (len(A),)
            Euclidean distances.
        """
        import numpy as np
        try:
            import cupy as cp
            from cupyx.scipy.spatial import cKDTree as GPU_KDTree
            CUPY_AVAILABLE = True
            gpu = True
            try:
                # detect CUDA presence
                _ = cp.cuda.runtime.getDeviceCount()
                if _ <= 0:
                    gpu = False
            except Exception:
                gpu = False
        except ImportError:
            CUPY_AVAILABLE = False
            gpu = False

        if A is None or B is None or len(A) == 0 or len(B) == 0:
            return np.zeros((len(A),), dtype=float)

        if gpu:
            try:
                A_gpu = cp.asarray(A, dtype=cp.float32)
                B_gpu = cp.asarray(B, dtype=cp.float32)
                tree = GPU_KDTree(B_gpu)
                dists, _ = tree.query(A_gpu, k=1)
                return cp.asnumpy(dists)
            except Exception as e:
                print(f"[nn_dists][warn] GPU KDTree failed → falling back to CPU: {e}")

        # CPU fallback (SciPy)
        try:
            from scipy.spatial import cKDTree as CPU_KDTree
            tree = CPU_KDTree(B)
            dists, _ = tree.query(A, k=1)
            return dists
        except Exception as e:
            print(f"[nn_dists][warn] CPU KDTree failed, using brute force: {e}")
            A = np.asarray(A, float)
            B = np.asarray(B, float)
            diff = A[:, None, :] - B[None, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            return np.min(dists, axis=1)

    @staticmethod
    def chamfer_symmetric(A, B):
        """Mean NN distance both directions."""
        A = CrackUtils._finite_xy(A); B = CrackUtils._finite_xy(B)
        return float(np.mean(CrackUtils._nn_dists(A,B))) + float(np.mean(CrackUtils._nn_dists(B,A)))

    @staticmethod
    def hausdorff_symmetric(A, B):
        """Max directed NN both ways."""
        A = CrackUtils._finite_xy(A); B = CrackUtils._finite_xy(B)
        da = CrackUtils._nn_dists(A, B); db = CrackUtils._nn_dists(B, A)
        if da.size == 0 or db.size == 0:
            return float('inf')
        return float(max(da.max(), db.max()))

    # --- DROP-IN REPLACEMENT in crackutils.py ---

    @staticmethod
    def frechet_discrete(A, B, max_points=800):
        """
        Iterative Eiter–Mannila discrete Fréchet distance.
        - No recursion (avoids RecursionError)
        - Resamples long polylines to <= max_points for robustness
        """
        A = CrackUtils._finite_xy(A)
        B = CrackUtils._finite_xy(B)
        if len(A) == 0 or len(B) == 0:
            return float('inf')

        # Optional safety downsampling by arclength (keeps geometry)
        if len(A) > max_points:
            A = CrackUtils._resample_by_arclen(A, N=max_points)
        if len(B) > max_points:
            B = CrackUtils._resample_by_arclen(B, N=max_points)

        n, m = len(A), len(B)
        # DP table of size (n x m)
        ca = np.full((n, m), np.inf, dtype=float)

        # helper to compute Euclidean distance quickly
        def dist(i, j):
            dx = A[i, 0] - B[j, 0]
            dy = A[i, 1] - B[j, 1]
            return np.hypot(dx, dy)

        ca[0, 0] = dist(0, 0)
        # first column
        for i in range(1, n):
            ca[i, 0] = max(ca[i-1, 0], dist(i, 0))
        # first row
        for j in range(1, m):
            ca[0, j] = max(ca[0, j-1], dist(0, j))

        # fill DP
        for i in range(1, n):
            Ai = A[i]  # small locality win
            for j in range(1, m):
                d = np.hypot(Ai[0] - B[j, 0], Ai[1] - B[j, 1])
                ca[i, j] = max(min(ca[i-1, j], ca[i-1, j-1], ca[i, j-1]), d)

        return float(ca[n-1, m-1])

    @staticmethod
    def tangent_angles(xy):
        xy = CrackUtils._finite_xy(xy)
        if len(xy) < 2: return np.array([])
        d = np.gradient(xy, axis=0)
        ang = np.arctan2(d[:,1], d[:,0])
        return ang

    @staticmethod
    def angle_error_degrees(A, B):
        # resample to same count for angle comparison
        Ar = CrackUtils._resample_by_arclen(A, N=400)
        Br = CrackUtils._resample_by_arclen(B, N=len(Ar))
        if len(Ar)==0 or len(Br)==0: return np.nan
        aA = CrackUtils.tangent_angles(Ar); aB = CrackUtils.tangent_angles(Br)
        n = min(len(aA), len(aB))
        if n == 0: return np.nan
        da = np.abs(np.unwrap(aA[:n]) - np.unwrap(aB[:n]))
        da = np.mod(da + pi, 2*pi) - pi
        return float(np.degrees(np.mean(np.abs(da))))

    @staticmethod
    def orthogonal_deviation(manual_xy, auto_xy, N=400, robust='median'):
        """Signed distance from manual to nearest auto, measured along manual normal."""
        M = CrackUtils._resample_by_arclen(manual_xy, N=N)
        A = CrackUtils._finite_xy(auto_xy)
        if len(M)==0 or len(A)==0:
            return dict(mean=np.nan, median=np.nan, rmse=np.nan, p95=np.nan)
        # manual normals
        d = np.gradient(M, axis=0)  # tangents
        norm = np.column_stack([-d[:,1], d[:,0]])
        nlen = np.maximum(1e-9, np.sqrt((norm**2).sum(1)))
        n = norm / nlen[:,None]
        # nearest auto → signed projection
        d2 = ((M[:,None,:] - A[None,:,:])**2).sum(2)
        idx = d2.argmin(1)
        v = A[idx] - M
        signed = (v * n).sum(1)
        absd = np.abs(signed)
        out = dict(
            mean=float(np.mean(signed)),
            median=float(np.median(signed)),
            rmse=float(np.sqrt(np.mean(absd**2))),
            p95=float(np.percentile(absd, 95))
        )
        return out

    @staticmethod
    def coverage_at_tau(A, B, tau_px=3.0):
        A = CrackUtils._finite_xy(A); B = CrackUtils._finite_xy(B)
        if len(A)==0 or len(B)==0: return dict(A_to_B=0.0, B_to_A=0.0)
        da = CrackUtils._nn_dists(A,B); db = CrackUtils._nn_dists(B,A)
        return dict(
            A_to_B=float(np.mean(da <= tau_px)),
            B_to_A=float(np.mean(db <= tau_px))
        )

    @staticmethod
    def length_ratio(A, B):
        La = CrackUtils._len_seg(CrackUtils._finite_xy(A)); Lb = CrackUtils._len_seg(CrackUtils._finite_xy(B))
        if La < 1e-9: return np.nan
        return float(abs(Lb - La) / La)

    @staticmethod
    def mask_iou(m1, m2):
        if m1 is None or m2 is None: return np.nan
        a = (np.asarray(m1)>0).astype(np.uint8)
        b = (np.asarray(m2)>0).astype(np.uint8)
        inter = int((a & b).sum())
        union = int((a | b).sum())
        return (inter / union) if union else float('nan')

    # --- DROP-IN REPLACEMENT in crackutils.py ---
    @staticmethod
    def compute_midline_metrics(auto_xy, man_xy, tau=3.0):
        """
        Return a dict with centerline metrics.
        Robust: guards Fréchet against huge inputs and exceptions.
        """
        A = CrackUtils._finite_xy(man_xy)
        B = CrackUtils._finite_xy(auto_xy)

        # Light downsampling for expensive ops
        A_ds = CrackUtils._resample_by_arclen(A, N=min(600, len(A) or 0))
        B_ds = CrackUtils._resample_by_arclen(B, N=min(600, len(B) or 0))

        out = {
            "chamfer_mean": float(CrackUtils.chamfer_symmetric(A, B)),
            "hausdorff": float(CrackUtils.hausdorff_symmetric(A, B)),
            "frechet_discrete": float('nan'),
            "orth_dev": CrackUtils.orthogonal_deviation(A, B, N=400),
            "angle_err_deg": float(CrackUtils.angle_error_degrees(A, B)),
            "length_ratio": float(CrackUtils.length_ratio(A, B)),
            "coverage": CrackUtils.coverage_at_tau(A, B, tau_px=tau)
        }

        # Safe Fréchet (iterative DP) on the downsampled curves
        try:
            if len(A_ds) >= 2 and len(B_ds) >= 2:
                out["frechet_discrete"] = float(CrackUtils.frechet_discrete(A_ds, B_ds, max_points=800))
        except Exception as e:
            # Don't crash metrics; just record NaN and log
            print(f"[metrics][warn] Fréchet failed: {e}")

        return out

    @staticmethod
    def widths_from_normals(n1_xy, n2_xy):
        """Take Nx2 arrays; return width vector (min-aligned length)."""
        n1 = CrackUtils._finite_xy(n1_xy); n2 = CrackUtils._finite_xy(n2_xy)
        m = min(len(n1), len(n2))
        if m < 2: return np.array([])
        d = np.sqrt(((n1[:m] - n2[:m])**2).sum(1))
        return d[np.isfinite(d)]

    @staticmethod
    def compare_widths(w_ref, w_pred):
        if w_ref.size == 0 or w_pred.size == 0: 
            return dict(MAE=np.nan, RMSE=np.nan, corr=np.nan)
        m = min(len(w_ref), len(w_pred))
        wr = w_ref[:m]; wp = w_pred[:m]
        mae = float(np.mean(np.abs(wr-wp)))
        rmse = float(np.sqrt(np.mean((wr-wp)**2)))
        corr = float(np.corrcoef(wr, wp)[0,1]) if m>2 else np.nan
        return dict(MAE=mae, RMSE=rmse, corr=corr)

    @staticmethod
    def _auto_cache_key(self):
        # Include any params that affect auto generation
        parts = [
            "v1",
            f"down={getattr(self, 'downsample_factor_box', None).value() if hasattr(self,'downsample_factor_box') else 'na'}",
            f"mu={getattr(self,'mu_box',None).value() if hasattr(self,'mu_box') else 'na'}",
            f"l={getattr(self,'l_box',None).value() if hasattr(self,'l_box') else 'na'}",
            f"p={getattr(self,'p_box',None).value() if hasattr(self,'p_box') else 'na'}",
            f"color={getattr(self, 'edge_track_color_box', None).currentText() if hasattr(self,'edge_track_color_box') else 'G'}"
        ]
        return "|".join(map(str, parts))

    def debug_plot_midlines(self, crack_id, cache_key=None, show_gt=True, save_path=None):
        """
        Plot overlay of manual vs auto midline (and GT mask if available).
        """
        ann = self.annotation.get("annotations", {})
        atomic = ann.get("atomic_cracks", {})
        crack = atomic.get(str(crack_id))
        if crack is None:
            print(f"⚠️ No crack {crack_id} found")
            return

        cache_key = cache_key or CrackUtils._auto_cache_key(self)
        auto_var = crack.get("variants", {}).get("auto", {}).get(cache_key)

        man_xy = np.array(crack.get("midline", []), float)
        auto_xy = np.array(auto_var["midline"], float) if auto_var else np.empty((0,2))

        # Start with image
        im = self.original_image.copy()

        # Optional GT mask outline
        if show_gt and getattr(self, "current_mask", None) is not None:
            contours, _ = cv2.findContours(self.current_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(im, contours, -1, (0,0,255), 1)  # blue outline

        # Plot with matplotlib
        plt.figure(figsize=(8,8))
        plt.imshow(im)
        if len(man_xy) > 1:
            plt.plot(man_xy[:,0], man_xy[:,1], 'g-', linewidth=2, label="Manual (GT midline)")
        if len(auto_xy) > 1:
            plt.plot(auto_xy[:,0], auto_xy[:,1], 'r-', linewidth=2, label="Auto midline")
        plt.legend()
        plt.title(f"Crack {crack_id} - cache_key={cache_key}")
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"✅ Saved debug plot → {save_path}")
        else:
            plt.show()

    # ---------------- JSON helpers (static) ----------------
    @staticmethod
    def _to_py(obj, ndigits=6):
        """
        Recursively converts NumPy / CuPy arrays, pandas, etc. to plain
        Python lists, rounding floats to `ndigits` decimals for compact JSON.
        """
        import numpy as np

        if obj is None:
            return None
        if isinstance(obj, (int, bool, str)):
            return obj
        if isinstance(obj, float):
            # round floats directly
            return round(obj, ndigits)
        if isinstance(obj, (list, tuple)):
            return [CrackUtils._to_py(x, ndigits) for x in obj]
        if isinstance(obj, dict):
            return {k: CrackUtils._to_py(v, ndigits) for k, v in obj.items()}
        if hasattr(obj, "tolist"):  # numpy / cupy array
            return CrackUtils._to_py(obj.tolist(), ndigits)
        return obj

    @staticmethod
    def safe_json_dump(data, path, compact=True):
        """Atomic JSON writer — supports compact (semi-human) or fully minified mode."""
        import os, tempfile, json
        d = CrackUtils._to_py(data)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path)+".", suffix=".tmp",
                                dir=os.path.dirname(path))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                if compact:
                    # minimal spacing, arrays inline
                    json.dump(d, f, ensure_ascii=False, indent=None, separators=(',', ':'))
                else:
                    # readable multi-line, smaller indent
                    json.dump(d, f, ensure_ascii=False, indent=1)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception:
            try: os.remove(tmp)
            except Exception: pass
            raise

    @staticmethod
    def _normals_to_json(normals, xmin, ymin, ndigits=ROUNDING_DIGITS):
        import numpy as np

        def to_xy2(arr):
            a = np.asarray(arr, float)
            if a.ndim == 2 and a.shape[1] == 2:       # already Nx2
                x, y = a[:, 0], a[:, 1]
            elif a.ndim == 2 and a.shape[0] == 2:     # 2xN ( [xlist, ylist] )
                x, y = a[0], a[1]
            elif a.ndim == 1:                         # degenerate 1-D
                x, y = a, np.full_like(a, np.nan, dtype=float)
            else:
                x = y = np.array([], float)

            x = x + float(xmin)
            y = y + float(ymin)

            out = np.stack([x, y], axis=1)
            # round
            if np.isfinite(out).any():
                out = np.round(out, ndigits=ndigits, where=np.isfinite(out))
            # JSON-safe NaNs
            out[~np.isfinite(out)] = None
            return out.tolist()

        # dict form: {"edge1":[xlist,ylist], "edge2":[xlist,ylist]} or {"edge1":Nx2,...}
        if isinstance(normals, dict):
            e1 = normals.get("edge1", [])
            e2 = normals.get("edge2", [])
            # accept either [xlist,ylist] or Nx2
            e1 = e1 if isinstance(e1, (list, tuple)) else []
            e2 = e2 if isinstance(e2, (list, tuple)) else []
            e1 = to_xy2(e1)
            e2 = to_xy2(e2)
            return {"edge1": e1, "edge2": e2}

        # tuple/list form: ((e1x,e1y), (e2x,e2y))
        try:
            (e1x, e1y), (e2x, e2y) = normals
            return {"edge1": to_xy2([e1x, e1y]), "edge2": to_xy2([e2x, e2y])}
        except Exception:
            return {"edge1": [], "edge2": []}
