import cracktools as ct
from helpers.crackhelpers import *

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

#This class is basically is all of the utility / save and load or unimportant functions that aren't directly accessible via a ui button or aren't important
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
    
    # crackutils.py  --- CrackUtils.change_image  (REPLACE WHOLE METHOD)
    def change_image(self):
        import os, json, cv2
        import numpy as np
        from skimage.segmentation import mark_boundaries  # for colored boundaries

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
            if mask_path:
                if mask_path.endswith('.npy'):
                    mask = np.load(mask_path)
                    mask = (mask > 0).astype(np.uint8) if mask.max() > 1 else mask.astype(np.uint8)
                else:
                    mask = cv2.imread(mask_path, 0)
                    mask = (mask > 0).astype(np.uint8) if mask is not None else None
                self.current_mask = mask

        im = self.original_image.copy()

        # -------- Load annotation data --------
        self.ann_name = os.path.join(self.save_folder, base_name + '.json')
        self.mask_name_bin = os.path.join(self.save_folder, base_name + '_mask.png')
        self.mask_name_255 = os.path.join(self.save_folder, base_name + '_mask255.png')
        self.mask = []               # in-memory rendered masks for this image
        self.annotation = {}

        H, W = im.shape[:2]

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

            # ---- Atomic cracks: cyan boundaries, midline (teal), edges (yellow), endpoints (magenta)
            for crack_id, crack in atomic.items():
                mask_full = reconstruct_full_mask_from_crack(crack, H, W)
                if np.any(mask_full):
                    # cyan-ish boundary
                    im = (mark_boundaries(im/255.0, (mask_full>0).astype(np.uint8),
                                        color=(0.0, 1.0, 1.0), background_label=0)*255).astype(np.uint8)
                    self.mask.append(mask_full)

                midline = np.array(crack.get("midline", []), dtype=float)
                if len(midline) > 1:
                    for i in range(1, len(midline)):
                        pt1 = (int(round(midline[i-1][0])), int(round(midline[i-1][1])))
                        pt2 = (int(round(midline[i][0])),  int(round(midline[i][1])))
                        cv2.line(im, pt1, pt2, (0,200,200), 2)

                edges = crack.get("geodesic_edges", {}) or {}
                for _, edge_pts in edges.items():
                    edge_pts = np.array(edge_pts, dtype=float)
                    if len(edge_pts) > 1:
                        for i in range(1, len(edge_pts)):
                            pt1 = (int(round(edge_pts[i-1][0])), int(round(edge_pts[i-1][1])))
                            pt2 = (int(round(edge_pts[i][0])),  int(round(edge_pts[i][1])))
                            cv2.line(im, pt1, pt2, (255,255,0), 2)

                up = crack.get("user_points", []) or []
                for p in up:
                    x, y = int(round(p[0])), int(round(p[1]))
                    if 0 <= x < W and 0 <= y < H:
                        endpoint_radius = max(3, int(min(H, W) * 0.0035))
                        cv2.circle(im, (x, y), endpoint_radius, (255, 0, 255), -1)

            # ---- Combined cracks: light red boundaries (pink)
            for crack_id, crack in list(combined.items()):
                members = [m for m in crack.get("members", []) if m in atomic]
                crack["members"] = members  # keep in sync in memory
                if not members:
                    continue  # skip rendering if no members remain
                mask_full = reconstruct_full_mask_from_crack(crack, H, W)
                if np.any(mask_full):
                    im = (mark_boundaries(im/255.0, (mask_full>0).astype(np.uint8),
                                        color=(1.0, 0.6, 0.6), background_label=0)*255).astype(np.uint8)

        # ---- Render main image to screen ----
        _, pixmap = numpy_to_qimage_and_scaled_pixmap(
            im.astype(np.uint8), self.ImageScreen.width(), self.ImageScreen.height(), is_gray=False
        )
        self.ImageScreen.setPixmap(pixmap)

        # ---- Update "all segments" preview (right-hand panel) ----
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

