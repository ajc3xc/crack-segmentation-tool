#!/usr/bin/env python3
"""
Improved Crack Mask Brush Editor

Changes:
- Higher mouse sampling density
- Brush max size = 80
- Controls panel permanently visible
- M toggles overlay/mask
- T toggles image-only/overlay
"""

import os
import sys
import numpy as np
import cv2

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt


# ============================================================
# Utilities
# ============================================================

def blend_overlay(image, mask, w_img=0.6, w_mask=0.4):
    H, W = image.shape[:2]
    mask_rgb = np.zeros((H, W, 3), np.uint8)
    mask_rgb[mask > 0] = (255, 255, 255)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return cv2.addWeighted(image, w_img, mask_rgb, w_mask, 0)


def mask_only(mask):
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
    rgb[mask > 0] = (255, 255, 255)
    return rgb


def to_qimage(img):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    return QtGui.QImage(img.data, w, h, img.strides[0],
                        QtGui.QImage.Format_RGB888).copy()


# ============================================================
# Editor Widget
# ============================================================

class ImageMaskEditor(QtWidgets.QLabel):

    def __init__(self, image, mask):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self.overlay_mode = True
        self.show_image_only = False
        self.brush_radius = 12
        self.paint_value = 255
        self.painting = False
        self.last_point = None

        self.undo_stack = []
        self.undo_limit = 40

        # View transform state (cursor-centered zoom).
        self.view_zoom = 1.0
        self.view_zoom_min = 0.5
        self.view_zoom_max = 40.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        self.setMinimumSize(600, 600)
        self.load_image_and_mask(image, mask)

    def load_image_and_mask(self, image, mask):
        self.image = image.astype(np.uint8)
        self.mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        self.original_mask = self.mask.copy()
        self.undo_stack.clear()
        self.overlay_mode = True
        self.show_image_only = False
        self.painting = False
        self.last_point = None
        self.view_zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.refresh()

    def _base_scale(self):
        img_h, img_w = self.image.shape[:2]
        lbl_w, lbl_h = max(1, self.width()), max(1, self.height())
        return min(lbl_w / max(1, img_w), lbl_h / max(1, img_h))

    def _effective_scale(self):
        return self._base_scale() * float(self.view_zoom)

    def _enforce_pan_bounds(self):
        img_h, img_w = self.image.shape[:2]
        lbl_w, lbl_h = max(1, self.width()), max(1, self.height())
        s = max(1e-8, self._effective_scale())
        disp_w = float(img_w) * s
        disp_h = float(img_h) * s

        if disp_w <= lbl_w:
            self.pan_x = (lbl_w - disp_w) * 0.5
        else:
            min_x = lbl_w - disp_w
            self.pan_x = min(0.0, max(min_x, self.pan_x))

        if disp_h <= lbl_h:
            self.pan_y = (lbl_h - disp_h) * 0.5
        else:
            min_y = lbl_h - disp_h
            self.pan_y = min(0.0, max(min_y, self.pan_y))

    def _screen_to_image(self, pos):
        s = max(1e-8, self._effective_scale())
        ix = (float(pos.x()) - self.pan_x) / s
        iy = (float(pos.y()) - self.pan_y) / s
        return ix, iy

    def _zoom_at(self, pos, factor):
        factor = float(factor)
        if factor <= 0:
            return

        old_zoom = float(self.view_zoom)
        new_zoom = max(self.view_zoom_min, min(self.view_zoom_max, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-9:
            return

        base = self._base_scale()
        old_s = max(1e-8, base * old_zoom)
        new_s = max(1e-8, base * new_zoom)

        mx, my = float(pos.x()), float(pos.y())
        ix = (mx - self.pan_x) / old_s
        iy = (my - self.pan_y) / old_s

        self.view_zoom = new_zoom
        self.pan_x = mx - ix * new_s
        self.pan_y = my - iy * new_s
        self._enforce_pan_bounds()
        self.refresh()

    # ========================================================
    # Brush
    # ========================================================

    def set_brush_radius(self, r):
        self.brush_radius = max(1, min(80, int(r)))

    def push_undo(self):
        if len(self.undo_stack) >= self.undo_limit:
            self.undo_stack.pop(0)
        self.undo_stack.append(self.mask.copy())

    def undo(self):
        if self.undo_stack:
            self.mask = self.undo_stack.pop()
            self.refresh()

    def reset_mask(self):
        self.push_undo()
        self.mask = self.original_mask.copy()
        self.refresh()

    # ========================================================
    # Painting (Higher Density Interpolation)
    # ========================================================

    def mousePressEvent(self, event):
        # In image-only mode, editing is disabled.
        if self.show_image_only and event.button() in (Qt.LeftButton, Qt.RightButton):
            return

        if event.button() == Qt.LeftButton:
            self.paint_value = 255
        elif event.button() == Qt.RightButton:
            self.paint_value = 0
        else:
            return

        self.push_undo()
        self.painting = True
        self.last_point = None
        self.paint_at(event.pos())

    def mouseMoveEvent(self, event):
        if self.painting and (not self.show_image_only):
            self.paint_at(event.pos())

    def mouseReleaseEvent(self, event):
        self.painting = False
        self.last_point = None

    def paint_at(self, pos):
        img_h, img_w = self.image.shape[:2]
        ixf, iyf = self._screen_to_image(pos)
        ix = int(np.round(ixf))
        iy = int(np.round(iyf))

        if not (0 <= ix < img_w and 0 <= iy < img_h):
            return

        if self.last_point is None:
            cv2.circle(self.mask, (ix, iy),
                       self.brush_radius, self.paint_value, -1)
        else:
            # Dense interpolation
            x0, y0 = self.last_point
            x1, y1 = ix, iy
            dist = int(np.hypot(x1 - x0, y1 - y0))
            steps = max(1, dist // 1)  # max density
            for i in range(steps + 1):
                t = i / steps
                xi = int(x0 + t * (x1 - x0))
                yi = int(y0 + t * (y1 - y0))
                cv2.circle(self.mask, (xi, yi),
                           self.brush_radius, self.paint_value, -1)

        self.last_point = (ix, iy)
        self.refresh()

    # ========================================================
    # Display
    # ========================================================

    def refresh(self):
        if self.show_image_only:
            display = self.image.copy()
        elif self.overlay_mode:
            display = blend_overlay(self.image, self.mask)
        else:
            display = mask_only(self.mask)

        lbl_w, lbl_h = max(1, self.width()), max(1, self.height())
        self._enforce_pan_bounds()
        s = float(self._effective_scale())

        M = np.float32([[s, 0.0, float(self.pan_x)],
                        [0.0, s, float(self.pan_y)]])
        canvas = cv2.warpAffine(
            display,
            M,
            (lbl_w, lbl_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        self._draw_minimap(canvas, display)

        self.setPixmap(QtGui.QPixmap.fromImage(to_qimage(canvas)))

    def _draw_minimap(self, canvas, src):
        ch, cw = canvas.shape[:2]
        ih, iw = src.shape[:2]
        if ch < 120 or cw < 180 or ih < 2 or iw < 2:
            return

        pad = 10
        max_w = min(260, max(120, int(cw * 0.24)))
        max_h = min(200, max(90, int(ch * 0.24)))
        aspect = float(iw) / float(max(ih, 1))
        mini_w = max_w
        mini_h = int(round(mini_w / max(aspect, 1e-8)))
        if mini_h > max_h:
            mini_h = max_h
            mini_w = int(round(mini_h * aspect))
        mini_w = max(80, min(mini_w, cw - 2 * pad))
        mini_h = max(60, min(mini_h, ch - 2 * pad))

        x0 = cw - mini_w - pad
        y0 = pad
        x1 = x0 + mini_w
        y1 = y0 + mini_h

        mini = cv2.resize(src, (mini_w, mini_h), interpolation=cv2.INTER_AREA)
        canvas[y0:y1, x0:x1] = mini
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 255, 255), 1)

        s = max(1e-8, self._effective_scale())
        vx0 = (0.0 - self.pan_x) / s
        vy0 = (0.0 - self.pan_y) / s
        vx1 = (float(cw) - self.pan_x) / s
        vy1 = (float(ch) - self.pan_y) / s
        vx0 = float(np.clip(vx0, 0.0, float(iw)))
        vy0 = float(np.clip(vy0, 0.0, float(ih)))
        vx1 = float(np.clip(vx1, 0.0, float(iw)))
        vy1 = float(np.clip(vy1, 0.0, float(ih)))
        if vx1 < vx0:
            vx0, vx1 = vx1, vx0
        if vy1 < vy0:
            vy0, vy1 = vy1, vy0

        rx0 = x0 + int(round((vx0 / float(iw)) * mini_w))
        ry0 = y0 + int(round((vy0 / float(ih)) * mini_h))
        rx1 = x0 + int(round((vx1 / float(iw)) * mini_w))
        ry1 = y0 + int(round((vy1 / float(ih)) * mini_h))
        rx0 = int(np.clip(rx0, x0, x1 - 1))
        ry0 = int(np.clip(ry0, y0, y1 - 1))
        rx1 = int(np.clip(rx1, rx0 + 1, x1))
        ry1 = int(np.clip(ry1, ry0 + 1, y1))
        cv2.rectangle(canvas, (rx0, ry0), (rx1, ry1), (0, 255, 255), 1)
        cv2.putText(canvas, "Minimap", (x0 + 6, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0 and not event.pixelDelta().isNull():
            delta = event.pixelDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        self._zoom_at(event.pos(), factor)
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._enforce_pan_bounds()
        self.refresh()

    def keyPressEvent(self, event):
        win = self.window()
        if event.key() in (Qt.Key_M,):
            self.overlay_mode = not self.overlay_mode
            if self.show_image_only:
                self.show_image_only = False
            self.refresh()
        elif event.key() == Qt.Key_T:
            # Toggle image-only vs overlay view.
            self.show_image_only = not self.show_image_only
            if self.show_image_only:
                self.painting = False
                self.last_point = None
            self.refresh()
        elif event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            anchor = self.mapFromGlobal(QtGui.QCursor.pos())
            if not self.rect().contains(anchor):
                anchor = QtCore.QPoint(self.width() // 2, self.height() // 2)
            self._zoom_at(anchor, 1.15)
        elif event.key() in (Qt.Key_Minus, Qt.Key_Underscore):
            anchor = self.mapFromGlobal(QtGui.QCursor.pos())
            if not self.rect().contains(anchor):
                anchor = QtCore.QPoint(self.width() // 2, self.height() // 2)
            self._zoom_at(anchor, 1.0 / 1.15)
        elif event.key() == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            self.undo()
        elif event.key() == Qt.Key_R:
            self.reset_mask()
        elif event.key() == Qt.Key_S:
            if hasattr(win, "save_mask"):
                win.save_mask()
        elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if hasattr(win, "save_and_close"):
                win.save_and_close()
            elif hasattr(win, "save_mask"):
                win.save_mask()
                if win is not None:
                    win.close()
        elif event.key() in (Qt.Key_Escape, Qt.Key_Q):
            if win is not None:
                win.close()
        else:
            super().keyPressEvent(event)


# ============================================================
# Main Window
# ============================================================

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, image, mask, out_path):
        super().__init__()
        self.setWindowTitle("Mask Brush Editor")

        self.out_path = out_path
        self.editor = ImageMaskEditor(image, mask)
        self._build_ui()
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            self.resize(screen.availableGeometry().size())
        # Run the same path as pressing "R" once startup/init is complete.
        QtCore.QTimer.singleShot(0, self.editor.reset_mask)

    def _build_ui(self):
        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self.editor, 1)

        side_panel = QtWidgets.QWidget()
        side_panel.setFixedWidth(250)
        side_layout = QtWidgets.QVBoxLayout(side_panel)
        side_layout.setContentsMargins(10, 10, 10, 10)

        info = QtWidgets.QLabel(
            "Controls:\n"
            "Left drag  = draw\n"
            "Right drag = erase\n"
            "Ctrl+Z     = undo\n"
            "R          = reset\n"
            "S          = save\n"
            "Enter      = save and close\n"
            "M          = toggle overlay/mask\n"
            "T          = toggle image-only/overlay\n"
            "Image-only = drawing disabled\n"
            "Esc/Q      = close without saving"
        )
        info.setWordWrap(True)
        info.setStyleSheet("font-size: 12px;")
        side_layout.addWidget(info)
        side_layout.addStretch(1)

        layout.addWidget(side_panel, 0)
        self.setCentralWidget(central)

        tb = QtWidgets.QToolBar()
        self.addToolBar(tb)

        tb.addAction("Save (S)", self.save_mask)
        tb.addAction("Save+Close (Enter)", self.save_and_close)
        tb.addAction("Undo (Ctrl+Z)", self.editor.undo)
        tb.addAction("Reset (R)", self.editor.reset_mask)

        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel(" Brush: "))

        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(80)
        self.slider.setValue(self.editor.brush_radius)
        self.slider.valueChanged.connect(self.editor.set_brush_radius)
        tb.addWidget(self.slider)

        self.spin = QtWidgets.QSpinBox()
        self.spin.setMinimum(1)
        self.spin.setMaximum(80)
        self.spin.setValue(self.editor.brush_radius)
        self.spin.valueChanged.connect(self.editor.set_brush_radius)
        self.spin.valueChanged.connect(self.slider.setValue)
        self.slider.valueChanged.connect(self.spin.setValue)
        tb.addWidget(self.spin)

    def save_mask(self):
        cv2.imwrite(self.out_path, self.editor.mask)
        self.statusBar().showMessage(f"Saved to {self.out_path}", 3000)

    def save_and_close(self):
        self.save_mask()
        self.close()


# ============================================================
# Dialog API for embedding in an existing Qt application
# ============================================================

class MaskEditorDialog(QtWidgets.QDialog):
    def __init__(self, image, mask, out_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Modify Ground Truth Mask")
        self.out_path = out_path
        self.editor = ImageMaskEditor(image, mask)
        self.saved = False
        self._build_ui()

        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            self.resize(int(avail.width() * 0.90), int(avail.height() * 0.90))

        QtCore.QTimer.singleShot(0, self.editor.reset_mask)

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        controls = QtWidgets.QHBoxLayout()
        btn_save = QtWidgets.QPushButton("Save (S)")
        self.btn_save = btn_save
        btn_save_close = QtWidgets.QPushButton("Save+Close (Enter)")
        btn_cancel = QtWidgets.QPushButton("Cancel (Esc)")
        btn_undo = QtWidgets.QPushButton("Undo (Ctrl+Z)")
        btn_reset = QtWidgets.QPushButton("Reset (R)")
        controls.addWidget(btn_save)
        controls.addWidget(btn_save_close)
        controls.addWidget(btn_cancel)
        controls.addWidget(btn_undo)
        controls.addWidget(btn_reset)
        controls.addStretch(1)
        controls.addWidget(QtWidgets.QLabel("Brush:"))

        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(80)
        self.slider.setValue(self.editor.brush_radius)
        self.spin = QtWidgets.QSpinBox()
        self.spin.setMinimum(1)
        self.spin.setMaximum(80)
        self.spin.setValue(self.editor.brush_radius)
        self.slider.valueChanged.connect(self.editor.set_brush_radius)
        self.slider.valueChanged.connect(self.spin.setValue)
        self.spin.valueChanged.connect(self.slider.setValue)
        self.spin.valueChanged.connect(self.editor.set_brush_radius)
        controls.addWidget(self.slider)
        controls.addWidget(self.spin)
        root.addLayout(controls)

        body = QtWidgets.QHBoxLayout()
        body.addWidget(self.editor, 1)

        info = QtWidgets.QLabel(
            "Controls:\n"
            "Left drag  = draw\n"
            "Right drag = erase\n"
            "Ctrl+Z     = undo\n"
            "R          = reset\n"
            "S          = save\n"
            "Enter      = save and close\n"
            "M          = toggle overlay/mask\n"
            "T          = toggle image-only/overlay\n"
            "Image-only = drawing disabled\n"
            "Esc/Q      = close without saving"
        )
        info.setWordWrap(True)
        info.setStyleSheet("font-size: 12px;")
        info_wrap = QtWidgets.QWidget()
        info_wrap.setFixedWidth(230)
        info_layout = QtWidgets.QVBoxLayout(info_wrap)
        info_layout.setContentsMargins(6, 6, 6, 6)
        info_layout.addWidget(info)
        info_layout.addStretch(1)
        body.addWidget(info_wrap, 0)
        root.addLayout(body, 1)

        btn_save.clicked.connect(self.save_mask)
        btn_save_close.clicked.connect(self.save_and_close)
        btn_cancel.clicked.connect(self.reject)
        btn_undo.clicked.connect(self.editor.undo)
        btn_reset.clicked.connect(self.editor.reset_mask)

    def _flash_saved_feedback(self, ms=1400):
        if hasattr(self, "btn_save"):
            old_style = self.btn_save.styleSheet()
            self.btn_save.setStyleSheet("background-color: #7ad27a; color: black; font-weight: bold;")
            QtCore.QTimer.singleShot(ms, lambda: self.btn_save.setStyleSheet(old_style))
        # short non-blocking popup-like feedback near top of dialog
        center = self.mapToGlobal(QtCore.QPoint(self.width() // 2, 24))
        QtWidgets.QToolTip.showText(center, "Saved", self, self.rect(), ms)

    def save_mask(self, close_after=False):
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        cv2.imwrite(self.out_path, self.editor.mask)
        self.saved = True
        self._flash_saved_feedback()
        if close_after:
            # Keep a short visible confirmation before closing.
            QtCore.QTimer.singleShot(350, self.accept)

    def save_and_close(self):
        self.save_mask(close_after=True)


def edit_mask_dialog(image, mask, out_path, parent=None):
    """
    Open a modal mask editor dialog inside an existing Qt app.
    Returns (saved_ok: bool, edited_mask_or_none).
    """
    dlg = MaskEditorDialog(image, mask, out_path, parent=parent)
    dlg.exec_()
    if dlg.saved:
        return True, dlg.editor.mask.copy()
    return False, None


# ============================================================
# Entry
# ============================================================

def main(img_path, mask_path, out_path):
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, 0)

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(image, mask, out_path)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    # -------------------------
    # Hardcoded convenience paths
    # -------------------------
    HARD_IMAGE_PATH = r"C:\Users\13144\Documents\Masters_Thesis\datasets\SUT_Compressed\Original_Image\8.jpg"
    HARD_MASK_PATH  = r"C:\Users\13144\Documents\Masters_Thesis\datasets\SUT_Compressed\Ground Truth\8.png"
    HARD_OUT_PATH   = r"test.png"

    # CLI override:
    #   python mask_brush_editor.py in_img.png in_mask.png out_mask.png
    if len(sys.argv) >= 4:
        img_p = sys.argv[1]
        mask_p = sys.argv[2]
        out_p = sys.argv[3]
    else:
        img_p = HARD_IMAGE_PATH
        mask_p = HARD_MASK_PATH
        out_p = HARD_OUT_PATH

    main(img_p, mask_p, out_p)
