from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QPainter, QPen, QColor, QCursor
from PyQt5.QtCore import Qt, QPoint, QPointF
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

min_crop_size = 16

class CrackAnnotator(QtWidgets.QWidget):
    """
    Extended CrackAnnotator:
    - Supports multiple midlines, one per connection (tuple of point indices)
    - Midline drawing only allowed in connection mode, on a selected connection without an existing midline
    - Midline starts at one endpoint, must end at the other
    - Hover over midline in connection mode to highlight and right-click to delete
    """
    def __init__(self, image=None, boxes=None, initial_points=None, initial_connections=None, initial_midlines=None):
        super().__init__()
        # Store & prepare image
        self.orig_image = image
        if image is not None:
            h, w, _ = image.shape
            qimg = QImage(image.data, w, h, image.strides[0], QImage.Format_RGB888)
            self.image_pixmap = QPixmap.fromImage(qimg)
            self.img_w, self.img_h = w, h
        else:
            self.image_pixmap = None
            self.img_w, self.img_h = 100, 100  # default size

        # Annotation state
        self.points = list(initial_points) if initial_points else []
        self.connections = list(initial_connections) if initial_connections else []

        # Midlines: dict[(i1,i2)] = [(x,y),...]
        self.midlines = dict(initial_midlines) if initial_midlines else {}

        # --- Adaptive point radius ---
        if image is not None:
            h, w, _ = image.shape
            min_dim = min(w, h)
        else:
            min_dim = 100
        self.point_radius = max(3, min(20, int(0.005 * min_dim)))

        self.connection_mode = False
        self.connecting_index = None
        self.hover_index = None
        self.hover_line_index = None
        self.hover_midline_key = None  # for highlighting

        # Bounding boxes
        self.boxes = boxes if boxes is not None else []

        # Zoom state
        self.scale = 1.0
        self.setMouseTracking(True)
        self.update_canvas_size()

        # Midline drawing state
        self.polyline_mode = False
        self.current_midline_key = None
        self.current_polyline = []
        self._is_dragging = False

        self.setFocusPolicy(Qt.StrongFocus)

    # ===== Midline helpers =====
    def start_midline_for_connection(self, conn_key):
        """Enable midline mode for a given connection (tuple of point indices sorted)"""
        self.polyline_mode = True
        self.current_midline_key = tuple(sorted(conn_key))
        self.current_polyline = []
        self._is_dragging = False
        self.update()

    def cancel_midline(self):
        self.polyline_mode = False
        self.current_midline_key = None
        self.current_polyline = []
        self._is_dragging = False
        self.update()

    def finish_midline(self):
        if self.polyline_mode and self.current_midline_key and len(self.current_polyline) >= 2:
            self.midlines[self.current_midline_key] = list(self.current_polyline)
        self.cancel_midline()

    def delete_midline(self, conn_key):
        conn_key = tuple(sorted(conn_key))
        if conn_key in self.midlines:
            del self.midlines[conn_key]
            self.update()

    def _add_poly_point(self, p):
        self.current_polyline.append((float(p[0]), float(p[1])))

    def _pop_poly_point(self):
        if self.current_polyline:
            self.current_polyline.pop()

    def clear_current_polyline(self):
        self.current_polyline = []
        self.update()

    # ===== Canvas/interaction =====
    def update_canvas_size(self):
        w = int(self.img_w * self.scale)
        h = int(self.img_h * self.scale)
        self.setMinimumSize(w, h)
        self.resize(w, h)
        self.update()

    def wheelEvent(self, event):
        f = 1.2 if event.angleDelta().y() > 0 else 1/1.2
        self.scale = max(0.1, min(10.0, self.scale * f))
        self.update_canvas_size()
        self.update()

    def mousePressEvent(self, event):
        p = self._to_image_coords(event.pos())

        # --- Drawing a midline ---
        if self.polyline_mode and self.current_midline_key:
            if event.button() == Qt.LeftButton:
                self._add_poly_point(p)
                self._is_dragging = True
                self.update()
            elif event.button() == Qt.RightButton:
                self._pop_poly_point()
                self.update()
            return

        # --- In connection mode: right-click delete midline if hovering ---
        if self.connection_mode and self.hover_midline_key and event.button() == Qt.RightButton:
            self.delete_midline(self.hover_midline_key)
            return

        # --- Original connection/point logic ---
        point_i = self._find_point_at(p)
        line_i = self._find_line_at(p)

        if event.button() == Qt.LeftButton:
            if not self.connection_mode:
                if point_i is None:
                    self.points.append(p)
                else:
                    self.connections = [(i1, i2) for i1, i2 in self.connections if i1 != point_i and i2 != point_i]
                    self.points.pop(point_i)
                    self.connections = [(i1 - (i1 > point_i), i2 - (i2 > point_i)) for i1, i2 in self.connections]
            else:
                if (line_i is not None) and (self.connecting_index is None) and (point_i is None):
                    self.connections.pop(line_i)
                elif point_i is not None:
                    if self.connecting_index is None:
                        self.connecting_index = point_i
                    elif self.connecting_index != point_i:
                        c = (self.connecting_index, point_i)
                        if c not in self.connections:
                            self.connections.append(c)
                        self.connecting_index = None
                    else:
                        self.connecting_index = None
                else:
                    self.connecting_index = None
        self.update()

    def mouseReleaseEvent(self, event):
        self._is_dragging = False
        # Check if midline should be auto-finished when both endpoints reached
        if self.polyline_mode and self.current_midline_key:
            end_pts = [self.points[i] for i in self.current_midline_key]
            if len(self.current_polyline) >= 2:
                # If last point is near second endpoint
                last = self.current_polyline[-1]
                if self._is_near_point(last, end_pts[1]):
                    self.finish_midline()

    def mouseMoveEvent(self, event):
        p = self._to_image_coords(event.pos())

        if self.polyline_mode and self._is_dragging and (event.buttons() & Qt.LeftButton):
            self._add_poly_point(p)
            self.update()
            return

        self.hover_index = self._find_point_at(p)
        self.hover_midline_key = self._find_midline_at(p)
        if self.connection_mode and self.connecting_index is None and self.hover_index is None:
            self.hover_line_index = self._find_line_at(p)
        else:
            self.hover_line_index = None
        self.update()

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key_Backspace, Qt.Key_Z):
            if self.polyline_mode and self.current_polyline:
                self._pop_poly_point()
                self.update()
                return
        if key in (Qt.Key_C, Qt.Key_Delete):
            if self.polyline_mode and self.current_polyline:
                self.clear_current_polyline()
                return
        super().keyPressEvent(event)

    def paintEvent(self, event):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)

        # Draw image
        if self.image_pixmap:
            img_w, img_h = self.img_w, self.img_h
            win_w, win_h = self.width(), self.height()
            scale = min(win_w / img_w, win_h / img_h)
            xoff = int((win_w - img_w * scale) / 2)
            yoff = int((win_h - img_h * scale) / 2)
            self._last_draw_scale, self._last_draw_xoff, self._last_draw_yoff = scale, xoff, yoff
            qp.drawPixmap(xoff, yoff, int(img_w * scale), int(img_h * scale), self.image_pixmap)
        else:
            scale, xoff, yoff = 1.0, 0, 0
            self._last_draw_scale, self._last_draw_xoff, self._last_draw_yoff = scale, xoff, yoff

        # Bounding boxes
        qp.setPen(QPen(QColor(0, 128, 255), 3))
        for bbox in self.boxes:
            xmin, ymin, xmax, ymax = [float(v) for v in bbox]
            qp.drawRect(int(round(xmin*scale+xoff)), int(round(ymin*scale+yoff)),
                        int(round((xmax-xmin)*scale)), int(round((ymax-ymin)*scale)))

        # Connections
        for idx, (i1, i2) in enumerate(self.connections):
            p1_img, p2_img = self.points[i1], self.points[i2]
            p1 = QPoint(int(round(p1_img[0]*scale+xoff)), int(round(p1_img[1]*scale+yoff)))
            p2 = QPoint(int(round(p2_img[0]*scale+xoff)), int(round(p2_img[1]*scale+yoff)))
            pen = QPen(QColor(0,0,0), 6 if (self.connection_mode and idx==self.hover_line_index and self.hover_index is None) else 4)
            qp.setPen(pen)
            qp.drawLine(p1, p2)
            self._draw_arrowhead(qp, p1, p2)

        # Points
        for i,(x,y) in enumerate(self.points):
            center = QPoint(int(round(x*scale+xoff)), int(round(y*scale+yoff)))
            brush = QColor(0,200,0) if i==self.hover_index or (self.connection_mode and i==self.connecting_index) else QColor(200,80,80)
            qp.setBrush(brush)
            qp.setPen(Qt.NoPen)
            qp.drawEllipse(center, int(self.point_radius*scale), int(self.point_radius*scale))

        # Midlines
        for key, poly in self.midlines.items():
            qp.setPen(QPen(QColor(0, 200, 200), 6 if key==self.hover_midline_key else 4))
            for i in range(1, len(poly)):
                p1 = QPoint(int(round(poly[i-1][0]*scale+xoff)), int(round(poly[i-1][1]*scale+yoff)))
                p2 = QPoint(int(round(poly[i][0]*scale+xoff)), int(round(poly[i][1]*scale+yoff)))
                qp.drawLine(p1, p2)

        # Current drawing polyline
        if self.polyline_mode and self.current_polyline:
            qp.setPen(QPen(QColor(255, 150, 0), 4))
            for i in range(1, len(self.current_polyline)):
                p1 = QPoint(int(round(self.current_polyline[i-1][0]*scale+xoff)),
                            int(round(self.current_polyline[i-1][1]*scale+yoff)))
                p2 = QPoint(int(round(self.current_polyline[i][0]*scale+xoff)),
                            int(round(self.current_polyline[i][1]*scale+yoff)))
                qp.drawLine(p1, p2)

    def toggle_mode(self):
        self.connection_mode = not self.connection_mode
        self.connecting_index = None
        self.update()

    # Geometry helpers
    def _to_image_coords(self, pos):
        return ((pos.x()-self._last_draw_xoff)/self._last_draw_scale,
                (pos.y()-self._last_draw_yoff)/self._last_draw_scale)

    def _find_point_at(self, pos):
        for i,(x,y) in enumerate(self.points):
            if (x-pos[0])**2 + (y-pos[1])**2 <= (self.point_radius/self.scale)**2:
                return i
        return None

    def _find_line_at(self, pos):
        thr = 7 / self.scale
        for idx,(i1,i2) in enumerate(self.connections):
            if self._dist_point_to_segment(pos, self.points[i1], self.points[i2]) < thr:
                return idx
        return None

    def _find_midline_at(self, pos):
        thr = 7 / self.scale
        for key, poly in self.midlines.items():
            for i in range(1, len(poly)):
                if self._dist_point_to_segment(pos, poly[i-1], poly[i]) < thr:
                    return key
        return None

    def _dist_point_to_segment(self, p, a, b):
        import numpy as np
        p,a,b = np.array(p), np.array(a), np.array(b)
        if np.all(a==b): return np.linalg.norm(p-a)
        t = max(0, min(1, np.dot(p-a,b-a)/np.dot(b-a,b-a)))
        proj = a + t*(b-a)
        return np.linalg.norm(p-proj)

    def _draw_arrowhead(self, qp, p1, p2):
        import math
        angle = math.atan2(p2.y()-p1.y(), p2.x()-p1.x())
        sz = int(10*self.scale)
        dx1, dy1 = sz*math.cos(angle-math.pi/8), sz*math.sin(angle-math.pi/8)
        dx2, dy2 = sz*math.cos(angle+math.pi/8), sz*math.sin(angle+math.pi/8)
        left = QPoint(int(p2.x()-dx1), int(p2.y()-dy1))
        right = QPoint(int(p2.x()-dx2), int(p2.y()-dy2))
        qp.setPen(Qt.NoPen)
        qp.setBrush(QColor(80,80,220))
        qp.drawPolygon(p2, left, right)

    def _is_near_point(self, a, b, thr=5):
        return (a[0]-b[0])**2 + (a[1]-b[1])**2 <= thr**2