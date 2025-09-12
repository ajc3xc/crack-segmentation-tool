from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint

class CrackAnnotator(QtWidgets.QWidget):
    def __init__(self, image=None, boxes=None, initial_points=None, initial_connections=None, initial_midlines=None):
        super().__init__()

        # --- image / geometry ---
        self.orig_image = image
        if image is not None:
            h, w, _ = image.shape
            qimg = QImage(image.data, w, h, image.strides[0], QImage.Format_RGB888)
            self.image_pixmap = QPixmap.fromImage(qimg)
            self.img_w, self.img_h = w, h
        else:
            self.image_pixmap = None
            self.img_w, self.img_h = 100, 100

        self.points = list(initial_points) if initial_points else []
        self.connections = [(min(a, b), max(a, b)) for a, b in (initial_connections or []) if a != b]

        min_dim = min(self.img_w, self.img_h) if self.img_w and self.img_h else 100
        self.point_radius = max(3, min(20, int(0.005 * min_dim)))
        self.boxes = boxes or []

        # --- zoom / pan / fit state ---
        self.scale = 1.0
        self.pan_x, self.pan_y = 0.0, 0.0
        self._fit_scale = 1.0        # computed when we can measure the viewport
        self._user_zoomed = False    # once True, we stop auto-fit on resize
        self._sa = None              # cached scroll area
        self._last_draw_scale = 1.0
        self._last_draw_xoff = 0.0
        self._last_draw_yoff = 0.0

        self.setMouseTracking(True)

        # --- modes/state you already had ---
        self.connection_mode = False
        self.connecting_index = None
        self.hover_index = None
        self.hover_line_index = None

        self.polyline_mode = False
        self.polyline = []
        self._is_drawing = False
        self._start_idx = None

        self.midlines = {}
        if isinstance(initial_midlines, dict):
            for k, poly in initial_midlines.items():
                try:
                    a, b = k.split("_")
                    i1, i2 = int(a), int(b)
                    if i1 != i2:
                        self.midlines[(min(i1, i2), max(i1, i2))] = [(float(x), float(y)) for x, y in poly]
                except:
                    pass

        self._hover_midline_key = None
        self._last_pair_error = None

        self._erase_timer = QtCore.QTimer()
        self._erase_timer.timeout.connect(lambda: (self._pop_poly_point(), self.update()))
        self._erase_start_time = None

        self.readonly_midlines = {}
        self.readonly_connections = []

        self.setFocusPolicy(Qt.StrongFocus)

        # try to hook & fit after layout settles
        QtCore.QTimer.singleShot(0, self._late_init)

    # ---------- FIT/SCROLL AREA HOOKS ----------
    def _late_init(self):
        self._hook_scroll_area()
        self._fit_to_view()

    def _hook_scroll_area(self):
        if self._sa:
            return
        sa = self._find_scroll_area()
        if sa is None:
            return
        self._sa = sa
        # Watch viewport resizes; refit until user zooms
        sa.viewport().installEventFilter(self)

    def eventFilter(self, obj, ev):
        # react to viewport resizes only until the user has zoomed
        if self._sa and obj is self._sa.viewport() and ev.type() == QtCore.QEvent.Resize:
            if not self._user_zoomed:
                # defer a tick so sizes are final
                QtCore.QTimer.singleShot(0, self._fit_to_view)
        return super().eventFilter(obj, ev)

    def _fit_to_view(self):
        if self.image_pixmap is None:
            return
        self._hook_scroll_area()
        if not self._sa:
            return

        vp = self._sa.viewport()
        vw, vh = max(1, vp.width()), max(1, vp.height())

        # compute fit scale to show the WHOLE image
        self._fit_scale = min(vw / self.img_w, vh / self.img_h)
        if self._fit_scale <= 0:
            self._fit_scale = 1.0

        self.scale = self._fit_scale
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update_canvas_size()

        # center inside the viewport
        cw, ch = int(self.img_w * self.scale), int(self.img_h * self.scale)
        hsb, vsb = self._sa.horizontalScrollBar(), self._sa.verticalScrollBar()
        hsb.setValue(max(0, (cw - vw) // 2))
        vsb.setValue(max(0, (ch - vh) // 2))
        self.update()

    # ---------- BASIC HELPERS ----------
    def _find_scroll_area(self):
        p = self.parent()
        while p is not None and not isinstance(p, QtWidgets.QAbstractScrollArea):
            p = p.parent()
        return p

    def update_canvas_size(self):
        w = int(self.img_w * self.scale)
        h = int(self.img_h * self.scale)
        self.setMinimumSize(w, h)
        self.resize(w, h)   # important for QScrollArea content size
        self.update()

    def _min_scale(self):
    # Don’t allow zoom-out beyond ~90% of fit-to-view
        return max(0.01, 0.9 * (self._fit_scale or 1.0))

    def _clamp_pan(self):
        # used only in the no-scroll-area fallback
        vw, vh = self.width(), self.height()
        cw, ch = int(self.img_w * self.scale), int(self.img_h * self.scale)
        self.pan_x = min(0, max(vw - cw, self.pan_x))
        self.pan_y = min(0, max(vh - ch, self.pan_y))

    # ---------- INPUT: ZOOM ----------
    def wheelEvent(self, event):
        # Ctrl+Wheel → let QScrollArea handle scrolling
        if event.modifiers() & Qt.ControlModifier:
            super().wheelEvent(event)
            return

        # Zoom factor
        f = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        old = self.scale
        new = max(0.1, min(10.0, old * f))  # allow zoom-out to 0.1

        if abs(new - old) < 1e-9:
            return

        # Mouse position in widget coordinates
        mx, my = event.pos().x(), event.pos().y()

        # Image coords under cursor (before zoom)
        img_x = (mx - self.pan_x) / old
        img_y = (my - self.pan_y) / old

        # Apply zoom
        self.scale = new

        # Adjust pan so cursor stays anchored
        self.pan_x = mx - img_x * new
        self.pan_y = my - img_y * new

        self.update()

    # ---------- YOUR EXISTING EDITOR LOGIC (unchanged) ----------
    def toggle_mode(self):
        self.connection_mode = not self.connection_mode
        self.connecting_index = None
        self.update()

    def set_mode_polyline(self, enabled: bool, confirm_cb=None):
        if not enabled and self._is_drawing:
            ok_to_discard = True
            if confirm_cb is not None:
                ok_to_discard = confirm_cb()
            if not ok_to_discard:
                return False
            self.polyline.clear()
            self._is_drawing = False
            self._start_idx = None
        self.polyline_mode = enabled
        if enabled:
            self.connecting_index = None
        self.update()
        return True

    def all_pairs_saturated(self):
        n = len(self.points)
        if n < 2:
            return True
        pairs = set(self.connections) | set(self.midlines.keys()) | set(self.readonly_midlines.keys()) | set(self.readonly_connections)
        return len(pairs) >= (n * (n - 1)) // 2

    def _sorted(self, i, j):
        return (i, j) if i < j else (j, i)

    def _add_poly_point(self, p):
        self.polyline.append((float(p[0]), float(p[1])))

    def _pop_poly_point(self):
        if self.polyline:
            self.polyline.pop()

    # ---------- PAINT ----------
    def paintEvent(self, event):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)

        if self.image_pixmap:
            scale = self.scale
            xoff  = self.pan_x
            yoff  = self.pan_y

            # expose transform for hit-testing
            self._last_draw_scale = scale
            self._last_draw_xoff  = xoff
            self._last_draw_yoff  = yoff

            qp.drawPixmap(int(xoff), int(yoff),
                        int(self.img_w * scale), int(self.img_h * scale),
                        self.image_pixmap)
        else:
            self._last_draw_scale, self._last_draw_xoff, self._last_draw_yoff = 1.0, 0.0, 0.0
            scale, xoff, yoff = 1.0, 0.0, 0.0

        # (leave the rest of your bounding boxes, connections, points, midlines, etc. unchanged)

        crop_xmin, crop_ymin = getattr(self, "crop_offset", (0, 0))

        def apply_offset(pt):
            return (pt[0] + crop_xmin, pt[1] + crop_ymin)

        # boxes
        qp.setPen(QPen(QColor(0, 128, 255), 3))
        for xmin, ymin, xmax, ymax in self.boxes:
            qp.drawRect(int((xmin) * scale + xoff), int((ymin) * scale + yoff),
                        int((xmax - xmin) * scale), int((ymax - ymin) * scale))

        # read-only connections
        qp.setPen(QPen(QColor(150, 150, 150), 2, Qt.DashLine))
        for i1, i2 in self.readonly_connections:
            if i1 < len(self.points) and i2 < len(self.points):
                p1 = apply_offset(self.points[i1])
                p2 = apply_offset(self.points[i2])
                qp.drawLine(QPoint(int(p1[0] * scale + xoff), int(p1[1] * scale + yoff)),
                            QPoint(int(p2[0] * scale + xoff), int(p2[1] * scale + yoff)))

        # read-only midlines
        qp.setPen(QPen(QColor(150, 150, 0), 2))
        for key, poly in self.readonly_midlines.items():
            for i in range(1, len(poly)):
                p1 = apply_offset(poly[i - 1])
                p2 = apply_offset(poly[i])
                qp.drawLine(QPoint(int(p1[0] * scale + xoff), int(p1[1] * scale + yoff)),
                            QPoint(int(p2[0] * scale + xoff), int(p2[1] * scale + yoff)))

        # editable connections
        for idx, (i1, i2) in enumerate(self.connections):
            if i1 < len(self.points) and i2 < len(self.points):
                p1 = apply_offset(self.points[i1])
                p2 = apply_offset(self.points[i2])
                thick = 6 if (self.connection_mode and self.connecting_index is None
                              and idx == self.hover_line_index and self.hover_index is None) else 4
                qp.setPen(QPen(QColor(0, 0, 0), thick))
                qp.drawLine(QPoint(int(p1[0] * scale + xoff), int(p1[1] * scale + yoff)),
                            QPoint(int(p2[0] * scale + xoff), int(p2[1] * scale + yoff)))

        # points
        for i, (x, y) in enumerate(self.points):
            x, y = apply_offset((x, y))
            center = QPoint(int(x * scale + xoff), int(y * scale + yoff))
            brush = QColor(0, 200, 0) if i == self.hover_index or (
                self.connection_mode and i == self.connecting_index) else QColor(200, 80, 80)
            qp.setBrush(brush)
            qp.setPen(Qt.NoPen)
            qp.drawEllipse(center, int(self.point_radius * scale), int(self.point_radius * scale))

        # editable midlines
        for key, poly in self.midlines.items():
            if len(poly) < 2:
                continue
            thick = 8 if (self.connection_mode and key == self._hover_midline_key) else 4
            qp.setPen(QPen(QColor(0, 200, 200), thick))
            for i in range(1, len(poly)):
                p1 = apply_offset(poly[i - 1])
                p2 = apply_offset(poly[i])
                qp.drawLine(QPoint(int(p1[0] * scale + xoff), int(p1[1] * scale + yoff)),
                            QPoint(int(p2[0] * scale + xoff), int(p2[1] * scale + yoff)))

        # live polyline
        if self.polyline_mode and len(self.polyline) >= 1:
            qp.setPen(QPen(QColor(0, 200, 200), 4))
            for i in range(1, len(self.polyline)):
                p1 = apply_offset(self.polyline[i - 1])
                p2 = apply_offset(self.polyline[i])
                qp.drawLine(QPoint(int(p1[0] * scale + xoff), int(p1[1] * scale + yoff)),
                            QPoint(int(p2[0] * scale + xoff), int(p2[1] * scale + yoff)))

    # ---------- rest of your existing methods ----------
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Backspace, Qt.Key_Z):
            if self.polyline_mode and self._is_drawing and self.polyline:
                self._pop_poly_point()
                self.update()
                return
        if event.key() == Qt.Key_Escape:
            if self.polyline_mode and self._is_drawing:
                self.polyline.clear()
                self._is_drawing = False
                self._start_idx = None
                self.update()
                return
        super().keyPressEvent(event)

    def _to_image_coords(self, pos):
        # Map from widget coords → image coords (match pan/scale logic in paintEvent)
        return ((pos.x() - self.pan_x) / self.scale,
                (pos.y() - self.pan_y) / self.scale)

    def _find_point_at(self, pos):
        r2 = (self.point_radius / self.scale) ** 2
        for i, (x, y) in enumerate(self.points):
            if (x - pos[0]) ** 2 + (y - pos[1]) ** 2 <= r2:
                return i
        return None

    def _find_line_at(self, pos):
        thr = 7 / self.scale
        for idx, (i1, i2) in enumerate(self.connections):
            x1, y1 = self.points[i1]
            x2, y2 = self.points[i2]
            if self._dist_point_to_segment(pos, (x1, y1), (x2, y2)) < thr:
                return idx
        return None

    def _dist_point_to_segment(self, p, a, b):
        import numpy as np
        p, a, b = np.array(p), np.array(a), np.array(b)
        if np.all(a == b):
            return float(np.linalg.norm(p - a))
        t = max(0, min(1, np.dot(p - a, b - a) / np.dot(b - a, b - a)))
        proj = a + t * (b - a)
        return float(np.linalg.norm(p - proj))

    def _midline_hit_test(self, screen_pos, px_thresh=9.0):
        if not self.midlines:
            return None
        s = self._last_draw_scale
        xo = self._last_draw_xoff
        yo = self._last_draw_yoff
        sx, sy = screen_pos.x(), screen_pos.y()

        import math
        def dseg(px, py, ax, ay, bx, by):
            vx, vy = bx - ax, by - ay
            wx, wy = px - ax, py - ay
            vv = vx * vx + vy * vy
            if vv <= 1e-12:
                return math.hypot(px - ax, py - ay)
            t = max(0.0, min(1.0, (wx * vx + wy * vy) / vv))
            qx, qy = ax + t * vx, ay + t * vy
            return math.hypot(px - qx, py - qy)

        best = 1e9
        hit = None
        for key, poly in self.midlines.items():
            for i in range(1, len(poly)):
                x1, y1 = poly[i - 1]
                x2, y2 = poly[i]
                ax, ay = x1 * s + xo, y1 * s + yo
                bx, by = x2 * s + xo, y2 * s + yo
                d = dseg(sx, sy, ax, ay, bx, by)
                if d < best:
                    best = d
                    hit = key
        return hit if best <= px_thresh else None

    def _delete_point_reindex(self, idx):
        self.connections = [(i1 - (i1 > idx), i2 - (i2 > idx))
                            for (i1, i2) in self.connections if i1 != idx and i2 != idx]
        new_mid = {}
        for (i1, i2), poly in self.midlines.items():
            if i1 == idx or i2 == idx:
                continue
            ni1 = i1 - (i1 > idx)
            ni2 = i2 - (i2 > idx)
            new_mid[(min(ni1, ni2), max(ni1, ni2))] = poly
        self.midlines = new_mid
        self.points.pop(idx)

    def _commit_midline(self, end_idx):
        start_idx = self._start_idx
        if start_idx is None or end_idx is None or start_idx == end_idx:
            return

        key = self._sorted(start_idx, end_idx)
        if (key in self.midlines) or (key in self.connections) \
        or (key in self.readonly_connections) or (key in self.readonly_midlines):
            self.polyline.clear()
            self._is_drawing = False
            self._start_idx = None
            self.update()
            return

        # Build the polyline in drawn order
        if len(self.polyline) >= 2:
            middle = list(self.polyline[1:-1])
            poly = [tuple(map(float, self.points[start_idx]))] + middle + [tuple(map(float, self.points[end_idx]))]
        else:
            poly = [tuple(map(float, self.points[start_idx])),
                    tuple(map(float, self.points[end_idx]))]

        # --- NEW: bounding box check ---
        if self.boxes:
            # assume just one box for now; extend easily if multiple
            xmin, ymin, xmax, ymax = self.boxes[0]
            outside = [(x, y) for (x, y) in poly if not (xmin <= x <= xmax and ymin <= y <= ymax)]
            if outside:
                print(f"[COMMIT] Segment rejected, {len(outside)} points outside bbox")
                # optional feedback dialog
                QtWidgets.QMessageBox.warning(self, "Out of bounds",
                    "Some segment points are outside the bounding box.\nCommit cancelled.")
                self.polyline.clear()
                self._is_drawing = False
                self._start_idx = None
                self.update()
                return
        # --- END NEW ---

        self.midlines[key] = poly
        self._last_polyline_start_idx = start_idx
        self._last_polyline_end_idx   = end_idx
        self._just_committed_midline  = True

        self.polyline.clear()
        self._is_drawing = False
        self._start_idx = None
        self.update()
    
    def mousePressEvent(self, event):
        # image coords under the cursor (using pan/scale)
        p = self._to_image_coords(event.pos())
        point_i = self._find_point_at(p)
        line_i = self._find_line_at(p)
        mid_key = self._midline_hit_test(event.pos(), 10.0)

        print(f"[PRESS] Click at {p}, point_i={point_i}, line_i={line_i}, mid_key={mid_key}, "
            f"_is_drawing={self._is_drawing}, polyline_mode={self.polyline_mode}, polyline_len={len(self.polyline)}")

        if mid_key in self.readonly_midlines:
            mid_key = None
        if line_i is not None and (self.connections[line_i] in self.readonly_connections):
            line_i = None

        # ----- Polyline mode -----
        if self.polyline_mode:
            if event.button() == Qt.LeftButton:
                if not self._is_drawing:
                    # first click while in polyline mode
                    if mid_key is not None:
                        self.midlines.pop(mid_key, None)
                        self._hover_midline_key = None
                        self.update()
                        return
                    if line_i is not None:
                        self.connections.pop(line_i)
                        self.update()
                        return
                    if point_i is None or len(self.points) < 2:
                        return
                    self._start_idx = point_i
                    sx, sy = self.points[self._start_idx]
                    self.polyline = [(float(sx), float(sy))]
                    self._is_drawing = True
                    print(f"[PRESS] START midline from {self._start_idx} at {self.points[self._start_idx]}")
                    self.update()
                    return
                else:
                    # second+ clicks while drawing
                    if getattr(self, "_just_committed_midline", False):
                        self._just_committed_midline = False
                        return
                    if point_i is not None and point_i != self._start_idx:
                        self._commit_midline(point_i)
                    elif point_i is None:
                        # add free point to the live polyline (in image coords!)
                        px, py = p
                        self.polyline.append((float(px), float(py)))
                        print(f"[PRESS] Added polyline point: {self.polyline[-1]}")
                        self.update()
                    return

            elif event.button() == Qt.RightButton:
                if self._is_drawing:
                    self._erase_timer.stop()
                    QtCore.QTimer.singleShot(500, lambda: (
                        self._erase_timer.start(75)
                        if (QtWidgets.QApplication.mouseButtons() & Qt.RightButton and self._is_drawing)
                        else None
                    ))
                    if len(self.polyline) > 1:
                        self._pop_poly_point()
                    else:
                        self.polyline = []
                        self._is_drawing = False
                        self._start_idx = None
                    self.update()
                return

        # ----- Normal connection / point modes -----
        if event.button() == Qt.LeftButton:
            if (not self.polyline_mode) and self.connection_mode and (mid_key is not None):
                self.midlines.pop(mid_key, None)
                self._hover_midline_key = None
                self.update()
                return

            if not self.connection_mode:
                # add/remove point
                if point_i is None:
                    # p is already in image coords
                    self.points.append((float(p[0]), float(p[1])))
                    print(f"[PRESS] Added new point at {p}")
                else:
                    if any(point_i in c for c in self.readonly_connections) or \
                    any(point_i in k for k in self.readonly_midlines.keys()):
                        return
                    self._delete_point_reindex(point_i)
            else:
                # connect / delete edges
                if (line_i is not None) and (self.connecting_index is None) and (point_i is None):
                    self.connections.pop(line_i)
                elif point_i is not None:
                    if self.connecting_index is None:
                        self.connecting_index = point_i
                    elif self.connecting_index != point_i:
                        c = self._sorted(self.connecting_index, point_i)
                        if c not in self.connections and c not in self.readonly_connections:
                            self.connections.append(c)
                        self.connecting_index = None
                    else:
                        self.connecting_index = None
                else:
                    self.connecting_index = None
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self._erase_timer.stop()

    def mouseMoveEvent(self, event):
        # use image coords for hit-tests & drawing
        p = self._to_image_coords(event.pos())
        point_i = self._find_point_at(p)

        if self.polyline_mode and self._is_drawing and (event.buttons() & Qt.LeftButton):
            if point_i is not None and point_i != self._start_idx:
                print(f"[MOVE] Hovering endpoint {point_i}, attempting commit")
                self._commit_midline(point_i)
                return
            else:
                # freehand: append image-space points
                self._add_poly_point(p)
                # print(f"[MOVE] Added freehand point {p}")
                self.update()
                return

        self.hover_index = self._find_point_at(p)
        if self.connection_mode and self.connecting_index is None and self.hover_index is None:
            self.hover_line_index = self._find_line_at(p)
        else:
            self.hover_line_index = None

        # midline hit-test wants screen coords; that's event.pos()
        self._hover_midline_key = self._midline_hit_test(event.pos(), 10.0) if not self._is_drawing else None
        self.update()
        
        
        
    def wheelEvent(self, event):
        # Ctrl+Wheel → let QScrollArea handle scrolling
        if event.modifiers() & Qt.ControlModifier:
            super().wheelEvent(event)
            return

        f = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        old = self.scale
        new = max(self._min_scale(), min(10.0, old * f))
        if abs(new - old) < 1e-9:
            return

        # widget coords
        mx, my = event.pos().x(), event.pos().y()
        # image coords under cursor before zoom
        img_x = (mx - self.pan_x) / old
        img_y = (my - self.pan_y) / old

        self.scale = new
        self._user_zoomed = True           # <- critical: blocks refit on minor resizes

        # keep cursor anchored
        self.pan_x = mx - img_x * new
        self.pan_y = my - img_y * new
        self.update()

    def toggle_mode(self):
        self.connection_mode = not self.connection_mode
        self.connecting_index = None
        # if user is already off fit scale, lock out refits triggered by layout tweaks
        if abs(self.scale - self._fit_scale) > 1e-6:
            self._user_zoomed = True
        self.update()

    def set_mode_polyline(self, enabled: bool, confirm_cb=None):
        if not enabled and self._is_drawing:
            ok_to_discard = True
            if confirm_cb is not None:
                ok_to_discard = confirm_cb()
            if not ok_to_discard:
                return False
            self.polyline.clear()
            self._is_drawing = False
            self._start_idx = None
        self.polyline_mode = enabled
        if enabled:
            self.connecting_index = None

        if abs(self.scale - self._fit_scale) > 1e-6:
            self._user_zoomed = True

        self.update()
        return True
    
    def _commit_midline(self, end_idx):
        start_idx = self._start_idx
        if start_idx is None or end_idx is None or start_idx == end_idx:
            return

        key = self._sorted(start_idx, end_idx)
        if (key in self.midlines) or (key in self.connections) \
        or (key in self.readonly_connections) or (key in self.readonly_midlines):
            self.polyline.clear()
            self._is_drawing = False
            self._start_idx = None
            self.update()
            return

        # Build the polyline in drawn order
        if len(self.polyline) >= 2:
            middle = list(self.polyline[1:-1])
            poly = [tuple(map(float, self.points[start_idx]))] + middle + [tuple(map(float, self.points[end_idx]))]
        else:
            poly = [tuple(map(float, self.points[start_idx])),
                    tuple(map(float, self.points[end_idx]))]

        # --- STRICT: all points must be inside the SAME bounding box (if any exist) ---
        if self.boxes:
            def inside_box(poly_pts, box):
                xmin, ymin, xmax, ymax = box
                return all(xmin <= x <= xmax and ymin <= y <= ymax for (x, y) in poly_pts)

            ok_single_box = any(inside_box(poly, b) for b in self.boxes)
            if not ok_single_box:
                QMessageBox.warning(self, "Out of bounds",
                    "The midline must lie entirely inside a single bounding box.\nCommit cancelled.")
                self.polyline.clear()
                self._is_drawing = False
                self._start_idx = None
                self.update()
                return
        # --- END STRICT ---

        self.midlines[key] = poly
        self._last_polyline_start_idx = start_idx
        self._last_polyline_end_idx   = end_idx
        self._just_committed_midline  = True

        self.polyline.clear()
        self._is_drawing = False
        self._start_idx = None
        self.update()
    
    def add_midline_auto(self, i1, i2, poly):
        """
        Add an automatically-computed midline for the pair (i1,i2).
        `poly` must be a list of (x, y) in IMAGE coordinates.
        Enforces: unique key, not read-only, and fully inside ONE bounding box.
        Returns True if added, False otherwise.
        """
        if i1 == i2:
            return False
        key = self._sorted(i1, i2)

        # duplicates / read-only
        if (key in self.midlines) or (key in self.connections) \
        or (key in self.readonly_connections) or (key in self.readonly_midlines):
            return False

        # single-box guard (if boxes exist)
        if self.boxes:
            def inside_box(poly_pts, box):
                xmin, ymin, xmax, ymax = box
                return all(xmin <= x <= xmax and ymin <= y <= ymax for (x, y) in poly_pts)
            if not any(inside_box(poly, b) for b in self.boxes):
                QMessageBox.warning(self, "Out of bounds",
                    f"Auto midline for pair {key} is not inside a single bounding box.\nInsertion cancelled.")
                return False

        # ok — insert
        self.midlines[key] = [ (float(x), float(y)) for (x,y) in poly ]
        self.update()
        return True