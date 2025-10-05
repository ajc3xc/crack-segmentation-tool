from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint

# I dare you to waste another week trying to improve the zooming functionality

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
        self.point_radius = max(3, min(20, int(0.006 * min_dim)))
        self.boxes = boxes or []

        # --- zoom / pan / fit state ---
        self.scale = 1.0
        self.pan_x, self.pan_y = 0.0, 0.0
        self._fit_scale = 1.0
        self._user_zoomed = False
        self._sa = None
        self._last_draw_scale = 1.0
        self._last_draw_xoff = 0.0
        self._last_draw_yoff = 0.0

        self.setMouseTracking(True)

        # --- modes/state ---
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

        # init crop tracking
        self._crop_x1, self._crop_y1 = 0, 0
        self._crop_x2, self._crop_y2 = self.img_w, self.img_h

        # hook & fit after layout settles
        QtCore.QTimer.singleShot(0, self._late_init)

        # --- NEW: enable pinch gestures (touchpad, Mac trackpad, etc.) ---
        self.grabGesture(Qt.PinchGesture)
        
        self._panning = False      # currently dragging the view?
        self._pan_last = None      # last mouse position while dragging


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

    '''def eventFilter(self, obj, ev):
        # react to viewport resizes only until the user has zoomed
        if self._sa and obj is self._sa.viewport() and ev.type() == QtCore.QEvent.Resize:
            if not self._user_zoomed:
                # defer a tick so sizes are final
                QtCore.QTimer.singleShot(0, self._fit_to_view)
        return super().eventFilter(obj, ev)'''
        
    def eventFilter(self, obj, ev):
        if self._sa and obj is self._sa.viewport() and ev.type() == QtCore.QEvent.Resize:
            if not self._user_zoomed:
                QtCore.QTimer.singleShot(0, self._fit_to_view)
            else:
                # After the user has zoomed, keep it neat on viewport size changes
                QtCore.QTimer.singleShot(0, self._enforce_bounds)
        return super().eventFilter(obj, ev)

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

    # ---------- YOUR EXISTING EDITOR LOGIC (unchanged) ----------
    def toggle_mode(self):
        self.connection_mode = not self.connection_mode
        self.connecting_index = None
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
        # Keep hitbox radius constant in screen space, not image space
        r = self.point_radius / self.scale
        r2 = r * r
        for i, (x, y) in enumerate(self.points):
            if (x - pos[0])**2 + (y - pos[1])**2 <= r2:
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

    # ---------- bbox helpers ----------
    def _point_box_index(self, x, y):
        """Return the index of the first bbox containing (x,y), or None."""
        for bi, (xmin, ymin, xmax, ymax) in enumerate(self.boxes):
            if xmin <= x <= xmax and ymin <= y <= ymax:
                return bi
        return None

    def _poly_inside_box_index(self, poly, box_index):
        """True if all (x,y) lie inside the given bbox index."""
        if box_index is None or not self.boxes:
            return False
        xmin, ymin, xmax, ymax = self.boxes[box_index]
        for (x, y) in poly:
            if not (xmin <= float(x) <= xmax and ymin <= float(y) <= ymax):
                return False
        return True

    # ---------- midlines ----------
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

        # STRICT: endpoints must be in the SAME bbox, and all poly points in that bbox
        if self.boxes:
            sx, sy = self.points[start_idx]
            ex, ey = self.points[end_idx]
            b1 = self._point_box_index(float(sx), float(sy))
            b2 = self._point_box_index(float(ex), float(ey))
            if (b1 is None) or (b2 is None) or (b1 != b2) or (not self._poly_inside_box_index(poly, b1)):
                QMessageBox.warning(self, "Out of bounds",
                    "The midline and both endpoints must lie inside the same bounding box.\nCommit cancelled.")
                self.polyline.clear()
                self._is_drawing = False
                self._start_idx = None
                self.update()
                return

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
        Enforces: unique key, not read-only, BOTH endpoints in the SAME bbox,
                  and the polyline lies entirely inside that bbox.
        Returns True if added, False otherwise.
        """
        if i1 == i2:
            return False
        key = self._sorted(i1, i2)

        # duplicates / read-only collisions
        if (key in self.midlines) or (key in self.connections) \
           or (key in self.readonly_connections) or (key in self.readonly_midlines):
            return False

        if self.boxes:
            # endpoint -> bbox index
            try:
                sx, sy = self.points[i1]
                ex, ey = self.points[i2]
            except Exception:
                return False

            b1 = self._point_box_index(float(sx), float(sy))
            b2 = self._point_box_index(float(ex), float(ey))

            # must both exist and match, and poly must be inside that same bbox
            if (b1 is None) or (b2 is None) or (b1 != b2) or (not self._poly_inside_box_index(poly, b1)):
                QMessageBox.warning(self, "Out of bounds",
                    f"Auto midline for pair {key} must be entirely inside a single bounding box with both endpoints.\nInsertion cancelled.")
                return False

        # ok — insert
        self.midlines[key] = [(float(x), float(y)) for (x, y) in poly]
        self.update()
        return True

    # ---------- mouse handlers ----------
    '''def mousePressEvent(self, event):
        # image coords under the cursor (using pan/scale)
        img_rect = QtCore.QRect(
            int(self._last_draw_xoff),
            int(self._last_draw_yoff),
            int(self.img_w * self.scale),
            int(self.img_h * self.scale),
        )
        if not img_rect.contains(event.pos()):
            return  # ignore clicks in gray margin

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
        self.update()'''

    def add_midline_auto(self, i1, i2, poly):
        """
        Add an automatically-computed midline for the pair (i1,i2).
        Enforces: both endpoints + all poly points inside SAME bounding box.
        """
        if i1 == i2:
            return False
        key = self._sorted(i1, i2)

        # duplicates / read-only
        if (key in self.midlines) or (key in self.connections) \
        or (key in self.readonly_connections) or (key in self.readonly_midlines):
            return False

        if self.boxes:
            try:
                sx, sy = self.points[i1]
                ex, ey = self.points[i2]
            except Exception:
                return False

            def point_box(x, y):
                for bi, (xmin, ymin, xmax, ymax) in enumerate(self.boxes):
                    if xmin <= x <= xmax and ymin <= y <= ymax:
                        return bi
                return None

            b1 = point_box(float(sx), float(sy))
            b2 = point_box(float(ex), float(ey))

            # endpoints must be in SAME bbox
            if b1 is None or b2 is None or b1 != b2:
                QMessageBox.warning(self, "Out of bounds",
                    f"Auto midline {key} endpoints are not in the same bounding box.\nInsertion cancelled.")
                return False

            # ensure all poly points are inside that same bbox
            xmin, ymin, xmax, ymax = self.boxes[b1]
            for (x, y) in poly:
                if not (xmin <= float(x) <= xmax and ymin <= float(y) <= ymax):
                    QMessageBox.warning(self, "Out of bounds",
                        f"Auto midline {key} poly not entirely inside its bounding box.\nInsertion cancelled.")
                    return False

        # success
        self.midlines[key] = [(float(x), float(y)) for (x, y) in poly]
        self.update()
        return True
        
    def _validated_set_midline(self, key, poly):
        """Internal helper to enforce bbox rules before saving a midline."""
        i1, i2 = key
        if i1 == i2:
            return False
        if (key in self.midlines) or (key in self.connections) \
        or (key in self.readonly_connections) or (key in self.readonly_midlines):
            return False

        if self.boxes:
            sx, sy = self.points[i1]
            ex, ey = self.points[i2]
            b1 = self._point_box_index(float(sx), float(sy))
            b2 = self._point_box_index(float(ex), float(ey))
            if b1 is None or b2 is None or b1 != b2 or not self._poly_inside_box_index(poly, b1):
                return False

        self.midlines[key] = [(float(x), float(y)) for (x, y) in poly]
        return True

    def add_midline_auto(self, i1, i2, poly):
        return self._validated_set_midline(self._sorted(i1, i2), poly)

    '''def paintEvent(self, event):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)

        if self.image_pixmap:
            sw = int(self.img_w * self.scale)
            sh = int(self.img_h * self.scale)

            # expose transform for hit-testing
            self._last_draw_scale = self.scale
            self._last_draw_xoff = 0
            self._last_draw_yoff = 0

            # draw image always at (0,0), scrollbars handle pan
            qp.drawPixmap(0, 0, sw, sh, self.image_pixmap)
        else:
            self._last_draw_scale, self._last_draw_xoff, self._last_draw_yoff = 1.0, 0.0, 0.0

        crop_xmin, crop_ymin = getattr(self, "crop_offset", (0, 0))
        def apply_offset(pt):
            return (pt[0] + crop_xmin, pt[1] + crop_ymin)
        scale = self._last_draw_scale
        xoff = self._last_draw_xoff
        yoff = self._last_draw_yoff

        # --- boxes ---
        qp.setPen(QPen(QColor(0, 128, 255), 3))
        for xmin, ymin, xmax, ymax in self.boxes:
            qp.drawRect(
                int(xmin * scale + xoff),
                int(ymin * scale + yoff),
                int((xmax - xmin) * scale),
                int((ymax - ymin) * scale),
            )

        # (leave your connections, points, midlines, polylines drawing logic unchanged)
        # Just remember: xoff,yoff are always 0 now.

        # --- read-only connections ---
        qp.setPen(QPen(QColor(150, 150, 150), 2, Qt.DashLine))
        for i1, i2 in self.readonly_connections:
            if i1 < len(self.points) and i2 < len(self.points):
                p1 = apply_offset(self.points[i1])
                p2 = apply_offset(self.points[i2])
                qp.drawLine(
                    QPoint(int(p1[0] * scale + xoff), int(p1[1] * scale + yoff)),
                    QPoint(int(p2[0] * scale + xoff), int(p2[1] * scale + yoff))
                )

        # --- read-only midlines ---
        qp.setPen(QPen(QColor(150, 150, 0), 2))
        for key, poly in self.readonly_midlines.items():
            for i in range(1, len(poly)):
                p1 = apply_offset(poly[i - 1])
                p2 = apply_offset(poly[i])
                qp.drawLine(
                    QPoint(int(p1[0] * scale + xoff), int(p1[1] * scale + yoff)),
                    QPoint(int(p2[0] * scale + xoff), int(p2[1] * scale + yoff))
                )

        # --- editable connections ---
        for idx, (i1, i2) in enumerate(self.connections):
            if i1 < len(self.points) and i2 < len(self.points):
                p1 = apply_offset(self.points[i1])
                p2 = apply_offset(self.points[i2])
                thick = 6 if (self.connection_mode and self.connecting_index is None
                            and idx == self.hover_line_index and self.hover_index is None) else 4
                qp.setPen(QPen(QColor(0, 0, 0), thick))
                qp.drawLine(
                    QPoint(int(p1[0] * scale + xoff), int(p1[1] * scale + yoff)),
                    QPoint(int(p2[0] * scale + xoff), int(p2[1] * scale + yoff))
                )

        # --- points ---
        for i, (x, y) in enumerate(self.points):
            x, y = apply_offset((x, y))
            center = QPoint(int(x * scale + xoff), int(y * scale + yoff))
            brush = QColor(0, 200, 0) if i == self.hover_index or (
                self.connection_mode and i == self.connecting_index) else QColor(200, 80, 80)
            qp.setBrush(brush)
            qp.setPen(Qt.NoPen)
            qp.drawEllipse(center, int(self.point_radius * scale), int(self.point_radius * scale))

        # --- editable midlines ---
        for key, poly in self.midlines.items():
            if len(poly) < 2:
                continue
            thick = 8 if (self.connection_mode and key == self._hover_midline_key) else 4
            qp.setPen(QPen(QColor(0, 200, 200), thick))
            for i in range(1, len(poly)):
                p1 = apply_offset(poly[i - 1])
                p2 = apply_offset(poly[i])
                qp.drawLine(
                    QPoint(int(p1[0] * scale + xoff), int(p1[1] * scale + yoff)),
                    QPoint(int(p2[0] * scale + xoff), int(p2[1] * scale + yoff))
                )

        # --- live polyline (manual midline in progress) ---
        if self.polyline_mode and len(self.polyline) >= 1:
            qp.setPen(QPen(QColor(0, 200, 200), 4))
            for i in range(1, len(self.polyline)):
                p1 = apply_offset(self.polyline[i - 1])
                p2 = apply_offset(self.polyline[i])
                qp.drawLine(
                    QPoint(int(p1[0] * scale + xoff), int(p1[1] * scale + yoff)),
                    QPoint(int(p2[0] * scale + xoff), int(p2[1] * scale + yoff))
                )'''

    #def _min_scale(self):
    #    # absolute minimum: fit-to-view (no smaller)
    #    return max(0.01, self._fit_scale or 1.0)
    
    def _min_scale(self):
        """
        Minimum zoom so the entire image always fits within the viewport.
        Prevents shimmying when zoomed all the way out.
        """
        if not self._sa or not self.img_w or not self.img_h:
            return 1.0

        vp = self._sa.viewport()
        vw, vh = max(1, vp.width()), max(1, vp.height())

        # Fit-to-view scale: image fits both width and height
        fit_scale = min(vw / float(self.img_w), vh / float(self.img_h))
        return max(0.01, fit_scale)

    
    def _fit_to_view(self):
        if self.image_pixmap is None:
            return
        self._hook_scroll_area()
        if not self._sa:
            return

        vp = self._sa.viewport()
        vw, vh = max(1, vp.width()), max(1, vp.height())

        # always start at native resolution
        base_scale = 1.0

        self._fit_scale = base_scale
        self.scale = base_scale

        sw, sh = int(self.img_w * self.scale), int(self.img_h * self.scale)

        # center if smaller than viewport, else align to top-left
        if sw < vw:
            self.pan_x = (vw - sw) // 2
        else:
            self.pan_x = 0

        if sh < vh:
            self.pan_y = (vh - sh) // 2
        else:
            self.pan_y = 0

        self._user_zoomed = False
        self.update_canvas_size()
        self.update()

        print(f"[FIT] img {self.img_w}x{self.img_h}, vp {vw}x{vh}, "
            f"scale={self.scale:.3f}, pan=({self.pan_x},{self.pan_y})")

    '''def update_canvas_size(self):
        """Update canvas so QScrollArea knows exact content size."""
        sw = int(self.img_w * self.scale)
        sh = int(self.img_h * self.scale)
        self.setMinimumSize(sw, sh)
        self.resize(sw, sh)  # content size = scaled image
        self.update()

    def _enforce_bounds(self, margin_frac=0.25):
        """
        Keep image within viewport bounds.
        margin_frac allows at most X fraction of viewport whitespace per side.
        """
        if not self._sa:
            return
        vp = self._sa.viewport()
        vw, vh = vp.width(), vp.height()
        sw, sh = int(self.img_w * self.scale), int(self.img_h * self.scale)

        # maximum allowed margin (so image can drift a little, but not disappear)
        max_x_margin = int(vw * margin_frac)
        max_y_margin = int(vh * margin_frac)

        # Clamp X
        min_x = vw - sw - max_x_margin
        max_x = max_x_margin
        self.pan_x = max(min_x, min(max_x, self.pan_x))

        # Clamp Y
        min_y = vh - sh - max_y_margin
        max_y = max_y_margin
        self.pan_y = max(min_y, min(max_y, self.pan_y))
    
    def event(self, ev):
        if ev.type() == QtCore.QEvent.Gesture:
            return self.gestureEvent(ev)
        return super().event(ev)

    def gestureEvent(self, ev: QtWidgets.QGestureEvent):
        pinch = ev.gesture(Qt.PinchGesture)
        if pinch:
            scale_change = pinch.scaleFactor()
            if abs(scale_change - 1.0) > 1e-3:
                center = pinch.centerPoint().toPoint()
                self._zoom_at(center, scale_change)
        return True

    def _zoom_at(self, pos: QtCore.QPoint, factor: float):
        self._hook_scroll_area()

        old = self.scale
        new = max(self._min_scale(), min(10.0, old * factor))
        if abs(new - old) < 1e-9:
            return

        if not self._sa:
            self.scale = new
            self._user_zoomed = True
            self.update_canvas_size()
            return

        vp = self._sa.viewport()
        vw, vh = vp.width(), vp.height()
        hsb, vsb = self._sa.horizontalScrollBar(), self._sa.verticalScrollBar()
        old_hval, old_vval = hsb.value(), vsb.value()

        mx, my = int(pos.x()), int(pos.y())

        # Anchor pixel BEFORE zoom
        img_x = (old_hval + mx) / max(old, 1e-9)
        img_y = (old_vval + my) / max(old, 1e-9)

        # Apply new scale
        self.scale = new
        self._user_zoomed = True
        new_w, new_h = int(self.img_w * new), int(self.img_h * new)
        self.setMinimumSize(new_w, new_h)
        self.resize(new_w, new_h)

        # Scroll target (ideal)
        new_hval = int(img_x * new - mx)
        new_vval = int(img_y * new - my)

        # Clamp
        if new_w > vw:
            new_hval = max(0, min(new_hval, new_w - vw))
        else:
            new_hval = (new_w - vw) // 2

        if new_h > vh:
            new_vval = max(0, min(new_vval, new_h - vh))
        else:
            new_vval = (new_h - vh) // 2

        # Apply
        hsb.setValue(new_hval)
        vsb.setValue(new_vval)

        # Debug: compute where that anchor ended up
        final_x = img_x * new - hsb.value()
        final_y = img_y * new - vsb.value()

        print("\n=== _zoom_at DEBUG ===")
        print(f"factor: {factor:.3f}")
        print(f"old scale: {old:.3f}, new scale: {new:.3f}")
        print(f"viewport size: {vw}x{vh}")
        print(f"image size: {self.img_w}x{self.img_h}")
        print(f"content size (new): {new_w}x{new_h}")
        print(f"cursor pos (widget): ({mx},{my})")
        print(f"scrollbar old: H={old_hval}, V={old_vval}, max=({hsb.maximum()},{vsb.maximum()})")
        print(f"anchor before zoom (img_x,img_y): ({img_x:.2f},{img_y:.2f})")
        print(f"scrollbar new: H={hsb.value()}, V={vsb.value()}, max=({hsb.maximum()},{vsb.maximum()})")
        print(f"anchor after zoom (viewport coords): ({final_x:.2f},{final_y:.2f})")
        print(f"cursor vs anchor delta: dx={final_x - mx:.2f}, dy={final_y - my:.2f})")
        print("======================\n")

        self.update()
   
    def wheelEvent(self, event: QtGui.QWheelEvent):
        # Let Ctrl+Wheel pass through for default scroll behavior if desired
        if event.modifiers() & Qt.ControlModifier:
            super().wheelEvent(event)
            return

        # Touchpads sometimes send pixelDelta instead of angleDelta
        delta = event.angleDelta().y()
        if delta == 0 and not event.pixelDelta().isNull():
            delta = event.pixelDelta().y()
        if delta == 0:
            return  # no useful delta

        factor = 1.2 if delta > 0 else 1 / 1.2
        self._zoom_at(event.pos(), factor)
        event.accept()

    def event(self, ev):
        if ev.type() == QtCore.QEvent.Gesture:
            return self.gestureEvent(ev)
        return super().event(ev)

    def gestureEvent(self, ev: QtWidgets.QGestureEvent):
        pinch = ev.gesture(Qt.PinchGesture)
        if pinch:
            scale_change = pinch.scaleFactor()
            if abs(scale_change - 1.0) > 1e-3:
                center = pinch.centerPoint().toPoint()
                self._zoom_at(center, scale_change)
        return True'''


    # ========= paintEvent (draw with float pan_x/pan_y) =========
    def paintEvent(self, event):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)

        # scaled image size
        sw = int(self.img_w * self.scale)
        sh = int(self.img_h * self.scale)

        # expose transform for hit-testing & overlays
        self._last_draw_scale = self.scale
        self._last_draw_xoff = float(self.pan_x)
        self._last_draw_yoff = float(self.pan_y)

        if self.image_pixmap:
            # draw the image at floating pan offset (cast to int for painting)
            qp.drawPixmap(int(self.pan_x), int(self.pan_y), sw, sh, self.image_pixmap)

        # convenient locals for overlay transforms
        scale = self._last_draw_scale
        xoff  = self._last_draw_xoff
        yoff  = self._last_draw_yoff

        crop_xmin, crop_ymin = getattr(self, "crop_offset", (0, 0))
        def apply_offset(pt):
            return (pt[0] + crop_xmin, pt[1] + crop_ymin)

        # --- boxes ---
        qp.setPen(QPen(QColor(0, 128, 255), 3))
        for xmin, ymin, xmax, ymax in self.boxes:
            qp.drawRect(
                int(xmin * scale + xoff),
                int(ymin * scale + yoff),
                int((xmax - xmin) * scale),
                int((ymax - ymin) * scale),
            )

        # --- read-only connections ---
        qp.setPen(QPen(QColor(150, 150, 150), 2, Qt.DashLine))
        for i1, i2 in self.readonly_connections:
            if i1 < len(self.points) and i2 < len(self.points):
                p1 = apply_offset(self.points[i1])
                p2 = apply_offset(self.points[i2])
                qp.drawLine(
                    QPoint(int(p1[0] * scale + xoff), int(p1[1] * scale + yoff)),
                    QPoint(int(p2[0] * scale + xoff), int(p2[1] * scale + yoff))
                )

        # --- read-only midlines ---
        qp.setPen(QPen(QColor(150, 150, 0), 2))
        for key, poly in self.readonly_midlines.items():
            for i in range(1, len(poly)):
                p1 = apply_offset(poly[i - 1])
                p2 = apply_offset(poly[i])
                qp.drawLine(
                    QPoint(int(p1[0] * scale + xoff), int(p1[1] * scale + yoff)),
                    QPoint(int(p2[0] * scale + xoff), int(p2[1] * scale + yoff))
                )

        # --- editable connections ---
        for idx, (i1, i2) in enumerate(self.connections):
            if i1 < len(self.points) and i2 < len(self.points):
                p1 = apply_offset(self.points[i1])
                p2 = apply_offset(self.points[i2])
                thick = 6 if (self.connection_mode and self.connecting_index is None
                            and idx == self.hover_line_index and self.hover_index is None) else 4
                qp.setPen(QPen(QColor(0, 0, 0), thick))
                qp.drawLine(
                    QPoint(int(p1[0] * scale + xoff), int(p1[1] * scale + yoff)),
                    QPoint(int(p2[0] * scale + xoff), int(p2[1] * scale + yoff))
                )

        # --- points ---
        for i, (x, y) in enumerate(self.points):
            x, y = apply_offset((x, y))
            center = QPoint(int(x * scale + xoff), int(y * scale + yoff))
            brush = QColor(0, 200, 0) if i == self.hover_index or (
                self.connection_mode and i == self.connecting_index) else QColor(200, 80, 80)
            qp.setBrush(brush)
            qp.setPen(Qt.NoPen)
            #qp.drawEllipse(center, int(self.point_radius * scale), int(self.point_radius * scale))
            # keep circle size constant on screen regardless of zoom
            r_screen = int(self.point_radius)
            qp.drawEllipse(center, r_screen, r_screen)

        # --- editable midlines ---
        qp.setPen(QPen(QColor(0, 200, 200), 4))
        for key, poly in self.midlines.items():
            if len(poly) < 2:
                continue
            thick = 8 if (self.connection_mode and key == self._hover_midline_key) else 4
            qp.setPen(QPen(QColor(0, 200, 200), thick))
            for i in range(1, len(poly)):
                p1 = apply_offset(poly[i - 1])
                p2 = apply_offset(poly[i])
                qp.drawLine(
                    QPoint(int(p1[0] * scale + xoff), int(p1[1] * scale + yoff)),
                    QPoint(int(p2[0] * scale + xoff), int(p2[1] * scale + yoff))
                )

        # --- live polyline (manual midline in progress) ---
        if self.polyline_mode and len(self.polyline) >= 1:
            qp.setPen(QPen(QColor(0, 200, 200), 4))
            for i in range(1, len(self.polyline)):
                p1 = apply_offset(self.polyline[i - 1])
                p2 = apply_offset(self.polyline[i])
                qp.drawLine(
                    QPoint(int(p1[0] * scale + xoff), int(p1[1] * scale + yoff)),
                    QPoint(int(p2[0] * scale + xoff), int(p2[1] * scale + yoff))
                )


    # ========= fit-to-view (center using float pan; keep widget = viewport size) =========
    def _fit_to_view(self):
        if self.image_pixmap is None:
            return
        self._hook_scroll_area()
        if not self._sa:
            return

        vp = self._sa.viewport()
        vw, vh = max(1, vp.width()), max(1, vp.height())

        # Start at native resolution (you can change to fit image: min(vw/img_w, vh/img_h))
        base_scale = 1.0
        self._fit_scale = base_scale
        self.scale = base_scale

        sw, sh = int(self.img_w * self.scale), int(self.img_h * self.scale)

        # center image in viewport (float pan)
        self.pan_x = (vw - sw) / 2.0
        self.pan_y = (vh - sh) / 2.0

        self._user_zoomed = False
        self.update_canvas_size()
        self.update()

        print(f"[FIT] img {self.img_w}x{self.img_h}, vp {vw}x{vh}, "
            f"scale={self.scale:.3f}, pan=({self.pan_x:.1f},{self.pan_y:.1f})")


    # ========= keep widget the size of the viewport (no scrollbars) =========
    def update_canvas_size(self):
        """Keep content size equal to viewport; all pan/zoom is float-rendered."""
        if self._sa:
            vp = self._sa.viewport()
            w, h = max(1, vp.width()), max(1, vp.height())
        else:
            # fallback before SA is known
            w = h = 1
        self.setMinimumSize(w, h)
        self.resize(w, h)
        self.update()


    # ========= bounds clamp for float pan =========
    '''def _enforce_bounds(self, margin_frac=0.25):
        """Keep image within viewport-ish bounds, allow some whitespace margin."""
        if not self._sa:
            return
        vp = self._sa.viewport()
        vw, vh = vp.width(), vp.height()
        sw, sh = float(self.img_w * self.scale), float(self.img_h * self.scale)

        max_x_margin = vw * margin_frac
        max_y_margin = vh * margin_frac

        # The image top-left (pan_x, pan_y) is allowed to drift within these margins
        min_x = vw - sw - max_x_margin
        max_x = max_x_margin
        min_y = vh - sh - max_y_margin
        max_y = max_y_margin

        before = (self.pan_x, self.pan_y)
        self.pan_x = max(min_x, min(max_x, self.pan_x))
        self.pan_y = max(min_y, min(max_y, self.pan_y))
        after = (self.pan_x, self.pan_y)

        if before != after:
            print(f"[BOUNDS] pan {before} -> {after}  (vw={vw},vh={vh}, sw={sw:.1f},sh={sh:.1f})")'''

    def _enforce_bounds(self, margin_frac=None):
        """
        Keep image neatly within viewport.
        - If the scaled image is smaller than the viewport on an axis -> hard center it on that axis.
        - If larger -> clamp pan on that axis with a small white margin.
        """
        if not self._sa:
            return

        vp = self._sa.viewport()
        vw, vh = float(vp.width()), float(vp.height())
        sw, sh = float(self.img_w) * float(self.scale), float(self.img_h) * float(self.scale)

        # configurable margin fraction (portion of viewport allowed as whitespace on each side)
        if margin_frac is None:
            margin_frac = getattr(self, "_pan_margin_frac", 0.05)  # default ~8%
        
        max_margin = int(min(vw, vh) * margin_frac)
        max_x_margin = max_margin
        max_y_margin = max_margin

        prev_pan_x, prev_pan_y = self.pan_x, self.pan_y

        # --- X axis ---
        if sw <= vw:
            # Image narrower than viewport: center exactly (no drift, no shimmy)
            self.pan_x = (vw - sw) * 0.5
        else:
            # Image wider than viewport: clamp with margin
            #max_x_margin = vw * margin_frac
            min_x = vw - sw - max_x_margin   # far left
            max_x = max_x_margin             # far right
            if self.pan_x < min_x:
                self.pan_x = min_x
            elif self.pan_x > max_x:
                self.pan_x = max_x

        # --- Y axis ---
        if sh <= vh:
            # Image shorter than viewport: center exactly
            self.pan_y = (vh - sh) * 0.5
        else:
            # Image taller than viewport: clamp with margin
            #max_y_margin = vh * margin_frac
            min_y = vh - sh - max_y_margin   # top
            max_y = max_y_margin             # bottom
            if self.pan_y < min_y:
                self.pan_y = min_y
            elif self.pan_y > max_y:
                self.pan_y = max_y

        if (prev_pan_x, prev_pan_y) != (self.pan_x, self.pan_y):
            print(f"[BOUNDS] clamped/centered pan: ({prev_pan_x:.1f},{prev_pan_y:.1f}) -> ({self.pan_x:.1f},{self.pan_y:.1f})  "
                f"(vw={vw:.0f},vh={vh:.0f}, sw={sw:.0f},sh={sh:.0f}, margin={margin_frac:.2f})")

    # ========= precise cursor-anchored zoom (NO scrollbar drift) + deep debug =========
    def _zoom_at(self, pos: QtCore.QPoint, factor: float):
        """Zoom with the anchor fixed under `pos` (widget coords), using float pan."""
        self._hook_scroll_area()

        old = float(self.scale)
        new = float(max(self._min_scale(), min(10.0, old * float(factor))))
        if abs(new - old) < 1e-9:
            print("[ZOOM] Skipped (new≈old)")
            return

        if not self._sa:
            self.scale = new
            self._user_zoomed = True
            self.update_canvas_size()
            print("[ZOOM] No scroll area yet. scale set to", new)
            return

        vp = self._sa.viewport()
        vw, vh = vp.width(), vp.height()

        mx, my = float(pos.x()), float(pos.y())

        # --- Anchor in image coords BEFORE zoom (using current pan/scale) ---
        img_x = (mx - self.pan_x) / max(old, 1e-9)
        img_y = (my - self.pan_y) / max(old, 1e-9)

        # --- Apply new scale ---
        self.scale = new
        self._user_zoomed = True

        # --- Adjust pan so that the same image pixel stays under the cursor ---
        # mx = pan_x + img_x * new  => pan_x = mx - img_x * new
        new_pan_x = mx - img_x * new
        new_pan_y = my - img_y * new

        # Debug before clamping
        before_pan = (self.pan_x, self.pan_y)
        unclamped_pan = (new_pan_x, new_pan_y)

        self.pan_x, self.pan_y = new_pan_x, new_pan_y

        # Clamp pan so the image doesn't disappear
        self._enforce_bounds()

        # Where does the anchor end up after clamping?
        final_x = img_x * self.scale + self.pan_x
        final_y = img_y * self.scale + self.pan_y

        # Sizes for debug
        sw, sh = int(self.img_w * self.scale), int(self.img_h * self.scale)

        print("\n=== _zoom_at DEBUG (float pan/zoom) ===")
        print(f"factor: {float(factor):.3f}")
        print(f"scale: {old:.3f} -> {self.scale:.3f}")
        print(f"viewport: {vw}x{vh}")
        print(f"image: {self.img_w}x{self.img_h}  scaled: {sw}x{sh}")
        print(f"cursor (widget): ({mx:.1f},{my:.1f})")
        print(f"pan before: ({before_pan[0]:.2f},{before_pan[1]:.2f})  unclamped: ({unclamped_pan[0]:.2f},{unclamped_pan[1]:.2f})  final: ({self.pan_x:.2f},{self.pan_y:.2f})")
        print(f"anchor (image coords before): ({img_x:.2f},{img_y:.2f})")
        print(f"anchor after -> viewport coords: ({final_x:.2f},{final_y:.2f})")
        print(f"cursor vs anchor delta: dx={final_x - mx:.2f}, dy={final_y - my:.2f}")
        print("=======================================\n")

        self.update()

    '''def _zoom_at(self, pos: QtCore.QPoint, factor: float):
        """Zoom with the anchor fixed under `pos` (widget coords)."""
        self._hook_scroll_area()

        old = self.scale
        new = max(self._min_scale(), min(10.0, old * factor))
        if abs(new - old) < 1e-9:
            return

        if not self._sa:
            self.scale = new
            self._user_zoomed = True
            self.update_canvas_size()
            return

        vp = self._sa.viewport()
        vw, vh = vp.width(), vp.height()
        hsb, vsb = self._sa.horizontalScrollBar(), self._sa.verticalScrollBar()
        old_hval, old_vval = hsb.value(), vsb.value()

        mx, my = int(pos.x()), int(pos.y())

        # Anchor pixel BEFORE zoom (in image coords)
        img_x = (old_hval + mx) / max(old, 1e-9)
        img_y = (old_vval + my) / max(old, 1e-9)

        # Apply new scale
        self.scale = new
        self._user_zoomed = True
        new_w, new_h = int(self.img_w * new), int(self.img_h * new)
        self.setMinimumSize(new_w, new_h)
        self.resize(new_w, new_h)

        # Scroll target (ideal to keep cursor fixed)
        new_hval = int(img_x * new - mx)
        new_vval = int(img_y * new - my)

        # Clamp / recenter if smaller than viewport
        if new_w > vw:
            new_hval = max(0, min(new_hval, new_w - vw))
        else:
            new_hval = (new_w - vw) // 2  # recenter X

        if new_h > vh:
            new_vval = max(0, min(new_vval, new_h - vh))
        else:
            new_vval = (new_h - vh) // 2  # recenter Y

        # Apply scrollbars
        hsb.setValue(new_hval)
        vsb.setValue(new_vval)

        # Debug info
        final_x = img_x * new - hsb.value()
        final_y = img_y * new - vsb.value()

        print("\n=== _zoom_at DEBUG ===")
        print(f"factor: {factor:.3f}")
        print(f"old scale: {old:.3f}, new scale: {new:.3f}")
        print(f"viewport size: {vw}x{vh}")
        print(f"image size: {self.img_w}x{self.img_h}")
        print(f"content size (new): {new_w}x{new_h}")
        print(f"cursor pos (widget): ({mx},{my})")
        print(f"scrollbar old: H={old_hval}, V={old_vval}, max=({hsb.maximum()},{vsb.maximum()})")
        print(f"anchor before zoom (img_x,img_y): ({img_x:.2f},{img_y:.2f})")
        print(f"scrollbar new: H={hsb.value()}, V={vsb.value()}, max=({hsb.maximum()},{vsb.maximum()})")
        print(f"anchor after zoom (viewport coords): ({final_x:.2f},{final_y:.2f})")
        print(f"cursor vs anchor delta: dx={final_x - mx:.2f}, dy={final_y - my:.2f})")
        print("======================\n")

        self.update()'''

    def _delete_point_reindex(self, idx):
        """Delete a point and reindex everything cleanly."""
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

        # --- FIX hover glitch ---
        self.hover_index = None
        self.hover_line_index = None
        self._hover_midline_key = None

        self.update()

    # ========= wheel zoom (uses float zoom_at; blocks default scroll) =========
    def wheelEvent(self, event: QtGui.QWheelEvent):
        # Let Ctrl+Wheel pass through for default scroll if you ever want it
        if event.modifiers() & Qt.ControlModifier:
            super().wheelEvent(event)
            return

        delta = event.angleDelta().y()
        if delta == 0 and not event.pixelDelta().isNull():
            delta = event.pixelDelta().y()
        if delta == 0:
            return

        factor = 1.2 if delta > 0 else 1.0 / 1.2
        self._zoom_at(event.pos(), factor)
        event.accept()


    # ========= gesture routing (accept to prevent scroll area from interfering) =========
    def event(self, ev: QtCore.QEvent):
        if ev.type() == QtCore.QEvent.Gesture:
            ev.accept()   # stop propagation to scrollbars/parent
            return self.gestureEvent(ev)
        return super().event(ev)


    def gestureEvent(self, ev: QtWidgets.QGestureEvent):
        pinch = ev.gesture(Qt.PinchGesture)
        if pinch:
            if pinch.changeFlags() & QtWidgets.QPinchGesture.ScaleFactorChanged:
                scale_change = pinch.scaleFactor()
                if abs(scale_change - 1.0) > 1e-3:
                    center = pinch.centerPoint().toPoint()
                    self._zoom_at(center, scale_change)
            ev.accept()
            return True
        return False


    # ========= mouse panning (middle button drag) with debug =========
    def mousePressEvent(self, event):
        # Only consider the image rect for editing clicks
        img_rect = QtCore.QRect(
            int(self._last_draw_xoff),
            int(self._last_draw_yoff),
            int(self.img_w * self.scale),
            int(self.img_h * self.scale),
        )

        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_last = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            print(f"[PAN] start at {self._pan_last}")
            return

        # ----- EXISTING EDITOR LOGIC (unchanged below) -----
        if not img_rect.contains(event.pos()):
            return  # ignore clicks in gray margin

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
                    if getattr(self, "_just_committed_midline", False):
                        self._just_committed_midline = False
                        return
                    if point_i is not None and point_i != self._start_idx:
                        self._commit_midline(point_i)
                    elif point_i is None:
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
                if point_i is None:
                    self.points.append((float(p[0]), float(p[1])))
                    print(f"[PRESS] Added new point at {p}")
                else:
                    if any(point_i in c for c in self.readonly_connections) or \
                    any(point_i in k for k in self.readonly_midlines.keys()):
                        return
                    self._delete_point_reindex(point_i)
            else:
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


    def mouseMoveEvent(self, event):
        if self._panning and self._pan_last is not None:
            dx = float(event.pos().x() - self._pan_last.x())
            dy = float(event.pos().y() - self._pan_last.y())
            self.pan_x += dx
            self.pan_y += dy
            self._pan_last = event.pos()
            self._enforce_bounds()
            print(f"[PAN] move dx={dx:.1f}, dy={dy:.1f} -> pan=({self.pan_x:.1f},{self.pan_y:.1f})")
            self.update()
            return

        # --- existing hover / drawing logic ---
        p = self._to_image_coords(event.pos())
        point_i = self._find_point_at(p)

        if self.polyline_mode and self._is_drawing and (event.buttons() & Qt.LeftButton):
            if point_i is not None and point_i != self._start_idx:
                print(f"[MOVE] Hovering endpoint {point_i}, attempting commit")
                self._commit_midline(point_i)
                return
            else:
                self._add_poly_point(p)
                self.update()
                return

        self.hover_index = self._find_point_at(p)
        if self.connection_mode and self.connecting_index is None and self.hover_index is None:
            self.hover_line_index = self._find_line_at(p)
        else:
            self.hover_line_index = None

        self._hover_midline_key = self._midline_hit_test(event.pos(), 10.0) if not self._is_drawing else None
        self.update()


    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton and self._panning:
            self._panning = False
            self._pan_last = None
            self.unsetCursor()
            print("[PAN] end")
            return

        if event.button() == Qt.RightButton:
            self._erase_timer.stop()
