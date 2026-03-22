#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap, QImage
from PyQt5.QtCore import Qt, QPointF

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
        # Sanitize any preloaded endpoints to image bounds.
        if self.points:
            xmax0 = float(max(0, self.img_w - 1))
            ymax0 = float(max(0, self.img_h - 1))
            pts = []
            for p in self.points:
                try:
                    x = float(p[0])
                    y = float(p[1])
                except Exception:
                    continue
                x = min(max(x, 0.0), xmax0)
                y = min(max(y, 0.0), ymax0)
                pts.append((x, y))
            self.points = pts
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
        self._active_poly_regions = []

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
        self._erase_timer.timeout.connect(self._erase_tick)
        self._erase_start_time = None

        # Optional overlay background state.
        self.overlay_pixmap = None
        self.use_overlay = False
        self.overlay_toggle_cb = None
        self._minimap_corner = "top_right"

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
            self._active_poly_regions = []
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

    def _undo_polyline_points(self, n=1):
        removed = 0
        for _ in range(max(1, int(n))):
            if not self.polyline:
                break
            self.polyline.pop()
            removed += 1
        if not self.polyline:
            self._is_drawing = False
            self._start_idx = None
        return removed

    def _erase_tick(self):
        if not self._is_drawing:
            self._erase_timer.stop()
            return
        removed = self._undo_polyline_points(2)
        if removed > 0:
            self.update()
        else:
            self._erase_timer.stop()

    def _point_radius_screen(self):
        r = self.point_radius * (self.scale ** 0.5)
        return max(1, min(16, r))

    # ---------- rest of your existing methods ----------
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Backspace, Qt.Key_Z):
            if self.polyline_mode and self._is_drawing and self.polyline:
                self._undo_polyline_points(2)
                self.update()
                return
        if event.key() == Qt.Key_M:
            self._toggle_minimap_corner()
            return
        if event.key() in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
            step = 36.0
            if event.modifiers() & Qt.ShiftModifier:
                step = 80.0
            if event.key() == Qt.Key_Left:
                self.pan_x += step
            elif event.key() == Qt.Key_Right:
                self.pan_x -= step
            elif event.key() == Qt.Key_Up:
                self.pan_y += step
            elif event.key() == Qt.Key_Down:
                self.pan_y -= step
            self._enforce_bounds()
            self.update()
            return
        if event.key() == Qt.Key_T:
            if callable(self.overlay_toggle_cb):
                self.overlay_toggle_cb()
                return
            if self.overlay_pixmap is not None:
                self.use_overlay = not self.use_overlay
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

    def _clamp_xy_to_image(self, x, y, edge_snap_px=1.0):
        """
        Clamp coordinates to image bounds and snap near-border points to edges.
        """
        x = float(x)
        y = float(y)
        xmax = float(max(0, self.img_w - 1))
        ymax = float(max(0, self.img_h - 1))

        if abs(x) <= edge_snap_px:
            x = 0.0
        elif abs(x - xmax) <= edge_snap_px:
            x = xmax
        if abs(y) <= edge_snap_px:
            y = 0.0
        elif abs(y - ymax) <= edge_snap_px:
            y = ymax

        x = min(max(x, 0.0), xmax)
        y = min(max(y, 0.0), ymax)
        return x, y


    '''def _find_point_at(self, pos):
        # Keep hitbox radius constant in screen space, not image space
        r = self._point_radius_screen()
        r2 = r * r
        for i, (x, y) in enumerate(self.points):
            if (x - pos[0])**2 + (y - pos[1])**2 <= r2:
                return i
        return None'''
    def _find_point_at(self, pos_img):
        # pos_img is in IMAGE coordinates
        r_screen = self._point_radius_screen()
        r_img = r_screen / self.scale   # convert to image space
        r2 = r_img * r_img

        for i, (x, y) in enumerate(self.points):
            dx = x - pos_img[0]
            dy = y - pos_img[1]
            if dx*dx + dy*dy <= r2:
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

    def _points_share_any_box(self, i1, i2, tol=0.0):
        """True if both point indices lie in at least one common bounding box."""
        if not self.boxes:
            return True
        if i1 < 0 or i2 < 0 or i1 >= len(self.points) or i2 >= len(self.points):
            return False
        x1, y1 = map(float, self.points[i1])
        x2, y2 = map(float, self.points[i2])
        for (xmin, ymin, xmax, ymax) in self.boxes:
            if (
                (xmin - tol) <= x1 <= (xmax + tol)
                and (ymin - tol) <= y1 <= (ymax + tol)
                and (xmin - tol) <= x2 <= (xmax + tol)
                and (ymin - tol) <= y2 <= (ymax + tol)
            ):
                return True
        return False

    def _containing_boxes_for_point(self, x, y, tol=0.0):
        """Return all bbox rectangles containing (x,y)."""
        out = []
        for (xmin, ymin, xmax, ymax) in (self.boxes or []):
            if (
                (xmin - tol) <= x <= (xmax + tol)
                and (ymin - tol) <= y <= (ymax + tol)
            ):
                out.append((float(xmin), float(ymin), float(xmax), float(ymax)))
        return out

    def _poly_point_within_active_regions(self, p):
        if not self._active_poly_regions:
            return True
        x, y = float(p[0]), float(p[1])
        for xmin, ymin, xmax, ymax in self._active_poly_regions:
            if xmin <= x <= xmax and ymin <= y <= ymax:
                return True
        return False

    def _abort_active_polyline_out_of_bounds(self):
        QMessageBox.warning(
            self,
            "Out of bounds",
            "Polyline left the active bounding box of its start endpoint.\n"
            "Current midline was reset.",
        )
        self.polyline.clear()
        self._is_drawing = False
        self._start_idx = None
        self._active_poly_regions = []
        self.update()

    # ---------- midlines ----------
    '''def _commit_midline(self, end_idx):
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
        self.update()'''
        
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
            self._active_poly_regions = []
            self.update()
            return

        # -------------------------------------------------
        # Build the polyline in drawn order
        # -------------------------------------------------
        if len(self.polyline) >= 2:
            middle = list(self.polyline[1:-1])
            poly = (
                [tuple(map(float, self.points[start_idx]))]
                + middle
                + [tuple(map(float, self.points[end_idx]))]
            )
        else:
            poly = [
                tuple(map(float, self.points[start_idx])),
                tuple(map(float, self.points[end_idx])),
            ]

        # -------------------------------------------------
        # Trim interior points near endpoints (manual artifact fix)
        # -------------------------------------------------
        import numpy as np

        def _trim_polyline_near_endpoint(poly, endpoint_idx, radius=1.0):
            if len(poly) < 3:
                return poly

            pts = np.asarray(poly, float)

            if endpoint_idx == 0:
                E = pts[0]
                i = 1
                while i < len(pts) and np.linalg.norm(pts[i] - E) <= radius:
                    i += 1
                if i <= 1:
                    return poly
                P = 0.5 * (E + pts[i])
                new_pts = np.vstack([E, P, pts[i:]])

            else:
                E = pts[-1]
                i = len(pts) - 2
                while i >= 0 and np.linalg.norm(pts[i] - E) <= radius:
                    i -= 1
                if i >= len(pts) - 2:
                    return poly
                P = 0.5 * (E + pts[i])
                new_pts = np.vstack([pts[: i + 1], P, E])

            if len(new_pts) < 2:
                return poly

            return new_pts.tolist()

        poly = _trim_polyline_near_endpoint(poly, endpoint_idx=0, radius=1)
        poly = _trim_polyline_near_endpoint(poly, endpoint_idx=-1, radius=1)

        # -------------------------------------------------
        # Final validation: endpoints must share at least one bbox
        # -------------------------------------------------
        if self.boxes:
            sx, sy = self.points[start_idx]
            ex, ey = self.points[end_idx]
            s_boxes = self._containing_boxes_for_point(float(sx), float(sy), tol=1.0)
            e_boxes = self._containing_boxes_for_point(float(ex), float(ey), tol=1.0)

            if not s_boxes or not e_boxes:
                QMessageBox.warning(
                    self, "Out of bounds",
                    "An endpoint is outside all bounding boxes.\nCommit cancelled."
                )
                self.polyline.clear()
                self._is_drawing = False
                self._start_idx = None
                self._active_poly_regions = []
                self.update()
                return

            if not set(s_boxes).intersection(set(e_boxes)):
                QMessageBox.warning(
                    self, "Out of bounds",
                    "Endpoints do not share a bounding box.\nCommit cancelled."
                )
                self.polyline.clear()
                self._is_drawing = False
                self._start_idx = None
                self._active_poly_regions = []
                self.update()
                return

        # -------------------------------------------------
        # Commit
        # -------------------------------------------------
        self.midlines[key] = poly
        self._last_polyline_start_idx = start_idx
        self._last_polyline_end_idx = end_idx
        self._just_committed_midline = True

        self.polyline.clear()
        self._is_drawing = False
        self._start_idx = None
        self._active_poly_regions = []
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

        # ok â€” insert
        self.midlines[key] = [(float(x), float(y)) for (x, y) in poly]
        self.update()
        return True

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
            if b1 is None or b2 is None or b1 != b2:
                QMessageBox.warning(
                    self,
                    "Out of bounds",
                    f"Auto midline {key} endpoints do not share a bounding box.\n"
                    "Connection was removed.",
                )
                if key in self.connections:
                    self.connections.remove(key)
                self.update()
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
                    QPointF(int(p1[0] * scale + xoff), int(p1[1] * scale + yoff)),
                    QPointF(int(p2[0] * scale + xoff), int(p2[1] * scale + yoff))
                )

        # --- read-only midlines ---
        qp.setPen(QPen(QColor(150, 150, 0), 2))
        for key, poly in self.readonly_midlines.items():
            for i in range(1, len(poly)):
                p1 = apply_offset(poly[i - 1])
                p2 = apply_offset(poly[i])
                qp.drawLine(
                    QPointF(int(p1[0] * scale + xoff), int(p1[1] * scale + yoff)),
                    QPointF(int(p2[0] * scale + xoff), int(p2[1] * scale + yoff))
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
                    QPointF(int(p1[0] * scale + xoff), int(p1[1] * scale + yoff)),
                    QPointF(int(p2[0] * scale + xoff), int(p2[1] * scale + yoff))
                )

        # --- points ---
        for i, (x, y) in enumerate(self.points):
            x, y = apply_offset((x, y))
            center = QPointF(int(x * scale + xoff), int(y * scale + yoff))
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
                    QPointF(int(p1[0] * scale + xoff), int(p1[1] * scale + yoff)),
                    QPointF(int(p2[0] * scale + xoff), int(p2[1] * scale + yoff))
                )

        # --- live polyline (manual midline in progress) ---
        if self.polyline_mode and len(self.polyline) >= 1:
            qp.setPen(QPen(QColor(0, 200, 200), 4))
            for i in range(1, len(self.polyline)):
                p1 = apply_offset(self.polyline[i - 1])
                p2 = apply_offset(self.polyline[i])
                qp.drawLine(
                    QPointF(int(p1[0] * scale + xoff), int(p1[1] * scale + yoff)),
                    QPointF(int(p2[0] * scale + xoff), int(p2[1] * scale + yoff))
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

    def _zoom_at(self, pos: QtCore.QPointF, factor: float):
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
    """def paintEvent(self, event):
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
                    QPointF(int(p1[0] * scale + xoff), int(p1[1] * scale + yoff)),
                    QPointF(int(p2[0] * scale + xoff), int(p2[1] * scale + yoff))
                )

        # --- read-only midlines ---
        qp.setPen(QPen(QColor(150, 150, 0), 2))
        '''for key, poly in self.readonly_midlines.items():
            for i in range(1, len(poly)):
                p1 = apply_offset(poly[i - 1])
                p2 = apply_offset(poly[i])
                qp.drawLine(
                    QPointF(int(p1[0] * scale + xoff), int(p1[1] * scale + yoff)),
                    QPointF(int(p2[0] * scale + xoff), int(p2[1] * scale + yoff))
                )'''
                
        # --- read-only midlines ---
        for key, rec in self.readonly_midlines.items():
            poly = rec.get("poly", [])
            if not poly or len(poly) < 2:
                continue
            tag = rec.get("tag", "manual")
            r,g,b = rec.get("color", (150,150,0))
            # thicker, solid for processed; dashed + semi-transparent for unprocessed
            if tag == "unprocessed":
                pen = QPen(QColor(r, g, b, 180), 3, Qt.DashLine)
            elif tag == "auto":
                pen = QPen(QColor(r, g, b), 3)  # solid green
            else:  # processed manual
                pen = QPen(QColor(r, g, b), 3)
            self.qp = self.qp if hasattr(self, "qp") else None  # (no-op; keeps linters happy)
            qp.setPen(pen)
            for i in range(1, len(poly)):
                p1x, p1y = poly[i-1]
                p2x, p2y = poly[i]
                # account for crop offset + pan/scale
                p1x += getattr(self, "crop_offset", (0,0))[0]
                p1y += getattr(self, "crop_offset", (0,0))[1]
                p2x += getattr(self, "crop_offset", (0,0))[0]
                p2y += getattr(self, "crop_offset", (0,0))[1]
                qp.drawLine(
                    QPointF(int(p1x * scale + xoff), int(p1y * scale + yoff)),
                    QPointF(int(p2x * scale + xoff), int(p2y * scale + yoff))
                )



        # --- editable midlines (manual) ---
        '''for key, entry in self.midlines.items():
            if not entry:
                continue
            if isinstance(entry, dict):
                poly = entry.get("poly", [])
                unprocessed = entry.get("unprocessed", False)
            else:
                poly = entry
                unprocessed = False
            if len(poly) < 2:
                continue

            # Orange dashed if unprocessed; cyan solid if processed
            color = QColor(255, 165, 0) if unprocessed else QColor(0, 200, 255)
            style = Qt.DashLine if unprocessed else Qt.SolidLine
            width = 5 if unprocessed else 4
            thick = 8 if (self.connection_mode and key == self._hover_midline_key) else width
            qp.setPen(QPen(color, thick, style))

            for i in range(1, len(poly)):
                p1 = apply_offset(poly[i - 1])
                p2 = apply_offset(poly[i])
                qp.drawLine(
                    QPointF(int(p1[0] * scale + xoff), int(p1[1] * scale + yoff)),
                    QPointF(int(p2[0] * scale + xoff), int(p2[1] * scale + yoff))
                )'''
                
        
                # --- editable midlines (manual) ---
        for key, entry in self.midlines.items():
            # Support both dict and direct list forms
            if isinstance(entry, dict):
                poly = entry.get("poly", [])
                unprocessed = entry.get("unprocessed", False)
            else:
                poly = entry
                unprocessed = False

            if not poly or len(poly) < 2:
                continue

            # Color/style based on processing state
            color = QColor(255, 165, 0) if unprocessed else QColor(0, 200, 255)
            style = Qt.DashLine if unprocessed else Qt.SolidLine
            width = 5 if unprocessed else 4
            thick = 8 if (self.connection_mode and key == getattr(self, "_hover_midline_key", None)) else width
            qp.setPen(QPen(color, thick, style))

            for i in range(1, len(poly)):
                p1 = apply_offset(poly[i - 1])
                p2 = apply_offset(poly[i])
                qp.drawLine(
                    QPointF(int(p1[0] * scale + xoff), int(p1[1] * scale + yoff)),
                    QPointF(int(p2[0] * scale + xoff), int(p2[1] * scale + yoff))
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
                    QPointF(int(p1[0] * scale + xoff), int(p1[1] * scale + yoff)),
                    QPointF(int(p2[0] * scale + xoff), int(p2[1] * scale + yoff))
                )

        # --- points ---
        #print(len(self.points))
        for i, (x, y) in enumerate(self.points):
            x, y = apply_offset((x, y))
            center = QPointF(int(x * scale + xoff), int(y * scale + yoff))
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
                    QPointF(int(p1[0] * scale + xoff), int(p1[1] * scale + yoff)),
                    QPointF(int(p2[0] * scale + xoff), int(p2[1] * scale + yoff))
                )

        # --- live polyline (manual midline in progress) ---
        if self.polyline_mode and len(self.polyline) >= 1:
            qp.setPen(QPen(QColor(0, 200, 200), 4))
            for i in range(1, len(self.polyline)):
                p1 = apply_offset(self.polyline[i - 1])
                p2 = apply_offset(self.polyline[i])
                qp.drawLine(
                    QPointF(int(p1[0] * scale + xoff), int(p1[1] * scale + yoff)),
                    QPointF(int(p2[0] * scale + xoff), int(p2[1] * scale + yoff))
                )"""
                
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
            # choose base or overlay pixmap
            pix = self.overlay_pixmap if (self.use_overlay and self.overlay_pixmap is not None) else self.image_pixmap
            # draw the image at floating pan offset (cast to int for painting)
            qp.drawPixmap(int(self.pan_x), int(self.pan_y), sw, sh, pix)

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
        midline_color = QColor(90, 200, 0)
        qp.setPen(QPen(midline_color, 2, Qt.DashLine))
        for i1, i2 in self.readonly_connections:
            if i1 < len(self.points) and i2 < len(self.points):
                p1 = apply_offset(self.points[i1])
                p2 = apply_offset(self.points[i2])
                qp.drawLine(
                    QPointF(int(p1[0] * scale + xoff), int(p1[1] * scale + yoff)),
                    QPointF(int(p2[0] * scale + xoff), int(p2[1] * scale + yoff))
                )

        # --- read-only midlines ---
        qp.setPen(QPen(QColor(150, 150, 0), 2))
        for key, rec in self.readonly_midlines.items():
            poly = rec.get("poly", [])
            if not poly or len(poly) < 2:
                continue
            tag = rec.get("tag", "manual")
            r, g, b = rec.get("color", (150, 150, 0))
            # thicker, solid for processed; dashed + semi-transparent for unprocessed
            if tag == "unprocessed":
                pen = QPen(QColor(r, g, b, 180), 3, Qt.DashLine)
            elif tag == "auto":
                pen = QPen(QColor(r, g, b), 3)  # solid auto color
            else:  # processed manual
                pen = QPen(QColor(r, g, b), 3)
            qp.setPen(pen)
            for i in range(1, len(poly)):
                p1x, p1y = poly[i - 1]
                p2x, p2y = poly[i]
                # account for crop offset + pan/scale
                p1x += getattr(self, "crop_offset", (0, 0))[0]
                p1y += getattr(self, "crop_offset", (0, 0))[1]
                p2x += getattr(self, "crop_offset", (0, 0))[0]
                p2y += getattr(self, "crop_offset", (0, 0))[1]
                qp.drawLine(
                    QPointF(int(p1x * scale + xoff), int(p1y * scale + yoff)),
                    QPointF(int(p2x * scale + xoff), int(p2y * scale + yoff))
                )

        # --- editable midlines (manual, with unprocessed flag support) ---
        for key, entry in self.midlines.items():
            # Support both dict and direct list forms
            if isinstance(entry, dict):
                poly = entry.get("poly", [])
                unprocessed = entry.get("unprocessed", False)
            else:
                poly = entry
                unprocessed = False

            if not poly or len(poly) < 2:
                continue

            # Color/style based on processing state
            color = QColor(255, 165, 0) if unprocessed else QColor(0, 200, 255)
            style = Qt.DashLine if unprocessed else Qt.SolidLine
            width = 5 if unprocessed else 4
            thick = 8 if (self.connection_mode and key == getattr(self, "_hover_midline_key", None)) else width
            qp.setPen(QPen(color, thick, style))

            for i in range(1, len(poly)):
                p1 = apply_offset(poly[i - 1])
                p2 = apply_offset(poly[i])
                qp.drawLine(
                    QPointF(int(p1[0] * scale + xoff), int(p1[1] * scale + yoff)),
                    QPointF(int(p2[0] * scale + xoff), int(p2[1] * scale + yoff))
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
                    QPointF(int(p1[0] * scale + xoff), int(p1[1] * scale + yoff)),
                    QPointF(int(p2[0] * scale + xoff), int(p2[1] * scale + yoff))
                )

        # --- points ---
        readonly_point_idxs = set()
        for (i1, i2) in self.readonly_connections:
            readonly_point_idxs.update([i1, i2])
        for (i1, i2) in self.readonly_midlines.keys():
            readonly_point_idxs.update([i1, i2])

        for i, (x, y) in enumerate(self.points):
            x, y = apply_offset((x, y))
            center = QPointF(int(x * scale + xoff), int(y * scale + yoff))
            is_readonly = i in readonly_point_idxs
            is_active = (i == self.hover_index) or (self.connection_mode and i == self.connecting_index)
            if is_readonly:
                # Saved/read-only endpoints: muted red, slightly brighter when hovered.
                brush = QColor(205, 110, 110) if is_active else QColor(175, 90, 90)
            else:
                brush = QColor(0, 200, 0) if is_active else QColor(200, 80, 80)
            qp.setBrush(brush)
            qp.setPen(Qt.NoPen)
            # keep circle size constant on screen regardless of zoom
            #r_screen = int(self.point_radius)
            r_screen = self._point_radius_screen()
            qp.drawEllipse(center, r_screen, r_screen)

        # --- editable midlines (simple cyan layer to keep behavior from older code) ---
        '''qp.setPen(QPen(QColor(0, 200, 200), 4))
        for key, poly in self.midlines.items():
            if isinstance(poly, dict):
                poly = poly.get("poly", [])
            if len(poly) < 2:
                continue
            thick = 8 if (self.connection_mode and key == getattr(self, "_hover_midline_key", None)) else 4
            qp.setPen(QPen(QColor(0, 200, 200), thick))
            for i in range(1, len(poly)):
                p1 = apply_offset(poly[i - 1])
                p2 = apply_offset(poly[i])
                qp.drawLine(
                    QPointF(int(p1[0] * scale + xoff), int(p1[1] * scale + yoff)),
                    QPointF(int(p2[0] * scale + xoff), int(p2[1] * scale + yoff))
                )'''

        # --- live polyline (manual midline in progress) ---
        if self.polyline_mode and len(self.polyline) >= 1:
            qp.setPen(QPen(QColor(0, 200, 200), 4))
            for i in range(1, len(self.polyline)):
                p1 = apply_offset(self.polyline[i - 1])
                p2 = apply_offset(self.polyline[i])
                qp.drawLine(
                    QPointF(int(p1[0] * scale + xoff), int(p1[1] * scale + yoff)),
                    QPointF(int(p2[0] * scale + xoff), int(p2[1] * scale + yoff))
                )

        self._draw_minimap(qp)

    def _toggle_minimap_corner(self):
        self._minimap_corner = (
            "bottom_right" if self._minimap_corner == "top_right" else "top_right"
        )
        self.update()

    def _draw_minimap(self, qp: QPainter):
        if self.image_pixmap is None:
            return

        cw = max(1, int(self.width()))
        ch = max(1, int(self.height()))
        iw = max(1, int(self.img_w))
        ih = max(1, int(self.img_h))
        if ch < 120 or cw < 180:
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
        y0 = pad if self._minimap_corner == "top_right" else (ch - mini_h - pad)
        x1 = x0 + mini_w
        y1 = y0 + mini_h

        mini_rect = QtCore.QRect(int(x0), int(y0), int(mini_w), int(mini_h))

        # Minimap should always use the original image background.
        # Build from raw image each paint so it cannot inherit overlay tint.
        mini_pix = self.image_pixmap
        try:
            if self.orig_image is not None:
                arr = self.orig_image
                if len(arr.shape) == 3 and arr.shape[2] == 3:
                    qmini = QImage(
                        arr.data,
                        arr.shape[1],
                        arr.shape[0],
                        arr.strides[0],
                        QImage.Format_RGB888,
                    ).copy()
                    mini_pix = QPixmap.fromImage(qmini)
        except Exception:
            pass

        qp.save()
        qp.setBrush(Qt.NoBrush)
        qp.drawPixmap(mini_rect, mini_pix)
        qp.setPen(QPen(QColor(255, 255, 255), 1))
        qp.drawRect(mini_rect)

        s = max(1e-8, float(self.scale))
        vx0 = (0.0 - float(self.pan_x)) / s
        vy0 = (0.0 - float(self.pan_y)) / s
        vx1 = (float(cw) - float(self.pan_x)) / s
        vy1 = (float(ch) - float(self.pan_y)) / s
        vx0 = float(max(0.0, min(float(iw), vx0)))
        vy0 = float(max(0.0, min(float(ih), vy0)))
        vx1 = float(max(0.0, min(float(iw), vx1)))
        vy1 = float(max(0.0, min(float(ih), vy1)))
        if vx1 < vx0:
            vx0, vx1 = vx1, vx0
        if vy1 < vy0:
            vy0, vy1 = vy1, vy0

        rx0 = x0 + int(round((vx0 / float(iw)) * mini_w))
        ry0 = y0 + int(round((vy0 / float(ih)) * mini_h))
        rx1 = x0 + int(round((vx1 / float(iw)) * mini_w))
        ry1 = y0 + int(round((vy1 / float(ih)) * mini_h))
        rx0 = int(max(x0, min(x1 - 1, rx0)))
        ry0 = int(max(y0, min(y1 - 1, ry0)))
        rx1 = int(max(rx0 + 1, min(x1, rx1)))
        ry1 = int(max(ry0 + 1, min(y1, ry1)))
        qp.setPen(QPen(QColor(0, 255, 255), 1))
        qp.setBrush(Qt.NoBrush)
        qp.drawRect(QtCore.QRect(int(rx0), int(ry0), int(rx1 - rx0), int(ry1 - ry0)))

        qp.setPen(QPen(QColor(255, 255, 255), 1))
        qp.drawText(int(x0 + 6), int(y1 - 6), "Minimap")
        qp.restore()

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
    def _zoom_at(self, pos: QtCore.QPointF, factor: float):
        """Zoom with the anchor fixed under `pos` (widget coords), using float pan."""
        self._hook_scroll_area()

        old = float(self.scale)
        new = float(max(self._min_scale(), min(10.0, old * float(factor))))
        if abs(new - old) < 1e-9:
            print("[ZOOM] Skipped (new ≈ old)")
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

    '''def _zoom_at(self, pos: QtCore.QPointF, factor: float):
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
            self._pan_last = event.localPos()
            self.setCursor(Qt.ClosedHandCursor)
            print(f"[PAN] start at {self._pan_last}")
            return

        # ----- EXISTING EDITOR LOGIC (unchanged below) -----
        if not img_rect.contains(event.pos()):
            return  # ignore clicks in gray margin

        p = self._to_image_coords(event.localPos())
        point_i = self._find_point_at(p)
        line_i = self._find_line_at(p)
        mid_key = self._midline_hit_test(event.localPos(), 10.0)

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
                    #dont delete old midlines while drawing (duh)
                    if point_i is not None:
                        mid_key = None
                    # Absolute priority enforcement
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
                    self._active_poly_regions = self._containing_boxes_for_point(float(sx), float(sy), tol=1.0)
                    if self.boxes and (not self._active_poly_regions):
                        QMessageBox.warning(
                            self,
                            "Out of bounds",
                            "Start endpoint is not inside any bounding box.\n"
                            "Cannot start manual polyline.",
                        )
                        self.polyline.clear()
                        self._is_drawing = False
                        self._start_idx = None
                        self._active_poly_regions = []
                        self.update()
                        return
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
                        if not self._poly_point_within_active_regions((px, py)):
                            self._abort_active_polyline_out_of_bounds()
                            return
                        self.polyline.append((float(px), float(py)))
                        print(f"[PRESS] Added polyline point: {self.polyline[-1]}")
                        self.update()
                    return

            elif event.button() == Qt.RightButton:
                if self._is_drawing:
                    self._erase_timer.stop()
                    QtCore.QTimer.singleShot(250, lambda: (
                        self._erase_timer.start(40)
                        if (QtWidgets.QApplication.mouseButtons() & Qt.RightButton and self._is_drawing)
                        else None
                    ))
                    self._undo_polyline_points(2)
                    self.update()
                return

        # ----- Normal connection / point modes -----
        if event.button() == Qt.LeftButton:
            if point_i is not None:
                        mid_key = None
                        line_i = None
            if (not self.polyline_mode) and self.connection_mode and (mid_key is not None):
                self.midlines.pop(mid_key, None)
                self._hover_midline_key = None
                self.update()
                return

            if not self.connection_mode:
                if point_i is None:
                    px, py = self._clamp_xy_to_image(p[0], p[1], edge_snap_px=1.0)
                    if self.boxes and (self._point_box_index(float(px), float(py)) is None):
                        QMessageBox.warning(
                            self,
                            "Out of bounds",
                            "Endpoint is outside all bounding boxes.\nPoint was not kept.",
                        )
                        return
                    self.points.append((px, py))
                    print(f"[PRESS] Added new point at ({px}, {py})")
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
                            if self._points_share_any_box(c[0], c[1], tol=1.0):
                                self.connections.append(c)
                            else:
                                QMessageBox.warning(
                                    self,
                                    "Out of bounds",
                                    "Connection endpoints do not share a bounding box.\n"
                                    "Connection was removed.",
                                )
                                if c in self.connections and c not in self.readonly_connections:
                                    self.connections.remove(c)
                        self.connecting_index = None
                    else:
                        self.connecting_index = None
                else:
                    self.connecting_index = None
            self.update()

    def mouseMoveEvent(self, event):
        if self._panning and self._pan_last is not None:
            dx = float(event.localPos().x() - self._pan_last.x())
            dy = float(event.localPos().y() - self._pan_last.y())
            self.pan_x += dx
            self.pan_y += dy
            self._pan_last = event.localPos()
            self._enforce_bounds()
            print(f"[PAN] move dx={dx:.1f}, dy={dy:.1f} -> pan=({self.pan_x:.1f},{self.pan_y:.1f})")
            self.update()
            return

        # --- existing hover / drawing logic ---
        p = self._to_image_coords(event.localPos())
        point_i = self._find_point_at(p)

        if self.polyline_mode and self._is_drawing and (event.buttons() & Qt.LeftButton):
            if point_i is not None and point_i != self._start_idx:
                print(f"[MOVE] Hovering endpoint {point_i}, attempting commit")
                self._commit_midline(point_i)
                return
            else:
                if not self._poly_point_within_active_regions(p):
                    self._abort_active_polyline_out_of_bounds()
                    return
                self._add_poly_point(p)
                self.update()
                return

        self.hover_index = self._find_point_at(p)
        if self.connection_mode and self.connecting_index is None and self.hover_index is None:
            self.hover_line_index = self._find_line_at(p)
        else:
            self.hover_line_index = None

        #self._hover_midline_key = self._midline_hit_test(event.localPos(), 10.0) if not self._is_drawing else None
        # Midline hover detection rules:
        #  - never while drawing
        #  - never when hovering a point
        if not self._is_drawing and self.hover_index is None:
            self._hover_midline_key = self._midline_hit_test(event.localPos(), 10.0)
        else:
            self._hover_midline_key = None
        self.update()

    def set_overlay_image(self, overlay_np):
        """
        Set an alternative background image (e.g., original+mask blend).

        overlay_np: HxWx3 uint8 RGB NumPy array with SAME size as orig_image.
        If None is passed, overlay is disabled.
        """
        if overlay_np is None:
            self.overlay_pixmap = None
            self.use_overlay = False
            self.update()
            return

        h, w, _ = overlay_np.shape
        qimg = QImage(
            overlay_np.data,
            w,
            h,
            overlay_np.strides[0],
            QImage.Format_RGB888
        )
        self.overlay_pixmap = QPixmap.fromImage(qimg)
        # do not force-enable here; let caller decide or call set_overlay_enabled
        self.update()

    def set_overlay_enabled(self, enabled: bool):
        """Turn overlay drawing on/off (if overlay_pixmap is set)."""
        self.use_overlay = bool(enabled)
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
