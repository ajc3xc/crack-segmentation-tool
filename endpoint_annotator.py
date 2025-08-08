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
    Points & connections + manual midlines.

    - Point mode: LMB on canvas adds; LMB on point deletes (and reindexes).
    - Connection mode: LMB two points = connect; LMB on a connection line = delete.
    - Manual mode: click an EXISTING point to start → draw by drag/click → click a DIFFERENT EXISTING point to finish.
      * Ends snap to endpoints
      * LMB on a hovered midline (in manual OR connection mode) deletes it
      * Backspace/Z undo last vertex while drawing
      * Esc cancels current midline
    - Prevents duplicate pair midlines; warns on abandon.

    Stored midlines: self.midlines[(i1,i2)] = [(x,y), ...] with i1<i2.
    """
    def __init__(self, image=None, boxes=None, initial_points=None, initial_connections=None, initial_midlines=None):
        super().__init__()
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
        # normalize connections to sorted tuples with unique set
        conns = []
        if initial_connections:
            for a,b in initial_connections:
                a,b = int(a), int(b)
                if a==b: continue
                pair = (a,b) if a<b else (b,a)
                if pair not in conns: conns.append(pair)
        self.connections = conns

        # point radius scales with image size
        min_dim = min(self.img_w, self.img_h)
        self.point_radius = max(3, min(20, int(0.005 * (min_dim if min_dim>0 else 500))))

        self.boxes = boxes if boxes is not None else []
        self.scale = 1.0
        self.setMouseTracking(True)
        self.update_canvas_size()

        # Classic connection mode
        self.connection_mode = False
        self.connecting_index = None
        self.hover_index = None
        self.hover_line_index = None

        # Manual midline mode
        self.polyline_mode = False
        self.polyline = []
        self._is_drawing = False
        self._start_idx = None

        # Saved midlines
        self.midlines = {}
        if isinstance(initial_midlines, dict):
            for k, poly in initial_midlines.items():
                try:
                    a, b = k.split("_"); i1, i2 = int(a), int(b)
                    if i1==i2: continue
                    key = (i1,i2) if i1<i2 else (i2,i1)
                    self.midlines[key] = [(float(x), float(y)) for x,y in poly]
                except: pass

        # Hovered midline key for feedback/deletion
        self._hover_midline_key = None

        self.setFocusPolicy(Qt.StrongFocus)

    # -------------- mode toggles --------------
    def toggle_mode(self):
        self.connection_mode = not self.connection_mode
        self.connecting_index = None
        self.update()

    def set_mode_polyline(self, enabled: bool, confirm_cb=None):
        """Enable/disable manual mode. If disabling with active draw, use confirm_cb() -> bool to confirm discard."""
        if not enabled and self._is_drawing:
            ok_to_discard = True
            if confirm_cb is not None:
                ok_to_discard = confirm_cb()
            if not ok_to_discard:
                return False
            self.polyline = []
            self._is_drawing = False
            self._start_idx = None
        self.polyline_mode = enabled
        if enabled:
            self.connecting_index = None
        self.update()
        return True

    # -------------- util --------------
    def all_pairs_saturated(self):
        """True if every unordered pair of points is represented either as a connection OR a midline."""
        n = len(self.points)
        if n < 2: return True
        pairs = set()
        pairs.update((min(a,b), max(a,b)) for (a,b) in self.connections)
        pairs.update(self.midlines.keys())
        need = (n*(n-1))//2
        return len(pairs) >= need

    def _sorted(self, i, j): return (i,j) if i<j else (j,i)
    def _add_poly_point(self, p): self.polyline.append((float(p[0]), float(p[1])))
    def _pop_poly_point(self): 
        if self.polyline: self.polyline.pop()

    # -------------- canvas plumbing --------------
    def update_canvas_size(self):
        w = int(self.img_w * self.scale); h = int(self.img_h * self.scale)
        self.setMinimumSize(w, h); self.resize(w, h); self.update()

    def wheelEvent(self, event):
        f = 1.2 if event.angleDelta().y() > 0 else 1/1.2
        self.scale = max(0.1, min(10.0, self.scale*f))
        self.update_canvas_size()

    # -------------- mouse/key --------------
    def mousePressEvent(self, event):
        p = self._to_image_coords(event.pos())
        point_i = self._find_point_at(p)
        line_i = self._find_line_at(p)  # connection hit test
        mid_key = self._midline_hit_test(event.pos(), 10.0)  # midline hit test (screen px)

        # manual mode first
        if self.polyline_mode:
            if event.button() == Qt.LeftButton:
                # delete hovered connection/midline by LMB (even in manual mode)
                if (not self._is_drawing) and (mid_key is not None):
                    self.midlines.pop(mid_key, None); self._hover_midline_key=None; self.update(); return
                if (not self._is_drawing) and (line_i is not None):
                    # delete connection by LMB in manual mode too
                    self.connections.pop(line_i); self.update(); return

                # start?
                if not self._is_drawing:
                    if point_i is None: return
                    if len(self.points) < 2: return
                    self._start_idx = point_i
                    sx, sy = self.points[self._start_idx]
                    self.polyline = [(float(sx), float(sy))]
                    self._is_drawing = True
                    self.update(); return
                else:
                    # trying to finish?
                    if point_i is not None and point_i != self._start_idx:
                        key = self._sorted(self._start_idx, point_i)
                        # block duplicate pair
                        if (key in self.midlines) or (key in [(min(a,b),max(a,b)) for (a,b) in self.connections]):
                            # show error via callback (the dialog will own the messagebox)
                            self._last_pair_error = key
                            self.update(); return
                        if len(self.polyline) >= 2:
                            # snap first/last vertices
                            sx, sy = self.points[key[0]]; ex, ey = self.points[key[1]]
                            poly = [(float(sx), float(sy))] + self.polyline[1:-1] + [(float(ex), float(ey))]
                            self.midlines[key] = poly
                            # reset and allow more midlines
                            self.polyline = []; self._is_drawing = False; self._start_idx = None
                            self.update(); return
                        # else keep drawing if too short
                    # else add vertex
                    self._add_poly_point(p); self.update(); return

            elif event.button() == Qt.RightButton:
                # undo/cancel
                if self._is_drawing:
                    if len(self.polyline) > 1: self._pop_poly_point()
                    else:
                        self.polyline=[]; self._is_drawing=False; self._start_idx=None
                    self.update()
                return

        # connection / point modes
        if event.button() == Qt.LeftButton:
            # delete hovered midline by LMB (allowed in connection mode too)
            if (not self.polyline_mode) and (mid_key is not None):
                self.midlines.pop(mid_key, None); self._hover_midline_key=None; self.update(); return

            if not self.connection_mode:
                if point_i is None:
                    self.points.append(p)
                else:
                    # delete point and reindex connections + midlines
                    self._delete_point_reindex(point_i)
            else:
                # connection mode
                if (line_i is not None) and (self.connecting_index is None) and (point_i is None):
                    # delete connection
                    self.connections.pop(line_i)
                elif point_i is not None:
                    if self.connecting_index is None:
                        self.connecting_index = point_i
                    elif self.connecting_index != point_i:
                        c = self._sorted(self.connecting_index, point_i)
                        # block duplicate
                        if c not in self.connections:
                            self.connections.append(c)
                        self.connecting_index = None
                    else:
                        self.connecting_index = None
                else:
                    self.connecting_index = None
            self.update(); return

    def mouseMoveEvent(self, event):
        p = self._to_image_coords(event.pos())
        # mid-draw drag adds vertices
        if self.polyline_mode and self._is_drawing and (event.buttons() & Qt.LeftButton):
            self._add_poly_point(p); self.update(); return

        self.hover_index = self._find_point_at(p)
        if self.connection_mode and self.connecting_index is None and self.hover_index is None:
            self.hover_line_index = self._find_line_at(p)
        else:
            self.hover_line_index = None

        # hover midline (for enlarge)
        self._hover_midline_key = self._midline_hit_test(event.pos(), 10.0) if (not self._is_drawing) else None
        self.update()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Backspace, Qt.Key_Z):
            if self.polyline_mode and self._is_drawing and self.polyline:
                self._pop_poly_point(); self.update(); return
        if event.key() == Qt.Key_Escape:
            if self.polyline_mode and self._is_drawing:
                self.polyline=[]; self._is_drawing=False; self._start_idx=None; self.update(); return
        super().keyPressEvent(event)

    # -------------- paint --------------
    def paintEvent(self, event):
        qp = QPainter(self); qp.setRenderHint(QPainter.Antialiasing)
        # fit image
        if self.image_pixmap:
            img_w, img_h = self.img_w, self.img_h
            win_w, win_h = self.width(), self.height()
            scale = min(win_w/img_w, win_h/img_h)
            xoff = int((win_w - img_w*scale)/2)
            yoff = int((win_h - img_h*scale)/2)
            self._last_draw_scale, self._last_draw_xoff, self._last_draw_yoff = scale, xoff, yoff
            qp.drawPixmap(xoff, yoff, int(img_w*scale), int(img_h*scale), self.image_pixmap)
        else:
            self._last_draw_scale, self._last_draw_xoff, self._last_draw_yoff = 1.0, 0, 0

        scale = self._last_draw_scale; xoff = self._last_draw_xoff; yoff = self._last_draw_yoff

        # boxes (blue)
        qp.setPen(QPen(QColor(0,128,255),3))
        for xmin,ymin,xmax,ymax in self.boxes:
            qp.drawRect(int(xmin*scale+xoff), int(ymin*scale+yoff),
                        int((xmax-xmin)*scale), int((ymax-ymin)*scale))

        # connections (black)
        for idx, (i1,i2) in enumerate(self.connections):
            x1,y1 = self.points[i1]; x2,y2 = self.points[i2]
            p1 = QPoint(int(x1*scale+xoff), int(y1*scale+yoff))
            p2 = QPoint(int(x2*scale+xoff), int(y2*scale+yoff))
            thick = 6 if (self.connection_mode and self.connecting_index is None and idx==self.hover_line_index and self.hover_index is None) else 4
            qp.setPen(QPen(QColor(0,0,0), thick))
            qp.drawLine(p1, p2)

        # points
        for i,(x,y) in enumerate(self.points):
            center = QPoint(int(x*scale+xoff), int(y*scale+yoff))
            brush = (QColor(0,200,0) if i==self.hover_index or (self.connection_mode and i==self.connecting_index) else QColor(200,80,80))
            qp.setBrush(brush); qp.setPen(Qt.NoPen)
            qp.drawEllipse(center, int(self.point_radius*scale), int(self.point_radius*scale))

        # midlines (cyan)
        for key, poly in self.midlines.items():
            if len(poly)<2: continue
            thick = 8 if (key == self._hover_midline_key) else 4
            qp.setPen(QPen(QColor(0,200,200), thick))
            for i in range(1, len(poly)):
                x1,y1 = poly[i-1]; x2,y2 = poly[i]
                qp.drawLine(QPoint(int(x1*scale+xoff), int(y1*scale+yoff)),
                            QPoint(int(x2*scale+xoff), int(y2*scale+yoff)))

        # current polyline
        if self.polyline_mode and len(self.polyline)>=1:
            qp.setPen(QPen(QColor(0,200,200), 4))
            for i in range(1, len(self.polyline)):
                x1,y1 = self.polyline[i-1]; x2,y2 = self.polyline[i]
                qp.drawLine(QPoint(int(x1*scale+xoff), int(y1*scale+yoff)),
                            QPoint(int(x2*scale+xoff), int(y2*scale+yoff)))

    # -------------- helpers --------------
    def _to_image_coords(self, pos):
        return ((pos.x()-getattr(self,"_last_draw_xoff",0))/getattr(self,"_last_draw_scale",1.0),
                (pos.y()-getattr(self,"_last_draw_yoff",0))/getattr(self,"_last_draw_scale",1.0))

    def _find_point_at(self, pos):
        r2 = (self.point_radius/self.scale)**2
        for i,(x,y) in enumerate(self.points):
            if (x-pos[0])**2 + (y-pos[1])**2 <= r2: return i
        return None

    def _dist_point_to_segment(self, p, a, b):
        import numpy as np
        p,a,b = np.array(p), np.array(a), np.array(b)
        if np.all(a==b): return float(np.linalg.norm(p-a))
        t = max(0,min(1,np.dot(p-a,b-a)/np.dot(b-a,b-a)))
        proj = a + t*(b-a); return float(np.linalg.norm(p-proj))

    def _find_line_at(self, pos):
        thr = 7/self.scale
        for idx,(i1,i2) in enumerate(self.connections):
            x1,y1 = self.points[i1]; x2,y2 = self.points[i2]
            if self._dist_point_to_segment(pos,(x1,y1),(x2,y2)) < thr: return idx
        return None

    def _midline_hit_test(self, screen_pos, px_thresh=9.0):
        if not self.midlines: return None
        s = self._last_draw_scale; xo = self._last_draw_xoff; yo = self._last_draw_yoff
        sx,sy = screen_pos.x(), screen_pos.y()

        import math
        def dseg(px,py, ax,ay, bx,by):
            vx,vy = bx-ax, by-ay; wx,wy = px-ax, py-ay
            vv = vx*vx+vy*vy
            if vv <= 1e-12: return math.hypot(px-ax, py-ay)
            t = max(0.0, min(1.0, (wx*vx+wy*vy)/vv))
            qx,qy = ax+t*vx, ay+t*vy
            return math.hypot(px-qx, py-qy)

        best = 1e9; hit=None
        for key, poly in self.midlines.items():
            for i in range(1,len(poly)):
                x1,y1 = poly[i-1]; x2,y2 = poly[i]
                ax,ay = x1*s+xo, y1*s+yo; bx,by = x2*s+xo, y2*s+yo
                d = dseg(sx,sy, ax,ay, bx,by)
                if d<best: best=d; hit=key
        return hit if best<=px_thresh else None

    def _delete_point_reindex(self, idx):
        # remove connections touching idx
        self.connections = [(i1 - (i1>idx), i2 - (i2>idx)) for (i1,i2) in self.connections if (i1!=idx and i2!=idx)]
        # remove midlines touching idx
        new_mid={}
        for (i1,i2), poly in self.midlines.items():
            if i1==idx or i2==idx: continue
            ni1 = i1 - (i1>idx); ni2 = i2 - (i2>idx)
            new_mid[(min(ni1,ni2), max(ni1,ni2))]=poly
        self.midlines = new_mid
        # remove point
        self.points.pop(idx)