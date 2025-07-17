#!/usr/bin/env python3
import sys, os, json
import numpy as np
import cv2
import cracktools as ct
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QWidget,
    QVBoxLayout, QHBoxLayout, QFileDialog, QSpinBox, QLabel
)
from PyQt5.QtGui import QPainter, QPen, QColor, QCursor, QImage, QPixmap
from PyQt5.QtCore import Qt, QPoint

class CrackAnnotator(QWidget):
    def __init__(self):
        super().__init__()
        # point/connection UI state
        self.points = []         # list of (x,y)
        self.connections = []    # list of (i1,i2)
        self.point_radius = 7
        self.connection_mode = False
        self.connecting_index = None
        self.hover_index = None
        self.hover_line_index = None

        # bounding box
        self.bb_pts = None       # np.array([[x0,y0],[x1,y1]])

        # image & annotation store
        self.original_image = None
        self.display_image = None
        self.ann_name = None
        self.annotation = {}

        self.setMouseTracking(True)
        self.setMinimumSize(800, 600)

    def load_image(self, fname):
        img = cv2.imread(fname)
        if img is None:
            raise RuntimeError(f"Could not open {fname}")
        self.original_image = img[:, :, ::-1]  # BGR→RGB
        self.display_image = self.original_image.copy()
        self.ann_name = os.path.splitext(fname)[0] + '.json'
        # reset state
        self.points = []
        self.connections = []
        self.bb_pts = None
        self.annotation = {
            "image_name": fname,
            "annotations": {
                "box": {},
                "points": [],
                "connections": [],
                "segmentation": []
            }
        }
        self.update()

    def toggle_mode(self):
        self.connection_mode = not self.connection_mode
        self.connecting_index = None
        self.update()

    def mouseMoveEvent(self, evt):
        pos = (evt.x(), evt.y())
        self.hover_index = self._find_point(pos)
        if self.connection_mode and self.connecting_index is None and self.hover_index is None:
            self.hover_line_index = self._find_line(pos)
        else:
            self.hover_line_index = None
        self.update()

    def leaveEvent(self, evt):
        self.hover_index = None
        self.hover_line_index = None
        self.update()

    def mousePressEvent(self, evt):
        pos = (evt.x(), evt.y())
        pi = self._find_point(pos)
        li = self._find_line(pos)

        if not self.connection_mode:
            # point mode: add/remove points
            if pi is None:
                self.points.append(pos)
            else:
                # remove point + associated connections
                self.points.pop(pi)
                self.connections = [
                    (i1-(i1>pi), i2-(i2>pi))
                    for i1,i2 in self.connections
                    if i1!=pi and i2!=pi
                ]
        else:
            # connection mode: add/remove connections
            if li is not None and self.connecting_index is None and pi is None:
                self.connections.pop(li)
            elif pi is not None:
                if self.connecting_index is None:
                    self.connecting_index = pi
                elif self.connecting_index != pi:
                    c = (self.connecting_index, pi)
                    if c not in self.connections:
                        self.connections.append(c)
                    self.connecting_index = None
                else:
                    self.connecting_index = None
            else:
                self.connecting_index = None
        self.update()

    def paintEvent(self, evt):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)

        # draw bounding box
        if self.bb_pts is not None:
            (x0,y0),(x1,y1) = self.bb_pts
            pen = QPen(QColor(0,200,0), 4)
            qp.setPen(pen)
            qp.drawRect(x0, y0, x1-x0, y1-y0)

        # draw connections
        for idx,(i1,i2) in enumerate(self.connections):
            p1 = QPoint(*self.points[i1])
            p2 = QPoint(*self.points[i2])
            if (self.connection_mode and self.connecting_index is None 
                and idx==self.hover_line_index and self.hover_index is None):
                pen = QPen(QColor(100,220,140), 6)
            else:
                pen = QPen(QColor(80,80,220), 4)
            qp.setPen(pen)
            qp.drawLine(p1,p2)
            self._draw_arrowhead(qp,p1,p2)

        # preview connection
        if (self.connection_mode and self.connecting_index is not None 
            and self.hover_index is not None 
            and self.hover_index!=self.connecting_index):
            qp.setPen(QPen(QColor(0,200,0),3,Qt.DashLine))
            x1,y1 = self.points[self.connecting_index]
            x2,y2 = self.points[self.hover_index]
            qp.drawLine(QPoint(x1,y1),QPoint(x2,y2))

        # draw points
        for i,(x,y) in enumerate(self.points):
            if i==self.hover_index or (self.connection_mode and i==self.connecting_index):
                qp.setBrush(QColor(0,200,0))
            else:
                qp.setBrush(QColor(200,80,80))
            qp.setPen(Qt.NoPen)
            qp.drawEllipse(QPoint(x,y), self.point_radius, self.point_radius)

    def _draw_arrowhead(self, qp, p1, p2):
        import math
        angle = math.atan2(p2.y()-p1.y(), p2.x()-p1.x())
        size = 10
        dx1 = size*math.cos(angle-math.pi/8)
        dy1 = size*math.sin(angle-math.pi/8)
        left = QPoint(int(p2.x()-dx1), int(p2.y()-dy1))
        dx2 = size*math.cos(angle+math.pi/8)
        dy2 = size*math.sin(angle+math.pi/8)
        right = QPoint(int(p2.x()-dx2), int(p2.y()-dy2))
        qp.setPen(Qt.NoPen)
        qp.setBrush(QColor(80,80,220))
        qp.drawPolygon(p2, left, right)

    def _find_point(self, pos):
        for i,(x,y) in enumerate(self.points):
            if (x-pos[0])**2 + (y-pos[1])**2 <= self.point_radius**2:
                return i
        return None

    def _find_line(self, pos):
        import numpy as np
        thresh = 7
        for idx,(i1,i2) in enumerate(self.connections):
            a = np.array(self.points[i1])
            b = np.array(self.points[i2])
            p = np.array(pos)
            if np.all(a==b):
                d = np.linalg.norm(p-a)
            else:
                t = np.clip(np.dot(p-a, b-a)/np.dot(b-a,b-a), 0,1)
                proj = a + t*(b-a)
                d = np.linalg.norm(p-proj)
            if d<thresh:
                return idx
        return None

    def export_annotation(self):
        # compose annotation dict
        ann = self.annotation
        # box
        if self.bb_pts is not None:
            ann["annotations"]["box"] = {0:{"bounding_box": self.bb_pts.tolist()}}
        # points
        ann["annotations"]["points"] = [list(p) for p in self.points]
        # connections
        ann["annotations"]["connections"] = [
            {"from":i1,"to":i2} for i1,i2 in self.connections
        ]
        with open(self.ann_name,'w') as f:
            json.dump(ann, f, indent=2)
        print(f"Saved JSON → {self.ann_name}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Connect Full Test")

        # widgets
        self.annot = CrackAnnotator()
        self.load_btn    = QPushButton("Load Image")
        self.box_btn     = QPushButton("Draw Box")
        self.clear_box   = QPushButton("Clear Box")
        self.mode_btn    = QPushButton("Switch to Conn Mode")
        self.process_btn = QPushButton("Process All")
        self.save_btn    = QPushButton("Save JSON")

        # parameters
        self.y_sb    = QSpinBox(); self.y_sb.setRange(0,100);  self.y_sb.setValue(10)
        self.x_sb    = QSpinBox(); self.x_sb.setRange(0,100);  self.x_sb.setValue(10)
        self.ds_sb   = QSpinBox(); self.ds_sb.setRange(1,8);    self.ds_sb.setValue(2)
        self.lam_sb  = QSpinBox(); self.lam_sb.setRange(1,100); self.lam_sb.setValue(1)
        self.p_sb    = QSpinBox(); self.p_sb.setRange(1,10);   self.p_sb.setValue(1)

        # layout
        ctrl = QHBoxLayout()
        for w in (self.load_btn, self.box_btn, self.clear_box,
                  self.mode_btn, self.process_btn, self.save_btn):
            ctrl.addWidget(w)
        ctrl.addStretch()
        for lbl, sb in (("Y-mgn",self.y_sb),("X-mgn",self.x_sb),
                        ("Down",self.ds_sb),("λ",self.lam_sb),("p",self.p_sb)):
            ctrl.addWidget(QLabel(lbl)); ctrl.addWidget(sb)

        vbox = QVBoxLayout()
        vbox.addLayout(ctrl)
        vbox.addWidget(self.annot)

        container = QWidget()
        container.setLayout(vbox)
        self.setCentralWidget(container)

        # signals
        self.load_btn.clicked.connect(self._on_load)
        self.box_btn.clicked.connect(self._on_box)
        self.clear_box.clicked.connect(self._on_clear_box)
        self.mode_btn.clicked.connect(self._on_toggle_mode)
        self.process_btn.clicked.connect(self._on_process)
        self.save_btn.clicked.connect(self.annot.export_annotation)

    def _on_load(self):
        fn,_ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg)")
        if fn:
            self.annot.load_image(fn)

    def _on_box(self):
        size = max(self.annot.width(), self.annot.height())//4
        pts,_ = ct.tools.Draw().bounding_box(self.annot.original_image, size)
        self.annot.bb_pts = np.array(pts, dtype=int)
        self.annot.update()

    def _on_clear_box(self):
        self.annot.bb_pts = None
        self.annot.update()

    def _on_toggle_mode(self):
        self.annot.toggle_mode()
        if self.annot.connection_mode:
            self.mode_btn.setText("Switch to Point Mode")
        else:
            self.mode_btn.setText("Switch to Conn Mode")

    def _on_process(self):
        A = self.annot
        if A.original_image is None:
            print("Load an image first.")
            return

        res = []
        y_m, x_m = self.y_sb.value(), self.x_sb.value()
        ds, lam, p = self.ds_sb.value(), self.lam_sb.value(), self.p_sb.value()

        for i1,i2 in A.connections:
            pt1, pt2 = A.points[i1], A.points[i2]
            # crop
            crop, pts_crop = ct.tools.image_crop(
                A.original_image, pt1, pt2, [pt1,pt2], y_m, x_m
            )
            # downsample
            img_ds = ct.tools.block_reduce(crop, (ds,ds,1), np.min)
            pts_ds = [tuple(np.array(p)/ds) for p in pts_crop]

            # orientation score → vesselness → cost
            osGF = ct.os.OrientationScoreTransform(
                img_ds[:,:,0]/255.0, size=32, nOrientations=8,
                design="N", inflectionPoint=3, mnOrder=3,
                splineOrder=3, overlapFactor=1, dcStdDev=1.0,
                directional=False
            )
            msv = ct.os.MultiScaleVesselness(
                osGF.real, ksi=1, scales=[1,2,3], method="LIF", sigmas_ext=1
            )
            cf = ct.os.CostFunction(ct.os.MultiScaleVesselnessFilter(msv), lambdaa=lam, p=p)

            # fast marching & full-size track
            track_ds = ct.tracking.fast_marching(cf, pts_ds[0], pts_ds[1], g11=1,g22=1,g33=1)
            track = ct.tools.track_crop_to_full(track_ds, pt1, pt2, y_m, x_m)

            # edge masks & tracking
            em1, em2 = ct.segmentation.edge_masks(
                A.original_image[:,:,0], np.array(track).T, window_half_size=3
            )
            te1, te2 = ct.segmentation.edges_tracking(
                crop[:,:,0], pts_crop, em1[:,:,None], em2[:,:,None],
                mu=0.1, l=1, p=1
            )
            e1 = ct.tools.track_crop_to_full(te1, pt1, pt2, y_m, x_m)
            e2 = ct.tools.track_crop_to_full(te2, pt1, pt2, y_m, x_m)

            # mask
            yx1 = np.array(e1)[::-1]; yx2 = np.array(e2)[::-1]
            m1 = ct.segmentation.create_mask(A.original_image, yx1[0], yx1[1])
            m2 = ct.segmentation.create_mask(A.original_image, yx2[0], yx2[1])
            mask = np.clip(m1+m2,0,1).tolist()

            res.append({
                "connection":[i1,i2],
                "centerline":np.array(track).T.tolist(),
                "edges":[e1,e2],
                "mask":mask
            })

        A.annotation["annotations"]["segmentation"] = res
        with open(A.ann_name,'w') as f:
            json.dump(A.annotation, f, indent=2)
        print(f"Processed {len(res)} connections → {A.ann_name}")

if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
