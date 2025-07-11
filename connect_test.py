import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QWidget, QVBoxLayout
)
from PyQt5.QtGui import QPainter, QPen, QColor, QCursor
from PyQt5.QtCore import Qt, QPoint

class CrackAnnotator(QWidget):
    def __init__(self):
        super().__init__()
        self.points = []  # List of (x, y)
        self.connections = []  # List of (from_idx, to_idx)
        self.point_radius = 7
        self.connection_mode = False
        self.connecting_index = None  # index of first point in connection
        self.hover_index = None       # for highlighting points on hover
        self.hover_line_index = None  # for highlighting lines on hover
        self.setMouseTracking(True)
        self.setMinimumSize(700, 500)

    def toggle_mode(self):
        self.connection_mode = not self.connection_mode
        self.connecting_index = None
        self.update()

    def mouseMoveEvent(self, event):
        pos = (event.x(), event.y())
        self.hover_index = self._find_point_at(pos)
        # Only highlight line if in connection mode, not creating a connection, and not hovering any point
        if self.connection_mode and self.connecting_index is None and self.hover_index is None:
            self.hover_line_index = self._find_line_at(pos)
        else:
            self.hover_line_index = None
        self.update()

    def leaveEvent(self, event):
        self.hover_index = None
        self.hover_line_index = None
        self.update()

    def mousePressEvent(self, event):
        pos = (event.x(), event.y())
        point_i = self._find_point_at(pos)
        line_i = self._find_line_at(pos)

        if not self.connection_mode:
            if point_i is None:
                # Add new point
                self.points.append(pos)
            else:
                # Remove point and its connections (any direction)
                self.connections = [
                    (i1, i2) for i1, i2 in self.connections
                    if i1 != point_i and i2 != point_i
                ]
                # Re-index connections
                self.points.pop(point_i)
                self.connections = [
                    (i1 - (i1 > point_i), i2 - (i2 > point_i)) for i1, i2 in self.connections
                ]
        else:
            # Only delete a line if NOT hovering any point
            if (line_i is not None) and (self.connecting_index is None) and (point_i is None):
                self.connections.pop(line_i)
            elif point_i is not None:
                # Only create connection, never delete in connection mode
                if self.connecting_index is None:
                    self.connecting_index = point_i
                elif self.connecting_index != point_i:
                    c = (self.connecting_index, point_i)
                    if c not in self.connections:
                        self.connections.append(c)
                    self.connecting_index = None
                else:
                    # Clicked same point while trying to connect -- just cancel, don't delete anything!
                    self.connecting_index = None
            else:
                self.connecting_index = None  # click background cancels
        self.update()

    def paintEvent(self, event):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)

        # Draw connections (directed: from->to, as arrow or line)
        for idx, (i1, i2) in enumerate(self.connections):
            p1 = QPoint(*self.points[i1])
            p2 = QPoint(*self.points[i2])
            # Only highlight this connection if hovered for deletion in connection mode and not hovering a point
            if (self.connection_mode and self.connecting_index is None and idx == self.hover_line_index and self.hover_index is None):
                pen = QPen(QColor(100, 220, 140), 6)  # Light green if hovered for deletion
            else:
                pen = QPen(QColor(80, 80, 220), 4)
            qp.setPen(pen)
            qp.drawLine(p1, p2)
            self._draw_arrowhead(qp, p1, p2)

        # Draw connecting line in green while waiting for second click (connection_mode)
        if self.connection_mode and self.connecting_index is not None and self.hover_index is not None and self.hover_index != self.connecting_index:
            qp.setPen(QPen(QColor(0, 200, 0), 3, Qt.DashLine))
            x1, y1 = self.points[self.connecting_index]
            x2, y2 = self.points[self.hover_index]
            qp.drawLine(QPoint(x1, y1), QPoint(x2, y2))

        # Draw points
        for i, (x, y) in enumerate(self.points):
            if self.hover_index == i:
                qp.setBrush(QColor(0, 200, 0))  # Green when hovered
            elif (self.connection_mode and i == self.connecting_index):
                qp.setBrush(QColor(0, 200, 0))
            else:
                qp.setBrush(QColor(200, 80, 80))
            qp.setPen(Qt.NoPen)
            qp.drawEllipse(QPoint(x, y), self.point_radius, self.point_radius)

    def _draw_arrowhead(self, qp, p1, p2):
        """Draw a small arrowhead at the end of the line (for direction)."""
        import math
        angle = math.atan2(p2.y() - p1.y(), p2.x() - p1.x())
        arrow_size = 10
        dx1 = arrow_size * math.cos(angle - math.pi / 8)
        dy1 = arrow_size * math.sin(angle - math.pi / 8)
        left = QPoint(
            int(p2.x() - dx1),
            int(p2.y() - dy1)
        )
        dx2 = arrow_size * math.cos(angle + math.pi / 8)
        dy2 = arrow_size * math.sin(angle + math.pi / 8)
        right = QPoint(
            int(p2.x() - dx2),
            int(p2.y() - dy2)
        )
        qp.setPen(Qt.NoPen)
        qp.setBrush(QColor(80, 80, 220))
        points = [p2, left, right]
        qp.drawPolygon(*points)

    def _find_point_at(self, pos):
        """Return index of point at pos, else None"""
        for i, (x, y) in enumerate(self.points):
            if (x - pos[0]) ** 2 + (y - pos[1]) ** 2 <= self.point_radius ** 2:
                return i
        return None

    def _find_line_at(self, pos):
        """Return index of directed line if pos is close to it, else None"""
        threshold = 7
        for idx, (i1, i2) in enumerate(self.connections):
            x1, y1 = self.points[i1]
            x2, y2 = self.points[i2]
            if self._dist_point_to_segment(pos, (x1, y1), (x2, y2)) < threshold:
                return idx
        return None

    def _dist_point_to_segment(self, p, a, b):
        # Compute minimum distance from point p to segment ab
        import numpy as np
        p = np.array(p)
        a = np.array(a)
        b = np.array(b)
        if np.all(a == b):
            return np.linalg.norm(p - a)
        t = max(0, min(1, np.dot(p - a, b - a) / np.dot(b - a, b - a)))
        proj = a + t * (b - a)
        return np.linalg.norm(p - proj)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crack Annotation Demo")
        self.annotator = CrackAnnotator()
        self.mode_button = QPushButton("Switch to Connection Mode")
        self.mode_button.setCheckable(True)
        self.mode_button.clicked.connect(self.switch_mode)
        layout = QVBoxLayout()
        layout.addWidget(self.mode_button)
        layout.addWidget(self.annotator)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self._update_button_text()

    def switch_mode(self):
        self.annotator.toggle_mode()
        self._update_button_text()

    def _update_button_text(self):
        if self.annotator.connection_mode:
            self.mode_button.setText("Switch to Point Mode")
            self.mode_button.setStyleSheet("background: #97e297;")
        else:
            self.mode_button.setText("Switch to Connection Mode")
            self.mode_button.setStyleSheet("background: #e2c297;")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
