import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

##### image functions ###########
def image_load(name):
    return cv2.imread(name)[:,:,::-1]

def show_image(image,frame_size = 0, pts = None, limits_y=None, limits_x=None,color='green', marker='+',markersize=12,cmap = 'gray'):
    if frame_size!=0:
        fig = plt.figure(figsize = (frame_size, frame_size))
    if pts != None:
        for p in pts:
            plt.plot(p[0],p[1],color=color, marker=marker,markersize=markersize)
        if limits_x!=None:
            plt.xlim([np.min(np.array(pts)[:,0]) - limits_x, np.max(np.array(pts)[:,0]) + limits_x])
        if limits_y!=None:
            plt.ylim([np.max(np.array(pts)[:,1]) + limits_y, np.min(np.array(pts)[:,1]) - limits_y])
            
    plt.imshow(image,cmap = cmap)
    
def draw_track(image,track,frame_size = 0,limits_x = None,limits_y = None, track_color = 'r',track_width = 2):
    if frame_size!=0:
        fig = plt.figure(figsize = (frame_size, frame_size))
    plt.imshow(image)
    plt.plot(track[0],track[1], color=track_color, linewidth=track_width)
    if limits_x!= None:
        plt.xlim([np.min(track[0]) - limits_x, np.max(track[0]) + limits_x])
    if limits_y!= None:
        plt.ylim([np.max(track[1]) + limits_y, np.min(track[1]) - limits_y])
    plt.show()
    
def scale_image(image):
    image_out = image - abs(np.min(image))
    s = 1/image_out.max()
    image_out = image_out*s
    return image_out
###################################

def redrow_lines(img, contours_x, contours_y, t, scale, color=(0, 255, 0)):
    """
    Redraw contour lines on an image.

    Parameters
    ----------
    img : np.ndarray
        The image to draw on
    contours_x, contours_y : list[list[float]]
        List of contour x and y coordinates (each sublist is one stroke)
    t : int
        Thickness factor
    scale : float
        Scaling factor for thickness
    color : tuple(int,int,int), optional
        BGR color for lines (default green)
    """
    img2 = img.copy()
    for cx, cy in zip(contours_x, contours_y):  # each stroke
        for i in range(len(cx) - 1):
            x1, y1 = int2(np.round(cx[i])),   int2(np.round(cy[i]))
            x2, y2 = int2(np.round(cx[i+1])), int2(np.round(cy[i+1]))
            img2 = cv2.line(
                img2,
                (x1, y1),
                (x2, y2),
                color=color,
                thickness=int2(np.ceil(t * scale))
            )
    return img2

def redrow_points(img,pts,t,scale):
    img2 = img.copy()
    for p in pts:
        x1v = int2(p[0]-10)
        x2v = int2(p[0]+10)
        y1v = int2(p[1])
        y2v = int2(p[1])
        
        x1h = int2(p[0])
        x2h = int2(p[0])
        y1h = int2(p[1]-10)
        y2h = int2(p[1]+10)
        
#         img2 = cv2.line(img2,(x1,y1),(x2,y2),color=(0,255,0),thickness=int2(np.ceil(t*scale)))
        img2 = cv2.line(img2,(x1h,y1h),(x2h,y2h),color=(0,255,0),thickness=int2(np.ceil(t*scale)))
        img2 = cv2.line(img2,(x1v,y1v),(x2v,y2v),color=(0,255,0),thickness=int2(np.ceil(t*scale)))
    return (img2)

def redrow_coordinates(img,x,y,t,scale):
    img2 = img.copy()
    # Red crosshair for current cursor position.
    img2 = cv2.line(img2,(x,img2.shape[0]),(x,0),color=(0,0,255),thickness=int2(np.ceil(t*scale)))
    img2 = cv2.line(img2,(img2.shape[1],y),(0,y),color=(0,0,255),thickness=int2(np.ceil(t*scale)))
    return (img2)

def redrow_bb(img,x,y,t,scale,pts,active,c,initial_pairs=0):
    img2 = img.copy()
    img2 = cv2.line(img2,(x,img2.shape[0]),(x,0),color=(0,0,255),thickness=int2(np.ceil(t*scale)))
    img2 = cv2.line(img2,(img2.shape[1],y),(0,y),color=(0,0,255),thickness=int2(np.ceil(t*scale)))
    if len(pts)>1:
        for i in range(0,len(pts)-1,2):
            x0 = int(pts[i][0])
            y0 = int(pts[i][1])
            x1 = int(pts[i+1][0])
            y1 = int(pts[i+1][1])
            pair_idx = int(i/2)
            # Previously pending (unsaved) boxes: dark green.
            # Newly drawn boxes in current session: lime green.
            color = (0, 140, 0) if pair_idx < int(initial_pairs) else (0,255,0)
            if len(c)>int(i/2):
                if c[int(i/2)] == 1:
                    color = (0,0,255)
                elif c[int(i/2)] == 2:
                    color = (0,255,0)
            img2 = cv2.line(img2,(x0,y0),(x1,y0),color=color,thickness=int2(np.ceil(t*scale)))
            img2 = cv2.line(img2,(x0,y0),(x0,y1),color=color,thickness=int2(np.ceil(t*scale)))
            img2 = cv2.line(img2,(x0,y1),(x1,y1),color=color,thickness=int2(np.ceil(t*scale)))
            img2 = cv2.line(img2,(x1,y0),(x1,y1),color=color,thickness=int2(np.ceil(t*scale)))
        
    if active == True and x!=None: 
        x1 = int(pts[-1][0])
        y1 = int(pts[-1][1])

        # Active box currently being drawn: lime green.
        active_color = (0,255,0)
        img2 = cv2.line(img2,(x,y),(x1,y),color=active_color,thickness=int2(np.ceil(t*scale)))
        img2 = cv2.line(img2,(x,y),(x,y1),color=active_color,thickness=int2(np.ceil(t*scale)))
        img2 = cv2.line(img2,(x,y1),(x1,y1),color=active_color,thickness=int2(np.ceil(t*scale)))
        img2 = cv2.line(img2,(x1,y),(x1,y1),color=active_color,thickness=int2(np.ceil(t*scale)))
    return (img2)

def drow_mask_lines(img,contours_x,contours_y,color,t=1):
#     flat_x = [item for sublist in contours_x for item in sublist]
#     flat_y = [item for sublist in contours_y for item in sublist]
    img2 = img.copy()
    for i in range(len(contours_x)-1):
        x1 = int2(np.round(contours_x[i]))
        x2 = int2(np.round(contours_x[i+1]))
        y1 = int2(np.round(contours_y[i]))
        y2 = int2(np.round(contours_y[i+1]))
        img2 = cv2.line(img2,(x1,y1),(x2,y2),color=color,thickness=int2(np.ceil(t)))
        
    x1 = int2(np.round(contours_x[0]))
    x2 = int2(np.round(contours_x[-1]))
    y1 = int2(np.round(contours_y[0]))
    y2 = int2(np.round(contours_y[-1]))
    img2 = cv2.line(img2,(x1,y1),(x2,y2),color=color,thickness=int2(np.ceil(t)))
    return (img2)

def int2(a):
    return (int(np.round(a)))

class Draw():
    def contours(self, image, scale, move_x=0, move_y=0, annotations=None, mode="add"):
        """
        Interactive contour drawing:
        - Green polylines drawn continuously with mouse
        - Red overlay for atomic + combined masks
        - Scroll wheel zoom/pan
        - Right click undo
        - ESC or 'X' closes window
        """
        self.image = image
        self.scale = scale
        self.drawing = False
        self.t = 2  # line thickness

        # Stroke storage
        self.contours_x, self.contours_y = [], []
        self.countur_x, self.countur_y = [], []

        # Pan/zoom state
        self.dx1 = self.dy1 = 0
        self.dx2 = self.dy2 = 1
        self.scale2x = self.scale2y = 1
        self.p = 0.1  # zoom step

        H, W = self.image.shape[:2]

        # --- Base layer (original + masks) ---
        base = self.image.copy()
        if base.ndim == 2:
            base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
            
        def get_final_cracks(annotations):
            ann = (annotations or {})
            atomic = ann.get("atomic_cracks", {}) or {}
            combined = ann.get("combined_cracks", {}) or {}

            # collect all atomic IDs that belong to a combined crack
            atomic_members = set()
            for cmb in combined.values():
                for m in cmb.get("members", []):
                    atomic_members.add(str(m))

            # final list = (atomic without membership) + (combined cracks)
            finals = []

            # non-member atomic cracks
            for cid, cr in atomic.items():
                if str(cid) not in atomic_members:
                    finals.append(cr)

            # combined cracks
            for cr in combined.values():
                finals.append(cr)

            return finals
        
        def draw_midline_overlay(img, annotations):
            """
            Overlay midlines + endpoints onto the contours drawing canvas.

            Rules:
            - Combined cracks → yellow midline, NO endpoints
            - Atomic cracks not in any combined → white midline + magenta endpoints
            - Atomic members of combined are NOT drawn
            """
            import cv2
            import numpy as np

            shown = img.copy()
            atomic = (annotations or {}).get("atomic_cracks", {}) or {}
            combined = (annotations or {}).get("combined_cracks", {}) or {}

            # ---- gather all atomic IDs that are members of combined cracks ----
            member_ids = set()
            for cmb in combined.values():
                members = cmb.get("members", []) or []
                for m in members:
                    member_ids.add(str(m))

            # ================================
            #   1) Draw combined midlines (yellow)
            # ================================
            def draw_crack(crack, color, thickness=2):
                ml = crack.get("midline")
                if not ml or len(ml) < 2:
                    return

                seg = []
                for p in ml:
                    # gap / pen-up marker
                    if (
                        p is None or
                        len(p) != 2 or
                        p[0] is None or
                        p[1] is None
                    ):
                        if len(seg) >= 2:
                            pts = np.asarray(seg, np.int32)
                            cv2.polylines(
                                shown,
                                [pts],
                                False,
                                color,
                                thickness,
                                cv2.LINE_AA
                            )
                        seg = []
                        continue

                    seg.append([float(p[0]), float(p[1])])

                # flush last segment
                if len(seg) >= 2:
                    pts = np.asarray(seg, np.int32)
                    cv2.polylines(
                        shown,
                        [pts],
                        False,
                        color,
                        thickness,
                        cv2.LINE_AA
                    )
            
            for cid, crack in combined.items():
                draw_crack(crack, (255, 255, 0))   # yellow

            for cid, crack in atomic.items():
                if str(cid) in member_ids:
                    continue
                draw_crack(crack, (255, 255, 255)) # white

                # atomic endpoints
                user_pts = crack.get("user_points") or []
                for p in user_pts:
                    if p is None:
                        continue
                    x, y = int(p[0]), int(p[1])

                    # outline
                    cv2.circle(shown, (x, y), 4, (0, 0, 0), 2, cv2.LINE_AA)
                    # magenta
                    cv2.circle(shown, (x, y), 4, (128, 0, 0), -1, cv2.LINE_AA)

            return shown

        def reconstruct_full_mask(crack, H, W):
            mc, bb = crack.get("mask_crop"), crack.get("mask_bbox")
            if mc is None or bb is None or not len(mc):
                return np.zeros((H, W), np.uint8)
            crop = np.array(mc, dtype=np.uint8)
            x0, y0, w, h = [int(v) for v in bb]
            x1, y1 = min(x0 + w, W), min(y0 + h, H)
            mask = np.zeros((H, W), np.uint8)
            mask[y0:y1, x0:x1] = crop[:y1 - y0, :x1 - x0]
            return (mask > 0).astype(np.uint8)

        try:
            if annotations is not None:
                ann = annotations or {}
                atomic  = ann.get("atomic_cracks", {}) or {}
                combined = ann.get("combined_cracks", {}) or {}

                # collect atomic IDs that belong to a combined crack
                member_ids = {
                    str(m)
                    for cmb in combined.values()
                    for m in (cmb.get("members", []) or [])
                }

                overlay = np.zeros_like(base, dtype=np.uint8)

                # -------- draw final atomic masks --------
                for cid, crack in atomic.items():
                    if str(cid) in member_ids:
                        continue  # skip atomic members of combined

                    mask = reconstruct_full_mask(crack, H, W)
                    if np.any(mask):
                        overlay[mask.astype(bool)] = (0, 0, 255)

                # -------- draw combined masks --------
                for crack in combined.values():
                    mask = reconstruct_full_mask(crack, H, W)
                    if np.any(mask):
                        overlay[mask.astype(bool)] = (0, 0, 255)

                base = cv2.addWeighted(base, 1.0, overlay, 0.5, 0)

        except Exception:
            pass


        # Layers
        committed = draw_midline_overlay(base.copy(), annotations)

        live_points = []

        contours_name = f"{mode.upper()} contours (Esc or 'X' closes) midline(Cyan=combined, White=atomic); mask(Red); non-combined atomic endpoints (Blue)"
        gui_normal_flag = getattr(cv2, "WINDOW_GUI_NORMAL", 0)
        # AUTOSIZE avoids OpenCV Qt viewport drag/pan behavior hijacking left-drag.
        cv2.namedWindow(contours_name, cv2.WINDOW_AUTOSIZE | gui_normal_flag)
        cv2.moveWindow(contours_name, move_x, move_y)

        if mode=='add':
                #green
                active_draw_color = (0,255,0) #green
                done_draw_color = (0,200,0)
        else:
                #blue
                active_draw_color = (0, 165, 255)   # bright orange
                done_draw_color   = (0, 140, 255)   # darker orange
        
        def redraw_committed():
            nonlocal committed
            committed = draw_midline_overlay(base.copy(), annotations)
            committed = redrow_lines(committed, self.contours_x, self.contours_y, self.t, 1, color=done_draw_color)

        def _apply_zoom(rx, ry, zoom_in):
            ddx = (W - (self.dx1 + self.dx2)) * self.p
            ddy = (H - (self.dy1 + self.dy2)) * self.p
            if zoom_in:
                self.dx1 = max(int2(self.dx1 + ddx * rx), 0)
                self.dx2 = max(int2(self.dx2 + ddx * (1 - rx)), 1)
                self.dy1 = max(int2(self.dy1 + ddy * ry), 0)
                self.dy2 = max(int2(self.dy2 + ddy * (1 - ry)), 1)
            else:
                self.dx1 = max(int2(self.dx1 - ddx * rx), 0)
                self.dx2 = max(int2(self.dx2 - ddx * (1 - rx)), 1)
                self.dy1 = max(int2(self.dy1 - ddy * ry), 0)
                self.dy2 = max(int2(self.dy2 - ddy * (1 - ry)), 1)

            self.scale2x = 1 - (self.dx1 + self.dx2) / W
            self.scale2y = 1 - (self.dy1 + self.dy2) / H

        def on_mouse(event, x, y, flags, param):
            nonlocal live_points

            # --- Adjust for zoom/pan offsets ---
            x_real = self.dx1 + x * self.scale * self.scale2x
            y_real = self.dy1 + y * self.scale * self.scale2y

            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.countur_x, self.countur_y = [x_real], [y_real]
                live_points = [(x_real, y_real)]

            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                self.countur_x.append(x_real)
                self.countur_y.append(y_real)
                live_points.append((x_real, y_real))

            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                self.countur_x.append(x_real)
                self.countur_y.append(y_real)
                self.contours_x.append(self.countur_x)
                self.contours_y.append(self.countur_y)
                redraw_committed()
                live_points = []
                
                # 🔑 reset current stroke so next stroke starts fresh
                self.countur_x, self.countur_y = [], []

            elif event == cv2.EVENT_RBUTTONDOWN:
                if not self.drawing and len(self.contours_x) > 0:
                    self.contours_x.pop()
                    self.contours_y.pop()
                    redraw_committed()
                    live_points = []

            elif event == cv2.EVENT_MOUSEWHEEL:
                # Restore original wheel zoom behavior for manual draw/erase.
                if flags > 0:
                    _apply_zoom(x / self.image_countur.shape[1], y / self.image_countur.shape[0], True)
                else:
                    _apply_zoom(x / self.image_countur.shape[1], y / self.image_countur.shape[0], False)

        cv2.setMouseCallback(contours_name, on_mouse)

        while True:
            display = committed.copy()

            # overlay live stroke (bright green while drawing)
                
            if self.drawing and len(live_points) > 1:
                for i in range(1, len(live_points)):
                    cv2.line(display,
                            (int2(live_points[i - 1][0]), int2(live_points[i - 1][1])),
                            (int2(live_points[i][0]), int2(live_points[i][1])),
                            active_draw_color, self.t)
            else:
                # show darker green for completed strokes
                display = redrow_lines(display, self.contours_x, self.contours_y, self.t, 1, color=done_draw_color)

            # crop to zoom region
            x1 = max(0, self.dx1)
            x2 = min(W - self.dx2, W)
            y1 = max(0, self.dy1)
            y2 = min(H - self.dy2, H)
            if x2 <= x1 or y2 <= y1:
                view = display
            else:
                view = display[y1:y2, x1:x2]

            new_w = int2(view.shape[1] / self.scale / self.scale2x)
            new_h = int2(view.shape[0] / self.scale / self.scale2y)
            self.image_countur = cv2.resize(view, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            cv2.imshow(contours_name, self.image_countur)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('+'), ord('='), ord(']')):
                _apply_zoom(0.5, 0.5, True)
            elif key in (ord('-'), ord('_'), ord('[')):
                _apply_zoom(0.5, 0.5, False)
            if key == 27 or cv2.getWindowProperty(contours_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()

        flat_x = [item for sublist in self.contours_x for item in sublist]
        flat_y = [item for sublist in self.contours_y for item in sublist]
        return np.array(flat_x) - 0.5, np.array(flat_y) - 0.5

    def clear_pending_segment(self):
        """Discard the currently drawn (but unsaved) polygon and clear the preview."""
        if hasattr(self, "manuall_x"): del self.manuall_x
        if hasattr(self, "manuall_y"): del self.manuall_y
        if hasattr(self, "pending_mode"): del self.pending_mode

        # Repaint the preview with only existing cracks (red)
        try:
            im = self.image.astype(np.uint8).copy()
            H, W = im.shape[:2]
            ann = self.annotation.get("annotations", {})
            atomic = (ann or {}).get("atomic_cracks", {})
            combined = (ann or {}).get("combined_cracks", {})

            def reconstruct_full_mask(crack):
                mc, bb = crack.get("mask_crop"), crack.get("mask_bbox")
                if mc is None or bb is None or not len(mc):
                    return np.zeros((H, W), np.uint8)
                crop = np.array(mc, dtype=np.uint8)
                x0, y0, w, h = [int(v) for v in bb]
                x1, y1 = min(x0 + w, W), min(y0 + h, H)
                mask = np.zeros((H, W), np.uint8)
                mask[y0:y1, x0:x1] = crop[:y1-y0, :x1-x0]
                return (mask > 0).astype(np.uint8)

            red = np.zeros_like(im)
            for crack in list(atomic.values()) + list(combined.values()):
                m = reconstruct_full_mask(crack)
                if np.any(m):
                    red[m.astype(bool)] = (255, 0, 0)
            im = cv2.addWeighted(im, 1, red, 0.35, 0)

            qimage = QImage(im, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            scaled = pixmap.scaled(self.manual_segment_screen.width(),
                                self.manual_segment_screen.height(),
                                Qt.KeepAspectRatio, Qt.FastTransformation)
            self.manual_segment_screen.setPixmap(scaled)
        except:
            pass
            
    def points(self,image,scale,t = 5,move_x = 0, move_y = 0):
        """
        image : array
            Image to drow on
        scale : int,float
            defins size of display window    
        """
        self.image = image
        self.image_countur = self.image.copy()
        self.scale = scale
        
        self.t = t
        self.p = 0.1
        self.pt1_x , self.pt1_y = None , None
        self.pts = []
        self.pt_x = []
        self.pt_y = []
        self.image2 = image.copy()
        self.dx = 0
        self.dy = 0
        self.dx1 = 0
        self.dx2 = 1
        self.dy1 = 0
        self.dy2 = 1
        self.scale2 = 1   
        self.scale2x = 1
        self.scale2y = 1
    
        
        cv2.namedWindow('draw points')
        cv2.moveWindow('draw points', move_x, move_y)
        cv2.setMouseCallback('draw points',self.put_points)

        self.image_countur = cv2.resize(self.image_countur,[int2(self.image_countur.shape[1]/scale),
                                                            int2(self.image_countur.shape[0]/scale)],
                                        interpolation = cv2.INTER_NEAREST)
        while(1):
            cv2.imshow('draw points',self.image_countur)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()

#         flat_x = [item for sublist in self.contours_x for item in sublist]
#         flat_y = [item for sublist in self.contours_y for item in sublist]

#         flat_x = np.array(flat_x) - 0.5
#         flat_y = np.array(flat_y)- 0.5

        return self.pts
        
        
    def put_points(self,event,x,y,flags,param):

        if event==cv2.EVENT_LBUTTONDOWN:
            
            self.pt1_x,self.pt1_y=self.dx1+x*self.scale*self.scale2x,self.dy1+y*self.scale*self.scale2y
            self.pts.append(np.array([self.pt1_x,self.pt1_y]))
            self.image2 = redrow_points(self.image,self.pts,self.t,np.mean([self.scale2x,self.scale2y]))
            cv2.line(self.image_countur,(int2(x-10),int2(y)),(int2(x+10),int2(y)),color=(0,255,0),thickness=1)
            cv2.line(self.image_countur,(int2(x),int2(y-10)),(int2(x),int2(y+10)),color=(0,255,0),thickness=1)
#             self.image_countur = self.image2.copy()

        elif event==cv2.EVENT_RBUTTONDOWN:
            if len(self.pts)>0:
                
                self.pts = self.pts[:-1]
                self.image2 = redrow_points(self.image,self.pts,self.t,np.mean([self.scale2x,self.scale2y]))
                self.image_countur = cv2.resize(self.image2[self.dy1:-self.dy2,self.dx1:-self.dx2,:],
                                                [int2(self.image.shape[1]/self.scale/self.scale2),
                                                 int2(self.image.shape[0]/self.scale/self.scale2)],
                            interpolation = cv2.INTER_NEAREST)

        elif event==cv2.EVENT_MOUSEWHEEL and flags>0:

            rx,ry = x/self.image_countur.shape[1],y/self.image_countur.shape[0]

            ddx = (self.image.shape[1]-(self.dx1+self.dx2))*self.p
            self.dx1 = np.max([int2(self.dx1+ddx*rx),0])
            self.dx2 = np.max([int2(self.dx2+ddx*(1-rx)),1])

            ddy = (self.image.shape[0]-(self.dy1+self.dy2))*self.p
            self.dy1 = np.max([int2(self.dy1+ddy*ry),0])
            self.dy2 = np.max([int2(self.dy2+ddy*(1-ry)),1])

            self.scale2x = 1-(self.dx1+self.dx2)/self.image.shape[1]
            self.scale2y = 1-(self.dy1+self.dy2)/self.image.shape[0]
    #         image2 = redrow_lines(image,contours_x,contours_y,t,scale*scale2x)
            self.image_countur = self.image2[self.dy1:-self.dy2,self.dx1:-self.dx2,:]
            self.image_countur = cv2.resize(self.image_countur,[int2(self.image_countur.shape[1]/self.scale/self.scale2x),
                                                    int2(self.image_countur.shape[0]/self.scale/self.scale2y)],
                                                    interpolation = cv2.INTER_NEAREST)
        elif event==cv2.EVENT_MOUSEWHEEL:
            rx,ry = x/self.image_countur.shape[1],y/self.image_countur.shape[0]

            ddx = (self.image.shape[1]-(self.dx1+self.dx2))*self.p
            self.dx1 = np.max([int2(self.dx1-ddx*rx),0])
            self.dx2 = np.max([int2(self.dx2-ddx*(1-rx)),1])

            ddy = (self.image.shape[0]-(self.dy1+self.dy2))*self.p
            self.dy1 = np.max([int2(self.dy1-ddy*ry),0])
            self.dy2 = np.max([int2(self.dy2-ddy*(1-ry)),1])

            self.scale2x = 1-(self.dx1+self.dx2)/self.image.shape[1]
            self.scale2y = 1-(self.dy1+self.dy2)/self.image.shape[0]
    #         image2 = redrow_lines(image,contours_x,contours_y,t,scale*scale2x)
            self.image_countur = self.image2[self.dy1:-self.dy2,self.dx1:-self.dx2,:]
            self.image_countur = cv2.resize(self.image_countur,[int2(self.image_countur.shape[1]/self.scale/self.scale2x),
                                                    int2(self.image_countur.shape[0]/self.scale/self.scale2y)],
                                                    interpolation = cv2.INTER_NEAREST)
            
    def _bb_toggle_hit(self, x, y):
        if not getattr(self, "_bb_has_alt", False):
            return False
        x0, y0, x1, y1 = getattr(self, "_bb_toggle_rect", (0, 0, 0, 0))
        return (x0 <= x <= x1) and (y0 <= y <= y1)

    def _bb_draw_toggle_button(self, frame):
        if not getattr(self, "_bb_has_alt", False):
            return frame
        x0, y0, x1, y1 = self._bb_toggle_rect
        out = frame.copy()
        cv2.rectangle(out, (x0, y0), (x1, y1), (30, 30, 30), -1)
        cv2.rectangle(out, (x0, y0), (x1, y1), (220, 220, 220), 1)
        label = "View: RAW" if self._bb_show_alt else "View: OVERLAY"
        cv2.putText(out, label, (x0 + 8, y0 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(out, "Toggle: T", (x0 + 8, y0 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        return out

    def _bb_refresh_view(self, x_real=None, y_real=None):
        if x_real is not None and y_real is not None:
            self._bb_last_cursor_real = (float(x_real), float(y_real))

        base = (self.image_alt if self._bb_show_alt else self.image).copy()
        if x_real is None or y_real is None:
            last = getattr(self, "_bb_last_cursor_real", None)
            if last is not None:
                lx, ly = last
                self.image2 = redrow_coordinates(
                    base, int(lx), int(ly), self.t, np.mean([self.scale2x, self.scale2y])
                )
                self.image2 = redrow_bb(
                    self.image2, int(lx), int(ly), self.t, np.mean([self.scale2x, self.scale2y]),
                    self.pts, self.active, self.c, initial_pairs=getattr(self, "_bb_initial_pairs", 0)
                )
            else:
                self.image2 = redrow_bb(
                    base, None, None, self.t, np.mean([self.scale2x, self.scale2y]),
                    self.pts, self.active, self.c, initial_pairs=getattr(self, "_bb_initial_pairs", 0)
                )
        else:
            self.image2 = redrow_coordinates(
                base, int(x_real), int(y_real), self.t, np.mean([self.scale2x, self.scale2y])
            )
            self.image2 = redrow_bb(
                self.image2, int(x_real), int(y_real), self.t, np.mean([self.scale2x, self.scale2y]),
                self.pts, self.active, self.c, initial_pairs=getattr(self, "_bb_initial_pairs", 0)
            )

        crop = self.image2[self.dy1:-self.dy2, self.dx1:-self.dx2, :]
        if crop.size == 0:
            crop = self.image2
        new_w = max(1, int2(crop.shape[1] / self.scale / max(self.scale2x, 1e-8)))
        new_h = max(1, int2(crop.shape[0] / self.scale / max(self.scale2y, 1e-8)))
        canvas = cv2.resize(crop, [new_w, new_h], interpolation=cv2.INTER_LINEAR)

        # Cap display image to monitor bounds so HighGUI window stays usable.
        max_w = int(max(1, getattr(self, "_bb_max_disp_w", canvas.shape[1])))
        max_h = int(max(1, getattr(self, "_bb_max_disp_h", canvas.shape[0])))
        ds = min(1.0, max_w / float(max(canvas.shape[1], 1)), max_h / float(max(canvas.shape[0], 1)))
        self._bb_display_scale = float(ds)
        if ds < 1.0:
            dw = max(1, int(round(canvas.shape[1] * ds)))
            dh = max(1, int(round(canvas.shape[0] * ds)))
            self.image_countur = cv2.resize(canvas, [dw, dh], interpolation=cv2.INTER_AREA)
        else:
            self.image_countur = canvas

    def _bb_get_view_bounds(self):
        H, W = self.image.shape[:2]
        x1 = int(max(0, self.dx1))
        x2 = int(min(W - self.dx2, W))
        y1 = int(max(0, self.dy1))
        y2 = int(min(H - self.dy2, H))
        if x2 <= x1:
            x1, x2 = 0, W
        if y2 <= y1:
            y1, y2 = 0, H
        return x1, x2, y1, y2

    def _bb_draw_minimap(self, frame):
        out = frame.copy()
        fh, fw = out.shape[:2]
        if fh < 80 or fw < 120:
            self._bb_minimap_rect = None
            return out

        src = self.image_alt if (self._bb_has_alt and self._bb_show_alt) else self.image
        H, W = src.shape[:2]
        pad = 10
        max_w = min(260, max(120, int(fw * 0.28)))
        max_h = min(190, max(80, int(fh * 0.28)))
        aspect = float(W) / float(max(H, 1))
        mini_w = max_w
        mini_h = int(round(mini_w / max(aspect, 1e-8)))
        if mini_h > max_h:
            mini_h = max_h
            mini_w = int(round(mini_h * aspect))
        mini_w = max(80, min(mini_w, fw - 2 * pad))
        mini_h = max(60, min(mini_h, fh - 2 * pad))

        x0 = max(pad, fw - mini_w - pad)
        if getattr(self, "_bb_minimap_anchor", "top_right") == "bottom_right":
            y0 = max(pad, fh - mini_h - pad)
        else:
            y0 = pad
        x1 = x0 + mini_w
        y1 = y0 + mini_h
        self._bb_minimap_rect = (x0, y0, x1, y1)

        mini = cv2.resize(src, (mini_w, mini_h), interpolation=cv2.INTER_AREA)
        out[y0:y1, x0:x1] = mini
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 1)

        vx1, vx2, vy1, vy2 = self._bb_get_view_bounds()
        rx0 = x0 + int(round((vx1 / max(W, 1)) * mini_w))
        rx1 = x0 + int(round((vx2 / max(W, 1)) * mini_w))
        ry0 = y0 + int(round((vy1 / max(H, 1)) * mini_h))
        ry1 = y0 + int(round((vy2 / max(H, 1)) * mini_h))
        rx0 = int(np.clip(rx0, x0, x1 - 1))
        ry0 = int(np.clip(ry0, y0, y1 - 1))
        rx1 = int(np.clip(rx1, rx0 + 1, x1))
        ry1 = int(np.clip(ry1, ry0 + 1, y1))
        cv2.rectangle(out, (rx0, ry0), (rx1, ry1), (0, 255, 255), 1)
        anchor_txt = "BR" if getattr(self, "_bb_minimap_anchor", "top_right") == "bottom_right" else "TR"
        cv2.putText(out, f"Minimap ({anchor_txt})", (x0 + 6, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def _bb_pan_to_minimap_xy(self, mx, my):
        if not getattr(self, "_bb_minimap_rect", None):
            return
        x0, y0, x1, y1 = self._bb_minimap_rect
        if x1 <= x0 or y1 <= y0:
            return
        px = float(np.clip((mx - x0) / float(x1 - x0), 0.0, 1.0))
        py = float(np.clip((my - y0) / float(y1 - y0), 0.0, 1.0))

        H, W = self.image.shape[:2]
        vx1, vx2, vy1, vy2 = self._bb_get_view_bounds()
        vw = max(1, vx2 - vx1)
        vh = max(1, vy2 - vy1)

        cx = int(round(px * (W - 1)))
        cy = int(round(py * (H - 1)))
        nx1 = int(np.clip(cx - vw // 2, 0, max(W - vw, 0)))
        ny1 = int(np.clip(cy - vh // 2, 0, max(H - vh, 0)))
        nx2 = nx1 + vw
        ny2 = ny1 + vh

        self.dx1 = nx1
        self.dx2 = max(1, W - nx2)
        self.dy1 = ny1
        self.dy2 = max(1, H - ny2)
        self.scale2x = 1 - (self.dx1 + self.dx2) / W
        self.scale2y = 1 - (self.dy1 + self.dy2) / H
        self._bb_refresh_view()

    def _bb_pan_pixels(self, shift_x, shift_y):
        H, W = self.image.shape[:2]
        vx1, vx2, vy1, vy2 = self._bb_get_view_bounds()
        vw = max(1, vx2 - vx1)
        vh = max(1, vy2 - vy1)
        nx1 = int(np.clip(vx1 + shift_x, 0, max(W - vw, 0)))
        ny1 = int(np.clip(vy1 + shift_y, 0, max(H - vh, 0)))
        nx2 = nx1 + vw
        ny2 = ny1 + vh
        self.dx1 = nx1
        self.dx2 = max(1, W - nx2)
        self.dy1 = ny1
        self.dy2 = max(1, H - ny2)
        self.scale2x = 1 - (self.dx1 + self.dx2) / W
        self.scale2y = 1 - (self.dy1 + self.dy2) / H
        self._bb_refresh_view()

    def _bb_get_screen_size(self):
        # Try native Windows metrics first; fallback to sane defaults.
        try:
            import ctypes
            user32 = ctypes.windll.user32
            return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
        except Exception:
            return 1920, 1080

    def _bb_event_to_canvas(self, x, y):
        s = float(max(getattr(self, "_bb_display_scale", 1.0), 1e-8))
        return float(x) / s, float(y) / s

    def bounding_box(self,image,scale,t = 5, move_x = 0, move_y = 0, image_alt=None, initial_pts=None):
        """
        image : array
            Image to drow on
        scale : int,float
            defins size of display window    
        """
        self.image = image
        if image_alt is not None:
            self.image_alt = image_alt
            if self.image_alt.shape[:2] != self.image.shape[:2]:
                self.image_alt = cv2.resize(
                    self.image_alt,
                    (self.image.shape[1], self.image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            if self.image_alt.ndim == 2:
                self.image_alt = cv2.cvtColor(self.image_alt, cv2.COLOR_GRAY2BGR)
        else:
            self.image_alt = None
        if self.image.ndim == 2:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        self.image_countur = self.image.copy()
        self.scale = scale
        
        self.t = t
        self.p = 0.1
        self.pt1_x , self.pt1_y = None , None
        self.pts = []
        if initial_pts is not None:
            try:
                for p in list(initial_pts):
                    arr = np.asarray(p, dtype=float).ravel()
                    if arr.size >= 2:
                        self.pts.append(np.array([arr[0], arr[1]], dtype=float))
            except Exception:
                self.pts = []
        self._bb_initial_pairs = int(len(self.pts) // 2)
        # Use neutral class marker for pre-existing pending boxes so class hotkeys
        # continue to apply only to newly drawn boxes.
        self.c = [0] * self._bb_initial_pairs
        self.pt_x = []
        self.pt_y = []
        self.image2 = image.copy()
        self.dx = 0
        self.dy = 0
        self.dx1 = 0
        self.dx2 = 1
        self.dy1 = 0
        self.dy2 = 1
        self.scale2 = 1   
        self.scale2x = 1
        self.scale2y = 1
        self.active = False
        self._bb_has_alt = self.image_alt is not None
        self._bb_show_alt = False
        self._bb_toggle_rect = (10, 10, 240, 48)
        self._bb_minimap_rect = None
        self._bb_drag_minimap = False
        self._bb_minimap_anchor = "top_right"
        self._bb_last_cursor_real = None
        self._bb_display_scale = 1.0
        sw, sh = self._bb_get_screen_size()
        self._bb_max_disp_w = max(320, int(sw * 0.92))
        self._bb_max_disp_h = max(240, int(sh * 0.86))
    
        
        # Linux/OpenCV Qt workaround: wheel zoom is intentionally not used here
        # because HighGUI Qt can switch to hand/pan viewport mode and hijack
        # left-drag interactions. Use +/- keyboard zoom for now.
        bb_name = "draw bb (Esc close; RightClick undo; +/- zoom; arrows pan; T toggle view; M minimap side)"
        gui_normal_flag = getattr(cv2, "WINDOW_GUI_NORMAL", 0)
        # AUTOSIZE avoids OpenCV Qt viewport drag/pan behavior hijacking left-drag.
        cv2.namedWindow(bb_name, cv2.WINDOW_AUTOSIZE | gui_normal_flag)
        safe_x = int(np.clip(move_x, 0, max(0, sw - 200)))
        safe_y = int(np.clip(move_y, 0, max(0, sh - 150)))
        cv2.moveWindow(bb_name, safe_x, safe_y)
        cv2.setMouseCallback(bb_name,self.bb)
        self._bb_refresh_view()

        def _apply_bb_zoom(rx, ry, zoom_in):
            ddx = (self.image.shape[1] - (self.dx1 + self.dx2)) * self.p
            ddy = (self.image.shape[0] - (self.dy1 + self.dy2)) * self.p
            if zoom_in:
                self.dx1 = np.max([int2(self.dx1 + ddx * rx), 0])
                self.dx2 = np.max([int2(self.dx2 + ddx * (1 - rx)), 1])
                self.dy1 = np.max([int2(self.dy1 + ddy * ry), 0])
                self.dy2 = np.max([int2(self.dy2 + ddy * (1 - ry)), 1])
            else:
                self.dx1 = np.max([int2(self.dx1 - ddx * rx), 0])
                self.dx2 = np.max([int2(self.dx2 - ddx * (1 - rx)), 1])
                self.dy1 = np.max([int2(self.dy1 - ddy * ry), 0])
                self.dy2 = np.max([int2(self.dy2 - ddy * (1 - ry)), 1])
            self.scale2x = 1 - (self.dx1 + self.dx2) / self.image.shape[1]
            self.scale2y = 1 - (self.dy1 + self.dy2) / self.image.shape[0]
            self._bb_refresh_view()

        while(1):
            frame = self._bb_draw_toggle_button(self.image_countur)
            frame = self._bb_draw_minimap(frame)
            cv2.imshow(bb_name,frame)
            key = cv2.waitKeyEx(1)
            key_ascii = key & 0xFF
            if key_ascii in (ord('+'), ord('='), ord(']')):
                _apply_bb_zoom(0.5, 0.5, True)
            elif key_ascii in (ord('-'), ord('_'), ord('[')):
                _apply_bb_zoom(0.5, 0.5, False)
            elif key_ascii in (ord('t'), ord('T')) and self._bb_has_alt:
                self._bb_show_alt = not self._bb_show_alt
                self._bb_refresh_view()
            elif key_ascii in (ord('m'), ord('M')):
                self._bb_minimap_anchor = "bottom_right" if self._bb_minimap_anchor == "top_right" else "top_right"
                self._bb_refresh_view()
            elif key in (2424832, 65361, 81):  # left
                vx1, vx2, vy1, vy2 = self._bb_get_view_bounds()
                step = max(10, int(0.08 * max(1, vx2 - vx1)))
                self._bb_pan_pixels(-step, 0)
            elif key in (2555904, 65363, 83):  # right
                vx1, vx2, vy1, vy2 = self._bb_get_view_bounds()
                step = max(10, int(0.08 * max(1, vx2 - vx1)))
                self._bb_pan_pixels(step, 0)
            elif key in (2490368, 65362, 82):  # up
                vx1, vx2, vy1, vy2 = self._bb_get_view_bounds()
                step = max(10, int(0.08 * max(1, vy2 - vy1)))
                self._bb_pan_pixels(0, -step)
            elif key in (2621440, 65364, 84):  # down
                vx1, vx2, vy1, vy2 = self._bb_get_view_bounds()
                step = max(10, int(0.08 * max(1, vy2 - vy1)))
                self._bb_pan_pixels(0, step)
            if key_ascii == 27:
                break
            if len(self.c)<int(len(self.pts)/2):
                if key_ascii == 49:
                    self.c.append(1)
                if key_ascii == 50:
                    self.c.append(2)
                
        cv2.destroyAllWindows()

#         flat_x = [item for sublist in self.contours_x for item in sublist]
#         flat_y = [item for sublist in self.contours_y for item in sublist]

#         flat_x = np.array(flat_x) - 0.5
#         flat_y = np.array(flat_y)- 0.5

        return self.pts,self.c

    def bb(self,event,x,y,flags,param):

        if event==cv2.EVENT_LBUTTONDOWN:
            if self._bb_minimap_rect is not None:
                x0, y0, x1, y1 = self._bb_minimap_rect
                if x0 <= x <= x1 and y0 <= y <= y1:
                    self._bb_drag_minimap = True
                    self._bb_pan_to_minimap_xy(x, y)
                    return
            if self._bb_toggle_hit(x, y):
                self._bb_show_alt = not self._bb_show_alt
                self._bb_refresh_view()
                return
        elif event==cv2.EVENT_MOUSEMOVE and self._bb_drag_minimap:
            self._bb_pan_to_minimap_xy(x, y)
            return
        elif event==cv2.EVENT_LBUTTONUP and self._bb_drag_minimap:
            self._bb_drag_minimap = False
            return

        if event==cv2.EVENT_LBUTTONDOWN:
            if self.active == False:
                self.active = True
                x_canvas, y_canvas = self._bb_event_to_canvas(x, y)
                self.pt1_x,self.pt1_y=self.dx1+x_canvas*self.scale*self.scale2x,self.dy1+y_canvas*self.scale*self.scale2y
                self.pts.append(np.array([self.pt1_x,self.pt1_y]))
    #             self.image2 = redrow_points(self.image,self.pts,1,1)
#                 cv2.line(self.image_countur,(int2(x-10),int2(y)),(int2(x+10),int2(y)),color=(0,255,0),thickness=1)
#                 cv2.line(self.image_countur,(int2(x),int2(y-10)),(int2(x),int2(y+10)),color=(0,255,0),thickness=1)
            elif self.active == True:
                self.active = False
                x_canvas, y_canvas = self._bb_event_to_canvas(x, y)
                x1,y1=self.dx1+x_canvas*self.scale*self.scale2x,self.dy1+y_canvas*self.scale*self.scale2y
                self.pt1_x,self.pt1_y=self.dx1+x_canvas*self.scale*self.scale2x,self.dy1+y_canvas*self.scale*self.scale2y
                self.pts.append(np.array([self.pt1_x,self.pt1_y]))
                self._bb_refresh_view(x1, y1)
#                 self.c.append(input('class (1-crack, 2-corrosion):'))
    #             self.image2 = redrow_points(self.image,self.pts,1,1)
#                 cv2.line(self.image_countur,(int2(x-10),int2(y)),(int2(x+10),int2(y)),color=(0,255,0),thickness=1)
#                 cv2.line(self.image_countur,(int2(x),int2(y-10)),(int2(x),int2(y+10)),color=(0,255,0),thickness=1)
                
#             self.image_countur = self.image2.copy()
        

        elif event==cv2.EVENT_RBUTTONDOWN:
            if len(self.pts)>0:
                if self.active == True:
                    self.pts = self.pts[:-1]
                    self.active = False
                elif self.active == False:
                    self.pts = self.pts[:-2]
                    self.c = self.c[:-1]
                self._bb_initial_pairs = min(getattr(self, "_bb_initial_pairs", 0), int(len(self.pts)//2))
                if len(self.pts)>0:
                    self._bb_refresh_view()
                else:
                    self._bb_refresh_view()

                    
        if event==cv2.EVENT_MOUSEMOVE:
            x_canvas, y_canvas = self._bb_event_to_canvas(x, y)
            x1,y1=self.dx1+x_canvas*self.scale*self.scale2x,self.dy1+y_canvas*self.scale*self.scale2y
            self._bb_refresh_view(x1, y1)
                

        elif event==cv2.EVENT_MOUSEWHEEL and flags>0:

            rx,ry = x/self.image_countur.shape[1],y/self.image_countur.shape[0]

            ddx = (self.image.shape[1]-(self.dx1+self.dx2))*self.p
            self.dx1 = np.max([int2(self.dx1+ddx*rx),0])
            self.dx2 = np.max([int2(self.dx2+ddx*(1-rx)),1])

            ddy = (self.image.shape[0]-(self.dy1+self.dy2))*self.p
            self.dy1 = np.max([int2(self.dy1+ddy*ry),0])
            self.dy2 = np.max([int2(self.dy2+ddy*(1-ry)),1])

            self.scale2x = 1-(self.dx1+self.dx2)/self.image.shape[1]
            self.scale2y = 1-(self.dy1+self.dy2)/self.image.shape[0]
            self._bb_refresh_view()
            
            
        elif event==cv2.EVENT_MOUSEWHEEL:
            rx,ry = x/self.image_countur.shape[1],y/self.image_countur.shape[0]

            ddx = (self.image.shape[1]-(self.dx1+self.dx2))*self.p
            self.dx1 = np.max([int2(self.dx1-ddx*rx),0])
            self.dx2 = np.max([int2(self.dx2-ddx*(1-rx)),1])

            ddy = (self.image.shape[0]-(self.dy1+self.dy2))*self.p
            self.dy1 = np.max([int2(self.dy1-ddy*ry),0])
            self.dy2 = np.max([int2(self.dy2-ddy*(1-ry)),1])

            self.scale2x = 1-(self.dx1+self.dx2)/self.image.shape[1]
            self.scale2y = 1-(self.dy1+self.dy2)/self.image.shape[0]
            self._bb_refresh_view()
            

def image_crop(image,start_point,end_point,pts,sides1 = 10,sides2 = 10):
    """Function cropps input image with a rectengular 
    box making "sides" pixels indent from endpoints."""

    y_bound1 = int(np.max([int(np.min([start_point[1],end_point[1]]))-sides1,0]))
    y_bound2 = int(np.max([int(np.max([start_point[1],end_point[1]]))+sides1,0]))
    x_bound1 = int(np.max([int(np.min([start_point[0],end_point[0]]))-sides2,0]))
    x_bound2 = int(np.max([int(np.max([start_point[0],end_point[0]]))+sides2,0]))
    img_cropp = image[y_bound1:y_bound2,x_bound1:x_bound2,:]
    pts_cropp = []
    for pt in pts:
        pts_cropp.append(pt-[x_bound1,y_bound1])
    return img_cropp,pts_cropp

def track_crop_to_full(track_crop,start_point,end_point,sides1,sides2):
    y_bound1 = int(np.max([int(np.min([start_point[1],end_point[1]]))-sides1,0]))
    x_bound1 = int(np.max([int(np.min([start_point[0],end_point[0]]))-sides2,0]))
    track_x = np.array(track_crop[0]) + x_bound1
    track_y = np.array(track_crop[1]) + y_bound1
    return [track_x.squeeze(),track_y.squeeze()]

from pathlib import Path

def get_files(folder='cracktools/crackimages', formats=['png','jpg'], basename=True):
    import re

    def natural_key(path_like):
        name = Path(path_like).name if not basename else str(path_like)
        stem = Path(name).name.lower()
        return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', stem)]

    exts = [e.lower() for e in formats]
    files = []
    for f in Path(folder).glob("*"):
        if f.suffix.lower().lstrip(".") in exts:
            files.append(f.name if basename else str(f))
    files.sort(key=natural_key)
    return files
