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

##### choose points on image ##################
'''pts = []
def put_points(img1):
    # mouse callback function
    def line_drawing(event,x,y,flags,param):
        global pts,pt
        if event==cv2.EVENT_LBUTTONDOWN:
            pts.append(np.array([x,y]))
            cv2.line(img,(x-10,y),(x+10,y),color=(0,255,0),thickness=1)
            cv2.line(img,(x,y-10),(x,y+10),color=(0,255,0),thickness=1)

    img = img1.copy()
    cv2.namedWindow('test draw')
    cv2.setMouseCallback('test draw',line_drawing)

    while(1):
        cv2.imshow('test draw',img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    
    return (np.array(pts))'''

'''def points(img1,scalar):
    global pts
    pts = []
    contur_points = put_points(cv2.resize(img1,\
                            (int(img1.shape[1]/scalar),int(img1.shape[0]/scalar))))
    contur_points = scalar*contur_points
    return contur_points'''

'''def redrow_lines(img,contours_x,contours_y,t,scale):
    flat_x = [item for sublist in contours_x for item in sublist]
    flat_y = [item for sublist in contours_y for item in sublist]
    img2 = img.copy()
    for i in range(len(flat_x)-1):
        x1 = int2(flat_x[i]-0.5)
        x2 = int2(flat_x[i+1]-0.5)
        y1 = int2(flat_y[i]-0.5)
        y2 = int2(flat_y[i+1]-0.5)
        img2 = cv2.line(img2,(x1,y1),(x2,y2),color=(0,255,0),thickness=int2(np.ceil(t*scale)))
    return (img2)'''
    
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
    img2 = cv2.line(img2,(x,img2.shape[0]),(x,0),color=(0,255,0),thickness=int2(np.ceil(t*scale)))
    img2 = cv2.line(img2,(img2.shape[1],y),(0,y),color=(0,255,0),thickness=int2(np.ceil(t*scale)))
    return (img2)

def redrow_bb(img,x,y,t,scale,pts,active,c):
    img2 = img.copy()
    img2 = cv2.line(img2,(x,img2.shape[0]),(x,0),color=(0,255,0),thickness=int2(np.ceil(t*scale)))
    img2 = cv2.line(img2,(img2.shape[1],y),(0,y),color=(0,255,0),thickness=int2(np.ceil(t*scale)))
    if len(pts)>1:
        for i in range(0,len(pts)-1,2):
            x0 = int(pts[i][0])
            y0 = int(pts[i][1])
            x1 = int(pts[i+1][0])
            y1 = int(pts[i+1][1])
            color = (255,0,0)
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

        img2 = cv2.line(img2,(x,y),(x1,y),color=(255,0,0),thickness=int2(np.ceil(t*scale)))
        img2 = cv2.line(img2,(x,y),(x,y1),color=(255,0,0),thickness=int2(np.ceil(t*scale)))
        img2 = cv2.line(img2,(x,y1),(x1,y1),color=(255,0,0),thickness=int2(np.ceil(t*scale)))
        img2 = cv2.line(img2,(x1,y),(x1,y1),color=(255,0,0),thickness=int2(np.ceil(t*scale)))
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
        cv2.namedWindow(contours_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(contours_name, 1200, 800)  # make window large by default
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

            # --- Zoom with scroll wheel ---
            elif event == cv2.EVENT_MOUSEWHEEL:
                rx, ry = x / self.image_countur.shape[1], y / self.image_countur.shape[0]
                ddx = (W - (self.dx1 + self.dx2)) * self.p
                ddy = (H - (self.dy1 + self.dy2)) * self.p
                if flags > 0:  # zoom in
                    self.dx1 = max(int2(self.dx1 + ddx * rx), 0)
                    self.dx2 = max(int2(self.dx2 + ddx * (1 - rx)), 1)
                    self.dy1 = max(int2(self.dy1 + ddy * ry), 0)
                    self.dy2 = max(int2(self.dy2 + ddy * (1 - ry)), 1)
                else:  # zoom out
                    self.dx1 = max(int2(self.dx1 - ddx * rx), 0)
                    self.dx2 = max(int2(self.dx2 - ddx * (1 - rx)), 1)
                    self.dy1 = max(int2(self.dy1 - ddy * ry), 0)
                    self.dy2 = max(int2(self.dy2 - ddy * (1 - ry)), 1)

                self.scale2x = 1 - (self.dx1 + self.dx2) / W
                self.scale2y = 1 - (self.dy1 + self.dy2) / H

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
            self.image_countur = cv2.resize(view, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            cv2.imshow(contours_name, self.image_countur)

            key = cv2.waitKey(1) & 0xFF
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
            
    def bounding_box(self,image,scale,t = 5, move_x = 0, move_y = 0):
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
        self.c = []
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
    
        
        bb_name = 'draw bb (Esc closes, RightClick deletes most recent)'
        cv2.namedWindow(bb_name)
        cv2.moveWindow(bb_name, move_x, move_y)
        cv2.setMouseCallback(bb_name,self.bb)

        self.image_countur = cv2.resize(self.image_countur,[int2(self.image_countur.shape[1]/scale),
                                                            int2(self.image_countur.shape[0]/scale)],
                                        interpolation = cv2.INTER_NEAREST)
        while(1):
            cv2.imshow(bb_name,self.image_countur)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            if len(self.c)<int(len(self.pts)/2):
                if cv2.waitKey(1) & 0xFF == 49:
                    self.c.append(1)
                if cv2.waitKey(1) & 0xFF == 50:
                    self.c.append(2)
                
        cv2.destroyAllWindows()

#         flat_x = [item for sublist in self.contours_x for item in sublist]
#         flat_y = [item for sublist in self.contours_y for item in sublist]

#         flat_x = np.array(flat_x) - 0.5
#         flat_y = np.array(flat_y)- 0.5

        return self.pts,self.c

    def bb(self,event,x,y,flags,param):

        if event==cv2.EVENT_LBUTTONDOWN:
            if self.active == False:
                self.active = True
                self.pt1_x,self.pt1_y=self.dx1+x*self.scale*self.scale2x,self.dy1+y*self.scale*self.scale2y
                self.pts.append(np.array([self.pt1_x,self.pt1_y]))
    #             self.image2 = redrow_points(self.image,self.pts,1,1)
#                 cv2.line(self.image_countur,(int2(x-10),int2(y)),(int2(x+10),int2(y)),color=(0,255,0),thickness=1)
#                 cv2.line(self.image_countur,(int2(x),int2(y-10)),(int2(x),int2(y+10)),color=(0,255,0),thickness=1)
            elif self.active == True:
                self.active = False
                x1,y1=self.dx1+x*self.scale*self.scale2x,self.dy1+y*self.scale*self.scale2y
                self.pt1_x,self.pt1_y=self.dx1+x*self.scale*self.scale2x,self.dy1+y*self.scale*self.scale2y
                self.pts.append(np.array([self.pt1_x,self.pt1_y]))
                self.image2 = redrow_bb(self.image,int(x1),int(y1),self.t,np.mean([self.scale2x,self.scale2y]),
                                        self.pts,self.active,self.c)
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
                if len(self.pts)>0:
                    self.image2 = redrow_bb(self.image,None,None,self.t,np.mean([self.scale2x,self.scale2y]),
                                        self.pts,self.active,self.c)
                    self.image_countur = cv2.resize(self.image2[self.dy1:-self.dy2,self.dx1:-self.dx2,:],
                                                    [int2(self.image.shape[1]/self.scale/self.scale2),
                                                     int2(self.image.shape[0]/self.scale/self.scale2)],
                                interpolation = cv2.INTER_NEAREST)

                    
        if event==cv2.EVENT_MOUSEMOVE:
            x1,y1=self.dx1+x*self.scale*self.scale2x,self.dy1+y*self.scale*self.scale2y
            self.image2 = redrow_coordinates(self.image,int(x1),int(y1),self.t,np.mean([self.scale2x,self.scale2y]))
            self.image2 = redrow_bb(self.image,int(x1),int(y1),self.t,np.mean([self.scale2x,self.scale2y]),
                                    self.pts,self.active,self.c)
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
    exts = [e.lower() for e in formats]
    files = []
    for f in Path(folder).glob("*"):
        if f.suffix.lower().lstrip(".") in exts:
            files.append(f.name if basename else str(f))
    return files
