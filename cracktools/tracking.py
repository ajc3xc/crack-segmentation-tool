import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import os
import psutil
# import sys

# Prevent accidental giant ndarray dumps in debug prints.
np.set_printoptions(threshold=32, edgeitems=2, linewidth=120)

'''try:
    import cupy as cp
    try:
        _ = cp.cuda.runtime.getDeviceCount()
        if _ > 0:
            CUPY_AVAILABLE = True
        else:
            raise RuntimeError("No CUDA device found")
    except Exception:
        import numpy as np
        cp = np
        CUPY_AVAILABLE = False
except ImportError:
    import numpy as np
    cp = np
    CUPY_AVAILABLE = False
   
def asnumpy(x):
    if CUPY_AVAILABLE:
        return cp.asnumpy(x)
    return x'''
    
try:
    import cupy as cp
    try:
        n_devices = cp.cuda.runtime.getDeviceCount()
        if n_devices > 0:
            CUPY_AVAILABLE = True

            # -------------------------------
            # 💾 GPU + Pinned Memory Pools
            # -------------------------------
            from cupy.cuda import pinned_memory

            # Device (VRAM) pool
            gpu_pool = cp.cuda.MemoryPool()
            cp.cuda.set_allocator(gpu_pool.malloc)

            # Host (CPU) pinned-memory pool
            pin_pool = pinned_memory.PinnedMemoryPool()
            pinned_memory.set_pinned_memory_allocator(pin_pool.malloc)

            # 👇 New universal safe limit setter
            if hasattr(pin_pool, "set_limit"):
                pin_pool.set_limit(size=1024 * 1024 * 1024)  # 1 GB
                pinned_limit = pin_pool.get_limit() / (1024**2)
            else:
                pinned_limit = "auto"

            #print(f"[cupy] ✅ CUDA detected ({n_devices} device(s))")
            #print(f"[cupy] Memory pools active (pinned_limit={pinned_limit} MiB)")

        else:
            raise RuntimeError("No CUDA device found")

    except Exception as e:
        print(f"[cupy] ⚠️ GPU unavailable or failed ({e}) — falling back to NumPy")
        import numpy as np
        cp = np
        CUPY_AVAILABLE = False

except ImportError:
    print("[cupy] ❌ CuPy not installed — using NumPy fallback")
    import numpy as np
    cp = np
    CUPY_AVAILABLE = False


def asnumpy(x):
    """Safe conversion to NumPy array whether x is CuPy or NumPy."""
    if CUPY_AVAILABLE and hasattr(cp, "asnumpy"):
        return cp.asnumpy(x)
    return x


def free_cupy_caches():
    """Manual cleanup — free all cached GPU + pinned blocks between images."""
    if not CUPY_AVAILABLE:
        return
    try:
        cp.get_default_memory_pool().free_all_blocks()
        if hasattr(cp.cuda.pinned_memory, "get_default_pinned_memory_pool"):
            cp.cuda.pinned_memory.get_default_pinned_memory_pool().free_all_blocks()
    except Exception as e:
        print(f"[cupy] cleanup warning: {e}")

def _rss_gib():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)

def _mem_point(tag, arr=None):
    rss = _rss_gib()
    msg = f"[MEM] {tag} | RSS={rss:.3f} GiB"
    if arr is not None:
        shape = getattr(arr, "shape", None)
        dtype = getattr(arr, "dtype", None)
        nbytes = getattr(arr, "nbytes", None)
        if shape is not None and dtype is not None and nbytes is not None:
            msg += f" | shape={shape} dtype={dtype} size={nbytes/(1024**3):.3f} GiB"
        else:
            msg += f" | type={type(arr).__module__}.{type(arr).__name__}"
    print(msg)

def mem_snapshot(tag, arrays=None, cupy_mod=None, do_gc=False):
    """
    Lightweight memory snapshot for debugging OOM retries.
    arrays: optional iterable of (name, arr)
    """
    import gc

    if do_gc:
        gc.collect()

    print(f"[MEMSNAP] {tag}")
    _mem_point("rss")

    if arrays:
        for name, arr in arrays:
            _mem_point(name, arr)

    cm = cupy_mod if cupy_mod is not None else (cp if CUPY_AVAILABLE else None)
    if cm is not None and hasattr(cm, "cuda"):
        try:
            free, total = cm.cuda.runtime.memGetInfo()
            print(f"[MEMSNAP][GPU] free={free/(1024**3):.2f} GiB total={total/(1024**3):.2f} GiB")
            try:
                print(f"[MEMSNAP][GPU] pool_used={cm.get_default_memory_pool().used_bytes()/(1024**3):.2f} GiB")
            except Exception:
                pass
            try:
                pinned_pool = cm.get_default_pinned_memory_pool()
                if hasattr(pinned_pool, "used_bytes"):
                    print(f"[MEMSNAP][GPU] pinned_used={pinned_pool.used_bytes()/(1024**3):.2f} GiB")
                elif hasattr(pinned_pool, "n_free_blocks"):
                    print(f"[MEMSNAP][GPU] pinned_free_blocks={pinned_pool.n_free_blocks()}")
            except Exception:
                pass
        except Exception as e:
            print(f"[MEMSNAP][GPU] unavailable: {e}")

def _format_exception_brief(e, max_chars=600):
    msg = f"{type(e).__name__}: {e}"
    msg = " ".join(str(msg).split())
    if len(msg) > max_chars:
        msg = msg[:max_chars] + " ... [truncated]"
    return msg


def asnumpy(x):
    """Safe conversion to NumPy array whether x is CuPy or NumPy."""
    if CUPY_AVAILABLE and hasattr(cp, "asnumpy"):
        return cp.asnumpy(x)
    return x


def free_cupy_caches():
    """Manual cleanup — free all cached GPU + pinned blocks between images."""
    if not CUPY_AVAILABLE:
        return
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.pinned_memory.get_default_pinned_memory_pool().free_all_blocks()
        cp._default_memory_pool = None
    except Exception as e:
        print(f"[cupy] cleanup warning: {e}")

def tang_len(start_point_x,start_point_y,end_point_x,end_point_y):
    """Function defines oriantation and direction of line that connects two points"""
    dx = end_point_x - start_point_x
    dy = end_point_y - start_point_y
    l = np.sqrt(dx**2+dy**2)
    ddx = dx/l
    ddy = dy/l
    return ddx,ddy,l

'''def rot_matrix(theta):
    """Rotation matrix"""
    return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])

def inter_val(img,point_x,point_y,method='linear'):
    """Value of non-integer pixel position"""
    x1 = np.floor(point_x).astype(int)
    y1 = np.floor(point_y).astype(int)
    if method=='closest':
        val = img[x1,y1]
    else :
        x2 = np.ceil(point_x).astype(int)
        y2 = np.ceil(point_y).astype(int)
        z11 = img[x1,y1]
        z12 = img[x1,y2]
        z21 = img[x2,y1]
        z22 = img[x2,y2]
        if type(point_x)==list or type(point_x) == np.ndarray:
            val = []
            for i in range(len(point_x)):
                f = scipy.interpolate.interp2d([y1[i],y2[i]], [x1[i],x2[i]], [[z11[i],z12[i]],[z21[i],z22[i]]], kind=method)
                val.append(f(point_y[i],point_x[i]))
        else :
            f = scipy.interpolate.interp2d([y1,y2], [x1,x2], [[z11,z12],[z21,z22]], kind=method)
            val = f(point_y,point_x)
    return val

def inter_val_color(img,point_x,point_y,method='linear'):
    """Value of non-integer pixel position for color image"""
    x1 = np.floor(point_x).astype(int)
    x2 = np.ceil(point_x).astype(int)
    if method=='closest':
        val = img[y1,x1]
    else:
        y1 = np.floor(point_y).astype(int)
        y2 = np.ceil(point_y).astype(int)
        if type(point_x)==list or type(point_x) == np.ndarray:
            vals = np.zeros((len(x1),3))
        else :
            vals = np.zeros((3))
        for c in range(3):
            z11 = img[y1,x1,c]
            z12 = img[y1,x2,c]
            z21 = img[y2,x1,c]
            z22 = img[y2,x2,c]
            if type(point_x)==list or type(point_x) == np.ndarray:
                val = []
                for i in range(len(point_x)):
                    f = scipy.interpolate.interp2d([y1[i],y2[i]], [x1[i],x2[i]], [[z11[i],z12[i]],[z21[i],z22[i]]], kind='linear')
                    val.append(f(point_y[i],point_x[i]))
                    
                vals[:,c] = val
            else :
                f = scipy.interpolate.interp2d([y1,y2], [x1,x2], [[z11,z12],[z21,z22]], kind='linear')
                val = f(point_y,point_x)
                vals[c] = val
    return vals

def anomaly_color(vals):
    mean = np.mean(vals,axis = 0)
    dev = vals-mean
    l = np.sqrt(dev[:,0]**2+dev[:,1]**2+dev[:,2]**2)
    return (np.argmax(l))

def Route_finder(img,start_point,end_point,
                long_step = 5,side_search_step = 0.5,side_search_range = 20, interpolation_method='linear',color_method='Min'):
    """
        Algorithm that tracks a crack between endpoints. Described in :
        Dare, P. M. (2002). AN OPERATIONAL APPLICATION OF AUTOMATIC FEATURE EXTRACTION: THE MEASUREMENT 
            OF CRACKS IN CONCRETE STRUCTURES, (Vol. 17, Issue 99)
    """

    start_point_x = start_point[0]
    start_point_y = start_point[1]
    end_point_x = end_point[0]
    end_point_y = end_point[1]
    trak_points_x = []
    trak_points_y = []
    l2 = 0
    l1 = 0
    l2_old = 0
    while l1>=long_step*3 or len(trak_points_x)<2:
        ddx1,ddy1,l1 = tang_len(start_point_x,start_point_y,end_point_x,end_point_y)   # l1 - distance to the endpoint. ddx,ddy - line orientation
        side_steps = np.arange(side_search_step,side_search_range+side_search_step,side_search_step)
        # choosing of points of interest
        side_points_x1 = start_point_x + ddy1*side_steps
        side_points_y1 = start_point_y - ddx1*side_steps
        side_points_x2 = start_point_x - ddy1*side_steps
        side_points_y2 = start_point_y + ddx1*side_steps
        side_points_x = np.concatenate((side_points_x1,side_points_x2),axis = 0) 
        side_points_y = np.concatenate((side_points_y1,side_points_y2),axis = 0) 
        if len(img.shape)==2:
            val = inter_val(img,side_points_x,side_points_y,method=interpolation_method)
            start_point_x = side_points_x[np.argmin(val)]
            start_point_y = side_points_y[np.argmin(val)]
        elif len(img.shape)==3:
            val = inter_val_color(img,side_points_x,side_points_y)
            if color_method == 'Min':
                arg = np.argmin(np.sum(val,axis = 1))
            if color_method == 'Max':
                arg = np.argmax(np.sum(val,axis = 1))
            if color_method == 'Anomaly':
                arg = anomaly_color(val)
            start_point_x = side_points_x[arg]
            start_point_y = side_points_y[arg]
        trak_points_x.append(start_point_x)
        trak_points_y.append(start_point_y)
        l2_old = l2
        ddx2,ddy2,l2 = tang_len(start_point_x,start_point_y,end_point_x,end_point_y)
        if l2_old<l2 and len(trak_points_x)>10:
            n = 10
        else :
            n = 1
        start_point_x = start_point_x+ddx2*long_step*n
        start_point_y = start_point_y+ddy2*long_step*n
        
    trak_points_x.append(start_point_x)
    trak_points_y.append(start_point_y)
    trak_points_x.append(end_point_x)
    trak_points_y.append(end_point_y)
    return [trak_points_x,trak_points_y]

def plot_track(image,start_point,end_point,track,x_bound = None,y_bound=None,size = None):
    """"show image with calculated track"""
    # if size != None:
    #     fig = plt.figure()
    #     fig.set_size_inches(size, size)
    plt.figure()
    plt.imshow(image)
    plt.plot(start_point[0],start_point[1],'go', markersize=12)
    plt.plot(end_point[0],end_point[1],'ro', markersize=12)
    if track != None:
        plt.plot(track[0],track[1],'r-',)
    if x_bound != None:
        plt.xlim([x_bound[0],x_bound[1]])
    if y_bound != None:
        plt.ylim([y_bound[0],y_bound[1]])
    plt.show()

def Fly_fisher(img,start_point,end_point,
              n_directions = 100,search_range = 10,search_step = 1,move_step = 3,angle = 45):
    """
        Second algorithm that tracks a crack between endpoints. Described in :
        Dare, P. M. (2002). AN OPERATIONAL APPLICATION OF AUTOMATIC FEATURE EXTRACTION: THE MEASUREMENT 
            OF CRACKS IN CONCRETE STRUCTURES, (Vol. 17, Issue 99)
    """
    start_point_x1 = start_point[1]
    start_point_y1 = start_point[0]
    end_point_x1 = end_point[1]
    end_point_y1 = end_point[0]
    trak_points_x = []
    trak_points_y = []
    l2 = 0
    l1 = 0
    l2_old = 0
    angle_rad = angle/(180/np.pi)
    while l1>=move_step*3 or len(trak_points_x)<2:
    # for i in range(2):
        l1_old = l1
        dx1,dy1,l1 = tang_len(start_point_x1,start_point_y1,end_point_x1,end_point_y1)
        direction_sums = []
        theta = np.arange(-angle_rad,angle_rad+angle_rad*2/(n_directions-1),angle_rad*2/(n_directions-1))
        R = rot_matrix(theta)
        a = [np.matmul(R[:,:,i],[dx1,dy1]) for i in range(R.shape[2])]
        dsx1 = [a[i][0] for i in range(len(a))] 
        dsy1 = [a[i][1] for i in range(len(a))] 

        steps = np.arange(1,search_range+search_step,search_step)
        v_sums = []
        for direction in range(len(dsx1)):
            search_points_x1 = start_point_x1 + dsx1[direction]*steps
            search_points_y1 = start_point_y1 + dsy1[direction]*steps
            vals = inter_val(img,search_points_x1,search_points_y1)
            v_sums.append(np.sum(vals))

        trak_points_x.append(start_point_x1)
        trak_points_y.append(start_point_y1)   

    #     if l1_old<l1 and len(trak_points_x)>10:
    #         n = 3
    #         print(n)
    #     else :
        n = 1
        start_point_x1 = start_point_x1 + dsx1[np.argmin(v_sums)] * move_step*n
        start_point_y1 = start_point_y1 + dsy1[np.argmin(v_sums)] * move_step*n


        if len(trak_points_x)%1 == 0:
            print('distance to the end =',l1, end='\r')

    trak_points_x.append(end_point_x1)
    trak_points_y.append(end_point_y1)  
    return [trak_points_y,trak_points_x]

def Dijsktra_grid(grid, start_point, end_point,transition_cost = None):
    """
        Dijkstra's algorithm to compute shortest path with grid as a cost function
    """
    start_x = int(start_point[0])
    start_y = int(start_point[1])
    end_x = int(end_point[0])
    end_y = int(end_point[1])
    if transition_cost == None:
        transition_cost = np.zeros((grid.shape[0],grid.shape[1],4))
    x = int(start_x)
    y = int(start_y)
    distmap=np.ones_like(grid,dtype=int)*np.Infinity
    distmap[y,x]=0
    finished=False
    prev_cell=np.ones_like(grid,dtype=int)*np.nan
    visited=np.zeros_like(grid,dtype=bool)
    count = 0
    i = 0
    while not finished:
        # move to right
        if x < grid.shape[1]-1:
            if distmap[y,x+1]>grid[y,x+1]+distmap[y,x]+transition_cost[y,x,0] and not visited[y,x+1]:
                distmap[y,x+1]=grid[y,x+1]+distmap[y,x]+transition_cost[y,x,0]
                prev_cell[y,x+1]=np.ravel_multi_index([y,x], (grid.shape[0],grid.shape[1]))

        # move to left
        if x > 0:
            if distmap[y,x-1]>grid[y,x-1]+distmap[y,x]+transition_cost[y,x,2] and not visited[y,x-1]:
                distmap[y,x-1]=grid[y,x-1]+distmap[y,x]+transition_cost[y,x,2]
                prev_cell[y,x-1]=np.ravel_multi_index([y,x], (grid.shape[0],grid.shape[1]))

        # move up
        if y > 0:
            if distmap[y-1,x]>grid[y-1,x]+distmap[y,x]+transition_cost[y,x,3] and not visited[y-1,x]:
                distmap[y-1,x]=grid[y-1,x]+distmap[y,x]+transition_cost[y,x,3]
                prev_cell[y-1,x]=np.ravel_multi_index([y,x], (grid.shape[0],grid.shape[1]))

        # move down
        if y < grid.shape[0]-1:
            if distmap[y+1,x]>grid[y+1,x]+distmap[y,x]+transition_cost[y,x,1] and not visited[y+1,x]:
                distmap[y+1,x]=grid[y+1,x]+distmap[y,x]+transition_cost[y,x,1]
                prev_cell[y+1,x]=np.ravel_multi_index([y,x], (grid.shape[0],grid.shape[1]))

        visited[y,x]=True

        dismaptemp=distmap.copy()
        dismaptemp[np.where(visited)]=np.Infinity
        minpost=np.unravel_index(np.argmin(dismaptemp),np.shape(dismaptemp))
        y,x=minpost[0],minpost[1]
        if x==end_x and y==end_y:
            finished=True
        count=count+1
        i+=1
        if i%10000 == 0:
            print('distance to the end = ',np.sqrt((x - end_x)**2 + (y - end_y)**2), end = '\r')


#     mattemp=grid.astype(float)
    x,y=end_x,end_y
    path_x=[]
    path_y=[]
#     mattemp[int(y),int(x)]=np.nan

    while 1:
        path_x.append(int(x))
        path_y.append(int(y))
        xxyy=np.unravel_index(int(prev_cell[int(y),int(x)]), (grid.shape[0],grid.shape[1]))
        x,y=xxyy[1],xxyy[0]
#         mattemp[int(y),int(x)]=np.nan
        if x==start_x and y==start_y:
            break
    path_x.append(int(x))
    path_y.append(int(y))

    return [path_x,path_y]'''
    
'''def fast_marching_2d(cost,start_point,end_point,l = 1, p = 6):
    mu = 0
    seeds = np.array([*start_point[::-1]])
    tips = np.array([*end_point[::-1]])
    b = np.array([0,cost.shape[0]])
    c = np.array([0,cost.shape[1]])
    sides = np.array([b,c])
    dims = np.array([cost.shape[0],cost.shape[1]])
    
    DxZ,DyZ = np.gradient(cost) 
    a11 = scipy.ndimage.gaussian_filter(mu*DxZ**2, 1, order=(0,0))
    a12 = scipy.ndimage.gaussian_filter(mu*DxZ*DyZ, 1, order=(0,0))
    a21 = scipy.ndimage.gaussian_filter(mu*DxZ*DyZ, 1, order=(0,0))
    a22 = scipy.ndimage.gaussian_filter(mu*DyZ**2, 1, order=(0,0))
    df = np.array([[1+a11,a12],[a21,1+a22]])
    metric1 = (0.0001+cost*l)**p*df

    metric = Riemann(metric1)
    hfmIn = Eikonal.dictIn({
        'model' : 'Riemann2',
        'seeds' : np.expand_dims(seeds,axis = 0),
        'arrayOrdering' : 'RowMajor',
        'tips' : np.expand_dims(tips,axis = 0),
        'metric' : metric})
    hfmIn['order']=2
    hfmIn.SetRect(sides = sides, dims = dims)
    hfmOut = hfmIn.Run()
    geos1 = [g.T for g in hfmOut['geodesics']]
    print('Done.')
    
    return [geos1[0][:,1],geos1[0][:,0]]'''
    


'''from agd import Eikonal
from agd.Metrics import AsymQuad,Riemann # Riemannian metric and \Asymmetric Quadratic Models
from agd import AutomaticDifferentiation as ad
from agd import LinearParallel as lp
from agd import FiniteDifferences as fd
from agd import Eikonal

from agd.LinearParallel import outer_self as Outer # outer product v \v^T of a vector with itself
norm = ad.Optimization.norm
import numpy as np; xp=np

def ReedsSheppMetricGFOld(GF,dims,g11,g22,g33):
    nx = dims[1]
    ny = dims[2]
    nt = dims[0]
#     GFinv = np.array([np.linalg.inv(GF[i,:,:,:,:]) for i in range(GF.shape[0])])
    GFinv = GF # inverse of identity matrix. much faster this way
    LIFtoEuclidean = np.zeros((dims[0],3,3))
    for t in range(0,nt):
        LIFtoEuclidean[t,:,:] = GLIFtoEuclideanOld(t*2*np.pi/nt)
    
    LIFtoEuclideaninv = np.array([np.linalg.inv(LIFtoEuclidean[i]) for i in range(LIFtoEuclidean.shape[0])])
    metric = np.zeros((dims[0],dims[1],dims[2],3,3))
    for t in range(nt):
        for x in range(nx):
            for y in range(ny):
                metric[t,x,y,:,:] = GGF(g11,g22,g33,GFinv[t,x,y],LIFtoEuclideaninv[t,:,:])
                         
    return metric

def GGF(g11,g22,g33,GFtoLIFinv,LIFtoEuclideaninv):
    GF = np.diag([g11,g22,g33])
    transformMatrix = np.dot(LIFtoEuclideaninv,GFtoLIFinv)
    G = np.dot(transformMatrix,np.dot(GF,transformMatrix.T))
    return G

def GLIFtoEuclideanOld(theta):
    return np.array([[np.cos(theta),np.sin(theta),0],[-np.sin(theta),np.cos(theta),0],[0,0,1]])

def IncludeCost(cost, metric):
    cost_sq = cost**2  # (nt, nx, ny)
    cost_exp = cost_sq[:, :, :, None, None]  # shape: (nt, nx, ny, 1, 1)
    metric_exp = metric[:, None, None, :, :]  # shape: (nt, 1, 1, 3, 3)
    return cost_exp * metric_exp  # broadcast to (nt, nx, ny, 3, 3)

def runReedsSheppGF(sides, dims, seeds, tips, metric, solver_dtype="float32"):
    #print(".")
    #metric = Riemann(cp.asarray(metric))
    metric = Riemann(xp.array(metric))
    #print("..")
    hfmIn = Eikonal.dictIn({
        'model' : 'Riemann3_Periodic',
        'seeds' : seeds,
        'arrayOrdering' : 'RowMajor',
        'tips' : tips,
        #'mode':'gpu',
        'metric' : metric,
        'verbosity': 0})
    #print("...")
    hfmIn.SetRect(sides = sides, dims = dims)
    #if hfmIn.mode=='gpu': 
    #    hfmIn.update({'model':'Riemann3','periodic':(True,False,False)})
    #print("....")
    hfmOut = hfmIn.Run()
    #print(".....")
    geos = [g.T for g in hfmOut['geodesics']]
    print('Done.')
    return geos

def GLIFtoEuclideanOld_vec(nt):
    """
    Vectorized version of GLIFtoEuclideanOld for all t.
    Returns [nt, 3, 3].
    """
    t = np.arange(nt) * 2 * np.pi / nt
    LIF = np.zeros((nt, 3, 3))
    LIF[:, 0, 0] = np.cos(t)
    LIF[:, 0, 1] = np.sin(t)
    LIF[:, 1, 0] = -np.sin(t)
    LIF[:, 1, 1] = np.cos(t)
    LIF[:, 2, 2] = 1.0
    return LIF

def ReedsSheppMetricGFOld(GF, dims, g11, g22, g33):
    """
    Memory-efficient version of ReedsSheppMetricGFOld.
    Assumes GF is unused (typically identity) and omits full spatial tiling.

    Returns:
        (nt, 3, 3) metric tensor to be broadcast later.
    """
    nt, nx, ny = dims[0], dims[1], dims[2]
    LIFtoEuclidean = GLIFtoEuclideanOld_vec(nt)     # (nt, 3, 3)
    LIFtoEuclideaninv = np.linalg.inv(LIFtoEuclidean)

    # Diagonal GF metric
    GFmat = np.diag([g11, g22, g33])

    # Compose per-theta metric: M_t = LIFinv @ GF @ LIFinv^T
    M = LIFtoEuclideaninv @ GFmat @ np.transpose(LIFtoEuclideaninv, (0, 2, 1))  # (nt, 3, 3)

    return M

from time import time
def fast_marching(os_cost,start_point,end_point,g11=1,g22=25,g33=25):
    NxCost = os_cost.shape[1]
    NyCost = os_cost.shape[2]
    NoCost = os_cost.shape[0]
    s_theta = 2*np.pi/NoCost
    gfLIF = np.zeros((NoCost,NxCost,NyCost,3,3))
    gfLIF[:,:,:,0,0] = 1
    gfLIF[:,:,:,1,1] = 1
    gfLIF[:,:,:,2,2] = 1

    dims = np.array([NoCost,NxCost,NyCost])
    sidesLIFmetric = np.array([[0,NxCost],[0,NyCost],[0,2*np.pi - s_theta]])
    print("Initial constructions donne")

    start_time = time()
    metricLIFOld = ReedsSheppMetricGFOld(gfLIF,dims,g11,g22,g33)
    print(f"ReedsSheppMetricGFOld = {time() - start_time}")
    import gc
    gc.collect()
    if CUPY_AVAILABLE:
        mempool = cp.get_default_memory_pool()
        pinned = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned.free_all_blocks()
    
    start_time = time()
    metricLIFinclCostOld = IncludeCost(os_cost**2, metricLIFOld)
    print(f"IncludeCost time = {time() - start_time}")

    metricLIFinclCostOld1 = metricLIFinclCostOld.transpose((3,4,1,2,0))

    a = np.array([0,2*np.pi])-s_theta/2
    b = np.array([0,NxCost])
    c = np.array([0,NyCost])
    sides = np.array([b,c,a])

    seeds = np.array([*start_point[::-1],np.pi/2])
    tips = np.array([*end_point[::-1],np.pi/2])

    metricLIFinclCostOld = np.reshape(metricLIFinclCostOld,(3,3,dims[0],dims[1],dims[2]))

    start_time = time()
    geos1 = runReedsSheppGF(sides, [dims[1],dims[2],dims[0]], [seeds], [tips], metricLIFinclCostOld1)
    #geos1 = runReedsSheppGF(cp.asarray(sides, dtype=cp.float32), cp.asarray([dims[1],dims[2],dims[0]]), [cp.asarray(seeds, dtype=cp.float32)], [cp.asarray(tips, dtype=cp.float32)], cp.asarray(metricLIFinclCostOld1, dtype=cp.float32))
    #geos1 = runReedsSheppGF(cp.asarray(sides), cp.asarray([dims[1],dims[2],dims[0]]), [cp.asarray(seeds, dtype=cp.float64)], [cp.asarray(tips, dtype=cp.float64)], cp.asarray(metricLIFinclCostOld1, dtype=cp.float64))
    print(f"runReedsSheppGF = {time() - start_time}")

    return [geos1[0][:,1],geos1[0][:,0]]'''
    
from agd import Eikonal
from agd.Metrics import AsymQuad, Riemann  # Riemannian metric and Asymmetric Quadratic Models
from agd import AutomaticDifferentiation as ad
from agd import LinearParallel as lp
from agd import FiniteDifferences as fd
from agd import Eikonal

from agd.LinearParallel import outer_self as Outer  # outer product v \v^T of a vector with itself
norm = ad.Optimization.norm
import numpy as np; xp = np

# ---------------------------------------------------------------------
# Safety guard: CUPY_AVAILABLE may or may not exist in this module
# ---------------------------------------------------------------------
try:
    CUPY_AVAILABLE
except NameError:
    CUPY_AVAILABLE = False

# ---------------------------------------------------------------------
# Shared helpers (unchanged math)
# ---------------------------------------------------------------------
def GGF(g11, g22, g33, GFtoLIFinv, LIFtoEuclideaninv):
    GF = np.diag([g11, g22, g33])
    transformMatrix = np.dot(LIFtoEuclideaninv, GFtoLIFinv)
    G = np.dot(transformMatrix, np.dot(GF, transformMatrix.T))
    return G

def GLIFtoEuclideanOld(theta):
    return np.array([
        [ np.cos(theta),  np.sin(theta), 0],
        [-np.sin(theta),  np.cos(theta), 0],
        [ 0            ,  0            , 1]
    ])

def GLIFtoEuclideanOld_vec(nt):
    """
    Vectorized version of GLIFtoEuclideanOld for all t.
    Returns [nt, 3, 3].
    """
    t = np.arange(nt) * 2 * np.pi / nt
    LIF = np.zeros((nt, 3, 3))
    LIF[:, 0, 0] = np.cos(t)
    LIF[:, 0, 1] = np.sin(t)
    LIF[:, 1, 0] = -np.sin(t)
    LIF[:, 1, 1] = np.cos(t)
    LIF[:, 2, 2] = 1.0
    return LIF

# ---------------------------------------------------------------------
# Original (unoptimized) metric builder – for "old_unoptimized"
# ---------------------------------------------------------------------
def ReedsSheppMetricGFOld_naive(GF, dims, g11, g22, g33):
    """
    Original nested-loop implementation.
    Returns metric[t, x, y, i, j] of shape (nt, nx, ny, 3, 3).
    """
    nx = dims[1]
    ny = dims[2]
    nt = dims[0]

    GFinv = GF  # inverse of identity matrix. much faster this way
    LIFtoEuclidean = np.zeros((dims[0], 3, 3))
    for t in range(0, nt):
        LIFtoEuclidean[t, :, :] = GLIFtoEuclideanOld(t * 2 * np.pi / nt)

    LIFtoEuclideaninv = np.array(
        [np.linalg.inv(LIFtoEuclidean[i]) for i in range(LIFtoEuclidean.shape[0])]
    )
    metric = np.zeros((dims[0], dims[1], dims[2], 3, 3))
    for t in range(nt):
        for x in range(nx):
            for y in range(ny):
                metric[t, x, y, :, :] = GGF(
                    g11, g22, g33, GFinv[t, x, y], LIFtoEuclideaninv[t, :, :]
                )
    return metric

# ---------------------------------------------------------------------
# Optimized metric builder – same math, less memory – for "old_optimized"/"new_optimized"
# ---------------------------------------------------------------------
def ReedsSheppMetricGFOld_vec(GF, dims, g11, g22, g33):
    """
    Memory-efficient version of ReedsSheppMetricGFOld.
    Assumes GF is unused (typically identity) and omits full spatial tiling.

    Returns:
        (nt, 3, 3) metric tensor; later broadcast over (x,y).
    """
    nt, nx, ny = dims[0], dims[1], dims[2]
    LIFtoEuclidean = GLIFtoEuclideanOld_vec(nt)   # (nt, 3, 3)
    LIFtoEuclideaninv = np.linalg.inv(LIFtoEuclidean)

    # Diagonal GF metric
    GFmat = np.diag([g11, g22, g33])

    # Compose per-theta metric: M_t = LIFinv @ GF @ LIFinv^T
    M = LIFtoEuclideaninv @ GFmat @ np.transpose(LIFtoEuclideaninv, (0, 2, 1))  # (nt, 3, 3)

    return M

# ---------------------------------------------------------------------
# Original IncludeCost – expansion implementation (old_unoptimized)
# ---------------------------------------------------------------------
def IncludeCost_naive(cost, metric):
    """
    Original cost inclusion:
      - cost is (nt, nx, ny)
      - internally squared again → overall cost^4 if upstream passes os_cost**2
    """
    cost = cost**2
    cost = np.expand_dims(cost, axis=3)
    cost = np.concatenate([cost, cost, cost], axis=3)
    cost = np.expand_dims(cost, axis=4)
    cost = np.concatenate([cost, cost, cost], axis=4)
    metric = metric * cost
    return metric  # shape (nt, nx, ny, 3, 3)

# ---------------------------------------------------------------------
# Optimized IncludeCost – broadcast implementation (old/new optimized)
# float32 + cp-aware
# ---------------------------------------------------------------------
def IncludeCost(cost, metric):
    """
    Broadcast-based IncludeCost (float32, cp-aware)

      - cost is (nt, nx, ny)
      - metric is:
          * (nt, 3, 3)  OR
          * (nt, nx, ny, 3, 3)
      - returns (nt, nx, ny, 3, 3) as float32
    """

    # Force float32 on active backend (CuPy or NumPy fallback)
    cost = cp.asarray(cost, dtype=cp.float32)
    metric = cp.asarray(metric, dtype=cp.float32)

    # cost^2 (still float32)
    cost_sq = cost * cost

    # Expand to (nt, nx, ny, 1, 1)
    cost_exp = cost_sq[..., None, None]

    if metric.ndim == 3:
        # (nt, 3, 3) → broadcast over x,y
        metric_exp = metric[:, None, None, :, :]
    else:
        # already (nt, nx, ny, 3, 3)
        metric_exp = metric

    out = cost_exp * metric_exp

    # Ensure float32 output (avoid accidental upcast)
    return out.astype(cp.float32, copy=False)

'''# ---------------------------------------------------------------------
# Riemann3_Periodic solver wrapper (unchanged except verbosity)
# ---------------------------------------------------------------------
def runReedsSheppGF(sides, dims, seeds, tips, metric, solver_dtype="float32"):

    # right before metric.get()
    import gc
    gc.collect()
    if CUPY_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    
    # If metric is CuPy → move to CPU explicitly
    if CUPY_AVAILABLE and isinstance(metric, cp.ndarray):
        metric = metric.get()  # GPU → CPU copy

    # AGD/HFMIO bindings expect a CPU NumPy float64 metric array.
    metric = np.ascontiguousarray(metric, dtype=np.float64)
    metric = Riemann(metric)

    hfmIn = Eikonal.dictIn({
        'model'        : 'Riemann3_Periodic',
        'seeds'        : seeds,
        'arrayOrdering': 'RowMajor',
        'tips'         : tips,
        'metric'       : metric,
        'verbosity'    : 0,
    })

    hfmIn.SetRect(sides=sides, dims=dims)
    hfmOut = hfmIn.Run()

    geos = [g.T for g in hfmOut['geodesics']]
    print('Done.')
    return geos

# ---------------------------------------------------------------------
# Ablation-ready fast_marching
# ---------------------------------------------------------------------
from time import time

def fast_marching(
    os_cost,
    start_point,
    end_point,
    g11=1,
    g22=25,
    g33=25,
    *,
    mode="new_optimized",       # "old_unoptimized" | "old_optimized" | "new_optimized"
    return_subtiming=False
):
    """
    Reeds–Shepp fast marching for midline tracking (Riemann3_Periodic).

    Modes
    -----
    - "old_unoptimized":
        * EXACT original behavior:
          - ReedsSheppMetricGFOld_naive (nested loops)
          - IncludeCost_naive (concat-based)
          - default anisotropy if g22/g33 not overridden: 100
    - "old_optimized":
        * Same math as original, but:
          - ReedsSheppMetricGFOld_vec (vectorized over theta)
          - IncludeCost (broadcast)
          - default g22=g33 forced to 100 if left at 25
    - "new_optimized" (default):
        * Same math as optimized, but:
          - default g22=g33 = 25 (your "new" behavior)

    Returns
    -------
    - If return_subtiming is False:
        [x_coords, y_coords]
    - If return_subtiming is True:
        ([x_coords, y_coords], subtiming_dict)
    """
    # -----------------------------------
    # Basic setup (same for all modes)
    # -----------------------------------
    t_all0 = time()
    _mem_point("FM entry", os_cost)

    NxCost = os_cost.shape[1]
    NyCost = os_cost.shape[2]
    NoCost = os_cost.shape[0]
    s_theta = 2 * np.pi / NoCost

    gfLIF = np.zeros((NoCost, NxCost, NyCost, 3, 3), dtype=np.float32)
    gfLIF[:, :, :, 0, 0] = 1.0
    gfLIF[:, :, :, 1, 1] = 1.0
    gfLIF[:, :, :, 2, 2] = 1.0
    _mem_point("After gfLIF alloc", gfLIF)
    
    dims = np.array([NoCost, NxCost, NyCost])
    sidesLIFmetric = np.array([[0, NxCost], [0, NyCost], [0, 2 * np.pi - s_theta]])
    print("Initial constructions done")

    # match old defaults for "old_*" modes when user leaves g22,g33 at new defaults
    if mode in ("old_unoptimized", "old_optimized") and g22 == 25 and g33 == 25:
        g22_local = 100
        g33_local = 100
    else:
        g22_local = g22
        g33_local = g33

    # we always pass cost**2 into IncludeCost, as in the original code
    cost_sq_input = os_cost**2
    _mem_point("After cost_sq_input", cost_sq_input)

    # timing placeholders
    t_metric = 0.0
    t_include = 0.0
    t_transpose = 0.0
    t_solver = 0.0

    # -----------------------------------
    # Metric build (per mode)
    # -----------------------------------
    if mode == "old_unoptimized":
        # 1) metric build – original nested implementation
        t0 = time()
        metricLIFOld = ReedsSheppMetricGFOld_naive(gfLIF, dims, g11, g22_local, g33_local)
        t_metric = time() - t0
        print(f"[fast_marching:{mode}] ReedsSheppMetricGFOld_naive = {t_metric:.4f} s")
        _mem_point("After metric build", metricLIFOld)

        # GC / GPU pool cleanup (optional, non-algorithmic)
        import gc
        gc.collect()
        if CUPY_AVAILABLE:
            import cupy as cp
            mempool = cp.get_default_memory_pool()
            pinned = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned.free_all_blocks()

        # 2) Include cost – original expansion
        t0 = time()
        metricLIFinclCostOld = IncludeCost_naive(cost_sq_input, metricLIFOld)
        t_include = time() - t0
        print(f"[fast_marching:{mode}] IncludeCost_naive = {t_include:.4f} s")
        _mem_point("After include cost", metricLIFinclCostOld)

    elif mode == "old_optimized":
        # 1) metric build – vectorized over theta
        t0 = time()
        metric_theta = ReedsSheppMetricGFOld_vec(gfLIF, dims, g11, g22_local, g33_local)  # (nt,3,3)
        t_metric = time() - t0
        print(f"[fast_marching:{mode}] ReedsSheppMetricGFOld_vec = {t_metric:.4f} s")
        _mem_point("After metric build", metric_theta)

        import gc
        gc.collect()
        if CUPY_AVAILABLE:
            import cupy as cp
            mempool = cp.get_default_memory_pool()
            pinned = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned.free_all_blocks()

        # 2) Include cost – broadcast version
        t0 = time()
        metricLIFinclCostOld = IncludeCost(cost_sq_input, metric_theta)  # (nt,nx,ny,3,3)
        t_include = time() - t0
        print(f"[fast_marching:{mode}] IncludeCost(broadcast) = {t_include:.4f} s")
        _mem_point("After include cost", metricLIFinclCostOld)

    elif mode == "new_optimized":
        # Same as old_optimized, but with "new" defaults (g22=g33=25 unless overridden)
        t0 = time()
        metric_theta = ReedsSheppMetricGFOld_vec(gfLIF, dims, g11, g22_local, g33_local)
        t_metric = time() - t0
        print(f"[fast_marching:{mode}] ReedsSheppMetricGFOld_vec = {t_metric:.4f} s")
        _mem_point("After metric build", metric_theta)

        import gc
        gc.collect()
        if CUPY_AVAILABLE:
            import cupy as cp
            mempool = cp.get_default_memory_pool()
            pinned = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned.free_all_blocks()

        t0 = time()
        metricLIFinclCostOld = IncludeCost(cost_sq_input, metric_theta)  # (nt,nx,ny,3,3)
        t_include = time() - t0
        print(f"[fast_marching:{mode}] IncludeCost(broadcast) = {t_include:.4f} s")
        _mem_point("After include cost", metricLIFinclCostOld)

    else:
        raise ValueError(f"Unknown fast_marching mode: {mode}")

    # -----------------------------------
    # Transpose / reshape for AGD API
    # -----------------------------------
    t0 = time()
    metricLIFinclCostOld1 = metricLIFinclCostOld.transpose((3, 4, 1, 2, 0))  # (3,3,Nx,Ny,No)
    t_transpose = time() - t0
    print(f"[fast_marching:{mode}] transpose = {t_transpose:.4f} s")
    _mem_point("After transpose", metricLIFinclCostOld1)

    a = np.array([0, 2 * np.pi]) - s_theta / 2
    b = np.array([0, NxCost])
    c = np.array([0, NyCost])
    sides = np.array([b, c, a])

    seeds = np.array([*start_point[::-1], np.pi / 2])
    tips = np.array([*end_point[::-1], np.pi / 2])

    # The AGD Riemann3_Periodic expects dims in (nx, ny, nt) order
    dims_agd = [dims[1], dims[2], dims[0]]

    # -----------------------------------
    # Run solver
    # -----------------------------------
    _mem_point("Before solver", metricLIFinclCostOld1)
    t0 = time()
    geos1 = runReedsSheppGF(sides, dims_agd, [seeds], [tips], metricLIFinclCostOld1)
    t_solver = time() - t0
    print(f"[fast_marching:{mode}] runReedsSheppGF = {t_solver:.4f} s")
    _mem_point("After solver")

    # -----------------------------------
    # Prepare outputs
    # -----------------------------------
    path = [geos1[0][:, 1], geos1[0][:, 0]]  # [x, y]
    t_total = time() - t_all0

    subtiming = {
        "fm_mode": mode,
        "fm_metric_build_sec": float(t_metric),
        "fm_include_cost_sec": float(t_include),
        "fm_transpose_sec": float(t_transpose),
        "fm_solver_sec": float(t_solver),
        "fm_total_sec": float(t_total),
    }

    if return_subtiming:
        return path, subtiming
    else:
        return path'''
# ---------------------------------------------------------------------
# Riemann3_Periodic solver wrapper (memory-stable)
# ---------------------------------------------------------------------
def runReedsSheppGF(sides, dims, seeds, tips, metric, solver_dtype):
    """
    Expects:
        metric: NumPy/CuPy array, shape (3,3,Nx,Ny,No)
        solver_dtype: "float32" or "float64" (CPU dtype passed to AGD)
    """

    import gc
    gc.collect()

    if solver_dtype not in ("float32", "float64"):
        raise ValueError(f"solver_dtype must be 'float32' or 'float64', got {solver_dtype!r}")

    # If metric is CuPy, transfer once in current dtype; cast on CPU as needed.
    if CUPY_AVAILABLE and hasattr(cp, "ndarray") and isinstance(metric, cp.ndarray):
        metric_np = metric.get(order="C")
    else:
        metric_np = metric

    target_dtype = np.float32 if solver_dtype == "float32" else np.float64
    metric_np = np.ascontiguousarray(metric_np, dtype=target_dtype)
    gc.collect()

    try:
        metric = Riemann(metric_np)

        hfmIn = Eikonal.dictIn({
            'model'        : 'Riemann3_Periodic',
            'seeds'        : seeds,
            'arrayOrdering': 'RowMajor',
            'tips'         : tips,
            'metric'       : metric,
            'verbosity'    : 0,
        })

        hfmIn.SetRect(sides=sides, dims=dims)
        hfmOut = hfmIn.Run()
    except TypeError as e:
        msg = str(e)
        if "set_array()" in msg and "'metric'" in msg:
            shape = getattr(metric_np, "shape", None)
            dtype = getattr(metric_np, "dtype", None)
            cflag = bool(metric_np.flags["C_CONTIGUOUS"]) if hasattr(metric_np, "flags") else None
            raise TypeError(
                f"HFM metric type mismatch for set_array('metric'): "
                f"shape={shape}, dtype={dtype}, C_CONTIGUOUS={cflag}, solver_dtype={solver_dtype}. "
                f"The binding expects numpy.ndarray[numpy.float64]."
            ) from None
        raise

    geos = [g.T for g in hfmOut['geodesics']]
    print("Done.")
    return geos

from time import time
import numpy as np
import gc

def fast_marching(
    os_cost,
    start_point,
    end_point,
    g11=1,
    g22=25,
    g33=25,
    *,
    mode="new_optimized",
    return_subtiming=False,
    solver_dtype="float32",
):
    t_all0 = time()
    _mem_point("FM entry", os_cost)

    # Ensure float32 early (preserve backend if already CuPy)
    if CUPY_AVAILABLE and hasattr(cp, "ndarray") and isinstance(os_cost, cp.ndarray):
        os_cost = os_cost.astype(cp.float32, copy=False)
    else:
        os_cost = np.asarray(os_cost, dtype=np.float32, order="C")
    _mem_point("FM os_cost float32", os_cost)

    NxCost = os_cost.shape[1]
    NyCost = os_cost.shape[2]
    NoCost = os_cost.shape[0]
    s_theta = 2 * np.pi / NoCost

    dims = np.array([NoCost, NxCost, NyCost], dtype=np.int32)
    print("Initial constructions done")

    if mode in ("old_unoptimized", "old_optimized") and g22 == 25 and g33 == 25:
        g22_local = 100
        g33_local = 100
    else:
        g22_local = g22
        g33_local = g33

    t_metric = t_include = t_transpose = t_solver = 0.0

    # -----------------------------------
    # Metric build
    # -----------------------------------

    if mode == "old_unoptimized":
        # Only allocate gfLIF for this mode
        gfLIF = np.zeros((NoCost, NxCost, NyCost, 3, 3), dtype=np.float32)
        gfLIF[:, :, :, 0, 0] = 1.0
        gfLIF[:, :, :, 1, 1] = 1.0
        gfLIF[:, :, :, 2, 2] = 1.0
        _mem_point("After gfLIF alloc", gfLIF)

        cost_sq_input = os_cost * os_cost
        _mem_point("After cost_sq_input", cost_sq_input)

        t0 = time()
        metricLIFOld = ReedsSheppMetricGFOld_naive(gfLIF, dims, g11, g22_local, g33_local)
        t_metric = time() - t0
        _mem_point("After metric build", metricLIFOld)

        del gfLIF
        gc.collect()

        t0 = time()
        metricLIFinclCostOld = IncludeCost_naive(cost_sq_input, metricLIFOld)
        t_include = time() - t0
        _mem_point("After include cost", metricLIFinclCostOld)

        del metricLIFOld, cost_sq_input
        gc.collect()

    else:
        # Optimized modes: NO gfLIF allocation
        cost_sq_input = os_cost * os_cost
        _mem_point("After cost_sq_input", cost_sq_input)

        t0 = time()
        metric_theta = ReedsSheppMetricGFOld_vec(
            None, dims, g11, g22_local, g33_local
        )
        t_metric = time() - t0
        _mem_point("After metric build", metric_theta)

        # Build the large metric tensor on CPU (NumPy) because the solver is CPU-only.
        # This avoids a later fragile CuPy -> NumPy transfer of the full 5D tensor.
        t0 = time()
        if CUPY_AVAILABLE and hasattr(cp, "ndarray"):
            cost_sq_np = cost_sq_input.get() if isinstance(cost_sq_input, cp.ndarray) else np.asarray(cost_sq_input, dtype=np.float32)
            metric_theta_np = metric_theta.get() if isinstance(metric_theta, cp.ndarray) else np.asarray(metric_theta, dtype=np.float32)
        else:
            cost_sq_np = np.asarray(cost_sq_input, dtype=np.float32)
            metric_theta_np = np.asarray(metric_theta, dtype=np.float32)

        cost_sq_np = np.ascontiguousarray(cost_sq_np, dtype=np.float32)
        metric_theta_np = np.ascontiguousarray(metric_theta_np, dtype=np.float32)
        metricLIFinclCostOld = (cost_sq_np[..., None, None] * metric_theta_np[:, None, None, :, :]).astype(np.float32, copy=False)
        t_include = time() - t0
        _mem_point("After include cost", metricLIFinclCostOld)

        del metric_theta, cost_sq_input, cost_sq_np, metric_theta_np
        gc.collect()
        if CUPY_AVAILABLE:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass

    # -----------------------------------
    # Transpose for solver
    # -----------------------------------
    t0 = time()
    metric_view = metricLIFinclCostOld.transpose((3, 4, 1, 2, 0))
    t_transpose = time() - t0
    _mem_point("After transpose (view)", metric_view)

    # Free float32 base tensor immediately
    del metricLIFinclCostOld
    gc.collect()

    # -----------------------------------
    # Build solver args
    # -----------------------------------
    solver_np_dtype = np.float32 if solver_dtype == "float32" else np.float64
    a = np.array([0, 2 * np.pi], dtype=solver_np_dtype) - s_theta / 2
    b = np.array([0, NxCost], dtype=solver_np_dtype)
    c = np.array([0, NyCost], dtype=solver_np_dtype)
    sides = np.array([b, c, a], dtype=solver_np_dtype)

    seeds = np.array([*start_point[::-1], np.pi / 2], dtype=solver_np_dtype)
    tips  = np.array([*end_point[::-1],   np.pi / 2], dtype=solver_np_dtype)
    dims_agd = [int(dims[1]), int(dims[2]), int(dims[0])]

    # -----------------------------------
    # Solver-boundary metric prep
    # - float32: keep metric_view on backend and let runReedsSheppGF transfer once
    # - float64: move to CPU float32 first, then chunk-cast on CPU (avoid VRAM spike)
    # -----------------------------------
    def _is_cupy_array(x):
        return CUPY_AVAILABLE and hasattr(cp, "ndarray") and isinstance(x, cp.ndarray)

    def _free_gpu_pools():
        if CUPY_AVAILABLE:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass

    metric_for_solver = metric_view
    if solver_dtype == "float64":
        _mem_point("Before metric CPU move", metric_view)

        if _is_cupy_array(metric_view):
            metric32_cpu = metric_view.get(order="C")
            del metric_view
            _free_gpu_pools()
        else:
            metric32_cpu = np.ascontiguousarray(metric_view, dtype=np.float32)
            del metric_view

        gc.collect()
        _mem_point("After metric CPU move (float32)", metric32_cpu)

        _mem_point("Before float64 cast", metric32_cpu)
        metric64 = np.empty(metric32_cpu.shape, dtype=np.float64, order="C")
        n_theta = metric32_cpu.shape[-1]
        for t in range(n_theta):
            metric64[..., t] = metric32_cpu[..., t].astype(np.float64, copy=False)

        del metric32_cpu
        gc.collect()

        _mem_point("After float64 cast (CPU)", metric64)
        metric_for_solver = metric64

    # -----------------------------------
    # Run solver
    # -----------------------------------
    _mem_point("Before solver (metric_view)", metric_for_solver)
    t0 = time()
    geos1 = runReedsSheppGF(
        sides, dims_agd, [seeds], [tips], metric_for_solver, solver_dtype=solver_dtype
    )
    t_solver = time() - t0
    print(f"[fast_marching:{mode}] runReedsSheppGF = {t_solver:.4f} s")

    # Free solver metric buffers immediately
    del metric_for_solver
    if 'metric_view' in locals():
        del metric_view
    if 'metric64' in locals():
        del metric64
    gc.collect()

    path = [geos1[0][:, 1], geos1[0][:, 0]]
    t_total = time() - t_all0

    subtiming = {
        "fm_mode": mode,
        "fm_metric_build_sec": float(t_metric),
        "fm_include_cost_sec": float(t_include),
        "fm_transpose_sec": float(t_transpose),
        "fm_solver_sec": float(t_solver),
        "fm_total_sec": float(t_total),
    }

    return (path, subtiming) if return_subtiming else path

def fast_marching_with_fallback(
    os_cost,
    start_point,
    end_point,
    *,
    g11,
    g22,
    g33,
    mode="new_optimized",
    solver_dtype="float32",
    max_downsample_attempts=(1, 2, 4),
):
    """
    Try fast marching at full resolution and retry with spatial downsampling on OOM.
    """
    import gc

    last_exception = None
    np_oom_type = getattr(getattr(np, "core", None), "_exceptions", None)
    np_oom_type = getattr(np_oom_type, "_ArrayMemoryError", ())

    for ds in max_downsample_attempts:
        try:
            print(f"\n=== FAST MARCHING ATTEMPT (downsample={ds}) ===")

            if ds == 1:
                cost_try = os_cost
                sp_try = start_point
                ep_try = end_point
            else:
                cost_try = os_cost[:, ::ds, ::ds]
                sp_try = (start_point[0] / ds, start_point[1] / ds)
                ep_try = (end_point[0] / ds, end_point[1] / ds)

            mem_snapshot(
                f"Before FM attempt ds={ds}",
                arrays=[("cost_try", cost_try)],
                cupy_mod=(cp if CUPY_AVAILABLE else None),
                do_gc=True,
            )

            path = fast_marching(
                cost_try,
                sp_try,
                ep_try,
                g11=g11,
                g22=g22,
                g33=g33,
                mode=mode,
                solver_dtype=solver_dtype,
            )

            if ds != 1:
                path = [np.asarray(path[0]) * ds, np.asarray(path[1]) * ds]

            print(f"Fast marching succeeded at downsample={ds}")
            return path

        except Exception as e:
            last_exception = e
            print(f"\nFast marching failed at downsample={ds}")
            print(f"Exception: {_format_exception_brief(e)}")

            mem_snapshot(
                f"After failure ds={ds}",
                cupy_mod=(cp if CUPY_AVAILABLE else None),
                do_gc=True,
            )

            if CUPY_AVAILABLE:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                except Exception:
                    pass

            gc.collect()

            msg = str(e)
            is_oom = (
                (np_oom_type and isinstance(e, np_oom_type))
                or ("OutOfMemory" in msg)
                or ("Unable to allocate" in msg)
            )
            if not is_oom:
                print("Non-memory exception, not retrying.")
                raise

            print("Retrying with next downsample level...")

    print("All fast marching attempts failed.")
    raise last_exception
