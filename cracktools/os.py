import numpy as np
import scipy
import cmath
import math
import sys
import skimage
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import scipy.signal
from skimage.filters import threshold_otsu
# from skimage import filters
# from skimage import measure
from time import time



def ErfSet(size,No,periodicity):
    """ErfSet retuns a set of 2 D error functions.This function is used to \
        cut the wavelets in two (in the spatial domain)"""
    out = np.zeros((No,size,size))
    for i in range(1,No+1):
        xx = -1
        for x in np.arange( -((size - 1) / 2) - ((size - 1) / 2)%1, (size - 1) / 2 - ((size - 1)/2)%1 +1):
            xx += 1
            yy = -1
            for y in np.arange( -((size - 1) / 2) - ((size - 1) / 2)%1, (size - 1) / 2 - ((size - 1)/2)%1 +1):
                yy += 1
                out[i-1,xx,yy] = 1/2 * (1 + scipy.special.erf(x * np.cos( (periodicity*(-1+i))/ No)\
                                                              + y * np.sin((periodicity*(-1+i))/No)))
    return out

def WindowGauss(size,sigma_s):
    """WindowGauss retuns the spatial Gauss envelope"""
    out = np.zeros((size,size))
    i = -1
    j = -1
    for x in np.arange( (- ((size - 1) / 2) - (size - 1) / 2 % 1), (size - 1) / 2 - (size-1) / 2 % 1 + 1,1):
        i = i+1
        j = -1
        for y in np.arange( (- ((size - 1) / 2) - (size - 1) / 2 % 1), (size - 1) / 2 - (size-1) / 2 % 1 + 1,1):
            j = j+1
            out[i,j] = math.e**( -( x**2 / (2 * sigma_s ** 2 )) -  y**2 / (2 * sigma_s ** 2 ))
            
    return out

def PolarCoordinateGridAngular(size):
    """
    PolarCoordinateGridRadial returns a matrix in which each element \
    gives the corresponding radial coordinate (with the origin in the \
    center of the matrix
    """
    m = np.zeros((size,size))
    centerx = np.ceil((size-1)/2)
    centery = centerx
    for i in range(size):
        for j in range(size):
            dx = i-centerx
            dy = j-centery
            m[i,j] = cmath.phase(complex(dx,dy))
    return m

def PolarCoordinateGridRadial(size):
    """
    PolarCoordinateGridRadial returns a matrix in which each element \
    gives the corresponding radial coordinate (with the origin in the \
    center of the matrix
    """
    m = np.zeros((size,size))
    centerx = np.ceil((size-1)/2)
    centery = centerx
    for i in range(size):
        for j in range(size):
            dx = centerx-i
            dy = centery-j
            m[i,j] = (np.sqrt(dx**2 + dy**2) + sys.float_info.epsilon) / ((size - 1) / 2)
    return m

def MnWindow(size,n,inflectionPoint):
    """MnWindow gives the radial windowing matrix for sampling the fourier \
        domain"""
    
    eps = sys.float_info.epsilon
    po_matrix = eps + 1 / (np.sqrt(2 * inflectionPoint**2 / (1 + 2*n))) * PolarCoordinateGridRadial(size)
    s = np.zeros_like(po_matrix)
    for k in range(n+1):
        s = s + math.e**(-po_matrix**2) * po_matrix**(2*k) / np.math.factorial(k)
    return s

def BSplineMatrixFunc(n,x):
    eps = sys.float_info.epsilon
    r = 0
    for i in np.arange((1 - n - 1) / 2, (n - 1 + 1) / 2+1):
        s = 0
        for k in range(n+2):
            binom_cof = scipy.special.binom(n+1, k)
            sign = np.sign(i + (n + 1) / 2 - k)
            s += binom_cof * (x + (n + 1) / 2 - k) ** (n + 1 - 1) * (-1)**k * sign

        f = 1/(2 * np.math.factorial(n+1-1)) * s
        if i < (n+1-1)/2:
            ic = np.heaviside((x - (i - 1/2 + eps)), 1) * np.heaviside(-(x - (i + 1/2)), 1)
        else :
            ic = np.heaviside((x - (i - 1/2 + eps)), 1) * np.heaviside(-(x - (i + 1/2 - eps)), 1)
        
        r += f*np.round(ic)
    return r

def CakeWaveletStackFourier(size, sPhi, splineOrder, overlapFactor, inflectionPoint, mnOrder, dcStdDev,
                            noSymmetry):
    """CakeWaveletStackFourier constructs the cake wavelets in the Fourier \
        domain (note that windowing in the spatial domain is still required \
        after this"""
    dcWindow = np.ones((size,size)) - WindowGauss(size,dcStdDev)
    mnWindow = MnWindow(size, mnOrder, inflectionPoint)
    angleGrid = PolarCoordinateGridAngular(size)
    sPhiOverlapped = sPhi/overlapFactor
    if noSymmetry == True:
        s = 2*np.pi
    else :
        s = np.pi
    
    out = np.array([], dtype=np.int64).reshape(0,size,size)
    for theta in np.arange(0, s, sPhiOverlapped):
        x = mod_offset(angleGrid - theta - np.pi / 2, 2*np.pi, -np.pi) / sPhi 
        f = dcWindow*mnWindow * BSplineMatrixFunc(splineOrder,x) / overlapFactor
        f = np.expand_dims(f,axis = 0)
        out = np.vstack([out,f])
    
    filters = np.vstack([out,np.expand_dims((1-dcWindow),axis = 0)])
    return filters

def CakeWaveletStack(size, nOrientations, design, inflectionPoint, mnOrder, splineOrder, overlapFactor,
                    dcStdDev, directional):
    
    noSymmetry = nOrientations%2 == 1
    dcSigma = (1/dcStdDev)*(size/(2*np.pi))
    filters = CakeWaveletStackFourier(size, 2*np.pi / nOrientations, splineOrder, overlapFactor,
                                      inflectionPoint, mnOrder, dcSigma, noSymmetry)
#     print(filters.shape)
    cakeF = filters[:-1,:,:]
#     print(cakeF.shape)
    dcFilter = filters[-1,:,:]
    if design == "M":
        cakeF = np.sqrt(cakeF)
        dcFilter = np.sqrt(dcFilter)

    cake = np.zeros_like(cakeF,dtype=np.complex_)
    for i in range(cakeF.shape[0]):
        cakeIF = RotateLeft(cakeF[i,:,:],np.floor(np.array([size,size])/2).astype(int))
       
        ##### ifftn gives result not similar to wolfram (gives conjucate)########
        cakeIF = np.conj(np.fft.ifftn(cakeIF))
        
        cakeIF = RotateRight(cakeIF,np.floor(np.array([size,size])/2).astype(int))
        cake[i,:,:] = cakeIF
        
    if directional:
        if not noSymmetry:
            cake = np.vstack([cake,np.conj(cake)])
        cake = cake*ErfSet(size, (overlapFactor*nOrientations), 2*np.pi)
    else :
        if not noSymmetry:
            cake = np.vstack([cake,np.conj(cake)])
    
    return cake

def mod_offset(arr,divv,offset):
    return arr-(arr-offset)//divv*divv

def RotateLeft(arr,k):
    if type(k) == int or type(k) == float:
        arr1 = arr[:k]
        arr2 = arr[k:]
        arr = np.concatenate((arr2,arr1),axis = 0)
        return arr
    if len(k) == 2 and len(arr.shape) == 2:
        arr1 = arr[:,:k[1]]
        arr2 = arr[:,k[1]:]
        arr = np.concatenate((arr2,arr1),axis = 1)
        arr1 = arr[:k[0],:]
        arr2 = arr[k[0]:,:]
        arr = np.concatenate((arr2,arr1),axis = 0)
        return arr
            
def RotateRight(arr,k):
    if type(k) == int or type(k) == float:
        arr1 = arr[:-k]
        arr2 = arr[-k:]
        arr = np.concatenate((arr2,arr1),axis = 0)
        return arr
    if len(k) == 2 and len(arr.shape) == 2:
        arr1 = arr[:,:-k[1]]
        arr2 = arr[:,-k[1]:]
        arr = np.concatenate((arr2,arr1),axis = 1)
        arr1 = arr[:-k[0],:]
        arr2 = arr[-k[0]:,:]
        arr = np.concatenate((arr2,arr1),axis = 0)
        return arr

def CheckWavelet(window_size = 70,size = 75, nOrientations = 32, design = "N", 
                inflectionPoint = 0.9, mnOrder = 8, splineOrder = 3,
                overlapFactor = 1, dcStdDev = 8, directional = False,display_orientations = 0,mode='real'):
    a = np.zeros((window_size, window_size))
    a[np.int(window_size/2),np.int(window_size/2)] = 1
    os_check = OrientationScoreTransform(a, size = size, nOrientations = nOrientations, design = design, inflectionPoint = inflectionPoint, mnOrder = mnOrder, splineOrder = splineOrder,
                              overlapFactor = overlapFactor, dcStdDev = dcStdDev, directional = directional)
    # for i in display_orientations:
    #     if mode=='real':
    #         plt.imshow(os_check[i,:,:].real)
    #         plt.show()
    #     elif mode=='imag':
    #         plt.imshow(os_check[i,:,:].imag)
    #         plt.show()
    return os_check[display_orientations,:,:].real
           
'''def OrientationScoreTransform(im, size, nOrientations, design = "N", inflectionPoint = 0.8, mnOrder = 8, splineOrder = 3,
                              overlapFactor = 1, dcStdDev = 8, directional = False):
    
    """
    directional     - Determines whenever the filter goes in both directions;
    design          - Indicates which design is used N = Subscript[N, \[Psi]] or M = Subscript[M, \[Psi]]
    inflectionPoint - Is the location of the inflection point as a factor in (positive) radial direction
    splineOrder     - Order of the B - Spline that is used to construct the wavelet
    mnOrder         - The order of the (Taylor expansion) gaussian decay used to construct the wavelet
    dcStdDev        - The standard deviation of the gaussian window (in the Spatial domain) \
                      that removes the center of the pie, to avoid long tails in the spatial domain
    overlapFactor   - How much the cakepieces overlaps in \[Phi] - direction, this can be \
                      seen as subsampling the angular direction
    """
    im = np.pad(im, pad_width=((size, size), (size, size)), mode='symmetric')
    start_time = time()
    cws = CakeWaveletStack(size, nOrientations, design, inflectionPoint, mnOrder, splineOrder, overlapFactor,
                    dcStdDev, directional)
    
    print(f"CakeWaveletStack time: {time() - start_time}")
    start_time = time()
    cwsP = np.pad(cws, ([0,0],[np.floor((im.shape[0]-cws.shape[1])/2).astype(int),
                           np.ceil((im.shape[0]-cws.shape[1])/2).astype(int)],
                    [np.floor((im.shape[1]-cws.shape[2])/2).astype(int),
                     np.ceil((im.shape[1]-cws.shape[2])/2).astype(int)]), mode='constant')
    os = WaveletTransform2D(cp.asarray(im), cp.asarray(cwsP.real))
    print(f"Wavelet Transform 2D time: {time() - start_time}")
    print(os.shape)
    os = os[:,size:-size,size:-size]
    print(im.shape, os.shape)
    return os'''
    
def OrientationScoreTransform(im, size, nOrientations, design="N", inflectionPoint=0.8,
                              mnOrder=8, splineOrder=3, overlapFactor=1,
                              dcStdDev=8, directional=False, top_extra_rows=0,
                              plot_orientations=(0,)):
    """
    Computes orientation score transform with optional top padding and debugging plots.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import cupy as cp
    from time import time

    print("=== OrientationScoreTransform DEBUG ===")
    print(f"Original image shape: {im.shape}")
    pad = size

    # Step 0: Add blank rows at the top
    if top_extra_rows > 0:
        im = np.vstack([
            np.zeros((top_extra_rows, im.shape[1]), dtype=im.dtype),
            im
        ])
        print(f"Added {top_extra_rows} throwaway rows → New shape: {im.shape}")

    # Step 1: Pad image symmetrically
    im_padded = np.pad(im, ((pad, pad), (pad, pad)), mode='symmetric')
    print(f"Padded image shape: {im_padded.shape}")

    # Step 2: Build Cake wavelets
    t0 = time()
    cws = CakeWaveletStack(size, nOrientations, design, inflectionPoint,
                           mnOrder, splineOrder, overlapFactor, dcStdDev, directional)
    print(f"CakeWaveletStack shape: {cws.shape} — time: {time() - t0:.2f}s")

    # Step 3: Pad wavelets to match image
    pad_y = (im_padded.shape[0] - cws.shape[1]) // 2
    pad_x = (im_padded.shape[1] - cws.shape[2]) // 2
    pad_y_extra = im_padded.shape[0] - cws.shape[1] - pad_y
    pad_x_extra = im_padded.shape[1] - cws.shape[2] - pad_x

    cwsP = np.pad(cws, ((0, 0), (pad_y, pad_y_extra), (pad_x, pad_x_extra)), mode='constant')

    # Step 4: Wavelet Transform
    t1 = time()
    os = WaveletTransform2D(cp.asarray(im_padded), cp.asarray(cwsP.real))
    os = cp.asnumpy(os)
    print(f"WaveletTransform2D done in {time() - t1:.2f}s → shape: {os.shape}")

    # Step 5: Crop away padding
    os_cropped = os[:, pad:-pad, pad:-pad]
    print(f"Cropped to original (with throwaway): {os_cropped.shape}")

    # Step 6: Remove the extra top rows
    if top_extra_rows > 0:
        os_cropped = os_cropped[:, top_extra_rows:, :]
        print(f"Removed top {top_extra_rows} throwaway rows → Final shape: {os_cropped.shape}")

    # Step 7: Optional visualization
    '''if plot_orientations:
        n = len(plot_orientations)
        plt.figure(figsize=(4 * n, 4))
        for idx, ori in enumerate(plot_orientations):
            plt.subplot(1, n, idx + 1)
            plt.imshow(os_cropped[ori].real, cmap='gray')
            plt.title(f'Orientation {ori}')
            plt.axis('off')
        plt.tight_layout()
        plt.show()'''

    print("=======================================")
    return os_cropped

def WaveletTransform2D(im, kernels):
    # im: shape (H, W)
    # kernels: shape (N, H, W)
    N = kernels.shape[0]
    H, W = im.shape
    imf = np.fft.fftn(im)                     # (H, W)
    kernelsf = np.fft.fftn(kernels, axes=(1,2)) # (N, H, W)
    # Broadcasting: kernelsf * imf[None, :, :]
    result_f = kernelsf * imf[None, :, :]       # (N, H, W)
    result = np.fft.ifftn(result_f, axes=(1,2)) # (N, H, W)

    # Apply RotateRight to each orientation in batch
    k = np.ceil(0.1 + np.array(im.shape) / 2).astype(int)
    result = np.roll(result, shift=-k[0], axis=1)
    result = np.roll(result, shift=-k[1], axis=2)
    return result

def Rescale(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def IntegerDigits(num):
    x = [int(a) for a in str(num)]
    return x

class ObjPositionOrientationData:
    def __init__(self,Data,Symmetry,Wavelets = None,InputData = None,
                 DcFilterImage = 0):
        self.Data = Data
        self.Symmetry = Symmetry
        self.Wavelets = Wavelets
        self.InputData = InputData
        self.DcFilterImage = DcFilterImage
        self.AngularResolution = Symmetry/Data.shape[0]
        self.FullOrientationList = np.arange(0,Symmetry,self.AngularResolution)

def LeftInvariantDerivative(osObj,sigmaSpatial,sigmaOD,order,symmetry,anglesMatrix):
        n = np.sum(np.array(order) == 1) + np.sum(np.array(order) == 2)+1
        scaledSigmaSpatial = 1/np.sqrt(n)*sigmaSpatial
        angularOrder = np.sum(np.array(order) == 3)
        der = OrientationDerivative(osObj.Data, scaledSigmaSpatial, sigmaOD, osObj.AngularResolution,
                                    symmetry, angularOrder)
        for dirr in order:
            if dirr == 3:
                continue
            if symmetry == np.pi:
                symmetry1 = -np.pi
            elif symmetry == -np.pi:
                symmetry1 = np.pi
            elif symmetry == 2*np.pi:
                symmetry1 = 2*np.pi
            
            der = SpatialDerivative(der, scaledSigmaSpatial, 0, dirr,anglesMatrix, symmetry1)
            
            
        return der

'''def OrientationScoreTensor3(osObj, sigmaSpatial, sigmaOrientation, method):
    # Only 2 cases: LIF (most used) or all components
    if method == "LIF":
        order_list = [11, 22]
    else:
        order_list = [11, 21, 31, 12, 22, 32, 13, 23, 33]

    sigmaOD = sigmaOrientation / osObj.AngularResolution
    shape = osObj.Data.shape + (3, 3)
    tensor = np.zeros(shape, dtype=osObj.Data.dtype)

    # Vectorized: orientations axis
    angles = np.arange(0, abs(osObj.Symmetry), osObj.AngularResolution)
    anglesMatrix = angles[:, None, None]  # (No, 1, 1), will broadcast

    symmetry = osObj.Symmetry

    # Helper for all derivatives, batched
    def get_derivative(order):
        # order: e.g. [1,1] (for dx-dx), [2,1] (dy-dx), [3,1] (dtheta-dx), etc
        # Map to string for caching?
        n = np.sum(np.array(order) == 1) + np.sum(np.array(order) == 2) + 1
        scaledSigmaSpatial = 1 / np.sqrt(n) * sigmaSpatial
        angularOrder = np.sum(np.array(order) == 3)

        # --- Orientation Derivative (for all orientations at once) ---
        periodicOS = CreatePeriodicOrientationAxes(osObj.Data, symmetry)
        sigma = scaledSigmaSpatial
        trunc = (4 * scaledSigmaSpatial + 1) / sigma
        # Gaussian blur, spatial, but NOT along orientation axis
        spatialBlurredOs = skimage.filters.gaussian(periodicOS, sigma=[0, sigma, sigma], truncate=trunc, preserve_range=True)
        # Gaussian derivative along orientation axis (batched)
        derivative = scipy.ndimage.gaussian_filter(
            spatialBlurredOs, [sigmaOD, 0.125, 0.125], order=[angularOrder, 0, 0], mode="wrap"
        )
        # Only keep first No slices if duplicated for symmetry
        if symmetry == np.pi or symmetry == -np.pi:
            derivative = derivative[0:osObj.Data.shape[0], :, :]
        derivative = derivative / (osObj.AngularResolution ** angularOrder)

        # --- Spatial Derivative(s) (for all orientations at once) ---
        for dirr in order:
            if dirr == 3:
                continue  # orientation derivative done above
            if symmetry == np.pi:
                symmetry1 = -np.pi
            elif symmetry == -np.pi:
                symmetry1 = np.pi
            elif symmetry == 2 * np.pi:
                symmetry1 = 2 * np.pi
            else:
                symmetry1 = symmetry

            # "periodicOS" for spatial, now it's orientation-batched already!
            orientationBlurredOS = CreatePeriodicOrientationAxes(derivative, symmetry1)
            trunc2 = (4 * scaledSigmaSpatial + 1) / scaledSigmaSpatial
            # Only first No orientations needed
            if symmetry1 == np.pi or symmetry1 == -np.pi:
                orientationBlurredOS = orientationBlurredOS[0:osObj.Data.shape[0], :, :]
            # dx, dy are both batched for all orientations
            dx = norm_gaussian_filter(orientationBlurredOS, [0, scaledSigmaSpatial, scaledSigmaSpatial],
                                      order=[0, 1, 0], truncate=trunc2, mode="nearest")
            dy = norm_gaussian_filter(orientationBlurredOS, [0, scaledSigmaSpatial, scaledSigmaSpatial],
                                      order=[0, 0, 1], truncate=trunc2, mode="nearest")
            # orientation-batched angles
            cos_theta = np.cos(anglesMatrix)
            sin_theta = np.sin(anglesMatrix)
            if dirr == 1:
                derivative = dx * cos_theta + dy * sin_theta
            elif dirr == 2:
                derivative = -dx * sin_theta + dy * cos_theta

        return derivative

    # Vectorized, no unnecessary axis shuffling
    for order in order_list:
        # order as [i, j] (two digits)
        order_digits = IntegerDigits(order)
        der = get_derivative(order_digits[::-1])
        tensor[..., order_digits[1] - 1, order_digits[0] - 1] = der

    return tensor'''

import cupy as cp
import cupyx.scipy.ndimage as nd
#from cupyx import fuse

@cp.fuse()
def rotate_directional(dx, dy, cos_theta, sin_theta, dirr):
    return cp.where(
        dirr == 1,
        dx * cos_theta + dy * sin_theta,
        -dx * sin_theta + dy * cos_theta
    )

def OrientationScoreTensor3_gpu(osObj, sigmaSpatial, sigmaOrientation, method):
    if method == "LIF":
        order_list = [11, 22]
    else:
        order_list = [11, 21, 31, 12, 22, 32, 13, 23, 33]

    sigmaOD = sigmaOrientation / osObj.AngularResolution
    shape = osObj.Data.shape + (3, 3)
    tensor = cp.zeros(shape, dtype=cp.float32)

    symmetry = osObj.Symmetry
    No = osObj.Data.shape[0]
    angles = cp.arange(0, abs(symmetry), osObj.AngularResolution)
    anglesMatrix = angles[:, None, None]
    cos_theta = cp.cos(anglesMatrix)
    sin_theta = cp.sin(anglesMatrix)

    # Handle symmetry
    periodicOS = cp.array(osObj.Data, dtype=cp.float32)
    if symmetry == -cp.pi:
        periodicOS = cp.concatenate([periodicOS, cp.conj(periodicOS)], axis=0)

    # Cache dictionary
    cache = {}

    def get_spatial_blur(data, sigma):
        key = f"spatial_{sigma:.4f}"
        if key not in cache:
            cache[key] = nd.gaussian_filter(data, sigma=[0, sigma, sigma],
                                            truncate=(4 * sigma + 1) / sigma, mode="nearest")
        return cache[key]

    def get_orientation_blur(data, sigmaOD, angularOrder):
        key = f"orient_{sigmaOD:.4f}_{angularOrder}"
        if key not in cache:
            cache[key] = nd.gaussian_filter(data, sigma=[sigmaOD, 0.125, 0.125],
                                            order=[angularOrder, 0, 0], mode="wrap", truncate=4)
        return cache[key]

    for order in order_list:
        order_digits = [int(x) for x in str(order)][::-1]
        n = cp.sum(cp.array(order_digits) == 1) + cp.sum(cp.array(order_digits) == 2) + 1
        scaledSigmaSpatial = 1.0 / cp.sqrt(n) * sigmaSpatial
        angularOrder = cp.sum(cp.array(order_digits) == 3)

        # --- Filtering ---
        spatialBlurred = get_spatial_blur(periodicOS, float(scaledSigmaSpatial))
        orientBlurred = get_orientation_blur(spatialBlurred, float(sigmaOD), int(angularOrder))

        if symmetry in [cp.pi, -cp.pi]:
            orientBlurred = orientBlurred[:No]

        derivative = orientBlurred / (osObj.AngularResolution ** angularOrder)

        for dirr in order_digits:
            if dirr == 3:
                continue

            if symmetry in [cp.pi, -cp.pi]:
                orientationBlurredOS = derivative[:No]
            else:
                orientationBlurredOS = derivative

            trunc2 = (4 * scaledSigmaSpatial + 1) / scaledSigmaSpatial

            dx = nd.gaussian_filter(orientationBlurredOS,
                                    sigma=[0, scaledSigmaSpatial, scaledSigmaSpatial],
                                    order=[0, 1, 0], truncate=trunc2, mode="nearest")

            dy = nd.gaussian_filter(orientationBlurredOS,
                                    sigma=[0, scaledSigmaSpatial, scaledSigmaSpatial],
                                    order=[0, 0, 1], truncate=trunc2, mode="nearest")

            derivative = rotate_directional(dx, dy, cos_theta, sin_theta, dirr)

        # Fill output tensor
        tensor[..., order_digits[1] - 1, order_digits[0] - 1] = derivative

    return tensor

def CreatePeriodicOrientationAxes(os,symmetry):
    if symmetry == np.pi or symmetry == 2*np.pi:
        return os
    elif symmetry == -np.pi:
        return np.concatenate([os,np.conj(os)],axis = 0)

def OrientationDerivative(derivativesIn, scaledSigmaSpatial, sigmaOriantation, angularResolution, symmetry, order):
    periodicOS = CreatePeriodicOrientationAxes(derivativesIn, symmetry)
    sigma = scaledSigmaSpatial
    trunc = (4*scaledSigmaSpatial + 1)/sigma
    spatialBlurredOs = skimage.filters.gaussian(periodicOS,sigma = [0,sigma,sigma],truncate = trunc)
    derivative = scipy.ndimage.gaussian_filter(spatialBlurredOs, [sigmaOriantation,0.125,0.125], order=[order,0,0], mode="wrap")
    if symmetry == np.pi or symmetry == -np.pi:
        derivative = derivative[0:derivativesIn.shape[0],:,:]
    derivative = derivative / (angularResolution**order)
    
    return derivative

def SpatialDerivative(derivativesIn, scaledSigmaSpatial, scaledSigmaOrientation, dirr,angles, symmetry):
    periodicOS = CreatePeriodicOrientationAxes(derivativesIn, symmetry)
    orientationBlurredOS = periodicOS
    trunc = (4*scaledSigmaSpatial + 1)/scaledSigmaSpatial
    if symmetry == np.pi or symmetry == -np.pi:
        orientationBlurredOS = orientationBlurredOS[0:derivativesIn.shape[0],:,:]
    dx = norm_gaussian_filter(orientationBlurredOS, [0,scaledSigmaSpatial,scaledSigmaSpatial],
                                       order=[0,1,0],truncate=trunc, mode="nearest")
    dy = norm_gaussian_filter(orientationBlurredOS, [0,scaledSigmaSpatial,scaledSigmaSpatial],
                                       order=[0,0,1],truncate=trunc, mode="nearest")
    if dirr == 1:
        derivative = dx*np.cos(angles) + dy*np.sin(angles)
    elif dirr == 2:
        derivative = -dx*np.sin(angles) + dy*np.cos(angles)
    return derivative

def norm_gaussian_filter(data,sigma,order,truncate = 4,mode = "wrap"):
#     print(sigma[1])
#     if sigma[1]<0.5:
    zeros = np.zeros_like(data)
    ind = [int(i/2) for i in data.shape]
    zeros[tuple(ind)] = 1
    weights = scipy.ndimage.gaussian_filter(zeros, sigma,
                                        order=order,truncate=truncate, mode=mode)

    r = scipy.ndimage.gaussian_filter(data, sigma,
                                        order=order,truncate=truncate, mode=mode)
    r = r/np.sum(abs(weights))
#     else :
#         r = scipy.ndimage.gaussian_filter(data, sigma,
#                                     order=order,truncate=truncate, mode=mode)
    return r

def CostFunctionVesselnessFiltering(U,ksi,zeta,sigma_s, method,sigmas_ext = 0, sigmaa_ext = 0):
    Nx = U.shape[1]
    Ny = U.shape[2]
    No = U.shape[0]
    betha = 0.75/sigma_s
    sigma1 = 0.5
    from time import time
    start_time = time()
    obj = ObjPositionOrientationData(U,2*np.pi,Wavelets = None,InputData = None,DcFilterImage = 0)
    #print(f"ObjPositionOrientationData time: {time() - start_time}")
    start_time = time()
    H = OrientationScoreTensor3_gpu(obj,0.5*sigma_s**2, 0.5*(2*betha*sigma_s)**2,method)
    #print(f"OrientationScoreTensor3 time: {time() - start_time}")

    M = cp.diag([1/ksi,zeta/ksi,1])

    if sigmas_ext!=0 or sigmaa_ext !=  0:
        start_time = time()
        H = ExternalRegularization(H,obj.FullOrientationList,sigmas_ext,sigmaa_ext)
        #print(f"ExternalRegularization time: {time() - start_time}")
    a = cp.ones((3,3))
    b = cp.dot(M,cp.dot(a,M))
    #Hess = cp.zeros((No,Nx,Ny,3,3))
    #Hess_old = Hess.copy()
    start_time = time()
    #Hess = cp.broadcast_to(b, (No, Nx, Ny, 3, 3)).copy()
    #print(np.count_nonzero(Hess != Hess_old))
    Hess = cp.empty((No, Nx, Ny, 3, 3), dtype=cp.float32)
    Hess[:] = b  # This uses broadcasting internally, but is more memory-friendly than .broadcast_to().copy()


    Hess = Hess*H
    #print(f"Hess time: {time() - start_time}")
    start_time = time()

    @cp.fuse()
    def fused_vesselness(lambda1, c, Q, sigma1, sigma2):
        S = lambda1**2 + c**2
        R = lambda1 / (c + 1e-6)  # avoids division by 0
        cost = cp.exp(-R**2 / (2 * sigma1**2)) * (1 - cp.exp(-S / (2 * sigma2)))
        return cost * (1 - cp.heaviside(-Q, 0))

    lambda1 = Hess[:,:,:,0,0]
    c = Hess[:,:,:,1,1]
    Q = c

    S_tmp = lambda1**2 + c**2
    sigma2 = 0.2 * cp.max(cp.abs(S_tmp))  # still outside the fused block

    cost = fused_vesselness(lambda1, c, Q, sigma1, sigma2)
    #print(f"Remaining costfunctionvesselnessfiltering: {time() - start_time}")
    
    return cost.get()

def CostFunction(oc,lambdaa, p):
    cost = 1/(1 + lambdaa*(oc)**p)
    return cost

'''def MultiScaleVesselness(U,ksi,zeta,sigmas_s,method,sigmas_ext = 0, sigmaa_ext = 0):
    """Ërosion gives not the same results!!!"""
    vesselnessfilter = []
    for sigma in sigmas_s:
        print(sigma)
        start_time = time()
        vesselness = CostFunctionVesselnessFiltering(U,ksi,zeta,sigma, method,sigmaa_ext = sigmaa_ext)
        print(f"CostFunctionVesselnessFiltering time: {time() - start_time}")
        import cupy as cp, gc
        gc.collect()
        mempool = cp.get_default_memory_pool()
        pinned = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned.free_all_blocks()
        start_time = time()
        pad = 5
        vesselness_pad = np.pad(vesselness,pad,mode = 'wrap')

        vesselnessErosion = scipy.ndimage.morphology.grey_erosion(vesselness_pad,size=(3,0,0))
        vesselnessErosion =  vesselnessErosion[pad:-pad,pad:-pad,pad:-pad]
        vesselnessErosion[:,:5,:] = 0
        vesselnessfilter.append(vesselnessErosion)
        print(f"vesselness remaining time: {time() - start_time}")
        
    return (vesselnessfilter)'''

def MultiScaleVesselness(U, ksi, zeta, sigmas_s, method, sigmas_ext=0, sigmaa_ext=0):
    """Applies vesselness filtering at multiple scales and shows debug plots."""
    import cupy as cp, gc
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.ndimage

    vesselnessfilter = []
    for idx, sigma in enumerate(sigmas_s):
        print(f"\n=== Vesselness for sigma={sigma} ===")
        start_time = time()
        vesselness = CostFunctionVesselnessFiltering(U, ksi, zeta, sigma, method, sigmaa_ext=sigmaa_ext)
        print(f"Filtering time: {time() - start_time:.2f}s")

        # --- Check before erosion ---
        minval = np.min(vesselness)
        maxval = np.max(vesselness)
        nz_count = np.count_nonzero(vesselness)
        print(f"Before erosion: min={minval:.4f}, max={maxval:.4f}, nonzero count={nz_count}")

        # --- Plot before erosion ---
        plt.figure(figsize=(6, 3))
        plt.imshow(np.max(vesselness, axis=0), cmap='hot')
        plt.title(f"Before erosion — σ={sigma}, max={maxval:.4f}")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

        # --- Erosion ---
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

        pad = 5
        vesselness_pad = np.pad(vesselness, pad, mode='wrap')
        vesselnessErosion = scipy.ndimage.grey_erosion(vesselness_pad, size=(3, 0, 0))
        vesselnessErosion = vesselnessErosion[pad:-pad, pad:-pad, pad:-pad]
        #vesselnessErosion[:, :5, :] = 0

        # --- Check after erosion ---
        minval2 = np.min(vesselnessErosion)
        maxval2 = np.max(vesselnessErosion)
        nz_count2 = np.count_nonzero(vesselnessErosion)
        print(f"After erosion:  min={minval2:.4f}, max={maxval2:.4f}, nonzero count={nz_count2}")

        # --- Plot after erosion ---
        plt.figure(figsize=(6, 3))
        plt.imshow(np.max(vesselnessErosion, axis=0), cmap='hot')
        plt.title(f"After erosion — σ={sigma}, max={maxval2:.4f}")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

        vesselnessfilter.append(vesselnessErosion)

    return vesselnessfilter

def MultiScaleVesselnessFilter(vesselnessfilters):
    sum1 = np.sum(vesselnessfilters, axis=0)  # shape (H, W)
    mu = np.max(sum1)
    cost = sum1 / mu
    return cost

def LeftInvariantFrame(theta):
    return np.array([[np.cos(theta),np.sin(theta),0],[-np.sin(theta),np.cos(theta),0],[0,0,1]])

def FromLeftInvariantFrame(orientationList,tensor):
    out = np.zeros_like(tensor)
    leftInvariantFrame = []
    for o in range(tensor.shape[0]):
        rot = LeftInvariantFrame(orientationList[o])
        for i in range(tensor.shape[1]):
            for j in range(tensor.shape[2]):
                ten = tensor[o,i,j,:,:]
                out[o,i,j,:,:] = np.dot(rot.T,np.dot(ten,rot))
    return out

def ToLeftInvariantFrame(orientationList,tensor):
    out = np.zeros_like(tensor)
    leftInvariantFrame = []
    for o in range(tensor.shape[0]):
        rot = LeftInvariantFrame(orientationList[o])
        for i in range(tensor.shape[1]):
            for j in range(tensor.shape[2]):
                ten = tensor[o,i,j,:,:]
                out[o,i,j,:,:] = np.dot(rot,np.dot(ten,rot.T))
    return out

def ExternalRegularization(tensor,orientations,sigmaSpatialExternal,sigmaAngularExternal):
    oriantations1 = np.sort(orientations)
    oriantations1 = oriantations1[1:] - oriantations1[:-1]
    sigmaAngularExternal = sigmaAngularExternal/np.mean(oriantations1)

    tensor = FromLeftInvariantFrame(orientations, tensor)
    tensor = norm_gaussian_filter(tensor,sigma = [sigmaAngularExternal,sigmaSpatialExternal,sigmaSpatialExternal,0,0],
                                  order = 0,mode='nearest')
    tensor = ToLeftInvariantFrame(orientations, tensor)
    return tensor