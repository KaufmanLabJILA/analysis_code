import imageio

import numpy as np
import pandas as pd
from numpy import array as arr

import os
import re
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

from skimage import restoration
from skimage.feature import peak_local_max
from skimage import img_as_float

def psf(x, w):
#     return 1/(np.pi*w/2)*np.exp(-2*(x**2)/w)
    return np.exp(-2*(x**2)/w)


def deconvolve(img, w, iters):
    a = round(w)
    x = np.arange(-a+1,a)
    y = np.arange(-a+1,a)
    
    xx, yy = np.meshgrid(x, y)
    psfM = psf(np.sqrt(xx**2+yy**2), w)
    
    img = img - img.min()
    norm = np.max(img)
    img = img/norm
    
    imgRL = restoration.richardson_lucy(img, psfM, iterations=iters)*norm
    return imgRL

def atomVal(img, mask, w = 6, iters = 20):
    return np.sum(mask*deconvolve(img, w, iters))

def getMasks(mimg, fftN = 2000, N = 10, wmask = 3, supersample = None):

    fimg = np.fft.fft2(mimg, s = (fftN,fftN))
    fimg = np.fft.fftshift(fimg)
    fimgAbs = np.abs(fimg)
    fimgArg = np.angle(fimg)

    # fimgMax = ndi.maximum_filter(fimg, size = 100, mode = 'constant')
    fMaxCoord = peak_local_max(fimgAbs, min_distance=100, threshold_rel=.1)
    # fMaxBool = peak_local_max(fimgAbs, min_distance=100, threshold_rel=.1, num_peaks = 4, indices=False)

    fMaxCoord = fMaxCoord[fMaxCoord[:,0]-fftN/2>-fftN/100] # Restrict to positive quadrant
    fMaxCoord = fMaxCoord[fMaxCoord[:,1]-fftN/2>-fftN/100] # Restrict to positive quadrant
    fMaxCoord = fMaxCoord[fMaxCoord.sum(axis=1)-fftN>fftN/100] # Restrict to positive quadrant

#     xsort = np.lexsort((fMaxCoord[:,0]+fMaxCoord[:,1]))
#     ysort = np.lexsort((fMaxCoord[:,1]+fMaxCoord[:,0]))

    xsort = np.argsort(fMaxCoord[:,1]+fMaxCoord[:,0]/2)
    ysort = np.argsort(fMaxCoord[:,0]+fMaxCoord[:,1]/2)
    xpeak, ypeak = fMaxCoord[xsort[0]], fMaxCoord[ysort[0]]

#     print(fMaxCoord)
#     print(xsort)
#     print(ysort)
    
    plt.imshow(fimgAbs)
    plt.plot(fMaxCoord[:,1], fMaxCoord[:,0],'g.')
    # plt.plot([xpeak[0], ypeak[0]],[xpeak[1],ypeak[1]],'r.')
    plt.plot([xpeak[1]],[xpeak[0]],'r.')
    plt.plot([ypeak[1]],[ypeak[0]],'b.')
    # plt.plot(0,1000,'r.')

    plt.show()

    freqs = np.fft.fftfreq(fftN)
    freqs = np.fft.fftshift(freqs)
    fx, fy = freqs[xpeak], freqs[ypeak]
    dx, dy = 1/fx, 1/fy
    # dx = arr([dx[1]/dy[0], dx[1]]) 
    # dy = arr([dy[0],dy[0]/dy[1]]) 
    
    phix, phiy = fimgArg[xpeak[0], xpeak[1]], fimgArg[ypeak[0], ypeak[1]]

    # if phix<0:
    #     phix = -phix
    # if phiy<0:
    #     phiy = -phiy

    normX = np.sqrt(np.sum(fx**2))
    dx = (1/normX)*(fx/normX)

    normY = np.sqrt(np.sum(fy**2))
    dy = (1/normY)*(fy/normY)
    
    dx[1]=-dx[1]
    dy[0]=-dy[0]
    tmp = dy[0]
    dy[0] = dx[1]
    dx[1] = tmp

    if supersample!= None:
        dx, dy = dx/supersample, dy/supersample
        N = (N-1)*supersample+1
    
    ns = np.arange(N)

    px = arr([(dx*ind) for ind in ns])
    py = arr([(dy*ind) for ind in ns])

    pts = arr([(x + py)+[(2*np.pi-phix)/(2*np.pi*normX), (2*np.pi-phiy)/(2*np.pi*normY)] for x in px]).reshape((N**2,2))

    plt.imshow(mimg)
    plt.plot(pts[:,1],pts[:,0],'r.')
    plt.show()
    
    x = np.arange(len(mimg[0]))
    y = np.arange(len(mimg[:,0]))

    xx, yy = np.meshgrid(x, y)

    masks = arr([psf(np.sqrt((xx-pts[i,1])**2+(yy-pts[i,0])**2), wmask) for i in range(len(pts))])
    plt.imshow(np.sum(masks, axis=0))
    plt.plot(pts[:,1],pts[:,0],'r.')
    plt.show()
    
    print(dx, dy)

    return masks