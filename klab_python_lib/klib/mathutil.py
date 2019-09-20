# Math tools for klablib

import imageio
import scipy.stats
import scipy.special

import numpy as np
import pandas as pd
from numpy import array as arr

import os
import re
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astropy.io import fits

from scipy.stats import sem 
from scipy import ndimage as ndi
from scipy.optimize import curve_fit

###########################

def radial_profile(data, center):
    """Returns radial average of matrix about user-defined center."""
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

def gaussian(x, x0, a, waist, y0):
    """Math: 1D Gaussian function, optics definition. params [x0, amp, waist, y0]"""
    return a*np.exp(-(2*(x-x0)**2)/(waist**2))+y0

def lorentz(x, x0, a, sig, y0):
    """Math: 1D Lorentz function"""
    return a*(sig/2)/((x-x0)**2+(sig/2)**2)+y0

def lor(x, a, k, x0):
    return np.abs(a)/(1+((x-x0)/(k/2))**2)

def twolor(x, a0, a1, k0, k1, x0, x1, y0):
    return y0 + lor(x, a0, k0, x0) + lor(x, a1, k1, x1)

def triplor(x, a0, a1, a2, kc, ks, x0, dx, y0):
    """Fit carrier and sidebands. Params: [a0, a1, a2, sigma_carrier, sigma_side, f_carrier, df_sidebands, y0]"""
    return y0+lor(x, a0, ks, x0-dx)+lor(x, a1, kc, x0) + lor(x, a2, ks, x0+dx)

def triplor2(x, a0, a1, a2, kc, ks, x0, dx1, dx2, y0):
    """Fit carrier and sidebands. Params: [a0, a1, a2, sigma_carrier, sigma_side, f_carrier, df_sidebands, y0]"""
    return y0+lor(x, a0, ks, x0-dx1)+lor(x, a1, kc, x0) + lor(x, a2, ks, x0+dx2)

def fivelor(x, a0, a1, a2, a3, a4, kc, ks, kss, x0, dx, dxx, y0):
    return y0 + lor(x, a0, kss, x0-dxx)+ lor(x, a1, ks, x0-dx)+lor(x, a2, kc, x0) + lor(x, a3, ks, x0+dx) +lor(x, a4, kss, x0+dxx)

def expfit(t, A, tau):
    return A*np.exp(-t/tau)

def cos(t, f, A, phi, y0):
    return A*np.cos(2*np.pi*f*t+phi) + y0

def cosFit(keyVals, dat, n = 0, ic = False):
    """Cosine fit. Parameter order: [A, f, phi, y0]"""
    """Can select nth color from default matplotlib color cycle."""
    xdat=arr(keyVals)
    ydat=arr(dat)
    
    if not ic:
        ymin = np.min(ydat)
        ymax = np.max(ydat)

        A = (ymax-ymin)/2
        f = .1/(xdat[1]-xdat[0])
        phi = 0
        y0 = np.mean(ydat)
    
        guess = [f, A, phi, y0]
    else:
        guess = ic
        
    print(guess)
    params, uncert = curve_fit(cos, xdat, ydat, p0=guess, maxfev=999999)
    
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.plot(xdat, cos(xdat, *params), "-", color=cycle[n])
    perr = np.sqrt(np.diag(uncert))
    return params, perr

def cosFitF(keyVals, dat, f, n = 0, ic = False):
    """Cosine fit. Parameter order: [f, A, phi, y0]"""
    """Can select nth color from default matplotlib color cycle."""
    xdat=arr(keyVals)
    ydat=arr(dat)
    
    if not ic:
        ymin = np.min(ydat)
        ymax = np.max(ydat)

        A = (ymax-ymin)/2
#         f = 1/(xdat[1]-xdat[0])
        phi = 0
        y0 = np.mean(ydat)
    
        guess = [A, phi, y0]
    else:
        guess = ic
        
#     print(guess)
    params, uncert = curve_fit(lambda x, amp, phase, y: cos(x, f, amp, phase, y), xdat, ydat, p0=guess, bounds = ([0, -np.pi, 0], [100, np.pi, 100]), maxfev = 9999)
    
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.plot(xdat, cos(xdat, f, *params), "-", color=cycle[n])
    perr = np.sqrt(np.diag(uncert))
    return params, perr

def dampedCos(t, A, tau, f, phi, y0):
    return A*np.exp(-t/tau)/2 * (np.cos(2*np.pi*f*t+phi)) + y0

def dampedCosFit(keyVals, dat, n = 0, ic = False):
    """Damped cosine fit. Parameter order: [A, tau, f, phi, y0]"""
    """Can select nth color from default matplotlib color cycle."""
    xdat=arr(keyVals)
    ydat=arr(dat)
    
    if not ic:
        ymin = np.min(ydat)
        ymax = np.max(ydat)

        A = ymax-ymin
        tau = np.mean(xdat)
        f = 1/(xdat[1]-xdat[0])
        phi = 0
        y0 = np.mean(ydat)
    
        guess = [A, tau, f, phi, y0]
    else:
        guess = ic
        
    print(guess)
    params, uncert = curve_fit(dampedCos, xdat, ydat, p0=guess, maxfev=999999)
    
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.plot(xdat, dampedCos(xdat, *params), "-", color=cycle[n])
    perr = np.sqrt(np.diag(uncert))
    return params, perr


def gausfit(keyVals, dat, y_offset=False, negative=False, n=0, guess = []):
    """1D gaussian fit, with or without Y-offset, and with positive or negative amplitude. Can select nth color from default matplotlib color cycle. returns gausparams [x0, a, waist, y0], perr"""
    xdat=keyVals
    ydat=arr(dat)
    if negative:
        i=ydat.argmin()
        a = -abs(ydat[i]-max(ydat))
        ihalf=np.argmin(np.abs(ydat-np.min(ydat)-(np.max(ydat)-np.min(ydat))/2)) #find position of half-maximum

    else:
        i=ydat.argmax()
        a = ydat[i]
        ihalf=np.argmin(np.abs(ydat-(np.max(ydat)-np.min(ydat))/2)) #find position of half-maximum

    x0 = xdat[i]
    y0 = ydat[0]
    #sig = np.abs(xdat[i]-xdat[ihalf])
    sig = (keyVals[-1]-keyVals[0])/5
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
#     plt.plot(xdat, ydat,".", color=cycle[n])

    if y_offset:
        if len(guess) == 0:
            guess = [x0, a, sig, y0]
#         print(guess)
        gauss_params, gauss_uncert = curve_fit(gaussian, xdat, ydat, p0=guess, maxfev=10000)
#         plt.plot(xdat, gaussian(xdat, *gauss_params), "-", color=cycle[n])
    else:
        if len(guess) == 0:
            guess = [x0, a, sig]
        #print(guess)
        gauss_params, gauss_uncert = curve_fit(lambda x, x0, a, sig: gaussian(x, x0, a, sig, 0), xdat, ydat, p0=guess, maxfev=100000)
#         plt.plot(xdat, gaussian(xdat, *gauss_params, 0), "-", color=cycle[n])

    perr = np.sqrt(np.diag(gauss_uncert))

    # plt.plot(xdat, lorentz(xdat, *lorentz_params), "r")
#     plt.xlabel('Modulation freq (MHz)')
#     plt.ylabel('ROI sum (arb)')
#     plt.show()

#     df=pd.DataFrame([gauss_params])
#     df.columns=['x0','a','sig']
#     df

    return gauss_params, perr

def gauss2d(xy, amp, x0, y0, theta, sig_x, sig_y):
    """Math: 2D Gaussian"""
    x, y =xy

    a = np.cos(theta)**2/(2*sig_x**2) + np.sin(theta)**2/(2*sig_y**2);
    b = -np.sin(2*theta)/(4*sig_x**2) + np.sin(2*theta)/(4*sig_y**2);
    c = np.sin(theta)**2/(2*sig_x**2) + np.cos(theta)**2/(2*sig_y**2);

    return amp * np.exp(-(a * (x - x0)**2 + 2 * b * (x - x0)**2 * (y - y0)**2 + c * (y - y0)**2))

def gaussFit2d(datc):
    """2D Gaussian fit to image matrix. params [amp, x0, y0, theta, w_x, w_y]"""
    rows=range(datc.shape[1])
    cols=range(datc.shape[0])
    x,y=np.meshgrid(rows,cols)
    x,y=x.flatten(),y.flatten()
    xy=[x,y]
    datf=datc.flatten()

    i = datf.argmax()
    ihalf=np.argmin(np.abs(datf-datf[i]/2)) #find position of half-maximum
    sig_x_guess = np.abs(x[i]-x[ihalf])+1
    sig_y_guess = np.abs(y[i]-y[ihalf])+1
    print(sig_x_guess,sig_y_guess)
    guess = [datf[i], x[i], y[i], 0, sig_x_guess, sig_y_guess]
    pred_params, uncert_cov = curve_fit(gauss2d, xy, datf, p0=guess, maxfev=100000)

    zpred = gauss2d(xy, *pred_params)
    #print('Predicted params:', pred_params)
    print('Residual, RMS(obs - pred)/mean:', np.sqrt(np.mean((datf - zpred)**2))/np.mean(datf))
    return zpred.reshape(datc.shape[0],datc.shape[1]), pred_params

def gaussianBeam(xy, I0, x0, y0, w0, a0):
    """Math: 2D Gaussian"""
    x, y =xy
    r = np.sqrt((x-x0)**2+(y-y0)**2)
    return I0 * np.exp(-2 * r**2 / w0**2) + a0

def gaussianBeamFit(datc):
    """2D Gaussian fit to image matrix. params [I0, x0, y0, w0, offset]"""
    rows=range(datc.shape[1])
    cols=range(datc.shape[0])
    x,y=np.meshgrid(rows,cols)
    x,y=x.flatten(),y.flatten()
    xy=[x,y]
    datf=datc.flatten()

    i = datf.argmax()
    ihalf=np.argmin(np.abs(datf-datf[i]/2)) #find position of half-maximumsync
    w0_guess = np.abs(x[i]-x[ihalf])+1
    offset = np.mean(datc)
    print(w0_guess)
    guess = [datf[i], x[i], y[i], w0_guess, offset]
    pred_params, uncert_cov = curve_fit(gaussianBeam, xy, datf, p0=guess, maxfev=100000)

    zpred = gaussianBeam(xy, *pred_params)
    print('Predicted params:', pred_params)
    print('Residual, RMS(obs - pred)/mean:', np.sqrt(np.mean((datf - zpred)**2))/np.mean(datf))
    return zpred.reshape(datc.shape[0],datc.shape[1]), pred_params

def gaussianBeam2D(xy, amp, x0, y0, theta, wx, wy, z0):
    """Math: 2D Gaussian w/ factor of two for optics formalism. params [amp, x0, y0, theta, wx, wy]"""
    x, y =xy

    a = 2*np.cos(theta)**2/(wx**2) + 2*np.sin(theta)**2/(wy**2);
    b = -np.sin(2*theta)/(wx**2) + np.sin(2*theta)/(wy**2);
    c = 2*np.sin(theta)**2/(wx**2) + 2*np.cos(theta)**2/(wy**2);

    return z0+amp * np.exp(-(a * (x - x0)**2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0)**2))

def gaussianBeamFit2D(datc, auto = True, mguess = []):
    """2D Gaussian fit to image matrix. params [amp, x0, y0, theta, w_x, w_y, z0]. Returns fitdata, params, perr"""
    rows=range(datc.shape[1])
    cols=range(datc.shape[0])
    x,y=np.meshgrid(rows,cols)
    x,y=x.flatten(),y.flatten()
    xy=[x,y]
    datf=datc.flatten()

    if auto:
        i = datf.argmax()
        ihalf=np.argmin(np.abs(datf-datf[i]/2)) #find position of half-maximum
        wx_guess = np.abs(x[i]-x[ihalf])+1
        wy_guess = np.abs(y[i]-y[ihalf])+1
#         print(wx_guess,wy_guess)
        guess = [datf[i], x[i], y[i], 0, wx_guess, wy_guess, 0]
    else:
        guess = mguess
    pred_params, uncert_cov = curve_fit(gaussianBeam2D, xy, datf, p0=guess, maxfev=1000, bounds = ([0,0,0,0,0,0,0],[np.inf,np.inf,np.inf,np.pi/2,np.inf,np.inf,np.inf]))

    perr = np.sqrt(np.diag(uncert_cov))
    zpred = gaussianBeam2D(xy, *pred_params)
    #print('Predicted params:', pred_params)
#     print('Residual, RMS(obs - pred)/mean:', np.sqrt(np.mean((datf - zpred)**2))/np.mean(datf))
    return zpred.reshape(datc.shape[0],datc.shape[1]), pred_params, perr

def beam_waist(z,z0,zr,w0,lam):
#     zr=np.pi*w0**2/lam
    return w0*np.sqrt(1+((z-z0)/zr)**2)

def waistFit(kvals,dat,lam):
    i=dat.argmin()
    guess=[kvals[i],1,dat[i]]
#     print(guess)
    pred_params, uncert = curve_fit(lambda z, z0, zr, w0: beam_waist(z, z0, zr, w0, lam), kvals, dat, p0=guess)
    perr = np.sqrt(np.diag(uncert))
    zpred = beam_waist(kvals, *pred_params, lam)
#     print('Predicted params (z0, zr, w0):', pred_params)
#     print('Residual, RMS(obs - pred):', np.sqrt(np.mean((dat - zpred)**2)))
    return zpred, pred_params, perr
