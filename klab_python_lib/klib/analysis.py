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

#####
# Dependencies for HDF file wrapper class, ExpClass
import h5py as h5
from colorama import Fore, Style
#####

from .plotutil import *
from .mathutil import *
from .imagutil import *

def get_threshold(evencounts, oddcounts, tmin = 2, tmax = 10, criteria = 'inf'):
    """Finds threshold that minimizes the infidelity between two images"""
    trial_threshes = np.arange(tmin, tmax, 0.1)
    infs = []
    losses = []
    tots = []
    for thresh in trial_threshes:
        odds = np.array([int(x) for x in oddcounts>thresh])
        evens = np.array([int(x) for x in evencounts>thresh])
        sums = odds + evens
        diffs = evens - odds
        aa = len(sums[sums > 1])
        vv = len(sums[sums < 1])
        av = len(sums[diffs == 1])
        va = len(sums[diffs < 0])        
        infs.append(va)
        losses.append(2*(av-va))
    tots = np.array(infs)+np.array(losses)
    #plt.plot(trial_threshes, infs)
    #plt.plot(trial_threshes, losses)
    #plt.plot(trial_threshes, tots)
    if criteria == 'inf':
        return trial_threshes[np.argmin(infs)]
    elif criteria == 'tot':
        return trial_threshes[np.argmin(tots)]

def hist_stats_old(roisums, thresh = -999, countsperphoton=70, fnum=-1, crit = 'tot', tmin = 0, i=0, roinum = -1, quiet = False, plots = True):
    lossfracs = []
    infidelities = []

#     flat = np.array(roisums[i,10:,:]).flatten('F')/countsperphoton
    flat = np.array(roisums[i,:,:]).flatten('F')/countsperphoton
    if roinum >= 0:
        flat = np.array(roisums[i,:,roinum]).flatten('F')/countsperphoton
    evencounts = flat[0::2]
    oddcounts = flat[1::2]


    oddcounts = oddcounts[np.round(evencounts, 2)!=0.0]
    evencounts = evencounts[np.round(evencounts, 2)!=0.0]

#     REMOVED. Previously here to account for lost images, this shouldn't happen any more, so removing this is a convenient check for if things are broken.
#     oddcounts = oddcounts[evencounts>-100]
#     evencounts = evencounts[evencounts>-100]
#     print(flat.shape)

    if thresh==-999:
        thresh = get_threshold(evencounts, oddcounts, criteria = crit, tmin = tmin)
    
    if plots:
        ne, bse, ps = plt.hist(evencounts,50, normed=True, histtype = 'step')
        plt.plot([thresh, thresh],[0, max(ne)],'--')
        #plt.show()

        n, bs, ps = plt.hist(oddcounts,50, normed=True, histtype = 'step')
        plt.plot([thresh, thresh],[0, max(n)],'--')
        #plt.axis([-5, 40, 0, 100])

        plt.xlabel("Photons Collected")
        plt.ylabel("Normalized Counts")
    #     plt.show()
    
    odds = np.array([int(x) for x in oddcounts>thresh])
    evens = np.array([int(x) for x in evencounts>thresh])
    sums = odds + evens
    diffs = evens - odds
    aa = len(sums[sums > 1])
    vv = len(sums[sums < 1])
    av = len(sums[diffs == 1])
    va = len(sums[diffs < 0])
    if not quiet:
        print('total pairs: ' + str(len(sums)))
        print('atom atom: ' + str(aa))
        print('void void: ' + str(vv))
        print('atom void: ' + str(av))
        print('void atom: ' + str(va))
        print(aa,vv,av)

    tot = aa+vv+av+va
    ff = (aa+av)/tot # fill fraction
    if ff > 0:
        infidelity = np.round(va*100/len(sums), 2)
        lossfrac = np.round((av-va)*100/(ff*len(sums)), 2)
        inf_err = np.round(np.sqrt(va)*100/len(sums), 2)
        
        z = 1.96 # corresponds to 95% CI
        if lossfrac>0:
            loss_err = 100*z*np.sqrt(.01*lossfrac*(1-.01*lossfrac)/len(sums))
        else:
            loss_err = 0
        #loss_err = np.round(np.sqrt(av-va)*100/(ff*len(sums)), 2)
    else:
#         print('fill fraction = 0')
        return -1, -1, -1, -1
    lossfracs.append(lossfrac)
    infidelities.append(infidelity)
    if not quiet:
        print('average infidelity: ' + str(infidelity)+ ' += '+ str(inf_err)+ ' percent')
        print('losses: ' + str(lossfrac)+ ' += '+ str(loss_err)+ ' percent')
        print('even/odd thresholds: ',thresh,thresh)
    return infidelity, inf_err, lossfrac, loss_err

def hist_stats(roisums, thresh = -999, countsperphoton=70, fnum=-1, crit = 'tot', tmin = 0, i=0, roinum = -1, quiet = False, plots = True, rng = None):
    lossfracs = []
    infidelities = []
# roisums_old: (variation, nreps*npics, rois), roisums: (variation, nreps, npics, rois)
# The issue below is that flatten mode 'F' (column order) is being used s.th. iterating by 2 jumps images and not ROIs. Easy fix for now but keep in mind for eventually handling 
#     flat = np.array(roisums[i,10:,:]).flatten('F')/countsperphoton
    a, b, c, d = roisums.shape
    roisums = roisums.reshape(a, b*c, d)
    flat = np.array(roisums[i,:,:]).flatten('F')/countsperphoton
    if roinum >= 0:
        flat = np.array(roisums[i,:,roinum]).flatten('F')/countsperphoton
    evencounts = flat[0::2]
    oddcounts = flat[1::2]


    oddcounts = oddcounts[np.round(evencounts, 2)!=0.0]
    evencounts = evencounts[np.round(evencounts, 2)!=0.0]

#     REMOVED. Previously here to account for lost images, this shouldn't happen any more, so removing this is a convenient check for if things are broken.
#     oddcounts = oddcounts[evencounts>-100]
#     evencounts = evencounts[evencounts>-100]
#     print(flat.shape)

    if thresh==-999:
        thresh = get_threshold(evencounts, oddcounts, criteria = crit, tmin = tmin)
    
    if plots:
        ne, bse, ps = plt.hist(evencounts,50, normed=True, range = rng, histtype = 'step', color = '0')
        plt.plot([thresh, thresh],[0, max(ne)],'k--')
        #plt.show()

        n, bs, ps = plt.hist(oddcounts,50, normed=True, range = rng, histtype = 'step', color = '0.5', zorder = 0)
        plt.plot([thresh, thresh],[0, max(n)],'--', color = '0.5', zorder = 0)
        #plt.axis([-5, 40, 0, 100])

        plt.xlabel("Photons Collected")
        plt.ylabel("Normalized Counts")
        plt.legend(['Even', 'Odd'])
        plt.show()
    
    odds = np.array([int(x) for x in oddcounts>thresh])
    evens = np.array([int(x) for x in evencounts>thresh])
    sums = odds + evens
    diffs = evens - odds
    aa = len(sums[sums > 1])
    vv = len(sums[sums < 1])
    av = len(sums[diffs == 1])
    va = len(sums[diffs < 0])
    if not quiet:
        print('total pairs: ' + str(len(sums)))
        print('atom atom: ' + str(aa))
        print('void void: ' + str(vv))
        print('atom void: ' + str(av))
        print('void atom: ' + str(va))
        print(aa,vv,av)
        print('total even: ', np.sum(evens))
        print('total odd: ', np.sum(odds))
        print('total loss: ', (np.sum(evens)-np.sum(odds))/np.sum(evens))

    tot = aa+vv+av+va
    ff = (aa+av)/tot # fill fraction
    if ff > 0:
        infidelity = np.round(va*100/len(sums), 2)
        lossfrac = np.round((av-va)*100/(ff*len(sums)), 2)
        inf_err = np.round(np.sqrt(va)*100/len(sums), 2)
        
        z = 1.96 # corresponds to 95% CI
        if lossfrac>0:
            loss_err = 100*z*np.sqrt(.01*lossfrac*(1-.01*lossfrac)/len(sums))
        else:
            loss_err = 0
        #loss_err = np.round(np.sqrt(av-va)*100/(ff*len(sums)), 2)
    else:
#         print('fill fraction = 0')
        return -1, -1, -1, -1
    lossfracs.append(lossfrac)
    infidelities.append(infidelity)
    if not quiet:
        print('average infidelity: ' + str(infidelity)+ ' += '+ str(inf_err)+ ' percent')
        print('losses: ' + str(lossfrac)+ ' += '+ str(loss_err)+ ' percent')
        print('even/odd thresholds: ',thresh,thresh)
    return infidelity, inf_err, lossfrac, loss_err

def hist_stats_roi(roisums, thresh = -999, countsperphoton=70, fnum=-1, crit = 'tot', tmin = 0, i=0, roinum = -1, quiet = False, plots = True, rng = None):
    lossfracs = []
    infidelities = []
# roisums_old: (variation, nreps*npics, rois), roisums: (variation, nreps, npics, rois)
# The issue below is that flatten mode 'F' (column order) is being used s.th. iterating by 2 jumps images and not ROIs. Easy fix for now but keep in mind for eventually handling 
#     flat = np.array(roisums[i,10:,:]).flatten('F')/countsperphoton
    nVar, nRep, nPic, nRoi = roisums.shape
    oddcounts = roisums[i,:,1,:]/countsperphoton
    evencounts = roisums[i,:,0,:]/countsperphoton

    if thresh==-999:
        thresh = get_threshold(evencounts, oddcounts, criteria = crit, tmin = tmin)
    
    if plots:
        ne, bse, ps = plt.hist(evencounts.flatten(),50, normed=True, range = rng, histtype = 'step', color = '0')
        plt.plot([thresh, thresh],[0, max(ne)],'k--')
        #plt.show()

        n, bs, ps = plt.hist(oddcounts.flatten(),50, normed=True, range = rng, histtype = 'step', color = '0.5', zorder = 0)
        plt.plot([thresh, thresh],[0, max(n)],'--', color = '0.5', zorder = 0)
        #plt.axis([-5, 40, 0, 100])

        plt.xlabel("Photons Collected")
        plt.ylabel("Normalized Counts")
        plt.legend(['Even', 'Odd'])
#         plt.savefig('hist.pdf', format='pdf', dpi=1000, bbox_inches='tight')
        plt.show()

    
    odds = np.greater(oddcounts, thresh).astype(int)
    evens = np.greater(evencounts, thresh).astype(int)
    
    sums = odds + evens
    diffs = evens - odds
    
    aa = np.greater(sums, 1).astype(int)
    vv = np.less(sums, 1).astype(int)
    av = np.equal(diffs, 1).astype(int)
    va = np.less(diffs, 0).astype(int)
    
    N = np.sqrt(nRoi).astype(int)
    av = av.reshape(nRep, N, N)
    va = va.reshape(nRep, N, N)
    
    avxp = np.roll(av, 1, axis = 1)
    avxp[:, 0, :] = 0
    avxm = np.roll(av, -1, axis = 1)
    avxm[:, -1, :] = 0
    avyp = np.roll(av, 1, axis = 2)
    avyp[:, :, 0] = 0
    avym = np.roll(av, -1, axis = 2)
    avym[:, :, -1] = 0
        
    av_va = (avxp + avxm + avyp + avym)*va

    aaf, vvf, avf, vaf = np.sum(aa), np.sum(vv), np.sum(av), np.sum(va)
    npair = len(sums.flatten())
    if not quiet:
        print('total pairs: ' + str(npair))
        print('atom atom: ' + str(aaf))
        print('void void: ' + str(vvf))
        print('atom void: ' + str(avf))
        print('void atom: ' + str(vaf))
        print('total even: ', np.sum(evens))
        print('total odd: ', np.sum(odds))
        print('total loss: ', (np.sum(evens)-np.sum(odds))/np.sum(evens))

    tot = aaf+vvf+avf+vaf
    ff = (aaf+avf)/tot # fill fraction
    if ff > 0:
        infidelity = np.round(vaf*100/npair, 2)
        lossfrac = np.round((avf-vaf)*100/(ff*npair), 2)
        inf_err = np.round(np.sqrt(vaf)*100/npair, 2)
        
        z = 1.96 # corresponds to 95% CI
        if lossfrac>0:
            loss_err = 100*z*np.sqrt(.01*lossfrac*(1-.01*lossfrac)/npair)
        else:
            loss_err = 0
        #loss_err = np.round(np.sqrt(avf-vaf)*100/(ff*len(sums)), 2)
    else:
#         print('fill fraction = 0')
        return -1, -1, -1, -1
    lossfracs.append(lossfrac)
    infidelities.append(infidelity)
    if not quiet:
        print('average infidelity: ' + str(infidelity)+ ' += '+ str(inf_err)+ ' percent')
        print('losses: ' + str(lossfrac)+ ' += '+ str(loss_err)+ ' percent')
        print('even/odd thresholds: ',thresh,thresh)
        
#     plt.imshow(np.sum(av, axis = 0))
#     plt.title('av')
#     plt.colorbar()
#     plt.show()
    
#     plt.imshow(np.sum(va, axis = 0))
#     plt.title('va')
#     plt.colorbar()
#     plt.show()
    
#     plt.imshow(np.sum(av_va, axis = 0))
#     plt.title("av_va")
#     plt.colorbar()
#     plt.show()    
    
    return infidelity, inf_err, lossfrac, loss_err

def img_stats(roisums, thresh = -999, countsperphoton=70, fnum=-1, crit = 'tot', tmin = 0, i=0, roinum = -1, quiet = False, plots = True):
    lossfracs = []
    infidelities = []

#     flat = np.array(roisums[i,10:,:]).flatten('F')/countsperphoton
    flat = np.array(roisums[i,:,:]).flatten('F')/countsperphoton
    if roinum >= 0:
        flat = np.array(roisums[i,:,roinum]).flatten('F')/countsperphoton
    evencounts = flat[0::2]
    oddcounts = flat[1::2]


#     oddcounts = oddcounts[np.round(evencounts, 2)!=0.0]
#     evencounts = evencounts[np.round(evencounts, 2)!=0.0]

#     REMOVED. Previously here to account for lost images, this shouldn't happen any more, so removing this is a convenient check for if things are broken.
#     oddcounts = oddcounts[evencounts>-100]
#     evencounts = evencounts[evencounts>-100]
#     print(flat.shape)

    if thresh==-999:
        thresh = get_threshold(evencounts, oddcounts, criteria = crit, tmin = tmin)
    
    if plots:
        ne, bse, ps = plt.hist(evencounts,50, normed=True, histtype = 'step')
        plt.plot([thresh, thresh],[0, max(ne)],'--')
        #plt.show()

        n, bs, ps = plt.hist(oddcounts,50, normed=True, histtype = 'step')
        plt.plot([thresh, thresh],[0, max(n)],'--')
        #plt.axis([-5, 40, 0, 100])

        plt.xlabel("Photons Collected")
        plt.ylabel("Normalized Counts")
    #     plt.show()
    
    odds = np.array([oddcounts>thresh])
    evens = np.array([evencounts>thresh])
    
    fill = evens
    losses = evens&~odds
    
    
#     sums = odds + evens
#     diffs = evens - odds
#     aa = len(sums[sums > 1])
#     vv = len(sums[sums < 1])
#     av = len(sums[diffs == 1])
#     va = len(sums[diffs < 0])
#     if not quiet:
#         print('total pairs: ' + str(len(sums)))
#         print('atom atom: ' + str(aa))
#         print('void void: ' + str(vv))
#         print('atom void: ' + str(av))
#         print('void atom: ' + str(va))
#         print(aa,vv,av)

#     tot = aa+vv+av+va
#     ff = (aa+av)/tot # fill fraction
#     if ff > 0:
#         infidelity = np.round(va*100/len(sums), 2)
#         lossfrac = np.round((av-va)*100/(ff*len(sums)), 2)
#         inf_err = np.round(np.sqrt(va)*100/len(sums), 2)
#         loss_err = np.round(np.sqrt(av+va)*100/(ff*len(sums)), 2)
#     else:
#         print('fill fraction = 0')
#         return -1, -1, -1, -1
#     lossfracs.append(lossfrac)
#     infidelities.append(infidelity)
#     if not quiet:
#         print('average infidelity: ' + str(infidelity)+ ' += '+ str(inf_err)+ ' percent')
#         print('losses: ' + str(lossfrac)+ ' += '+ str(loss_err)+ ' percent')
#         print('even/odd thresholds: ',thresh,thresh)
#     return infidelity, inf_err, lossfrac, loss_err
    return fill, losses

def spect_plot(keysorts, counts, points, guess):
    """Triple lorentz fit with convenient printing. Guess params: [a0, a1, a2, sigma_carrier, sigma_side, f_carrier, df_sidebands, y0]"""
#     guess = [.2, .25, .03, .05, .05, 79.98, .2, 0]

    ks = keysorts[0]
    kss = np.linspace(np.min(ks),np.max(ks),1000)
    cts = np.mean(counts, axis = 0)
    pts = np.sum(points, axis = 0)
    z = 1.96 # corresponds to 95% CI
    error = z*np.sqrt(cts*(1-cts)/pts)

    popt, pcov = curve_fit(triplor, ks, cts, p0 = guess, maxfev=100000)

#     guess = [.1, .1, .1, .1, .1, .05, .05, 80.02, .2, 0]
#     popt, pcov = curve_fit(fivelor, ks, cts, p0 = guess, maxfev=100000000)

    x = (popt[-3]-ks)*1e3
    xs = np.linspace(np.max(x),np.min(x),1000)
    
#     n=np.max(cts)
    n=1
    
#     plt.errorbar(x, cts, yerr = error,fmt='.')
#     plt.plot(xs, triplor(kss, *popt))

    # plt.plot([popt[5], popt[5]],[0, .15], 'r-')
    # plt.plot([popt[5]-popt[6], popt[5]-popt[6]],[0, .15], 'r-')
    # plt.plot([popt[5]+popt[6], popt[5]+popt[6]],[0, .15], 'r-')
    
    print("Blue SB:")
    print(np.abs(popt[0]))
    print('+-')
    print(np.sqrt(pcov[0][0]))

    print("Red SB:")
    print(np.abs(popt[2]))
    print('+-')
    print(np.sqrt(pcov[2][2]))

    print("ratio:")
    print(np.abs(popt[0]/popt[2]))
    print('ratio bound:')
    print(np.abs(popt[0])/(np.abs(popt[2])+np.sqrt(pcov[2][2])))

    print("splitting:")
    print(str(1000*popt[6])+' kHz')
    print('+-')
    print(str(1000*np.sqrt(pcov[6][6]))+' kHz')
    print(popt)
    return x, cts/n, error/n, xs, triplor(kss, *popt)/n

def find_rois(dat, roi_size, filter_size, threshold, bg_offset, display=True):
    """Find peaks in image and define set of ROIs. Returns rois, bgrois."""
    # dat=np.mean(dat,axis=0)
    # dat=ndi.gaussian_filter(dat,1.5)
    dmin=ndi.filters.minimum_filter(dat,filter_size)
    dmax=ndi.filters.maximum_filter(dat,filter_size)

    maxima = (dat==dmax)
    diff = ((dmax-dmin)>threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndi.label(maxima)
    xys = np.array(ndi.center_of_mass(dat, labeled, range(1, num_objects+1)))

#     sort = np.lexsort((xys[:,0],xys[:,1]))
#     xys = xys[sort]
    
#     keep = [True]
#     for i in range(len(xys)-1):
#         xy = xys[i]
#         xyp = xys[i+1]
#         if np.abs(xyp[0]-xy[0])<roi_size/10:
# #         or np.abs(xyp[1]-xy[1])<roi_size/5:
#             keep.append(False)
#         else:
#             keep.append(True)
    
#     xys = xys[keep]
    
    rois=[[int(xy[1])-roi_size, int(xy[1])+roi_size, int(xy[0])-roi_size, int(xy[0])+roi_size] for xy in xys]
    bgrois = [[roi[0]+bg_offset[0],roi[1]+bg_offset[0],roi[2]+bg_offset[1],roi[3]+bg_offset[1]] for roi in rois]

    corners = arr(rois)[:,[0,2]]
    
    for i in range(len(corners)-1):
        if corners[i+1,1]-corners[i,1] < roi_size:
            corners[i+1,1] = corners[i,1]
    
# #     Find average separation.
#     xvals = corners[:,0].sort()
#     diff = [y - x for x, y in zip(*[iter(xvals)] * 2)]
#     davg = sum(diff) / len(diff)
    
    # corners = (corners/roiSettings[0]).astype(int)
#     corners = np.fix((corners-[min(corners[:,0]), min(corners[:,1])])/(roi_size))
    sort = np.lexsort((corners[:,0], corners[:,1]))
#     sort0 = np.argsort(corners[:,1])
    
    rois = arr(rois)[sort]
    bgrois = arr(bgrois)[sort]
    xys = xys[sort]
    
    rois = rois[corners[:,0]>10]
    corners = corners[corners[:,0]>10]
    rois = rois[corners[:,1]<dat.shape[0]-10]
    
    if display:
        fig, ax = plt.subplots(figsize = (15,15))
        ax.imshow(dat)
#         ax = plt.gca()
        for i in range(len(rois)):
            roi = rois[i]
            ax.add_patch(
            patches.Rectangle((roi[0], roi[2]), roi[1]-roi[0], roi[3]-roi[2], fill = False, color = 'black'))
            ax.add_patch(
            patches.Rectangle((roi[0]+bg_offset[0], roi[2]+bg_offset[1]), roi[1]-roi[0], roi[3]-roi[2], fill = False, color = 'red'))

            plt.text(xys[i,1],xys[i,0],str(i), color = 'white')
        # plt.plot(xys[:, 1], xys[:, 0], 'r.')
        plt.show()
    print("Peaks found:" + str(len(rois)))
    return rois, bgrois


def get_projections(image, roi):
    """Bin image within roi in x and y directions"""
    if len(image.shape) == 3:
        image = np.mean(image, axis = 0)
    image = image[roi[2]:roi[3],roi[0]:roi[1]]
    #image = image - ndi.filters.gaussian_filter(image, sigma=10)
    x_proj = np.sum(image, axis = 0)
    y_proj = np.sum(image, axis = 1)
    #plt.plot(x_proj)
    #plt.plot(y_proj)
    return x_proj, y_proj

# def findroi(dat, roi_size, bg_offset, display=True):
#     """Automatically define ROI around brightest pixel in image, and offset background ROI"""
#     mdat=np.mean(dat, axis=0)
#     i=mdat.argmax()
#     rows=range(mdat.shape[1])
#     cols=range(mdat.shape[0])
#     x,y=np.meshgrid(rows,cols)
#     x,y=x.flatten(),y.flatten()
#     x0=x[i]
#     y0=y[i]

#     roi=[x0-roi_size, x0+roi_size, y0-roi_size, y0+roi_size]

#     if display:
#         plt.imshow(mdat)
#         ax1 = plt.gca()
#         ax1.add_patch(
#         patches.Rectangle((roi[0], roi[2]), roi[1]-roi[0], roi[3]-roi[2], fill = False, color = 'black'))
#         ax1.add_patch(
#         patches.Rectangle((roi[0]+bg_offset[0], roi[2]+bg_offset[1]), roi[1]-roi[0], roi[3]-roi[2], fill = False, color = 'red'))
#         plt.show()
#     return roi


#
# returns a list of rois, for looking at multiple spots
#
#

def get_rois(n, m, spot_offset, x0, y0, w):
    rois = []
    for i in range(n):
        for ii in range(m):
            xi = int(x0+i*spot_offset +ii*spot_offset)
            yi = int(y0+i*spot_offset -ii*spot_offset)
            roi = [xi, xi + w, yi, yi + w]
            rois.append(roi)
    return rois


#
#  Calculates the difference in roi sums between two regions on an image.  roi [x1, x2, y1, y1] defines the positive
#  value.  Offset (x_o, y_o) defines the offset of the background region from the image region.  If passed a single
#  image, will use that.  If passed a list of images, will calculate the mean image, then proceed with processing the
#  resulting image.
#

def get_roi_sum(image, roi, bg_offset, display=True, bgsub = False):
    """Get sum in rectangular region of image, with the option of displaying the region of interest over a plot of the data."""
    #if len(image.shape) == 3:
    #    image = np.mean(image, axis = 0)
    if display:
        plt.imshow(image)
        ax1 = plt.gca()
        ax1.add_patch(
        patches.Rectangle((roi[0], roi[2]), roi[1]-roi[0], roi[3]-roi[2], fill = False, color= 'black'))
        ax1.add_patch(
        patches.Rectangle((roi[0]+bg_offset[0], roi[2]+bg_offset[1]), roi[1]-roi[0], roi[3]-roi[2], fill = False, color = 'red'))
    #imsum = np.sum(image[roi[0]:roi[1],roi[2]:roi[3]])
    #bgsum = np.sum(image[roi[2]+bg_offset[1]:roi[3]+bg_offset[1],roi[0]+bg_offset[0]:roi[1]+bg_offset[0]])
    imsum = np.sum(image[roi[2]:roi[3],roi[0]:roi[1]])
    if bgsub:
        bgsum = np.sum(image[roi[2]+bg_offset[1]:roi[3]+bg_offset[1],roi[0]+bg_offset[0]:roi[1]+bg_offset[0]])
        return imsum - bgsum
    return imsum


def get_max(image, roi, bg_offset, display=True, bgsub = False):
    """Get sum in rectangular region of image, with the option of displaying the region of interest over a plot of the data."""
    #if len(image.shape) == 3:
    #    image = np.mean(image, axis = 0)
    if display:
        plt.imshow(image)
        ax1 = plt.gca()
        ax1.add_patch(
        patches.Rectangle((roi[0], roi[2]), roi[1]-roi[0], roi[3]-roi[2], fill = False, color = 'black'))
        ax1.add_patch(
        patches.Rectangle((roi[0]+bg_offset[0], roi[2]+bg_offset[1]), roi[1]-roi[0], roi[3]-roi[2], fill = False, color = 'red'))
    #imsum = np.sum(image[roi[0]:roi[1],roi[2]:roi[3]])
    #bgsum = np.sum(image[roi[2]+bg_offset[1]:roi[3]+bg_offset[1],roi[0]+bg_offset[0]:roi[1]+bg_offset[0]])
    immax = max(image[roi[2]:roi[3],roi[0]:roi[1]])
    return immax