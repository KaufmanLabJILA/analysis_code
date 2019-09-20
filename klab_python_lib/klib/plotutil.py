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

#
# plots the rois on an image
#
def plotrois(img, rois, bg_offset = 0, bgs = False):
    plt.imshow(img)
    ax1 = plt.gca()
    for roi in rois:
        ax1.add_patch(
        patches.Rectangle((roi[0], roi[2]), roi[1]-roi[0], roi[3]-roi[2], fill = False, color = 'black'))
        if bgs:
            ax1.add_patch(
            patches.Rectangle((roi[0]+bg_offset[0], roi[2]+bg_offset[1]), roi[1]-roi[0], roi[3]-roi[2], fill = False, color = 'red'))
    plt.show()