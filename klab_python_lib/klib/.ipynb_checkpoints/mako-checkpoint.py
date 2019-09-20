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

from PIL import Image

def loadBMPs(path, nameFormat):
    names = arr(os.listdir(path))
    print(names)
    inds = arr([int((i.replace(nameFormat,"")).replace(".bmp","")) for i in names])
    sort = np.argsort(inds)
    names = names[sort]
    imgs = []
    for name in names:
        img  = arr(Image.open(path+name))
        imgs.append(img)
    
    return imgs

def sortImgs(exp, imgs):
    """Sort list of imported images (imgs) by hdf5 experiment object (exp) returned from Chimera."""
    key = exp.key
    reps = exp.reps
    sort = np.argsort(key)
    shape = imgs.shape
    imgs = np.reshape(imgs, (len(key), reps, shape[-2], shape[-1]))
    return key[sort], imgs[sort]