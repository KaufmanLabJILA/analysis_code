# Shared dependencies:

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
from mpl_toolkits.mplot3d import Axes3D
from astropy.io import fits

from scipy.stats import sem 
from scipy import ndimage as ndi
from scipy.optimize import curve_fit

from skimage import restoration
from skimage.feature import peak_local_max
from skimage import img_as_float

from PIL import Image

#####
# Dependencies for HDF file wrapper class, ExpClass
import h5py as h5
from colorama import Fore, Style
#####