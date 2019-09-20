# Data export functions for Adam.

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

def toCSV(xname, yname, x, y, filename = "tmpdata"):
    np.savetxt(filename, data, delimiter = ',')