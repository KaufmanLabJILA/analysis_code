# Kaufman lab library for common analysis functions. Only commit stable functions, and make sure to comment code well. Single use/experimental funcitons should be saved in a jupyter notebook in the relevant data folder.
# import imageio
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astropy.io import fits
from scipy import ndimage as ndi
from scipy.optimize import curve_fit

#####
# Dependencies for HDF file wrapper class, ExpClass
import h5py as h5
from colorama import Fore, Style
from numpy import array as arr
# import numpy as np
#####
def load_data(fnum, dire=os.getcwd()):
    """Load img_<fnum>.fits files into list of arrays."""
    os.chdir(dire)
    fname = 'img_' + str(fnum)+'.fits'
    with fits.open(fname) as hdul:
        hdul.info()
        images = hdul[0].data
    return images

def load_img(imgNumber, logNumber, dataAddress=os.getcwd()):
    """Wrapper to load image data from fits file, and log data from """
    """Chimera HDF5 file."""
    with ExpFile(dataAddress+'/Raw Data/', logNumber) as exp:
        # exp.print_all()
        key = exp.get_old_key()
        nreps = exp.get_info_arr()[3]

    # print(key)

    keyVals = key[1]
    nconfigs = len(keyVals)
    # keyVals = np.linspace(1,10, nconfigs)

    images = load_data(imgNumber, dire=dataAddress)
    images = images-ndi.filters.gaussian_filter(images, sigma=10)
    return keyVals, nreps, nconfigs, images