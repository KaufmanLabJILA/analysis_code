
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

    rois=[[int(xy[1])-roi_size, int(xy[1])+roi_size, int(xy[0])-roi_size, int(xy[0])+roi_size] for xy in xys]
    bgrois = [[roi[0]+bg_offset[0],roi[1]+bg_offset[0],roi[2]+bg_offset[1],roi[3]+bg_offset[1]] for roi in rois]


    if display:
        plt.imshow(dat)
        ax1 = plt.gca()
        for i in range(len(rois)):
            roi = rois[i]
            ax1.add_patch(
            patches.Rectangle((roi[0], roi[2]), roi[1]-roi[0], roi[3]-roi[2], fill = False, color = 'black'))
            ax1.add_patch(
            patches.Rectangle((roi[0]+bg_offset[0], roi[2]+bg_offset[1]), roi[1]-roi[0], roi[3]-roi[2], fill = False, color = 'red'))

            plt.text(xys[i,1],xys[i,0],str(i))
        # plt.plot(xys[:, 1], xys[:, 0], 'r.')
        plt.show()
    print("Peaks found:" + str(num_objects))
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

def findroi(dat, roi_size, bg_offset, display=True):
    """Automatically define ROI around brightest pixel in image, and offset background ROI"""
    mdat=np.mean(dat, axis=0)
    i=mdat.argmax()
    rows=range(mdat.shape[1])
    cols=range(mdat.shape[0])
    x,y=np.meshgrid(rows,cols)
    x,y=x.flatten(),y.flatten()
    x0=x[i]
    y0=y[i]

    roi=[x0-roi_size, x0+roi_size, y0-roi_size, y0+roi_size]

    if display:
        plt.imshow(mdat)
        ax1 = plt.gca()
        ax1.add_patch(
        patches.Rectangle((roi[0], roi[2]), roi[1]-roi[0], roi[3]-roi[2], fill = False, color = 'black'))
        ax1.add_patch(
        patches.Rectangle((roi[0]+bg_offset[0], roi[2]+bg_offset[1]), roi[1]-roi[0], roi[3]-roi[2], fill = False, color = 'red'))
        plt.show()
    return roi

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

def triplor(x, a0, a1, a2, kc, ks, x0, dx, y0):
    return y0+lor(x, a0, ks, x0-dx)+lor(x, a1, kc, x0) + lor(x, a2, ks, x0+dx)

def fivelor(x, a0, a1, a2, a3, a4, kc, ks, kss, x0, dx, y0):
    return y0 + lor(x, a0, ks, x0-2*dx)+ lor(x, a1, ks, x0-dx)+lor(x, a2, kc, x0) + lor(x, a3, ks, x0+dx) +lor(x, a4, ks, x0+2*dx)

def expfit(t, A, tau):
    return A*np.exp(-t/tau)

def gausfit(keyVals, dat, y_offset=False, negative=False, n=0):
    """1D gaussian fit, with or without Y-offset, and with positive or"""
    """negative amplitude. Can select nth color from default matplotlib"""
    """color cycle."""
    xdat=keyVals
    ydat=np.array(dat)
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
    sig = np.abs(xdat[i]-xdat[ihalf])

    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.plot(xdat, ydat,".", color=cycle[n])

    if y_offset:
        guess = [x0, a, sig, y0]
        print(guess)
        gauss_params, gauss_uncert = curve_fit(gaussian, xdat, ydat, p0=guess, maxfev=10000)
        plt.plot(xdat, gaussian(xdat, *gauss_params), "-", color=cycle[n])
    else:
        guess = [x0, a, sig]
        print(guess)
        gauss_params, gauss_uncert = curve_fit(lambda x, x0, a, sig: gaussian(x, x0, a, sig, 0), xdat, ydat, p0=guess, maxfev=100000)
        plt.plot(xdat, gaussian(xdat, *gauss_params, 0), "-", color=cycle[n])

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
    """2D Gaussian fit to image matrix. params [amp, x0, y0, theta, w_x, w_y, z0]"""
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
    print(guess)
    pred_params, uncert_cov = curve_fit(lambda z, z0, zr, w0: beam_waist(z, z0, zr, w0, lam), kvals, dat, p0=guess)
    zpred = beam_waist(kvals, *pred_params, lam)
    print('Predicted params (z0, zr, w0):', pred_params)
    print('Residual, RMS(obs - pred):', np.sqrt(np.mean((dat - zpred)**2)))
    return zpred, pred_params

# HDF File wrapper class
class ExpFile:
    """
    a wrapper around an hdf5 file for easier handling and management.
    """

    def __init__(self, dataAddress, file_id=None):
        """
        if you give the constructor a file_id number, it will automatically fill the relevant member variables.
        """
        # copy the current value of the address
        self.f = None
        self.key_name = None
        self.key = None
        self.pics = None
        self.reps = None
        self.experiment_time = None
        self.experiment_date = None
        self.data_addr = dataAddress
        if file_id is not None:
            self.f = self.open_hdf5(fileID=file_id)
            self.key_name, self.key = self.get_old_key()
            self.pics = self.get_pics()
            self.reps = self.f['Master-Parameters']['Repetitions'][0]
            self.experiment_time, self.experiment_date = self.get_experiment_time_and_date()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self.f.close()

    def open_hdf5(self, fileID=None):

        if type(fileID) == int:
            path = self.data_addr + "data_" + str(fileID) + ".h5"
        else:
            # assume a file address itself
            path = fileID
        file = h5.File(path, 'r')
        return file

    def get_key(self):
        """
        :param file:
        :return:
        """
        keyNames = []
        keyValues = []
        foundOne = False
        for var in self.f['Master-Parameters']['Seq #1 Variables']:
            if not self.f['Master-Parameters']['Seq #1 Variables'][var].attrs['Constant']:
                foundOne = True
                keyNames.append(var)
                keyValues.append(arr(self.f['Master-Parameters']['Seq #1 Variables'][var]))
        if foundOne:
            if len(keyNames) > 1:
                return keyNames, arr(transpose(arr(keyValues)))
            else:
                return keyNames[0], arr(keyValues[0])
        else:
            return 'No-Variation', arr([1])


    def get_old_key(self):
        """
        :param file:
        :return:
        """
        keyNames = []
        keyValues = []
        foundOne = False
        for var in self.f['Master-Parameters']['Variables']:
            if not self.f['Master-Parameters']['Variables'][var].attrs['Constant']:
                foundOne = True
                keyNames.append(var)
                keyValues.append(arr(self.f['Master-Parameters']['Variables'][var]))
        if foundOne:
            if len(keyNames) > 1:
                return keyNames, arr(np.transpose(arr(keyValues)))
            else:
                return keyNames[0], arr(keyValues[0])
        else:
            return 'No-Variation', arr([1])

    def get_pics(self):
        try:
            p_t = arr(self.f['Andor']['Pictures'])
            pics = p_t.reshape((p_t.shape[0], p_t.shape[2], p_t.shape[1]))
        except KeyError:
            return None
        return pics

    def print_all(self):
        self.__print_hdf5_obj(self.f,'')

    def print_all_groups(self):
        self.__print_groups(self.f,'')

    def __print_groups(self, obj, prefix):
        """
        Used recursively to print the structure of the file.
        obj can be a single file or a group or dataset within.
        """
        for o in obj:
            if o == 'Functions':
                print(prefix, o)
                self.print_functions(prefix=prefix+'\t')
            elif o == 'Master-Script' or o == "Seq. 1 NIAWG-Script":
                print(prefix,o)
            elif type(obj[o]) == h5._hl.group.Group:
                print(prefix, o)
                self.__print_groups(obj[o], prefix + '\t')
            elif type(obj[o]) == h5._hl.dataset.Dataset:
                print(prefix, o)
            #else:
            #    raise TypeError('???')

    def __print_hdf5_obj(self, obj, prefix):
        """
        Used recursively in other print functions.
        obj can be a single file or a group or dataset within.
        """
        for o in obj:
            if o == 'Functions':
                print(prefix, o)
                self.print_functions(prefix=prefix+'\t')
            elif o == 'Master-Script' or o == "Seq. 1 NIAWG-Script":
                print(prefix,o)
                self.print_script(obj[o])
            elif type(obj[o]) == h5._hl.group.Group:
                print(prefix, o)
                self.__print_hdf5_obj(obj[o], prefix + '\t')
            elif type(obj[o]) == h5._hl.dataset.Dataset:
                print(prefix, o, ':',end='')
                self.__print_ds(obj[o],prefix+'\t')
            else:
                raise TypeError('???')

    def print_functions(self, brief=True, prefix=''):
        """
        print the list of all functions which were created at the time of the experiment.
        if not brief, print the contents of every function.
        """
        for func in self.f['Master-Parameters']['Functions']:
            print(prefix,'-',func,end='')
            if not brief:
                print(': \n---------------------------------------')
                # I think it's a bug that this is nested like this.
                for x in self.f['Master-Parameters']['Functions'][func]:
                    for y in self.f['Master-Parameters']['Functions'][func][x]:
                        print(Style.DIM, y.decode('UTF-8'),end='')
                print('\n---------------------------------------\n')
            print('')

    def print_master_script(self):
        # A shortcut
        self.print_script(self.f['Master-Parameters']['Master-Script'])

    def print_niawg_script(self):
        # A shortcut
        self.print_script(self.f['NIAWG']['Seq. 1 NIAWG-Script'])


    def print_script(self, script):
        """
        special formatting used for printing long scripts which are stored as normal numpy bytes.
        """
        print(Fore.GREEN,'\n--------------------------------------------')
        for x in script:
            print(x.decode('UTF-8'),end='')
        print('\n--------------------------------------------\n\n', Style.RESET_ALL)

    def __print_ds(self, ds, prefix):
        """
        Print dataset
        """
        if type(ds) != h5._hl.dataset.Dataset:
            raise TypeError('Tried to print non dataset as dataset.')
        else:
            if len(ds) > 0:
                if type(ds[0]) == np.bytes_:
                    print(' "',end='')
                    for x in ds:
                        print(x.decode('UTF-8'),end='')
                    print(' "',end='')
                elif type(ds[0]) in [np.uint8, np.uint16, np.uint32, np.uint64,
                                     np.int8, np.int16, np.int32, np.int64,
                                     np.float32, np.float64]:
                    for x in ds:
                        print(x,end=' ')
                else:
                    print(' type:', type(ds[0]), ds[0])
            print('')

    def print_pic_info(self):
        print('Number of Pictures:', self.pics.shape[0])
        print('Picture Dimensions:', self.pics.shape[1],'x',self.pics.shape[2])

    def get_basic_info(self):
        """
        Some quick easy to read summary info
        """
        self.print_pic_info()
        print('Variations:', len(self.key))
        print('Repetitions:', self.reps)
        print('Experiment started at (H:M:S) ', self.experiment_time, ' on (Y-M-D)', self.experiment_date)

    def get_info_arr(self):
        """
        Get basic run information in array form.
        """
        return 'Variations:', len(self.key), 'Repetitions:', self.reps, 'Start time (H:M:S) ', self.experiment_time, ' Start date (Y-M-D)', self.experiment_date

    def get_experiment_time_and_date(self):
        date = ''.join([x.decode('UTF-8') for x in self.f['Miscellaneous']['Run-Date']])
        time = ''.join([x.decode('UTF-8') for x in self.f['Miscellaneous']['Time-Of-Logging']])
        return time, date
