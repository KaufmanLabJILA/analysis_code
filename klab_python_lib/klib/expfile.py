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

#     def get_key(self):
#         """
#         :param file:
#         :return:
#         """
#         keyNames = []
#         keyValues = []
#         foundOne = False
#         for var in self.f['Master-Parameters']['Seq #1 Variables']:
#             if not self.f['Master-Parameters']['Seq #1 Variables'][var].attrs['Constant']:
#                 foundOne = True
#                 keyNames.append(var)
#                 keyValues.append(arr(self.f['Master-Parameters']['Seq #1 Variables'][var]))
#         if foundOne:
#             if len(keyNames) > 1:
#                 return keyNames, arr(transpose(arr(keyValues)))
#             else:
#                 return keyNames[0], arr(keyValues[0])
#         else:
#             return 'No-Variation', arr([1])


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
            elif o == 'Master-Script' or o == "DDS-Script" or o == "Moog-Script":
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
                for file in self.f['Master-Parameters']['Functions'][func]:
                   self.print_script(self.f['Master-Parameters']['Functions'][func][file])
            print('')

    def print_master_script(self):
        # A shortcut
        self.print_script(self.f['Master-Parameters']['Master-Script'])

#     def print_niawg_script(self):
#         # A shortcut
#         self.print_script(self.f['NIAWG']['Seq. 1 NIAWG-Script'])

    def print_variables(self):
        self.__print_hdf5_obj(self.f['Master-Parameters']['Variables'],'')
        
    def print_DDS_script(self):
        self.print_script(self.f['DDS-Parameters']['DDS-Script'])
        
    def print_moog_script(self):
        self.print_script(self.f['Moog-Parameters']['Moog-Script'])

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
