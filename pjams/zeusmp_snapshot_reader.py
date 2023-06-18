# -*- coding: utf-8 -*-
import sys, os
VICO_loc = '/scratch/ecg6wm/VICO'
sys.path.append(VICO_loc)

import numpy as np
import fnmatch
from pyhdf import SD as h4SD
import os


def convert_zeusmp_to_npy (filename, path='./'):
    """
    Given an ZEUS-MP HDF4 snapshot, convert the data to CGS units and then
    return them as numpy arrays.

    This function reads a single snapshot; it is up to the user to splice
    together snapshots from different MPI ranks (if applicable).

    """

    # read file
    f = h4SD.SD(path+filename)
    x3 = f.select('fakeDim0')[:]
    x2 = f.select('fakeDim1')[:]
    x1 = f.select('fakeDim2')[:]

    # Data-set-2: v1
    # Data-set-3: v2
    # Data-set-4: v3
    # Data-set-5: b1
    # Data-set-6: b2
    # Data-set-7: b3
    # Data-set-8: d
    # Data-set-9: e
    v1 = f.select('Data-Set-2')
    v1 = v1[:].squeeze().T
    v2 = f.select('Data-Set-3')
    v2 = v2[:].squeeze().T
    v3 = f.select('Data-Set-4')
    v3 = v3[:].squeeze().T
    b1 = f.select('Data-Set-5')
    b1 = b1[:].squeeze().T
    b2 = f.select('Data-Set-6')
    b2 = b2[:].squeeze().T
    b3 = f.select('Data-Set-7')
    b3 = b3[:].squeeze().T
    d  = f.select('Data-Set-8')
    d  = d[:].squeeze().T
    e  = f.select('Data-Set-9')
    e  = e[:].squeeze().T

    # close file
    f.end()

    return x1, x2, x3, {'d':d, 'e':e, 'v1':v1, 'v2':v2, 'v3':v3, 'b1':b1, 'b2':b2, 'b3':b3}

def read_zeusmp_snapshot(run='OP', snap=0, path='./'):
    """Read a ZEUS-MP snapshot (complete, across all MPI ranks).
       The assumed format of the snapshot is `hdfIIAABBCC.DDD`, where `II` is a two character
       ID tag given by keyword `run`, `DDD` is the 3-digit snapshot number given by keyword `snap`,
       while `AA`, `BB`, and `CC` represent the zero-based MPI index in the 3rd, 2nd, and 1st dimension.

    """

    from itertools import product
    assert isinstance(snap,int)

    # figure out how many files there are in total
    files = fnmatch.filter(os.listdir(path), 'hdf'+run+'*.{:03d}'.format(snap))
    nfiles = len(files)
    if nfiles == 0:
        raise Exception('Snapshot {} not found!'.format(snap))
    files = sorted(files)

    # first pass; we want to know how many MPI slices there are in each direction
    nmpi1, nmpi2, nmpi3 = 0, 0, 0
    for ff in files:
        # split filename to discover MPI indices; this is hard-wired and based on the assumed formatting (see docstring above)
        pre, suf = ff.split(run)
        mid, suf = suf.split('.')
        i1, i2, i3 = int(mid[0:2]), int(mid[2:4]), int(mid[4:6]) # MPI rank indices
        # determine total number of slices in each direction
        nmpi1, nmpi2, nmpi3 = max(nmpi1, i1), max(nmpi2, i2), max(nmpi3, i3)

    nmpi1 += 1
    nmpi2 += 1
    nmpi3 += 1
    print("Number of MPI tiles in each direction:",nmpi1,nmpi2,nmpi3)

    var_dict = {'d':[], 'e':[], 'v1':[], 'v2':[], 'v3':[], 'b1':[], 'b2':[], 'b3':[]}
    x1, x2, x3, q = convert_zeusmp_to_npy(files[0],path=path) # need to read one file to get dimensions
    n1, n2, n3 = q['d'].shape
    # all files are (reasonably) assumed to have the same dimensions
    # initialize large empty array for each variable
    for k in var_dict:
        var_dict[k] = np.empty((nmpi1*n1,nmpi2*n2,nmpi3*n3),order='F')
    print("Shape of each array in ONE file:",(n1,n2,n3))
    print("Shape of each variable array:",var_dict['d'].shape)

    # second pass; now we read data from the files
    n = 1
    x1_slices, x2_slices, x3_slices = [], [], []
    # the order of this nested iterable is last to first index (i.e. `range(nmpi3)` is only looped over once)
    for i,j,k in product(range(nmpi1),range(nmpi2),range(nmpi3)):
        # reconstruct file name
        ff = 'hdf'+run+'{0:02d}{1:02d}{2:02d}.{3:03d}'.format(i,j,k,snap)
        print('Reading '+ff+'... ({:d}/{:d})'.format(n,nfiles))
        x1, x2, x3, q = convert_zeusmp_to_npy(ff,path=path)
        n += 1
        # build up length arrays
        if j == 0 and k == 0:
            x1_slices.append(x1)
        if k == 0 and i == 0:
            x2_slices.append(x2)
        if i == 0 and j == 0:
            x3_slices.append(x3)
        # offsets into variable arrays
        ilo, ihi = i * n1, (i+1) * n1
        jlo, jhi = j * n2, (j+1) * n2
        klo, khi = k * n3, (k+1) * n3
        # build up variable arrays
        for k in var_dict:
            var_dict[k][ilo:ihi,jlo:jhi,klo:khi] = q[k]

    # concatenate slices into single arrays
    x1_all = np.concatenate(x1_slices)
    x2_all = np.concatenate(x2_slices)
    x3_all = np.concatenate(x3_slices)

    return x1_all, x2_all, x3_all, var_dict

class EmptyObject(object):
    def __init__(self):
        pass

class ScaleFactors(object):

    def __init__(self):
        # create attributes and assign default values for variable stored in the snapshot
        self.__vars__ = ['l','t','d','e','v','B']
        self.units = EmptyObject()
        for k in self.__vars__:
            setattr(self, k, 1.0)
            setattr(self.units, k, "--")
        self.v = self.l / self.t # velocity
        # derived variables (not in the snapshot)
        self.T = self.e / self.d # temperature (propto e1/d)

    def print(self):
        # print out each scale factor and associated unit
        print("Var |  Factor   |  Unit")
        print("--------------------------")
        for k in self.__vars__:
            print("{0:3s} | {1:06.3e} | {2:8s}".format(k,getattr(self,k),getattr(self.units,k)))
