import sys, os, inspect
import sys
sys.path.append('/Users/emigardiner/VICO/pjams-ionization/pjams-ionization/')

from zeusmp_snapshot_reader import read_zeusmp_snapshot
from zeusmp_snapshot_reader import ScaleFactors
from snapshot import *
from basic_snapshot import *
import plot

import numpy as np

FREQS = np.array([0.01, 0.05, 0.1, 0.5, 1, 5.3, 23, 43, 100, 230])*10**9 # Hz
VICO_loc = '/Users/emigardiner/VICO/pjams-ionization'



#################################################
#### z Segments
#################################################


def get_z_segments(x1, targets=np.linspace(0,26000,27)):
    zseg = np.zeros(len(targets), dtype=int)
    for tt, target in enumerate(targets):
        for ii, xx in enumerate(x1[:-1]):
            if x1[ii] < target and x1[ii+1] > target:
                # print('x1[%d]=%.2f, x1[%d]=%.2f'
                #     % (ii, x1, ii+1, x1_lr[ii+1]))
                zseg[tt] = ii
    if zseg[-1]==0:
        zseg[-1]= len(x1)-1

    zval = np.zeros(len(zseg))
    for ii in range(len(zseg)):
        zval[ii] = x1[zseg[ii]]

    zmid = np.zeros(len(zseg)-1)
    for ii in range(len(zmid)):
        zmid[ii] = np.average([zval[ii], zval[ii+1]])
    return zseg, zval, zmid

#################################################
#### Calculate and Save Flux Profiles
#################################################

def flux_in_slice(snap, InuA, zmin, zmax, norm=True):
    Fnu=0
    for ii in range(zmin, zmax):
        for jj in range(len(snap.x2)):
            Fnu += (
                InuA[ii,jj] * snap.del1[ii] * snap.del2[jj] 
                / snap.r_cm**2 * 206264.5**2 # mJy 
            )
    if norm: Fnu = Fnu/(snap.x1[zmax]-snap.x1[zmin]) * 1000 # au
    return(Fnu)
              
def save_fluxprof(snap, ratio=False):
    data_path = (VICO_loc+'/Data/'+snap.name+'/')
    if ratio:
        filename = (data_path+'flux_prof_ratio.npz') 
        np.savez(filename, flux_prof_ratio = snap.flux_prof_ratio)
    else: 
        filename = (data_path+'flux_prof.npz')
        np.savez(filename, flux_prof = snap.flux_prof)
    
def load_fluxprof(snap, ratio=False, debug=True):
    data_path = (VICO_loc+'/Data/'+snap.name+'/')
    if ratio:
        file= np.load((data_path+'flux_prof_ratio.npz'))
        snap.flux_prof_ratio = file['flux_prof_ratio']
    else: 
        file = np.load((data_path+'flux_prof.npz'))
        snap.flux_prof = file['flux_prof']
    if debug: print(snap, 'flux_profile loaded')


def make_flux_profiles(snap, zseg, ratio=False, debug=False, norm=True, save=False):
    flux_profile = np.zeros((len(FREQS), len(zseg)-1))
    for ff in [5,9]:
        if ratio:
            snap.load_intensity_variables_const(FREQS[ff], False)
            InuA = snap.InuA_const
        else:
            snap.load_intensity_variables(FREQS[ff])
            InuA = snap.InuA
        for ii in range(len(zseg)-1):
            flux_profile[ff,ii] = flux_in_slice(snap, InuA, zseg[ii], zseg[ii+1], norm)
        if debug: 
            print('%s avg flux vs height arrays made (nu=%s)' 
                  % (snap.name, str(FREQS[ff]/10**9)))
    if ratio:
        snap.flux_prof_ratio = flux_profile
    else: 
        snap.flux_prof= flux_profile
    if save: 
        save_fluxprof(snap, ratio)
    if debug:
        print('flux profiles saved')


#################################################
#### Prep Resolution Snapshots
#################################################

def prep_res_snapshots(rnums = np.array([9, 21, 39, 54])):

    snaps_lr = np.empty_like(rnums, dtype=snapshot)
    snaps_mr = np.empty_like(rnums, dtype=snapshot)
    snaps_hr = np.empty_like(rnums, dtype=snapshot)

    # get full snapshot data from one at each res
    lr_snap = snapshot(snap=rnums[0], name='Snap009_nlr', read_zeusmp=False)
    mr_snap = snapshot(snap=rnums[0], name='Snap009_n', read_zeusmp=False)
    hr_snap = snapshot(snap=rnums[0], name='Snap009_nhr', read_zeusmp=False)

    # get basic snapshots for the rest
    for ii, num in enumerate(rnums):
        snaps_lr[ii] = basic_snapshot(snap=num, name = ('Snap%03d_nlr' % num))
        snaps_mr[ii] = basic_snapshot(snap=num, name = ('Snap%03d_n' % num))
        snaps_hr[ii] = basic_snapshot(snap=num, name = ('Snap%03d_nhr' % num))

    # assign res positions to the rest
    for ii, snap in enumerate(snaps_lr):
        snap.x1 = lr_snap.x1
        snap.x2 = lr_snap.x2
        snap.x3 = lr_snap.x3
        snap.calculate_deltas()

    for ii, snap in enumerate(snaps_mr):
        snap.x1 = mr_snap.x1
        snap.x2 = mr_snap.x2
        snap.x3 = mr_snap.x3
        snap.calculate_deltas()

    for ii, snap in enumerate(snaps_hr):
        snap.x1 = hr_snap.x1
        snap.x2 = hr_snap.x2
        snap.x3 = hr_snap.x3
        snap.calculate_deltas()

    return snaps_lr, snaps_mr, snaps_hr

def prep_spec_snapshots(rnums = np.array([9, 21, 39, 54]), debug=False):

    snaps_lr = np.empty_like(rnums, dtype=snapshot)
    snaps_mr = np.empty_like(rnums, dtype=snapshot)
    snaps_hr = np.empty_like(rnums, dtype=snapshot)

    # get basic snapshots for the rest
    for ii, num in enumerate(rnums):
        snaps_lr[ii] = basic_snapshot(snap=num, name = ('Snap%03d_nlr' % num))
        snaps_mr[ii] = basic_snapshot(snap=num, name = ('Snap%03d_n' % num))
        snaps_hr[ii] = basic_snapshot(snap=num, name = ('Snap%03d_nhr' % num))

    # assign res positions to the rest
    for snaps in [snaps_lr, snaps_mr, snaps_hr]:
        for ii, snap in enumerate(snaps):
            snap.load_fluxes(debug=debug)
            # snap.load_scalefluxes_const(False)

    return snaps_lr, snaps_mr, snaps_hr