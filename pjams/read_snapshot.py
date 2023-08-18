import io, os, sys, inspect, types
sys.path.append('/Users/emigardiner/VICO/pjams-ionization/pjams/')

from zeusmp_snapshot_reader import read_zeusmp_snapshot
from zeusmp_snapshot_reader import ScaleFactors
from snapshot import snapshot 
from basic_snapshot import basic_snapshot

import numpy as np
import argparse

from datetime import datetime
from tqdm import tqdm


# DEFINE VARIABLES
# set this to be your own folder location
VICO_loc = '/Users/emigardiner/VICO/pjams-ionization'

frequencies = np.array([.01, .05, .1, .5, 1, 5.3, 23, 43, 100, 230 ]) # GHz
frequencies *= 10**9 # Hz
r_kpc = 1
scales = np.array([500,1000,2000,4000,8000,16000,32000]) # AU
heights = np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
                   10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000,
                   18000, 19000, 20000, 21000, 22000, 23000, 24000, 25000])


# DEFINE FUNCTIONS
def plot_intensity_read_flux(snapshot, frequencies=frequencies, r_kpc=r_kpc, scales=scales):
    Fluxes = np.zeros((3, len(frequencies)))
    snapshot.load_shock_variables()

    scalefluxfile = open((VICO_loc+'/RadioEmissions/FluxOutputs/scale_fluxes_'+snapshot.name+'.txt'), 'a')
    heightfluxfile = open((VICO_loc+'/RadioEmissions/FluxOutputs/height_fluxdensities_'+snapshot.name+'.txt'), 'a')
    ScaleFluxes = np.zeros((3, len(frequencies), len(scales)))
    zScaleFluxes = np.zeros((2, len(frequencies), len(scales)))
    HeightFluxDensities = np.zeros((2, len(frequencies), len(heights)))
    
    for f in range(len(frequencies)):
        snapshot.make_intensity_plots(nu=frequencies[f], r_kpc=r_kpc) # also loads intensity variables
        Fluxes[0,f] = snapshot.FnuA
        Fluxes[1,f] = snapshot.FnuB
        Fluxes[2,f] = snapshot.FnuC


        print(snapshot.name, 'Output at', datetime.now(), file=scalefluxfile)
        print('\n', snapshot.name, 'Output at', datetime.now(), file=scalefluxfile)
        print('nu (GHz) \tFluxA (mJy) \tFluxB (mJy) \tFluxC (mJy) \tx1 range (AU) \tx2 range (AU), \tx3 range (AU)', file=scalefluxfile)
        for s in range(len(scales)):
            FA, FB, FC, zFA, zFB = snapshot.calculate_scale_fluxes(scales[s], file=scalefluxfile, nu = frequencies[f], 
                                                                   r_kpc=1, printToScreen=False)
            ScaleFluxes[0,f,s] = FA
            ScaleFluxes[1,f,s] = FB
            ScaleFluxes[2,f,s] = FC
            zScaleFluxes[0,f,s] = zFA
            zScaleFluxes[1,f,s] = zFB
        for h in range(len(heights)):
            hFA = snapshot.calculate_height_fluxes(heights[h], file=heightfluxfile, nu = frequencies[f], 
                                                              r_kpc=1, width=1000, printToScreen=False)
            HeightFluxDensities[0,f,h] = hFA
                           
    scalefluxfile.close()
    np.savez((VICO_loc+'/Data/'+snapshot.name+'/'+snapshot.name+'_ScaleFluxes.npz'), 
             ScaleFluxes = ScaleFluxes, zScaleFluxes = zScaleFluxes, HeightFluxDensities = HeightFluxDensities)
    return ScaleFluxes, zScaleFluxes, HeightFluxDensities

def read_snap(snapname, snapnum, path = VICO_loc+'/Data/', read_zeusmp=False, 
              calculate_intensity_vars=False, calculate_flux=False):
    Snap = snapshot(snap=snapnum, name = snapname, path = path, read_zeusmp = read_zeusmp) 
    if(read_zeusmp):
        Snap.calculate_all_shock_variables()
        Snap.save_shock_variables()
    else:
        Snap.load_shock_variables()
    if(calculate_intensity_vars):
        for freq in tqdm(frequencies):
            Snap.calculate_all_intensity_variables(nu=freq, r_kpc=1)
        # Snap.calculate_all_intensity_variables(nu=23 * 10**9, r_kpc=1)
        # Snap.calculate_all_intensity_variables(nu=43 * 10**9, r_kpc=1)
        # Snap.calculate_all_intensity_variables(nu=100 * 10**9, r_kpc=1)
        # Snap.calculate_all_intensity_variables(nu=230 * 10**9, r_kpc=1)
        # Snap.calculate_all_intensity_variables(nu=.01 * 10**9, r_kpc=1)
        # Snap.calculate_all_intensity_variables(nu=.05 * 10**9, r_kpc=1)
        # Snap.calculate_all_intensity_variables(nu=.1 * 10**9, r_kpc=1)
        # Snap.calculate_all_intensity_variables(nu=.5 * 10**9, r_kpc=1)
        # Snap.calculate_all_intensity_variables(nu=1 * 10**9, r_kpc=1)
    if(calculate_flux):
        plot_intensity_read_flux(Snap)
        # Snap.scale_fluxes_const(frequencies=frequencies, const=.01, scales=scales)
        Snap.scale_fluxes_const(frequencies=frequencies, const=False, scales=scales)

# SETUP ARGS
def _setup_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('num', action='store', type=int,
                        help="snapshot number")
    parser.add_argument('--lr', '--lores', action='store_true', dest='lr_flag', default=False,
                        help='if low res')
    parser.add_argument('--hr', '--hires', action='store_true', dest='hr_flag', default=False,
                        help='if high res')
    parser.add_argument('--data_path', action='store', type=str, dest='data_path', 
                        default='/Users/emigardiner/VICO/pjams-ionization/Data/LOWRES',
                        help='path to snapshot simulation data')
    args = parser.parse_args()
    return args


# MAIN
def main():
    start_time = datetime.now()
    print("-----------------------------------------")
    print(f"starting at {start_time}")
    print("-----------------------------------------")

    # set up args
    args = _setup_argparse()

    # snapshot name
    name = 'Snap%03d_n' % args.num
    if args.lr_flag:
        name = name + 'lr'
    elif args.hr_flag:
        name = name = 'hr'
    print(f"making {name}")

    # read snapshot
    read_snap(name, args.num, path=args.data_path, 
              read_zeusmp=True, calculate_intensity_vars=True, calculate_flux=True, 
              )

    end_time = datetime.now()
    print("-----------------------------------------")
    print(f"ending at {end_time}")
    print(f"total time: {end_time - start_time}")
    print("-----------------------------------------")

if __name__ == "__main__":
    main()

