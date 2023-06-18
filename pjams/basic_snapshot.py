""" 
    Model for snapshot objects without all the data
"""


import sys, os
VICO_loc = '/Users/emigardiner/VICO/pjams-ionization'

sys.path.append(VICO_loc)
from zeusmp_snapshot_reader import read_zeusmp_snapshot
from zeusmp_snapshot_reader import ScaleFactors

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime


FREQS = np.array([0.01, 0.05, 0.1, 0.5, 1, 5.3, 23, 43, 100, 230])*10**9 # Hz

class basic_snapshot:

    def __init__(self, snap, name, npz_location=None):
        self.name = name

        if npz_location is None:
            npz_location = npz_location = VICO_loc+'/Data/'+self.name+'/'+self.name+'_simulation_data.npz'

    ##### DELTAS AND VOLUMES #####
    ##############################
    # calculates del1, del2, and del3 in centimeters
    def calculate_deltas(self):
        x1, x2, x3 = self.x1, self.x2, self.x3
        del1 = np.zeros(len(x1))
        del2 = np.zeros(len(x2))
        del3 = np.zeros(len(x3))
        for i in range(len(x1)): 
            if(i>0 and i < (len(x1) -1)):
                del1[i] = (x1[i+1] - x1[i-1])/2 
            elif(i > 0): #last element
                del1[i] = (x1[i] - x1[i-1]) 
            elif(i < len(x1)-1): # First element
                del1[i] = (x1[i+1] - x1[i])              
        for j in range(len(x2)): 
            if(j>0 and j < (len(x2) -1)):
                del2[j] = (x2[j+1] - x2[j-1])/2
            elif(j > 0): #last element
                del2[j] = (x2[j] - x2[j-1])
            elif(j < len(x2)-1): # First element
                del2[j] = (x2[j+1] - x2[j])       
        for k in range(len(x3)): 
            if(k>0 and k < (len(x3) -1)):
                del3[k] = (x3[k+1] - x3[k-1])/2
            elif(k > 0): #last element
                del3[k] = (x3[k] - x3[k-1])
            elif(k < len(x3)-1): # First element
                del3[k] = (x3[k+1] - x3[k])
        self.del1 = del1 * (1.496*10**13) #cm
        self.del2 = del2 * (1.496*10**13) #cm
        self.del3 = del3 * (1.496*10**13) #cm
        print('del1[], del2[], del3[] complete, units: cm')

    # calculate volumes array in cm^3
    def calculate_volumes(self):
        self.volumes = np.zeros((len(self.x1), len(self.x2), len(self.x3)))
        for i in range(len(self.x1)):
            for j in range(len(self.x2)):
                for k in range(len(self.x3)):
                    self.volumes[i,j,k] = self.del1[i] * self.del2[j] * self.del3[k]
        print('volumes[][][] complete, units: cm^3')

    ####################################################################################
    ################## load all intensity and flux variables ###########################
    ####################################################################################
    
    def load_intensity_variables(self, nu, r_kpc=1):
        # self.make_grids_arcsec
        data_path = (VICO_loc+'/Data/'+self.name+'/')
        loaded_intensity_arrays = np.load((data_path+self.name+'_'+str(nu/10**9)+'GHz_'+str(r_kpc)+'kpc_arrays.npz'))
        self.r_AU = 2.0624 * 10**8 * r_kpc # AU    
        self.r_cm = 3.0857 * 10**21 * r_kpc # cm
        self.emission_coefs = loaded_intensity_arrays['emission_coefs'] 
        self.absorption_coefs = loaded_intensity_arrays['absorption_coefs']
        self.tauA = loaded_intensity_arrays['tauA']
        self.tauB = loaded_intensity_arrays['tauB'] 
        self.tauC = loaded_intensity_arrays['tauC'] 
        self.source_functions = loaded_intensity_arrays['source_functions'] 
        self.InuA = loaded_intensity_arrays['InuA'] 
        self.InuB = loaded_intensity_arrays['InuB']
        self.InuC = loaded_intensity_arrays['InuC']
        self.FnuA = loaded_intensity_arrays['FnuA']
        self.FnuB = loaded_intensity_arrays['FnuB']
        self.FnuC = loaded_intensity_arrays['FnuC']


    # load all intensity variables AND SCALE FLUXES at a given frequency    
    def load_intensity_variables_const(self, nu, const, r_kpc=1):
        # self.make_grids_arcsec
        data_path = (VICO_loc+'/Data/'+self.name+'/')
        self.load_intensity_variables(nu, r_kpc)
        loaded_intensity_arrays = np.load((data_path+self.name+'_'+str(nu/10**9)+'GHz_'+str(r_kpc)+'kpc_arrays_const'+str(const)+'.npz'))
        self.InuA_const = loaded_intensity_arrays['InuA_const'] 
        self.FnuA_const = loaded_intensity_arrays['FnuA_const']
     
     
     # load scale fluxes for a given constant scale factor
    def load_scalefluxes_const(self, const, r_kpc=1):
        file = np.load(VICO_loc+'/Data/'+self.name+'/'+self.name+'_ScaleFluxes_const'+str(const)+'.npz')
        if(const):
            self.ScaleFluxes_const = file['ScaleFluxes_const']
            print('ScaleFluxes_const[f,s] loaded')
            return self.ScaleFluxes_const
        else:
            self.ScaleFluxes_ratio = file['ScaleFluxes_const'] 
            print('ScaleFluxes_ratio[f,s] loaded')  
            return self.ScaleFluxes_ratio
    
    def load_fluxes(self, debug=True):
        data_path = (VICO_loc+'/Data/')
        flux_data = np.load(data_path+self.name+'/'+self.name+'_ScaleFluxes.npz')
        self.ScaleFluxes = flux_data['ScaleFluxes']
        if debug: print('ScaleFluxes[f,s] loaded')  
        # self.zScaleFluxes = flux_data['zScaleFluxes']
        # self.HeightFluxDensities = flux_data['HeightFluxDensities']
        flux_data.close()
        ratio_data = np.load(data_path+self.name+'/'+self.name+'_ScaleFluxes_constFalse.npz')
        self.ScaleFluxes_ratio = ratio_data['ScaleFluxes_const']
        if debug: print('ScaleFluxes_ratio[f,s] loaded')  
        ratio_data.close()