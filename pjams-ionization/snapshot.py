import sys, os, inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir)

import sys, os
VICO_loc = '/Users/emigardiner/VICO/pjams-ionization'
sys.path.append(VICO_loc)
from zeusmp_snapshot_reader import read_zeusmp_snapshot
from zeusmp_snapshot_reader import ScaleFactors

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime


class Constants:
    # constant class variables
    m_H = 1.4 * 1.67 * 10**-24 # g   
    mu = 1
    Bdivk = 157800 # B/k [K]
    c = 2.998 * 10**10 # cm/s
    h_bar = 1.0546 * 10**-27 # cm^2 g/s
    e = 4.8032*10**-10
    c = 2.998 * 10**10 # cm/s
    h = 6.6261 * 10**-27 # cm^2 g/s
    m_e = 9.10938 *10**-28 # g
    k_B = 1.3807 * 10**-16 # cm^2 g s^-2 K^-1   = erg/K
    
class snapshot:
    def __init__(self, snap, name, path = './', scale_x_AU = 0.012, scale_d_gcm3 = 10**-12, 
                 scale_v_kms = .6, scale_b_G = 4.7, read_zeusmp = True, npz_location = 'unspecified'):
        self.name = name
        if(npz_location == 'unspecified'):
            npz_location = VICO_loc+'/Data/'+self.name+'/'+self.name+'_simulation_data.npz'
        if(read_zeusmp):
            self.x1, self.x2, self.x3, self.q = read_zeusmp_snapshot(snap=snap, path = path)
            self.x1 *= scale_x_AU #AU
            self.x2 *= scale_x_AU #AU
            self.x3 *= scale_x_AU #AU
            self.q['d'] *= scale_d_gcm3 #g/cm^3
            self.q['v1'] *= scale_v_kms #km/s
            self.q['v2'] *= scale_v_kms #km/s
            self.q['v3'] *= scale_v_kms #km/s
            self.q['b1'] /= scale_b_G #Gtask
            self.q['b2'] /= scale_b_G #G
            self.q['b3'] /= scale_b_G #G
            np.savez(npz_location, 
                     x1=self.x1, x2=self.x2, x3=self.x3, q=self.q)
            print('Simulation data saved to npz file.')
        else:
            simulation_data = np.load(npz_location, allow_pickle=True)
            self.x1 = simulation_data['x1']
            self.x2 = simulation_data['x2']
            self.x3 = simulation_data['x3']
            self.q = simulation_data['q'][()]
            print('Simulation data loaded from npz file.')
        print(name+' initialized')

        
    # make grids for plotting, with axes in AU
    # set midpoint indices mid1, mid2, and mid3    
    def make_grids(self):
        print('Dimensions')
        print('x1:', len(self.x1), 'x2:', len(self.x2), 'x3:', len(self.x3))
        self.X2_1, self.X1_2 = np.meshgrid(self.x2, self.x1)
        self.X3_1, self.X1_3 = np.meshgrid(self.x3, self.x1)
        self.X3_2, self.X2_3 = np.meshgrid(self.x3, self.x2)
        self.X2_3, self.X3_2 = np.meshgrid(self.x2, self.x3)
        print('X1_2:', self.X1_2.shape, 'X2_1:', self.X2_1.shape) # z-x
        print('X1_3:', self.X1_3.shape, 'X3_1:', self.X3_1.shape) # z-y
        print('X2_3:', self.X2_3.shape, 'X3_2:', self.X3_2.shape) # x-y
        self.mid1 = (int)((len(self.x1))/2 - 1)
        self.mid2 = (int)((len(self.x2))/2 - 1)
        self.mid3 = (int)((len(self.x3))/2 - 1)
        self.X1_2v, self.X2_1v = np.meshgrid(self.x1, self.x2)
        print('mid1 =', self.mid1, 'mid2 =', self.mid2, 'mid3 =', self.mid3)
        print('AU grid complete')
    
    ##### DELTAS AND VOLUMES #####
    ##############################
    # calculates del1, del2, and del3 in centimeters
    def calculate_deltas(self):
        self.del1 = np.zeros(len(self.x1))
        self.del2 = np.zeros(len(self.x2))
        self.del3 = np.zeros(len(self.x3))
        for i in range(len(self.x1)): 
            if(i>0 and i < (len(self.x1) -1)):
                self.del1[i] = (self.x1[i+1] - self.x1[i-1])/2 
            elif(i > 0): #last element
                self.del1[i] = (self.x1[i] - self.x1[i-1]) 
            elif(i < len(self.x1)-1): # First element
                self.del1[i] = (self.x1[i+1] - self.x1[i])              
        for j in range(len(self.x2)): 
            if(j>0 and j < (len(self.x2) -1)):
                self.del2[j] = (self.x2[j+1] - self.x2[j-1])/2
            elif(j > 0): #last element
                self.del2[j] = (self.x2[j] - self.x2[j-1])
            elif(j < len(self.x2)-1): # First element
                self.del2[j] = (self.x2[j+1] - self.x2[j])       
        for k in range(len(self.x3)): 
            if(k>0 and k < (len(self.x3) -1)):
                self.del3[k] = (self.x3[k+1] - self.x3[k-1])/2
            elif(k > 0): #last element
                self.del3[k] = (self.x3[k] - self.x3[k-1])
            elif(k < len(self.x3)-1): # First element
                self.del3[k] = (self.x3[k+1] - self.x3[k])
        self.del1 *= (1.496*10**13) #cm
        self.del2 *= (1.496*10**13) #cm
        self.del3 *= (1.496*10**13) #cm
        print('del1[], del2[], del3[] complete, units: cm')

    # calculate volumes array in cm^3
    def calculate_volumes(self):
        self.volumes = np.zeros((len(self.x1), len(self.x2), len(self.x3)))
        for i in range(len(self.x1)):
            for j in range(len(self.x2)):
                for k in range(len(self.x3)):
                    self.volumes[i,j,k] = self.del1[i] * self.del2[j] * self.del3[k]
        print('volumes[][][] complete, units: cm^3')

    
    ############## SHOCK VARIABLES ###############################
    
    
    ##### VELOCITIES ######
    #######################
    # Function for calculating velocity differences
    # Inputs: comparison cell name (face), current cell coordinates (i,j,k)
    # Output: v_comparison cell - v_current cell if the current cell and comparison cell 
    #   are valid; 0, otherwise 
    def vel_diff(self, face, i, j, k):
        if(face == 'm1'):
            if(i>0):
                return self.q['v1'][i-1,j,k] - self.q['v1'][i,j,k]
        elif(face == 'p1'):
            if(i<len(self.x1)-1):
                return self.q['v1'][i+1,j,k] - self.q['v1'][i,j,k]
        elif(face == 'm2'):
            if(j>0):
                return self.q['v2'][i,j-1,k] - self.q['v2'][i,j,k]
        elif(face == 'p2'):
            if(j<len(self.x2)-1):
                return self.q['v2'][i,j+1,k] - self.q['v2'][i,j,k]
        elif(face == 'm3'):
            if(k>0):
                return self.q['v3'][i,j,k-1] - self.q['v3'][i,j,k]
        elif(face == 'p3'):
            if(k<len(self.x3)-1):
                return self.q['v3'][i,j,k+1] - self.q['v3'][i,j,k]
        return 0
    
    #checks if velocity difference is incoming or not, returns self if it is, returns 0 if not
    def vel_in(self, face, i, j, k):
        v = self.vel_diff(face, i, j, k)
        if((face == 'm1' or face == 'm2' or face=='m3') and v<0): v = 0
        if((face == 'p1' or face == 'p2' or face=='p3') and v>0): v = 0
        return v
    
    
    ##### TEMPERATURES #####
    ########################
    # Function for calculating shock temperatures
    # input: v_s in km/s
    # output: shock temperature in K
    # does not check for M>>1
    def temp(self, v_s):
        return ((1.38 * 10**7) * (Constants.mu / (1.4/2.3)) * (v_s / 1000)**2)

    # same but takes input of face, i, j, k
    def temp2(self, face, i, j, k):
        v_s = self.vel_in(face,i,j,k)
        return self.temp(v_s) 
    
    # Calculate face temperatures in K
    def calculate_face_temps(self):
        self.temperatures_m1 = np.zeros((len(self.x1), len(self.x2), len(self.x3)))
        self.temperatures_p1 = np.zeros((len(self.x1), len(self.x2), len(self.x3)))
        self.temperatures_m2 = np.zeros((len(self.x1), len(self.x2), len(self.x3)))
        self.temperatures_p2 = np.zeros((len(self.x1), len(self.x2), len(self.x3)))
        self.temperatures_m3 = np.zeros((len(self.x1), len(self.x2), len(self.x3)))
        self.temperatures_p3 = np.zeros((len(self.x1), len(self.x2), len(self.x3)))
        for i in range(len(self.x1)):
            for j in range(len(self.x2)):
                for k in range(len(self.x3)):
                    self.temperatures_m1[i,j,k] = self.temp2('m1',i,j,k)
                    self.temperatures_p1[i,j,k] = self.temp2('p1',i,j,k)
                    self.temperatures_m2[i,j,k] = self.temp2('m2',i,j,k)
                    self.temperatures_p2[i,j,k] = self.temp2('p2',i,j,k)
                    self.temperatures_m3[i,j,k] = self.temp2('m3',i,j,k)
                    self.temperatures_p3[i,j,k] = self.temp2('p3',i,j,k)
        print('temperatures_side[][][] complete, units: K')
        
    # Calculate average temperature, weighting the shock temperature of each face by the face-adjacent 
    # flux. Requires face temperatures arrays be instantiated
    def flux_weighted_temp(self,i,j,k):
        flux_tot = 0
        t_tot = 0
        if(i>0 and self.temperatures_m1[i,j,k]>0):
            t_tot += np.abs(self.temperatures_m1[i,j,k] * self.q['d'][i-1,j,k] * self.vel_in('m1',i,j,k))
            flux_tot += np.abs(self.q['d'][i-1,j,k] * self.vel_in('m1',i,j,k))
        if(i<len(self.x1)-1  and self.temperatures_p1[i,j,k]>0):
            t_tot += np.abs(self.temperatures_p1[i,j,k] * self.q['d'][i+1,j,k] * self.vel_in('p1',i,j,k))
            flux_tot += np.abs(self.q['d'][i+1,j,k] * self.vel_in('p1',i,j,k))
        if(j>0  and self.temperatures_m2[i,j,k]>0):
            t_tot += np.abs(self.temperatures_m2[i,j,k] * self.q['d'][i,j-1,k] * self.vel_in('m2',i,j,k))
            flux_tot += np.abs(self.q['d'][i,j-1,k] * self.vel_in('m2',i,j,k))
        if(j<len(self.x2)-1  and self.temperatures_p2[i,j,k]>0):
            t_tot += np.abs(self.temperatures_p2[i,j,k] * self.q['d'][i,j+1,k] * self.vel_in('p2',i,j,k))
            flux_tot += np.abs(self.q['d'][i,j+1,k] * self.vel_in('p2',i,j,k))
        if(k>0  and self.temperatures_m3[i,j,k]>0):
            t_tot += np.abs(self.temperatures_m3[i,j,k] * self.q['d'][i,j,k-1] * self.vel_in('m3',i,j,k))
            flux_tot += np.abs(self.q['d'][i,j,k-1] * self.vel_in('m3',i,j,k))
        if(k<len(self.x3)-1  and self.temperatures_p3[i,j,k]>0):
            t_tot += np.abs(self.temperatures_p3[i,j,k] * self.q['d'][i,j,k+1] * self.vel_in('p3',i,j,k))
            flux_tot += np.abs(self.q['d'][i,j,k+1] * self.vel_in('p3',i,j,k))
        if (flux_tot>0):
            return (t_tot/flux_tot) 
        else:
            return(0)
    
    # Calculate flux weighted temperatures
    def calculate_fw_temps(self):
        self.temperatures = np.zeros((len(self.x1), len(self.x2), len(self.x3)))
        for i in range(len(self.x1)):
            for j in range(len(self.x2)):
                for k in range(len(self.x3)):
                    self.temperatures[i,j,k] = self.flux_weighted_temp(i,j,k)
        print('temperatures[][][] complete, units: K')
    
    
    ##### IONIZATION FRACTIONS #####  
    ################################
    # Function for calculation ionization fraction
    # input: temperature
    # output: n(H+)/[n(H) + n(H+)]
    def ion_fraction(self,temp):
        if (temp<300):
            return 0
        HdivHplus = (2**4 / 3**(3/2) * (1/137.04)**3 * Constants.Bdivk/temp * np.exp(Constants.Bdivk/temp))
        return (HdivHplus + 1)**(-1)

    # same but takes input of face, i, j, k
    def ion_fraction2(self,face,i,j,k):
        temp = self.temp2(face,i,j,k)
        return self.ion_fraction(temp)    

    def calculate_face_ionfracs(self):
        # Calculate all ionization fractions n(A+)/[n(A+) + n(A+)]            
        self.ionf_m1 = np.zeros((len(self.x1), len(self.x2), len(self.x3)))
        self.ionf_p1 = np.zeros((len(self.x1), len(self.x2), len(self.x3)))
        self.ionf_m2 = np.zeros((len(self.x1), len(self.x2), len(self.x3)))
        self.ionf_p2 = np.zeros((len(self.x1), len(self.x2), len(self.x3)))
        self.ionf_m3 = np.zeros((len(self.x1), len(self.x2), len(self.x3)))
        self.ionf_p3 = np.zeros((len(self.x1), len(self.x2), len(self.x3)))
        for i in range(len(self.x1)):
            for j in range(len(self.x2)):
                for k in range(len(self.x3)):
                    self.ionf_m1[i,j,k] = self.ion_fraction2('m1',i,j,k)
                    self.ionf_p1[i,j,k] = self.ion_fraction2('p1',i,j,k)
                    self.ionf_m2[i,j,k] = self.ion_fraction2('m2',i,j,k)
                    self.ionf_p2[i,j,k] = self.ion_fraction2('p2',i,j,k)
                    self.ionf_m3[i,j,k] = self.ion_fraction2('m3',i,j,k)
                    self.ionf_p3[i,j,k] = self.ion_fraction2('p3',i,j,k)
        print('ionf_side[][][] complete, units: none')
    
    # Calculate flux-weighted ionization fractions
    def flux_weighted_ionf(self,i,j,k):
        flux_tot = 0
        if_tot = 0
        if(i>0):
            if_tot += np.abs(self.ionf_m1[i,j,k] * self.q['d'][i-1,j,k] * self.vel_in('m1',i,j,k))
            flux_tot += np.abs(self.q['d'][i-1,j,k] * self.vel_in('m1',i,j,k))
        if(i<len(self.x1)-1):
            if_tot += np.abs(self.ionf_p1[i,j,k] * self.q['d'][i+1,j,k] * self.vel_in('p1',i,j,k))
            flux_tot += np.abs(self.q['d'][i+1,j,k] * self.vel_in('p1',i,j,k))
        if(j>0):
            if_tot += np.abs(self.ionf_m2[i,j,k] * self.q['d'][i,j-1,k] * self.vel_in('m2',i,j,k))
            flux_tot += np.abs(self.q['d'][i,j-1,k] * self.vel_in('m2',i,j,k))
        if(j<len(self.x2)-1):
            if_tot += np.abs(self.ionf_p2[i,j,k] * self.q['d'][i,j+1,k] * self.vel_in('p2',i,j,k))
            flux_tot += np.abs(self.q['d'][i,j+1,k] * self.vel_in('p2',i,j,k))
        if(k>0):
            if_tot += np.abs(self.ionf_m3[i,j,k] * self.q['d'][i,j,k-1] * self.vel_in('m3',i,j,k))
            flux_tot += np.abs(self.q['d'][i,j,k-1] * self.vel_in('m3',i,j,k))
        if(k<len(self.x3)-1):
            if_tot += np.abs(self.ionf_p1[i,j,k] * self.q['d'][i,j,k+1] * self.vel_in('p3',i,j,k))
            flux_tot += np.abs(self.q['d'][i,j,k+1] * self.vel_in('p3',i,j,k))
        if(flux_tot>0):
            return (if_tot/flux_tot)
        else: return 0
    
    # Calculate flux-weighted ionization fractions for each cell
    # requires face ion fraction arrays be instantiated
    def calculate_ion_fracs(self):
        self.ion_fractions = np.zeros((len(self.x1), len(self.x2), len(self.x3)))
        for i in range(len(self.x1)):
            for j in range(len(self.x2)):
                for k in range(len(self.x3)):
                    self.ion_fractions[i,j,k] = self.flux_weighted_ionf(i,j,k)
        print('ion_fractions[][][] complete, units: none')
        
    # Calculate n_H+ volume densities
    def calculate_volume_densities(self):
        self.volume_densities = np.zeros((len(self.x1), len(self.x2), len(self.x3)))
        for i in range(len(self.x1)):
            for j in range(len(self.x2)):
                for k in range(len(self.x3)):
                    self.volume_densities[i,j,k] = self.ion_fractions[i,j,k] * self.q['d'][i,j,k] / Constants.m_H
        print('volume_densities[][][] complete, units: cm^-3')
 


    ################### Cooling times ##########
    def cooling_time(self, T, n_H, chi):
        if(T>10**5):
            C = 1.1*10**-22 # erg cm^3 s^-1
            numer = 3 * Constants.k_B * T**1.7
            denom = 2 * C * chi * n_H * 10**4.2
        else:
            D = 3.98*10**-30
            numer = (3/2) * Constants.k_B * T**-0.6
            denom = D * chi * n_H 
        if(denom == 0): 
            return np.inf
        return numer/denom

    def cell_cooling_time(self, i, j, k):
        T = self.temperatures[i,j,k]
        n_H = self.q['d'][i,j,k] / Constants.m_H
        chi = self.ion_fractions[i,j,k]
        return(self.cooling_time(T, n_H, chi))
    
    def calculate_cooling_times(self):
        self.cooling_times = np.zeros((len(self.x1), len(self.x2), len(self.x3)))
        for i in range(len(self.x1)):
            for j in range(len(self.x2)):
                for k in range(len(self.x3)):
                    self.cooling_times[i,j,k] = self.cell_cooling_time(i,j,k)
        print('cooling_times[][][] complete, units: s')

        data_path = (VICO_loc+'/Data/'+self.name+'/')
        np.savez((data_path+self.name+'_cooling_times.npz'), 
                 cooling_times = self.cooling_times)
        print('cooling_times[][][] saved, units: s')

    def load_cooling_times(self):
        data_path = (VICO_loc+'/Data/'+self.name+'/')
        cooling_times_file = np.load(data_path+self.name+'_cooling_times.npz')
        self.cooling_times = cooling_times_file['cooling_times']
    
    
    ############### Flow times ########################

    def delta_s(self, face, i,j,k):
        if((face == 'm1') or (face=='p1')): return self.del1[i]
        elif((face == 'm2') or (face=='p2')): return self.del2[j]
        elif((face == 'm3') or (face == 'p3')): return self.del3[k]

    def cell_flow_time(self, i,j,k):
        numer = 0
        denom = 0
        if(i>0):
            v = np.abs(self.vel_in('m1',i,j,k))
            ds = self.del1[i]
            if(v>0):
                numer += np.abs(ds/v * self.q['d'][i-1,j,k] * v)
                denom += np.abs(self.q['d'][i-1,j,k] * v)
        if(i<len(self.x1)-1):
            v = np.abs(self.vel_in('p1',i,j,k))
            ds = self.del1[i]
            if(v>0):
                numer += np.abs(ds/v * self.q['d'][i+1,j,k] * v)
                denom += np.abs(self.q['d'][i+1,j,k] * v)
        if(j>0):
            v = np.abs(self.vel_in('m2',i,j,k))
            ds = self.del2[j]
            if(v>0):
                numer += np.abs(ds/v * self.q['d'][i,j-1,k] * v)
                denom += np.abs(self.q['d'][i,j-1,k] * v)
        if(j<len(self.x2)-1):
            v = np.abs(self.vel_in('p2',i,j,k))
            ds = self.del2[j]
            if(v>0):
                numer += np.abs(ds/v * self.q['d'][i,j+1,k] * v)
                denom += np.abs(self.q['d'][i,j+1,k] * v)
        if(k>0):
            v = np.abs(self.vel_in('m3',i,j,k))
            ds = self.del3[k]
            if(v>0):
                numer += np.abs(ds/v * self.q['d'][i,j,k-1] * v)
                denom += np.abs(self.q['d'][i,j,k-1] * v)
        if(k<len(self.x3)-1):
            v = np.abs(self.vel_in('p3',i,j,k))
            ds = self.del3[k]
            if(v>0):
                numer += np.abs(ds/v * self.q['d'][i,j,k+1] * v)
                denom += np.abs(self.q['d'][i,j,k+1] * v) 
        # unconverted units: cm/(km/s) 
        if(denom>0 and numer>0):
            return 2*numer/denom /100/1000 # s
        else:
            return 10**-10 #no shocks in cell

    ### calculate for whole Snapshot ###
    def calculate_flow_times(self):
        self.flow_times = np.zeros((len(self.x1), len(self.x2), len(self.x3)))
        for i in range(len(self.x1)):
            for j in range(len(self.x2)):
                for k in range(len(self.x3)):
                    self.flow_times[i,j,k] = self.cell_flow_time(i,j,k)
        print('flow_times[][][] complete, units: s')

        data_path = (VICO_loc+'/Data/'+self.name+'/')
        np.savez((data_path+self.name+'_flow_times.npz'), 
                 flow_times = self.flow_times)
        print('flow_times[][][] saved, units: s')
        
    def load_flow_times(self):
        data_path = (VICO_loc+'/Data/'+self.name+'/')
        flow_times_file = np.load(data_path+self.name+'_flow_times.npz')
        self.flow_times = flow_times_file['flow_times']
    

    
    ####################################
    ####################################
    # Calculate all shock variables, independent of distance and frequency
    def calculate_all_shock_variables(self):
        self.calculate_deltas()               # del1[], del2[]. del3[]
        self.calculate_volumes()              # volumes[][][]
        self.calculate_face_temps()           # temperatures_side[][][]
        self.calculate_fw_temps()             # temperatures[][][]
        self.calculate_face_ionfracs()        # ionf_side[][][]
        self.calculate_ion_fracs()            # ion_fractions[][][]
        self.calculate_volume_densities()     # volume_densities[][][]
        self.calculate_cooling_times()        # cooling_times[][][]
        self.calculate_flow_times()           # flow_times[][][]

        
    # Save shock variable arrays with np.savez
    def save_shock_variables(self):
        data_path = (VICO_loc+'/Data/'+self.name+'/')
        np.savez((data_path+self.name+'_shock_arrays.npz'), del1 = self.del1, del2 = self.del2, del3 = self.del3, 
                 volumes = self.volumes, temperatures = self.temperatures, ion_fractions = self.ion_fractions, 
                 volume_densities = self.volume_densities)
        print('del1[], del2[], del3[] saved, units: cm')
        print('volumes[][][] saved, units: cm^3')
        print('temperatures[][][] saved, units: K')        
        print('ion_fractions[][][] saved, units: none')        
        print('volume_densities[][][] saved, units: cm^-3')
        
        
    # Load shock variables from saved array with np.load
    def load_shock_variables(self, with_times = True):
        self.make_grids()
        data_path = (VICO_loc+'/Data/'+self.name+'/')
        loaded_shock_arrays = np.load(data_path+self.name+'_shock_arrays.npz')
        self.del1 = loaded_shock_arrays['del1']
        self.del2 = loaded_shock_arrays['del2']
        self.del3 = loaded_shock_arrays['del3']
        self.volumes = loaded_shock_arrays['volumes']
        self.temperatures = loaded_shock_arrays['temperatures']
        self.ion_fractions = loaded_shock_arrays['ion_fractions']
        self.volume_densities = loaded_shock_arrays['volume_densities']  
        print('del1[], del2[], del3[] loaded, units: cm')
        print('volumes[][][] loaded, units: cm^3')
        print('temperatures[][][] loaded, units: K')        
        print('ion_fractions[][][] loaded, units: none')        
        print('volume_densities[][][] loaded, units: cm^-3')
        if(with_times):
            self.load_cooling_times()
            self.load_flow_times()

    
    
    ###################################################################################
    ##########################                       ##################################
    ########################## EMISSIONS / INTENSITY ##################################
    ##########################                       ##################################
    ###################################################################################
    # Below is frequency dependent
    
    
    #####################    
    # Function for calculating gaunt factor g_FF
    # inputs T4 = temp/10^4 K and nu9 = freq/10^9Hz = freq/GHz
    def gaunt_factor(self,T4,nu9): 
        return 5.96 * T4**.15 * nu9**(-.1)
    # Function for calculating emission coefficient in cgs units
    # inputs: T in K and nu in Hz
    
    
    ##### EMISSION COEFFICIENTS #####
    #################################
    def emission_coef(self,T, nu, n_H):
        if(T<10**3):
            return 0
        g_ff = self.gaunt_factor(T/(10**4), nu/(10**9))
        j_nu = (3.86 * g_ff * Constants.e**6 / (Constants.m_e**2 * Constants.c**3) 
                * (Constants.m_e / (Constants.k_B*T))**(1/2)
                * n_H**2 * np.exp(-Constants.h*nu / (Constants.k_B*T))) 
        return j_nu

    def emission_coef2(self,i,j,k,nu):
        T = self.temperatures[i,j,k]
        n_H = self.volume_densities[i,j,k]
        return self.emission_coef(T, nu, n_H)
    
    # Calculate emission coefficients in erg cm^-3 Hz^-1 s^-1 sr^-1
    def calculate_emission_coefs(self, nu):
        self.emission_coefs = np.zeros((len(self.x1), len(self.x2), len(self.x3)))
        for i in range(len(self.x1)):
            for j in range(len(self.x2)):
                for k in range(len(self.x3)):
                    self.emission_coefs[i,j,k] = self.emission_coef2(i,j,k,nu) # erg cm^-3 Hz^-1 s^-1 sr^-1
        print('emission_coefs[][][] complete')
        
                  
                    
    ##### ABSORPTION COEFFICIENTS #####
    ###################################
    # Function for calculating absorption coefficient
    # inputs: T in K and nu in Hz
    def absorption_coef(self, T, nu, n_H):
        if(T<100):
            T=100
        g_ff = self.gaunt_factor(T/(10**4), nu/(10**9))
        kappa_nu = 1.9296 * n_H**2 * Constants.e**6 * g_ff / (
                    Constants.m_e**(3/2) * Constants.c * (Constants.k_B * T)**(3/2) * nu**2)
        if (kappa_nu < 10**-30): kappa_nu = 10**-30
        return kappa_nu
    
    def absorption_coef2(self,i,j,k,nu):
        T = self.temperatures[i,j,k]
        n_H = self.volume_densities[i,j,k]
        return self.absorption_coef(T, nu, n_H)
    
    def calculate_absorption_coefs(self, nu):
        self.absorption_coefs = np.zeros((len(self.x1), len(self.x2), len(self.x3)))
        for i in range(len(self.x1)):
            for j in range(len(self.x2)):
                for k in range(len(self.x3)):
                    self.absorption_coefs[i,j,k] = self.absorption_coef2(i,j,k,nu)
        print('absorption_coefs[][][] complete')
    
    ##### OPTICAL DEPTHS & SOURCE FUNCTIONS#####
    ##########################
    # Calculate optical depths in each direction
    def calculate_taus(self):
        self.tauA = np.zeros((len(self.x1), len(self.x2), len(self.x3)))
        self.tauB = np.zeros((len(self.x1), len(self.x2), len(self.x3)))
        self.tauC = np.zeros((len(self.x1), len(self.x2), len(self.x3)))
        for i in range(len(self.x1)):
            for j in range(len(self.x2)):
                for k in range(len(self.x3)):
                    self.tauA[i,j,k] = self.absorption_coefs[i,j,k] * self.del3[k]
                    self.tauB[i,j,k] = self.absorption_coefs[i,j,k] * self.del2[j]
                    self.tauC[i,j,k] = self.absorption_coefs[i,j,k] * self.del1[i]
        print('tauA[][][], tauB[][][], tauC[][][] complete')
                    
    # Calculate the source function for each cell
    def calculate_sourcefuncs(self):
        self.source_functions = self.emission_coefs / self.absorption_coefs
        self.source_functions[np.isnan(self.source_functions)] = 0
        print('source_functions[][][] complete')
        
        
    ##### FRAME INTENSITIES #####
    #############################
    # Get intensity for a given column and frame, using cgs units
    def get_frameA_intensity(self,i,j):
        # start with emission from backmost cell
        I_cur = self.emission_coefs[i,j,0] * self.del3[0]     
        # iterate through cells moving forward
        for k in range(1, len(self.x3)): #len(x3)):
            I_prev = I_cur #inputs intensity of previous cell to calculate intensity of current cell
            I_cur = (I_prev * np.exp(-1*self.tauA[i,j,k]) 
                     + (self.source_functions[i,j,k])  * (1 - np.exp(-1 * self.tauA[i,j,k])))
        return(I_cur) # erg cm^-2 s^-1 Hz^-1 sr^-1
    def get_frameB_intensity(self,i,k):
        # start with emission from backmost cell
        I_cur = self.emission_coefs[i,0,k] * self.del2[0] # erg cm^-2 s^-1 Hz^-1 sr^-1
        # iterate through cells moving forward
        for j in range(1, len(self.x2)): #len(x3)):
            I_prev = I_cur #inputs intensity of previous cell to calculate intensity of current cell
            I_cur = (I_prev * np.exp(-1*self.tauB[i,j,k]) 
                     + (self.source_functions[i,j,k])  * (1 - np.exp(-1 * self.tauB[i,j,k])))
        return(I_cur) # erg cm^-2 s^-1 Hz^-1 sr^-1
    def get_frameC_intensity(self,j,k):
        # start with emission from backmost cell
        I_cur = self.emission_coefs[0,j,k] * self.del1[0] # erg cm^-2 s^-1 Hz^-1 sr^-1
        # iterate through cells moving forward
        for i in range(1, len(self.x1)): #len(x3)):
            I_prev = I_cur #inputs intensity of previous cell to calculate intensity of current cell
            I_cur = (I_prev * np.exp(-1*self.tauC[i,j,k]) 
                     + (self.source_functions[i,j,k])  * (1 - np.exp(-1 * self.tauC[i,j,k])))
        return(I_cur) # erg cm^-2 s^-1 Hz^-1 sr^-1
    
    # Calculate intensity from each plane in mJy/arcsec^2
    def calculate_intensities(self):
        self.InuA = np.zeros((len(self.x1), len(self.x2)))
        self.InuB = np.zeros((len(self.x1), len(self.x3)))
        self.InuC = np.zeros((len(self.x2), len(self.x3)))
        for i in range(len(self.x1)):
            for j in range(len(self.x2)):
                self.InuA[i,j] = self.get_frameA_intensity(i,j) * 10**26  / (4.2545 * 10**10) # mJy/arcsec^2 
        for i in range(len(self.x1)):
            for k in range(len(self.x3)):
                self.InuB[i,k] = self.get_frameB_intensity(i,k) * 10**26  / (4.2545 * 10**10) # mJy/arcsec^2 
        for j in range(len(self.x2)):
            for k in range(len(self.x3)):
                self.InuC[j,k] = self.get_frameC_intensity(j,k) * 10**26  / (4.2545 * 10**10) # mJy/arcsec^2 
        print('InuA[][], InuB[][], InuC[][] complete')
        
    
    ##### DISTANCE DEPENDENCE #####
    ###############################ma
    # (below is distance dependent, input r_kpc)
    def make_grids_arcsec(self, r_kpc=1):
        self.r_AU = 2.0624 * 10**8 * r_kpc # AU
        # convert axes to arcsec
        self.X1_2_as = self.X1_2 / self.r_AU * 206264.5 # arcsec 
        self.X2_1_as = self.X2_1 / self.r_AU * 206264.5 # arcsec 
        self.X1_3_as = self.X1_3 / self.r_AU * 206264.5 # arcsec 
        self.X3_1_as = self.X3_1 / self.r_AU * 206264.5 # arcsec
        self.X2_3_as = self.X2_3 / self.r_AU * 206264.5 # arcsec 
        self.X3_2_as = self.X3_2 / self.r_AU * 206264.5 # arcsec
        self.x1_as = self.x1 / self.r_AU * 206264.5 # arcsec
        self.x2_as = self.x2 / self.r_AU * 206264.5 # arcsec
        self.x3_as = self.x3 / self.r_AU * 206264.5 # arcsec
        print('arcsec grid complete')

        # calculate monochromatic flux F_nu in mJy
    def calculate_fluxes(self):
        self.FnuA = 0
        self.FnuB = 0
        self.FnuC = 0
        for i in range(len(self.x1)):
            for j in range(len(self.x2)):
                self.FnuA += self.InuA[i,j] * self.del1[i] * self.del2[j] / self.r_cm**2 * 206264.5**2 # mJy
        for i in range(len(self.x1)):
            for k in range(len(self.x3)):
                self.FnuB += self.InuB[i,k] * self.del1[i] * self.del3[k] / self.r_cm**2 * 206264.5**2 # mJy
        for j in range(len(self.x2)):
            for k in range(len(self.x3)):
                self.FnuC += self.InuC[j,k] * self.del2[j] * self.del3[k] / self.r_cm**2 * 206264.5**2 # mJy
        print('FnuA, FnuB, FnuC complete, units: mJy')
   
    
    # Calculate all intensity variables for a given frequency 
    def calculate_all_intensity_variables(self, nu, r_kpc = 1):    
        data_path = (VICO_loc+'/Data/'+self.name+'/')
        print('nu = '+str(nu/10**9)+'GHz, r = '+str(r_kpc)+'kpc')
        self.r_AU = 2.0624 * 10**8 * r_kpc # AU
        self.r_cm = 3.0857 * 10**21 * r_kpc # cm
        self.make_grids()
        self.make_grids_arcsec(r_kpc)
        self.calculate_emission_coefs(nu)     # emission_coefs[][][]
        self.calculate_absorption_coefs(nu)   # absorption_coefs[][][]
        self.calculate_taus()                 # tauA[][][], tauB[][][], tauC[][][]
        self.calculate_sourcefuncs()          # source_functions[][][]
        self.calculate_intensities()          # InuA[][], InuB[][], InuC[][]
        self.calculate_fluxes()               # FnuA, FnuB, FnuC      
        np.savez((data_path+self.name+'_'+str(nu/10**9)+'GHz_'+str(r_kpc)+'kpc_arrays.npz'), 
                emission_coefs = self.emission_coefs, absorption_coefs = self.absorption_coefs,
                tauA = self.tauA, tauB = self.tauB, tauC = self.tauC, 
                source_functions = self.source_functions, 
                InuA = self.InuA, InuB = self.InuB, InuC = self.InuC,
                FnuA = self.FnuA, FnuB = self.FnuB, FnuC = self.FnuC,
                nu = nu, r_kpc = r_kpc)
        self.calculate_all_intensity_variables_const(nu, .01)
        self.calculate_all_intensity_variables_const(nu, False)
        
    def load_intensity_variables(self, nu, r_kpc=1):
        self.make_grids_arcsec
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
        

         
    def delete_intermediary_arrays(self):
        del self.del1
        del self.del2
        del self.del3
        del self.volumes
        del self.temperatures_m1
        del self.temperatures_p1
        del self.temperatures_m2
        del self.temperatures_p2
        del self.temperatures_m3
        del self.temperatures_p3
        del self.ion_fractions_m1
        del self.ion_fractions_p1
        del self.ion_fractions_m2
        del self.ion_fractions_p2
        del self.ion_fractions_m3
        del self.ion_fractions_p3
        del self.volume_densities
        
        
        
        
        ############# Making Plots #####################
        ################################################
        
    def make_intensity_plots(self, nu, r_kpc=1, min = -3, max = 2):
        self.make_grids()
        self.make_grids_arcsec(r_kpc=r_kpc)
        
        self.load_intensity_variables(nu, r_kpc)
        now = datetime.now()

        # Make specific intensity_nu log plots for given frequency in mJy/as
        gs_Inu = gridspec.GridSpec(4,4)
        fig_Inu = plt.figure(figsize = (12,10))
        axA_Inu, axB_Inu, axC_Inu = plt.subplot(gs_Inu[:2,:2]), plt.subplot(gs_Inu[:2,2:]), plt.subplot(gs_Inu[2:4, 1:3])
        fig_Inu.suptitle(('Specific Intensity at '+str(nu/10**9)+'GHz, '+str(r_kpc)+'kpc of '+self.name), fontsize=18)
        fig_Inu.tight_layout()
        axA_Inu.set_xlabel('z (arcsec)')
        axA_Inu.set_ylabel('x (arcsec)')
        axA_Inu.set_aspect(1)
        axB_Inu.set_xlabel('z (arcsec)')
        axB_Inu.set_ylabel('y (arcsec)')
        axB_Inu.set_aspect(1)
        axC_Inu.set_xlabel('x (arcsec)')
        axC_Inu.set_ylabel('y (arcsec)')
        axC_Inu.set_aspect(1)

        levels_Inu = np.linspace(min, max, 101)

        contA_Inu = axA_Inu.contourf(self.X1_2_as , self.X2_1_as, np.log10(self.InuA), levels=levels_Inu, cmap =  'jet')
        fig_Inu.colorbar(contA_Inu, label = 'log($I_{'+str(nu/10**9)+'\mathrm{GHz}}$ / [mJy/as])', ax=axA_Inu)

        contB_Inu = axB_Inu.contourf(self.X1_3_as, self.X3_1_as, np.log10(self.InuB), levels=levels_Inu, cmap =  'jet')
        fig_Inu.colorbar(contB_Inu, label = 'log($I_{'+str(nu/10**9)+'\mathrm{GHz}}$ / [mJy/as])', ax=axB_Inu)

        contC_Inu = axC_Inu.contourf(self.X2_3_as, self.X3_2_as, np.log10(self.InuC), levels=levels_Inu, cmap =  'jet')
        fig_Inu.colorbar(contC_Inu, label = 'log($I_{'+str(nu/10**9)+'\mathrm{GHz}}$ / [mJy/as])', ax=axC_Inu)

        fig_Inu.savefig(VICO_loc+'/RadioEmissions/IntensityPlots/'+self.name+'_I_'+str(nu/10**9)+'GHz_'+str(r_kpc)+'kpc.png')
        fig_Inu.show()


        fluxfile = open((VICO_loc+'/RadioEmissions/FluxOutputs/flux_'+self.name+'.txt'), 'a')
        print('\nOutput at:', now, file=fluxfile)
        print(('F_%.2fGHz A: %e mJy' % ((nu/10**9), self.FnuA)), file = fluxfile)
        print(('F_%.2fGHz B: %e mJy' % ((nu/10**9), self.FnuB)), file = fluxfile)
        print(('F_%.2fGHz C: %e mJy' % ((nu/10**9), self.FnuC)), file = fluxfile)
        fluxfile.close()

        print('F_%.2fGHz A: %e mJy' % ((nu/10**9), self.FnuA))
        print('F_%.2fGHz B: %e mJy' % ((nu/10**9), self.FnuB))
        print('F_%.2fGHz C: %e mJy' % ((nu/10**9), self.FnuC))
        
    def make_intensity_plots(self, nu, r_kpc=1, min = -3, max = 2):
        self.make_grids()
        self.make_grids_arcsec(r_kpc=r_kpc)
        
        self.load_intensity_variables(nu, r_kpc)
        now = datetime.now()

        # Make specific intensity_nu log plots for given frequency in mJy/as
        gs_Inu = gridspec.GridSpec(4,4)
        fig_Inu = plt.figure(figsize = (12,10))
        axA_Inu, axB_Inu, axC_Inu = plt.subplot(gs_Inu[:2,:2]), plt.subplot(gs_Inu[:2,2:]), plt.subplot(gs_Inu[2:4, 1:3])
        fig_Inu.suptitle(('Specific Intensity at '+str(nu/10**9)+'GHz, '+str(r_kpc)+'kpc of '+self.name), fontsize=18)
        fig_Inu.tight_layout()
        axA_Inu.set_xlabel('z (arcsec)')
        axA_Inu.set_ylabel('x (arcsec)')
        axA_Inu.set_aspect(1)
        axB_Inu.set_xlabel('z (arcsec)')
        axB_Inu.set_ylabel('y (arcsec)')
        axB_Inu.set_aspect(1)
        axC_Inu.set_xlabel('x (arcsec)')
        axC_Inu.set_ylabel('y (arcsec)')
        axC_Inu.set_aspect(1)

        levels_Inu = np.linspace(min, max, 101)

        contA_Inu = axA_Inu.contourf(self.X1_2_as , self.X2_1_as, np.log10(self.InuA), levels=levels_Inu, cmap =  'jet')
        fig_Inu.colorbar(contA_Inu, label = 'log($I_{'+str(nu/10**9)+'\mathrm{GHz}}$ / [mJy/as])', ax=axA_Inu)

        contB_Inu = axB_Inu.contourf(self.X1_3_as, self.X3_1_as, np.log10(self.InuB), levels=levels_Inu, cmap =  'jet')
        fig_Inu.colorbar(contB_Inu, label = 'log($I_{'+str(nu/10**9)+'\mathrm{GHz}}$ / [mJy/as])', ax=axB_Inu)

        contC_Inu = axC_Inu.contourf(self.X2_3_as, self.X3_2_as, np.log10(self.InuC), levels=levels_Inu, cmap =  'jet')
        fig_Inu.colorbar(contC_Inu, label = 'log($I_{'+str(nu/10**9)+'\mathrm{GHz}}$ / [mJy/as])', ax=axC_Inu)

        fig_Inu.savefig(VICO_loc+'/RadioEmissions/IntensityPlots/'+self.name+'_I_'+str(nu/10**9)+'GHz_'+str(r_kpc)+'kpc.png')
        fig_Inu.show()
        
    # Inputs: nu in (Hz), r_kpc in (kpc) scale in (AU)
    # nu and r_kpc must be the same as alreadu loaded
    # returns fluxes for given scale
    def calculate_scale_fluxes(self, scale, file, nu, r_kpc=1, printToScreen = False):
        scale_FnuA = 0
        scale_FnuB = 0
        scale_FnuC = 0
        zscale_FnuA = 0
        zscale_FnuB = 0
        max1 = np.argmax(self.x1[self.x1 <= (scale)])
        max2 = np.argmax(self.x2[self.x2 <= scale])
        max3 = np.argmax(self.x3[self.x3 <= scale])
        min1 = (np.where(self.x1 == (np.amin(self.x1[self.x1 >= (-1*scale)]))))[0][0]
        min2 = (np.where(self.x2 == (np.amin(self.x2[self.x2 >= (-1*scale)]))))[0][0]
        min3 = (np.where(self.x3 == (np.amin(self.x3[self.x3 >= (-1*scale)]))))[0][0]
        for i in range(min1, max1+1):
            for j in range(min2, max2+1):
                scale_FnuA += self.InuA[i,j] * self.del1[i] * self.del2[j] / self.r_cm**2 * 206264.5**2 # mJy
            for j in range(len(self.x2)):
                zscale_FnuA += self.InuA[i,j] * self.del1[i] * self.del2[j] / self.r_cm**2 * 206264.5**2 # mJy
        for i in range(min1, max1+1):
            for k in range(min3, max3+1):
                scale_FnuB += self.InuB[i,k] * self.del1[i] * self.del3[k] / self.r_cm**2 * 206264.5**2 # mJy
            for k in range(len(self.x3)):
                zscale_FnuB += self.InuB[i,k] * self.del1[i] * self.del3[k] / self.r_cm**2 * 206264.5**2 # mJy
        for j in range(min2, max2+1):
            for k in range(min3, max3+1):
                scale_FnuC += self.InuC[j,k] * self.del2[j] * self.del3[k] / self.r_cm**2 * 206264.5**2 # mJy
        scale_FnuA *= 2 # to account for both jets
        scale_FnuB *= 2 # to account for both jets
        zscale_FnuA *= 2 # to account for both jets
        zscale_FnuB *= 2 # to account for both jets
        print('%8.2f \t%2.6f \t%2.6f \t%2.6f \t%6.0f to %6.0f \t%6.0f to %6.0f \t%6.0f to %6.0f' %  
            ((nu/10**9), scale_FnuA, scale_FnuB, scale_FnuC, self.x1[min1], self.x1[max1], 
             self.x2[min2], self.x2[max2], self.x3[min3],
            self.x3[max3]), file = file)
        print('%8.2f \t%2.6f \t%2.6f \t\t \t%6.0f to %6.0f \t%6.0f to %6.0f \t%6.0f to %6.0f' %  
            ((nu/10**9), zscale_FnuA, zscale_FnuB, self.x1[min1], self.x1[max1], 
             self.x2[0], self.x2[len(self.x2)-1], self.x3[0],
            self.x3[len(self.x3)-1]), file = file)
        if(printToScreen):
            print('%8.2f \t%2.6f \t%2.6f \t%2.6f \t%6.0f to %6.0f \t%6.0f to %6.0f \t%6.0f to %6.0f' %  
                ((nu/10**9), scale_FnuA, scale_FnuB, scale_FnuC, self.x1[min1], self.x1[max1], 
                 self.x2[min2], self.x2[max2], self.x3[min3],self.x3[max3]))
            print('%8.2f \t%2.6f \t%2.6f \t\t \t%6.0f to %6.0f \t%6.0f to %6.0f \t%6.0f to %6.0f' %  
                  ((nu/10**9), zscale_FnuA, zscale_FnuB, self.x1[min1], self.x1[max1], 
                   self.x2[0], self.x2[len(self.x2)-1], self.x3[0], self.x3[len(self.x3)-1]))
        return(scale_FnuA, scale_FnuB, scale_FnuC, zscale_FnuA, zscale_FnuB)
    
        # Inputs: nu in (Hz), r_kpc in (kpc), z height in (AU), x or y width in (AU)
    # nu and r_kpc must be the same as alreadu loaded
    # returns fluxes for given scale
    def calculate_height_fluxes_cumulative(self, height, file, nu, width = 10000, r_kpc=1, printToScreen = False):
        height_FnuA = 0
        height_FnuB = 0
        max1 = np.argmax(self.x1[self.x1 <= (height)])
        max2 = np.argmax(self.x2[self.x2 <= width/2])
        max3 = np.argmax(self.x3[self.x3 <= width/2])
        min1 = (np.where(self.x1 == (np.amin(self.x1[self.x1 >= (-1*height)]))))[0][0]
        min2 = (np.where(self.x2 == (np.amin(self.x2[self.x2 >= (-1*width/2)]))))[0][0]
        min3 = (np.where(self.x3 == (np.amin(self.x3[self.x3 >= (-1*width/2)]))))[0][0]
        for i in range(min1, max1+1):
            for j in range(min2, max2+1):
                height_FnuA += self.InuA[i,j] * self.del1[i] * self.del2[j] / self.r_cm**2 * 206264.5**2 # mJy
        for i in range(min1, max1+1):
            for k in range(min3, max3+1):
                height_FnuB += self.InuB[i,k] * self.del1[i] * self.del3[k] / self.r_cm**2 * 206264.5**2 # mJy
        height_FnuA *= 2 # to account for both jets
        height_FnuB *= 2 # to account for both jets
        print('%8.2f \t%2.6f \t%2.6f \t%6.0f to %6.0f \t%6.0f to %6.0f \t%6.0f to %6.0f' %  
            ((nu/10**9), height_FnuA, height_FnuB, self.x1[min1], self.x1[max1], 
             self.x2[min2], self.x2[max2], self.x3[min3],
            self.x3[max3]), file = file)
        if(printToScreen):
            print('%8.2f \t%2.6f \t%2.6f \t%6.0f to %6.0f \t%6.0f to %6.0f \t%6.0f to %6.0f' %  
                ((nu/10**9), height_FnuA, height_FnuB, self.x1[min1], self.x1[max1], 
                 self.x2[min2], self.x2[max2], self.x3[min3],self.x3[max3]))
        return(height_FnuA, height_FnuB)
    
    
    def calculate_height_fluxes(self, height, nu, width = 10000, r_kpc=1, file = False, printToScreen = False):
        height_FnuA = 0
        h1 = np.argmax(self.x1[self.x1 <= (height)])
        max2 = np.argmax(self.x2[self.x2 <= width/2])
        min2 = (np.where(self.x2 == (np.amin(self.x2[self.x2 >= (-1*width/2)]))))[0][0]
        for j in range(min2, max2+1):
            height_FnuA += self.InuA[h1,j] * self.del1[h1] * self.del2[j] / self.r_cm**2 * 206264.5**2 # mJy
        height_FnuA *= 2 # to account for both jets
        if(file != False):
            print('%8.2f \t%2.6f \t%6.0f to %6.0f' %  
                ((nu/10**9), height_FnuA, self.x2[min2], self.x2[max2]), file = file)
        if(printToScreen):
            print('%8.2f \t%2.6f \t%6.0f to %6.0f' %  
                ((nu/10**9), height_FnuA, self.x2[min2], self.x2[max2]))
        return(height_FnuA)
    
    
  
    
    
    
######################################################################
#######                 Cooling-Time Scaled                   ########
######                 Intensity and Fluxes                   ########
######################################################################
######################################################################    
# const=False refers to scale_factor = t_cool/t_flow

########################## Intensities ###############################

    # Get intensity for a given column and frame, using cgs units
    # gets the intensity for a single column at i,j
    def get_frameA_intensity_const(self,i,j,const):
        # start with emission from backmost cell
        I_cur = self.emission_coefs[i,j,0] * self.del3[0]*const    
        # iterate through cells moving forward
        for k in range(1, len(self.x3)): #len(x3)): 
            if(const): scale_factor = const
            else: 
                scale_factor = self.cooling_times[i,j,k]/self.flow_times[i,j,k]
                if(scale_factor>1): scale_factor=1 # can't be more than 1
            I_prev = I_cur #inputs intensity of previous cell to calculate intensity of current cell
            I_cur = (I_prev * np.exp(-1*self.tauA[i,j,k]*scale_factor) 
                     + (self.emission_coefs[i,j,k]/self.absorption_coefs[i,j,k])  * (1 - np.exp(-1 * self.tauA[i,j,k]*scale_factor)))
        return(I_cur) # erg cm^-2 s^-1 Hz^-1 sr^-1

    # gets the intensities for all columns of a snapshot
    # must first load the intensity variables
    def calculate_intensities_const(self, const):
        self.InuA_const = np.zeros((len(self.x1), len(self.x2)))
        for i in range(len(self.x1)):
            for j in range(len(self.x2)):
                self.InuA_const[i,j] = self.get_frameA_intensity_const(i,j, const) * 10**26  / (4.2545 * 10**10) # mJy/arcsec^2 

    # calculates flux of entire snapshot
    # based on whatever intensity nu and const is loaded
    def calculate_fluxes_const(self):
        self.FnuA_const = 0
        for i in range(len(self.x1)):
            for j in range(len(self.x2)):
                self.FnuA_const += self.InuA_const[i,j] * self.del1[i] * self.del2[j] / self.r_cm**2 * 206264.5**2 # mJy
        self.FnuA_const 

            
            
    ############## ONLY NEED TO RUN THE FOLLOWING 2 #################                       
    # calculate and save all intensity variables at a given frequency and constant scale factor  
    def calculate_all_intensity_variables_const(self, nu, const, r_kpc = 1):    
        self.load_intensity_variables(nu, r_kpc)
        data_path = (VICO_loc+'/Data/'+self.name+'/')
        if(const):
            print('nu = '+str(nu/10**9)+'GHz, r = '+str(r_kpc)+'kpc, SCALE = '+str(const))
        else:
            print('nu = '+str(nu/10**9)+'GHz, r = '+str(r_kpc)+'kpc, SCALE = t_cool/t_flow')
        self.calculate_intensities_const(const)    # InuA[][], InuB[][], InuC[][]
        self.calculate_fluxes_const()         # FnuA, FnuB, FnuC
        np.savez((data_path+self.name+'_'+str(nu/10**9)+'GHz_'+str(r_kpc)+'kpc_arrays_const'+str(const)+'.npz'), 
                InuA_const = self.InuA_const, FnuA_const = self.FnuA_const,
                nu = nu, r_kpc = r_kpc)

    
    # load all intensity variables AND SCALE FLUXES at a given frequency    
    def load_intensity_variables_const(self, nu, const, r_kpc=1):
        self.make_grids_arcsec
        data_path = (VICO_loc+'/Data/'+self.name+'/')
        self.load_intensity_variables(nu, r_kpc)
        loaded_intensity_arrays = np.load((data_path+self.name+'_'+str(nu/10**9)+'GHz_'+str(r_kpc)+'kpc_arrays_const'+str(const)+'.npz'))
        self.InuA_const = loaded_intensity_arrays['InuA_const'] 
        self.FnuA_const = loaded_intensity_arrays['FnuA_const']

 
    # calculate scale fluxes for a given frequency
    def calculate_scale_fluxes_const(self, scale, nu, const, r_kpc=1):
        self.load_intensity_variables_const(nu, const=const)
        scale_FnuA_const = 0
        max1 = np.argmax(self.x1[self.x1 <= (scale)])
        max2 = np.argmax(self.x2[self.x2 <= scale])
        min1 = (np.where(self.x1 == (np.amin(self.x1[self.x1 >= (-1*scale)]))))[0][0]
        min2 = (np.where(self.x2 == (np.amin(self.x2[self.x2 >= (-1*scale)]))))[0][0]
        for i in range(min1, max1+1):
            for j in range(min2, max2+1):
                scale_FnuA_const += self.InuA_const[i,j] * self.del1[i] * self.del2[j] / self.r_cm**2 * 206264.5**2 # mJy
        scale_FnuA_const *= 2 # to account for both jets
        return(scale_FnuA_const)

    # calculate, save, and return all scale fluxes
    def scale_fluxes_const(self, frequencies, const, scales, r_kpc=1):
        self.ScaleFluxes_const = np.zeros((len(frequencies), len(scales)))
        for f in range(len(frequencies)):
            for s in range(len(scales)):
                FA = self.calculate_scale_fluxes_const(scales[s], nu = frequencies[f], const=const, r_kpc=1)
                self.ScaleFluxes_const[f,s] = FA
                print('nu = %fGHz, scale=%d au, Flux = %e mJy' %(frequencies[f]/10**9, scales[s], FA))
        np.savez((VICO_loc+'/Data/'+self.name+'/'+self.name+'_ScaleFluxes_const'+str(const)+'.npz'), 
                 ScaleFluxes_const = self.ScaleFluxes_const)
        return self.ScaleFluxes_const            

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
    

    ####################################################################################
    ################## load all intensity and flux variables ###########################
    ####################################################################################
    def load_fluxes(self):
        data_path = (VICO_loc+'/Data/')
        flux_data = np.load(data_path+self.name+'/'+self.name+'_ScaleFluxes.npz')
        self.ScaleFluxes = flux_data['ScaleFluxes']
        self.zScaleFluxes = flux_data['zScaleFluxes']
        self.HeightFluxDensities = flux_data['HeightFluxDensities']
        flux_data.close()
        ratio_data = np.load(data_path+self.name+'/'+self.name+'_ScaleFluxes_constFalse.npz')
        self.ScaleFluxes_ratio = ratio_data['ScaleFluxes_const']
        ratio_data.close()
        
        # self.load_scalefluxes_const(.01)
        # self.load_scalefluxes_const(False)
