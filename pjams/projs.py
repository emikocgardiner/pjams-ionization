import sys, os, inspect
import sys
sys.path.append('/Users/emigardiner/VICO/pjams-ionization/pjams-ionization/')

from zeusmp_snapshot_reader import read_zeusmp_snapshot
from zeusmp_snapshot_reader import ScaleFactors
from snapshot import *
from basic_snapshot import basic_snapshot, FREQS, VICO_loc
import plot
import ionfrac

import numpy as np

import matplotlib.colors as colors
from obspy.imaging.cm import viridis_white_r 

SAVELOC = VICO_loc+'/pillowfiles/projs_fig'
frequencies = FREQS
v_mins = ionfrac.V_MINS


#########################################
slicescales = np.array([4000, 25000])
vmin_ionfrac, vmax_ionfrac = -3, 0       
v = len(v_mins)-1
vmin_iontemp, vmax_iontemp = 0, 7       
vmin_intensity_noratio, vmax_intensity_noratio = -3, 2.8   
vmin_intensity_ratio, vmax_intensity_ratio = -4, 0   
##########################################



#########################################
#### New ionfrac cmap
#########################################

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap('jet')
new_cmap = truncate_colormap(cmap, 0.1, 1)




#########################################
#### Mass Ionization Fraction
#########################################

def mass_ionfrac_proj_pcolormesh(Snap, year, scale, vmin, vmax, v, cmap=new_cmap,
                            saveloc=SAVELOC, show_cbar= False, 
                            show_xlabels=True, show_ylabels=True, 
                            cbar_pad=.2, show_freq=False, vertical_cbar=False, 
                            show_contours=False, show_legend=False):
    Snap.X1_2v, Snap.X2_1v = np.meshgrid(Snap.x1, Snap.x2)
    v_min=v_mins[v]
            
    # Make projection log plots for snapshot
    fig = plt.figure(figsize = (4,4))
    fig.tight_layout()
    ax = plt.subplot()
    ax.set_aspect(1)
    ax.set_xlim(-scale/2, scale/2)
    ax.set_ylim(0, scale)
    ax.set_facecolor('black')
    
    cont = ax.pcolormesh(Snap.X2_1v , Snap.X1_2v, np.rot90(np.log10(Snap.ionfrac_mass[v,:,:])), 
                               vmin=vmin, vmax=vmax, cmap = cmap, shading='nearest')
    
    if(show_xlabels):
        ax.set_xticks([-scale*.25, 0, scale*.25])
        ax.set_xlabel('$x$ (au)', fontsize=28)
        ax.tick_params(axis='x', labelsize=20)
    else:
        ax.tick_params(bottom=False)
        ax.set_xticks([])
    if(show_ylabels):
        ax.set_yticks([scale*.25,scale*.5,scale*.75])
        ax.set_ylabel('$z$ (au)', fontsize=28)
        ax.tick_params(axis='y', labelsize=20)
    else:
        ax.tick_params(left=False)
        ax.set_yticks([])
    if(show_cbar):
        if(vertical_cbar):
            cbar = fig.colorbar(cont, orientation = 'vertical', 
                                fraction=0.038, pad=cbar_pad, extend='min')
            cbar.set_label(label = (r'log10($\langle\chi_\mathrm{H+}\rangle$ )'),
                          fontsize = 18)
            cbar.ax.tick_params(rotation=45, labelsize = 13)
        else:
            cbar = fig.colorbar(cont, orientation = 'horizontal', 
                                fraction=0.038, pad=cbar_pad, extend='min')
            cbar.set_label(label = (r'log10($\langle\chi_\mathrm{H+}\rangle$ )'),
                          fontsize=18)
            cbar.ax.tick_params(rotation=45, labelsize=13)
            
    if(show_contours):
        cntr = ax.contour(Snap.X2_1v , Snap.X1_2v, np.rot90(Snap.maxv1), 
                       levels = [1,10,100], colors = ['darkorange', 'red', 'magenta']) #linewidths = [1, 1.5, 1])
        h,_ = cntr.legend_elements()
        if(show_legend):
            ax.legend([h[0], h[1], h[2]], ['1 km/s', '10km/s', '100km/s'], loc = 'upper center', bbox_to_anchor=(.5, -.65))
    
#     ax.text(.02, .22, (r'$\mathbf{\langle\chi_{H+}\rangle, mass}$'+'\n'+year+'\n'+'$\mathbf{v_{min}}$='+str(v_min)+' km/s'), ha='left', va='top', 
#             transform=ax.transAxes, fontsize = 13, color='white', weight='bold')
    ax.text(.02, .3, (r'$\mathbf{\langle\chi_{H+}\rangle}$'+'\n'+year),
            ha='left', va='top', transform=ax.transAxes, 
            fontsize = 28, weight='bold', color='white')
    
    if (saveloc != False): 
        filename = saveloc+'/'+Snap.name+'_ionfrac_mass_v'+str(v_mins[v])+'kms_'+str(scale)+'AU.png'
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        return filename
    
def mass_ionfrac_proj_const_pcolormesh(Snap, year, const, scale, vmin, vmax, v, cmap=new_cmap,
                            saveloc=SAVELOC, show_cbar= False, 
                            show_xlabels=True, show_ylabels=True, 
                            cbar_pad=.2, show_freq=False, vertical_cbar=False, 
                            show_contours=False, show_legend=False):
    Snap.X1_2v, Snap.X2_1v = np.meshgrid(Snap.x1, Snap.x2)
    v_min=v_mins[v]
            
    # Make projection log plots for snapshot
    fig = plt.figure(figsize = (4,4))
    fig.tight_layout()
    ax = plt.subplot()
    ax.set_aspect(1)
    ax.set_xlim(-scale/2, scale/2)
    ax.set_ylim(0, scale)
    ax.set_facecolor('black')
    
    if(const):
        cont = ax.pcolormesh(Snap.X2_1v , Snap.X1_2v, np.rot90(np.log10(Snap.ionfrac_mass_const[v,:,:])), 
                               vmin=vmin, vmax=vmax, cmap = cmap, shading='nearest')
    else: 
        cont = ax.pcolormesh(Snap.X2_1v , Snap.X1_2v, np.rot90(np.log10(Snap.ionfrac_mass_ratio[v,:,:])), 
                               vmin=vmin, vmax=vmax, cmap = cmap, shading='nearest')
    
    if(show_xlabels):
        ax.set_xticks([-scale*.25, 0, scale*.25])
        ax.set_xlabel('$x$ (au)', fontsize=28)
        ax.tick_params(axis='x', labelsize=20)
    else:
        ax.tick_params(bottom=False)
        ax.set_xticks([])
    if(show_ylabels):
        ax.set_yticks([scale*.25,scale*.5,scale*.75])
        ax.set_ylabel('$z$ (au)', fontsize=28)
        ax.tick_params(axis='y', labelsize=20)
    else:
        ax.tick_params(left=False)
        ax.set_yticks([])
    if(show_cbar):
        if(vertical_cbar):
            cbar = fig.colorbar(cont, orientation = 'vertical', 
                                fraction=0.038, pad=cbar_pad, extend='min')
            cbar.set_label(label = (r'log10($\langle\chi_\mathrm{H+}\rangle$ )'),
                          fontsize = 18)
            cbar.ax.tick_params(rotation=45, labelsize = 13)
        else:
            cbar = fig.colorbar(cont, orientation = 'horizontal', 
                                fraction=0.038, pad=cbar_pad, extend='min')
            cbar.set_label(label = (r'log10($\langle\chi_\mathrm{H+}\rangle$ )'),
                          fontsize=18)
            cbar.ax.tick_params(rotation=45, labelsize=13)
            
    if(show_contours):
        cntr = ax.contour(Snap.X2_1v , Snap.X1_2v, np.rot90(Snap.maxv1), 
                       levels = [1,10,100], colors = ['darkorange', 'red', 'magenta']) #linewidths = [1, 1.5, 1])
        h,_ = cntr.legend_elements()
        if(show_legend):
            ax.legend([h[0], h[1], h[2]], ['1 km/s', '10km/s', '100km/s'], loc = 'upper center', bbox_to_anchor=(.5, -.65))
    
#     ax.text(.02, .22, (r'$\mathbf{\langle\chi_{H+}\rangle, mass}$'+'\n'+year+'\n'+'$\mathbf{v_{min}}$='+str(v_min)+' km/s'), ha='left', va='top', 
#             transform=ax.transAxes, fontsize = 13, color='white', weight='bold')
    ax.text(.02, .3, (r'$\mathbf{\langle\chi_{H+}\rangle}$'+'\n'+year),
            ha='left', va='top', transform=ax.transAxes, 
            fontsize = 28, weight='bold', color='white')
    
    if ((saveloc) and (const)): 
        filename = saveloc+'/'+Snap.name+'_ionfrac_mass_const_v'+str(v_mins[v])+'kms_'+str(scale)+'AU.png'             
        fig.savefig(filename, bbox_inches='tight')
        return filename
    elif ((saveloc)):  #  and (!const)
        filename = saveloc+'/'+Snap.name+'_ionfrac_mass_ratio_v'+str(v_mins[v])+'kms_'+str(scale)+'AU.png'             
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        return filename
    

##################################
#### Intensity
##################################
freq_strings=np.array(['.01', '.05', '0.1', '0.5', '1.0', 
                       '5.3', '23', '43', '100', '230']) 

def intensity_proj_pcolormesh(Snap, year, scale, f, vmin, vmax, cmap='CMRmap',
                        saveloc=False, show=False, show_cbar= False, show_xlabels=True, show_ylabels=True, cbar_pad=.2,
                             vertical_cbar=False):
    nu=frequencies[f]
    Snap.load_intensity_variables(nu)
    Snap.X1_2v, Snap.X2_1v = np.meshgrid(Snap.x1, Snap.x2)
    fig = plt.figure(figsize = (4,4))
    fig.tight_layout()
    ax = plt.subplot()
    ax.set_aspect(1)
    ax.set_xlim(-scale/2, scale/2)
    ax.set_ylim(0, scale)
    ax.set_facecolor('black')

    cont = ax.pcolormesh(Snap.X2_1v, Snap.X1_2v, np.log10(np.rot90(Snap.InuA[:,:])), 
                               vmin=vmin, vmax=vmax, cmap = cmap, shading='nearest')
    if(show_xlabels):
        ax.set_xticks([-scale*.25, 0, scale*.25])
        ax.set_xlabel('$x$ (au)', fontsize=28)
        ax.tick_params(axis='x', labelsize=20)
    else:
        ax.tick_params(bottom=False)
        ax.set_xticks([])
    if(show_ylabels):
        ax.set_yticks([scale*.25,scale*.5,scale*.75])
        ax.set_ylabel('$z$ (au)', fontsize=28)
        ax.tick_params(axis='y', labelsize=20)
    else:
        ax.tick_params(left=False)
        ax.set_yticks([])
        
    if(show_cbar):
        if(vertical_cbar):
            cbar = fig.colorbar(cont, orientation = 'vertical', 
                                fraction=0.038, pad=cbar_pad)
            cbar.set_label(label = (r'log($I_\nu$ / [mJy/as])'),
                          fontsize = 18)
            cbar.ax.tick_params(rotation=45, labelsize = 12)
        else:
            cbar = fig.colorbar(cont, orientation = 'horizontal', 
                                fraction=0.038, pad=cbar_pad)
            cbar.set_label(label = (r'log($I_\nu$ / [mJy/as])'),
                          fontsize=18)
            cbar.ax.tick_params(rotation=45, labelsize=12)
    ax.text(.02, .29, (freq_strings[f]+'GHz'), ha='left', va='top', 
            transform=ax.transAxes, fontsize = 20, color='white', weight='bold')
    ax.text(.02, .3, ('\n'+year), ha='left', va='top', 
            transform=ax.transAxes, fontsize = 28, color='white', weight='bold')
    if (saveloc != False): 
        filename = saveloc+'/'+Snap.name+'_'+str(nu/10**9)+'GHzIntensity_'+str(scale)+'AU.png' 
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        return(filename)
        
def intensity_proj_const_pcolormesh(Snap, year, const, scale, f, vmin, vmax, cmap='CMRmap',
                        saveloc=False, show=False, show_cbar= False, show_xlabels=True, show_ylabels=True, cbar_pad=.2,
                             vertical_cbar=False):
    nu=frequencies[f]
    Snap.load_intensity_variables(nu)
    Snap.load_intensity_variables_const(nu=nu, const=const)
    
    Snap.X1_2v, Snap.X2_1v = np.meshgrid(Snap.x1, Snap.x2)
    fig = plt.figure(figsize = (4,4))
    fig.tight_layout()
    ax = plt.subplot()
    ax.set_aspect(1)
    ax.set_xlim(-scale/2, scale/2)
    ax.set_ylim(0, scale)
    ax.set_facecolor('black')
    
    cont = ax.pcolormesh(Snap.X2_1v, Snap.X1_2v, np.log10(np.rot90(Snap.InuA_const[:,:])), 
                               vmin=vmin, vmax=vmax, cmap = cmap, shading='nearest')
    if(show_xlabels):
        ax.set_xticks([-scale*.25, 0, scale*.25])
        ax.set_xlabel('$x$ (au)', fontsize=28)
        ax.tick_params(axis='x', labelsize=20)
    else:
        ax.tick_params(bottom=False)
        ax.set_xticks([])
    if(show_ylabels):
        ax.set_yticks([scale*.25,scale*.5,scale*.75])
        ax.set_ylabel('$z$ (au)', fontsize=28)
        ax.tick_params(axis='y', labelsize=20)
    else:
        ax.tick_params(left=False)
        ax.set_yticks([])
        
    if(show_cbar):
        if(vertical_cbar):
            cbar = fig.colorbar(cont, orientation = 'vertical', 
                                fraction=0.038, pad=cbar_pad)
            cbar.set_label(label = (r'log($I_\nu$ / [mJy/as])'),
                          fontsize = 18)
            cbar.ax.tick_params(rotation=45, labelsize = 12)
        else:
            cbar = fig.colorbar(cont, orientation = 'horizontal', 
                                fraction=0.038, pad=cbar_pad)
            cbar.set_label(label = (r'log($I_\nu$ / [mJy/as])'),
                          fontsize=18)
            cbar.ax.tick_params(rotation=45, labelsize=12)
    ax.text(.02, .29, (freq_strings[f]+'GHz'), ha='left', va='top', 
            transform=ax.transAxes, fontsize = 20, color='white', weight='bold')
    ax.text(.02, .3, ('\n'+year), ha='left', va='top', 
            transform=ax.transAxes, fontsize = 28, color='white', weight='bold')
    if ((saveloc) and (const)): 
        filename = saveloc+'/'+Snap.name+'_'+str(nu/10**9)+'GHzIntensity_const_'+str(scale)+'AU.png'             
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        return filename
    elif ((saveloc)):  #  and (!const)
        filename = saveloc+'/'+Snap.name+'_'+str(nu/10**9)+'GHzIntensity_ratio_'+str(scale)+'AU.png'             
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        return filename
    

#########################################################
#### IonTemp Projection Map Methods
#########################################################

def iontemp_proj_pcolormesh(Snap, year, scale, vmin, vmax, 
                            cmap='hot', nu = 5.3*10**9,
                            saveloc=False, show_cbar= False, 
                            show_xlabels=True, show_ylabels=True, 
                            cbar_pad=.2, vertical_cbar=False):
    TprojA = np.zeros((len(Snap.x1), len(Snap.x2)))
    for i in range(len(Snap.x1)):
        for j in range(len(Snap.x2)):
            TprojA[i,j] = (np.sum(Snap.temperatures[i,j,:] * Snap.ion_fractions[i,j,:] * Snap.q['d'][i,j,:] * Snap.del3) 
                           / np.sum(Snap.ion_fractions[i,j,:] * Snap.q['d'][i,j,:] * Snap.del3))
    # Make temperature projection log plots for snapshot    
    
    
    Snap.X1_2v, Snap.X2_1v = np.meshgrid(Snap.x1, Snap.x2)
    fig = plt.figure(figsize = (4,4))
    fig.tight_layout()
    ax = plt.subplot()
    ax.set_aspect(1)
    ax.set_xlim(-scale/2, scale/2)
    ax.set_ylim(0, scale)
    ax.set_facecolor('black')

    cont = ax.pcolormesh(Snap.X2_1v, Snap.X1_2v, np.log10(np.rot90(TprojA)), 
                               vmin=vmin, vmax=vmax, cmap = cmap, shading='nearest')
    if(show_xlabels):
        ax.set_xticks([-scale*.25, 0, scale*.25])
        ax.set_xlabel('$x$ (au)', fontsize=28)
        ax.tick_params(axis='x', labelsize=20)
    else:
        ax.tick_params(bottom=False)
        ax.set_xticks([])
    if(show_ylabels):
        ax.set_yticks([scale*.25,scale*.5,scale*.75])
        ax.set_ylabel('$z$ (au)', fontsize=28)
        ax.tick_params(axis='y', labelsize=20)
    else:
        ax.tick_params(left=False)
        ax.set_yticks([])
    if(show_cbar):
        if(vertical_cbar):
            cbar = fig.colorbar(cont, orientation = 'vertical', 
                                fraction=0.038, pad=cbar_pad)
            cbar.set_label(label = (r'log($\langle T_\mathrm{H+}\rangle$ / [K])'),
                          fontsize = 18)
            cbar.ax.tick_params(rotation=45, labelsize = 13)
        else:
            cbar = fig.colorbar(cont, orientation = 'horizontal', 
                                fraction=0.038, pad=cbar_pad)
            cbar.set_label(label = (r'log($\langle T_\mathrm{H+}\rangle$ / [K])'),
                          fontsize=18)
            cbar.ax.tick_params(rotation=45, labelsize=13)
    
    ax.text(.02, .3, (r'$\mathbf{\langle T_{H+}\rangle}$'+'\n'+year), ha='left', va='top', 
            transform=ax.transAxes, fontsize = 28, weight='bold', color='white')
    if (saveloc != False): 
        filename = saveloc+'/'+Snap.name+'_iontemp_'+str(scale)+'AU.png' 
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        return(filename)

def iontemp_proj_ratio_pcolormesh(Snap, year, scale, vmin, vmax, 
                            cmap='hot', nu = 5.3*10**9,
                            saveloc=False, show_cbar= False, 
                            show_xlabels=True, show_ylabels=True, 
                            cbar_pad=.2, vertical_cbar=False, const=False):
    TprojA = np.zeros((len(Snap.x1), len(Snap.x2)))
    for i in range(len(Snap.x1)):
        for j in range(len(Snap.x2)):
            TprojA[i,j] = (np.sum(Snap.temperatures[i,j,:] * Snap.ion_fractions[i,j,:] 
                                  * Snap.q['d'][i,j,:] * Snap.del3 
                                  * Snap.scale_factors[i,j,:]) 
                           / np.sum(Snap.ion_fractions[i,j,:] 
                                    * Snap.q['d'][i,j,:] * Snap.del3
                                    * Snap.scale_factors[i,j,:]))
    # Make temperature projection log plots for snapshot    
    
    Snap.X1_2v, Snap.X2_1v = np.meshgrid(Snap.x1, Snap.x2)
    fig = plt.figure(figsize = (4,4))
    fig.tight_layout()
    ax = plt.subplot()
    ax.set_aspect(1)
    ax.set_xlim(-scale/2, scale/2)
    ax.set_ylim(0, scale)
    ax.set_facecolor('black')

    cont = ax.pcolormesh(Snap.X2_1v, Snap.X1_2v, np.log10(np.rot90(TprojA)), 
                               vmin=vmin, vmax=vmax, cmap = cmap, shading='nearest')
    if(show_xlabels):
        ax.set_xticks([-scale*.25, 0, scale*.25])
        ax.set_xlabel('$x$ (au)', fontsize=28)
        ax.tick_params(axis='x', labelsize=20)
    else:
        ax.tick_params(bottom=False)
        ax.set_xticks([])
    if(show_ylabels):
        ax.set_yticks([scale*.25,scale*.5,scale*.75])
        ax.set_ylabel('$z$ (au)', fontsize=28)
        ax.tick_params(axis='y', labelsize=20)
    else:
        ax.tick_params(left=False)
        ax.set_yticks([])
    if(show_cbar):
        if(vertical_cbar):
            cbar = fig.colorbar(cont, orientation = 'vertical', 
                                fraction=0.038, pad=cbar_pad)
            cbar.set_label(label = (r'log($\langle T_\mathrm{H+}\rangle$ / [K])'),
                          fontsize = 18)
            cbar.ax.tick_params(rotation=45, labelsize = 13)
        else:
            cbar = fig.colorbar(cont, orientation = 'horizontal', 
                                fraction=0.038, pad=cbar_pad)
            cbar.set_label(label = (r'log($\langle T_\mathrm{H+}\rangle$ / [K])'),
                          fontsize=18)
            cbar.ax.tick_params(rotation=45, labelsize=13)
    
    ax.text(.02, .3, (r'$\mathbf{\langle T_{H+}\rangle}$'+'\n'+year), ha='left', va='top', 
            transform=ax.transAxes, fontsize = 28, weight='bold', color='white')
    if ((saveloc) and (const)): 
        filename = saveloc+'/'+Snap.name+'_iontemp_const_'+str(scale)+'AU.png'             
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        return filename
    elif ((saveloc)):  #  and (!const)
        filename = saveloc+'/'+Snap.name+'_iontemp_ratio_'+str(scale)+'AU.png'             
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        return filename

#################################################################
#### Projection Images, All
#################################################################
def proj_images(Snap, year, scale, saveloc = SAVELOC, 
                 show_cbar=False, xlabels=False, cbar_pad=.15):
    # Ionization Fraction
    file0 = mass_ionfrac_proj_pcolormesh(Snap, year, scale, v = v,
                            vmin=vmin_ionfrac, vmax = vmax_ionfrac, 
                            show_xlabels = xlabels, show_ylabels = True,
                            show_contours = False, show_legend = False, 
                            show_cbar = False, saveloc=saveloc)
    # Temperature Projection
    file1 = iontemp_proj_pcolormesh(Snap, year, scale,  
                            vmin_iontemp, vmax_iontemp,
                            show_xlabels = xlabels, show_ylabels = False, 
                            show_cbar=False, saveloc=saveloc)
    
    # 5.3 GHz Intensity
    file2 = intensity_proj_pcolormesh(Snap, year, scale, 5, 
                              vmin_intensity_noratio, vmax_intensity_noratio, 
                             show_xlabels = xlabels, show_ylabels = False, 
                              show_cbar=False, saveloc=saveloc)
    # 230 GHz
    file3 = intensity_proj_pcolormesh(Snap, year, scale, 9,
                              vmin_intensity_noratio, vmax_intensity_noratio, 
                             show_xlabels = xlabels, show_ylabels = False, 
                              show_cbar=False, saveloc=saveloc)
    return (file0, file1, file2, file3)

def proj_const_images(Snap, year, const, scale, saveloc = False, 
                 show_cbar=False, xlabels=False, cbar_pad=.15):
    # Ionization Fraction
    file0 = mass_ionfrac_proj_const_pcolormesh(Snap, year, const, scale, v = v,
                            vmin=vmin_ionfrac, vmax = vmax_ionfrac, 
                            show_xlabels = xlabels, show_ylabels = True,
                            show_contours = False, show_legend = False, 
                            show_cbar = False, saveloc=saveloc)
    # Temperature Projection
    file1 = iontemp_proj_ratio_pcolormesh(Snap, year, scale,  
                            vmin_iontemp, vmax_iontemp,
                            show_xlabels = xlabels, show_ylabels = False, 
                            show_cbar=False, saveloc=saveloc)

    
    # 5.3 GHz Intensity
    file2 = intensity_proj_const_pcolormesh(Snap, year, const, scale, 5, 
                              vmin_intensity_ratio, vmax_intensity_ratio, 
                             show_xlabels = xlabels, show_ylabels = False, 
                              show_cbar=False, saveloc=saveloc)
    # 230 GHz
    file3 = intensity_proj_const_pcolormesh(Snap, year, const, scale, 9,
                              vmin_intensity_ratio, vmax_intensity_ratio, 
                             show_xlabels = xlabels, show_ylabels = False, 
                              show_cbar=False, saveloc=saveloc)
    return (file0, file1, file2, file3)