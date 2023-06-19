import sys, os, inspect
import sys
sys.path.append('/Users/emigardiner/VICO/pjams-ionization/pjams-ionization/')

from zeusmp_snapshot_reader import read_zeusmp_snapshot
from zeusmp_snapshot_reader import ScaleFactors
from snapshot import *
from basic_snapshot import basic_snapshot, VICO_loc
import plot

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from IPython.display import Image as DisplayImage

SAVELOC = VICO_loc+'/pillowfiles/ionfrac_fig'

FREQS = np.array([0.01, 0.05, 0.1, 0.5, 1, 5.3, 23, 43, 100, 230])*10**9 # Hz
VICO_loc = '/Users/emigardiner/VICO/pjams-ionization'
MYFONT = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf', 65)
SMALLFONT = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf', 55)
VMIN = -3
VMAX = 0
V_MINS = np.array([1, 10, 100])

r_kpc = 1
heights_and_scales = np.load(VICO_loc+'/Data/heights_and_scales.npz')
scales = heights_and_scales['scales'] # AU
heights = heights_and_scales['heights'] # AU  

colors = plot.COLORS
colors_cont = ['#8080ff', '#ff99ff', 'white']


#################################################
#### Max velocity Arrays for Contours
#################################################
v_mins = V_MINS

def max_velocity_proj_array(snap, v_min):
    snap.maxv1 = np.zeros((len(snap.x1), len(snap.x2)))
    for i in range(len(snap.x1)):
        for j in range(len(snap.x2)):
            snap.maxv1[i,j] = np.max(snap.q['v1'][i,j,:])


import matplotlib.colors as mplcols


#################################################
#### Load ionfrac arrays 
# #################################################


def load_average_ionfrac_array(Snap, debug=False):
    data_path = (VICO_loc+'/Data/'+Snap.name+'/')
    loaded_average_ionfracs = np.load((data_path+Snap.name+'_average_ionfrac_arrays.npz'))
    Snap.ionfrac_mass = loaded_average_ionfracs['ionfrac_mass']
    Snap.ionfrac_vol =  loaded_average_ionfracs['ionfrac_vol']
    Snap.ionfrac_emis = loaded_average_ionfracs['ionfrac_emis']
    if debug: print(Snap.name + ' ionfrac arrays loaded')
    
def load_average_ionfrac_array_ratio(Snap, const=False, debug=False):
    data_path = (VICO_loc+'/Data/'+Snap.name+'/')
    loaded_average_ionfracs = np.load((data_path+Snap.name+'_average_ionfrac_arrays_const'+str(const)+'.npz'))
    Snap.ionfrac_mass_ratio = loaded_average_ionfracs['ionfrac_mass']
    Snap.ionfrac_vol_ratio =  loaded_average_ionfracs['ionfrac_vol']
    Snap.ionfrac_emis_ratio = loaded_average_ionfracs['ionfrac_emis']
    if debug: print(Snap.name + ' ionfrac_ratio arrays loaded')







#################################################
#### Ionfrac new colormap
#################################################

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mplcols.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap('jet')
new_cmap = truncate_colormap(cmap, 0.1, 1)



#################################################
#### Mass pcolormesh
#################################################


def mass_ionfrac_pcolormesh(
        Snap, year, scale, vmin, vmax, v, cmap=new_cmap, 
        saveloc=SAVELOC, show_cbar= False, cbar_pad=.2, vertical_cbar=False, 
        show_xlabels=True, show_ylabels=True, show_contours=False, lw_cont=1.5, colors_cont=colors_cont, show_legend=False,
        name=None, xx_name=0.02, yy_name=0.22, namecolor='white', namefs=13):
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
        ax.tick_params(axis='x', labelsize=24)
    else:
        ax.tick_params(bottom=False)
        ax.set_xticks([])
    if(show_ylabels):
        ax.set_yticks([scale*.25,scale*.5,scale*.75])
        ax.set_ylabel('$z$ (au)', fontsize=28)
        ax.tick_params(axis='y', labelsize=24)
    else:
        ax.tick_params(left=False)
        ax.set_yticks([])
    if(show_cbar):
        if(vertical_cbar):
            fig.colorbar(cont, label = r'log10($\langle\chi_\mathrm{H+}\rangle$ )', 
                 orientation = 'vertical', fraction=0.038, pad=cbar_pad, extend='min').ax.tick_params(rotation=45)
        else:
            fig.colorbar(cont, label = (r'log10($\langle\chi_\mathrm{H+}\rangle$ )'), 
                     orientation = 'horizontal', fraction=0.038, pad=cbar_pad, extend='min').ax.tick_params(rotation=45)
    
    if(show_contours):
        cntr = ax.contour(Snap.X2_1v , Snap.X1_2v, np.rot90(Snap.maxv1), linewidths=lw_cont, 
                       levels = [1,10,100], colors=colors_cont) #linewidths = [1, 1.5, 1])
        h,_ = cntr.legend_elements()
        if(show_legend):
            ax.legend([h[0], h[1], h[2]], ['1 km/s', '10km/s', '100km/s'], loc = 'upper center', bbox_to_anchor=(.5, -.65))
    
    if name is not None:
        ax.text(xx_name, yy_name, name, transform=ax.transAxes, fontsize = namefs, color=namecolor, weight='bold')

    if (saveloc is not None): 
        filename = saveloc+'/'+Snap.name+'_ionfrac_mass_v'+str(v_mins[v])+'kms_'+str(scale)+'AU.png'
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        return filename
    return fig
        

        
def mass_ionfrac_const_pcolormesh(
        Snap, year, scale, vmin, vmax, v, const=False, cmap=new_cmap, saveloc=SAVELOC, 
        show_cbar= False, cbar_pad=.2, show_freq=False, vertical_cbar=False, 
        show_xlabels=True, show_ylabels=True, show_contours=False, lw_cont=1.5, colors_cont=colors_cont, show_legend=False,
        name=None, xx_name=0.02, yy_name=0.22, namecolor='white', namefs=13):
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
        ax.tick_params(axis='x', labelsize=24)
    else:
        ax.tick_params(bottom=False)
        ax.set_xticks([])
    if(show_ylabels):
        ax.set_yticks([scale*.25,scale*.5,scale*.75])
        ax.set_ylabel('$z$ (au)', fontsize=28)
        ax.tick_params(axis='y', labelsize=24)
    else:
        ax.tick_params(left=False)
        ax.set_yticks([])
    if(show_cbar):
        if(vertical_cbar):
            fig.colorbar(cont, label = r'log10($\langle\chi_\mathrm{H+}\rangle$ )', 
                 orientation = 'vertical', fraction=0.038, pad=cbar_pad, extend='min').ax.tick_params(rotation=45)
        else:
            fig.colorbar(cont, label = (r'log10($\langle\chi_\mathrm{H+}\rangle$ )'), 
                     orientation = 'horizontal', fraction=0.038, pad=cbar_pad, extend='min').ax.tick_params(rotation=45)
    
    if(show_contours):
        cntr = ax.contour(Snap.X2_1v , Snap.X1_2v, np.rot90(Snap.maxv1), linewidths=lw_cont, 
                       levels = [1,10,100], colors=colors_cont) #linewidths = [1, 1.5, 1])
        h,_ = cntr.legend_elements()
        if(show_legend):
            ax.legend([h[0], h[1], h[2]], ['1 km/s', '10km/s', '100km/s'], loc = 'upper center', bbox_to_anchor=(.5, -.65))
    
    if name is not None:
        ax.text(xx_name, yy_name, name, transform=ax.transAxes, fontsize = namefs, color=namecolor, weight='bold')

    if ((saveloc) and (const)): 
        filename = saveloc+'/'+Snap.name+'_ionfrac_mass_const_v'+str(v_mins[v])+'kms_'+str(scale)+'AU.png'
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        return filename
    elif ((saveloc)):  #  and (!const)
        filename = saveloc+'/'+Snap.name+'_ionfrac_mass_ratio_v'+str(v_mins[v])+'kms_'+str(scale)+'AU.png'
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        return filename
    return fig
   
#################################################
#### Volume pcolormesh
#################################################     
        
def vol_ionfrac_pcolormesh(Snap, year, scale, vmin, vmax, v, cmap=new_cmap,
                            saveloc=SAVELOC, show=False, show_cbar= False, lw_cont=1.5, colors_cont=colors_cont, 
                            show_xlabels=True, show_ylabels=True, 
                            cbar_pad=.2, show_freq=False, vertical_cbar=False, 
                            show_contours=False, show_legend=False):
    Snap.X1_2v, Snap.X2_1v = np.meshgrid(Snap.x1, Snap.x2)
#     print(v)
    v_min=v_mins[v]

    # Make projection log plots for snapshot
    fig = plt.figure(figsize = (4,4))
    fig.tight_layout()
    ax = plt.subplot()
    ax.set_aspect(1)
    ax.set_xlim(-scale/2, scale/2)
    ax.set_ylim(0, scale)
    ax.set_facecolor('black')
    
    cont = ax.pcolormesh(Snap.X2_1v , Snap.X1_2v, np.rot90(np.log10(Snap.ionfrac_vol[v,:,:])), 
                               vmin=vmin, vmax=vmax, cmap = cmap, shading='nearest')
    if(show_xlabels):
        ax.set_xticks([-scale*.25, 0, scale*.25])
        ax.set_xlabel('$x$ (au)', fontsize=28)
        ax.tick_params(axis='x', labelsize=24)
    else:
        ax.tick_params(bottom=False)
        ax.set_xticks([])
    if(show_ylabels):
        ax.set_yticks([scale*.25,scale*.5,scale*.75])
        ax.set_ylabel('$z$ (au)', fontsize=28)
        ax.tick_params(axis='y', labelsize=24)
    else:
        ax.tick_params(left=False)
        ax.set_yticks([])
    if(show_cbar):
        if(vertical_cbar):
            fig.colorbar(cont, label = r'log10($\langle\chi_\mathrm{H+}\rangle$ )', 
                 orientation = 'vertical', fraction=0.038, pad=cbar_pad, extend='min').ax.tick_params(rotation=45)
        else:
            fig.colorbar(cont, label = (r'log10($\langle\chi_\mathrm{H+}\rangle$ )'), 
                     orientation = 'horizontal', fraction=0.038, pad=cbar_pad, extend='min').ax.tick_params(rotation=45)
      
    if(show_contours):
        cntr = ax.contour(Snap.X2_1v , Snap.X1_2v, np.rot90(Snap.maxv1), linewidths=lw_cont, 
                       levels = [1,10,100], colors=colors_cont) #linewidths = [1, 1.5, 1])
        h,_ = cntr.legend_elements()
        if(show_legend):
            ax.legend([h[0], h[1], h[2]], ['1 km/s', '10km/s', '100km/s'], loc = 'upper center', bbox_to_anchor=(.5, -.65))
    
#     ax.text(.02, .22, (r'$\mathbf{\langle\chi_{H+}\rangle, vol}$'+'\n'+year+'\n'+'$\mathbf{v_{min}}$='+str(v_min)+' km/s'), ha='left', va='top', 
#             transform=ax.transAxes, fontsize = 13, color='white', weight='bold')
    if (saveloc != False): 
        filename = saveloc+'/'+Snap.name+'_ionfrac_vol_v'+str(v_mins[v])+'kms_'+str(scale)+'AU.png'
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        return filename
    return fig

def vol_ionfrac_const_pcolormesh(Snap, year, scale, vmin, vmax, v, const=False, cmap=new_cmap,
                            saveloc=SAVELOC, show=False, show_cbar= False, lw_cont=1.5, colors_cont=colors_cont, 
                            show_xlabels=True, show_ylabels=True, 
                            cbar_pad=.2, show_freq=False, vertical_cbar=False, 
                            show_contours=False, show_legend=False):
    Snap.X1_2v, Snap.X2_1v = np.meshgrid(Snap.x1, Snap.x2)
#     print(v)
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
        cont = ax.pcolormesh(Snap.X2_1v , Snap.X1_2v, np.rot90(np.log10(Snap.ionfrac_vol_const[v,:,:])), 
                               vmin=vmin, vmax=vmax, cmap = cmap, shading='nearest')
    else:
        cont = ax.pcolormesh(Snap.X2_1v , Snap.X1_2v, np.rot90(np.log10(Snap.ionfrac_vol_ratio[v,:,:])), 
                               vmin=vmin, vmax=vmax, cmap = cmap, shading='nearest')
    if(show_xlabels):
        ax.set_xticks([-scale*.25, 0, scale*.25])
        ax.set_xlabel('$x$ (au)', fontsize=28)
        ax.tick_params(axis='x', labelsize=24)
    else:
        ax.tick_params(bottom=False)
        ax.set_xticks([])
    if(show_ylabels):
        ax.set_yticks([scale*.25,scale*.5,scale*.75])
        ax.set_ylabel('$z$ (au)', fontsize=28)
        ax.tick_params(axis='y', labelsize=24)
    else:
        ax.tick_params(left=False)
        ax.set_yticks([])
    if(show_cbar):
        if(vertical_cbar):
            fig.colorbar(cont, label = r'log10($\langle\chi_\mathrm{H+}\rangle$ )', 
                 orientation = 'vertical', fraction=0.038, pad=cbar_pad, extend='min').ax.tick_params(rotation=45)
        else:
            fig.colorbar(cont, label = (r'log10($\langle\chi_\mathrm{H+}\rangle$ )'), 
                     orientation = 'horizontal', fraction=0.038, pad=cbar_pad, extend='min').ax.tick_params(rotation=45)
      
    if(show_contours):
        cntr = ax.contour(Snap.X2_1v , Snap.X1_2v, np.rot90(Snap.maxv1), linewidths=lw_cont, 
                       levels = [1,10,100], colors=colors_cont) #linewidths = [1, 1.5, 1])
        h,_ = cntr.legend_elements()
        if(show_legend):
            ax.legend([h[0], h[1], h[2]], ['1 km/s', '10km/s', '100km/s'], loc = 'upper center', bbox_to_anchor=(.5, -.65))
    
#     ax.text(.02, .22, (r'$\mathbf{\langle\chi_{H+}\rangle, vol}$'+'\n'+year+'\n'+'$\mathbf{v_{min}}$='+str(v_min)+' km/s'), ha='left', va='top', 
#             transform=ax.transAxes, fontsize = 13, color='white', weight='bold')
    if ((saveloc) and (const)): 
        filename = saveloc+'/'+Snap.name+'_ionfrac_vol_const_v'+str(v_mins[v])+'kms_'+str(scale)+'AU.png'
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        return filename
    elif ((saveloc)):  #  and (!const)
        filename = saveloc+'/'+Snap.name+'_ionfrac_vol_ratio_v'+str(v_mins[v])+'kms_'+str(scale)+'AU.png'
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        return filename
    return fig


 
#################################################
#### Emis pcolormesh
#################################################
   
        
def emis_ionfrac_pcolormesh(Snap, year, scale, f, vmin, vmax, v, cmap=new_cmap,
                            saveloc=SAVELOC, show=False, show_cbar= False, lw_cont=1.5, colors_cont=colors_cont, 
                            show_xlabels=True, show_ylabels=True, 
                            cbar_pad=.2, show_freq=False, vertical_cbar=False, 
                            show_contours=False, show_legend=False, 
                           with_delta = True):
    Snap.X1_2v, Snap.X2_1v = np.meshgrid(Snap.x1, Snap.x2)
    v_min=v_mins[v]
    nu=FREQS[f]
           
    # Make projection log plots for snapshot
    fig = plt.figure(figsize = (4,4))
    fig.tight_layout()
    ax = plt.subplot()
    ax.set_aspect(1)
    ax.set_xlim(-scale/2, scale/2)
    ax.set_ylim(0, scale)
    ax.set_facecolor('black')
    
    cont = ax.pcolormesh(Snap.X2_1v , Snap.X1_2v, np.rot90(np.log10(Snap.ionfrac_emis[f,v,:,:])), 
                               vmin=vmin, vmax=vmax, cmap = cmap, shading='nearest')
#     ax.plot(Snap.x2[9], Snap.x1[0], 
#             color='cyan', markersize=50, marker='+')
#     ax.plot(Snap.x2[9], Snap.x1[165], 
#             color='cyan', markersize=50, marker='+')
#     ax.plot(Snap.x2[270], Snap.x1[0], 
#             color='cyan', markersize=50, marker='+')
#     ax.plot(Snap.x2[270], Snap.x1[165], 
#             color='cyan', markersize=50, marker='+')

    if(show_xlabels):
        ax.set_xticks([-scale*.25, 0, scale*.25])
        ax.set_xlabel('$x$ (au)', fontsize=28)
        ax.tick_params(axis='x', labelsize=24)
    else:
        ax.tick_params(bottom=False)
        ax.set_xticks([])
    if(show_ylabels):
        ax.set_yticks([scale*.25,scale*.5,scale*.75])
        ax.set_ylabel('$z$ (au)', fontsize=28)
        ax.tick_params(axis='y', labelsize=24)
    else:
        ax.tick_params(left=False)
        ax.set_yticks([])
    if(show_cbar):
        if(vertical_cbar):
            fig.colorbar(cont, label = r'log10($\langle\chi_\mathrm{H+}\rangle$ )', 
                 orientation = 'vertical', fraction=0.038, pad=cbar_pad, extend='min').ax.tick_params(rotation=45)
        else:
            fig.colorbar(cont, label = (r'log10($\langle\chi_\mathrm{H+}\rangle$ )'), 
                     orientation = 'horizontal', fraction=0.038, pad=cbar_pad, extend='min').ax.tick_params(rotation=45)   
    
    if(show_contours):
        cntr = ax.contour(Snap.X2_1v , Snap.X1_2v, np.rot90(Snap.maxv1), linewidths=lw_cont, 
                       levels = [1,10,100], colors=colors_cont) #linewidths = [1, 1.5, 1])
        h,_ = cntr.legend_elements()
        if(show_legend):
            ax.legend([h[0], h[1], h[2]], ['1 km/s', '10km/s', '100km/s'], loc = 'upper center', bbox_to_anchor=(.5, -.65))
    
#     ax.text(.02, .22, (r'$\mathbf{\langle\chi_{H+}\rangle, emis}$'+'\n'+year+'\n'+'$\mathbf{v_{min}}$='+str(v_min)+' km/s'), ha='left', va='top', 
#             transform=ax.transAxes, fontsize = 13, color='white', weight='bold')
    if (saveloc != False): 
        filename = saveloc+'/'+Snap.name+'_ionfrac_emis_v'+str(v_mins[v])+'kms_'+str(scale)+'AU.png'
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        return filename
    return fig
    
        
def emis_ionfrac_const_pcolormesh(Snap, year, scale, f, vmin, vmax, v, const=False, cmap=new_cmap,
                            saveloc=SAVELOC, show=False, show_cbar= False, lw_cont=1.5, colors_cont=colors_cont, 
                            show_xlabels=True, show_ylabels=True, 
                            cbar_pad=.2, show_freq=False, vertical_cbar=False, 
                            show_contours=False, show_legend=False, 
                           with_delta = True):
    Snap.X1_2v, Snap.X2_1v = np.meshgrid(Snap.x1, Snap.x2)
    v_min=v_mins[v]
    nu=FREQS[f]
           
    # Make projection log plots for snapshot
    fig = plt.figure(figsize = (4,4))
    fig.tight_layout()
    ax = plt.subplot()
    ax.set_aspect(1)
    ax.set_xlim(-scale/2, scale/2)
    ax.set_ylim(0, scale)
    ax.set_facecolor('black')
    
    if(const):
        cont = ax.pcolormesh(Snap.X2_1v , Snap.X1_2v, np.rot90(np.log10(Snap.ionfrac_emis_const[f,v,:,:])), 
                               vmin=vmin, vmax=vmax, cmap = cmap, shading='nearest')
    else:
        cont = ax.pcolormesh(Snap.X2_1v , Snap.X1_2v, np.rot90(np.log10(Snap.ionfrac_emis_ratio[f,v,:,:])), 
                               vmin=vmin, vmax=vmax, cmap = cmap, shading='nearest')
    if(show_xlabels):
        ax.set_xticks([-scale*.25, 0, scale*.25])
        ax.set_xlabel('$x$ (au)', fontsize=28)
        ax.tick_params(axis='x', labelsize=24)
    else:
        ax.tick_params(bottom=False)
        ax.set_xticks([])
    if(show_ylabels):
        ax.set_yticks([scale*.25,scale*.5,scale*.75])
        ax.set_ylabel('$z$ (au)', fontsize=28)
        ax.tick_params(axis='y', labelsize=24)
    else:
        ax.tick_params(left=False)
        ax.set_yticks([])
    if(show_cbar):
        if(vertical_cbar):
            fig.colorbar(cont, label = r'log10($\langle\chi_\mathrm{H+}\rangle$ )', 
                 orientation = 'vertical', fraction=0.038, pad=cbar_pad, extend='min').ax.tick_params(rotation=45)
        else:
            fig.colorbar(cont, label = (r'log10($\langle\chi_\mathrm{H+}\rangle$ )'), 
                     orientation = 'horizontal', fraction=0.038, pad=cbar_pad, extend='min').ax.tick_params(rotation=45)   
    
    if(show_contours):
        cntr = ax.contour(Snap.X2_1v , Snap.X1_2v, np.rot90(Snap.maxv1), linewidths=lw_cont,
                       levels = [1,10,100], colors=colors_cont) #linewidths = [1, 1.5, 1])
        h,_ = cntr.legend_elements()
        if(show_legend):
            ax.legend([h[0], h[1], h[2]], ['1 km/s', '10km/s', '100km/s'], loc = 'upper center', bbox_to_anchor=(.5, -.65))
    
#     ax.text(.02, .22, (r'$\mathbf{\langle\chi_{H+}\rangle, emis}$'+'\n'+year+'\n'+'$\mathbf{v_{min}}$='+str(v_min)+' km/s'), ha='left', va='top', 
#             transform=ax.transAxes, fontsize = 15, color='white', weight='bold')
    if ((saveloc) and (const)): 
        filename = saveloc+'/'+Snap.name+'_ionfrac_emis_const_v'+str(v_mins[v])+'kms_'+str(scale)+'AU.png'
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        return filename
    elif ((saveloc)):  #  and (!const)
        filename = saveloc+'/'+Snap.name+'_ionfrac_emis_ratio_v'+str(v_mins[v])+'kms_'+str(scale)+'AU.png'
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        return filename
    return fig
