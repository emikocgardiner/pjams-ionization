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
        print(stats(Snap.ionfrac_emis_ratio))
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


#####################################################################################################
################### 3D Average ionization fractions (not for table) ######################################
#####################################################################################################



def mass_avg_3d_ionfrac(Snap, v_cutoff, scale, cooled=False):
    numer = 0
    denom = 0
    for i in range(len(Snap.x1)):
        if((Snap.x1[i]) <= scale):
            for j in range(len(Snap.x2)):
                if(np.abs(Snap.x2[j] <= scale/2)):
                    for k in range(len(Snap.x3)):
                        if(Snap.q['v1'][i,j,k]>=v_cutoff):
                            numer += (Snap.ion_fractions[i,j,k] * Snap.q['d'][i,j,k] * 
                                      Snap.del1[i] * Snap.del2[j] * Snap.del3[k])
                            denom += (Snap.q['d'][i,j,k] * 
                                      Snap.del1[i] * Snap.del2[j] * Snap.del3[k])
    if(denom==0): return 0
    return numer/denom

def vol_avg_3d_ionfrac(Snap, v_cutoff, scale):
    numer = 0
    denom = 0
    for i in range(len(Snap.x1)):
        if((Snap.x1[i]) <= scale):
            for j in range(len(Snap.x2)):
                if(np.abs(Snap.x2[j] <= scale/2)):
                    for k in range(len(Snap.x3)):
                        if(Snap.q['v1'][i,j,k]>=v_cutoff):
                            numer += (Snap.ion_fractions[i,j,k]  * 
                                      Snap.del1[i] * Snap.del2[j] * Snap.del3[k])
                            denom += (Snap.del1[i] * Snap.del2[j] * Snap.del3[k])
    if(denom==0): return 0
    return numer/denom
    
def emis_avg_3d_ionfrac(Snap, nu, v_cutoff, scale):
    Snap.load_intensity_variables(nu)
    numer = 0
    denom = 0
    for i in range(len(Snap.x1)):
        if((Snap.x1[i]) <= scale):
            for j in range(len(Snap.x2)):
                if(np.abs(Snap.x2[j] <= scale/2)):
                    for k in range(len(Snap.x3)):
                        if(Snap.q['v1'][i,j,k]>=v_cutoff):
                            numer += (Snap.ion_fractions[i,j,k] * Snap.emission_coefs[i,j,k] * 
                                      Snap.del3[k])
                            denom += (Snap.emission_coefs[i,j,k] * 
                                      Snap.del3[k])
    if(denom==0): return 0
    return numer/denom


#####################################################################################################
################### Map Average ionization fractions for Table ######################################
#####################################################################################################


def calculate_column_mass_densities(Snap):
    Snap.column_mass_densities = np.zeros((len(Snap.x1), len(Snap.x2)))
    for i in range(len(Snap.x1)):
        for j in range(len(Snap.x2)):
            numer = 0
            denom = 0
            for k in range(len(Snap.x3)):
                numer += (Snap.q['d'][i,j,k] * Snap.del3[k]) #number density
                denom += Snap.del3[k]
            if (denom == 0):
                Snap.column_mass_densities[i,j] = 0
            else:
                Snap.column_mass_densities[i,j] = numer/denom  

def mass_avg_map_ionfrac(Snap, v, scale, cooled=False):
    ionfrac_mass = Snap.ionfrac_mass_ratio if cooled else Snap.ionfrac_mass

    numer = 0
    denom = 0
    for i in range(len(Snap.x1)):
        if((Snap.x1[i]) <= scale):
            for j in range(len(Snap.x2)):
                if(np.abs(Snap.x2[j] <= scale/2)):
                    numer += (ionfrac_mass[v,i,j] * Snap.column_mass_densities[i,j] 
                              * Snap.del1[i] * Snap.del2[j])
                    denom += Snap.column_mass_densities[i,j] * Snap.del1[i] * Snap.del2[j]
    Snap.avg_map_ionfrac_mass = numer/denom
    return numer/denom

def vol_avg_map_ionfrac(Snap, v, scale, cooled=False):
    ionfrac_vol = Snap.ionfrac_vol_ratio if cooled else Snap.ionfrac_vol
    numer = 0
    denom = 0
    for i in range(len(Snap.x1)):
        if((Snap.x1[i]) <= scale):
            for j in range(len(Snap.x2)):
                if(np.abs(Snap.x2[j] <= scale/2)):
                    numer += ionfrac_vol[v,i,j] * Snap.del1[i] * Snap.del2[j]
                    denom += Snap.del1[i] * Snap.del2[j]
    Snap.avg_map_ionfrac_vol = numer/denom
    return numer/denom
    
def emis_avg_map_ionfrac(Snap, f, v, scale, cooled=False):
    ionfrac_emis = Snap.ionfrac_emis_ratio if cooled else Snap.ionfrac_emis
    InuA = Snap.InuA_const if cooled else Snap.InuA
    # InuA = Snap.InuA
    Snap.load_intensity_variables(FREQS[f])
    numer = 0
    denom = 0
    for i in range(len(Snap.x1)):
        if((Snap.x1[i]) <= scale):
            for j in range(len(Snap.x2)):
                if(np.abs(Snap.x2[j] <= scale/2)):
                    numer += ionfrac_emis[f,v,i,j] * InuA[i,j] * Snap.del1[i] * Snap.del2[j]
                    denom += InuA[i,j] * Snap.del1[i] * Snap.del2[j]
    Snap.avg_map_ionfrac_emis = numer/denom
    return numer/denom



#####################################################################################################
################### Profiles ######################################
#####################################################################################################

def mass_ionfrac_average(Snap, year, xmin, xmax, zmin, zmax, v_min, ymin=0, ymax=False, to_print=False):
    if(not ymax): ymax = len(Snap.x3)
    if(to_print): print('(%.1f <= z < %.1f), (%.1f <= x < %.1f), (%.1f <= y <= %.1f)' % 
                         (Snap.x1[zmin], Snap.x1[zmax], Snap.x2[xmin], Snap.x2[xmax], Snap.x3[ymin], Snap.x3[ymax-1]))
    numer = 0
    denom = 0
    for i in range(zmin, zmax):
        for j in range(xmin, xmax):
            for k in range(len(Snap.x3)):
                if(Snap.q['v1'][i,j,k] >= v_min):
                    numer += Snap.ion_fractions[i,j,k] * Snap.q['d'][i,j,k] * Snap.del3[k]
                    denom += Snap.q['d'][i,j,k] * Snap.del3[k]
    if(denom==0): return 0
    return(numer/denom)
def vol_ionfrac_average(Snap, year, xmin, xmax, zmin, zmax, v_min, ymin=0, ymax=False, to_print=False):
    if(not ymax): ymax = len(Snap.x3)
    if(to_print): print('(%.1f <= z < %.1f), (%.1f <= x < %.1f), (%.1f <= y <= %.1f)' % 
                         (Snap.x1[zmin], Snap.x1[zmax], Snap.x2[xmin], Snap.x2[xmax], Snap.x3[ymin], Snap.x3[ymax-1]))
    numer = 0
    denom = 0
    for i in range(zmin, zmax):
        for j in range(xmin, xmax):
            for k in range(len(Snap.x3)):
                if(Snap.q['v1'][i,j,k] >= v_min):
                    numer += Snap.ion_fractions[i,j,k] * Snap.del3[k]
                    denom += Snap.del3[k]
    if(denom==0): return 0
    return(numer/denom)

def emis_ionfrac_average(Snap, year, nu, xmin, xmax, zmin, zmax, v_min, ymin=0, ymax=False, to_print=False):
    Snap.load_intensity_variables(nu)
    if(not ymax): ymax = len(Snap.x3)
    if(to_print): print('(%.1f <= z < %.1f), (%.1f <= x < %.1f), (%.1f <= y <= %.1f)' % 
                         (Snap.x1[zmin], Snap.x1[zmax], Snap.x2[xmin], Snap.x2[xmax], Snap.x3[ymin], Snap.x3[ymax-1]))
    numer = 0
    denom = 0
    for i in range(zmin, zmax):
        for j in range(xmin, xmax):
            for k in range(len(Snap.x3)):
                if(Snap.q['v1'][i,j,k] >= v_min):
                    numer += Snap.ion_fractions[i,j,k] * Snap.emission_coefs[i,j,k] * Snap.del3[k] #with delta
                    denom += Snap.emission_coefs[i,j,k] * Snap.del3[k]
    if(denom==0): return 0
    return(numer/denom)

def make_mass_ionfrac_arrays(
        Snap, Year, 
        ionfrac_heights, z_segments, xsquare, xslice, ):
    Snap.mass_ionfrac_xsquare = np.zeros((len(v_mins), len(ionfrac_heights)))
    Snap.mass_ionfrac_xslice = np.zeros((len(v_mins), len(ionfrac_heights)))
    for i in range(25):
        for v in range(len(v_mins)):
            Snap.mass_ionfrac_xsquare[v,i] = mass_ionfrac_average(Snap, Year, xsquare[0], xsquare[1], z_segments[i], z_segments[i+1], v_min=v_mins[v])
            Snap.mass_ionfrac_xslice[v,i] = mass_ionfrac_average(Snap, Year, xslice[0], xslice[1], z_segments[i], z_segments[i+1], v_min=v_mins[v])
    print('%s mass ionfrac arrays made' % (Year))
def make_vol_ionfrac_arrays(
        Snap, Year, 
        ionfrac_heights, z_segments, xsquare, xslice, ):
    Snap.vol_ionfrac_xsquare = np.zeros((len(v_mins), len(ionfrac_heights)))
    Snap.vol_ionfrac_xslice = np.zeros((len(v_mins), len(ionfrac_heights)))
    for i in range(25):
        for v in range(len(v_mins)):
            Snap.vol_ionfrac_xsquare[v,i] = vol_ionfrac_average(Snap, Year, xsquare[0], xsquare[1], z_segments[i], z_segments[i+1], v_min=v_mins[v])
            Snap.vol_ionfrac_xslice[v,i] = vol_ionfrac_average(Snap, Year, xslice[0], xslice[1], z_segments[i], z_segments[i+1], v_min=v_mins[v])
    print('%s vol ionfrac arrays made' % (Year))
def make_emis_ionfrac_arrays(
        Snap, Year, nu,
        ionfrac_heights, z_segments, xsquare, xslice, ):
    Snap.emis_ionfrac_xsquare = np.zeros((len(v_mins), len(ionfrac_heights)))
    Snap.emis_ionfrac_xslice = np.zeros((len(v_mins), len(ionfrac_heights)))
    for i in range(25):
        for v in range(len(v_mins)):
            Snap.emis_ionfrac_xsquare[v,i] = emis_ionfrac_average(Snap, Year, nu, xsquare[0], xsquare[1], z_segments[i], z_segments[i+1], v_min=v_mins[v])
            Snap.emis_ionfrac_xslice[v,i] = emis_ionfrac_average(Snap, Year, nu, xslice[0], xslice[1], z_segments[i], z_segments[i+1], v_min=v_mins[v])
    print('%s emis ionfrac arrays made (nu=%s)' % (Year, str(nu/10**9)))


def save_ionfrac_profile(Snap):
    data_path = (VICO_loc+'/Data/'+Snap.name+'/')
    np.savez((data_path+Snap.name+'_ionfrac_profile.npz'), 
             mass_ionfrac_xsquare = Snap.mass_ionfrac_xsquare, mass_ionfrac_xslice = Snap.mass_ionfrac_xslice,
             vol_ionfrac_xsquare = Snap.vol_ionfrac_xsquare, vol_ionfrac_xslice = Snap.vol_ionfrac_xslice, 
             emis_ionfrac_xsquare = Snap.emis_ionfrac_xsquare, emis_ionfrac_xslice = Snap.emis_ionfrac_xslice)
def load_ionfrac_profile(Snap):
    data_path = (VICO_loc+'/Data/'+Snap.name+'/')
    loaded_ionfrac_profile= np.load((data_path+Snap.name+'_ionfrac_profile.npz'))
    Snap.mass_ionfrac_xsquare = loaded_ionfrac_profile['mass_ionfrac_xsquare']
    Snap.mass_ionfrac_xslice =  loaded_ionfrac_profile['mass_ionfrac_xslice']
    Snap.vol_ionfrac_xsquare = loaded_ionfrac_profile['vol_ionfrac_xsquare']
    Snap.vol_ionfrac_xslice = loaded_ionfrac_profile['vol_ionfrac_xslice']
    Snap.emis_ionfrac_xsquare =  loaded_ionfrac_profile['emis_ionfrac_xsquare']
    Snap.emis_ionfrac_xslice = loaded_ionfrac_profile['emis_ionfrac_xslice']
    print(Snap, 'ionfrac_profile loaded')

#############3 ratio

def mass_ionfrac_average_ratio(Snap, year, xmin, xmax, zmin, zmax, v_min, ymin=0, ymax=False, to_print=False):
    if(not ymax): ymax = len(Snap.x3)
    if(to_print): print('(%.1f <= z < %.1f), (%.1f <= x < %.1f), (%.1f <= y <= %.1f)' % 
                         (Snap.x1[zmin], Snap.x1[zmax], Snap.x2[xmin], Snap.x2[xmax], Snap.x3[ymin], Snap.x3[ymax-1]))

    # scale_factor = Snap.cooling_times/Snap.flow_times
    # scale_factor[scale_factor > 1.0] = 1.0
    # zero_vels = np.ones_like(scale_factor)
    # zero_vels[Snap.q['v1'] < v_min] = 0 # exclude cells without v_z >= v_cutoff
    # numer = np.sum(Snap.ion_fractions * Snap.q['d'] * Snap.del3[np.newaxis,np.newaxis,:] * scale_factor * zero_vels)
    # denom = np.sum(Snap.q['d'] * Snap.del3[np.newaxis,np.newaxis,:] * zero_vels)
    numer = 0
    denom = 0
    for i in range(zmin, zmax):
        for j in range(xmin, xmax):
            for k in range(len(Snap.x3)):
                if(Snap.q['v1'][i,j,k] >= v_min):
                    scale_factor = (Snap.cooling_times[i,j,k]
                                    /Snap.flow_times[i,j,k])
                    if(scale_factor>1): scale_factor=1 
                        # can't be more than 1  
                    numer += (Snap.ion_fractions[i,j,k] * 
                              Snap.q['d'][i,j,k] 
                              * Snap.del3[k] 
                              * scale_factor)
                    denom += Snap.q['d'][i,j,k] * Snap.del3[k]
    if denom==0: return 0
    return(numer/denom)

def vol_ionfrac_average_ratio(Snap, year, xmin, xmax, zmin, zmax, v_min, ymin=0, ymax=False, to_print=False):
    if(not ymax): ymax = len(Snap.x3)
    if(to_print): print('(%.1f <= z < %.1f), (%.1f <= x < %.1f), (%.1f <= y <= %.1f)' % 
                         (Snap.x1[zmin], Snap.x1[zmax], Snap.x2[xmin], Snap.x2[xmax], Snap.x3[ymin], Snap.x3[ymax-1]))

    # scale_factor = Snap.cooling_times/Snap.flow_times
    # scale_factor[scale_factor > 1.0] = 1.0
    # zero_vels = np.ones_like(scale_factor)
    # zero_vels[Snap.q['v1'] < v_min] = 0 # exclude cells without v_z >= v_cutoff
    # numer = np.sum(Snap.ion_fractions * Snap.del3[np.newaxis,np.newaxis,:] * scale_factor * zero_vels)
    # denom = np.sum(Snap.del3[np.newaxis,np.newaxis,:] * zero_vels)
    numer = 0 
    denom = 0
    for i in range(zmin, zmax):
        for j in range(xmin, xmax):
            for k in range(len(Snap.x3)):
                scale_factor = (Snap.cooling_times[i,j,k]
                                    /Snap.flow_times[i,j,k])
                if(scale_factor>1): scale_factor=1 
                    # can't be more than 1
                if(Snap.q['v1'][i,j,k] >= v_min):
                    numer += (Snap.ion_fractions[i,j,k] 
                              * Snap.del3[k] 
                              * scale_factor)
                    denom += Snap.del3[k]
    if(denom==0): return 0
    return(numer/denom)

def emis_ionfrac_average_ratio(Snap, year, nu, xmin, xmax, zmin, zmax, v_min, ymin=0, ymax=False, to_print=False):
    Snap.load_intensity_variables(nu)
    if(not ymax): ymax = len(Snap.x3)
    if(to_print): print('(%.1f <= z < %.1f), (%.1f <= x < %.1f), (%.1f <= y <= %.1f)' % 
                         (Snap.x1[zmin], Snap.x1[zmax], Snap.x2[xmin], Snap.x2[xmax], Snap.x3[ymin], Snap.x3[ymax-1]))
    # scale_factor = (Snap.cooling_times/Snap.flow_times)
    # scale_factor[scale_factor>1] = 1
    # zero_vels = np.ones_like(scale_factor) 
    # zero_vels[Snap.q['v1'] < v_min] = 0 # exclude cells without v_z >= v_cutoff
    # numer = np.sum(scale_factor * Snap.ion_fractions * Snap.emission_coefs * Snap.del3[np.newaxis,np.newaxis,:] * zero_vels)
    # denom = np.sum(scale_factor * Snap.emission_coefs * Snap.del3[np.newaxis, np.newaxis,:] * zero_vels)
    numer = 0
    denom = 0
    for i in range(zmin, zmax):
        for j in range(xmin, xmax):
            for k in range(len(Snap.x3)):
                if(Snap.q['v1'][i,j,k] >= v_min):
                    scale_factor = (Snap.cooling_times[i,j,k]
                                        /Snap.flow_times[i,j,k])
                    if(scale_factor>1): scale_factor=1 
                        # can't be more than 1  
                    numer += Snap.ion_fractions[i,j,k] * Snap.emission_coefs[i,j,k] * Snap.del3[k] * scale_factor #with delta
                    denom += Snap.emission_coefs[i,j,k] * Snap.del3[k] * scale_factor
    if denom ==0: return 0
    return(numer/denom)

def make_mass_ionfrac_ratio_arrays(Snap, Year,
        ionfrac_heights, z_segments, xsquare, xslice, ):
    Snap.mass_ionfrac_xsquare = np.zeros((len(v_mins), 
                                   len(ionfrac_heights)))
    Snap.mass_ionfrac_xslice = np.zeros((len(v_mins), 
                                   len(ionfrac_heights)))
    for i in range(25):
        for v in range(len(v_mins)):
            Snap.mass_ionfrac_xsquare[v,i] = \
            mass_ionfrac_average_ratio(Snap, Year, 
                                xsquare[0], xsquare[1], 
                        z_segments[i], z_segments[i+1], 
                                       v_min=v_mins[v])
            Snap.mass_ionfrac_xslice[v,i] = \
            mass_ionfrac_average_ratio(Snap, Year, 
                                xslice[0], xslice[1], 
                        z_segments[i], z_segments[i+1], 
                                       v_min=v_mins[v])
    print('%s mass ionfrac ratio arrays made' % (Year))

def make_vol_ionfrac_ratio_arrays(Snap, Year,
        ionfrac_heights, z_segments, xsquare, xslice, ):
    Snap.vol_ionfrac_xsquare = np.zeros((len(v_mins), 
                                len(ionfrac_heights)))
    Snap.vol_ionfrac_xslice = np.zeros((len(v_mins), 
                                len(ionfrac_heights)))
    for i in range(25):
        for v in range(len(v_mins)):
            Snap.vol_ionfrac_xsquare[v,i] = \
            vol_ionfrac_average_ratio(Snap, Year, 
                                xsquare[0], xsquare[1], 
                        z_segments[i], z_segments[i+1], 
                                      v_min=v_mins[v])
            Snap.vol_ionfrac_xslice[v,i] = \
            vol_ionfrac_average_ratio(Snap, Year, 
                                xslice[0], xslice[1], 
                        z_segments[i], z_segments[i+1], 
                                      v_min=v_mins[v])
    print('%s vol ionfrac ratio arrays made' % (Year))

def make_emis_ionfrac_ratio_arrays(Snap, Year, nu,
        ionfrac_heights, z_segments, xsquare, xslice, ):
    Snap.emis_ionfrac_xsquare = np.zeros((len(v_mins), 
                                len(ionfrac_heights)))
    Snap.emis_ionfrac_xslice = np.zeros((len(v_mins), 
                                len(ionfrac_heights)))
    for i in range(25):
        for v in range(len(v_mins)):
            Snap.emis_ionfrac_xsquare[v,i] = \
            emis_ionfrac_average_ratio(Snap, Year, nu, 
                                xsquare[0], xsquare[1], 
                        z_segments[i], z_segments[i+1], 
                                       v_min=v_mins[v])
            Snap.emis_ionfrac_xslice[v,i] = \
            emis_ionfrac_average_ratio(Snap, Year, nu, 
                                xslice[0], xslice[1], 
                        z_segments[i], z_segments[i+1], 
                                       v_min=v_mins[v])
    print('%s emis ionfrac ratio arrays made (nu=%s)' % (Year, str(nu/10**9)))
    
             
def save_ionfrac_ratio_profile(Snap):
    data_path = (VICO_loc+'/Data/'+Snap.name+'/')
    np.savez((data_path+Snap.name
              +'_ionfrac_ratio_profile.npz'), 
             mass_ionfrac_xsquare = \
             Snap.mass_ionfrac_xsquare, 
             mass_ionfrac_xslice = \
             Snap.mass_ionfrac_xslice,
             vol_ionfrac_xsquare = \
             Snap.vol_ionfrac_xsquare, 
             vol_ionfrac_xslice = \
             Snap.vol_ionfrac_xslice, 
             emis_ionfrac_xsquare = \
             Snap.emis_ionfrac_xsquare, 
             emis_ionfrac_xslice = \
             Snap.emis_ionfrac_xslice)
    
def load_ionfrac_ratio_profile(Snap):
    data_path = (VICO_loc+'/Data/'+Snap.name+'/')
    loaded_ionfrac_profile= np.load((data_path+Snap.name
                        +'_ionfrac_ratio_profile.npz'))
    Snap.mass_ionfrac_xsquare = \
    loaded_ionfrac_profile['mass_ionfrac_xsquare']
    Snap.mass_ionfrac_xslice =  \
    loaded_ionfrac_profile['mass_ionfrac_xslice']
    Snap.vol_ionfrac_xsquare = \
    loaded_ionfrac_profile['vol_ionfrac_xsquare']
    Snap.vol_ionfrac_xslice = \
    loaded_ionfrac_profile['vol_ionfrac_xslice']
    Snap.emis_ionfrac_xsquare =  \
    loaded_ionfrac_profile['emis_ionfrac_xsquare']
    Snap.emis_ionfrac_xslice = \
    loaded_ionfrac_profile['emis_ionfrac_xslice']
    print(Snap, 'ionfrac_ratio_profile loaded')

    ####################### NEW FLUX PROFILES ######################
    ################################################################
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

def mass_chi_in_slice(snap, v_min, ii_min, ii_max, ratio=False):
    """ ii_min = index of z_min, ii_max = index of z_max"""
    scale_factor = snap.cooling_times/snap.flow_times
    if ratio is False:
        scale_factor = np.ones_like(scale_factor)
    else:
        scale_factor[scale_factor > 1.0] = 1.0
    zero_factor = np.ones_like(scale_factor)
    zero_factor[snap.v1<v_min] = 0

    numer = (scale_factor * snap.q['d'] * snap.del3[np.newaxis,np.newaxis,:] * zero_factor
             * snap.ion_fractions)
    denom = (scale_factor * snap.q['d'] * snap.del3[np.newaxis,np.newaxis,:] * zero_factor)


def make_mass_chi_profiles(snap, zseg, ratio=False, v_mins=v_mins):
    chi_profile = np.zeros(len(v_mins),len(zseg)-1 )
    for v_min in v_mins:
        for ii in range(len(zseg)-1):
            mass_chi_in_slice(snap, v_min, zseg[ii], zseg[ii+1], ratio=ratio)