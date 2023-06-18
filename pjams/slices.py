import sys, os, inspect
import sys
sys.path.append('/Users/emigardiner/VICO/pjams-ionization/pjams-ionization/')

from zeusmp_snapshot_reader import read_zeusmp_snapshot
from zeusmp_snapshot_reader import ScaleFactors
from snapshot import *
from basic_snapshot import basic_snapshot, FREQS, VICO_loc
import plot

import numpy as np

import matplotlib.colors as colors
from obspy.imaging.cm import viridis_white_r 

SAVELOC = VICO_loc+'/pillowfiles/slices_fig'

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap('jet')
new_jet_cmap = truncate_colormap(cmap, 0, .5)

def density_slice_pcolormesh(Snap, year, scale, vmin, vmax, cmap='cividis', 
                        saveloc=False, show=False, show_cbar= False, 
                             show_xlabels=True, show_ylabels=True, cbar_pad=.2):
    Snap.X1_2v, Snap.X2_1v = np.meshgrid(Snap.x1, Snap.x2)
    fig = plt.figure(figsize = (4,4))
    fig.tight_layout()
    ax = plt.subplot()
    ax.set_aspect(1)
    ax.set_xlim(-scale/2, scale/2)
    ax.set_ylim(0, scale)
    ax.set_facecolor('darkkhaki')

    cont = ax.pcolormesh(Snap.X2_1v, Snap.X1_2v, np.log10(np.rot90(Snap.q['d'][:,:,Snap.mid3])), 
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
        cbar = fig.colorbar(cont, orientation = 'horizontal', 
                            fraction=0.038, pad=cbar_pad)
        cbar.set_label(label = (r'log($\rho$ / [g/cm^3])'), 
                       fontsize=20,)
        cbar.ax.tick_params(rotation=45, labelsize = 18)
    ax.text(.02, .3, (r'$\mathbf{\rho}$'+'\n'+year), ha='left', va='top', 
            transform=ax.transAxes, fontsize = 28, weight='bold', color='black')#, alpha=.5)
    filename = saveloc+'/'+Snap.name+'_0_density_'+str(scale)+'AU.png'
    if (saveloc != False): 
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    return filename


def zvelocity_slice_pcolormesh(Snap, year, scale, vmin, vmax, cmap=viridis_white_r, 
                               saveloc=False, show=False, show_cbar= False, 
                               show_xlabels=True, show_ylabels=True, cbar_pad=.2):
    Snap.X1_2v, Snap.X2_1v = np.meshgrid(Snap.x1, Snap.x2)
    fig = plt.figure(figsize = (4,4))
    fig.tight_layout()
    ax = plt.subplot()
    ax.set_aspect(1)
    ax.set_xlim(-scale/2, scale/2)
    ax.set_ylim(0, scale)
    ax.set_facecolor('white')

    cont = ax.pcolormesh(Snap.X2_1v, Snap.X1_2v, np.log10(np.rot90(Snap.q['v1'][:,:,Snap.mid3])), 
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
        cbar = fig.colorbar(cont, orientation = 'horizontal', 
                            fraction=0.038, pad=cbar_pad)
        cbar.set_label(label = (r'log($v_z$ / [km/s])'), 
                       fontsize=20)
        cbar.ax.tick_params(rotation=45, labelsize = 18)
    ax.text(.02, .3, (r'$\mathbf{v_z}$'+'\n'+year), ha='left', va='top', 
            transform=ax.transAxes, fontsize = 28, weight='bold', color='black')#, alpha=.5)
    filename = saveloc+'/'+Snap.name+'_1_zvelocity_'+str(scale)+'AU.png'
    if (saveloc != False): 
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    return filename

        
        
def temp_slice_pcolormesh(Snap, year, scale, vmin, vmax, cmap='hot', 
                        saveloc=False, show=False, show_cbar= False, 
                          show_xlabels=True, show_ylabels=True, cbar_pad=.2):
    Snap.X1_2v, Snap.X2_1v = np.meshgrid(Snap.x1, Snap.x2)
    fig = plt.figure(figsize = (4,4))
    fig.tight_layout()
    ax = plt.subplot()
    ax.set_aspect(1)
    ax.set_xlim(-scale/2, scale/2)
    ax.set_ylim(0, scale)
    ax.set_facecolor('black')

    cont = ax.pcolormesh(Snap.X2_1v, Snap.X1_2v, np.log10(np.rot90(Snap.temperatures[:,:,Snap.mid3])), 
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
        cbar = fig.colorbar(cont, orientation = 'horizontal', 
                            fraction=0.038, pad=cbar_pad)
        cbar.set_label(label = (r'log($T$ / [K])'), 
                       fontsize=20)
        cbar.ax.tick_params(rotation=45, labelsize = 18)
    ax.text(.02, .3, (r'$\mathbf{T}$'+'\n'+year), ha='left', va='top', 
            transform=ax.transAxes, fontsize = 28, weight='bold', color='white')#, alpha=.5)
    filename = saveloc+'/'+Snap.name+'_2_temp_'+str(scale)+'AU.png'
    if (saveloc != False): 
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    return filename

        
def ionfrac_slice_pcolormesh(Snap, year, scale, vmin, vmax, cmap='viridis', 
                        saveloc=False, show=False, show_cbar= False, 
                             show_xlabels=True, show_ylabels=True, cbar_pad=.2):
    Snap.X1_2v, Snap.X2_1v = np.meshgrid(Snap.x1, Snap.x2)
    fig = plt.figure(figsize = (4,4))
    fig.tight_layout()
    ax = plt.subplot()
    ax.set_aspect(1)
    ax.set_xlim(-scale/2, scale/2)
    ax.set_ylim(0, scale)
    ax.set_facecolor('black')

    cont = ax.pcolormesh(Snap.X2_1v, Snap.X1_2v, np.log10(np.rot90(Snap.ion_fractions[:,:,Snap.mid3])), 
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
        cbar = fig.colorbar(cont, orientation = 'horizontal', 
                            fraction=0.038, pad=cbar_pad)
        cbar.set_label(label = (r'log($\chi_\mathrm{H+}$)'), 
                      fontsize=20)
        cbar.ax.tick_params(rotation=45, labelsize = 18)
    ax.text(.02, .3, (r'$\mathbf{\chi_{H+}}$'+'\n'+year), ha='left', va='top', 
            transform=ax.transAxes, fontsize = 28, weight='bold', color='white')#, alpha=.5)
    filename = saveloc+'/'+Snap.name+'_3_ionfrac_'+str(scale)+'AU.png'
    if (saveloc != False): 
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    return filename

        
def emis_slice_pcolormesh(Snap, year, scale, vmin, vmax, cmap='magma', nu = 5.3*10**9,
                        saveloc=False, show=False, show_cbar= False, 
                          show_xlabels=True, show_ylabels=True, cbar_pad=.2):
    Snap.load_intensity_variables(nu)
    Snap.X1_2v, Snap.X2_1v = np.meshgrid(Snap.x1, Snap.x2)
    fig = plt.figure(figsize = (4,4))
    fig.tight_layout()
    ax = plt.subplot()
    ax.set_aspect(1)
    ax.set_xlim(-scale/2, scale/2)
    ax.set_ylim(0, scale)
    ax.set_facecolor('black')

    cont = ax.pcolormesh(Snap.X2_1v, Snap.X1_2v, np.log10(np.rot90(Snap.emission_coefs[:,:,Snap.mid3]))+26, 
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
        cbar = fig.colorbar(cont, orientation = 'horizontal', 
                            fraction=0.038, pad=cbar_pad)
        cbar.set_label(label = (r'log($j_{ff,\nu}$ / [mJy cm$^{-1}$)]'),
                      fontsize=20)
        cbar.ax.tick_params(rotation=45, labelsize = 18)
    ax.text(.02, .3, (r'$\mathbf{j_{ff,\nu}}$'+'\n'+year), ha='left', va='top', 
                transform=ax.transAxes, fontsize = 28, weight='bold', color='white')#, alpha=.5)

    filename = saveloc+'/'+Snap.name+'_4_emissivity_'+str(scale)+'AU.png'
    if (saveloc != False):
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    return filename

        