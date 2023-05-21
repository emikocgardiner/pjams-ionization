#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 16:46:52 2017

@author: Vivi
"""

import pylab as pl
import numpy as np
#from matplotlib.font_manager import FontProperties
#from matplotlib.ticker import MaxNLocator
#from matplotlib.ticker import MultipleLocator
#import pickle
from scipy import optimize
from scipy import stats
#from matplotlib.ticker import ScalarFormatter
#from scipy.stats import mode
#from matplotlib.pyplot import annotate
#import matplotlib.patches as patches
#from scipy.stats import ks_2samp
#from scipy.stats import mode
#import itertools
#from astroquery.vizier import Vizier
#import astropy.units as u
#import astropy.coordinates as coord
#from scipy.integrate import simps
#from numpy import trapz
#from pprint import pprint as pp
#import math
#from read_SOMA_radio_data_table_vjosh import get_radio_data
#import matplotlib
#from matplotlib.backends.backend_pgf import FigureCanvasPgf
#matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
#
##import matplotlib.pyplot as plt
#
#pgf_with_latex = {
#    "text.usetex": True,            # use LaTeX to write all text
#    "pgf.rcfonts": False,           # Ignore Matplotlibrc
#    "pgf.preamble": [
#        r'\usepackage{color}'     # xcolor for colours
#    ]
#}
#matplotlib.rcParams.update(pgf_with_latex)

from read_SOMA_radio_data_table import get_radio_data,get_radio_data_lit,get_infrared_data

#from itertools import cycle

powerlaw2=lambda freq,tt,alpha: tt*freq**alpha

#this is to calculate the power-law fit and error in the fit

def curve_and_bootstrap(nu, fluxX, fluxXerr, doPlot=False, histAxis='', histLabel=''):

    #do the fit
    (p,p_conv)=optimize.curve_fit(powerlaw2,nu,fluxX,sigma=fluxXerr)

    #bootstrap
    allpnew=[]
    for i in range(1000):
        newfluxX=fluxX + fluxXerr*pl.randn(len(fluxX))
        try:
            (pnew,pnew_conv)=optimize.curve_fit(powerlaw2,nu,newfluxX,sigma=fluxXerr)
            allpnew.append(pnew[1])
        except:
            continue

    if doPlot:        
#        histfig=figure().gca()
#        figure()
        histAxis.hist(allpnew,bins=40,normed=True)
        x1=pl.linspace(min(allpnew),max(allpnew),1000)
        histfit=stats.norm.pdf(x1,scale=np.std(allpnew),loc=np.mean(allpnew))
        histAxis.set_xlim(min(allpnew),max(allpnew)) 
        histAxis.plot(x1,histfit,'r-',lw=3)
        histAxis.set_title(r'$\sigma$= %.3f'%(np.std(allpnew)))
        
        histAxis.text(.1,.85,histLabel,transform=histAxis.transAxes)
#        savefig('G11_hist.ps')       

        #example formatted string 
        #x='blah %.3f and %i' % (74.11122233344, 13.0)

    #Error from the fit (curve_fit)
    #This is the error of my alpha from curve_fit
    print('std allpnew='+str(np.std(allpnew)))
    return (p, np.std(allpnew))




def log_calc(fluxX, fluxXerr):
    fluxXerr = np.sqrt(fluxXerr**2 + (0.1*fluxX)**2) #Accurate to within 10%
    fluxXlog = np.log10(fluxX)
    fluxXerrlog = fluxXerr/ (fluxX * np.log(10))    
    fluxXerrlogpos = -fluxXlog + np.log10(fluxX+fluxXerr)
    fluxXerrlogneg = fluxXlog - np.log10(fluxX-fluxXerr)
    return (fluxXerr, fluxXlog, fluxXerrlog, fluxXerrlogpos, fluxXerrlogneg)



# origin can be 'single_component', 'component_combination', 'SOMA', or 'intermediate'
def get_radio_flux( source, comp='A', origin='single_componenent', fractional_error=0.10, component_combinations={}, imstat_results={}):
    ## values from the core scale
    if origin=='single_component':
          flux= get_radio_data(source, comp, 'FluxJy')/1e-3
          freq= get_radio_data(source, comp, 'Freq')
          fluxerr = get_radio_data(source, comp, 'sig_flux')/1e-3
          fluxerr = np.sqrt(np.array(fluxerr)**2 + (fractional_error*np.array(flux))**2)
          fluxerr = list(fluxerr)
          
    elif origin=='component_combination':
          flux= component_combinations[source]['combined'][comp]['flux_comb']
          freq= component_combinations[source]['combined'][comp]['freq_comb'] 
          fluxerr = fractional_error*flux
    
    elif origin in [ 'SOMA', 'intermediate' ]:
          bands = imstat_results[source][origin].keys()
          flux = [imstat_results[source][origin][band]['region']['flux'][0]/1e-3 for band in bands]
          freq = [imstat_results[source]['frequency'][band]/1e9 for band in bands]
          beamarea= [imstat_results[source]['beamarea'][band] for band in bands]
          rms = [imstat_results[source][origin][band]['rms']['rms'][0]/1e-3 for band in bands]
          npts = [imstat_results[source][origin][band]['region']['npts'][0] for band in bands]
          fluxerr = [rms[i] * np.sqrt( npts[i] / beamarea[i] ) for i in range(len(bands)) ]
          fluxerr = np.sqrt(np.array(fluxerr)**2 + (fractional_error*np.array(flux))**2)
          fluxerr = list(fluxerr)                   
#    else:
#        freq,flux,fluxerr=0,0,0        
                  
    return (freq,flux,fluxerr)



def get_radio_flux_lit( source, origin='intermediate' ):
    
    flux= get_radio_data_lit(source, origin, 'fluxJy')/1e-3
    freq= get_radio_data_lit(source, origin, 'freqGHz')
    try:
        fluxerr = np.array( get_radio_data_lit(source, origin, 'sig_flux'), dtype='float64')/1e-3 
    except:
#        raise
        fluxerr = 0.1*flux
    
    return (freq,flux,fluxerr)



def plot_radio_flux( freq,flux,fluxerr, upper='zero', color='r' ):

      flux_I20_err, flux_log, flux_I20_errlog, errlogpos, errlogneg = log_calc( np.array(flux), np.array(fluxerr) )
      errlogneg[errlogneg!=errlogneg] = 10#2* errlogpos[errlogneg!=errlogneg]
      
      for i,radio_freq in enumerate(freq):
          if upper=='zero':
              is_limit = flux[i] > 0
          else:
              is_limit = flux[i] > 3*fluxerr[i]

            ## not upper limits
          if ( is_limit ):               
                flux_errlogneg= [errlogneg[i]] 
                flux_errlogpos= [errlogpos[i]] 
                lim_values_radio= [0]
                radio_flux= flux[i]
                fmt= 'o'

 
            ## upper limits:
          else:
                flux_errlogneg= [0.5]    
                flux_errlogpos= [0]
                lim_values_radio= [1]
                radio_flux= 3*fluxerr[i]
                fmt= ''

                
#            print radio_flux, flux[i],fluxerr[i]
          pl.errorbar([np.log10(radio_freq)], [np.log10(radio_flux)], yerr=[flux_errlogneg, flux_errlogpos],\
                            xerr=None,fmt=fmt, color=color, markeredgewidth=1.5,alpha=0.6, uplims=lim_values_radio, capsize=5)
            


def fit_and_plot_alpha(freq, flux, flux_err, method='exclude', text_offset=0, alpha_text='', color='k', source='', ax1=None ):
    
    if len(freq) >= 2:     

        # to fit upper limits as data
        if method=='upper':
          fit_flux = np.array([ f if (f>0) else 3*flux_err[i] for i,f in enumerate(flux) ])
          pComp, pComp_err = curve_and_bootstrap(freq, fit_flux, flux_err, doPlot=False, histAxis=ax1, histLabel=source+' SOMA')
 
           # to exclude upper limits from fit
        elif method=='exclude':   
            hits = np.where( np.array(flux) > 0 )[0]
            pComp, pComp_err = curve_and_bootstrap( np.array(freq)[hits], np.array(flux)[hits], np.array(flux_err)[hits], doPlot=False, histAxis=ax1, histLabel=source+' SOMA')      

       # use bottom of upper limit arrow as data
        elif method=='bottom':
            
            fit_flux = np.array([ f if (f>0) else 1.5*flux_err[i] for i,f in enumerate(flux) ])
            pComp, pComp_err = curve_and_bootstrap(freq, fit_flux, flux_err, doPlot=False, histAxis=ax1, histLabel=source+' SOMA')
                    
        if type(freq) == type([]):
            freq = sorted(freq)
        pl.plot(np.log10(freq),\
                 np.log10(powerlaw2(freq, pComp[0],pComp[1])),'--',color=color, alpha=0.7)#,'--b')
        print freq,type(freq)

        if np.any( np.array(flux) < 0 ):
          pl.text(.42,text_offset+0.3,r'$\alpha_{'+alpha_text+r'} < $'+str(round(pComp[1],1)).replace('-','$-$'), transform=pl.gca().transAxes, fontsize=18, color=color, alpha=0.8)
        else:      
          pl.text(.42,text_offset+0.3,r'$\alpha_{'+alpha_text+r'}=$'+str(round(pComp[1],1)).replace('-','$-$')+'$\pm$'+\
                    str(round(pComp_err,1)).replace('-','$-$'),\
                 transform=pl.gca().transAxes, fontsize=18, color=color, alpha=0.8)

        return (pComp, pComp_err)
    
    else:
        return ('--','--')
    
    
#return origin of data: measured, lit, or none
# scale can be: core, intermediate, or SOMA
def source_check_scale( source, scale, component_combinations={}, imstat_results={}  ):
 #   print get_radio_data_lit(source, scale, 'scale')
        
    if len(get_radio_data_lit(source, scale, 'scale')) > 0: 
        return 'lit'

    scale = 'single_component' if scale=='core' else scale
    try:
        rf = get_radio_flux( source, origin=scale, component_combinations=component_combinations, imstat_results=imstat_results)
        if len(rf[0]) == 0: raise ValueError
        return 'measured'
    except:
#        raise
        return 'none'
    





def get_infrared( source ):
    
    flux_with = get_infrared_data(source, 'Flux_with_backgJy')/1e-3
    flux_without = get_infrared_data(source, 'Flux_with_out_backgJy')/1e-3
    freq= get_infrared_data(source, 'Freq')
    
    
    err= np.max([abs(flux_without - flux_with),0.1*flux_with], axis=0)#/f11[i]
    err_logpos = -np.log10(flux_with) + np.log10(flux_with + err)
    err_logneg =  np.log10(flux_with) - np.log10(flux_with - err)
    
#    hits = np.where( np.isnan(err_logneg)==True )[0]
    err_logneg[np.isnan(err_logneg)] = 10.

   
#    try:
#        fluxerr = float( get_radio_data_lit(source, origin, 'sig_flux')/1e-3 )
#    except:
#        fluxerr = 0.1*flux
    
    return (freq,flux_with,err,err_logneg,err_logpos)




def plot_infrared_data( freq,flux_with,err,err_logneg, err_logpos, color='m' ):

#      flux_I20_err, flux_log, flux_I20_errlog, errlogpos, errlogneg = log_calc( np.array(flux), np.array(fluxerr) )
 #     errlogneg[errlogneg!=errlogneg] = 10#2* errlogpos[errlogneg!=errlogneg]
      
      for i,ir_freq in enumerate(freq):
 #         is_limit = err_logneg[i] == 0.5   

          
          if (freq[i] >= 37470.) : is_limit = True
          else: is_limit = False
          
#          if (err_logneg[i] == 0.5) :   #(8um)
#              flux_err
        
#          else:
#              is_limit = False
              
          
          
          flux_errlogpos= [err_logpos[i]] 
          ir_flux= flux_with[i]
          fmt= color+'o'
         

            ## not upper limits
          if not is_limit:               ## shortcut for True
                flux_errlogneg= [err_logneg[i]] 
                lim_values_ir= [0]
 
            ## upper limits:
          else:
                flux_errlogneg= [0.5]    
                flux_errlogpos= [2]
                lim_values_ir= [1]
#                ir_flux= flux_with[i]
#                fmt= color

                
#            print radio_flux, flux[i],fluxerr[i]
#      pl.errorbar([np.log10(freq)], [np.log10(flux_with)], yerr=[err_logneg, err_logpos],\
 #                       xerr=None,fmt='ro',  markeredgewidth=1.5,alpha=0.5, uplims=lim_values_ir, capsize=5)

          pl.errorbar([np.log10(ir_freq)], [np.log10(ir_flux)], yerr=[flux_errlogneg, flux_errlogpos],\
                            xerr=None,fmt=fmt,  markeredgewidth=1.5,alpha=0.6, uplims=lim_values_ir, capsize=5)

