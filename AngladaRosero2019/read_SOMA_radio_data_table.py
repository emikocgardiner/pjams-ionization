#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 18:31:26 2017

@author: Vivi
"""

## This script is for reading the radio data for the SOMA sources for the core scales
## modified with the table that I made by hand including the results from imfit
import numpy as np
import sys, os
VICO_loc = '/scratch/ecg6wm/VICO'
sys.path.append(VICO_loc+'/AngladaRosero2019')

radio_dat = np.genfromtxt( VICO_loc+'/AngladaRosero2019/AngladaData/SOMA_Radio_data_table_core_v2.txt', names=True, dtype=None )

def get_radio_data(source,comp,column):
    hits = np.where( (radio_dat['source'] == source) & (radio_dat['comp'] == comp) )
    if type(column) == type('str'):
        return radio_dat[hits][column]
    elif type(column) == type([]):
        return [radio_dat[hits][this_column] for this_column in column]



radio_dat_lit = np.genfromtxt( VICO_loc+'/AngladaRosero2019/AngladaData/soma_flux_lit.txt', names=True, dtype=None )

def get_radio_data_lit(source,origin,column):
    hits = np.where( (radio_dat_lit['source'] == source) & (radio_dat_lit['scale'] == origin) )
    if type(column) == type('str'):
        return radio_dat_lit[hits][column]
    elif type(column) == type([]):
        return [radio_dat_lit[hits][this_column] for this_column in column]


infrared_data = np.genfromtxt( VICO_loc+'/AngladaRosero2019/AngladaData/SOMA_infrared_data_table.txt' , names=True, dtype=None )

def get_infrared_data(source,column):
    hits = np.where( (infrared_data['source'] == source) )
    if type(column) == type('str'):
        return infrared_data[hits][column]
    elif type(column) == type([]):
        return [infrared_data[hits][this_column] for this_column in column]


#get_radio_data_lit(source, 'SOMA', 'fluxJy')

#flux = get_radio_data_lit('Cepheus_A', 'core', ['fluxJy'])
# get some radio data
#hits = np.where( (radio_dat['source'] == 'G35.20-0.74') & (radio_dat['comp'] == 'A') )
#freq = radio_dat[hits]['Freq']
#flux = radio_dat[hits]['FluxJy']
#
#
## or like this
#flux2 = radio_dat[(radio_dat['source'] == 'G35.20-0.74') & (radio_dat['comp'] == 'A')]['FluxJy']
#
#
#
## or like this

#freq,flux = get_radio_data('G35.20-0.74', 'A', ['Freq','FluxJy'])
#flux = get_radio_data('G35.20-0.74', 'A', ['FluxJy'])
#
#
#import pylab as pl
#pl.plot(flux,freq,'o')
