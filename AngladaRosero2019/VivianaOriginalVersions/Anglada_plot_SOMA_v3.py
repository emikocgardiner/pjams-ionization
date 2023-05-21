import pylab as pl
import numpy as np
from scipy import stats
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator
import pickle
from pprint import pprint as pp
import seaborn as sns
from def_seds import *



##VARR 07.19.17

## Here I will plot only object within their mm core

do_print= False
do_plot= False



#example: data_dict['18089-1732']




Yichen_models_dict_dir ='Yichen_models_dict.p'
Yichen_models_dict = pickle.load( open( Yichen_models_dict_dir, 'rb' ) )


radio_SOMA_dict_dir ='radio_SOMA_dict.p'
radio_SOMA_dict = pickle.load( open( radio_SOMA_dict_dir, 'rb' ) )

best_Yichen_fits_dict_dir_new_model_Kei ='best_Yichen_fits_dict_new_model_Kei.p'
best_Yichen_fits_dict_new_model_Kei = pickle.load( open( best_Yichen_fits_dict_dir_new_model_Kei, 'rb' ) )

best_Yichen_fits_dict_dir ='best_Yichen_fits_dict.p'
best_Yichen_fits_dict = pickle.load( open( best_Yichen_fits_dict_dir, 'rb' ) )


SOMA_sources_dir =\
                   'SOMA_sources.npy'
SOMA_sources = np.load( SOMA_sources_dir )




#infile_M10 = '/Users/Vivi/PhD_projects/codes/mom_rate_plot/Survey_momentum_rate_v4/Refs/viviana/mcore10.sigma1.dat'
infile_M60 = 'mcore60.sigma1.dat'
#infile_M1000 = '/Users/Vivi/PhD_projects/codes/mom_rate_plot/Survey_momentum_rate_v4/Refs/viviana/mcore1000.sigma1.dat'







#M_star_M10, r_star_M10, L_star_M10, T_star_M10, Q_star_M10, rad_lum_star_M10 = \
#        pl.loadtxt(infile_M10, unpack=True ,usecols=[0, 1, 2, 3, 4, 5])

M_star_M60, r_star_M60, L_star_M60, T_star_M60, Q_star_M60, rad_lum_star_M60 = \
        pl.loadtxt(infile_M60, unpack=True ,usecols=[0, 1, 2, 3, 4, 5])

#M_star_M1000, r_star_M1000, L_star_M1000, T_star_M1000, Q_star_M1000, rad_lum_star_M1000 = \
#        pl.loadtxt(infile_M1000, unpack=True ,usecols=[0, 1, 2, 3, 4, 5])



#




## Table with bol luminosities from best models chi2

lum_bol_tab = '/Users/Vivi/postdoc_UF/SOMA/paper_VLA_SOMA/Tables/latex_tab_chi2_models_v3.txt'
x=np.genfromtxt(lum_bol_tab,dtype=None,delimiter='&')
L = [x1.replace('[9pt]','').lstrip('\t').rstrip(' \\\t') for x1 in x['f14']]
aL = 1e4*np.array(L,dtype='float')
for i,line in enumerate(x):
	if line['f0'] != '': source = line['f0']
	else: x[i]['f0'] = source

sources=np.unique(x['f0'])
L_dict={}
for key in sources:
    hits = np.where(x['f0'] == key)[0]
#   print key, aL[hits[0]],min(aL[hits]),max(aL[hits])
    L_key = key.strip().replace(' ','_').replace('$','')
    for S_key in SOMA_sources:
            
        #print S_key, L_key
        if S_key.startswith(L_key):
                
                L_key = S_key
#           print 'replacing'
#           print S_key,L_key
        L_dict[L_key] = {}
        L_dict[L_key]['best']=aL[hits[0]]
#        if min(aL[hits]) == aL[hits[0]]:
#            L_dict[L_key]['min']=min(aL[hits])*0.99            
#        else:
        L_dict[L_key]['min']=min(aL[hits]) 
        L_dict[L_key]['max']=max(aL[hits])

for L_key in L_dict.keys():
    print L_dict[L_key]['min'],L_dict[L_key]['best'],L_dict[L_key]['max']

#print bla



## Calculating radio lum using 4.9 GHz data and for sources
## with positive spectral index

def Radio_Lum_calc( data_dict, data_dict_radio, this_source, this_comp ):

    if data_dict_radio[ this_source ][ this_comp ]['Flux_4_9'] != '\\nodata' and \
       data_dict_radio[this_source][this_comp]['mm_associat'] == 'y' and\
           (data_dict_radio[ source ][ comp ]['Spec_ind'] != '\\nodata') and \
           (float( data_dict_radio[ source ][ comp ]['Spec_ind']) >= 0.2) and \
           (float( data_dict_radio[ source ][ comp ]['Spec_ind']) < 1.9):

        
        rad_lum_4_9 = (float( data_dict[ this_source ]['dist']))**2 * \
                   (float( data_dict_radio[ this_source ][ this_comp ]['Flux_4_9'] )*1e-3)
        ##Radio luminosity just at 4.9 GHz I need the flux in mJy
        log_rad_lum_4_9 = pl.log10(rad_lum_4_9)
#        pdot_4_9 = 10*10**(-3.5)* rad_lum_4_9

    else:
        rad_lum_4_9 = 'nodata'
        log_rad_lum_4_9 = 'nodata'
#        pdot_4_9 = 'nodata'

    return [this_source, this_comp, data_dict_radio[ this_source ][ this_comp ]['Flux_4_9'], \
                rad_lum_4_9, log_rad_lum_4_9]






## New dictionary with the Literature tables

data_dict_Lit = {}
Lit_sources = []
#infile='/Users/Vivi/PhD_projects/Survey/codes/mom_rate_plot/Survey_momentum_rate_v4/Reference_table.txt'
infile='/Users/Vivi/PhD_projects/codes/mom_rate_plot/Survey_momentum_rate_v4/Reference_table.txt'

##Reading the input from the table
for line in open(infile, 'r'):
    l1 = line.split()
    if l1==[]: continue
    skipChars = ['#']
    if line[0] in skipChars: continue

#    if not 'data_list' in locals(): data_list = [l1]
#    else: data_list += [l1]
    if not data_dict_Lit.has_key( l1[0] ):
            data_dict_Lit[ l1[0] ] = {}
            Lit_sources.append( l1[0] )

    data_dict_Lit[ l1[0] ] = {
                              'Dist':   l1[1],
                              'Flux':  l1[2],
                              'Radio_lum': l1[3],
                              'Lum_bol': l1[4],
                              'Pdot': l1[5],
                              'Refs':l1[6]}



pickle.dump( data_dict_Lit, open( 'data_dict_Lit.p', 'wb' ) )
np.save('Lit_sources', Lit_sources)


## Calculating Pdot from data
## scale factor from 1.8 cm to 6 cm = 2.06
## see my notebook for details
def Pdot_calc( data_dict_Lit, this_source, scale_factor = 1.0):

    if data_dict_Lit[ this_source ]['Flux'] != 'na':
        rad_lum_Lit = (float( data_dict_Lit[ this_source ]['Dist']))**2 * \
                   (float( data_dict_Lit[ this_source ]['Flux'] )*1e-3)/scale_factor
        ##Radio luminosity I need the flux in mJy
#        Pdot_Lit = 10*10**(-3.5)* rad_lum_Lit

    else:
        rad_lum_Lit = 'na'
#        Pdot_Lit = 'na'

    return [this_source, data_dict_Lit[ this_source ]['Flux'], \
                rad_lum_Lit]

################## Scaife's sources
if do_print: print '## Source          Flux(uJy)   Rad_Lum      Pdot'
###Scaife sources fluxes are at 1.8 cm, therefore I need to scale it
for source in Lit_sources:
    if data_dict_Lit[ source ]['Refs'] == 'Scaife_2011' or \
       data_dict_Lit[ source ]['Refs'] == 'Scaife_2012':

        Sc_ret = Pdot_calc( data_dict_Lit, source, 2.06)
        ##Adding the new parameters to the dictionary for each source
        data_dict_Lit[ source ].update({'Rad_Lum_calc': Sc_ret[2]})
 

        line1 = '%-18.22s %-10.5s  %8.2e   '%( tuple(Sc_ret) )
        if do_print: print line1
##################

################## Moscadelli et al 2016 sources
if do_print: print '## Source          Flux(uJy)   Rad_Lum      Pdot'
### sources fluxes are at  C-band, therefore I don't need to scale it
for source in Lit_sources:
    if data_dict_Lit[ source ]['Refs'] == 'Mosca_2016':

        Mosca_ret = Pdot_calc( data_dict_Lit, source, 1.0)
        ##Adding the new parameters to the dictionary for each source
        data_dict_Lit[ source ].update({'Rad_Lum_calc': Mosca_ret[2]})
 

        line1 = '%-18.22s %-10.5s  %8.2e   '%( tuple(Mosca_ret) )
        if do_print: print line1
##################
        

################## Rest of sources

if do_print: print '\n \n'
if do_print: print '## Source          Flux(uJy)   Rad_Lum      Pdo'
### The rest of sources: I am not scaling them:
for source in Lit_sources:
    
    if not data_dict_Lit[ source ]['Refs'] == 'Scaife_2011' and \
       data_dict_Lit[ source ]['Refs'] != 'Scaife_2012' and \
       data_dict_Lit[ source ]['Flux'] != 'na':

        Others_ret = Pdot_calc( data_dict_Lit, source, 1.)  ## this apply pretty much to Ang92, Rod08 and Kurtz95
        ##Adding the new parameters to the dictionary for each source
        data_dict_Lit[ source ].update({'Rad_Lum_calc': Others_ret[2]})


        line1 = '%-18.22s %-10.5s  %8.2e  '%( tuple(Others_ret) )
        if do_print: print line1

##################


##################

## Lyman Continuum line: data from Thompson 1984

##Thompson 1984

logL_Th, logNe_Th, logNL_Th = [], [], []


##Reading the input from the table
infile='/Users/Vivi/PhD_projects/codes/mom_rate_plot/Survey_momentum_rate_v4/'
for line in open(infile+'INPUT_Thompson_1984.txt', 'r'):
    l1 = line.split()
    if l1==[]: continue
    skipChars = ['#']
    if line[0] in skipChars: continue
    this_logL_Th= float(l1[0])
    this_logNe_Th= float(l1[1])
    this_logNL_Th= float(l1[2])

    logL_Th.append( this_logL_Th )
    logNe_Th.append( this_logNe_Th )
    logNL_Th.append( this_logNL_Th )
    


##Calculating the number of Lyman cont photons
##using equation from Monge
T_e= 1e4    #K
nu= 6.       #GHz at 5 cm


rad_lum_Th=  2.08e-46 * 10**(np.array(logNL_Th))* nu**-0.1 * T_e**0.45   ##scaife 2012
rad_lum_Th2= 1.32e-46 * 10**(np.array(logNL_Th))* nu**-0.1 * T_e**0.5   ##Solving from Kurtz et al. 1994

## The equations are equivalent. I will use the one from Scaife 2012

##################

### Pdot is from Literature of my sources grouped in types

def My_sources_plot_info( data_dict, source):
    
    if (data_dict[source]['Pdot_10e-3'] != '--') and \
       (data_dict[source].has_key ('Rad_Lum_4.9_Comps')):
        
        This_Pdot_from_Lit = float(data_dict[source]['Pdot_10e-3'])*1e-3
        This_Rad_Lum_calc_4_9 = float(data_dict[source]['Rad_Lum_4.9_Comps'])
        This_bol_Lum = data_dict[source]['Lum_bol']

    else:
        This_Pdot_from_Lit = '--'
        This_Rad_Lum_calc_4_9 = '--'
        This_bol_Lum = data_dict[source]['Lum_bol']
        

        
    return [source, This_Pdot_from_Lit, This_Rad_Lum_calc_4_9, This_bol_Lum]




#####################


def Radio_Lum_calc_v2( data_dict, data_dict_radio, source, comp ):

    if data_dict_radio[ source ][ comp ]['Flux_4_9'] != '\\nodata':
        rad_lum_4_9 = (float( data_dict[ source ]['dist']))**2 * \
                   (float( data_dict_radio[ source ][ comp ]['Flux_4_9'] )*1e-3)
        ##Radio luminosity just at 4.9 GHz I need the flux in mJy

    else:
        rad_lum_4_9 = 'nodata'

    return [source, comp, data_dict_radio[ source ][ comp ]['Flux_4_9'], rad_lum_4_9]









## Fig1
                
scale= 1.36    #scaling Anglada95 and slide talks data from 3.6 cm to 6 cm, assuming alpha=0.6



pl.figure(9, figsize=(9,9))

#sns.set(rc={"figure.figsize": (7.5,7.5)})
#np.random.seed(sum(map(ord, "palettes")))
#sns.palplot(sns.color_palette("hls", 8))



Fig9='Anglada_plot'
fig9=pl.gcf()
ax9 = fig9.add_axes([.15,.15, .8, .75])



#pl.loglog(10**np.array(logL_Th), rad_lum_Th,'-k', linewidth=3,label='_nolegend_',alpha=0.8)
logL_bol_cl, logNe_05_cl, logNe_95_cl = [], [], []

#cluster_file= '/Users/Vivi/PhD_projects/Survey/codes/Relations/Lyman_lum_plot/lbin_cluster_Lbol-Nlym.dat'
cluster_file= '/Users/Vivi/PhD_projects/codes/Relations/lbin_cluster_Lbol-Nlym.dat'


##Reading the input from the table
for line in open(cluster_file, 'r'):
    l1 = line.split()
    if l1==[]: continue
    skipChars = ['#']
    if line[0] in skipChars: continue
    this_logL_bol_cl= float(l1[0])
    this_logNe_05_cl= float(l1[1])
    this_logNe_95_cl= float(l1[2])

    logL_bol_cl.append( this_logL_bol_cl )
    logNe_05_cl.append( this_logNe_05_cl )
    logNe_95_cl.append( this_logNe_95_cl )




#x= np.linspace(1, 1e6, 100)

rad_lum_cesaroni_data=  2.08e-46 * 10**(np.array(logNe_95_cl))* nu**-0.1 * T_e**0.45 

#pl.loglog(10**pl.array(logL_Th), 10**pl.array(logNL_Th), 'k-', linewidth=3)
pl.loglog(10**pl.array(logL_bol_cl), rad_lum_cesaroni_data, 'k-', linewidth=3)


for source in Lit_sources:
    
##    if  data_dict_Lit[ source ]['Refs'] == 'Anglada_92' and \
##        data_dict_Lit[ source ]['Lum_bol'] != 'na': ## Nothing here
##        
##        Ang_92 = pl.loglog(float(data_dict_Lit[ source ]['Lum_bol']),\
##                data_dict_Lit[ source ]['Rad_Lum_calc'],'.y', markersize=16, \
##                markeredgewidth=1.5,alpha=0.5)

    


    if  data_dict_Lit[ source ]['Refs'] == 'Anglada_95'   and \
        data_dict_Lit[ source ]['Lum_bol'] != 'na':
        
        Ang_95 = pl.loglog(float(data_dict_Lit[ source ]['Lum_bol']),\
                float(data_dict_Lit[ source ]['Radio_lum'])/scale,'.y', markersize=16, \
                markeredgewidth=1.5,alpha=0.5)



        

     
        

    scale_k = 0.95  ## asumming an ~flat spectrum -0.1

## HII sources from Kurtz et al. 1994
## These sources are at 3.6 cm, so I need to scale them to 6 cm
    Unresolved_Kurtz_94= ['G10.841-2.592', 'G28.200-0.049', 'G48.61+0.02', 'G76.383-0.621',\
                      'G138.295+1.555', 'G189.030+0.784', 'G189.876+0.516',\
                      'G206.543-16.347']

    if  data_dict_Lit[ source ]['Refs'] == 'Kurtz_94':
        if source in Unresolved_Kurtz_94:
            continue
        
        
##            Kur_94_U = pl.loglog(float(data_dict_Lit[ source ]['Lum_bol']),\
##                    float(data_dict_Lit[ source ]['Rad_Lum_calc']),'xk', markersize=10, \
##                    markeredgewidth=2.5,alpha=0.5)# Kurtz's HII regions
        else:
            Kur_94 = pl.loglog(float(data_dict_Lit[ source ]['Lum_bol']),\
                    float(data_dict_Lit[ source ]['Rad_Lum_calc'])/scale_k,'xk', markersize=10, \
                    markeredgewidth=2.5,alpha=0.5)# Kurtz's HII regions

    

## bol lum from the best model fit of Yichen
## rad lum around 5 GHz


#other = ['Cepheus_A', 'G45.47+0.05']

#other = ['Cepheus_A']#, 'G45.47+0.05']

#other = ['AFGL_4029', 'AFGL_437', 'NGC_7538_IRS9']
other = ['AFGL_437', 'NGC_7538_IRS9']


## scaled using their spectral indices in inner scale
## I excluded AFGL 4029 since the alpha is 0.7 and the scale factor is 0.96 there is not much difference
scale_8GHz = 0.497  # alpha =1.38  
scale_5_3GHz = 0.94   # alpha = 1

##gray #7f7f7f
new_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#0f0bc6',
              '#9467bd', '#8c564b', '#e377c2', '#d62728',
              '#bcbd22', '#17becf']
i=0
#for i in range(np.size(SOMA_sources))
for source in SOMA_sources:
    i=i+1
    
    flux= get_radio_data(source, 'A', 'FluxJy')/1e-3   #mJy
    freq= get_radio_data(source, 'A', 'Freq')

    print source, flux, freq
    
    

    ##flux[1] is at ~5 GHz

    if (source not in other and source != 'Cepheus_A'):
        rad_lum_SOMA= Yichen_models_dict[source]['distance[kpc]']**2*\
                                                     flux[0]

#        print source, flux[1], freq[1],rad_lum_SOMA, Yichen_models_dict[source]['best_bol_lum']
        print source, flux[0], freq[0],rad_lum_SOMA, Yichen_models_dict[source]['best_bol_lum']

    if source == 'Cepheus_A':

        rad_lum_SOMA= Yichen_models_dict[source]['distance[kpc]']**2*\
                                                     flux[0] * scale_8GHz

    elif source in other:

        rad_lum_SOMA= Yichen_models_dict[source]['distance[kpc]']**2*\
                                                     flux[0] * scale_5_3GHz       

    
        print 'other',source, flux[0], freq[0],rad_lum_SOMA, Yichen_models_dict[source]['best_bol_lum']

    pl.errorbar( L_dict[source]['best'],rad_lum_SOMA,\
                 xerr=[[L_dict[source]['best']-L_dict[source]['min']],\
                       [L_dict[source]['max']-L_dict[source]['best']]],\
                 ecolor=new_colors[i],elinewidth=2,capsize=7,capthick=3, alpha=0.5)

    SOMA_17 = pl.loglog(L_dict[source]['best'],\
                rad_lum_SOMA,'o', color=new_colors[i], markersize=14, \
                markeredgewidth=1.5,alpha=0.5,label=source)

 
    pl.legend(loc=2, numpoints=1,prop=FontProperties(size='medium'), ncol=2)
    

leg= fig9.legend([Ang_95[0], Kur_94[0]],\
              ['Jets low-Mass YSO: Anglada et al. 1995', 'UC/HC HII: Kurtz et al. 1994'],'lower right', \
              prop=FontProperties(size='medium'), numpoints=1)
leg.get_frame().set_alpha(0.2)



    





pl.text(4e-1,1e3, r'Lyman Continuum from ZAMS',fontsize=20)
pl.text(1.5e2,2e-4, r'Lyman Continuum from YSO',fontsize=20, color='teal')

pl.ylabel(r'$S_{\nu}d^{2}$ (mJy kpc$^{2}$)', fontsize=20)
#pl.xlabel(r'L (L$_{\odot}$)', fontsize=20)
pl.xlabel(r'L$_{bol}$ (L$_{\odot}$)', fontsize='20')




pl.gca().set_ylim(1e-4,1.0e6)
pl.gca().set_xlim(1e-1,1.45e6)

pl.gca().tick_params('both', length=10, width=1, which='major')
pl.gca().tick_params('both', length=4, width=1, which='minor')


##x0,x1 = ax6.get_xlim()
##y0,y1 = ax6.get_ylim()
##ax6.set_aspect(abs(x1-x0)/abs(y1-y0))

## Theretical fit from the low mass sources
plot_lum_v2= pl.array(pl.gcf().gca().get_xlim())
#rad_lum_an_v2= 10**(-2.1)*plot_lum_v2**(0.6)
rad_lum_an_v2= 8*10**(-3)*plot_lum_v2**(0.6)

pl.loglog(plot_lum_v2, rad_lum_an_v2,'--k')#,color=mygray)#,'--b')
pl.text(5e-1,5e-0,r'$S_{\nu}d^{2} =\,8 \times 10^{-3}(L_{bol})^{0.6}$',fontsize=22)

label_size = 14
xlabels = ax9.get_xticklabels()
for item in xlabels: item.set_size(label_size)
ylabels = ax9.get_yticklabels()
for item in ylabels: item.set_size(label_size)
pl.tick_params(labelsize=18)




##########


## Tanaka's 2016 data
#pl.loglog(L_star_M10, rad_lum_star_M10,'-', color= 'teal', linewidth=3,label='_nolegend_',alpha=0.8)
pl.loglog(L_star_M60, rad_lum_star_M60,'-', color= 'teal', linewidth=3,label='_nolegend_',alpha=0.8)
#pl.loglog(L_star_M1000, rad_lum_star_M1000,'-', color= 'teal', linewidth=3,label='_nolegend_',alpha=0.8)












pl.show()

    
    



