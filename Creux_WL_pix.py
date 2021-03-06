import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18
import astropy.io.fits as fits
from scipy.ndimage import gaussian_filter
import matplotlib

### data
data = pd.read_csv('../../shared_folder/result_cu1jemw77zite8se.csv')
cleaned_data = data[(data['MAG_r'] != 99.) & (data['MAG_i'] != 99.)]
RAmin, RAmax = 207,222
DECmin, DECmax = 50, 58

#print('Do you want to plot the data ? (y/n)')
plot = 'n'#input()
if plot == 'y' :
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(cleaned_data['Z_B'], bins = 100)

def slice_data(lower, upper, plot = False): #bins : 0.3-0.5, 0.7-0.9
    binned_data = cleaned_data[(cleaned_data['Z_B'] > lower) & (cleaned_data['Z_B'] < upper)]
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        scat = ax.scatter(binned_data['ALPHA_J2000'], binned_data['DELTA_J2000'], s = 0.3, c = r)
        plt.colorbar(scat, label = '$z$')
        ax.set_xlabel(r'$\alpha_{J2000}$(°)')
        ax.set_ylabel(r'$\delta_{J2000}$(°)')
        ax.set_title(f'Galaxy distribution between z = {lower} and z = {upper}')
    return binned_data
    
### density map
RA_Nbins, DEC_Nbins = 80, 150
lower_z, higher_z = 0.3, 0.5
data_in_z_bin = cleaned_data[(cleaned_data['Z_B'] > lower_z) & (cleaned_data['Z_B'] < higher_z)]
data_map, edges = np.histogramdd((data_in_z_bin['ALPHA_J2000'],data_in_z_bin['DELTA_J2000']),bins=(RA_Nbins,DEC_Nbins))
#fits.writeto('density_map.fits',data_map)
def plot_density_map(data_map):
    plt.figure()
    plt.imshow(data_map.T,origin='lower',cmap='jet',extent=(RAmin,RAmax,DECmin,DECmax),aspect='auto')
    plt.xlabel(r'$\alpha_{J2000}$')
    plt.ylabel(r'$\delta_{J2000}$')
    plt.colorbar(label='$N_{gal}$')
    plt.show()

### mask
# density map
w = np.where(data_map==0)
mask = np.zeros((RA_Nbins, DEC_Nbins))
mask[w] = 1
#fits.writeto('mask.fits',mask.flatten())
# dtfe map
zmin, zmax = 0.7, 0.9
data_map = fits.getdata('dtfe_map_z_'+str(zmin)+'_'+str(zmax)+'_512x512.fits')
percentile = 15
w = np.where(data_map<=np.percentile(data_map,percentile))
mask = np.zeros((data_map.shape[0], data_map.shape[1]))
mask[w] = 1
#flat_mask = mask.flatten()
#fits.writeto('dtfe_map_z_0.3_0.5_512x512_mask.fits',flat_mask)
#matplotlib.image.imsave('dtfe_map_z_0.3_0.5_512x512_mask.png',mask)

### mean density in mask
data_map2 = np.copy(data_map)
data_map2[w] = data_map.mean()
#fits.writeto('dtfe_map_z_0.3_0.5_512x512_meanDinMask.fits',data_map2)

### max density in mask
data_map2 = np.copy(data_map)
data_map2[w] = data_map.max()
#fits.writeto('dtfe_map_z_'+str(zmin)+'_'+str(zmax)+'_512x512_maxDinMask.fits',data_map2)

def plot_mask(mask):
    plt.figure()
    #plt.imshow(data_map,origin='lower',cmap='jet',extent=(RAmin,RAmax,DECmin,DECmax),aspect='auto')
    #plt.imshow(mask,origin='lower',cmap='binary_r',extent=(RAmin,RAmax,DECmin,DECmax),aspect='auto',alpha=0.4)
    plt.imshow(mask,origin='lower',cmap='binary',extent=(RAmin,RAmax,DECmin,DECmax),aspect='auto')
    plt.xlabel(r'$\alpha_{J2000}$')
    plt.ylabel(r'$\delta_{J2000}$')
    plt.colorbar(label='$N_{gal}$')
    plt.show()
    

### critical points
pers = 0.15
f = np.load('dtfe_map_z_'+str(zmin)+'_'+str(zmax)+'_512x512_maxDinMask_adims3_sourceMinDownSkl_'+str(pers)+'.npz')
crit_pos = f['crit_pos']
crit_type = f['crit_type']
def plot_density_map_cp():
    plt.figure()
    data_map_smoothed = gaussian_filter(data_map,sigma=3)
    plt.imshow(data_map_smoothed,origin='lower',cmap='jet',aspect='auto',extent=(RAmin,RAmax,DECmin,DECmax))
    plt.xlabel(r'$\alpha_{J2000}$')
    plt.ylabel(r'$\delta_{J2000}$')
    plt.colorbar(label=r'$\rho$')
    
    w = np.where(crit_type==0)[0]
    #X = RAmin + crit_pos[1,w] * (RAmax-RAmin)/data_map_smoothed.shape[0]
    #Y = DECmin + crit_pos[0,w] * (DECmax-DECmin)/data_map_smoothed.shape[1]
    X = RAmin + crit_pos[0,w] * (RAmax-RAmin)/data_map.shape[0]
    Y = DECmin + crit_pos[1,w] * (DECmax-DECmin)/data_map.shape[1]
    plt.scatter(X,Y,marker='*',s=150,color='magenta')
    #plt.scatter(crit_pos[1,w],crit_pos[0,w],marker="*",color="k")
    
    plt.title('pers='+str(pers))
    plt.show()

# saving the min in RA,DEC coordinates
w = np.where(crit_type==0)[0]
X = RAmin + crit_pos[0,w] * (RAmax-RAmin)/data_map.shape[0]
Y = DECmin + crit_pos[1,w] * (DECmax-DECmin)/data_map.shape[1]
np.savez('dtfe_map_z_'+str(zmin)+'_'+str(zmax)+'_512x512_maxDinMask_voids.npz',RA=X,DEC=Y)

if __name__=='__main__':
    #plot_mask(mask)
    #plot_density_map(data_map2)
    plot_density_map_cp()
    
    
    
    print()
