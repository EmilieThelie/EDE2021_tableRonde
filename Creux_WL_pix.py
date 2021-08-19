import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18

### data
data = pd.read_csv('/home/raphael/Téléchargements/result_cu1jemw77zite8se.csv')
cleaned_data = data[(data['MAG_r'] != 99.) & (data['MAG_i'] != 99.)]
RAmin, RAmax = 207,222
DECmin, DECmax = 50, 58

### density map
Npix = 100

print('Do you want to plot the data ? (y/n)')
plot = input()
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

if __name__=="__main__":
    lower_z, higher_z = 0.3, 0.5
    data_in_z_bin = cleaned_data[(cleaned_data['Z_B'] > lower_z) & (cleaned_data['Z_B'] < higher_z)].value
    
    data_map = np.zeros((Npix,Npix))
    
    data_map = np.histogram(data_in_z_bin,bins=np.linspace(data_in_z_bin.min(),data_in_z_bin.max(),))
