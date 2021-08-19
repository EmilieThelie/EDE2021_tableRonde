import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18


data = pd.read_csv('/home/raphael/Téléchargements/result_cu1jemw77zite8se.csv')
cleaned_data = data[(data['MAG_r'] != 99.) & (data['MAG_i'] != 99.)]

print('Do you want to plot the data ? (y/n)')
plut = input()
if plot == 'y' :
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(cleaned_data['Z_B'], bins = 100)
print(cleaned_data)

def slice_data(lower, upper):
    return binned_data

def main():
    return 0