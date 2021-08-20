import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import Creux_WL as creux
from scipy.spatial import Delaunay
from itertools import repeat
from multiprocessing import Pool
from scipy.interpolate import griddata
from astropy.cosmology import Planck15
from scipy.ndimage import gaussian_filter as gaussf
from matplotlib.colors import LogNorm
from astropy.io import fits

def how_many(tri,num_points):
    there_are=np.where(tri.simplices==num_points)[0]
    return there_are

def area_triangle(points):
    return 0.5*np.abs(np.dot(points[:,0],np.roll(points[:,1],1))-np.dot(points[:,1],np.roll(points[:,0],1)))

def area_delaunay(inputs):
    tri,num=inputs
    a=area_triangle(tri.points[tri.simplices[num]])
    return a

def get_areas(tri,the_pool):
    shape=tri.simplices.shape[0]
    areas=np.empty(shape)
    all_inputs=zip(repeat(tri),np.arange(shape))
    areas=the_pool.map(area_delaunay,all_inputs)
    return np.array(areas)

def vol_tetrahedron(points):
    return abs(np.dot((points[0]-points[3]),np.cross((points[1]-points[3]),(points[2]-points[3]))))/6.

def vol_delaunay(inputs):
     tri,num=inputs
     a=vol_tetrahedron(tri.points[tri.simplices[num]])
     return a

def get_volumes(tri,the_pool):
    shape=tri.simplices.shape[0]
    volumes=np.empty(shape)
    all_inputs=zip(repeat(tri),np.arange(shape))
    volumes=the_pool.map(vol_delaunay,all_inputs)
    return np.array(volumes)

def get_densities3d(inputs):
    tri,num,volumes=inputs
    l=how_many(tri,num)
    true_vol=np.sum(volumes[l])/4.
    return 1./true_vol

def densities3d(tri,the_pool,volumes):
    shape=tri.points.shape[0]
    dens=np.empty(shape)
    all_inputs=zip(repeat(tri),np.arange(shape),repeat(volumes))
    dens=the_pool.map(get_densities3d,all_inputs)
    return np.array(dens)

def get_densities2d(inputs):
    tri,num,areas=inputs
    l=how_many(tri,num)
    true_vol=np.sum(areas[l])/3.
    return 1./true_vol

def densities2d(tri,the_pool,areas):
    shape=tri.points.shape[0]
    dens=np.empty(shape)
    all_inputs=zip(repeat(tri),np.arange(shape),repeat(areas))
    dens=the_pool.map(get_densities2d,all_inputs)
    return np.array(dens)

def map_dtfe3d(x,y,z,size):
    tab=np.vstack((x,y,z)).T
    tri=Delaunay(tab)
    the_pool=Pool()
    volumes=get_volumes(tri,the_pool)
    d=densities3d(tri,the_pool,volumes)
    the_pool.close()
    x_m=np.linspace(np.min(x),np.max(x),size)
    y_m=np.linspace(np.min(y),np.max(y),size)
    z_m=np.linspace(np.min(z),np.max(z),size)
    x_m,y_m,z_m=np.meshgrid(x_m,y_m,z_m)
    grid=griddata(tab,d,(x_m,y_m,z_m),method='linear')
    grid[np.isnan(grid)]=0
    return grid

def map_dtfe2d(x,y,size):
    print('First line')
    tab=np.vstack((x,y)).T
    print('Second line')
    tri=Delaunay(tab)
    print('Third line')
    the_pool=Pool()
    print('Fourth line')
    areas=get_areas(tri,the_pool)
    print('Fifth line')
    d=densities2d(tri,the_pool,areas)
    print('Sixth line')
    the_pool.close()
    print('Seventh line')
    x_m=np.linspace(np.min(x),np.max(x),size)
    print('Eighth line')
    y_m=np.linspace(np.min(y),np.max(y),size)
    print('Ninth line')
    x_m,y_m=np.meshgrid(x_m,y_m)
    print('Tenth line')
    grid=griddata(tab,d,(x_m,y_m),method='linear')
    print('Eleventh line')
    grid[np.isnan(grid)]=0
    print('Twelfth line')
    return grid

def make_dtfemap(lower, upper, size):
    binned_data = creux.slice_data(lower, upper)
    
    ra = binned_data['ALPHA_J2000']
    dec = binned_data['DELTA_J2000']
    
    print('Just before making the map')
    dtfe_map = map_dtfe2d(ra, dec, size)
    print('I just did the map !')
    np.save(f'../dtfe_map_z_{lower}_{upper}_{size}x{size}.npy', dtfe_map)

def plot_dtfemap(lower, upper, size, smooth = 0, vmin = None, vmax = None, norm = None):
    
    binned_data = creux.slice_data(lower, upper)
    ra = binned_data['ALPHA_J2000']
    dec = binned_data['DELTA_J2000']
    corners = [np.linspace(np.min(ra), np.max(ra), size), np.linspace(np.min(dec), np.max(dec), size)]
    
    dtfe_map = np.load(f'../dtfe_map_z_{lower}_{upper}_{size}x{size}.npy')
    dtfe_map = gaussf(dtfe_map, smooth)
    fits.writeto(f'../dtfe_map_z_{lower}_{upper}_{size}x{size}.fits', data = dtfe_map)
    print(np.min(dtfe_map), np.max(dtfe_map), np.mean(dtfe_map))
    fig = plt.figure(figsize = (15,15))
    ax = fig.add_subplot(111)
    field = ax.pcolormesh(corners[0], corners[1], dtfe_map, vmin = vmin, vmax = vmax, norm = norm, cmap = 'seismic')
    plt.colorbar(field, label = 'DTFE density')
    ax.set_title(f'DTFE map of the galaxy density field between z = {lower}, z = {upper}')
    ax.set_xlabel(r'$\alpha_{J2000}$(°)')
    ax.set_ylabel(r'$\delta_{J2000}$(°)')
    plt.savefig(f'../dtfe_map_z_{lower}_{upper}_{size}x{size}.png')
    plt.show()
