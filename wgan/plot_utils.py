5# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 12:36:54 2018

@author: Juan Jose Zamudio
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wgan_utils import power_spectrum_np, power_spectrum_np_2d
from torch.autograd import Variable
import torch
import itertools

from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from scipy.signal import medfilt

import matplotlib as mpl

mean_5=14592.24
std_5=922711.56
max_5=4376932000
#mask_value=10**2

#Bootstrap
mean_l5 = 2.7784111
std_l5 = 1.5777067
max_l5 =22.199614


def plot_loss(datalist, ylabel, log_, save_plot='', t='', med_filter=False, show_plot=False):
    
    plt.figure(figsize=(20,10))
    lw =1
    
    if med_filter == True:
        
        datalist = medfilt(datalist, kernel_size=199)
        lw = 3
    
    if ylabel=='Wasserstein loss':
        
        plt.plot([-x for x in datalist], linewidth = lw, color='b')
        #plt.ylim(0,max([-x for x in datalist]))
        
    else:
        plt.plot([x for x in datalist], linewidth = lw , color='b')
        
    plt.ylabel(ylabel, fontsize=16)
    plt.yticks(fontsize=14)
    
    if ylabel=='Generator loss':
        plt.xlabel('Epoch', fontsize=16)
    else:
        plt.xlabel('Iterations', fontsize=16)
    
    plt.grid()
    
    if show_plot == True:
        plt.show()
        
    if save_plot != '':
        plt.savefig(save_plot+ "wass_loss_" + str(t) + ".png", bbox_inches='tight')
    plt.close()
    

def visualize_cube(cube,  
             edge_dim, fig=None, ax=None,
             fig_size=(10,10),
             start_cube_index_x=0,
             start_cube_index_y=0,
             start_cube_index_z=0,
             norm_multiply=1e2,
             size_magnitude = False,
             save_fig = False,
            raw_cube_max = False,
            show_plot=False):

    """Takes as input;
    - cube: A 3d numpy array to visualize (scaled to [0,1],
    - edge_dim: edge length,
    - fig_size: Figure size for the plot,
    - norm_multiply: Multiplication factor to enable matplotlib to 'see' the particles,
    - color_map: A maplotlib colormap of your choice,
    Returns: 
    - The cube visualization
    - Also saves PNG file
    PROBLEMS:
    - Plotting everypoint (not truncating) with colorscale from 0 to 1 takes a really long time
    - size = magnitude -> some dots get too big to obscure view
    """
    
    #time_start = timeit.default_timer()
        
    cube_size = edge_dim
    edge = np.array([*range(cube_size)])
    
    end_x = start_cube_index_x + cube_size
    end_y = start_cube_index_y + cube_size
    end_z = start_cube_index_z + cube_size
    
    if fig == None:
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111, projection='3d')

    data_value = cube[start_cube_index_x:end_x, start_cube_index_y:end_y, start_cube_index_z:end_z]
    
    x,y,z = edge,edge,edge
    product = [*itertools.product(x,y,z)]
    
    X = np.array([product[k][0] for k in [*range(len(product))]])
    Y = np.array([product[k][1] for k in [*range(len(product))]])
    Z = np.array([product[k][2] for k in [*range(len(product))]])
    
    ## map data to 1d array that corresponds to the axis values in the product array
    data_1dim = np.array([data_value[X[i]][Y[i]][Z[i]] for i in [*range(len(product))]])
#     print("data_1dim = " + str(data_1dim))
    
#     initial_mean = np.mean(data_1dim) - stdev_to_white*np.std(data_1dim)
#     mask = data_1dim / raw_cube_max > 0.
#     mask = mask.astype(np.int)
    
    n = -262000
    #n = 64*64*64-1
    sorted_data = np.sort(data_1dim)
    n_max = sorted_data[n]
    mask = data_1dim > n_max
    mask = mask.astype(np.int)    
#     """
#     Plot Randomly Selected Points
#     """
#     p = 0.05 # the proportion of all points to be plotted
#     mask = np.random.binomial(n = 1, p = p, size = len(data_1dim))
    """
    Masking part of the data to speed up plotting time
    """
    data_1dim = np.multiply(mask,data_1dim)

    X, Y, Z, data_1dim = [axis[np.where(data_1dim> -0.1)] for axis in [X,Y,Z,data_1dim]]
    
    if True:

        #cmap = colors.LinearSegmentedColormap.from_list("", ["white","blue"])
        #cmap = viridis'
        #new_cmap = truncate_colormap(cmap, minval = 0.2, maxval = 0.3, n=10000)
        ## IGNORE BELOW 3D PLOT FORMATTING 

        cube_definition = [(start_cube_index_x, start_cube_index_x, start_cube_index_x),
                          (start_cube_index_x, start_cube_index_x+edge_dim, start_cube_index_x),
                          (start_cube_index_x + edge_dim, start_cube_index_x, start_cube_index_x),
                          (start_cube_index_x, start_cube_index_x, start_cube_index_x+edge_dim)]

        cube_definition_array = [
            np.array(list(item))
            for item in cube_definition
        ]

        points = []
        points += cube_definition_array
        vectors = [
            cube_definition_array[1] - cube_definition_array[0],
            cube_definition_array[2] - cube_definition_array[0],
            cube_definition_array[3] - cube_definition_array[0]
        ]

        points += [cube_definition_array[0] + vectors[0] + vectors[1]]
        points += [cube_definition_array[0] + vectors[0] + vectors[2]]
        points += [cube_definition_array[0] + vectors[1] + vectors[2]]
        points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

        points = np.array(points)
        edges = [
            [points[0], points[3], points[5], points[1]],
            [points[1], points[5], points[7], points[4]],
            [points[4], points[2], points[6], points[7]],
            [points[2], points[6], points[3], points[0]],
            [points[0], points[2], points[4], points[1]],
            [points[3], points[6], points[7], points[5]]
        ]

        faces = Poly3DCollection(edges, linewidths=1, edgecolors='k',)
        faces.set_facecolor((0,0,1,0)) ## set transparent facecolor to the cube

        ax.add_collection3d(faces)

        ax.scatter(points[:,0], 
                   points[:,1], 
                   points[:,2], 
                   s=0)

        ax.set_aspect('equal')
        
        new_cmap='viridis'
        
        v = plt.cm.get_cmap(new_cmap)
        rgba = v(0.0)
        
        ax.w_xaxis.set_pane_color(rgba)
        ax.w_yaxis.set_pane_color(rgba)
        ax.w_zaxis.set_pane_color(rgba)
        

        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        
        
        ax.xaxis.set_major_locator(MultipleLocator(edge_dim))
        ax.yaxis.set_major_locator(MultipleLocator(edge_dim))
        ax.zaxis.set_major_locator(MultipleLocator(edge_dim))

        ax.grid(False)

        ax.set_xlim3d(0,edge_dim)
        ax.set_ylim3d(0,edge_dim)
        ax.set_zlim3d(0,edge_dim)
    #     ax.get_frame_on()

        ax.xaxis._axinfo['tick']['inward_factor'] = 0
        ax.xaxis._axinfo['tick']['outward_factor'] = 0
        ax.yaxis._axinfo['tick']['inward_factor'] = 0
        ax.yaxis._axinfo['tick']['outward_factor'] = 0
        ax.zaxis._axinfo['tick']['inward_factor'] = 0
        ax.zaxis._axinfo['tick']['outward_factor'] = 0

        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.axes.zaxis.set_ticklabels([])
        #ax.get_zaxis().set_ticks([])
        
        ax.scatter(X, Y, Z,      
                   c=data_1dim, 
                   cmap=new_cmap,
                   s=2,           
                   alpha = 1,
                   vmin= -0.1,
                   vmax= 0.36,
                   edgecolors="face")
        
        if show_plot== True:
           plt.show()
        
        if save_fig != '':
            fig.savefig(save_fig, bbox_inches='tight', dpi = 1000)
        
        #return fig    
        #plt.close(fig)

def visualize2d(real, fake, log=False, save='', t='', show_plot=True):
  
    #min_= 0
    #max_= 0.22
    
    min_= -0.05
    max_= 0.27

    cols=8
    rows=2
    color = 'viridis'
    
    fig, axes = plt.subplots(nrows = rows, ncols=cols, figsize=(16,4))
    
    for ax, row in zip(axes[:,0], ['Generated', 'Real']):
        ax.set_ylabel(row, rotation=90, fontsize=16)
    
    mf, mr= 0,0
    for ax in axes.flat:
        #Plot only half of the mini-batch
        if mf<cols:
            if log==True:
                im = ax.imshow(np.log(fake[mf][0].mean(axis=2)), aspect='equal', cmap = color,
                       interpolation=None, vmin=min_, vmax=max_)
            else:
                im = ax.imshow((fake[mf][0].mean(axis=2)), aspect='equal', cmap = color,
                       interpolation=None, vmin=min_, vmax=max_)
            mf+=1
        
        else:
            if log==True:
                im = ax.imshow(np.log(real[mr][0].mean(axis=2)), aspect='equal', cmap = color,
                       interpolation=None, vmin=min_, vmax=max_)
            else:
                im = ax.imshow((real[mr][0].mean(axis=2)), aspect='equal', cmap = color,
                       interpolation=None, vmin=min_, vmax=max_)
                
            mr+=1
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout() 
    
    if show_plot == True:
        plt.show()
    
    if save!='':
        fig.savefig(save+'samples_'+str(t)+'.png')
           
    plt.close()
    #plt.show()  

def visualize2d_2d(real, fake, log=False, save='', t='', show_plot=True):
    min_= -0.0001
    max_= 0.3
    
    #min_= 0
    #max_= 0.22
    
    #max_=(max_l5-mean_l5) / std_l5
    
    cols=8
    rows=2
    
    fig, axes = plt.subplots(nrows=2, ncols=cols, figsize=(16,4))
    
    for ax, row in zip(axes[:,0], ['Generated', 'Real']):
        ax.set_ylabel(row, rotation=90, fontsize=16)
    
    mf, mr=0,0
    for ax in axes.flat:
        #Plot only half of the mini-batch
        if mf<cols:
            if log==True:
                im = ax.imshow(np.log(fake[mf][0]), aspect='equal', #cmap='Blues',
                       interpolation=None, vmin=min_, vmax=max_)
            else:
                im = ax.imshow((fake[mf][0]), aspect='equal', #cmap='Blues',
                       interpolation=None, vmin=min_, vmax=max_)
            mf+=1
        
        else:
            if log==True:
                im = ax.imshow(np.log(real[mr][0]), aspect='equal', #cmap='Blues',
                       interpolation=None, vmin=min_, vmax=max_)
            else:
                im = ax.imshow((real[mr][0]), aspect='equal', #cmap='Blues',
                       interpolation=None, vmin=min_, vmax=max_)
                
            mr+=1
        ax.set_xticks([])
        ax.set_yticks([])
    #fig.subplots_adjust(right=.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #fig.colorbar(ax, cax=cbar_ax)
    fig.tight_layout() 
    
    if show_plot == True:
        plt.show()
    
    if save!='':
        fig.savefig(save+'samples_'+str(t)+'.png')
    

    plt.close()    
  
    
    
def plot_densities(fake, real, num_plots, log_):
    fig, ax = plt.subplots(figsize=(20,10))
    ax.plot(xlabel='big', ylabel='big', title='Density')
    sns.set(rc={'figure.figsize':(20,10),"lines.linewidth": 1})
    #plt.subplots(figsize=(20,10))
    ax.set_ylim(0,9)
    #ax.set_xlim(0, )
    bw=0.005
    eps=0
    grid=200
    n=0
    if log_==True:
        fake=np.log(fake+eps)
        real=np.log(real+eps)
    
    for m in range(fake.shape[0]):
        n+=1
        
        if m <= num_plots:
            if n==1:
                sns.kdeplot(fake[m][0].flatten(), ax=ax, bw=bw, label='Generated', color='red');
                sns.kdeplot(real[m][0].flatten(), ax=ax, bw=bw,  label='Real' ,   color='blue');
            else:
                sns.kdeplot(fake[m][0].flatten(), ax=ax, bw=bw, color='red');
                sns.kdeplot(real[m][0].flatten(), ax=ax, bw=bw, color='blue');
        
        ax.set_title("Kernel densitiy estimates of log HI mass",fontsize=25)
        ax.tick_params(labelsize=20)
        plt.setp(ax.get_legend().get_texts(), fontsize='20')
    plt.show()

def adjust_cube(cube, log = False):
    #new_cube = cube[6.881349e-10]
    if log == True:
        cube[cube < 0] = 0
    else:
        cube[cube < 6.881349e-10] = 0
    
    return cube



def histogram_mean_confint(noise, real, log_plot,  t, save_plot, show_plot, adjust=False, d2=False):
    """
    Plots the mean and confidence intervals for each bin of the histogram
    Args:
        real(): real data
        epoch(integer): epoch number
        file_name(string): name of the file
        hd (integer) : if 0 it's a histogram, if 1 it's a pdf
        
    """
    batch_size = noise.shape[0]
    
#     plt.figure(figsize = (20,10))
    plt.figure(figsize = (14,8))
    plot_min = min(float(noise.min()), float(real.min()))
    plot_max = max(float(noise.max()), float(real.max()))
#     plt.xlim(0.01, 0.5)
    plt.xlim(plot_min, plot_max)
    plt.ylim(0, 20)
    font_size = 14
    
#     bins = np.linspace(0, 0.5, 500)
#     bins = np.linspace(plot_min, 0.5, 500)
    
    if d2 == False:
        bins = np.linspace(plot_min, plot_max, 500)
    else:
        bins = np.linspace(plot_min, plot_max, 80)
        
    for m in range(min(batch_size, 64)):

        #if len(real.shape) == 5:
        real_viz = real[m][0].flatten()
        noise_viz = noise[m][0].flatten()
        
        if adjust == True:
            noise_viz = adjust_cube(noise_viz, log=True)
        
        #elif len(real.shape) == 4:
         #   real_viz = real[m].flatten()
          #  noise_viz = noise[m].flatten()   
      #  else:
       #     raise NotImplementedError

        bin_vals_real, _ , _ = plt.hist(real_viz, bins = bins, 
                                 color = "b" , log = False, alpha = 0.00, 
                                 normed=True, label='Real')
        bin_vals_noise, _ , _ = plt.hist(noise_viz, bins = bins, 
                             color = "b" , log = False, alpha = 0.00, 
                             normed=True, label='Generated')

        if m == 0:
            bin_vals_m_real = bin_vals_real
            bin_vals_m_noise = bin_vals_noise
        else:
            bin_vals_m_real = np.column_stack((bin_vals_m_real,bin_vals_real))
            bin_vals_m_noise = np.column_stack((bin_vals_m_noise,bin_vals_noise))

    # take column wise mean
    col_means_real = np.mean(bin_vals_m_real, axis = 1)
    col_means_noise = np.mean(bin_vals_m_noise, axis = 1)

    # calculate column wise stddev
    col_stddev_real = np.std(bin_vals_m_real, axis = 1)
    col_stddev_noise = np.std(bin_vals_m_noise, axis = 1)

    # plot means and confidence interval
    bins = bins[1:]
    plt.errorbar(x = bins, 
                 y = col_means_real, 
                 yerr = col_stddev_real, linestyle=None, marker='o',capsize=3, 
                 markersize = 2, color = "blue", alpha = 0.25)
    plt.errorbar(x = bins, 
                 y = col_means_noise, 
                 yerr = col_stddev_noise, linestyle=None, marker='o',capsize=3, 
                 markersize = 2, color = "red", alpha = 0.25)

    if d2 == False:
        plt.xlim(-0.4, 0.6)
        plt.ylim(0, 7)
    else:
        plt.xlim(0, 0.8)
        #plt.ylim(0, 10)
        
    plt.tick_params(axis='both', labelsize=font_size)
    plt.title('Empirical Distributions of Real (blue) and Generated Samples (red)', fontsize=font_size)
#     plt.legend(fontsize=font_size)
    
    if save_plot!='':
        plt.savefig(save_plot+ "hist_meanstddev_" + str(t) + ".png", 
                bbox_inches='tight') 
    
    if show_plot:
        plt.show() 
        
    plt.close()
    
    

    
def plot_power_spec( generated_cube,   # should be inverse_transformed
                    raw_cube_mean,
                    s_size,
                    log_scale=True,
                    threads=1, 
                    MAS="CIC", 
                    axis=0, 
                    BoxSize=75.0/2048*64,
                    save_plot='',
                    t='',
                    d2=False):
    """Takes as input;
    - Real cube: (batch_size x 1 x n x n x n) torch cuda FloatTensor,
    - Generated copy: (batch_size x 1 x n x n x n) torch cuda FloatTensor,
    - constant assignments: threads, MAS, axis, BoxSize.
    Returns;
    - Power spectrum plots of both cubes
    in the same figure.
    """
    
    
    if d2 == True:
        stats = pd.read_csv('Pk_stats_2d.csv', index_col = [0])   
        generated_cube = generated_cube.reshape(-1,
                                            1,
                                            generated_cube.shape[2],
                                            generated_cube.shape[2])
    else:
        stats = pd.read_csv('Pk_stats.csv', index_col = [0])
        generated_cube = generated_cube.reshape(-1,
                                            1,
                                            generated_cube.shape[2],
                                            generated_cube.shape[2],
                                            generated_cube.shape[2])
    
    plt.figure(figsize=(14,8))
    
    for cube_no in range(generated_cube.shape[0]):
        
        delta_gen_cube = generated_cube[cube_no][0]
        
        #k_real ,Pk_real = power_spectrum_np(cube = delta_real_cube, mean_raw_cube = raw_cube_mean, SubBoxSize=BoxSize)
        if d2 == False:    
            k_gen,  Pk_gen  = power_spectrum_np(cube = delta_gen_cube,  mean_raw_cube = raw_cube_mean, SubBoxSize=BoxSize)
        else:
             k_gen,  Pk_gen  = power_spectrum_np_2d(cube = delta_gen_cube,  mean_raw_cube = raw_cube_mean, SubBoxSize=BoxSize)
            
        #plt.plot(np.log(k_real), np.log(Pk_real), color="b", alpha = 0.8, label="Real", linewidth = 2)
        plt.plot(k_gen, Pk_gen, color="r", alpha = 0.8, label="Generated", linewidth = 2)
    
    plt.errorbar(stats['k'], stats['mean'],  yerr=  stats['std'],  color='b', ecolor='lightblue', elinewidth=3, 
                 alpha=0.8, label='Mean of real samples' , fmt='o' )
    
    plt.rcParams["font.size"] = 16
    plt.title("Power Spectrum Comparison - (Blue: Real, Red: Generated)")
    plt.xlabel('k')
    plt.ylabel('Pk.k3D')

    
    if log_scale == True:
        plt.xscale('log')
        plt.yscale('log')

        plt.xlabel('log10k')
        plt.ylabel('log10(Pk.k3D)')
        
    
    if save_plot!='':
        plt.savefig(save_plot + 'powerspectrum_' + str(t) + '.png', 
                 bbox_inches='tight')
   
    #plt.show()
    plt.close()
    
    

    
    
def generate_images(G, no_images, epoch, dataset, save_plot):
    #f file has to be opened globally
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 8,shuffle=False, num_workers=0, drop_last = True)
    data_iter = iter(dataloader)
    
    for n in range(no_images):
        #real  = get_samples(0, s_sample=s_sample, nsamples=8, test_coords={'x': [0, 1023], 'y': [0, 1023], 'z': [0, 1023]})
        #real=real.reshape(-1, 1 ,real.shape[2], real.shape[2], real.shape[2])
        
        real = data_iter.next().numpy()
        
        noise = torch.FloatTensor(8, nz,1 , 1, 1).normal_(0,10)
        fake = G(Variable(noise)).detach().numpy()
        
        visualize2d(real, fake, log= False, save=save_plot, t=epoch)     
    return

def generate_3d_images(no_images, dataset, save_plot):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle=False, num_workers=0, drop_last = True)
    data_iter = iter(dataloader)
    for n in range(no_images):
        
        real = data_iter.next().numpy()
        real = real[0][0]

        noise = torch.FloatTensor(1, nz,1 , 1, 1).normal_(0,10)
        fake = netG(Variable(noise)).detach().numpy()
        fake = fake[0][0]

        visualize_cube(cube=real, edge_dim=s_sample,        
                 start_cube_index_x=0, start_cube_index_y=0, start_cube_index_z=0,
                 fig_size=(20,20),
                 norm_multiply=1e2, size_magnitude = False, raw_cube_max = 1,
                save_fig = save_plot)

        visualize_cube(cube=fake, edge_dim=s_sample,        
                 start_cube_index_x=0, start_cube_index_y=0, start_cube_index_z=0,
                 fig_size=(20,20),
                 norm_multiply=1e2, size_magnitude = False, raw_cube_max = 1,
                save_fig = save_plot)


def get_latents(mode, nb_latents=8, filter_latents=1):
    
    if mode == 'Gaussian':
        latents = np.random.randn(nb_latents, nz, 1, 1, 1).astype(np.float32)
    
        latents = ndimage.gaussian_filter(latents, [filter_latents, 0, 0, 0, 0], mode='wrap')
        latents /= np.sqrt(np.mean(latents**2))
    
    elif mode == 'Linear':
        #v_new = x v1 + (1-x) v2
        alphas = np.linspace(0, 1, num = nb_latents)
        
        z1 = torch.FloatTensor(nz, 1).normal_(0, 1).numpy()
        z2 = torch.FloatTensor(nz, 1).normal_(0, 1).numpy()
        
        latents = 0
        
        n=0
        for a in alphas:
            new_v = a * z1 + (1-a) * z2
            
            if n == 0:
                latents = new_v
            else:
                latents = np.hstack((latents, new_v))
            n+=1
        
        latents = latents.T
        latents = latents.reshape(nb_latents, nz, 1, 1, 1)

    else:
        print('Mode not implemented')
  
    return torch.from_numpy(latents)

def plot_interpolations(G, num_vectors, mode, dim, plot_show=True, save='', fig_size=(20,10)):
    
    z = get_latents(mode = mode , nb_latents = num_vectors, filter_latents=1)
    fake = G(Variable(z)).detach().numpy()
    cols = num_vectors
    rows = 1
    
    if dim == '3D' or dim == 'both':
        fig = plt.figure(figsize=fig_size)
        #fig.suptitle(mode +' interpolations in latent space', fontsize=26)
        
        for n in range(1, num_vectors+1):
            ax = fig.add_subplot(rows, num_vectors, n, projection='3d')
            
            visualize_cube(cube=fake[n-1][0], edge_dim=s_sample, fig = fig, ax = ax,
                 start_cube_index_x=0, start_cube_index_y=0, start_cube_index_z=0,
                 norm_multiply=1e2, size_magnitude = False, raw_cube_max = 1,
                save_fig = '')
            
        fig.tight_layout() 
        
    if dim == '2D' or dim == 'both':
        min_= -0.05   #-0.1
        max_= 0.27    #0.36
        
       
        color = 'viridis'
        
        fig, axes = plt.subplots(nrows = rows, ncols=cols, figsize=(fig_size[0]-2, fig_size[1]-2))
        n=0
        for ax in axes.flat:
            ax.imshow(fake[n][0].mean(axis=2), aspect='equal', cmap = color,
                       interpolation=None, vmin=min_, vmax=max_)
            n+=1
            
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout() 
    
    
    if plot_show == True:
        plt.show()
    
    if save!='':
        fig.savefig(save+'samples_'+str(t)+'.png')
           
    plt.close()





